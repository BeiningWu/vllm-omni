# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS quantization routing, fused loading and activation precision."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.attention import selector
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.mossimage import mossimage_transformer as model
from vllm_omni.quantization import ComponentQuantizationConfig, int8_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _cpu_quantize(weight, scale=None):
    assert scale is None
    scale = weight.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (weight.float() / scale).round().clamp(-127, 127).to(torch.int8), scale, None


class _CPUScaledInt8Kernel:
    """Replace device kernels while retaining the real online quantization loader."""

    layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")

    def apply_weights(self, layer, x, bias=None):
        assert layer.weight.dtype == torch.int8
        assert x.dtype == torch.bfloat16
        weight = (layer.weight.t().float() * layer.weight_scale).to(x.dtype)
        return F.linear(x, weight, bias)


@pytest.fixture(autouse=True)
def single_rank_cpu(monkeypatch):
    tp_group = Mock(rank_in_group=0, world_size=1)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    monkeypatch.setattr(model, "get_sequence_parallel_world_size", lambda: 1)
    # Only backend selection and device kernels are replaced. Configuration,
    # real TP linear layers, lazy weight loading and all transformer math remain.
    monkeypatch.setattr(int8_config.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(int8_config.current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(
        int8_config.current_omni_platform,
        "get_diffusion_attn_backend_cls",
        lambda **_kwargs: DiffusionAttentionBackendEnum.TORCH_SDPA.get_path(),
    )
    monkeypatch.setattr(int8_config, "init_int8_linear_kernel", lambda **_kwargs: _CPUScaledInt8Kernel())
    monkeypatch.setattr(int8_config.ops, "scaled_int8_quant", _cpu_quantize)
    selector._cached_get_backend_cls.cache_clear()
    yield
    selector._cached_get_backend_cls.cache_clear()


def _make_transformer(quant_config=None, construction_dtype=torch.bfloat16):
    config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with (
        set_default_torch_dtype(construction_dtype),
        set_current_diffusion_config(config),
        set_forward_context(omni_diffusion_config=config),
    ):
        transformer = model.MOSSImageTransformer(
            od_config=config, dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=8, freq_dim=8, latent_channels=4
        ).eval()
    return transformer, config


def _checkpoint_weights(transformer):
    """Recreate checkpoint q/k/v and gate/up layouts from initialized BF16 weights."""
    weights = []
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for name, parameter in transformer.named_parameters():
            if parameter.ndim > 1:
                torch.nn.init.normal_(parameter, std=0.05)
            else:
                parameter.fill_(0 if name.endswith("bias") else 1)
            value = parameter.detach().clone()
            if ".proj.qkv." in name:
                for shard, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                    weights.append((name.replace(".proj.qkv.", f".proj.{shard}."), chunk.contiguous()))
            elif name.startswith("text_embedder.") and ".ffn_up_gate." in name:
                for shard, chunk in zip(("ffn_up", "ffn_gate"), value.chunk(2, dim=0)):
                    weights.append((name.replace(".ffn_up_gate.", f".{shard}."), chunk.contiguous()))
            else:
                weights.append((name, value))
    weights.append(("rope.bands", transformer.rope.bands.clone()))
    return weights


def _int8_config(ignored_layers=None):
    return int8_config.DiffusionInt8Config(ignored_layers=ignored_layers)


def _linears(transformer):
    return {name: layer for name, layer in transformer.named_modules() if isinstance(layer, LinearBase)}


@torch.no_grad()
def test_online_int8_loads_main_dit_and_preserves_text_refinement_and_bfloat16_activations():
    baseline, _ = _make_transformer()
    weights = _checkpoint_weights(baseline)
    transformer, config = _make_transformer(_int8_config())
    transformer.load_weights(iter(weights))

    expected_linears = {
        *(f"layers.{i}.proj.{name}" for i in range(2) for name in ("qkv", "o", "ffn_up_gate", "ffn_down")),
        *(f"text_embedder.blocks.{i}.{name}" for i in range(2) for name in ("qkv", "proj", "ffn_up_gate", "ffn_down")),
    }
    assert set(_linears(transformer)) == expected_linears
    for name, layer in _linears(transformer).items():
        source = baseline.get_submodule(name).weight
        if name.startswith("text_embedder."):
            assert isinstance(layer.quant_method, UnquantizedLinearMethod)
            assert layer.weight.dtype == torch.bfloat16
            torch.testing.assert_close(layer.weight, source, atol=0, rtol=0)
            continue
        assert isinstance(layer.quant_method, int8_config.Int8OnlineLinearMethod)
        assert layer.weight.dtype == torch.int8
        assert layer._already_called_process_weights_after_loading
        expected_weight, expected_scale, _ = _cpu_quantize(source)
        torch.testing.assert_close(layer.weight, expected_weight.t(), atol=0, rtol=0)
        torch.testing.assert_close(layer.weight_scale, expected_scale, atol=0, rtol=0)

    # Embedders, timestep/AdaLN modulation and the output head stay in BF16.
    for layer in (
        transformer.video_embedder,
        transformer.text_embedder.proj,
        transformer.time_embedding[0],
        transformer.shared_block_modulation,
        transformer.final_linear_video,
    ):
        assert layer.weight.dtype == torch.bfloat16

    latents = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
    text = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    with set_forward_context(omni_diffusion_config=config):
        output = transformer(latents, 500.0, text).sample
    assert output.shape == latents.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("ignored", "fused_name"),
    [
        (["layers.0.proj.qkv"], "layers.0.proj.qkv"),
        ([f"layers.0.proj.{name}" for name in ("q", "k", "v")], "layers.0.proj.qkv"),
        (["layers.0.proj.ffn_up_gate"], "layers.0.proj.ffn_up_gate"),
    ],
)
def test_ignored_layers_keep_the_entire_fused_linear_in_bfloat16(ignored, fused_name):
    transformer, _ = _make_transformer(_int8_config(ignored))
    linears = _linears(transformer)
    assert isinstance(linears[fused_name].quant_method, UnquantizedLinearMethod)
    assert linears[fused_name].weight.dtype == torch.bfloat16
    assert isinstance(linears["layers.1.proj.qkv"].quant_method, int8_config.Int8OnlineLinearMethod)


def test_partial_fused_layer_exclusions_are_rejected():
    with pytest.raises(ValueError, match="some but not all shards"):
        _make_transformer(_int8_config(["layers.0.proj.q"]))


@pytest.mark.parametrize("component", ["transformer", "text_encoder"])
def test_component_quantization_routes_to_the_transformer(component):
    quant_config = _int8_config()
    config = ComponentQuantizationConfig(component_configs={component: quant_config}, default_config=None)
    transformer, _ = _make_transformer(config)
    expected_quantized = component == "transformer"
    assert (transformer.quant_config is quant_config) == expected_quantized
    for name, layer in _linears(transformer).items():
        quantized_layer = expected_quantized and name.startswith("layers.")
        assert isinstance(layer.quant_method, int8_config.Int8OnlineLinearMethod) == quantized_layer
        assert layer.params_dtype == torch.bfloat16
        if not quantized_layer:
            assert layer.weight.dtype == torch.bfloat16


def test_component_config_preserves_fused_layer_exclusions():
    quant_config = _int8_config([f"layers.0.proj.{name}" for name in ("q", "k", "v")])
    config = ComponentQuantizationConfig(component_configs={"transformer": quant_config}, default_config=None)
    transformer, _ = _make_transformer(config)
    assert isinstance(transformer.layers[0].proj["qkv"].quant_method, UnquantizedLinearMethod)
    assert isinstance(transformer.layers[1].proj["qkv"].quant_method, int8_config.Int8OnlineLinearMethod)


@torch.no_grad()
def test_default_bfloat16_matches_a_component_config_without_transformer_quantization():
    baseline, baseline_config = _make_transformer()
    weights = _checkpoint_weights(baseline)
    config = ComponentQuantizationConfig(component_configs={"text_encoder": _int8_config()}, default_config=None)
    transformer, transformer_config = _make_transformer(config)
    transformer.load_weights(iter(weights))
    assert all(isinstance(layer.quant_method, UnquantizedLinearMethod) for layer in _linears(transformer).values())
    latents = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
    text = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    with set_forward_context(omni_diffusion_config=baseline_config):
        expected = baseline(latents, 500.0, text).sample
    with set_forward_context(omni_diffusion_config=transformer_config):
        actual = transformer(latents, 500.0, text).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_bfloat16_forward_after_converting_a_float32_model():
    transformer, config = _make_transformer(construction_dtype=torch.float32)
    transformer.to(dtype=torch.bfloat16)
    _checkpoint_weights(transformer)
    assert transformer.layers[0].proj["qkv"].params_dtype == torch.float32
    latents = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
    text = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    with set_forward_context(omni_diffusion_config=config):
        actual = transformer(latents, 500.0, text).sample
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
