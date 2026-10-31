# MOSS INT8 local image evidence for PR #9627

Actual pretrained outputs generated at `8ecd7d3de8d37ac11dfb193692d9685a040c6e49` on an RTX 4060 Ti 16 GB. Both variants use the same checkpoint, prompt, seed and request settings in [comparison.json](comparison.json).

| BF16 reference | INT8 main DiT |
| --- | --- |
| ![BF16](moss-bf16-4060ti.png) | ![INT8](moss-int8-4060ti.png) |

This local run uses eager execution, TORCH_SDPA and CPU layer offload for both the DiT and text encoder, with pinned memory disabled. A local loading helper initially places the text encoder on CPU in both variants so the full model fits. The production transformer, forward paths and real INT8 kernels are unchanged; no transformer or quantization-kernel mocks were used.

For this single pair, PSNR is 32.8114 dB and SSIM is 0.989868. These measure output drift for one prompt/seed, not general image quality. The separate A800 latency and device-peak results in the PR description were collected earlier at `f5523fd3b` using FlashAttention without this local offload setup.

[Model manifest](model-manifest.json) records the 37 file hashes matching ModelScope snapshot `1f920e15816c63f25466e4e37cf50647976aa5c2`. The assets live on a separate fork branch to keep binaries out of the feature PR diff.
