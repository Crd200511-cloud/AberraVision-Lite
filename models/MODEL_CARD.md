# Model card: AberraVision Lite

- Architecture: five 3×3 convolutional layers, channels 2–32–32–32–32–1, residual addition to Wiener; 28,641 parameters.
- Inputs: unclipped linear grayscale simulated observation and its Wiener restoration, in this order. Target is not an input.
- Scope: fixed PSF a=0.15 RMS waves, q step 0.25 λf/D, 129×129 cropped PSF, σ=0.01 Gaussian noise, K=0.01, periodic boundary.
- Data: project-specific 100/20/30 original-image split inside official DIV2K train HR. See `../docs/DATA_SOURCE.md`.
- Training: one formal 30-epoch run from a fresh initialization; seed 20261013; Adam lr=0.0005; L1; CPU float32. A separate small pilot preceded it.
- Selected epoch: 25, using minimum validation L1; test results do not select the checkpoint.
- Test: 30 held-out source photos, fixed center 256×256, equal output clipping and data_range=1. CNN average PSNR 25.0668 dB and SSIM 0.74971; paired gains +1.5893 dB and +0.09861.
- SHA-256: `05c43f9425fcfb3e23fad3c6e75d3ff1c00501f1226609fc5de9388727397ba3`.
- Format: PyTorch state_dict plus configuration and provenance; load with `weights_only=True`. It contains no optimizer state.
- Intended use: educational reproduction and academic research under the stated simulated conditions.
- Not validated: real cameras, unknown aberrations, other noise levels, color, out-of-distribution images, cross-seed stability. Restored details are estimates, not evidence of physical recovery beyond the diffraction limit.
- Rights: third-party training photographs are not redistributed. No unrestricted commercial rights in those data are asserted. See `../THIRD_PARTY_NOTICES.md`.
- Development: substantial AI assistance; actual user contributions recorded in `../PARTICIPATION.md`.
