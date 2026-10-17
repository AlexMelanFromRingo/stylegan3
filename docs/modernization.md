# Modernization notes

What was changed to run this code on current software, why, and how it was measured. Network pickles are unchanged and
compatible in both directions: the official pretrained models load and produce the same images, and pickles trained with
this code load in the official code (the network classes keep their parameters and buffers).

Measurements: RTX 4080 SUPER (16 GB) under WSL2, Python 3.12, PyTorch 2.14 with CUDA 13.0. Training steps vary by up to
±20% between runs on this machine (WSL2 / WDDM scheduling and power limits), the numbers below are typical values of
repeated runs.

## Installation

| Problem with the official setup | Fix |
|---|---|
| `pyimgui` (visualizer) has no wheels for Python ≥ 3.12 and its sources no longer build | the visualizer uses [imgui-bundle](https://github.com/pthom/imgui_bundle) through `gui_utils/imgui_compat.py`, which exposes the subset of the pyimgui API the widgets use |
| `distutils` is removed in Python 3.12, `pkg_resources` no longer installed | replaced (`dnnlib/util.py`, `torch_utils/ops/*_gradfix.py`) |
| the CUDA kernels need a system CUDA toolkit matching PyTorch's CUDA version (e.g. Ubuntu 24.04 ships CUDA 12.0, current PyTorch wheels use 13.0) | the toolkit published by NVIDIA as pip packages is assembled into a `CUDA_HOME` automatically (`torch_utils/custom_ops.py`) |
| a kernel that fails to compile aborts the program | the ops fall back to native PyTorch implementations with a warning (`STYLEGAN_CUSTOM_OPS=auto|off|require`) |
| `cpp_extension.load()` of recent PyTorch versions does not register the module, the plugin import failed after a successful build | the returned module is used |
| pinned 2021 versions (`environment.yml`, Docker image 21.08) | `requirements.txt` with lower bounds only |

## Runtime fixes

- Training: `torch.utils.data.Sampler.__init__` no longer accepts the dataset (`InfiniteSampler`), and `torch.optim.Adam`
  rejects `betas=[0, 0.99]` (mixed int / float).
- FID: `scipy.linalg.sqrtm(..., disp=False)` fails with SciPy ≥ 1.18 - after all the features were computed. The Fréchet
  distance is now computed from an eigendecomposition in PyTorch (`frechet_distance()`), which gives the same value
  (relative difference ~1e-13, ~1e-7 for rank deficient covariances, e.g. fewer images than features) and is ~15x faster than `sqrtm` for 2048 features.
- Visualizer:
  - GLFW ≥ 3.4 prefers Wayland and PyOpenGL ≥ 3.1.7 switches to EGL whenever `WAYLAND_DISPLAY` is set (WSLg, GNOME, KDE...), which
    left PyOpenGL looking at the wrong context: `Attempt to retrieve context when no valid context`. Both now use X11 / GLX
    when an X display is available (`gui_utils/__init__.py`).
  - On WSL2, Mesa rendered with `llvmpipe` (CPU) instead of the GPU, the D3D12 driver is selected when present.
  - "Separate process" rendering: the process was forked when the global multiprocessing start method was already set,
    and CUDA cannot be initialized in a forked process. It uses its own `spawn` context now, and no longer hangs at exit.
  - FFT view: `matplotlib.cm.get_cmap()` was removed in matplotlib 3.9.
  - The font is taken from imgui-bundle instead of being downloaded at startup.

## Native implementations of the custom ops

Used when the CUDA kernels are not available. Both support gradients of any order, like the CUDA ops.

- `upfirdn2d`: upsampling as polyphase convolutions (one ordinary convolution with `up²` output phases, interleaved) instead of
  zero-stuffing the image, downsampling as a strided convolution that only computes the kept pixels, separable filters as two 1D
  passes. The backward pass is the same op with transposed parameters - the reference implementation instead relies on PyTorch's
  double backward of grouped convolutions, which loops over the channels and is extremely slow (R1 regularization).
  cuDNN's transposed convolutions turned out 2-6x slower than the polyphase formulation for these shapes.
- `filtered_lrelu`: like the CUDA op, only 2 bits per value of the upsampled activations are kept for the backward pass
  (negative / clamped), packed 4 per byte. The reference implementation keeps the upsampled tensor itself.

The native ops produce the same images as the kernels up to rounding (official StyleGAN3-T/R and StyleGAN2 256x256 models:
mean absolute difference 0.0003 - 0.001 on images in [-1, 1], i.e. about 1/8 of an 8-bit level). They cannot match the speed of the
fused CUDA kernel, which keeps the 4-16x larger intermediate images in on-chip memory:

| StyleGAN3-T 256x256 | CUDA kernels | native |
|---|---|---|
| training, batch 16 (`--batch-gpu=8`) | ~66 s/kimg, 5.6 GB | ~277 s/kimg, 7.8 GB |
| generation, batch 1 | 14.6 ms | 35 ms |
| generation, batch 1, StyleGAN3-R | 16.5 ms | 61 ms |

## Training speed and memory

- The magnitude EMA of every synthesis layer (`update_emas=True`) copied the activations to float32, squared and averaged them.
  It is now a single float32 reduction (`torch.linalg.vector_norm`): 4.8 ms → 0.5 ms and no 1.4 GB temporary for a
  16x512x148x148 activation.
- The G_ema update uses multi-tensor (`torch._foreach_*`) kernels instead of two kernels per parameter, Adam runs fused.
- Reference: the official numbers for StyleGAN3-T at 256x256 on one GPU are 89 s/kimg (V100) and 50 s/kimg (A100). With the
  kernels, this code runs at 63 - 69 s/kimg on an RTX 4080 SUPER (`--batch=16`).

Investigated and not adopted, because they did not help on this setup:

- TF32 (disabled by the official code for accuracy): no measurable difference with deterministic cuDNN algorithms. The
  convolutions of the low resolution layers are a small part of the time.
- Non-fused modulated convolutions (modulate the input, shared weights, demodulate the output - what StyleGAN2 uses for
  training): equivalent results, but the generator step became ~2x slower, so the grouped convolution of the official code stays.
- `channels_last` for the FP16 layers of the discriminator, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: within the noise.

On Windows and WSL2, the main cause of slow training is running close to the GPU memory limit: the driver then moves memory
to the system RAM instead of failing, and training becomes several times slower. A smaller `--batch-gpu` is the fix.

## New options

- `--loss=rpgan` (`training/loss.py: RpGANLoss`): relativistic pairing loss with R1 + R2 penalties from
  [R3GAN](https://arxiv.org/abs/2501.05441). Experimental, see the README.
- `fd_dinov2_50k_full`, `fd_dinov2_10k_full` (`metrics/`): Fréchet distance with DINOv2 ViT-L/14 features
  ([Stein et al. 2023](https://arxiv.org/abs/2306.04675)), loaded through `torch.hub`.
