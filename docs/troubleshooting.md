# Troubleshooting

Our PyTorch code uses custom [CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) to speed up some of the network layers.  Getting these to run can sometimes be a hassle.

This page aims to give guidance on how to diagnose and fix run-time problems related to these extensions.

## The custom CUDA kernels are optional

Every custom op also has a native PyTorch implementation. When the kernels cannot be compiled, a warning is printed once per op,
with the reason, and the native implementation is used: generation, the visualizer and training all work, training is about
4x slower and uses more GPU memory. The behavior is controlled with the `STYLEGAN_CUSTOM_OPS` environment variable:

- `auto` (default): try to compile the kernels, fall back to the native implementation with a warning.
- `off`: do not try to compile the kernels (no warning, faster startup).
- `require`: fail if the kernels cannot be compiled, with the full build error - useful to debug the compilation.

The native and the compiled ops produce the same images up to rounding (the mean difference is about 1/8 of an 8-bit level).

## Installing dependencies

The compilation needs:

- **A CUDA toolkit with the same major version as the CUDA version PyTorch was built with** (`python -c "import torch; print(torch.version.cuda)"`).
  - PyTorch invokes `nvcc` to compile our CUDA kernels.
  - On Linux, the easiest is to install the toolkit with pip into the same environment, it is found automatically. For PyTorch built with CUDA 13.0:
    `pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-cuda-cccl==13.0.*"`.
    Pin all four packages to the same version: `nvidia-nvvm` from a newer release produces PTX that the older `ptxas` rejects
    ("Unsupported .version").
  - A system wide installation ([CUDA toolkit](https://developer.nvidia.com/cuda-toolkit), `CUDA_HOME`, `nvcc` on the `PATH`) is used when its version matches.
    A mismatching system toolkit (e.g. the `nvidia-cuda-toolkit` package of Ubuntu 24.04 is CUDA 12.0) is skipped in favor of the pip one.
- **ninja** (`pip install ninja`). It is found also when the virtual environment is not activated.
- **GCC** (Linux) or **Visual Studio** with the C++ workload (Windows).
  - With a compiler newer than what `nvcc` officially supports, the build uses `--allow-unsupported-compiler`.

#### Why is CUDA toolkit installation necessary?

The PyTorch package contains the required CUDA toolkit libraries needed to run PyTorch, so why is a separate CUDA toolkit installation required?  Our models use custom CUDA kernels to implement operations such as efficient resampling of 2D images.  PyTorch code invokes the CUDA compiler at run-time to compile these kernels on first-use.  The tools and libraries required for this compilation are not bundled in PyTorch and thus a CUDA toolkit installation is required - NVIDIA publishes it as pip packages as well.

## Things to try

- Run with `STYLEGAN_CUSTOM_OPS=require` to see the full build error instead of the fallback warning.
- Completely remove: `$HOME/.cache/torch_extensions` (Linux) or `C:\Users\<username>\AppData\Local\torch_extensions\torch_extensions\Cache` (Windows) and re-run StyleGAN3 python code.
- Run ninja in `$HOME/.cache/torch_extensions` to see that it builds.
- Inspect the `build.ninja` in the build directories under `$HOME/.cache/torch_extensions` and check CUDA tools and versions are consistent with what you intended to use.

## Visualizer

- `OpenGL.error.Error: Attempt to retrieve context when no valid context`: GLFW and PyOpenGL use different window systems.
  On Linux, `gui_utils` makes both use X11 (GLX) when `DISPLAY` is set, and EGL otherwise. If you set `PYOPENGL_PLATFORM`
  yourself, it has to match the platform GLFW uses.
- Slow interface on WSL2: check that the OpenGL renderer is the GPU (`glxinfo -B | grep renderer` should name the GPU, not `llvmpipe`).
  The visualizer sets `GALLIUM_DRIVER=d3d12` on WSL when Mesa's D3D12 driver is installed (`sudo apt install mesa-utils libgl1-mesa-dri`).
- The training speed is very low on Windows / WSL2 when the GPU memory is nearly full: the driver moves memory to the system RAM
  instead of raising an out of memory error. Use a smaller `--batch-gpu`, or set "CUDA - Sysmem Fallback Policy" to
  "Prefer No Sysmem Fallback" in the NVIDIA Control Panel to get an error instead.
