# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys

# OpenGL platform setup. Must happen before OpenGL is imported, which imgui-bundle and gl_utils do on import.
if sys.platform.startswith('linux'):
    # GLFW >= 3.4 prefers Wayland and PyOpenGL >= 3.1.7 switches to EGL whenever WAYLAND_DISPLAY is set, which leaves
    # PyOpenGL looking at the wrong context ("no valid context"). When an X display is available, both use X11 / GLX.
    if os.environ.get('DISPLAY') and 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'glx'
    elif os.environ.get('WAYLAND_DISPLAY') and 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    # WSL: Mesa may fall back to software rendering (llvmpipe) instead of using the GPU through its D3D12 driver.
    _dri_dirs = ['/usr/lib/x86_64-linux-gnu/dri', '/usr/lib/aarch64-linux-gnu/dri', '/usr/lib64/dri', '/usr/lib/dri']
    if os.path.exists('/dev/dxg') and 'GALLIUM_DRIVER' not in os.environ and any(os.path.isfile(os.path.join(d, 'd3d12_dri.so')) for d in _dri_dirs):
        os.environ['GALLIUM_DRIVER'] = 'd3d12'
