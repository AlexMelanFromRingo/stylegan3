# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
import uuid
import warnings

import torch
import torch.utils.cpp_extension
from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

# The compiled CUDA kernels are an optimization. Every op also has a native PyTorch implementation, which is used
# when the kernels cannot be compiled (no CUDA toolkit / C++ compiler, or a toolkit that does not match PyTorch).
#   STYLEGAN_CUSTOM_OPS=auto     try to compile the kernels, fall back to the native implementation with a warning (default)
#   STYLEGAN_CUSTOM_OPS=off      never compile, always use the native implementation
#   STYLEGAN_CUSTOM_OPS=require  fail if the kernels cannot be compiled
mode = os.environ.get('STYLEGAN_CUSTOM_OPS', 'auto').lower()

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_mangled_gpu_name():
    name = torch.cuda.get_device_name().lower()
    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

#----------------------------------------------------------------------------
# CUDA toolkit discovery. The kernels have to be compiled with a CUDA toolkit whose major version matches the one
# PyTorch was built with. When the system toolkit is missing or does not match, the toolkit published by NVIDIA as pip
# packages (e.g. `pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*"` for a PyTorch built with CUDA 13.0)
# is used instead, without any system wide installation.

def _nvcc_version(nvcc):
    try:
        out = subprocess.run([nvcc, '--version'], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'release (\d+)\.(\d+)', out)
    return (int(match.group(1)), int(match.group(2))) if match else None

def _torch_cuda_version():
    return tuple(int(v) for v in torch.version.cuda.split('.')[:2]) if torch.version.cuda else None

def _pip_cuda_home():
    # Linux only: assembles a CUDA_HOME directory out of symlinks to the nvidia-* pip packages
    if os.name == 'nt':
        return None
    for site in sys.path:
        nvidia = os.path.join(site, 'nvidia')
        for nvcc in sorted(glob.glob(os.path.join(nvidia, '*', 'bin', 'nvcc'))):
            version = _nvcc_version(nvcc)
            if version is None or version[0] != _torch_cuda_version()[0]:
                continue
            home = os.path.join(torch.utils.cpp_extension._get_build_directory('cuda_home', verbose=False), f'{version[0]}.{version[1]}') # pylint: disable=protected-access
            if not os.path.isfile(os.path.join(home, 'bin', 'nvcc')):
                tmp = f'{home}-tmp-{uuid.uuid4().hex}'
                for sub in ['include', 'lib64']:
                    os.makedirs(os.path.join(tmp, sub))
                nvcc_root = os.path.dirname(os.path.dirname(nvcc))
                os.symlink(os.path.join(nvcc_root, 'bin'), os.path.join(tmp, 'bin'))
                if os.path.isdir(os.path.join(nvcc_root, 'nvvm')):
                    os.symlink(os.path.join(nvcc_root, 'nvvm'), os.path.join(tmp, 'nvvm'))
                # merge the headers and libraries of all the nvidia packages
                for src, dst in [('include', 'include'), ('lib', 'lib64')]:
                    for pkg_dir in sorted(glob.glob(os.path.join(nvidia, '*', src))):
                        for entry in os.listdir(pkg_dir):
                            target = os.path.join(tmp, dst, entry)
                            if not os.path.lexists(target):
                                os.symlink(os.path.join(pkg_dir, entry), target)
                cudart = sorted(glob.glob(os.path.join(tmp, 'lib64', 'libcudart.so.*')))
                if cudart and not os.path.lexists(os.path.join(tmp, 'lib64', 'libcudart.so')):
                    os.symlink(cudart[0], os.path.join(tmp, 'lib64', 'libcudart.so'))
                try:
                    os.replace(tmp, home)
                except OSError:
                    shutil.rmtree(tmp, ignore_errors=True)
            return home
    return None

def _setup_build_tools():
    # ninja from the pip package, also when the python environment is not activated
    if shutil.which('ninja') is None:
        try:
            import ninja # pylint: disable=import-outside-toplevel
            os.environ['PATH'] = ninja.BIN_DIR + os.pathsep + os.environ.get('PATH', '')
        except ImportError:
            pass

    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    system_version = _nvcc_version(os.path.join(cuda_home, 'bin', 'nvcc')) if cuda_home else None
    if torch.version.cuda is None or (system_version is not None and system_version[0] == _torch_cuda_version()[0]):
        return

    pip_home = _pip_cuda_home()
    if pip_home is not None:
        os.environ['CUDA_HOME'] = pip_home
        torch.utils.cpp_extension.CUDA_HOME = pip_home
    elif system_version is None:
        raise RuntimeError(f'No CUDA toolkit found. Install the CUDA {torch.version.cuda} toolkit, or `pip install "nvidia-cuda-nvcc=={torch.version.cuda}.*" "nvidia-cuda-cccl=={torch.version.cuda}.*"` (Linux)')
    else:
        raise RuntimeError(f'The CUDA toolkit in {cuda_home} is version {system_version[0]}.{system_version[1]}, but PyTorch was built with CUDA {torch.version.cuda}. Install a matching toolkit, or `pip install "nvidia-cuda-nvcc=={torch.version.cuda}.*" "nvidia-cuda-cccl=={torch.version.cuda}.*"` (Linux)')

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        _setup_build_tools()

        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Some containers set TORCH_CUDA_ARCH_LIST to a list that can either
        # break the build or unnecessarily restrict what's available to nvcc.
        # Unset it to let nvcc decide based on what's available on the
        # machine.
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        #
        # EDIT: We now do it regardless of TORCH_EXTENSIOS_DIR, in order to work
        # around the *.cu dependency bug in ninja config.
        #
        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1: # and ('TORCH_EXTENSIONS_DIR' in os.environ):

            # Compute combined hash digest for all source files.
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            # Select cached build directory name.
            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{_get_mangled_gpu_name()}')

            if not os.path.isdir(cached_build_dir):
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir) # atomic
                except OSError:
                    # source directory already exists, delete tmpdir and its contents.
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir): raise

            # Compile.
            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            module = torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs)
        else:
            module = torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)

        # Load. Recent PyTorch versions return the module without registering it in sys.modules.
        if module is None:
            module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache dict.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Returns the compiled plugin, or None if the custom ops are disabled or could not be compiled.
# The ops then use their native PyTorch implementations.

_failed_plugins = dict()

def try_get_plugin(module_name, **kwargs):
    assert mode in ['auto', 'off', 'require'], f'STYLEGAN_CUSTOM_OPS must be auto, off or require, not {mode}'
    if mode == 'off' or not torch.cuda.is_available():
        return None
    if module_name in _failed_plugins:
        return None
    try:
        return get_plugin(module_name, **kwargs)
    except Exception as e: # pylint: disable=broad-except
        if mode == 'require':
            raise
        reason = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        _failed_plugins[module_name] = reason
        warnings.warn(f'Could not compile the custom CUDA kernels of "{module_name}" ({reason}). Using the native PyTorch implementation instead, '
            'which is a bit slower. Set STYLEGAN_CUSTOM_OPS=off to skip the compilation, or see docs/troubleshooting.md.', RuntimeWarning)
        return None

#----------------------------------------------------------------------------
