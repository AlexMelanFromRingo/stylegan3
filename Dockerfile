# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# A pinned environment, as an alternative to `pip install -r requirements.txt`: the NGC image fixes Python, PyTorch,
# CUDA and the compiler for the custom kernels (nvcc 13.4, matching PyTorch), docker/requirements.lock the rest.
# Build:  docker build --tag stylegan3 .
# Run:    see "Docker" in README.md (needs the NVIDIA Container Toolkit for --gpus).
FROM nvcr.io/nvidia/pytorch:26.09-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenGL / EGL / GLX for the visualizer (Mesa, including the D3D12 driver used on WSL2).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libglx-mesa0 libgl1-mesa-dri && \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /workspace
