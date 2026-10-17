# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Docker is optional, see README.md. The NGC image includes PyTorch, the CUDA toolkit and the compilers for the custom kernels.
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Keep the PyTorch build of the image: install everything else from requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN grep -v '^torch' /tmp/requirements.txt > /tmp/requirements-no-torch.txt && pip install -r /tmp/requirements-no-torch.txt

WORKDIR /workspace

RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]
