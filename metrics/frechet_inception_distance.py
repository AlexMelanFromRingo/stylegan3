# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import torch
from . import metric_utils

#----------------------------------------------------------------------------

def frechet_distance(mu1, sigma1, mu2, sigma2):
    """||mu1 - mu2||^2 + tr(sigma1 + sigma2 - 2 sqrt(sigma1 sigma2)).
    The trace of the matrix square root is the sum of the square roots of the eigenvalues of
    sqrt(sigma1) sigma2 sqrt(sigma1), a symmetric matrix with the same eigenvalues as sigma1 sigma2.
    Same result as scipy.linalg.sqrtm, much faster, and independent of the scipy version
    (sqrtm's `disp` argument, used here before, was removed in scipy 1.18)."""
    mu1, sigma1, mu2, sigma2 = (torch.as_tensor(t, dtype=torch.float64) for t in (mu1, sigma1, mu2, sigma2))
    eigvals, eigvecs = torch.linalg.eigh(sigma1)
    sqrt_sigma1 = (eigvecs * eigvals.clamp(min=0).sqrt()) @ eigvecs.T
    trace_covmean = torch.linalg.eigvalsh(sqrt_sigma1 @ sigma2 @ sqrt_sigma1).clamp(min=0).sqrt().sum()
    return float((mu1 - mu2).square().sum() + sigma1.trace() + sigma2.trace() - 2 * trace_covmean)

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    return frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Frechet distance in the feature space of DINOv2 ViT-L/14 instead of Inception-v3 (FD_DINOv2, Stein et al. 2023),
# which agrees much better with human judgments of image quality. Values are not comparable to FID.

def compute_fd_dinov2(opts, max_real, num_gen):
    detector_url = 'torchhub://facebookresearch/dinov2/dinov2_vitl14'

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs={},
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs={},
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')
    return frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

#----------------------------------------------------------------------------
