# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 2D images."""

import os
import numpy as np
import torch

from .. import custom_ops
from .. import misc
from . import conv2d_gradfix

#----------------------------------------------------------------------------

_plugin = None
_plugin_tried = False

def _init():
    global _plugin, _plugin_tried
    if not _plugin_tried:
        _plugin_tried = True
        _plugin = custom_ops.try_get_plugin(
            module_name='upfirdn2d_plugin',
            sources=['upfirdn2d.cpp', 'upfirdn2d.cu'],
            headers=['upfirdn2d.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler'],
        )
    return _plugin is not None

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = torch.outer(f, f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'`, `'native'` or `'cuda'` (default: `'cuda'`).
                     `'cuda'` uses the compiled CUDA kernels when they are available, and the
                     native implementation otherwise.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'native', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    if impl in ['cuda', 'native']:
        return _upfirdn2d_native(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------

_upfirdn2d_cuda_cache = dict()

def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    # Parse arguments.
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]

    # Forward op.
    class Upfirdn2dCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0) # Convert separable-1 into full-1x1.
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda

#----------------------------------------------------------------------------
# Native PyTorch implementation, used when the CUDA kernels are not available. Faster and much more memory efficient
# than the reference implementation: upsampling uses a transposed convolution instead of materializing the zero-stuffed
# image, downsampling a strided convolution that only computes the kept pixels, and separable filters are applied as two
# 1D passes. Like the CUDA op, the backward pass is the same op with transposed parameters, so gradients of any order are
# computed without going through PyTorch's (very slow) double backward of grouped convolutions.

def _upfirdn2d_pass(x, f, up, down, pad0, pad1, gain, dim):
    """One pass over the height (dim=2), the width (dim=3), or both (dim=None, 2D filter). `f` is a correlation kernel,
    `up`, `down`, `pad0` and `pad1` are (y, x) pairs."""
    batch_size, num_channels, in_height, in_width = x.shape
    w = f if dim is None else (f.unsqueeze(0) if dim == 3 else f.unsqueeze(1))
    kh, kw = w.shape
    (uy, ux), (dy, dx), (py0, px0), (py1, px1) = up, down, pad0, pad1

    # Every channel is filtered with the same kernel: fold the channels into the batch dimension.
    x = x.reshape(batch_size * num_channels, 1, in_height, in_width)
    w = (w * gain).to(x.dtype)[np.newaxis, np.newaxis]

    if uy == ux == 1:
        x = torch.nn.functional.pad(x, [px0, px1, py0, py1])
        x = torch.nn.functional.conv2d(x, w, stride=[dy, dx])
    else:
        # Compute the full upsampled output (of a transposed convolution), then shift it to the requested padding.
        out_h = in_height * uy + py0 + py1 - kh + 1
        out_w = in_width * ux + px0 + px1 - kw + 1
        x = _polyphase_conv_transpose2d(x, w[0, 0].flip([0, 1]), uy, ux)
        sy, sx = kh - 1 - py0, kw - 1 - px0
        x = torch.nn.functional.pad(x, [-sx, out_w + sx - x.shape[3], -sy, out_h + sy - x.shape[2]])
        if dy > 1 or dx > 1:
            x = x[:, :, ::dy, ::dx]
    return x.reshape(batch_size, num_channels, *x.shape[2:])

def _polyphase_conv_transpose2d(x, w, uy, ux):
    """`conv_transpose2d(x, w, stride=[uy, ux])` for a single channel `x` of shape [N, 1, H, W] and a kernel `w` of shape [kh, kw],
    zero padded at the end. Computed as uy * ux ordinary convolutions, one per output phase, which are interleaved: cuDNN's
    transposed convolutions are several times slower for these shapes."""
    kh, kw = w.shape
    ky, kx = -(-kh // uy), -(-kw // ux)
    w = torch.nn.functional.pad(w, [0, kx * ux - kw, 0, ky * uy - kh])
    phases = w.reshape(ky, uy, kx, ux).permute(1, 3, 0, 2).flip([2, 3]).reshape(uy * ux, 1, ky, kx)
    x = torch.nn.functional.pad(x, [kx - 1, kx - 1, ky - 1, ky - 1])
    z = torch.nn.functional.conv2d(x, phases)
    n, _, zh, zw = z.shape
    return z.reshape(n, uy, ux, zh, zw).permute(0, 3, 1, 4, 2).reshape(n, 1, zh * uy, zw * ux)

def _upfirdn2d_conv(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    if not flip_filter:
        f = f.flip(list(range(f.ndim))) # upfirdn2d convolves, conv2d computes a correlation
    if f.ndim == 2:
        return _upfirdn2d_pass(x, f, (upy, upx), (downy, downx), (pady0, padx0), (pady1, padx1), gain, dim=None)
    x = _upfirdn2d_pass(x, f, (1, upx), (1, downx), (0, padx0), (0, padx1), 1, dim=3)
    return _upfirdn2d_pass(x, f, (upy, 1), (downy, 1), (pady0, 0), (pady1, 0), gain, dim=2)

def _upfirdn2d_native_transpose(dy, f, x_shape, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Gradient of `upfirdn2d(x, f, up, down, padding, flip_filter, gain)` with respect to `x` of shape `x_shape`,
    for the output gradient `dy`. Differentiable, as it is again an upfirdn2d op."""
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    if f is not None and f.ndim == 1 and f.shape[0] == 1:
        f = f.square().unsqueeze(0)
    _, _, ih, iw = x_shape
    _, _, oh, ow = dy.shape
    fw, fh = _get_filter_size(f)
    p = [
        fw - padx0 - 1,
        iw * upx - ow * downx + padx0 - upx + 1,
        fh - pady0 - 1,
        ih * upy - oh * downy + pady0 - upy + 1,
    ]
    return _upfirdn2d_native(up=[downx, downy], down=[upx, upy], padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

_upfirdn2d_native_cache = dict()

def _upfirdn2d_native(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Native PyTorch implementation of `upfirdn2d()` with gradients of arbitrary order.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_native_cache:
        return _upfirdn2d_native_cache[key]

    class Upfirdn2dNative(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0) # Convert separable-1 into full-1x1.
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = _upfirdn2d_conv(x, f, up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1], flip_filter=flip_filter, gain=gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            dx = None
            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_native_transpose(dy, f, ctx.x_shape, up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1], flip_filter=flip_filter, gain=gain)
            assert not ctx.needs_input_grad[1]
            return dx, None

    _upfirdn2d_native_cache[key] = Upfirdn2dNative
    return Upfirdn2dNative

#----------------------------------------------------------------------------

def filter2d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy, impl=impl)

#----------------------------------------------------------------------------

def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------
