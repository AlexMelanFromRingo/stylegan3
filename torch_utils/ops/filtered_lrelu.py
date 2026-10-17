# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch
import warnings

from .. import custom_ops
from .. import misc
from . import upfirdn2d
from . import bias_act

#----------------------------------------------------------------------------

_plugin = None
_plugin_tried = False

def _init():
    global _plugin, _plugin_tried
    if not _plugin_tried:
        _plugin_tried = True
        _plugin = custom_ops.try_get_plugin(
            module_name='filtered_lrelu_plugin',
            sources=['filtered_lrelu.cpp', 'filtered_lrelu_wr.cu', 'filtered_lrelu_rd.cu', 'filtered_lrelu_ns.cu'],
            headers=['filtered_lrelu.h', 'filtered_lrelu.cu'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler'],
        )
    return _plugin is not None

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0] # width, height

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1

#----------------------------------------------------------------------------

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    r"""Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
       it so that the footprint of all output pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'`, `'native'` or `'cuda'` (default: `'cuda'`).
                     `'cuda'` uses the compiled CUDA kernels when they are available, and the
                     native implementation otherwise.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'native', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    if impl in ['cuda', 'native']:
        return _filtered_lrelu_native(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b)
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)

#----------------------------------------------------------------------------

@misc.profiled_function
def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Slow and memory-inefficient reference implementation of `filtered_lrelu()` using
    existing `upfirdn2n()` and `bias_act()` ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

    # Compute using existing ops.
    x = bias_act.bias_act(x=x, b=b) # Apply bias.
    x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter) # Upsample.
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp) # Bias, leaky ReLU, clamp.
    x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter) # Downsample.

    # Check output shape & dtype.
    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x

#----------------------------------------------------------------------------

# Native PyTorch implementation, used when the CUDA kernels are not available. It follows the design of the CUDA op:
# the upsampled activations are not kept for the backward pass, only 2 bits per value (negative, clamped), packed
# into bytes, and the backward pass is made of differentiable upfirdn2d ops, so gradients of any order are supported.

_code_shifts = dict()

def _pack_codes(neg, clamped):
    codes = neg.to(torch.uint8)
    if clamped is not None:
        codes = codes | (clamped.to(torch.uint8) << 1)
    width = codes.shape[-1]
    if width % 4:
        codes = torch.nn.functional.pad(codes, [0, 4 - width % 4])
    codes = codes.reshape(*codes.shape[:-1], -1, 4)
    return codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)

def _unpack_codes(packed, width):
    key = packed.device
    if key not in _code_shifts:
        _code_shifts[key] = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    codes = (packed.unsqueeze(-1) >> _code_shifts[key]) & 3
    return codes.flatten(-2)[..., :width]

_filtered_lrelu_native_cache = dict()

def _filtered_lrelu_native(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Native PyTorch implementation of `filtered_lrelu()`.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp) if clamp is not None else None

    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_native_cache:
        return _filtered_lrelu_native_cache[key]

    up_kwargs = dict(up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
    down_kwargs = dict(down=down, flip_filter=flip_filter)
    is_identity = lambda f, scale, pad: (f is None or (f.ndim == 2 and f.numel() == 1 and float(f) == 1)) and scale == 1 and pad == [0, 0, 0, 0]

    class FilteredLReluNative(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fu, fd, b): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if b is not None:
                x = x + b.to(x.dtype).reshape(1, -1, 1, 1)

            # Upsample. Skipped for the identity, e.g. in ToRGB layers.
            y = x if is_identity(fu, up, [px0, px1, py0, py1]) else upfirdn2d._upfirdn2d_conv(x, fu, **up_kwargs)

            # Gain, leaky ReLU, clamp.
            y = y * gain
            neg = None
            if slope != 1:
                neg = y < 0
                y = torch.where(neg, y * slope, y)
            clamped = None
            if clamp is not None:
                clamped = y.abs() > clamp
                y = y.clamp(-clamp, clamp)

            # Downsample.
            out = y if is_identity(fd, down, [0, 0, 0, 0]) else upfirdn2d._upfirdn2d_conv(y, fd, **down_kwargs)

            codes = None
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                codes = _pack_codes(neg if neg is not None else torch.zeros_like(y, dtype=torch.bool), clamped)
            ctx.save_for_backward(fu, fd, codes)
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.has_bias = b is not None
            return out

        @staticmethod
        def backward(ctx, dout): # pylint: disable=arguments-differ
            fu, fd, codes = ctx.saved_tensors
            dx = db = None

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                # Gradient of the downsampling.
                dy = dout if is_identity(fd, down, [0, 0, 0, 0]) else upfirdn2d._upfirdn2d_native_transpose(dout, fd, ctx.y_shape, **down_kwargs)

                # Gradient of the activation: gain, times slope where negative, zero where clamped.
                codes = _unpack_codes(codes, ctx.y_shape[-1])
                scale = torch.where((codes & 1).bool(), gain * slope, gain).to(dy.dtype)
                if clamp is not None:
                    scale = scale.masked_fill((codes & 2).bool(), 0)
                dy = dy * scale

                # Gradient of the upsampling.
                dx = dy if is_identity(fu, up, [px0, px1, py0, py1]) else upfirdn2d._upfirdn2d_native_transpose(dy, fu, ctx.x_shape, **up_kwargs)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, None, None, db

    _filtered_lrelu_native_cache[key] = FilteredLReluNative
    return FilteredLReluNative

_filtered_lrelu_cuda_cache = dict()

def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    # Lookup from cache.
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]

    # Forward op.
    class FilteredLReluCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels (faster than separable).
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale factor is 1.
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            # Missing sign input tensor.
            if si is None:
                si = torch.empty([0])

            # Missing bias tensor.
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)

            # Construct internal sign tensor only if gradients are needed.
            write_signs = (si.numel() == 0) and (x.requires_grad or b.requires_grad)

            # Warn if input storage strides are not in decreasing order due to e.g. channels-last layout.
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn("low-performance memory layout detected in filtered_lrelu input", RuntimeWarning)

            # Call C++/Cuda plugin if datatype is supported.
            if x.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                    warnings.warn("filtered_lrelu called with non-default cuda stream but concurrent execution is not supported", RuntimeWarning)
                y, so, return_code = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1

            # No Cuda kernel found? Fall back to generic implementation. Still more memory efficient than the reference implementation because
            # only the bit-packed sign tensor is retained for gradient computation.
            if return_code < 0:
                warnings.warn("filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback", RuntimeWarning)

                y = x.add(b.unsqueeze(-1).unsqueeze(-1)) # Add bias.
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter) # Upsample.
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain, slope, clamp, write_signs) # Activation function and sign handling. Modifies y in-place.
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter) # Downsample.

            # Prepare for gradient computation.
            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx  = None # 0
            dfu = None; assert not ctx.needs_input_grad[1]
            dfd = None; assert not ctx.needs_input_grad[2]
            db  = None # 3
            dsi = None; assert not ctx.needs_input_grad[4]
            dsx = None; assert not ctx.needs_input_grad[5]
            dsy = None; assert not ctx.needs_input_grad[6]

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[0] - 1) + (fd.shape[0] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up ** 2) / (down ** 2)
                ff = (not flip_filter)
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0]  - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda

#----------------------------------------------------------------------------
