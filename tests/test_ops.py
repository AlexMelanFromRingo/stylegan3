# Tests of the native implementations of the custom ops and of the fixes for current library versions.
# They run on the CPU as well: python -m pytest tests

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dnnlib
from torch_utils import misc
from torch_utils.ops import filtered_lrelu, upfirdn2d
from metrics.frechet_inception_distance import frechet_distance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _random_upfirdn2d_config(rng):
    upx = rng.choice([1, 2, 4]); upy = upx if rng.random() < 0.8 else rng.choice([1, 2, 4])
    downx = rng.choice([1, 2]); downy = downx if rng.random() < 0.8 else rng.choice([1, 2])
    separable = rng.random() < 0.6
    k = rng.choice([1, 2, 3, 4, 6, 12]) if separable else rng.choice([2, 3, 4])
    shape = [k] if separable else [k, rng.choice([k, k + 1])]
    padding = [rng.randint(-3, k + 3) for _ in range(4)]
    return dict(up=[upx, upy], down=[downx, downy], padding=padding, flip_filter=rng.random() < 0.5, gain=rng.choice([1, 2.5])), shape

def test_upfirdn2d_native_matches_reference():
    rng = random.Random(0)
    checked = 0
    for _ in range(200):
        kwargs, shape = _random_upfirdn2d_config(rng)
        f = torch.randn(shape, dtype=torch.float64, device=device)
        x = torch.randn(2, 3, rng.randint(4, 16), rng.randint(4, 16), dtype=torch.float64, device=device)
        try:
            ref = upfirdn2d.upfirdn2d(x, f.float(), impl='ref', **kwargs)
        except AssertionError:
            continue # invalid configuration (output smaller than the filter)
        out = upfirdn2d.upfirdn2d(x, f.float(), impl='native', **kwargs)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-4)
        checked += 1
    assert checked > 100

@pytest.mark.parametrize('seed', range(4))
def test_upfirdn2d_native_gradients(seed):
    rng = random.Random(seed)
    while True:
        kwargs, shape = _random_upfirdn2d_config(rng)
        f = torch.randn(shape, device=device)
        x = torch.randn(1, 2, 6, 7, dtype=torch.float64, device=device, requires_grad=True)
        fn = lambda t: upfirdn2d.upfirdn2d(t, f, impl='native', **kwargs)
        try:
            fn(x)
            break
        except (AssertionError, RuntimeError):
            continue
    assert torch.autograd.gradcheck(fn, (x,))
    assert torch.autograd.gradgradcheck(fn, (x,))

@pytest.mark.parametrize('up,down,radial', [(2, 2, False), (4, 2, False), (2, 2, True), (1, 1, False)])
def test_filtered_lrelu_native(up, down, radial):
    torch.manual_seed(0)
    fu = upfirdn2d.setup_filter(torch.randn(6 * up).abs().tolist(), device=device) if up > 1 else None
    fd = (torch.rand(12, 12) if radial else torch.rand(12)).to(device) if down > 1 else None
    fd = fd / fd.sum() if fd is not None else None
    kwargs = dict(up=up, down=down, padding=[5, 4, 5, 4] if up > 1 else 0, gain=np.sqrt(2), slope=0.2, clamp=3)
    x = (torch.randn(2, 3, 12, 12, dtype=torch.float64, device=device) * 3).requires_grad_()
    b = torch.randn(3, dtype=torch.float64, device=device).requires_grad_()

    ref = filtered_lrelu.filtered_lrelu(x, fu, fd, b, impl='ref', **kwargs)
    out = filtered_lrelu.filtered_lrelu(x, fu, fd, b, impl='native', **kwargs)
    assert torch.allclose(out, ref, atol=1e-10)

    g = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, (x, b), g)
    out_grads = torch.autograd.grad(out, (x, b), g)
    assert all(torch.allclose(a, r, atol=1e-10) for a, r in zip(out_grads, ref_grads))

    xs = x.detach()[:1, :1].clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda t: filtered_lrelu.filtered_lrelu(t, fu, fd, None, impl='native', **kwargs), (xs,))

def test_frechet_distance_matches_sqrtm():
    scipy_linalg = pytest.importorskip('scipy.linalg')
    rng = np.random.default_rng(0)
    a = rng.standard_normal((400, 16)) @ rng.standard_normal((16, 16))
    b = a[:200] * 1.1 + 0.2 + rng.standard_normal((200, 16))
    mu1, mu2, s1, s2 = a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    expected = np.square(mu1 - mu2).sum() + np.trace(s1 + s2 - 2 * scipy_linalg.sqrtm(s1 @ s2).real)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(expected, rel=1e-8)
    assert frechet_distance(mu1, s1, mu1, s1) == pytest.approx(0, abs=1e-6)

def test_helpers():
    assert misc.torch_version_at_least(1, 11)
    assert not misc.torch_version_at_least(99, 0)
    assert dnnlib.util.strtobool('Yes') and not dnnlib.util.strtobool('off')
    with pytest.raises(ValueError):
        dnnlib.util.strtobool('maybe')
