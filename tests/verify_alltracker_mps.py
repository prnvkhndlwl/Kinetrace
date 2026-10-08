"""AllTracker on an Apple GPU (Mac install audit P0-1), checked on any machine.

Metal's adaptive pool (F.interpolate mode='area') needs every input side to be a
multiple of its output side; CorrBlock's fifth halving of the 1/8 feature map
broke that, so the first Track press failed on every Mac. Metal's rule is
emulated here on the CPU:
  1. the unpatched path raises at 640x480 (this check fails on the old code);
  2. the patched path (alltracker_backend.AreaPoolOffDevice) builds the
     correlation pyramid at every working size the app uses for 640x480,
     1280x720, 1920x1080, 2704x1520 and 3840x2160, bit-identical to plain torch;
  3. a real AllTracker window (checkpoint permitting) gives bit-identical tracks
     with the patch, and the unpatched stream raises;
  4. the patch is installed for 'mps' only (CUDA / CPU keep the vendored call).
CPU only, ~20 s. The real Apple-GPU run is verify_alltracker.py section 7."""
import os
import sys

os.environ["KINETRACE_DEVICE"] = "cpu"     # the emulation runs on the CPU on every machine
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kinetrace import alltracker_backend as at  # noqa: E402

assert at.available(), "models/alltracker (vendored repo) is missing: run install.py --alltracker-only"
if str(at.REPO_DIR) not in sys.path:
    sys.path.insert(0, str(at.REPO_DIR))
import nets.blocks as blocks  # noqa: E402
from nets.blocks import CorrBlock, InputPadder  # noqa: E402

PLAIN_F = blocks.F
METAL_MSG = "Adaptive pool MPS: input sizes must be divisible by output sizes."


def metal_interpolate(input, size=None, scale_factor=None, mode="nearest", *args, **kw):
    """torch's interpolate with Metal's restriction on area pooling (the error the audit saw)."""
    if mode == "area" and not at._area_divisible(input.shape[2:], size, scale_factor):
        raise RuntimeError(METAL_MSG + " Non-divisible input sizes are not implemented on MPS device yet.")
    return F.interpolate(input, size, scale_factor, mode, *args, **kw)


class MetalF:
    """torch.nn.functional as the old code saw it on a Mac."""
    def __getattr__(self, name):
        return getattr(F, name)

    interpolate = staticmethod(metal_interpolate)


def patched():
    return at.AreaPoolOffDevice(F, device_type="cpu", native=metal_interpolate)


def working_size(w, h, max_dim):
    """tracker.py's rule: longest side <= max_dim, then InputPadder's multiple of 64."""
    s = min(1.0, max_dim / max(w, h))
    ww, wh = max(2, round(w * s)), max(2, round(h * s))
    p = InputPadder((1, 3, wh, ww))
    hp = wh + p._pad[2] + p._pad[3]
    wp = ww + p._pad[0] + p._pad[1]
    return ww, wh, hp // 8, wp // 8


def pyramid(f_module, h8, w8, seed=0):
    g = torch.Generator().manual_seed(seed)
    fmap1 = torch.randn(1, 8, 2, 2, generator=g)
    fmap2 = torch.randn(1, 8, h8, w8, generator=g)
    blocks.F = f_module
    try:
        return CorrBlock(fmap1, fmap2, 5, 4).corr_pyramid
    finally:
        blocks.F = PLAIN_F


# ---- 1. the old path fails the way the Mac did ----
_, _, h8, w8 = working_size(640, 480, 1024)
try:
    pyramid(MetalF(), h8, w8)
except RuntimeError as e:
    assert METAL_MSG in str(e), e
    print(f"1. unpatched CorrBlock at 640x480 (feature map {w8}x{h8}) raises as on the Mac: OK")
else:
    sys.exit("1. FAILED: the emulated Metal pool did not raise at 640x480 - the emulation is wrong")

# ---- 2. patched pyramid at every working size, bit-identical ----
n_cpu = 0
for (w, h) in ((640, 480), (1280, 720), (1920, 1080), (2704, 1520), (3840, 2160)):
    for max_dim in (1024, 896, 768, 640, 512):
        ww, wh, h8, w8 = working_size(w, h, max_dim)
        ref = pyramid(PLAIN_F, h8, w8)
        shim = patched()
        calls = []
        shim._F = type("Spy", (), {"__getattr__": lambda s, n: getattr(F, n),
                                   "interpolate": staticmethod(lambda *a, **k: calls.append(1) or
                                                               F.interpolate(*a, **k))})()
        got = pyramid(shim, h8, w8)
        assert len(ref) == len(got) == 5
        for a, b in zip(ref, got):
            assert np.array_equal(a.numpy(), b.numpy()), f"pyramid differs at {w}x{h} / {max_dim}"
        n_cpu += len(calls)
    print(f"2. {w}x{h}: patched pyramid at working sizes 1024..512 equals plain torch: OK")
assert n_cpu > 0, "no size needed the CPU path: the test no longer covers the bug"
print(f"   ({n_cpu} non-divisible poolings took the CPU path)")

# ---- 3. a real AllTracker window, patched vs unpatched ----
if at.is_cached():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (260, 360, 3), np.uint8)
    frames = [np.ascontiguousarray(base[10 - i // 2:250 - i // 2, 10 + i:330 + i]) for i in range(17)]  # 320x240
    xy = np.array([[60.0, 50.0], [200.0, 120.0], [300.0, 220.0]], np.float32)

    def run(f_module):
        blocks.F = f_module
        try:
            s = at.AllTrackerStream()
            assert str(s.device) == "cpu"
            s.start(frames[0], xy)
            out = None
            for fr in frames[1:]:
                out = s.push(fr) or out
            return out
        finally:
            blocks.F = PLAIN_F

    ref = run(PLAIN_F)
    got = run(patched())
    assert ref is not None and got is not None
    for a, b in zip(ref[1:], got[1:]):
        assert np.array_equal(a, b), "tracks differ with the patch"
    try:
        run(MetalF())
    except RuntimeError as e:
        assert METAL_MSG in str(e), e
    else:
        sys.exit("3. FAILED: the unpatched stream did not raise under Metal's rule")
    print("3. AllTracker window at 320x240: patched tracks bit-identical, unpatched raises: OK")
else:
    print("3. skipped: the AllTracker checkpoint is not downloaded (no download in a CPU suite)")

# ---- 4. installed for Apple's GPU only ----
for dev in ("cuda", "cpu"):
    at._install_area_pool_fix(dev)
    assert blocks.F is PLAIN_F, f"the patch must not touch {dev}"
at._install_area_pool_fix("mps")
assert isinstance(blocks.F, at.AreaPoolOffDevice) and blocks.F._device_type == "mps"
at._install_area_pool_fix("mps")
assert blocks.F._F is PLAIN_F, "installed twice"
blocks.F = PLAIN_F
print("4. patch installed for mps only, once: OK")
print("ALL ALLTRACKER MPS CHECKS PASSED")
