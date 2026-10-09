"""AllTracker as the worker's point model, headless against ground truth:
full run, tails, pause, confidence sanity on the dot video; LK-refined 4K
accuracy; ROI restart contiguity. GPU, ~4 min (regenerates test4k on demand)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])

from kinetrace.alltracker_backend import available, is_cached  # noqa: E402
from kinetrace.tracker import PointSpec, TrackingWorker  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

assert available(), "models/alltracker (vendored repo) is missing"
print("checkpoint cached:", is_cached())

VID = os.path.join(ROOT, "test600.mp4")
if not os.path.exists(VID + ".gt.npz"):
    subprocess.run([sys.executable, os.path.join(ROOT, "make_test_video.py"), VID, "--seed", "0"], check=True)
GT = np.load(VID + ".gt.npz")["gt"]
T = GT.shape[0]


def run_segment(video, start, seeds, n_frames, refine=True, roi=True, pause_after=None, backend="alltracker"):
    K = len(seeds)
    tracks = np.full((n_frames, K, 2), np.nan, np.float32)
    conf = np.zeros((n_frames, K), np.float32)
    ev = {"emits": 0, "finished": None, "error": None, "autopaused": None, "rows": []}
    specs = [PointSpec(i, np.asarray(s, np.float32).copy()) for i, s in enumerate(seeds)]
    w = TrackingWorker(video, start, None, None, FrameCache(256 * 1024 ** 2), n_frames, refine=refine,
                       specs=specs, roi=roi, autopause=False, point_backend=backend)

    def on_chunk(w0, tr, vi, cf, members, frames):
        L = tr.shape[0]
        tracks[w0:w0 + L] = tr
        conf[w0:w0 + L] = cf
        ev["emits"] += 1
        ev["rows"].append((w0, L))
        assert np.isfinite(cf).all() and (cf >= 0).all() and (cf <= 1).all()
        if pause_after and ev["emits"] >= pause_after:
            w.request_pause()

    w.chunk_ready.connect(on_chunk)
    w.finished_ok.connect(lambda last, p: ev.update(finished=(last, p)))
    w.autopaused.connect(lambda f, pid: ev.update(autopaused=(f, pid)))
    w.error.connect(lambda m: ev.update(error=m))
    w.run()
    if ev["error"]:
        print(ev["error"])
        sys.exit("worker error")
    return tracks, conf, ev


def report(name, tracks, gt, start, end):
    seg = tracks[start:end]
    e = np.linalg.norm(seg - gt[start:end], axis=2)
    ok = np.isfinite(e).all(axis=1)
    print(f"{name}: mean {np.nanmean(e):.2f} px, max {np.nanmax(e):.2f} px, frames {int(ok.sum())}/{end - start}")
    return float(np.nanmean(e)), float(np.nanmax(e)), int(ok.sum())


# ---- 7. every picture size of the Mac install audit (P0-1: the Apple GPU failed at 640x480) ----
# the worker's own working size (alltracker_max_dim) decides what AllTracker sees; run it with
# KINETRACE_DEVICE=cpu as well to compare a GPU's numbers with the CPU's
def check_sizes():
    from kinetrace.device import pick_device  # noqa: E402
    print("device:", pick_device()[1])
    OUT = os.path.join(ROOT, "tests", "out")
    os.makedirs(OUT, exist_ok=True)
    for (sw, sh) in ((640, 480), (1280, 720), (1920, 1080), (2704, 1520)):
        vid = os.path.join(OUT, f"at_size_{sw}x{sh}.mp4")
        if not os.path.exists(vid + ".gt.npz"):
            subprocess.run([sys.executable, os.path.join(ROOT, "make_test_video.py"), vid, "--frames", "40",
                            "--size", f"{sw}x{sh}", "--dots", "4", "--seed", "3"], check=True)
        gts = np.load(vid + ".gt.npz")["gt"]
        trs, _, evs = run_segment(vid, 0, gts[0], 40, refine=True, roi=False)
        assert evs["finished"] == (39, False) and np.isfinite(trs[:, :, 0]).all(), f"{sw}x{sh} coverage"
        ms, _, _ = report(f"size {sw}x{sh} (40f, refined)", trs, gts, 0, 40)
        assert ms < 2.5, f"accuracy at {sw}x{sh}"


if "--sizes" in sys.argv:          # only section 7 (e.g. KINETRACE_DEVICE=cpu for the comparison)
    check_sizes()
    print("ALL ALLTRACKER SIZE CHECKS PASSED")
    sys.exit(0)


# ---- 1. full run from frame 0 ----
tr, cf, ev = run_segment(VID, 0, GT[0], T, refine=False)
assert ev["finished"] == (T - 1, False), ev["finished"]
assert np.isfinite(tr[:, :, 0]).all(), "coverage gap"
assert np.allclose(tr[0], GT[0], atol=1e-4), "seed row must be exact"
m, mx, n = report("alltracker full 600", tr, GT, 0, T)
assert m < 2.5 and mx < 12, "accuracy on the dot video"
frac = (cf[1:] > 0.25).mean()
print(f"  confidence above the pause threshold: {frac:.1%}, median {np.median(cf[1:]):.3f}")
assert frac > 0.95

# ---- 2. resume mid-video ----
tr2, _, ev2 = run_segment(VID, 250, GT[250], T)
assert ev2["finished"][0] == T - 1 and np.isfinite(tr2[250:, :, 0]).all() and np.isnan(tr2[:250]).all()
report("resume@250", tr2, GT, 250, T)

# ---- 3. short tails near EOF (the flush path: 10, 5, 1, 16, 8 frames) ----
for start in (590, 595, 599, 584, 592):
    trS, _, evS = run_segment(VID, start, GT[start], T)
    assert np.isfinite(trS[start:T, :, 0]).all(), f"gap in tail start={start}"
    assert evS["finished"][0] == T - 1, evS["finished"]
    mS, _, _ = report(f"tail@{start} ({T - start}f)", trS, GT, start, T)
    assert mS < 8.0

# ---- 4. pause path ----
trP, _, evP = run_segment(VID, 0, GT[0], T, pause_after=3)
last, paused = evP["finished"]
assert paused and 16 <= last < T - 1 and np.isfinite(trP[:last + 1, :, 0]).all()
print(f"pause OK: stopped at frame {last} after 3 emits")

# ---- 5. 4K with LK refinement + ROI (restart contiguity) ----
VID4 = os.path.join(ROOT, "test4k.mp4")
if not os.path.exists(VID4 + ".gt.npz"):
    subprocess.run([sys.executable, os.path.join(ROOT, "make_test_video.py"), VID4, "--frames", "300",
                    "--size", "3840x2160", "--dots", "4", "--seed", "7"], check=True)
GT4 = np.load(VID4 + ".gt.npz")["gt"]
tr4, cf4, ev4 = run_segment(VID4, 0, GT4[0], 300, refine=True, roi=True)
assert ev4["finished"] == (299, False) and np.isfinite(tr4[:, :, 0]).all(), "4K coverage"
m4, mx4, _ = report("alltracker 4K refined+ROI", tr4, GT4, 0, 300)
assert m4 < 2.0, "4K refined accuracy"
tr4r, _, _ = run_segment(VID4, 0, GT4[0], 300, refine=False, roi=False)
m4r, _, _ = report("alltracker 4K raw full-frame", tr4r, GT4, 0, 300)
print(f"refinement + ROI delta at 4K: {m4r - m4:+.2f} px (positive = they help)")

# ---- 6. CoTracker3 remains the same worker path (sanity) ----
trc, _, evc = run_segment(VID, 0, GT[0], 120, refine=False, backend="cotracker3")
mc, _, _ = report("cotracker3 (backend switch sanity, 120f)", trc, GT, 0, 120)
assert mc < 3.0
check_sizes()
print("ALL ALLTRACKER CHECKS PASSED")
