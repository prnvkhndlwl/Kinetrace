"""ROI-zoom tracking: engagement policy, restart coverage, and 4K accuracy.

- Wide constellation at 1080p (zoom gain < 2x): ROI must decline and results
  must equal the full-frame baseline.
- Tight constellation at 4K (zoom ~5x): ROI must engage, auto-restart when the
  target approaches the crop edge, keep frame coverage contiguous across
  restarts, and be at least as accurate as full-frame.
- A target near the PICTURE border (the crop clipped against it) must keep its
  crop for the whole run: a crop side on the border is not an edge to escape
  (I119: it used to restart every window, grow and drop ROI altogether).
Prerequisite: test4k.mp4 (regenerate with
  python make_test_video.py test4k.mp4 --frames 300 --size 3840x2160 --dots 4 --seed 7)."""
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])
from kinetrace.tracker import PointSpec, TrackingWorker
from kinetrace.video_source import FrameCache

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)


def run(vid, specs, n_frames, roi):
    K = len(specs)
    tracks = np.full((n_frames, K, 2), np.nan, np.float32)
    emitted = np.zeros(n_frames, bool)
    w = TrackingWorker(vid, 0, None, None, FrameCache(400 * 1024 * 1024),
                       n_frames, refine=True, specs=specs, roi=roi, autopause=False)

    def on_chunk(w0, tr, vi, cf, mem, fr):
        tracks[w0:w0 + tr.shape[0]] = tr
        emitted[w0:w0 + tr.shape[0]] = True

    w.chunk_ready.connect(on_chunk)
    w.error.connect(lambda m: (print(m), sys.exit("worker error")))
    w.run()
    return tracks, emitted


def with_crop_log(fn):
    log = []
    orig = TrackingWorker._compute_crop
    TrackingWorker._compute_crop = lambda self, specs, *a, **k: log.append(
        c := orig(self, specs, *a, **k)) or c
    try:
        out = fn()
    finally:
        TrackingWorker._compute_crop = orig
    return out, log


# ---- 1. 1080p wide constellation: ROI declines, matches baseline ----
VID1 = os.path.join(SCRATCH, "roi1080.mp4")
if not os.path.exists(VID1 + ".gt.npz"):
    subprocess.run([sys.executable, os.path.join(ROOT, r"make_test_video.py"), VID1,
                    "--frames", "300", "--size", "1920x1080", "--dots", "3",
                    "--seed", "5"], check=True)
GT1 = np.load(VID1 + ".gt.npz")["gt"]
specs1 = [PointSpec(i, GT1[0, i].astype(np.float32)) for i in range(3)]
(tr_on, em_on), crops = with_crop_log(lambda: run(VID1, specs1, 300, roi=True))
assert not any(c is not None for c in crops), \
    "marginal zoom (<2x) must not engage ROI"
tr_off, _ = run(VID1, specs1, 300, roi=False)
e_on = np.linalg.norm(tr_on - GT1[:300], axis=-1).mean()
e_off = np.linalg.norm(tr_off - GT1[:300], axis=-1).mean()
print(f"1080p wide: roi-flag on {e_on:.2f} px vs off {e_off:.2f} px (ROI declined)")
assert abs(e_on - e_off) < 0.5

# ---- 2. 4K tight constellation: engage, restart, stay contiguous, not worse ----
VID2 = os.path.join(ROOT, r"test4k.mp4")
if not os.path.exists(VID2 + ".gt.npz"):
    subprocess.run([sys.executable, os.path.join(ROOT, r"make_test_video.py"), VID2,
                    "--frames", "300", "--size", "3840x2160", "--dots", "4",
                    "--seed", "7"], check=True)
GT2 = np.load(VID2 + ".gt.npz")["gt"]
specs2 = [PointSpec(0, GT2[0, 0].astype(np.float32)),
          PointSpec(1, GT2[0, 0].astype(np.float32), kind="group", radius=27.0)]
(tr4, em4), crops4 = with_crop_log(lambda: run(VID2, specs2, 300, roi=True))
engaged = [c for c in crops4 if c is not None]
print(f"4K tight: {len(crops4)} segments, {len(engaged)} with ROI, "
      f"first crop {engaged[0] if engaged else None}")
assert engaged, "ROI must engage on a tight 4K constellation"
assert em4.all(), "frame coverage must be contiguous across ROI restarts"
assert np.isfinite(tr4[:, :, 0]).all(), "NaN gap across ROI restarts"
zoom = 3840 / max(engaged[0][2], engaged[0][3])
assert zoom >= 2.0, f"engaged crop zoom {zoom:.1f} below policy minimum"
tr4b, _ = run(VID2, specs2, 300, roi=False)
e4_on = np.linalg.norm(tr4 - GT2[:300, 0][:, None], axis=-1).mean(axis=0)
e4_off = np.linalg.norm(tr4b - GT2[:300, 0][:, None], axis=-1).mean(axis=0)
print(f"4K point err: roi {e4_on[0]:.2f} vs off {e4_off[0]:.2f} px | "
      f"group err: roi {e4_on[1]:.2f} vs off {e4_off[1]:.2f} px "
      f"({len(crops4) - 1} restarts)")
assert e4_on.mean() < e4_off.mean() + 0.5, "ROI must not be worse than full-frame"

# seed rows stay exact through the ROI path
assert np.allclose(tr4[0, 0], GT2[0, 0], atol=1e-3), "ROI seed row not exact"

# ---- 3. a target near the PICTURE border keeps its crop: no restart churn (I119) ----
VID3 = os.path.join(SCRATCH, "roi_border.mp4")
T3, BW, BH = 96, 1920, 1080          # 1080p: the 768x576 minimum crop is a 2.5x zoom
rng = np.random.default_rng(11)
bg3 = cv2.GaussianBlur(rng.integers(20, 110, (BH, BW, 3), dtype=np.uint8), (0, 0), 2.0)
t3 = np.arange(T3)
paths3 = {"top-left": np.stack([46 + 6 * np.sin(t3 / 9), 40 + 5 * np.cos(t3 / 11)], axis=1),
          "bottom-right": np.stack([BW - 46 + 6 * np.sin(t3 / 8), BH - 40 + 5 * np.cos(t3 / 10)], axis=1)}
paths3 = {k: np.round(p) for k, p in paths3.items()}   # drawn at integer centres = the truth
vw = cv2.VideoWriter(VID3, cv2.VideoWriter_fourcc(*"mp4v"), 30, (BW, BH))
for f in range(T3):
    fr = bg3.copy()
    for p in paths3.values():
        cv2.circle(fr, (int(p[f, 0]), int(p[f, 1])), 8, (230, 230, 255), -1, lineType=cv2.LINE_AA)
    vw.write(fr)
vw.release()
for corner, p in paths3.items():
    specs3 = [PointSpec(0, p[0].astype(np.float32))]
    (tr3, em3), crops3 = with_crop_log(lambda: run(VID3, specs3, T3, roi=True))
    e3 = np.linalg.norm(tr3[:, 0] - p, axis=1)
    print(f"picture-border target ({corner}): {len(crops3)} segment(s), crop {crops3[0]}, "
          f"err mean {np.nanmean(e3):.2f} px")
    assert crops3[0] is not None, "ROI must engage on a small target at 1080p"
    assert len(crops3) == 1, f"restart churn at the picture border: crops {crops3}"
    assert em3.all() and np.isfinite(tr3[:, 0, 0]).all()
    assert np.nanmean(e3) < 2.0, f"border target tracked badly ({np.nanmean(e3):.2f} px)"
os.remove(VID3)

print("ROI PASSED")
