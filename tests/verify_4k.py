"""4K precision: raw CoTracker (through 1280 working res) vs LK-refined, vs ground truth.
Also reports throughput and worker RSS to sanity-check memory behavior."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])
from kinetrace.tracker import TrackingWorker
from kinetrace.video_source import FrameCache

VID = os.path.join(ROOT, r"test4k.mp4")
GT = np.load(VID + ".gt.npz")["gt"]
T = len(GT)


def run(refine):
    tracks = np.full((T, 4, 2), np.nan, np.float32)
    err_holder = {}
    w = TrackingWorker(VID, 0, GT[0].astype(np.float32), [0, 1, 2, 3],
                       FrameCache(1 * 1024**3), T, refine=refine)
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr:
                          tracks.__setitem__(slice(w0, w0 + len(tr)), tr))
    w.error.connect(lambda m: err_holder.update(e=m))
    t0 = time.time()
    w.run()
    dt = time.time() - t0
    if "e" in err_holder:
        print(err_holder["e"]); sys.exit(1)
    err = np.linalg.norm(tracks - GT, axis=-1)
    return err, T / dt


err_raw, fps_raw = run(refine=False)
err_ref, fps_ref = run(refine=True)
print(f"4K raw    : mean {err_raw.mean():.2f} px, p95 {np.percentile(err_raw, 95):.2f}, "
      f"max {err_raw.max():.2f}  @ {fps_raw:.1f} fps")
print(f"4K refined: mean {err_ref.mean():.2f} px, p95 {np.percentile(err_ref, 95):.2f}, "
      f"max {err_ref.max():.2f}  @ {fps_ref:.1f} fps")
assert err_ref.mean() < err_raw.mean(), "refinement must improve 4K accuracy"
assert err_ref.mean() < 2.0, f"refined mean {err_ref.mean():.2f} px misses the <2px target"
print("4K PRECISION PASSED")
