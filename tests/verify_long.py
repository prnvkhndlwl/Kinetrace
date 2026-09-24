"""5000-frame run: bounded RAM, accuracy at 1080p (refinement auto-on), throughput.
Plus unicode-path smoke test."""
import os
import ctypes
import ctypes.wintypes as wt
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])
from cotracker_app.tracker import TrackingWorker
from cotracker_app.video_source import FrameCache, probe_video, VideoSource


def rss_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC(cb=ctypes.sizeof(PMC))
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wt.HANDLE  # 64-bit pseudo-handle, not c_int
    fn = getattr(k32, "K32GetProcessMemoryInfo", None) \
        or ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
    fn.restype = wt.BOOL
    ok = fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    assert ok and pmc.WorkingSetSize > 0, "RSS probe failed — memory guard inactive"
    return pmc.WorkingSetSize / 1024**2


# ---- unicode path ----
uni = Path(os.path.join(ROOT, r"тест_видео_日本語.mp4"))
shutil.copy(os.path.join(ROOT, r"test600.mp4"), uni)
try:
    info = probe_video(str(uni))
    src = VideoSource(str(uni), FrameCache(50 * 1024**2))
    ok = src.get_frame(100) is not None
    src.close()
    print(f"unicode path: probe {info.n_frames} frames, frame100 decode={'OK' if ok else 'FAIL'}")
    assert ok and info.n_frames == 600
finally:
    uni.unlink(missing_ok=True)

# ---- 5000-frame tracked run with RSS sampling ----
VID = os.path.join(ROOT, r"test5000.mp4")
GT = np.load(VID + ".gt.npz")["gt"]
T, D = GT.shape[:2]
tracks = np.full((T, D, 2), np.nan, np.float32)
rss_samples = []

cache = FrameCache(1 * 1024**3)  # 1 GB budget for this test
w = TrackingWorker(VID, 0, GT[0].astype(np.float32), list(range(D)), cache, T, refine=True)
w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: (
    tracks.__setitem__(slice(w0, w0 + len(tr)), tr),
    rss_samples.append(rss_mb()) if (w0 // 8) % 25 == 0 else None))
w.error.connect(lambda m: (print(m), sys.exit(1)))
t0 = time.time()
w.run()
dt = time.time() - t0

err = np.linalg.norm(tracks - GT, axis=-1)
print(f"5000f 1080p: mean {np.nanmean(err):.2f} px, p95 {np.nanpercentile(err, 95):.2f} px, "
      f"{T / dt:.1f} fps ({dt:.0f}s total)")
print(f"RSS first/mid/last: {rss_samples[0]:.0f} / {rss_samples[len(rss_samples)//2]:.0f} / "
      f"{rss_samples[-1]:.0f} MB  (n={len(rss_samples)})")
assert np.isfinite(tracks).all()
assert np.nanmean(err) < 6.0
growth = rss_samples[-1] - rss_samples[len(rss_samples) // 2]
assert growth < 500, f"RSS still growing late in the run (+{growth:.0f} MB after midpoint)"
print("LONG RUN PASSED")
