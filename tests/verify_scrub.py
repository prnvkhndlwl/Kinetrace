"""Scrubbing efficiency: backward-step batching, idle prefetch, cache reset.

- Backward steps must stay frame-accurate AND get dramatically cheaper after
  the first one (the batch decode caches the run-up).
- SeekService must prefetch ahead after serving a request, so forward steps
  are cache hits; latest-wins collapsing must survive request spam.
- Shift+C must drop the cache, rebuild the decoder, and re-decode identically."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtWidgets import QApplication

# QApplication (not QCoreApplication): the cache-reset section drives the real
# MainWindow, and widgets need a GUI application object
app = QApplication([])
from kinetrace.video_source import (FrameCache, PREFETCH_AHEAD, SEEK_BACK_PREFETCH,
                                        SeekService, VideoSource)

VID = os.path.join(ROOT, r"test600.mp4")

# ---- backward batch: correctness + cache warm-up ----
cache = FrameCache(500 * 1024 * 1024)
src = VideoSource(VID, cache)
_ = src.get_frame(400)                       # play forward to 400
ref = {i: src.get_frame(i).copy() for i in (395, 396, 397)}  # fresh decodes forward
cache.clear()
src.seek(0)
_ = src.get_frame(400)
t0 = time.perf_counter()
f399 = src.get_frame(399)                    # backward step: batch decode lands 387..399
t_first = time.perf_counter() - t0
for i in range(398, 388, -1):                # ten more backward steps: all cache hits
    t0 = time.perf_counter()
    f = src.get_frame(i)
    dt = time.perf_counter() - t0
    assert f is not None
    assert dt < 0.02, f"backward step to {i} not a cache hit ({dt * 1000:.1f} ms)"
assert np.array_equal(src.get_frame(396), ref[396]), "backward batch broke frame accuracy"
print(f"backward steps OK: first {t_first * 1000:.1f} ms (batch), rest are cache hits")
# the batch really landed the run-up in the cache
assert all(cache.get(i) is not None for i in range(399 - SEEK_BACK_PREFETCH, 400))
src.close()

# ---- SeekService prefetch + latest-wins ----
cache2 = FrameCache(500 * 1024 * 1024)
served = []
svc = SeekService(VID, cache2)
svc.frame_ready.connect(lambda i, f: served.append(i))
svc.start()
svc.request(100)
t0 = time.time()
while not served and time.time() - t0 < 10:
    app.processEvents()
    time.sleep(0.005)
assert served == [100], served
time.sleep(0.4)                               # let the idle prefetch run
missing = [j for j in range(101, 101 + PREFETCH_AHEAD) if cache2.get(j) is None]
assert not missing, f"prefetch left gaps: {missing}"
print(f"prefetch OK: frames 101..{100 + PREFETCH_AHEAD} warmed after serving 100")

# latest-wins: spam requests, only the newest needs to be served last
served.clear()
for i in range(200, 260, 3):
    svc.request(i)
t0 = time.time()
while (not served or served[-1] != 257) and time.time() - t0 < 15:
    app.processEvents()
    time.sleep(0.005)
assert served and served[-1] == 257, f"latest request must win: {served[-5:]}"
assert len(served) < 21, f"request spam must collapse, served {len(served)} frames"
print(f"latest-wins OK ({len(served)} of 20 spammed requests decoded)")
svc.stop()

svc.stop()

# ---- a failed read is not the end of the video; an exception never kills the thread (I40) ----
from kinetrace import video_source as _vs       # noqa: E402


class _FaultySource(VideoSource):
    """get_frame fails as told for given frames (consumed in order); opens can fail too."""
    faults: dict = {}
    fail_open = 0

    def __init__(self, *a, **k):
        if _FaultySource.fail_open > 0:
            _FaultySource.fail_open -= 1
            raise ValueError("simulated: the share is not reachable")
        super().__init__(*a, **k)

    def get_frame(self, idx):
        acts = _FaultySource.faults.get(idx)
        if acts:
            act = acts.pop(0)
            if act == "raise":
                raise MemoryError("simulated 4K allocation failure")
            return None                     # "none": the decoder gave no picture
        return super().get_frame(idx)


def _run_service(n_frames, faults, reqs, fail_open=0):
    _FaultySource.faults = {k: list(v) for k, v in faults.items()}
    _FaultySource.fail_open = fail_open
    sig = []
    real = _vs.VideoSource
    _vs.VideoSource = _FaultySource
    try:
        s = SeekService(VID, FrameCache(200 * 1024 * 1024), n_frames=n_frames)
        s.frame_ready.connect(lambda i, f: sig.append(("frame", i)))
        s.eof_truncated.connect(lambda i: sig.append(("eof", i)))
        s.decode_failed.connect(lambda i, m: sig.append(("failed", i, m)))
        s.start()
        for idx in reqs:
            s.request(idx)
            t0 = time.time()
            while not any(e[1] == idx for e in sig) and time.time() - t0 < 10:
                app.processEvents()
                time.sleep(0.005)
        t0 = time.time()
        while time.time() - t0 < 0.2:
            app.processEvents()
            time.sleep(0.005)
        alive = s.isRunning()
        s.stop()
    finally:
        _vs.VideoSource = real
    return [e[:2] for e in sig], sig, alive


ev, raw, alive = _run_service(600, {50: ["raise"]}, [50, 100])
assert ev == [("frame", 50), ("frame", 100)] and alive, (raw, alive)
ev, raw, alive = _run_service(600, {300: ["none"]}, [300])
assert ev == [("frame", 300)], f"a transient read failure shortened the video: {raw}"
ev, raw, alive = _run_service(600, {300: ["none", "none"]}, [300, 100])
assert ev == [("failed", 300), ("frame", 100)] and alive, raw
assert raw[0][2], "decode_failed must say why"
ev, raw, alive = _run_service(None, {300: ["none", "none"]}, [300])
assert ev == [("eof", 300)], f"without a verified count a persistent failure is still EOF: {raw}"
ev, raw, alive = _run_service(None, {400: ["raise", "raise"]}, [400, 100])
assert ev == [("failed", 400), ("frame", 100)] and "MemoryError" in raw[0][2] and alive, raw
ev, raw, alive = _run_service(600, {}, [100], fail_open=1)
assert ev == [("failed", -1), ("frame", 100)] and alive, raw
print("SeekService faults OK: transient failure retried on a fresh capture, a frame that exists but "
      "will not decode is reported (not EOF), an exception or a failed open keeps the thread serving")

# ---- Shift+C failsafe: cache cleared, decoder rebuilt, frame re-decoded ----
from PySide6.QtCore import QEvent, Qt          # noqa: E402
from PySide6.QtGui import QKeyEvent            # noqa: E402
from PySide6.QtWidgets import QMessageBox      # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)

gui = app
from kinetrace.app import MainWindow, READY  # noqa: E402

win = MainWindow()
win.show()
win._open_video(VID)
t0 = time.time()
while win.state != READY and time.time() - t0 < 20:
    gui.processEvents()
    time.sleep(0.005)
assert win.state == READY, "video did not open"

win._goto(120)
t0 = time.time()
# wait for frame 120 ITSELF: opening the video already put frame 0 in the
# cache, so "any frame cached" is not the same as "the seek has landed"
while win.cache.get(120) is None and time.time() - t0 < 20:
    gui.processEvents()
    time.sleep(0.005)
n_before, bytes_before = win.cache.stats()
assert n_before > 0, "expected frames in the cache after seeking"
ref = win.cache.get(120)
assert ref is not None, "frame 120 was not decoded within 20 s"
ref = ref.copy()
old_service = win.seek

win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.ShiftModifier))
gui.processEvents()
assert win.seek is not old_service, "Shift+C must rebuild the scrubbing decoder"

# the frame must come back byte-identical from a genuinely fresh decode
t0 = time.time()
while win.cache.get(120) is None and time.time() - t0 < 15:
    gui.processEvents()
    time.sleep(0.005)
fresh = win.cache.get(120)
assert fresh is not None, "current frame was not re-decoded after the reset"
assert np.array_equal(fresh, ref), "re-decoded frame differs from the original"
assert win.session is not None and win.current == 120, "reset must not move the playhead"
print(f"Shift+C failsafe OK: dropped {n_before} frames "
      f"({bytes_before / 1024**2:.0f} MB), re-decoded frame 120 identically")

win._dev_probe.wait(180_000)
win.close()
gui.processEvents()
if os.path.exists(VID + ".cotracker.npz"):
    os.remove(VID + ".cotracker.npz")

print("SCRUB PASSED")
