"""Scrubbing efficiency: backward-step batching, idle prefetch, cache reset.

- Backward steps must stay frame-accurate AND get dramatically cheaper after
  the first one (the batch decode caches the run-up).
- SeekService must prefetch ahead after serving a request, so forward steps
  are cache hits; latest-wins collapsing must survive request spam.
- Shift+C must drop the cache, rebuild the decoder, and re-decode identically.
- (I306) Two 4K cameras in Sync all, 15 F and 15 B presses: the other camera's
  steps come from its window (no re-decode), its cached frames equal the file
  read forward, one frame budget is split between the decoding cameras.
- (G204) A typed jump to an uncached frame veils the nearest-frame preview."""
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
from _clean import forget_recovery  # noqa: E402
from kinetrace.video_source import (FrameCache, SCRUB_WINDOW, SEEK_BACK_PREFETCH,
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
# the idle fill: a first move leans forward, below keeps one keyframe interval or a third of the window
below = max(SEEK_BACK_PREFETCH, SCRUB_WINDOW // 3)
win_frames = list(range(100 - below, 100)) + list(range(101, 101 + SCRUB_WINDOW - below))
t0 = time.time()
while any(cache2.get(j) is None for j in win_frames) and time.time() - t0 < 10:
    time.sleep(0.02)
missing = [j for j in win_frames if cache2.get(j) is None]
assert not missing, f"the scrub window left gaps: {missing}"
print(f"scrub window OK: frames {win_frames[0]}..{win_frames[-1]} warmed after serving 100")

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

# ---- (I306, w4) the scrub window on a 4K file: every frame it serves or caches is the file read forward ----
import hashlib                                  # noqa: E402
from kinetrace.video_source import FrameReader  # noqa: E402

V4K = os.path.join(ROOT, "test4k.mp4")          # 3840x2160, 300 frames, a keyframe every 12
rd = FrameReader(V4K, 0)
REF4K = []                                      # SHA-256 of each frame read forward from frame 0 (25 MB each)
import cv2                                      # noqa: E402
for f in range(300):
    REF4K.append(hashlib.sha256(cv2.cvtColor(rd.read(f), cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
rd.release()
fb4k = 3840 * 2160 * 3
cache4 = FrameCache((SCRUB_WINDOW + 2) * fb4k)  # exactly what a camera's window gets: eviction is exercised
svc4 = SeekService(V4K, cache4, n_frames=300)
src4 = VideoSource(V4K, cache4)
seen, bad, gaps = {}, [], []
seq = (list(range(100, 116)) + list(range(114, 94, -1)) + [230] + list(range(229, 219, -1))
       + list(range(221, 226)) + [10] + list(range(9, -1, -1)) + [299, 298, 297])
for idx in seq:                                 # what SeekService.run does, step by step
    svc4.request(idx)
    svc4._target = None
    fr = src4.get_frame(idx)
    if hashlib.sha256(fr.tobytes()).hexdigest() != REF4K[idx]:
        bad.append(("served", idx))
    svc4._fill(src4, idx, fr.nbytes)
    for f, arr in list(cache4._data.items()):
        if seen.get(f) is not arr:              # each cached array checked once
            seen[f] = arr
            if hashlib.sha256(arr.tobytes()).hexdigest() != REF4K[f]:
                bad.append(("cached", f))
    want = [f for a, b in svc4._plan_fill(idx, fb4k) for f in range(a, b + 1)]
    gaps += [(idx, f) for f in want + [idx] if cache4.get(f) is None]
src4.close()
assert not bad, f"(I306) the window holds a frame that is not the file read forward: {bad[:8]}"
assert not gaps, f"(I306) the window lost frames it had filled: {gaps[:8]}"
print(f"scrub window exact OK: {len(seq)} moves on 4K, {len(seen)} cached frames, each == the file read forward")

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
        assert not hasattr(s, "eof_truncated"), "the eof_truncated signal was removed (never fired in the app)"
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
assert ev == [("failed", 300)], f"without a verified count a persistent failure is a decode_failed too: {raw}"
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
forget_recovery(VID)

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
forget_recovery(VID)

# ---- (I306, G204) the scrub window: two 4K cameras stepped with real F / B presses ----
import shutil                                   # noqa: E402
import cv2                                      # noqa: E402
from PySide6.QtTest import QTest                # noqa: E402
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog   # noqa: E402
from kinetrace import cameraorder               # noqa: E402
from kinetrace.app import STEP_GAP_S            # noqa: E402
from kinetrace.video_source import DEFAULT_CACHE_BYTES, FrameReader   # noqa: E402

QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(), self.result())[1]
OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
CAM_A, CAM_B = (os.path.join(OUT, f"scrub_window_{c}.mp4") for c in ("a", "b"))
for _p in (CAM_A, CAM_B):                       # 3840x2160, 300 frames, a keyframe every 12
    shutil.copyfile(os.path.join(ROOT, "test4k.mp4"), _p)
forget_recovery(CAM_A, CAM_B)
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([CAM_B], ""))


FAILS = []


def check(ok, msg):
    """Collect a failure and go on, so one run shows every section that fails."""
    if not ok:
        FAILS.append(msg)
        print("FAIL: " + msg)


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        gui.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def settle(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        gui.processEvents()
        time.sleep(0.005)


win = MainWindow()
win.resize(1400, 900)
win.show()
win._open_video(CAM_A)
pump(lambda: win.state == READY, 30, "open camera A")
win._add_video_dialog()
settle(0.5)
assert win.project.n_views == 2, "the second camera was not added"
assert win._sync_mode == "all" and win.grid.visible_indices() == [0, 1], "Sync all, both cameras shown"
win._goto(100)
pump(lambda: win._views[1].cache.get(100) is not None and win._lock is None, 30, "both cameras at 100")
settle(1.5)

misses = {0: [], 1: []}
_orig_slow = win._on_seek_slow


def _count_slow(view, idx):
    misses.setdefault(view, []).append(idx)
    _orig_slow(view, idx)


win._on_seek_slow = _count_slow                 # the services' lambdas look this up at call time


def step(key, n):
    for _ in range(n):
        start = win.current
        QTest.keyClick(win, key)
        pump(lambda: win.current != start and win._lock is None, 10, "a step")
        settle(max(1.2, 3 * STEP_GAP_S))        # a lone step; the idle fill runs


step(Qt.Key_F, 15)
fwd = list(misses[1])
assert win.current == 115, win.current
step(Qt.Key_B, 15)
back = misses[1][len(fwd):]
assert win.current == 100, win.current
print(f"scrub window: companion misses forward {len(fwd)}/15, backward {len(back)}/15; "
      f"working camera {len(misses[0])}/30")
check(len(fwd) <= 1, f"(I306) the other camera re-decoded on forward steps: {fwd}")
check(len(back) <= 2, f"(I306) the other camera re-decoded on backward steps: {back}")

# every picture the window served is the frame read forward from the start of the file
rd = FrameReader(CAM_B, 95)
for f in range(95, 120):
    bgr = rd.read(f)
    got = win._views[1].cache.get(f)
    if got is not None:
        assert np.array_equal(got, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), f"cached frame {f} differs from the file"
rd.release()
print("scrub window frames exact OK (cache == read forward)")

# (w2) the budget: the other camera holds a window in Sync all, the floor when it does not decode
fb = 3840 * 2160 * 3
floor = max(96 * 1024 ** 2, 3 * fb)       # a camera that does not decode keeps what it had before
comp = win._views[1].cache.max_bytes
check(comp >= min(42 * fb, DEFAULT_CACHE_BYTES // 2),
      f"(I306) the other camera's cache holds {comp // fb} 4K frames in Sync all")
check(sum(rt.cache.max_bytes for rt in win._views) <= DEFAULT_CACHE_BYTES + floor,
      "(I306) the cameras' caches together are over the frame budget")
win.act_sync_active.trigger()
settle(0.3)
assert win._views[1].cache.max_bytes <= floor, "Active view only: the other camera keeps only the floor"
assert win._views[0].cache.max_bytes > floor
win.act_sync_all.trigger()
settle(0.3)
try:
    from kinetrace.video_source import scrub_budget   # noqa: E402
except ImportError:
    scrub_budget = None
check(scrub_budget is not None, "(I306) no scrub_budget: the frame budget is not split between the cameras")
for n in ((1, 2, 4, 10) if scrub_budget is not None else ()):
    split = scrub_budget(DEFAULT_CACHE_BYTES, {v: fb for v in range(n)}, set(range(n)), 0)
    total = sum(b for b, _w in split.values())
    assert total <= DEFAULT_CACHE_BYTES + n * floor, f"{n} cameras: {total} over the budget"
    assert all(w >= 1 for _b, w in split.values()), f"{n} cameras: a decoding camera has no window"
print("scrub budget OK: one budget split between the decoding cameras")

# (w3, G204) the nearest-frame preview is veiled: one camera shown, a typed jump far away
QTest.mouseClick(win.cameras._rows[1].btn_eye, Qt.LeftButton)
settle(0.3)
assert win.grid.visible_indices() == [0]
win.spin.setFocus()
win.spin.selectAll()
QTest.keyClicks(win.spin, "250")
QTest.keyClick(win.spin, Qt.Key_Return)
assert win.current == 250 and win.cache.get(250) is None
check(win.canvas._stale.isVisible(), "(G204) the nearest frame is shown under frame 250's markers, unveiled")
assert not win.canvas.is_stale(), "a preview veil must not read as an out-of-date camera"
pump(lambda: win.cache.get(250) is not None, 20, "frame 250")
settle(0.2)
assert not win.canvas._stale.isVisible(), "the veil stays after the real frame arrived"
# a camera switch while the stand-in is up: no veil carries over, and the camera left gets its real frame
QTest.mouseClick(win.cameras._rows[1].btn_eye, Qt.LeftButton)
settle(0.5)
assert win.grid.visible_indices() == [0, 1]
for typed in ("40", "45"):                      # two quick jumps: one continuous move, so no lockstep
    win.spin.setFocus()
    win.spin.selectAll()
    QTest.keyClicks(win.spin, typed)
    QTest.keyClick(win.spin, Qt.Key_Return)
assert win.current == 45 and win._lock is None and win.cache.get(45) is None
check(win.canvas.is_preview(), "(G204) the stand-in during a continuous move is veiled")
cv1 = win.grid.canvas(1)
QTest.mouseClick(cv1.viewport(), Qt.LeftButton, Qt.NoModifier, cv1.viewport().rect().center())
settle(0.05)
assert win.project.active == 1, "a click on the other camera works in it"
cv0 = win.grid.canvas(0)
assert not cv0.is_preview(), "the camera left kept its stand-in veil"   # (B may show its OWN stand-in)
pump(lambda: win._views[0].cache.get(45) is not None, 20, "camera A's frame 45")
settle(0.3)
assert cv0._raw_rgb is win._views[0].cache.get(45), "the camera left must show its real frame 45"
win._set_active_view(0)
settle(0.3)
print("preview veil OK (G204)")

# (I306) Help > System Check names the window and its verdict
from kinetrace.video_source import describe_scrub   # noqa: E402
shown = []
win._text_report = lambda title, text, *a, **k: shown.append(text)
win._dev_probe.wait(180_000)
win.act_syscheck.trigger()
assert shown and "Scrubbing:" in shown[0] and "each of the 2 cameras shown" in shown[0], shown[-1:]
assert "+-20 frames" in shown[0] and " - good:" in shown[0] and "Hand check:" in shown[0], shown[0][-400:]
assert all(ord(c) < 128 for c in shown[0].split("Scrubbing:")[1]), "the scrubbing lines are ASCII"
del win._text_report
GB = 1024 ** 3
assert " - ok:" in describe_scrub(4 * GB, [(30, 12, fb)] * 4)            # inside a keyframe interval: ok
assert " - poor:" in describe_scrub(4 * GB, [(6, 12, fb)] * 10)          # less than one interval: poor
print("System Check scrubbing line OK")

win._dev_probe.wait(180_000)
win.close()
gui.processEvents()
forget_recovery(CAM_A, CAM_B)
assert not FAILS, f"{len(FAILS)} scrub-window check(s) failed"

print("SCRUB PASSED")
