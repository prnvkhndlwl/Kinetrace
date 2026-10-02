"""Offscreen GUI: opening videos / projects never looks frozen (G31: the
wait before everything was loaded looked like a frozen app).

Reading a 4K file takes ~2.3 s (measured on a real clip); these small clips read
in a blink, so `probe_video` is slowed down here (same result, same progress
calls). Checked with real key / mouse events:
  File -> Open Video: the loading card appears, names the file and each step
  ("... frames ... checking the frame rate", "Checking that all N frames can be
  read"), counts them on its bar, blocks clicks and hotkeys, the window keeps
  repainting (the spinner turns), and it goes once the first picture is on screen;
  Cancel (or Esc) while the file is read: nothing changes (the open project stays);
  a quick open does not flash the card;
  adding a camera no longer freezes the window (the probe used to run ON the GUI
  thread: ~2.3 s per 4K file) -- `_add_view` still returns when the camera is in;
  several cameras are read at once (3 slowed files in about the time of one) and
  the card counts them; a 2-camera project opens with every camera read at once;
  closing during an open waits for it rather than tearing a half-built project.

Run: .venv\\Scripts\\python.exe tests\\verify_loading.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.project import Project
from kinetrace.session import TrackingSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
W, H, N = 320, 240, 40
PERF = float(os.environ.get("KINETRACE_PERF_SCALE", "1"))
paths = []
for c in range(3):
    path = os.path.join(OUT, f"loading_cam{c}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 50 + 40 * c, np.uint8)
        cv2.circle(img, (40 + 5 * f, 120), 8, (240, 240, 240), -1)
        vw.write(img)
    vw.release()
    paths.append(path)
    forget_recovery(path)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

import kinetrace.app as appmod  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

real_probe = appmod.probe_video
SLOW = {"s": 0.0}


def slow_probe(path, vfr_samples=60, progress=None):
    """The real probe, with each stage stretched like a 4K file's."""
    def paced(stage, facts):
        if progress is not None:
            progress(stage, facts)
        time.sleep(SLOW["s"] / 3)
    return real_probe(path, vfr_samples, paced)


appmod.probe_video = slow_probe
app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1400, 900)
win.show()
app.setActiveWindow(win)


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t > timeout:
            raise TimeoutError(what)


ticks = []
sp = win.overlay.spinner
sp._timer.timeout.connect(lambda: ticks.append(time.perf_counter()))

# ---- 1. Open Video: the card, its steps, input blocked, a live window -----------------
SLOW["s"] = 1.2
t0 = time.perf_counter()
win._open_video(paths[0])
pump(0.35)
ov = win.overlay
assert ov.isVisible(), "the loading card is up while the file is read"
assert ov.title.text() == "Opening loading_cam0.mp4", ov.title.text()
seen = {ov.detail.text()}
steps = {ov.bar.value()}
assert not win.menuBar().isEnabled(), "the menus are off while opening"
QTest.keyClick(win, Qt.Key_N)                          # a hotkey during the open does nothing
assert not win.btn_add.isChecked()
while win._busy_stack:                                 # the card is gone at the first picture
    seen.add(ov.detail.text())
    steps.add(ov.bar.value())
    pump(0.02)
    if time.perf_counter() - t0 > 20 * PERF:
        raise TimeoutError("the open")
dt = time.perf_counter() - t0
assert win.state == READY and win.canvas._raw_rgb is not None, "it goes once the first picture is on screen"
assert not ov.isVisible() and win.menuBar().isEnabled()
assert any("frames" in s and "frame rate" in s for s in seen), seen
assert any("Checking that all 40 frames can be read" in s for s in seen), seen
assert ov.bar.maximum() == appmod.OPEN_STAGES and max(steps) >= 3, (steps, ov.bar.maximum())
n_ticks = len([t for t in ticks if t >= t0])
assert n_ticks > 15, f"the window kept repainting during the {dt:.1f} s open: {n_ticks} spinner turns"
print(f"open video: card up with {len(seen)} step messages, bar to {max(steps)}/{appmod.OPEN_STAGES}, input "
      f"blocked, {n_ticks} spinner turns in {dt:.1f} s, gone at the first picture OK")

# ---- 2. Cancel (and Esc) while the file is read: nothing changes ------------------------
before = win.project
win._open_video(paths[1])
pump(0.35)
assert win.overlay.isVisible() and win.overlay.btn_cancel.isVisible()
QTest.mouseClick(win.overlay.btn_cancel, Qt.LeftButton)
pump(0.05)
assert not win._loading and not win.overlay.isVisible(), "Cancel closes the card"
assert "cancelled" in win.statusBar().currentMessage()
pump(SLOW["s"] + 0.5)                                  # the abandoned read finishes: ignored
assert win.project is before and os.path.basename(win.info.path) == "loading_cam0.mp4", "nothing changed"
win._open_video(paths[1])
pump(0.35)
QTest.keyClick(win, Qt.Key_Escape)
pump(0.05)
assert not win._loading and win.project is before, "Esc cancels too"
pump(SLOW["s"] + 0.5)
print("Cancel and Esc during the read leave the open project untouched; the late result is dropped OK")

# ---- 3. a quick open does not flash the card ------------------------------------------
SLOW["s"] = 0.0
flashed = []
win.overlay.showEvent = lambda ev: flashed.append(1)
win._open_video(paths[1])
wait(lambda: win.state == READY and not win._busy_stack and os.path.basename(win.info.path) == "loading_cam1.mp4",
     10 * PERF, "quick open")
del win.overlay.showEvent
print(f"a quick open: the card {'flashed' if flashed else 'did not flash'} (it waits {win.overlay.DELAY_MS} ms) OK")

# ---- 4. adding a camera keeps the window alive ------------------------------------------
SLOW["s"] = 1.0
t0 = time.perf_counter()
ok = win._add_view(paths[2])
dt = time.perf_counter() - t0
n_ticks = len([t for t in ticks if t >= t0])
assert ok and win.project.n_views == 2
assert n_ticks > 10, f"the window repainted while the camera was read ({dt:.1f} s): {n_ticks} turns"
assert not win._loading and win.menuBar().isEnabled()
print(f"adding a camera: {n_ticks} spinner turns in {dt:.1f} s (it froze the window before), returns True OK")

# ---- 5. several cameras are read at once, and counted -----------------------------------
SLOW["s"] = 0.9
counts = []
_orig_step = win._busy_step
win._busy_step = lambda d=None, v=None, t=None: (counts.append((v, t)), _orig_step(d, v, t))[1]
t0 = time.perf_counter()
res = win._probe_many(paths, "Opening 3 cameras")
dt = time.perf_counter() - t0
win._busy_step = _orig_step
assert all(not isinstance(r, str) for r in res.values()), res
assert dt < 1.8 * SLOW["s"] * PERF + 0.8, f"3 files read at once in {dt:.2f} s (one takes {SLOW['s']} s)"
assert (3, 3) in counts or (2, 3) in counts, counts
print(f"3 slowed files read at once in {dt:.2f} s (one alone: {SLOW['s']:.1f} s), the card counted them OK")

# ---- 6. a 2-camera project opens with every camera read at once --------------------------
PROJ = os.path.join(OUT, "loading_project.kinetrace")
Project([TrackingSession(paths[0], N, 30.0, W, H), TrackingSession(paths[1], N, 30.0, W, H)],
        ["camA", "camB"], [0, 0]).save(PROJ)
forget_recovery(paths[0])
win.project.dirty = False
SLOW["s"] = 1.0
titles = set()
t0 = time.perf_counter()
win._open_project_from_path(PROJ)
while win._busy_stack or win.project.n_views != 2 or os.path.basename(win.info.path) != "loading_cam0.mp4":
    if win.overlay.isVisible():
        titles.add(win.overlay.title.text())
    pump(0.02)
    if time.perf_counter() - t0 > 20 * PERF:
        raise TimeoutError("the project open")
dt = time.perf_counter() - t0
assert "Opening loading_project.kinetrace" in titles, titles
assert dt < 2.0 * SLOW["s"] * PERF + 1.5, f"both cameras read at once: {dt:.2f} s (one takes {SLOW['s']} s)"
assert [os.path.basename(s.video_path) for s in win.project.sessions] == ["loading_cam0.mp4", "loading_cam1.mp4"]
print(f"a 2-camera project opened in {dt:.2f} s with both cameras read at once (one alone {SLOW['s']:.1f} s) OK")

# ---- 7. closing during an open waits for it ------------------------------------------
SLOW["s"] = 1.0
win._open_video(paths[2])
pump(0.3)
assert win._loading
win.close()
pump(0.1)
assert win.isVisible() and not win._loading, "close during a read cancels it and waits"
wait(lambda: not win.isVisible(), 5 * PERF, "the deferred close")
print("closing during an open cancels the read and closes afterwards OK")

appmod.probe_video = real_probe
win._dev_probe.wait(15000)
for path in paths:
    forget_recovery(path)
print("LOADING PASSED")
