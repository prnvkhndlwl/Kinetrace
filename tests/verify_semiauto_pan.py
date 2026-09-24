"""Semi-automatic step-tracking, pan tool, and the single-scrubber layout.

- Layout: there is NO separate QSlider — the timeline panel is the one and
  only scrubber (a slider handle's half-width inset can never align with the
  lane pixels); the frame spinbox lives in the control panel.
- Pan (✋ / H): left-drag pans instead of editing; works while interaction is
  disabled (tracking); no point is ever added.
- Semi-automatic mode: F tracks exactly one frame forward and pauses; the next
  F re-seeds from the (possibly corrected) current positions; falls back to
  plain navigation where nothing is seedable; Space steps too; mode persists
  in ui_state."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

VID = os.path.join(ROOT, r"test600.mp4")
GT = np.load(VID + ".gt.npz")["gt"]
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication([])
from cotracker_app.app import MainWindow, READY, TRACKING
from cotracker_app.timeline import GUTTER_W

win = MainWindow()
win.resize(1280, 860)
win.show()


def key(k, mods=Qt.NoModifier):
    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, k, mods))
    app.processEvents()


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def mouse(widget, etype, pos, button=Qt.LeftButton):
    ev = QMouseEvent(etype, QPointF(*pos), QPointF(*pos), button,
                     button if etype != QEvent.MouseButtonRelease else Qt.NoButton,
                     Qt.NoModifier)
    getattr(widget, {QEvent.MouseButtonPress: "mousePressEvent",
                     QEvent.MouseMove: "mouseMoveEvent",
                     QEvent.MouseButtonRelease: "mouseReleaseEvent"}[etype])(ev)
    app.processEvents()


win._open_video(VID)
pump(lambda: win.state == READY, 20, "open")
s = win.session
app.processEvents()

# ---- 1. the timeline IS the scrubber: no separate slider exists ----
assert not hasattr(win, "slider"), "the QSlider must be gone — timeline is the scrubber"
win._goto(42)
assert win.spin.value() == 42, "frame box must follow navigation"
win.spin.setValue(7)
win.spin.editingFinished.emit()
app.processEvents()
assert win.current == 7, "typing a frame number must still seek"
win._goto(0)
print("single-scrubber layout OK (no QSlider; timeline + frame box seek)")

# ---- 2. pan tool: left-drag pans, adds nothing; H toggles; works uninteractive ----
c = win.canvas
key(Qt.Key_H)
assert win.btn_pan.isChecked() and c._pan_mode, "H must toggle the pan tool"
for _ in range(6):
    c.scale(1.25, 1.25)  # zoom so scrollbars have range
c._user_zoomed = True
app.processEvents()
h0 = c.horizontalScrollBar().value()
n_before = s.n_points
mouse(c, QEvent.MouseButtonPress, (400, 300))
mouse(c, QEvent.MouseMove, (300, 260))
mouse(c, QEvent.MouseButtonRelease, (300, 260))
assert c.horizontalScrollBar().value() != h0, "left-drag in pan mode must pan"
assert s.n_points == n_before, "pan drag must not add a point"
c.set_interactive(False)  # tracking state: pan must still work
h1 = c.horizontalScrollBar().value()
mouse(c, QEvent.MouseButtonPress, (400, 300))
mouse(c, QEvent.MouseMove, (340, 300))
mouse(c, QEvent.MouseButtonRelease, (340, 300))
assert c.horizontalScrollBar().value() != h1, "pan must work while uninteractive"
c.set_interactive(True)
key(Qt.Key_H)
assert not c._pan_mode
# stray clicks must NOT add points — placement requires the N/＋Add arm
mouse(c, QEvent.MouseButtonPress, (400, 300))
mouse(c, QEvent.MouseButtonRelease, (400, 300))
assert s.n_points == n_before, "a stray click must never add a point"
key(Qt.Key_N)
assert win.btn_add.isChecked() and c._place_mode, "N must arm placement (crosshair)"
mouse(c, QEvent.MouseButtonPress, (400, 300))
mouse(c, QEvent.MouseButtonRelease, (400, 300))
assert s.n_points == n_before + 1, "armed click must place the point"
assert not win.btn_add.isChecked() and not c._place_mode, "placement is one-shot"
key(Qt.Key_N)
key(Qt.Key_Escape)
assert not win.btn_add.isChecked(), "Esc must cancel an armed placement"
win._on_delete(s.n_points - 1)
c.fit()
print("pan tool + N place mode OK")

# ---- Space = preview play/pause; T = track ----
cur0 = win.current
key(Qt.Key_Space)
assert win.btn_play.isChecked(), "Space must start preview playback"
key(Qt.Key_Space)
assert not win.btn_play.isChecked(), "Space must pause preview playback"
win._goto(cur0)
print("space play/pause OK")

# ---- 3. semi-automatic stepping ----
win.act_mode_semi.trigger()
assert win._track_mode == "semi"
start = 100
win._goto(start)
win._on_add(float(GT[start, 0, 0]), float(GT[start, 0, 1]))
pid = s.n_points - 1
assert win.btn_track.text().startswith("Step"), win.btn_track.text()

def step_and_wait(expect_frame):
    key(Qt.Key_F)
    pump(lambda: win.state == TRACKING, 120, "step start (model load on first)")
    pump(lambda: win.state == READY, 60, "step finish")
    pump(lambda: win.current == expect_frame, 10, f"arrive at {expect_frame}")

step_and_wait(start + 1)
assert s.tracked[start + 1, pid], "step must track the next frame"
err1 = np.linalg.norm(s.tracks[start + 1, pid] - GT[start + 1, 0])
for i in range(2, 5):
    step_and_wait(start + i)
    assert s.tracked[start + i, pid]
errs = np.linalg.norm(s.tracks[start + 1:start + 5, pid] - GT[start + 1:start + 5, 0], axis=1)
print(f"semi-auto steps OK: frames {start + 1}..{start + 4} tracked, "
      f"err {errs.mean():.2f} px mean / {errs.max():.2f} max")
assert errs.mean() < 6.0

# a correction feeds the NEXT step (re-seed from what you see)
corr = GT[start + 4, 0] + np.array([4.0, -3.0])
win._on_place(pid, float(corr[0]), float(corr[1]))
step_and_wait(start + 5)
assert np.allclose(s.tracks[start + 4, pid], corr, atol=0.5), \
    "step must keep the corrected seed exactly"
d = np.linalg.norm(s.tracks[start + 5, pid] - corr)
print(f"corrected re-seed OK (seed kept exact; next frame {d:.1f} px from the correction)")

# Space also steps in semi mode
win._toggle_tracking()
pump(lambda: win.state == TRACKING, 60, "space step start")
pump(lambda: win.state == READY, 60, "space step finish")
assert win.current == start + 6 and s.tracked[start + 6, pid]
print("space-steps OK")

# F falls back to plain navigation where nothing is seedable
win._goto(400)
assert not s.seedable_at(400)
key(Qt.Key_F)
assert win.state == READY and win.current == 401, "F must just move when nothing seedable"
assert not s.tracked[401, pid]
print("fallback navigation OK")

# mode persists through ui_state
win._sync_ui_state()
assert s.ui_state["track_mode"] == "semi"
win.act_mode_auto.trigger()
assert win._track_mode == "auto" and win.btn_track.text().startswith("Track")
print("mode persistence + switch-back OK")

win._dev_probe.wait(180_000)
win.close()
app.processEvents()
if os.path.exists(VID + ".cotracker.npz"):
    os.remove(VID + ".cotracker.npz")
print("SEMIAUTO+PAN PASSED")
