"""Manual annotation: with a point selected in the
panel, a plain (unarmed) left click on the video places THAT point at the
current frame by hand, replacing what the tracker put there; Shift+< / Shift+>
jump to the selected point's first / last frame with data (, and . step between the
hand-placed ones). Also pins:
nothing selected = nothing edited, a drag is not a click, Ctrl+Z takes the
click back, the flags survive a project round trip, and the timeline paints
the marks without error."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

VID = os.path.join(ROOT, "test600.mp4")
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication([])
from kinetrace.app import MainWindow, READY

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


def click(x, y, mods=Qt.NoModifier):
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, mods, vp)
    app.processEvents()


win._open_video(VID)
pump(lambda: win.state == READY, 20, "open")
s = win.session
T = s.n_frames

# two points with fake "tracker" data over frames 0..199
win._on_add(100.0, 100.0)
win._on_add(300.0, 200.0)
L = 200
tw = np.zeros((L, 2, 2), np.float32)
tw[:, 0] = (100.0, 100.0)
tw[:, 1] = (300.0, 200.0)
s.write_segment(0, tw, np.ones((L, 2), bool), [0, 1], np.full((L, 2), 0.9, np.float32))
assert s.manual[:, 0].sum() == 0 and s.manual[:, 1].sum() == 0, "write_segment clears manual"
win._refresh_point_list()
win.canvas.fit()
app.processEvents()

# ---- nothing selected: a plain click edits nothing ----
win._deselect()
app.processEvents()
assert win.selected is None
before = s.tracks.copy()
click(200, 150)
assert np.array_equal(np.nan_to_num(before), np.nan_to_num(s.tracks)), "stray click edited data"
assert s.n_points == 2, "stray click added a point"

# ---- selected point + plain click at a frame WITH tracker data: overwrite by hand ----
win._on_select(1)
win._goto(50)
click(320, 210)
assert s.n_points == 2, "annotation must not create a point"
assert s.manual[50, 1] and s.tracked[50, 1], "frame 50 not flagged manual"
assert np.allclose(s.tracks[50, 1], (320, 210), atol=0.6), s.tracks[50, 1]
assert s.confidence[50, 1] == 1.0
assert not s.manual[50, 0], "the other point must be untouched"
assert np.allclose(s.tracks[50, 0], (100, 100))
print("annotate-by-click on a tracked frame: OK")

# ---- a frame without data: same gesture, same point ----
win._goto(400)
assert not s.tracked[400, 1]
click(330, 220)
assert s.n_points == 2 and s.manual[400, 1] and s.tracked[400, 1]
assert np.allclose(s.tracks[400, 1], (330, 220), atol=0.6)

# ---- a third annotation, then the navigation keys ----
win._goto(120)
click(310, 205)
assert sorted(s.manual_frames(1).tolist()) == [50, 120, 400]
# Shift+< / Shift+> walk the point's DATA (on a tracked point the old
# hand-placed-only rule looked like a dead key). P2 was tracked over
# 0..199 and hand-placed at 400, so its data runs 0..199 plus 400.
win._goto(10)
key(Qt.Key_Greater, Qt.ShiftModifier)
assert win.current == 400, f"Shift+> went to {win.current}"
key(Qt.Key_Less, Qt.ShiftModifier)
assert win.current == 0, f"Shift+< went to {win.current}"
# the keys are offered through the application filter too (focus in a spin box)
win.spin.setFocus()
app.processEvents()
QTest.keyClick(win.spin, Qt.Key_Greater, Qt.ShiftModifier)
app.processEvents()
assert win.current == 400, f"Shift+> via the event filter went to {win.current}"
# the hand-placed frames themselves are what , and . step between
win._goto(0)
key(Qt.Key_Period)
assert win.current == 50, f". went to {win.current}"
key(Qt.Key_Period)
assert win.current == 120, f". went to {win.current}"
key(Qt.Key_Comma)
assert win.current == 50, f", went to {win.current}"
# and the point's own menu names them, with the frame numbers
_menu, _acts = win.canvas._build_context_menu(1)
assert "(50)" in _acts["go_manual_first"].text() and "(400)" in _acts["go_manual_last"].text()
win._point_menu_extra_action(_acts["go_manual_last"], _acts, 1)
assert win.current == 400
_menu.deleteLater()
# a point with data but none placed by hand still jumps (P1 is tracked 0..199)
win._on_select(0)
assert len(s.manual_frames(0)) == 0
key(Qt.Key_Greater, Qt.ShiftModifier)
assert win.current == 199, f"Shift+> on a tracked point went to {win.current}"
print("Shift+< / Shift+> navigation: OK")

# ---- a drag without N is not a click: nothing edited ----
win._on_select(1)
win._goto(200)
p0 = win.canvas.mapFromScene(QPointF(200.0, 150.0))
p1 = win.canvas.mapFromScene(QPointF(260.0, 190.0))
QTest.mousePress(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseMove(win.canvas.viewport(), p0 + QPoint(15, 15))
QTest.mouseMove(win.canvas.viewport(), p1)
QTest.mouseRelease(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, p1)
app.processEvents()
assert not s.manual[200, 1] and s.n_points == 2, "an unarmed drag edited data"

# ---- Ctrl+Z takes the last click back ----
win._goto(300)
click(340, 230)
assert s.manual[300, 1]
win._undo_run()
assert not s.manual[300, 1] and not s.tracked[300, 1], "undo did not revert the click"
assert s.manual[50, 1] and s.manual[120, 1] and s.manual[400, 1]
print("drag-is-not-a-click + undo: OK")

# ---- armed placement (N) is unchanged: a new point ----
win._on_select(1)
win._goto(50)
key(Qt.Key_N)
assert win.btn_add.isChecked()
click(500, 300)
assert s.n_points == 3, "armed click on a frame with data must add a point"
assert not win.btn_add.isChecked()

# ---- timeline paints the marks; flags survive a project round trip ----
win.timeline.repaint()
app.processEvents()
proj = os.path.join(SCRATCH, "annotate_rt.kinetrace")
win.project.save(proj)
from kinetrace.project import Project
back = Project.load(proj)
s2 = back.session
assert sorted(s2.manual_frames(1).tolist()) == [50, 120, 400], s2.manual_frames(1)
assert np.allclose(s2.tracks[50, 1], (320, 210), atol=0.6)
print("timeline paint + project round trip: OK")

win._dev_probe.wait(30000)
win.close()
app.processEvents()
forget_recovery(VID)
print("verify_annotate PASSED")
