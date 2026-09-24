"""Offscreen test: F/B/Shift navigation, +/- keyboard zoom, and follow-v2
panning (selected point, gap-hold, group fit with auto zoom-out)."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

VID = os.path.join(ROOT, r"test600.mp4")
GT = np.load(VID + ".gt.npz")["gt"]
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


win._open_video(VID)
pump(lambda: win.state == READY, 20, "open")

# ---- F / B / Shift+F / Shift+B ----
win._goto(100)
key(Qt.Key_F)
assert win.current == 101, win.current
key(Qt.Key_B)
assert win.current == 100
win.step_spin.setValue(25)
key(Qt.Key_F, Qt.ShiftModifier)
assert win.current == 125, win.current
key(Qt.Key_B, Qt.ShiftModifier)
assert win.current == 100
key(Qt.Key_B, Qt.ShiftModifier)  # clamps at 75... then to 50, 25, 0
for _ in range(5):
    key(Qt.Key_B, Qt.ShiftModifier)
assert win.current == 0, "must clamp at frame 0"
print("F/B navigation OK (incl. step jumps and clamping)")

# ---- follow: zoom in, put a point far from the view center ----
win._goto(0)
win._on_add(600.0, 440.0)   # bottom-right corner of 640x480
win.selected = 0
c = win.canvas
c.fit()
app.processEvents()
for _ in range(6):          # zoom to ~3.8x so scrollbars have range
    c.scale(1.25, 1.25)
c.centerOn(50, 50)          # look at the far corner; point now off-view
app.processEvents()
vp = c.viewport().rect()
p_before = c.mapFromScene(600.0, 440.0)
assert not vp.contains(p_before), "test setup: point should start off-view"

win.btn_follow.setChecked(True)
win._refresh_overlay()      # what every frame change calls
app.processEvents()
p_after = c.mapFromScene(600.0, 440.0)
assert vp.contains(p_after), f"follow did not bring point into view: {p_after}"
print(f"follow OK: point moved into viewport at ({p_after.x():.0f}, {p_after.y():.0f})")

# follow disabled -> view stays put
c.centerOn(50, 50)
app.processEvents()
win.btn_follow.setChecked(False)
win._refresh_overlay()
app.processEvents()
p3 = c.mapFromScene(600.0, 440.0)
assert not vp.contains(p3), "follow must be inert when toggled off"
print("follow toggle OK")

# ---- follow during a real tracking run at high zoom ----
win.btn_follow.setChecked(True)
win._goto(0)
win._on_place(0, float(GT[0, 0, 0]), float(GT[0, 0, 1]))  # move point onto dot 0
win._toggle_tracking()
pump(lambda: win.state != READY, 60, "track start")
pump(lambda: win.state == READY, 300, "track finish")
pos = win.session.tracks[win.current, 0]
p_final = c.mapFromScene(float(pos[0]), float(pos[1]))
assert vp.contains(p_final), "followed point not in view after tracked run"
print("follow-during-tracking OK")

# ---- +/- keyboard zoom (video, anchored; exact factors) ----
z0 = c.transform().m11()
key(Qt.Key_Plus)
assert abs(c.transform().m11() - z0 * 1.25) < 1e-6, "+ must zoom in 1.25x"
key(Qt.Key_Minus)
assert abs(c.transform().m11() - z0) < 1e-6, "- must zoom back out"
key(Qt.Key_Equal)  # unshifted "+" key on US layouts
assert abs(c.transform().m11() - z0 * 1.25) < 1e-6 and c._user_zoomed
key(Qt.Key_Minus)
print("keyboard zoom OK")

# ---- Shift + +/- zooms the TIMELINE time axis, video zoom untouched ----
tl = win.timeline
assert tl._view == (0, 599)
zv = c.transform().m11()
key(Qt.Key_Plus, Qt.ShiftModifier)
v0, v1 = tl._view
assert (v1 - v0) < 599, "Shift++ must zoom the time axis"
assert abs(c.transform().m11() - zv) < 1e-9, "video zoom must not move on Shift+±"
for _ in range(10):
    key(Qt.Key_Minus, Qt.ShiftModifier)
assert tl._view == (0, 599), "Shift+- must zoom the time axis back out (clamped)"
print("timeline keyboard zoom OK")

# ---- Help menu reference dialog opens ----
win._show_hotkeys()
assert win._hotkeys_dlg.isVisible()
win._hotkeys_dlg.close()
print("help dialog OK")

# ---- follow v2: a gap in the selected track holds the view still ----
s = win.session
win._goto(599)
win._on_add(50.0, 50.0)          # new point, data ONLY at 599, becomes selected
h0, v0 = c.horizontalScrollBar().value(), c.verticalScrollBar().value()
win._goto(300)                   # selected point has no data here
assert (c.horizontalScrollBar().value(), c.verticalScrollBar().value()) == (h0, v0), \
    "gap in the selected track must not move the view"
print("gap-hold OK")

# ---- follow v2: no selection keeps ALL points visible, zooming out to fit ----
win._on_add(600.0, 440.0)        # second point with data at 300, far from dot 0
win._deselect()
for _ in range(8):               # deep zoom via direct scale (no follow trigger)
    c.scale(1.25, 1.25)
c._user_zoomed = True
z_deep = c.transform().m11()
win._refresh_overlay()           # auto-frame fits the spread points
assert c.transform().m11() < z_deep, "auto-frame must zoom OUT to fit all points"
vp2 = c.viewport().rect()
for pt in (s.tracks[300, 0], [600.0, 440.0]):
    pp = c.mapFromScene(float(pt[0]), float(pt[1]))
    assert vp2.contains(pp), f"point {pt} left out of the fitted view"
print("auto-frame fit-all (zoom out) OK")

# ---- auto-frame also zooms IN on a tight cluster (bbox+buffer semantics) ----
win._on_place(0, 300.0, 300.0)
win._on_place(win.session.n_points - 1, 340.0, 330.0)
c.fit()                          # start from whole-video view
fit_scale = c.transform().m11()
win._refresh_overlay()
assert c.transform().m11() > fit_scale, \
    "auto-frame must zoom IN when all points sit in a small area"
for pt in ((300.0, 300.0), (340.0, 330.0)):
    assert c.viewport().rect().contains(c.mapFromScene(*pt))
print("auto-frame zoom-in OK")

win.close()
app.processEvents()
print("KEYS+FOLLOW PASSED")

