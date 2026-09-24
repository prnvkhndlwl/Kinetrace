"""Offscreen GUI: keyboard focus discipline and the control bar. Real key events (QTest) are sent to whatever widget holds
the focus — a spin box, the point list, a toolbar button — and the hotkeys
must still act; spin boxes give the focus back after an edit; the marker
size applies at once; the spin-box steppers are actually drawn; Follow is off
by default and R fits the whole frame regardless of it; typing in the rename
editor is NOT hijacked.

Run: .venv\\Scripts\\python.exe tests\\verify_ui_focus.py   (needs test600.mp4)
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

from kinetrace import theme

VID = os.path.join(ROOT, "test600.mp4")
if not os.path.exists(VID):
    sys.exit("test600.mp4 missing — python make_test_video.py test600.mp4 --seed 0")
for stale in (VID + ".cotracker.npz",):
    if os.path.exists(stale):
        os.remove(stale)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace.app import READY, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1500, 950)
win.show()


def pump(seconds=0.15):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


# ---- G4: H with no video must not arm the (disabled) pan tool ---------------
app.setActiveWindow(win)
pump()
QTest.keyClick(win, Qt.Key_H)
pump()
assert not win.btn_pan.isChecked(), "H with no video armed the pan tool"
print("H ignored without a video OK")

win._open_video(VID)
for _ in range(300):
    pump(0.05)
    if win.state == READY:
        break
assert win.state == READY
app.setActiveWindow(win)
pump()

# ---- G1: every menu shows its entries' tooltips ------------------------------
from PySide6.QtWidgets import QMenu  # noqa: E402
tipped = [m for m in win.findChildren(QMenu) if any(a.toolTip() and a.toolTip() != a.text().replace("&", "")
                                                   for a in m.actions())]
assert tipped, "no menu carries a tooltip?"
# (the menu bar's own overflow menu is Qt's, untitled, and never used at the window's minimum width)
hidden = [m.title() or type(m.parent()).__name__ for m in win.findChildren(QMenu)
          if not m.toolTipsVisible() and m.title()]
assert not hidden, f"menus that hide their tooltips: {hidden}"
print(f"all {len(win.findChildren(QMenu))} menus show their tooltips OK")

# ---- G2: the control bar folds to icons instead of widening the window -------
full_w = win._controls_full_w
assert win._controls.minimumWidth() < full_w - 150, (win._controls.minimumWidth(), full_w)
w0 = win.width()
win.resize(win.minimumSizeHint().width(), 900)
pump(0.3)
assert win._controls.width() < full_w and win._controls_compact
assert win.btn_pan.toolButtonStyle() == Qt.ToolButtonIconOnly, "the least-needed label folds first"
assert win.btn_track.text().startswith("Track"), "the Track button keeps its text"
assert win.btn_follow.toolTip(), "an icon-only button must keep its tooltip"
win.resize(full_w + 600, 900)
pump(0.3)
assert not win._controls_compact and win.btn_follow.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
# a bar a little short folds only the least-needed labels, one button at a time
short = win._label_saving[win._compact_order[0]] // 2 + 1
win.resize(win.width() - (win._controls.width() - full_w) - short, 900)
pump(0.3)
assert win._controls_level == 1, (win._controls_level, win._controls.width(), full_w)
assert win.btn_pan.toolButtonStyle() == Qt.ToolButtonIconOnly
assert win.btn_add.toolButtonStyle() == Qt.ToolButtonTextBesideIcon, "Add keeps its label longest"
assert win.marker_spin.value() == 3, "markers are 3 px by default"
win.resize(1500, 950)
pump(0.3)
print(f"control bar: {full_w} px with labels, folds to icons below it (window min "
      f"{win.minimumSizeHint().width()} px) OK")

# ---- defaults ------------------------------------------------------------
assert not win.btn_follow.isChecked(), "Follow must be OFF by default"
assert not win.canvas._follow_enabled
from kinetrace.session import DEFAULT_UI_STATE
assert DEFAULT_UI_STATE["follow"] is False
print("Follow off by default OK")

# ---- the steppers are drawn -----------------------------------------------
rules = theme.spin_arrow_rules()
assert "image: url(" in rules, "spin arrow images were not generated"
img = win.marker_spin.grab().toImage()
w, h = img.width(), img.height()
# the button column, away from the border and the hairline
col = np.array([[img.pixelColor(x, y).value() for x in range(w - 14, w - 3)] for y in range(3, h - 3)])
assert col.max() - col.min() > 80, f"spin steppers still invisible (value range {col.min()}-{col.max()})"
print(f"spin-box steppers drawn OK (value range {col.min()}-{col.max()})")

# ---- hotkeys with focus in a spin box --------------------------------------
win._goto(100)
pump()
win.marker_spin.setFocus()
pump()
assert app.focusWidget() is win.marker_spin or app.focusWidget().parent() is win.marker_spin
QTest.keyClick(app.focusWidget(), Qt.Key_F)
pump()
assert win.current == 101, f"F with the marker box focused did not advance (frame {win.current})"
QTest.keyClick(app.focusWidget(), Qt.Key_B)
pump()
assert win.current == 100, "B with the marker box focused did not step back"
QTest.keyClick(app.focusWidget(), Qt.Key_F, Qt.ShiftModifier)
pump()
assert win.current == 100 + win.step_spin.value(), "Shift+F with a spin box focused"
# digits still type into the box, Enter applies and RELEASES the focus
win.marker_spin.setFocus()
win.marker_spin.selectAll()
QTest.keyClicks(app.focusWidget(), "12")
pump()
assert win.marker_spin.value() == 12, win.marker_spin.value()
assert abs(win.canvas._marker_radius - 12) < 1e-6, "marker size must apply while typing, not on Enter"
QTest.keyClick(app.focusWidget(), Qt.Key_Return)
pump()
assert app.focusWidget() is not win.marker_spin and (app.focusWidget() is None or app.focusWidget().parent() is not win.marker_spin), \
    "the marker box must release the keyboard after Enter"
QTest.keyClick(win, Qt.Key_F)
pump()
assert win.current == 100 + win.step_spin.value() + 1, "F right after editing the marker size"
# the same for the step box: type a value, then Shift+F uses it immediately
win.step_spin.setFocus()
win.step_spin.selectAll()
QTest.keyClicks(app.focusWidget(), "5")
QTest.keyClick(app.focusWidget(), Qt.Key_Return)
pump()
assert win.step_spin.value() == 5
QTest.keyClick(win, Qt.Key_F, Qt.ShiftModifier)
pump()
assert win.current == 100 + 10 + 1 + 5, win.current
# a minus typed into the (negative-capable) camera offset box must stay text:
# the video-zoom hotkey must not fire from inside a spin box
win.step_spin.setFocus()
z0 = win.canvas.transform().m11()
QTest.keyClick(app.focusWidget(), Qt.Key_Minus)
pump()
assert abs(win.canvas.transform().m11() - z0) < 1e-9, "'-' inside a spin box must not zoom the video"
win.step_spin.clearFocus()
print("hotkeys survive spin-box focus; spin boxes release focus after Enter; marker size applies live OK")

# ---- hotkeys with focus in the point list -----------------------------------
win._goto(200)
pump()
pid = win.session.add_point(200, 320.0, 240.0)
win._refresh_point_list()
pump()
win.point_list.setFocus()
win.point_list.setCurrentRow(0)
pump()
QTest.keyClick(win.point_list, Qt.Key_F)
pump()
assert win.current == 201, "F with the point list focused (type-ahead must not eat it)"
QTest.keyClick(win.point_list, Qt.Key_N)
pump()
assert win.btn_add.isChecked(), "N with the point list focused must arm placement"
QTest.keyClick(win.point_list, Qt.Key_Escape)
pump()
assert not win.btn_add.isChecked(), "Esc with the point list focused must cancel placement"
print("hotkeys survive point-list focus OK")

# ---- typing a NAME is never hijacked -----------------------------------------
item = win.point_list.item(0)
win.point_list.editItem(item)
pump()
editor = app.focusWidget()
assert isinstance(editor, QLineEdit), f"rename editor not focused: {editor}"
before = win.current
QTest.keyClicks(editor, "fbn")
pump()
assert win.current == before and not win.btn_add.isChecked(), "letters typed into the rename editor triggered hotkeys"
assert editor.text().endswith("fbn")
QTest.keyClick(editor, Qt.Key_Escape)
pump()
print("rename editor keeps its letters OK")

# ---- toolbar buttons never take the focus; Space plays, it does not toggle them
for b in (win.btn_follow, win.btn_roi, win.btn_autopause, win.btn_add):
    assert int(b.focusPolicy()) == int(Qt.NoFocus), f"{b.text()} still takes keyboard focus"
was = win.btn_roi.isChecked()
win.btn_roi.setFocus()
pump()
QTest.keyClick(win, Qt.Key_Space)
pump()
assert win.btn_roi.isChecked() == was, "Space toggled a toolbar button instead of play/pause"
assert win.btn_play.isChecked(), "Space must start preview playback"
QTest.keyClick(win, Qt.Key_Space)
pump()
assert not win.btn_play.isChecked()
print("toolbar buttons keep out of the focus chain; Space = play/pause OK")

# ---- R fits the whole frame, and stays fitted (Follow is off) -----------------
win.canvas.zoom_step(4.0)
pump()
zoomed = win.canvas.transform().m11()
win.marker_spin.setFocus()          # the worst case: a focused spin box
QTest.keyClick(app.focusWidget(), Qt.Key_R)
pump()
fitted = win.canvas.transform().m11()
assert fitted < zoomed * 0.5, "R did not reset the view"
win._goto(win.current + 1)
pump()
assert abs(win.canvas.transform().m11() - fitted) < 1e-6, "the view moved on its own after R (auto-frame must be off)"
print("R resets the view and it stays put OK")

# ---- the camera panel's offset box gives the focus back too ------------------
win.marker_spin.clearFocus()
row = win.cameras._rows[0] if win.cameras._rows else None
if row is not None:
    assert int(row.spin.focusPolicy()) == int(Qt.ClickFocus)

# ---- every other feature, driven by real keys from odd focus places ----------
from PySide6.QtWidgets import QInputDialog
QInputDialog.getText = staticmethod(lambda *a, **k: ("jump", True))
QInputDialog.getItem = staticmethod(lambda *a, **k: ("jump", True))
# the timeline takes click focus and must pass the letters on
QTest.mouseClick(win.timeline, Qt.LeftButton, pos=win.timeline.rect().center())
pump()
f0 = win.current
QTest.keyClick(app.focusWidget() or win, Qt.Key_F)
pump()
assert win.current == f0 + 1, "F with the timeline focused"
span0 = win.timeline._view[1] - win.timeline._view[0]
QTest.keyClick(app.focusWidget() or win, Qt.Key_Plus, Qt.ShiftModifier)
pump()
assert win.timeline._view[1] - win.timeline._view[0] < span0, "Shift++ with the timeline focused must zoom the timeline"
z0 = win.canvas.transform().m11()
QTest.keyClick(app.focusWidget() or win, Qt.Key_Plus)
pump()
assert win.canvas.transform().m11() > z0 * 1.1, "plain + with the timeline focused must zoom the video"
QTest.keyClick(win, Qt.Key_R)
pump()
# E twice marks an event, even with a spin box focused
n_ev = len(win.session.events)
win.step_spin.setFocus()
QTest.keyClick(app.focusWidget(), Qt.Key_E)
pump()
win._goto(win.current + 3)
pump()
QTest.keyClick(app.focusWidget() or win, Qt.Key_E)
pump()
assert len(win.session.events) == n_ev + 1, "E/E with a spin box focused did not mark an event"
# H toggles the pan tool from a spin box
win.step_spin.setFocus()
was_pan = win.btn_pan.isChecked()
QTest.keyClick(app.focusWidget(), Qt.Key_H)
pump()
assert win.btn_pan.isChecked() != was_pan, "H with a spin box focused"
QTest.keyClick(app.focusWidget() or win, Qt.Key_H)
pump()
# Delete INSIDE a spin box is an editing key: the selected point must survive
win.point_list.setCurrentRow(0)
win._select(0) if hasattr(win, "_select") else None
n_pts = win.session.n_points
win.marker_spin.setFocus()
QTest.keyClick(app.focusWidget(), Qt.Key_Delete)
pump()
assert win.session.n_points == n_pts, "Delete typed into a spin box deleted a point"
win.marker_spin.clearFocus()
# Delete with the LIST focused deletes the selected point
win.point_list.setFocus()
win.point_list.setCurrentRow(0)
pump()
QTest.keyClick(win.point_list, Qt.Key_Delete)
pump()
assert win.session.n_points == n_pts - 1, "Delete with the point list focused must delete the selected point"
print("timeline focus, events, pan, delete OK")

# ---- the 3D window: frame keys still work while it is the active window ------
win._toggle_view3d(True)
pump()
assert win.view3d is not None and win.view3d.isVisible()
app.setActiveWindow(win.view3d)
win.view3d.chk_points.setFocus()
pump()
f0 = win.current
QTest.keyClick(app.focusWidget() or win.view3d, Qt.Key_F)
pump()
assert win.current == f0 + 1, "F from the 3D window must step the frame"
win.view3d.hide()
app.setActiveWindow(win)
pump()
print("3D window keeps the frame keys OK")

# ---- I68: the Segment & Points panel floated: its focus still takes the hotkeys --
win.dock.setFloating(True)
pump(0.3)
if win.point_list.count():
    app.setActiveWindow(win.dock)
    win.point_list.setFocus()
    pump()
    f0 = win.current
    QTest.keyClick(app.focusWidget() or win.point_list, Qt.Key_F)
    pump()
    assert win.current == f0 + 1, "F with the floated panel focused must step the frame"
win.dock.setFloating(False)
app.setActiveWindow(win)
pump(0.3)
print("floated panel keeps the hotkeys OK")

# ---- G6: an open menu owns the keyboard -----------------------------------------
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QMenu as _QM  # noqa: E402
_g6 = {}


def _keys_in_menu():
    pop = QApplication.activePopupWidget()
    _g6["open"] = isinstance(pop, _QM)
    _g6["f0"], _g6["n0"] = win.current, win.session.n_points
    if pop is not None:
        QTest.keyClick(pop, Qt.Key_F)          # must NOT step the video behind the menu
        QTest.keyClick(pop, Qt.Key_Delete)     # must NOT delete a point behind it
        _g6["f1"], _g6["n1"] = win.current, win.session.n_points
        QTest.keyClick(pop, Qt.Key_Escape)     # must close the menu


QTimer.singleShot(300, _keys_in_menu)
win.btn_skeleton.showMenu()                    # blocks until the menu closes
pump()
assert _g6.get("open"), "the Skeleton menu did not open"
assert _g6["f1"] == _g6["f0"] and _g6["n1"] == _g6["n0"], f"hotkeys acted behind an open menu: {_g6}"
assert QApplication.activePopupWidget() is None, "Escape did not close the menu"
print("an open menu keeps the keyboard: Escape closes it, nothing acts behind it OK")

win._dev_probe.wait(15000)
win.close()
if os.path.exists(VID + ".cotracker.npz"):
    os.remove(VID + ".cotracker.npz")
print("UI FOCUS PASSED")
