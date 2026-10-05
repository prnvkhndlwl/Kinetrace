"""The point's right-click menu and the Shift+< / Shift+> jumps
("shift + >" looked dead on an ordinary tracked point, and points deserved
the segment row's menu).

Offscreen, no GPU. Covers: Shift+< / Shift+> on a TRACKED point (its first /
last frame with data), on a point with hand-placed frames, with nothing
selected (the segment's silhouette), and with nothing at all; then the
frame-aware half of the point menu — jump to first / last frame, first / last
hand-placed frame, first doubtful stretch, clear this frame, clear the
timeline's selected window, clear the whole track — from BOTH entry points
(the canvas marker menu and the POINTS list), with undo.
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

VID = os.path.join(ROOT, "test600.mp4")
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication([])
from kinetrace.app import MainWindow, READY

win = MainWindow()
win.resize(1280, 860)
win.show()


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def key(k, mods=Qt.NoModifier):
    """Press a key and return the status message the HANDLER left. Read before
    pumping: the decoder's own "Seeking frame N" message lands during the pump
    and would overwrite it."""
    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, k, mods))
    msg = win.statusBar().currentMessage()
    app.processEvents()
    return msg


win._open_video(VID)
pump(lambda: win.state == READY, 20, "open")
s = win.session

def settle(seconds=0.4):
    """Let the decoder's own "Seeking frame ..." status message land before we
    read the status bar (it would otherwise clobber the handler's message)."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


# ---- nothing to jump to at all --------------------------------------------------
win._goto(300)
settle()
msg = key(Qt.Key_Greater, Qt.ShiftModifier)
assert win.current == 300 and "Nothing to jump to" in msg, msg

# ---- an ordinary TRACKED point (no hand placements): the keys must work -----------
s.add_point(0, 100.0, 100.0)
L = 200
tw = np.zeros((L, 1, 2), np.float32)
tw[:, 0] = (150.0, 120.0)
conf = np.full((L, 1), 0.9, np.float32)
conf[120:150, 0] = 0.2                       # a doubtful stretch
s.write_segment(20, tw, np.ones((L, 1), bool), [0], conf)   # data on frames 20..219
s.clear_window([0], 0, 19)          # drop the seed at frame 0: data is exactly 20..219
s.manual[:] = False
win._refresh_point_list()
win._on_select(0)
assert len(s.manual_frames(0)) == 0 and len(s.data_frames(0)) == 200
win._goto(400)
msg = key(Qt.Key_Greater, Qt.ShiftModifier)
assert win.current == 219, win.current
assert "last frame with data" in msg, msg
key(Qt.Key_Less, Qt.ShiftModifier)
assert win.current == 20, win.current
# through the application event filter (focus in the point list), like a real user
win.layers.setFocus()
app.processEvents()
QTest.keyClick(win.layers, Qt.Key_Greater, Qt.ShiftModifier)
app.processEvents()
assert win.current == 219, win.current
print("Shift+< / Shift+> on a tracked point OK")

# ---- hand-placed frames are reported, not required --------------------------------
s.set_position(90, 0, 151.0, 121.0)
s.set_position(110, 0, 152.0, 122.0)
win._goto(0)
msg = key(Qt.Key_Greater, Qt.ShiftModifier)
assert win.current == 219 and "2 of them hand-placed" in msg, msg
key(Qt.Key_Period)                    # . still steps between hand-placed frames
assert win.current == 219
win._goto(0)
key(Qt.Key_Period)
assert win.current == 90, win.current
print("hand-placed stepping unchanged OK")

# ---- with nothing selected, the keys walk the silhouette ----------------------------
s.ensure_animal()
bm = np.zeros((s.height, s.width), bool)
bm[200:300, 280:380] = True
for f in range(50, 91):
    s.masks.set(f, bm, 9.0)
win._deselect()
win._goto(300)
msg = key(Qt.Key_Greater, Qt.ShiftModifier)
assert win.current == 90, win.current
assert "silhouette" in msg, msg
key(Qt.Key_Less, Qt.ShiftModifier)
assert win.current == 50, win.current
print("segment fallback OK")

# ---- the point menu: navigation ------------------------------------------------------
win._on_select(0)
win._goto(300)
menu, acts = win.canvas._build_context_menu(0)
for k in ("go_first", "go_last", "go_manual_first", "go_manual_last", "go_low",
          "clear_here", "clear_window", "clear_all"):
    assert k in acts, k
assert "(20)" in acts["go_first"].text() and "(219)" in acts["go_last"].text()
assert "(90)" in acts["go_manual_first"].text() and "(110)" in acts["go_manual_last"].text()
assert "140" in acts["go_low"].text() or "139" in acts["go_low"].text(), acts["go_low"].text()
assert acts["go_first"].isEnabled() and acts["go_low"].isEnabled()
assert not acts["clear_here"].isEnabled(), "no data on frame 300"
assert not acts["clear_window"].isEnabled(), "no timeline window selected"
assert acts["clear_all"].isEnabled() and "200" in acts["clear_all"].text()
win._point_menu_extra_action(acts["go_last"], acts, 0)
assert win.current == 219
win._point_menu_extra_action(acts["go_manual_first"], acts, 0)
assert win.current == 90
win._point_menu_extra_action(acts["go_low"], acts, 0)
assert win.current == 140, win.current        # the low-confidence run starts here
menu.deleteLater()
print("point menu navigation OK")

# ---- the point menu: clear this frame, undo --------------------------------------------
win._goto(100)
menu, acts = win.canvas._build_context_menu(0)
assert acts["clear_here"].isEnabled() and "frame 100" in acts["clear_here"].text()
win._point_menu_extra_action(acts["clear_here"], acts, 0)
assert not s.tracked[100, 0] and s.tracked[101, 0] and int(s.tracked[:, 0].sum()) == 199
win._undo_run()
assert s.tracked[100, 0] and int(s.tracked[:, 0].sum()) == 200, "Ctrl+Z restores the frame"
print("clear-this-frame + undo OK")

# ---- the point menu: clear a selected window ---------------------------------------------
win.timeline.sel_range = (150, 219)
win.timeline.sel_rows = [0]
win.timeline.sel_seg = False
menu, acts = win.canvas._build_context_menu(0)
assert acts["clear_window"].isEnabled() and "150" in acts["clear_window"].text()
win._point_menu_extra_action(acts["clear_window"], acts, 0)
assert int(s.tracked[:, 0].sum()) == 130 and s.tracked[149, 0] and not s.tracked[200, 0]
print("clear-selected-window OK")

# ---- the same menu from the POINTS list ---------------------------------------------------
s.add_point(0, 400.0, 300.0)                       # a second point, to be sure of the pid
win._refresh_point_list()
menu, acts = win.canvas._build_context_menu(1)
assert "(0)" in acts["go_first"].text(), acts["go_first"].text()
win._goto(300)
win._point_menu_extra_action(acts["go_first"], acts, 1)
assert win.current == 0
menu.deleteLater()

# ---- the point menu: clear the whole track, the point survives -------------------------------
menu, acts = win.canvas._build_context_menu(0)
win._point_menu_extra_action(acts["clear_all"], acts, 0)
assert int(s.tracked[:, 0].sum()) == 0 and s.n_points == 2, "the point itself must stay"
menu, acts = win.canvas._build_context_menu(0)
assert not acts["go_first"].isEnabled() and not acts["clear_all"].isEnabled()
win._undo_run()
assert int(s.tracked[:, 0].sum()) == 130
print("clear-whole-track keeps the point OK")

# ---- the canvas's own entries still dispatch (the extra hook must not swallow them) --------
menu, acts = win.canvas._build_context_menu(0)
assert not win._point_menu_extra_action(acts["rename"], acts, 0)
assert not win._point_menu_extra_action(acts["delete"], acts, 0)
assert not win._point_menu_extra_action(None, acts, 0)
# ---- keyframe interpolation between hand placements ---------------------------------
s.set_position(150, 0, 300.0, 200.0)
s.set_position(170, 0, 340.0, 260.0)                # keys at 90, 110, 150, 170 (90 / 110 already placed)
s.clear_window([0], 151, 169)                          # a gap between the last two keys
s.clear_window([0], 95, 99)                            # and one between the first two
assert not s.tracked[160, 0] and not s.tracked[97, 0]
menu, acts = win.canvas._build_context_menu(0)
assert acts["interp_fill"].isEnabled() and "4 placed frames" in acts["interp_fill"].text(), acts["interp_fill"].text()
before_120 = s.tracks[120, 0].copy()                   # tracked data between keys: untouched by "fill"
win._point_menu_extra_action(acts["interp_fill"], acts, 0)
assert s.tracked[160, 0] and s.tracked[97, 0], "the gaps are filled"
assert abs(s.confidence[160, 0] - s.INTERP_CONF) < 1e-6 and not s.manual[160, 0]
assert np.allclose(s.tracks[120, 0], before_120), "fill never touches tracked frames"
x160 = s.tracks[160, 0]
assert 290 < x160[0] < 350 and 190 < x160[1] < 270, x160        # between the neighbouring keys
assert win.act_undo.isEnabled()
win._undo_run()
assert not s.tracked[160, 0], "Ctrl+Z undoes the fill"
menu, acts = win.canvas._build_context_menu(0)
win._point_menu_extra_action(acts["interp_replace"], acts, 0)
assert s.tracked[160, 0] and not np.allclose(s.tracks[120, 0], before_120), "replace overwrites tracked frames between keys"
assert s.manual[150, 0] and np.allclose(s.tracks[150, 0], [300.0, 200.0]), "the keys themselves are kept"
win._undo_run()
assert np.allclose(s.tracks[120, 0], before_120)
# fewer than two hand-placed frames: nothing happens, and the entry says so
s2_pid = s.add_point(20, 50.0, 50.0)
menu, acts = win.canvas._build_context_menu(s2_pid)
assert not acts["interp_fill"].isEnabled()
assert s.interpolate_keyframes(s2_pid) == (0, None)
print("keyframe interpolation between hand placements OK")
win._on_anchor_toggled(0, True)
assert s.points[0].anchor
print("canvas entries untouched OK")

win._dev_probe.wait(30000)
win.close()
app.processEvents()
forget_recovery(VID)
print("verify_point_menu PASSED")
