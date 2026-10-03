"""Offscreen GUI: timeline panel, event windows, rename collisions, R-key fit,
group gesture handler, and EXACT project state restore (frame, selection,
zoom/pan, toggles, events). No tracking runs — fast, no GPU needed."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

VID = os.path.join(ROOT, r"test600.mp4")
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
_dialog_reply = {"text": "swing", "ok": True}
QInputDialog.getText = staticmethod(lambda *a, **k: (_dialog_reply["text"],
                                                     _dialog_reply["ok"]))
QInputDialog.getItem = staticmethod(lambda *a, **k: (_dialog_reply["text"],
                                                     _dialog_reply["ok"]))

app = QApplication([])
from kinetrace.app import MainWindow, READY
from kinetrace.timeline import EVENTS_H, GUTTER_W

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
s = win.session
tlp = win.timeline
app.processEvents()

# ---- R fits the view (0 no longer does) ----
c = win.canvas
c.scale(3.0, 3.0)
c._user_zoomed = True
z_before = c.transform().m11()
key(Qt.Key_R)
assert c.transform().m11() != z_before and not c._user_zoomed, "R must fit/reset the view"
print("R fit OK")

# ---- onboarding strip: the hint is guidance, NOT a fifth numbered step ----
from PySide6.QtWidgets import QFrame, QSizePolicy

ob = win.onboarding
_lay = ob.layout()
_items = [_lay.itemAt(i).widget() for i in range(_lay.count())]
_rule = [w for w in _items
         if isinstance(w, QFrame) and w.frameShape() == QFrame.VLine]
assert _rule, "a rule must separate the numbered chips from the hint"
assert _items.index(ob._chips[-1]) < _items.index(_rule[0]) < _items.index(ob.hint), \
    "order must be: chips -> rule -> hint"
assert ob.hint.alignment() & Qt.AlignRight, "the hint sits hard right, away from step 4"
# the QLabel pitfall: a longer hint must never resize the strip or refit the canvas
ob.set_state([True, False, False, False], "short")
app.processEvents()
_geom = (ob.minimumSizeHint().width(), ob.sizeHint().width(), ob.height())
for _h in ("", "Press S, then click the animal on the video", "x" * 400):
    ob.set_state([True, False, False, False], _h)
    app.processEvents()
    assert (ob.minimumSizeHint().width(), ob.sizeHint().width(), ob.height()) == _geom, \
        f"a {len(_h)}-char hint resized the onboarding strip"
assert ob.hint.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
assert ob.hint.text().startswith("ⓘ"), "the hint is marked as info, not a step"
# G5: a hint too long for the strip keeps its FIRST words (what to press) and ends in
# an ellipsis, the whole text in its tooltip -- it used to lose its start instead
from PySide6.QtWidgets import QLabel as _QL  # noqa: E402
from kinetrace.widgets import ElidedLabel  # noqa: E402
_long = "Press N and click the animal, then Track -- " + "and more words " * 40
assert isinstance(ob.hint, ElidedLabel), "the onboarding hint must elide, not clip"
_el = ElidedLabel("ⓘ  " + _long, pad=16)
_el.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
_el.show()                              # a hidden widget gets no resize events
_el.resize(260, 24)
app.processEvents()
_shown = _QL.text(_el)
assert _shown.startswith("ⓘ  Press N"), f"the hint lost its start: {_shown[:40]!r}"
assert _shown.endswith("…") and len(_shown) < len(_el.text()), "a cut hint ends in an ellipsis"
assert _el.toolTip() == _el.text() == "ⓘ  " + _long, "the whole hint is in the tooltip and text()"
_el.resize(20000, 24)
app.processEvents()
assert _QL.text(_el) == _el.text(), "a hint that fits is shown whole"
_el.deleteLater()
win._refresh_onboarding()
print("onboarding hint separated from the steps OK")

# ---- the manual is IN the app (Help → User Manual, F1) ----
from kinetrace.widgets import MANUAL_PATH

assert MANUAL_PATH.exists(), f"the manual must ship with the app: {MANUAL_PATH}"
win._show_manual()
app.processEvents()
_man = win._manual_dlg
assert _man.isVisible(), "F1 / Help → User Manual must open it"
_txt = _man.view.toPlainText()
assert len(_txt) > 10_000, f"the manual rendered only {len(_txt)} chars"
assert "**" not in _txt, "markdown leaked through unrendered"
# the sections a newcomer depends on must actually be there
for _needed in ("What this program does", "Your first tracking session",
                "Checking the result and fixing mistakes", "Glossary",
                "Getting your numbers out"):
    assert _needed in _txt, f"the manual is missing the '{_needed}' section"
assert _man.contents.count() >= 15, "the contents sidebar must list the sections"
# tables survive Qt's markdown reader (the export and hotkey tables are tables)
assert len(_man.view.document().rootFrame().childFrames()) >= 5, "tables did not render"
# jumping: the contents list and the manual's own links both scroll somewhere
_man.contents.setCurrentRow(_man.contents.count() - 1)
app.processEvents()
assert _man.view.verticalScrollBar().value() > 0, "the contents list must scroll the view"
_man.view.verticalScrollBar().setValue(0)
from PySide6.QtCore import QUrl

# Look the anchor up rather than hard-coding "#15-glossary": inserting a
# section renumbers every heading after it, and this assertion is about links
# working, not about the glossary happening to be section 15.
_glossary = next((s for s in _man._anchors if s.endswith("-glossary")), None)
assert _glossary, f"no glossary anchor among {sorted(_man._anchors)[:8]}…"
_man._on_anchor(QUrl("#" + _glossary))
app.processEvents()
assert _man.view.verticalScrollBar().value() > 0, "in-document links must jump"
_man._find_next()                       # empty box must be a no-op, not a crash
_man.find.setText("silhouette")
_man._find_next()
assert _man.view.textCursor().hasSelection(), "find must select a match"
_man.close()
app.processEvents()
print(f"in-app manual OK ({len(_txt):,} chars, {_man.contents.count()} sections)")

# ---- Settings lives on the Segment ▾ button, not in the View menu ----
# It configures the segmentation model, so it belongs next to the model choice.
# The window must still OWN the action or Ctrl+, would stop working (a QAction
# in a tool button's menu has no shortcut context of its own).
_view_entries = []
for _a in win.menuBar().actions():
    if _a.text() == "&View":
        _view_entries = [x.text() for x in _a.menu().actions()]
assert _view_entries, "the View menu must still exist"
assert not any("Settings" in t for t in _view_entries), \
    f"Settings must not be in the View menu any more: {_view_entries}"
assert any(x is win.act_settings for x in win._seg_menu.actions()), \
    "Settings must be the last entry of the Segment dropdown"
assert win.act_settings in win.actions(), \
    "the window must own the action, or Ctrl+, stops working once it leaves the menu"
assert win.act_settings.shortcut().toString() == "Ctrl+,"
print("Settings lives only on the Segment dropdown, Ctrl+, still bound OK")

# ---- lane bookkeeping updates on add; splitter resize grows visible lanes ----
win._on_add(100.0, 100.0)
win._on_add(300.0, 200.0)
app.processEvents()
assert tlp._laid_out_n == 2, "refresh must notice the lane-count change"
from PySide6.QtWidgets import QSplitter
split = win.centralWidget()
assert isinstance(split, QSplitter), "canvas/timeline must sit in a vertical splitter"
rows_before = tlp._max_rows()
total = sum(split.sizes())
split.setSizes([total - 420, 420])   # drag the handle up: more timeline space
app.processEvents()
assert tlp._max_rows() > rows_before, \
    f"resizing the panel must add visible lanes ({rows_before} -> {tlp._max_rows()})"
split.setSizes([total - 200, 200])
app.processEvents()
print("splitter-resizable timeline OK")
L = 200
fake = np.tile(np.array([[150.0, 150.0], [310.0, 210.0]], np.float32), (L, 1, 1))
conf = np.ones((L, 2), np.float32)
conf[100:140, 1] = 0.2  # a low-confidence stretch for point 2
s.write_segment(0, fake, np.ones((L, 2), bool), [0, 1], conf)
app.processEvents()

# ---- timeline click -> seek (pixel <-> frame mapping) ----
tlp_w = tlp._lane_w()
target = 300
x = tlp._x_of(target)
ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(x, EVENTS_H + 8), QPointF(x, EVENTS_H + 8),
                 Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
tlp.mousePressEvent(ev)
app.processEvents()
assert abs(win.current - target) <= 1, f"timeline seek landed at {win.current}, wanted ~{target}"
assert tlp._frame_at(tlp._x_of(0)) == 0 and tlp._frame_at(tlp._x_of(599)) == 599
print("timeline click-to-seek OK")

# ---- timeline gutter click selects the point ----
tlp.point_selected.emit(1)
app.processEvents()
assert win.selected == 1
print("timeline point select OK")

# ---- E-key two-press event flow ----
win._goto(120)
key(Qt.Key_E)
assert win._pending_event == 120 and tlp.pending_event == 120
win._goto(180)
key(Qt.Key_E)  # QInputDialog monkeypatched -> "swing"
assert win._pending_event is None and len(s.events) == 1
e = s.events[0]
assert (e.name, e.start, e.end) == ("swing", 120, 180)
# Esc cancels a pending mark
key(Qt.Key_E)
assert win._pending_event is not None
key(Qt.Key_Escape)
assert win._pending_event is None and len(s.events) == 1
# reversed marking swaps
win._goto(90)
key(Qt.Key_E)
win._goto(40)
_dialog_reply["text"] = "reversed"
key(Qt.Key_E)
assert s.events[1].start == 40 and s.events[1].end == 90
# events menu lists them (action for each event + the mark action)
win._refresh_events_ui()
labels = [a.text() for a in win.m_events.actions() if a.text()]
assert any("swing" in t for t in labels), labels
print("event E-flow OK")

# ---- event ribbon click jumps to its start ----
win._goto(0)
ex = tlp._x_of(150)  # inside "swing" [120-180]
ev2 = QMouseEvent(QEvent.MouseButtonPress, QPointF(ex, EVENTS_H / 2), QPointF(ex, EVENTS_H / 2),
                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
tlp.mousePressEvent(ev2)
app.processEvents()
assert win.current == 120, f"ribbon click must jump to event start, got {win.current}"
assert tlp.sel_range == (120, 180), "ribbon click must also SELECT the event window"
key(Qt.Key_Escape)
assert tlp.sel_range is None
# a SINGLE-FRAME event's ribbon (drawn >= 3 px) must be hittable as drawn
s.add_event("blip", 500, 500)
tlp.refresh()
bx = tlp._x_of(500) + 2.0  # inside the 3 px minimum ribbon, outside frame 500's own pixel? still hits
assert tlp._event_at(bx, EVENTS_H / 2) == len(s.events) - 1, \
    "short-event ribbon must hit-test in pixel space"
s.remove_event(len(s.events) - 1)
print("event ribbon jump OK (incl. single-frame ribbon hit)")

# ---- gestures must not survive interaction being disabled ----
c2 = win.canvas
c2._press_scene = QPointF(50, 50)
c2._press_view = QPointF(50, 50)
c2._g, c2._gbutton = "circle", Qt.LeftButton      # R5: one gesture state (was _circle_active / _dragging flags)
c2.set_interactive(False)
assert c2._press_scene is None and c2._g == "none", \
    "set_interactive(False) must cancel in-flight gestures"
c2.set_interactive(True)
print("gesture cancel on disable OK")

# ---- rename collision auto-suffix through the GUI paths ----
_dialog_reply["text"] = s.points[0].name  # rename P2 to P1's name via dialog
win._on_rename(1)
assert s.points[1].name == f"{s.points[0].name} (2)", s.points[1].name
item = win.point_list.item(1)
item.setText(s.points[0].name)  # rename via the list editor to a taken name
app.processEvents()
assert s.points[1].name.endswith("(2)") and s.points[0].name != s.points[1].name
assert win.point_list.item(1).text() == s.points[1].name, "list must show the applied name"
print("rename collision OK")

# ---- circle gesture handler -> group point ----
win._on_add_group(320.0, 240.0, 25.0)
g = s.n_points - 1
assert s.points[g].kind == "group" and s.points[g].radius == 25.0
assert s.tracked[win.current, g]
print("group gesture handler OK")

# ---- appearance-lock menu: enabled+checkable on points, disabled on groups ----
win._refresh_overlay()
menu_p, acts_p = win.canvas._build_context_menu(0)
assert acts_p["anchor"].isEnabled() and acts_p["anchor"].isCheckable()
assert not acts_p["anchor"].isChecked()
menu_g, acts_g = win.canvas._build_context_menu(g)
assert not acts_g["anchor"].isEnabled(), \
    "groups must show the item disabled (with the why), not hide it"
win.canvas.anchor_toggled.emit(0, True)
assert s.points[0].anchor is True, "anchor toggle must reach the session"
_, acts_p2 = win.canvas._build_context_menu(0)
assert acts_p2["anchor"].isChecked(), "menu must reflect the current anchor state"
print("anchor menu OK")

# ---- marker size: applies to markers, scales hit test, persists ----
win.marker_spin.setValue(3)
assert win.canvas._marker_radius == 3.0
assert win.canvas._markers[0].rect().width() == 6.0, "existing markers must resize"
win.marker_spin.setValue(20)
assert win.canvas._markers[0].rect().width() == 40.0
from PySide6.QtCore import QPointF as _QP
c_hit = win.canvas
c_hit.fit()
app.processEvents()
assert c_hit._hit_test(_QP(*s.tracks[win.current, 0])) == 0, \
    "big markers must stay grabbable at their edge"
win.marker_spin.setValue(4)
print("marker size OK")

# ---- EXACT state restore through save/load ----
win._goto(321)
win._on_select(1)
win.btn_follow.setChecked(False)
win.btn_autopause.setChecked(False)
win.btn_roi.setChecked(True)
c.scale(2.5, 2.5)
c._user_zoomed = True
c.centerOn(200, 150)
app.processEvents()
view_before = c.view_state()
proj = os.path.join(SCRATCH, "state_restore.kinetrace")
win.project_path = None
win._sync_ui_state()
s.save(proj)

old_session = s
win._open_project_from_path(proj)
pump(lambda: win.session is not old_session and win.state == READY, 30,
     "project reopen (session swap)")
pump(lambda: win.current == 321, 10, "frame restore")
app.processEvents()  # let the queued view restore run
s2 = win.session
assert win.current == 321, "frame restore"
assert win.selected == 1, "selection restore"
assert win.btn_follow.isChecked() is False, "follow toggle restore"
assert win.btn_autopause.isChecked() is False, "autopause toggle restore"
assert win.btn_roi.isChecked() is True, "roi toggle restore"
assert win.marker_spin.value() == 4 and win.canvas._marker_radius == 4.0, \
    "marker size restore"
assert len(s2.events) == 2 and s2.events[0].name == "swing", "events restore"
assert s2.points[g].kind == "group" and s2.points[g].radius == 25.0, "group restore"
assert s2.points[0].anchor is True, "appearance-lock flag restore"
view_after = win.canvas.view_state()
assert abs(view_after["zoom"] - view_before["zoom"]) < 1e-6, \
    f"zoom restore: {view_after['zoom']} vs {view_before['zoom']}"
assert abs(view_after["center_x"] - view_before["center_x"]) < 2.0
assert abs(view_after["center_y"] - view_before["center_y"]) < 2.0
print(f"exact state restore OK (frame {win.current}, sel {win.selected}, "
      f"zoom {view_after['zoom']:.2f})")

# ---- events survive into exports ----
mat_path = os.path.join(SCRATCH, "tl_events.mat")
s2.export_mat(mat_path)
from scipy.io import loadmat
m = loadmat(mat_path)
assert [str(n[0]) for n in m["event_names"].squeeze(0)] == ["swing", "reversed"], \
    m["event_names"]
s2.export_events_csv(os.path.join(SCRATCH, "tl_events.csv"))
lines = open(os.path.join(SCRATCH, "tl_events.csv")).read().splitlines()
assert lines[1].startswith("swing,120,180") and lines[2].startswith("reversed,40,90")
print("event exports OK")

# (the project was reloaded above: rebind to the live session)
s = win.session

# ---- time zoom: +/- over the panel, mapping stays exact, clamps at full ----
tlp.zoom_time(4.0)
v0, v1 = tlp._view
assert (v1 - v0) < 599, "zoom_time must narrow the view span"
for f in (v0, (v0 + v1) // 2, v1):
    assert tlp._frame_at(tlp._x_of(f)) == f, "frame<->pixel mapping must stay exact"
tlp.zoom_time(1 / 100.0)
assert tlp._view == (0, 599), "zooming far out must clamp to the full video"
print("time zoom OK")

# ---- frame-window selection: shift+drag, event window, Esc ----
sx0, sx1 = tlp._x_of(50), tlp._x_of(80)
tlp.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(sx0, EVENTS_H + 8),
                                QPointF(sx0, EVENTS_H + 8), Qt.LeftButton,
                                Qt.LeftButton, Qt.ShiftModifier))
tlp.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(sx1, EVENTS_H + 8),
                               QPointF(sx1, EVENTS_H + 8), Qt.NoButton,
                               Qt.LeftButton, Qt.ShiftModifier))
tlp.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(sx1, EVENTS_H + 8),
                                  QPointF(sx1, EVENTS_H + 8), Qt.LeftButton,
                                  Qt.NoButton, Qt.NoModifier))
assert tlp.sel_range is not None
r0, r1 = tlp.sel_range
assert abs(r0 - 50) <= 1 and abs(r1 - 80) <= 1, f"shift+drag selected {tlp.sel_range}"
key(Qt.Key_Escape)
assert tlp.sel_range is None, "Esc must clear the frame-window selection"
tlp.select_event_window(0)   # "swing" = [120, 180]
assert tlp.sel_range == (120, 180), "event window must be selectable for bulk ops"

# ---- bulk clear: Delete blanks the window for the panel-selected points ----
win.point_list.item(0).setSelected(True)
win.point_list.item(1).setSelected(False)
assert win._selected_pids() == [0]
assert s.tracked[150, 0] and s.tracked[150, 1]
key(Qt.Key_Delete)
assert not s.tracked[120:181, 0].any(), "window must be cleared for the selected point"
assert s.tracked[120:181, 1].all(), "unselected point must keep its data"
assert s.tracked[100, 0] and s.tracked[190, 0], "outside the window stays intact"
assert np.isnan(s.tracks[150, 0]).all() and s.confidence[150, 0] == 0
assert tlp.sel_range is None, "selection consumed by the clear"
assert win.act_undo.isEnabled()
win._undo_run()
assert s.tracked[150, 0], "Ctrl+Z must restore the cleared window"
print("frame-window bulk clear + undo OK")

# ---- bulk point deletion: multi-select + Delete removes them, undo restores ----
n0 = s.n_points
win.point_list.item(0).setSelected(True)
win.point_list.item(1).setSelected(True)
assert win._selected_pids() == [0, 1]
key(Qt.Key_Delete)   # QMessageBox.question monkeypatched -> Yes
assert s.n_points == n0 - 2, "both selected points must be deleted"
win._undo_run()
assert s.n_points == n0, "Ctrl+Z must restore deleted points"
print("bulk point deletion + undo OK")

# ---- event TYPES: repeat occurrences share color; menu groups them ----
win._goto(300)
key(Qt.Key_E)
win._goto(340)
_dialog_reply["text"] = "swing"          # getItem path: existing names offered
key(Qt.Key_E)
assert len(s.events) == 3
assert s.events[2].name == "swing" and s.events[2].color == s.events[0].color, \
    "repeat occurrence must reuse the event type's color"
assert s.events[1].color != s.events[0].color, "distinct types keep distinct colors"
win._refresh_events_ui()
subs = [a for a in win.m_events.actions() if a.menu() is not None]
assert any("swing" in a.text() and "2" in a.text() for a in subs), \
    f"multi-occurrence types must group into a submenu: {[a.text() for a in win.m_events.actions()]}"
# ribbon tooltip knows its occurrence index (via the same-name lookup)
assert tlp._event_at(tlp._x_of(320), EVENTS_H / 2) == 2
s.remove_event(2)
win._refresh_events_ui()
print("event types / occurrences OK")

# ---- Premiere-style ruler + scroll strip ----
for span, w_px in ((40000, 1200), (600, 1100), (37, 900), (5000, 300)):
    st = tlp._tick_step(span, w_px)
    assert str(st)[0] in "125" and set(str(st)[1:]) <= {"0"}, \
        f"tick step must be 1/2/5x10^k: {st} (span {span})"
    assert st * w_px / span >= 45, f"ticks too dense: {st} for span {span}"
tlp.zoom_time(8.0)
assert tlp._zoomed()
v0a, v1a = tlp._view
span_a = v1a - v0a
cur_before = win.current
strip_y = tlp.height() - 3                  # inside the scroll strip zone
target_x = GUTTER_W + tlp._lane_w() * 0.8   # strip maps the WHOLE video: 80% in
tlp.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(target_x, strip_y),
                                QPointF(target_x, strip_y), Qt.LeftButton,
                                Qt.LeftButton, Qt.NoModifier))
tlp.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(target_x, strip_y),
                                  QPointF(target_x, strip_y), Qt.LeftButton,
                                  Qt.NoButton, Qt.NoModifier))
v0b, v1b = tlp._view
assert (v1b - v0b) == span_a, "strip drag must pan, not zoom"
mid = (v0b + v1b) / 2
assert abs(mid - 0.8 * 599) <= span_a, f"strip click must center the view there (mid {mid})"
assert win.current == cur_before, "the strip moves the VIEW, never the playhead"
tlp.zoom_time(1 / 100.0)
assert not tlp._zoomed()
print("adaptive ruler + scroll strip OK")

# ---- lane marquee: Shift+drag picks frames AND lanes; Delete clears just those
# (the segment lane is fabricated here — no GPU, no segmenter run) ----
from kinetrace.timeline import ANIMAL_H, LANE_H

s.ensure_animal()
_m = np.zeros((480, 640), bool)
_m[200:300, 250:400] = True
for _f in range(400, 501):
    s.masks.set(_f, _m, 8.0)
_L2 = 101
s.write_segment(400, np.tile(np.array([[150.0, 150.0], [310.0, 210.0]], np.float32),
                             (_L2, 1, 1)),
                np.ones((_L2, 2), bool), [0, 1], np.ones((_L2, 2), np.float32))
win._refresh_point_list()
win._refresh_animal_panel()
tlp.refresh()
tlp.resize(1200, EVENTS_H + ANIMAL_H + LANE_H * 6 + 6)
app.processEvents()
assert tlp._animal_h() == ANIMAL_H, "the segment lane must appear once an animal exists"
assert s.masks.n_masked() == 101 and s.tracked[450, 0] and s.tracked[450, 1]


def marquee(f0, f1, y0, y1):
    """Shift+drag from (frame f0, y0) to (frame f1, y1)."""
    a, b = QPointF(tlp._x_of(f0), y0), QPointF(tlp._x_of(f1), y1)
    tlp.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, a, a, Qt.LeftButton,
                                    Qt.LeftButton, Qt.ShiftModifier))
    tlp.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, b, b, Qt.NoButton,
                                   Qt.LeftButton, Qt.ShiftModifier))
    tlp.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, b, b, Qt.LeftButton,
                                      Qt.NoButton, Qt.NoModifier))
    app.processEvents()


seg_y = EVENTS_H + 2 + ANIMAL_H // 2
row_y = [tlp._row_rect_y(r)[0] + LANE_H // 2 for r in range(3)]

# 1. along the SEGMENT lane -> silhouettes only, points untouched
win.point_list.clearSelection()          # the panel must not leak into this
marquee(420, 440, seg_y, seg_y)
assert tlp.sel_seg and tlp.sel_rows == [], \
    f"segment-lane drag must target the silhouettes alone: {tlp.sel_rows}, {tlp.sel_seg}"
assert "silhouettes" in tlp._sel_hint()
key(Qt.Key_Delete)
assert s.masks.n_masked() == 101 - 21, "the selected silhouettes must be gone"
assert not s.masks.has(430) and s.masks.has(410) and s.masks.has(450)
assert s.tracked[420:441, :2].all(), "a segment-lane delete must not touch points"
assert tlp.sel_range is None, "the selection is consumed by the delete"
win._undo_run()
assert s.masks.n_masked() == 101, "Ctrl+Z must restore the silhouettes"

# 2. along ONE point lane -> that point only, even though the panel says otherwise
win.point_list.item(1).setSelected(True)   # panel selection must lose to the marquee
marquee(420, 440, row_y[0], row_y[0])
assert tlp.sel_rows == [0] and not tlp.sel_seg, f"point-lane drag: {tlp.sel_rows}"
key(Qt.Key_Delete)
assert not s.tracked[420:441, 0].any(), "the dragged lane must be cleared"
assert s.tracked[420:441, 1].all(), "a lane the drag missed must keep its data"
assert s.masks.has(430), "a point-lane delete must not touch the silhouettes"
win._undo_run()
assert s.tracked[430, 0], "Ctrl+Z must restore the cleared track"

# 3. across BOTH lanes -> both cleared, and ONE undo brings both back
win.point_list.clearSelection()
marquee(420, 440, seg_y, row_y[1])
assert tlp.sel_seg and tlp.sel_rows == [0, 1], f"spanning drag: {tlp.sel_rows}"
key(Qt.Key_Delete)
assert not s.masks.has(430) and not s.tracked[420:441, :2].any(), \
    "a drag over both must clear both"
assert s.masks.has(450) and s.tracked[450, 0], "outside the window stays intact"
win._undo_run()
assert s.masks.has(430) and s.tracked[430, 0] and s.tracked[430, 1], \
    "one Ctrl+Z must restore points AND silhouettes"

# 4. in the RULER -> lanes unspecified, so Delete falls back to the point panel
win.point_list.item(0).setSelected(True)
win.point_list.item(1).setSelected(False)
marquee(420, 440, EVENTS_H / 2, EVENTS_H / 2)
assert tlp.sel_rows is None and not tlp.sel_seg, "a ruler drag must not name lanes"
key(Qt.Key_Delete)
assert not s.tracked[420:441, 0].any() and s.tracked[420:441, 1].all(), \
    "the panel selection decides when the drag named no lane"
assert s.masks.has(430), "the fallback path must never touch the silhouettes"
win._undo_run()

# 5. the band is painted over the covered lanes only, and an event window still
#    selects the whole panel height
marquee(420, 440, seg_y, seg_y)
by0, by1 = tlp._sel_band_y()
assert by0 == EVENTS_H + 2 and by1 == EVENTS_H + 2 + ANIMAL_H, \
    f"the band must cover the segment lane alone: {by0}-{by1}"
tlp.select_event_window(0)
assert tlp.sel_rows is None and tlp._sel_band_y() == (0.0, float(tlp.height()))
key(Qt.Key_Escape)
assert tlp.sel_range is None

# 6. the REAL key path: a drag leaves focus on the panel, so Delete must
#    propagate from the panel up to the window (the panel keeps only its zoom keys)
win.point_list.clearSelection()
tlp.setFocus()
marquee(420, 440, seg_y, seg_y)
app.sendEvent(tlp, QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
app.processEvents()
assert not s.masks.has(430), "Delete must reach the window while the panel has focus"
win._undo_run()
assert s.masks.has(430)
win.point_list.clearSelection()
print("timeline lane marquee (segment / points / both) OK")

# ---- timeline aggregation stays fast at 40k frames ----
import time as _t
from kinetrace.session import TrackingSession
big = TrackingSession("x.mp4", 40000, 30.0, 3840, 2160)
for i in range(10):
    big.add_point(0, 100.0 * i + 10, 100.0)
big.write_segment(0, np.random.rand(40000, 10, 2).astype(np.float32) * 800 + 100,
                  np.ones((40000, 10), bool), list(range(10)))
tlp.set_session(big)
tlp.resize(1200, tlp.sizeHint().height())
t0 = _t.perf_counter()
tlp._ensure_columns()
dt = _t.perf_counter() - t0
t0 = _t.perf_counter()
tlp._ensure_columns()  # cached
dt2 = _t.perf_counter() - t0
print(f"40k x 10 column aggregation: {dt * 1000:.1f} ms cold, {dt2 * 1000:.3f} ms cached")
assert dt < 0.25 and dt2 < 0.005, "timeline aggregation too slow"

# the background torch-import probe must finish before interpreter teardown
# (this test runs no tracking, so nothing else waits on it)
win._dev_probe.wait(180_000)
win.close()
app.processEvents()
forget_recovery(VID)
print("TIMELINE+EVENTS PASSED")
