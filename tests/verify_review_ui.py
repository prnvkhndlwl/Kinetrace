"""Code review 2026-10-02, the canvas / timeline / widgets / viewgrid / camerapanel /
folderimport fixes (G65, I199, I216, I217, G85-G90, G127-G131, R4-R6), each with a
check that FAILS on the old code. Real mouse / key / wheel events (QTest) on bare
widgets, no video decode, no GPU.

`KT_PKG_ROOT=<folder>` runs the same checks against another tree's `kinetrace`
package (how "fails on old" was proved: a `git archive` of the base commit). Each
section reports its own failures and the script goes on, so one run on the old
code lists everything it gets wrong."""
import os
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("KT_PKG_ROOT", ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QColor, QContextMenuEvent, QMouseEvent, QTextDocument, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton, QWidget

app = QApplication([])
EXC: list = []
sys.excepthook = lambda t, v, tb: EXC.append("".join(traceback.format_exception(t, v, tb))[-400:])

from kinetrace import theme  # noqa: E402
from kinetrace.canvas import VideoCanvas  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond, detail="") -> None:
    print(("ok    " if cond else "FAIL  ") + name + ("" if cond else f"   [{detail}]"))
    if not cond:
        FAILS.append(name)


def section(fn):
    """Run one group of checks; an exception is a failure of the group (old code).
    `KT_SKIP=name,name` leaves groups out (the old code's I217 paint error leaves a
    broken QPainter behind that aborts the process later)."""
    only = [n for n in os.environ.get("KT_ONLY", "").split(",") if n]
    if fn.__name__ in os.environ.get("KT_SKIP", "").split(",") or (only and fn.__name__ not in only):
        print(f"skip  {fn.__name__}")
        return fn
    try:
        fn()
    except Exception as e:      # noqa: BLE001
        FAILS.append(fn.__name__)
        print(f"FAIL  {fn.__name__}: raised {type(e).__name__}: {str(e)[:160]}")
    return fn


def pump(ms=0):
    if ms:
        QTest.qWait(ms)
    app.processEvents()


# ----------------------------------------------------------------- canvas helpers

def mk_canvas(w=200, h=150, size=(400, 300)):
    c = VideoCanvas()
    c.resize(*size)
    c.show()
    c.set_video_size(w, h)
    c.set_frame(np.full((h, w, 3), 255, np.uint8))
    pump()
    return c


def vp_pos(c, x, y) -> QPoint:
    return c.mapFromScene(QPointF(float(x), float(y)))


def ev_move(pos, buttons=Qt.NoButton):
    return QMouseEvent(QEvent.MouseMove, QPointF(pos), QPointF(pos), Qt.NoButton, buttons, Qt.NoModifier)


def drag(c, a, b, mods=Qt.NoModifier):
    vp = c.viewport()
    QTest.mousePress(vp, Qt.LeftButton, mods, a)
    QTest.mouseMove(vp, a + QPoint(12, 12))
    QTest.mouseMove(vp, b)
    QTest.mouseRelease(vp, Qt.LeftButton, mods, b)
    pump()


def wheel(dx, dy, mods=Qt.NoModifier, pos=QPointF(100, 100)):
    return QWheelEvent(pos, pos, QPoint(0, 0), QPoint(dx, dy), Qt.NoButton, mods, Qt.NoScrollPhase, False)


# ------------------------------------------------------------------------ G65

@section
def g65_second_click():
    c = mk_canvas()
    got = []
    c.annotate_requested.connect(lambda x, y: got.append((x, y)))
    QTest.mouseDClick(c.windowHandle(), Qt.LeftButton, Qt.NoModifier, vp_pos(c, 100, 70))
    pump()
    check("G65 a double click (press, release, double-click, release) places twice", len(got) == 2, got)
    # the polygon double-click still closes the polygon, with the corners clicked
    c.set_place_mode(True)
    c.set_region_shape("polygon")
    regs = []
    c.region_requested.connect(lambda kind, pts: regs.append((kind, pts)))
    wh = c.windowHandle()
    for p in ((30, 30), (170, 40), (150, 120)):
        QTest.mouseClick(wh, Qt.LeftButton, Qt.NoModifier, vp_pos(c, *p))
    pump()
    check("G65 polygon corners wait for the close", c.polygon_in_progress() and not regs, regs)
    QTest.mouseDClick(wh, Qt.LeftButton, Qt.NoModifier, vp_pos(c, 40, 125))
    pump()
    check("G65 a polygon double-click closes it with every corner clicked",
          len(regs) == 1 and regs[0][0] == "polygon" and len(regs[0][1]) == 4 and not c.polygon_in_progress(),
          regs)


# ----------------------------------------------------------------------- I216

@section
def i216_clamped_release():
    c = mk_canvas(200, 150, (700, 500))          # the picture is letterboxed: room to drag past its edge
    c.set_place_mode(True)
    c.set_region_shape("rect")
    regs = []
    c.region_requested.connect(lambda kind, pts: regs.append(pts))
    a = vp_pos(c, 150, 100)
    far = QPoint(698, a.y() + 60)
    check("I216 the release point is beyond the picture", c.mapToScene(far).x() > 199.5, c.mapToScene(far).x())
    drag(c, a, far)
    ok = len(regs) == 1 and all(0.0 <= x <= 199.5 and 0.0 <= y <= 149.5 for x, y in regs[0])
    check("I216 a rectangle dragged past the edge ends AT the edge", ok, regs)
    # the segment box
    c.set_place_mode(False)
    c.set_animal_mode(True)
    boxes = []
    c.animal_box.connect(lambda *b: boxes.append(b))
    drag(c, vp_pos(c, 100, 60), QPoint(698, vp_pos(c, 100, 120).y()))
    ok = len(boxes) == 1 and boxes[0][2] <= 199.5 and boxes[0][3] <= 149.5
    check("I216 the segment box dragged past the edge ends AT the edge", ok, boxes)
    # a clean armed click and an armed circle drag still work
    c.set_animal_mode(False)
    c.set_place_mode(True)
    c.set_region_shape("circle")
    adds, groups = [], []
    c.add_requested.connect(lambda x, y: adds.append((x, y)))
    c.group_requested.connect(lambda x, y, r: groups.append((x, y, r)))
    QTest.mouseClick(c.viewport(), Qt.LeftButton, Qt.NoModifier, vp_pos(c, 60, 60))
    c.set_place_mode(True)
    drag(c, vp_pos(c, 100, 75), vp_pos(c, 140, 75))
    check("armed click adds, armed drag draws a circle group", len(adds) == 1 and len(groups) == 1
          and abs(groups[0][2] - 40.0) < 1.5, (adds, groups))


# ------------------------------------------------------------------------ G86

@section
def g86_shift_at_press():
    c = mk_canvas()
    c.set_animal_mode(True)
    clicks = []
    c.animal_click.connect(lambda x, y, pos: clicks.append(pos))
    p = vp_pos(c, 80, 60)
    vp = c.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.ShiftModifier, p)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, p)          # Shift let go a moment early
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, p)
    check("G86 Shift held at the press = a 'not the segment' click, even if released early",
          clicks == [False, True], clicks)


# ------------------------------------------------------------------------ G87

@section
def g87_lost_release():
    c = mk_canvas(200, 150, (400, 300))
    c.scale(4.0, 4.0)
    c._user_zoomed = True
    pump()
    hb = c.horizontalScrollBar()
    vp = c.viewport()
    centre = QPoint(200, 150)
    # (a) the pan tool: a press, a drag (pans), then the release never arrives
    hb.setValue(hb.maximum() // 2)
    h_start = hb.value()
    c.set_pan_mode(True)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, centre)
    QTest.mouseMove(vp, centre + QPoint(-40, 0))
    h1 = hb.value()
    check("G87 (setup) the pan drag pans", h1 != h_start, (h_start, h1))
    c.mouseMoveEvent(ev_move(centre + QPoint(-90, 0)))               # no button held any more
    h2 = hb.value()
    c.mouseMoveEvent(ev_move(centre + QPoint(-140, 0)))
    check("G87 a pan whose button is no longer held stops", hb.value() == h2, (h1, h2, hb.value()))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, centre)
    c.set_pan_mode(False)
    # (b) an unarmed press whose release is lost must not turn into a pan
    pump()
    hb.setValue(hb.maximum() // 2)
    h0 = hb.value()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, centre)
    c.mouseMoveEvent(ev_move(centre + QPoint(60, 0)))                # far beyond the drag threshold, no button
    c.mouseMoveEvent(ev_move(centre + QPoint(90, 0)))
    check("G87 a plain press whose release was lost does not pan", hb.value() == h0, (h0, hb.value()))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, centre)
    # (c) cancel_gesture (Esc) ends a pan
    c.set_pan_mode(True)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, centre)
    QTest.mouseMove(vp, centre + QPoint(-20, 0))
    c.cancel_gesture()
    h3 = hb.value()
    c.mouseMoveEvent(ev_move(centre + QPoint(-80, 0), Qt.LeftButton))
    check("G87 cancel_gesture ends a pan", hb.value() == h3, (h3, hb.value()))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, centre)
    c.set_pan_mode(False)


@section
def g87_long_press_refused_with_left_down():
    c = mk_canvas(200, 150, (400, 300))
    metas = [SimpleNamespace(name="a", color=(255, 0, 0), display=True)]
    c.set_points(np.array([[100.0, 75.0]], np.float32), np.ones(1, bool), metas, None)
    menus = []
    c._context_menu = lambda pid, pos: menus.append(pid)
    wh = c.windowHandle()
    on = vp_pos(c, 100, 75)
    off = vp_pos(c, 40, 40)
    QTest.mousePress(wh, Qt.RightButton, Qt.NoModifier, on)
    pump(650)
    QTest.mouseRelease(wh, Qt.RightButton, Qt.NoModifier, on)
    check("G87 (setup) a long right press on a marker opens its menu", menus == [0], menus)
    menus.clear()
    QTest.mousePress(wh, Qt.RightButton, Qt.NoModifier, on)
    QTest.mousePress(wh, Qt.LeftButton, Qt.NoModifier, off)          # the left button goes down meanwhile
    pump(650)
    check("G87 the long right-press menu refuses while the left button is down", menus == [], menus)
    QTest.mouseRelease(wh, Qt.LeftButton, Qt.NoModifier, off)
    QTest.mouseRelease(wh, Qt.RightButton, Qt.NoModifier, on)
    pump()


# ------------------------------------------------------------------------ G91

@section
def g91_pan_tool_right_click():
    c = mk_canvas(200, 150, (400, 300))
    metas = [SimpleNamespace(name="a", color=(255, 0, 0), display=True)]
    c.set_points(np.array([[100.0, 75.0]], np.float32), np.ones(1, bool), metas, None)
    cleared = []
    c.clear_frame_requested.connect(lambda pid: cleared.append(pid))
    on = vp_pos(c, 100, 75)
    QTest.mouseClick(c.viewport(), Qt.RightButton, Qt.NoModifier, on)
    check("(setup) a right click on a marker clears it on this frame", cleared == [0], cleared)
    cleared.clear()
    c.set_pan_mode(True)
    QTest.mouseClick(c.viewport(), Qt.RightButton, Qt.NoModifier, on)
    check("G91 with the pan tool on, a right click on a marker does nothing to it", cleared == [], cleared)
    c.set_pan_mode(False)


# ------------------------------------------------------------------------ G85

def _img(c):
    return c._loupe.pixmap().toImage()


@section
def g85_loupe_redraws():
    c = mk_canvas(200, 150, (400, 300))
    c.set_loupe(True)
    rng = np.random.default_rng(1)
    c.set_frame(rng.integers(0, 255, (150, 200, 3), dtype=np.uint8))
    QTest.mouseMove(c.viewport(), QPoint(5, 5))          # (a move to where the pointer already is sends nothing)
    QTest.mouseMove(c.viewport(), vp_pos(c, 100, 75))
    pump()
    check("G85 (setup) the loupe shows over the video", c._loupe.isVisible(),
          (getattr(c, "_cursor_view", "n/a"), c._interactive, c._loupe_enabled, c._native_size))
    a = _img(c)
    c.set_frame(rng.integers(0, 255, (150, 200, 3), dtype=np.uint8))          # F: a new frame, the cursor still
    b = _img(c)
    check("G85 a new frame redraws the loupe under a still cursor", a != b)
    c.set_display_filter("bright")
    d = _img(c)
    check("G85 a display filter redraws it", b != d)
    c.zoom_step(1.25)
    e = _img(c)
    check("G85 zoom redraws it", d != e)
    c.set_interactive(False)
    check("G85 the loupe hides when interaction is revoked", not c._loupe.isVisible())
    c.set_interactive(True)
    c.set_display_filter("none")
    # near the picture edge the crop is padded, the crosshair stays on the true pixel
    c.fit()
    c.set_frame(np.full((150, 200, 3), 255, np.uint8))
    QTest.mouseMove(c.viewport(), QPoint(5, 5))
    QTest.mouseMove(c.viewport(), vp_pos(c, 1, 1))
    pump()
    img = _img(c)
    px = QColor(img.pixel(10, 100))
    check("G85 outside the picture the loupe is black, not a stretched edge",
          c._loupe.isVisible() and (px.red(), px.green(), px.blue()) == (0, 0, 0),
          (px.red(), px.green(), px.blue()))


# ------------------------------------------------------------------------ G129

@section
def g129_bones_to_hidden_points():
    c = mk_canvas()
    metas = [SimpleNamespace(name="a", color=(255, 0, 0), display=True),
             SimpleNamespace(name="b", color=(0, 255, 0), display=False)]
    pos = np.array([[50.0, 50.0], [100.0, 100.0]], np.float32)
    c.set_points(pos, np.ones(2, bool), metas, None)
    c.set_bones([(0, 1)])
    check("G129 no bone to a landmark hidden in POINTS", c._bones_item.path().isEmpty())
    metas[1].display = True
    c.set_points(pos, np.ones(2, bool), metas, None)
    check("G129 the bone is back when it is shown", not c._bones_item.path().isEmpty())


# ------------------------------------------------------------------------ G131

@section
def g131_canvas_wheel():
    c = mk_canvas()
    z0 = c.transform().m11()
    c.scale(1.5, 1.5)
    z1 = c.transform().m11()
    c.wheelEvent(wheel(120, 0))
    check("G131 a horizontal swipe does not zoom the video", abs(c.transform().m11() - z1) < 1e-9,
          (z0, z1, c.transform().m11()))
    c.wheelEvent(wheel(0, 120))
    z2 = c.transform().m11()
    check("G131 one notch zooms one step", abs(z2 / z1 - 1.25) < 1e-6, z2 / z1)
    for _ in range(3):
        c.wheelEvent(wheel(0, -40))
    check("G131 small touchpad deltas add up to one step", abs(c.transform().m11() / z2 - 1 / 1.25) < 1e-6,
          c.transform().m11() / z2)


@section
def r4_dead_drag_code_gone():
    c = VideoCanvas()
    gone = [n for n in ("_dragging", "_drag_moved", "_panning", "_plain_press", "_trails", "update_marker")
            if hasattr(c, n)]
    check("R4 the pre-G59 marker-drag code is gone", not gone, gone)
    check("R4 the two signals nobody emits (and the app no longer connects) are gone",
          not hasattr(c, "point_moved") and not hasattr(c, "move_committed"))
    check("R4 Follow is off by default on a bare canvas", c._follow_enabled is False)


# ----------------------------------------------------------------- timeline

from kinetrace.session import TrackingSession  # noqa: E402
from kinetrace import timeline as tm  # noqa: E402
from kinetrace.timeline import EVENTS_H, GUTTER_W, TimelinePanel  # noqa: E402


def mk_session(n_frames=100, n_points=1):
    s = TrackingSession("x.mp4", n_frames, 30.0, 640, 480)
    for i in range(n_points):
        s.add_point(0, 10.0 + i, 10.0)
    return s


@section
def i217_one_frame_video():
    tl = TimelinePanel()
    tl.resize(600, 120)
    s1 = mk_session(1, 1)
    EXC.clear()
    tl.set_session(s1)
    tl.grab()
    check("I217 a one-frame video with a point paints without an error", not EXC, EXC)
    check("I217 its view is the one frame", tl._view == (0, 0), tl._view)
    tl.zoom_time(2.0)
    tl.zoom_fit()
    tl.grab()
    check("I217 zoom on a one-frame video is harmless", not EXC and tl._view == (0, 0) and tl._frame_at(1e9) == 0,
          (EXC, tl._view))
    tl.set_session(None)
    tl.grab()
    check("I217 no session paints", not EXC, EXC)


@section
def g88_menu_hit_order():
    s = mk_session(100, 1)
    s.add_event("swing", 20, 40)
    s.set_note(70, "a note")
    tl = TimelinePanel()
    tl.resize(800, 200)
    tl.set_session(s)
    seen = []
    saved = tm._pop
    tm._pop = lambda menu, pos: seen.append([a.text() for a in menu.actions() if not a.isSeparator()])

    def right_click(x, y):
        seen.clear()
        p = QPoint(int(x), int(y))
        tl.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse, p, p))
        return seen[0] if seen else []
    try:
        tl.select_event_window(0)                           # what a click on the event does
        menu = right_click(tl._x_of(30), EVENTS_H // 2)
        check("G88 right click on a selected event: its own menu (rename / delete), not the bulk clear",
              any(t.startswith("Rename event") for t in menu) and not any(t.startswith("Clear") for t in menu),
              menu)
        tl.sel_range = (60, 80)
        menu = right_click(tl._x_of(70), EVENTS_H - 3)
        check("G88 right click on a note inside the selection: the note menu",
              any(t.startswith("Edit note at frame 70") for t in menu), menu)
        menu = right_click(tl._x_of(65), tl._lanes_y0() + 3)
        check("G88 inside the selection (not on a note / event) the bulk menu still comes",
              any("Extend selection" in t for t in menu), menu)
        menu = right_click(tl._x_of(90), EVENTS_H // 2)
        check("G88 the bare ruler offers a note", any(t.startswith("Add a note at frame 90") for t in menu), menu)
    finally:
        tm._pop = saved


@section
def g90_scroll_clamped():
    s = mk_session(100, 25)
    tl = TimelinePanel()
    tl.resize(600, 120)
    tl.show()
    tl.set_session(s)
    pump()
    tl._scroll = 20
    tl.resize(600, 600)
    pump()
    tl.grab()
    check("G90 enlarging the panel pulls the scroll back (no lane hidden above)",
          tl._scroll == max(0, 25 - tl._max_rows()), (tl._scroll, tl._max_rows()))
    tl.close()


@section
def r6_timeline_geometry():
    s = mk_session(100, 1)
    tl = TimelinePanel()
    tl.resize(800, 120)
    tl.set_session(s)
    check("R6 set_session shows the whole video", tl._view == (0, 99), tl._view)
    tl._set_view(10, 15)                                    # 16 cells: frames 10..25
    check("R6 _frame_at stays inside the view", tl._frame_at(1e9) == 25 and tl._frame_at(-1e9) == 10,
          (tl._frame_at(1e9), tl._frame_at(-1e9)))
    tl._pan_time(-tl._lane_w())                             # a lane width = the 16 cells shown
    check("R6 panning by a lane width moves span + 1 cells", tl._view == (26, 41), tl._view)
    check("R6 disagree_threshold: 5 px at 1920, scaled for a wider picture",
          tm.disagree_threshold(1920) == 5.0 and tm.disagree_threshold(3840) == 10.0
          and tm.disagree_threshold(640) == 5.0)
    s2 = TrackingSession("y.mp4", 50, 30.0, 3840, 2160)
    s2.add_point(0, 1.0, 1.0)
    tl.set_session(s2)
    tl.set_disagreement(np.zeros((50, 1), np.float32))
    check("R6 the panel's default threshold is that rule", tl._disagree_px == 10.0, tl._disagree_px)


@section
def g131_timeline_wheel():
    s = mk_session(100, 25)
    tl = TimelinePanel()
    tl.resize(600, 120)
    tl.show()
    tl.set_session(s)
    pump()
    tl.wheelEvent(wheel(120, 0))
    check("G131 a horizontal swipe does not scroll the lanes", tl._scroll == 0, tl._scroll)
    tl.wheelEvent(wheel(0, -120))
    check("G131 one notch down scrolls one lane", tl._scroll == 1, tl._scroll)
    for _ in range(3):
        tl.wheelEvent(wheel(0, -40))
    check("G131 small deltas add up to one lane", tl._scroll == 2, tl._scroll)
    tl._set_view(10, 20)
    tl.wheelEvent(wheel(120, 0, Qt.ControlModifier))
    check("G131 Ctrl + a horizontal swipe does not zoom the time axis", tl._view == (10, 30), tl._view)
    tl.close()


# ------------------------------------------------------------------- widgets

from kinetrace import widgets as W  # noqa: E402


def shown_text(label) -> str:
    """What the label actually shows, whichever text format it is in."""
    from PySide6.QtGui import Qt as _QtGui                  # Qt.mightBeRichText lives in QtGui
    t = label.text()
    rich = label.textFormat() == Qt.RichText or (label.textFormat() == Qt.AutoText and _QtGui.mightBeRichText(t))
    if rich:
        d = QTextDocument()
        d.setHtml(t)
        return d.toPlainText()
    return t


@section
def g89_toast():
    host = QWidget()
    host.resize(800, 600)
    host.show()
    t = W.Toast(host)
    t.show_message("Stopped: 5 < 6 & 7 > 3 done", "info", 6000)
    t.show_message("<b>Run verdict</b>: all good", "success", 6000)
    txt = shown_text(t)
    check("G89 a plain notice followed by an HTML one shows both as written (no tags shown)",
          "Stopped: 5 < 6 & 7 > 3 done" in txt and "Run verdict: all good" in txt and "<b>" not in txt, txt)
    t.hide()
    t.show_message("<b>Run verdict</b>: all good", "success", 6000)
    t.show_message("line one\nline two", "info", 6000)
    txt = shown_text(t)
    check("G89 an HTML notice followed by a plain one keeps its line break",
          "Run verdict: all good" in txt and "line one\nline two" in txt, repr(txt))
    t.hide()
    t.show_message("first verdict", "warn", 6000)
    t.show_message("second note", "info", 6000)
    t.show_message("second note", "info", 6000)
    txt = shown_text(t)
    check("G89 a repeated notice refreshes itself and keeps the pair",
          "first verdict" in txt and txt.count("second note") == 1, txt)
    t.hide()
    t.show_message("boom", "error", 6000)
    t.show_message("x", "info", 6000)
    t.show_message("y", "info", 6000)
    txt = shown_text(t)
    check("G89 the colour is the most severe of the notices SHOWN",
          "boom" not in txt and t._level == "info", (txt, t._level))
    t.hide()
    called = []
    t.show_message("open me", "info", 6000, on_click=lambda: called.append(1))
    t.show_message("another", "info", 6000)
    check("G89 the action stays with its notice (tooltip)", t.toolTip() == "click to open it", t.toolTip())
    QTest.mouseClick(t, Qt.LeftButton)
    pump(50)
    check("G89 a click runs the action of the notice that has one", called == [1] and not t.isVisible(), called)
    host.close()


@section
def g130_manual():
    d = W.ManualDialog()
    d.resize(1000, 700)
    d.show()
    pump()
    n = d.contents.count()
    check("(setup) the manual has a contents list", n > 8, n)
    d.contents.setCurrentRow(6)
    pump()
    bar = d.view.verticalScrollBar()
    away = bar.value()
    check("(setup) the entry scrolls the manual", away > 0, away)
    bar.setValue(0)
    rect = d.contents.visualItemRect(d.contents.item(6))
    QTest.mouseClick(d.contents.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    pump()
    check("G130 clicking the current contents entry again jumps", bar.value() > 0, bar.value())
    opened = []

    class FakeDS:
        @staticmethod
        def openUrl(u):
            opened.append(u.toString())
            return True
    W.QDesktopServices = FakeDS
    d._on_anchor(QUrl("../README.md"))
    check("G130 the manual's file link opens (resolved against the manual's folder)",
          len(opened) == 1 and opened[0].startswith("file:") and opened[0].endswith("README.md"), opened)
    d.close()


# ------------------------------------------------------- viewgrid / camera panel

@section
def g127_active_outline():
    from kinetrace.viewgrid import ViewGrid
    g = ViewGrid()
    g.resize(700, 320)
    g.set_count(2)
    g.show()
    g.set_active(1)
    pump()
    img = g.grab().toImage()

    def edge(i):
        cell = g._cells[i]
        o = cell.mapTo(g, QPoint(0, 0))
        return QColor(img.pixel(o.x(), o.y() + 100))
    acc, hair = QColor(theme.ACCENT), QColor(theme.HAIRLINE)
    a1, a0 = edge(1), edge(0)
    check("G127 the active camera's outline is drawn in the accent colour", a1.rgb() == acc.rgb(),
          (a1.name(), acc.name()))
    check("G127 the others carry the hairline", a0.rgb() == hair.rgb(), (a0.name(), hair.name()))
    g.set_active(0)
    pump()
    img = g.grab().toImage()
    check("G127 the outline follows the active view", edge(0).rgb() == acc.rgb() and edge(1).rgb() == hair.rgb())
    g.set_count(1)
    pump()
    m = g._cells[0].layout().contentsMargins()
    check("G127 a lone view stays bare (no frame, no margin)", (m.left(), m.top()) == (0, 0)
          and g._cells[0].styleSheet() == "", (m.left(), g._cells[0].styleSheet()))
    g.close()


@section
def g128_reference_nudges():
    from kinetrace.camerapanel import CameraPanel
    p = CameraPanel()
    p.update_rows(["camA", "camB"], [0.0, 5.0], 0, ["s", "s"])
    nudges = lambda row: [b for b in row.findChildren(QToolButton) if b.text() in ("◂", "▸")]  # noqa: E731
    r0, r1 = nudges(p._rows[0]), nudges(p._rows[1])
    check("G128 the reference row's nudge buttons are disabled with its offset box",
          len(r0) == 2 and not any(b.isEnabled() for b in r0) and not p._rows[0].spin.isEnabled(),
          [b.isEnabled() for b in r0])
    check("G128 the other rows' nudges stay on", len(r1) == 2 and all(b.isEnabled() for b in r1))


# ----------------------------------------------------------------- folderimport

@section
def i199_probe_thread_never_dropped():
    from kinetrace import folderimport as fi
    from kinetrace import video_source
    d = tempfile.mkdtemp()
    for n in ("a.mp4", "b.mp4"):
        open(os.path.join(d, n), "wb").close()
    entered = []

    class _Cap:
        def isOpened(self):
            return False

    def slow_open(path, *a, **k):
        entered.append(path)
        time.sleep(3.0)                       # a stalled open (a share, a damaged file)
        return _Cap()
    real = video_source.open_capture
    video_source.open_capture = slow_open
    try:
        dlg = fi.VideoFolderDialog(None, d)
        t0 = time.time()
        while not entered and time.time() - t0 < 5:
            pump(20)
        check("(setup) the header thread is inside a stalled read", bool(entered))
        t0 = time.time()
        dlg.stop_probe()
        dt = time.time() - t0
        check("I199 stop_probe does not block the dialog on a stalled read", dt < 1.0, f"{dt:.2f}s")
        orphans = list(fi._ORPHANS)
        check("I199 the running header thread is kept referenced (never dropped)",
              len(orphans) == 1 and orphans[0].isRunning(), orphans)
        dlg.deleteLater()
        pump()
        t0 = time.time()
        while fi._ORPHANS and time.time() - t0 < 8:
            pump(50)
        check("I199 it leaves the list when it has finished", not fi._ORPHANS, fi._ORPHANS)
        fi.wait_orphans(100)                  # the helper the app can call at close
    finally:
        video_source.open_capture = real


# --------------------------------------------------------------------------- end

check("no unexpected exception reached sys.excepthook", not EXC, EXC)
if FAILS:
    print("VERIFY_REVIEW_UI FAILED: " + "; ".join(FAILS))
    sys.exit(1)
print("VERIFY_REVIEW_UI PASSED")
