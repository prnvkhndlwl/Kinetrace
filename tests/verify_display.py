"""Workflow + display features: trails with a
future path, onion skin, loupe, display-only filters, rectangle / polygon
regions, the hand-marked hidden flag (kept but not exported, timeline window
marking, undo), frame notes + event notes + annotator, next / previous
hand-placed frame keys, next / previous low-confidence stretch keys, and the
project round trip of all of it."""
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
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

VID = os.path.join(ROOT, "test600.mp4")
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
_next_text = {"v": ""}
QInputDialog.getMultiLineText = staticmethod(lambda *a, **k: (_next_text["v"], True))
QInputDialog.getText = staticmethod(lambda *a, **k: (_next_text["v"], True))

app = QApplication([])
from cotracker_app.app import MainWindow, READY
from cotracker_app.canvas import apply_display_filter
from cotracker_app.tracker import sample_members

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

# ---- pure helpers -------------------------------------------------------
rgb = (np.random.default_rng(0).integers(0, 255, size=(120, 160, 3))).astype(np.uint8)
prev = np.roll(rgb, 3, axis=1)
for kind in ("none", "contrast", "bright", "diff"):
    out = apply_display_filter(rgb, kind, prev)
    assert out.shape == rgb.shape and out.dtype == np.uint8, kind
assert apply_display_filter(rgb, "diff", None) is rgb, "no previous frame: unfiltered"
sq = np.array([[-20, -10], [20, -10], [20, 10], [-20, 10]], np.float32)
mem = sample_members(np.array([100.0, 100.0], np.float32), 22.4, 640, 480, sq)
assert 6 <= len(mem) <= 17 and np.allclose(mem[0], (100, 100)), mem.shape
assert (np.abs(mem[:, 0] - 100) <= 16.1).all() and (np.abs(mem[:, 1] - 100) <= 8.1).all(), "members inside"
tri = np.array([[0, -30], [30, 30], [-30, 30]], np.float32)
mem = sample_members(np.array([200.0, 200.0], np.float32), 30, 640, 480, tri)
assert len(mem) >= 4
print("display filters + polygon member sampling OK")

# ---- fake tracks over frames 0..299 for two points ------------------------
win._on_add(100.0, 100.0)
win._on_add(300.0, 200.0)
L = 300
tw = np.zeros((L, 2, 2), np.float32)
tw[:, 0, 0] = 100 + np.arange(L) * 0.5
tw[:, 0, 1] = 100
tw[:, 1] = (300.0, 200.0)
conf = np.full((L, 2), 0.9, np.float32)
conf[120:140, 0] = 0.2            # a low-confidence stretch on P1
conf[200:210, 1] = 0.3            # and one on P2
s.write_segment(0, tw, np.ones((L, 2), bool), [0, 1], conf)
win._refresh_point_list()
win.canvas.fit()
app.processEvents()

# ---- trails / onion / loupe / filters through the menus ------------------
win._goto(50)
win._set_trail_len(60)
win.act_trail_future.setChecked(True)
mo = win.canvas._motion
assert len(mo.past) == 2 and len(mo.past[0]) == 51, len(mo.past[0]) if mo.past else None
assert len(mo.future) == 2 and len(mo.future[0]) == 61, "future path"
key(Qt.Key_O)
assert win.act_onion.isChecked() and mo.ghost_prev is not None and mo.ghost_next is not None
assert np.allclose(mo.ghost_prev[0], tw[49, 0]) and np.allclose(mo.ghost_next[0], tw[51, 0])
key(Qt.Key_O)
assert not win.act_onion.isChecked() and mo.ghost_prev is None
win._set_trail_len(0)
assert mo.past == [] and mo.future == []
win._set_trail_len(30)
key(Qt.Key_L)
assert win.act_loupe.isChecked() and win.canvas._loupe_enabled
vp = win.canvas.mapFromScene(QPointF(120.0, 100.0))
QTest.mouseMove(win.canvas.viewport(), vp)
app.processEvents()
assert win.canvas._loupe.isVisible(), "loupe should show over the video"
key(Qt.Key_L)
assert not win.canvas._loupe.isVisible()
for k in ("contrast", "bright", "diff", "none"):
    win._set_display_filter(k)
    assert win.canvas.display_filter() == k
    win._goto(win.current + 1)          # a new frame renders through the filter
    app.processEvents()
print("trails / onion / loupe / filters OK")

# ---- rectangle + polygon regions --------------------------------------------
win._deselect()
win._set_region_shape("rect")
key(Qt.Key_N)
assert win.btn_add.isChecked()
p0 = win.canvas.mapFromScene(QPointF(400.0, 300.0))
p1 = win.canvas.mapFromScene(QPointF(460.0, 340.0))
QTest.mousePress(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseMove(win.canvas.viewport(), p0 + QPoint(10, 10))
QTest.mouseMove(win.canvas.viewport(), p1)
QTest.mouseRelease(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, p1)
app.processEvents()
assert s.n_points == 3 and s.points[2].kind == "group" and s.points[2].shape == "rect", \
    (s.n_points, s.points[-1].kind, s.points[-1].shape)
assert len(s.points[2].outline) == 4 and np.allclose(s.tracks[win.current, 2], (430, 320), atol=1.0)
assert not win.btn_add.isChecked()
win._set_region_shape("polygon")
key(Qt.Key_N)
for x, y in ((500, 100), (560, 110), (540, 170), (490, 150)):
    click(x, y)
assert win.canvas.polygon_in_progress()
key(Qt.Key_Return)
assert s.n_points == 4 and s.points[3].shape == "polygon" and len(s.points[3].outline) == 4
assert not win.canvas.polygon_in_progress()
# Esc cancels a half-drawn polygon without adding anything
key(Qt.Key_N)
click(50, 400)
click(90, 400)
key(Qt.Key_Escape)
assert s.n_points == 4 and not win.canvas.polygon_in_progress() and not win.btn_add.isChecked()
win._set_region_shape("circle")
# outlines travel with the fitted centre
ol = s.points[3].outline_at((600.0, 300.0))
assert ol is not None and np.allclose(ol.mean(axis=0), (600, 300), atol=1e-3)
print("rectangle / polygon regions OK")

# ---- hidden (occluded) flag ---------------------------------------------------
win._on_select(0)
win._goto(60)
key(Qt.Key_X, Qt.ShiftModifier)
assert s.occluded[60, 0] and s.tracked[60, 0], "hidden keeps the data"
assert not s.exportable[60, 0] and s.exportable[61, 0]
key(Qt.Key_X, Qt.ShiftModifier)
assert not s.occluded[60, 0]
win._occlude_window(70, 79, [0], True)
assert s.occluded[70:80, 0].all() and not s.occluded[69, 0] and not s.occluded[70, 1]
win._undo_run()
assert not s.occluded[70:80, 0].any(), "undo reverts the window mark"
win._occlude_window(70, 79, [0, 1], True)
csv = os.path.join(SCRATCH, "display_wide.csv")
s.export_csv(csv)
rows = open(csv, encoding="utf-8").read().splitlines()
assert rows[71].split(",")[1] == "" and rows[69].split(",")[1] != "", "hidden cells export blank"
s.export_mat(os.path.join(SCRATCH, "display.mat"))
from scipy.io import loadmat
m = loadmat(os.path.join(SCRATCH, "display.mat"))
assert np.isnan(m["tracks"][75, 0, 0]) and not np.isnan(m["tracks"][65, 0, 0])
assert bool(m["occluded"][75, 0])
win.timeline.repaint()
app.processEvents()
print("hidden flag + exports OK")

# ---- notes + annotator ----------------------------------------------------------
win._apply_annotator("PK")
assert s.annotator == "PK"
win._goto(33)
_next_text["v"] = "tail hidden behind rock"
key(Qt.Key_N, Qt.ShiftModifier)
assert 33 in s.notes and s.notes[33]["author"] == "PK", s.notes
_next_text["v"] = ""
win._edit_note(33)
assert 33 not in s.notes
_next_text["v"] = "strike starts"
win._edit_note(40)
s.add_event("strike", 40, 55, note="fast one")
assert s.events[-1].note == "fast one" and s.events[-1].author == "PK"
s.update_event(len(s.events) - 1, note="edited")
assert s.events[-1].note == "edited"
ev_csv = os.path.join(SCRATCH, "display_events.csv")
s.export_events_csv(ev_csv)
txt = open(ev_csv, encoding="utf-8").read()
assert "strike,40,55,edited,PK" in txt and "note,40,40,strike starts,PK" in txt, txt
win._on_events_changed()
print("notes + annotator OK")

# ---- next / previous hand-placed + low-confidence keys --------------------------
win._on_select(0)
for f in (20, 90, 150):
    win._goto(f)
    click(float(tw[f, 0, 0]) + 40, 140)   # away from the marker: a marker click selects/drags
assert sorted(s.manual_frames(0).tolist()) == [20, 90, 150]
win._goto(0)
key(Qt.Key_Period)
assert win.current == 20, win.current
key(Qt.Key_Period)
assert win.current == 90
key(Qt.Key_Period)
key(Qt.Key_Period)                      # past the last: stays
assert win.current == 150
key(Qt.Key_Comma)
assert win.current == 90
win._deselect()
win._goto(0)
key(Qt.Key_J)
assert win.current == 120, win.current    # P1's low stretch
key(Qt.Key_J)
assert win.current == 200, win.current    # P2's
key(Qt.Key_J, Qt.ShiftModifier)
assert win.current == 120
win._on_select(1)
win._goto(0)
key(Qt.Key_J)
assert win.current == 200, "selected point only"
print("hand-placed + low-confidence navigation OK")

# ---- project round trip ---------------------------------------------------------
win._set_display_filter("contrast")
win._set_trail_len(120)
win.act_onion.setChecked(True)
proj = os.path.join(SCRATCH, "display_rt.cotrk")
win._sync_ui_state()
win.project.save_npz(proj)
from cotracker_app.project import Project
back = Project.load_npz(proj)
s2 = back.session
assert s2.points[2].shape == "rect" and len(s2.points[2].outline) == 4
assert s2.points[3].shape == "polygon"
assert s2.occluded[75, 0] and s2.occluded[75, 1] and not s2.occluded[60, 0]
assert s2.notes[40]["text"] == "strike starts" and s2.annotator == "PK"
assert s2.events[-1].note == "edited" and s2.events[-1].author == "PK"
assert s2.ui_state["display_filter"] == "contrast" and s2.ui_state["trail_len"] == 120
assert s2.ui_state["onion"] is True
# reopen through the app: the view options come back
win._open_project_from_path(proj)
pump(lambda: win.state == READY and win.session is not s, 30, "reopen")
assert win.canvas.display_filter() == "contrast" and win._trail_len == 120 and win.act_onion.isChecked()
win._set_display_filter("none")
print("project round trip OK")

win._dev_probe.wait(30000)
win.close()
app.processEvents()
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)
print("verify_display PASSED")
