"""Track follows the selection, each point with its own tracker (G61-G64, GPU group).

[1] nothing selected -> nothing is tracked (Track says what to select); one
    selected point -> only that one; Ctrl+A -> everything.
[2] a point's own tracker: the AT / CT / MS tag on its row, Track ▾ -> Point
    model gives it to the selected points, saved with the project (points.csv
    `tracker`) and back on open.
[3] AllTracker + CoTracker3 points in one selection: two passes over the same
    frames, the Track button says so, one Ctrl+Z undoes both; a second pass
    never runs past where the first stopped; when it stops earlier, the first
    pass's points go back to their pre-run data after it (one end frame).
[4] several selected points: the menu for all of them (clear here + Ctrl+Z,
    delete + Ctrl+Z, tracker for all) and a short right click on one of their
    markers clears all of them on this frame, keeping the selection.
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from _clean import forget_recovery  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication.instance() or QApplication([])
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

VID = os.path.join(ROOT, "test600.mp4")
OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
forget_recovery(VID)
win = MainWindow()
win.resize(1400, 900)
win.show()


def pump(t=0.05):
    t0 = time.time()
    while time.time() - t0 < t:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t0 = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def run_to_end(timeout=300):
    wait(lambda: win.state == TRACKING or win._passes is not None, 30, "the run starts")
    wait(lambda: win._passes is None and win.state == READY and win._multi is None, timeout, "the run ends")
    pump(0.2)


def select(*pids):
    win.point_list.clearSelection()
    for q in pids:
        win.point_list.item(q).setSelected(True)
    pump()


def last_frame(q):
    fr = np.nonzero(s.tracked[:, q])[0]
    return int(fr.max()) if len(fr) else None


win._open_video(VID)
wait(lambda: win.state == READY, 30, "open")
s = win.session
for xy in ((100, 100), (300, 200), (420, 330)):
    win._on_add(*xy)
pump(0.2)

# ---- [1] the selection decides
print("[1] only what is selected")
win._deselect()
pump()
assert win._track_blocked is not None and "Select what to track" in win._track_blocked, win._track_blocked
win._toggle_tracking()
pump(0.3)
assert win.state == READY and int(s.tracked.sum()) == 3, "nothing selected = nothing tracked"
assert "Select what to track" in win.toast.text()
select(1)
assert "1 point" in win.btn_track.text(), win.btn_track.text()
win._start_tracking(stop_after=30)
run_to_end()
assert [int(s.tracked[:31, q].sum()) for q in range(3)] == [1, 31, 1], "only the selected point"
win._undo_run()
pump()
win.act_select_all.trigger()
pump()
assert sorted(win._selected_pids()) == [0, 1, 2] and "3 points" in win.btn_track.text()
win._on_add(500, 400)          # adding a point selects just the new one (no pile-up)
pump()
assert win._selected_pids() == [3], win._selected_pids()
win._on_delete(3) if hasattr(win, "_on_delete") else None
pump()
print("  nothing selected tracks nothing; one point tracks one; Ctrl+A selects all OK")

# ---- [2] a point's own tracker
print("[2] trackers per point")
select(0)
win.act_pm_alltracker.trigger() if win.act_pm_alltracker.isEnabled() else win.act_pm_cotracker.trigger()
select(2)
win.act_pm_cotracker.trigger()
pump()
at = "alltracker" if win.act_pm_alltracker.isEnabled() else "cotracker3"
assert s.points[0].tracker == at and s.points[2].tracker == "cotracker3", (s.points[0].tracker, s.points[2].tracker)
assert win.point_list.iconSize().width() >= 30 and "CoTracker3" in win.point_list.item(2).toolTip()
proj = os.path.join(OUT, "track_selection.kinetrace")
win.project.save(proj)
from kinetrace.project import Project  # noqa: E402

back = Project.load(proj).sessions[0]
assert [p.tracker for p in back.points[:3]] == [at, s.points[1].tracker, "cotracker3"]
print("  tags on the rows, Track ▾ sets the selected points' tracker, saved and read back OK")

# ---- [3] two passes
if at == "alltracker":
    print("[3] AllTracker + CoTracker3: two passes")
    select(0, 2)
    assert "2 passes" in win.btn_track.text(), win.btn_track.text()
    win._goto(0, force=True)
    select(0, 2)
    win._toggle_tracking()
    run_to_end(600)
    l0, l2 = last_frame(0), last_frame(2)
    print(f"  pass 1 (AllTracker) to {l0}, pass 2 (CoTracker3) to {l2}")
    assert l0 is not None and l2 is not None and l2 <= l0, (l0, l2)
    assert "Two passes" in win.toast.text()
    win._undo_run()
    pump()
    assert int(s.tracked[:, 0].sum()) == 1 and int(s.tracked[:, 2].sum()) == 1, "one Ctrl+Z undoes both passes"
    # a second pass never runs past the first: pretend pass 1 stopped at frame 40
    win._goto(0, force=True)
    win._passes = {"queue": [[s.points[2].name]], "groups": [[s.points[0].name], [s.points[2].name]],
                   "view": 0, "frame": 0, "step": False, "every": False, "snap": s.snapshot(), "msnaps": None,
                   "done": [{"lasts": {0: 40}, "fail": None, "user_stop": False, "error": False}]}
    win._passes_next()
    run_to_end()
    assert last_frame(2) == 40, f"the second pass ran to {last_frame(2)}, past the first pass's 40"
    win._undo_run()
    pump()
    # the end-frame rule: pass 2 ended at 60, pass 1 at 100 -> pass 1 back to its pre-run data after 60
    snap = s.snapshot()
    s.write_segment(0, np.full((101, 1, 2), 50.0, np.float32), np.ones((101, 1), bool), [0],
                    np.full((101, 1), 0.9, np.float32))
    win._passes = {"queue": [], "groups": [[s.points[0].name], [s.points[2].name]], "view": 0, "frame": 0,
                   "step": False, "every": False, "snap": snap, "msnaps": None,
                   "done": [{"lasts": {0: 100}, "fail": None, "user_stop": False, "error": False},
                            {"lasts": {0: 60}, "fail": None, "user_stop": False, "error": False}]}
    win._passes_finish()
    assert last_frame(0) == 60, last_frame(0)
    win._undo_run()
    pump()
    print("  two passes over the same frames, one Ctrl+Z; a second pass stays inside the first; one end frame OK")
else:
    print("[3] skipped: AllTracker is not installed here")

# ---- [4] several selected: the menu for all, and the short right click
print("[4] several selected points")
win._goto(0, force=True)
win._set_tracker([0, 1, 2], "cotracker3")      # one tracker: a single bounded run below
select(0, 1, 2)
win._start_tracking(stop_after=20)
run_to_end()
win._goto(10, force=True)
select(0, 1)
menu, acts = win._build_multi_menu([0, 1])
labels = [a.text() for a in menu.actions()]
assert any("Clear the 2 selected points on frame 10" in t for t in labels), labels
assert any("Delete the 2 points" in t for t in labels)
win._multi_menu_action(acts["clear_here"], acts, [0, 1])
assert not s.tracked[10, 0] and not s.tracked[10, 1] and s.tracked[10, 2], "clear here: the two selected only"
win._undo_run()
assert s.tracked[10, 0] and s.tracked[10, 1]
tr = next(a for a, k in acts["tracker"].items() if k == "cotracker3")
win._multi_menu_action(tr, acts, [0, 1])
assert s.points[0].tracker == s.points[1].tracker == "cotracker3"
menu.deleteLater()
win._multi_menu_action(acts["delete"], acts, [0, 1])
assert s.n_points == 1, s.n_points
win._undo_run()
pump()
assert s.n_points == 3, "Ctrl+Z brings the deleted points back"
# a short right click on one of several selected markers clears all of them here
select(0, 1)
win.canvas.fit()
pump(0.1)
vp = win.canvas.viewport()
QTest.mouseClick(vp, Qt.RightButton, Qt.NoModifier, win.canvas.mapFromScene(QPointF(*s.tracks[10, 0])))
pump()
assert not s.tracked[10, 0] and not s.tracked[10, 1] and s.tracked[10, 2]
assert sorted(win._selected_pids()) == [0, 1], "the selection is kept"
win._undo_run()
# a long right press on one of them opens the menu for all (the canvas hook)
seen = []
win.canvas.multi_menu = lambda pid, pos: seen.append(pid) or True
QTest.mousePress(vp, Qt.RightButton, Qt.NoModifier, win.canvas.mapFromScene(QPointF(*s.tracks[10, 0])))
pump(0.8)
QTest.mouseRelease(vp, Qt.RightButton, Qt.NoModifier, win.canvas.mapFromScene(QPointF(*s.tracks[10, 0])))
pump()
assert seen == [0], seen
print("  the menu for several points (clear here, delete, tracker; each one Ctrl+Z), a right click clears all "
      "selected here, a long press opens the menu for all OK")

win.close()
pump(0.3)
win._dev_probe.wait(30000)
forget_recovery(VID)
print("verify_track_selection PASSED")
