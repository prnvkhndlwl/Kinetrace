"""Code review 2026-10-02, app.py region 2 (selection .. run handlers .. ui_state).

One plain script, offscreen, QMessageBox / QInputDialog stubbed like the other GUI suites. Every
check below FAILS on code-review-fixes' app.py (3bdda88) and passes now; the ID is in each label.

Sections
  [1] one camera, real key / mouse events: selection (G66 G67 G80 G81), undo steps (G68), clears
      (G82 G84), hints and menus (G102 G124), rows (G126), saved scope (G103), overwrite question
      (G101), run ends (I241), the test's Use (I205), a real bounded run (G99 I189).
  [2] the passes (G63): what can start (I203), per-camera bounds (I204), the end (I202 G100 G99).
  [3] three calibrated cameras: the live 3D verdict (G83), fractional frames (I253), rounding
      (I258), rename across cameras (G68), the re-track (G69 G125), every-camera runs (I184 I196
      I189), the segment riding along (I185).
  [4] a real two-pass step run on the GPU (I203 G100).

Run: .venv\\Scripts\\python.exe tests\\verify_review_app2.py   (the GPU sections need the models)
"""
import os
import shutil
import sys
import time
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox, QProgressDialog  # noqa: E402

from _clean import forget_recovery  # noqa: E402

ASKED: list = []
ANSWER = [QMessageBox.Yes]


def _question(*a, **k):
    ASKED.append(a[2] if len(a) > 2 else "")
    return ANSWER[0]


QMessageBox.question = staticmethod(_question)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))

app = QApplication.instance() or QApplication([])
from kinetrace import alltracker_backend  # noqa: E402
from kinetrace.app import READY, TRACKING, MainWindow, _SideRun  # noqa: E402
from kinetrace.calib import CameraCalibration, NoUndistort, dlt_from_camera  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
VID = os.path.join(ROOT, "test600.mp4")
FAILS: list = []


def pump(t=0.1):
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


def check(label, cond, detail=""):
    if callable(cond):                      # evaluated here: old code may raise where new code answers
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001
            cond, detail = False, f"raised {e!r}"
    cond = bool(cond)
    print(("  ok    " if cond else "  FAIL  ") + label + ("" if cond or isinstance(detail, str) and not detail
                                                          else f"   [{detail}]"))
    if not cond:
        FAILS.append(label)


class Rec:
    """Records the toast's messages (and the click action of each) without showing them."""

    def __init__(self, win):
        self.win = win
        self.msgs: list = []
        self._orig = win.toast.show_message
        win.toast.show_message = self.show

    def show(self, text, level="info", ms=6000, on_click=None):
        self.msgs.append((text, level, on_click))

    def text(self):
        return " | ".join(m[0] for m in self.msgs)

    def clear(self):
        self.msgs.clear()


def new_window(video):
    forget_recovery(video)
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    w._open_video(video)
    wait(lambda: w.state == READY, 60, "open")
    return w


def row_center(win, row):
    return win.layers.visualItemRect(win.layers.point_item(row)).center()


def select_rows(win, *rows):
    win.layers.clearSelection()
    for r in rows:
        win.layers.point_item(r).setSelected(True)
    pump(0.05)


def close(win):
    win.close()
    pump(0.3)
    win._dev_probe.wait(30000)


REAL_AT = alltracker_backend.available
alltracker_backend.available = lambda: True      # the tag logic only; no model is loaded here

# ================================================================== [1] one camera
print("[1] one camera")
win = new_window(VID)
s = win.session
for xy in ((100, 100), (300, 200), (420, 330), (500, 400)):
    win._on_add(*xy)
pump(0.2)
names = [m.name for m in s.points]
rec = Rec(win)

# ---- G66 / G80 / G81 / G67: the selection
select_rows(win, 1, 2, 3)
win._on_select(0)
check("G66 a click on P1 with P2-P4 selected leaves just P1", win._selected_pids() == [0], win._selected_pids())
check("G66 the Track button follows (T tracked the stale set)", "1 point" in win.btn_track.text(), win.btn_track.text())

win.layers.clearSelection()
pump(0.05)
QTest.mouseClick(win.layers.viewport(), Qt.LeftButton, Qt.NoModifier, row_center(win, 1))
QTest.mouseClick(win.layers.viewport(), Qt.LeftButton, Qt.ControlModifier, row_center(win, 2))
check("G80 real clicks: P2 + P3 selected, P3 current", win._selected_pids() == [1, 2] and win.selected == 2,
      (win._selected_pids(), win.selected))
QTest.mouseClick(win.layers.viewport(), Qt.LeftButton, Qt.ControlModifier, row_center(win, 2))
check("G80 Ctrl+click-deselecting the current row moves it into the selection",
      win._selected_pids() == [1] and win.selected == 1, (win._selected_pids(), win.selected))

win._deselect()
win.act_select_all.trigger()
pump(0.05)
check("G80 Ctrl+A gives a current point", win.selected is not None and win.selected in win._selected_pids(),
      win.selected)
before = int(s.manual[win.current].sum())
win._on_annotate(222.0, 111.0)
check("G80 a click on the video after Ctrl+A places the current point",
      lambda: win.selected is not None and bool(s.manual[win.current, win.selected]))
win._undo_run()
win._deselect()
win.act_select_all.trigger()
pump(0.05)
win.selected = None                        # the state the old code left after Ctrl+A
win._set_tracker([0], "cotracker3")        # any change that rebuilds the list
check("G81 a rebuild keeps a Ctrl+A selection", win._selected_pids() == [0, 1, 2, 3], win._selected_pids())
win._undo_run()
win._deselect()

select_rows(win, 1, 2)
win._on_select(1)
select_rows(win, 1, 2)
win.selected = 1
win._on_delete(0)
check("G67 deleting P1 keeps P2 + P3 selected (by name)",
      [s.points[q].name for q in win._selected_pids()] == names[1:3], [s.points[q].name for q in win._selected_pids()])
win._undo_run()
pump(0.1)
check("G67 Ctrl+Z brings P1 back", s.n_points == 4)
win._deselect()

# ---- G68: each its own undo step
win._goto(5, force=True)
win._on_select(1)
win._on_annotate(150.0, 150.0)            # the earlier edit
pid = 1
old = s.points[pid].name
win._apply_rename(pid, "Snout")
win._undo_run()
check("G68 Ctrl+Z after a rename takes back the rename only",
      s.points[pid].name == old and bool(s.manual[5, pid]), (s.points[pid].name, bool(s.manual[5, pid])))
win._undo_run()
win._on_annotate(151.0, 151.0)
t0 = s.points[pid].tracker
win._set_tracker([pid], "spot")
win._undo_run()
check("G68 a tracker change is its own step", s.points[pid].tracker == t0 and bool(s.manual[5, pid]),
      (s.points[pid].tracker, bool(s.manual[5, pid])))
win._undo_run()
win._on_annotate(152.0, 152.0)
win._on_anchor_toggled(pid, True)
win._undo_run()
check("G68 the anchor toggle is its own step", not s.points[pid].anchor and bool(s.manual[5, pid]))
win._undo_run()
win._on_annotate(153.0, 153.0)
win.layers.point_item(pid).setCheckState(0, Qt.Unchecked)
check("G68 (display off)", not s.points[pid].display)
win._undo_run()
check("G68 the display checkbox is its own step", s.points[pid].display and bool(s.manual[5, pid]))
win._undo_run()

# ---- G126: rows after in-place edits
it = win.layers.point_item(pid)
it.setText(0, "")
pump(0.05)
check("G126 an emptied name keeps the old one", it.text(0) == s.part_name(pid) != "", it.text(0))
q = s.add_empty_point()
win._refresh_point_list()
dim = lambda r: win.layers.point_item(r).foreground(0).style() != Qt.NoBrush      # noqa: E731
check("G126 a point with no position is dimmed", dim(q))
win._on_select(q)
win._on_annotate(80.0, 80.0)
check("G126 its first placement undims the row", not dim(q))
win._clear_window(0, s.n_frames - 1, [q], True, False)
check("G126 clearing its whole track dims it again", dim(q))
win._undo_run()

# ---- G82 / G84: clears
win._goto(30, force=True)
win._on_select(2)
win._on_annotate(210.0, 210.0)
marker = s.snapshot()
win._undo_snap = marker
win.statusBar().clearMessage()
win._clear_window(400, 410, [2], True, False)
check("G82 a clear that removes nothing takes no snapshot", win._undo_snap is marker)
check("G82 ... and says so", "Nothing to clear" in win.statusBar().currentMessage(), win.statusBar().currentMessage())
select_rows(win, 0, 3)
menu, acts = win._build_multi_menu([0, 3])
win._goto(500, force=True)
menu2, acts2 = win._build_multi_menu([0, 3])
check("G82 multi 'clear on frame' is disabled when no selected point has data there",
      not acts2["clear_here"].isEnabled() and acts["clear_here"].isEnabled() in (True, False))
menu.deleteLater()
menu2.deleteLater()
win._goto(30, force=True)
win._on_select(0)
win._on_annotate(60.0, 60.0)
win._on_select(3)
win._on_annotate(61.0, 61.0)
select_rows(win, 0, 3)
win._undo_snap = None
s.tracked[30, 3] = False
win._multi_clear([0, 3], 30, 30)
check("G82 the multi-clear message counts what was cleared", "1 of the 2 points" in win.statusBar().currentMessage(),
      win.statusBar().currentMessage())
win._undo_run()
s.tracked[30, 3] = True

s.ensure_animal()
s.animal.add_click(10, 320.0, 240.0, True)
bitmap = np.zeros((s.height, s.width), bool)
bitmap[200:300, 280:380] = True
for f in range(10, 41):
    s.masks.set(f, bitmap, 9.0)
win._refresh_animal_panel()
win._goto(30, force=True)
for q2 in (0, 3):
    win._on_select(q2)
    win._on_annotate(200.0 + q2, 200.0)
win.act_select_all.trigger()
pump(0.05)
check("G84 (setup) the segment row is selected", win._segment_selected())
win._on_clear_frame(0)
check("G84 a short right click on one of several selected markers never clears the silhouette",
      s.masks.has(30) and not s.tracked[30, 0] and not s.tracked[30, 3], (s.masks.has(30), s.tracked[30, 0]))
win._undo_run()
win._deselect()

# ---- G102 / G124: hints and the Tracker menus
b = s.add_ball(0, 60.0, 60.0)
win._refresh_point_list()
for k in range(8):
    win._goto(1 + k, force=True)
    win._hint_corrections(b)
check("G102 no 'corrected often' hint for a ball marker", not win._spot_hints, win._spot_hints)
rec.clear()
for k in range(8):
    win._goto(1 + k, force=True)
    win._hint_corrections(2)
hint = [m for m in rec.msgs if m[2] is not None]
check("G102 (setup) the hint appears for a plain point", bool(hint))
seen = []
win._test_point_models = lambda pid: seen.append(pid)
win._on_delete(0)                                  # the rows shift
target = s.pid_by_name(names[2])
hint[-1][2]()
check("G102 its click looks the point up by name (the index shifted)", seen == [target], (seen, target))
win._undo_run()
del win._test_point_models

g = s.add_point(0, 300.0, 300.0, kind="group", radius=20.0)
win._refresh_point_list()
win._set_tracker([g], "spot")
check("G124 a region keeps its tracker when Moving spot is chosen", s.points[g].tracker != "spot", s.points[g].tracker)
select_rows(win, g, 0)
menu, acts = win._build_multi_menu([g])
spot_act = next(a for a, k in acts["tracker"].items() if k == "spot")
check("G124 the multi menu does not offer Moving spot for regions only", not spot_act.isEnabled())
menu.deleteLater()
menu, acts = win.canvas._build_context_menu(g)
win._extend_point_menu(menu, acts, g)
spot_act = next(a for a, k in acts["tracker"].items() if k == "spot")
check("G124 nor does the point's own menu", not spot_act.isEnabled())
menu.deleteLater()
win._on_delete(g)
win._deselect()

# ---- G103: the run scope is part of the saved working state
select_rows(win, 1, 2)
win._on_select(1)
win.layers.point_item(2).setSelected(True)
pump(0.05)
win._sync_ui_state()
saved = list(s.ui_state.get("selected_names", []))
check("G103 the selected names go into ui_state", saved == [names[1], names[2]], saved)
win._deselect()
win._apply_ui_state()
check("G103 and come back", [s.points[q].name for q in win._selected_pids()] == [names[1], names[2]],
      win._selected_pids())
win._deselect()
s.masks.set(30, bitmap, 9.0)
win._select_segments([0])
win._sync_ui_state()
win._deselect()
win._apply_ui_state()
check("G103 the segment's row is restored too", win._segment_selected())
win._deselect()

# ---- I205 / G68: the test's Use
spot_settings = types.SimpleNamespace(to_dict=lambda: {"cue": "bright", "radius": 9.0}, describe=lambda: "bright")
res = types.SimpleNamespace(model="spot", settings=spot_settings, label="Moving spot")
win._set_tracker([2], "spot")
win._undo_snap = None
s.dirty = False
win._apply_test_choice(2, res)
check("I205 Use marks the project changed even when the point is already on Moving spot", s.dirty)
check("I205 the settings are stored", s.points[2].spot == {"cue": "bright", "radius": 9.0}, s.points[2].spot)
win._set_tracker([2], "cotracker3")
win._apply_test_choice(2, res)
win._undo_run()
check("G68 Use is one step: Ctrl+Z puts the tracker back", s.points[2].tracker == "cotracker3", s.points[2].tracker)
win._undo_run()

# ---- I241: the run end shared by the finished and the error path
win.state = TRACKING
win._user_paused = True
win._step_run = True
win._undo_snap = s.snapshot()
win.act_undo.setEnabled(False)
win._model_dialog = QProgressDialog("x", "Cancel", 0, 0, win)
win.worker = types.SimpleNamespace(wait=lambda ms=0: True, isRunning=lambda: False, decode_failed_at=None,
                                   _autopause_reason="")
win._on_track_error("boom")
check("I241 an error resets _user_paused", not win._user_paused)
check("I241 an error puts Ctrl+Z back on", win.act_undo.isEnabled())
check("I241 an error closes the models dialog", win._model_dialog is None)
win.state = TRACKING
win._model_dialog = QProgressDialog("x", "Cancel", 0, 0, win)
win.worker = types.SimpleNamespace(wait=lambda ms=0: True, isRunning=lambda: False, _autopause_reason="",
                                   _ball_ended={}, _spot_ended={})
win._autopause_info = None
win._run_start = win.current
win._on_track_finished(win.current, True)
check("I241 a finished run closes a models dialog still open", win._model_dialog is None)

# ---- G99 / I189: a real bounded run (the points' own tracker)
win._deselect()
win._set_tracker([0, 1, 2, 3], "cotracker3" if not alltracker_backend.available() else "alltracker")
alltracker_backend.available = lambda: False       # CoTracker3 for the real runs below (no vendored code needed)
win._set_tracker([0, 1, 2, 3], "cotracker3")
win._goto(0, force=True)
for q2 in range(4):
    win._on_select(q2)
    win._on_annotate(*[(100, 100), (300, 200), (420, 330), (500, 400)][q2])
win._on_select(0)
check("I189 (setup) the start frame is hand-placed", bool(s.manual[0, 0]))
win._start_tracking(stop_after=6)
check("G99 a bounded run is not a semi-automatic step", win._step_run is False)
wait(lambda: win.state == READY, 240, "bounded run")
pump(0.3)
check("I189 the run keeps the start frame's hand-placed flag", bool(s.manual[0, 0]) and not s.manual[1:7, 0].any(),
      (bool(s.manual[0, 0]), int(s.manual[1:7, 0].sum())))
win._goto(6, force=True)
win._on_annotate(*[float(v) for v in s.tracks[6, 0]])
win.act_mode_semi.trigger()
win._on_select(0)
win._track_step()
wait(lambda: win.state == READY, 120, "step")
pump(0.3)
check("I189 a semi-automatic step after a correction keeps that click hand-placed",
      bool(s.manual[6, 0]) and s.tracked[7, 0], (bool(s.manual[6, 0]), bool(s.tracked[7, 0])))
win.act_mode_auto.trigger()
alltracker_backend.available = lambda: True
win._undo_run()
close(win)

# ---- G101: the overwrite question counts the run's own points
print("[1b] the overwrite question")
LONG = os.path.join(OUT, "review_app2_long.mp4")
if not os.path.isfile(LONG):
    vw = cv2.VideoWriter(LONG, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for f in range(2100):
        img = np.full((48, 64, 3), 60, np.uint8)
        cv2.circle(img, (10 + f % 40, 24), 3, (230, 230, 230), -1)
        vw.write(img)
    vw.release()
win = new_window(LONG)
s = win.session
win._on_add(20.0, 20.0)
s.tracked[:2090, 0] = True
s.tracks[:2090, 0] = (20.0, 20.0)
s.visibility[:2090, 0] = True
win._goto(10, force=True)
win._on_add(30.0, 30.0)                  # a new point D: selected alone
pump(0.1)
ANSWER[0] = QMessageBox.No
ASKED.clear()
win._start_tracking()
check("G101 a new point D alone does not ask about overwriting another point's frames",
      not [a for a in ASKED if "overwrite" in a], ASKED)
win._on_select(0)
ASKED.clear()
win._start_tracking()
check("G101 the point that has the tracked frames does", any("overwrite" in a for a in ASKED), ASKED)
ANSWER[0] = QMessageBox.Yes
close(win)

# ================================================================== [2] the passes
print("[2] passes")
win = new_window(VID)
s = win.session
rec = Rec(win)
for xy in ((100, 100), (300, 200), (420, 330)):
    win._on_add(*xy)
pump(0.1)
calls: list = []
win._start_tracking = lambda **k: calls.append(k)
s.points[0].tracker = "alltracker"
s.points[1].tracker = "cotracker3"
s.points[2].tracker = "cotracker3"
for q2 in range(3):
    s.set_position(10, q2, 100.0 + 50 * q2, 100.0)
win._goto(10, force=True)
s.clear_window([0], 10, 10)                       # P1 (AllTracker) has no position here
select_rows(win, 0, 1)
win._on_select(1)
select_rows(win, 0, 1)
win._update_track_button()
check("I203 the button counts only passes that can start", "passes" not in win.btn_track.text(), win.btn_track.text())
calls.clear()
win._toggle_tracking()
check("I203 pass 1 with nothing to start from no longer cancels the press",
      calls and calls[0].get("only_pids") == [1], calls)
check("I203 the user is told the others were left out", "no position on frame 10" in rec.text(), rec.text())
win._passes = None

# two passes that both start; spot / ball / derived points are not listed under AllTracker (G100)
s.points[0].tracker = "alltracker"
s.set_position(10, 0, 100.0, 100.0)
b = s.add_ball(10, 60.0, 60.0)
sp = s.add_point(10, 200.0, 50.0)
s.points[sp].tracker = "spot"
win._refresh_point_list()
select_rows(win, 0, 1, b, sp)
win._on_select(0)
select_rows(win, 0, 1, b, sp)
calls.clear()
win._start_tracking = lambda **k: (calls.append(k), setattr(win, "state", TRACKING))      # pass 1 "starts"
win._toggle_tracking()
win.state = READY
win._start_tracking = lambda **k: calls.append(k)
lab = (win._passes or {}).get("labels") or []
check("G100 the pass labels name each point under what follows it",
      bool(lab) and f"AllTracker ({s.points[0].name})" in lab[0] and f"ball markers ({s.points[b].name})" in lab[0]
      and f"Moving spot ({s.points[sp].name})" in lab[0], lab)
check("I203 (setup) two passes are planned", win._passes is not None and len(win._passes["groups"]) == 2)
win._passes = None
win._deselect()

# G99: pass 2 is bounded, not a step
calls.clear()
win._start_pass([1], False, False, segment=False, stops={0: 40}, quiet=True)
check("G99 pass 2 asks for a bound and is not a step",
      calls and calls[0].get("stop_after") == 40 and calls[0].get("step") is False, calls)
calls.clear()
win._start_pass([1], True, False, segment=False, stops={0: 40}, quiet=True)
check("G99 a semi-automatic pass is a step", calls and calls[0].get("step") is True, calls)
calls.clear()
win._start_pass([1], False, False, segment=False, stops={3: 40}, quiet=True)
check("I204 a pass 2 for a camera the first pass did not run in does not run unbounded", not calls, calls)

# G100 / I202 / I203: the end of a two-pass run
rec.clear()
snap0 = s.snapshot()


def passes_state(done, step=False, labels=("AllTracker (P1)", "CoTracker3 (P2)")):
    return {"queue": [], "groups": [[s.points[0].name], [s.points[1].name]], "labels": list(labels),
            "view": 0, "frame": 10, "step": step, "every": False, "done": done, "snap": snap0, "msnaps": None}


win._goto(10, force=True)
win._passes = passes_state([{"lasts": {0: 50}, "fail": None, "user_stop": False, "error": False, "group": 0},
                             {"lasts": {0: 30}, "fail": None, "user_stop": True, "error": False, "group": 1}])
win._passes_finish()
txt = rec.text()
check("G100 X during pass 2 reads 'stopped by you', not 'did not run'",
      "stopped by you at frame 30" in txt and "did not run" not in txt, txt)
check("G100 the playhead ends where the run ended, not back at the start", win.current == 30, win.current)
rec.clear()
win._goto(10, force=True)
win._passes = passes_state([{"lasts": {0: 11}, "fail": None, "user_stop": False, "error": False, "group": 0},
                            {"lasts": {0: 11}, "fail": None, "user_stop": False, "error": False, "group": 1}], step=True)
win._passes_finish()
check("G100 a clean semi-automatic step is a status line, not a notice", not rec.msgs, rec.text())
check("G100 ... and the playhead stays on the stepped frame", win.current == 11, win.current)
check("G100 ... saying what each tracker did", "Two passes" in win.statusBar().currentMessage(),
      win.statusBar().currentMessage())
rec.clear()
win._goto(10, force=True)
win._passes = passes_state([{"lasts": {0: 50}, "fail": None, "user_stop": False, "error": False, "group": 0},
                            {"lasts": {}, "fail": None, "user_stop": False, "error": False, "skipped": True,
                             "reason": "none of its points has a position on frame 10 in the cameras the first "
                                       "pass ran in", "group": 1}])
win._passes_finish()
check("I203 a pass that could not start says why", "not run (none of its points has a position" in rec.text(),
      rec.text())

# I203 / I204 in _passes_next: per-camera bounds from the cameras' OWN frames
rec.clear()
got: list = []
win._start_pass = lambda *a, **k: got.append((a, k))
win._goto(10, force=True)
win._passes = {"queue": [[s.points[1].name]], "groups": [[s.points[0].name], [s.points[1].name]],
               "view": 0, "frame": 10, "step": False, "every": False,
               "done": [{"lasts": {0: 30, 1: 2}, "fail": None, "user_stop": False, "error": False, "group": 0}],
               "snap": snap0, "msnaps": None}
win._passes_next()
check("I204 a camera whose first pass tracked nothing is not given a bound below its start",
      got and got[0][1].get("stops") == {0: 30}, got)
del win._start_pass
win._passes = None
close(win)

# ================================================================== [3] three calibrated cameras
print("[3] three calibrated cameras")
from kinetrace.calib import Calibration  # noqa: E402

W3, H3, N3, FPS3 = 640, 480, 60, 30.0


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K3 = np.array([[700.0, 0, W3 / 2], [0, 700.0, H3 / 2], [0, 0, 1]])
cams = []
for a in (0.2, 1.3, 2.4):
    pos = np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.6])
    R, tt = look_at(pos)
    cams.append(CameraCalibration(dlt_from_camera(K3, R, tt), W3, H3, NoUndistort(), pixel_origin=0.0))


def world(f):
    return np.array([[0.15 * np.cos(f * 0.05), 0.15 * np.sin(f * 0.05), 0.05]])


def truth(c, f):
    return cams[c].project(world(f))[0]


paths, sessions = [], []
for c, cal in enumerate(cams):
    path = os.path.join(OUT, f"review_app2_cam{c}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS3, (W3, H3))
    for f in range(N3):
        img = np.full((H3, W3, 3), 40 + 10 * c, np.uint8)
        u, v = cal.project(world(f))[0]
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (230, 230, 230), -1)
        vw.write(img)
    vw.release()
    paths.append(path)
    sessions.append(TrackingSession(path, N3, FPS3, W3, H3))
    forget_recovery(path)
PROJ = os.path.join(OUT, "review_app2.kinetrace")
Project(sessions, ["camA", "camB", "camC"], [0, 0, 0]).save(PROJ)


def open_rig():
    rd = os.environ.get("KINETRACE_RECOVERY_DIR")        # no unsaved-work copy of the last window may come back
    if rd and os.path.isdir(rd):
        shutil.rmtree(rd, ignore_errors=True)
        os.makedirs(rd, exist_ok=True)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    w._open_project_from_path(PROJ)
    wait(lambda: w.state == READY and w.project is not None and w.project.n_views == 3, 60, "open rig")
    w.project.calibration = Calibration(list(cams), source="synthetic")
    w._guides_on()
    return w


win = open_rig()
p = win.project
rec = Rec(win)
F = 20
win._goto(F, force=True)
pump(0.3)
win._on_add(*truth(0, F))
name = p.sessions[0].points[0].name
win._set_active_view(1)
win._on_annotate(*truth(1, F))
tri = win._triangulate(win._observations(name, F, exclude=-1))
check("G83 two cameras never read 'good' (the Reconstruct report caps them at ok)",
      tri is not None and tri["verdict"] == "ok" and "errors along the line are invisible" in tri.get("note", ""), tri)
check("G83 the live sentence says it", "(ok, 2 cams: errors along the line are invisible" in win._residual_sentence(name),
      win._residual_sentence(name))
win._refresh_guides()
notes = [n for cv in win.grid.canvases for n in cv._guides.notes]
check("G83 and so does the note beside the point", bool(notes) and all("errors along the line" in n[2] for n in notes),
      notes)
win._set_active_view(2)
win._on_annotate(*truth(2, F))
tri3 = win._triangulate(win._observations(name, F, exclude=-1))
check("G83 three cameras can still say 'good'", tri3 is not None and tri3["verdict"] == "good" and not tri3.get("note"), tri3)

# ---- I253: the other cameras are read at the fractional frame, like Reconstruct
win._set_active_view(0)
win._goto(F, force=True)
p.set_offset(1, 0.5)
win._set_active_view(1)
win._goto(F + 1, force=True)
win._on_annotate(*truth(1, F + 1))
win._set_active_view(0)
win._goto(F, force=True)
obs = {c: uv for c, _cal, uv in win._observations(name, F, exclude=0)}
sb = p.sessions[1]
j = sb.pid_by_name(name)
mid = (sb.tracks[F, j].astype(np.float64) + sb.tracks[F + 1, j].astype(np.float64)) / 2
check("I253 a half-frame offset reads the midpoint of the two neighbouring frames",
      1 in obs and np.allclose(obs[1], mid, atol=1e-6), (obs.get(1), mid))
p.set_offset(1, 0.0)

# ---- I258: the disagreement band uses map_frame's tie rule (no skipped frames at a .5 offset)
p.set_offset(1, 0.5)
win._set_active_view(1)
T = 30
fake = types.SimpleNamespace(per_cam=np.ones((T, 1, 3), np.float32), t0=0, n_frames=T, names=[name], xyz=None)
p.reconstruction = fake
got: list = []
win.timeline.set_disagreement = lambda a: got.append(a)
win._update_disagreement()
arr = got[-1]
rows = np.flatnonzero(np.isfinite(arr[:, 0])) if arr is not None else np.array([])
check("I258 at a .5 offset the band covers consecutive frames (np.round skipped every other one)",
      len(rows) >= T - 2 and bool(np.all(np.diff(rows) == 1)), rows[:12])
p.reconstruction = None
del win.timeline.set_disagreement
p.set_offset(1, 0.0)
win._set_active_view(0)

# ---- G68: a rename is ONE undo step in every camera
win._goto(F + 5, force=True)
win._on_select(0)
win._on_annotate(*truth(0, F + 5))               # an earlier edit (hand-placed on camA)
old = p.sessions[0].points[0].name
win._apply_rename(0, "Beak")
check("G68 (setup) renamed everywhere", [sv.points[0].name for sv in p.sessions] == ["Beak"] * 3)
win._undo_run()
check("G68 Ctrl+Z puts the old name back in EVERY camera (it left the others split)",
      [sv.points[0].name for sv in p.sessions] == [old] * 3 and bool(p.sessions[0].manual[F + 5, 0]),
      [sv.points[0].name for sv in p.sessions])
win._undo_run()
win._set_tracker([0], "spot")
check("G68 (setup) the tracker is set in every camera", [sv.points[0].tracker for sv in p.sessions] == ["spot"] * 3)
win._undo_run()
check("G68 and one Ctrl+Z puts it back in every camera", [sv.points[0].tracker for sv in p.sessions] == [""] * 3,
      [sv.points[0].tracker for sv in p.sessions])

# ---- G68: the test's Use is one step across the cameras too
use = types.SimpleNamespace(model="cotracker3", settings=None, label="CoTracker3")
win._apply_test_choice(0, use)
check("G68 (setup) Use set the tracker in every camera", [sv.points[0].tracker for sv in p.sessions] == ["cotracker3"] * 3)
win._undo_run()
check("G68 one Ctrl+Z puts the tracker back in every camera after Use",
      [sv.points[0].tracker for sv in p.sessions] == [""] * 3, [sv.points[0].tracker for sv in p.sessions])

# ---- G125 / G69: the automatic re-track
counted = []
orig_rc = win._refresh_companions
win._refresh_companions = lambda *a, **k: counted.append(1) or orig_rc(*a, **k)
win._retrack_restore({"snaps": {0: p.sessions[0].snapshot()}})
check("G125 answering No refreshes the other cameras", counted)
win._refresh_companions = orig_rc
from kinetrace import retrack as _rt  # noqa: E402

snapA, snapB = p.sessions[0].snapshot(), p.sessions[1].snapshot()
orig_cells, orig_verdict, orig_rec = _rt.cells_summary, _rt.verdict, win._reconstruct_3d
_rt.cells_summary = lambda *a, **k: {"median_px": 2.0, "max_px": 3.0}
_rt.verdict = lambda *a, **k: ("better", "better")
win._reconstruct_3d = lambda *a, **k: None
win._retrack = {"jobs": [], "done": [1], "thr": [5.0, 5.0, 5.0], "stretches": [],
                "before": {"median_px": 9.0, "max_px": 12.0}, "snaps": {0: snapA, 1: snapB},
                "prev_active": 0, "prev_frame": 3}
win._undo_snap = None
win._retrack_finish()
check("G69 Keep sets the undo point to the re-track's own snapshots",
      win._undo_snap is snapA and win._undo_extra.get(1) is snapB, (win._undo_snap is snapA, list(win._undo_extra)))
check("G69 ... and Ctrl+Z is on", win.act_undo.isEnabled())
_rt.cells_summary, _rt.verdict, win._reconstruct_3d = orig_cells, orig_verdict, orig_rec

# ---- I202 / I204 on the rig
win._goto(F, force=True)
win._set_active_view(0)
snaps = {v: p.sessions[v].snapshot() for v in (0, 1, 2)}
win._passes = {"queue": [], "groups": [[name], [name]], "labels": ["AllTracker (a)", "CoTracker3 (b)"],
               "view": 0, "frame": F, "step": False, "every": True, "snap": None, "msnaps": snaps,
               "done": [{"lasts": {0: 30, 1: 25}, "fail": (1, 25, 0), "user_stop": False, "error": False, "group": 0},
                        {"lasts": {0: 30, 1: 25}, "fail": None, "user_stop": False, "error": False, "group": 1}]}
win._passes_finish()
check("I202 an every-camera run that stopped in another camera can be undone (the switch no longer wipes the "
      "undo point)", win._undo_snap is not None and 0 in win._undo_extra and p.active == 1,
      (win._undo_snap is not None, list(win._undo_extra), p.active))
for v in (1, 2):                                   # every camera has the point on frame F
    win._set_active_view(v)
    win._on_annotate(*truth(v, F))
win._set_active_view(0)
win._goto(F, force=True)
check("I204 (setup) all three cameras have the point here", all(sv.tracked[F, 0] for sv in p.sessions))
rec.clear()
calls = []
win._start_tracking = lambda **k: calls.append(k)
win._passes = None
win._start_pass([0], False, True, segment=False, stops={0: 40}, quiet=True)
check("I204 (setup) camA starts alone, bounded by its own first pass", len(calls) == 1 and calls[0].get("stop_after") == 40, calls)
check("I204 a camera the first pass did not run in is not blamed for a missing position",
      "has no position" not in rec.text(), rec.text())
calls.clear()
win._start_pass([0], False, True, segment=False, stops={1: 40}, quiet=True)
check("I204 a pass 2 that has no camera to run in does not run", not calls, calls)
del win._start_tracking

# ---- I184 / I185: the segment in the run
for v in (0, 1, 2):
    sv = p.sessions[v]
    sv.ensure_animal()
    sv.animal.add_click(F, 320.0, 240.0, True)
win._new_point()
for sv in p.sessions:
    sv.points[sv.n_points - 1].source = "silhouette"
    sv.points[sv.n_points - 1].spec = "centroid"
    sv.move_points([sv.n_points - 1], 0)             # (G153) a derived landmark belongs to its animal
win._refresh_point_list()
dname = p.sessions[0].points[-1].name
win._set_active_view(0)
win._goto(F, force=True)
pid0 = 0
select_rows(win, pid0, p.sessions[0].n_points - 1)
jobs = win._multi_jobs(False)
check("I184 every-camera runs that re-segment also fill the derived landmarks of the run",
      bool(jobs) and all(dname in [p.sessions[j["view"]].points[q].name for q in j["pids"]] for j in jobs), jobs)
win._deselect()
win._on_select(pid0)
# (G153, G156, G160) the point is the animal's, and the animal holds its points: then its silhouette comes
s0 = p.sessions[0]
s0.move_points([pid0], 0)
check("G160 an animal's point alone runs alone (holding is opt-in)", not win._run_segment({pid0}))
s0.segments[0].hold = True
check("I185 the animal holds its points: a selected point of it brings its silhouette, its row not selected",
      not win._segment_selected() and win._run_segment({pid0}))
s0.points[pid0].free = True
check("I185 a point marked 'may leave its silhouette' does not bring it", not win._run_segment({pid0}))
s0.points[pid0].free = False
win._update_track_button()
check("I185 the Track button's tooltip says the silhouette rides along",
      "so the silhouette rides along" in win.btn_track.toolTip(), win.btn_track.toolTip())
win._deselect()
# the pass holding the head landmark goes first and carries the segment (I185)
s0 = p.sessions[0]
win._set_active_view(0)
win._goto(F, force=True)
win._on_add(300.0, 200.0)
q_head = s0.n_points - 1
s0.points[0].tracker = "alltracker"
s0.points[q_head].tracker = "cotracker3"
s0.move_points([q_head], 0)
s0.set_head(q_head)                                     # (G157) the animal's head
check("I185 (setup) the head is the CoTracker3 point", s0.head_pid() == q_head)
groups = win._tracker_passes({0, q_head})
check("I185 in a two-pass run the pass holding the head runs FIRST", groups[0] == [q_head] and groups[1] == [0],
      groups)
s0.clear_skeleton(0)
s0.points[q_head].name = "q"                             # (no head by name either)
groups = win._tracker_passes({0, q_head})
check("I185 (no skeleton: AllTracker first as before)", groups[0] == [0] and groups[1] == [q_head], groups)
win._deselect()
# these edits mark the project changed (G153's move / head); the close must not save them into the shared
# project file the next section opens
for sv in win.project.sessions:
    sv.dirty = False
win.project.dirty = False
close(win)

# ---- I196 / I189: a real every-camera run (Moving spot: no model)
print("[3b] an every-camera run")
win = open_rig()
p = win.project
F = 20
win._goto(F, force=True)
pump(0.3)
win._on_add(*truth(0, F))
nm = p.sessions[0].points[0].name
for v in (1, 2):
    win._set_active_view(v)
    win._on_annotate(*truth(v, F))
for v in (2, 1, 0):
    win._set_active_view(v)
    win._goto(F + 1, force=True)
    win._on_annotate(*truth(v, F + 1))
win._set_active_view(0)
win._goto(F, force=True)
win._set_tracker([0], "spot")
win.act_track_all.setChecked(True)
win._deselect()
win._on_select(0)
check("I189 (setup) the start frame is hand-placed in every camera", all(sv.manual[F, 0] for sv in p.sessions))
win._toggle_tracking()
wait(lambda: win._multi is not None or win.state == TRACKING, 60, "every-camera start")
wait(lambda: win._multi is None and win.state == READY, 240, "every-camera end")
pump(0.6)
check("I189 every camera keeps its start frame hand-placed after the run",
      all(sv.manual[F, 0] for sv in p.sessions), [bool(sv.manual[F, 0]) for sv in p.sessions])
check("I196 the finished run let go of its workers (no _SideRun left on the window)",
      not win.findChildren(_SideRun), len(win.findChildren(_SideRun)))
check("(the run did track)", all(sv.tracked[F + 3, 0] for sv in p.sessions))
close(win)

# ================================================================== [4] a real two-pass step
print("[4] a real two-pass run on the models")
alltracker_backend.available = REAL_AT
if not REAL_AT():
    print("  (AllTracker is not installed here: section [4] skipped)")
else:
    win = new_window(VID)
    s = win.session
    rec = Rec(win)
    win.act_mode_semi.trigger()
    pump(0.1)
    win._on_add(100.0, 100.0)
    win._on_add(300.0, 200.0)
    win._on_add(420.0, 330.0)
    for q2 in range(3):
        s.points[q2].tracker = "alltracker" if q2 == 0 else "cotracker3"
    win._refresh_point_list()
    # P1 (AllTracker) was placed on frame 0 only; at frame 5 only the CoTracker3 points have a position
    win._goto(5, force=True)
    for q2 in (1, 2):
        win._on_select(q2)
        win._on_annotate(*[(100, 100), (300, 200), (420, 330)][q2])
    select_rows(win, 0, 1)
    win._on_select(0)
    select_rows(win, 0, 1)
    win._toggle_tracking()
    try:
        wait(lambda: win.state == READY and win._passes is None and win._multi is None, 40, "step 1")
    except TimeoutError:
        pass                         # the old code left a pass queued that never started
    pump(0.3)
    win._passes = None
    check("I203 with the AllTracker point unplaced here, the press steps the CoTracker3 point",
          bool(s.tracked[6, 1]) and not s.tracked[6, 0], (bool(s.tracked[6, 1]), bool(s.tracked[6, 0])))
    win._undo_run()
    win._goto(5, force=True)
    win._on_select(0)
    win._on_annotate(100.0, 100.0)
    select_rows(win, 0, 1)
    win._on_select(0)
    select_rows(win, 0, 1)
    rec.clear()
    status: list = []
    _orig_show = win.statusBar().showMessage
    win.statusBar().showMessage = lambda m, ms=0: (status.append(m), _orig_show(m, ms))[1]
    win._toggle_tracking()
    wait(lambda: win.state == READY and win._passes is None and win._multi is None, 240, "step 2")
    pump(0.3)
    check("G100 a semi-automatic two-pass step tracks both points one frame",
          bool(s.tracked[6, 0]) and bool(s.tracked[6, 1]) and not s.tracked[7, 0] and not s.tracked[7, 1])
    check("G100 ... and its summary is a status line, not a 12 s notice",
          not [m for m in rec.msgs if "Two passes" in m[0]] and any("Two passes" in m for m in status),
          (rec.text(), status))
    check("G100 ... with the playhead on the stepped frame", win.current == 6, win.current)
    win._undo_run()
    pump(0.1)
    check("one Ctrl+Z undoes both passes of the step", not s.tracked[6, 0] and not s.tracked[6, 1])
    close(win)


# ---------------------------------------------------------------- the end
if FAILS:
    print(f"VERIFY_REVIEW_APP2 FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("VERIFY_REVIEW_APP2 PASSED")
