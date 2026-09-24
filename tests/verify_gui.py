"""Scripted end-to-end GUI test (offscreen QPA): open video, add points, track,
pause-correct-resume, undo, export, project round-trip."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

VID = os.path.join(ROOT, r"test600.mp4")
GT = np.load(VID + ".gt.npz")["gt"]
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)

# autosave from earlier runs would trigger a resume prompt; remove for determinism
for leftover in (VID + ".cotracker.npz",):
    if os.path.exists(leftover):
        os.remove(leftover)

# never let a dialog block the scripted run
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"critical dialog: {a[2] if len(a)>2 else a}")))

app = QApplication([])
from cotracker_app.app import MainWindow, READY, TRACKING

win = MainWindow()
win.show()


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timeout waiting for: {what}")


# ---- open video ----
win._open_video(VID)
pump(lambda: win.state == READY, 20, "video open")
assert win.n_frames == 600 and win.session is not None
assert win.btn_track.isEnabled() is False, "Track must be disabled with no points"

# ---- add 4 points at GT frame 0 ----
for d in range(4):
    win._on_add(float(GT[0, d, 0]), float(GT[0, d, 1]))
assert win.session.n_points == 4 and win.point_list.count() == 4
assert win.btn_track.isEnabled(), "Track should enable once points exist"

# ---- full tracking run ----
win.point_list.clearSelection()   # run scope = panel selection; none = everything
win._toggle_tracking()
pump(lambda: win.state == TRACKING, 120, "tracking start (model load)")
pump(lambda: win.state == READY, 300, "tracking finish")
s = win.session
assert s.tracked.all(), "not all frames tracked"
err = np.linalg.norm(s.tracks - GT, axis=-1)
print(f"full run mean err {err.mean():.2f} px, max {err.max():.2f} px")
assert err.mean() < 6.0
assert win.act_undo.isEnabled()

# ---- pause mid-run ----
win._goto(0)
win.point_list.clearSelection()   # run scope = panel selection; none = everything
win._toggle_tracking()
pump(lambda: win.state == TRACKING, 60, "second run start")
pump(lambda: win.current > 100, 120, "progress past frame 100")
win._pause_tracking()
pump(lambda: win.state == READY, 30, "pause")
paused_at = win.current
print(f"paused at frame {paused_at}")
assert 100 < paused_at < 599

# ---- A: mis-placed seed must track the PLACED location (user intent honored) ----
win._goto(250)
wrong = GT[250, 0] + np.array([50.0, 30.0])  # empty static background
win._on_place(0, float(wrong[0]), float(wrong[1]))
assert s.manual[250, 0] and np.allclose(s.tracks[250, 0], wrong, atol=0.5)
pre_run = s.tracks.copy()  # undo restores to the state at Track-click (incl. the edit)
win.point_list.clearSelection()   # run scope = panel selection; none = everything
win._toggle_tracking()
pump(lambda: win.state == TRACKING, 60, "resume start")
pump(lambda: win.state == READY, 300, "resume finish")
assert np.allclose(s.tracks[250, 0], wrong, atol=0.5), "seed frame must keep user value"
# the seeded background spot is static: the track must stay near it, not follow the dot
d_static = np.linalg.norm(s.tracks[251:290, 0] - wrong, axis=-1)
print(f"mis-seed: distance from placed spot @251-290: mean {d_static.mean():.2f} px")
assert d_static.mean() < 12.0, "tracker did not follow the user-placed location"
assert not s.manual[251:, 0].any(), "manual flags beyond seed row must clear"

# ---- undo last run restores the pre-run tracks exactly ----
win._undo_run()
assert np.allclose(s.tracks, pre_run, equal_nan=True), "undo did not restore pre-run state"
print("undo restored pre-run state")

# ---- B: correction ON the dot at 250 -> re-track -> follows the dot accurately ----
win._goto(250)
for d in range(4):
    win._on_place(d, float(GT[250, d, 0]), float(GT[250, d, 1]))
win.point_list.clearSelection()   # run scope = panel selection; none = everything
win._toggle_tracking()
pump(lambda: win.state == TRACKING, 60, "clean re-track start")
pump(lambda: win.state == READY, 300, "clean re-track finish")
err_b = np.linalg.norm(s.tracks[250:, 0] - GT[250:, 0], axis=-1)
print(f"correction-on-dot: err @250-600: mean {err_b.mean():.2f} px")
assert err_b.mean() < 6.0

# ---- export all formats ----
csv_path = os.path.join(SCRATCH, "gui_tracks.csv")
win.session.export_csv(csv_path)
win.session.export_tsv_sparse(os.path.join(SCRATCH, "gui_tracks.tsv"))
win.session.export_mat(os.path.join(SCRATCH, "gui_tracks.mat"))

# ---- project save / load round-trip ----
proj = os.path.join(SCRATCH, "gui_test.cotrk")
win.project_path = None
win.session.current_frame = 321
win.session.save_npz(proj)
win._open_project_from_path(proj)
pump(lambda: win.state == READY and win.current == 321, 30, "project reopen at saved frame")
assert win.session.n_points == 4
print("project round-trip OK (reopened at frame 321)")

# ---- run scope = the panel selection: only the selected point is re-tracked ----
s = win.session
win._goto(300)
s.clear_window(list(range(4)), 301, 599)          # everything after 300 is blank now
assert not s.tracked[301:].any()
win.point_list.clearSelection()
win.point_list.setCurrentRow(2)                    # a single selected point
assert win._selected_pids() == [2] and win._run_scope()[0] == {2}
assert "selected point" in win.btn_track.toolTip()
win._start_tracking(stop_after=340)
pump(lambda: win.state == TRACKING, 10, "scoped run start")
pump(lambda: win.state == READY, 120, "scoped run end")
assert s.tracked[301:341, 2].all(), "the selected point must be tracked"
assert not s.tracked[301:, [0, 1, 3]].any(), "unselected points must be left alone"
# select everything -> all points track
win._goto(300)                                     # all four have data here
for r in range(4):
    win.point_list.item(r).setSelected(True)
assert win._run_scope()[0] is None and "all 4 point" in win.btn_track.toolTip(), win.btn_track.toolTip()
win._start_tracking(stop_after=330)
pump(lambda: win.state == TRACKING, 10, "full run start")
pump(lambda: win.state == READY, 120, "full run end")
assert s.tracked[301:331].all(), "all points must be tracked when all are selected"
print("run scope follows the panel selection OK")

win.close()
app.processEvents()
print("GUI E2E PASSED")
