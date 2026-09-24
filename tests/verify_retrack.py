"""Automatic epipolar-constrained re-tracking (retrack.py + the app), GPU.

A 3-camera synthetic rig (known DLT cameras, three landmarks drawn as small
bright dots on a dark background so the point tracker can follow them). The
tracks are written exactly into every camera, then camera C's "head" is slid
25 px off over frames 30-60: after a reconstruction that camera disagrees
there (the magenta band). The planner must find exactly that stretch and put
the ray target within 1.5 px of the truth; the automatic run (re-seed on the
rays, re-track through the stretch with the real worker, reconstruct again)
must bring the slid frames back within 2.5 px of the truth and report
BETTER; the Undo answer must restore everything.

Run: .venv\\Scripts\\python.exe tests\\verify_retrack.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession
from kinetrace import retrack

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(4)
W, H = 640, 480
N_FR = 120
FPS = 60.0
NAMES = ["head", "hip", "tail"]
COLORS = [(80, 220, 255), (255, 200, 80), (200, 120, 255)]


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])
cams = []
for k, a in enumerate((0.3, 2.4, 4.4)):
    pos = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.5 + 0.4 * k])
    R, t = look_at(pos)
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))


def world_at(t_ref):
    s = float(t_ref)
    hip = np.array([0.3 * np.cos(s * 0.05), 0.3 * np.sin(s * 0.05), 0.003 * s])
    return np.stack([hip + [0.08, 0.0, 0.03], hip, hip - [0.09, 0.0, 0.01]])


sessions, paths, truth = [], [], []
for c, cal in enumerate(cams):
    p = os.path.join(OUT, f"retrack_cam{c}.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    s = TrackingSession(p, N_FR, FPS, W, H)
    for nm in NAMES:
        s.add_landmark(nm)
    tr = np.zeros((N_FR, 3, 2))
    for f in range(N_FR):
        img = np.full((H, W, 3), 28, np.uint8)
        # a faint texture so the tracker has context, then a distinct dot per landmark
        img[..., 0] = (28 + 12 * np.sin(np.arange(W) / 19.0)[None, :] + 8 * np.cos(np.arange(H) / 13.0)[:, None]).astype(np.uint8)
        uv = cal.project(world_at(f))
        for j in range(3):
            cv2.circle(img, (int(round(uv[j][0])), int(round(uv[j][1]))), 5, COLORS[j], -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(uv[j][0])), int(round(uv[j][1]))), 2, (20, 20, 20), -1, cv2.LINE_AA)
            tr[f, j] = np.round(uv[j])
            s.set_position(f, j, float(tr[f, j, 0]), float(tr[f, j, 1]))
        s.manual[f] = False
        vw.write(img)
    vw.release()
    sessions.append(s)
    paths.append(p)
    truth.append(tr)
SLID_CAM, SLID_PID, SLID = 2, 0, (30, 60)
sc = sessions[SLID_CAM]
for f in range(SLID[0], SLID[1] + 1):
    sc.tracks[f, SLID_PID] = truth[SLID_CAM][f, SLID_PID] + [25.0, 0.0]
proj = Project(sessions, ["camA", "camB", "camC"], [0.0, 0.0, 0.0])
proj.calibration = Calibration(cams)
PROJ = os.path.join(OUT, "retrack.cotrk")
proj.save_npz(PROJ)
for p in paths:
    if os.path.exists(p + ".cotracker.npz"):
        os.remove(p + ".cotracker.npz")

ANSWERS = {"question": QMessageBox.Yes}
QMessageBox.question = staticmethod(lambda *a, **k: ANSWERS["question"])
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from kinetrace.app import READY, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(cond, timeout, what):
    t0 = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


win._open_project_from_path(PROJ)
wait(lambda: win.state == READY and win.project is not None and win.project.n_views == 3, 120, "open")
p = win.project
assert p.calibration is not None
win._reconstruct_3d(quiet=True)
pump(0.2)
assert p.reconstruction is not None and p.reconstruction.per_cam is not None
thr = win._disagree_thresholds()

# ---- the planner ------------------------------------------------------------------------
plan = retrack.plan(p, thr)
print(f"planner: {len(plan)} stretch(es): " + "; ".join(
    f"{p.name(st.view)} {st.name} {st.local0}-{st.local1} {st.median_px:.1f} px, target {st.target}" for st in plan))
# with three cameras the slide spreads the residual over all three (8-10 px each): the
# leave-one-out test must blame camera C alone and name it in the others' reasons
doable = [st for st in plan if st.target is not None]
skipped = [st for st in plan if st.target is None]
assert len(doable) == 1, plan
st = doable[0]
assert st.view == SLID_CAM and st.name == NAMES[SLID_PID]
assert st.rest_px < 1.0 < st.full_px, (st.rest_px, st.full_px)
assert len(skipped) == 2 and all(p.name(SLID_CAM) in q.reason for q in skipped), [q.reason for q in skipped]
assert abs(st.local0 - SLID[0]) <= 1 and abs(st.local1 - SLID[1]) <= 1, (st.local0, st.local1)
assert st.target is not None and st.n_rays == 2
d = np.linalg.norm(st.target - truth[SLID_CAM][st.local0, SLID_PID])
print(f"ray target {d:.2f} px from the truth on frame {st.local0}")
assert d < 1.5, d
before = retrack.cells_summary(p, plan)
assert before["median_px"] > thr[SLID_CAM], before
assert win.act_retrack.isEnabled()
# nothing to do on a clean rig: the planner says so
clean = Project.load_npz(PROJ)
clean.calibration = Calibration(cams)
clean.sessions[SLID_CAM].tracks[SLID[0]:SLID[1] + 1, SLID_PID] = truth[SLID_CAM][SLID[0]:SLID[1] + 1, SLID_PID]
from kinetrace.calib import reconstruct  # noqa: E402
clean.reconstruction = reconstruct(clean.sessions, clean.calibration, clean.rates, clean.offsets, (0, N_FR - 1))
assert retrack.plan(clean, thr) == [], "a clean rig has no disagreeing stretch"
print("planner OK")

# ---- the automatic run, kept ---------------------------------------------------------------
t0 = time.time()
win._retrack_dialog()               # question -> Yes: runs
wait(lambda: win._retrack is None and win.state == READY, 900, "re-track run")
pump(0.3)
res = win._retrack_last
print(f"re-track finished in {time.time() - t0:.0f}s: verdict {res['verdict']}, "
      f"{res['before']['median_px']:.1f} px -> {res['after']['median_px']:.1f} px (median), {res['n_done']} stretch(es)")
assert res is not None and res["n_done"] == 1
assert res["verdict"] == "better", res
sc = p.sessions[SLID_CAM]
got = sc.tracks[SLID[0]:SLID[1] + 1, SLID_PID]
err = np.linalg.norm(got - truth[SLID_CAM][SLID[0]:SLID[1] + 1, SLID_PID], axis=1)
print(f"slid frames after re-tracking: error median {np.nanmedian(err):.2f} px, max {np.nanmax(err):.2f} px, "
      f"tracked {int(np.isfinite(got).all(axis=1).sum())}/{SLID[1] - SLID[0] + 1}")
assert np.isfinite(got).all(axis=1).sum() >= SLID[1] - SLID[0] - 1
assert np.nanmedian(err) < 2.5, np.nanmedian(err)
assert res["after"]["median_px"] < thr[SLID_CAM]
# (the re-seeded frame is the run's seed: write_segment clears the hand-placed flag where the model
#  writes, exactly as for a user click followed by Track - the position itself is the ray target)
assert p.active == 0 and win.current == 0, "the working camera and frame are restored"
assert retrack.plan(p, thr) == [], "no stretch left after the fix"
print("automatic re-track (kept) OK")

# ---- the Undo answer restores every camera ----------------------------------------------------
for f in range(SLID[0], SLID[1] + 1):
    sc.tracks[f, SLID_PID] = truth[SLID_CAM][f, SLID_PID] + [25.0, 0.0]
sc.manual[SLID[0]:SLID[1] + 1, SLID_PID] = False
sc._touch()
win._reconstruct_3d(quiet=True)
pump(0.2)
snap_before = sc.tracks.copy()
ANSWERS["question"] = QMessageBox.No          # the run's confirm AND the keep question say No...
win._retrack_dialog()                          # ... so this one does not even start
pump(0.2)
assert win._retrack is None and np.allclose(sc.tracks, snap_before, equal_nan=True)


class _Seq:
    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, *a, **k):
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


QMessageBox.question = staticmethod(_Seq([QMessageBox.Yes, QMessageBox.No]))   # run: Yes, keep: No
win._retrack_dialog()
wait(lambda: win._retrack is None and win.state == READY, 900, "re-track run 2")
pump(0.3)
assert np.allclose(sc.tracks, snap_before, equal_nan=True), "Undo must restore the slid camera exactly"
print("Undo answer restores everything OK")
win.close()
pump(0.3)
for pth in paths:
    if os.path.exists(pth + ".cotracker.npz"):
        os.remove(pth + ".cotracker.npz")
print("verify_retrack PASSED")
