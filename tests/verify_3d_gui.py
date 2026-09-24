"""Offscreen GUI: the 3D layer wired into the app — a 3-camera synthetic
project (known DLT cameras, a landmark constellation flying a known path,
disc silhouettes) opened through the real MainWindow, then Import
Calibration -> Estimate Sub-frame Offsets -> Reconstruct -> Carve Volume ->
3D view -> exports -> project round-trip. No GPU: the tracks and silhouettes
are written into the sessions directly (the tracker has its own suites).

Run: .venv\\Scripts\\python.exe tests\\verify_3d_gui.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.RandomState(11)

W, H = 320, 240
N_FR = [90, 180, 90]                  # camera 2 runs at 2x
FPS = [60.0, 120.0, 60.0]
TRUE_OFF = [0.0, 20.4, -5.3]          # local frames at reference instant 0
RATES = [1.0, 2.0, 1.0]
NAMES = ["head", "hip", "tail"]
BODY_R = 0.05                         # the "animal": a sphere of 5 cm around the hip


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[420.0, 0, W / 2], [0, 420.0, H / 2], [0, 0, 1]])
cams = []
for k, a in enumerate((0.3, 2.4, 4.4)):
    pos = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.5 + 0.4 * k])
    R, t = look_at(pos)
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))


def world_at(t_ref):
    s = np.asarray(t_ref, np.float64)[..., None]
    hip = np.stack([0.3 * np.cos(s[..., 0] * 0.06), 0.3 * np.sin(s[..., 0] * 0.06), 0.003 * s[..., 0]], -1)
    return np.stack([hip + [0.06, 0.0, 0.02], hip, hip - [0.07, 0.0, 0.01]], axis=-2)   # (..., 3, 3)


# ---- synthetic videos, sessions (tracks + silhouettes), project file --------
sessions = []
paths = []
for c, cal in enumerate(cams):
    p = os.path.join(SCRATCH, f"cam3d_{c}.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS[c], (W, H))
    s = TrackingSession(p, N_FR[c], FPS[c], W, H)
    for nm in NAMES:
        s.add_landmark(nm)
    s.ensure_animal()
    for f in range(N_FR[c]):
        t_ref = (f - TRUE_OFF[c]) / RATES[c]
        img = np.full((H, W, 3), 30, np.uint8)
        X = world_at(t_ref)
        uv = cal.project(X)
        # silhouette: the sphere around the hip, drawn from its projected radius
        hip_uv = uv[1]
        edge = cal.project(X[1] + [0.0, 0.0, BODY_R])[0]
        r_px = max(2.0, float(np.linalg.norm(edge - hip_uv)))
        mask = np.zeros((H, W), np.uint8)
        if 0 <= t_ref <= 80:
            cv2.circle(img, (int(round(hip_uv[0])), int(round(hip_uv[1]))), int(round(r_px)), (200, 200, 200), -1)
            cv2.circle(mask, (int(round(hip_uv[0])), int(round(hip_uv[1]))), int(round(r_px)), 1, -1)
            for j in range(3):
                s.set_position(f, j, float(uv[j][0] + rng.normal(0, 0.2)), float(uv[j][1] + rng.normal(0, 0.2)))
            s.masks.set(f, mask.astype(bool), 9.0)
        vw.write(img)
    vw.release()
    sessions.append(s)
    paths.append(p)
proj = Project(sessions, ["camA", "camB", "camC"], [round(o) for o in TRUE_OFF])
assert proj.rates == RATES, proj.rates
PROJ = os.path.join(SCRATCH, "test3d_gui.kinetrace")
proj.save(PROJ)
csv = os.path.join(SCRATCH, "test3d_gui_dltCoefs.csv")
np.savetxt(csv, np.stack([c.coefs for c in cams], 1), delimiter=",", fmt="%.10g")
for p in paths:
    forget_recovery(p)

# ---- the app ---------------------------------------------------------------
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace import view3d as v3  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


win._open_project_from_path(PROJ)
for _ in range(200):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == 3:
        break
assert win.project is not None and win.project.n_views == 3 and win.state == READY
assert win.project.rates == RATES
print("3-camera project opened through the app OK")

# calibration dialog: choose the csv, MATLAB convention would be wrong here -> pick OpenCV
_orig_exec = v3.CalibrationDialog.exec


def _fake_exec(self):
    assert self.load(csv)
    self.conv.setCurrentIndex(2)          # OpenCV, 0-based: how the synthetic cameras were built
    self._accept()
    return QDialog.Accepted


v3.CalibrationDialog.exec = _fake_exec
# before a calibration the 3D entries stay clickable and SAY what they need (I115)
assert win.act_calib.isEnabled() and win.act_recon.isEnabled()
_said = []
_orig_info = QMessageBox.information
QMessageBox.information = staticmethod(lambda *a, **k: (_said.append(str(a[2]) if len(a) > 2 else ""),
                                                        QMessageBox.Ok)[1])
win._reconstruct_3d()
QMessageBox.information = _orig_info
assert _said and "calibration" in _said[0].lower(), _said
assert win.project.reconstruction is None
win._import_calibration()
pump()
p = win.project
assert p.calibration is not None and len(p.calibration) == 3
assert p.calibration.cameras[1].pixel_origin == 0.0 and not p.calibration.cameras[1].y_flip
assert win.act_recon.isEnabled() and win.act_offsets3d.isEnabled() and win.act_hull.isEnabled()
print("calibration imported through the dialog OK")

# ---- the import dialog: conventions, sizes, what the importer could not read ---------
_asked = []
_q_prev = QMessageBox.question
QMessageBox.question = staticmethod(lambda *a, **k: _asked.append(a[1]) or QMessageBox.Yes)
# (I96) an OpenCV K + R/t TEXT file is 0-based by construction: the convention is
# locked (the .txt form kept the MATLAB default and shifted every track by 1 px),
# and the moved world origin survives the dialog
_Kt = np.array([[2783.0, 0, 1920.0], [0, 2784.0, 1080.0], [0, 0, 1.0]])
_Tt = np.eye(4)
_Tt[:3, 3] = [-0.5, 0.0, 0.02]
_rows_t = lambda M: "\n".join(" ".join(f"{v:.7f}" for v in row) for row in M)   # noqa: E731
_txt = os.path.join(SCRATCH, "test3d_gui_krt.txt")
open(_txt, "w", encoding="utf-8").write(
    f"Intrinsics at 3840x2160\nK1=[{_rows_t(_Kt)}]\nK2=[{_rows_t(_Kt)}]\nT12=[{_rows_t(_Tt)}]\n")
dlg = v3.CalibrationDialog(win, ["left", "right"], [(3840, 2160)] * 2)
assert dlg.load(_txt)
assert not dlg.conv.isEnabled() and v3.CONVENTIONS[dlg.conv.currentIndex()][1] == 0.0, dlg.conv.currentIndex()
assert "No lens distortion in this file" in dlg.info.text(), dlg.info.text()
dlg._accept()
_rc = dlg.result_calibration
assert _rc is not None and [c.pixel_origin for c in _rc.cameras] == [0.0, 0.0], [c.pixel_origin for c in _rc.cameras]
assert _rc.origin_shift is not None and np.allclose(_rc.origin_shift, dlg._loaded.origin_shift)
dlg.deleteLater()
# (I99) a dltCoefs.csv records no picture sizes: none is shown or compared, so a
# correct non-identity mapping is not argued with (views of mixed sizes)
dlg = v3.CalibrationDialog(win, ["A", "B", "C"], [(W, H), (848, 480), (W, H)])
assert dlg.load(csv)
_labels = [dlg._combos[0].itemText(k) for k in range(dlg._combos[0].count())]
assert _labels == ["cam 1", "cam 2", "cam 3"], _labels
assert "does not record picture sizes" in dlg.info.text(), dlg.info.text()
dlg.conv.setCurrentIndex(2)
dlg._combos[1].setCurrentIndex(2)
dlg._combos[2].setCurrentIndex(1)
_asked.clear()
dlg._accept()
assert not _asked and dlg.result_calibration is not None, _asked
assert dlg.result_calibration.cameras[1].width == 848 and np.allclose(dlg.result_calibration.cameras[1].coefs,
                                                                       cams[2].coefs)
dlg.deleteLater()
# (I102) a DLTdv project whose lens store cannot be read SAYS so, instead of
# telling the user to use the DLTdv project file they just chose
from scipy.io import savemat as _savemat  # noqa: E402
import scipy.io as _sio  # noqa: E402
_dvp = os.path.join(SCRATCH, "test3d_gui_dvProject.mat")
_savemat(_dvp, {"udExport": {"data": {"dltcoef": np.stack([c.coefs for c in cams], 1),
                                      "movsizes": np.array([[H, W]] * 3)}}})
dlg = v3.CalibrationDialog(win, ["camA", "camB", "camC"], [(W, H)] * 3)
assert dlg.load(_dvp) and "carries no lens undistortion" in dlg.info.text(), dlg.info.text()
_lm = _sio.loadmat
_sio.loadmat = lambda *a, **k: dict(_lm(*a, **k), __function_workspace__=np.frombuffer(b"not a MAT stream" * 8,
                                                                                         np.uint8))
try:
    assert dlg.load(_dvp)
finally:
    _sio.loadmat = _lm
assert "could not be read" in dlg.info.text() and "needs the DLTdv project file" not in dlg.info.text(), \
    dlg.info.text()
dlg.deleteLater()
QMessageBox.question = _q_prev
print("import dialog: K + R/t text locked to 0-based, no invented CSV sizes, lens-store failures said OK "
      "(I96, I99, I102)")

# reconstruct with whole-frame offsets, then refine
win._reconstruct_3d()
pump()
r0 = p.reconstruction
assert r0 is not None and r0.names == NAMES
valid = np.isfinite(r0.xyz).all(axis=2)
assert valid.sum() > 150, valid.sum()
t = np.arange(r0.t0, r0.t0 + r0.n_frames)
e0 = np.linalg.norm(r0.xyz - world_at(t), axis=2)
assert win.view3d is not None and win.view3d.isVisible() and win.act_view3d.isChecked()
win._estimate_offsets_dialog()
pump()
assert np.abs(np.array(p.offsets) - np.array(TRUE_OFF)).max() < 0.08, p.offsets
r1 = p.reconstruction
e1 = np.linalg.norm(r1.xyz - world_at(np.arange(r1.t0, r1.t0 + r1.n_frames)), axis=2)
assert np.nanmean(e1) < 0.4 * np.nanmean(e0) and np.nanmean(e1) < 1.5e-3, (np.nanmean(e0), np.nanmean(e1))
print(f"reconstruct + sub-frame offsets OK: {np.nanmean(e0) * 1e3:.2f} mm -> {np.nanmean(e1) * 1e3:.2f} mm, "
      f"offsets {np.round(p.offsets, 3).tolist()} vs true {TRUE_OFF}")
# ---- epipolar guides, snap-to-rays, per-camera disagreement on the timeline ----------
from kinetrace.calib import closest_on_polyline as _cop
s_act = win.session
pid0 = s_act.pid_by_name(NAMES[0])
mid = int(np.nonzero(s_act.tracked[:, pid0])[0][len(np.nonzero(s_act.tracked[:, pid0])[0]) // 2])
win._goto(mid)
win._on_select(pid0)
pump(0.1)
guides = win._epipolar_guides(pid0)
assert len(guides) == 2, f"two other cameras see {NAMES[0]}: {len(guides)} guides"
truth = s_act.tracks[mid, pid0].copy()          # a copy: set_position below writes into the array
for pts, colour, cam in guides:
    d = np.linalg.norm(_cop(pts, truth) - truth)
    assert d < 1.5, f"guide from {cam} misses the true position by {d:.2f} px"
assert win.canvas._guides.lines and len(win.canvas._guides.lines) == 2, "guides drawn on the canvas"
win.act_epipolar.setChecked(False)
pump(0.05)
assert not win.canvas._guides.lines, "the View toggle hides the guides"
win.act_epipolar.setChecked(True)
pump(0.05)
# snap: displace the landmark here, then let the other cameras' rays put it back
s_act.set_position(mid, pid0, float(truth[0] + 30.0), float(truth[1] - 20.0))
win._snap_to_epipolar(pid0)
pump(0.05)
back = s_act.tracks[mid, pid0]
assert np.linalg.norm(back - truth) < 1.0, f"snap landed {np.linalg.norm(back - truth):.2f} px off"
assert s_act.manual[mid, pid0], "a snapped frame is hand-placed"
win._undo_run()
assert np.allclose(s_act.tracks[mid, pid0], [truth[0] + 30.0, truth[1] - 20.0]), "Ctrl+Z undoes the snap"
s_act.set_position(mid, pid0, float(truth[0]), float(truth[1]))
# the snap entry exists in the point menu, with the camera count
menu, acts = win.canvas._build_context_menu(pid0)
assert "2 cameras" in acts["snap_epipolar"].text() and acts["snap_epipolar"].isEnabled()
# per-camera disagreement: present on the timeline after a reconstruction, clean on clean data
assert p.reconstruction.per_cam is not None and p.reconstruction.per_cam.shape[2] == 3
win._update_disagreement()
dis = win.timeline._disagree
assert dis is not None and dis.shape == (s_act.n_frames, s_act.n_points) and np.isfinite(dis).any()
assert np.nanmax(dis) < 2.0, f"clean synthetic cameras must not disagree ({np.nanmax(dis):.2f} px)"
# a landmark slid onto the wrong spot in THIS camera shows up as disagreement here
frames_bad = np.nonzero(s_act.tracked[:, pid0])[0][:20]
saved = s_act.tracks[frames_bad, pid0].copy()
for fb in frames_bad:
    s_act.set_position(int(fb), pid0, float(saved[list(frames_bad).index(fb), 0] + 25.0), float(saved[list(frames_bad).index(fb), 1]))
win._reconstruct_3d(quiet=True)
pump(0.1)
dis2 = win.timeline._disagree
loc = np.round(p.rates[p.active] * np.arange(p.reconstruction.t0, p.reconstruction.t0 + p.reconstruction.n_frames)
               + p.offsets[p.active]).astype(int)
bad_cells = dis2[frames_bad, pid0]
assert np.nanmedian(bad_cells) > win.timeline._disagree_px, f"the slid frames must exceed the band threshold ({np.nanmedian(bad_cells):.1f} px)"
win.timeline.refresh()
pump(0.05)
assert win.timeline._dis_col is not None and (win.timeline._dis_col[:, pid0] > win.timeline._disagree_px).any()
# the automatic re-track's planner (retrack.py) finds exactly this stretch in this camera
from kinetrace import retrack  # noqa: E402
plan_ = retrack.plan(p, win._disagree_thresholds())
mine = [st for st in plan_ if st.view == p.active and st.name == NAMES[0]]
assert len(mine) == 1, [(st.view, st.name, st.local0, st.local1) for st in plan_]
st0 = mine[0]
# the first frames of the slide have only two cameras (camC starts later): there nobody can
# be blamed, so the stretch may begin a few frames in
assert int(frames_bad[0]) <= st0.local0 <= int(frames_bad[0]) + 8 and abs(st0.local1 - int(frames_bad[-1])) <= 1, \
    (st0.local0, st0.local1)
assert st0.target is not None and st0.n_rays == 2
truth0 = saved[st0.local0 - int(frames_bad[0])]           # the true position on the stretch's FIRST frame
assert np.linalg.norm(st0.target - truth0) < 2.5, f"ray target {np.linalg.norm(st0.target - truth0):.2f} px off"   # 2 cameras, 0.2 px track noise at 320 px wide
assert win.act_retrack.isEnabled()
print(f"re-track planner: 1 stretch, frames {st0.local0}-{st0.local1}, {st0.median_px:.1f} px off, target within "
      f"{np.linalg.norm(st0.target - truth0):.2f} px OK")
for k, fb in enumerate(frames_bad):
    s_act.set_position(int(fb), pid0, float(saved[k, 0]), float(saved[k, 1]))
s_act.manual[frames_bad, pid0] = False
win._reconstruct_3d(quiet=True)
pump(0.1)
print("epipolar guides, snap to rays and per-camera disagreement band OK")
# ---- kinematics export through the app (the smoothing question stubbed to Automatic) ----
from PySide6.QtWidgets import QInputDialog  # noqa: E402
QInputDialog.getItem = staticmethod(lambda *a, **k: (a[3][0], True))
kin_out = os.path.join(SCRATCH, "test3d_gui_kinematics.csv")
written_k = win._export_one("kin", kin_out)
assert len(written_k) == 2 and all(os.path.exists(w) for w in written_k), written_k
assert "Smoothing: automatic" in open(written_k[1], encoding="utf-8").read()
print("3D kinematics export OK")

# the camera panel shows the fractional offsets and the x2 rate
row = win.cameras._rows[1]
assert abs(row.spin.value() - p.offsets[1]) < 1e-3 and row.rate.text() == "×2", (row.spin.value(), row.rate.text())

# volume hull at a mid-flight frame (in the reference view's numbering)
win._goto(40)
pump()
win._carve_hull_here()
pump()
assert 40 in win._hull_cache, "hull not carved"
verts, faces, h = win._hull_cache[40]
V = 4 / 3 * np.pi * BODY_R ** 3
assert 0.9 * V < h.volume() < 2.2 * V, (h.volume(), V)     # 3 views: an over-estimate, but the right size
assert np.linalg.norm(h.centroid() - world_at(40.0)[1]) < 0.01
assert win.act_export_mesh.isEnabled()
win.view3d._render()
img = win.view3d.last_image
assert img is not None and img.std() > 5
win.view3d.save_png(os.path.join(SCRATCH, "test3d_gui_view.png"))
# orbit / zoom interaction changes the picture
az0 = win.view3d.azimuth
win.view3d.azimuth += 40
win.view3d._render()
assert not np.array_equal(win.view3d.last_image, img)
print(f"volume hull OK: {h.n_views} views, {h.volume() * 1e6:.1f} cm3 (sphere {V * 1e6:.1f} cm3), "
      f"{len(faces)} triangles; 3D view renders")

# exports: xyz through the export dialog, mesh through its dialog
QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (os.path.join(SCRATCH, "test3d_gui_xyz.csv"),
                     "3D landmarks — xyz per reference frame + residual sidecar (*.csv)"))
win._export_dialog()
pump()
assert os.path.exists(os.path.join(SCRATCH, "test3d_gui_xyz.csv"))
assert os.path.exists(os.path.join(SCRATCH, "test3d_gui_xyz_xyzres.csv"))
rows = open(os.path.join(SCRATCH, "test3d_gui_xyz.csv"), encoding="utf-8").read().splitlines()
assert rows[0] == "frame," + ",".join(f"{n}_{ax}" for n in NAMES for ax in "XYZ")
QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (os.path.join(SCRATCH, "test3d_gui_hull.obj"), "Wavefront OBJ (*.obj)"))
win._export_mesh()
pump()
obj = open(os.path.join(SCRATCH, "test3d_gui_hull.obj"), encoding="utf-8").read().splitlines()
assert sum(1 for l in obj if l.startswith("f ")) == len(faces)
# "Everything" includes the xyz file
QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (os.path.join(SCRATCH, "test3d_gui_all.csv"),
                     "Everything — all of the above with one base name (*.csv)"))
win._export_dialog()
pump()
assert os.path.exists(os.path.join(SCRATCH, "test3d_gui_all_xyz.csv"))
print("exports OK (xyz CSV + residual sidecar, OBJ mesh, Everything)")

# project round trip keeps calibration, fractional offsets, rates and the 3D result
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (PROJ, "Kinetrace project (*.kinetrace)"))
win._save_project_as()
pump()
back = Project.load(PROJ)
assert back.calibration is not None and len(back.calibration) == 3
assert np.allclose(back.offsets, p.offsets) and back.rates == RATES
assert back.reconstruction is not None and np.allclose(back.reconstruction.xyz, r1.xyz, equal_nan=True)
print("project round trip with calibration + 3D OK")

# no cameras -> the 3D actions are off; teardown hides the view
win._teardown_video()
win.project = None
win._apply_state()
assert not win.act_calib.isEnabled() and not win.act_recon.isEnabled()
assert not win.view3d.isVisible() and not win.act_view3d.isChecked()
win._dev_probe.wait(15000)
win.close()
for p_ in paths:
    forget_recovery(p_)
print("VERIFY 3D GUI PASSED")
