"""Native wand calibration through the app (offscreen, no GPU).

A 4-camera synthetic rig films a 0.5 m wand waved through a 1 m volume, a
dropped ball and three floor marks; the tracks are written into the sessions
with 0.3 px noise (one camera offset by 5 frames). The project is opened in
the real MainWindow, the wand wizard is driven page by page (auto focal,
gravity from the ball), and the result must be GOOD: wand score < 0.5 %,
g ratio within 3 %, camera spacing within 1 % of truth. Then: it becomes the
project calibration, Reconstruct recovers the 0.5 m floor distances to 3 mm,
Export Calibration writes the .kcal.json / dltCoefs.csv / report and the
.kcal.json re-imports through the Import Calibration dialog with its own
convention locked. A second run with the three-reference-point frame puts
the floor origin at (0,0,0) and +X at (0.5,0,0).

Run: .venv\\Scripts\\python.exe tests\\verify_wand_gui.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWizard

from kinetrace.calib import CameraCalibration, NoUndistort, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.RandomState(5)


def ascii_(s):
    return str(s).encode("ascii", "replace").decode()

W, H = 640, 480
N_CAM = 4
FPS = 60.0
T = 260
OFFS = [0, 5, 0, 0]                   # camera 2 started 5 frames later
WAND_L = 0.5
G = 9.81


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


FOCAL = [600.0, 650.0, 600.0, 580.0]
cams = []
true_pos = []
for k in range(N_CAM):
    a = 0.4 + k * 2 * np.pi / N_CAM
    pos = np.array([2.7 * np.cos(a), 2.7 * np.sin(a), 1.3 + 0.3 * (k % 2)])
    true_pos.append(pos)
    R, t = look_at(pos, np.array([0.0, 0.0, 1.0]))
    K = np.array([[FOCAL[k], 0, W / 2], [0, FOCAL[k], H / 2], [0, 0, 1]])
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))
true_pos = np.array(true_pos)


def wand_at(t):
    """Both ends at reference frame t (metres): a Lissajous centre, tumbling."""
    s = t / 60.0
    c = np.array([0.85 * np.sin(1.3 * s), 0.85 * np.sin(0.9 * s + 1.0), 1.0 + 0.7 * np.sin(0.7 * s)])
    th, ph = 2.1 * s, 1.4 * s + 0.5
    d = np.array([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)])
    return c + 0.5 * WAND_L * d, c - 0.5 * WAND_L * d


DROP_T0, DROP_T1 = 150, 175      # 0.42 s of fall: 0.85 m, the ball stays in every picture


def ball_at(t):
    if t < DROP_T0 or t > DROP_T1:
        return None
    dt = (t - DROP_T0) / FPS
    return np.array([0.2, 0.1, 1.55 - 0.5 * G * dt * dt])


FLOOR = {"floor_origin": np.array([0.0, 0.0, 0.0]), "floor_x": np.array([0.5, 0.0, 0.0]),
         "floor_y": np.array([0.0, 0.5, 0.0])}

sessions, paths = [], []
for c, cal in enumerate(cams):
    p = os.path.join(SCRATCH, f"wandcam_{c}.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_fr = T + OFFS[c]
    s = TrackingSession(p, n_fr, FPS, W, H)
    for nm in ("wand A", "wand B", "ball", *FLOOR):
        s.add_landmark(nm)
    img = np.full((H, W, 3), 25, np.uint8)
    for f in range(n_fr):
        t = f - OFFS[c]
        if 0 <= t < T:
            A, B = wand_at(t)
            pts = {"wand A": A, "wand B": B, **FLOOR}
            b = ball_at(t)
            if b is not None:
                pts["ball"] = b
            for nm, X in pts.items():
                if c == 1 and nm == "wand A" and 100 <= t < 120:
                    continue                      # a gap in one camera: must not break anything
                uv = cal.project(X[None])[0]
                if 2 <= uv[0] < W - 2 and 2 <= uv[1] < H - 2:
                    s.set_position(f, s.pid_by_name(nm), float(uv[0] + rng.normal(0, 0.3)),
                                   float(uv[1] + rng.normal(0, 0.3)))
        vw.write(img)
    vw.release()
    s.manual[:] = False
    sessions.append(s)
    paths.append(p)
proj = Project(sessions, [f"cam{c + 1}" for c in range(N_CAM)], OFFS)
PROJ = os.path.join(SCRATCH, "wand_gui.cotrk")
proj.save_npz(PROJ)
for p in paths:
    if os.path.exists(p + ".cotracker.npz"):
        os.remove(p + ".cotracker.npz")

# ---- the app -----------------------------------------------------------------
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace import calibwizard as cw  # noqa: E402
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
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == N_CAM:
        break
assert win.project is not None and win.project.n_views == N_CAM and win.state == READY
print("4-camera wand project opened through the app OK")


def drive(self, frame_mode="drop"):
    """Stand-in for WandWizard.exec: walk the pages like a user would."""
    self.show()
    self.restart()
    pump(0.1)
    assert self.currentPage() is self.page_intro and self.page_intro.isComplete()
    # (I36) the manual says "Use this calibration": the button must say so while the wizard is open
    assert self.buttonText(QWizard.FinishButton) == "Use this calibration", self.buttonText(QWizard.FinishButton)
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_wand
    assert self.page_wand.end_a.currentText() == "wand A" and self.page_wand.end_b.currentText() == "wand B", \
        (self.page_wand.end_a.currentText(), self.page_wand.end_b.currentText())
    self.page_wand.length.setValue(WAND_L)
    self.page_wand.unit.setCurrentIndex(0)
    assert self.page_wand._n_frames >= 200, self.page_wand._n_frames
    assert self.page_wand.isComplete()
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_cams
    assert self.page_cams.r_auto.isChecked()
    names = self.page_cams.extra_names()
    assert "ball" in names and "floor_origin" in names, names
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_frame
    if frame_mode == "drop":
        # (I26) a landmark named 'ball' whose fall is found: gravity is the default
        assert self.page_frame.mode() == "drop", self.page_frame.mode()
        self.page_frame.r_drop.setChecked(True)
        assert self.page_frame.drop_name.currentText() == "ball"
        assert self.page_frame.drop_t0.value() == DROP_T0 and self.page_frame.drop_t1.value() == DROP_T1, \
            (self.page_frame.drop_t0.value(), self.page_frame.drop_t1.value())
    else:
        self.page_frame.r_axes.setChecked(True)
        self.page_frame.ax_o.setCurrentText("floor_origin")
        self.page_frame.ax_x.setCurrentText("floor_x")
        self.page_frame.ax_y.setCurrentText("floor_y")
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_run and not self.page_run.isComplete()
    self.page_run._run()
    t0 = time.time()
    while self.result_calibration is None and self.page_run.btn_run.isEnabled() is False:
        pump(0.1)
        if time.time() - t0 > 300:
            raise TimeoutError("calibration did not finish")
    assert self.result_calibration is not None, self.page_run.report.toPlainText()[:800]
    assert self.page_run.isComplete()
    self.accept()
    return QDialog.Accepted


cw.WandWizard.exec = drive
win._wand_wizard()
pump(0.2)
p = win.project
assert p.calibration is not None and len(p.calibration) == N_CAM
res, grav = win._wand_result
rep = res.report
print(f"verdict {rep['verdict']}: wand score {rep['wand_score_pct']:.3f} %, rmse {rep['reproj_rmse_all']:.3f} px, "
      f"focal {[round(f) for f in rep['focal_px']]}, g ratio {grav['g_ratio']:.4f}")
assert rep["verdict"] == "good", rep["verdict_reasons"]
assert rep["wand_score_pct"] < 0.5
assert abs(grav["g_ratio"] - 1.0) < 0.03, grav
assert grav["applied"] is True and grav["frames"] == [DROP_T0, DROP_T1], grav
assert rep["camera_names"] == list(p.names), rep["camera_names"]         # (I34) messages use cam1..cam4
assert res.unit == "m" and p.calibration.unit == "m"
for c in range(N_CAM):
    assert abs(rep["focal_px"][c] - FOCAL[c]) / FOCAL[c] < 0.03, (c, rep["focal_px"][c], FOCAL[c])
for i, j, d in rep["camera_distances"]:
    truth = float(np.linalg.norm(true_pos[int(i)] - true_pos[int(j)]))
    assert abs(d - truth) / truth < 0.01, (i, j, d, truth)
# gravity frame: Z up -> every camera centre sits above the drop point's origin by ~its true height difference
centres = np.array([c.center() for c in p.calibration.cameras])
b0 = ball_at(DROP_T0)
for k in range(N_CAM):
    assert abs((centres[k, 2]) - (true_pos[k, 2] - b0[2])) < 0.03, (k, centres[k], true_pos[k])
print("wizard result matches the synthetic truth OK")

# ---- reconstruct with it ---------------------------------------------------------
win._reconstruct_3d(quiet=True)
pump(0.3)
r = p.reconstruction
assert r is not None
idx = {nm: i for i, nm in enumerate(r.names)}
fo, fx, fy = (r.xyz[:, idx[n]] for n in ("floor_origin", "floor_x", "floor_y"))
dx = np.nanmedian(np.linalg.norm(fx - fo, axis=1))
dy = np.nanmedian(np.linalg.norm(fy - fo, axis=1))
assert abs(dx - 0.5) < 0.003 and abs(dy - 0.5) < 0.003, (dx, dy)
wa, wb = r.xyz[:, idx["wand A"]], r.xyz[:, idx["wand B"]]
wl = np.linalg.norm(wa - wb, axis=1)
print(f"reconstructed: floor distances {dx:.4f} / {dy:.4f} m, wand {np.nanmean(wl):.4f} +- {np.nanstd(wl):.4f} m")
assert abs(np.nanmean(wl) - WAND_L) < 0.003

# ---- export + re-import ----------------------------------------------------------
kcal = os.path.join(SCRATCH, "wand_gui.kcal.json")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (kcal, ""))
win._export_calibration()
pump(0.2)
for suf in ("wand_gui.kcal.json", "wand_gui_dltCoefs.csv", "wand_gui_report.txt"):
    assert os.path.exists(os.path.join(SCRATCH, suf)), suf
back = v3.load_calibration_file(kcal)
assert len(back) == N_CAM and back.cameras[0].pixel_origin == 0.0
assert np.allclose(back.cameras[2].coefs, p.calibration.cameras[2].coefs)
dlg = v3.CalibrationDialog(win, list(p.names), win._view_sizes(), SCRATCH)
assert dlg.load(kcal) and not dlg.conv.isEnabled() and dlg.conv.currentIndex() == 2
dlg._accept()
assert dlg.result_calibration is not None and dlg.result_calibration.cameras[1].pixel_origin == 0.0
# the MATLAB csv projects the same world point one pixel over
from kinetrace.calib import Calibration, dlt_project
mat = Calibration.load_dlt_csv(os.path.join(SCRATCH, "wand_gui_dltCoefs.csv"), pixel_origin=1.0)
X = np.array([[0.1, 0.2, 1.0]])
u0 = dlt_project(p.calibration.cameras[0].coefs, X)
u1 = dlt_project(mat.cameras[0].coefs, X)
assert np.allclose(u1 - u0, 1.0, atol=1e-6), (u0, u1)
assert not os.path.exists(os.path.join(SCRATCH, "wand_gui_dltCoefs_README.txt")), \
    "no lens correction: the csv stands on its own, no caveat file"
print("export / re-import OK")

# ---- (I33) exporting a DIFFERENT calibration must not carry this wand run's report ----
import json as _json  # noqa: E402
own_cal = p.calibration
p.calibration = mat                                   # e.g. an imported easyWand / DLTdv csv
kcal2 = os.path.join(SCRATCH, "wand_gui_imported.kcal.json")
rep2 = os.path.join(SCRATCH, "wand_gui_imported_report.txt")
for f_ in (kcal2, rep2):
    if os.path.exists(f_):
        os.remove(f_)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (kcal2, ""))
win._export_calibration()                             # win._wand_result still holds the wand run
pump(0.2)
d2 = _json.loads(open(kcal2, encoding="utf-8").read())
assert d2["report"] == {} and d2["gravity"] == {}, (list(d2["report"])[:5], d2["gravity"])
assert not os.path.exists(rep2), "a wand report was written for a calibration it does not describe"
assert np.allclose(d2["cameras"][0]["coefs"], mat.cameras[0].coefs)
p.calibration = own_cal
print("export of another calibration carries no stale wand report OK")

# ---- (I32) a lens-corrected calibration's dltCoefs.csv says what it needs ----------------
import copy as _copy  # noqa: E402
from kinetrace.calib import OpenCVUndistort  # noqa: E402
res_l = _copy.deepcopy(res)
K0 = np.array([[FOCAL[0], 0, W / 2], [0, FOCAL[0], H / 2], [0, 0, 1.0]])
res_l.cameras[0].undistort = OpenCVUndistort(K0, np.array([-0.25, 0.08, 0, 0, 0]))
lpath = os.path.join(SCRATCH, "wand_gui_lens.kcal.json")
written = cw.save_calibration_files(res_l, grav, lpath)
names_w = [os.path.basename(w_) for w_ in written]
assert "wand_gui_lens_dltCoefs_README.txt" in names_w, names_w
readme = open(os.path.join(SCRATCH, "wand_gui_lens_dltCoefs_README.txt"), encoding="utf-8").read()
assert "UNDISTORTED" in readme and "cam1 (column 1)" in readme and "cam2" not in readme, readme[:400]
rtxt = open(os.path.join(SCRATCH, "wand_gui_lens_report.txt"), encoding="utf-8").read()
assert "NOTE:" in rtxt and "lens-corrected pixels in cam1" in rtxt and "distortion_estimated" in rtxt, rtxt[:600]
assert cw.dlt_csv_caveat(res_l.to_calibration(), list(p.names)) and cw.dlt_csv_caveat(res.to_calibration()) is None
print("lens-corrected dltCoefs.csv comes with its README + report note OK")

# ---- three-reference-point frame ------------------------------------------------
cw.WandWizard.exec = lambda self: drive(self, "axes")
win._wand_wizard()
pump(0.2)
win._reconstruct_3d(quiet=True)
pump(0.3)
r = p.reconstruction
idx = {nm: i for i, nm in enumerate(r.names)}
fo = np.nanmedian(r.xyz[:, idx["floor_origin"]], axis=0)
fx = np.nanmedian(r.xyz[:, idx["floor_x"]], axis=0)
fy = np.nanmedian(r.xyz[:, idx["floor_y"]], axis=0)
assert np.linalg.norm(fo) < 0.005, fo
assert np.allclose(fx, [0.5, 0, 0], atol=0.005), fx
assert np.allclose(fy, [0, 0.5, 0], atol=0.005), fy
b = np.nanmedian(r.xyz[:, idx["ball"]], axis=0)
assert b[2] > 1.0, b                    # up is up
print("three-reference-point frame OK")

# project round trip keeps the calibration
p.save_npz(PROJ)
back = Project.load_npz(PROJ)
assert back.calibration is not None and np.allclose(back.calibration.cameras[3].coefs, p.calibration.cameras[3].coefs)

# =============================================================================
# Release-sweep regressions on in-memory projects (same rig, no MainWindow)
# =============================================================================
from kinetrace import wanddata as wd  # noqa: E402
from kinetrace import wand as wand_mod  # noqa: E402
from kinetrace import lens as lens_mod  # noqa: E402


def mem_project(points, offs=None, tag="mem"):
    """The 4-camera rig filming `points` {name: f(t) -> 3D point or None} at
    reference frames t; camera c's frame f shows t = f - offs[c]."""
    offs = offs or [0] * N_CAM
    ss = []
    for c, cal in enumerate(cams):
        n_fr = T + offs[c]
        s = TrackingSession(os.path.join(SCRATCH, f"{tag}_{c}.mp4"), n_fr, FPS, W, H)
        for nm in points:
            s.add_landmark(nm)
        for f in range(n_fr):
            t = f - offs[c]
            for nm, fn in points.items():
                X = fn(t)
                if X is None:
                    continue
                uv = cal.project(np.asarray(X, float)[None])[0]
                if 2 <= uv[0] < W - 2 and 2 <= uv[1] < H - 2:
                    s.set_position(f, s.pid_by_name(nm), float(uv[0] + rng.normal(0, 0.3)),
                                   float(uv[1] + rng.normal(0, 0.3)))
        s.manual[:] = False
        ss.append(s)
    return Project(ss, [f"cam{c + 1}" for c in range(N_CAM)], offs)


def wand_end(k, t_lo=0, t_hi=T - 1):
    return lambda t: wand_at(t)[k] if t_lo <= t <= t_hi and 0 <= t < T else None


def ball_bounce(t):
    """Held still 100-149, falls 150-175 (lands exactly at 175), bounces, rests."""
    if t < 100 or t >= T:
        return None
    z_top, dt_fall = 1.55, (DROP_T1 - DROP_T0) / FPS
    floor = z_top - 0.5 * G * dt_fall ** 2
    if t <= DROP_T0:
        return np.array([0.2, 0.1, z_top])
    if t <= DROP_T1:
        s_ = (t - DROP_T0) / FPS
        return np.array([0.2, 0.1, z_top - 0.5 * G * s_ * s_])
    z, v, tt_ = floor, 0.6 * G * dt_fall, 0.0             # bounce up with 60 % of the impact speed
    target = (t - DROP_T1) / FPS
    while tt_ < target - 1e-9:
        h = min(1e-3, target - tt_)
        v -= G * h
        z += v * h
        if z < floor:
            z, v = floor, -0.6 * v if abs(v) > 0.3 else 0.0
        tt_ += h
    return np.array([0.2, 0.1, z])


def new_wizard(proj):
    wz = cw.WandWizard(None, proj, SCRATCH)
    wz.show()
    wz.restart()
    pump(0.05)
    return wz


def goto(wz, page):
    while wz.currentPage() is not page:
        wz.next()
        pump(0.02)


def known_focal(wz):
    """Start from the true focal lengths: a fast solve (no grid search)."""
    wz.page_cams.r_known.setChecked(True)
    for sp, f in zip(wz.page_cams._spins, FOCAL):
        sp.setValue(int(f))


def run_and_wait(wz, timeout=120):
    wz.page_run._run()
    t0_ = time.time()
    while not wz.page_run.btn_run.isEnabled():
        pump(0.05)
        if time.time() - t0_ > timeout:
            raise TimeoutError("calibration did not finish")
    pump(0.1)


def close_wizard(wz):
    wz.reject()
    pump(0.05)
    wz.deleteLater()
    pump(0.05)


floor_at = {nm: (lambda X: (lambda t: X if 0 <= t < T else None))(X) for nm, X in FLOOR.items()}

# ---- (I30) a camera that started late does not cut the other cameras' wand frames --------
P_late = mem_project({"wand A": wand_end(0, 0, 119), "wand B": wand_end(1, 0, 119)}, offs=[0, 0, 0, -100],
                     tag="late")
uv_l, fr_l = wd.collect_wand(P_late, "wand A", "wand B", max_frames=100000)
print(f"late camera: reference_window {wd.reference_window(P_late)}, sampling_window {wd.sampling_window(P_late)}, "
      f"wand frames {len(fr_l)} (reference {int(fr_l.min())}..{int(fr_l.max())})")
assert wd.sampling_window(P_late) == (0, T - 1) and wd.reference_window(P_late) == (100, T - 1)
assert len(fr_l) >= 110 and fr_l.min() < 100, (len(fr_l), fr_l.min())
assert np.isnan(uv_l[fr_l < 100][:, 3]).all(), "the late camera has no data before it started"
wz = new_wizard(P_late)
goto(wz, wz.page_wand)
assert wz.page_wand._n_frames >= 110 and wz.page_wand.isComplete(), wz.page_wand._n_frames
close_wizard(wz)
print("wand frames recorded before the last camera started are used OK")

# ---- (I26) the frame page's default follows the data ------------------------------------
P_mark = mem_project({"wand A": wand_end(0), "wand B": wand_end(1),
                      "corner mark": lambda t: np.array([0.3, -0.2, 0.0]) if 0 <= t < T else None}, tag="mark")
wz = new_wizard(P_mark)
goto(wz, wz.page_frame)
assert wz.page_frame.mode() == "none", "a static mark is not a dropped object: default is Neither"
assert wd.find_fall(P_mark, "corner mark") is None
P_bounce = mem_project({"wand A": wand_end(0), "wand B": wand_end(1), "ball": ball_bounce}, offs=[0, 30, 0, 0],
                       tag="bounce")
ff = wd.find_fall(P_bounce, "ball")
print(f"ball held 100-149, falls {DROP_T0}-{DROP_T1}, bounces: find_fall -> {ff[:2]}{ascii_(ff[2]) if ff else ''}")
assert ff is not None and DROP_T0 - 2 <= ff[0] <= DROP_T0 + 2 and DROP_T1 - 2 <= ff[1] <= DROP_T1, ff
wz2 = new_wizard(P_bounce)
goto(wz2, wz2.page_frame)
f2 = wz2.page_frame
assert f2.mode() == "drop" and f2.drop_name.currentText() == "ball"
assert (f2.drop_t0.value(), f2.drop_t1.value()) == ff[:2], (f2.drop_t0.value(), f2.drop_t1.value())
assert f2.drop_t0.minimum() == -30, "a camera that started before the reference: negative instants selectable"
close_wizard(wz2)
print("frame page defaults: static mark -> Neither, bouncing ball -> its fall only OK")

# ---- (I25 through the wizard) forcing the static mark as the drop keeps the calibration --
known_focal(wz)
wz.page_frame.r_drop.setChecked(True)
assert wz.page_frame.drop_name.currentText() == "corner mark" and wz.page_frame.isComplete()
assert "No free fall" in wz.page_frame.drop_note.text()
goto(wz, wz.page_run)
run_and_wait(wz)
assert wz.result_calibration is not None, wz.page_run.report.toPlainText()[:500]
g_m = wz.gravity
assert g_m["applied"] is False, g_m
assert wz.result.report["verdict"] != "good" and "NOT applied" in wz.result.report["frame"]
assert "not used" in wz.page_run.report.toPlainText(), wz.page_run.report.toPlainText()[:400]
close_wizard(wz)
print(f"static mark forced as the drop: calibration kept, vertical NOT set ({ascii_(g_m['why'])[:60]}) OK")

# ---- (I29) reference points clicked on ONE frame; a failed alignment keeps the solve ------
P_axes = mem_project({"wand A": wand_end(0), "wand B": wand_end(1), "ball": ball_at,
                      **{nm: (lambda X: (lambda t: X if t == 40 else None))(X) for nm, X in FLOOR.items()}},
                     tag="axes")
wz = new_wizard(P_axes)
goto(wz, wz.page_cams)
known_focal(wz)
goto(wz, wz.page_frame)
fp = wz.page_frame
fp.r_axes.setChecked(True)
fp.ax_o.setCurrentText("floor_origin")
fp.ax_x.setCurrentText("floor_x")
fp.ax_y.setCurrentText("floor_y")
pump(0.05)
assert fp.ax_t.value() == 40, ("the reference frame must follow the chosen points", fp.ax_t.value())
assert fp.isComplete() and "frame 40" in fp.ax_note.text(), fp.ax_note.text()
fp.ax_t.setValue(41)
assert not fp.isComplete() and not wz.button(QWizard.NextButton).isEnabled(), "Next blocked on a frame without them"
assert "frame 40 has all three" in fp.ax_note.text(), fp.ax_note.text()
fp.ax_t.setValue(40)
goto(wz, wz.page_run)
fp.ax_t.setValue(41)                                  # behind the page's back: the run itself must cope
run_and_wait(wz)
assert wz.result_calibration is not None, "the wand calibration must survive a failed axes step"
rr = wz.result.report
assert "NOT applied" in rr["frame"] and rr["verdict"] != "good", rr["frame"]
assert any("could NOT be set" in x and "reference frame 41" in x for x in rr["verdict_reasons"]), rr["verdict_reasons"][-1]
print("axes: frame follows the picked points, Next blocked on a bad frame, a failed alignment keeps the solve OK")

# ---- (I28) Back / Next: an unchanged setup keeps the result, a changed one drops it --------
fp.ax_t.setValue(40)
fp.r_none.setChecked(True)
goto(wz, wz.page_run)
run_and_wait(wz)
assert wz.result_calibration is not None and wz.page_run.isComplete()
for _ in range(3):
    wz.back()
    pump(0.02)
assert wz.currentPage() is wz.page_wand
goto(wz, wz.page_run)
assert wz.result_calibration is not None and wz.button(QWizard.FinishButton).isEnabled(), "same settings: kept"
for _ in range(3):
    wz.back()
    pump(0.02)
wz.page_wand.length.setValue(25.0)
wz.page_wand.unit.setCurrentIndex(1)                  # cm
goto(wz, wz.page_run)
assert wz.result_calibration is None and wz.result is None and not wz.page_run.isComplete()
assert not wz.button(QWizard.FinishButton).isEnabled(), "Finish must not hand over the 0.5 m result"
assert "Press" in wz.page_run.report.toPlainText()
# a run still going when the settings change is dropped too
wz.page_run._run()
for _ in range(3):
    wz.back()
    pump(0.02)
wz.page_wand.length.setValue(30.0)
goto(wz, wz.page_run)
t0_ = time.time()
while wz.page_run._thread.isRunning() or not wz.page_run.btn_run.isEnabled():
    pump(0.05)
    assert time.time() - t0_ < 60
pump(0.2)
assert wz.result_calibration is None, "a run made with 25 cm must not become the 30 cm result"
# the unit the run was made with, whatever the combo says when it ends
wz.page_run._run()
wz.page_wand.unit.setCurrentIndex(2)                  # mm, changed during the run
t0_ = time.time()
while not wz.page_run.btn_run.isEnabled():
    pump(0.05)
    assert time.time() - t0_ < 60
pump(0.1)
assert wz.result is not None and wz.result.unit == "cm" and wz.result_calibration.unit == "cm", \
    (wz.result.unit, wz.result_calibration.unit)
assert abs(wz.result.report["wand_mean"] - 30.0) < 1e-6
close_wizard(wz)
print("stale results: kept when nothing changed, dropped after a change or mid-run, unit from the run OK")

# ---- (I31) a lens profile made at another picture size is refused -------------------------
Kc = [np.array([[FOCAL[c], 0, W / 2], [0, FOCAL[c], H / 2], [0, 0, 1.0]]) for c in range(N_CAM)]
big = lens_mod.LensProfile(2 * W, 2 * H, Kc[0] * np.array([[2], [2], [1]]), np.zeros(5), False, 0.2, 30, "test")
big_path = big.save(os.path.join(SCRATCH, "big_mode.klens.json"))
ok_prof = lens_mod.LensProfile(W, H, Kc[0], np.zeros(5), False, 0.2, 30, "test")
ok_path = ok_prof.save(os.path.join(SCRATCH, "cam1_ok.klens.json"))
argus = os.path.join(SCRATCH, "two_cams_argus.txt")
with open(argus, "w", encoding="ascii") as fh:
    for c in range(2):
        fh.write(f"{c + 1} {FOCAL[c]} {W} {H} {W / 2 + 1} {H / 2 + 1} 1 0 0 0 0 0\n")
_orig_open = QFileDialog.getOpenFileName
wz = new_wizard(P_axes)
goto(wz, wz.page_cams)
pc = wz.page_cams
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (big_path, ""))
pc._load_lens(0)
assert P_axes.lenses[0] is None, "a 1280x960 profile must not be attached to a 640x480 camera"
assert "1280" in pc.lens_labels[0].text() and "640" in pc.lens_labels[0].text(), pc.lens_labels[0].text()
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (argus, ""))
pc._load_lens(2)
# the SAME rule as the lens wizard (lens.argus_profile_for): the file's camera numbers decide
assert P_axes.lenses[2] is None and "no line for camera 3" in pc.lens_labels[2].text(), pc.lens_labels[2].text()
pc._load_lens(1)
assert P_axes.lenses[1] is not None and "camera 2" in P_axes.lenses[1].source, P_axes.lenses[1].source
assert abs(P_axes.lenses[1].f_square - FOCAL[1]) < 1e-9
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ok_path, ""))
pc._load_lens(0)
assert P_axes.lenses[0] is not None and "640" in pc.lens_labels[0].text()
# a wrong-size profile attached elsewhere is shown and NOT used
P_axes.lenses[3] = big
pc._refresh_lens_rows()
assert "NOT used" in pc.lens_labels[3].text() and pc.usable_lens(3) is None and pc.lens_models()[3] is None
QFileDialog.getOpenFileName = _orig_open
print("lens profiles: wrong size refused / flagged, Argus lines matched to cameras OK")

# ---- (I35) the distortion tick is honoured for the cameras without a profile ------------
P_axes.lenses[3] = None
P_axes.lenses[1] = None                               # profile on cam1 only
pc._refresh_lens_rows()
pc.distort.setChecked(True)
assert pc.distortion_per_camera() == [False, True, True, True]
seen_kw = {}
_orig_cal = wand_mod.calibrate_wand


def _capture(**kw):
    seen_kw.update(kw)
    raise wand_mod.WandError("stub: kwargs captured")


wand_mod.calibrate_wand = _capture
goto(wz, wz.page_run)
assert "fitted for the 3 camera(s) without a profile" in wz.page_run.summary.text(), wz.page_run.summary.text()
run_and_wait(wz)
wand_mod.calibrate_wand = _orig_cal
assert seen_kw["estimate_distortion"] == [False, True, True, True], seen_kw["estimate_distortion"]
assert seen_kw["names"] == ["cam1", "cam2", "cam3", "cam4"] and seen_kw["estimate_focal"] is True
assert "stub" in wz.page_run.report.toPlainText()
for _ in range(2):
    wz.back()
    pump(0.02)
for c in range(N_CAM):
    P_axes.lenses[c] = lens_mod.LensProfile(W, H, Kc[c], np.zeros(5), False, 0.2, 30, "test")
pc._refresh_lens_rows()
assert not pc.distort.isEnabled() and pc.distortion_per_camera() == [False] * N_CAM
close_wizard(wz)
P_axes.lenses = [None] * N_CAM
print("distortion tick: fitted for un-profiled cameras only, disabled when all have a profile OK")

# ---- (I36) Cancel during a long solve returns at once and leaves no thread running -------
SLOW = {"steps": 0}


def _slow(**kw):
    prog = kw["progress"]
    for i in range(400):                               # ~20 s if nobody cancels
        prog(i / 400.0, "slow stub")
        SLOW["steps"] += 1
        time.sleep(0.05)
    raise wand_mod.WandError("slow stub finished")


wand_mod.calibrate_wand = _slow
wz = new_wizard(P_axes)
goto(wz, wz.page_run)
wz.page_run._run()
pump(0.3)
th_ = wz.page_run._thread
assert th_.isRunning()
t0_ = time.time()
wz.reject()
dt_cancel = time.time() - t0_
wand_mod.calibrate_wand = _orig_cal
assert dt_cancel < 3.0 and not th_.isRunning(), (dt_cancel, th_.isRunning())
assert 0 < SLOW["steps"] < 100, SLOW                   # it ran, and it was stopped - not finished
pump(0.1)
wz.deleteLater()
pump(0.05)
print(f"Cancel during a solve: closed in {dt_cancel:.2f} s, no thread left running OK")

# ---- the wizard opens with ONE camera too: review, not refuse -------------------------
win._open_video(paths[0])
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == 1:
        break
assert win.project.n_views == 1
s1 = win.session
pa = s1.add_point(10, 100.0, 100.0, name="wand A")
pb = s1.add_point(10, 160.0, 120.0, name="wand B")
for f in range(10, 60):
    s1.set_position(f, pa, 100.0 + f, 100.0)
    s1.set_position(f, pb, 160.0 + f, 120.0)
SEEN = {}


def drive_single(self):
    self.show()
    self.restart()
    pump(0.1)
    SEEN["intro"] = (self.page_intro.isComplete(), self.page_intro.check.text())
    self.next()
    pump(0.1)
    SEEN["on_wand"] = self.currentPage() is self.page_wand
    SEEN["names"] = [self.page_wand.end_a.itemText(i) for i in range(self.page_wand.end_a.count())]
    SEEN["complete"] = self.page_wand.isComplete()
    SEEN["warn"] = self.page_wand.warn.text()
    SEEN["table"] = self.page_wand.table.text()
    self.reject()
    return QDialog.Rejected


cw.WandWizard.exec = drive_single
win._wand_wizard()
pump(0.2)
assert SEEN, "the wizard must OPEN with one camera"
assert SEEN["intro"][0] and "ONE camera" in SEEN["intro"][1], SEEN["intro"]
assert SEEN["on_wand"] and "wand A" in SEEN["names"] and "wand B" in SEEN["names"], SEEN["names"]
assert not SEEN["complete"], "one camera cannot go on to a calibration"
assert "second camera" in SEEN["warn"] and "50" in SEEN["table"], (SEEN["warn"][:80], SEEN["table"])
print("wand wizard with one camera: opens, reports coverage, explains what is missing OK")

win._dev_probe.wait(60000)
win.close()
pump(0.2)
for pth in paths:
    if os.path.exists(pth + ".cotracker.npz"):
        os.remove(pth + ".cotracker.npz")
print("WAND GUI PASSED")
