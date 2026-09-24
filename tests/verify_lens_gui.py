"""Lens calibration through the app, then a wand calibration that uses it
(offscreen, no GPU).

Three synthetic cameras with STRONG radial distortion film a checkerboard
video each and a 0.5 m wand + dropped ball + floor marks. The lens wizard is
driven for every camera (3D → Calibrate a Lens): verdict GOOD, focal within
1 %, profile attached to the right camera. The wand wizard then picks the
profiles up (straightened points, focal from the checkerboard): verdict
GOOD, camera spacing within 1 %, floor distances to 3 mm, and the resulting
calibration undistorts. The same wand run WITHOUT the lens profiles must be
measurably worse. Lens file save / load and the project round trip of the
profiles are covered too.

Run: .venv\\Scripts\\python.exe tests\\verify_lens_gui.py
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

from cotracker_app.calib import CameraCalibration, OpenCVUndistort, dlt_from_camera
from cotracker_app.project import Project
from cotracker_app.session import TrackingSession

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.RandomState(9)

W, H = 640, 480
N_CAM = 3
FPS = 60.0
T = 240
WAND_L = 0.5
G = 9.81
PATTERN = (9, 6)
SQUARE = 0.024
FOCAL = [600.0, 620.0, 590.0]
DIST = np.array([-0.25, 0.07, 0.0, 0.0, 0.0])


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


Ks, cams, true_pos = [], [], []
for k in range(N_CAM):
    a = 0.5 + k * 2 * np.pi / N_CAM
    pos = np.array([2.6 * np.cos(a), 2.6 * np.sin(a), 1.2 + 0.3 * (k % 2)])
    true_pos.append(pos)
    R, t = look_at(pos, np.array([0.0, 0.0, 1.0]))
    K = np.array([[FOCAL[k], 0, W / 2 + 3 * k], [0, FOCAL[k] * 1.003, H / 2 - 2 * k], [0, 0, 1]])
    Ks.append(K)
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, OpenCVUndistort(K, DIST), pixel_origin=0.0))
true_pos = np.array(true_pos)


def wand_at(t):
    s = t / 60.0
    c = np.array([0.8 * np.sin(1.3 * s), 0.8 * np.sin(0.9 * s + 1.0), 1.0 + 0.6 * np.sin(0.7 * s)])
    th, ph = 2.1 * s, 1.4 * s + 0.5
    d = np.array([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)])
    return c + 0.5 * WAND_L * d, c - 0.5 * WAND_L * d


DROP_T0, DROP_T1 = 150, 175


def ball_at(t):
    if t < DROP_T0 or t > DROP_T1:
        return None
    dt = (t - DROP_T0) / FPS
    return np.array([0.2, 0.1, 1.55 - 0.5 * G * dt * dt])


FLOOR = {"floor_origin": np.array([0.0, 0.0, 0.0]), "floor_x": np.array([0.5, 0.0, 0.0]),
         "floor_y": np.array([0.0, 0.5, 0.0])}


def render_board_video(path, K, dist, n_frames, seed):
    """Checkerboard at many poses through the distorted camera (curved edges
    rendered faithfully by subdividing each square)."""
    r = np.random.RandomState(seed)
    cols, rows = PATTERN
    nx, ny = cols + 1, rows + 1
    SUB = 3
    gx, gy = np.meshgrid(np.arange(-1, nx * SUB + 1) / SUB, np.arange(-1, ny * SUB + 1) / SUB)
    board = np.stack([gx.ravel() * SQUARE, gy.ravel() * SQUARE, np.zeros(gx.size)], axis=1)
    board -= np.array([(nx - 2) / 2.0 * SQUARE, (ny - 2) / 2.0 * SQUARE, 0.0])
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for _ in range(n_frames):
        z = r.uniform(0.4, 0.8)
        u = r.uniform(0.03, 0.97) * W
        v = r.uniform(0.03, 0.97) * H
        xc = (u - K[0, 2]) / K[0, 0] * z
        yc = (v - K[1, 2]) / K[1, 1] * z
        rvec = np.deg2rad(35.0) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        pts, _ = cv2.projectPoints(board.reshape(-1, 1, 3), rvec, np.array([xc, yc, z]), K, dist)
        pts = pts.reshape(ny * SUB + 2, nx * SUB + 2, 2)
        img = np.full((H, W, 3), 140, np.uint8)
        for yy in range(ny * SUB):
            for xx in range(nx * SUB):
                quad = np.array([pts[yy + 1, xx + 1], pts[yy + 1, xx + 2], pts[yy + 2, xx + 2], pts[yy + 2, xx + 1]],
                                np.float32)
                if not np.isfinite(quad).all() or np.abs(quad).max() > 4 * max(W, H):
                    continue
                col = 15 if ((xx // SUB) + (yy // SUB)) % 2 == 0 else 245
                cv2.fillConvexPoly(img, np.round(quad * 16).astype(np.int32), (col, col, col), cv2.LINE_AA, 4)
        img = np.clip(img.astype(np.int16) + r.normal(0, 3, (H, W, 1)).astype(np.int16), 0, 255).astype(np.uint8)
        vw.write(cv2.GaussianBlur(img, (0, 0), 0.6))
    vw.release()


sessions, paths, board_videos = [], [], []
for c, cal in enumerate(cams):
    p = os.path.join(SCRATCH, f"lenscam_{c}.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    s = TrackingSession(p, T, FPS, W, H)
    for nm in ("wand A", "wand B", "ball", *FLOOR):
        s.add_landmark(nm)
    img = np.full((H, W, 3), 25, np.uint8)
    for f in range(T):
        A, B = wand_at(f)
        pts = {"wand A": A, "wand B": B, **FLOOR}
        b = ball_at(f)
        if b is not None:
            pts["ball"] = b
        for nm, X in pts.items():
            uv = cal.project(X[None])[0]                 # DISTORTED raw pixels
            if 2 <= uv[0] < W - 2 and 2 <= uv[1] < H - 2:
                s.set_position(f, s.pid_by_name(nm), float(uv[0] + rng.normal(0, 0.3)),
                               float(uv[1] + rng.normal(0, 0.3)))
        vw.write(img)
    vw.release()
    s.manual[:] = False
    sessions.append(s)
    paths.append(p)
    bv = os.path.join(SCRATCH, f"lensboard_{c}.mp4")
    render_board_video(bv, Ks[c], DIST, 80, seed=20 + c)
    board_videos.append(bv)
proj = Project(sessions, [f"cam{c + 1}" for c in range(N_CAM)], [0, 0, 0])
PROJ = os.path.join(SCRATCH, "lens_gui.cotrk")
proj.save_npz(PROJ)
for p in paths:
    if os.path.exists(p + ".cotracker.npz"):
        os.remove(p + ".cotracker.npz")

# ---- the app -----------------------------------------------------------------
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from cotracker_app.app import READY, MainWindow  # noqa: E402
from cotracker_app import calibwizard as cw  # noqa: E402
from cotracker_app import lenswizard as lw  # noqa: E402
from cotracker_app import lens  # noqa: E402
from cotracker_app import lens  # noqa: E402

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
p = win.project
assert len(p.lenses) == N_CAM and all(l is None for l in p.lenses)
print("3-camera distorted project opened OK")

# ---- lens wizard for every camera ----------------------------------------------
current_cam = {"c": 0}
scans = {}


def drive_lens(self):
    self.show()
    self.restart()
    pump(0.1)
    assert self.currentPage() is self.page_intro
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_video
    c = current_cam["c"]
    self.page_video.path.setText(board_videos[c])
    # (I79) the combo shows ANOTHER camera while the scan runs and is corrected
    # afterwards: the lens must still go to the camera shown when Next is pressed
    self.page_video.cam.setCurrentIndex((c + 1) % N_CAM)
    self.page_video.r_auto.setChecked(True)
    self.page_video.square.setValue(24.0)
    self.page_video.unit.setCurrentIndex(0)
    assert abs(self.page_video.square_m() - 0.024) < 1e-9
    self.page_video._run()
    t0 = time.time()
    while not self.page_video.btn_run.isEnabled():
        pump(0.1)
        if time.time() - t0 > 300:
            raise TimeoutError("the board scan did not finish")
    assert self.scan is not None and len(self.scan.corners) >= 10, self.page_video.status.text()
    assert self.page_video.isComplete()
    scans[c] = self.scan
    self.page_video.cam.setCurrentIndex(c)

    # ---- the review board ------------------------------------------------
    self.next()
    pump(0.2)
    assert self.currentPage() is self.page_review, "the scan must lead to the review board"
    assert self.result_view == c, f"(I79) the camera shown at Next is the target: {self.result_view} vs {c}"
    rv = self.page_review.review
    n = len(self.scan.corners)
    assert len(rv._tiles) == n, f"a tile per board found: {len(rv._tiles)} vs {n}"
    assert all(len(th.shape) == 3 for th in self.scan.thumbs), "every tile has a picture"
    assert 3 <= len(rv.chosen()) <= n, "an automatic choice is made for you"
    assert self.page_review.isComplete(), "and it is fitted, so Next is available"
    auto_n = len(rv.chosen())
    first = self.result_profile
    assert first is not None

    # the tick on each image, and the bulk buttons
    rv._set_all(True)
    assert len(rv.chosen()) == n, "All ticks every board"
    rv._set_all(False)
    assert len(rv.chosen()) == 0, "None clears them"
    # (I74) the profile belongs to the boards it was fitted from: a changed
    # choice holds Next until a refit (or until the choice is put back)
    assert not self.page_review.isComplete(), "a changed choice must not leave the old fit attachable"
    rv._auto()
    assert len(rv.chosen()) == auto_n, "and the automatic choice is reproducible"
    assert self.page_review.isComplete(), "the fitted choice put back makes Next available again"
    drop = rv.chosen()[0]
    rv._on_toggle(drop, False)
    assert drop not in rv.chosen() and len(rv.chosen()) == auto_n - 1, "one image can be dropped"
    assert not self.page_review.isComplete(), "(I74) one unticked board after the fit holds Next"
    assert "Fit" in self.page_review.status.text(), self.page_review.status.text()
    rv._on_toggle(drop, True)
    assert len(rv.chosen()) == auto_n, "and put back"
    assert self.page_review.isComplete()

    # a corner dragged by hand must move that view's corners and its error
    i_edit = rv.chosen()[0]
    before = self.scan.corners[i_edit].copy()
    err_before = rv.errors[i_edit]
    self.scan.corners[i_edit][4] += [9.0, -7.0]      # what the editor writes back
    rv.edited.add(i_edit)
    rv.errors[i_edit] = lens.per_view_errors([self.scan.corners[i_edit]], self.pattern,
                                             self.square, rv.prof)[0]
    assert rv.errors[i_edit] > err_before, "a nudged corner reprojects worse"
    rv.corner_rev += 1                                # ... and the editor's accept does this
    rv.changed.emit()
    assert not self.page_review.isComplete(), "(I74) an edited corner holds Next until a refit"
    self.scan.corners[i_edit] = before                # put it back for the real fit
    rv.errors[i_edit] = err_before

    if c == 0:
        # (I81) the corner editor reads its frame on a worker thread and opens
        # when it arrives; a frame that cannot be read is SAID, not ignored
        from cotracker_app import boardreview as brv
        import cotracker_app.video_source as vsrc
        import threading
        threads, opened, warned = [], [], []
        real_oc, real_exec, real_warn = vsrc.open_capture, brv.CornerEditor.exec, QMessageBox.warning
        vsrc.open_capture = lambda pth: (threads.append(threading.current_thread() is threading.main_thread()),
                                         real_oc(pth))[1]
        brv.CornerEditor.exec = lambda dlg: (opened.append(dlg.bgr.shape), QDialog.Rejected)[1]
        QMessageBox.warning = staticmethod(lambda *a, **k: (warned.append(a[2] if len(a) > 2 else ""),
                                                            QMessageBox.Ok)[1])
        try:
            rv._edit(i_edit)
            t0 = time.time()
            while not opened and time.time() - t0 < 30:
                pump(0.05)
            assert opened and opened[0][:2] == (H, W), f"the editor opened on the full frame: {opened}"
            assert threads and not any(threads), "the frame was read off the GUI thread"
            real_video = self.scan.video
            gone = os.path.join(SCRATCH, "moved_away_board.mp4")
            self.scan.video = gone
            rv._edit(i_edit)
            t0 = time.time()
            while not warned and time.time() - t0 < 30:
                pump(0.05)
            self.scan.video = real_video
            assert warned and "moved_away_board.mp4" in warned[0], f"the missing file is named: {warned}"
            assert "moved_away_board.mp4" in rv.summary.text(), rv.summary.text()
            assert len(opened) == 1, "no editor for a frame that could not be read"
            assert QApplication.activeModalWidget() is None, "no dialog left open"
        finally:
            vsrc.open_capture, brv.CornerEditor.exec = real_oc, real_exec
            QMessageBox.warning = real_warn
        rv.stop_reader()
        print("  corner editor: frame read off the GUI thread; a moved video is reported OK")

    # refit from a chosen subset
    keep = rv.chosen()[:max(8, auto_n // 2)]
    rv._set_all(False)
    for k in keep:
        rv._on_toggle(k, True)
    assert len(rv.chosen()) == len(keep)
    self.page_review._fit()
    pump(0.2)
    assert self.result_profile is not None, self.page_review.status.text()
    assert self.result_profile is not first, "refitting produced a new profile"
    assert self.result_profile.n_views <= len(keep), \
        f"the fit used only the ticked boards: {self.result_profile.n_views} of {len(keep)}"
    assert np.isfinite(rv.errors).any(), "every view is re-scored after a refit"
    assert self.page_review.isComplete(), "a refit makes Next available"
    # put the good spread back so the rest of the suite checks a sound lens
    rv._auto()
    self.page_review._fit()
    pump(0.2)
    assert self.result_profile is not None

    # (I78) the "where the board went" picture lights up the boards the fit used
    cov = {}
    real_cov = lw.lens.coverage_image
    lw.lens.coverage_image = lambda scan, chosen=None, max_w=640: (cov.setdefault("used", chosen),
                                                                    real_cov(scan, chosen, max_w))[1]
    try:
        self.next()
        pump(0.1)
    finally:
        lw.lens.coverage_image = real_cov
    assert self.currentPage() is self.page_result
    used = cov.get("used")
    assert used is not None and set(used) <= set(rv.chosen()), "only ticked boards are shown as used"
    assert len(set(used)) == self.result_profile.n_views, (len(set(used)), self.result_profile.n_views)
    assert self.page_result.report.toPlainText()
    assert not self.page_result.pic_before.pixmap().isNull()
    self.accept()
    return QDialog.Accepted


lw.LensWizard.exec = drive_lens
for c in range(N_CAM):
    current_cam["c"] = c
    win._lens_wizard()
    pump(0.2)
    prof = p.lenses[c]
    assert prof is not None, c
    rep = prof.report
    print(f"cam{c + 1}: verdict {rep['verdict']}, f {prof.f_square:.1f} (true {FOCAL[c] * 1.0015:.1f}), "
          f"rms {prof.rms:.2f} px, {prof.n_views} views, bend {rep['distortion_border_px']:.0f} px, model {rep['model']}")
    assert rep["verdict"] == "good", rep["verdict_reasons"]
    assert abs(prof.f_square - FOCAL[c] * 1.0015) / FOCAL[c] < 0.01
    assert not prof.fisheye
print("lens wizard attached a GOOD profile to every camera OK")

# save / load the lens file through the result page
lensfile = os.path.join(SCRATCH, "cam1.klens.json")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (lensfile, ""))
wiz = lw.LensWizard(win, p, board_videos[0], 0, SCRATCH)
wiz.result_profile = p.lenses[0]
wiz.result_view = 0
wiz.page_result._save()
assert os.path.exists(lensfile)
back = lens.LensProfile.load(lensfile)
assert np.allclose(back.K, p.lenses[0].K) and back.report["verdict"] == "good"
print("lens file save / load OK")

# ---- "I already have a lens file" reaches the Attach button (I76) --------------------
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (lensfile, ""))
kept_lens1 = p.lenses[1]


def drive_loaded(self):
    self.show()
    self.restart()
    pump(0.1)
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_video
    self.page_video.cam.setCurrentIndex(1)
    self.page_video._load()
    assert self.page_video.isComplete(), self.page_video.status.text()
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_result, \
        f"(I76) a loaded profile goes straight to the result page, not {type(self.currentPage()).__name__}"
    assert self.page_result.isComplete() and self.button(lw.QWizard.FinishButton).isVisible()
    assert "cam2" in self.buttonText(lw.QWizard.FinishButton), self.buttonText(lw.QWizard.FinishButton)
    self.accept()
    return QDialog.Accepted


lw.LensWizard.exec = drive_loaded
p.lenses[1] = None
win._lens_wizard()
pump(0.2)
assert p.lenses[1] is not None and np.allclose(p.lenses[1].K, back.K), "the loaded profile is attached"
p.lenses[1] = kept_lens1
print("a lens file loaded in the wizard is attached through Finish OK")

# ---- a profile for another picture size is refused, with both sizes named (lens#2) ---
big = TrackingSession(os.path.join(SCRATCH, "lens_bigcam.mp4"), 10, FPS, 2 * W, 2 * H)
proj_sz = Project([sessions[0], big], ["cam1", "bigcam"], [0, 0])
wz = lw.LensWizard(win, proj_sz, board_videos[0], 1, SCRATCH)
wz.show()
wz.restart()
pump(0.1)
wz.next()
pump(0.1)
vp = wz.page_video
assert vp.cam.currentIndex() == 1
vp._load()                                        # the 640 x 480 cam1 profile, on a 1280 x 960 camera
st = vp.status.text()
assert not vp.isComplete(), "a 640 x 480 profile must not go to a 1280 x 960 camera"
assert f"{W} x {H}" in st and f"{2 * W} x {2 * H}" in st and "bigcam" in st, st
assert not vp.validatePage()
vp.cam.setCurrentIndex(0)                         # the camera it belongs to
assert vp.isComplete() and vp.validatePage() and wz.result_view == 0
# the same for a scanned board video
wz.scan = scans[0]
wz.result_profile = None
vp._loaded_path = None
vp.cam.setCurrentIndex(1)
assert not vp.isComplete() and "checkerboard video" in vp.status.text(), vp.status.text()
vp.cam.setCurrentIndex(0)
assert vp.isComplete()

# ---- a multi-camera Argus file: the line for the camera shown, refused when absent (I80)
wz.scan = None
argus2 = os.path.join(SCRATCH, "lens_gui_argus.txt")
with open(argus2, "w") as fh:
    fh.write(f"1 600 {W} {H} 319.5 239.5 1 -0.25 0.07 0 0 0\n2 620 {W} {H} 322.5 237.5 1 -0.25 0.07 0 0 0\n")
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (argus2, ""))
proj3 = Project([sessions[0], sessions[1], sessions[2]], ["cam1", "cam2", "cam3"], [0, 0, 0])
wz3 = lw.LensWizard(win, proj3, board_videos[0], 2, SCRATCH)
wz3.show()
wz3.restart()
pump(0.1)
wz3.next()
pump(0.1)
vp3 = wz3.page_video
assert vp3.cam.currentIndex() == 2
vp3._load()
assert wz3.result_profile is None and not vp3.isComplete(), "no line for camera 3: nothing attached"
assert "camera 3" in vp3.status.text() and "cam3" in vp3.status.text(), vp3.status.text()
vp3.cam.setCurrentIndex(1)                        # the line follows the camera
assert wz3.result_profile is not None and wz3.result_profile.f_square == 620, wz3.result_profile
assert abs(wz3.result_profile.principal[0] - 322.5) < 1e-9, "(I75) the principal point as written"
assert "camera 2" in vp3.status.text() and vp3.isComplete()
wz3.next()
pump(0.1)
assert wz3.currentPage() is wz3.page_result and wz3.result_view == 1
for w_ in (wz, wz3):
    w_.done(0)
pump(0.1)
print("picture-size check and Argus line per camera in the lens wizard OK")


# ---- wand wizard using the profiles -------------------------------------------
def drive_wand(self):
    self.show()
    self.restart()
    pump(0.1)
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_wand
    self.page_wand.length.setValue(WAND_L)
    self.page_wand.unit.setCurrentIndex(0)
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_cams
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_frame
    self.page_frame.r_drop.setChecked(True)
    assert self.page_frame.drop_name.currentText() == "ball"
    self.next()
    pump(0.1)
    assert self.currentPage() is self.page_run
    self.page_run._run()
    t0 = time.time()
    while self.result_calibration is None and self.page_run.btn_run.isEnabled() is False:
        pump(0.1)
        if time.time() - t0 > 300:
            raise TimeoutError("wand calibration did not finish")
    assert self.result_calibration is not None, self.page_run.report.toPlainText()[:800]
    self.accept()
    return QDialog.Accepted


cw.WandWizard.exec = drive_wand
win._wand_wizard()
pump(0.2)
res, grav = win._wand_result
rep = res.report
print(f"wand WITH lenses: verdict {rep['verdict']}, wand score {rep['wand_score_pct']:.3f} %, "
      f"rmse {rep['reproj_rmse_all']:.3f} px, focal {[round(f) for f in rep['focal_px']]}, g ratio {grav['g_ratio']:.4f}")
assert rep["verdict"] == "good", rep["verdict_reasons"]
assert rep["reproj_rmse_all"] < 1.0
assert "lens_profiles" in rep and all(rep["lens_profiles"])
err_with = []
for i, j, d in rep["camera_distances"]:
    truth = float(np.linalg.norm(true_pos[int(i)] - true_pos[int(j)]))
    err_with.append(abs(d - truth) / truth)
assert max(err_with) < 0.01, err_with
assert all(c.undistort.kind == "opencv" for c in p.calibration.cameras), "the calibration must keep the lens correction"
win._reconstruct_3d(quiet=True)
pump(0.3)
r = p.reconstruction
idx = {nm: i for i, nm in enumerate(r.names)}
fo, fx = r.xyz[:, idx["floor_origin"]], r.xyz[:, idx["floor_x"]]
dx = float(np.nanmedian(np.linalg.norm(fx - fo, axis=1)))
print(f"  floor distance reconstructed {dx:.4f} m, camera spacing error max {100 * max(err_with):.2f} %")
assert abs(dx - 0.5) < 0.003, dx
print("wand calibration with lens profiles OK")

# ---- the same without the profiles: must be measurably worse ---------------------
saved_lenses = list(p.lenses)
p.lenses = [None] * N_CAM
win._wand_wizard()
pump(0.2)
res0, grav0 = win._wand_result
rep0 = res0.report
err_without = []
for i, j, d in rep0["camera_distances"]:
    truth = float(np.linalg.norm(true_pos[int(i)] - true_pos[int(j)]))
    err_without.append(abs(d - truth) / truth)
print(f"wand WITHOUT lenses: verdict {rep0['verdict']}, rmse {rep0['reproj_rmse_all']:.3f} px, "
      f"camera spacing error max {100 * max(err_without):.2f} %")
assert rep0["reproj_rmse_all"] > 2 * rep["reproj_rmse_all"] or max(err_without) > 2 * max(err_with), \
    "distortion should hurt the pinhole solve"
p.lenses = saved_lenses

# ---- project round trip of the profiles ---------------------------------------------
p.save_npz(PROJ)
back = Project.load_npz(PROJ)
assert len(back.lenses) == N_CAM and all(l is not None for l in back.lenses)
assert np.allclose(back.lenses[2].K, p.lenses[2].K) and back.lenses[2].report["verdict"] == "good"
print("project round trip of lens profiles OK")

win._dev_probe.wait(60000)
win.close()
pump(0.2)
for pth in paths:
    if os.path.exists(pth + ".cotracker.npz"):
        os.remove(pth + ".cotracker.npz")
print("LENS GUI PASSED")
