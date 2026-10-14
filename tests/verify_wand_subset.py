"""Wand calibration of a chosen subset of the cameras (I272-I274, G182; offscreen, no GPU).

A synthetic 5-camera rig (1280 x 720, 100 fps) films a 0.5 m wand. Camera 3 has the wand landmarks
but no clicks, as on a big rig where not every camera saw the wand. Checked:

  A. the wizard ticks the cameras that have wand clicks (camera 3 unticked, with the reason), and
     Calibrate gives a calibration with camera 3 NOT calibrated and the others good (I272);
  B. the user's click on camera 5's tick leaves it out too;
  C. with every camera ticked, the wizard calls the solver with exactly the arguments it always did,
     and the calibration is bit-identical to a direct call (np.array_equal on every coefficient);
  D. a calibration with a gap saves and loads (calibration.json, .kcal.json) with the gap kept (I273);
  E. Reconstruct with a gap = Reconstruct of a project of only the calibrated cameras (np.array_equal);
  F. a camera whose offset is 4 frames wrong is found by the wand timing check, which suggests the
     shift that puts it right; a correct rig gets no suggestion; a camera that shares too few wand
     frames is "cannot time", never a suggestion (I274);
  G. exports and the import dialog with a camera that is not calibrated;
  H. the real window: wizard (one camera 3 frames off: the suggestion, the user's click on Apply) ->
     calibration of cameras 1, 2, 4, 5 -> 3D -> Reconstruct; Ctrl+Z (Edit -> Undo) puts the offsets
     AND the calibration back;
  I. the wizard's suggestion box: the user's click on "Calibrate with the offsets as they are" keeps
     the offsets and calibrates anyway.

Run: .venv\\Scripts\\python.exe tests\\verify_wand_subset.py
"""
import os
import sys
import time
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from kinetrace import calib
from kinetrace import calibio
from kinetrace import wand as wand_mod
from kinetrace import wanddata as wd
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)

W, H = 1280, 720
FPS = 100.0
T = 600
WAND_L = 0.5
N_CAM = 5
FOCAL = [1000.0, 1050.0, 980.0, 1020.0, 990.0]
NO_WAND = 2                          # camera 3 (0-based 2) has the landmarks but no clicks


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


CAMS = []
for k in range(N_CAM):
    a = 0.3 + k * 2 * np.pi / N_CAM
    pos = np.array([2.8 * np.cos(a), 2.8 * np.sin(a), 1.2 + 0.35 * (k % 2)])
    R, t = look_at(pos, np.array([0.0, 0.0, 1.0]))
    K = np.array([[FOCAL[k], 0, (W - 1) / 2], [0, FOCAL[k], (H - 1) / 2], [0, 0, 1]])
    CAMS.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))


def wand_at(t):
    """Both ends at (fractional) reference frame t: a Lissajous centre at changing speeds,
    the wand tumbling - motion in every direction."""
    s = np.asarray(t, np.float64) / FPS
    c = np.stack([0.55 * np.sin(1.9 * s) + 0.15 * np.sin(5.3 * s),
                  0.55 * np.sin(1.3 * s + 1.0) + 0.12 * np.cos(4.1 * s),
                  1.0 + 0.4 * np.sin(1.1 * s) + 0.08 * np.sin(6.7 * s)], -1)
    th, ph = 2.6 * s, 1.7 * s + 0.5
    d = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], -1)
    return c + 0.5 * WAND_L * d, c - 0.5 * WAND_L * d


def build(no_wand=(NO_WAND,), offsets=None, error=None, tag="sub", seed=3, few=None, rates=None):
    """The rig as a project: camera c's local frame f shows reference instant f - offsets[c];
    the project is told offsets + error (a sync mistake). Cameras in `no_wand` get the
    landmarks but no clicks."""
    rng = np.random.RandomState(seed)
    offsets = list(offsets or [0, 3, -2, 5, 1])
    error = list(error or [0] * N_CAM)
    rates = list(rates or [1.0] * N_CAM)               # camera c records at FPS * rates[c]
    ss = []
    for c, cal in enumerate(CAMS):
        n_fr = int(T * rates[c]) + 10
        s = TrackingSession(os.path.join(SCRATCH, f"{tag}_cam{c + 1}.mp4"), n_fr, FPS * rates[c], W, H)
        ja, jb = s.add_landmark("wand A"), s.add_landmark("wand B")
        if c not in no_wand:
            tt = (np.arange(n_fr, dtype=np.float64) - offsets[c]) / rates[c]
            A, B = wand_at(tt)
            inside = (tt >= 0) & (tt <= T - 1)
            if few is not None and c == few[0]:          # this camera saw the wand on a few frames only
                inside &= (tt >= 100) & (tt < 100 + few[1])
            for j, X in ((ja, A), (jb, B)):
                uv = cal.project(X) + rng.normal(0, 0.3, (n_fr, 2))
                vis = inside & (uv[:, 0] >= 2) & (uv[:, 0] < W - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < H - 2)
                s.tracks[vis, j] = uv[vis]
                s.tracked[vis, j] = True
        s.manual[:] = False
        s._touch()
        ss.append(s)
    told = [float(o + e) for o, e in zip(offsets, error)]
    base = told[0]
    told = [o - base for o in told]
    return Project(ss, [f"cam{c + 1}" for c in range(N_CAM)], told)


# ---- the wizard, in memory -------------------------------------------------------------
app = QApplication.instance() or QApplication([])
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace import calibwizard as cw  # noqa: E402


def pump(seconds=0.05):
    t_end = time.time() + seconds
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.005)


def new_wizard(proj):
    wz = cw.WandWizard(None, proj, SCRATCH)
    wz.show()
    wz.restart()
    pump()
    return wz


def goto(wz, page):
    while wz.currentPage() is not page:
        wz.next()
        pump(0.02)


def known_focal(wz):
    wz.page_cams.r_known.setChecked(True)
    for sp, f in zip(wz.page_cams._spins, FOCAL):
        sp.setValue(int(f))


def run_and_wait(wz, timeout=180):
    wz.page_run._run()
    t0 = time.time()
    while not wz.page_run.btn_run.isEnabled():
        pump(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError("calibration did not finish")
    pump(0.1)


def close_wizard(wz):
    wz.reject()
    pump()
    wz.deleteLater()
    pump()


def calibrate_through_wizard(proj, untick=()):
    """Open the wizard, (click the ticks in `untick` off), known focal lengths, Neither frame, Calibrate."""
    wz = new_wizard(proj)
    goto(wz, wz.page_wand)
    wz.page_wand.length.setValue(WAND_L)
    for c in untick:
        cb = wz.page_wand.ticks[c]
        assert cb.isChecked(), f"cam{c + 1} should start ticked"
        QTest.mouseClick(cb, Qt.LeftButton, pos=cb.rect().center())      # the user's click
        pump()
        assert not cb.isChecked(), f"the click did not untick cam{c + 1}"
    goto(wz, wz.page_cams)
    known_focal(wz)
    goto(wz, wz.page_frame)
    wz.page_frame.r_none.setChecked(True)
    goto(wz, wz.page_run)
    run_and_wait(wz)
    return wz


def wand_score(cal_, proj):
    rec = calib.reconstruct(proj.sessions, cal_, proj.rates, proj.offsets, (0, T - 1), ["wand A", "wand B"])
    L = np.linalg.norm(rec.xyz[:, 0] - rec.xyz[:, 1], axis=1)
    L = L[np.isfinite(L)]
    return 100.0 * L.std() / L.mean(), len(L)


# ---- the sections ----------------------------------------------------------------------
def sec_a():
    """The cameras with wand clicks are ticked; Calibrate leaves camera 3 out."""
    P = build(tag="suba")
    wz = calibrate_through_wizard(P)
    try:
        cal_ = wz.result_calibration
        assert cal_ is not None, "Calibrate failed: " + wz.page_run.report.toPlainText()[:400]
        ticks = [cb.isChecked() for cb in wz.page_wand.ticks]
        assert ticks == [c != NO_WAND for c in range(N_CAM)], ticks
        assert "no wand" in wz.page_wand.ticks[NO_WAND].text().lower(), wz.page_wand.ticks[NO_WAND].text()
        assert len(cal_) == N_CAM and cal_.camera(NO_WAND) is None, [c is None for c in cal_.cameras]
        assert cal_.views() == [0, 1, 3, 4], cal_.views()
        rep = wz.result.report
        # "ok", not "good", only because this rig's wand covers ~20 % of each picture (a coverage note)
        assert rep["verdict"] in ("good", "ok") and rep["wand_score_pct"] < 0.5, rep["verdict_reasons"]
        assert not any("cam3" in r for r in rep["verdict_reasons"]), rep["verdict_reasons"]
        assert rep["project_views"] == [0, 1, 3, 4] and rep["camera_names"] == ["cam1", "cam2", "cam4", "cam5"]
        html = wz.page_run.report.toPlainText()
        assert "cam3" in html and "not calibrated" in html, html[:600]
        score, n = wand_score(cal_, P)
        assert score < 0.5 and n > 300, (score, n)
        print(f"A. camera 3 (no clicks) unticked, calibration of 1, 2, 4, 5: verdict {rep['verdict']}, "
              f"wand score {score:.3f} % over {n} frames; the report lists cam3 as not calibrated OK")
    finally:
        close_wizard(wz)


def sec_b():
    """The user unticks camera 5: it is left out too."""
    P = build(tag="subb")
    wz = calibrate_through_wizard(P, untick=(4,))
    try:
        cal_ = wz.result_calibration
        assert cal_ is not None, "Calibrate failed: " + wz.page_run.report.toPlainText()[:400]
        assert cal_.views() == [0, 1, 3], cal_.views()
        assert cal_.camera(4) is None and cal_.camera(NO_WAND) is None
        print("B. clicking camera 5's tick off leaves it out: calibration of 1, 2, 4 OK")
    finally:
        close_wizard(wz)


def sec_c():
    """Every camera ticked: today's arguments, bit-identical result."""
    P = build(no_wand=(), tag="subc")
    seen = {}
    real = wand_mod.calibrate_wand

    def capture(*a, **k):
        seen.update(k)
        return real(*a, **k)

    wand_mod.calibrate_wand = capture
    try:
        wz = calibrate_through_wizard(P)
    finally:
        wand_mod.calibrate_wand = real
    try:
        assert wz.result_calibration is not None, wz.page_run.report.toPlainText()[:400]
        # what the wizard always passed for this rig (no lens profiles, known focal lengths)
        uv, frames = wd.collect_wand(P, "wand A", "wand B", max_frames=400)
        today = dict(wand_uv=uv, wand_length=WAND_L, sizes=[(W, H)] * N_CAM, coverage_uv=uv,
                     coverage_bg_uv=None, frame_ids=frames, focal=[float(int(f)) for f in FOCAL],
                     principal=None, bg_uv=None, estimate_focal=True, estimate_distortion=False,
                     unit=seen.get("unit"), names=[P.name(c) for c in range(N_CAM)])
        for k, v in today.items():
            got = seen.get(k, "<missing>")
            if isinstance(v, np.ndarray):
                assert isinstance(got, np.ndarray) and np.array_equal(got, v, equal_nan=True), k
            else:
                assert got == v, (k, got, v)
        assert set(seen) - {"progress"} == set(today), sorted(set(seen) ^ set(today))
        ref = real(**today).to_calibration()
        got_cal = wz.result_calibration
        assert len(got_cal) == N_CAM and got_cal.views() == list(range(N_CAM))
        for a_, b_ in zip(ref.cameras, got_cal.cameras):
            assert np.array_equal(a_.coefs, b_.coefs), "an all-camera calibration changed"
        print("C. all cameras ticked: the solver gets today's arguments; coefficients bit-identical OK")
    finally:
        close_wizard(wz)


def gap_calibration():
    cams_ = [CameraCalibration(c.coefs.copy(), W, H, NoUndistort(), pixel_origin=0.0, rmse=0.4) for c in CAMS]
    cams_[NO_WAND] = None
    return Calibration(cams_, "m", "wand (Kinetrace)")


def sec_d():
    """A gap survives calibration.json and .kcal.json."""
    cal_ = gap_calibration()
    P = build(tag="subd")
    P.calibration = cal_
    path = os.path.join(SCRATCH, "wand_subset.kinetrace")
    P.save(path)
    back = Project.load(path).calibration
    assert back is not None and len(back) == N_CAM and back.camera(NO_WAND) is None, back
    for v in back.views():
        assert np.array_equal(back.camera(v).coefs, cal_.camera(v).coefs)
    kpath = os.path.join(SCRATCH, "wand_subset.kcal.json")
    import json
    with open(kpath, "w", encoding="utf-8") as fh:
        json.dump(calibio.calibration_to_kcal(cal_), fh)
    kb = calibio.load_kcal(kpath)
    assert len(kb) == N_CAM and kb.camera(NO_WAND) is None and kb.views() == [0, 1, 3, 4]
    for v in kb.views():
        assert np.array_equal(kb.camera(v).coefs, cal_.camera(v).coefs)
    # (review) a calibration with nothing calibrated left saves (as no calibration) instead of failing
    P.calibration = Calibration([None] * N_CAM, "m")
    P.save(path)
    assert Project.load(path).calibration is None
    # removing the calibrated cameras one by one leaves no calibration, not a list of gaps
    P2 = build(tag="subd2")
    P2.calibration = Calibration([None, cal_.camera(1), None, None, None], "m")
    P2.remove_view(1)
    assert P2.calibration is None, P2.calibration
    P2.save(os.path.join(SCRATCH, "wand_subset_none.kinetrace"))
    print("D. a not-calibrated camera survives calibration.json and .kcal.json; an all-gap calibration saves "
          "as none OK")


def sec_e():
    """Reconstruct with a gap = the calibrated cameras alone."""
    P = build(no_wand=(), tag="sube")
    cal_ = gap_calibration()
    rec = calib.reconstruct(P.sessions, cal_, P.rates, P.offsets, (0, T - 1), ["wand A", "wand B"])
    keep = [0, 1, 3, 4]
    sub_cal = Calibration([cal_.camera(v) for v in keep], "m")
    rec2 = calib.reconstruct([P.sessions[v] for v in keep], sub_cal, [P.rates[v] for v in keep],
                             [P.offsets[v] for v in keep], (0, T - 1), ["wand A", "wand B"])
    assert np.array_equal(rec.xyz, rec2.xyz, equal_nan=True), "a gap changed the 3D"
    assert np.array_equal(rec.residual, rec2.residual, equal_nan=True)
    assert rec.per_cam.shape[-1] == N_CAM and np.isnan(rec.per_cam[..., NO_WAND]).all()
    assert np.array_equal(rec.per_cam[..., keep], rec2.per_cam, equal_nan=True)
    print(f"E. Reconstruct with camera 3 not calibrated = cameras 1, 2, 4, 5 alone "
          f"({int(np.isfinite(rec.xyz[..., 0]).sum())} points, np.array_equal) OK")


def sec_f():
    """A 4-frame sync mistake on camera 2 is found; a correct rig gets no suggestion."""
    P = build(no_wand=(), error=[0, 4, 0, 0, 0], tag="subf")
    # (I280) the inlier band scales like every residual verdict (long side / 1920, never below 1): 1.5 px
    # at 1280x720, not 1.0
    seen_thr, real_share = [], wd._inlier_share
    wd._inlier_share = lambda xa, xb, thr: (seen_thr.append(thr), real_share(xa, xb, thr))[1]
    try:
        tc = wd.timing_check(P, "wand A", "wand B", list(range(N_CAM)), [None] * N_CAM)
    finally:
        wd._inlier_share = real_share
    assert seen_thr and all(t_ == 1.5 for t_ in seen_thr), sorted(set(seen_thr))
    assert tc.suggestions == {1: -4}, (tc.suggestions, tc.lines)
    P_ok = build(no_wand=(), tag="subf_ok")
    tc_ok = wd.timing_check(P_ok, "wand A", "wand B", list(range(N_CAM)), [None] * N_CAM)
    assert tc_ok.suggestions == {} and tc_ok.verdict == "fine", (tc_ok.suggestions, tc_ok.lines)
    assert all(v == 0 for v in tc_ok.shift.values()), tc_ok.shift
    # camera 5 saw the wand on 20 frames only (and is 6 frames off): it cannot be timed - no suggestion
    P_few = build(no_wand=(), error=[0, 0, 0, 0, 6], few=(4, 20), tag="subf_few")
    tc_few = wd.timing_check(P_few, "wand A", "wand B", list(range(N_CAM)), [None] * N_CAM)
    assert tc_few.shift[4] is None and 4 not in tc_few.suggestions, (tc_few.shift, tc_few.lines)
    assert any("cannot check the timing of cam5" in ln for ln in tc_few.lines), tc_few.lines
    # (review) cameras at other frame rates: camera 2 records at half the rate and is 1 of ITS frames off;
    # the shifts are fitted in reference time, so only camera 2 gets a suggestion, of its own -1 frame
    P_rate = build(no_wand=(), error=[0, 1, 0, 0, 0], rates=[1.0, 0.5, 1.0, 1.0, 1.0], tag="subf_rate")
    assert abs(P_rate.rates[1] - 0.5) < 1e-12, P_rate.rates
    tc_rate = wd.timing_check(P_rate, "wand A", "wand B", list(range(N_CAM)), [None] * N_CAM)
    assert tc_rate.suggestions == {1: -1}, (tc_rate.suggestions, tc_rate.shift, tc_rate.lines)
    print(f"F. camera 2 told 4 frames late: the timing check suggests -4 ({tc.lines[0][:70]}...); "
          f"a correct rig: no suggestion OK")


def sec_g():
    """Exports name the calibrated cameras; the import dialog keeps a camera not calibrated."""
    import json
    cal_ = gap_calibration()
    names = [f"cam{c + 1}" for c in range(N_CAM)]
    m = calibio.to_models(cal_, names)
    assert [c.name for c in m.cameras] == ["cam1", "cam2", "cam4", "cam5"], [c.name for c in m.cameras]
    assert any("cam3 is not calibrated" in n for n in m.notes), m.notes
    toml = os.path.join(SCRATCH, "wand_subset_anipose.toml")
    calibio.write_anipose(m, toml)
    got = [c.name for c in calibio.read_anipose(toml)]
    assert got == ["cam1", "cam2", "cam4", "cam5"], got
    stem = os.path.join(SCRATCH, "wand_subset_files")
    files = cw.save_calibration_files(None, None, stem + ".kcal.json", cal=cal_)
    csv = np.loadtxt(stem + "_dltCoefs.csv", delimiter=",")
    assert csv.shape == (11, 4), csv.shape
    readme = open(stem + "_dltCoefs_README.txt", encoding="utf-8").read()
    assert "column 3 = camera 4" in readme and "camera 3 is not calibrated" in readme, readme[:400]
    assert any(f.endswith("_dltCoefs_README.txt") for f in files)
    # the import dialog: the gap file onto a 5-camera project keeps camera 3 not calibrated
    from kinetrace import view3d as v3
    dlg = v3.CalibrationDialog(None, names, [(W, H)] * N_CAM)
    assert dlg.load(stem + ".kcal.json")
    assert [cb.currentData() for cb in dlg._combos] == [0, 1, -1, 3, 4], [cb.currentData() for cb in dlg._combos]
    assert dlg._combos[NO_WAND].currentText() == "not calibrated"
    # the user also sets camera 5 to "not calibrated" in its list, then OK
    cb5 = dlg._combos[4]
    cb5.setCurrentIndex(cb5.findData(-1))
    dlg._accept()
    rc = dlg.result_calibration
    assert rc is not None and rc.views() == [0, 1, 3], rc
    assert np.array_equal(rc.camera(3).coefs, cal_.camera(3).coefs)
    dlg.deleteLater()
    # a file with FEWER cameras than the project: the cameras left over are "not calibrated"
    small = Calibration([cal_.camera(0), cal_.camera(1), cal_.camera(3)], "m")
    kp = os.path.join(SCRATCH, "wand_subset_small.kcal.json")
    with open(kp, "w", encoding="utf-8") as fh:
        json.dump(calibio.calibration_to_kcal(small), fh)
    dlg = v3.CalibrationDialog(None, names, [(W, H)] * N_CAM)
    assert dlg.load(kp)
    assert [cb.currentData() for cb in dlg._combos] == [0, 1, 2, -1, -1], [cb.currentData() for cb in dlg._combos]
    dlg._accept()
    assert dlg.result_calibration is not None and dlg.result_calibration.views() == [0, 1, 2]
    dlg.deleteLater()
    # (review) a saved gap calibration loses its TRAILING gaps: [c1, c2, null, c4] onto 5 cameras keeps
    # each camera's own entry (camera 4 = entry 4), camera 5 not calibrated - never packed by size
    short = Calibration([cal_.camera(0), cal_.camera(1), None, cal_.camera(3)], "m")
    kp2 = os.path.join(SCRATCH, "wand_subset_short.kcal.json")
    with open(kp2, "w", encoding="utf-8") as fh:
        json.dump(calibio.calibration_to_kcal(short), fh)
    dlg = v3.CalibrationDialog(None, names, [(W, H)] * N_CAM)
    assert dlg.load(kp2)
    assert [cb.currentData() for cb in dlg._combos] == [0, 1, -1, 3, -1], [cb.currentData() for cb in dlg._combos]
    dlg._accept()
    rc2 = dlg.result_calibration
    assert rc2.views() == [0, 1, 3] and np.array_equal(rc2.camera(3).coefs, cal_.camera(3).coefs)
    dlg.deleteLater()
    pump()
    print("G. exports hold cam1, 2, 4, 5 under their names, the DLT csv says which column is which; "
          "the import dialog keeps 'not calibrated' cameras OK")


def sec_h():
    """Through the real window: the wizard's calibration of cameras 1, 2, 4, 5 becomes the project's,
    the 3D menu's Reconstruct runs on them and says camera 3 is left out."""
    import cv2
    from PySide6.QtWidgets import QDialog
    from _clean import forget_recovery
    from kinetrace.app import READY, MainWindow
    P = build(tag="subh", error=[0, 0, 0, 3, 0])              # camera 4 is told 3 frames late
    wrong = float(P.offsets[3])
    paths = [s.video_path for s in P.sessions]
    img = np.full((H, W, 3), 30, np.uint8)
    for pth, s in zip(paths, P.sessions):
        vw = cv2.VideoWriter(pth, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for _ in range(s.n_frames):
            vw.write(img)
        vw.release()
        forget_recovery(pth)
    proj_path = os.path.join(SCRATCH, "wand_subset_gui.kinetrace")
    P.save(proj_path)
    real_critical = QMessageBox.critical
    QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))
    real_exec = cw.WandWizard.exec
    win = MainWindow()
    win.show()
    try:
        win._open_project_from_path(proj_path)
        for _ in range(300):
            pump(0.05)
            if win.state == READY and win.project is not None and win.project.n_views == N_CAM:
                break
        assert win.state == READY and win.project.n_views == N_CAM

        def drive(self):
            self.show()
            self.restart()
            pump(0.1)
            goto(self, self.page_wand)
            self.page_wand.length.setValue(WAND_L)
            assert [cb.isChecked() for cb in self.page_wand.ticks] == [True, True, False, True, True]
            goto(self, self.page_cams)
            known_focal(self)
            goto(self, self.page_frame)
            self.page_frame.r_none.setChecked(True)
            goto(self, self.page_run)
            run_and_wait(self)
            pg = self.page_run
            assert self.result_calibration is None and pg.timing_box.isVisible(), pg.report.toPlainText()[:300]
            assert "cam4" in pg.timing_text.text(), pg.timing_text.text()
            QTest.mouseClick(pg.btn_apply_offsets, Qt.LeftButton)       # the user's click: Apply
            pump(0.1)
            t0 = time.time()
            while not pg.btn_run.isEnabled():
                pump(0.05)
                assert time.time() - t0 < 180, "calibration after Apply did not finish"
            pump(0.1)
            assert self.result_calibration is not None, pg.report.toPlainText()[:400]
            assert "Timing check" in pg.report.toPlainText()
            self.accept()
            return QDialog.Accepted

        cw.WandWizard.exec = drive
        win._wand_wizard()
        pump(0.2)
        p = win.project
        assert p.calibration is not None and p.calibration.views() == [0, 1, 3, 4], p.calibration
        assert abs(p.offsets[3] - (wrong - 3)) < 1e-9, (p.offsets, wrong)
        pump(0.1)
        assert win.act_recon.isEnabled()
        win.act_recon.trigger()                       # 3D -> Reconstruct, as the menu does
        for _ in range(200):
            pump(0.05)
            if p.reconstruction is not None:
                break
        r = p.reconstruction
        assert r is not None, "Reconstruct did not run with cameras 1, 2, 4, 5 calibrated"
        ia, ib = r.names.index("wand A"), r.names.index("wand B")
        L = np.linalg.norm(r.xyz[:, ia] - r.xyz[:, ib], axis=1)
        assert np.isfinite(L).sum() > 300 and np.nanstd(L) / np.nanmean(L) < 0.01, (np.isfinite(L).sum(),)
        assert np.isnan(r.per_cam[..., NO_WAND]).all()
        t = win.toast.text()
        assert "cam3" in t and "not calibrated" in t, t
        # (review) Carve Volume counts only the CALIBRATED cameras' silhouettes: two calibrated cameras
        # with three silhouettes is still "needs three cameras", never a two-camera sliver
        full_cal = p.calibration
        p.calibration = Calibration([full_cal.camera(0), full_cal.camera(1), None, None, None], "m")
        real_masks, real_info = win._masks_at_instant, QMessageBox.information
        seen = []
        win._masks_at_instant = lambda t, name=None: [np.ones((H, W), bool)] * N_CAM
        QMessageBox.information = staticmethod(lambda *a, **k: seen.append(a[1]) or QMessageBox.Ok)
        try:
            win.act_hull.trigger()                    # 3D -> Carve Volume at This Frame
            pump(0.2)
        finally:
            win._masks_at_instant, QMessageBox.information = real_masks, real_info
            p.calibration = full_cal
        assert "Volume needs three cameras" in seen, seen
        # Edit -> Undo: the offsets AND the calibration made at them go back, in one step
        assert win.act_undo.isEnabled()
        win.activateWindow()
        win.canvas.setFocus()
        QTest.keyClick(win.canvas, Qt.Key_Z, Qt.ControlModifier)       # the user's Ctrl+Z
        pump(0.2)
        assert abs(p.offsets[3] - wrong) < 1e-9 and p.calibration is None and p.reconstruction is None, \
            (p.offsets, p.calibration)
        # (review) Apply again, then the user edits camera 2's offset in the CAMERAS panel: Undo must leave
        # that edit (and the calibration) alone instead of taking them back with the Apply
        win._wand_wizard()
        pump(0.2)
        assert abs(p.offsets[3] - (wrong - 3)) < 1e-9 and p.calibration is not None
        cal_now = p.calibration
        nudge = win.cameras._rows[1].nudges[1]              # CAMERAS: camera 2's "one frame later" button
        nudge.click()
        nudge.click()
        pump(0.1)
        hand = float(p.offsets[1])
        if win.act_undo.isEnabled():
            win.act_undo.trigger()
            pump(0.2)
        assert abs(p.offsets[1] - hand) < 1e-9 and abs(p.offsets[3] - (wrong - 3)) < 1e-9 \
            and p.calibration is cal_now, (p.offsets, p.calibration)
        print(f"H. through the window: cam4 3 frames off -> suggestion, Apply -> calibration of 1, 2, 4, 5 -> "
              f"3D -> Reconstruct: {int(np.isfinite(L).sum())} wand frames, the toast names cam3; Undo puts the "
              f"offset and the calibration back, but never a later hand edit OK")
    finally:
        cw.WandWizard.exec = real_exec
        QMessageBox.critical = real_critical
        if hasattr(win, "_dev_probe"):
            win._dev_probe.wait(60000)
        win.project.dirty = False if win.project is not None else None
        win.close()
        pump(0.2)
        win.deleteLater()
        pump(0.1)
        for pth in paths:
            forget_recovery(pth)


def sec_i():
    """The suggestion box's other button: calibrate with the offsets as they are."""
    P = build(no_wand=(), error=[0, 4, 0, 0, 0], tag="subi")
    before = list(P.offsets)
    wz = new_wizard(P)
    try:
        goto(wz, wz.page_wand)
        wz.page_wand.length.setValue(WAND_L)
        goto(wz, wz.page_cams)
        known_focal(wz)
        goto(wz, wz.page_frame)
        wz.page_frame.r_none.setChecked(True)
        goto(wz, wz.page_run)
        run_and_wait(wz)
        pg = wz.page_run
        assert pg.timing_box.isVisible() and "cam2" in pg.timing_text.text(), pg.timing_text.text()
        QTest.mouseClick(pg.btn_keep_offsets, Qt.LeftButton)        # the user's click
        pump(0.1)
        t0 = time.time()
        while not pg.btn_run.isEnabled():
            pump(0.05)
            assert time.time() - t0 < 180
        pump(0.1)
        assert list(P.offsets) == before, (P.offsets, before)
        assert wz.result_calibration is not None or "failed" in pg.report.toPlainText().lower()
        print("I. 'Calibrate with the offsets as they are': offsets unchanged, the solve runs OK")
    finally:
        close_wizard(wz)


SECTIONS = [sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, sec_g, sec_h, sec_i]

if __name__ == "__main__":
    failed = []
    for fn in SECTIONS:
        try:
            fn()
        except Exception as e:      # noqa: BLE001 - every section reports, then the run fails
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {str(e)[:400]}".encode("ascii", "replace").decode())
            traceback.print_exc(limit=2)
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("PASSED")
