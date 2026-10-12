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
     shift that puts it right; a correct rig gets no suggestion (I274).

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


def build(no_wand=(NO_WAND,), offsets=None, error=None, tag="sub", seed=3):
    """The rig as a project: camera c's local frame f shows reference instant f - offsets[c];
    the project is told offsets + error (a sync mistake). Cameras in `no_wand` get the
    landmarks but no clicks."""
    rng = np.random.RandomState(seed)
    offsets = list(offsets or [0, 3, -2, 5, 1])
    error = list(error or [0] * N_CAM)
    ss = []
    for c, cal in enumerate(CAMS):
        n_fr = T + 10
        s = TrackingSession(os.path.join(SCRATCH, f"{tag}_cam{c + 1}.mp4"), n_fr, FPS, W, H)
        ja, jb = s.add_landmark("wand A"), s.add_landmark("wand B")
        if c not in no_wand:
            tt = np.arange(n_fr, dtype=np.float64) - offsets[c]
            A, B = wand_at(tt)
            inside = (tt >= 0) & (tt <= T - 1)
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
        assert wz.result.report["verdict"] == "good", wz.result.report["verdict_reasons"]
        score, n = wand_score(cal_, P)
        assert score < 0.5 and n > 300, (score, n)
        print(f"A. camera 3 (no clicks) unticked, calibration of 1, 2, 4, 5: verdict good, wand score "
              f"{score:.3f} % over {n} frames OK")
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
    print("D. a not-calibrated camera survives calibration.json and .kcal.json OK")


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
    tc = wd.timing_check(P, "wand A", "wand B", list(range(N_CAM)), [None] * N_CAM)
    assert tc.suggestions == {1: -4}, (tc.suggestions, tc.lines)
    P_ok = build(no_wand=(), tag="subf_ok")
    tc_ok = wd.timing_check(P_ok, "wand A", "wand B", list(range(N_CAM)), [None] * N_CAM)
    assert tc_ok.suggestions == {}, (tc_ok.suggestions, tc_ok.lines)
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
    pump()
    print("G. exports hold cam1, 2, 4, 5 under their names, the DLT csv says which column is which; "
          "the import dialog keeps 'not calibrated' cameras OK")


SECTIONS = [sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, sec_g]

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
