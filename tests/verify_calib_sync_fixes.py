"""Calibration, 3D and sync fixes (docs/AUDIT.md I340-I356, the 2026-10-25 full sweep's PR 3). Every section
FAILS on main 393f21e; the GUI halves (G222-G224, the per-camera sub-frame search the app asks for) are in
verify_calib_sync_gui.py. No GPU, no video.

  I340  dltCoefs.csv and the DLTdv xyzpts written for one calibration reproject onto each other
        (left-handed easyWand / DLTdv rig; K + R/t rig with camera 1 at the file's origin)
  I341  a MATLAB v7 dvProject column of zeros is a camera that is NOT calibrated; a 1-camera v7 file loads
  I342  dltCoefs.csv with a header row loads for 11 cameras and for 1 camera
  I343  a lens with fx ~= fy undistorts bit-identically after its JSON round trip
  I344  a fisheye click where the model has no inverse undistorts to NaN and gives no 3D point
  I345  a lens model that folds inside the picture is not called sound (border_check, calib_frame_ok)
  I346  an instant a hair below a whole frame is that frame (first frame after a gap)
  I347  the drop-worst option (max_residual) is gone
  I348  the sub-frame search never scores an offset on fewer cells than it chose
  I349  the robust sub-frame pass does not blame a camera it cannot tell from the others
  I350  motion sync from the start of the recording finds a camera that started later
  I351  motion sync of a camera at twice the rate lands within one local frame; the sub-frame search takes
        one width per camera
  I352  the wand's bad-click floor is each camera's own
  I353  a 0.13-frame wobble between the wand's timing windows is not a dropped frame
  I354  the wand drop's "still" band is the same for a portrait and a landscape camera
  I355  a ChArUco frame two cameras share counts whatever a third camera grazes
  I356  the multi-camera xypts sidecar says how the other cameras are sampled, with their offsets

.venv\\Scripts\\python.exe tests\\verify_calib_sync_fixes.py
"""
import inspect
import json
import os
import shutil
import sys
import types
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")
OUT = os.path.join(HERE, "out", "calib_sync_fixes")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

import numpy as np  # noqa: E402
import scipy.io  # noqa: E402

from kinetrace import autoexport, calib, calibio, charuco, lens, sync, wand, wanddata, wandsync  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, Reconstruction, dlt_from_camera, dlt_project  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def section(name, fn):
    print(name, flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, f"{name.split()[0]} ran without an error", repr(e))


def look_at(C, up=(0, 0, 1)):
    C = np.asarray(C, float)
    z = -C / np.linalg.norm(C)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


# ------------------------------------------------------------------ I340
def _i340():
    K = np.array([[1500.0, 0, 959.5], [0, 1500.0, 539.5], [0, 0, 1]])
    F = np.diag([1.0, 1.0, -1.0])
    cams = []
    for C in ((3, -1, 1), (-2, -2.5, 1.2), (0.5, 3, 0.8)):
        R, t = look_at(C)
        cams.append(CameraCalibration(dlt_from_camera(K, R @ F, t), 1920, 1080))
    cal = Calibration(cams, unit="m")
    check(calib.world_is_left_handed(cal), "the rig is left-handed (as easyWand / DLTdv rigs are)")
    X = np.random.default_rng(0).uniform(-0.3, 0.3, (20, 3)) @ F.T
    names = [f"p{j}" for j in range(20)]
    rec = Reconstruction(0, names, X[None], np.zeros((1, 20)), np.full((1, 20), 3))
    csv = os.path.join(OUT, "lh_dltCoefs.csv")
    calibio.dlt_csv_matlab(cal, csv)
    coefs = np.loadtxt(csv, delimiter=",")

    def miss(pts, cams_, coefs_):
        worst = 0.0
        for c, cam in enumerate(cams_):
            clicks = dlt_project(cam.coefs, X) + 1.0             # DLTdv's 1-based pixels of the same landmark
            worst = max(worst, float(np.nanmax(np.linalg.norm(dlt_project(coefs_[:, c], pts) - clicks, axis=1))))
        return worst

    # the exports/ folder's xyzpts (autoexport) and Ctrl+E's DLTdv kind (write_points3d with the models)
    proj = SimpleNamespace(calibration=cal, reconstruction=rec, names=["cam1", "cam2", "cam3"])
    auto = os.path.join(OUT, "lh_auto_xyzpts.csv")
    autoexport._write_points_in_camera_world(calibio, proj, auto)
    pa = np.loadtxt(auto, delimiter=",", skiprows=1).reshape(-1, 3)
    m = miss(pa, cams, coefs)
    check(m < 1e-6, "exports/ xyzpts reproject through dltCoefs.csv onto the clicks", f"{m:.3f} px")
    ctrl = os.path.join(OUT, "lh_ctrle_xyzpts.csv")
    calibio.write_points3d(rec, ctrl, "dltdv", models=calibio.to_models(cal, ["cam1", "cam2", "cam3"]))
    pc = np.loadtxt(ctrl, delimiter=",", skiprows=1).reshape(-1, 3)
    m = miss(pc, cams, coefs)
    check(m < 1e-6, "Ctrl+E DLTdv xyzpts reproject through dltCoefs.csv onto the clicks", f"{m:.3f} px")
    check(np.array_equal(pc, np.round(X, 6)) or float(np.max(np.abs(pc - X))) < 1e-6,
          "the DLTdv xyzpts are the calibration's own world (no Z flip)")
    # the K + R/t formats keep their (mirrored) world, with their points
    an = os.path.join(OUT, "lh_anipose.csv")
    calibio.write_points3d(rec, an, "anipose", models=calibio.to_models(cal, ["cam1", "cam2", "cam3"]))
    back, _ = calibio.read_points3d(an)
    check(float(np.nanmax(np.abs(back.xyz[0] - X * np.array([1, 1, -1])))) < 1e-5,
          "the Anipose points stay mirrored with their (K + R/t) cameras")

    # K + R/t: camera 1 at the file's origin (the OpenCV stereo convention)
    Kk = np.array([[1200.0, 0, 959.5], [0, 1200.0, 539.5], [0, 0, 1]])
    a = np.radians(-10.0)                                    # camera 2: 0.6 m to the right, turned in 10 deg
    R2 = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    krt = [{"K": Kk, "R": np.eye(3), "t": np.zeros(3), "width": 1920, "height": 1080},
           {"K": Kk, "R": R2, "t": -R2 @ np.array([0.6, 0.0, 0.0]), "width": 1920, "height": 1080}]
    calk = Calibration.from_krt(krt, unit="m")
    check(calk.origin_shift is not None, "a K + R/t import moved its origin (camera 1 sits on the file's)")
    Xk = np.random.default_rng(1).uniform(-0.2, 0.2, (10, 3)) + np.array([0.3, 0.0, 2.5]) - calk.origin_shift
    reck = Reconstruction(0, names[:10], Xk[None], np.zeros((1, 10)), np.full((1, 10), 2))
    csvk = os.path.join(OUT, "krt_dltCoefs.csv")
    calibio.dlt_csv_matlab(calk, csvk)
    ck = np.loadtxt(csvk, delimiter=",")
    pk_path = os.path.join(OUT, "krt_xyzpts.csv")
    calibio.write_points3d(reck, pk_path, "dltdv", models=calibio.to_models(calk, ["cam1", "cam2"]))
    pk = np.loadtxt(pk_path, delimiter=",", skiprows=1).reshape(-1, 3)
    worst = 0.0
    for c, cam in enumerate(calk.cameras):
        clicks = dlt_project(cam.coefs, Xk) + 1.0
        worst = max(worst, float(np.nanmax(np.linalg.norm(dlt_project(ck[:, c], pk) - clicks, axis=1))))
    check(worst < 1e-6, "a K + R/t calibration's dltCoefs.csv and xyzpts reproject onto the clicks",
          f"{worst:.3f} px")


# ------------------------------------------------------------------ I341 / I342
def _i341():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    L1 = dlt_from_camera(K, np.eye(3), np.array([0.0, 0.0, 5.0]))
    p = os.path.join(OUT, "zero_dvProject.mat")
    scipy.io.savemat(p, {"udExport": {"data": {"dltcoef": np.column_stack([L1, np.zeros(11)]),
                                               "movsizes": np.array([[1080, 1920], [1080, 1920]], float)}}})
    cal = Calibration.load_dltdv_project(p)
    check(cal.camera(1) is None, "a v7 column of zeros is a camera that is not calibrated",
          type(cal.camera(1)).__name__)
    check(cal.camera(0) is not None and np.allclose(cal.camera(0).coefs, L1), "the calibrated column is kept")
    p1 = os.path.join(OUT, "one_dvProject.mat")
    scipy.io.savemat(p1, {"udExport": {"data": {"dltcoef": L1.reshape(11, 1),
                                                "movsizes": np.array([[1080, 1920]], float)}}})
    try:
        c1 = Calibration.load_dltdv_project(p1)
        ok = c1.camera(0) is not None and np.allclose(c1.camera(0).coefs, L1)
        why = ""
    except Exception as e:  # noqa: BLE001
        ok, why = False, repr(e)
    check(ok, "a one-camera v7 dvProject loads", why)
    from kinetrace import dltdvproject
    zero, good = np.zeros(11), L1
    check(calib.dlt_column_calibrated(good) and not calib.dlt_column_calibrated(zero)
          and dltdvproject.DLTdvProject.calibrated.__code__.co_names.count("dlt_column_calibrated") > 0,
          "one 'calibrated' rule shared by both readers")


def _i342():
    rng = np.random.default_rng(3)
    for ncam in (1, 10, 11, 12):
        for header in (False, True):
            if ncam == 1 and not header:
                continue
            M = rng.uniform(-5, 5, (11, ncam))
            pth = os.path.join(OUT, "c%d_%s_dltCoefs.csv" % (ncam, "h" if header else "n"))
            with open(pth, "w", encoding="ascii") as fh:
                if header:
                    fh.write(",".join("cam%d" % (k + 1) for k in range(ncam)) + "\n")
                for r in M:
                    fh.write(",".join("%.10g" % v for v in r) + "\n")
            try:
                cal = Calibration.load_dlt_csv(pth)
                ok = len(cal.cameras) == ncam and all(np.allclose(cal.cameras[k].coefs, M[:, k], atol=1e-8)
                                                       for k in range(ncam))
                why = f"{len(cal.cameras)} cameras"
            except Exception as e:  # noqa: BLE001
                ok, why = False, repr(e)
            check(ok, f"dltCoefs.csv with {ncam} camera(s), header {header}: loads every column", why)


# ------------------------------------------------------------------ I343
def _i343():
    K = np.array([[1300.004, 0, 960], [0, 1299.996, 540], [0, 0, 1.0]])
    m = lens.LensProfile(1920, 1080, K, np.array([-0.2, 0.05, 0, 0, 0.0])).undistort_model()
    d = json.loads(json.dumps(m.to_json()))
    m2 = calib.undistort_from_json(d)
    pts = np.array([[1900.0, 1000.0], [0.0, 0.0], [1500.0, 200.0]])
    a, b = m.undistort(pts), m2.undistort(pts)
    check(np.array_equal(a, b), "a lens with fx ~= fy undistorts bit-identically after Save + reopen",
          f"{np.nanmax(np.abs(a - b)):.3g} px")
    plain = calib.OpenCVUndistort(K, np.array([-0.2, 0.05, 0, 0, 0.0]))
    check("P" not in plain.to_json(), "a model whose P is K writes no P (files unchanged)")


# ------------------------------------------------------------------ I344 / I345
def _i344():
    Kf = np.array([[600, 0, 959.5], [0, 600, 539.5], [0, 0, 1.0]])
    m = calib.OpenCVUndistort(Kf, np.array([-0.3, 0, 0, 0.0]), fisheye=True)
    u = m.undistort([[0.0, 0.0], [959.5, 539.5], [700.0, 400.0]])
    check(np.isnan(u[0]).all(), "a fisheye corner with no inverse undistorts to NaN", str(u[0]))
    check(np.isfinite(u[1:]).all(), "points the fisheye model describes still undistort")
    c0 = dlt_from_camera(Kf, np.eye(3), np.array([0.05, 0, 2.0]))
    c1 = dlt_from_camera(Kf, np.eye(3), np.array([-0.05, 0, 2.0]))
    sess = SimpleNamespace(tracks=np.array([[[0.0, 0.0]]]), exportable=np.array([[True]]))
    ct = calib.calib_tracks(sess, CameraCalibration(c0, 1920, 1080, m))
    uv = np.stack([ct[0, 0], dlt_project(c1, np.zeros((1, 3)))[0]])
    xyz, res, ncam, _ = calib.triangulate_batch(np.stack([c0, c1]), uv[None])
    check(not np.isfinite(xyz).any() and ncam[0] < 2, "that click gives no 3D point",
          f"xyz {xyz[0]}, residual {res[0]}")


def _fold_case(fv, k1, k2):
    Kv = np.array([[fv, 0, 959.5], [0, fv, 539.5], [0, 0, 1.0]])
    dv = np.array([k1, k2, 0, 0, 0.0])
    rf = np.sqrt((-3 * k1 - np.sqrt(9 * k1 * k1 - 20 * k2)) / (10 * k2))       # first root of d(r D(r))/dr
    peak = rf * (1 + k1 * rf ** 2 + k2 * rf ** 4) * fv
    return Kv, dv, peak


def _i345():
    c = np.array([959.5, 539.5])
    d = -c / np.linalg.norm(c)
    corner = float(np.hypot(959.5, 539.5))
    for fv, k1, k2 in ((1700.0, -0.85, 0.325), (1700.0, -0.40, 0.045)):
        Kv, dv, peak = _fold_case(fv, k1, k2)
        chk = lens.LensProfile(1920, 1080, Kv, dv).border_check()
        check(peak < corner, f"(the f {fv:.0f} k1 {k1} k2 {k2} model folds inside the picture)")
        check(chk["runaway"] and chk["valid_frac"] < 1.0,
              f"border_check calls the folded model (k1 {k1}) a runaway", str(chk))
        mv = calib.OpenCVUndistort(Kv, dv)
        past = c + np.arange(peak + 0.25, corner + 0.01, 0.25)[:, None] * d
        _, ok = calib.calib_frame_ok(CameraCalibration(np.zeros(11), 1920, 1080, mv), past)
        check(not ok.any(), f"calib_frame_ok refuses every raw point past the fold (k1 {k1})",
              f"{int(ok.sum())} of {len(ok)} accepted")
        inside = c + np.array([0.5 * peak, 0.9 * peak])[:, None] * d
        check(np.isfinite(mv.undistort(inside)).all(), f"points inside the fold still undistort (k1 {k1})")
    # a sound wide lens (I73's GoPro-like standard model) is untouched
    Ks = np.array([[1000.0, 0, 959.5], [0, 1000.0, 539.5], [0, 0, 1]])
    sound = lens.LensProfile(1920, 1080, Ks, np.array([-0.25, 0.07, 0, 0, -0.008])).border_check()
    check(not sound["runaway"] and sound["valid_frac"] == 1.0, "a sound wide lens is still sound", str(sound))


# ------------------------------------------------------------------ I346
def _i346():
    ct = np.full((10, 1, 2), np.nan)
    ct[4, 0] = (100, 100)
    lt = calib.local_frame(np.array([3.0]), 1.2, 0.4)
    check(float(lt[0]) < 4.0, "(the instant is a hair below local frame 4)", repr(float(lt[0])))
    out = calib.sample_tracks_at(ct, lt)
    check(np.array_equal(out[0, 0], [100.0, 100.0]), "the first frame after a gap gives its click",
          str(out[0, 0]))
    ct2 = np.full((10, 1, 2), np.nan)
    ct2[3, 0], ct2[4, 0] = (90, 90), (100, 100)
    lt2 = np.array([3.5])
    check(np.allclose(calib.sample_tracks_at(ct2, lt2)[0, 0], [95, 95]), "a real half frame still interpolates")


# ------------------------------------------------------------------ I347
def _i347():
    for fn in (calib.triangulate_batch, calib.reconstruct, calib._Prepared.triangulate):
        check("max_residual" not in inspect.signature(fn).parameters, f"{fn.__name__} has no max_residual")


# ------------------------------------------------------------------ I348 / I349 rigs
def _two_cam_rig(T=100):
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
    c0 = dlt_from_camera(K, np.eye(3), np.array([0.3, 0, 3.0]))
    R1 = np.array([[0.8660254, 0, -0.5], [0, 1, 0], [0.5, 0, 0.8660254]])
    c1 = dlt_from_camera(K, R1, np.array([-0.3, 0, 3.0]))
    t = np.arange(T)
    X = np.column_stack([0.3 * np.sin(t / 7.0), 0.2 * np.cos(t / 5.0), 0.1 * np.sin(t / 3.0)])

    def sess(coefs, nan_rows=None):
        tr = dlt_project(coefs, X).reshape(T, 1, 2).astype(np.float64)
        if nan_rows is not None:
            tr[nan_rows] = np.nan
        return SimpleNamespace(tracks=tr, exportable=np.ones((T, 1), bool),
                               points=[SimpleNamespace(name="p")], n_frames=T)
    cal = Calibration([CameraCalibration(c0, 1920, 1080), CameraCalibration(c1, 1920, 1080)])
    return cal, sess, c0, c1


def _i348():
    T = 100
    cal, sess, c0, c1 = _two_cam_rig(T)
    full = []                                   # per triangulation: kept cells / cells with data
    orig = calib._Prepared.triangulate

    def spy(self, uv, min_cams=2, *a, **k):
        seen = np.isfinite(uv).all(axis=-1)                    # (T, N, C)
        full.append(int((seen.sum(axis=-1) >= 2).sum()))
        return orig(self, uv, min_cams, *a, **k)
    calib._Prepared.triangulate = spy
    try:
        offs, before, after = calib.estimate_offsets([sess(c0), sess(c1, slice(50, 60))], cal, [1, 1], [0, 0.6],
                                                     (0, T - 1), report={})
    finally:
        calib._Prepared.triangulate = orig
    counts = sorted(set(full))
    check(len(counts) == 1, "every offset the search scores is scored on the same cells",
          f"cell counts {counts[:6]}...")
    check(abs(offs[1]) < 0.05, "and it still finds the true offset 0", f"{offs[1]:.3f}")
    # one width per camera
    seen = []
    orig_s = calib._Prepared.samples

    def spy_s(self, rates, offsets, tt, keep=None):
        seen.append(float(offsets[1]))
        return orig_s(self, rates, offsets, tt, keep)
    calib._Prepared.samples = spy_s
    try:
        r = calib.estimate_offsets([sess(c0), sess(c1)], cal, [1, 1], [0, 0.0], (0, T - 1), search=[1.0, 2.0],
                                   passes=1)
        reach, why = max(abs(np.array(seen))), ""
    except Exception as e:  # noqa: BLE001
        reach, why, r = 0.0, repr(e), None
    finally:
        calib._Prepared.samples = orig_s
    check(r is not None and reach >= 1.9, "estimate_offsets takes one search width per camera (+/-2 here)",
          why or f"reach {reach:.2f}")


def _i349():
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
    coefs = np.stack([dlt_from_camera(K, np.eye(3), np.array([0.05, 0, 2.0])),
                      dlt_from_camera(K, np.eye(3), np.array([-0.05, 0, 2.0])),
                      dlt_from_camera(K, np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]), np.array([0, 0, 2.0]))])
    cal = Calibration([CameraCalibration(coefs[c], 1920, 1080) for c in range(3)])
    T = 120
    t = np.arange(T)
    X = np.column_stack([0.2 * np.sin(t / 9.0), np.zeros(T), 0.1 * np.sin(t / 4.0)])     # the cameras' plane
    good_blamed = bad_kept = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        sessions = []
        for c in range(3):
            tr = dlt_project(coefs[c], X) + rng.normal(0, 0.2, (T, 2))
            if c == 2:
                tr[40:42, 0] += 50.0
            sessions.append(SimpleNamespace(tracks=tr.reshape(T, 1, 2), exportable=np.ones((T, 1), bool),
                                            points=[SimpleNamespace(name="p")], n_frames=T))
        prep = calib._Prepared(sessions, cal, None)
        before = [np.isfinite(ct[:, 0]).all(axis=1).copy() for ct in prep.ct]
        calib._drop_bad_observations(prep, [1, 1, 1], [0.0, 0.0, 0.0], t)
        removed = {c: set(np.nonzero(before[c] & ~np.isfinite(prep.ct[c][:, 0]).all(axis=1))[0].tolist())
                   for c in range(3)}
        if np.isfinite(prep.ct[2][40:42, 0]).all(axis=1).any():
            bad_kept += 1
        if (removed[0] & {40, 41} or removed[1] & {40, 41}) and not (removed[2] & {40, 41}):
            good_blamed += 1
    check(good_blamed == 0, "an ambiguous cell never blames a good camera alone", f"{good_blamed} of 40")
    check(bad_kept == 0, "the +50 px frames are never kept", f"{bad_kept} of 40")
    from kinetrace import retrack
    check(getattr(calib, "BLAME_MARGIN", None) == 1.5 and "BLAME_MARGIN" in inspect.getsource(retrack._blame),
          "one 'clearly the culprit' margin shared with Re-track")


# ------------------------------------------------------------------ I350 / I351
def _with_fake_motion(fake, fn):
    orig = sync.motion_signal
    sync.motion_signal = fake
    try:
        return fn()
    finally:
        sync.motion_signal = orig


def _i350():
    rng = np.random.default_rng(0)
    base = rng.exponential(1, 9000) * (rng.random(9000) < 0.05) * 20 + rng.normal(0, 0.1, 9000)
    full = {"ref": base[:6000], "cam": base[300:6300]}

    def fake(path, f0, f1, progress=None, should_cancel=None, thumb_w=160):
        return np.asarray(full[path][max(0, f0):f1], float)

    def run(rr):
        r = sync.estimate_offsets_from_motion(["ref", "cam"], [1.0, 1.0], rr, 400, prior=None)[0]
        return r.offset, r.result.verdict
    got = _with_fake_motion(fake, lambda: {rr: run(rr) for rr in ((0, 3000), (100, 3100), (1000, 4000))})
    for rr, (o, v) in got.items():
        check(o == -300 and v == "clear", f"motion sync over {rr} finds the camera started 300 frames later",
              f"{o} {v}")


def _i351():
    rng = np.random.default_rng(1)
    centres, widths, amps = rng.uniform(0, 1200, 25), rng.uniform(3, 12, 25), rng.uniform(0.5, 2.0, 25)

    def activity(Tq):
        Tq = np.asarray(Tq, np.float64)[..., None]
        return (amps * np.exp(-0.5 * ((Tq - centres) / widths) ** 2)).sum(-1)
    CAM = {}

    def fake(path, f0, f1, progress=None, should_cancel=None, thumb_w=160):
        r, o = CAM[path]
        g = np.arange(max(0, int(f0)), int(f1))
        a = activity((g - o) / r)
        out = np.zeros(len(g))
        out[1:] = np.abs(np.diff(a))
        return out
    S = 30

    def errors(r):
        errs = []
        for o in np.linspace(-0.8 * r * S, 0.8 * r * S, 201):
            CAM["ref"], CAM["cam"] = (1.0, 0.0), (r, float(o))
            cs = sync.estimate_offsets_from_motion(["ref", "cam"], [1.0, r], (100, 1000), S, prior=[None, 0.0])[0]
            errs.append(cs.offset - o)
        return np.array(errs)
    e1, e2 = _with_fake_motion(fake, lambda: (errors(1.0), errors(2.0)))
    check(np.abs(e1).max() <= 0.52 + 1e-9, "same rate: as before (within half a frame and a tie)",
          f"{np.abs(e1).max():.3f}")
    check(np.abs(e2).max() <= 1.0, "a camera at twice the rate lands within one of its own frames",
          f"worst {np.abs(e2).max():.3f}, mean {e2.mean():+.3f}")


# ------------------------------------------------------------------ I352
def _wand_run(sizes, bad_px=3.5, seed=1, bad_cam=2):
    rng = np.random.default_rng(seed)
    C = len(sizes)
    Ks, Rs, ts = [], [], []
    for k, (W, H) in enumerate(sizes):
        a = 0.3 + k * np.pi / 2.2
        R, t = look_at(np.array([2.8 * np.cos(a), 2.8 * np.sin(a), 1.2 + 0.35 * (k % 2)]))
        f = 0.8 * W
        Ks.append(np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]]))
        Rs.append(R)
        ts.append(t)
    F, L = 150, 0.5
    ctr = rng.uniform([-0.4, -0.4, 0.7], [0.4, 0.4, 1.3], (F, 3))
    d = rng.normal(size=(F, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ends = np.stack([ctr - 0.5 * L * d, ctr + 0.5 * L * d], axis=1)
    uv = np.full((F, C, 2, 2), np.nan)
    for c in range(C):
        sc = calib.residual_scale(*sizes[c])
        Xc = ends.reshape(-1, 3) @ Rs[c].T + ts[c]
        x = Xc @ Ks[c].T
        uv[:, c] = (x[:, :2] / x[:, 2:3]).reshape(F, 2, 2) + rng.normal(0, 0.3 * sc, (F, 2, 2))
    bad = [(fr, bad_cam, 0) for fr in (10, 40, 70, 100, 130)]
    for fr, c, e in bad:
        uv[fr, c, e, 0] += bad_px
    res = wand.calibrate_wand(uv, L, sizes, focal=[K[0, 0] for K in Ks], estimate_focal=False)
    rej = {(o["row"], o["cam"], o["end"]) for o in res.report["outliers"] if o["kind"] == "wand"}
    return [b in rej for b in bad], res


def _i352():
    caught, res = _wand_run([(3840, 2160), (3840, 2160), (1280, 720)])
    check(all(caught), "a 720p camera's 3.5 px mis-clicks are caught beside two 4K cameras", str(caught))


# ------------------------------------------------------------------ I353 / I354
def _i353():
    def mk(rng, o1):
        return {"range": rng, "n_frames": rng[1] - rng[0] + 1, "offsets": [0.0, o1],
                "report": {"per_view": {1: "sharp"}}, "sigma": np.array([np.nan, 0.001])}

    def run(three, six):
        def fake(rv, cal, a, b, frames, n, progress=None, cancelled=None):
            if n == 3:
                return [mk((100 * k, 100 * k + 99), three[k]) for k in range(3)]
            return [mk((50 * k, 50 * k + 49), six[k]) for k in range(6)]
        orig = wandsync._part_estimates
        wandsync._part_estimates = fake
        try:
            return wandsync._window_check(types.SimpleNamespace(n_views=2), None, "a", "b", np.arange(300))[
                "per_camera"][1]
        finally:
            wandsync._part_estimates = orig
    for three, six in (([0.0, 0.13, 0.12], [0.0, 0.0, 0.13, 0.12, 0.13, 0.12]),
                       ([0.0, 0.0, 0.12], [0.0, 0.0, 0.0, 0.01, 0.12, 0.12])):
        r = run(three, six)
        check(r["pattern"] == "unsteady" and r["verdict"] == "ok",
              f"windows {three}: a small wobble reads unsteady (ok), not a dropped frame",
              f"{r['pattern']} {r['verdict']}")
    r = run([0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    check(r["pattern"] == "jump", "a whole-frame step is still a jump", r["pattern"])
    check(wandsync.JUMP_MIN_FRAMES == 0.5, "one jump size for both checks")


def _i354():
    seen = []

    def fake_sample(project, name, t):
        uv = np.zeros((len(t), len(project.sessions), 2))
        uv[:, :, 1] = np.arange(len(t))[:, None] * 5.0
        return uv

    def fake_fall(uv, seen_, still, min_frames):
        seen.append(np.array(still, float))
        return None
    o1, o2 = wanddata._sample_name, wanddata._fall_in_run
    wanddata._sample_name, wanddata._fall_in_run = fake_sample, fake_fall
    try:
        sess = [types.SimpleNamespace(width=3840, height=2160), types.SimpleNamespace(width=2160, height=3840),
                types.SimpleNamespace(width=1920, height=1080)]
        wanddata.find_fall(types.SimpleNamespace(sessions=sess), "ball", 0, 50)
    finally:
        wanddata._sample_name, wanddata._fall_in_run = o1, o2
    st = seen[0]
    check(st[0] == st[1], "a portrait 4K camera's 'still' band equals the landscape one", str(st))
    check(abs(st[0] - 3.072) < 1e-9 and abs(st[2] - 1.536) < 1e-9, "landscape bands unchanged (4K, 1080p)", str(st))


# ------------------------------------------------------------------ I355
def _i355():
    spec = charuco.BoardSpec()
    N = spec.n_corners
    rng = np.random.default_rng(1)
    v0, v1 = rng.uniform(100, 1800, (N, 2)), rng.uniform(100, 1800, (N, 2))
    v2 = np.full((N, 2), np.nan)
    v2[:5] = rng.uniform(100, 1800, (5, 2))

    def scan(views):
        C = len(views)
        return charuco.RigScan(spec, [(1920, 1080)] * C, ["a", "b", "c"][:C], [0], [[0.0] * C], [views],
                               [None] * C)
    wand_uv, _, bg, _, _ = charuco.pairs_and_extras(scan([v0, v1, v2]))
    check(len(wand_uv) == 1, "two cameras that share the board give a frame whatever a third grazes",
          f"{len(wand_uv)} frames")
    if len(wand_uv):
        check(np.isnan(wand_uv[0, 2]).all() and (len(bg) == 0 or np.isnan(bg[:, 2]).all()),
              "the grazing camera (6 shared ids with nobody) is left out of that frame")
    w2, _, _, _, _ = charuco.pairs_and_extras(scan([v0, v1, np.full((N, 2), np.nan)]))
    check(len(w2) == 1, "two cameras alone: as before")


# ------------------------------------------------------------------ I356
def _i356():
    W, H = 1920, 1080
    s0, s1 = TrackingSession("a.mp4", 50, 120.0, W, H), TrackingSession("b.mp4", 60, 120.0, W, H)
    for s in (s0, s1):
        s.add_landmark("p")
    OFF = 7.37
    for fr in range(50):
        s0.set_position(fr, 0, 100.0 + 10 * fr, 500.0)
    for fr in range(60):
        s1.set_position(fr, 0, 100.0 + 10 * (fr - OFF), 500.0)
    p = Project([s0, s1], ["a", "b"], [0.0, OFF])
    out = os.path.join(OUT, "multi_xypts.csv")
    files = p.export_multi_dltdv(out)
    side = open(files[1], encoding="utf-8").read()
    check("same instant" not in side and "nearest whole frame" in side,
          "the sidecar says the other cameras are at their nearest whole frame")
    check("7.37" in side, "and gives each camera's offset")
    rows = np.loadtxt(out, delimiter=",", skiprows=1)
    check(abs(rows[20, 2] - 1 - 296.3) < 1e-6, "the xypts rows themselves are unchanged", str(rows[20]))


for name, fn in (("I340 the DLTdv pair in one world", _i340),
                 ("I341 DLTdv v7 projects", _i341),
                 ("I342 dltCoefs.csv headers", _i342),
                 ("I343 the lens projection matrix saved exactly", _i343),
                 ("I344 the fisheye sentinel", _i344),
                 ("I345 a folded lens", _i345),
                 ("I346 an instant a hair below a frame", _i346),
                 ("I347 no drop-worst option", _i347),
                 ("I348 the sub-frame search's scored cells", _i348),
                 ("I349 blame only when clear", _i349),
                 ("I350 motion sync from the start", _i350),
                 ("I351 motion sync at another rate", _i351),
                 ("I352 the wand floor per camera", _i352),
                 ("I353 wobble vs dropped frame", _i353),
                 ("I354 the 'still' band of a portrait camera", _i354),
                 ("I355 ChArUco frames per pair", _i355),
                 ("I356 the multi-camera xypts sidecar", _i356)):
    section(name, fn)

print("\nverify_calib_sync_fixes: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
