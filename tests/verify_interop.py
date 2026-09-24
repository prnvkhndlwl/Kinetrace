"""Other programs' files in and out of Kinetrace (no GPU).

[1] 2D tracks (trackio.py): DeepLabCut CSV, DLTdv / Argus xypts (one camera,
    all cameras, top-left and bottom-left, with and without the sidecar),
    SLEAP (1.x analysis CSV and sleap-io's sleap / instances / points / frames
    layouts), Kinetrace's own tracks.csv. Each is checked two ways: a file
    written by hand the way the other program writes it (a hand-computed
    point), and a round trip through Kinetrace's own export. Refusals
    (multi-animal, training data, several tracks, a locale-mangled file) say
    why. `apply` keeps cells the file has no data for, leaves out positions
    outside the picture or the video and says so.
[2] Calibrations (calibio.py): an OpenCV-style rig, a left-handed 1-based
    (easyWand / DLTdv style) one, a bottom-left one and an LWM lens become K,
    R, t cameras that project within 1e-6 px of Kinetrace (LWM: fitted, within
    half a pixel); Anipose / OpenCV YAML + JSON / MATLAB files round-trip
    exactly and re-import; MATLAB's 1-based, transposed fields and Blender's
    lens / shift / camera matrix are checked against hand values; lens
    profiles, offsets and 3D points (Kinetrace / DLTdv / Anipose) round-trip.
[3] File -> Import Tracks through the app (offscreen): asks for the video
    first, Ctrl+Z, a refusal, one camera out of an all-cameras file; mask
    images in (File -> Import -> Silhouettes), polygons / PNGs out.
[4] The command-line converter as a real subprocess: every subcommand, the
    exit codes of `check` (0 clean, 1 warnings, 2 errors) and the refusals.

Run: .venv\\Scripts\\python.exe tests\\verify_interop.py
"""
import os
import shutil
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from kinetrace import projectfile, trackio  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "interop")
if os.path.isdir(OUT):
    shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
fails = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        fails.append(what)


def write(name, text):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p


def refused(path, words):
    try:
        trackio.read(path)
    except trackio.TrackImportError as e:
        return all(w in str(e) for w in words), str(e)
    return False, "not refused"


def session(T=120, W=640, H=480, name="v.mp4"):
    return TrackingSession(os.path.join(OUT, name), T, 30.0, W, H)


def imported_into(path, T=120, **kw):
    s = session(T)
    imp = trackio.read(path)
    summ = trackio.apply(s, imp, **kw)
    return s, imp, summ


# a source session: three points, gaps, low confidence, one leaving the picture
T, W, H = 120, 640, 480
src = session()
rng = np.random.default_rng(1)
for nm in ("snout", "tail base", "left foot"):
    src.add_point(0, 10.0, 10.0, name=nm)
xy = np.stack([np.linspace(50, 600, T), np.linspace(40, 440, T)], -1).astype(np.float32)
for j in range(3):
    w = xy + rng.normal(0, 3, xy.shape).astype(np.float32) + 20 * j
    src.write_segment(0, w[:, None, :], np.ones((T, 1), bool), [j], rng.uniform(0.3, 1.0, (T, 1)).astype(np.float32))
src.clear_window([1], 30, 44)                              # a gap
src.tracks[100:, 2] = (700.0, 100.0)                       # left the picture: blanked the app's way
src.tracked[100:, 2] = False
exp = src.exportable


def same_as_src(s, decimals, what, conf_tol=None):
    ok_cells = (s.tracked[:, :3] == exp).all()
    d = np.nanmax(np.abs(s.tracks[:, :3] - src.tracks[:, :3])[exp]) if exp.any() else 0.0
    check(ok_cells and d <= 0.5 * 10 ** -decimals + 1e-4, f"{what}: the same cells, positions within "
          f"{0.5 * 10 ** -decimals:g} px", f"cells equal {ok_cells}, max diff {d}")
    if conf_tol is not None:
        dc = np.abs(s.confidence[:, :3] - src.confidence[:, :3])[exp].max()
        check(dc <= conf_tol, f"{what}: confidence carried", f"max diff {dc}")


# ------------------------------------------------------------------ 1
print("\n[1a] DeepLabCut CSV")
p = os.path.join(OUT, "dlc_roundtrip.csv")
src.export_dlc_csv(p)
check(trackio.detect(p) == "dlc", "recognised from its header rows")
s, imp, summ = imported_into(p)
check([q.name for q in s.points] == ["snout", "tail base", "left foot"], "body parts become points by name")
same_as_src(s, 3, "round trip through Kinetrace's DLC export", conf_tol=5e-5)
p = write("dlc_real.csv",
          "scorer,DLC_resnet50_lizardSep1shuffle1_100000,DLC_resnet50_lizardSep1shuffle1_100000,"
          "DLC_resnet50_lizardSep1shuffle1_100000\n"
          "bodyparts,snout,snout,snout\ncoords,x,y,likelihood\n"
          "0,12.5,20.25,0.99\n1,13.0,21.0,0.40\n2,,,\n3,639.4,479.4,1.0\n")
s, imp, summ = imported_into(p)
check(np.allclose(s.tracks[0, 0], (12.5, 20.25)) and abs(s.confidence[1, 0] - 0.40) < 1e-6,
      "a DeepLabCut file: pixel positions taken as they are (both put pixel centres on whole numbers)")
check(not s.tracked[2, 0] and s.tracked[3, 0], "blank cells stay empty; the last pixel is inside")
check("DLC_resnet50" in " ".join(imp.notes), "the scorer is reported")
ok, msg = refused(write("dlc_multi.csv", "scorer,S,S,S\nindividuals,a1,a1,a1\nbodyparts,snout,snout,snout\n"
                                          "coords,x,y,likelihood\n0,1,2,0.9\n"), ["multi-animal", "one animal"])
check(ok, "a multi-animal file is refused with the reason", msg)
ok, msg = refused(write("dlc_train.csv", "scorer,me,me\nbodyparts,snout,snout\ncoords,x,y\n"
                                          "labeled-data/vid/img000.png,10,20\n"), ["training-data"])
check(ok, "a training-data file (image names) is refused with the reason", msg)

print("\n[1b] DLTdv / Argus xypts")
p = os.path.join(OUT, "one_xypts.csv")
src.export_dltdv_csv(p)
check(trackio.detect(p) == "dltdv", "recognised from pt1_cam1_X columns")
s, imp, summ = imported_into(p)
check([q.name for q in s.points] == ["snout", "tail base", "left foot"], "names from the _pointnames sidecar")
same_as_src(s, 3, "DLTdv8 convention (first pixel 1, top-left) round trip")
p = os.path.join(OUT, "flip_xypts.csv")
src.export_dltdv_csv(p, flip_y=True)
s, imp, summ = imported_into(p)
check(imp.flip_y, "the sidecar's bottom-left convention is honoured")
same_as_src(s, 3, "bottom-left (older DLTdv / Argus) round trip")
p = write("bare_xypts.csv", "pt1_cam1_X,pt1_cam1_Y\n1,1\nNaN,NaN\n640,480\n")
s, imp, summ = imported_into(p)
check(np.allclose(s.tracks[0, 0], (0, 0)) and np.allclose(s.tracks[2, 0], (639, 479)) and not s.tracked[1, 0],
      "no sidecar: DLTdv8's pixel (1, 1) is Kinetrace's (0, 0), NaN is no data")
check(s.points[0].name == "pt1" and imp.notes and "pt1" in imp.notes[0], "and the points are called pt1, ... (said)")
# all cameras: a project whose second camera starts 5 frames later
a, b = session(name="a.mp4"), session(name="b.mp4")
for s_ in (a, b):
    s_.add_point(0, 1.0, 1.0, name="snout")
    s_.add_point(0, 1.0, 1.0, name="tail base")
    s_.write_segment(0, (xy[:, None, :] + rng.normal(0, 2, (T, 2, 2))).astype(np.float32),
                     np.ones((T, 2), bool), [0, 1])
proj = Project([a, b], ["cam A", "cam B"], [0.0, -5.0])
p = os.path.join(OUT, "multi_xypts.csv")
proj.export_multi_dltdv(p)
imp = trackio.read(p)
check(imp.n_cameras == 2 and imp.rows == "reference" and imp.names == ["snout", "tail base"],
      "an all-cameras file: two cameras, rows in the reference camera's frames, names from its sidecar")
worst = 0.0
for c, s_ in enumerate((a, b)):
    t = session(name=f"t{c}.mp4")
    trackio.apply(t, imp, camera=c, frame_of_row=lambda r, c=c: proj.map_frame(0, c, r))
    m = s_.exportable & t.tracked
    worst = max(worst, float(np.abs(t.tracks - s_.tracks)[m].max()))
    if c == 1:
        check(not t.tracked[T - 5:].any() and t.tracked[:T - 5].all(),
              "camera B's frames come back through the offset (its last 5 frames are after camera A's end)")
check(worst <= 5e-5 + 1e-4, "each camera's positions come back", f"max diff {worst}")

print("\n[1c] SLEAP")
v14 = ("track,frame_idx,instance.score,head.x,head.y,head.score,tail.x,tail.y,tail.score\n"
       "track_0,0,0.93,101.5,55.25,0.97,140.0,60.0,0.88\n"
       "track_0,1,,102.0,56.0,,,,\n"
       "track_0,3,0.51,103.0,57.0,0.20,141.0,61.0,0.30\n")
p = write("sleap14.csv", v14)
check(trackio.detect(p) == "sleap", "SLEAP 1.x analysis CSV recognised")
s, imp, summ = imported_into(p)
check(imp.names == ["head", "tail"] and np.allclose(s.tracks[0, 0], (101.5, 55.25))
      and abs(s.confidence[0, 1] - 0.88) < 1e-6, "nodes become points; node scores are the confidence")
check(s.manual[1, 0] and s.confidence[1, 0] == 1.0 and not s.tracked[1, 1],
      "a user-labelled instance is hand-placed; its missing node stays empty")
check(not s.tracked[2].any() and "track_0" in " ".join(imp.notes), "a frame without a row stays empty; track said")
sio_sleap = ("track,frame_idx,instance.score,head.score,head.x,head.y,tail.score,tail.x,tail.y\n"
             "track_0,0,0.93,0.97,101.5,55.25,0.88,140.0,60.0\n"
             "track_0,1,0.70,0.9,999.0,999.0,0.9,999.0,999.0\n"
             "track_0,1,,,102.0,56.0,,,\n"
             "track_0,3,0.51,0.20,103.0,57.0,0.30,141.0,61.0\n")
sio_inst = ("frame_idx,track,track_score,score,head.x,head.y,head.score,tail.x,tail.y,tail.score\n"
            "0,track_0,0.8,0.93,101.5,55.25,0.97,140.0,60.0,0.88\n1,track_0,0.8,,102.0,56.0,,,,\n"
            "3,track_0,0.8,0.51,103.0,57.0,0.20,141.0,61.0,0.30\n")
sio_pts = ("frame_idx,node,x,y,track,track_score,instance_score,score\n"
           "0,head,101.5,55.25,track_0,0.8,0.93,0.97\n0,tail,140.0,60.0,track_0,0.8,0.93,0.88\n"
           "1,head,102.0,56.0,track_0,0.8,,\n"
           "3,head,103.0,57.0,track_0,0.8,0.51,0.20\n3,tail,141.0,61.0,track_0,0.8,0.51,0.30\n")
sio_frames = ("frame_idx,inst0.track,inst0.track_score,inst0.score,inst0.head.x,inst0.head.y,inst0.head.score,"
              "inst0.tail.x,inst0.tail.y,inst0.tail.score\n"
              "0,track_0,0.8,0.93,101.5,55.25,0.97,140.0,60.0,0.88\n1,track_0,0.8,,102.0,56.0,,,,\n"
              "2,,,,,,,,,\n3,track_0,0.8,0.51,103.0,57.0,0.20,141.0,61.0,0.30\n")
ref_s = s
for label, text in (("sleap-io 'sleap' (alphabetical columns, a user label beside a prediction)", sio_sleap),
                    ("sleap-io 'instances'", sio_inst), ("sleap-io 'points'", sio_pts),
                    ("sleap-io 'frames'", sio_frames)):
    p = write("sio.csv", text)
    s, imp, summ = imported_into(p)
    same = (np.array_equal(s.tracked, ref_s.tracked) and np.allclose(s.tracks, ref_s.tracks, equal_nan=True)
            and np.allclose(s.confidence, ref_s.confidence) and np.array_equal(s.manual, ref_s.manual))
    check(same, f"{label}: the same result as the analysis CSV")
ok, msg = refused(write("sleap_two.csv", v14 + "track_1,0,0.9,300,300,0.9,320,320,0.9\n"), ["2 tracks", "one animal"])
check(ok, "two tracks are refused with the reason", msg)

print("\n[1d] Kinetrace tracks.csv (the project file's own)")
kp = os.path.join(OUT, "src.kinetrace")
src.save(kp)
with zipfile.ZipFile(kp) as z:
    name = next(n for n in z.namelist() if n.endswith("/tracks.csv"))
    p = write("tracks.csv", z.read(name).decode("utf-8"))
check(trackio.detect(p) == "kinetrace", "recognised from frame, point, x, y")
s, imp, summ = imported_into(p)
m = exp
check(np.array_equal(s.tracked[:, :3], src.tracked) and np.array_equal(s.tracks[:, :3][m], src.tracks[m])
      and np.array_equal(s.confidence[:, :3][m], src.confidence[m]), "bit-exact positions and confidence")

print("\n[1e] into a camera that already has data")
s = session()
s.add_point(0, 5.0, 5.0, name="snout")
s.set_position(50, 0, 77.0, 88.0)
before = s.tracks[50, 0].copy()
p = write("partial.csv", "scorer,S,S,S\nbodyparts,snout,snout,snout\ncoords,x,y,likelihood\n"
                         "0,10,10,0.9\n50,,,\n60,700,10,0.9\n130,10,10,0.9\n")
imp = trackio.read(p)
summ = trackio.apply(s, imp)
check(s.n_points == 1 and summ["updated"] == 1 and summ["new"] == 0, "a known name updates that point")
check(np.array_equal(s.tracks[50, 0], before) and s.manual[50, 0], "a cell the file has no data for is kept")
check(np.allclose(s.tracks[0, 0], (10, 10)) and not s.manual[0, 0], "a cell it has replaces the old one")
check(summ["outside"] == 1 and summ["beyond"] == 1 and "outside the picture" in summ["sentence"]
      and "frames this video does not have" in summ["sentence"], "outside the picture / video: left out and said",
      summ["sentence"])

print("\n[1f] files that are not tracks")
ok, msg = refused(write("other.csv", "time,value\n0,1\n"), ["DeepLabCut", "DLTdv", "SLEAP"])
check(ok, "an unknown table names the formats Kinetrace reads", msg)
ok, msg = refused(write("semi.csv", "frame;point;x;y\n0;a;1,5;2,5\n"), ["';'", "locale"])
check(ok, "a semicolon / decimal-comma file is explained", msg)

# ------------------------------------------------------------------ 2
print("\n[2a] calibration -> K [R | t] cameras (calibio.to_models)")
import cv2  # noqa: E402

from kinetrace import calibio  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, LWMUndistort, NoUndistort,  # noqa: E402
                             OpenCVUndistort, dlt_from_camera)

CW, CH = 1920, 1080


def rig(n=3, dist=True):
    """n cameras on an arc looking at the origin, 3 m away (x_cam = R X + t)."""
    cams = []
    for i in range(n):
        a = np.radians(-30 + 30 * i)
        C = np.array([3 * np.sin(a), -0.4 + 0.2 * i, -3 * np.cos(a)])
        z = -C / np.linalg.norm(C)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        R = np.vstack([x, np.cross(z, x), z])
        K = np.array([[1400.0 + 20 * i, 0, 959.5 + 3 * i], [0, 1402.0 + 20 * i, 539.5 - 2 * i], [0, 0, 1]])
        d = np.array([-0.21, 0.08, 0.0005, -0.0003, -0.01]) if dist else np.zeros(5)
        cams.append({"K": K, "R": R, "t": -R @ C, "dist": d, "width": CW, "height": CH})
    return cams


def worst_diff(cal_a, models):
    """Largest pixel difference between Kinetrace's projection and the model's,
    over points of the working volume."""
    rng_ = np.random.default_rng(5)
    X = rng_.uniform(-0.5, 0.5, (400, 3))
    worst = 0.0
    for cam, m in zip(cal_a.cameras, models.cameras):
        a_ = cam.project(X)
        b_ = m.project(models.world_to_export(X))
        ok_ = np.isfinite(a_).all(1) & np.isfinite(b_).all(1)
        worst = max(worst, float(np.abs(a_[ok_] - b_[ok_]).max()))
    return worst


truth = rig()
cal = Calibration.from_krt(truth, unit="m", source="synthetic")
ms = calibio.to_models(cal, names=["A", "B", "C"])
check(not ms.mirrored and max(m.check_px for m in ms.cameras) < 1e-6,
      "an OpenCV-style calibration: the exported cameras reproduce it", str([m.check_px for m in ms.cameras]))
errK = max(np.abs(m.K - c["K"]).max() for m, c in zip(ms.cameras, truth))
errR = max(np.abs(m.R - c["R"]).max() for m, c in zip(ms.cameras, truth))
errt = max(np.abs(m.t - c["t"]).max() for m, c in zip(ms.cameras, truth))
errd = max(np.abs(m.dist - c["dist"]).max() for m, c in zip(ms.cameras, truth))
check(errK < 1e-6 and errR < 1e-9 and errt < 1e-9 and errd < 1e-9,
      "... and K, distortion, R, t come back as they were, in the file's own world (origin moved back)",
      f"K {errK:.2e} R {errR:.2e} t {errt:.2e} d {errd:.2e}")
# a DLTdv / easyWand-style calibration: 1-based pixels, a left-handed world
F = np.diag([1.0, 1.0, -1.0])
lh = []
for c in rig(dist=False):
    K1 = c["K"].copy()
    K1[:2, 2] += 1.0                                  # MATLAB pixels
    L = dlt_from_camera(K1, c["R"] @ F, c["t"])       # world mirrored: X' = F X
    lh.append(CameraCalibration(L, CW, CH, NoUndistort(), pixel_origin=1.0))
cal_lh = Calibration(lh, "m", "left-handed")
ms_lh = calibio.to_models(cal_lh)
check(ms_lh.mirrored and any("mirrored" in n for n in ms_lh.notes), "a left-handed world is exported mirrored, "
      "and said")
check(worst_diff(cal_lh, ms_lh) < 1e-6, "... and still projects every point where Kinetrace does",
      f"{worst_diff(cal_lh, ms_lh):.2e}")
check(max(np.abs(m.K - c["K"]).max() for m, c in zip(ms_lh.cameras, rig())) < 1e-6,
      "the 1-based principal point comes back 0-based")
# y counted from the bottom edge
yf = []
for c in rig(dist=False):
    A = np.array([[1, 0, 0], [0, -1, CH - 1], [0, 0, 1.0]])     # ours -> bottom-left 0-based
    L = dlt_from_camera(A @ c["K"], c["R"], c["t"])
    yf.append(CameraCalibration(L, CW, CH, NoUndistort(), pixel_origin=0.0, y_flip=True))
cal_yf = Calibration(yf, "m")
ms_yf = calibio.to_models(cal_yf)
check(worst_diff(cal_yf, ms_yf) < 1e-6 and max(np.abs(m.K - c["K"]).max() for m, c in zip(ms_yf.cameras, rig()))
      < 1e-6, "a bottom-left calibration gives the ordinary top-left camera")
# DLTdv's LWM lens correction: fitted with an OpenCV model, error reported
lw = []
for c in rig():
    gx, gy = np.meshgrid(np.linspace(0, CW - 1, 25), np.linspace(0, CH - 1, 15))
    raw = np.column_stack([gx.ravel(), gy.ravel()])
    und = cv2.undistortPoints(raw.reshape(-1, 1, 2), c["K"], c["dist"], None, None, c["K"]).reshape(-1, 2)
    L = dlt_from_camera(c["K"], c["R"], c["t"])
    lw.append(CameraCalibration(L, CW, CH, LWMUndistort(raw, und), pixel_origin=0.0))
cal_lw = Calibration(lw, "m")
ms_lw = calibio.to_models(cal_lw)
fitpx = max(m.lens_fit_px for m in ms_lw.cameras)
check(0 < fitpx < 0.5 and worst_diff(cal_lw, ms_lw) < 0.5, "an LWM lens is fitted with an OpenCV model within half a "
      "pixel, and the fit's worst error is recorded", f"fit {fitpx:.3f}, projection {worst_diff(cal_lw, ms_lw):.3f}")
check(any("Check:" in n for n in ms_lw.notes), "the export says how close it is, in pixels")

print("\n[2b] calibration files: Anipose, OpenCV YAML / JSON, MATLAB, Blender")
for label, fname, wr in (("Anipose calibration.toml", "calibration.toml", calibio.write_anipose),
                         ("OpenCV YAML", "cameras.yml", calibio.write_opencv),
                         ("OpenCV JSON", "cameras.json", calibio.write_opencv),
                         ("MATLAB .mat", "cameras.mat", calibio.write_matlab)):
    p = os.path.join(OUT, fname)
    wr(ms, p)
    back = calibio.read_cameras(p)
    d = max(max(np.abs(a_.K - b_.K).max(), np.abs(a_.R - b_.R).max(), np.abs(a_.t - b_.t).max(),
                np.abs(a_.dist - b_.dist).max()) for a_, b_ in zip(ms.cameras, back))
    re_cal = calibio.from_models(back, unit="m")
    X = np.random.default_rng(2).uniform(-0.5, 0.5, (200, 3))
    worst = max(float(np.abs(m.project(X) - b_.project(X)).max()) for m, b_ in zip(ms.cameras, back))
    check(len(back) == 3 and d < 1e-9 and [b_.name for b_ in back] == ["A", "B", "C"] and worst < 1e-6,
          f"{label}: written and read back exactly", f"param diff {d:.2e}, px {worst:.2e}")
    re_ms = calibio.to_models(re_cal)
    check(worst_diff(re_cal, re_ms) < 1e-6 and max(np.abs(a_.t - b_.t).max() for a_, b_ in
                                                     zip(ms.cameras, re_ms.cameras)) < 1e-6,
          f"{label}: imported as a Kinetrace calibration, it exports the same cameras again")
from scipy.io import loadmat  # noqa: E402
mm = loadmat(os.path.join(OUT, "cameras.mat"), squeeze_me=True, struct_as_record=False)["cameras"][0]
check(abs(mm.K[0, 2] - (truth[0]["K"][0, 2] + 1)) < 1e-9 and np.allclose(mm.IntrinsicMatrix, mm.K.T)
      and np.allclose(mm.RotationMatrix, truth[0]["R"].T) and list(mm.ImageSize) == [CH, CW],
      "MATLAB: 1-based principal point, IntrinsicMatrix = K', RotationMatrix = R', ImageSize = [rows cols]")
hand = write("hand.toml", "[cam_0]\nname = \"left\"\nsize = [ 1280, 1024,]\n"
                          "matrix = [ [ 1000.0, 0.0, 639.5,], [ 0.0, 1000.0, 511.5,], [ 0.0, 0.0, 1.0,],]\n"
                          "distortions = [ -0.1, 0.0, 0.0, 0.0, 0.0,]\nrotation = [ 0.0, 0.0, 0.0,]\n"
                          "translation = [ 0.0, 0.0, 2.0,]\n\n[cam_1]\nname = \"right\"\nsize = [ 1280, 1024,]\n"
                          "matrix = [ [ 1000.0, 0.0, 639.5,], [ 0.0, 1000.0, 511.5,], [ 0.0, 0.0, 1.0,],]\n"
                          "distortions = [ 0.0, 0.0, 0.0, 0.0, 0.0,]\nrotation = [ 0.0, 0.3, 0.0,]\n"
                          "translation = [ -0.5, 0.0, 2.0,]\n\n[metadata]\nadjusted = false\nerror = 0.2\n")
hm = calibio.read_cameras(hand)
check(len(hm) == 2 and hm[0].name == "left" and hm[0].project(np.zeros((1, 3)))[0].tolist() == [639.5, 511.5],
      "an aniposelib-style file (trailing commas): the world origin projects onto the principal point")
blend = os.path.join(OUT, "cameras_blender.py")
calibio.write_blender(ms, blend)
src_b = open(blend, encoding="utf-8").read()
compile(src_b, blend, "exec")
ns = {}
exec(src_b.split("scene = bpy.context.scene")[0].replace("import bpy", "").replace(
    "from mathutils import Matrix", ""), ns)
c0 = ns["CAMERAS"][0]
check(abs(c0["lens"] - truth[0]["K"][0, 0] * 36.0 / CW) < 1e-9 and abs(c0["shift_x"]) < 1e-12
      and abs(c0["shift_y"]) < 1e-12, "Blender: lens from fx, no shift for a centred principal point ((w - 1) / 2)")
Mw = np.array(c0["matrix"])
check(np.allclose(Mw[:3, 3], ms.cameras[0].center()) and np.allclose(Mw[:3, 2], -ms.cameras[0].R[2]),
      "Blender: the camera sits at its centre and looks down its -Z along the optical axis")

print("\n[2c] lens profiles, camera offsets, 3D points")
from kinetrace.lens import LensProfile  # noqa: E402
prof = LensProfile(CW, CH, truth[0]["K"], truth[0]["dist"], False, 0.31, 40, "test")
for fname in ("lens.yml", "lens.json", "lens.txt"):
    p = os.path.join(OUT, fname)
    calibio.write_lens(prof, p)
    back = calibio.read_lens(p)
    check(np.allclose(back.K, prof.K, atol=1e-9) and np.allclose(back.dist, prof.dist, atol=1e-12)
          and (back.width, back.height) == (CW, CH), f"lens profile as {fname}: read back exactly")
try:
    calibio.write_lens(LensProfile(CW, CH, truth[0]["K"], np.zeros(4), True), os.path.join(OUT, "fish.txt"))
    check(False, "a fisheye lens cannot be written as an Argus line")
except calibio.CalibFormatError as e:
    check("fisheye" in str(e), "a fisheye lens cannot be written as an Argus line (said)")
pj = Project([session(name="o1.mp4"), session(name="o2.mp4"), session(name="o3.mp4")],
             ["cam A", "cam B", "cam C"], [0.0, -4.5, 12.25])
p = os.path.join(OUT, "offsets.csv")
calibio.write_offsets(pj, p)
got_off = calibio.read_offsets(p, pj)
check([(v, o) for v, o, _ in got_off] == [(0, 0.0), (1, -4.5), (2, 12.25)], "camera offsets: written and read by name")
from kinetrace.calib import Reconstruction  # noqa: E402
Tn, Nn = 30, 2
xyz3 = np.random.default_rng(3).uniform(-1, 1, (Tn, Nn, 3))
xyz3[5, 1] = np.nan
rec = Reconstruction(10, ["snout", "tail base"], xyz3, np.full((Tn, Nn), 0.4), np.full((Tn, Nn), 3, np.int32), "m")
for kind in ("kinetrace", "dltdv", "anipose"):
    p = os.path.join(OUT, f"points_{kind}.csv")
    calibio.write_points3d(rec, p, kind)
    back, notes = calibio.read_points3d(p)
    off = rec.t0 - back.t0
    got = back.xyz[off:off + Tn] if back.t0 <= rec.t0 else None
    check(got is not None and back.names == rec.names and np.allclose(got, xyz3, atol=1e-6, equal_nan=True),
          f"3D points as {kind}: frames, names and positions come back")
p = os.path.join(OUT, "points_mirrored.csv")
calibio.write_points3d(rec, p, "kinetrace", models=ms_lh)
back, _ = calibio.read_points3d(p)
check(np.allclose(back.xyz[:, :, 2], -xyz3[:, :, 2], equal_nan=True, atol=1e-6),
      "3D points written beside mirrored cameras are mirrored the same way")
gone = os.path.join(OUT, "no_such_file")
said_missing = []
for label, fn in (("3D points", lambda: calibio.read_points3d(gone + ".csv")),
                  ("offsets", lambda: calibio.read_offsets(gone + ".csv", pj)),
                  ("Anipose", lambda: calibio.read_cameras(gone + ".toml")),
                  ("OpenCV", lambda: calibio.read_cameras(gone + ".yml")),
                  ("MATLAB", lambda: calibio.read_cameras(gone + ".mat"))):
    try:
        fn()
        said_missing.append(f"{label}: no error")
    except calibio.CalibFormatError as e:
        if "no_such_file" not in str(e):
            said_missing.append(f"{label}: {e}")
    except Exception as e:  # noqa: BLE001
        said_missing.append(f"{label}: {type(e).__name__}")
check(not said_missing, "a missing file is refused with its name, in every reader (not a crash)", str(said_missing))

print("\n[3] through the app: File -> Import Tracks")
import cv2  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

said = {"warn": [], "info": []}
QMessageBox.warning = staticmethod(lambda *a, **k: (said["warn"].append(str(a[2]) if len(a) > 2 else ""),
                                                    QMessageBox.Ok)[1])
QMessageBox.information = staticmethod(lambda *a, **k: (said["info"].append(str(a[2]) if len(a) > 2 else ""),
                                                        QMessageBox.Ok)[1])
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
vid = os.path.join(OUT, "v.mp4")
vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
for f in range(T):
    fr = np.full((H, W, 3), 40, np.uint8)
    cv2.circle(fr, (int(xy[f, 0]), int(xy[f, 1])), 6, (255, 255, 255), -1)
    vw.write(fr)
vw.release()
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (vid, ""))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
forget_recovery(vid)


def pump(sec):
    import time
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.005)


win = MainWindow()
win.show()
dlc = os.path.join(OUT, "dlc_roundtrip.csv")
win._import_tracks_dialog(dlc)                 # no video yet: asks for it first, then imports
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.session is not None and win.session.n_points == 3:
        break
check(said["info"] and "DeepLabCut" in said["info"][0], "with no video open it names the file's format and asks "
      "for the video", str(said["info"]))
check(win.session is not None and win.session.n_points == 3, "the video opens and the tracks arrive")
if win.session is not None and win.session.n_points == 3:
    d = np.abs(win.session.tracks[:, :3] - src.tracks)[exp].max()
    check(np.array_equal(win.session.tracked, exp) and d <= 5e-4 + 1e-4, "at the right frames and positions", str(d))
    win._undo_run()
    check(win.session.n_points == 0, "Ctrl+Z undoes the import")
win._import_tracks_dialog(os.path.join(OUT, "dlc_multi.csv"))
check(said["warn"] and "multi-animal" in said["warn"][-1] and win.session.n_points == 0,
      "a refused file says why and changes nothing")
QInputDialog.getItem = staticmethod(lambda *a, **k: ("camera 2 of the file", True))
win._import_tracks_dialog(os.path.join(OUT, "multi_xypts.csv"))
pump(0.2)
# the file's rows are camera A's frames; camera B's frame k sits in row k + 5
got = win.session
ok = got.n_points == 2 and np.array_equal(got.tracked[5:, :2], b.exportable[:T - 5]) \
    and np.abs(got.tracks[5:, :2] - b.tracks[:T - 5])[b.exportable[:T - 5]].max() < 2e-4
check(ok, "an all-cameras file into a one-camera project: the camera asked for is imported, row by row")
# silhouettes: mask images in (File -> Import -> Silhouettes), polygons / PNGs out
import json  # noqa: E402
mdir = os.path.join(OUT, "masks_in")
os.makedirs(mdir, exist_ok=True)
truth_masks = {}
for f in (5, 6, 40):
    mk = np.zeros((H, W), np.uint8)
    cv2.ellipse(mk, (int(xy[f, 0]), int(xy[f, 1])), (60, 35), 20, 0, 360, 255, -1)
    truth_masks[f] = mk > 0
    cv2.imwrite(os.path.join(mdir, f"mask_{f:06d}.png"), mk)
cv2.imwrite(os.path.join(mdir, "notes.txt.png"), np.zeros((4, 4), np.uint8))   # no frame number: ignored
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: mdir)
win._import_masks()
m = win.session.masks
check(m is not None and m.n_masked() == 3 and win.session.animal is not None,
      "three mask images become the camera's segment on their frames")
iou = min((m.rasterize(f, H, W) & t).sum() / (m.rasterize(f, H, W) | t).sum() for f, t in truth_masks.items())
check(iou > 0.98, "the stored outlines match the images", f"IoU {iou:.3f}")
sj = os.path.join(OUT, "sil.json")
win._export_one("sil_json", sj)
dj = json.load(open(sj, encoding="utf-8"))
check(sorted(dj["frames"]) == ["40", "5", "6"] and dj["width"] == W, "silhouettes out as polygons per frame (JSON)")
win._export_one("sil_png", os.path.join(OUT, "sil.png"))
outs = sorted(os.listdir(os.path.join(OUT, "sil_masks")))
back_png = cv2.imread(os.path.join(OUT, "sil_masks", outs[-1]), cv2.IMREAD_GRAYSCALE) > 0 if outs else None
check(len(outs) == 3 and back_png is not None and (back_png & truth_masks[40]).sum() / (back_png | truth_masks[40]).sum()
      > 0.98, "silhouettes out as one PNG per frame", str(outs))
check(win._undo_snap is None, "a mask import that CREATES the segment is not an undo step (the message says how "
      "to remove it)")
more = os.path.join(OUT, "masks_more")
os.makedirs(more, exist_ok=True)
cv2.imwrite(os.path.join(more, "mask_000070.png"), truth_masks[40].astype(np.uint8) * 255)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: more)
win._import_masks()
check(win.session.masks.n_masked() == 4, "a second folder adds to the segment")
win._undo_run()
check(win.session.masks.n_masked() == 3, "Ctrl+Z undoes an import onto an existing segment")
bad = os.path.join(OUT, "masks_bad")
os.makedirs(bad, exist_ok=True)
cv2.imwrite(os.path.join(bad, "m_000001.png"), np.zeros((10, 10), np.uint8))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: bad)
said["warn"] = []
win._import_masks()
check(said["warn"] and "size of the video" in said["warn"][-1], "masks of another size are refused with the reason")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
win.close()
pump(0.3)
forget_recovery(vid)

# ------------------------------------------------------------------ 4
print("\n[4] the command-line converter (python -m kinetrace.convert), run as a program")
import subprocess  # noqa: E402


def convert(*args):
    r = subprocess.run([sys.executable, "-m", "kinetrace.convert", *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr).strip()


kc = os.path.join(OUT, "synth.kcal.json")
import json as _json  # noqa: E402
with open(kc, "w", encoding="utf-8") as fh:
    _json.dump(calibio.calibration_to_kcal(cal), fh)
code, out = convert("calibration", kc, os.path.join(OUT, "cli_calibration.toml"))
back = calibio.read_cameras(os.path.join(OUT, "cli_calibration.toml"))
check(code == 0 and len(back) == 3 and max(np.abs(b_.K - c["K"]).max() for b_, c in zip(back, truth)) < 1e-6,
      "calibration: a .kcal.json into Anipose's calibration.toml", out)
dltc = os.path.join(OUT, "cli_dltCoefs.csv")
code, out = convert("calibration", kc, dltc)
check(code == 0 and os.path.exists(dltc), "calibration: into a dltCoefs.csv (MATLAB pixels)", out)
code, out = convert("calibration", dltc, os.path.join(OUT, "cli_cams.yml"))
check(code == 2 and "--size" in out, "a dltCoefs.csv without --size is refused with what to add", out)
code, out = convert("calibration", dltc, os.path.join(OUT, "cli_cams.yml"), "--size", f"{CW}x{CH}")
yb = calibio.read_cameras(os.path.join(OUT, "cli_cams.yml")) if code == 0 else []
ok_px = False
if yb:
    Xw = np.random.default_rng(9).uniform(-0.3, 0.3, (50, 3))
    ref_c = calibio.load_calibration(dltc)
    for c_ in ref_c.cameras:
        c_.width, c_.height = CW, CH
    ms_ = calibio.to_models(ref_c)
    ok_px = max(float(np.abs(m_.project(Xw) - y_.project(Xw)).max()) for m_, y_ in zip(ms_.cameras, yb)) < 1e-6
check(code == 0 and ok_px, "... and with --size it becomes OpenCV cameras (the 1-based pixels converted)", out)
code, out = convert("tracks", os.path.join(OUT, "dlc_roundtrip.csv"), os.path.join(OUT, "cli_xypts.csv"),
                    "--to", "dltdv", "--size", f"{W}x{H}", "--frames", T)
s, imp, summ = imported_into(os.path.join(OUT, "cli_xypts.csv")) if code == 0 else (None, None, None)
check(code == 0 and s is not None and np.array_equal(s.tracked[:, :3], exp), "tracks: DeepLabCut CSV into DLTdv8 xypts",
      out)
code, out = convert("tracks", os.path.join(OUT, "sleap14.csv"), os.path.join(OUT, "cli_sleap.kinetrace"),
                    "--video", vid)
check(code == 0 and projectfile.load(os.path.join(OUT, "cli_sleap.kinetrace")).sessions[0].n_points == 2,
      "tracks: a SLEAP CSV into a Kinetrace project for its video", out)
code, out = convert("points3d", os.path.join(OUT, "points_kinetrace.csv"), os.path.join(OUT, "cli_p3.csv"),
                    "--to", "anipose")
back3, _ = calibio.read_points3d(os.path.join(OUT, "cli_p3.csv")) if code == 0 else (None, None)
check(code == 0 and back3 is not None and back3.names == rec.names, "points3d: Kinetrace CSV into Anipose's", out)
# a project to work on: the source session, saved for the converter
kp2 = os.path.join(OUT, "cli_project.kinetrace")
src_copy = session(name="v.mp4")
trackio.apply(src_copy, trackio.read(os.path.join(OUT, "dlc_roundtrip.csv")))
trackio.import_masks_png(src_copy, mdir)
projectfile.save(Project([src_copy], ["top"]), kp2)
code, out = convert("check", kp2)
check(code == 0 and out.endswith("OK"), "check: a sound project is OK (exit 0)", out)
code, out = convert("info", kp2)
check(code == 0 and "'top'" in out and "3 point(s)" in out, "info: says what the project holds", out)
moved = os.path.join(OUT, "elsewhere")
os.makedirs(moved, exist_ok=True)
shutil.copy(kp2, os.path.join(moved, "p.kinetrace"))
code, out = convert("check", os.path.join(moved, "p.kinetrace"))
check(code == 0, "check: a copy in another folder still finds its video (by the recorded absolute path)", out)
os.rename(vid, vid + ".away")
code, out = convert("check", os.path.join(moved, "p.kinetrace"))
os.rename(vid + ".away", vid)
check(code == 1 and "video is not found" in out, "check: a video that cannot be found is a warning (exit 1)", out)
bad_zip = write("broken.kinetrace", "not a zip at all")
code, out = convert("check", bad_zip)
check(code == 2 and out.startswith("error:"), "check: a damaged file is an error (exit 2), said in a sentence", out)
code, out = convert("import", kp2, "--tracks", os.path.join(OUT, "sleap14.csv"), "--out",
                    os.path.join(OUT, "cli_imported.kinetrace"))
got_p = projectfile.load(os.path.join(OUT, "cli_imported.kinetrace")) if code == 0 else None
check(got_p is not None and got_p.sessions[0].n_points == 5 and os.path.exists(kp2),
      "import: SLEAP tracks into a project, written as a new file", out)
code, out = convert("import", kp2, "--tracks", os.path.join(OUT, "sleap14.csv"))
check(code == 0 and os.path.exists(kp2 + ".bak") and projectfile.load(kp2).sessions[0].n_points == 5,
      "import: into the project itself, the previous version kept as .bak", out)
folder = os.path.join(OUT, "unzipped")
with zipfile.ZipFile(kp2) as z_:
    z_.extractall(folder)
code, out = convert("pack", folder, os.path.join(OUT, "cli_packed.kinetrace"))
pk = projectfile.load(os.path.join(OUT, "cli_packed.kinetrace")) if code == 0 else None
ref_p = projectfile.load(kp2)
check(pk is not None and np.array_equal(pk.sessions[0].tracks, ref_p.sessions[0].tracks, equal_nan=True),
      "pack: an unzipped (hand-edited) folder back into one file, unchanged", out)
code, out = convert("masks", kp2, os.path.join(OUT, "cli_masks.json"))
check(code == 0 and len(_json.load(open(os.path.join(OUT, "cli_masks.json"), encoding="utf-8"))["frames"]) == 3,
      "masks: a project's silhouettes as polygons", out)
pjo = os.path.join(OUT, "cli_offsets_project.kinetrace")
projectfile.save(pj, pjo)
code, out = convert("offsets", pjo, os.path.join(OUT, "cli_offsets.csv"))
check(code == 0 and [(v, o_) for v, o_, _ in calibio.read_offsets(os.path.join(OUT, "cli_offsets.csv"), pj)]
      == [(0, 0.0), (1, -4.5), (2, 12.25)], "offsets: a project's camera offsets as CSV", out)
code, out = convert("tracks", os.path.join(OUT, "dlc_multi.csv"), os.path.join(OUT, "x.csv"), "--size", "640x480",
                    "--frames", "10")
check(code == 2 and "multi-animal" in out, "a refused file: exit 2 and the reason", out)
# docs/FORMAT.md's "Writing a project for Kinetrace" example, exactly as printed there
mine = os.path.join(OUT, "myproject")
os.makedirs(os.path.join(mine, "cameras", "cam1"), exist_ok=True)
write(os.path.join("myproject", "kinetrace.json"), '{"format": "kinetrace-project", "format_version": 1,\n'
      ' "cameras": [{"folder": "cam1", "name": "cam1"}]}\n')
write(os.path.join("myproject", "project.json"), '{"cameras": [{"name": "cam1", "folder": "cam1",\n'
      ' "video": {"relative_path": "../v.mp4"},\n "n_frames": 120, "fps": 30, "width": 640, "height": 480}]}\n')
write(os.path.join("myproject", "cameras", "cam1", "tracks.csv"), "frame,point,x,y\n0,snout,120.5,88.25\n"
      "1,snout,121.0,88.0\n")
fmt_doc = open(os.path.join(ROOT, "docs", "FORMAT.md"), encoding="utf-8").read()
check('"video": {"relative_path": "../clip.mp4"}' in fmt_doc and "0,snout,120.5,88.25" in fmt_doc,
      "docs/FORMAT.md still prints this example (tested here with this suite's video instead of clip.mp4)")
code, out = convert("check", mine)
check(code == 0, "FORMAT.md's three-file project checks clean", out)
code, out = convert("pack", mine, os.path.join(OUT, "myproject.kinetrace"))
mp = projectfile.load(os.path.join(OUT, "myproject.kinetrace")) if code == 0 else None
check(mp is not None and mp.sessions[0].points[0].name == "snout"
      and np.allclose(mp.sessions[0].tracks[0, 0], (120.5, 88.25)) and mp.sessions[0].confidence[1, 0] == 1.0,
      "... and packs into a .kinetrace with its point and positions", out)
w2 = MainWindow()
w2.show()
w2._open_project_from_path(os.path.join(mine, "kinetrace.json"))   # File -> Open Project on the folder's file
for _ in range(300):
    pump(0.05)
    if w2.state == READY and w2.project is not None:
        break
check(w2.project is not None and w2.session.n_points == 1 and w2.session.tracked[1, 0],
      "the unzipped folder also opens in the app (choosing its kinetrace.json)")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
w2.close()
pump(0.3)

print("\n" + "=" * 62)
if fails:
    print(f"verify_interop FAILED: {len(fails)}")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("verify_interop PASSED")
