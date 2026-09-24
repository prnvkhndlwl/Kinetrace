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
[2] File -> Import Tracks through the app (offscreen): asks for the video
    first, Ctrl+Z, a refusal, one camera out of an all-cameras file.

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

print("\n[2] through the app: File -> Import Tracks")
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
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
win.close()
pump(0.3)
forget_recovery(vid)

print("\n" + "=" * 62)
if fails:
    print(f"verify_interop FAILED: {len(fails)}")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("verify_interop PASSED")
