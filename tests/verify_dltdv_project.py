"""File > Import > DLTdv Project (I310, G211): a DLTdv8 *_dvProject.mat becomes a Kinetrace project.

Offscreen; no GPU. The synthetic project (tests/_synth_dvproject.py) has KNOWN answers: three cameras (ratios
1, 1, 2; DLTdv offsets 0, -4, 3), two points, camera 1's LWM lens (v7.3 copy only), DLTdv's 3D = the truth, and
camera 2's video drawn 9 frames later than the project says. Every import is made by the user's gesture
(Alt+F, I, D) with the file dialogs and the two import dialogs stubbed. Checked:
  (a) the MATLAB v7 and v7.3 copies read the same arrays;
  (b) cameras, names, rates [1, 1, 2], offsets by DLTdv's rule, offsets[0] == 0;
  (c) every click at camera c's frame F_c(r) = ratio*(r+1) + offset - 1, pixels (x - 1, y - 1), as hand clicks;
  (d) the calibration is internal (0-based) with camera 1's LWM lens;
  (e) the frame check offers +9 on camera 2 only; taking it moves camera 2's clicks AND offset by 9, the 3D is
      the same array either way;
  (f) the import report's 3D check is good against DLTdv's stored 3D;
  (g) refusals in words: not a DLTdv project, camera 1's video missing, a camera of another picture size;
  (h) File > Import > Calibration reads the v7.3 copy (it failed: "Please use HDF reader ...");
  (i) nothing is written beside the .mat.
"""
import os
import re
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

import _synth_dvproject as S

OUT = os.path.join(ROOT, "tests", "out", "dvsynth")
MISSING = os.path.join(ROOT, "tests", "out", "dvsynth_missing")
GOPRO = os.path.join(ROOT, "tests", "out", "dvsynth_gopro")       # cam1/clip.mp4, cam2/clip.mp4, cam3/clip.mp4
ONEFILE = os.path.join(ROOT, "tests", "out", "dvsynth_onefile")   # only cam1/clip.mp4 is here
for d in (OUT, MISSING, GOPRO, ONEFILE):
    if os.path.isdir(d):
        shutil.rmtree(d)
B = S.build(OUT)
os.makedirs(MISSING, exist_ok=True)
NO_VIDEOS = S.write_v7(os.path.join(MISSING, "lost_dvProject.mat"))
WRONG_SIZE = S.write_v7(os.path.join(OUT, "size_dvProject.mat"), sizes={1: (1280, 720)})
RIG = [f"X:/rig/cam{c + 1}/clip.mp4" for c in range(3)]          # a GoPro rig: one file name in every camera folder
for c in range(3):
    os.makedirs(os.path.join(GOPRO, f"cam{c + 1}"))
    shutil.copyfile(B["videos"][c], os.path.join(GOPRO, f"cam{c + 1}", "clip.mp4"))
os.makedirs(os.path.join(ONEFILE, "cam1"))
shutil.copyfile(B["videos"][0], os.path.join(ONEFILE, "cam1", "clip.mp4"))
GOPRO_DV = S.write_v7(os.path.join(GOPRO, "rig_dvProject.mat"), files=RIG)
ONEFILE_DV = S.write_v7(os.path.join(ONEFILE, "rig_dvProject.mat"), files=RIG)
# two names that differ only in capitals, and camera 1 never clicked the first of them
NAMES_DV = S.write_v7(os.path.join(OUT, "names_dvProject.mat"), names=["Head", "head"], drop=((0, 0),))
NOT_DV = os.path.join(OUT, "other.mat")
from scipy.io import savemat  # noqa: E402
savemat(NOT_DV, {"something": np.eye(3)})

said: list[str] = []                       # every message box the app showed, title + text


def _box(answer):
    def f(parent, title, text, *a, **k):
        said.append(f"{title}: {text}")
        return answer
    return staticmethod(f)


_answer = {"question": QMessageBox.Yes}
QMessageBox.question = staticmethod(lambda parent, title, text, *a, **k: (said.append(f"{title}: {text}"),
                                                                           _answer["question"])[1])
QMessageBox.warning = _box(QMessageBox.Ok)
QMessageBox.information = _box(QMessageBox.Ok)
QMessageBox.critical = _box(QMessageBox.Ok)
_pick = {"file": "", "dir": ""}
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (_pick["file"], ""))
_asked = {"dir": 0}
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: (_asked.__setitem__("dir", _asked["dir"] + 1),
                                                                  _pick["dir"])[1])

app = QApplication([])
from _clean import forget_recovery  # noqa: E402
forget_recovery(*B["videos"])
from kinetrace.app import MainWindow, READY  # noqa: E402

FAILS: list[str] = []


def check(ok, what, got=None):
    print(("  ok   " if ok else "  FAIL ") + what + ("" if ok or got is None else f"  (got {got!r})"))
    if not ok:
        FAILS.append(what)


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(cond, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    check(False, what + " (timed out)")
    return False


win = MainWindow()
win.show()
app.setActiveWindow(win)
pump(0.3)

# ---- the feature exists where the user looks: File > Import > DLTdv Project...
check(hasattr(win, "act_import_dltdv"), "File > Import has a DLTdv Project entry")
if not hasattr(win, "act_import_dltdv"):
    print("FAILED:", FAILS)
    sys.exit(1)
from kinetrace import clickshift, dltdvimport, dltdvproject  # noqa: E402
from kinetrace.calib import LWMUndistort, NoUndistort, reconstruct  # noqa: E402

seen = {"check": [], "report": []}
_offer = {"take": True}


def _frame_exec(self):
    seen["check"].append(self)
    if not _offer["take"]:
        for box in self.use_boxes:
            if box is not None:
                box.setChecked(False)
    return QDialog.Accepted


dltdvimport.ClickShiftDialog.exec = _frame_exec
dltdvimport.ImportReportDialog.exec = lambda self: (seen["report"].append(self), QDialog.Accepted)[1]


def import_by_gesture(path, folder="", timeout=120):
    """Alt+F (File), I (Import), D (DLTdv Project...) with the file dialog answering `path`."""
    _pick["file"], _pick["dir"] = path, folder
    n_rep = len(seen["report"])
    QTest.keyClick(win, Qt.Key_F, Qt.AltModifier)
    pump(0.2)
    m = QApplication.activePopupWidget()
    check(m is not None, "Alt+F opens the File menu")
    QTest.keyClick(m, Qt.Key_I)
    pump(0.2)
    sub = QApplication.activePopupWidget()
    check(sub is not None and sub is not m, "I opens File > Import")
    n_said = len(said)
    QTest.keyClick(sub, Qt.Key_D)
    wait(lambda: len(seen["report"]) > n_rep or len(said) > n_said, timeout,
         "the import ends in its report or in a message")
    pump(0.3)


def clicks_at(p, v):
    """{(frame, point name): (x, y)} of camera v's hand clicks."""
    s = p.sessions[v]
    out = {}
    for i, pt in enumerate(s.points):
        fr = np.flatnonzero(s.tracked[:, i])
        for f in fr:
            out[(int(f), pt.name)] = (float(s.tracks[f, i, 0]), float(s.tracks[f, i, 1]))
    return out


def expected(v, shift=0):
    xy = S.clicks()
    out = {}
    for r in range(xy.shape[0]):
        for k, nm in enumerate(S.POINTS):
            if np.isfinite(xy[r, k, v, 0]):
                out[(int(S.local_frame(v, r, shift)), nm)] = (xy[r, k, v, 0] - 1.0, xy[r, k, v, 1] - 1.0)
    return out


# ---- (a) the two MATLAB forms read the same
d7, d73 = dltdvproject.read(B["v7"]), dltdvproject.read(B["v73"])
same = all(np.array_equal(getattr(d7, f), getattr(d73, f), equal_nan=True)
           for f in ("xy", "xyz", "res", "dltcoef", "sizes", "offsets", "ratios", "frame_rates"))
check(same and d7.point_names == d73.point_names == S.POINTS and d7.video_files == d73.video_files == S.VIDEOS,
      "(a) the v7 and v7.3 copies read the same arrays, names and video files")
check(isinstance(d73.lenses[0], LWMUndistort) and isinstance(d73.lenses[1], NoUndistort)
      and all(isinstance(x, NoUndistort) for x in d7.lenses), "(a) camera 1's lens only in the v7.3 copy")

# ---- (h) File > Import > Calibration now reads a v7.3 dvProject
from kinetrace import calibio  # noqa: E402
try:
    cal73 = calibio.load_calibration(B["v73"])
    check(len(cal73.cameras) == 3 and isinstance(cal73.cameras[0].undistort, LWMUndistort)
          and cal73.cameras[0].pixel_origin == 1.0, "(h) a v7.3 dvProject's calibration loads with its lens")
except Exception as e:  # noqa: BLE001
    check(False, "(h) a v7.3 dvProject's calibration loads", f"{type(e).__name__}: {e}")

# ---- the frame check on its own: +9 on camera 2, nothing on 1 and 3
for v in range(3):
    fit = clickshift.check_camera(B["videos"][v], *d73.check_inputs(v), S.FPS[v])
    want = S.PLANTED[v]
    check(fit.best == want and (fit.offer == (want or None)),
          f"frame check camera {v + 1}: best {want:+d}, offer {'+9' if want else 'none'} ({fit.verdict})",
          (fit.best, fit.offer, fit.verdict))

# ---- (b)-(f), (i): the v7.3 copy, offer taken
listing = sorted(os.listdir(os.path.dirname(B["v73"])))
import_by_gesture(B["v73"])
p = win.project
check(p is not None and p.n_views == 3 and win.state == READY, "(b) three cameras open",
      None if p is None else p.n_views)
if p is not None and p.n_views == 3:
    check([os.path.basename(s.video_path) for s in p.sessions] == S.VIDEOS, "(b) the videos found in the parent folder")
    check(np.allclose(p.rates, [1, 1, 2]), "(b) rates 1, 1, 2", p.rates)
    check(p.offsets[0] == 0 and np.allclose(p.offsets, [0, -4 + 9, 4]), "(b) offsets 0, 5 (= -4 + the taken +9), 4",
          p.offsets)
    for v in range(3):
        got, want = clicks_at(p, v), expected(v, S.PLANTED[v])
        check(got.keys() == want.keys() and all(np.allclose(got[k], want[k], atol=1e-4) for k in want),
              f"(c) camera {v + 1}: every click at F(r) + its shift, pixels - 1 ({len(want)} clicks)",
              (len(got), len(want)))
        s = p.sessions[v]
        check(bool(np.all(s.manual[s.tracked])) and bool(np.all(s.confidence[s.tracked] == 1.0)),
              f"(c) camera {v + 1}: clicks are hand clicks")
        rows = np.flatnonzero(np.isfinite(S.clicks()[:, :, v, 0]).any(axis=1))
        check(all(p.local_index(v, S.local_frame(0, r, 0)) == S.local_frame(v, r, S.PLANTED[v]) for r in rows),
              f"(c) camera {v + 1}: local_index(F_1(r)) == F_c(r) for every clicked row")
    cal = p.calibration
    check(cal is not None and cal.is_internal() and isinstance(cal.camera(0).undistort, LWMUndistort),
          "(d) the calibration is internal, camera 1 keeps its LWM lens")
    check(len(seen["check"]) == 1, "(e) the frame-check dialog was shown once")
    if seen["check"]:
        fits = seen["check"][-1].fits
        check([f.offer for f in fits] == [None, 9, None], "(e) it offered +9 on camera 2 only",
              [f.offer for f in fits])
    check(len(seen["report"]) == 1, "(f) the import report was shown")
    if seen["report"]:
        txt = seen["report"][-1].text
        check(re.search(r"3D check: good", txt) is not None, "(f) the report's 3D check is good", txt[:400])
        check("hand check" in txt.lower(), "(f) the report gives a hand check")
        print("  --- the report ---\n" + txt.encode("ascii", "replace").decode() + "\n  ---")
    check(p.dirty and win.project_path is None, "the new project is unsaved (Save asks where)")
    taken = p
    rec_taken = reconstruct(p.sessions, p.calibration, p.rates, p.offsets, (0, S.N_REF - 1))
check(sorted(os.listdir(os.path.dirname(B["v73"]))) == listing, "(i) nothing written beside the .mat")

# ---- (e) declining the offer: camera 2 keeps DLTdv's frames, the 3D is the same array
_offer["take"] = False
import_by_gesture(B["v73"])
p = win.project
if p is not None and p.n_views == 3:
    check(np.allclose(p.offsets, [0, -4, 4]), "(e) declined: offsets as the project says", p.offsets)
    got, want = clicks_at(p, 1), expected(1, 0)
    check(got.keys() == want.keys(), "(e) declined: camera 2's clicks where DLTdv put them")
    rec = reconstruct(p.sessions, p.calibration, p.rates, p.offsets, (0, S.N_REF - 1))
    check(np.array_equal(rec.xyz, rec_taken.xyz, equal_nan=True),
          "(e) the 3D is the same array whether the shift is taken or not (only the pictures move)")
_offer["take"] = True

# ---- the v7 copy through the menu: the same clicks; no lens (scipy cannot write one, so the file has none)
import_by_gesture(B["v7"])
p = win.project
if p is not None and p.n_views == 3:
    check(all(clicks_at(p, v).keys() == expected(v, S.PLANTED[v]).keys() for v in range(3)),
          "(b) v7 copy: every camera's clicks at the same frames as the v7.3 copy's")
    check(isinstance(p.calibration.camera(0).undistort, NoUndistort), "(d) v7 copy: no lens (none in the file)")

# ---- (j) a GoPro rig: the same file name in every camera's folder, none beside the .mat: each camera finds ITS
# file in the sub-folder named like its stored folder, without asking (it asked, then refused camera 1)
_asked["dir"] = 0
import_by_gesture(GOPRO_DV)
p = win.project
check(_asked["dir"] == 0, "(j) a GoPro rig's videos are found in their camera folders without asking", _asked["dir"])
check(p is not None and [os.path.relpath(s.video_path, GOPRO) for s in p.sessions]
      == [os.path.join(f"cam{c + 1}", "clip.mp4") for c in range(3)],
      "(j) each camera gets its own clip.mp4", None if p is None else [s.video_path for s in p.sessions])

# ---- (k) only camera 1's clip.mp4 exists and the user points at its folder: camera 2 / 3 must NOT be given the same
# file (it built a half project after replacing the one on screen); No to "import without them" changes nothing
prev, n0 = win.project, len(said)
_answer["question"] = QMessageBox.No
import_by_gesture(ONEFILE_DV, folder=os.path.join(ONEFILE, "cam1"))          # the user picks camera 1's folder
_answer["question"] = QMessageBox.Yes
check(any("video 2" in m and "not found" in m for m in said[n0:]),
      "(k) cameras 2 and 3 are named as not found (never camera 1's file again)", said[n0:])
check(win.project is prev, "(k) ... and No leaves the project on screen as it was")

# ---- (l) "Head" and "head", camera 1 never clicked "Head": every camera has both points in DLTdv's order and the
# same name means the same DLTdv point in every camera (they were paired the wrong way round)
import_by_gesture(NAMES_DV)
p = win.project
if p is not None and p.n_views == 3:
    check(all([pt.name for pt in s.points] == ["Head", "head (2)"] for s in p.sessions),
          "(l) every camera: Head, head (2), in DLTdv's order", [[pt.name for pt in s.points] for s in p.sessions])
    xy = S.clicks()
    s0 = p.sessions[0]
    k2 = [pt.name for pt in s0.points].index("head (2)")
    want = {int(S.local_frame(0, r)) for r in range(xy.shape[0]) if np.isfinite(xy[r, 1, 0, 0])}
    check(set(np.flatnonzero(s0.tracked[:, k2]).tolist()) == want and not s0.tracked[:, 1 - k2].any(),
          "(l) camera 1: its clicks are on 'head (2)' (DLTdv's 2nd point), none on 'Head'")
    if seen["report"]:
        check("3D check: good" in seen["report"][-1].text, "(l) the 3D check is good (points paired right)",
              seen["report"][-1].text[-600:])

# ---- (n) I313: DLTdv undistorts clicks with camud (camera 2: a profile .txt's 3-iteration Bouguet model, camera 3:
# the GoPro fisheye), never with the LWM it stores. Kinetrace ran that LWM backwards: ~1e-4 of the volume off
# DLTdv's own 3D. Each camera must get DLTdv's model back from the control points and the 3D must be DLTdv's.
import_by_gesture(B["lens"])
p = win.project
if p is not None and p.n_views == 3:
    und = [p.calibration.camera(v).undistort for v in range(3)]
    check([getattr(u, "kind", "none") for u in und] == ["none", "dltdv", "dltdv"]
          and [getattr(u, "model", None) for u in und] == [None, "brown3", "ocam"],
          "(n) cameras 2 and 3 undistort with DLTdv's own model (brown3, ocam)",
          [(getattr(u, "kind", "none"), getattr(u, "model", None)) for u in und])
    dvl = dltdvproject.read(B["lens"])
    rec = reconstruct(p.sessions, p.calibration, p.rates, p.offsets, (0, S.N_REF - 1))
    cmp = dltdvproject.compare_3d(dvl, rec, [0, 1, 2], [0, 0, 0])
    check(cmp.get("n", 0) > 100 and cmp["rel"] <= 1e-9 and cmp.get("missing", 1) == 0,
          "(n) Kinetrace's 3D is DLTdv's stored 3D (median <= 1e-9 of the volume, none missing)",
          {k: cmp.get(k) for k in ("n", "median", "rel", "missing")})

# ---- (g) refusals, in words
n0, prev = len(said), win.project
import_by_gesture(NOT_DV)
check(any("not a DLTdv" in m for m in said[n0:]), "(g) a .mat that is not a DLTdv project is refused in words",
      said[n0:])
check(win.project is prev, "(g) ... and the project on screen is untouched")
n0 = len(said)
import_by_gesture(NO_VIDEOS, folder="")
check(any("cam1_side.mp4" in m and "not found" in m.lower() for m in said[n0:]),
      "(g) camera 1's video missing: refused, naming the file", said[n0:])
check(win.project is prev, "(g) ... and the project on screen is untouched")
n0 = len(said)
import_by_gesture(WRONG_SIZE)
p = win.project
check(any("1280" in m and "640" in m for m in said[n0:]) or
      (seen["report"] and "1280" in seen["report"][-1].text),
      "(g) a camera whose picture size differs is named with both sizes")
check(p is not None and p.n_views == 2, "(g) ... and left out (the user said yes)", None if p is None else p.n_views)

win._dev_probe.wait(60_000)
win.close()
app.processEvents()
forget_recovery(*B["videos"])
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("VERIFY_DLTDV_PROJECT PASSED")
