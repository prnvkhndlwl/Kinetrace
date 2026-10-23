"""Hand-placed frames are the truth (I320, I316, I317, I319, G217, G218; plan dev/work/2026-10-20-silhouette-hand-mix,
PR A). The owner: "hand clicked points are the truth. they never get replaced unless the user reclicks it or
deletes it".

Offscreen; no GPU (the runs use Moving spot, which needs no model, and a mock silhouette). Sections:
  (a) the session rules: `write_segment(keep_hand=True, own=...)` keeps every array of a hand cell and every
      cell outside its bounds, the default write is today's (and clears `imported` with `manual`);
      `clear_window(keep_hand=True)`; `set_source` keeps the hand frames both ways; `imported` is a point array.
  (b) imported clicks (D7): a DLTdv project imported by the user's gesture marks its clicks `imported`; a click
      and timeline Delete clear the mark, Ctrl+Z brings it back, Swap carries it; the saved CSV has an
      `imported` column; save / reopen and a recovery copy keep it; a recovery copy without `imported.npy`
      (written before the update) still opens.
  (c) Moving spot (I316, I317): a click ahead survives a Track press and the track restarts from it; a spot
      that vanishes and is clicked where it comes back is cut, not stopped; a point with no position on the
      start frame joins at its click; a spot that drifted onto a glint follows the real spot after the click
      (SpotRun.reseed's speed rule); clicks every third frame = one hand stretch, one re-seed.
  (d) silhouette landmarks take clicks (G218) and a silhouette run keeps them; the Data source switch keeps
      the hand frames and its question gives both counts.
  (e) the Track pop-up (G217, D9 choices 1-2): it asks when there is data ahead, with counts over every camera
      of the press; Fresh run = today's run, then the rest erased (never remembered); Keep my clicks; "Use
      this every time"; F never asks.
Every check FAILS on the code before the change for the reason it names (no `keep_hand`, no `imported`, no
re-seed, refusals, no pop-up). Run: .venv\\Scripts\\python.exe tests\\verify_hand_truth.py
"""
import csv
import glob
import os
import shutil
import sys
import time
import traceback
import zipfile

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "hand_truth")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

ASK = {"answer": QMessageBox.Yes, "asked": []}           # every question: (title, text)


def _question(parent, title, text, *a, **k):
    ASK["asked"].append((str(title), str(text)))
    return ASK["answer"]


QMessageBox.question = staticmethod(_question)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
PICK = {"file": "", "dir": "", "save": ""}
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (PICK["file"], ""))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: PICK["dir"])
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (PICK["save"], ""))

app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from _synth_spots import write_video  # noqa: E402

import kinetrace.app as appmod  # noqa: E402
from kinetrace import projectfile as pf  # noqa: E402
from kinetrace import recovery, segmenter, spots  # noqa: E402
from kinetrace import tracker as trk  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.session import POINT_ARRAYS, TrackingSession  # noqa: E402

FAILS: list = []


def check(label, cond, detail=None):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001 - the old code raises where the new one answers
            cond, detail = False, f"raised {e!r}"
    cond = bool(cond)
    line = ("  ok    " if cond else "  FAIL  ") + label + ("" if cond or detail is None else f"   [{detail}]")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not cond:
        FAILS.append(label)
    return cond


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.003)


def wait(cond, timeout):
    t0 = time.time()
    while True:
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > timeout:
            return False
        pump(0.02)


def section(fn):
    """Run one section; an exception is ONE failure (the old code lacks what the section drives)."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))


# ---- the Track pop-up (D9), stubbed: it records what it said and answers POP["pick"]
POP = {"seen": [], "pick": "keep", "remember": False, "fresh_box": []}
HAS_POPUP = hasattr(appmod, "TrackChoiceDialog")
if HAS_POPUP:
    def _pop_exec(self):
        POP["seen"].append(self.message())
        self.pick(POP["pick"])
        POP["fresh_box"].append((POP["pick"], self.remember_box.isEnabled()))
        if POP["remember"] and self.remember_box.isEnabled():
            self.remember_box.setChecked(True)
        return QDialog.Accepted
    appmod.TrackChoiceDialog.exec = _pop_exec

# ---- every run the app launches (the worker, for its re-seed log)
LAUNCHED: list = []
_orig_launch = MainWindow._launch_run


def _spy_launch(self, w, driver=None):
    LAUNCHED.append(w)
    return _orig_launch(self, w, driver)


MainWindow._launch_run = _spy_launch

# ================================================================= the clips (320 x 240 gray, bright spots)
W, H = 320, 240
_rng = np.random.RandomState(7)
BG = cv2.GaussianBlur((_rng.rand(H, W) * 90 + 40).astype(np.float32), (0, 0), 3)
YY, XX = np.mgrid[0:H, 0:W]


def spot_clip(name, centres):
    """A clip with a bright Gaussian spot at each centre array (n, 2); NaN = not drawn there."""
    n = len(centres[0])
    frames = []
    for f in range(n):
        img = BG.copy()
        for c in centres:
            x, y = c[f]
            if np.isfinite(x):
                img = img + 150.0 * np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / (2 * 2.0 ** 2))
        g = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(np.dstack([g, g, g]))
    path = os.path.join(OUT, name + ".mp4")
    write_video(path, frames, fps=30.0)
    return path


N1 = 80
LINE = np.stack([40.0 + 2.0 * np.arange(N1), np.full(N1, 120.0)], 1)       # 2 px / frame to the right
VID1 = spot_clip("line", [LINE])
GAP = LINE.copy()
GAP[45:52] = np.nan                                                        # hidden 45-51, back at 52
VID_GAP = spot_clip("gap", [GAP])
# the glint scene (I317): until frame 29 only G is there (the run follows it); from 30 the real spot A appears
# 20 px below G's last place and moves on at 2 px / frame while G stays where it was
HG = 30
G = np.stack([40.0 + 2.0 * np.arange(N1), np.full(N1, 100.0)], 1)
G[HG:] = G[HG - 1]
A = np.full((N1, 2), np.nan)
A[HG:] = np.stack([G[HG - 1, 0] + 2.0 * np.arange(1, N1 - HG + 1), np.full(N1 - HG, 120.0)], 1)
VID_GLINT = spot_clip("glint", [G, A])
N3 = 260
LONG = np.stack([20.0 + 1.0 * np.arange(N3), 120.0 + 20.0 * np.sin(np.arange(N3) / 30.0)], 1)
VID_LONG = spot_clip("long", [LONG])
VID_B = spot_clip("line_b", [LINE + np.array([0.0, 30.0])])               # a second camera
forget_recovery(VID1, VID_GAP, VID_GLINT, VID_LONG, VID_B)


# ================================================================= the window
def new_window(video):
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(video)
    ok = wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    pump(0.3)
    win._goto(0, force=True)
    wait(lambda: win.cache.get(0) is not None, 10)
    win.canvas.fit()
    pump(0.1)
    check(f"(setup) {os.path.basename(video)} opens", ok)
    return win


def close(win):
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


def canvas_click(win, x, y, mods=Qt.NoModifier):
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, mods, vp)
    pump(0.05)


def click_at(win, f, x, y):
    """Go to frame f and click the selected point there (an unarmed click = a hand placement)."""
    win._goto(f, force=True)
    pump(0.05)
    canvas_click(win, x, y)


def add_spot_point(win, gt, frames=(0, 1)):
    """N + a click on the first frame, then a click on the others (Moving spot reads the speed of 0 -> 1)."""
    win.act_pm_spot.trigger()
    f0 = frames[0]
    win._goto(f0, force=True)
    pump(0.05)
    win.btn_add.setChecked(True)
    canvas_click(win, *gt[f0])
    pid = win.session.n_points - 1
    for f in frames[1:]:
        click_at(win, f, *gt[f])
    return pid


def track_press(win, start=0, key=Qt.Key_T, mods=Qt.NoModifier, timeout=120):
    """Select all, go to `start`, press the key: True when a run was launched and has ended."""
    win.act_select_all.trigger()
    win._goto(start, force=True)
    pump(0.05)
    n0 = len(LAUNCHED)
    QTest.keyClick(win, key, mods)
    started = wait(lambda: len(LAUNCHED) > n0, 15)
    ended = wait(lambda: win.state == READY and win.worker is None and win._passes is None
                 and win._multi is None, timeout)
    pump(0.2)
    return started and ended


def hand(s, f, p):
    return bool(s.manual[f, p] and s.tracked[f, p])


# ================================================================= (a) the session rules
def section_a():
    print("(a) the session rules")
    s = TrackingSession(VID1, 40, 30.0, W, H)
    p = s.add_point(0, 10.0, 10.0)
    s.manual[0, p] = False                    # frame 0 = tracker output, the hand cells are 5 and 6
    for f in range(1, 40):
        s.tracks[f, p] = (10.0 + f, 10.0)
        s.tracked[f, p] = s.visibility[f, p] = True
        s.confidence[f, p] = 0.8
    s.set_position(5, p, 99.0, 99.0)
    s.set_position(6, p, 98.0, 98.0)
    before = {a.name: getattr(s, a.name).copy() for a in POINT_ARRAYS}
    tw = np.full((20, 1, 2), 50.0, np.float32)
    vw = np.ones((20, 1), bool)
    cw = np.full((20, 1), 0.7, np.float32)

    def kept():
        s.write_segment(0, tw, vw, [p], cw, keep_hand=True)
        return all(np.array_equal(getattr(s, a.name)[5:7, p], before[a.name][5:7, p], equal_nan=True)
                   for a in POINT_ARRAYS)
    check("(a) write_segment(keep_hand=True) leaves every array of a hand cell as it was", kept)
    check("(a) ... and writes the other rows of the window", lambda: np.allclose(s.tracks[0:5, p], 50.0)
          and np.allclose(s.tracks[7:20, p], 50.0) and not s.manual[7:20, p].any())
    for a in POINT_ARRAYS:
        getattr(s, a.name)[:] = before[a.name]

    def bounded():
        s.write_segment(0, tw, vw, [p], cw, keep_hand=True, own=[(0, 9)])
        return (np.allclose(s.tracks[7:10, p], 50.0)
                and np.array_equal(s.tracks[10:20, p], before["tracks"][10:20, p], equal_nan=True))
    check("(a) write_segment(own=[(lo, hi)]) writes a column only inside its bounds", bounded)
    for a in POINT_ARRAYS:
        getattr(s, a.name)[:] = before[a.name]
    check("(a) `imported` is a per-point array (in POINT_ARRAYS, so undo / swap / save carry it)",
          lambda: "imported" in [a.name for a in POINT_ARRAYS] and s.imported.shape == (40, 1))

    def default_write():
        s.imported[5, p] = True
        s.write_segment(0, tw, vw, [p], cw)
        return np.allclose(s.tracks[5, p], 50.0) and not s.manual[5, p] and not s.imported[5, p]
    check("(a) the default write is today's (it writes over a hand cell) and clears `imported` with `manual`",
          default_write)
    for a in POINT_ARRAYS:
        getattr(s, a.name)[:] = before[a.name]

    def clear_keep():
        n = s.clear_window([p], 0, 39, keep_hand=True)
        return (n == 38 and hand(s, 5, p) and hand(s, 6, p) and tuple(s.tracks[5, p]) == (99.0, 99.0)
                and not s.tracked[7:, p].any() and not s.tracked[:5, p].any())
    check("(a) clear_window(keep_hand=True) keeps the hand cells and clears the rest", clear_keep)
    for a in POINT_ARRAYS:
        getattr(s, a.name)[:] = before[a.name]
    check("(a) clear_window() without it still clears the hand cells (Delete on the timeline)",
          lambda: s.clear_window([p], 0, 39) == 40 and not s.tracked[:, p].any())
    for a in POINT_ARRAYS:
        getattr(s, a.name)[:] = before[a.name]
    s.set_source(p, "silhouette", "tip")
    check("(a) set_source -> silhouette keeps the hand frames and erases the tracker's",
          lambda: hand(s, 5, p) and hand(s, 6, p) and int(s.tracked[:, p].sum()) == 2 and s.points[p].derived,
          int(s.tracked[:, p].sum()))
    # the silhouette fills the other frames, then the point goes back to appearance tracking
    s.write_segment(10, np.full((10, 1, 2), 70.0, np.float32), np.ones((10, 1), bool), [p],
                    np.full((10, 1), 0.9, np.float32))
    s.set_source(p, "track")
    check("(a) set_source -> track keeps the hand frames and erases the silhouette's",
          lambda: hand(s, 5, p) and hand(s, 6, p) and int(s.tracked[:, p].sum()) == 2 and not s.points[p].derived,
          int(s.tracked[:, p].sum()))


# ================================================================= (b) imported clicks (D7)
def section_b():
    print("(b) imported clicks: a DLTdv project imported by the user's gesture")
    import _synth_dvproject as S
    from kinetrace import dltdvimport
    dv = os.path.join(OUT, "dv")
    B = S.build(dv)
    forget_recovery(*B["videos"])
    dltdvimport.ClickShiftDialog.exec = lambda self: QDialog.Accepted
    dltdvimport.ImportReportDialog.exec = lambda self: QDialog.Accepted
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    win.activateWindow()
    pump(0.3)
    PICK["file"], PICK["dir"] = B["v7"], ""
    QTest.keyClick(win, Qt.Key_F, Qt.AltModifier)
    pump(0.2)
    m = QApplication.activePopupWidget()
    QTest.keyClick(m, Qt.Key_I)
    pump(0.2)
    QTest.keyClick(QApplication.activePopupWidget(), Qt.Key_D)
    ok = wait(lambda: win.project is not None and win.project.n_views == 3 and win.state == READY, 120)
    pump(0.5)
    if not check("(b setup) File > Import > DLTdv Project made a 3-camera project", ok):
        close(win)
        return
    p = win.project
    s = p.sessions[0]
    mh = s.manual & s.tracked
    check("(b setup) the DLTdv clicks are hand-placed (as today)", mh.sum() > 10, int(mh.sum()))
    check("(b) ... and marked imported, exactly on the hand-placed cells",
          lambda: np.array_equal(s.imported, mh) and s.imported.any())
    j = 0
    fr = np.flatnonzero(mh[:, j])
    f = int(fr[len(fr) // 2])
    win._set_active_view(0)
    pump(0.2)
    win._on_select(j)
    win._goto(f, force=True)
    wait(lambda: win.cache.get(f) is not None, 10)
    win.canvas.fit()
    pump(0.1)
    x, y = (float(v) for v in s.tracks[f, j])
    canvas_click(win, x + 3, y + 2)
    check("(b) a click of yours on that frame makes it your click (imported cleared, still hand-placed)",
          lambda: hand(s, f, j) and not s.imported[f, j])
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.project.sessions[0]
    check("(b) Ctrl+Z brings the imported mark back", lambda: bool(s.imported[f, j]) and hand(s, f, j))
    # timeline Delete over two frames of that lane
    from kinetrace.timeline import LANE_H
    tlp = win.timeline
    yr = tlp._row_rect_y(j)[0] + LANE_H // 2
    a_, b_ = QPointF(tlp._x_of(f), yr), QPointF(tlp._x_of(f + 1), yr)
    tlp.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, a_, a_, Qt.LeftButton, Qt.LeftButton, Qt.ShiftModifier))
    tlp.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, b_, b_, Qt.NoButton, Qt.LeftButton, Qt.ShiftModifier))
    tlp.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, b_, b_, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    app.processEvents()
    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    pump(0.2)
    s = win.project.sessions[0]
    check("(setup) timeline Delete cleared that frame", not s.tracked[f, j])
    check("(b) ... and its imported mark", lambda: not s.imported[f, j])
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.project.sessions[0]
    # Point Tools Swap walks POINT_ARRAYS: the mark goes with the click
    k = 1

    def swapped():
        imp_j, imp_k = s.imported[:, j].copy(), s.imported[:, k].copy()
        n = s.swap_points(j, k, 0, s.n_frames - 1)
        ok_ = n > 0 and np.array_equal(s.imported[:, j], imp_k) and np.array_equal(s.imported[:, k], imp_j)
        s.swap_points(j, k, 0, s.n_frames - 1)
        return ok_
    check("(b) Point Tools Swap carries the imported marks with the clicks", swapped)
    # save: the CSV says which clicks are imported
    target = os.path.join(OUT, "dv_saved.kinetrace")
    PICK["save"] = target
    saved = win._save_project_as()
    check("(b setup) the project saves", saved and os.path.isfile(os.path.join(target, "kinetrace.json")))
    rows_ok, header_ok = [], []
    for path in glob.glob(os.path.join(target, "cameras", "*", "tracks", "*.csv")):
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            header_ok.append("imported" in (rd.fieldnames or []))
            for r in rd:
                if r.get("hand_placed") in ("1", "True", "true"):
                    rows_ok.append(r.get("imported") in ("1", "True", "true"))
    check("(b) every landmark CSV has an `imported` column", header_ok and all(header_ok), header_ok[:3])
    check("(b) ... = 1 on every imported click", rows_ok and all(rows_ok), (len(rows_ok), sum(rows_ok)))
    check("(b) save -> reopen keeps the marks", lambda: np.array_equal(pf.load(target).sessions[0].imported,
                                                                       win.project.sessions[0].imported))
    # the recovery copy (an edit makes the project dirty)
    s = win.project.sessions[0]
    canvas_click(win, x + 4, y + 4)
    win._autosave(wait=True)
    zp = recovery.paths(win._project_id)[0]
    check("(b setup) the recovery copy is written", os.path.isfile(zp))
    check("(b) the recovery copy keeps the marks",
          lambda: np.array_equal(pf.read(str(zp))[0].sessions[0].imported, win.project.sessions[0].imported))
    old = os.path.join(OUT, "old_layout.kinetrace")
    with zipfile.ZipFile(zp) as zin, zipfile.ZipFile(old, "w") as zout:
        for it in zin.infolist():
            if not it.filename.endswith("/imported.npy"):
                zout.writestr(it, zin.read(it.filename))
    check("(b) a recovery copy written before the update (no imported.npy) still opens, with no marks",
          lambda: not pf.read(old)[0].sessions[0].imported.any() and pf.read(old)[0].sessions[0].tracked.any())
    close(win)


# ================================================================= (c) Moving spot
def section_c():
    print("(c) Moving spot: clicks are kept and the spot restarts from them")
    # c1: a click ahead survives the run, exactly
    win = new_window(VID1)
    s = win.session
    pid = add_spot_point(win, LINE)
    click_at(win, 30, LINE[30, 0] + 1.5, LINE[30, 1] - 1.0)
    c30 = s.tracks[30, pid].copy()
    check("(c setup) the click at 30 is a hand cell", hand(s, 30, pid))
    ran = track_press(win)
    check("(c setup) T ran Moving spot from frame 0", ran and s.tracked[60, pid])
    check("(c) the click at 30 is still there, exactly, and still hand-placed",
          lambda: hand(s, 30, pid) and np.array_equal(s.tracks[30, pid], c30), (s.tracks[30, pid], c30))
    check("(c) ... and the track goes on after it on the spot",
          lambda: s.tracked[31:75, pid].all() and np.hypot(*(s.tracks[40, pid] - LINE[40])) < 1.5)
    close(win)
    # c2: hidden 45-51, clicked at 52 where it comes back: cut, not stopped
    win = new_window(VID_GAP)
    s = win.session
    pid = add_spot_point(win, GAP)
    click_at(win, 52, *GAP[52])
    ran = track_press(win)
    check("(c setup) T ran", ran)
    check("(c) the run is not stopped where the spot vanished: it is tracked again from the click at 52",
          lambda: s.tracked[53:75, pid].all() and np.hypot(*(s.tracks[70, pid] - GAP[70])) < 1.5,
          (int(s.tracked[53:75, pid].sum()), win.toast.text()[:80]))
    check("(c) ... with no data where it was hidden (cut, not guessed)",
          lambda: not s.tracked[46:52, pid].any() and s.tracked[:45, pid].all(),
          np.flatnonzero(s.tracked[40:53, pid]) + 40)
    close(win)
    # c3: a point with no position on the start frame joins at its first click
    win = new_window(VID1)
    s = win.session
    p0 = add_spot_point(win, LINE)
    win.layers.clearSelection()                # N with a point selected continues it: a NEW point here
    pump(0.05)
    win.selected = None
    win.btn_add.setChecked(True)
    win._goto(20, force=True)
    pump(0.05)
    canvas_click(win, *LINE[20])
    q = s.n_points - 1
    click_at(win, 21, *LINE[21])
    check("(c setup) the second point starts at 20", q != p0 and not s.tracked[0, q] and hand(s, 20, q))
    track_press(win)
    check("(c) a selected point with no position on the start frame joins the run at its click",
          lambda: s.tracked[22:70, q].all() and np.hypot(*(s.tracks[50, q] - LINE[50])) < 1.5,
          int(s.tracked[:, q].sum()))
    close(win)
    # c4 (I317): the run followed the glint; the user clicks the real spot at 30
    win = new_window(VID_GLINT)
    s = win.session
    pid = add_spot_point(win, G)
    click_at(win, HG, *A[HG])
    track_press(win)
    check("(c) after the click on the real spot it follows the real spot, not the glint it was on (I317)",
          lambda: all(np.hypot(*(s.tracks[f, pid] - A[f])) < 2.0 for f in range(HG + 2, HG + 30)),
          [tuple(np.round(s.tracks[f, pid], 1)) for f in (HG + 2, HG + 10)])
    close(win)
    # the speed rule of SpotRun.reseed, on the same frames (no app)
    def speed_rule():
        cap = cv2.VideoCapture(VID_GLINT)
        frames = []
        for _ in range(HG + 1):
            ok, fr = cap.read()
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        run = spots.SpotRun([(0, G[0], (2.0, 0.0), None)], (W, H), 0)
        run.resolve(frames[0])
        for i in range(HG):
            run.step(i, frames[i])
        far = run.reseed(0, HG, A[HG], {HG: A[HG]})                     # no neighbouring click, far from G
        near = run.reseed(0, HG, A[HG], {HG - 1: A[HG] - (2.0, 0.0), HG: A[HG]})
        return far.vel is None and np.allclose(near.vel, (2.0, 0.0))
    check("(c) SpotRun.reseed: speed from the neighbouring clicks, else unknown when the click is far from "
          "the old track (never click minus the drifted position)", speed_rule)
    # c5: clicks every third frame = one hand stretch, one re-seed, every click intact
    win = new_window(VID_LONG)
    s = win.session
    pid = add_spot_point(win, LONG)
    clicks = {}
    for f in range(40, 239, 3):
        click_at(win, f, LONG[f, 0] + 0.7, LONG[f, 1])
        clicks[f] = s.tracks[f, pid].copy()
    track_press(win, timeout=180)
    w = LAUNCHED[-1] if LAUNCHED else None
    check("(c) clicks with up to 2 frames between them are ONE stretch: one re-seed, at its last click",
          lambda: [f for f, _p in getattr(w, "_reseed_log", [])] == [238], getattr(w, "_reseed_log", None))
    check("(c) ... and every click is intact", lambda: all(hand(s, f, pid) and np.array_equal(s.tracks[f, pid], c)
                                                          for f, c in clicks.items()))
    close(win)


# ================================================================= (d) silhouette landmarks (G218)
def blob(cx, cy, w=W, h=H):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (30, 7), 0, 0, 360, 1, -1)
    return m.astype(bool)


class MockAnimal:
    """SAM stand-in: an ellipse round the line's spot on every frame (no weights, deterministic)."""

    def new_session(self, start, size):
        w, h = size

        class S:
            def step(self, native, idx, prompts):
                m = blob(*LINE[min(idx, N1 - 1)], w, h)
                return segmenter.FrameMasks(idx, [1], m[None], np.array([8.0], np.float32), (w, h), (w, h))
        return S()


def section_d():
    print("(d) silhouette landmarks take clicks; the Data source switch keeps them")
    win = new_window(VID1)
    s = win.session
    s.ensure_animal()
    s.animal.add_click(0, float(LINE[0, 0]), float(LINE[0, 1]), True)
    for f in range(N1):
        s.masks.set(f, blob(*LINE[f]), 8.0)
    s.apply_skeleton({"name": "d", "landmarks": ["snoot", "tailtip"], "bones": [], "head": "snoot",
                      "derived": {"tailtip": "tip"}})
    win._refresh_point_list()
    win._refresh_animal_panel()
    dpid = next(q for q in range(s.n_points) if s.part_name(q) == "tailtip")
    win._on_select(dpid)
    click_at(win, 10, LINE[10, 0] + 25, LINE[10, 1] + 1)
    c10 = s.tracks[10, dpid].copy()
    check("(d) a click on a silhouette landmark places it by hand on that frame", lambda: hand(s, 10, dpid))
    # a silhouette run (mock SAM, the animal's row selected) fills its other frames and keeps the click
    saved = trk.get_segmenter
    trk.get_segmenter = lambda be, **k: MockAnimal()
    try:
        win.layers.clearSelection()
        win._animal_item(0).setSelected(True)
        win._goto(0, force=True)
        pump(0.05)
        n0 = len(LAUNCHED)
        QTest.keyClick(win, Qt.Key_T)
        wait(lambda: len(LAUNCHED) > n0, 15)
        wait(lambda: win.state == READY and win.worker is None and win._passes is None, 120)
        pump(0.2)
    finally:
        trk.get_segmenter = saved
    check("(d setup) the silhouette run filled the landmark", s.tracked[30:60, dpid].all())
    check("(d) ... and kept the click exactly", lambda: hand(s, 10, dpid) and np.array_equal(s.tracks[10, dpid], c10))
    close(win)
    # the Data source switch, by the point's LAYERS menu
    win = new_window(VID1)
    s = win.session
    pid = s.add_point(0, 50.0, 60.0)
    for f in range(1, 30):
        s.tracks[f, pid] = (50.0 + f, 60.0)
        s.tracked[f, pid] = s.visibility[f, pid] = True
        s.confidence[f, pid] = 0.8
    s.manual[0, pid] = False
    s.set_position(5, pid, 90.0, 90.0)
    s.set_position(6, pid, 91.0, 91.0)
    s.ensure_animal()
    win._refresh_point_list()
    chosen = {"text": None}

    def pick_menu(menu, pos):
        def walk(m):
            for act in m.actions():
                if act.menu() is not None:
                    hit = walk(act.menu())
                    if hit is not None:
                        return hit
                elif act.text().startswith("From silhouette: tail tip"):
                    return act
            return None
        hit = walk(menu)
        chosen["text"] = hit.text() if hit is not None else None
        return hit
    orig_pop = appmod.pop_menu
    appmod.pop_menu = pick_menu
    try:
        ASK["asked"].clear()
        ASK["answer"] = QMessageBox.Yes
        lw = win.layers
        r = lw.visualItemRect(lw.point_item(pid)).center()
        QApplication.sendEvent(lw.viewport(), QContextMenuEvent(QContextMenuEvent.Mouse, r, lw.viewport().mapToGlobal(r)))
        pump(0.3)
    finally:
        appmod.pop_menu = orig_pop
    check("(d setup) the LAYERS menu offered Data source -> From silhouette: tail tip", chosen["text"] is not None)
    asked = " | ".join(t for _ti, t in ASK["asked"])
    check("(d) the question gives both counts: 28 tracked frames erased, 2 hand-placed kept",
          "28" in asked and "2 hand" in asked and "kept" in asked, asked[:160])
    check("(d) ... and the switch keeps the hand frames", lambda: s.points[pid].derived and hand(s, 5, pid)
          and hand(s, 6, pid) and int(s.tracked[:, pid].sum()) == 2, int(s.tracked[:, pid].sum()))
    close(win)


# ================================================================= (e) the Track pop-up (D9 choices 1-2)
def section_e():
    print("(e) the Track pop-up")
    check("(e) the Track pop-up exists (TrackChoiceDialog)", HAS_POPUP)
    # a reference run on the vanishing clip with nothing ahead: no pop-up
    win = new_window(VID_GAP)
    s = win.session
    pid = add_spot_point(win, GAP)
    n_seen = len(POP["seen"])
    track_press(win)
    check("(e) no data ahead -> no pop-up", len(POP["seen"]) == n_seen)
    ref = {a.name: getattr(s, a.name)[:45, pid].copy() for a in POINT_ARRAYS}
    last_ref = int(np.flatnonzero(s.tracked[:, pid]).max())
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.session
    # old data ahead (tracked 50-79) and a click ahead at 60: the press asks, with counts
    for f in range(50, N1):
        s.tracks[f, pid] = (LINE[f, 0] + 5.0, LINE[f, 1])
        s.tracked[f, pid] = s.visibility[f, pid] = True
        s.confidence[f, pid] = 0.8
    s.set_position(60, pid, *LINE[60])
    pre = s.snapshot()
    POP["pick"], POP["remember"] = "fresh", True
    n_seen = len(POP["seen"])
    track_press(win)
    s = win.session
    msg = POP["seen"][-1] if len(POP["seen"]) > n_seen else ""
    check("(e) data ahead -> Track asks first", len(POP["seen"]) > n_seen)
    check("(e) ... saying how much is ahead (31 tracked frames, 2 of them clicks)",
          "31" in msg and "2 " in msg and "click" in msg, msg[:160])
    check("(e) Fresh run: the frames it tracked are today's run, bit for bit",
          lambda: all(np.array_equal(getattr(s, a.name)[:45, pid], ref[a.name], equal_nan=True) for a in POINT_ARRAYS))
    check("(e) ... then everything after the last frame it reached is erased (old data and the click at 60)",
          lambda: not s.tracked[last_ref + 1:, pid].any(), np.flatnonzero(s.tracked[:, pid])[-3:])
    check("(e) Fresh run cannot be remembered (its 'use every time' box is off)",
          bool(POP["fresh_box"]) and POP["fresh_box"][-1] == ("fresh", False), POP["fresh_box"][-1:])
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.session
    check("(e) one Ctrl+Z restores everything (the old data and the click)",
          lambda: all(np.array_equal(getattr(s, a.name), getattr(pre, a.name), equal_nan=True) for a in POINT_ARRAYS))
    # Keep my clicks, remembered
    POP["pick"], POP["remember"] = "keep", True
    n_seen = len(POP["seen"])
    track_press(win)
    s = win.session
    check("(e) Keep my clicks keeps the click at 60", lambda: hand(s, 60, pid) and np.allclose(s.tracks[60, pid], LINE[60],
                                                                                              atol=1.0))
    check("(e) Fresh was not remembered: the next press asked again", len(POP["seen"]) == n_seen + 1)
    POP["remember"] = False
    n_seen = len(POP["seen"])
    track_press(win)
    check("(e) 'Use this every time' -> the next press does not ask", len(POP["seen"]) == n_seen)
    check("(e) ... and Track > 'When there is data ahead' shows Keep my clicks",
          lambda: win.act_ahead_keep.isChecked() and not win.act_ahead_ask.isChecked())
    win._sync_ui_state()
    check("(e) ... saved with the view state", lambda: win.session.ui_state.get("ahead_choice") == "keep")
    if hasattr(win, "act_ahead_ask"):
        win.act_ahead_ask.trigger()
        pump(0.05)
    # F (semi-automatic) never asks
    win.act_mode_semi.trigger()
    n_seen = len(POP["seen"])
    win._goto(10, force=True)
    win.act_select_all.trigger()
    QTest.keyClick(win, Qt.Key_F)
    wait(lambda: win.state == READY and win.worker is None, 30)
    pump(0.2)
    check("(e) F never asks", len(POP["seen"]) == n_seen)
    win.act_mode_auto.trigger()
    close(win)
    # every camera: the counts cover the other camera's data too
    win = new_window(VID1)
    if not check("(e setup) a second camera is added", win._add_view(VID_B)):
        close(win)
        return
    wait(lambda: win.project is not None and win.project.n_views == 2 and win.state == READY, 60)
    pump(0.3)
    p = win.project
    win.act_pm_spot.trigger()
    win._set_active_view(0)
    pump(0.2)
    add_spot_point(win, LINE)
    name = p.sessions[0].points[0].name
    win._set_active_view(1)
    pump(0.2)
    sb = p.sessions[1]
    jb = sb.pid_by_name(name)
    win.layers.setCurrentItem(win.layers.point_item(jb))
    lineb = LINE + np.array([0.0, 30.0])
    for f in (0, 1):
        click_at(win, f, *lineb[f])
    for f in range(40, 60):                                     # data ahead in camera 2 only
        sb.tracks[f, jb] = lineb[f]
        sb.tracked[f, jb] = sb.visibility[f, jb] = True
        sb.confidence[f, jb] = 0.8
    win._set_active_view(0)
    pump(0.2)
    win.act_track_all.setChecked(True)
    POP["pick"], POP["remember"] = "keep", False
    n_seen = len(POP["seen"])
    track_press(win, timeout=180)
    win.act_track_all.setChecked(False)
    msg = POP["seen"][-1] if len(POP["seen"]) > n_seen else ""
    check("(e) an every-camera press asks when only ANOTHER camera has data ahead, and counts it",
          len(POP["seen"]) > n_seen and "20" in msg and "camera" in msg, msg[:160])
    close(win)


for fn in (section_a, section_b, section_c, section_d, section_e):
    section(fn)
appmod.MainWindow._launch_run = _orig_launch
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_hand_truth: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
