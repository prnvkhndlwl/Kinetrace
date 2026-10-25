"""A tracker per camera (D8, G216) and the "Hand only" tracker (D10, G219); plan
dev/work/2026-10-20-silhouette-hand-mix, PR D. The owner: "have each point run its individual tracker set up by
the user", a different tracker for the same point in different cameras, and a per-point "Hand only (no tracker)".

Offscreen; no GPU and no model: the Moving spot runs are real, and AllTracker / CoTracker3 points are followed by a
STAND-IN model worker (the worker's model loop replaced by one that writes the seed moved 0.5 px a frame and
records which camera ran which model on which points), so the app's pass planning, every-camera runs, bounds and
end-of-run handling all run for real. tests/verify_tracker_per_camera_models.py runs the real models (GPU).
Sections:
  (a) the Tracker menu (LAYERS right-click): "Only in this camera" changes this camera alone, the top entries
      every camera; LAYERS shows the tag of the camera on screen; the menu says when cameras differ; one Ctrl+Z;
      saved and reopened; a point new to a camera gets the project's point model; Hand only is never the
      project's point model.
  (b) Hand only with Moving spot runs: never tracked (named in the run's status line), its clicks stay; only
      Hand-only points selected -> Track says so and starts nothing; the pop-up counts and a Fresh run leave
      its clicks out; every camera: tracked where its camera has a tracker, untouched where it is Hand only;
      no corrections hint for it.
  (c) every camera, each with its own tracker (stand-in models): the snout on AllTracker in camera 1 and
      CoTracker3 in camera 2, a third camera holding only CoTracker3 points -> one press tracks every point in
      every camera with its own tracker; pass k = model k; a camera's later pass is bounded by its own earlier
      pass only where that pass ran (review R0-C2).
  (d) Track > Test the point models > Use sets the tracker in this camera only (R0-M11).
  (e) 3D > Re-track Disagreeing Stretches skips a Hand-only point (named, nothing run).
  (f) the DeepLabCut export's scorer of a camera whose points are all placed by hand.
Every check FAILS on the code before the change for the reason it names (no camera scope in the Tracker menu, no
Hand only, one pass plan for every camera). Run: .venv\\Scripts\\python.exe tests\\verify_tracker_per_camera.py
"""
import os
import shutil
import sys
import time
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "tracker_per_camera")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

ASK = {"answer": QMessageBox.Yes, "asked": [], "info": []}      # questions / notices: (title, text)


def _question(parent, title, text, *a, **k):
    ASK["asked"].append((str(title), str(text)))
    return ASK["answer"]


def _info(parent, title, text, *a, **k):
    ASK["info"].append((str(title), str(text)))
    return QMessageBox.Ok


QMessageBox.question = staticmethod(_question)
QMessageBox.information = staticmethod(_info)
QMessageBox.warning = staticmethod(_info)
QMessageBox.critical = staticmethod(_info)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
PICK = {"file": "", "save": "", "filter": ""}
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (PICK["file"], ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (PICK["save"], PICK["filter"]))

app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402
from _synth_spots import write_video  # noqa: E402

import kinetrace.app as appmod  # noqa: E402
import kinetrace.layers as layersmod  # noqa: E402
from kinetrace import tracker as trk  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.session import POINT_ARRAYS  # noqa: E402

POP = stub_track_choice("keep")                 # the Track pop-up: answered, its sentence recorded
appmod._alltracker_available = lambda: True     # the stand-in needs no AllTracker code (nothing loads)
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


# ================================================================= the stand-in point model
RUNS: list = []             # every model worker: {video, backend, names, start, end, last}
STOP_AT: dict = {}          # (video basename, backend) -> frame where the stand-in reports its first point lost
_orig_run_steps = trk.TrackingWorker._run_steps


def _standin_steps(self, src):
    """The model loop of an AllTracker / CoTracker3 worker without a model: row 0 = the seeds, then each seed moved
    0.5 px a frame to the right, in 8-frame chunks, with the frames for the display. A worker with no model point
    (Moving spot, silhouettes, balls) runs for real."""
    if not self.specs:
        return (yield from _orig_run_steps(self, src))
    rec = {"video": os.path.basename(self.video_path), "backend": self.point_backend,
           "pids": [sp.pid for sp in self.specs], "start": self.start_frame, "end": self.n_frames, "last": None}
    RUNS.append(rec)
    self.model_loading.emit()
    self.started_ok.emit()
    stop = STOP_AT.get((rec["video"], self.point_backend))
    f, last = self.start_frame, self.start_frame - 1
    while f < self.n_frames and not self._pause and (stop is None or f < stop):
        n = min(8, self.n_frames - f) if stop is None else min(8, self.n_frames - f, stop - f)
        tr = np.full((n, self.n_cols, 2), np.nan, np.float32)
        vi = np.zeros((n, self.n_cols), bool)
        cf = np.zeros((n, self.n_cols), np.float32)
        for i, sp in enumerate(self.specs):
            for j in range(n):
                tr[j, i] = np.asarray(sp.seed, np.float32) + np.array([0.5 * (f + j - self.start_frame), 0.0],
                                                                      np.float32)
            vi[:, i] = True
            cf[:, i] = 0.9
        frames = [(f + j, src.get_frame(f + j)) for j in range(n)]
        self.chunk_ready.emit(f, tr, vi, cf, {}, frames)
        last = f + n - 1
        f += n
        yield
    rec["last"] = last
    if stop is not None and f >= stop and not self._pause:
        self._autopause_hit = (int(stop), int(self.specs[0].pid))      # the first point "lost" there
        self._autopause_reason = "lowconf"
        self.autopaused.emit(*self._autopause_hit)
    self.finished_ok.emit(last, self._pause or self._autopause_hit is not None)


trk.TrackingWorker._run_steps = _standin_steps

LAUNCHED: list = []         # every run the app launches (the working camera's worker)
_orig_launch = MainWindow._launch_run


def _spy_launch(self, w, driver=None):
    LAUNCHED.append(w)
    return _orig_launch(self, w, driver)


MainWindow._launch_run = _spy_launch

TAGS: dict = {}             # LAYERS: the tracker tag last drawn on each point's row, by name
_orig_icon = layersmod.point_icon


def _spy_icon(meta, tag=""):
    TAGS[meta.name] = tag
    return _orig_icon(meta, tag)


layersmod.point_icon = _spy_icon

# ================================================================= the clips (320 x 240 gray, bright spots)
W, H = 320, 240
N = 60
_rng = np.random.RandomState(11)
BG = cv2.GaussianBlur((_rng.rand(H, W) * 90 + 40).astype(np.float32), (0, 0), 3)
YY, XX = np.mgrid[0:H, 0:W]
TRUTH = [np.stack([40.0 + 2.0 * np.arange(N), np.full(N, y)], 1) for y in (60.0, 120.0, 180.0)]


def spot_clip(name, centres):
    frames = []
    for f in range(len(centres[0])):
        img = BG.copy()
        for c in centres:
            x, y = c[f]
            img = img + 150.0 * np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / (2 * 2.0 ** 2))
        g = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(np.dstack([g, g, g]))
    path = os.path.join(OUT, name + ".mp4")
    write_video(path, frames, fps=30.0)
    return path


CAMS = [spot_clip(f"cam{k}", TRUTH) for k in range(3)]
forget_recovery(*CAMS)


# ================================================================= the window and the user's gestures
def new_window(video):
    forget_recovery(video)
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
        for sv in win.project.sessions:
            sv.dirty = False
    win.close()
    pump(0.3)
    win._dev_probe.wait(30000)


def add_cameras(win, paths):
    n0 = win.project.n_views if win.project is not None else 0
    for path in paths:
        win._add_view(path)
        wait(lambda: win.state == READY and not win._loading, 60)
    pump(0.3)
    return win.project is not None and win.project.n_views == n0 + len(paths)


def canvas_click(win, x, y):
    from PySide6.QtCore import QPointF
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    pump(0.05)


def select_rows(win, *pids):
    win.layers.clearSelection()
    for q in pids:
        win.layers.point_item(q).setSelected(True)
    pump(0.05)


def new_point(win, xy, name=None):
    """N + a click on the video (frame 0): a new point; renamed `name` (all cameras) when given."""
    win._goto(0, force=True)
    pump(0.05)
    win.btn_add.setChecked(True)
    canvas_click(win, *xy)
    pid = win.session.n_points - 1
    if name is not None:
        win._apply_rename(pid, name)
        pump(0.05)
    return pid


def place(win, v, name, f, xy):
    """In camera v: select the point `name` in LAYERS and click it on frame f (a hand placement)."""
    win._set_active_view(v)
    pump(0.1)
    s = win.session
    q = s.pid_by_name(name)
    select_rows(win, q)
    win._on_select(q)
    win._goto(f, force=True)
    pump(0.05)
    canvas_click(win, *xy)
    return q


def menu_texts(m, depth=0):
    out = []
    for a in m.actions():
        out.append("  " * depth + a.text())
        if a.menu() is not None:
            out += menu_texts(a.menu(), depth + 1)
    return out


def layers_menu(win, pid, path):
    """Right-click `pid`'s LAYERS row (it alone selected) and choose the entry at `path` (text prefixes, submenu
    first) -> {"chosen": its text or None, "texts": every entry of the menu, indented by depth}."""
    got = {"chosen": None, "texts": []}

    def pick(menu, pos):
        got["texts"] = menu_texts(menu)
        menu.deleteLater()              # as `pop_menu` does (a shown menu kept alive crashed a later save)
        m, hit = menu, None
        for i, want in enumerate(path):
            hit = next((a for a in m.actions() if a.text().startswith(want)), None)
            if hit is None:
                return None
            if i < len(path) - 1:
                if hit.menu() is None:
                    return None
                m = hit.menu()
        if hit is None or not hit.isEnabled():
            return None
        got["chosen"] = hit.text()
        return hit
    select_rows(win, pid)
    win._on_select(pid)
    orig = appmod.pop_menu
    appmod.pop_menu = pick
    try:
        lw = win.layers
        r = lw.visualItemRect(lw.point_item(pid)).center()
        QApplication.sendEvent(lw.viewport(), QContextMenuEvent(QContextMenuEvent.Mouse, r,
                                                                lw.viewport().mapToGlobal(r)))
        pump(0.2)
    finally:
        appmod.pop_menu = orig
    return got


def set_tracker_here(win, v, name, key_label, key):
    """Camera v: right-click the point -> Tracker -> Only in this camera -> `key_label`. On the old code (no such
    entry) the tracker is set straight on that camera's point, so the run checks after it still test the run."""
    win._set_active_view(v)
    pump(0.1)
    s = win.session
    q = s.pid_by_name(name)
    got = layers_menu(win, q, ["Tracker", "Only in this camera", key_label])
    if got["chosen"] is None:
        s.points[q].tracker = key
    return got["chosen"] is not None


def track_press(win, start=0, every=False, timeout=120, mods=Qt.NoModifier, select_all=True):
    """Select all, go to `start`, T: True when a run was launched and the whole press has ended."""
    win.act_track_all.setChecked(bool(every))
    if select_all:
        win.act_select_all.trigger()
    win._goto(start, force=True)
    pump(0.05)
    n0 = len(LAUNCHED)
    QTest.keyClick(win, Qt.Key_T, mods)
    started = wait(lambda: len(LAUNCHED) > n0, 15)
    ended = wait(lambda: win.state == READY and win.worker is None and win._passes is None and win._multi is None,
                 timeout)
    pump(0.3)
    return started and ended


def ctrl_z(win):
    """Ctrl+Z as the user presses it: in the ACTIVE window (offscreen, a run can leave no window active and the key
    then reaches no action; verify_manual_mode does the same, review R1-M4)."""
    win.activateWindow()
    pump(0.05)
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)


class Toasts:
    def __init__(self, win):
        self.msgs: list = []
        win.toast.show_message = lambda text, level="info", ms=6000, on_click=None: self.msgs.append(str(text))

    def text(self):
        return " | ".join(self.msgs)


class Status:
    def __init__(self, win):
        self.msgs: list = []
        win.statusBar().messageChanged.connect(lambda t: self.msgs.append(str(t)) if t else None)

    def text(self):
        return " | ".join(self.msgs)


def cells(s, q):
    return {a.name: getattr(s, a.name)[:, q].copy() for a in POINT_ARRAYS}


def same_cells(s, q, before):
    return all(np.array_equal(getattr(s, a.name)[:, q], before[a.name], equal_nan=True) for a in POINT_ARRAYS)


def last_tracked(s, q):
    t = np.flatnonzero(s.tracked[:, q])
    return int(t.max()) if len(t) else -1


# ================================================================= (a) the Tracker menu, per camera
def section_a():
    print("(a) the Tracker menu: this camera only / every camera")
    win = new_window(CAMS[0])
    check("(a setup) a second camera", add_cameras(win, [CAMS[1]]))
    p = win.project
    toasts = Toasts(win)
    win._set_active_view(0)
    pump(0.1)
    win.act_pm_cotracker.trigger()                       # the project's point model: CoTracker3
    q0 = new_point(win, TRUTH[0][0], "snout")
    place(win, 1, "snout", 0, TRUTH[0][0])
    s0, s1 = p.sessions
    win._set_active_view(0)
    pump(0.1)
    got = layers_menu(win, s0.pid_by_name("snout"), ["Tracker", "Only in this camera", "Moving spot"])
    check("(a) camera 1: Tracker -> Only in this camera -> Moving spot exists", got["chosen"] is not None,
          [t for t in got["texts"] if "Tracker" in t or t.startswith("  ")][:12])
    check("(a) ... and changes camera 1 only (camera 2 keeps the project's point model)",
          s0.points[q0].tracker == "spot" and s1.points[s1.pid_by_name("snout")].tracker in ("", "cotracker3"),
          (s0.points[q0].tracker, s1.points[s1.pid_by_name("snout")].tracker))
    check("(a) LAYERS shows camera 1's tracker on its row (MS)", TAGS.get("snout") == "MS", TAGS.get("snout"))
    win._set_active_view(1)
    pump(0.2)
    check("(a) camera 2 on screen: LAYERS shows ITS tracker (CT)", TAGS.get("snout") == "CT", TAGS.get("snout"))
    got = layers_menu(win, s1.pid_by_name("snout"), ["Tracker", "zzz"])
    menu = " / ".join(got["texts"])
    check("(a) the menu says the cameras differ (CoTracker3 here, Moving spot in the other camera)",
          "CoTracker3 here" in menu and "Moving spot in" in menu, [t for t in got["texts"] if "here" in t])
    # every camera at once, one Ctrl+Z
    got = layers_menu(win, s1.pid_by_name("snout"), ["Tracker", "AllTracker"])
    check("(a) Tracker -> AllTracker (the top entries) gives it to every camera",
          got["chosen"] is not None and s0.points[q0].tracker == "alltracker"
          and s1.points[s1.pid_by_name("snout")].tracker == "alltracker",
          (s0.points[q0].tracker, s1.points[s1.pid_by_name("snout")].tracker))
    ctrl_z(win)
    pump(0.2)
    s0, s1 = p.sessions
    check("(a) one Ctrl+Z puts every camera back (Moving spot / CoTracker3)",
          s0.points[s0.pid_by_name("snout")].tracker == "spot"
          and s1.points[s1.pid_by_name("snout")].tracker in ("", "cotracker3"),
          (s0.points[s0.pid_by_name("snout")].tracker, s1.points[s1.pid_by_name("snout")].tracker))
    # Hand only, in camera 2 only
    got = layers_menu(win, s1.pid_by_name("snout"), ["Tracker", "Only in this camera", "Hand only"])
    check("(a) Tracker -> Only in this camera -> Hand only (no tracker)", got["chosen"] is not None
          and s1.points[s1.pid_by_name("snout")].tracker == "hand" and s0.points[q0].tracker == "spot",
          (got["chosen"], s1.points[s1.pid_by_name("snout")].tracker))
    win._refresh_point_list()
    check("(a) LAYERS tags a Hand-only point 'H'", TAGS.get("snout") == "H", TAGS.get("snout"))
    item = win.layers.point_item(s1.pid_by_name("snout"))
    check("(a) ... and its row's tooltip says it is placed by hand only",
          item is not None and "by hand only" in item.toolTip(0), item.toolTip(0) if item is not None else None)
    # Hand only is never the project's point model (review RE-M10)
    texts = [a.text() for a in win.btn_track.menu().actions()]
    check("(a) Track > Point model offers no Hand only (it is per point)",
          not any(t.startswith("Point model") and "Hand" in t for t in texts), [t for t in texts if "Point" in t])
    # saved and opened again, per camera
    target = os.path.join(OUT, "percam.kinetrace")
    PICK["save"] = target
    saved = win._save_project_as()
    close(win)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    if saved:
        win._open_project_from_path(target)
        wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
        pump(0.3)
    p = win.project
    check("(a) save, close, open: each camera's tracker comes back (Moving spot / Hand only)",
          lambda: saved and p.sessions[0].points[p.sessions[0].pid_by_name("snout")].tracker == "spot"
          and p.sessions[1].points[p.sessions[1].pid_by_name("snout")].tracker == "hand",
          [[m.tracker for m in sv.points] for sv in p.sessions] if p is not None else None)
    # a point new to a camera gets the project's point model, not another camera's choice
    win._set_active_view(0)
    pump(0.1)
    win._deselect()
    pump(0.05)
    win.act_pm_cotracker.trigger()
    check("(a setup) a third camera", add_cameras(win, [CAMS[2]]))
    win._set_active_view(2)
    pump(0.2)
    s2 = p.sessions[2]
    check("(a) a point new to a camera gets the project's point model (CT), not camera 1's Moving spot",
          s2.pid_by_name("snout") is not None and TAGS.get("snout") == "CT"
          and s2.points[s2.pid_by_name("snout")].tracker == "", (TAGS.get("snout"),
                                                                 [m.tracker for m in s2.points]))
    close(win)
    del toasts


# ================================================================= (b) Hand only, Moving spot runs
def section_b():
    print("(b) Hand only")
    win = new_window(CAMS[0])
    s = win.session
    toasts = Toasts(win)
    status = Status(win)
    win.act_pm_spot.trigger()                            # project's point model: Moving spot (real runs)
    a = new_point(win, TRUTH[0][0], "spot a")
    layers_menu(win, a, ["Tracker", "Moving spot"])
    b = new_point(win, TRUTH[1][0], "hand b")
    for f in (10, 20):
        place(win, 0, "hand b", f, TRUTH[1][f])
    got = layers_menu(win, b, ["Tracker", "Hand only"])
    check("(b) Tracker -> Hand only (no tracker), one camera", got["chosen"] is not None
          and s.points[b].tracker == "hand", (got["chosen"], s.points[b].tracker, got["texts"][-8:]))
    if got["chosen"] is None:
        s.points[b].tracker = "hand"
    before = cells(s, b)
    status.msgs.clear()
    ran = track_press(win)
    w = LAUNCHED[-1] if LAUNCHED else None
    check("(b setup) Track ran (Moving spot)", ran)
    check("(b) a Hand-only point is never tracked: not in the run", w is not None and b not in w.point_ids,
          None if w is None else w.point_ids)
    check("(b) ... its cells (its clicks on 0, 10, 20) are exactly as before", same_cells(s, b, before),
          np.flatnonzero(s.tracked[:, b]))
    check("(b) ... the Moving spot point beside it was tracked to the end", last_tracked(s, a) == N - 1,
          last_tracked(s, a))
    check("(b) the run's status line names it (placed by hand only)", "hand b" in status.text()
          and "by hand only" in status.text(), status.text()[:300])
    # only Hand-only points selected: Track says so and starts nothing
    select_rows(win, b)
    win._on_select(b)
    win._goto(0, force=True)
    win._update_track_button()
    tip = win.btn_track.toolTip()
    check("(b) only Hand-only points selected: the Track button says they are placed by hand only",
          "by hand only" in tip, tip[:200])
    n0 = len(LAUNCHED)
    toasts.msgs.clear()
    QTest.keyClick(win, Qt.Key_T)
    pump(0.5)
    check("(b) ... T starts nothing", len(LAUNCHED) == n0 and win.state == READY, len(LAUNCHED) - n0)
    check("(b) ... and says why (placed by hand only)", "by hand only" in toasts.text(), toasts.text()[:200])
    # the pop-up counts and a Fresh run leave its clicks out
    for f in range(40, N):                               # an earlier run's data ahead of spot a
        s.tracks[f, a] = TRUTH[0][f] + (5.0, 0.0)
        s.tracked[f, a] = s.visibility[f, a] = True
        s.manual[f, a] = False
        s.confidence[f, a] = 0.8
    s.clear_window([a], 1, 39)
    before = cells(s, b)
    POP.answer = "fresh"
    n_said = len(POP.said)
    track_press(win)
    said = POP.said[-1] if len(POP.said) > n_said else ""
    check("(b) the pop-up counts only the tracked point's data ahead (20 frames, no click)",
          "20 tracked frames" in said and "click" not in said, said[:200])
    check("(b) a Fresh run never erases a Hand-only point's clicks", same_cells(s, b, before),
          np.flatnonzero(s.tracked[:, b]))
    POP.answer = "keep"
    # no corrections hint for a Hand-only point (it has no tracker to change; the project's point model is a
    # point MODEL here, which the hint is about)
    win._deselect()
    pump(0.05)
    win.act_pm_cotracker.trigger()
    for f in range(30, 40):                            # data of an earlier run, then corrected by hand
        s.tracks[f, b] = TRUTH[1][f]
        s.tracked[f, b] = s.visibility[f, b] = True
        s.manual[f, b] = False
    toasts.msgs.clear()
    for f in range(30, 36):
        place(win, 0, "hand b", f, TRUTH[1][f] + (1.0, 0.0))
    check("(b) correcting a Hand-only point by hand gives no 'try another point model' hint",
          "You have corrected" not in toasts.text(), toasts.text()[:200])
    close(win)
    # every camera: tracked where its camera gives it a tracker, untouched where it is Hand only
    win = new_window(CAMS[0])
    check("(b setup) a second camera", add_cameras(win, [CAMS[1]]))
    p = win.project
    win._set_active_view(0)
    pump(0.1)
    a = new_point(win, TRUTH[0][0], "spot a")
    layers_menu(win, a, ["Tracker", "Moving spot"])           # every camera
    new_point(win, TRUTH[1][0], "wing")
    place(win, 0, "wing", 20, TRUTH[1][20])
    for v in (1,):
        place(win, v, "spot a", 0, TRUTH[0][0])
        place(win, v, "wing", 0, TRUTH[1][0])
    ok_menu = set_tracker_here(win, 1, "wing", "Moving spot", "spot")
    ok_menu &= set_tracker_here(win, 0, "wing", "Hand only", "hand")
    check("(b setup) wing: Hand only in camera 1, Moving spot in camera 2 (by the menu)", ok_menu)
    s0, s1 = p.sessions
    before0 = cells(s0, s0.pid_by_name("wing"))
    for f in range(30, 50):                                    # camera 2's wing has an earlier run's data ahead
        q = s1.pid_by_name("wing")
        s1.tracks[f, q] = TRUTH[1][f] + (4.0, 0.0)
        s1.tracked[f, q] = s1.visibility[f, q] = True
        s1.confidence[f, q] = 0.8
    win._set_active_view(0)
    pump(0.1)
    n_said = len(POP.said)
    ran = track_press(win, every=True)
    said = POP.said[-1] if len(POP.said) > n_said else ""
    check("(b setup) the every-camera press ran", ran)
    check("(b) every camera: the wing is tracked in camera 2 (Moving spot there)",
          last_tracked(s1, s1.pid_by_name("wing")) == N - 1, last_tracked(s1, s1.pid_by_name("wing")))
    check("(b) ... and untouched in camera 1, where it is Hand only (its click at 20 kept)",
          same_cells(s0, s0.pid_by_name("wing"), before0), np.flatnonzero(s0.tracked[:, s0.pid_by_name("wing")]))
    check("(b) the pop-up counted camera 2's wing data, not camera 1's Hand-only click (20 frames, no click)",
          "20 tracked frames" in said and "click" not in said, said[:200])
    close(win)
    del toasts


# ================================================================= (c) every camera, each its own tracker
def section_c():
    print("(c) every camera, each with its own tracker (stand-in models)")
    win = new_window(CAMS[0])
    check("(c setup) three cameras", add_cameras(win, CAMS[1:]))
    p = win.project
    names = ["snout", "tail", "ear"]
    win._set_active_view(0)
    pump(0.1)
    win.act_pm_cotracker.trigger()
    for j, nm in enumerate(names):
        new_point(win, TRUTH[j][0], nm)
    for v in (1, 2):
        for j, nm in enumerate(names):
            place(win, v, nm, 0, TRUTH[j][0])
    plan = {0: {"snout": "alltracker", "tail": "cotracker3", "ear": "alltracker"},
            1: {"snout": "cotracker3", "tail": "cotracker3", "ear": "alltracker"},
            2: {"snout": "cotracker3", "tail": "cotracker3", "ear": "cotracker3"}}     # camera 3: CoTracker3 only
    label = {"alltracker": "AllTracker", "cotracker3": "CoTracker3"}
    ok_menu = True
    for v, d in plan.items():
        for nm, key in d.items():
            ok_menu &= set_tracker_here(win, v, nm, label[key], key)
    check("(c setup) each camera's trackers set by Tracker -> Only in this camera", ok_menu)
    check("(c) ... and each camera keeps its own (no camera's choice overwrote another's)",
          all(p.sessions[v].points[p.sessions[v].pid_by_name(nm)].tracker == key
              for v, d in plan.items() for nm, key in d.items()),
          [[m.tracker for m in sv.points] for sv in p.sessions])
    for v, d in plan.items():                       # the old code: the run check below still tests the run
        for nm, key in d.items():
            p.sessions[v].points[p.sessions[v].pid_by_name(nm)].tracker = key
    win._set_active_view(0)
    pump(0.1)
    win.act_select_all.trigger()
    win.act_track_all.setChecked(True)
    win._update_track_button()
    check("(c) the Track button says two passes", "2 passes" in win.btn_track.text(), win.btn_track.text())
    snap = [sv.snapshot() for sv in p.sessions]
    RUNS.clear()
    ran = track_press(win, every=True, timeout=180)
    check("(c setup) the every-camera press ran", ran)
    vid = {os.path.basename(CAMS[v]): v for v in range(3)}

    def by_cam():
        out = {v: [] for v in range(3)}
        for k, r in enumerate(RUNS):
            v = vid.get(r["video"])
            if v is not None:
                sv = p.sessions[v]
                out[v].append((k, r["backend"], sorted(sv.points[q].name for q in r["pids"]), r["end"]))
        return out
    got = by_cam()
    check("(c) every point in every camera is tracked to the last frame",
          all(last_tracked(sv, sv.pid_by_name(nm)) == N - 1 for sv in p.sessions for nm in names),
          [[last_tracked(sv, sv.pid_by_name(nm)) for nm in names] for sv in p.sessions])
    want = {v: sorted((key, nm) for nm, key in d.items()) for v, d in plan.items()}
    have = {v: sorted((b, nm) for _k, b, nms, _e in runs for nm in nms) for v, runs in got.items()}
    check("(c) each camera tracked each point with ITS OWN tracker (snout: AllTracker in 1, CoTracker3 in 2)",
          have == want, have)
    check("(c) the camera holding only CoTracker3 points ran CoTracker3, and no AllTracker",
          [b for _k, b, _n, _e in got[2]] == ["cotracker3"], got[2])
    order = [r["backend"] for r in RUNS]
    check("(c) pass 1 = AllTracker (cameras 1 and 2), pass 2 = CoTracker3 (cameras 1, 2, 3)",
          sorted(order[:2]) == ["alltracker"] * 2 and sorted(order[2:]) == ["cotracker3"] * 3, order)
    ctrl_z(win)
    pump(0.3)
    check("(c) one Ctrl+Z undoes both passes in every camera",
          all(np.array_equal(p.sessions[v].tracked, snap[v].tracked) for v in range(3)))
    # camera 2's AllTracker pass stops at frame 30 (a lost point): camera 2's CoTracker3 pass stops there too,
    # camera 3 (whose first pass did not run) goes to the end (R0-C2)
    STOP_AT[(os.path.basename(CAMS[1]), "alltracker")] = 30
    RUNS.clear()
    track_press(win, every=True, timeout=180)
    STOP_AT.clear()
    got = by_cam()
    ct2 = [e for _k, b, _n, e in got[1] if b == "cotracker3"]
    ct3 = [e for _k, b, _n, e in got[2] if b == "cotracker3"]
    check("(c) a camera's later pass is bounded by its own earlier pass (camera 2's CoTracker3 pass ends at 30)",
          ct2 and max(ct2) <= 31, got[1])
    check("(c) ... and a camera the earlier pass did not run in is not bounded (camera 3 to the end)",
          ct3 == [N] and all(last_tracked(p.sessions[2], p.sessions[2].pid_by_name(nm)) == N - 1 for nm in names),
          (got[2], [last_tracked(p.sessions[2], p.sessions[2].pid_by_name(nm)) for nm in names]))
    ctrl_z(win)
    pump(0.3)
    # AllTracker in the working camera only, CoTracker3 in the two others: pass 1 runs in camera 1 alone, pass 2
    # in cameras 2 and 3 together; no camera is "missing a position", and one Ctrl+Z restores all three
    for nm in names:
        set_tracker_here(win, 0, nm, "AllTracker", "alltracker")
        for v in (1, 2):
            set_tracker_here(win, v, nm, "CoTracker3", "cotracker3")
    win._set_active_view(0)
    pump(0.1)
    snap = [sv.snapshot() for sv in p.sessions]
    toasts = Toasts(win)
    RUNS.clear()
    track_press(win, every=True, timeout=180)
    got = by_cam()
    check("(c) AllTracker in camera 1 alone, then CoTracker3 in cameras 2 and 3: every point tracked to the end",
          [b for _k, b, _n, _e in got[0]] == ["alltracker"] and [b for _k, b, _n, _e in got[1]] == ["cotracker3"]
          and [b for _k, b, _n, _e in got[2]] == ["cotracker3"]
          and all(last_tracked(sv, sv.pid_by_name(nm)) == N - 1 for sv in p.sessions for nm in names), got)
    check("(c) ... with no 'only this camera can start' notice for the cameras the first pass leaves to the second",
          "only this camera can start" not in toasts.text(), toasts.text()[:200])
    ctrl_z(win)
    pump(0.3)
    check("(c) ... and one Ctrl+Z restores every camera (the first pass's camera too)",
          all(np.array_equal(p.sessions[v].tracked, snap[v].tracked) for v in range(3)))
    win.act_track_all.setChecked(False)
    close(win)


# ================================================================= (d) the point-model test's Use
class _FakeTest:
    """Track > Test the point models: the dialog answered "Use CoTracker3"."""
    def __init__(self, parent, name, *a, on_use=None, **k):
        self.on_use = on_use

    def exec(self):
        class R:
            model, settings, label = "cotracker3", None, "CoTracker3"
        self.on_use(R())
        return 0

    def deleteLater(self):
        pass


def section_d():
    print("(d) the point-model test's Use")
    import kinetrace.pointtest as pt
    win = new_window(CAMS[0])
    check("(d setup) a second camera", add_cameras(win, [CAMS[1]]))
    p = win.project
    toasts = Toasts(win)
    win._set_active_view(0)
    win.act_pm_spot.trigger()                            # the project's point model: Moving spot
    q = new_point(win, TRUTH[0][0], "snout")
    place(win, 1, "snout", 0, TRUTH[0][0])
    win._set_active_view(0)
    pump(0.1)
    select_rows(win, q)
    win._on_select(q)
    orig = pt.PointModelTest
    pt.PointModelTest = _FakeTest
    try:
        win.act_test_models.trigger()
        pump(0.2)
    finally:
        pt.PointModelTest = orig
    s0, s1 = p.sessions
    check("(d) Use gives the point the model in THIS camera", s0.points[q].tracker == "cotracker3",
          s0.points[q].tracker)
    check("(d) ... and leaves the other camera alone (R0-M11)", s1.points[s1.pid_by_name("snout")].tracker == "",
          s1.points[s1.pid_by_name("snout")].tracker)
    check("(d) ... and says so", "this camera" in toasts.text(), toasts.text()[:200])
    close(win)


# ================================================================= (e) 3D re-track skips a Hand-only point
def section_e():
    print("(e) 3D > Re-track Disagreeing Stretches")
    from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera
    from kinetrace.project import Project
    from kinetrace.session import TrackingSession
    RW, RH, RN = 640, 480, 60
    K = np.array([[700.0, 0, RW / 2], [0, 700.0, RH / 2], [0, 0, 1]])

    def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
        z = target - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        return R, -R @ pos
    cams = []
    for k, ang in enumerate((0.3, 2.4, 4.4)):
        pos = np.array([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.5 + 0.4 * k])
        R, t = look_at(pos)
        cams.append(CameraCalibration(dlt_from_camera(K, R, t), RW, RH, NoUndistort(), pixel_origin=0.0))

    def world_at(f):
        hip = np.array([0.3 * np.cos(f * 0.05), 0.3 * np.sin(f * 0.05), 0.003 * f])
        return np.stack([hip + [0.08, 0.0, 0.03], hip, hip - [0.09, 0.0, 0.01]])
    sessions, paths = [], []
    for c, cal in enumerate(cams):
        path = os.path.join(OUT, f"rig_cam{c}.mp4")
        frames = []
        s = TrackingSession(path, RN, 60.0, RW, RH)
        for nm in ("head", "hip", "tail"):
            s.add_landmark(nm)
        for f in range(RN):
            img = np.full((RH, RW, 3), 28, np.uint8)
            uv = cal.project(world_at(f))
            for j in range(3):
                cv2.circle(img, (int(round(uv[j][0])), int(round(uv[j][1]))), 5, (200, 200, 255), -1)
                s.set_position(f, j, float(np.round(uv[j][0])), float(np.round(uv[j][1])))
            s.manual[f] = False
            frames.append(img)
        write_video(path, frames, fps=60.0)
        sessions.append(s)
        paths.append(path)
    for f in range(20, 41):                                   # camera C's head slid 25 px off
        sessions[2].tracks[f, 0] += (25.0, 0.0)
    proj = Project(sessions, ["camA", "camB", "camC"], [0.0, 0.0, 0.0])
    proj.calibration = Calibration(cams)
    target = os.path.join(OUT, "rig.kinetrace")
    proj.save(target)
    forget_recovery(*paths)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_project_from_path(target)
    ok = wait(lambda: win.state == READY and win.project is not None and win.project.n_views == 3, 120)
    pump(0.3)
    check("(e setup) the calibrated 3-camera rig opens", ok)
    p = win.project
    win._set_active_view(2)
    pump(0.2)
    ok = set_tracker_here(win, 2, "head", "Hand only", "hand")
    check("(e setup) camC's head: Hand only (by the menu)", ok)
    win.act_recon.trigger()                            # 3D > Reconstruct 3D Landmarks (Ctrl+3)
    pump(0.5)
    check("(e setup) the 3D result is there", p.reconstruction is not None)
    ASK["asked"].clear()
    ASK["info"].clear()
    ASK["answer"] = QMessageBox.No                     # the old code offers to re-track it: decline
    n0 = len(LAUNCHED)
    win.act_retrack.trigger()
    pump(0.5)
    ASK["answer"] = QMessageBox.Yes
    said = " | ".join(t for _ti, t in ASK["asked"] + ASK["info"])
    check("(e) the re-track does not offer camC's Hand-only head", not ASK["asked"] and len(LAUNCHED) == n0
          and win._retrack is None, said[:300])
    check("(e) ... it names it as skipped: placed by hand only", "head" in said and "by hand only" in said,
          said[:300])
    win.project.dirty = False
    close(win)


# ================================================================= (f) the DeepLabCut scorer
def section_f():
    print("(f) the DeepLabCut scorer")
    win = new_window(CAMS[0])
    s = win.session
    for j, nm in enumerate(("a", "b")):
        q = new_point(win, TRUTH[j][0], nm)
        layers_menu(win, q, ["Tracker", "Hand only"])
        if s.points[q].tracker != "hand":
            s.points[q].tracker = "hand"
    out = os.path.join(OUT, "hand_dlc.csv")
    PICK["save"] = out
    PICK["filter"] = next(f[0] for f in win.EXPORT_FORMATS if f[2] == "dlc")
    win.act_export.trigger()
    wait(lambda: os.path.exists(out), 30)
    pump(0.3)
    first = open(out, encoding="utf-8").readline().strip() if os.path.exists(out) else ""
    check("(f) a camera whose points are all placed by hand exports the scorer Kinetrace_Hand",
          first.startswith("scorer,Kinetrace_Hand,"), first[:80])
    PICK["filter"] = ""
    close(win)


# ================================================================= (g) two cameras (the plan's hand check (e))
def section_g():
    print("(g) two cameras: a pass that only the other camera runs")
    win = new_window(CAMS[0])
    check("(g setup) a second camera", add_cameras(win, [CAMS[1]]))
    p = win.project
    win._set_active_view(0)
    pump(0.1)
    win.act_pm_cotracker.trigger()
    new_point(win, TRUTH[0][0], "snout")
    place(win, 1, "snout", 0, TRUTH[0][0])
    sa, sb = p.sessions
    vid = {os.path.basename(CAMS[v]): v for v in range(2)}
    toasts = Toasts(win)

    def press(a_key, b_key, label, **kw):
        set_tracker_here(win, 0, "snout", {"alltracker": "AllTracker", "cotracker3": "CoTracker3",
                                           "hand": "Hand only"}[a_key], a_key)
        set_tracker_here(win, 1, "snout", {"alltracker": "AllTracker", "cotracker3": "CoTracker3",
                                           "hand": "Hand only"}[b_key], b_key)
        win._set_active_view(0)
        pump(0.1)
        before = [cells(sv, sv.pid_by_name("snout")) for sv in p.sessions]
        RUNS.clear()
        toasts.msgs.clear()
        ran = track_press(win, **kw)
        got = sorted((vid.get(r["video"]), r["backend"]) for r in RUNS)
        return ran, got, before

    # hand check (e): the snout on AllTracker in camera 1 and CoTracker3 in camera 2, Every camera, T
    for a_key, b_key in (("alltracker", "cotracker3"), ("cotracker3", "alltracker")):
        ran, got, _b = press(a_key, b_key, "", every=True)
        check(f"(g) camera 1 {a_key}, camera 2 {b_key}: one press tracks the snout in both cameras, each with its own",
              ran and got == [(0, a_key), (1, b_key)]
              and last_tracked(sa, sa.pid_by_name("snout")) == N - 1
              and last_tracked(sb, sb.pid_by_name("snout")) == N - 1,
              (got, last_tracked(sa, sa.pid_by_name("snout")), last_tracked(sb, sb.pid_by_name("snout")),
               toasts.text()[:160]))
        if a_key == "alltracker":
            last = toasts.msgs[-1] if toasts.msgs else ""
            check("(g) the end notice says which camera the CoTracker3 pass ran in (R1-M2)",
                  f"to frame {N - 1} in {p.name(1)}" in last, last[:200])
        ctrl_z(win)
        pump(0.3)
    # Hand only here, tracked in the other camera: Every camera (and Shift+T) runs it there, leaves it here
    for kw, how in (({"every": True}, "Every camera, T"), ({"every": False, "mods": Qt.ShiftModifier}, "Shift+T")):
        ran, got, before = press("hand", "alltracker", "", **kw)
        check(f"(g) Hand only in camera 1, AllTracker in camera 2, {how}: camera 2 tracks it, camera 1 leaves it",
              ran and got == [(1, "alltracker")] and last_tracked(sb, sb.pid_by_name("snout")) == N - 1
              and same_cells(sa, sa.pid_by_name("snout"), before[0]),
              (ran, got, last_tracked(sb, sb.pid_by_name("snout")), toasts.text()[:160]))
        if ran:
            ctrl_z(win)
            pump(0.3)
    # a pass that cannot start anywhere is named, with the model it is
    set_tracker_here(win, 0, "snout", "CoTracker3", "cotracker3")
    set_tracker_here(win, 1, "snout", "AllTracker", "alltracker")
    q = sb.pid_by_name("snout")
    kept = cells(sb, q)
    sb.clear_window([q], 0, 0)                              # camera 2 has no snout on its frame of this instant
    win._set_active_view(0)
    pump(0.1)
    toasts.msgs.clear()
    track_press(win, every=True)
    check("(g) a pass that cannot start anywhere is named (snout, the AllTracker pass)",
          "snout" in toasts.text() and "AllTracker" in toasts.text() and "no position" in toasts.text(),
          toasts.text()[:200])
    ctrl_z(win)
    pump(0.3)
    for a in POINT_ARRAYS:
        getattr(sb, a.name)[:, q] = kept[a.name]
    # (R1-M1) the AllTracker pass runs in camera 1 for a Moving spot point alone: the snout (CoTracker3 here,
    # AllTracker in camera 2 where it has no position) must not ride in it with the project's default model
    win._set_active_view(0)
    pump(0.1)
    d = new_point(win, TRUTH[1][0], "dot")
    set_tracker_here(win, 0, "dot", "Moving spot", "spot")
    set_tracker_here(win, 0, "snout", "CoTracker3", "cotracker3")
    set_tracker_here(win, 1, "snout", "AllTracker", "alltracker")
    q = sb.pid_by_name("snout")
    kept = cells(sb, q)
    sb.clear_window([q], 0, 0)
    win._set_active_view(0)
    pump(0.1)
    RUNS.clear()
    track_press(win, every=True)
    got = [(vid.get(r["video"]), r["backend"], sorted(sa.points[i].name for i in r["pids"])) for r in RUNS]
    check("(g) a point of the other model never rides in a pass with the project's model: the snout runs once, "
          "with CoTracker3 (R1-M1)", got == [(0, "cotracker3", ["snout"])]
          and last_tracked(sa, sa.pid_by_name("dot")) == N - 1, (got, last_tracked(sa, d)))
    ctrl_z(win)
    pump(0.3)
    for a in POINT_ARRAYS:
        getattr(sb, a.name)[:, q] = kept[a.name]
    win.act_track_all.setChecked(False)
    close(win)


for fn in (section_a, section_b, section_c, section_d, section_e, section_f, section_g):
    section(fn)

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("  - " + f.encode("ascii", "backslashreplace").decode("ascii"))
    sys.exit(1)
print("verify_tracker_per_camera PASSED")
