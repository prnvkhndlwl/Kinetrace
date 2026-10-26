"""Save, open and recovery safety (docs/AUDIT.md I326-I334, the 2026-10-25 full sweep's PR 1). Every section
FAILS on main 6039022 and makes the user's gesture through the real window (offscreen, no GPU, every dialog
stubbed), except the fault-injection loop of I326, which drives projectfile's commit step by step:

  I326  a case-only rename (snout -> Snout) and a save that fails or is cut off keeps snout.csv
        (Ctrl+S with another landmark's file held open; the k-th move of every commit raising / a power cut
        followed by finish_interrupted, for a landmark and for a camera folder)
  I327  a landmark whose file is missing opens with a notice naming it; Save makes no empty file for it
  I328  notes / events outside the video open with a notice naming the rows; Save keeps the rows as they are
  I329  File > Open of a damaged project with Discard keeps the open project's recovery copy
  I330  no recovery copy is written while a Save writes (1 ms autosave ticks, many Ctrl+S)
  I331  Cancel on the close question does not leave "the user pressed X" set
  I332  Help > Add Kinetrace to the Start menu leaves the settings and recovery folders where they were
  I333  the recovery copy written at the end of an auto-paused run holds the data after the cut
  I334  the auto-pause cut keeps the user's "hidden here" marks

.venv\\Scripts\\python.exe tests\\verify_save_safety.py
"""
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")
OUT = os.path.join(HERE, "out", "save_safety")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

from kinetrace import projectfile as pf  # noqa: E402
from kinetrace import recovery  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import Event, TrackingSession  # noqa: E402

FAILS = []
EXC = []
LOG = {"question": [], "warning": [], "critical": [], "info": []}
ASK = {"answer": QMessageBox.Yes, "open": "", "save": "", "text": ""}
RNG = np.random.default_rng(7)


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def section(title):
    """Run the function below as a section: an exception is a failed check, not a crash."""
    def deco(fn):
        print(f"\n[{title}]", flush=True)
        try:
            fn()
        except Exception as e:      # noqa: BLE001 - on the old code a section may end here
            import traceback
            check(False, f"{title}: raised {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc(limit=4)
        return fn
    return deco


def _question(*a, **k):
    title = str(a[1]) if len(a) > 1 else ""
    LOG["question"].append(title)
    if title == "Save changes?":
        return ASK["answer"]
    if title == "Unsaved changes found":
        return QMessageBox.No
    return QMessageBox.Yes


def _note(kind):
    def f(*a, **k):
        LOG[kind].append(" | ".join(str(x) for x in a[1:3]))
        return QMessageBox.Ok
    return staticmethod(f)


QMessageBox.question = staticmethod(_question)
QMessageBox.information = _note("info")
QMessageBox.warning = _note("warning")
QMessageBox.critical = _note("critical")
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (ASK["save"], ""))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: OUT)
QInputDialog.getText = staticmethod(lambda *a, **k: (ASK["text"], True))
app = QApplication.instance() or QApplication([])
sys.excepthook = lambda t, v, tb: EXC.append(f"{t.__name__}: {v}")
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def clip(name, n=60):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    base = RNG.integers(0, 255, (280, 320 + 4 * n, 3)).astype(np.uint8)
    for f in range(n):
        vw.write(np.ascontiguousarray(base[20:260, 4 * f:4 * f + 320]))
    vw.release()
    return path


def saved_project(name, points=("snout", "tail"), n=60, notes=None, events=None):
    """A saved one-camera project on a real clip, every landmark tracked on every frame."""
    s = TrackingSession(clip(name + ".mp4", n), n, 30.0, 320, 240)
    for j, pn in enumerate(points):
        s.add_point(0, 40.0 + 20 * j, 60.0, name=pn)
    s.tracks[:] = RNG.uniform(10, 230, (n, s.n_points, 2)).astype(np.float32)
    s.tracked[:] = True
    s.visibility[:] = True
    s.confidence[:] = 0.9
    for f, text in (notes or {}).items():
        s.notes[f] = {"text": text, "author": "", "time": ""}
    s.events.extend(events or [])
    path = os.path.join(OUT, name + ".kinetrace")
    pf.save(Project([s]), path)
    return path


def tracks_dir(path):
    cams = [d for d in (Path(path) / "cameras").iterdir() if d.is_dir()]
    return cams[0] / "tracks"


def settle():
    for _ in range(400):
        pump(0.05)
        if w.state == READY and w.project is not None and not w._loading:
            break
    pump(0.3)


def open_project(path, answer=QMessageBox.Yes):
    """File > Open Project (its dialog stubbed to `path`); `answer` = the close question's, if asked."""
    ASK.update(answer=answer, open=os.path.join(path, "kinetrace.json"))
    w.act_open_proj.trigger()
    settle()


def ctrl_s():
    w.activateWindow()
    w.setFocus()
    QTest.keyClick(w, Qt.Key_S, Qt.ControlModifier)
    pump(0.3)
    w._finish_writer()
    pump(0.1)


def clear_log():
    for v in LOG.values():
        v.clear()


# ================================================================== I326, projectfile level
def mini(points=("a", "snout"), T=40):
    s = TrackingSession(os.path.join(OUT, "videos", "cam1.mp4"), T, 30.0, 640, 480)
    for j, pn in enumerate(points):
        s.add_point(0, 10.0 + j, 20.0 + j, name=pn)
    s.tracks[:] = RNG.uniform(0, 600, (T, s.n_points, 2)).astype(np.float32)
    s.tracked[:] = True
    s.visibility[:] = True
    return Project([s], ["cam1"])


def state(p):
    out = {}
    for name, s in zip(p.names, p.sessions):
        for j, pt in enumerate(s.points):
            out[(name.casefold(), pt.name.casefold())] = (s.tracked[:, j].copy(),
                                                          np.nan_to_num(s.tracks[:, j].astype(np.float64), nan=-1.0))
    return out


def same_state(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1])
                                        for k in a)


def faulted_save(p, root, k, mode):
    """pf.save with the k-th file move of its commit failing: mode "raise" = an error (the commit's own
    rollback runs), "crash" = a power cut (no rollback; .saving/ and .history/pending.json stay behind for
    the next open). -> ("saved" | "failed", moves counted)."""
    calls = {"n": 0, "in": False, "crashed": False}
    real = (pf._commit, pf._move_aside, pf._replace, pf._rollback, shutil.rmtree)

    def commit(*a, **kw):
        calls["in"] = True
        try:
            return real[0](*a, **kw)
        finally:
            calls["in"] = False

    def hit():
        if calls["in"]:
            calls["n"] += 1
            if calls["n"] == k:
                if mode == "crash":
                    calls["crashed"] = True
                    raise SystemExit("power cut")
                raise PermissionError("held open by another program")

    def move(src, dst):
        hit()
        return real[1](src, dst)

    def rep(src, dst):
        hit()
        return real[2](src, dst)

    def rmtree(path, *a, **kw):
        if calls["crashed"] and Path(path).name == pf.SAVING_DIR:
            return                      # a power cut leaves .saving/ behind
        return real[4](path, *a, **kw)
    pf._commit, pf._move_aside, pf._replace, shutil.rmtree = commit, move, rep, rmtree
    if mode == "crash":
        pf._rollback = lambda root, pending: None
    try:
        pf.save(p, root)
        return "saved", calls["n"]
    except (PermissionError, SystemExit):
        return "failed", calls["n"]
    finally:
        pf._commit, pf._move_aside, pf._replace, pf._rollback, shutil.rmtree = real


@section("I326 a case-only rename survives a save that fails at any step (fault injection)")
def _i326_loop():
    for kind in ("landmark", "camera"):
        for mode in ("raise", "crash"):
            n_steps, bad, k = None, [], 1
            while n_steps is None or k <= n_steps:
                root = os.path.join(OUT, f"fault_{kind}_{mode}_{k}.kinetrace")
                p = mini()
                pf.save(p, root)
                before = state(pf.load(root))
                if kind == "landmark":
                    p.rename_landmark("snout", "Snout")
                else:
                    p.names[0] = "Cam1"
                after = state(p)
                if n_steps is None:             # how many moves this commit makes (an unfaulted dry count)
                    probe = os.path.join(OUT, f"fault_{kind}_{mode}_count.kinetrace")
                    q = mini()
                    pf.save(q, probe)
                    if kind == "landmark":
                        q.rename_landmark("snout", "Snout")
                    else:
                        q.names[0] = "Cam1"
                    n_steps = faulted_save(q, probe, 10 ** 9, "raise")[1]
                got, _n = faulted_save(p, root, k, mode)
                if mode == "crash":
                    pf.finish_interrupted(root)
                try:
                    now = state(pf.load(root))
                except Exception as e:  # noqa: BLE001
                    bad.append(f"step {k}: reopen raised {type(e).__name__}")
                    k += 1
                    continue
                want = before if got == "failed" else after
                if not same_state(now, want):
                    lost = [key for key in want if key not in now or not np.array_equal(now[key][0], want[key][0])]
                    bad.append(f"step {k} ({got}): {lost or 'positions differ'}")
                k += 1
            check(n_steps > 2 and not bad, f"{kind} rename by case, {mode}: every one of {n_steps} commit steps "
                  "leaves the project as the save before (or as the new one once it counted)", "; ".join(bad[:4]))


# ================================================================== the window
w = MainWindow()
w.resize(1400, 900)
w.show()
pump(0.5)


@section("I326 rename snout -> Snout + Ctrl+S with another landmark's file held open keeps snout.csv")
def _i326_gui():
    path = saved_project("i326", ("a", "snout"))
    tracks = tracks_dir(path)
    before = (tracks / "snout.csv").read_bytes()
    open_project(path)
    s = w.session
    w._on_select(s.pid_by_name("a"))
    w.activateWindow()
    QTest.keyClick(w, Qt.Key_Delete)            # delete landmark a (its file goes on the next save)
    pump(0.2)
    check(s.pid_by_name("a") is None, "the Delete key removed landmark a")
    ASK["text"] = "Snout"
    w._on_rename(s.pid_by_name("snout"))        # the Rename point dialog, answered "Snout"
    pump(0.1)
    clear_log()
    with open(tracks / "a.csv", "rb"):          # Windows refuses to move a file this process holds open
        ctrl_s()
    check(bool(LOG["critical"]), "the failed save said so", str(LOG))
    left = [n for n in os.listdir(tracks) if n.lower() == "snout.csv"]
    check(bool(left) and (tracks / left[0]).read_bytes() == before,
          "snout.csv is still on disk, byte-identical to the last save", f"{sorted(os.listdir(tracks))}")
    q = pf.load(path).sessions[0]
    j = next((i for i, x in enumerate(q.points) if x.name.lower() == "snout"), None)
    check(j is not None and int(q.tracked[:, j].sum()) == 60, "reopened: snout has all 60 frames",
          f"{[x.name for x in q.points]}")


@section("I327 a landmark whose file is missing opens with a notice; Save makes no empty file for it")
def _i327():
    path = saved_project("i327", ("snout", "tail"))
    tracks = tracks_dir(path)
    os.remove(tracks / "tail.csv")
    shutil.rmtree(Path(path) / ".cache", ignore_errors=True)
    clear_log()
    open_project(path)
    said = [t for t in LOG["warning"] + LOG["info"] if "tail.csv" in t]
    check(bool(said) and "tail" in said[0], "the notice names the landmark and its missing file", str(LOG))
    w.session.dirty = True
    ctrl_s()
    check(not (tracks / "tail.csv").exists(), "Ctrl+S did not create an empty tail.csv",
          f"{sorted(os.listdir(tracks))}")
    check((tracks / "snout.csv").exists(), "snout.csv is still written")
    pid = w.session.pid_by_name("tail")
    w.session.set_position(5, pid, 100.0, 120.0)
    w.session.dirty = True
    ctrl_s()
    q = pf.load(path).sessions[0]
    j = next((i for i, x in enumerate(q.points) if x.name == "tail"), None)
    check((tracks / "tail.csv").exists() and j is not None and bool(q.tracked[5, j]),
          "once tail has a position, Ctrl+S writes its file", f"{sorted(os.listdir(tracks))}")


@section("I328 notes / events outside the video open with a notice naming the rows; Save keeps them")
def _i328():
    path = saved_project("i328", ("snout",), notes={5: "tail down"},
                         events=[Event("glide", 3, 8, (255, 200, 0), "", "")])
    cam = tracks_dir(path).parent
    notes, events = cam / "notes.csv", cam / "events.csv"
    nt = notes.read_text(encoding="utf-8")
    et = events.read_text(encoding="utf-8")
    nt2 = re.sub(r"(?m)^5,", "500,", nt, count=1)
    et2 = re.sub(r"(?m)^glide,3,8,", "glide,-5,900,", et, count=1)
    check(nt2 != nt and et2 != et, "set-up: the note moved to frame 500 and the event to -5 .. 900 by hand")
    notes.write_text(nt2, encoding="utf-8", newline="")
    events.write_text(et2, encoding="utf-8", newline="")
    nb, eb = notes.read_bytes(), events.read_bytes()
    shutil.rmtree(Path(path) / ".cache", ignore_errors=True)
    clear_log()
    n_exc = len(EXC)
    open_project(path)
    said = " ".join(LOG["warning"] + LOG["info"])
    check("notes.csv row 2" in said and "events.csv row 2" in said, "the notice names both rows", said[:300])
    w.activateWindow()
    for key in (Qt.Key_End, Qt.Key_Home, Qt.Key_Right, Qt.Key_Left):
        QTest.keyClick(w, key)
        pump(0.05)
    w.session.dirty = True
    ctrl_s()                                    # the save also writes the project's exports/
    check(notes.read_bytes() == nb and events.read_bytes() == eb, "Ctrl+S kept both rows exactly as they were")
    check(len(EXC) == n_exc, "no exception in the timeline, the save or its exports", str(EXC[n_exc:])[:300])


@section("I329 File > Open of a damaged project with Discard keeps the open project's recovery copy")
def _i329():
    a = saved_project("i329a")
    c = saved_project("i329c")
    b = os.path.join(OUT, "i329b.kinetrace")
    os.makedirs(b, exist_ok=True)
    with open(os.path.join(b, "kinetrace.json"), "w", encoding="utf-8") as fh:
        fh.write("{")
    open_project(a)
    w.session.set_position(3, 0, 90.0, 90.0)
    w.session.dirty = True
    w._autosave(wait=True)
    pump(0.3)
    pid = w._project_id
    check(recovery.find(pid) is not None, "set-up: the open project's unsaved work has its recovery copy")
    clear_log()
    open_project(b, answer=QMessageBox.Discard)
    check(any("Could not open project" in t for t in LOG["critical"]), "the damaged project could not be opened",
          str(LOG["critical"])[:200])
    check(w._project_id == pid and w.project.dirty, "the old project is still open, still unsaved")
    check(recovery.find(pid) is not None, "its recovery copy is still there")
    open_project(c, answer=QMessageBox.Discard)
    check(w._project_id != pid and recovery.find(pid) is None,
          "control: opening a good project with Discard does drop the old work")
    w.session.set_position(3, 0, 91.0, 91.0)
    w.session.dirty = True
    w._autosave(wait=True)
    pump(0.3)
    pc = w._project_id
    clear_log()
    open_project(c, answer=QMessageBox.Discard)        # the same project again, its changes discarded
    check("Unsaved changes found" not in LOG["question"] and w.project is not None and not w.project.dirty
          and recovery.find(pc) is None, "re-opening the same project with Discard does not offer the work back",
          f"{LOG['question']}")


@section("I330 no recovery copy is written while a Save writes")
def _i330():
    path = saved_project("i330")
    open_project(path, answer=QMessageBox.Discard)
    pid = w._project_id
    hits, n_ticks = [], [0]

    def tick():
        n_ticks[0] += 1
        w._autosave()
    tm = QTimer()
    tm.setInterval(1)
    tm.timeout.connect(tick)
    tm.start()
    N = 120
    for i in range(N):
        w.session.set_position(0, 0, 50.0 + i * 0.01, 60.0)      # an edit: unsaved, a new signature
        w.session.dirty = True
        w._recovery_sig = None
        recovery.discard(pid)
        w.activateWindow()
        QTest.keyClick(w, Qt.Key_S, Qt.ControlModifier)
        stats = getattr(w._save_worker, "stats", "missing")
        clean = not w.project.dirty
        pump(0.03)
        w._finish_writer()
        pump(0.01)
        if clean and (recovery.find(pid) is not None or stats is None):
            hits.append(i)
    tm.stop()
    check(n_ticks[0] > N and not hits, f"{N} saves under 1 ms autosave ticks: none left a recovery copy or lost "
          "its 'files updated' line", f"{len(hits)} did: {hits[:8]}")


class FakeWorker:
    """What `_on_track_finished_run` reads from a finished TrackingWorker (no thread, no model): the end of a
    run that auto-paused is handed to the app's own end-of-run code, as the worker's finished signal does.
    (No CPU tracker here leaves unreliable frames after the frame it stops at, so a real run has no cut.)"""

    def __init__(self, pids, last):
        self.point_ids = list(pids)
        self._autopause_reason = "lost"
        self._ball_ended, self._spot_ended, self._seg_ended = {}, {}, {}
        self.animals = []
        self.resume_cuts = {}
        self.emitted_to = int(last)
        self._kt_restarts = 0

    def isRunning(self):
        return False

    def wait(self, ms=0):
        return True


def fill_run(s, pid, f0, f1):
    s.tracks[f0:f1 + 1, pid] = np.stack([50.0 + np.arange(f1 - f0 + 1), np.full(f1 - f0 + 1, 60.0)], 1)
    s.tracked[f0:f1 + 1, pid] = True
    s.visibility[f0:f1 + 1, pid] = True
    s.confidence[f0:f1 + 1, pid] = 0.9
    s.manual[f0:f1 + 1, pid] = False
    s._touch()


def end_run_with_autopause(pid, fail, last):
    w.worker = FakeWorker([pid], last)
    w.state = TRACKING
    w._autopause_info = (int(fail), int(pid))
    w._run_keep = True
    w._run_hand = None
    w._on_track_finished(int(last), True)
    pump(0.4)


@section("I331 Cancel on the close question does not leave 'the user pressed X' set")
def _i331():
    path = saved_project("i331")
    open_project(path, answer=QMessageBox.Discard)
    w.session.set_position(2, 0, 70.0, 70.0)
    w.session.dirty = True
    ASK["answer"] = QMessageBox.Cancel
    clear_log()
    w.close()
    pump(0.3)
    check("Save changes?" in LOG["question"] and w.isVisible(), "the close question was asked; Cancel kept the window")
    check(w._user_paused is False, "no pause is left pending after Cancel", f"_user_paused={w._user_paused}")
    got = []
    real = w._passes_record
    w._passes_record = lambda res, fail, user: got.append(bool(user))
    w._passes = {"frame": 0}
    s = w.session
    pid = s.pid_by_name("snout")
    try:
        end_run_with_autopause(pid, 15, 29)
    finally:
        w._passes_record = real
        w._passes = None
    check(got == [False], "a two-pass run that then auto-pauses is recorded as an auto-pause, not as your stop",
          f"user_stop={got}")


@section("I333 / I334 the end of an auto-paused run: recovery copy after the cut, hidden marks kept")
def _i333():
    path = saved_project("i333")
    open_project(path, answer=QMessageBox.Discard)
    s = w.session
    pid = s.pid_by_name("tail")
    s.tracked[:, pid] = False
    fill_run(s, pid, 0, 29)
    s.set_occluded_window([pid], 20, 24, True)          # the user: "hidden here" on frames 20-24
    w.session.dirty = True
    w._recovery_sig = None
    end_run_with_autopause(pid, 15, 29)                  # the run lost tail at frame 15, emitted up to 29
    w._finish_writer()
    pump(0.3)
    check(int(s.tracked[15:30, pid].sum()) == 0, "set-up: the cut removed frames 15-29 from the session")
    check(int(s.occluded[20:25, pid].sum()) == 5, "I334: the 5 hidden marks are still there after the cut",
          f"{int(s.occluded[20:25, pid].sum())} of 5")
    zp = recovery.paths(w._project_id)[0]
    q = pf.load(str(zp)).sessions[0]
    check(int(q.tracked[15:30, pid].sum()) == 0, "I333: the recovery copy on disk holds the data after the cut",
          f"{int(q.tracked[15:30, pid].sum())} cut cells in the copy")
    check(int(q.occluded[20:25, pid].sum()) == 5, "I334: ... and the hidden marks")


START_MENU_CHILD = r'''
import os, sys
root, fake = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from PySide6.QtWidgets import QApplication, QMessageBox
app = QApplication([sys.argv[0], "-platform", "minimal"])
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
for name in ("information", "warning", "critical"):
    setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
from kinetrace import recovery, winapp
from kinetrace.app import MainWindow
real = winapp.add_start_menu
winapp.add_start_menu = lambda r=None: real(fake)      # this folder's own files are never written
w = MainWindow()
before = (os.environ.get("QT_QPA_PLATFORM"), str(recovery.settings_path()), str(recovery.folder()[0]))
w.act_start_menu.trigger()
app.processEvents()
print("TRIGGERED")
after = (os.environ.get("QT_QPA_PLATFORM"), str(recovery.settings_path()), str(recovery.folder()[0]))
print("before", before)
print("after ", after)
print("SAME" if before == after else "MOVED")
sys.stdout.flush()
os._exit(0 if before == after else 1)
'''


@section("I332 Help > Add Kinetrace to the Start menu leaves settings and recovery where they were")
def _i332():
    fake = os.path.join(OUT, "fake_install")
    os.makedirs(os.path.join(fake, ".venv"), exist_ok=True)
    child = os.path.join(OUT, "start_menu_child.py")
    with open(child, "w", encoding="utf-8") as fh:
        fh.write(START_MENU_CHILD)
    env = {k: v for k, v in os.environ.items()
           if k not in ("QT_QPA_PLATFORM", "KINETRACE_RECOVERY_DIR", recovery.SETTINGS_ENV)}
    env["KINETRACE_START_MENU_DIR"] = os.path.join(OUT, "start_menu")     # never the user's real Start menu
    r = subprocess.run([sys.executable, child, ROOT, fake], env=env, capture_output=True, text=True, timeout=180)
    out = (r.stdout or "") + (r.stderr or "")[-600:]
    print("    " + "\n    ".join((r.stdout or "").strip().splitlines()))
    check("TRIGGERED" in out, "the child window ran the menu entry", out[-400:])
    check(r.returncode == 0 and "SAME" in out, "the platform setting, the settings file and the recovery folder "
          "are unchanged after the menu entry", out[-400:])




ASK["answer"] = QMessageBox.Discard
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_save_safety: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
