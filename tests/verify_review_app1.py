"""Code review 2026-10-02, the app.py region 1 fixes (I166, I167, I188, I206, I223, G70, G71, G92,
G104, G105, G109, G120, I233, I234, I259, G132, G133, G134, G139, I258, M9, G68 part, R11), each
with a check that FAILS on the old code. Offscreen MainWindow, dialogs stubbed, REAL key / mouse
events where the finding is about input. No GPU, no model.

`KT_PKG_ROOT=<folder>` runs the same checks against another tree's `kinetrace` package (how
"fails on old" was proved: a `git archive` of the base commit). Each section reports its own
failures and the script goes on, so one run on the old code lists everything it gets wrong.
`KT_ONLY=a,b` runs only those sections.

Run: .venv\\Scripts\\python.exe tests\\verify_review_app1.py
"""
import os
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("KT_PKG_ROOT", ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(errors="replace")

OUT = os.path.join(ROOT, "tests", "out", "review_app1")
if os.path.isdir(OUT):
    shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("KINETRACE_RECOVERY_DIR", os.path.join(OUT, "recovery"))
os.environ.setdefault("KINETRACE_LOG_DIR", os.path.join(OUT, "logs"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QMenu,  # noqa: E402
                               QMessageBox, QPlainTextEdit)

ASK = {"answer": QMessageBox.Yes, "close": QMessageBox.Discard, "asked": [], "warned": [], "texts": []}


def _question(*a, **k):
    title = a[1] if len(a) > 1 else ""
    ASK["asked"].append(title)
    ASK["texts"].append(str(a[2]) if len(a) > 2 else "")
    if title == "Save changes?":
        return ASK["close"]
    return ASK["answer"]


def _warning(*a, **k):
    ASK["warned"].append((a[1] if len(a) > 1 else "", str(a[2]) if len(a) > 2 else ""))
    return QMessageBox.Ok


QMessageBox.question = staticmethod(_question)
QMessageBox.warning = staticmethod(_warning)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))

app = QApplication.instance() or QApplication([])
EXC: list = []
sys.excepthook = lambda t, v, tb: EXC.append("".join(traceback.format_exception(t, v, tb))[-400:])

from kinetrace import app as appmod, projectfile, recovery, skeletons  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond, detail="") -> None:
    print(("ok    " if cond else "FAIL  ") + name + ("" if cond else f"   [{detail}]"), flush=True)
    if not cond:
        FAILS.append(name)


def section(fn):
    only = [n for n in os.environ.get("KT_ONLY", "").split(",") if n]
    if only and fn.__name__ not in only:
        print(f"skip  {fn.__name__}")
        return fn
    try:
        fn()
    except Exception as e:      # noqa: BLE001
        FAILS.append(fn.__name__)
        print(f"FAIL  {fn.__name__}: raised {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(limit=3)
    return fn


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def until(cond, sec=15.0) -> bool:
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


def clip(path, n=60, w=320, h=240, shift=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.RandomState(3)
    base = rng.randint(0, 255, (h + 40, w + 600, 3)).astype(np.uint8)
    for f in range(n):
        x = 4 * (f + shift)
        vw.write(np.ascontiguousarray(base[20:20 + h, x:x + w]))
    vw.release()
    return path


A = clip(os.path.join(OUT, "vid", "camA.mp4"), 90)
B = clip(os.path.join(OUT, "vid", "camB.mp4"), 60, shift=5)
C = clip(os.path.join(OUT, "vid", "camC.mp4"), 60, shift=9)


def forget(*videos):
    want = {os.path.normcase(os.path.abspath(str(v))) for v in videos}
    for info in recovery.scan():
        if any(os.path.normcase(os.path.abspath(v)) in want for v in info.get("videos", [])):
            recovery.discard(info["project_id"])


def window(video=None, extra=(), then=None):
    forget(A, B, C)
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    if video:
        w._open_video(video, then=then)
        assert until(lambda: w.state == READY and w.project is not None, 30), "the video did not open"
        pump(0.3)
        for e in extra:
            assert w._add_view(e), "could not add " + str(e)
        w._refresh_companions()                   # (the Add video button does this after _add_view)
        pump(0.4)
    return w


def close(w):
    ASK["close"] = QMessageBox.Discard
    w.close()
    pump(0.3)


def click(w, x, y, button=Qt.LeftButton):
    vp = w.canvas.viewport()
    QTest.mouseClick(vp, button, Qt.NoModifier, w.canvas.mapFromScene(QPointF(x, y)))
    pump(0.05)


def place_points(s, names, frames=range(10), x0=40.0):
    """Hand-placed points on `frames`, one per name."""
    out = []
    for k, nm in enumerate(names):
        pid = s.add_point(frames[0], x0 + 20 * k, 50.0, name=nm)
        for f in frames:
            s.set_position(f, pid, x0 + 20 * k + f, 50.0 + f)
        out.append(pid)
    return out


# ================================================================== G104
@section
def g104_error_hint():
    tb = ('Traceback (most recent call last):\n  File "kinetrace/balls.py", line 403, in step\n'
          '    x = crop[y0:y1]\nValueError: could not broadcast input array from shape (3,4) into shape (5,4)')
    check("G104 a stack frame at line 403 is not an HTTP 403", appmod._model_error_hint(tb) == "",
          appmod._model_error_hint(tb)[:60])
    tb2 = ('Traceback (most recent call last):\n  File "kinetrace/spots.py", line 12, in setup\n'
           '    download_blob()\nRuntimeError: bad value for setup (download key missing)')
    check("G104 the word 'download' in an echoed line is not a failed download",
          appmod._model_error_hint(tb2) == "", appmod._model_error_hint(tb2)[:60])
    gated = ('Traceback (most recent call last):\n  File "x.py", line 1, in f\n    g()\n'
             'huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo')
    check("G104 a gated repository is still named", "gated on Hugging Face" in appmod._model_error_hint(gated))
    oom = 'Traceback...\n  File "a.py", line 5\n    m()\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB'
    check("G104 out of memory is still named", "ran out of memory" in appmod._model_error_hint(oom))
    net = ('Traceback (most recent call last):\n  File "u.py", line 9, in g\n    urlopen(x)\n'
           'urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>')
    check("G104 a network failure is still named", "could not be downloaded" in appmod._model_error_hint(net))


# ================================================================== R4 / follow-ups (a) (b) (c)
@section
def followups_signals():
    from kinetrace.canvas import VideoCanvas
    from kinetrace.video_source import SeekService
    w = window(A, [B])
    check("(a) the two canvas signals nobody emits are gone",
          not hasattr(VideoCanvas, "point_moved") and not hasattr(VideoCanvas, "move_committed"))
    check("(b) the duplicate camera-switch handler is gone", not hasattr(w, "_on_canvas_clicked"))
    check("(c) the unreachable eof_truncated path is gone",
          not hasattr(w, "_on_eof_truncated") and not hasattr(SeekService, "eof_truncated"))
    # a real click on the companion still switches to it (viewgrid.view_activated)
    cv1 = w.grid.canvas(1)
    QTest.mouseClick(cv1.viewport(), Qt.LeftButton, Qt.NoModifier, cv1.viewport().rect().center())
    pump(0.3)
    check("(b) a real click on a companion camera switches to it", w.project.active == 1, w.project.active)
    check("R11 the decoded-frame bookkeeping lives on the runtime", hasattr(w._views[0], "bad_frames"))
    close(w)


# ================================================================== I223
@section
def i223_unique_names():
    d = os.path.join(OUT, "rig")
    p1 = clip(os.path.join(d, "cam1", "GX010001.mp4"), 30)
    p2 = clip(os.path.join(d, "cam2", "GX010001.mp4"), 30, shift=3)
    p3 = clip(os.path.join(d, "cam3", "GX010001.mp4"), 30, shift=6)
    w = window(A)
    w._import_folder([p1, p2, p3], None)
    assert until(lambda: w.project is not None and w.project.n_views == 3 and not w._busy_stack, 30)
    names = list(w.project.names)
    check("I223 three cameras with one file name get three names", len(set(n.lower() for n in names)) == 3, names)
    check("I223 the parent folders tell them apart", names == ["cam1", "cam2", "cam3"], names)
    # adding one by one: a clash gets a suffix or its folder
    q = clip(os.path.join(OUT, "other", "camA.mp4"), 30)
    w2 = window(A)
    assert w2._add_view(q)
    check("I223 a camera added with the same file name is named differently",
          len(set(n.lower() for n in w2.project.names)) == 2, w2.project.names)
    close(w)
    close(w2)


# ================================================================== I166
@section
def i166_import_over_project():
    d = os.path.join(OUT, "imp")
    p1 = clip(os.path.join(d, "a.mp4"), 30)
    p2 = clip(os.path.join(d, "b.mp4"), 30, shift=3)
    target = os.path.join(d, "imp.kinetrace")
    w = window(A)
    ASK["asked"].clear()
    w._import_folder([p1, p2], target)
    assert until(lambda: w.project is not None and w.project.n_views == 2 and not w._busy_stack, 30)
    check("I166 the first import writes the project", projectfile.is_project(target))
    s = w.project.sessions[0]
    place_points(s, ["keep_me"], range(5))
    s.dirty = True
    assert w._save_project()
    n_before = len(projectfile.read(target)[0].sessions[0].points)
    check("I166 a point is saved in the project", n_before == 1, n_before)
    # import the same folder again: a project is already there
    ASK["asked"].clear()
    ASK["answer"] = QMessageBox.No
    w._import_folder([p1, p2], target)
    assert until(lambda: w.project is not None and w.project.n_views == 2 and not w._busy_stack, 30)
    pump(0.5)
    asked = list(ASK["asked"])
    ASK["answer"] = QMessageBox.Yes
    after = projectfile.read(target)[0].sessions[0]
    check("I166 re-importing over an existing project asks first", "Replace project?" in asked, asked)
    check("I166 answering No leaves the saved project alone", len(after.points) == 1,
          [p.name for p in after.points])
    check("I166 the import stays open, unsaved", w.project_path is None)
    close(w)


# ================================================================== I167
@section
def i167_video_recovery_camera():
    forget(A, B)
    w1 = MainWindow()
    w1.resize(1200, 800)
    w1.show()
    w1._open_video(A)
    assert until(lambda: w1.state == READY and w1.project is not None, 30)
    assert w1._add_view(B)
    pump(0.4)
    sa, sb = w1.project.sessions
    place_points(sa, ["pa"], range(10), 40.0)
    place_points(sb, ["pb"], range(20, 30), 100.0)
    w1.project.sync_landmarks()
    w1.project.names[0], w1.project.names[1] = "front", "side"
    w1.project.dirty = True
    w1._autosave(wait=True)
    pump(0.5)
    check("I167 the first window wrote a recovery for both videos",
          any(B in (i.get("videos") or []) for i in recovery.scan()) or
          any(os.path.normcase(os.path.abspath(B)) == os.path.normcase(os.path.abspath(v))
              for i in recovery.scan() for v in i.get("videos", [])), recovery.scan())
    # a second window opens ONLY camera B's video (60 frames; camera A has 90)
    ASK["asked"].clear()
    ASK["answer"] = QMessageBox.Yes
    w2 = MainWindow()
    w2.resize(1200, 800)
    w2.show()
    w2._open_video(B)
    assert until(lambda: w2.state == READY and w2.project is not None, 40)
    pump(0.5)
    asked = list(ASK["asked"])
    p = w2.project
    check("I167 the recovery was offered (camera B's frame count compared, not camera A's)",
          "Restore unsaved work?" in asked, (asked, ASK["warned"][-2:]))
    check("I167 both cameras came back", p.n_views == 2, p.n_views)
    if p.n_views == 2:
        pb = p.sessions[p.active].pid_by_name("pb")
        check("I167 the opened video is the active camera, with ITS tracks",
              os.path.normcase(os.path.abspath(p.sessions[p.active].video_path)) ==
              os.path.normcase(os.path.abspath(B)) and pb is not None
              and int(p.sessions[p.active].tracked[:, pb].sum()) == 10,
              (p.active, p.sessions[p.active].video_path))
        other = p.sessions[1 - p.active]
        pa = other.pid_by_name("pa")
        check("I167 the other camera keeps its own video and tracks",
              os.path.normcase(os.path.abspath(other.video_path)) == os.path.normcase(os.path.abspath(A))
              and pa is not None and int(other.tracked[:, pa].sum()) == 10, other.video_path)
        check("I167 camera names survive", sorted(p.names) == ["front", "side"], p.names)
    close(w2)
    close(w1)
    # one camera: its name / settings come back too
    forget(A)
    w3 = window(A)
    w3.project.names[0] = "lonely"
    place_points(w3.session, ["solo"], range(8))
    w3.project.dirty = True
    w3._autosave(wait=True)
    pump(0.4)
    w4 = MainWindow()
    w4.resize(1200, 800)
    w4.show()
    w4._open_video(A)
    assert until(lambda: w4.state == READY and w4.project is not None, 30)
    pump(0.4)
    check("I167 a one-camera recovery keeps the camera's name", w4.project.names[0] == "lonely", w4.project.names)
    check("I167 and its points", w4.session.pid_by_name("solo") is not None)
    close(w4)
    close(w3)


# ================================================================== I188
@section
def i188_preview_binding():
    from kinetrace.segmenter import summarize_mask
    mask = np.zeros((240, 320), bool)
    mask[100:140, 100:160] = True
    summ = summarize_mask(mask, 1.0, 9.0)
    summ["frame"] = 0
    real_run = appmod._MaskPreviewWorker.run

    def slow_run(self):
        time.sleep(0.5)
        self.done.emit(dict(summ))

    appmod._MaskPreviewWorker.run = slow_run
    try:
        # (1) the video is replaced while SAM works: the outline must not land in the next video
        w = window(A)
        w.btn_animal.setChecked(True)
        click(w, 160, 120)
        check("I188 a preview is running", w._preview is not None)
        w._open_video(B, then=lambda info: w.session.ensure_animal())     # the next video already has a segment
        assert until(lambda: w.state == READY and w.session.video_path == B, 30)
        pump(1.2)
        s = w.session
        check("I188 a late preview is not written into the NEXT video's session",
              s.masks is None or not s.masks.has(0), s.masks.n_masked() if s.masks is not None else None)
        check("I188 the busy cursor is back", QApplication.overrideCursor() is None)
        close(w)
        # (2) the camera it was clicked in is removed
        w = window(A, [B])
        w._set_active_view(1)
        sB = w.project.sessions[1]
        w.btn_animal.setChecked(True)
        click(w, 160, 120)
        check("I188 camera B's preview is running", w._preview is not None and w._preview.target_session is sB)
        w._remove_view(1)
        pump(1.2)
        check("I188 a preview of a removed camera is dropped", sB.masks is None or not sB.masks.has(0))
        check("I188 the remaining camera is untouched",
              w.session.masks is None or not w.session.masks.has(0))
        check("I188 the busy cursor is back (removed camera)", QApplication.overrideCursor() is None)
        close(w)
    finally:
        appmod._MaskPreviewWorker.run = real_run


# ================================================================== I206
@section
def i206_hull_cache():
    w = window(A, [B])
    fake = (None, None, None)
    w._hull_cache[3] = fake
    w._on_view_offset(1, 2.0)
    check("I206 a retime clears the carved volumes even with no 3D result", not w._hull_cache, list(w._hull_cache))
    check("I206 and says so", "carved volume" in w.statusBar().currentMessage(), w.statusBar().currentMessage())
    w._hull_cache[3] = fake
    w._on_view_offset(1, 2.0)                       # the same value: nothing changed, nothing dropped
    check("I206 an offset set to the same value drops nothing", bool(w._hull_cache))
    w._hull_cache.clear()
    w._hull_cache[4] = fake
    assert w._add_view(C)
    check("I206 adding a camera clears the carved volumes", not w._hull_cache)
    w._hull_cache[4] = fake
    w._remove_view(2)
    check("I206 removing a camera clears the carved volumes", not w._hull_cache)
    close(w)


# ================================================================== G70 + G68 part
@section
def g70_undo_points():
    w = window(A, [B])
    w._preview_mask = lambda: None                  # no model in this suite
    s = w.session
    place_points(s, ["p1"], range(6))
    w._refresh_point_list()
    # a segment click: the previous undo step must not be left to undo something else
    w._undo_snap = s.snapshot()
    w.act_undo.setEnabled(True)
    w.btn_animal.setChecked(True)
    click(w, 100, 100)
    check("G70 a segment click leaves no older undo step behind",
          w._undo_snap is None and not w.act_undo.isEnabled(), w._undo_snap)
    check("G70 and the status line says the clicks are not undoable",
          "cannot be undone" in w.statusBar().currentMessage(), w.statusBar().currentMessage())
    # prompt removal and a box
    w._undo_snap = s.snapshot()
    w.act_undo.setEnabled(True)
    w._on_animal_box(10, 10, 50, 50)
    check("G70 a segment box clears the older undo step", w._undo_snap is None)
    w._undo_snap = s.snapshot()
    w.act_undo.setEnabled(True)
    w._on_prompt_remove(0)
    check("G70 removing a click clears the older undo step", w._undo_snap is None)
    w.btn_animal.setChecked(False)
    # a skeleton template: one step, in every camera
    t = {"name": "tiny", "head": "nose", "landmarks": ["nose", "tail", "paw"], "bones": [["nose", "tail"]],
         "derived": {}, "note": ""}
    w._undo_snap = None
    w.act_undo.setEnabled(False)
    n0 = [x.n_points for x in w.project.sessions]
    w._apply_skeleton_template(t)
    check("G70 a skeleton template takes an undo point", w.act_undo.isEnabled() and w._undo_snap is not None)
    n1 = [x.n_points for x in w.project.sessions]
    check("G70 the template gave landmarks to every camera", all(b > a for a, b in zip(n0, n1)), (n0, n1))
    w._undo_run()
    check("G70 Ctrl+Z takes the template back in EVERY camera", [x.n_points for x in w.project.sessions] == n0,
          [x.n_points for x in w.project.sessions])
    check("G70 and its skeleton", w.session.skeleton is None, w.session.skeleton)
    # forgetting the skeleton
    w._apply_skeleton_template(t)
    w._undo_snap = None
    w._clear_skeleton()
    check("G70 forgetting the skeleton takes an undo point", w._undo_snap is not None)
    w._undo_run()
    check("G70 Ctrl+Z brings the skeleton back", w.session.skeleton is not None and
          w.session.skeleton.get("name") == "tiny")
    # a data-source change on a point with NO data
    pid = w.session.pid_by_name("paw")
    w._undo_snap = None
    w._on_source_change(pid, "tip")
    check("G70 a data-source change without data takes an undo point", w._undo_snap is not None)
    check("G70 the point is derived now", w.session.points[pid].source == "silhouette")
    w._undo_run()
    check("G70 Ctrl+Z puts its source back", w.session.points[pid].source == "track")
    # G68 (the part in this region): may-leave-the-segment
    w._undo_snap = None
    w._on_free_toggled(pid, True)
    check("G68 a free-point toggle takes an undo point", w._undo_snap is not None)
    w._undo_run()
    check("G68 Ctrl+Z puts it back", not w.session.points[pid].free)
    close(w)


# ================================================================== G71
@section
def g71_loading_input():
    w = window(A, [B])
    calls = []
    w._toggle_tracking = lambda *a, **k: calls.append(1)
    tok = w._busy_push("Starting the tracking engine", "loading", 0, None)
    pump(0.2)
    check("G71 a blocking level is open", w._loading)
    w.dock.setFloating(True)
    pump(0.3)
    app.setActiveWindow(w.dock)
    w.point_list.setFocus()
    pump(0.1)
    QTest.keyClick(w.point_list, Qt.Key_T)
    pump(0.1)
    check("G71 T in the floated panel starts nothing while the card is up", calls == [], calls)
    w.dock.setFloating(False)
    app.setActiveWindow(w)
    pump(0.3)
    f0 = w.current
    QTest.keyClick(w, Qt.Key_Right)
    QTest.keyClick(w, Qt.Key_End)
    pump(0.1)
    check("G71 the arrow / End shortcuts do not move the frame meanwhile", w.current == f0, (f0, w.current))
    QTest.keyClick(w, Qt.Key_T)
    pump(0.1)
    check("G71 nor does T in the main window", calls == [], calls)
    w._busy_pop(tok)
    pump(0.2)
    check("G71 the card is gone", not w._loading and not w._busy_stack)
    QTest.keyClick(w, Qt.Key_Right)
    pump(0.1)
    check("G71 the arrow key works again afterwards", w.current == f0 + 1, (f0, w.current))
    QTest.keyClick(w, Qt.Key_T)
    pump(0.1)
    check("G71 and T reaches tracking again", calls == [1], calls)
    close(w)


# ================================================================== G92
@section
def g92_remove_view_tools():
    w = window(A, [B, C])
    w._set_active_view(2)
    pump(0.2)
    w.btn_add.setChecked(True)
    pump(0.1)
    w._mark_event()                                  # E: the start of an event, marked in camera C
    check("G92 Add is armed on the working canvas", w.canvas._place_mode)
    check("G92 an event start is pending", w._pending_event is not None)
    w._remove_view(0)                                # active moves from 2 to 1: another canvas
    pump(0.3)
    check("G92 removing a camera drops a half-marked event", w._pending_event is None)
    check("G92 Add is armed on the canvas that is working now", w.btn_add.isChecked() and w.canvas._place_mode,
          (w.btn_add.isChecked(), w.canvas._place_mode))
    check("G92 and on no other canvas", sum(1 for cv in w.grid.canvases if cv._place_mode) == 1,
          [cv._place_mode for cv in w.grid.canvases])
    w.btn_animal.setChecked(True)
    w._remove_view(0)
    pump(0.3)
    check("G92 removing a camera puts the segment tool down", not w.btn_animal.isChecked())
    close(w)


# ================================================================== G105
@section
def g105_system_check():
    import kinetrace.device as dev
    real = (dev.cached_device, dev.describe)
    t0 = {"t": None}

    class FakeProbe:
        def isRunning(self):
            return time.monotonic() - t0["t"] < 0.8

        def wait(self, ms=0):
            end = t0["t"] + 0.8
            time.sleep(max(0.0, min(ms / 1000.0, end - time.monotonic())))
            return True

    w = window(A)
    ticks = []
    timer = QTimer()
    timer.setInterval(30)
    timer.timeout.connect(lambda: ticks.append(1))
    real_exec = QDialog.exec
    try:
        t0["t"] = time.monotonic()
        dev.cached_device = lambda: ("x" if time.monotonic() - t0["t"] >= 0.8 else None)
        dev.describe = lambda *a, **k: "FAKE SYSTEM CHECK"
        w._dev_probe.wait(60000)
        w._dev_probe = FakeProbe()
        QDialog.exec = lambda self: 0
        timer.start()
        w._show_system_check()
        n = len(ticks)
        timer.stop()
    finally:
        dev.cached_device, dev.describe = real
        QDialog.exec = real_exec
    check("G105 the window keeps repainting while the check waits for PyTorch", n >= 5, n)
    w._dev_probe = SimpleNamespace(isRunning=lambda: False, wait=lambda ms=0: True)
    close(w)


# ================================================================== G109
@section
def g109_import_ends_pending_wait():
    d = os.path.join(OUT, "imp2")
    p1 = clip(os.path.join(d, "a.mp4"), 30)
    p2 = clip(os.path.join(d, "b.mp4"), 30, shift=3)
    w = window(A)
    old = w._busy_push("Opening something earlier", "waiting for its picture", 5)
    w._first_frame_token = old
    w._busy_set_passive(old)
    w._import_folder([p1, p2], None)
    assert until(lambda: w.project is not None and w.project.n_views == 2, 30)
    pump(1.5)
    check("G109 the earlier open's card is gone once the import has its picture",
          not w._busy_stack and w._first_frame_token is None, (w._busy_stack, w._first_frame_token))
    close(w)


# ================================================================== G120
@section
def g120_segment_row_and_derived_text():
    w = window(A)
    w._preview_mask = lambda: None
    s = w.session
    s.add_landmark("tail_tip", "silhouette", "tip")
    w._refresh_point_list()
    w.point_list.setCurrentRow(0)
    w.point_list.item(0).setSelected(True)
    w._update_track_button()
    msg = w._track_blocked or ""
    check("G120 a derived landmark alone is not told to be placed by hand", "place it here" not in msg and msg,
          msg[:100])
    # S + a real click: the segment's row is selected, so Track runs it
    w.point_list.clearSelection()
    w.btn_animal.setChecked(True)
    pump(0.1)
    click(w, 160, 120)
    pump(0.2)
    check("G120 the new segment's row is selected", w.animal_list.count() == 1 and
          w.animal_list.item(0).isSelected(), w.animal_list.count())
    check("G120 so Track says it covers the segment", "segment" in w.btn_track.text(), w.btn_track.text())
    check("G120 and is not blocked", w._track_blocked is None, w._track_blocked)
    close(w)


# ================================================================== I233
@section
def i233_annotator_every_camera():
    w = window(A, [B])
    w.project.dirty = False
    w._apply_annotator("Dr. Rivera")
    check("I233 every camera's session carries the annotator",
          [x.annotator for x in w.project.sessions] == ["Dr. Rivera", "Dr. Rivera"],
          [x.annotator for x in w.project.sessions])
    check("I233 and is marked changed", all(x.dirty for x in w.project.sessions))
    w._apply_annotator("")
    close(w)


# ================================================================== I234
@section
def i234_deleted_landmark_stays_deleted():
    w = window(A, [B])
    t = {"name": "tri", "head": "nose", "landmarks": ["nose", "belly", "tail"],
         "bones": [["nose", "belly"], ["belly", "tail"]], "derived": {}, "note": ""}
    w._apply_skeleton_template(t)
    pid = w.session.pid_by_name("belly")
    w._on_delete(pid)
    check("I234 the landmark left every camera", all(x.pid_by_name("belly") is None for x in w.project.sessions))
    assert w._add_view(C)
    new = w.project.sessions[-1]
    names = [[m.name for m in x.points] for x in w.project.sessions]
    check("I234 a camera added afterwards does not bring it back",
          all("belly" not in n for n in names), names)
    check("I234 the others have the same list", len({tuple(n) for n in names}) == 1, names)
    check("I234 the new camera's skeleton has no bone to it",
          new.skeleton is None or all("belly" not in b for b in new.skeleton.get("bones", [])), new.skeleton)
    check("I234 the point list shows the working camera's points", w.point_list.count() == w.session.n_points)
    close(w)


# ================================================================== I259
@section
def i259_events_menu_leak():
    w = window(A)
    s = w.session
    s.add_event("run", 1, 5)
    s.add_event("run", 10, 15)
    s.add_event("jump", 20, 25)
    s.set_note(3, "a note")
    for _ in range(25):
        w._refresh_events_ui()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    pump(0.1)
    n = len(w.m_events.findChildren(QMenu))
    check("I259 rebuilding the Events menu does not leak its submenus (2 live, was 50)", n <= 2, n)
    close(w)


# ================================================================== G132 + follow-up (d)
@section
def g132_custom_skeleton():
    real_dir = skeletons.SKELETON_DIR
    skeletons.SKELETON_DIR = skeletons.Path(os.path.join(OUT, "skel"))
    real_exec = QDialog.exec
    fill = {}

    def fake_exec(self):
        if self.windowTitle() != "Custom skeleton":
            return real_exec(self)
        edits = self.findChildren(QPlainTextEdit)
        edits[0].setPlainText("left-hip\nleft-knee\nleft-ankle")
        edits[1].setPlainText(fill["bones"])
        for le in self.findChildren(appmod.QLineEdit):
            if le.text() == "my skeleton":
                le.setText(fill.get("name", "mine"))
        return QDialog.Accepted

    QDialog.exec = fake_exec
    try:
        w = window(A)
        fill["bones"] = "left-hip - left-knee\nleft-knee-left-ankle\nghost - left-hip"
        ASK["warned"].clear()
        w._custom_skeleton_dialog()
        bones = (w.session.skeleton or {}).get("bones", [])
        check("G132 'left-hip - left-knee' keeps both hyphenated names", ["left-hip", "left-knee"] in bones, bones)
        check("G132 a line with no spaced dash is split where both halves are landmarks",
              ["left-knee", "left-ankle"] in bones, bones)
        warn = " ".join(t for _, t in ASK["warned"])
        check("G132 the dropped line is named", "ghost - left-hip" in warn, warn[:200])
        # (d) the same name again: asked before it is replaced
        ASK["asked"].clear()
        w._custom_skeleton_dialog()
        check("(d) saving over an existing skeleton asks first", "Replace the saved skeleton?" in ASK["asked"],
              ASK["asked"])
        path = skeletons.user_template_path("mine")
        before = path.read_text(encoding="utf-8")
        fill["bones"] = "left-hip - left-knee"
        ASK["answer"] = QMessageBox.No
        w._custom_skeleton_dialog()
        check("(d) No keeps the file", path.read_text(encoding="utf-8") == before)
        ASK["answer"] = QMessageBox.Yes
        w._custom_skeleton_dialog()
        check("(d) Yes replaces it", path.read_text(encoding="utf-8") != before)
        close(w)
    finally:
        QDialog.exec = real_exec
        skeletons.SKELETON_DIR = real_dir


# ================================================================== G133
@section
def g133_shift_c_other_cameras():
    w = window(A, [B])
    f = w.project.map_frame(0, 1, w.current)
    check("G133 camera B shows its picture before", until(lambda: w._views[1].cache.get(f) is not None, 10))
    w._reset_frame_cache()
    ok = until(lambda: w._views[1].cache.get(f) is not None, 10)
    check("G133 Shift+C makes the other cameras decode their picture again", ok)
    close(w)


# ================================================================== G134
@section
def g134_veil_does_not_carry():
    w = window(A, [B])
    w._set_active_view(1)
    w._set_sync_mode("active")
    pump(0.2)
    w._goto(5)
    pump(0.3)
    check("G134 camera A is veiled (Active view only)", w.grid.canvas(0).is_stale())
    w._open_video(C)
    assert until(lambda: w.state == READY and w.session.video_path == C, 30)
    pump(0.5)
    check("G134 the next video starts un-veiled", all(not cv.is_stale() for cv in w.grid.canvases),
          [cv.is_stale() for cv in w.grid.canvases])
    close(w)


# ================================================================== G139
@section
def g139_track_label_width():
    w = window(A)
    s = w.session
    place_points(s, ["alpha", "beta", "gamma"], range(4))
    s.ensure_animal()
    w._refresh_point_list()
    w._refresh_animal_panel()
    w._select_all_tracked()
    pump(0.2)
    w._update_track_button()
    w.setMinimumWidth(0)
    w.resize(600, 700)                    # as narrow as the layout allows
    pump(0.5)
    label = w.btn_track.text()
    check("G139 the Track label is a long one", "segment" in label and "points" in label, label)
    check("G139 the Track button is as wide as its label", w.btn_track.width() >= w.btn_track.sizeHint().width(),
          (w.btn_track.width(), w.btn_track.sizeHint().width()))
    check("G139 the bar's labels folded against the live width",
          w._controls.width() >= w._controls.minimumWidth(), (w._controls.width(), w._controls.minimumWidth()))
    close(w)


# ================================================================== I258
@section
def i258_reference_instant_rule():
    w = window(A, [B])
    w.project.set_offset(1, 0.5)
    w._set_active_view(1)
    pump(0.3)
    w._goto(3)
    pump(0.2)
    want = w.project.map_frame(1, 0, 3)
    check("I258 the reference instant agrees with map_frame at a half-frame offset",
          w._reference_instant() == want, (w._reference_instant(), want))
    close(w)


# ================================================================== M9 / R11
@section
def m9_docs_and_dead_bits():
    h = appmod.HOTKEYS_HTML
    check("M9 the reference no longer says a marker is dragged", "drag a marker" not in h)
    check("M9 it says markers are never dragged and a left drag pans", "points are never dragged" in h)
    check("M9 a short right click clears (several selected: all of them)", "clears all of them on this frame" in h)
    check("M9 a long right press opens the point menu", "hold right" in h and "menu" in h)
    check("M9 Every camera tracks the selected points (no 'or all')", "— or all —" not in h)
    check("M9 Ctrl+A is listed", "Ctrl+A" in h)
    w = window(A)
    check("M9 the Every camera tooltip says only the selected points", "or all" not in w.act_track_all.toolTip(),
          w.act_track_all.toolTip()[:120])
    check("R11 one constant for the manual heading", appmod.MANUAL_WHICH_MODEL == "Which point model should I use?")
    check("R22 QPushButton is no longer imported", not hasattr(appmod, "QPushButton"))
    close(w)


print()
if EXC:
    print("unhandled exceptions:", *EXC, sep="\n  ")
    FAILS.append("exceptions")
if FAILS:
    print(f"VERIFY_REVIEW_APP1 FAILED: {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("VERIFY_REVIEW_APP1 PASSED")
