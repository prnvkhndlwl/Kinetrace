"""The window's behaviour rows of the 2026-10-25 sweep (sweep PR 7, dev/work/2026-10-28-ui-behaviour-sweep). Offscreen,
CPU, every dialog stubbed; each section makes the user's gesture and FAILS on eaee38b.
  [1] (I369) open project A, then B at once: A's late "finished" never retires B's frame check, no 2 s freeze
  [1b] (I369, review) a camera removed while its frame check runs: the other cameras' checks still run
  [2] (G226) Every camera, the point placed here only in the other cameras: T runs there, as the label says
  [3] (G227) trails: up to 300 frames the same picture pixel for pixel; 20 x 1000 frames repaint under 33 ms,
      longer trails thinned within each stretch with data, never across a gap
  [4] (I370) close while the tracking worker does not stop in 20 s: it is retired, the close waits for it
  [5] (I371, G228) a Track press that starts nothing (last frame, nothing to start from, a declined run) leaves
      Ctrl+Z as it was; T on the last frame says there is nothing ahead
  [6] (I373) a ball's size hint is its size at the start frame (else the nearest earlier, else the nearest later)
  [7] (I374) 3D > Carve Volume (Ctrl+4) on a 50 mm ball: the same hull in a metre and a mm world, never cut off
  [8] (G229) names with "<" or "&" read as typed: a run's stop notice, a marked-up notice, a refused name typed in
      LAYERS
  [9] (G230) Save As onto the project opened without camB names camB and what replacing it drops; No changes nothing
  [10] (I375) has_click_after: the same answers as the old scan after every kind of edit; the Track label with 15
       cameras x 20 points x 40,000 frames costs under 5 ms a playhead move (20.7 ms on eaee38b)
  [11] (G231) Body > Side-by-side View (Ctrl+6) greyed out until there is a body pose; an open window stays closable
  [12] (I377) a project whose videos sit in ../../vid opens with plain video paths (no "..")
Run: .venv\\Scripts\\python.exe tests\\verify_ui_behaviour.py   (UI_ONLY=fc,... runs only those sections)
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
OUT = os.path.join(HERE, "out", "ui_behaviour")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_SETTINGS"] = os.path.join(OUT, "settings.ini")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QUESTIONS: list = []
ANSWER = [QMessageBox.Yes]                # what every question is answered (a section may set No)
QMessageBox.question = staticmethod(lambda *a, **k: (QUESTIONS.append(a[1:3]), ANSWER[0])[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
WARNINGS: list = []
QMessageBox.warning = staticmethod(lambda *a, **k: (WARNINGS.append(str(a[1])), QMessageBox.Ok)[1])
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _stubs import stub_track_choice  # noqa: E402

from kinetrace import threads  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

TRACK_CHOICE = stub_track_choice("keep")
FAILS: list = []


def check(label, cond, detail=None):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001
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


class Toasts:
    def __init__(self, win):
        self.msgs = []
        real = win.toast.show_message
        win.toast.show_message = lambda text, *a, **k: (self.msgs.append(str(text)), real(text, *a, **k))[1]

    def said(self, *words):
        return any(all(w.lower() in m.lower() for w in words) for m in self.msgs)


def make_video(name, n=40, w=160, h=120):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    rng = np.random.default_rng(len(name))
    bg = cv2.GaussianBlur(rng.integers(30, 90, size=(h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    for f in range(n):
        fr = bg.copy()
        cv2.circle(fr, (40 + f, 50), 5, (60, 60, 230), -1, lineType=cv2.LINE_AA)
        vw.write(fr)
    vw.release()
    return path


def make_project(name, videos, n=40, w=160, h=120):
    """A saved project folder with one point per camera, tracked on every frame."""
    sessions = []
    for v in videos:
        s = TrackingSession(v, n, 30.0, w, h)
        q = s.add_point(0, 40.0, 50.0)
        s.tracks[:, q] = np.stack([40.0 + np.arange(n), np.full(n, 50.0)], axis=1)
        s.tracked[:, q] = s.visibility[:, q] = True
        s.confidence[:, q] = 0.9
        sessions.append(s)
    proj = Project(sessions, [f"cam{chr(65 + i)}" for i in range(len(videos))], [0.0] * len(videos))
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    proj.save(path)
    return path


def new_window():
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    pump(0.2)
    return win


def open_project(win, path, timeout=30):
    """File > Open Project, the dialog answered with the project's kinetrace.json."""
    real = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (os.path.join(path, "kinetrace.json"), ""))
    try:
        win._open_project_dialog()
    finally:
        QFileDialog.getOpenFileName = real
    return wait(lambda: win.state == READY and win.project is not None and not win._loading, timeout)


def close(win):
    if win.project is not None:
        win.project.dirty = False
        for s in win.project.sessions:
            s.dirty = False
    win.close()
    pump(0.3)


# ---------------------------------------------------------------------------------------------------------------
def section_fc():
    print("[1] (I369) open A, then B at once: A's late 'finished' leaves B's frame check alone")
    from kinetrace import framecheck
    va, vb = make_video("fc/a.mp4"), make_video("fc/b.mp4")
    pa, pb = make_project("fc/a.kinetrace", [va]), make_project("fc/b.kinetrace", [vb])
    started, done = [], []
    real_run = framecheck.FrameCheckWorker.run

    def slow_run(self):
        """A check that takes a while (a 4K file on a share): it ends when cancelled (a little late, so its
        `finished` is still queued when the next project's check has started) or after 3 s."""
        started.append(os.path.basename(self.path))
        t0 = time.time()
        while not self._cancel and time.time() - t0 < 3.0:
            time.sleep(0.01)
        if not self._cancel:
            done.append(os.path.basename(self.path))
    framecheck.FrameCheckWorker.run = slow_run
    win = new_window()
    try:
        check("(setup) project A opens", open_project(win, pa))
        check("(setup) A's frame check started", wait(lambda: "a.mp4" in started, 10), started)
        check("(setup) project B opens while A's check runs", open_project(win, pb))
        wait(lambda: "b.mp4" in started, 10)
        t_last, gaps = [time.perf_counter()], []

        def tick():
            now = time.perf_counter()
            gaps.append(now - t_last[0])
            t_last[0] = now
        from PySide6.QtCore import QTimer
        beat = QTimer(interval=5)
        beat.timeout.connect(tick)
        beat.start()
        pump(0.6)                                         # A's queued `finished` is delivered here
        beat.stop()
        job = win._fc_job
        check("[1] B's frame check is still the current one after A's late 'finished'",
              job is not None and os.path.basename(getattr(job, "path", "")) == "b.mp4",
              getattr(job, "path", None))
        check("[1] ... and it is not set aside as an orphan", job is not None and job not in threads.ORPHANS)
        check("[1] the window did not freeze meanwhile (longest event-loop gap < 0.2 s)",
              gaps and max(gaps) < 0.2, round(max(gaps or [0]), 3))
        check("[1] B's check runs to its end", wait(lambda: "b.mp4" in done, 6), done)
        wait(lambda: win._fc_job is None, 5)
    finally:
        framecheck.FrameCheckWorker.run = real_run
    close(win)


def section_every():
    print("[2] (G226) Every camera, the point tracked here only in the OTHER cameras: T starts them, as the label says")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    vids = [make_video(f"every/cam{i}.mp4", n=80) for i in range(3)]
    p0 = make_project("every/three.kinetrace", vids, n=80)
    win = new_window()
    check("(setup) a three-camera project opens", open_project(win, p0))
    p = win.project
    F = 30
    q0 = p.sessions[0].add_empty_point()                 # a landmark with no data in the working camera
    p.sync_landmarks()
    name = p.sessions[0].points[q0].name
    for v in (1, 2):
        sv = p.sessions[v]
        sv.set_position(F, sv.pid_by_name(name), 50.0, 60.0)
    win._set_active_view(0)
    win._refresh_point_list()
    win.layers.select_only([q0])
    win._on_point_selection_changed()
    win.act_track_all.setChecked(True)
    win._goto(F, force=True)
    pump(0.2)
    win._update_track_button()
    label = win.btn_track.text()
    started, toasts = [], Toasts(win)
    real = MainWindow._start_multi_tracking
    MainWindow._start_multi_tracking = lambda self, **k: started.append(
        [j["view"] for j in self._multi_jobs(k.get("step", False), names=k.get("names"), model=k.get("model"),
                                             rest=k.get("rest", True))])
    try:
        win.activateWindow()
        QTest.keyClick(win, Qt.Key_T)
        pump(0.3)
    finally:
        MainWindow._start_multi_tracking = real
    check("(setup) the button says it runs in 2 cameras", "2 cams" in label, label)
    check("[2] T starts the run in the two cameras that have the point", started == [[1, 2]], (started, toasts.msgs))
    check("[2] ... and says nothing about 'no position'", not toasts.said("no position"), toasts.msgs)
    close(win)


def _old_trail_paint(self, painter):
    """`_MotionOverlay.paint` as it was on eaee38b (one pen + one line per segment): the reference picture."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QPen
    from kinetrace.canvas import GHOST_RADIUS
    scale = painter.worldTransform().m11() or 1.0
    painter.setRenderHint(painter.RenderHint.Antialiasing, True)
    for i in range(len(self.colors)):
        if i >= len(self.display) or not self.display[i]:
            continue
        color, bold = self.colors[i], i == self.selected
        pts = self.past[i] if i < len(self.past) else None
        if pts is not None and len(pts) >= 2:
            k = len(pts) - 1
            for j in range(k):
                a, b = pts[j], pts[j + 1]
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                painter.setPen(QPen(QColor(*color, int(40 + 200 * (j + 1) / k)), (2.2 if bold else 1.5) / scale))
                painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))
        pts = self.future[i] if i < len(self.future) else None
        if pts is not None and len(pts) >= 2:
            k = len(pts) - 1
            for j in range(k):
                a, b = pts[j], pts[j + 1]
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                painter.setPen(QPen(QColor(*color, int(30 + 150 * (k - j) / k)), 1.2 / scale, Qt.DashLine))
                painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))
        r = GHOST_RADIUS / scale
        cur = self.current[i] if self.current is not None and i < len(self.current) else None
        for ghost, style in ((self.ghost_prev, Qt.SolidLine), (self.ghost_next, Qt.DashLine)):
            if ghost is None or i >= len(ghost) or not np.isfinite(ghost[i]).all():
                continue
            g = ghost[i]
            painter.setPen(QPen(QColor(*color, 170), 1.2 / scale, style))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(float(g[0]), float(g[1])), r, r)
            if cur is not None and np.isfinite(cur).all():
                painter.setPen(QPen(QColor(*color, 110), 1.0 / scale, Qt.DotLine))
                painter.drawLine(QPointF(float(g[0]), float(g[1])), QPointF(float(cur[0]), float(cur[1])))


def _chains(segs):
    """Segments (a, b) joined where one ends and the next starts: [(first, ..., last)] per chain."""
    out = []
    for a, b in segs:
        if out and out[-1][-1] == a:
            out[-1].append(b)
        else:
            out.append([a, b])
    return out


def section_trails():
    print("[3] (G227) long trails: the same picture, drawn in under 33 ms (20 points, 1000 frames past + future)")
    import types
    from PySide6.QtGui import QImage, QPainter
    from kinetrace.canvas import VideoCanvas
    cv = VideoCanvas()
    cv.resize(1920, 1080)
    cv.set_video_size(3840, 2160)
    rng = np.random.default_rng(0)
    n = 20
    metas = [types.SimpleNamespace(color=(int(40 + 10 * i) % 256, 200, 80), display=True, name=f"P{i}", kind="point",
                                   source="track", anchor=False, free=False, derived=False, is_ball=False,
                                   segment=None) for i in range(n)]
    pos = (rng.random((n, 2)) * 1000 + 1000).astype(np.float32)

    def scene(length, nan_every=0):
        cv.set_trails(length, True)
        tr = [np.cumsum(rng.normal(0, 3, size=(length + 1, 2)), 0).astype(np.float32) + 1000 for _ in range(n)]
        fu = [np.cumsum(rng.normal(0, 3, size=(length + 1, 2)), 0).astype(np.float32) + 1000 for _ in range(n)]
        if nan_every:
            for a in tr + fu:
                a[rng.random(len(a)) < 1.0 / nan_every] = np.nan          # frames with no data: gaps in the trail
        cv.set_points(pos, np.ones(n, bool), metas, 3, tr, fu)
        cv._motion.ghost_prev = pos + 3.0
        cv._motion.ghost_next = pos - 3.0
        cv._motion.current = pos

    def picture(paint):
        img = QImage(1920, 1080, QImage.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.scale(0.5, 0.5)                                 # 4K video in a 1080p view
        t0 = time.perf_counter()
        paint(p)
        dt = time.perf_counter() - t0
        p.end()
        return img, dt
    mo = cv._motion
    for length, nan_every, sel in ((10, 0, None), (100, 0, None), (300, 40, 3)):
        scene(length, nan_every)
        mo.selected = sel
        ref, _ = picture(lambda p: _old_trail_paint(mo, p))
        new, _ = picture(lambda p: mo.paint(p, None))
        check(f"[3] {length}-frame trails (gaps every ~{nan_every or '-'} frames): the same picture, pixel for pixel",
              new == ref)
    # longer than TRAIL_DRAWN (owner: thinned only if still over 33 ms): every n-th frame of each stretch with data
    from PySide6.QtCore import Qt
    from kinetrace.canvas import TRAIL_DRAWN

    class Recorder:
        def __init__(self):
            self.lines, self.alpha = [], []

        def setPen(self, pen):
            self.cur = pen.color().alpha()

        def drawLines(self, lines):
            self.lines += [(ln.x1(), ln.y1(), ln.x2(), ln.y2()) for ln in lines]
            self.alpha += [self.cur] * len(lines)
    scene(1000, 50)
    bad, worst = [], 0
    for pts in mo.past + mo.future:
        k = len(pts) - 1
        alphas = (40 + 200 * np.arange(1, k + 1) / k).astype(int)
        rec = Recorder()
        mo._trail(rec, pts, (200, 200, 80), alphas, 1.0, Qt.SolidLine)
        at = {(float(x), float(y)): f for f, (x, y) in enumerate(pts.tolist()) if np.isfinite([x, y]).all()}
        segs = [(at[(x0, y0)], at[(x1, y1)]) for x0, y0, x1, y1 in rec.lines]
        ok = np.isfinite(pts).all(axis=1)
        edge = np.diff(np.r_[0, ok.astype(np.int8), 0])
        runs = [(a, b - 1) for a, b in zip(np.nonzero(edge == 1)[0], np.nonzero(edge == -1)[0]) if b - a >= 2]
        worst = max(worst, len(segs))
        if any(not ok[a:b + 1].all() or b <= a for a, b in segs):
            bad.append("crosses a frame without data")
        if sorted(segs) != segs or [(s[0], s[-1]) for s in _chains(segs)] != runs:
            bad.append("does not run from each stretch's first frame to its last")
        if [al for al in rec.alpha] != [int(alphas[b - 1]) for _a, b in segs]:
            bad.append("fade differs")
    check(f"[3] 1000-frame trails with gaps: drawn through every n-th frame of each stretch (at most "
          f"{TRAIL_DRAWN} + one per gap segments), never across a gap, the same fade", not bad and worst < 2 * TRAIL_DRAWN,
          (sorted(set(bad)), worst))
    scene(1000, 0)
    t_old = min(picture(lambda p: _old_trail_paint(mo, p))[1] for _ in range(3))
    t_new = min(picture(lambda p: mo.paint(p, None))[1] for _ in range(3))
    print(f"    20 x 1000 frames past + future: {t_old * 1000:.0f} ms before, {t_new * 1000:.0f} ms now")
    check("[3] 20 points x 1000 frames past + future repaint in under 33 ms", t_new < 0.033, f"{t_new * 1000:.0f} ms")
    cv.deleteLater()


def section_close():
    print("[4] (I370) closing while the tracking worker does not stop in the close's 20 s: the window waits for it")
    from PySide6.QtCore import QThread
    from kinetrace import app as appmod

    class Stuck(QThread):
        """A worker in a long model load: it does not look at the pause flag; it ends 1.5 s later."""
        def request_pause(self):
            pass

        def run(self):
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.5:
                time.sleep(0.01)
    win = new_window()
    w = Stuck()
    w.start()
    win.worker, win.state = w, appmod.TRACKING
    seen = []
    real_wait_all, real_time = threads.wait_all, time.time
    threads.wait_all = lambda ms=30000: (seen.append(w in threads.ORPHANS), real_wait_all(ms))[1]
    calls = {"n": 0}

    def clock():                                          # the close's 20 s wait runs out at once
        calls["n"] += 1
        return real_time() + (21.0 if calls["n"] > 1 else 0.0)
    time.time = clock
    try:
        win.close()                                       # the user closes the window
    finally:
        time.time, threads.wait_all = real_time, real_wait_all
    running = w.isRunning()
    check("[4] the worker that outlived the wait is kept until it ends (in the registry the close waits for)",
          seen and seen[0], seen)
    check("[4] ... so the window closed only after it ended", not running)
    w.wait(5000)


def click_canvas(win, x, y):
    """The user's click on the video at scene point (x, y): it places the selected point there."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    cv = win.canvas
    QTest.mouseClick(cv.viewport(), Qt.LeftButton, Qt.NoModifier, cv.mapFromScene(QPointF(float(x), float(y))))
    pump(0.1)


def select(win, pids):
    """The LAYERS selection becomes `pids` (what clicking the rows does)."""
    win.layers.select_only(list(pids))
    win._on_point_selection_changed()
    pump(0.1)


def key(win, k, mod=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    win.activateWindow()
    QTest.keyClick(win, k, mod if mod is not None else Qt.NoModifier)
    pump(0.15)


def section_undo():
    print("[5] (I371, G228) a Track press that starts nothing leaves Ctrl+Z as it was; T on the last frame says so")
    from PySide6.QtCore import Qt
    from kinetrace import device
    from kinetrace.app import TRACKING
    v = make_video("undo/clip.mp4", n=40)
    win = new_window()
    win._open_video(v)
    check("(setup) the video opens", wait(lambda: win.state == READY and win.session is not None
                                            and not win._loading, 30))
    s = win.session
    last = s.n_frames - 1
    p1 = s.add_point(0, 40.0, 50.0, name="P1")
    s.points[p1].tracker = "spot"                         # Moving spot: no point model to load on the old code
    win._refresh_point_list()
    toasts = Toasts(win)
    seen_tracking = []

    def press_t():
        n0 = len(seen_tracking)
        real_launch = win._launch_run
        win._launch_run = lambda w: (seen_tracking.append(1), real_launch(w))[1]
        try:
            key(win, Qt.Key_T)
            wait(lambda: win.state == READY and win.worker is None and not win._run_in_progress(), 30)
            pump(0.2)
        finally:
            del win._launch_run
        return len(seen_tracking) > n0

    # A: T on the last frame, right after a click there
    select(win, [p1])
    key(win, Qt.Key_End)
    check("(setup) the playhead is on the last frame", win.current == last, win.current)
    click_canvas(win, 60.0, 50.0)
    check("(setup) the click placed P1 on the last frame", s.tracked[last, p1])
    ran = press_t()
    check("[5] T on the last frame says there is nothing ahead to track", toasts.said("last frame", "nothing"),
          toasts.msgs[-2:])
    check("[5] ... and starts no run", not ran and win.state == READY)
    key(win, Qt.Key_Z, Qt.ControlModifier)
    check("[5] ... and Ctrl+Z still takes back the click made before it", not win.session.tracked[last, p1])

    # B: "Nothing to start from" (a Moving spot point with no position here, a click ahead, Fresh run)
    s = win.session
    p2 = s.add_point(12, 100.0, 60.0, name="P2")
    s.points[p2].tracker = "spot"
    win._refresh_point_list()
    win._goto(5, force=True)
    select(win, [p1])
    click_canvas(win, 45.0, 50.0)                         # the edit Ctrl+Z should take back
    check("(setup) the click placed P1 on frame 5", s.tracked[5, p1])
    select(win, [p2])
    pump(0.1)
    TRACK_CHOICE.answer = "fresh"
    try:
        ran = press_t()
    finally:
        TRACK_CHOICE.answer = "keep"
    check("(setup) the press reached 'Nothing to start from'", toasts.said("nothing to start from"), toasts.msgs[-2:])
    key(win, Qt.Key_Z, Qt.ControlModifier)
    check("[5] after a press that found nothing to start from, Ctrl+Z still takes back the click before it",
          not win.session.tracked[5, p1])

    # C: the "Many restarts at once" question answered No
    s = win.session
    p3 = s.add_point(6, 120.0, 70.0, name="P3")
    s.points[p3].tracker = "cotracker3"
    s.set_position(20, p3, 125.0, 70.0)                   # a click ahead: a re-seed session
    win._refresh_point_list()
    win._goto(6, force=True)
    select(win, [p1])
    click_canvas(win, 46.0, 50.0)
    check("(setup) the click placed P1 on frame 6", s.tracked[6, p1])
    select(win, [p3])
    pump(0.1)
    real_vram, n_q = device.cached_vram_gb, len(QUESTIONS)
    device.cached_vram_gb = lambda *a, **k: 0.01          # a card too small: the question is asked
    ANSWER[0] = QMessageBox.No
    try:
        ran = press_t()
    finally:
        device.cached_vram_gb, ANSWER[0] = real_vram, QMessageBox.Yes
    check("(setup) the press asked 'Many restarts at once' and was declined",
          any("restarts" in str(q[0]).lower() for q in QUESTIONS[n_q:]) and not ran, QUESTIONS[n_q:])
    key(win, Qt.Key_Z, Qt.ControlModifier)
    check("[5] after a declined run, Ctrl+Z still takes back the click before it", not win.session.tracked[6, p1])
    check("(setup) no run was left going", win.state != TRACKING)
    close(win)


def section_ball():
    print("[6] (I373) a ball tracked from an early frame gets the size it has THERE as its hint")
    from PySide6.QtCore import Qt
    import kinetrace.tracker as trk
    v = make_video("ball/clip.mp4", n=40)
    win = new_window()
    win._open_video(v)
    check("(setup) the video opens", wait(lambda: win.state == READY and win.session is not None
                                            and not win._loading, 30))
    s = win.session
    b = s.add_ball(0, 60.0, 60.0, name="ball")
    for f in range(20):
        s.set_position(f, b, 60.0 + f, 60.0)
        s.radius[f, b] = 10.0 if f < 10 else 40.0      # 10 px on frames 0-9, 40 px from frame 10
    s.radius[3:6, b] = np.nan                           # no size measured on frames 3-5
    s.radius[0:2, b] = np.nan                           # ... nor on 0-1
    s._touch()
    win._refresh_point_list()
    got = []

    class Worker:
        def __init__(self, *a, **k):
            got.append([bs.radius for bs in (k.get("balls") or [])])
            self.point_ids = []
    real_worker = trk.TrackingWorker
    win._launch_run = lambda w, driver=None: None        # nothing really runs
    trk.TrackingWorker = Worker
    try:
        for start, want, what in ((2, 10.0, "frame 2 (10 px there)"), (12, 40.0, "frame 12 (40 px there)"),
                                  (4, 10.0, "frame 4 (none there: the nearest earlier one, 10 px)"),
                                  (0, 10.0, "frame 0 (none before it: the nearest later one, 10 px)")):
            win._goto(start, force=True)
            select(win, [b])
            n0 = len(got)
            key(win, Qt.Key_T)
            r = got[n0][0] if len(got) > n0 and got[n0] else None
            check(f"[6] Track from {what}: size hint {want:g} px", r == want, r)
    finally:
        trk.TrackingWorker = real_worker
        del win._launch_run
    close(win)


def _hull_project(unit, scale):
    """3 cameras (f 1500 px, 1920 x 1080, 2 m away, 120 deg apart), one landmark at the centre of a 50 mm ball,
    each camera's silhouette = the ball's disc; `scale` = calibration units per mm."""
    from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera
    W_, H_, F_ = 1920, 1080, 1500.0
    K_ = np.array([[F_, 0, W_ / 2], [0, F_, H_ / 2], [0, 0, 1.0]])
    sessions, cams = [], []
    for k in range(3):
        a = np.deg2rad(120.0 * k)
        pos = np.array([2000.0 * np.cos(a), 2000.0 * np.sin(a), 300.0]) * scale
        z = -pos / np.linalg.norm(pos)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        cal = CameraCalibration(dlt_from_camera(K_, R, -R @ pos), W_, H_, NoUndistort(), pixel_origin=0.0)
        cams.append(cal)
        v = os.path.join(OUT, f"hull_{unit}", f"cam{k}.mp4")
        os.makedirs(os.path.dirname(v), exist_ok=True)
        vw = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W_, H_))
        for _f in range(3):
            vw.write(np.zeros((H_, W_, 3), np.uint8))
        vw.release()
        s = TrackingSession(v, 3, 30.0, W_, H_)
        uv = np.asarray(cal.project(np.zeros((1, 3))))[0]
        q = s.add_landmark("c")
        for f in range(3):
            s.set_position(f, q, float(uv[0]), float(uv[1]))
        i = s.add_segment("ball")
        m = np.zeros((H_, W_), np.uint8)
        cv2.circle(m, (int(round(uv[0])), int(round(uv[1]))), int(round(F_ * 50.0 * scale / np.linalg.norm(pos))),
                   255, -1)
        for f in range(3):
            s.write_mask(f, m > 0, 1.0, 1.0, i=i)
        sessions.append(s)
    proj = Project(sessions, ["cam0", "cam1", "cam2"], [0.0, 0.0, 0.0])
    proj.calibration = Calibration(cams, unit=unit, source="synthetic")
    path = os.path.join(OUT, f"hull_{unit}", "hull.kinetrace")
    proj.save(path)
    return path


def section_hull():
    print("[7] (I374) 3D > Carve Volume in a mm world: the same hull as in metres, not cut off")
    from PySide6.QtCore import Qt
    true_vol = 4.0 / 3.0 * np.pi * 50.0 ** 3
    for unit, scale in (("m", 0.001), ("mm", 1.0)):
        win = new_window()
        ok = open_project(win, _hull_project(unit, scale), timeout=60)
        check(f"(setup) the {unit} project opens", ok)
        win.act_recon.trigger()                           # Ctrl+3: the landmark's 3D position
        pump(0.5)
        n_w = len(WARNINGS)
        key(win, Qt.Key_4, Qt.ControlModifier)            # Ctrl+4: Carve Volume at This Frame
        wait(lambda: win._hull_cache, 120)
        cache = list(win._hull_cache.values())
        vol = float(cache[0][2].volume()) / scale ** 3 if cache else float("nan")
        cut = any("cut off" in w_.lower() for w_ in WARNINGS[n_w:])
        check(f"[7] {unit}: the 50 mm ball's hull is {vol:,.0f} mm3 (sphere {true_vol:,.0f}; a 3-view hull "
              "over-estimates, within 2x)", 0.5 < vol / true_vol < 2.0, f"{vol:,.0f}")
        check(f"[7] {unit}: and not 'cut off'", not cut, WARNINGS[n_w:])
        close(win)


def shown(win) -> str:
    """The notice as it reads on screen (the Toast's rich text rendered to plain text)."""
    from PySide6.QtGui import QTextDocument
    doc = QTextDocument()
    doc.setHtml(win.toast.text())
    return doc.toPlainText()


def rename_row(win, pid, text):
    """Rename the way the user does: double-click the point's LAYERS row, type the name, Enter."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    win.activateWindow()
    item = win.layers.point_item(pid)
    par = item.parent()
    while par is not None:
        par.setExpanded(True)
        par = par.parent()
    win.layers.scrollToItem(item)
    pump(0.05)
    at = win.layers.visualItemRect(item).center()
    QTest.mouseClick(win.layers.viewport(), Qt.LeftButton, Qt.NoModifier, at)
    pump(0.2 + QApplication.doubleClickInterval() / 1000.0)
    QTest.mouseDClick(win.layers.viewport(), Qt.LeftButton, Qt.NoModifier, win.layers.visualItemRect(item).center())
    pump(0.2)
    ed = QApplication.focusWidget()
    if type(ed).__name__ != "QLineEdit":
        return False
    QTest.keyClick(ed, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(ed, text)
    QTest.keyClick(ed, Qt.Key_Return)
    pump(0.2)
    return True


def section_names():
    print("[8] (G229) a name with '<' or '&' in it reads exactly as typed in every notice")
    from PySide6.QtCore import Qt
    path = os.path.join(OUT, "names", "gone.mp4")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    bg = cv2.GaussianBlur(np.random.default_rng(9).integers(30, 90, size=(120, 160, 3), dtype=np.uint8), (0, 0), 1.5)
    for f in range(60):
        fr = bg.copy()
        if f < 20:                                        # the dot is gone from frame 20 on
            cv2.circle(fr, (40 + f, 60), 5, (230, 230, 230), -1, lineType=cv2.LINE_AA)
        vw.write(fr)
    vw.release()
    win = new_window()
    win._open_video(path)
    check("(setup) the video opens", wait(lambda: win.state == READY and win.session is not None
                                            and not win._loading, 30))
    s = win.session
    tip = s.add_point(0, 40.0, 60.0, name="<tip>")
    s.points[tip].tracker = "spot"
    other = s.add_empty_point()
    s.points[other].name = "a<i>b"
    third = s.add_point(0, 120.0, 30.0, name="R&D")
    win._refresh_point_list()
    # (a) a run's stop notice (markup in it): Moving spot loses <tip> where its dot is gone
    win._goto(0, force=True)
    select(win, [tip])
    key(win, Qt.Key_T)
    wait(lambda: win.state == READY and win.worker is None and not win._run_in_progress(), 60)
    pump(0.3)
    txt = shown(win)
    check("[8] the run's stop notice names <tip> as typed", "<tip>" in txt, txt[:160])
    # (b) any notice that marks up a point's name (the app's own <b>): a<i>b stays a<i>b, not "ab" in italics
    win.toast.show_message(f"<b>{s.points[other].name}</b> sits on a dot about 9 px across.", "info", 5000)
    pump(0.1)
    txt = shown(win)
    check("[8] a notice that marks up a<i>b names it as typed (no italics)", "a<i>b sits on a dot" in txt, txt[:160])
    # (c) a plain notice: a new name that may not be used, typed in LAYERS
    check("(setup) the LAYERS name editor opened", rename_row(win, third, "=R&D<b>x"))
    txt = shown(win)
    check("[8] the refused name reads as typed: =R&D<b>x", "=R&D<b>x" in txt, txt[:160])
    close(win)


def _folder_digest(path):
    import hashlib
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(path)):
        for f in sorted(files):
            fp = os.path.join(root, f)
            h.update(os.path.relpath(fp, path).encode())
            with open(fp, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def section_saveas():
    print("[9] (G230) Save As onto the project opened without a camera names the camera it would drop")
    va, vb = make_video("saveas/vid/camA.mp4"), make_video("saveas/vid/camB.mp4")
    proj = make_project("saveas/two.kinetrace", [va, vb])
    os.replace(vb, vb + ".moved")                         # camB's video is not found when the project opens
    win = new_window()
    real_open = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))    # "Locate video": cancelled
    try:
        ok = open_project(win, proj)
    finally:
        QFileDialog.getOpenFileName = real_open
    check("(setup) the project opens with camA only", ok and win.project.n_views == 1, win.project and
          win.project.n_views)
    before = _folder_digest(proj)
    n_q = len(QUESTIONS)
    real_save = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (proj, ""))  # File > Save As: the same folder
    ANSWER[0] = QMessageBox.No
    try:
        win._save_project_as()
    finally:
        QFileDialog.getSaveFileName, ANSWER[0] = real_save, QMessageBox.Yes
    asked = QUESTIONS[n_q:]
    text = " ".join(str(q[1]) for q in asked)
    check("[9] Save As asks before replacing it", asked, asked)
    check("[9] ... and the question names camB, whose points, tracks and calibration it would drop",
          "camB" in text and "calibration" in text, text[:200])
    check("[9] answered No, the project folder is untouched", _folder_digest(proj) == before)
    close(win)
    os.replace(vb + ".moved", vb)


def _old_has_click_after(s, pid, frame, end=None):
    """`TrackingSession.has_click_after` as it was on eaee38b (a scan of the rows after `frame`): the reference."""
    from kinetrace.session import in_frame
    end = s.n_frames if end is None else min(s.n_frames, int(end))
    if not (0 <= pid < s.n_points) or end <= int(frame) + 1:
        return False
    rows = slice(int(frame) + 1, end)
    ok = s.manual[rows, pid] & s.tracked[rows, pid] & s.visibility[rows, pid] & ~s.occluded[rows, pid]
    return bool(ok.any() and in_frame(s.tracks[rows, pid][ok], s.width, s.height).any())


def section_label():
    print("[10] (I375) Every camera with 15 cameras x 20 points x 40,000 frames: the Track label keeps up with the "
          "playhead, the same answers")
    # the same answers, after every kind of edit
    rng = np.random.default_rng(4)
    s = TrackingSession("fuzz.mp4", 300, 30.0, 320, 240)
    for k in range(5):
        s.add_point(int(rng.integers(0, 300)), 50.0 + 30 * k, 60.0)
    bad = 0
    for step in range(60):
        q, f = int(rng.integers(0, 5)), int(rng.integers(0, 300))
        kind = step % 5
        if kind == 0:
            s.set_position(f, q, float(rng.uniform(0, 320)), float(rng.uniform(0, 240)))        # a click
        elif kind == 1:
            s.set_position(f, q, -5.0, float(rng.uniform(0, 240)))                            # outside the picture
        elif kind == 2:
            s.clear_window([q], f, min(299, f + int(rng.integers(0, 30))))                     # Delete
        elif kind == 3:
            s.set_occluded(f, q, bool(rng.integers(0, 2)))                                     # marked hidden
        else:
            s.set_position(f, q, 100.0, 100.0)
            s.manual[f, q] = False                                                             # a tracked row
            s._touch()
        for _ in range(40):
            p_, f_ = int(rng.integers(-1, 6)), int(rng.integers(-1, 302))     # (a frame < -1 wrapped the old slice)
            e_ = None if rng.random() < 0.4 else int(rng.integers(0, 320))
            bad += s.has_click_after(p_, f_, e_) != _old_has_click_after(s, p_, f_, e_)
    check("[10] 2,400 questions after 60 edits (clicks, out of the picture, Delete, hidden, tracked rows): every "
          "answer the same as the old scan", bad == 0, f"{bad} differ")
    # the label on every playhead move
    from kinetrace import cameraorder
    from PySide6.QtWidgets import QDialogButtonBox
    real_exec = cameraorder.CameraOrderDialog.exec
    cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(),
                                                       self.result())[1]
    ncam, T, npts = 15, 40000, 20
    vids = [make_video(f"label/cam{i}.mp4", n=40, w=64, h=48) for i in range(ncam)]
    win = new_window()
    win._open_video(vids[0])
    wait(lambda: win.state == READY and not win._loading, 30)
    real_files = QFileDialog.getOpenFileNames
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (vids[1:], ""))
    try:
        win._add_video_dialog()                           # File > Add Camera Videos, the other 14
        pump(1.0)
    finally:
        QFileDialog.getOpenFileNames, cameraorder.CameraOrderDialog.exec = real_files, real_exec
    p = win.project
    check("(setup) 15 cameras", p is not None and p.n_views == ncam, p and p.n_views)
    for v in range(ncam):                                 # 40,000-frame cameras (the runtimes say the same length)
        p.sessions[v] = TrackingSession(vids[v], T, 30.0, 3840, 2160)
        win._views[v].n_frames = T
    for _ in range(npts):
        p.sessions[0].add_empty_point()
    p.sync_landmarks()
    for sv in p.sessions:
        for q in range(sv.n_points):
            sv.set_position(39990, q, float(rng.integers(100, 3000)), float(rng.integers(100, 2000)))
    win._set_active_view(0)
    win._refresh_point_list()
    win.act_select_all.trigger()
    win.act_track_all.setChecked(True)
    times = []
    for f in range(20000, 20020):
        win.current = f                                   # the playhead moves one frame
        t0 = time.perf_counter()
        win._update_track_button()
        times.append(time.perf_counter() - t0)
    per = float(np.median(times)) * 1000
    print(f"    _update_track_button: {per:.1f} ms per playhead move (median of 20)")
    check("[10] the Track label costs under 5 ms per playhead move", per < 5.0, f"{per:.1f} ms")
    close(win)


def section_body():
    print("[11] (G231) Body > Side-by-side View is greyed out until there is a body pose; an open window stays closable")
    from PySide6.QtCore import Qt
    from kinetrace import body
    v = make_video("body/clip.mp4", n=30)
    win = new_window()
    win._open_video(v)
    check("(setup) the video opens", wait(lambda: win.state == READY and win.session is not None
                                            and not win._loading, 30))
    win._apply_state()
    check("[11] with no body pose the menu entry is greyed out", not win.act_body_view.isEnabled())
    key(win, Qt.Key_6, Qt.ControlModifier)                # Ctrl+6
    check("[11] ... and Ctrl+6 opens no window", not win.act_body_view.isChecked()
          and getattr(win, "body_win", None) is None)
    s = win.session
    s.body = body.BodyTrack(s.n_frames, body.rig_of("coco17"), 1)
    s.body.set_person(3, 0, joints2d=np.full((17, 2), 50.0), conf=np.ones(17), score=0.9)
    s._touch()
    win._apply_state()
    check("[11] with a body pose it is enabled", win.act_body_view.isEnabled())
    key(win, Qt.Key_6, Qt.ControlModifier)
    check("(setup) Ctrl+6 opened the window", wait(lambda: win.act_body_view.isChecked(), 5))
    s.body = None                                          # Body > Remove ... while the window is open
    s._touch()
    win._apply_state()
    check("[11] an open window can still be closed from the menu", win.act_body_view.isEnabled())
    win.act_body_view.trigger()
    pump(0.2)
    close(win)


def section_paths():
    print("[12] (I377) a video found through the project folder is shown by its plain path, no '..'")
    va, vb = make_video("paths/vid/camA.mp4"), make_video("paths/vid/camB.mp4")
    proj = make_project("paths/cams/two.kinetrace", [va, vb])           # the videos sit beside, in ../../vid
    win = new_window()
    check("(setup) the project opens", open_project(win, proj))
    got = [s.video_path for s in win.project.sessions]
    check("[12] both cameras' video paths are plain (no '..')", all(".." not in str(g) for g in got), got)
    check("[12] ... and the same files", all(os.path.samefile(g, v) for g, v in zip(got, (va, vb))), got)
    close(win)


def section_fc_remove():
    print("[1b] (I369, review) a camera removed while its frame check runs: the other cameras' checks still run")
    from kinetrace import framecheck
    vids = [make_video(f"fcr/c{i}.mp4") for i in range(3)]
    proj = make_project("fcr/three.kinetrace", vids)
    started, done = [], []
    real_run = framecheck.FrameCheckWorker.run

    def run(self):
        """Camera c1's check takes a while (until cancelled, at most 3 s); the others end at once."""
        name = os.path.basename(self.path)
        started.append(name)
        t0 = time.time()
        while name == "c1.mp4" and not self._cancel and time.time() - t0 < 3.0:
            time.sleep(0.01)
        if not self._cancel:
            done.append(name)
    framecheck.FrameCheckWorker.run = run
    win = new_window()
    try:
        check("(setup) the three-camera project opens", open_project(win, proj))
        check("(setup) camera c1's check is running", wait(lambda: "c1.mp4" in started and "c1.mp4" not in done, 10),
              (started, done))
        win.cameras.remove_requested.emit(1)              # CAMERAS: remove camera c1 (its question answered Yes)
        pump(0.3)
        check("(setup) camera c1 was removed", win.project.n_views == 2)
        check("[1b] camera c2's check still runs to its end", wait(lambda: "c2.mp4" in done, 6), (started, done))
        wait(lambda: win._fc_job is None, 5)
    finally:
        framecheck.FrameCheckWorker.run = real_run
    close(win)


SECTIONS = {"fc": section_fc, "fcr": section_fc_remove, "every": section_every, "trails": section_trails, "close": section_close,
            "undo": section_undo, "ball": section_ball, "hull": section_hull, "names": section_names,
            "saveas": section_saveas, "label": section_label, "body": section_body, "paths": section_paths}
ONLY = {x.strip() for x in os.environ.get("UI_ONLY", "").split(",") if x.strip()}
for sect, fn in SECTIONS.items():
    if ONLY and sect not in ONLY:
        continue
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_ui_behaviour: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
