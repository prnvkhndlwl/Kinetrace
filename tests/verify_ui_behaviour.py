"""The window's behaviour rows of the 2026-10-25 sweep (sweep PR 7, dev/work/2026-10-28-ui-behaviour-sweep). Offscreen,
CPU, every dialog stubbed; each section makes the user's gesture and FAILS on eaee38b.
  [1] (I369) open project A, then B at once: A's late "finished" never retires B's frame check, no 2 s freeze
  [2] (G226) Every camera, the point placed here only in the other cameras: T runs there, as the label says
  [3] (G227) trails: up to 300 frames the same picture pixel for pixel; 20 x 1000 frames repaint under 33 ms,
      longer trails thinned within each stretch with data, never across a gap
  [4] (I370) close while the tracking worker does not stop in 20 s: it is retired, the close waits for it
  [5] (I371, G228) a Track press that starts nothing (last frame, nothing to start from, a declined run) leaves
      Ctrl+Z as it was; T on the last frame says there is nothing ahead
  [6] (I373) a ball's size hint is its size at the start frame (else the nearest earlier, else the nearest later)
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


SECTIONS = {"fc": section_fc, "every": section_every, "trails": section_trails, "close": section_close,
            "undo": section_undo, "ball": section_ball}
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
