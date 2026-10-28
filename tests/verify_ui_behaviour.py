"""The window's behaviour rows of the 2026-10-25 sweep (sweep PR 7, dev/work/2026-10-28-ui-behaviour-sweep). Offscreen,
CPU, every dialog stubbed; each section makes the user's gesture and FAILS on eaee38b.
  [1] (I369) open project A, then B at once: A's late "finished" never retires B's frame check, no 2 s freeze
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
QMessageBox.question = staticmethod(lambda *a, **k: (QUESTIONS.append(a[1:3]), QMessageBox.Yes)[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
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


SECTIONS = {"fc": section_fc}
ONLY = {x.strip() for x in os.environ.get("UI_ONLY", "").split(",") if x.strip()}
for key, fn in SECTIONS.items():
    if ONLY and key not in ONLY:
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
