"""Offscreen GUI: one Track press tracks the points in EVERY camera (G29, GPU group).

Three synthetic cameras, each with a white dot (a tracked point) and an orange
ball (a ball marker, SAM + circle fit). The dot is placed in two cameras, the
ball in all three. With Track > Every camera ticked:
  the Track button says how many cameras a run covers; T tracks camA, camB and
  camC one after another (the dot where it was placed, the ball everywhere),
  each within a few px of the truth, and ends back in camA;
  one Ctrl+Z undoes the run in every camera;
  semi-automatic F steps ONE frame in every camera;
  X during the run stops the whole queue (the cameras after it are left alone).

Run: .venv\\Scripts\\python.exe tests\\verify_track_all.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.project import Project
from kinetrace.session import TrackingSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
W, H, N, FPS, R_BALL, F0 = 640, 480, 60, 30.0, 18, 5
rng = np.random.default_rng(4)

dot = np.zeros((3, N, 2))
ball = np.zeros((3, N, 2))
paths, sessions = [], []
for c in range(3):
    dot[c] = np.array([150.0 + 40 * c, 200.0]) + np.outer(np.arange(N), [3.0, 1.0 + 0.3 * c])
    ball[c] = np.array([420.0, 320.0 - 20 * c]) + np.outer(np.arange(N), [-3.0, -1.0])
    path = os.path.join(OUT, f"trackall_cam{c}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    xs, ys = np.arange(W), np.arange(H)
    # a STILL textured background, as filmed ground is (noise redrawn every frame made
    # the small dot drift, which tests the tracker, not tracking in every camera)
    bg = np.full((H, W, 3), 90, np.uint8)
    bg[..., 1] = (90 + 18 * np.sin(xs / 31.0 + c)[None, :] + 12 * np.cos(ys / 19.0)[:, None]).astype(np.uint8)
    bg = cv2.add(bg, np.repeat(rng.integers(0, 12, (H, W, 1), dtype=np.uint8), 3, axis=2))
    for k in range(N):
        img = bg.copy()
        cv2.circle(img, tuple(int(round(v)) for v in dot[c, k]), 7, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(int(round(v)) for v in ball[c, k]), R_BALL, (30, 120, 235), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    paths.append(path)
    sessions.append(TrackingSession(path, N, FPS, W, H))
    forget_recovery(path)
dot_r, ball_r = np.round(dot), np.round(ball)
PROJ = os.path.join(OUT, "trackall.kinetrace")
Project(sessions, ["camA", "camB", "camC"], [0, 0, 0]).save(PROJ)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))

from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1600, 1000)
win.show()


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t > timeout:
            raise TimeoutError(what)


win._open_project_from_path(PROJ)
wait(lambda: win.state == READY and win.project is not None and win.project.n_views == 3, 60, "open")
p = win.project
sa, sb, sc = p.sessions
win._goto(F0)
pump(0.3)

# the dot in camA and camB; the ball in all three
win.btn_add.setChecked(True)
win.canvas.add_requested.emit(float(dot_r[0, F0, 0]), float(dot_r[0, F0, 1]))
pump(0.1)
name_dot = sa.points[0].name
win.act_add_ball.trigger()
win.canvas.add_requested.emit(float(ball_r[0, F0, 0]), float(ball_r[0, F0, 1]))
pump(0.1)
name_ball = sa.points[1].name
assert [s.n_points for s in p.sessions] == [2, 2, 2] and all(s.points[1].is_ball for s in p.sessions)
for v, marks in ((1, (name_dot, name_ball)), (2, (name_ball,))):
    win._set_active_view(v)
    pump(0.2)
    s = p.sessions[v]
    for nm in marks:
        j = s.pid_by_name(nm)
        win.point_list.setCurrentRow(j)
        truth = dot_r if nm == name_dot else ball_r
        win._on_annotate(float(truth[v, F0, 0]), float(truth[v, F0, 1]))
        pump(0.05)
assert sb.points[sb.pid_by_name(name_ball)].ball_prompts, "a click on a ball in another camera is its SAM prompt"
win._set_active_view(0)
win._deselect()
win.point_list.clearSelection()
pump(0.2)
assert win.current == F0

# ---- 1. Track > Every camera: one press, every camera ------------------------------------
assert "cams" not in win.btn_track.text(), "off by default"
win.act_track_all.setChecked(True)
pump(0.05)
assert win.btn_track.text().endswith("· 3 cams"), win.btn_track.text()
t0 = time.time()
QTest.keyClick(win, Qt.Key_T)
wait(lambda: win._multi is not None or win.state == TRACKING, 30, "the queue starts")
seen = set()
while win._multi is not None or win.state != READY:
    if win.state == TRACKING:
        seen.add(p.active)
    pump(0.05)
    if time.time() - t0 > 900:
        raise TimeoutError("the multi-camera run")
pump(0.3)
assert seen == {0, 1, 2}, f"every camera was tracked, one after another: {seen}"
assert p.active == 0, "back in the working camera"


def err(s, v, nm, truth, frames):
    j = s.pid_by_name(nm)
    got = s.tracks[frames, j]
    ok = s.tracked[frames, j]
    return float(np.median(np.linalg.norm(got[ok] - truth[v, frames][ok], axis=1))) if ok.any() else np.inf, int(ok.sum())


frames = np.arange(F0, N)
if os.environ.get("TRACKALL_DEBUG"):
    for v in (0, 1):
        s = p.sessions[v]
        j = s.pid_by_name(name_dot)
        e = np.linalg.norm(s.tracks[frames, j] - dot_r[v, frames], axis=1)
        print(f"cam{v} dot errors:", np.round(e[::5], 1), "conf", np.round(s.confidence[frames[::5], j], 2))
for v, s in enumerate(p.sessions):
    eb, nb = err(s, v, name_ball, ball_r, frames)
    assert nb >= len(frames) - 3 and eb < 2.5, f"cam{v}: ball {nb}/{len(frames)} frames, median {eb:.2f} px"
    if v < 2:
        ed, nd = err(s, v, name_dot, dot_r, frames)
        assert nd >= len(frames) - 3 and ed < 3.0, f"cam{v}: dot {nd}/{len(frames)} frames, median {ed:.2f} px"
    else:
        assert not s.tracked[:, s.pid_by_name(name_dot)].any(), "camC never had the dot: nothing to track"
    print(f"  cam{v}: ball {nb} frames median {eb:.2f} px" + (f", dot {nd} frames median {ed:.2f} px" if v < 2 else ""))
assert "Tracked in every camera" in win.toast.text(), win.toast.text()
print(f"G29: one T tracked the point in the 2 cameras that have it and the ball in all 3, one camera after "
      f"another, back in camA ({time.time() - t0:.0f} s) OK")

# ---- 2. one Ctrl+Z, every camera -------------------------------------------------------
win._undo_run()
pump(0.2)
for v, s in enumerate(p.sessions):
    jb, jd = s.pid_by_name(name_ball), s.pid_by_name(name_dot)
    assert int(s.tracked[:, jb].sum()) == 1 and s.tracked[F0, jb], f"cam{v}: the ball is back to its one click"
    assert int(s.tracked[:, jd].sum()) == (1 if v < 2 else 0), f"cam{v}: the dot is back to its click"
print("G29: one Ctrl+Z undoes the run in every camera OK")

# ---- 3. semi-automatic: F steps one frame in every camera -------------------------------
win._goto(F0)                       # the run ended on its last frame; after the undo only F0 has data
win._deselect()
win.point_list.clearSelection()
win.act_mode_semi.trigger()
pump(0.05)
assert win.btn_track.text().endswith("· 3 cams") and win.btn_track.text().startswith("Step"), win.btn_track.text()
QTest.keyClick(win, Qt.Key_F)
wait(lambda: win._multi is None and win.state == READY, 300, "the multi-camera step")
pump(0.2)
for v, s in enumerate(p.sessions):
    jb = s.pid_by_name(name_ball)
    assert s.tracked[F0 + 1, jb] and not s.tracked[F0 + 2:, jb].any(), f"cam{v}: exactly one frame stepped"
assert p.active == 0 and win.current == F0 + 1, (p.active, win.current)
print("G29: semi-automatic F steps one frame in every camera and stays in camA OK")

# ---- 4. X stops the whole run ---------------------------------------------------------
win.act_mode_auto.trigger()
QTest.keyClick(win, Qt.Key_T)
wait(lambda: win.state == TRACKING and p.active == 0, 120, "camA tracking")
pump(0.5)
win._pause_tracking()
wait(lambda: win._multi is None and win.state == READY, 120, "the stop")
pump(0.2)
jb = sb.pid_by_name(name_ball)
assert not sb.tracked[F0 + 2:, jb].any(), "camB was never started after X"
assert "Stopped" in win.toast.text() and "Not tracked yet" in win.toast.text(), win.toast.text()
print("G29: X during the run stops every camera after the current one OK")

win._dev_probe.wait(15000)
win.close()
for path in paths:
    forget_recovery(path)
print("TRACK ALL PASSED")
