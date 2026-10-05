"""Offscreen GUI: one Track press tracks the points in EVERY camera (G29, I141, GPU group).

Three synthetic cameras, each with a white dot (a tracked point) and an orange
ball (a ball marker, SAM + circle fit). The dot is placed in two cameras, the
ball in all three. With Track > Every camera ticked:
  the Track button says how many cameras a run covers; T tracks camA, camB and
  camC AT THE SAME TIME (I141: all three under way before any is done, the
  working camera never switches mid-run, the others drawn live), the dot where
  it was placed, the ball everywhere, each within a few px of the truth, and
  ends in camA;
  each camera tracked ALONE gives exactly the same numbers (bit-identical);
  one Ctrl+Z undoes the run in every camera;
  semi-automatic F steps ONE frame in every camera;
  X during the run stops every camera at once, at about the same frame;
  with only one camera able to start, Track says which camera lacks what (G32).

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
        win.layers.setCurrentItem(win.layers.point_item(j))
        truth = dot_r if nm == name_dot else ball_r
        win._on_annotate(float(truth[v, F0, 0]), float(truth[v, F0, 1]))
        pump(0.05)
assert sb.points[sb.pid_by_name(name_ball)].ball_prompts, "a click on a ball in another camera is its SAM prompt"
win._set_active_view(0)
win._deselect()
win.act_select_all.trigger()   # Track tracks only what is selected (G61): all of it
pump(0.2)
assert win.current == F0

# ---- 1. Track > Every camera: one press, every camera ------------------------------------
assert "cams" not in win.btn_track.text(), "off by default"
win.act_track_all.setChecked(True)
pump(0.05)
assert win.btn_track.text().endswith("· 3 cams"), win.btn_track.text()
def progress(v):
    s = p.sessions[v]
    fr = np.nonzero(s.tracked[:, s.pid_by_name(name_ball)])[0]
    return int(fr.max()) if len(fr) else -1


side_draws = [0]
_real_side = win._render_side
win._render_side = lambda: (side_draws.__setitem__(0, side_draws[0] + 1), _real_side())[1]
t0 = time.time()
QTest.keyClick(win, Qt.Key_T)
wait(lambda: win._multi is not None or win.state == TRACKING, 30, "the run starts")
seen, together, side_trail = set(), False, False
while win._multi is not None or win.state != READY:
    if win.state == TRACKING:
        seen.add(p.active)
        past = win.grid.canvas(1)._motion.past or []
        if any(np.isfinite(x).all(axis=1).sum() > 2 for x in past):
            side_trail = True           # another camera draws its trail while it tracks (G33)
        prog = [progress(v) for v in range(3)]
        if min(prog) > F0 + 4 and max(prog) < N - 1:
            together = True             # every camera under way, none finished yet
    pump(0.05)
    if time.time() - t0 > 900:
        raise TimeoutError("the multi-camera run")
pump(0.3)
assert together, "the cameras were not tracked at the same time (I141)"
assert seen == {0}, f"the working camera stays the working camera during the run: {seen}"
assert side_draws[0] > 3, "the other cameras are drawn live in their own views"
assert side_trail, "with Every camera ticked, the other cameras draw their trails during the run (G33)"
assert p.active == 0, "ends in the working camera"
del win._render_side


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
print(f"I141: one T tracked the point in the 2 cameras that have it and the ball in all 3, all at the same "
      f"time, drawn live, ending in camA ({time.time() - t0:.0f} s) OK")
TOGETHER = {v: {k: getattr(p.sessions[v], k).copy() for k in ("tracks", "confidence", "radius", "visibility",
                                                               "tracked", "manual")} for v in range(3)}

# ---- 2. one Ctrl+Z, every camera -------------------------------------------------------
win._undo_run()
pump(0.2)
for v, s in enumerate(p.sessions):
    jb, jd = s.pid_by_name(name_ball), s.pid_by_name(name_dot)
    assert int(s.tracked[:, jb].sum()) == 1 and s.tracked[F0, jb], f"cam{v}: the ball is back to its one click"
    assert int(s.tracked[:, jd].sum()) == (1 if v < 2 else 0), f"cam{v}: the dot is back to its click"
print("G29: one Ctrl+Z undoes the run in every camera OK")

# ---- 2b. each camera ALONE: exactly the numbers of the simultaneous run (I141) ----------
win.act_track_all.setChecked(False)
for v in range(3):
    win._set_active_view(v)
    pump(0.2)
    win._goto(F0)
    win._deselect()
    win.act_select_all.trigger()   # Track tracks only what is selected (G61): all of it
    pump(0.1)
    win._toggle_tracking()
    wait(lambda: win.state == READY and win.worker is None, 300, f"cam{v} alone")
    pump(0.2)
for v in range(3):
    s = p.sessions[v]
    for k, a in TOGETHER[v].items():
        b = getattr(s, k)
        same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
        assert same, f"cam{v} {k}: tracked alone differs from tracked together"
win._set_active_view(0)
pump(0.2)
for s in p.sessions:                    # back to the clicks for the next part
    s.clear_window(list(range(s.n_points)), F0 + 1, N - 1)
win.act_track_all.setChecked(True)
print("I141: each camera tracked alone gives exactly the same tracks, confidence, radii and flags OK")

# ---- 3. semi-automatic: F steps one frame in every camera -------------------------------
win._goto(F0)                       # the run ended on its last frame; after the undo only F0 has data
win._deselect()
win.act_select_all.trigger()   # Track tracks only what is selected (G61): all of it
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

# ---- 4. X stops every camera at once (I141) ------------------------------------------
win.act_mode_auto.trigger()
QTest.keyClick(win, Qt.Key_T)
wait(lambda: win.state == TRACKING and p.active == 0, 120, "the run")
wait(lambda: min(progress(v) for v in range(3)) > F0 + 4, 120, "every camera under way")
win._pause_tracking()
wait(lambda: win._multi is None and win.state == READY, 120, "the stop")
pump(0.2)
lasts = [progress(v) for v in range(3)]
assert all(x > F0 + 4 for x in lasts) and max(lasts) < N - 1, f"every camera tracked until the stop: {lasts}"
assert max(lasts) - min(lasts) <= 16, f"they stop at about the same frame: {lasts}"
assert "Stopped" in win.toast.text() and "Not tracked yet" not in win.toast.text(), win.toast.text()
assert p.active == 0, "back in the camera being watched"
print(f"I141: X stops every camera at once (last frames {lasts}) OK")

# ---- 5. Every camera ticked, only this camera can start: said, not silent (G32) ------------
jd_a, jd_b = sa.pid_by_name(name_dot), sb.pid_by_name(name_dot)
f_a = int(np.nonzero(sa.tracked[:, jd_a])[0].max())
win._goto(f_a)
pump(0.2)
fb = p.map_frame(0, 1, f_a)
kept = bool(sb.tracked[fb, jd_b])
sb.tracked[fb, jd_b] = False            # camB lacks the dot on its frame of this instant
win.layers.clearSelection()
win.layers.setCurrentItem(win.layers.point_item(jd_a))
win.layers.point_item(jd_a).setSelected(True)
called = []
win._start_tracking = lambda *a, **k: called.append(k)
win._toggle_tracking()
del win._start_tracking
sb.tracked[fb, jd_b] = kept
txt = win.toast.text()
assert called == [{}], f"the working camera still tracks, alone: {called}"
assert "only this camera can start" in txt and "camB has no position for " + name_dot in txt \
    and "camC has no position" in txt, txt
print("G32: Every camera with one camera able to start says which cameras lack the point OK")

# ---- 6. the driver itself: CoTracker3 (a private model per camera) and a broken camera ------
from kinetrace.tracker import MultiTrackingWorker, TrackingWorker  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402


def worker(c, path=None):
    w = TrackingWorker(path or paths[c], F0, dot_r[c, F0].astype(np.float32)[None], [0], FrameCache(256 << 20),
                       N, autopause=False, point_backend="cotracker3")
    out = {"tr": np.full((N, 1, 2), np.nan, np.float32), "cf": np.zeros((N, 1), np.float32), "err": None}
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr, o=out: (o["tr"].__setitem__(slice(w0, w0 + len(tr)), tr),
                                                                   o["cf"].__setitem__(slice(w0, w0 + len(cf)), cf)))
    w.error.connect(lambda m, o=out: o.__setitem__("err", m))
    return w, out


solo = []
for c in (0, 1):
    w, o = worker(c)
    w.run()                                   # on this thread: the signals arrive directly
    solo.append(o)
ws = [worker(0), worker(1), worker(2, os.path.join(OUT, "no_such_video.mp4"))]
MultiTrackingWorker([w for w, _o in ws]).run()
for c in (0, 1):
    a, b = solo[c], ws[c][1]
    assert b["err"] is None and np.array_equal(a["tr"], b["tr"], equal_nan=True) and np.array_equal(a["cf"], b["cf"]), \
        f"cam{c}: CoTracker3 together differs from alone"
assert ws[2][1]["err"] and np.isnan(ws[2][1]["tr"]).all(), "the broken camera reports its error and writes nothing"
print("I141: the driver runs CoTracker3 cameras bit-identically to alone, and a broken camera stops only itself OK")

win._dev_probe.wait(15000)
win.close()
for path in paths:
    forget_recovery(path)
print("TRACK ALL PASSED")
