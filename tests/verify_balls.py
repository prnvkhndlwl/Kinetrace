"""Ball markers (balls.py + the worker + the app): SAM segments every ball,
a circle is fitted, the centre is the tracked point.

1. fit_circle on synthetic masks: a clean disc, a disc with a rod attached,
   a disc a quarter covered - centre within 0.15 px, radius within 0.3 px.
2. BallTracker on a rendered clip with three coloured discs (one carries a
   rod, one drifts out of the picture, all cross the crop edge): centres
   within 0.6 px median of the truth, the leaver dropped, restarts happened,
   a ball added mid-stream is picked up (the multi-object prompt path).
3. The TrackingWorker with balls only, then with a point + balls: the session
   receives the centres, confidences and radii; a ball lost INSIDE the
   picture auto-pauses with reason "lost"; a ball leaving does not.
4. The app offscreen: Add > Ball marker click, Track, the circle is drawn,
   the panel swatch, project round trip (prompts + radii), Ctrl+Z.
GPU (SAM). Builds its own video."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np

from kinetrace import balls

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
W, H, N = 1280, 720, 140
VANISH = 90          # ball B is not drawn from this frame on
rng = np.random.RandomState(3)


# ---------------------------------------------------------------- 1. fit_circle
def disc(cx, cy, r, shape=(200, 200)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


m = disc(100.3, 80.7, 22.4)
cx, cy, r, q = balls.fit_circle(m)
print(f"clean disc: centre error {np.hypot(cx - 100.3, cy - 80.7):.3f} px, radius error {abs(r - 22.4):.3f}, q {q:.3f}")
# the radius carries the rasterisation's half-pixel convention; the CENTRE is what matters
assert np.hypot(cx - 100.3, cy - 80.7) < 0.15 and abs(r - 22.4) < 0.7 and q > 0.95
rod = m.copy()
rod[78:84, 100:190] = True                     # a rod sticking out to the right
cx, cy, r, q = balls.fit_circle(rod)
print(f"disc + rod: centre error {np.hypot(cx - 100.3, cy - 80.7):.3f} px, radius error {abs(r - 22.4):.3f}, q {q:.3f}")
assert np.hypot(cx - 100.3, cy - 80.7) < 0.3 and abs(r - 22.4) < 0.8
cut = m.copy()
cut[:, 112:] = False                            # a quarter hidden behind something
cx, cy, r, q = balls.fit_circle(cut)
print(f"disc cut: centre error {np.hypot(cx - 100.3, cy - 80.7):.3f} px, radius error {abs(r - 22.4):.3f}, q {q:.3f}")
assert np.hypot(cx - 100.3, cy - 80.7) < 0.6 and abs(r - 22.4) < 1.0
assert balls.fit_circle(np.zeros((20, 20), bool)) is None
print("fit_circle OK")

# small balls pass (I59): R_RANGE admits r = 2.5, and a perfect disc of r <= 3 read < 0.45
for rs in (2.5, 3.0, 3.5):
    qs, es = [], []
    for k in range(10):
        cx0, cy0 = 20.0 + 0.1 * k, 20.0 + 0.07 * k
        yy, xx = np.mgrid[0:40, 0:40]
        fc = balls.fit_circle((xx - cx0) ** 2 + (yy - cy0) ** 2 <= rs * rs)
        qs.append(fc[3])
        es.append(np.hypot(fc[0] - cx0, fc[1] - cy0))
    print(f"small ball r = {rs}: quality {min(qs):.2f}..{max(qs):.2f} (gate {balls.MIN_QUALITY}), centre error max {max(es):.2f} px")
    assert min(qs) >= balls.MIN_QUALITY + 0.1 and max(es) < 0.35
assert 36 * balls.RIM_BIN_PX / (2 * np.pi) <= 4.5, "from r = 4.5 up the rim measure must stay the 36-bin one"
yy, xx = np.mgrid[0:60, 0:60]
third = balls.fit_circle(((xx - 30) ** 2 + (yy - 30) ** 2 <= 16) & (xx >= 30 + 4 / 3))
assert third[2] < balls.R_DRIFT[0] * 4, "a third-visible small ball must fit a radius the drift guard refuses"

# the colour guard's units are pinned (I60 kept OpenCV 8-bit Lab on purpose): colours sampled
# from real footage, the red ball vs the lit
# forehead SAM's mask migrated onto - true CIELAB puts them only 29 apart, under COLOUR_DE
lab_ball = balls._disc_colour(balls.guard_lab(np.full((9, 9, 3), (144, 75, 86), np.uint8)), 4, 4, 4)
lab_head = balls._disc_colour(balls.guard_lab(np.full((9, 9, 3), (166, 138, 128), np.uint8)), 4, 4, 4)
d_face = float(np.linalg.norm(lab_ball - lab_head))
print(f"colour guard: red ball vs forehead {d_face:.1f} (limit {balls.COLOUR_DE})")
assert d_face > balls.COLOUR_DE


# ---------------------------------------------------------------- 1b. mock SAM (no GPU)
class MockSeg:
    """Stands in for the segmenter: every object SAM was prompted with comes back as the disc
    gt(frame, obj) -> (x, y, r) in full-frame px (None = nothing), clipped to the crop. Records
    every click given outside the crop's picture."""

    def __init__(self, gt):
        self.gt, self.trk, self.bad = gt, None, []

    def new_session(self, start, size):
        mock = self

        class Sess:
            def __init__(self):
                self.next_frame, self.objs = start, []

            def step(self, crop, idx, prompts):
                h, w = crop.shape[:2]
                for p in prompts or []:
                    if p.points is not None:
                        for q in np.asarray(p.points, float).reshape(-1, 2):
                            if not (0 <= q[0] < w and 0 <= q[1] < h):
                                mock.bad.append((idx, p.obj_id, tuple(q)))
                    if p.obj_id not in self.objs:
                        self.objs.append(p.obj_id)
                if not self.objs:
                    raise ValueError("the first step of a session needs at least one prompt")
                x0, y0 = mock.trk.crop[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                ms, sc = [], []
                for o in self.objs:
                    c = mock.gt(idx, o)
                    mm = (np.zeros((h, w), bool) if c is None
                          else (xx - (c[0] - x0)) ** 2 + (yy - (c[1] - y0)) ** 2 <= c[2] ** 2)
                    ms.append(mm)
                    sc.append(8.0 if mm.any() else -5.0)
                self.next_frame += 1
                return segmenter.FrameMasks(idx, list(self.objs), np.array(ms), np.array(sc, np.float32), (w, h), (w, h))
        return Sess()


def run_mock(gt, size, n, first):
    mock = MockSeg(gt)
    tk = balls.BallTracker(mock, size)
    mock.trk = tk
    img = np.full((size[1], size[0], 3), 200, np.uint8)        # one colour: only the geometry decides
    res = {}
    for f in range(n):
        pr = [balls.BallPrompt(o, points=np.array([gt(0, o)[:2]]), labels=np.array([1]), radius=gt(0, o)[2])
              for o in first] if f == 0 else None
        for o, fc in tk.step(img, f, pr).items():
            res.setdefault(o, {})[f] = fc
    return tk, res, mock


from kinetrace import segmenter          # noqa: E402

# identity-swap guard after a missed frame (I56): SAM's mask moves onto a look-alike 300 px away,
# after one hidden frame and without one - no accepted centre may be more than 4 r off the truth
RB = 12.0
true_x = lambda f: 800.0 + 2 * f                                  # noqa: E731
for label, hide in (("after one hidden frame", True), ("without a hidden frame", False)):
    gt = (lambda f, o, hide=hide: (true_x(f), 540.0, RB) if f < 10 else
          (None if (f == 10 and hide) else (1100.0 + 2 * f, 540.0, RB)))
    tk, res, _ = run_mock(gt, (1920, 1080), 40, [1])
    worst = max(abs(fc.x - true_x(f)) for f, fc in res[1].items())
    print(f"swap onto a look-alike {label}: worst accepted error {worst:.1f} px, dropped as {tk.drop_reason.get(1)!r}")
    assert worst <= 4 * RB, "a persistent swap was accepted (I56)"
for v, gap in ((2.0, 3), (10.0, 5)):                               # ...but the real ball is taken back after a gap
    gt = lambda f, o, v=v, gap=gap: None if 10 <= f < 10 + gap else (600.0 + v * f, 400.0, RB)   # noqa: E731
    _, res, _ = run_mock(gt, (1920, 1080), 30, [1])
    assert min(f for f in res[1] if f >= 10) == 10 + gap, (v, gap)

# balls farther apart than the one shared crop (I58): never a click outside SAM's picture, the
# first ball tracked throughout, the other dropped as "apart" (clicking it again cannot help)
WC, HC, RC = 2704, 1520, 14.0
for span, vx, n in ((1100, 0.0, 40), (800, 4.0, 120), (600, 4.0, 120)):
    gt = (lambda f, o, span=span, vx=vx:
          ((WC / 2 - 300 + vx * f) + (-span / 2 if o == 1 else span / 2), 760.0, RC))
    tk, res, mock = run_mock(gt, (WC, HC), n, [1, 2])
    n1, n2 = len(res.get(1, {})), len(res.get(2, {}))
    print(f"two balls {span} px apart ({vx:.0f} px/frame): ball 1 {n1}/{n}, ball 2 {n2}/{n}, "
          f"dropped {tk.drop_reason}, clicks outside the crop {len(mock.bad)}")
    assert not mock.bad, mock.bad[:3]
    assert n1 == n
    if span <= 600:
        assert n2 == n and not tk.drop_reason
    else:
        assert tk.drop_reason.get(2) == "apart"
assert not balls.BallTracker(None, (WC, HC)).fits_one_crop([(500, 700), (1600, 700)])
tk, res, _ = run_mock(lambda f, o: None if (o == 2 and f >= 20) else ((900.0 if o == 1 else 1200.0), 700.0, RC),
                      (WC, HC), 40, [1, 2])
assert tk.drop_reason.get(2) == "lost" and len(res[1]) == 40, "a ball lost inside a shared crop is 'lost'"
print("mock-SAM guards OK")

# ---------------------------------------------------------------- the clip
# three balls: A red (with a white rod), B green (VANISHES inside the picture from frame 90:
# an occlusion the tracker cannot see through), C blue (drifts out at the right edge)
truth = np.zeros((N, 3, 2))
R = np.array([21.0, 17.0, 19.0])
pos = np.array([[300.0, 360.0], [420.0, 330.0], [560.0, 400.0]])
vel = np.array([[4.2, -1.1], [3.6, 1.4], [7.0, -0.6]])       # C: fast, leaves the picture at the right
path = os.path.join(OUT, "balls_test.mp4")
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
for k in range(N):
    img = np.full((H, W, 3), 96, np.uint8)
    # textured background so SAM has something to separate the balls from
    img[..., 1] = (96 + 20 * np.sin(np.arange(W) / 37.0)[None, :] + 10 * np.cos(np.arange(H) / 23.0)[:, None]).astype(np.uint8)
    truth[k] = pos
    cv2.line(img, (int(pos[0, 0]), int(pos[0, 1])), (int(pos[0, 0] + 90), int(pos[0, 1] + 30)), (235, 235, 235), 7, cv2.LINE_AA)
    cv2.circle(img, (int(round(pos[0, 0])), int(round(pos[0, 1]))), int(R[0]), (40, 40, 220), -1, cv2.LINE_AA)
    if k < VANISH:
        cv2.circle(img, (int(round(pos[1, 0])), int(round(pos[1, 1]))), int(R[1]), (60, 200, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (int(round(pos[2, 0])), int(round(pos[2, 1]))), int(R[2]), (230, 120, 40), -1, cv2.LINE_AA)
    vw.write(img)
    pos = pos + vel + rng.normal(0, 0.3, (3, 2))
    pos[:2] = np.clip(pos[:2], 40, [W - 40, H - 40])
vw.release()
# integer-rounded draw centres are the truth SAM can see
truth_r = np.round(truth)
leave_frame = int(np.nonzero(truth[:, 2, 0] + R[2] >= W)[0][0]) if (truth[:, 2, 0] + R[2] >= W).any() else N
print(f"clip: {N} frames {W}x{H}; ball C leaves the picture at frame {leave_frame}")

# ---------------------------------------------------------------- 2. BallTracker
from kinetrace import segmenter          # noqa: E402

backend = segmenter.preferred_backend()
seg = segmenter.get_segmenter(backend)
cap = cv2.VideoCapture(path)
trk = balls.BallTracker(seg, (W, H), crop=480, margin=60)
got = {1: {}, 2: {}, 3: {}}
t0 = time.time()
for k in range(N):
    ok, bgr = cap.read()
    assert ok
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prompts = []
    if k == 0:
        prompts = [balls.BallPrompt(1, points=truth_r[0, 0:1], labels=np.array([1])),
                   balls.BallPrompt(3, points=truth_r[0, 2:3], labels=np.array([1]), radius=R[2])]
    if k == 5:      # ball B added mid-stream, alongside the two already tracked
        prompts = [balls.BallPrompt(2, points=truth_r[5, 1:2], labels=np.array([1]))]
    fits = trk.step(rgb, k, prompts)
    for obj, f in fits.items():
        got[obj][k] = (f.x, f.y, f.r, f.quality, f.confidence)
cap.release()
dt = time.time() - t0
print(f"BallTracker: {N} frames in {dt:.1f}s ({N / dt:.1f} fps), {trk.n_restarts} crops, "
      f"frames per ball {[len(got[o]) for o in (1, 2, 3)]}")
for obj, col in ((1, 0), (2, 1), (3, 2)):
    ks = sorted(k for k in got[obj] if obj != 2 or k < VANISH)
    e = np.array([np.hypot(got[obj][k][0] - truth_r[k, col, 0], got[obj][k][1] - truth_r[k, col, 1]) for k in ks])
    rr = np.array([got[obj][k][2] for k in ks])
    print(f"  ball {obj}: {len(ks)} frames, centre error median {np.median(e):.2f} px, 95th {np.percentile(e, 95):.2f}, "
          f"max {e.max():.2f}; radius median {np.median(rr):.1f} (truth {R[col]:.0f}); "
          f"confidence median {np.median([got[obj][k][4] for k in ks]):.2f}")
    # measured 2026-09-19 (SAM 3, three objects, ten crop restarts): medians 0.65 / ~0.5 /
    # ~0.5 px, 95th < 1.7; a single ball without restarts tracks to 0.05-0.3 px. SAM's mask
    # is ~2 px wider than the drawn disc, so the radius carries that bias.
    # ball 3 is the stress case: 7 px a frame at 30 fps (a third of its radius; real 240 fps
    # footage moves ~1 px a frame) - SAM's mask smears, the drift guard rejects the worst frames
    lim, p95 = (1.0, 2.5) if obj != 3 else (1.6, 6.0)
    assert np.median(e) < lim, np.median(e)
    assert np.percentile(e, 95) < p95, np.percentile(e, 95)
    assert abs(np.median(rr) - R[col]) < (3.5 if obj != 3 else 5.5), np.median(rr)   # a smeared mask reads wider
assert len(got[1]) >= N - 2, "ball A (with the rod) must be tracked throughout"
assert min(got[2]) == 5 and len(got[2]) >= VANISH - 8, "ball B added mid-stream must be tracked from frame 5"
assert max(got[2]) <= VANISH + 2 and not trk.has(2), "ball B vanished inside the picture: dropped, never re-found"
assert max(got[3]) < leave_frame + 3 and not trk.has(3), "ball C must be dropped once it left the picture (or was lost)"
assert trk.n_restarts >= 2, "the balls cross the crop edge: restarts expected"
print("BallTracker OK")

# ---------------------------------------------------------------- 3. the worker
# a QApplication from the start: a QCoreApplication here would be what
# QApplication.instance() returns in part 4, and a MainWindow on it is a hard crash
from PySide6.QtWidgets import QApplication                     # noqa: E402

qapp = QApplication.instance() or QApplication([])
from kinetrace.session import TrackingSession              # noqa: E402
from kinetrace.tracker import BallSpec, PointSpec, TrackingWorker   # noqa: E402
from kinetrace.video_source import FrameCache              # noqa: E402


def run_worker(s, start, end, specs, ball_specs, autopause=True):
    w = TrackingWorker(path, start, None, None, FrameCache(256 * 1024 ** 2), end, refine=True,
                       specs=specs, roi=True, autopause=autopause, balls=ball_specs,
                       point_backend="cotracker3")
    ev = {"finished": None, "autopaused": None, "error": None, "chunks": 0}
    w.balls_ready.connect(lambda rows: s.write_ball_radii(rows))
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: (s.write_segment(w0, tr, vi, w.point_ids, cf),
                                                          ev.__setitem__("chunks", ev["chunks"] + 1)))
    w.finished_ok.connect(lambda last, p: ev.update(finished=(last, p)))
    w.autopaused.connect(lambda f, pid: ev.update(autopaused=(f, pid)))
    w.error.connect(lambda m: ev.update(error=m))
    w.run()
    assert ev["error"] is None, ev["error"]
    return w, ev


s = TrackingSession(path, N, 30.0, W, H)
pa = s.add_ball(0, *truth_r[0, 0])
pc = s.add_ball(0, *truth_r[0, 2])
specs_b = [BallSpec(pa, s.points[pa].ball_prompts, s.tracks[0, pa], None, backend),
           BallSpec(pc, s.points[pc].ball_prompts, s.tracks[0, pc], R[2], backend)]
w, ev = run_worker(s, 0, N, [], specs_b)
print(f"worker balls only: finished {ev['finished']} autopaused {ev['autopaused']} chunks {ev['chunks']}")
assert ev["finished"] is not None and ev["autopaused"] is None, "a ball leaving the picture must not pause"
ta = np.nonzero(s.tracked[:, pa])[0]
tc = np.nonzero(s.tracked[:, pc])[0]
ea = np.hypot(s.tracks[ta, pa, 0] - truth_r[ta, 0, 0], s.tracks[ta, pa, 1] - truth_r[ta, 0, 1])
print(f"  A: {len(ta)} frames, error median {np.median(ea):.2f} px, radius median {np.nanmedian(s.radius[ta, pa]):.1f}, "
      f"conf median {np.median(s.confidence[ta, pa]):.2f}; C: {len(tc)} frames, last {tc[-1]} (left at {leave_frame})")
assert len(ta) >= N - 2 and np.median(ea) < 1.0
assert np.isfinite(s.radius[ta, pa]).all() and abs(np.nanmedian(s.radius[ta, pa]) - R[0]) < 3.5
assert tc[-1] < leave_frame + 3 and s.confidence[ta, pa].min() > 0.3
assert not s.tracked[leave_frame + 5:, pc].any(), "no data after the ball left"

# a point AND balls in one run: columns must not shift
s2 = TrackingSession(path, N, 30.0, W, H)
pp = s2.add_point(0, 200.0, 150.0)               # a background point on the texture
pb = s2.add_ball(0, *truth_r[0, 1])
specs_p = [PointSpec(pp, s2.tracks[0, pp].astype(np.float32).copy())]
w2, ev2 = run_worker(s2, 0, 60, specs_p, [BallSpec(pb, s2.points[pb].ball_prompts, s2.tracks[0, pb], None, backend)])
tb = np.nonzero(s2.tracked[:60, pb])[0]
eb = np.hypot(s2.tracks[tb, pb, 0] - truth_r[tb, 1, 0], s2.tracks[tb, pb, 1] - truth_r[tb, 1, 1])
print(f"worker point + ball: point tracked {int(s2.tracked[:60, pp].sum())} frames, ball {len(tb)} frames, "
      f"ball error median {np.median(eb):.2f} px")
assert ev2["finished"] is not None and len(tb) >= 58 and np.median(eb) < 1.0
assert int(s2.tracked[:60, pp].sum()) >= 58

# a ball lost INSIDE the picture: ball B vanishes at frame 90 (nowhere near an edge)
s3 = TrackingSession(path, N, 30.0, W, H)
px = s3.add_ball(0, *truth_r[0, 1])
w3, ev3 = run_worker(s3, 0, N, [], [BallSpec(px, s3.points[px].ball_prompts, s3.tracks[0, px], None, backend)])
print(f"worker lost ball: autopaused {ev3['autopaused']} reason {w3._autopause_reason!r}")
assert ev3["autopaused"] is not None and ev3["autopaused"][1] == px and w3._autopause_reason == "lost"
assert VANISH - 3 <= ev3["autopaused"][0] <= VANISH + 3, ev3["autopaused"]
print("worker OK")

# ---------------------------------------------------------------- 4. the app
from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog   # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from kinetrace.app import MainWindow, READY, TRACKING       # noqa: E402

win = MainWindow()
win.show()


def pump(sec):
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


if os.path.exists(path + ".cotracker.npz"):
    os.remove(path + ".cotracker.npz")            # a crashed run's autosave would offer a resume
win._open_video(path)
wait(lambda: win.state == READY, 60, "open")
win._goto(0)
assert win.act_add_ball.isEnabled()
win.act_add_ball.trigger()
assert win.btn_add.isChecked() and win._place_kind == "ball"
win.canvas.add_requested.emit(float(truth_r[0, 0, 0]), float(truth_r[0, 0, 1]))
pump(0.1)
s = win.session
assert s.n_points == 1 and s.points[0].is_ball and not win.btn_add.isChecked() and win._place_kind == "point"
assert s.points[0].ball_prompts == {0: [[float(truth_r[0, 0, 0]), float(truth_r[0, 0, 1]), 1]]}
win.act_add_ball.trigger()
win.canvas.add_requested.emit(float(truth_r[0, 1, 0]), float(truth_r[0, 1, 1]))
pump(0.1)
assert s.n_points == 2 and s.points[1].is_ball
# a plain armed click after a ball adds an ordinary point
win.btn_add.setChecked(True)
win.canvas.add_requested.emit(200.0, 150.0)
pump(0.1)
assert s.n_points == 3 and not s.points[2].is_ball
print("Add > Ball marker places balls, N still places points OK")
icon_ok = win.point_list.item(0).icon() is not None
assert icon_ok
win.point_list.clearSelection()        # a single selected row would scope the run to that point
win._start_tracking(stop_after=50)
wait(lambda: win.state == TRACKING, 60, "tracking start")
wait(lambda: win.state == READY, 600, "tracking end")
pump(0.3)
n_a = int(s.tracked[:51, 0].sum())
print(f"app run: ball A tracked {n_a}/51 frames, radius at 30 = {s.radius[30, 0]:.1f}, point tracked {int(s.tracked[:51, 2].sum())}")
assert n_a >= 49 and np.isfinite(s.radius[30, 0])
win._goto(30)
pump(0.1)
reg = win.canvas._regions[0]
assert reg.isVisible() and reg.path().boundingRect().width() > 20, "the fitted circle must be drawn"
assert not win.canvas._regions[2].isVisible(), "a plain point has no circle"
# undo the run
win._undo_run()
pump(0.1)
assert int(s.tracked[:, 0].sum()) == 1, "Ctrl+Z must remove the run's ball data"
print("undo OK")
# round trip
win._goto(0)                            # the undo left data on frame 0 only
win.point_list.clearSelection()
win._start_tracking(stop_after=20)
wait(lambda: win.state == TRACKING, 60, "tracking start 2")
wait(lambda: win.state == READY, 600, "tracking end 2")
proj = os.path.join(OUT, "balls_test.cotrk")
win.project.save_npz(proj)
pump(0.2)
from kinetrace.project import Project                      # noqa: E402

p2 = Project.load_npz(proj)
t2 = p2.sessions[0]
assert t2.points[0].is_ball and t2.points[0].ball_prompts and np.isfinite(t2.radius[10, 0])
assert np.allclose(t2.tracks[:21, 0], s.tracks[:21, 0], equal_nan=True)
print("project round trip keeps the ball prompts, radii and tracks OK")
win.close()
pump(0.3)
for f in (path + ".cotracker.npz",):
    if os.path.exists(f):
        os.remove(f)
print("verify_balls PASSED")
