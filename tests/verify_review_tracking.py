"""Regression checks for the tracking-engine fixes of the 2026-10-02 code review
(docs/AUDIT.md: I186, I187, I191, I192, I193, I194, I195, I240, I251, I252, I256, I257, G118, R3).

Every check fails on the code before the review and passes now (run it against the base copy with
KINETRACE_REVIEW_ROOT=<folder holding the old kinetrace/> to see the FAIL lines). A plain script: ASCII
output, "VERIFY_REVIEW_TRACKING PASSED" at the end, non-zero exit on any failure. Uses the GPU (real
CoTracker3 for the checks that need a point model); builds its own clips under tests/out/review_tracking.
"""
import os
import sys
import tempfile
import time
import traceback
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("KINETRACE_REVIEW_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

import kinetrace  # noqa: E402
from _synth_spots import make_scene, write_video  # noqa: E402
from kinetrace import pointtest, segmenter, spots  # noqa: E402
from kinetrace import tracker as trk  # noqa: E402
from kinetrace import video_source as vs  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

print("testing", os.path.dirname(os.path.abspath(kinetrace.__file__)))
WT = os.path.dirname(HERE)
OUT = os.path.join(WT, "tests", "out", "review_tracking")
os.makedirs(OUT, exist_ok=True)
V600 = os.path.join(WT, "test600.mp4")
V4K = os.path.join(WT, "test4k.mp4")
GT600 = np.load(V600 + ".gt.npz")["gt"]
GT4K = np.load(V4K + ".gt.npz")["gt"]
FAILS = []


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except BaseException as e:      # noqa: BLE001 - every check reports, the run goes on
            FAILS.append(name)
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]} ({os.path.basename(tb.filename)}:{tb.lineno})")
        return fn
    return deco


def run_worker(w):
    """Run a TrackingWorker synchronously, recording what it emits."""
    rec = {"chunks": [], "fin": None, "err": None, "paused": None, "balls": []}
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mm, fr: rec["chunks"].append((w0, tr.copy(), vi.copy(), cf.copy())))
    w.finished_ok.connect(lambda last, p: rec.__setitem__("fin", (last, p)))
    w.error.connect(lambda m: rec.__setitem__("err", m))
    w.autopaused.connect(lambda f, pid: rec.__setitem__("paused", (f, pid)))
    w.balls_ready.connect(lambda r: rec["balls"].extend(r))
    w.run()
    return rec


def column(rec, k, n=None):
    """{frame: (x, y)} for output column k; the LAST write of a frame wins, as in the session."""
    out = {}
    for w0, tr, vi, cf in rec["chunks"]:
        for i in range(tr.shape[0]):
            p = tr[i, k]
            out[w0 + i] = None if not np.isfinite(p).all() else tuple(p)
    return out


def data_frames(rec, k):
    return sorted(f for f, p in column(rec, k).items() if p is not None)


# ---------------------------------------------------------------- I186 / I251 / I252 / G118: the point-model test
class _Sig:
    def __init__(self):
        self.fns = []

    def connect(self, fn):
        self.fns.append(fn)

    def emit(self, *a):
        for fn in self.fns:
            fn(*a)


class FakeModelWorker:
    """Stands in for TrackingWorker in pointtest: a model that follows a straight clicked line
    perfectly, except that a run started at frame 0 drifts 20 px from frame 12 on. Emits 16-row
    windows every 8 frames like the real models."""
    truth = staticmethod(lambda g: np.array([10.0 * g, 5.0]))
    starts = []

    def __init__(self, video_path, start, seed, ids, cache, n_end, refine=True, specs=None, roi=True,
                 autopause=False, point_backend="alltracker"):
        self.start, self.n_end = int(start), int(n_end)
        self.chunk_ready, self.error = _Sig(), _Sig()
        self.paused = False
        FakeModelWorker.starts.append(self.start)

    def request_pause(self):
        self.paused = True

    def run(self):
        f0 = self.start
        w0 = f0
        while w0 < self.n_end and not self.paused:
            L = min(16, self.n_end - w0)
            tr = np.zeros((L, 1, 2), np.float32)
            for i in range(L):
                p = self.truth(w0 + i)
                if f0 == 0 and w0 + i >= 12:
                    p = p + [0.0, 20.0]
                tr[i, 0] = p
            self.chunk_ready.emit(w0, tr, np.ones((L, 1), bool), np.ones((L, 1), np.float32), {}, [])
            w0 += 8


@check("I186 pointtest: a re-run after a correction judges only the clicks after its start")
def _():
    import kinetrace.tracker as tk
    real = tk.TrackingWorker
    tk.TrackingWorker = FakeModelWorker
    FakeModelWorker.starts = []
    try:
        clicks = {f: np.array([10.0 * f, 5.0]) for f in range(30)}
        t = pointtest._TestRun(V600, FrameCache(1 << 20), 600, clicks, (640, 480), ["alltracker"], True)
        t.limit = 6.0
        r = t._model("alltracker")
    finally:
        tk.TrackingWorker = real
    assert r.drifts == 1 and r.corrections == 1 and r.stops == 0, (r.corrections, r.drifts, r.stops, FakeModelWorker.starts)
    assert FakeModelWorker.starts == [0, 12], FakeModelWorker.starts


@check("I251 the scorer counts only the clicked frames it reached and says where the video stopped")
def _():
    fr, gt = make_scene("sea", n=30)
    clicks = {f: gt[f] for f in range(30)}
    st = spots.candidate_settings(1.5)[:2]
    res = spots.score_spot_settings(((f, fr[f]) for f in range(10)), clicks, st, (960, 540), 8.0)
    assert res[0].frames == 9, res[0].frames                     # clicked frames 1..9 of 29
    assert getattr(res[0], "unread", None) == 10, getattr(res[0], "unread", "n/a")
    res = spots.score_spot_settings(((f, fr[f]) for f in range(30)), clicks, st, (960, 540), 8.0)
    assert res[0].frames == 29 and res[0].unread is None


@check("I251 the test thread names the frame the video would not deliver and leaves later clicks out")
def _():
    fr, gt = make_scene("sea", n=60)
    vid = os.path.join(OUT, "sea45.mp4")
    write_video(vid, fr[:45])                       # the file holds 45 frames; the app believes 80
    clicks = {f: gt[f] + 0.3 for f in range(20, 50)}
    t = pointtest._TestRun(vid, FrameCache(1 << 28), 80, clicks, (960, 540), [], True)
    res = t._run()
    assert t.unread == 45, t.unread
    assert max(t.clicks) == 44 and all(r.frames == 24 for r in res), ([r.frames for r in res], max(t.clicks))
    assert res[0].unread == 45 and "Frame 45" in spots.verdict_text("P1", res)


@check("I252 the dialog's thread stop is idempotent (one wait, one orphan)")
def _():
    class FakeThread:
        def __init__(self):
            self.waits, self.cancels = 0, 0
            self.finished = _Sig()

        def isRunning(self):
            return True

        def cancel(self):
            self.cancels += 1

        def wait(self, ms):
            self.waits += 1
            return False

    retired = []
    fake_app = types.ModuleType("kinetrace.app")
    fake_app._retire = lambda th: retired.append(th)
    saved = sys.modules.get("kinetrace.app")
    sys.modules["kinetrace.app"] = fake_app
    pointtest._ORPHANS.clear()
    try:
        th = FakeThread()
        dlg = types.SimpleNamespace(_thread=th)
        pointtest.PointModelTest._stop_thread(dlg)      # reject()
        pointtest.PointModelTest._stop_thread(dlg)      # closeEvent() right after it
        assert th.waits == 1 and th.cancels == 1, (th.waits, th.cancels)
        assert pointtest._ORPHANS == [th], pointtest._ORPHANS
        assert retired == [th], "the app's registry (what closeEvent waits for) gets it once"
    finally:
        pointtest._ORPHANS.clear()
        if saved is None:
            sys.modules.pop("kinetrace.app", None)
        else:
            sys.modules["kinetrace.app"] = saved


@check("G118 the verdict and the button say 'for <point>', not 'for this project'")
def _():
    r_spot = spots.TestResult("Moving spot: bright spot, search 6 px", "spot", spots.SpotSettings("bright", 6, 0, 1.5),
                              frames=10, first_ok=10, errors=[1.0])
    txt = spots.verdict_text("spot_1", [r_spot])
    assert "for spot_1" in txt and "for this project" not in txt, txt
    r_at = spots.TestResult("AllTracker", "alltracker", frames=10, first_ok=10, errors=[0.5])
    txt = spots.verdict_text("spot_1", [r_at])
    assert "for spot_1" in txt and "for this project" not in txt, txt
    dlg = pointtest.PointModelTest(None, "spot_1", V600, FrameCache(1 << 20), 600, (640, 480),
                                   {f: np.array([1.0 * f, 5.0]) for f in range(25)}, [], True, False)
    dlg._thread = types.SimpleNamespace(_cancel=False, limit=6.0, look=None, unread=None)
    dlg._on_done([r_spot])
    assert "for spot_1" in dlg.btn_use.text() and "for this project" not in dlg.btn_use.text(), dlg.btn_use.text()
    assert "spot_1" in dlg.btn_use.text()


# ---------------------------------------------------------------- I187 / I194: Moving spot
def blob_frame(w=200, h=200, spots_=((100.0, 100.0, 150.0), (115.0, 100.0, 200.0)), sigma=1.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.full((h, w), 60.0, np.float32)
    for x, y, a in spots_:
        g += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


@check("I187 one click off the spot: Moving spot stops instead of adopting the strongest glint")
def _():
    img = blob_frame()
    for vel in (None, np.array([2.0, 0.0])):
        t = spots.SpotTracker(spots.SpotSettings("bright", 20.0, 0.0, 1.5), (106.0, 100.0), vel, (200, 200))
        t.start(0, img)
        assert not t.ref, "the click sat on no peak: no reference strength"
        fix = t.step(1, img)
        assert fix is None and t.stopped is not None and t.stopped[1] == "missing", (vel, fix, t.stopped)
    # with a reference (the click ON the spot) nothing changes: the spot is followed
    t = spots.SpotTracker(spots.SpotSettings("bright", 6.0, 0.0, 1.5), (100.0, 100.0), np.array([0.0, 0.0]), (200, 200))
    t.start(0, img)
    assert t.ref and t.step(1, img) is not None


@check("I194 a change-cue run from frame 9 / 10 reads its look-ahead (usable frames, not read frames)")
def _():
    fr, gt = make_scene("river", n=60)
    st = spots.SpotSettings("change", 25.0, 2.5, 1.5)

    def follow(start, n_video):
        v = gt[start + 1] - gt[start]
        run = spots.SpotRun([(0, gt[start], v, st)], (fr[0].shape[1], fr[0].shape[0]), start)
        run.resolve(fr[start])
        run.prefill(lambda k: fr[k] if 0 <= k < len(fr) else None, n_video)
        n = 0
        for f in range(start, 40):
            if 0 not in run.step(f, fr[f]):
                break
            n += 1
        return n, run.trackers[0].stopped

    for s0 in (9, 10):
        n, stopped = follow(s0, len(fr))
        assert stopped is None and n == 40 - s0, (s0, n, stopped)


@check("I194 a one-step run near the video's start reads its look-ahead past the run's end")
def _():
    fr, gt = make_scene("river", n=60)
    vid = os.path.join(OUT, "river60.mp4")
    write_video(vid, fr)
    st = {"cue": "change", "radius": 25.0, "speed_gain": 2.5, "sigma": 1.5}
    w = trk.TrackingWorker(vid, 3, None, None, FrameCache(1 << 28), 5, specs=[],
                           spots=[trk.SpotSpec(0, gt[3], gt[4] - gt[3], st)], autopause=True)
    rec = run_worker(w)
    assert rec["err"] is None, rec["err"]
    assert data_frames(rec, 0) == [3, 4] and rec["paused"] is None, (data_frames(rec, 0), rec["paused"])


@check("I194 one rule: the scorer and the worker seed the history alike")
def _():
    fr, gt = make_scene("river", n=40)
    grey = lambda k: cv2.cvtColor(fr[k], cv2.COLOR_RGB2GRAY)                          # noqa: E731
    past = [(k, grey(k)) for k in range(0, 9, 2)]
    ahead = [(k, grey(k)) for k in range(11, 40, 2)]
    h = spots.ChangeHistory()
    h.seed(past, ahead, 9)
    assert len(h.frames_for(9)) >= spots.MIN_HIST, len(h.frames_for(9))
    p2, a2 = spots.read_background(lambda k: fr[k] if k < len(fr) else None, 9)
    assert [k for k, _ in p2] == [0, 2, 4, 6, 8] and a2 and a2[0][0] == 11, ([k for k, _ in p2], [k for k, _ in a2][:3])


# ---------------------------------------------------------------- I191 / I257 / I192: the worker's loops
class MockBallSeg:
    """SAM stand-in for balls (tests/verify_balls.py's): the disc gt(frame, obj) -> (x, y, r), clipped to the crop."""

    def __init__(self, gt):
        self.gt = gt

    def new_session(self, start, size):
        mock = self
        x0, y0 = sys._getframe(1).f_locals["self"].crop[:2]

        class Sess:
            def __init__(self):
                self.next_frame, self.objs = start, []

            def step(self, crop, idx, prompts):
                h, w = crop.shape[:2]
                for p in prompts or []:
                    if p.obj_id not in self.objs:
                        self.objs.append(p.obj_id)
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


def with_mock_balls(gt, fn):
    saved = trk.get_segmenter
    mock = MockBallSeg(gt)
    trk.get_segmenter = lambda backend, **kw: mock
    try:
        return fn()
    finally:
        trk.get_segmenter = saved


@check("I191 a ball dropped earlier takes the user's later click; a run with every ball gone ends")
def _():
    R = 14.0
    # both balls run off the right edge of the 640 px picture (no pause); B comes back on frame 50
    a = lambda f: (400.0 + 8 * f, 120.0, R) if 400 + 8 * f < 660 else None                     # noqa: E731
    b = lambda f: ((500.0 + 6 * f, 300.0, R) if 500 + 6 * f < 660 else                         # noqa: E731
                   ((300.0 + 3 * (f - 50), 300.0, R) if f >= 50 else None))
    gt = lambda f, o: {1000: a, 1001: b}[o](f)                                                 # noqa: E731

    def balls_run(b_prompts, n=70):
        balls = [trk.BallSpec(1000, {0: [[400.0, 120.0, 1]]}, backend="stub"),
                 trk.BallSpec(1001, b_prompts, backend="stub")]
        w = trk.TrackingWorker(V600, 0, None, None, FrameCache(1 << 27), n, specs=[], balls=balls)
        return w, run_worker(w)

    # the ball leaves through the right edge, A is lost too (the picture's edge: no pause), and the user
    # clicked B again on frame 50 where it re-enters
    w, rec = with_mock_balls(gt, lambda: balls_run({0: [[500.0, 300.0, 1]], 50: [[300.0, 300.0, 1]]}))
    got = data_frames(rec, 1)
    assert rec["err"] is None, rec["err"]
    assert rec["paused"] is None, rec["paused"]
    assert any(f >= 50 for f in got) and got[-1] >= 65, f"the clicked ball is followed again: {got[-5:]}"
    # no later click: with every ball gone the run ends, it does not decode to the video's end
    w, rec = with_mock_balls(gt, lambda: balls_run({0: [[500.0, 300.0, 1]]}))
    assert rec["fin"] is not None and rec["fin"][0] < 50, rec["fin"]


@check("I257 the hand-over after every point was dropped only goes on while something can give data")
def _():
    import kinetrace.spots as spots_mod

    class _Run:                                             # the spot run itself is not needed here
        trackers = {}

        def __init__(self, *a, **k):
            pass

        def resolve(self, rgb):
            pass

        def prefill(self, *a, **k):
            pass

    for spot_gone, expect_hand_over in ((False, True), (True, False)):
        calls = []
        w = trk.TrackingWorker(V600, 0, None, None, FrameCache(1 << 20), 600, autopause=False,
                               specs=[trk.PointSpec(0, GT600[0, 0].astype(np.float32))],
                               spots=[trk.SpotSpec(7, GT600[0, 1], None, {})])

        def seg_stub(src, *a, w=w, spot_gone=spot_gone):
            if spot_gone:
                w._spot_gone.add(7)                         # the spot has stopped
            return "restart", 40, [None]                    # ... and every point is dropped at a restart
            yield                                           # noqa: unreachable -- makes it a generator

        def animal_stub(src, start, calls=calls):
            calls.append(start)
            return 599
            yield                                           # noqa: unreachable -- makes it a generator

        w._segment_steps, w._animal_only_steps = seg_stub, animal_stub
        saved = (trk.get_model, spots_mod.SpotRun)
        trk.get_model = lambda **kw: (None, "cpu")
        spots_mod.SpotRun = _Run
        rec = {}
        w.finished_ok.connect(lambda last, p: rec.__setitem__("fin", (last, p)))
        try:
            w.run()
        finally:
            trk.get_model, spots_mod.SpotRun = saved
        assert bool(calls) == expect_hand_over, (spot_gone, calls)
        assert rec["fin"] == ((599, False) if expect_hand_over else (40, False)), rec


@check("I192 a Moving-spot stop in a run that also tracks a model point: the point ends at the stop frame too")
def _():
    # the sea scene's small spot (vanishes on frame 45) + a dark-ringed red disc the point models follow cleanly
    fr, gt = make_scene("sea", n=60, vanish_at=45)
    dot = lambda f: np.array([100.0 + 6 * f, 450.0 - 4 * f], np.float32)       # noqa: E731
    frames = []
    for f, img in enumerate(fr):
        img = img.copy()
        x, y = int(round(float(dot(f)[0]))), int(round(float(dot(f)[1])))
        cv2.circle(img, (x, y), 15, (20, 20, 20), -1)
        cv2.circle(img, (x, y), 9, (230, 60, 60), -1)
        cv2.circle(img, (x + 3, y - 2), 3, (250, 250, 90), -1)
        frames.append(img)
    vid = os.path.join(OUT, "sea_dot_vanish45.mp4")
    write_video(vid, frames)
    for backend in ("cotracker3", "alltracker"):
        w = trk.TrackingWorker(vid, 0, None, None, FrameCache(1 << 29), 60, refine=True, point_backend=backend,
                               specs=[trk.PointSpec(0, dot(0))],
                               spots=[trk.SpotSpec(5, gt[0], gt[1] - gt[0], {})], autopause=True)
        rec = run_worker(w)
        assert rec["err"] is None, rec["err"]
        pt, sp = data_frames(rec, 0), data_frames(rec, 1)
        assert sp[-1] == 44 and w._spot_ended == {5: (45, "missing")}, (sp[-3:], w._spot_ended)
        assert rec["paused"] == (45, 5) and w._autopause_reason == "spot"
        assert pt[-1] >= 44, f"{backend}: the model point ended at {pt[-1]}, 15 frames early; the spot at {sp[-1]}"
        assert pt == list(range(pt[0], pt[-1] + 1)) and rec["fin"][0] >= 44, (rec["fin"], pt[-3:])


# ---------------------------------------------------------------- I193: derived landmarks keep their orientation
@check("I193 an overlap row is oriented against the frame before it, not the last row computed")
def _():
    W, H = 320, 240
    w = trk.TrackingWorker(V600, 0, None, None, FrameCache(1 << 20), 100,
                           specs=[trk.PointSpec(0, np.array([160.0, 120.0], np.float32))],
                           animal=trk.AnimalSpec({0: [(160.0, 120.0, 1)]}, {}, None, "stub"),
                           derived=[trk.DerivedSpec(5, "tip")], head_pid=0)
    w._frame_wh = (W, H)

    def mask_at(f):
        m = np.zeros((H, W), np.uint8)
        # a lizard-like body turning 30 degrees a frame: a long ellipse with a thick end (the head)
        th = np.radians(30.0 * f)
        c = np.array([160.0, 120.0])
        u = np.array([np.cos(th), np.sin(th)])
        cv2.ellipse(m, tuple(int(round(v)) for v in c), (80, 14), float(np.degrees(th)), 0, 360, 1, -1)
        cv2.circle(m, tuple(int(round(v)) for v in c + u * 80), 22, 1, -1)
        return m.astype(bool)

    for f in range(0, 40):
        w._mask_hist[f] = (mask_at(f), 1.0)
        ys, xs = np.nonzero(w._mask_hist[f][0])
        w._summ[f] = {"bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                      "area": int(len(xs)), "score": 8.0}
    # the head landmark sits MID-body: the anchor check fails and every row is oriented by continuity
    head = np.array([160.0, 120.0], np.float32)
    first = {}
    for f in range(0, 16):                                    # the first window, in order
        ml = w._midline_for(f, head + 0.01 * f)
        first[f] = None if ml is None else ml.head.copy()
    assert all(v is not None for v in first.values())
    # the second window re-emits frames 8..: the head moved a little, so every overlap row is recomputed
    again = {}
    for f in range(8, 24):
        ml = w._midline_for(f, head + 0.5 + 0.01 * f)
        again[f] = ml.head.copy()
    for f in range(8, 16):
        d = float(np.linalg.norm(again[f] - first[f]))
        assert d < 6.0, f"frame {f}: the recomputed overlap row's head moved {d:.0f} px against its first emission"


# ---------------------------------------------------------------- I240: the decoder's failures
class _Damaged:
    """VideoSource.read_next failing at one frame; `mode`: 'none' = returns None, 'raise' = raises."""

    def __init__(self, at, mode, once=False):
        self.at, self.mode, self.once, self.hit = at, mode, once, False
        self.orig = vs.VideoSource.read_next

    def __enter__(self):
        me = self

        def read_next(src):
            if src._pos == me.at and not (me.once and me.hit):
                me.hit = True
                if me.mode == "raise":
                    raise RuntimeError("simulated decoder failure")
                return None
            return me.orig(src)
        vs.VideoSource.read_next = read_next
        return self

    def __exit__(self, *a):
        vs.VideoSource.read_next = self.orig


def point_run(start, end, backend="cotracker3", **kw):
    w = trk.TrackingWorker(V600, start, None, None, FrameCache(1 << 28), end, refine=False, point_backend=backend,
                           specs=[trk.PointSpec(0, GT600[start, 0].astype(np.float32))], **kw)
    return w, run_worker(w)


@check("I240 a frame a fresh capture reads is a hiccup: 'press Track again', not 'damaged'")
def _():
    with _Damaged(300, "none", once=True):
        w, rec = point_run(250, 600)
    assert rec["fin"] is None and rec["err"] and "Frame 300" in rec["err"], rec["err"]
    assert "Press Track again" in rec["err"] and "damaged at that frame" not in rec["err"], rec["err"]
    assert w.decode_failed_at == 300 and w.decode_transient
    assert data_frames(rec, 0)[-1] == 299


@check("I240 a decode exception still emits the frames already read, then says the I40 sentence")
def _():
    real = trk._frame_decodes
    trk._frame_decodes = lambda path, idx: False if idx == 300 else real(path, idx)     # truly damaged there
    try:
        with _Damaged(300, "raise"):
            w, rec = point_run(250, 600)
    finally:
        trk._frame_decodes = real
    assert rec["err"] and "Traceback" not in rec["err"] and "Frame 300 of the video could not be decoded" in rec["err"], rec["err"]
    assert data_frames(rec, 0)[-1] == 299, data_frames(rec, 0)[-3:]          # the tail of the frames read (300 - 250 = 50 = 6 x 8 + 2)
    # the same for the AllTracker flush
    trk._frame_decodes = lambda path, idx: False if idx == 300 else real(path, idx)
    try:
        with _Damaged(300, "raise"):
            w, rec = point_run(250, 600, backend="alltracker")
    finally:
        trk._frame_decodes = real
    assert rec["err"] and "Frame 300 of the video could not be decoded" in rec["err"], rec["err"]
    assert data_frames(rec, 0)[-1] == 299, data_frames(rec, 0)[-3:]


# ---------------------------------------------------------------- I256: KINETRACE_ALLTRACKER_MAX_DIM
@check("I256 KINETRACE_ALLTRACKER_MAX_DIM above 1024 sets AllTracker's working size")
def _():
    from kinetrace import alltracker_backend as at
    sizes = []

    class StubStream:
        def start(self, frame, q):
            sizes.append(frame.shape[:2])

        def push(self, frame):
            return None

        def flush(self):
            return None

    saved = (at.get_alltracker, at.AllTrackerStream, os.environ.get("KINETRACE_ALLTRACKER_MAX_DIM"))
    at.get_alltracker = lambda **kw: (object(), "cuda")
    at.AllTrackerStream = StubStream
    try:
        for env, want in ((None, 1024), ("1536", 1536)):
            if env is None:
                os.environ.pop("KINETRACE_ALLTRACKER_MAX_DIM", None)
            else:
                os.environ["KINETRACE_ALLTRACKER_MAX_DIM"] = env
            sizes.clear()
            w = trk.TrackingWorker(V4K, 0, None, None, FrameCache(1 << 28), 10, refine=False, roi=False,
                                   point_backend="alltracker", specs=[trk.PointSpec(0, GT4K[0, 0].astype(np.float32))])
            run_worker(w)
            assert sizes and max(sizes[0]) == want, (env, sizes)
    finally:
        at.get_alltracker, at.AllTrackerStream = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("KINETRACE_ALLTRACKER_MAX_DIM", None)
        else:
            os.environ["KINETRACE_ALLTRACKER_MAX_DIM"] = saved[2]


# ---------------------------------------------------------------- I195: models load from models/hf whatever HF_HOME is
@check("I195 segmenter loads read this folder's models/hf even when the user has HF_HOME set")
def _():
    import subprocess
    tmp = tempfile.mkdtemp(prefix="hfcache_")
    other = tempfile.mkdtemp(prefix="hfhome_")
    code = r'''
import os, sys
sys.path.insert(0, r"%s")
from kinetrace import segmenter
from pathlib import Path
import huggingface_hub
rev = "0123456789abcdef0123456789abcdef01234567"
mine = Path(r"%s")
snap = mine / "hub" / "models--facebook--sam2.1-hiera-base-plus" / "snapshots" / rev
snap.mkdir(parents=True)
(snap / "config.json").write_text("{}")
segmenter.HF_DIR = mine
(mine / "token").write_text("hf_test_token")
assert os.environ["HF_HOME"] == r"%s", "the user's HF_HOME is kept"
try:                      # what a load did before: it looked in the USER's cache and found nothing
    huggingface_hub.hf_hub_download("facebook/sam2.1-hiera-base-plus", "config.json", revision=rev,
                                    local_files_only=True)
    raise SystemExit("a load without cache_dir must not find it")
except huggingface_hub.errors.LocalEntryNotFoundError:
    pass
kw = segmenter.hf_cache_args()
assert kw["cache_dir"] == str(mine / "hub") and kw.get("token") == "hf_test_token", kw
p = huggingface_hub.hf_hub_download("facebook/sam2.1-hiera-base-plus", "config.json", revision=rev,
                                    local_files_only=True, **kw)
assert os.path.samefile(p, snap / "config.json"), p
print("OK")
''' % (ROOT, tmp, other)
    env = dict(os.environ, HF_HOME=other)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout + r.stderr)[-400:]


@check("I195 a real SAM load works with HF_HOME pointing elsewhere (when SAM 2.1 is cached here)")
def _():
    import subprocess
    if not segmenter.model_is_cached("sam2.1-base-plus") or segmenter.local_dir("sam2.1-base-plus") is not None:
        print("     (SAM 2.1 base+ is not in models/hf here: skipped)")
        return
    other = tempfile.mkdtemp(prefix="hfhome_")
    code = ("import sys; sys.path.insert(0, r'%s')\nfrom kinetrace import segmenter\n"
            "segmenter.Segmenter('sam2.1-base-plus'); print('LOADED')\n" % ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, HF_HOME=other), timeout=600)
    assert "LOADED" in r.stdout, (r.stdout + r.stderr)[-300:]


# ---------------------------------------------------------------- R3: the snap / dilation caches are pruned
@check("R3 label images behind the window are dropped (the caches keep one window, not MASK_HIST frames)")
def _():
    w = trk.TrackingWorker(V600, 0, None, None, FrameCache(1 << 20), 100,
                           specs=[trk.PointSpec(0, np.array([100.0, 75.0], np.float32))],
                           animal=trk.AnimalSpec({0: []}, {}, None), constrain_pids=[0])
    w._refined = {}
    for f in range(40):
        w._snap_cache[f] = (None, None)
        w._dil_cache[f] = None
    w._constrain_to_mask(30, np.zeros((4, 1, 2), np.float32), w.specs, [("point", 0)], first_seg=False)
    assert min(w._snap_cache) == 30 and min(w._dil_cache) == 30, (min(w._snap_cache), min(w._dil_cache))


if FAILS:
    print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("VERIFY_REVIEW_TRACKING PASSED")
