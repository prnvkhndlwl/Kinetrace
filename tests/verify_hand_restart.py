"""AllTracker / CoTracker3 points start again at the user's clicks (I316, PR B of dev/work/2026-10-20-silhouette-hand-mix,
plan steps 16-18). GPU; the models in models/.

The owner: "each point run its individual tracker set up by the user and for each point the corresponding tracker
reseeds if it finds a handclicked point on that frame ... If another point in the same frame does not have a
handclicked point then it should just continue like normal". A clicked point gets its OWN tracker session from its
click on (`tracker.RunDriver`), the run's other points stay in the run's own tracker, bit-identical.

Bare driver (tracker level):
  [1] CoTracker3 and AllTracker, a click at h = s+1, s+8, s+16, s+17 (every phase of the 8-frame windows): the
      session's row h is the click exactly, and the run's own worker emits exactly what it emits with no click
  [2] a dot that vanishes and comes back, clicked where it returns: the run goes on (no stop), the dot is cut from
      where it was lost to the click, and tracked again from the click (today: the run stops at the loss)
  [3] X with 3 live sessions: everything stops in under a second
  [4] a synthetic animal: after a click on its head (the run's own head point was placed mid-body) the silhouette
      landmarks are measured from the session's head (D2); a held point's session is kept on the run's live
      silhouettes
App (real keys):
  [5] three points, clicks on two of them ahead, T: the clicks stay exactly, the clicked points follow their
      clicks, the unclicked point is the same as without the clicks, the playhead never goes back
  [6] X while three sessions run: the run stops in under a second, every point tracked to within 16 frames
  [7] two cameras, a click in camera B only, a different tracker per camera: camera A as without the click,
      camera B from its click
  [8] the vanishing dot through the app: no stop, nothing in the hidden stretch, the dot from its click on
  [12] a point with no position on the start frame joins at its click: alone (the run starts as its session), and
       beside a Moving spot point with the project default AllTracker (its session uses ITS tracker, CoTracker3)
  [13] a synthetic animal pair (SAM replaced by their true silhouettes): both heads clicked on one frame - each
       animal's landmarks anchored on its own head's session; a Hand-only head anchors them on its clicks, and
       between them they are measured without an anchor, never capped
  [14] X after a point was cut, before its click: the frames after the stop keep their old data (D9 choice 2)
  ([12]-[14] and [11]'s Track anyway / refusal: review findings #1, #3, #4 and nits, each failing on e5a5022)
Every check FAILS on main (there is no RunDriver; a run writes round the clicks and a lost point stops it).
Run: .venv\\Scripts\\python.exe tests\\verify_hand_restart.py
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
OUT = os.path.join(HERE, "out", "hand_restart")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402

import kinetrace.tracker as trk  # noqa: E402
from kinetrace import alltracker_backend, cameraorder  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

TRACK_CHOICE = stub_track_choice()     # (G217) the Track pop-up answers "keep my clicks"
cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(), self.result())[1]

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


V600 = os.path.join(ROOT, "test600.mp4")
GT = np.load(V600 + ".gt.npz")["gt"]
HAVE_AT = alltracker_backend.available()
CACHE = 1 << 29


# ===================================================================== bare driver helpers
class Rec:
    """What a worker (or the driver) emits, copied as it arrives."""

    def __init__(self, w, chunks=True):
        self.chunks, self.fin, self.auto, self.err = [], [], [], []
        if chunks:
            w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: self.chunks.append(
                (int(w0), np.array(tr, copy=True), np.array(vi, copy=True), np.array(cf, copy=True))))
        w.finished_ok.connect(lambda last, p: self.fin.append((int(last), bool(p))))
        w.autopaused.connect(lambda f, p: self.auto.append((int(f), int(p))))
        w.error.connect(lambda m: self.err.append(str(m)))

    def track(self, col: int, n: int) -> np.ndarray:
        """Column `col`'s rows, the last write of each frame winning (what a session keeps)."""
        out = np.full((n, 2), np.nan, np.float32)
        for w0, tr, _vi, _cf in self.chunks:
            for i in range(tr.shape[0]):
                if 0 <= w0 + i < n:
                    out[w0 + i] = tr[i, col]
        return out

    def first_row(self, col: int, frame: int):
        """The FIRST value the worker emitted for `frame` (a session's seed row)."""
        for w0, tr, _vi, _cf in self.chunks:
            if w0 <= frame < w0 + tr.shape[0]:
                return tr[frame - w0, col]
        return None


def same_chunks(a: Rec, b: Rec) -> bool:
    if len(a.chunks) != len(b.chunks):
        return False
    for x, y in zip(a.chunks, b.chunks):
        if x[0] != y[0] or not all(np.array_equal(p, q, equal_nan=True) for p, q in zip(x[1:], y[1:])):
            return False
    return True


def worker(video, start, n, pts: dict, backend, cache, **kw):
    specs = [trk.PointSpec(int(p), np.asarray(xy, np.float32).copy()) for p, xy in pts.items()]
    return trk.TrackingWorker(video, start, None, None, cache, n, refine=True, specs=specs, roi=True,
                              autopause=True, point_backend=backend, **kw)


# ===================================================================== [1] a session at every window phase
def section1():
    print("[1] bare driver: a session from a click, CoTracker3 and AllTracker, every window phase")
    n = 60
    four = {k: GT[0, k] for k in range(4)}
    for backend in ("cotracker3", "alltracker"):
        if backend == "alltracker" and not HAVE_AT:
            check("(setup) AllTracker is installed", False)
            continue
        base = worker(V600, 0, n, four, backend, FrameCache(CACHE))
        rb = Rec(base)
        base.run()
        check(f"(setup) {backend}: the run without clicks ends at {n - 1}", rb.fin and rb.fin[-1][0] == n - 1, rb.fin)
        for h in (1, 8, 16, 17):
            click = (float(GT[h, 1, 0]) + 1.0, float(GT[h, 1, 1]) - 0.6)        # on the dot, off its centre
            cache = FrameCache(CACHE)
            prim = worker(V600, 0, n, four, backend, cache, resume_pids={1})
            sess = worker(V600, h, n, {1: click}, backend, cache, cache_first=True, emit_frames=False)
            rp, rs = Rec(prim), Rec(sess)
            drv = trk.RunDriver(prim, [trk.SessionJob(h, sess, {1: (h, n - 1)})])
            rd = Rec(drv, chunks=False)
            drv.run()
            row = rs.first_row(0, h)
            c32 = np.asarray(click, np.float32)                 # tracks are stored as float32
            check(f"[1] {backend}, click at {h}: the session's row {h} is the click exactly",
                  row is not None and np.array_equal(np.asarray(row, np.float32), c32),
                  None if row is None else tuple(float(v) for v in row))
            tr = rs.track(0, n)
            off = np.array([1.0, -0.6])
            err = np.linalg.norm(tr[h + 1:] - (GT[h + 1:n, 1] + off), axis=1)
            own = np.linalg.norm(rb.track(1, n)[h + 1:] - GT[h + 1:n, 1], axis=1)     # the tracker's own error there
            check(f"[1] {backend}, click at {h}: ... and it follows the clicked spot from there (median "
                  f"{np.nanmedian(err):.2f} px; the same tracker from frame 0 is {np.nanmedian(own):.2f} px off the dot)",
                  np.isfinite(err).all() and float(np.median(err)) < float(np.median(own)) + 0.75)
            check(f"[1] {backend}, click at {h}: the run's own worker emits exactly what it emits with no click",
                  same_chunks(rp, rb), (len(rp.chunks), len(rb.chunks)))
            check(f"[1] {backend}, click at {h}: the run ends once, at {n - 1}, with no error",
                  rd.fin == [(n - 1, False)] and not rd.err and not rd.auto, (rd.fin, rd.err[-1:], rd.auto))
            check(f"[1] {backend}, click at {h}: the session decoded no frame itself (it read the run's)",
                  getattr(getattr(sess, "source_obj", None), "decoded", -1) == 0,
                  getattr(getattr(sess, "source_obj", None), "decoded", None))


# ===================================================================== the vanishing dot
W, H, T1, GONE, BACK = 640, 480, 260, 60, 170


def make_bg(seed):
    r = np.random.default_rng(seed)
    bg = r.integers(30, 90, size=(H, W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(bg, (0, 0), 1.5)


def dot(frame, x, y, color):
    cv2.circle(frame, (round(x), round(y)), 9, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (round(x), round(y)), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)


VAN = os.path.join(OUT, "vanish.mp4")
AX = 200 + 60 * np.sin(np.arange(T1) / 40)
AY = 200 + 40 * np.cos(np.arange(T1) / 55)
BX = 460 + 50 * np.sin(np.arange(T1) / 47)
BY = 320 + 45 * np.cos(np.arange(T1) / 38)
_bg_a, _bg_b = make_bg(3), make_bg(77)
_vw = cv2.VideoWriter(VAN, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for _f in range(T1):
    _fr = (_bg_a if _f < GONE else _bg_b).copy()           # the background changes when dot A goes (a real loss)
    if not GONE <= _f < BACK:
        dot(_fr, AX[_f], AY[_f], (60, 60, 230))
    dot(_fr, BX[_f], BY[_f], (60, 230, 60))
    _vw.write(_fr)
_vw.release()


def section2():
    print("[2] bare driver: a dot lost, clicked where it comes back")
    n = T1
    pts = {0: (AX[0], AY[0]), 1: (BX[0], BY[0])}
    base = worker(VAN, 0, n, pts, "cotracker3", FrameCache(CACHE))
    rb = Rec(base)
    base.run()
    check("(setup) without a click the run stops where dot A is lost (today)", rb.auto and GONE <= rb.auto[0][0] < BACK
          and rb.auto[0][1] == 0, rb.auto)
    cache = FrameCache(CACHE)
    click = (float(AX[BACK]), float(AY[BACK]))
    prim = worker(VAN, 0, n, pts, "cotracker3", cache, resume_pids={0})
    sess = worker(VAN, BACK, n, {0: click}, "cotracker3", cache, cache_first=True, emit_frames=False)
    rp, rs = Rec(prim), Rec(sess)
    drv = trk.RunDriver(prim, [trk.SessionJob(BACK, sess, {0: (BACK, n - 1)})])
    rd = Rec(drv, chunks=False)
    drv.run()
    check("[2] with the click the run does not stop: it ends at the clip's end", rd.fin == [(n - 1, False)] and not rd.auto,
          (rd.fin, rd.auto))
    cut = getattr(prim, "resume_cuts", {}).get(0)
    check("[2] the run's worker cut dot A where it was lost (before the click)", cut is not None and GONE <= cut < BACK,
          cut)
    tr0 = rp.track(0, n)
    check("[2] ... and emits nothing for it once the loss is confirmed (16 frames after the cut; the app clears the "
          "rows before)", cut is not None and np.isnan(tr0[cut + 16:]).all())
    trs = rs.track(0, n)
    err = np.linalg.norm(trs[BACK:] - np.stack([AX[BACK:n], AY[BACK:n]], axis=1), axis=1)
    check(f"[2] dot A is tracked again from the click to the end (median {np.nanmedian(err):.2f} px)",
          np.isfinite(err).all() and float(np.median(err)) < 2.0)
    trb = rp.track(1, n)
    check("[2] dot B is tracked to the end", np.isfinite(trb[n - 10:]).all())


# ===================================================================== [3] X with three live sessions
def section3():
    print("[3] bare driver: X with three live sessions")
    n = 600
    cache = FrameCache(CACHE)
    four = {k: GT[0, k] for k in range(4)}
    prim = worker(V600, 0, n, four, "cotracker3", cache, resume_pids={1, 2, 3})
    jobs = []
    for k in (1, 2, 3):
        sw = worker(V600, 4 + k, n, {k: GT[4 + k, k]}, "cotracker3", cache, cache_first=True, emit_frames=False)
        jobs.append(trk.SessionJob(4 + k, sw, {k: (4 + k, n - 1)}))
    drv = trk.RunDriver(prim, jobs)
    rd = Rec(drv, chunks=False)
    drv.start()
    live = wait(lambda: drv.n_live >= 3, 60)
    pump(0.5)
    t0 = time.time()
    drv.request_pause()
    done = wait(lambda: drv.isFinished(), 10)
    dt = time.time() - t0
    pump(0.2)
    check("(setup) three sessions were live", live)
    check(f"[3] X stopped the run and its three sessions in {dt:.2f} s (< 1 s)", done and dt < 1.0)
    check("[3] ... and the run says it was paused, once", len(rd.fin) == 1 and rd.fin[0][1], rd.fin)


# ===================================================================== [4] the synthetic animal: head (D2) and a held point
def section4():
    print("[4] bare driver: a head clicked (D2) and a held point's session on the live silhouettes")
    import _synth
    vid = _synth.build_video(os.path.join(OUT, "synth_animals.mp4"))
    n, h = 60, 10
    g0 = _synth.gt_frame(0)[1][0]
    s = TrackingSession(vid, _synth.T, 25.0, _synth.W, _synth.H)
    head = s.add_point(0, float(g0["centre"][0]), float(g0["centre"][1]), name="snout")   # mid-body: wrong
    s.ensure_animal()
    s.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
    tip = s.add_landmark("tail_tip", "silhouette", "tip")
    animal = trk.AnimalSpec({0: list(s.animal.prompts[0])}, {}, None)

    def prim_worker(cache, **kw):
        return trk.TrackingWorker(vid, 0, None, None, cache, n, refine=True,
                                  specs=[trk.PointSpec(head, s.tracks[0, head].astype(np.float32).copy())],
                                  roi=True, autopause=False, animal=animal, derived=[trk.DerivedSpec(tip, "tip")],
                                  head_pid=head, **kw)

    base = prim_worker(FrameCache(CACHE))
    rb = Rec(base)
    base.run()
    lost_base = sorted(f for f in base._anchor_lost if h + 16 <= f < n)
    check("(setup) with the head placed mid-body the anchor is lost on the frames after the click too",
          len(lost_base) >= 0.8 * (n - h - 16), len(lost_base))
    gh = _synth.gt_frame(h)[1][0]["head"]
    click = (float(gh[0]), float(gh[1]))
    cache = FrameCache(CACHE)
    rows: dict = {}
    sess = trk.TrackingWorker(vid, h, None, None, cache, n, refine=True,
                              specs=[trk.PointSpec(head, np.asarray(click, np.float32))], roi=True, autopause=False,
                              cache_first=True, emit_frames=False, keep_rows={head: rows})
    hook = trk.head_hook(h, {h: click}, [(h, n - 1, rows)])
    prim = prim_worker(cache, resume_pids=set(), head_xy={head: hook})
    rp = Rec(prim)
    drv = trk.RunDriver(prim, [trk.SessionJob(h, sess, {head: (h, n - 1)})])
    drv.run()
    lost = sorted(f for f in prim._anchor_lost if h + 16 <= f < n)
    check(f"[4] after the head click the silhouette landmarks are anchored on the session's head (anchor lost on "
          f"{len(lost)} of {n - h - 16} frames, was {len(lost_base)})", len(lost) <= 0.1 * (n - h - 16))
    tip_col = prim.point_ids.index(tip)
    cf = np.full(n, np.nan)
    for w0, _tr, _vi, c in rp.chunks:
        for i in range(c.shape[0]):
            cf[w0 + i] = c[i, tip_col]
    later = cf[h + 16:n]
    check("[4] ... so the tail tip's confidence is no longer capped there",
          np.nanmedian(later) > trk.ANCHOR_LOST_CONF + 0.05, float(np.nanmedian(later)))
    # a held point (kept on the silhouette) clicked at h: its session is kept on the run's LIVE silhouettes
    s2 = TrackingSession(vid, _synth.T, 25.0, _synth.W, _synth.H)
    eye0 = _synth.gt_frame(0)[1][0]["eye"]
    held = s2.add_point(0, float(eye0[0]), float(eye0[1]), name="eye")
    s2.ensure_animal()
    s2.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
    animal2 = trk.AnimalSpec({0: list(s2.animal.prompts[0])}, {}, None)
    cache2 = FrameCache(CACHE)
    prim2 = trk.TrackingWorker(vid, 0, None, None, cache2, n, refine=True,
                               specs=[trk.PointSpec(held, np.asarray(eye0, np.float32))], roi=True, autopause=False,
                               animal=animal2, constrain_pids=[held], on_body_pids=[held], resume_pids=set())
    eye_h = _synth.gt_frame(h)[1][0]["eye"]
    live = trk.LiveMasks(prim2, 0)
    sess2 = trk.TrackingWorker(vid, h, None, None, cache2, n, refine=True,
                               specs=[trk.PointSpec(held, np.asarray(eye_h, np.float32))], roi=True, autopause=False,
                               stored_masks=live, constrain_pids=[held], on_body_pids=[held],
                               cache_first=True, emit_frames=False)
    rs2 = Rec(sess2)
    drv2 = trk.RunDriver(prim2, [trk.SessionJob(h, sess2, {held: (h, n - 1)})])
    drv2.run()
    check("[4] the held point's session read the run's live silhouette on every frame it tracked",
          live.served >= n - h - 1, getattr(live, "served", None))
    tr = rs2.track(0, n)
    inside = []
    for f in range(h, n):
        m = _synth.gt_frame(f)[1][0]["mask"]
        x, y = int(round(tr[f, 0])), int(round(tr[f, 1]))
        inside.append(np.isfinite(tr[f]).all() and 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and bool(m[y, x]))
    check(f"[4] ... and stays on the body ({sum(inside)} of {len(inside)} frames)", all(inside))


# ===================================================================== [9] the exit, a session's error, X at the end
def section9():
    print("[9] bare driver: a held point's exit with a click ahead; a session's error; X after the run's own end")
    import _synth
    vid = os.path.join(OUT, "synth_animals.mp4")
    if not os.path.isfile(vid):
        vid = _synth.build_video(vid)
    n, h = 50, 12
    g0 = _synth.gt_frame(0)[1][0]
    s = TrackingSession(vid, _synth.T, 25.0, _synth.W, _synth.H)
    held = s.add_point(0, 40.0, 40.0, name="eye")             # placed on the background: it leaves the body at once
    s.ensure_animal()
    s.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
    animal = trk.AnimalSpec({0: list(s.animal.prompts[0])}, {}, None)

    def prim(cache, **kw):
        return trk.TrackingWorker(vid, 0, None, None, cache, n, refine=True,
                                  specs=[trk.PointSpec(held, np.array([40.0, 40.0], np.float32))], roi=True,
                                  autopause=False, animal=animal, constrain_pids=[held], on_body_pids=[held], **kw)
    base = prim(FrameCache(CACHE))
    rb = Rec(base)
    base.run()
    check("(setup) without a click a held point that leaves its silhouette stops the run (exit), Auto-pause off",
          rb.auto and getattr(base, "_autopause_reason", "") == "exit", (rb.auto, getattr(base, "_autopause_reason", "")))
    cache = FrameCache(CACHE)
    p1 = prim(cache, resume_pids={held})
    eye_h = _synth.gt_frame(h)[1][0]["eye"]
    sess = trk.TrackingWorker(vid, h, None, None, cache, n, refine=True,
                              specs=[trk.PointSpec(held, np.asarray(eye_h, np.float32))], roi=True, autopause=False,
                              stored_masks=trk.LiveMasks(p1, 0), constrain_pids=[held], on_body_pids=[held],
                              cache_first=True, emit_frames=False)
    rs = Rec(sess)
    drv = trk.RunDriver(p1, [trk.SessionJob(h, sess, {held: (h, n - 1)})])
    rd = Rec(drv, chunks=False)
    drv.run()
    cut = p1.resume_cuts.get(held)
    check("[9] with a click ahead the exit is a cut, not a stop: the run ends at the clip's end (review R2-I1)",
          rd.fin == [(n - 1, False)] and not rd.auto and cut is not None and cut <= h, (rd.fin, rd.auto, cut))
    check("[9] ... and the held point is tracked from its click on", np.isfinite(rs.track(0, n)[h:]).all())
    # a session's error is the run's ONE error, and the run stops
    cache = FrameCache(CACHE)
    four = {k: GT[0, k] for k in range(4)}
    p2 = worker(V600, 0, 300, four, "cotracker3", cache, resume_pids={1})
    s2 = worker(V600, 20, 300, {1: GT[20, 1]}, "cotracker3", cache, cache_first=True, emit_frames=False)

    def boom(*a, **k):
        raise RuntimeError("a fault inside a session")
    s2._segment_steps = boom
    rp2 = Rec(p2)
    drv2 = trk.RunDriver(p2, [trk.SessionJob(20, s2, {1: (20, 299)})])
    rd2 = Rec(drv2, chunks=False)
    drv2.run()
    last2 = max((w0 + tr.shape[0] - 1 for w0, tr, _v, _c in rp2.chunks), default=-1)
    check("[9] a session's error is the run's one error (no end, no auto-pause), and the run's own worker stops",
          len(rd2.err) == 1 and "a fault inside a session" in rd2.err[0] and not rd2.fin and not rd2.auto
          and last2 < 299, (len(rd2.err), rd2.fin, last2))
    # a fault of the DRIVER's own (not a worker's): the run's one error too, every worker stopped (R2-M3, N2)
    cache = FrameCache(CACHE)
    p4 = worker(V600, 0, 300, four, "cotracker3", cache, resume_pids={1})
    s4 = worker(V600, 20, 300, {1: GT[20, 1]}, "cotracker3", cache, cache_first=True, emit_frames=False)
    rp4 = Rec(p4)
    drv4 = trk.RunDriver(p4, [trk.SessionJob(20, s4, {1: (20, 299)})])
    rd4 = Rec(drv4, chunks=False)
    real_peak = trk.peak_sessions

    def broken(*a, **k):
        raise RuntimeError("a fault in the driver")
    trk.peak_sessions = broken                  # the driver's own bookkeeping at the first session's start
    try:
        drv4.run()
    finally:
        trk.peak_sessions = real_peak
    last4 = max((w0 + tr.shape[0] - 1 for w0, tr, _v, _c in rp4.chunks), default=-1)
    check("[9] a fault in the driver itself is the run's one error, and the run's own worker stops",
          len(rd4.err) == 1 and "a fault in the driver" in rd4.err[0] and not rd4.fin and last4 < 299,
          (len(rd4.err), rd4.fin, last4))
    # X right after the run's own worker ended: the sessions still running bound the run's last frame (R2-M1)
    cache = FrameCache(CACHE)
    p3 = worker(V600, 0, 120, four, "cotracker3", cache, resume_pids={1})
    s3 = worker(V600, 60, 120, {1: GT[60, 1]}, "cotracker3", cache, cache_first=True, emit_frames=False)
    drv3 = trk.RunDriver(p3, [trk.SessionJob(60, s3, {1: (60, 119)})])
    rd3 = Rec(drv3, chunks=False)
    rs3 = Rec(s3)
    p3.finished_ok.connect(lambda *_a: drv3.request_pause(), Qt.DirectConnection)
    drv3.run()
    s3_last = max((w0 + tr.shape[0] - 1 for w0, tr, _v, _c in rs3.chunks), default=-1)
    check("[9] X right after the run's own worker ended: the run says paused, at the session's last frame",
          rd3.fin and rd3.fin[0][1] and rd3.fin[0][0] == s3_last < 119, (rd3.fin, s3_last))


# ===================================================================== app helpers
def new_window(video):
    forget_recovery(video)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(video)
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    pump(0.3)
    return win


def close(win):
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


def press_t(win, start=0, key=Qt.Key_T, mods=Qt.NoModifier, timeout=600):
    win.act_select_all.trigger()
    win._goto(start, force=True)
    pump(0.05)
    win.activateWindow()
    QTest.keyClick(win, key, mods)
    wait(lambda: win.state != READY or win._passes is not None, 15)
    return wait(lambda: win.state == READY and win.worker is None and win._passes is None and win._multi is None,
                timeout)


def add_points(win, xys, tracker):
    s = win.session
    for x, y in xys:
        win._on_add(float(x), float(y))
    for k in range(s.n_points):
        s.points[k].tracker = tracker
    win._refresh_point_list()


def click_at(win, pid, f, xy):
    """A hand click of `pid` on frame `f` (the data state a click leaves; the gesture itself is PR A's)."""
    win.session.set_position(f, pid, float(xy[0]), float(xy[1]))


# ===================================================================== [5] the app: clicks ahead on two of three points
def section5():
    print("[5] app: three points, clicks ahead on two, T")
    win = new_window(V600)
    add_points(win, [GT[0, k] for k in range(3)], "cotracker3")
    s = win.session
    ok = press_t(win)
    check("(setup) the run without clicks ended", ok)
    base = s.tracks[:, 0].copy(), s.tracked[:, 0].copy(), s.confidence[:, 0].copy()
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)
    s = win.session
    off = np.array([2.0, -1.0])
    clicks = {1: [30, 90], 2: [45]}
    for pid, fs in clicks.items():
        for f in fs:
            click_at(win, pid, f, GT[f, pid] + off)
    shown = []
    orig_tick = win._render_tick

    def tick():
        orig_tick()
        if win.state != READY:
            shown.append(win.current)
    win._render_tick = tick
    win._render_timer.timeout.disconnect()
    win._render_timer.timeout.connect(win._render_tick)
    ok = press_t(win)
    s = win.session
    check("(setup) the run with the clicks ended", ok)
    check("[5] every click is kept exactly and is still hand-placed",
          lambda: all(s.manual[f, pid] and np.allclose(s.tracks[f, pid], GT[f, pid] + off, atol=1e-4)
                      for pid, fs in clicks.items() for f in fs))
    for pid, fs in clicks.items():
        f = fs[0]
        err = np.linalg.norm(s.tracks[f + 1:f + 25, pid] - (GT[f + 1:f + 25, pid] + off), axis=1)
        old = np.linalg.norm(s.tracks[f + 1:f + 25, pid] - GT[f + 1:f + 25, pid], axis=1)
        check(f"[5] P{pid + 1} follows its click at {f}: it restarted there (median {np.nanmedian(err):.2f} px from the "
              f"clicked spot, {np.nanmedian(old):.2f} px from the dot it tracked before)",
              np.isfinite(err).all() and float(np.median(err)) < 1.5 and float(np.median(err)) < float(np.median(old)))
    check("[5] the unclicked point is exactly what the run without clicks gave it",
          lambda: np.array_equal(s.tracks[:, 0], base[0], equal_nan=True) and np.array_equal(s.tracked[:, 0], base[1])
          and np.array_equal(s.confidence[:, 0], base[2]))
    back = [i for i in range(1, len(shown)) if shown[i] < shown[i - 1]]
    check(f"[5] the playhead never went back during the run ({len(shown)} frames shown)", shown and not back,
          back[:3])
    close(win)


# ===================================================================== [6] the app: X while three sessions run
def section6():
    print("[6] app: X while three sessions run")
    win = new_window(V600)
    add_points(win, [GT[0, k] for k in range(4)], "cotracker3")
    for k in (1, 2, 3):
        click_at(win, k, 4 + k, GT[4 + k, k])
    win.act_select_all.trigger()
    win._goto(0, force=True)
    pump(0.05)
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_T)
    started = wait(lambda: win.state != READY and getattr(win.worker, "n_live", 0) >= 3, 60)
    pump(1.0)
    t0 = time.time()
    QTest.keyClick(win, Qt.Key_X)
    done = wait(lambda: win.state == READY and win.worker is None, 10)
    dt = time.time() - t0
    s = win.session
    check("(setup) three sessions were live", started)
    check(f"[6] X stopped the run in {dt:.2f} s (< 1 s)", started and done and dt < 1.0)
    stop = win.current
    lasts = [int(np.flatnonzero(s.tracked[:, k])[-1]) if s.tracked[:, k].any() else -1 for k in range(4)]
    check(f"[6] every point is tracked to within 16 frames of the stop (stop {stop}, last frames {lasts})",
          started and all(stop - 16 <= f for f in lasts) and 10 < stop < 590)
    close(win)


# ===================================================================== [7] two cameras, a click in camera B only
def section7():
    print("[7] app: two cameras, a click in camera B only, a tracker per camera")
    if not HAVE_AT:
        check("(setup) AllTracker is installed", False)
        return
    mir = os.path.join(OUT, "mirror.mp4")
    cap = cv2.VideoCapture(V600)
    vw = cv2.VideoWriter(mir, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for _ in range(200):
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(cv2.flip(fr, 1))
    cap.release()
    vw.release()
    forget_recovery(mir)
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([mir], ""))

    def two_cams():
        win = new_window(V600)
        win._add_video_dialog()
        wait(lambda: win.project.n_views == 2 and win.state == READY and not win._loading, 60)
        pump(0.5)
        add_points(win, [GT[0, 0]], "cotracker3")
        sb = win.project.sessions[1]
        j = sb.pid_by_name(win.session.points[0].name)
        sb.set_position(0, j, 639.0 - float(GT[0, 0, 0]), float(GT[0, 0, 1]))
        sb.points[j].tracker = "alltracker"            # camera B tracks it with AllTracker (D8)
        win.act_track_all.setChecked(True)
        return win, j

    win, j = two_cams()
    ok = press_t(win, timeout=900)
    a0 = win.project.sessions[0].tracks[:200, 0].copy()
    b0 = win.project.sessions[1].tracks[:200, j].copy()
    check("(setup) the every-camera run without clicks ended", ok)
    close(win)
    win, j = two_cams()
    sb = win.project.sessions[1]
    f = 40
    off = np.array([2.0, -1.0])
    click = (float(b0[f, 0]) + 2.0, float(b0[f, 1]) - 1.0)        # 2 px right, 1 up of camera B's own track
    sb.set_position(f, j, *click)
    ok = press_t(win, timeout=900)
    sa, sb = win.project.sessions[0], win.project.sessions[1]
    check("(setup) the every-camera run with the click ended", ok)
    check("[7] camera A (no click) is exactly what it was without the click",
          lambda: np.array_equal(sa.tracks[:200, 0], a0, equal_nan=True))
    err = np.linalg.norm(sb.tracks[f + 1:f + 25, j] - (b0[f + 1:f + 25] + off), axis=1)
    same = np.linalg.norm(sb.tracks[f + 1:f + 25, j] - b0[f + 1:f + 25], axis=1)
    check(f"[7] camera B follows its click from frame {f}: its track without the click, moved by the click "
          f"(median {np.nanmedian(err):.2f} px from that; {np.nanmedian(same):.2f} px from the old track)",
          np.isfinite(err).all() and float(np.median(err)) < float(np.median(same))
          and float(np.median(same)) > 0.5 and sb.manual[f, j])
    close(win)


# ===================================================================== [8] the vanishing dot through the app
def section8():
    print("[8] app: the vanishing dot, clicked where it comes back")
    win = new_window(VAN)
    add_points(win, [(AX[0], AY[0]), (BX[0], BY[0])], "cotracker3")
    click_at(win, 0, BACK, (AX[BACK], AY[BACK]))
    ok = press_t(win)
    s = win.session
    check("(setup) the run ended", ok)
    check("[8] the run did not stop at the loss: dot B is tracked to the end", lambda: s.tracked[T1 - 10:, 1].all(),
          np.flatnonzero(s.tracked[:, 1])[-1:])
    check("[8] dot A has no data from where it was lost to its click (the last 40 frames before it)",
          lambda: not s.tracked[BACK - 40:BACK, 0].any(), np.flatnonzero(s.tracked[GONE:BACK, 0])[-1:] + GONE)
    err = np.linalg.norm(s.tracks[BACK + 1:T1, 0] - np.stack([AX[BACK + 1:T1], AY[BACK + 1:T1]], axis=1), axis=1)
    check(f"[8] dot A is tracked again from its click (median {np.nanmedian(err):.2f} px)",
          np.isfinite(err).all() and float(np.median(err)) < 2.0)
    close(win)


# ===================================================================== [10] two passes, [11] the memory question
def section10():
    print("[10] app: a two-pass press (AllTracker, then CoTracker3), the second pass's point clicked")
    if not HAVE_AT:
        check("(setup) AllTracker is installed", False)
        return
    win = new_window(V600)
    add_points(win, [GT[0, 0], GT[0, 1]], "cotracker3")
    s = win.session
    s.points[0].tracker = "alltracker"
    win._refresh_point_list()
    ok = press_t(win, timeout=900)
    base = s.tracks[:, 0].copy()
    b1 = s.tracks[:, 1].copy()
    check("(setup) the two-pass run without clicks ended", ok)
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)
    s = win.session
    f, off = 50, np.array([2.0, -1.0])
    click_at(win, 1, f, b1[f] + off)
    ok = press_t(win, timeout=900)
    s = win.session
    check("(setup) the two-pass run with the click ended", ok)
    check("[10] the first pass's point (AllTracker, no click) is exactly what it was",
          lambda: np.array_equal(s.tracks[:, 0], base, equal_nan=True))
    err = np.linalg.norm(s.tracks[f + 1:f + 25, 1] - (b1[f + 1:f + 25] + off), axis=1)
    same = np.linalg.norm(s.tracks[f + 1:f + 25, 1] - b1[f + 1:f + 25], axis=1)
    check(f"[10] the second pass's point restarted at its click (median {np.nanmedian(err):.2f} px from its old track "
          f"moved by the click, {np.nanmedian(same):.2f} px from the old track)",
          np.isfinite(err).all() and float(np.median(err)) < float(np.median(same)) and float(np.median(same)) > 0.5)
    close(win)


def section11():
    print("[11] app: restarts that may not fit the graphics card: the press asks first")
    import kinetrace.device as dev
    win = new_window(V600)
    add_points(win, [GT[0, k] for k in range(3)], "cotracker3")
    for k in (1, 2):
        click_at(win, k, 20, GT[20, k])
    asked = []
    real_q, real_vram = QMessageBox.question, dev.cached_vram_gb
    QMessageBox.question = staticmethod(lambda *a, **k: (asked.append(a[2] if len(a) > 2 else ""), QMessageBox.No)[1])
    dev.cached_vram_gb = lambda: 0.5                      # a 0.5 GB card (the estimate here: ~0.9 GB)
    try:
        before = win.session.tracked.copy()
        press_t(win, timeout=60)
    finally:
        QMessageBox.question, dev.cached_vram_gb = real_q, real_vram
    s = win.session
    check("[11] on a small card the press asks before restarting the clicked points (and says what it needs)",
          any("graphics card" in m for m in asked), asked[-1:])
    check("[11] ... and answered No, nothing is tracked", lambda: np.array_equal(s.tracked, before) and win.worker is None)
    close(win)
    # (review nits) "Track anyway" is the user's answer: the run does not refuse again; and the run's own refusal (the
    # estimate fitted the card, its free memory at the start did not) is said in words, not as a crash
    import torch
    cu = torch.cuda
    real_mem = (cu.mem_get_info, cu.memory_reserved, cu.memory_allocated)
    real_w, real_c = QMessageBox.warning, QMessageBox.critical
    shown = []
    QMessageBox.warning = staticmethod(lambda *a, **k: (shown.append(("warning", a[1] if len(a) > 1 else "")),
                                                        QMessageBox.Ok)[1])
    QMessageBox.critical = staticmethod(lambda *a, **k: (shown.append(("critical", a[1] if len(a) > 1 else "")),
                                                         QMessageBox.Ok)[1])
    cu.mem_get_info = lambda *a: (int(1e7), int(8e10))       # 10 MB free when the run looks: short for any session
    cu.memory_reserved = cu.memory_allocated = lambda *a: 0
    try:
        for anyway in (True, False):
            win = new_window(V600)
            add_points(win, [GT[0, k] for k in range(3)], "cotracker3")
            for k in (1, 2):
                click_at(win, k, 20, GT[20, k])
            shown.clear()
            dev.cached_vram_gb = (lambda: 0.5) if anyway else (lambda: 80.0)     # asks / fits the estimate
            ok = press_t(win, timeout=300)
            s = win.session
            if anyway:
                check("[11] answered Track anyway: the run does not refuse again, the clicked points are tracked to the "
                      "end", ok and s.tracked[590:, 1].all() and s.tracked[590:, 2].all() and not shown, shown)
            else:
                check("[11] the run's own refusal (its free memory short) is said in words, not as a crash",
                      ok and any(kd == "warning" and "memory" in t for kd, t in shown)
                      and not any(kd == "critical" for kd, _t in shown), shown)
            close(win)
    finally:
        QMessageBox.warning, QMessageBox.critical = real_w, real_c
        dev.cached_vram_gb = real_vram
        cu.mem_get_info, cu.memory_reserved, cu.memory_allocated = real_mem


# ===================================================================== [12] joiners
def section12():
    print("[12] app: a point with no position on the start frame joins at its click")
    win = new_window(V600)
    s = win.session
    win._goto(30, force=True)
    win._on_add(float(GT[30, 0, 0]), float(GT[30, 0, 1]))     # placed on frame 30 only
    s.points[0].tracker = "cotracker3"
    win._refresh_point_list()
    click = s.tracks[30, 0].copy()
    toasts = []
    real_toast = win.toast.show_message
    win.toast.show_message = lambda m, *a, **k: (toasts.append(str(m)), real_toast(m, *a, **k))
    ok = press_t(win, start=0)
    check("[12] one CoTracker3 point clicked on frame 30, Track from frame 0: the run starts (review #1: 'Nothing to "
          "start from')", ok and not any("Nothing to start" in m for m in toasts), toasts[-2:])
    err = np.linalg.norm(s.tracks[31:, 0] - GT[31:600, 0], axis=1)
    check(f"[12] ... nothing before its click, the click exact, tracked from it to the end (median {np.nanmedian(err):.2f} "
          "px)", lambda: not s.tracked[:30, 0].any() and np.array_equal(s.tracks[30, 0], click) and s.manual[30, 0]
          and s.tracked[31:, 0].all() and float(np.median(err)) < 2.0)
    close(win)
    # beside a Moving spot point, the project default AllTracker: the joiner's session is ITS tracker's
    win = new_window(V600)
    s = win.session
    win._on_add(float(GT[0, 1, 0]), float(GT[0, 1, 1]))
    s.points[0].tracker = "spot"
    j = s.add_point(30, float(GT[30, 0, 0]), float(GT[30, 0, 1]))     # placed on frame 30 only
    s.points[j].tracker = "cotracker3"
    win._refresh_point_list()
    win._point_backend = "alltracker"
    built = []
    real_sw = win._session_worker

    def rec(*a, **k):
        w = real_sw(*a, **k)
        built.append((list(w.point_ids), w.point_backend))
        return w
    win._session_worker = rec
    press_t(win, start=0)
    check("[12] a CoTracker3 point joining beside a Moving spot point (project default AllTracker) runs in a CoTracker3 "
          "session (review #1: the default's model)", any(j in p_ for p_, _b in built)
          and all(b == "cotracker3" for p_, b in built if j in p_), built)
    close(win)


# ===================================================================== [13] heads the run does not track itself
class _TwoAnimalsSAM:
    """SAM stand-in: the synthetic pair's true silhouettes (object id = the animal's place in the run + 1)."""

    def __init__(self, ks):
        self.ks = list(ks)

    def new_session(self, start, size):
        import _synth
        from kinetrace import segmenter
        w, h, ks = int(size[0]), int(size[1]), self.ks

        class S:
            def step(self, native, idx, prompts):
                gts = _synth.gt_frame(int(idx))[1]
                m = np.stack([cv2.resize(gts[k]["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                              .astype(bool) for k in ks])
                return segmenter.FrameMasks(int(idx), [i + 1 for i in range(len(ks))], m,
                                            np.full(len(ks), 8.0, np.float32), (w, h), (_synth.W, _synth.H))
        return S()


def animals_window(ks, head_tracker="cotracker3"):
    """The synthetic pair (`ks` of animals 0 and 1), each with its head on its true head on frame 0 and a
    silhouette-derived tail tip; -> (window, {k: head pid}, {k: tip pid})."""
    import _synth
    vid = _synth.build_video(os.path.join(OUT, "synth_animals.mp4"))
    win = new_window(vid)
    s = win.session
    heads, tips = {}, {}
    for j, k in enumerate(ks):
        g = _synth.gt_frame(0)[1][k]
        a = s.ensure_animal() if j == 0 else s.segments[s.add_segment()]
        a.add_click(0, float(g["centre"][0]), float(g["centre"][1]), True)
        idx = s.n_segments - 1
        hp = s.add_point(0, float(g["head"][0]), float(g["head"][1]), name="snout")
        tp = s.add_landmark("tip", "silhouette", "tip")
        s.move_points([hp, tp], idx)
        s.points[hp].tracker = head_tracker
        s.set_head(hp)
        heads[k], tips[k] = hp, tp
    win._refresh_point_list()
    return win, heads, tips


def section13():
    print("[13] app: two heads clicked on one frame; a Hand-only head (D10)")
    import _synth
    real_seg = trk.get_segmenter
    try:
        trk.get_segmenter = lambda be, **k: _TwoAnimalsSAM([0, 1])
        win, heads, tips = animals_window([0, 1])
        s = win.session
        h, n = 10, _synth.T
        for k, hp in heads.items():
            gh = _synth.gt_frame(h)[1][k]["head"]
            click_at(win, hp, h, (float(gh[0]) + 1.0, float(gh[1])))
        ok = press_t(win)
        check("(setup) both animals were tracked", ok and len(win._run_segments()) == 2, win._run_segments())
        for k in (0, 1):
            cf = s.confidence[h + 16:n, tips[k]]
            check(f"[13] both heads clicked on frame {h}: animal {k}'s tail tip is anchored on its own head's session, "
                  f"not capped (median confidence {np.nanmedian(cf):.2f}; review #3)",
                  np.isfinite(cf).sum() > 0.8 * (n - h - 16) and float(np.nanmedian(cf)) > trk.ANCHOR_LOST_CONF + 0.05)
        close(win)
        # a Hand-only head, clicked on every 10th frame: the midline starts at each click; between them it is
        # measured without an anchor (as with no head point), never capped
        trk.get_segmenter = lambda be, **k: _TwoAnimalsSAM([0])
        win, heads, tips = animals_window([0], head_tracker="hand")
        s = win.session
        hp, clicks = heads[0], {}
        for f in range(10, n, 10):
            gh = _synth.gt_frame(f)[1][0]["head"]
            clicks[f] = (float(gh[0]), float(gh[1]))
            click_at(win, hp, f, clicks[f])
        ok = press_t(win)
        mid = s.seg_masks[0].midline
        d = [float(np.linalg.norm(mid[f][0] - np.asarray(xy))) if f in mid else np.inf for f, xy in clicks.items()]
        check("(setup) the animal was tracked to the end", ok and len(mid) >= n - 2, len(mid))
        check(f"[13] a Hand-only head anchors its animal's midline on each frame it was clicked (median "
              f"{np.median(d):.1f} px from the click; review #4)", float(np.median(d)) < 3.0, np.round(d, 1))
        rest = [f for f in range(1, n) if f not in clicks]
        cf = s.confidence[rest, tips[0]]
        check(f"[13] ... and between its clicks the landmarks are not capped (median confidence {np.nanmedian(cf):.2f})",
              float(np.nanmedian(cf)) > trk.ANCHOR_LOST_CONF + 0.05)
        check("[13] ... its clicks are kept exactly", all(s.manual[f, hp] and np.array_equal(s.tracks[f, hp],
                                                                                            np.float32(xy))
                                                         for f, xy in clicks.items()))
        close(win)
    finally:
        trk.get_segmenter = real_seg


# ===================================================================== [14] X after a cut: the old data after the stop
def section14():
    print("[14] app: X after a point was cut, before its click")
    win = new_window(VAN)
    add_points(win, [(AX[0], AY[0]), (BX[0], BY[0])], "cotracker3")
    s = win.session
    old = (np.stack([AX, AY], axis=1) + 3.0).astype(np.float32)          # an older track of dot A, 3 px off
    s.tracks[1:BACK, 0] = old[1:BACK]
    s.tracked[1:BACK, 0] = s.visibility[1:BACK, 0] = True
    s.confidence[1:BACK, 0] = 0.9
    click_at(win, 0, BACK, (AX[BACK], AY[BACK]))
    win.act_select_all.trigger()
    win._goto(0, force=True)
    pump(0.05)
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_T)
    wait(lambda: win.state != READY, 30)
    wait(lambda: win.state == READY or win.current >= GONE + 45, 120)
    QTest.keyClick(win, Qt.Key_X)
    wait(lambda: win.state == READY and win.worker is None, 60)
    stop = win.current
    check("(setup) X stopped the run after dot A was lost and before its click", GONE + 30 < stop < BACK - 10, stop)
    check(f"[14] dot A is cut from where it was lost to the stop (frame {stop})",
          lambda: not s.tracked[GONE + 20:stop + 1, 0].any())
    check("[14] ... and the frames after the stop keep their old data up to the click (D9 choice 2; it was erased)",
          lambda: s.tracked[stop + 1:BACK, 0].all() and np.array_equal(s.tracks[stop + 1:BACK, 0], old[stop + 1:BACK]))
    close(win)


ONLY ={int(x) for x in os.environ.get("HAND_RESTART_ONLY", "").split(",") if x.strip()}   # e.g. "1,2" (a subset)
for k, fn in enumerate((section1, section2, section3, section4, section5, section6, section7, section8, section9,
                         section10, section11, section12, section13, section14), 1):
    if ONLY and k not in ONLY:
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
print("\nverify_hand_restart: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
