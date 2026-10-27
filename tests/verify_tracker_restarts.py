"""Tracker restarts, lost-point blanking, the re-anchor and the first pause (I335-I339, the 2026-10-25 full
sweep's PR 2: dev/work/2026-10-27-tracker-restarts). GPU for [1]-[3], [6], [7]; [4] and [5] need no model.

  [1] I335: a still point B covered (briefly) at the moment ROI moves its crop, because point A runs to the
      crop's edge: B must not follow the cover (today it is re-asked on the cover's picture and follows it at
      full confidence, ~1000 px off). The restart starts from the latest recent frame where every point is in
      view. A synthetic 4K clip; the restart frame r comes from a pilot run without the cover.
  [2] I335, a cover longer than the window (the sweep's clip): Auto-pause on -> the run auto-pauses on B with
      the reason "hidden" inside the cover; Auto-pause off -> B is blank from that frame and named in
      `worker.hidden_at_restart`.
  [3] I336: a 10:1 textured bar turns 90 deg, then moves (ROI restarts): the region's centre keeps following
      the bar after the restart (today it freezes on the background, 357 px off).
  [4] I337 (no model: the point model is scripted): with Auto-pause off, a point lost from frame g0 is blank on
      EVERY frame of the dip, whatever g0's place in the 8-frame windows (today 9-16 of the first 16).
  [5] I338 (no model): the re-anchor keeps LK's position when the seed's look agrees with it to sub-pixel
      level, and still snaps a track that is more than ~1 px off.
  [6] I339, its own fresh process (no model loaded yet): T, then X as soon as the run starts: the window is
      back to READY in < 1 s with nothing written; the next T tracks normally.
  [7] I336: after a run over the turning bar, Track pressed again (QTest T) on the turned bar samples the
      region's members on the bar (today on the drawn, unturned rectangle: 4 of 17 on the bar).
Every check FAILS on 393f21e.
Run: .venv\\Scripts\\python.exe tests\\verify_tracker_restarts.py   (TRACKER_RESTARTS_ONLY=1,4 for a subset)
"""
import os
import shutil
import subprocess
import sys
import time
import traceback
import types
from collections import deque

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
COLD = "--cold" in sys.argv
OUT = os.path.join(HERE, "out", "tracker_restarts")
MEDIA = os.path.join(OUT, "media")              # the synthetic clips, kept between runs (deterministic)
RUN = os.path.join(OUT, "cold" if COLD else "run")
shutil.rmtree(RUN, ignore_errors=True)
os.makedirs(RUN, exist_ok=True)
os.makedirs(MEDIA, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(RUN, "recovery")
os.environ["KINETRACE_SETTINGS"] = os.path.join(RUN, "settings.ini")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402

import kinetrace.tracker as trk  # noqa: E402
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402
from kinetrace.tracker import CONF_PAUSE_RUN, PointSpec, TrackingWorker  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

TRACK_CHOICE = stub_track_choice()     # (G217) the Track pop-up answers "keep my clicks"
V600 = os.path.join(ROOT, "test600.mp4")
V4K = os.path.join(ROOT, "test4k.mp4")
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


def new_window(video):
    forget_recovery(video)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(video)
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 120)
    pump(0.3)
    return win


def close(win):
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


def press_t(win, start=0, timeout=900):
    win.act_select_all.trigger()
    win._goto(start, force=True)
    pump(0.05)
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_T)
    wait(lambda: win.state != READY or win._passes is not None, 15)
    return wait(lambda: win.state == READY and win.worker is None and win._passes is None and win._multi is None,
                timeout)


# ===================================================================== synthetic 4K clips
W4, H4 = 3840, 2160


def tex(rng, h, w, sigma):
    t = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigma)
    t = (t - t.mean()) / (t.std() + 1e-6) * 40 + 128
    return np.clip(t, 0, 255).astype(np.uint8)


def paste(fr, patch, c):
    h, w = patch.shape[:2]
    x0, y0 = int(round(c[0] - w / 2)), int(round(c[1] - h / 2))
    x1, y1 = x0 + w, y0 + h
    if x1 <= 0 or y1 <= 0 or x0 >= W4 or y0 >= H4:
        return
    px0, py0 = max(0, -x0), max(0, -y0)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W4, x1), min(H4, y1)
    fr[y0c:y1c, x0c:x1c] = patch[py0:py0 + (y1c - y0c), px0:px0 + (x1c - x0c)]


# [1] / [2]: point A on a patch running +25 px / frame, point B on a still patch, a 140 px cover passing over B
OT = 140
A0 = np.array([1500.0, 1000.0], np.float32)
B0 = np.array([1500.0, 1300.0], np.float32)


def cover_centre(t, on, off, leave):
    """The cover slides in from the left at 20 px / frame, sits on B over frames on..off, then leaves downwards
    at `leave` px / frame. None = no cover."""
    if on is None:
        return None
    if t < on:
        return np.array([B0[0] - 20.0 * (on - t), B0[1]])
    if t <= off:
        return B0.astype(np.float64).copy()
    return np.array([B0[0], B0[1] + leave * (t - off)])


def occ_clip(name, on=None, off=None, leave=20.0):
    path = os.path.join(MEDIA, f"occ_{name}.avi")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(7)
    bg = cv2.resize(tex(rng, H4 // 4, W4 // 4, 1.5), (W4, H4), interpolation=cv2.INTER_CUBIC)
    pa, pb, po = tex(rng, 48, 48, 1.0), tex(rng, 48, 48, 1.0), tex(rng, OT, OT, 1.0)
    fw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W4, H4))
    for t in range(OT):
        fr = bg.copy()
        paste(fr, pa, A0 + np.array([25.0 * t, 0.0]))
        paste(fr, pb, B0)
        oc = cover_centre(t, on, off, leave)
        if oc is not None:
            paste(fr, po, oc)
        fw.write(fr)
    fw.release()
    return path


# [3] / [7]: a 400 x 40 textured bar turns 90 deg over frames 0-60 about its centre, then moves +30 px / frame
BT = 121
C0 = np.array([1500.0, 1000.0])
BAR = np.array([[-200, -20], [200, -20], [200, 20], [-200, 20]], np.float32)


def bar_pose(t):
    if t <= 60:
        return C0.copy(), 90.0 * t / 60.0
    return C0 + np.array([30.0 * (t - 60), 0.0]), 90.0


def bar_clip():
    path = os.path.join(MEDIA, "bar.avi")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(11)
    bg = cv2.resize(tex(rng, H4 // 4, W4 // 4, 1.5), (W4, H4), interpolation=cv2.INTER_CUBIC)
    bar = tex(rng, 40, 400, 1.0)
    S = 440
    canvas = np.zeros((S, S, 3), np.uint8)
    m = np.zeros((S, S), np.uint8)
    canvas[S // 2 - 20:S // 2 + 20, S // 2 - 200:S // 2 + 200] = bar
    m[S // 2 - 20:S // 2 + 20, S // 2 - 200:S // 2 + 200] = 255
    fw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W4, H4))
    for t in range(BT):
        c, ang = bar_pose(t)
        R = cv2.getRotationMatrix2D((S / 2 - 0.5, S / 2 - 0.5), -ang, 1.0)     # positive = clockwise on screen
        R[0, 2] += c[0] - (S / 2 - 0.5)
        R[1, 2] += c[1] - (S / 2 - 0.5)
        img = cv2.warpAffine(canvas, R, (W4, H4), flags=cv2.INTER_LINEAR)
        mm = cv2.warpAffine(m, R, (W4, H4), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
        fw.write((bg * (1 - mm) + img * mm).astype(np.uint8))
    fw.release()
    return path


def bar_on(t):
    """The bar's outline on frame t (absolute px), for a 'members on the bar' count."""
    c, ang = bar_pose(t)
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (BAR @ R.T + c).astype(np.float32)


def on_bar(pts, t):
    poly = bar_on(t).reshape(-1, 1, 2)
    return sum(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts)


# ===================================================================== a bare worker run
class Run:
    """One TrackingWorker run on this thread; the session's rule: the LAST write of each frame counts."""

    def __init__(self, path, specs, n, roi=True, autopause=False, backend="cotracker3"):
        self.starts = []
        orig = TrackingWorker._compute_crop
        log = self.starts

        def wrapped(w, sp, seg_start=None, model_wh=None):
            c = orig(w, sp, seg_start, model_wh)
            log.append(seg_start)
            return c
        K = len(specs)
        self.tracks = np.full((n, K, 2), np.nan, np.float32)
        self.vis = np.zeros((n, K), bool)
        self.conf = np.zeros((n, K), np.float32)
        self.written = np.zeros(n, bool)
        self.auto, self.err, self.fin = [], [], []

        def on(w0, tr, vi, cf, mem, fr):
            L = len(tr)
            self.tracks[w0:w0 + L] = tr[:, :K]
            self.vis[w0:w0 + L] = vi[:, :K]
            self.conf[w0:w0 + L] = cf[:, :K]
            self.written[w0:w0 + L] = True
        TrackingWorker._compute_crop = wrapped
        try:
            w = TrackingWorker(path, 0, None, None, FrameCache(1024 ** 3), n, roi=roi, autopause=autopause,
                               specs=specs, point_backend=backend, emit_frames=False)
            w.chunk_ready.connect(on)
            w.autopaused.connect(lambda f, pid: self.auto.append((int(f), int(pid))))
            w.error.connect(lambda m: self.err.append(m))
            w.finished_ok.connect(lambda last, p: self.fin.append((int(last), bool(p))))
            w.run()
        finally:
            TrackingWorker._compute_crop = orig
        self.worker = w
        self.restarts = [s for s in self.starts if s]


# ===================================================================== [1] a brief cover at a restart
R_FIRST = {}


def first_restart():
    if "r" not in R_FIRST:
        pilot = Run(occ_clip("none"), [PointSpec(0, A0.copy()), PointSpec(1, B0.copy())], OT)
        R_FIRST["r"] = pilot.restarts[0] if pilot.restarts else None
        R_FIRST["err"] = pilot.err
    return R_FIRST["r"]


def section1():
    print("[1] I335: a still point covered (briefly) when ROI moves its crop")
    r = first_restart()
    if not check(f"(setup) the pilot run without a cover restarts its crop (first restart at frame {r})",
                 r is not None and 8 < r < 60, R_FIRST.get("err")):
        return
    path = occ_clip(f"short{r}", r - 3, r + 3, 40.0)
    ref = Run(path, [PointSpec(0, A0.copy()), PointSpec(1, B0.copy())], OT, roi=False)
    eb_ref = np.linalg.norm(ref.tracks[:, 1] - B0, axis=1)
    check(f"(setup) ROI off, B stays on its patch after the cover (max {np.nanmax(eb_ref[r + 25:]):.2f} px)",
          np.isfinite(eb_ref[r + 25:]).all() and np.nanmax(eb_ref[r + 25:]) < 3.0, ref.err)
    run = Run(path, [PointSpec(0, A0.copy()), PointSpec(1, B0.copy())], OT, roi=True)
    eb = np.linalg.norm(run.tracks[:, 1] - B0, axis=1)
    late = eb[r + 25:]
    check(f"[1] ROI on: B does not follow the cover (B's error from frame {r + 25}: median "
          f"{np.nanmedian(late):.2f}, max {np.nanmax(late) if np.isfinite(late).any() else float('nan'):.2f} px; "
          f"restarts at {run.restarts})",
          np.isfinite(late).all() and float(np.max(late)) < 3.0, run.err)
    a_end = 90                          # A leaves the picture around frame 94
    check("[1] every frame up to A's exit was written, A tracked on all of them (contiguous across restarts)",
          run.written[:a_end].all() and np.isfinite(run.tracks[:a_end, 0]).all())


# ===================================================================== [2] a cover longer than the window
def section2():
    print("[2] I335: a cover longer than the window at a restart")
    r = first_restart()
    if r is None:
        check("(setup) the pilot's first restart", False)
        return
    on, off = r - 8, r + 12
    path = occ_clip(f"long{r}", on, off, 20.0)
    specs = lambda: [PointSpec(0, A0.copy()), PointSpec(1, B0.copy())]   # noqa: E731
    run = Run(path, specs(), OT, roi=True, autopause=True)
    reason = getattr(run.worker, "_autopause_reason", "")
    check(f"[2] Auto-pause on: the run auto-pauses on B with the reason 'hidden' inside the cover "
          f"(auto-pauses {run.auto}, reason {reason!r}, cover {on}-{off})",
          len(run.auto) == 1 and run.auto[0][1] == 1 and reason == "hidden" and on <= run.auto[0][0] <= off,
          run.err)
    if run.auto:
        f = run.auto[0][0]
        eb = np.linalg.norm(run.tracks[:on - 6, 1] - B0, axis=1)
        check(f"[2] B's track before the cover is kept (max {np.nanmax(eb):.2f} px over 0-{on - 7})",
              np.isfinite(eb).all() and float(np.max(eb)) < 3.0)
    off_run = Run(path, specs(), OT, roi=True, autopause=False)
    hid = list(getattr(off_run.worker, "hidden_at_restart", None) or [])
    f = hid[0][1] if hid and hid[0][0] == 1 else None
    check(f"[2] Auto-pause off: B is named in hidden_at_restart ({hid})", f is not None and on <= f <= off)
    if f is not None:
        check(f"[2] Auto-pause off: B is blank from frame {f} to the end, A tracked to its exit",
              np.isnan(off_run.tracks[f:, 1]).all() and np.isfinite(off_run.tracks[:90, 0]).all())
    else:
        eb = np.linalg.norm(off_run.tracks[off + 25:, 1] - B0, axis=1)
        check("[2] Auto-pause off: B never follows the cover (no data, or on its patch)",
              not np.isfinite(eb).any() or float(np.nanmax(eb)) < 3.0,
              f"max {np.nanmax(eb) if np.isfinite(eb).any() else float('nan'):.1f} px")


# ===================================================================== [3] a turning bar at a restart
def section3():
    print("[3] I336: a 10:1 region that turned 90 deg, at an ROI restart")
    path = bar_clip()
    radius = float(np.max(np.hypot(BAR[:, 0], BAR[:, 1])))       # as app._on_add_region sets it
    run = Run(path, [PointSpec(0, C0.astype(np.float32), "group", radius, False, BAR.copy())], BT, roi=True)
    gt = np.stack([bar_pose(t)[0] for t in range(BT)])
    e = np.linalg.norm(run.tracks[:, 0] - gt, axis=1)
    s1 = next((s for s in run.restarts if s >= 60), None)
    check(f"(setup) the region tracks the turn before any restart (max {np.nanmax(e[:61]):.2f} px over 0-60; "
          f"restarts at {run.restarts})", np.isfinite(e[:61]).all() and np.nanmax(e[:61]) < 5.0, run.err)
    if not check("(setup) a restart after the turn", s1 is not None):
        return
    after = e[s1:]
    check(f"[3] the centre keeps following the bar after the restart at {s1} (median "
          f"{np.nanmedian(after):.2f}, max {np.nanmax(after) if np.isfinite(after).any() else float('nan'):.2f} "
          f"px; frame 90: {e[90]:.1f} px)",
          np.isfinite(after).all() and float(np.median(after)) < 5.0 and float(np.max(after)) < 10.0)


# ===================================================================== [4] blanking, the point model scripted
class _FakeModel:
    step = 8
    interp_shape = (384, 512)


def section4():
    print("[4] I337: Auto-pause off, a lost point is blank on every frame of its dip, whatever the phase")
    END = 80
    xy = np.array([200.0, 150.0], np.float32)
    orig_get, orig_step = trk.get_model, TrackingWorker._model_step
    counts = {}
    try:
        trk.get_model = lambda **k: (_FakeModel(), "cpu")
        for g0 in range(16, 24):
            calls = {"n": 0}

            def step(self, model, frames, device, first, queries, g0=g0, calls=calls):
                if first:
                    return None
                calls["n"] += 1
                fed = 8 + 8 * calls["n"]                 # frames fed when this window is processed
                L = min(16, fed)
                f = np.arange(fed - L, fed)
                tr = np.tile(xy, (L, 1, 1)).astype(np.float32)
                vis = (f < g0)[:, None]
                conf = np.where(f < g0, 0.9, 0.1).astype(np.float32)[:, None]
                return tr, vis, conf
            TrackingWorker._model_step = step
            run = Run(V600, [PointSpec(0, xy.copy())], END, roi=False, autopause=False)
            gone = np.isnan(run.tracks[g0:, 0]).all(axis=1)
            counts[g0] = (int(gone[:CONF_PAUSE_RUN].sum()), int(gone.sum()), END - g0, run.err)
    finally:
        trk.get_model, TrackingWorker._model_step = orig_get, orig_step
    for g0, (n16, n, span, err) in counts.items():
        check(f"[4] loss from frame {g0} (phase {g0 % 8}): {n16}/16 of its first frames blank, {n}/{span} of the dip",
              n16 == CONF_PAUSE_RUN and n == span and not err, err or None)


# ===================================================================== [5] the re-anchor
def _anchor_stub(refine=True):
    ns = types.SimpleNamespace(_gate=15.0, _do_refine=refine, _lk_win=43, _frame_wh=(1000, 1000),
                               _templates={}, _template_frac={}, _refined={})
    ns._reanchor = lambda g, t, p: TrackingWorker._reanchor(ns, g, t, p)
    return ns


def section5():
    print("[5] I338: the re-anchor snaps only when the seed's look is more than ~1 px away")
    rng = np.random.default_rng(3)
    g0 = cv2.GaussianBlur(rng.uniform(0, 255, (1000, 1000)).astype(np.float32), (0, 0), 2.0)
    g0 = cv2.normalize(g0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    seed = np.array([500.4, 500.0], np.float32)
    sp = PointSpec(0, seed.copy(), anchor=True)
    same, errs = [], []
    for dx in np.round(np.arange(0.1, 1.0, 0.1), 2):
        g1 = cv2.warpAffine(g0, np.float32([[1, 0, dx], [0, 1, 0]]), (1000, 1000), flags=cv2.INTER_CUBIC)
        truth = np.array([500.4 + dx, 500.0])
        tracks = np.array([[seed], [[500.4 + dx + 0.5, 500.0]]], np.float32)
        vis = np.ones((2, 1), bool)
        lk = TrackingWorker._refine_window(_anchor_stub(), 0, tracks.copy(), vis, deque([(0, g0), (1, g1)]), 0,
                                           {})[1, 0]
        ns = _anchor_stub()
        TrackingWorker._capture_templates(ns, g0, [sp])
        out = TrackingWorker._refine_window(ns, 0, tracks.copy(), vis, deque([(0, g0), (1, g1)]), 0, {0: sp})[1, 0]
        same.append(bool(np.array_equal(out, lk)))
        errs.append((float(np.linalg.norm(lk - truth)), float(np.linalg.norm(out - truth))))
    check(f"[5] sub-pixel shifts 0.1-0.9 px: the anchored output is LK's own on all 9 "
          f"({sum(same)}/9; mean error anchored {np.mean([e[1] for e in errs]):.3f} vs LK "
          f"{np.mean([e[0] for e in errs]):.3f} px)", all(same))
    # a track that is off by d px (no LK: the model's position alone), the picture moved by 0.3 px
    g1 = cv2.warpAffine(g0, np.float32([[1, 0, 0.3], [0, 1, 0]]), (1000, 1000), flags=cv2.INTER_CUBIC)
    truth = np.array([500.7, 500.0])
    res = {}
    for d in (0.5, 1.5, 3.0, 6.0):
        ns = _anchor_stub(refine=False)
        TrackingWorker._capture_templates(ns, g0, [sp])
        tracks = np.array([[seed], [[500.7 + d, 500.0]]], np.float32)
        out = TrackingWorker._refine_window(ns, 0, tracks, np.ones((2, 1), bool), deque([(0, g0), (1, g1)]), 0,
                                            {0: sp})[1, 0]
        res[d] = float(np.linalg.norm(out - truth))
    check(f"[5] a track 0.5 px off is left alone (error {res[0.5]:.3f} px)", abs(res[0.5] - 0.5) < 1e-4)
    check(f"[5] tracks 1.5 / 3 / 6 px off are snapped back (errors {res[1.5]:.3f} / {res[3.0]:.3f} / "
          f"{res[6.0]:.3f} px)", max(res[1.5], res[3.0], res[6.0]) < 0.15)


# ===================================================================== [6] X during the first model load
def section6():
    print("[6] I339: X right after the first Track of a session (its own process, no model loaded)")
    gt = np.load(V4K + ".gt.npz")["gt"]
    win = new_window(V4K)
    win._on_add(float(gt[0, 0, 0]), float(gt[0, 0, 1]))
    win._refresh_point_list()
    s = win.session
    win.act_select_all.trigger()
    win._goto(0, force=True)
    pump(0.3)
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_T)
    started = wait(lambda: win.state == TRACKING, 60)
    t0 = time.time()
    QTest.keyClick(win.canvas.viewport(), Qt.Key_X)
    done = wait(lambda: win.state == READY and win.worker is None, 30)
    dt = time.time() - t0
    check("(setup) the run started", started)
    check(f"[6] X during the model load: back to READY in {dt:.2f} s (< 1 s)", started and done and dt < 1.0)
    check(f"[6] nothing was written ({int(s.tracked[:, 0].sum())} tracked frame(s): the click only)",
          int(s.tracked[1:, 0].sum()) == 0)
    msg = win.statusBar().currentMessage()
    check(f"[6] the window says so ({msg!r})", "Paused before any frames" in msg)
    ok = press_t(win, 0)
    n = int(s.tracked[:, 0].sum())
    check(f"[6] the next T tracks normally ({n} of {s.n_frames} frames)", ok and n == s.n_frames)
    close(win)


# ===================================================================== [7] a resumed Track on the turned bar
def section7():
    print("[7] I336: Track pressed again on the turned bar uses its current turn")
    path = bar_clip()
    win = new_window(path)
    win._goto(0, force=True)
    pump(0.2)
    win._on_add_region("rect", (BAR + C0).tolist())
    s = win.session
    for k in range(s.n_points):
        s.points[k].tracker = "cotracker3"
    win._refresh_point_list()
    ok = press_t(win, 0)
    e60 = float(np.linalg.norm(s.tracks[60, 0] - C0)) if s.tracked[60, 0] else float("nan")
    check(f"(setup) the first run tracked the bar through its turn (frame 60: {e60:.2f} px off)",
          ok and e60 < 5.0)
    seen = []
    orig = trk.sample_members

    def logged(center, radius, w, h, outline=None):
        pts = orig(center, radius, w, h, outline)
        seen.append(pts.copy())
        return pts
    trk.sample_members = logged
    try:
        win.activateWindow()
        win._goto(60, force=True)
        pump(0.1)
        QTest.keyClick(win, Qt.Key_T)
        wait(lambda: len(seen) > 0, 120)
        win._pause_tracking()
        wait(lambda: win.state == READY and win.worker is None, 60)
    finally:
        trk.sample_members = orig
    n_on = on_bar(seen[0], 60) if seen else 0
    check(f"[7] the resumed run's members sit on the turned bar ({n_on} of {len(seen[0]) if seen else 0})",
          bool(seen) and n_on >= len(seen[0]) - 2)
    close(win)


def finish(name):
    for w_ in QApplication.topLevelWidgets():
        probe = getattr(w_, "_dev_probe", None)
        if probe is not None:
            probe.wait(10000)
    print(f"\n{name}: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if COLD:
    try:
        section6()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check("(section6 ran to its end)", False, repr(e))
    finish("verify_tracker_restarts [6]")

ONLY = {int(x) for x in os.environ.get("TRACKER_RESTARTS_ONLY", "").split(",") if x.strip()}
if not ONLY or 6 in ONLY:
    # first, before this process loads any model: [6] needs a cold start
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold"], capture_output=True, text=True,
                       errors="replace", timeout=1800)
    print(p.stdout.rstrip(), flush=True)
    if p.returncode != 0 and p.stderr.strip():
        print(p.stderr.strip()[-2000:], flush=True)
    check("[6] the cold-start checks (own process) passed", p.returncode == 0, f"exit {p.returncode}")
for k, fn in ((1, section1), (2, section2), (3, section3), (4, section4), (5, section5), (7, section7)):
    if ONLY and k not in ONLY:
        continue
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))
finish("verify_tracker_restarts")
