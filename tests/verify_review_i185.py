"""I185 part 2 (code review 2026-10-02, owner go-ahead 2026-10-03): the SECOND pass of a two-pass Track
(AllTracker points, then CoTracker3 points) is kept on the silhouettes the first pass just made. Before,
only pass 1 carried the segment, so the second pass's on-body landmarks ran with no constraint, no stop
where a landmark leaves the animal and no off-body demotion. The worker takes `stored_masks` (the
session's MaskTrack) and drives the SAME constraint code from it; no SAM, nothing emitted.

Sections (every check FAILS on the code before the change: the old worker has no `stored_masks`, the old
app starts pass 2 with no silhouette)
  [1] a bare worker (no model, no video): the on-body rules from stored silhouettes, identical to the
      live-segment path on the same mask: nudge, exit stop, merged / off-body demotion, a frame with no
      stored silhouette = no constraint, nothing without the input
  [2] real CoTracker3 runs on test600 with a mock segmenter: stored == live, bit for bit; the landmark
      that slides off the silhouette stops the run at that frame; nudged / demoted; a gap in the stored
      silhouettes; no masks emitted
  [3] the app: a real two-pass run (AllTracker + head, then CoTracker3) with a mock SAM in pass 1: pass 2
      is kept on the silhouettes and stops where its landmark leaves the animal; the Track tooltip says so
  [4] every-camera two-pass plumbing: each camera's pass 2 gets THAT camera's silhouettes; (G180) any run
      of a held point is kept on the saved silhouettes, a point not held gets none

Run: .venv\\Scripts\\python.exe tests\\verify_review_i185.py   (the models are used: CoTracker3 and AllTracker)
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("KINETRACE_REVIEW_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from _clean import forget_recovery  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))

app = QApplication.instance() or QApplication([])
from kinetrace import alltracker_backend, segmenter  # noqa: E402
from kinetrace import tracker as trk  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.segmenter import MaskTrack, working_size  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

print("testing", os.path.dirname(os.path.abspath(trk.__file__)))
WT = os.path.dirname(HERE)
OUT = os.path.join(WT, "tests", "out", "review_i185")
os.makedirs(OUT, exist_ok=True)
V600 = os.path.join(WT, "test600.mp4")
GT = np.load(V600 + ".gt.npz")["gt"]            # (600, 4, 2)
FAILS: list = []
HALF = (100, 100)                                # the mock silhouette: a rectangle round dot 0, +-HALF px
HAS_INPUT = "stored_masks" in trk.TrackingWorker.__init__.__code__.co_varnames


def check(label, cond, detail=""):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001 - the old code raises where the new answers
            cond, detail = False, f"raised {e!r}"
    cond = bool(cond)
    print(("  ok    " if cond else "  FAIL  ") + label + ("" if cond or not detail else f"   [{detail}]"), flush=True)
    if not cond:
        FAILS.append(label)


def pump(t=0.1):
    t0 = time.time()
    while time.time() - t0 < t:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t0 = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


# ================================================================= the mock silhouette
def rect_mask(f, w=640, h=480, half=HALF):
    cx, cy = GT[f, 0]
    x0, x1 = int(round(cx - half[0])), int(round(cx + half[0]))
    y0, y1 = int(round(cy - half[1])), int(round(cy + half[1]))
    m = np.zeros((h, w), bool)
    m[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
    return m


class MockAnimal:
    """SAM stand-in: the rectangle round dot 0 of test600 on every frame (no weights, deterministic)."""

    def new_session(self, start, size):
        w, h = size

        class S:
            def step(self, native, idx, prompts):
                m = rect_mask(idx, w, h)
                return segmenter.FrameMasks(idx, [1], m[None], np.array([8.0], np.float32), (w, h), (w, h))
        return S()


def stored_track(n, absent=(), w=640, h=480):
    """The session's silhouettes as the app stores them (outline polygons from `summarize_mask`)."""
    mt = MaskTrack(n, w, h)
    for f in range(n):
        if f not in absent:
            mt.set_from_work(f, rect_mask(f, w, h), (1.0, 1.0), 8.0)
    return mt


# where dot 1 (P1) leaves the rectangle by more than the exit band (3 % of the diagonal, >= 8 px), from the
# ground truth
N = 100
_d = GT[:, 1] - GT[:, 0]
dist = np.hypot(np.maximum(np.abs(_d[:, 0]) - HALF[0], 0), np.maximum(np.abs(_d[:, 1]) - HALF[1], 0))
band = max(8.0, 0.03 * float(np.hypot(2 * HALF[0] + 1, 2 * HALF[1] + 1)))
EXPECT = int(np.nonzero(dist > band)[0][0])
print(f"(dot 1 leaves the silhouette by more than {band:.1f} px at frame {EXPECT})")


# ================================================================= [1] a bare worker
def section1():
    print("[1] the on-body rules from stored silhouettes (bare worker)")
    NW, NH = 2000, 1500
    WW, WH = working_size(NW, NH)
    SX, SY = NW / WW, NH / WH

    def nat(xw, yw):
        return np.array([(xw + 0.5) * SX - 0.5, (yw + 0.5) * SY - 0.5], np.float32)

    def to_work(p):
        return (p[0] + 0.5) / SX - 0.5, (p[1] + 0.5) / SY - 0.5

    ell = np.zeros((WH, WW), np.uint8)
    cv2.ellipse(ell, (512, 384), (300, 150), 0, 0, 360, 1, -1)
    big = MaskTrack(40, NW, NH)
    for f in range(40):
        if f != 5:                                   # frame 5 has no silhouette
            big.set_from_work(f, ell.astype(bool), (SX, SY), 8.0)
    empty = MaskTrack(40, NW, NH)

    def bare(seeds, constrain, on_body=None, stored=big, live_of=None):
        specs = [trk.PointSpec(i, np.asarray(s, np.float32)) for i, s in enumerate(seeds)]
        kw = {}
        if live_of is not None:
            kw["animal"] = trk.AnimalSpec({0: []}, {}, None)
        else:
            kw["stored_masks"] = stored
        w = trk.TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100, specs=specs,
                               constrain_pids=constrain, on_body_pids=on_body, **kw)
        w._refined = {}
        if live_of is not None:                      # the live path's per-frame state, taken from the stored worker
            w._sxy = live_of._sxy
            w._mask_hist = dict(live_of._mask_hist)
            w._summ = dict(live_of._summ)
        else:
            w._stored_setup(NW, NH)
            for f in range(8):
                w._stored_step(f)
        return w

    seeds = [nat(512, 384), nat(812, 384), nat(300, 384), nat(512, 10)]
    out_map = [("point", i) for i in range(4)]
    L = 4

    def drive(w):
        tr = np.zeros((L, 4, 2), np.float32)
        for i in range(L):
            tr[i, 0] = nat(512, 384)                                   # inside
            tr[i, 1] = nat(814, 384)                                   # 2 working px outside the right edge: jitter
            tr[i, 2] = nat(300, 384) if i == 0 else nat(900, 384)      # leaves at row 1 by ~90 working px
            tr[i, 3] = nat(512, 10)                                    # free, far outside: untouched
        w._constrain_to_mask(2, tr, w.specs, out_map, first_seg=False)
        return tr

    ws = bare(seeds, [0, 1, 2])
    check("stored: the worker built the silhouette state (working size, per-axis scale)",
          lambda: ws._stored_work == (WW, WH) and np.allclose(ws._sxy, (SX, SY)) and 2 in ws._mask_hist
          and 2 in ws._summ)
    tr_s = drive(ws)
    check("stored: the landmark that leaves the silhouette stops the run at that frame",
          lambda: ws._exit_hit == (3, 2) and ws._autopause_hit == (3, 2) and ws._autopause_reason == "exit",
          ws._exit_hit)
    check("stored: its data is blank from that frame, and only from it",
          lambda: np.isfinite(tr_s[0, 2]).all() and np.isnan(tr_s[1:, 2]).all())
    mk = ws._mask_hist[3][0]
    xw, yw = to_work(tr_s[1, 1])
    check("stored: outline jitter is nudged onto the silhouette, not an exit",
          lambda: mk[int(round(yw)), int(round(xw))] and 1 in ws._snapped[3] and 2 not in ws._snapped[3])
    check("stored: an inside point and a free point are never touched",
          lambda: np.allclose(tr_s[:, 0], nat(512, 384)) and np.allclose(tr_s[:, 3], nat(512, 10)))

    wl = bare(seeds, [0, 1, 2], live_of=ws)
    tr_l = drive(wl)
    check("stored == the live-segment path on the same mask (positions, exit, snapped / merged sets)",
          lambda: np.array_equal(tr_s, tr_l, equal_nan=True) and ws._exit_hit == wl._exit_hit
          and ws._snapped == wl._snapped and ws._collapsed == wl._collapsed)

    # merged landmarks: clearly apart when clicked, now on one pixel
    wm = bare([nat(300, 384), nat(700, 384)], [0, 1])
    trm = np.zeros((L, 2, 2), np.float32)
    trm[:, 0] = nat(500, 384)
    trm[:, 1] = nat(501, 384)
    wm._constrain_to_mask(0, trm, wm.specs, [("point", 0), ("point", 1)], first_seg=False)
    check("stored: two landmarks that merged are flagged", lambda: all(wm._collapsed[f] == {0, 1} for f in range(L)),
          wm._collapsed)
    cf = np.ones((L, 2), np.float32)
    wm._demote_merged(0, cf)
    check("stored: ... and demoted like an off-body point", lambda: (cf <= trk.OFF_BODY_CONF).all(), cf)

    # off-body demotion: an on-body point (not constrained) outside the dilated silhouette
    wd = bare([nat(512, 384), nat(100, 700)], [0], on_body=[0, 1])
    out_tr = np.zeros((L, 2, 2), np.float32)
    out_tr[:, 0] = nat(512, 384)
    out_tr[:, 1] = nat(100, 700)
    out_cf = np.ones((L, 2), np.float32)
    wd._demote_off_body(0, out_tr, out_cf)
    check("stored: an on-body point off the animal is demoted, the one on it is not",
          lambda: (out_cf[:, 1] <= trk.OFF_BODY_CONF).all() and (out_cf[:, 0] == 1.0).all(), out_cf)

    # a frame with no stored silhouette = no constraint, never a crash
    wg = bare(seeds, [0, 1, 2])
    wg._stored_step(5)
    trg = np.zeros((1, 4, 2), np.float32)
    trg[0, 1] = nat(900, 384)                        # far outside
    trg[0, 2] = nat(900, 384)
    before = trg.copy()
    wg._constrain_to_mask(5, trg, wg.specs, out_map, first_seg=False)
    check("no stored silhouette on a frame: nothing constrained, nothing stops, no crash",
          lambda: 5 not in wg._mask_hist and np.array_equal(trg, before) and wg._exit_hit is None)
    we = bare(seeds, [0, 1, 2], stored=empty)
    we._stored_step(3)
    tre = np.zeros((1, 4, 2), np.float32)
    tre[0, 1] = nat(900, 384)
    we._constrain_to_mask(3, tre, we.specs, out_map, first_seg=False)
    check("a track with no silhouette at all: the same", lambda: we._exit_hit is None and not we._mask_hist)

    # no input, no animal: the old behaviour -- nothing is constrained
    w0 = trk.TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100,
                            specs=[trk.PointSpec(i, np.asarray(s, np.float32)) for i, s in enumerate(seeds)],
                            constrain_pids=[0, 1, 2])
    w0._refined = {}
    tr0 = drive(w0)
    check("without the input nothing is constrained (unchanged): the exited landmark keeps its data",
          lambda: w0._exit_hit is None and np.isfinite(tr0[1:, 2]).all())

    # a live segment wins over a stored one
    wb = trk.TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100,
                            specs=[trk.PointSpec(0, nat(512, 384))], animal=trk.AnimalSpec({0: []}, {}, None),
                            stored_masks=big)
    check("a live segment wins when both are given", lambda: wb._stored is None)


# ================================================================= [2] real CoTracker3 runs
def run_real(stored=None, live=False, n=N, constrain=(0, 1), on_body=(0, 1, 2)):
    specs = [trk.PointSpec(i, GT[0, i].astype(np.float32).copy()) for i in range(3)]
    saved = trk.get_segmenter
    kw = {}
    if live:
        trk.get_segmenter = lambda be, **k: MockAnimal()
        kw["animal"] = trk.AnimalSpec({0: [(float(GT[0, 0, 0]), float(GT[0, 0, 1]), 1)]}, {}, None, "stub")
    if stored is not None:
        kw["stored_masks"] = stored
    try:
        w = trk.TrackingWorker(V600, 0, None, None, FrameCache(1 << 28), n, specs=specs, constrain_pids=list(constrain),
                               on_body_pids=list(on_body), point_backend="cotracker3",
                               autopause=False, **kw)   # the demoted off-body point would pause the run on low confidence
        rec = {"chunks": [], "masks": 0, "paused": None, "fin": None, "err": None, "reason": None}
        w.chunk_ready.connect(lambda w0, tr, vi, cf, mm, fr: rec["chunks"].append((w0, tr.copy(), vi.copy(), cf.copy())))
        w.masks_ready.connect(lambda s: rec.__setitem__("masks", rec["masks"] + 1))
        w.autopaused.connect(lambda f, p: rec.__setitem__("paused", (f, p)))
        w.finished_ok.connect(lambda last, p: rec.__setitem__("fin", (last, p)))
        w.error.connect(lambda m: rec.__setitem__("err", m))
        w.run()
        rec["reason"] = w._autopause_reason
        rec["ncols"] = w.n_cols
        return rec
    finally:
        trk.get_segmenter = saved


def stitch(rec, k):
    tr, cf = {}, {}
    for w0, t, v, c in rec["chunks"]:
        for i in range(t.shape[0]):
            tr[w0 + i], cf[w0 + i] = t[i, k], c[i, k]
    return tr, cf


def section2():
    print("[2] real runs on test600 (CoTracker3, mock segmenter)")
    track = stored_track(N + 8)
    rec_s = run_real(stored=track)
    rec_l = run_real(live=True)
    rec_n = run_real()
    for nm, r in (("stored", rec_s), ("live", rec_l), ("control", rec_n)):
        check(f"({nm} run finished without an error)", r["err"] is None and r["fin"] is not None, r["err"])
    check("stored == live: every chunk of the run, bit for bit (tracks, visibility, confidence)",
          lambda: len(rec_s["chunks"]) == len(rec_l["chunks"]) and all(
              a[0] == b[0] and np.array_equal(a[1], b[1], equal_nan=True) and np.array_equal(a[2], b[2])
              and np.array_equal(a[3], b[3]) for a, b in zip(rec_s["chunks"], rec_l["chunks"])),
          (len(rec_s["chunks"]), len(rec_l["chunks"])))
    check("stored == live: the same stop and the same finish", lambda: rec_s["paused"] == rec_l["paused"]
          and rec_s["fin"] == rec_l["fin"] and rec_s["reason"] == rec_l["reason"], (rec_s["paused"], rec_l["paused"]))
    f_stop, pid_stop = rec_s["paused"] or (None, None)
    check("the landmark that slides off the silhouette stops the run at that frame (exit)",
          lambda: pid_stop == 1 and rec_s["reason"] == "exit" and abs(f_stop - EXPECT) <= 4,
          (rec_s["paused"], EXPECT))
    p1, _c1 = stitch(rec_s, 1)
    p1n, _ = stitch(rec_n, 1)
    check("... and its data is blank from that frame on, present before it",
          lambda: all(np.isfinite(p1[f]).all() for f in range(0, f_stop))
          and all(np.isnan(p1[f]).all() for f in p1 if f >= f_stop))
    check("(control: no input) it is never stopped and keeps its data to the end of the run",
          lambda: rec_n["paused"] is None and rec_n["fin"][0] == N - 1 and np.isfinite(p1n[N - 1]).all(),
          rec_n["paused"])
    on = lambda f, p: rect_mask(f)[int(round(min(max(p[1], 0), 479))), int(round(min(max(p[0], 0), 639)))]  # noqa: E731
    kept = list(range(1, f_stop))
    strays = [f for f in kept if dist[f] > 0]
    check("in-band jitter is nudged onto the silhouette (every kept frame is on it; the control strays off it)",
          lambda: all(on(f, p1[f]) for f in kept) and not all(on(f, p1n[f]) for f in strays),
          (sum(bool(on(f, p1[f])) for f in kept), len(kept)))
    _p2s, c2s = stitch(rec_s, 2)
    _p2n, c2n = stitch(rec_n, 2)
    check("the on-body point that is not on the animal is demoted (confidence <= 0.2), as in the live run",
          lambda: all(c2s[f] <= trk.OFF_BODY_CONF for f in range(1, f_stop)) and np.median(list(c2n.values())) > 0.5,
          (max(c2s.values()), np.median(list(c2n.values()))))
    check("no masks emitted, no derived column, the session's silhouettes are not touched",
          lambda: rec_s["masks"] == 0 and rec_s["ncols"] == 3 and track.n_masked() == N + 8,
          (rec_s["masks"], rec_s["ncols"]))
    # a gap in the stored silhouettes: the exit falls in it, so the run goes on until the next silhouette
    gap = tuple(range(EXPECT - 12, EXPECT + 14))
    rec_g = run_real(stored=stored_track(N + 8, absent=gap))
    check("a gap of stored silhouettes: no crash, no constraint on its frames, the stop comes after it",
          lambda: rec_g["err"] is None and rec_g["paused"] is not None and rec_g["paused"][0] >= EXPECT + 14,
          (rec_g["err"], rec_g["paused"], gap[0], gap[-1]))
    # (G180) one run, two animals: A segmented live (its row selected), B's held point kept on B's SAVED
    # silhouettes (only its point selected)
    specs = [trk.PointSpec(i, GT[0, i].astype(np.float32).copy()) for i in range(3)]
    saved = trk.get_segmenter
    trk.get_segmenter = lambda be, **k: MockAnimal()
    rec_m = {"masks": 0, "paused": None, "err": None, "fin": None}
    try:
        live_a = trk.AnimalSpec({0: [(float(GT[0, 0, 0]), float(GT[0, 0, 1]), 1)]}, {}, None, "stub", index=0,
                                name="A", on_body={0}, constrain={0})
        stored_b = trk.AnimalSpec(index=1, name="B", stored=stored_track(N + 8), on_body={1, 2}, constrain={1})
        wm = trk.TrackingWorker(V600, 0, None, None, FrameCache(1 << 28), N, specs=specs, point_backend="cotracker3",
                                autopause=False, animals=[live_a, stored_b])
        wm.masks_ready.connect(lambda summ: rec_m.__setitem__("masks", rec_m["masks"] + 1))
        wm.autopaused.connect(lambda f, p: rec_m.__setitem__("paused", (f, p)))
        wm.finished_ok.connect(lambda last, p: rec_m.__setitem__("fin", (last, p)))
        wm.error.connect(lambda m: rec_m.__setitem__("err", m))
        wm.run()
        rec_m["reason"] = wm._autopause_reason
    finally:
        trk.get_segmenter = saved
    check("G180 a live animal and a stored one in ONE run: no error, the live one's silhouettes are emitted",
          lambda: rec_m["err"] is None and rec_m["masks"] > 0, (rec_m["err"], rec_m["masks"]))
    check("G180 ... and B's held point still stops where it leaves B's saved silhouette",
          lambda: rec_m["paused"] is not None and rec_m["paused"][1] == 1 and rec_m["reason"] == "exit"
          and abs(rec_m["paused"][0] - EXPECT) <= 4, (rec_m["paused"], rec_m.get("reason"), EXPECT))


# ================================================================= [3] the app: a real two-pass run
def select_rows(win, *rows):
    win.layers.clearSelection()
    for r in rows:
        win.layers.point_item(r).setSelected(True)
    pump(0.05)


def close(win):
    win.close()
    pump(0.3)
    win._dev_probe.wait(30000)


class Toasts:
    def __init__(self, win):
        self.msgs = []
        win.toast.show_message = lambda text, level="info", ms=6000, on_click=None: self.msgs.append(text)


CLIP = os.path.join(OUT, "clip100.mp4")


def make_clip():
    cap = cv2.VideoCapture(V600)
    vw = cv2.VideoWriter(CLIP, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for _ in range(100):
        ok, fr = cap.read()
        vw.write(fr)
    cap.release()
    vw.release()


def section3():
    print("[3] the app: a real two-pass run")
    if not alltracker_backend.available():
        print("  (AllTracker is not installed here: section [3] skipped)")
        return
    make_clip()
    forget_recovery(CLIP)
    win = MainWindow()
    win.resize(1500, 950)
    win.show()
    win._open_video(CLIP)
    wait(lambda: win.state == READY, 60, "open")
    s = win.session
    toasts = Toasts(win)
    win._on_add(*[float(v) for v in GT[0, 0]])
    win._on_add(*[float(v) for v in GT[0, 1]])
    s.points[0].tracker = "alltracker"
    s.points[1].tracker = "cotracker3"
    s.ensure_animal()
    s.animal.add_click(0, float(GT[0, 0, 0]), float(GT[0, 0, 1]), True)
    # (G153, G156, G160) both points are the animal's, it holds them, point 0 is its head
    s.move_points([0, 1], 0)
    s.animal.hold = True
    s.set_head(0)
    win._refresh_point_list()
    win._goto(0, force=True)
    select_rows(win, 0, 1)
    win._on_select(0)
    select_rows(win, 0, 1)
    win._animal_item(0).setSelected(True)            # (G180) the animal's row = its silhouette, with the points
    pump(0.05)
    win._update_track_button()
    check("the Track button's tooltip says the second pass is kept on the first pass's silhouettes",
          "kept on the silhouettes the first one makes" in win.btn_track.toolTip(), win.btn_track.toolTip())
    runs = []
    orig_launch = win._launch_run

    def spy(w, *a, **k):
        runs.append({"w": w, "masks": 0})
        w.masks_ready.connect(lambda summ, r=runs[-1]: r.__setitem__("masks", r["masks"] + 1))
        return orig_launch(w, *a, **k)

    win._launch_run = spy
    saved = trk.get_segmenter
    trk.get_segmenter = lambda be, **k: MockAnimal()
    try:
        win._toggle_tracking()
        wait(lambda: win.state == READY and win._passes is None and win._multi is None and len(runs) >= 1, 900,
             "the two-pass run")
        pump(0.5)
    finally:
        trk.get_segmenter = saved
    check("(setup) two passes ran, the first with the segment", len(runs) == 2 and runs[0]["w"].animal is not None,
          [type(r["w"]).__name__ for r in runs])
    w2 = runs[1]["w"] if len(runs) > 1 else None
    check("the second pass runs no segmenter and is handed the session's silhouettes",
          lambda: w2.animal is None and getattr(w2, "_stored", None) is s.masks)
    check("... with the same on-body rules as a run with the segment (its point is constrained and on-body)",
          lambda: w2.constrain == {1} and w2.on_body == {1}, (getattr(w2, "constrain", None), getattr(w2, "on_body", None)))
    check("... and it emits no silhouettes of its own (pass 1 did)",
          lambda: runs[1]["masks"] == 0 and runs[0]["masks"] > 0, (runs[0]["masks"], runs[1]["masks"] if len(runs) > 1 else None))
    last1 = int(np.nonzero(s.tracked[:, 1])[0].max()) if s.tracked[:, 1].any() else -1
    check("the CoTracker3 landmark that leaves the animal ends the run: its data stops just before that frame",
          lambda: EXPECT - 8 <= last1 <= EXPECT + 1, (last1, EXPECT))
    last0 = int(np.nonzero(s.tracked[:, 0])[0].max()) if s.tracked[:, 0].any() else -1
    check("... and the first pass's point is trimmed to where the second pass's run ended (not on to frame 99)",
          lambda: last1 <= last0 <= EXPECT + 12, (last0, last1))
    check("... and the notice says which point was lost",
          lambda: any("was lost" in m or "stopped" in m for m in toasts.msgs), toasts.msgs[-2:])
    close(win)


# ================================================================= [4] every-camera plumbing
def section4():
    print("[4] every camera: each pass 2 gets that camera's silhouettes")
    make_clip()
    forget_recovery(CLIP)
    proj = os.path.join(OUT, "i185_rig.kinetrace")
    sessions = [TrackingSession(CLIP, 100, 30.0, 640, 480) for _ in range(3)]
    Project(sessions, ["camA", "camB", "camC"], [0, 0, 0]).save(proj)
    win = MainWindow()
    win.resize(1500, 950)
    win.show()
    win._open_project_from_path(proj)
    wait(lambda: win.state == READY and win.project is not None and win.project.n_views == 3, 60, "open rig")
    p = win.project
    for v in range(3):
        sv = p.sessions[v]
        win._set_active_view(v)
        win._goto(0, force=True)
        for i in range(2):
            if v == 0:
                win._on_add(*[float(x) for x in GT[0, i]])      # the other cameras get placeholders (G19)
            else:
                win._on_select(i)
                win._on_annotate(*[float(x) for x in GT[0, i]])
        sv.points[0].tracker = "alltracker"
        sv.points[1].tracker = "cotracker3"
        sv.ensure_animal()
        sv.animal.add_click(0, float(GT[0, 0, 0]), float(GT[0, 0, 1]), True)
        if v != 2:                                    # camera C: its first pass did not carry a segment
            for f in range(40):
                sv.masks.set_from_work(f, rect_mask(f), (1.0, 1.0), 8.0)
    for sv in p.sessions:                             # (G156, G180) both points are the animal's, it holds them
        sv.move_points([0, 1], 0)
        sv.segments[0].hold = True
    check("(setup) both points have a position on frame 0 in every camera",
          all(bool(sv.tracked[0, 0]) and bool(sv.tracked[0, 1]) for sv in p.sessions),
          [(sv.n_points, [bool(sv.tracked[0, q]) for q in range(sv.n_points)]) for sv in p.sessions])
    win._set_active_view(0)
    win._goto(0, force=True)
    win._passes = {"queue": [], "groups": [], "view": 0, "frame": 0, "step": False, "every": True,
                   "done": [{"lasts": {0: 30, 1: 30, 2: 30}, "fail": None, "user_stop": False, "group": 0}],
                   "snap": None, "msnaps": None, "seg_views": {0, 1}}
    built = {}
    for v in range(3):
        win._set_active_view(v)
        win._goto(0, force=True)
        built[v] = win._start_tracking(stop_after=30, only_pids=[1], quiet=True, build_only=True, segment=False)
    check("camera A's pass 2 gets camera A's silhouettes (not B's, not C's) and runs no segmenter",
          lambda: built[0] is not None and built[0]._stored is p.sessions[0].masks and built[0].animal is None)
    check("camera B's pass 2 gets camera B's silhouettes",
          lambda: built[1] is not None and built[1]._stored is p.sessions[1].masks)
    check("camera C has no saved silhouette: its pass 2 is not held to anything",
          lambda: built[2] is not None and getattr(built[2], "_stored", None) is None
          and not built[2].constrain, (built[2], getattr(built[2], "constrain", None), win.project.active))
    win._passes = None
    win._set_active_view(0)
    win._goto(0, force=True)
    alone = win._start_tracking(stop_after=30, only_pids=[1], quiet=True, build_only=True, segment=False)
    check("G180 any run of a held point is kept on the saved silhouettes (not only a two-pass run's second pass)",
          lambda: alone is not None and alone._stored is p.sessions[0].masks and alone.animal is None
          and alone.constrain == {1})
    p.sessions[0].segments[0].hold = False
    free = win._start_tracking(stop_after=30, only_pids=[1], quiet=True, build_only=True, segment=False)
    check("G180 ... and a point its animal does not hold gets none",
          lambda: free is not None and getattr(free, "_stored", None) is None)
    close(win)


ONLY = os.environ.get("I185_SECTIONS", "1234")        # a development aid: I185_SECTIONS=4 runs one section
if HAS_INPUT:
    if "1" in ONLY:
        section1()
    if "2" in ONLY:
        section2()
else:
    check("the worker takes the stored silhouettes (stored_masks)", False, "the old worker has no such input")
if "3" in ONLY:
    section3()
if "4" in ONLY:
    section4()

# ================================================================= the end
if FAILS:
    print(f"VERIFY_REVIEW_I185 FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("VERIFY_REVIEW_I185 PASSED")
