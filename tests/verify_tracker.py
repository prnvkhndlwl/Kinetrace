"""Headless validation of tracker.py against synthetic ground truth."""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])

from kinetrace.tracker import TrackingWorker, get_model, pick_device
from kinetrace.video_source import FrameCache

VID = os.path.join(ROOT, r"test600.mp4")
GT = np.load(VID + ".gt.npz")["gt"]  # (600, 4, 2)
T, D = GT.shape[:2]

print("device:", pick_device()[1])


def run_segment(start, seeds, refine, n_frames=600, pause_after_emits=None, allow_error=False):
    """Run a worker synchronously; return (tracks (T,K,2) NaN-filled, vis, events)."""
    K = len(seeds)
    tracks = np.full((n_frames, K, 2), np.nan, np.float32)
    vis = np.zeros((n_frames, K), bool)
    conf = np.zeros((n_frames, K), np.float32)
    events = {"emits": 0, "finished": None, "error": None, "rows": [], "autopaused": None,
              "conf": conf}
    w = TrackingWorker(VID, start, np.array(seeds, np.float32), list(range(K)),
                       FrameCache(200 * 1024 * 1024), n_frames, refine=refine)

    def on_chunk(w0, tr, vi, cf, members, frames):
        L = tr.shape[0]
        tracks[w0:w0 + L] = tr
        vis[w0:w0 + L] = vi
        conf[w0:w0 + L] = cf
        assert np.isfinite(cf).all() and (cf >= 0).all() and (cf <= 1).all(), \
            "confidence must be finite in [0, 1]"
        assert members == {}, "plain-point run must not emit group members"
        events["emits"] += 1
        events["rows"].append((w0, L))
        if pause_after_emits and events["emits"] >= pause_after_emits:
            w.request_pause()

    w.chunk_ready.connect(on_chunk)
    w.autopaused.connect(lambda f, pid: events.update(autopaused=(f, pid)))
    w.finished_ok.connect(lambda last, paused: events.update(finished=(last, paused)))
    w.error.connect(lambda msg: events.update(error=msg))
    w.run()  # synchronous: direct-connection signals fire inline
    if events["error"] and not allow_error:
        print(events["error"])
        sys.exit(f"worker error (start={start}, refine={refine})")
    return tracks, vis, events


def report(name, tracks, start, end):
    errs = []
    for d in range(tracks.shape[1]):
        seg = tracks[start:end, d]
        ok = np.isfinite(seg[:, 0])
        err = np.linalg.norm(seg[ok] - GT[start:end][ok, d], axis=1)
        errs.append((err.mean(), err.max(), ok.sum()))
    mean_all = np.mean([e[0] for e in errs])
    print(f"{name}: per-point mean {[f'{e[0]:.2f}' for e in errs]} px, "
          f"max {max(e[1] for e in errs):.2f} px, frames {errs[0][2]}")
    return mean_all


# ---- 1. full run from frame 0, raw vs refined ----
seeds0 = GT[0]  # exact dot centers at frame 0
tr_raw, vis_raw, ev = run_segment(0, seeds0, refine=False)
assert ev["finished"] == (599, False), ev["finished"]
assert ev["autopaused"] is None, "auto-pause must not trigger on a clean video"
# clean tracking should be confidently above the pause threshold nearly always
frac_conf = (ev["conf"] > 0.35).mean()
print(f"confidence: {frac_conf:.1%} of cells above pause threshold, "
      f"median {np.median(ev['conf']):.3f}")
assert frac_conf > 0.95, f"confidence suspiciously low on clean video: {frac_conf:.1%}"
# coverage: every frame 0..599 has data
assert np.isfinite(tr_raw[:, :, 0]).all(), "coverage gap in full run"
# seed row must be exact
assert np.allclose(tr_raw[0], seeds0, atol=1e-4), "seed row not exact"
m_raw = report("raw   full", tr_raw, 0, 600)

tr_ref, vis_ref, _ = run_segment(0, seeds0, refine=True)
m_ref = report("refine full", tr_ref, 0, 600)
assert m_raw < 6.0, f"raw mean error too high: {m_raw:.2f}"
assert m_ref < 6.0, f"refined mean error too high: {m_ref:.2f}"
print(f"refinement delta: {m_raw - m_ref:+.3f} px (positive = refinement helps)")

# ---- 2. re-seed mid-video (correction resume semantics) ----
tr2, _, ev2 = run_segment(250, GT[250], refine=True)
assert ev2["finished"][0] == 599
assert np.isfinite(tr2[250:, :, 0]).all() and np.isnan(tr2[:250]).all()
report("resume@250", tr2, 250, 600)

# ---- 3. short tails near EOF (592 = exactly one 8-frame step, the init-only edge) ----
for start in (590, 595, 599, 584, 592):
    trS, _, evS = run_segment(start, GT[start], refine=True)
    n_seg = 600 - start
    assert np.isfinite(trS[start:600, :, 0]).all(), f"gap in short segment start={start}"
    assert evS["finished"][0] == 599, evS["finished"]
    m = report(f"tail@{start} ({n_seg}f)", trS, start, 600)
    assert m < 8.0

# ---- 4. pause path ----
trP, _, evP = run_segment(0, seeds0, refine=False, pause_after_emits=3)
last, paused = evP["finished"]
assert paused is True and 16 <= last < 599, evP["finished"]
assert np.isfinite(trP[: last + 1, :, 0]).all() and np.isnan(trP[last + 1 + 8:]).all()
print(f"pause OK: stopped at frame {last} after 3 emits")

# ---- 5. a frame that cannot be decoded mid-video is NOT the end of the video (I40) ----
from kinetrace import video_source as vs  # noqa: E402

_read_next = vs.VideoSource.read_next


def _fails_at_300(self):
    """Sequential decode refuses frame 300 (a damaged packet); seeking past it works."""
    if self._pos == 300:
        return None
    return _read_next(self)


vs.VideoSource.read_next = _fails_at_300
try:
    trD, _, evD = run_segment(250, GT[250], refine=False, allow_error=True)
finally:
    vs.VideoSource.read_next = _read_next
print(f"decode failure at 300: finished={evD['finished']}, error={str(evD['error'])[:90]!r}...")
assert evD["finished"] is None, "a damaged frame must not be reported as a completed run"
assert evD["error"] and "Frame 300" in evD["error"] and "could not be decoded" in evD["error"], evD["error"]
assert "Traceback" not in evD["error"], "the user gets a sentence, not a traceback"
assert np.isfinite(trD[250:300, :, 0]).all(), "everything before the damaged frame must be emitted"
assert np.isnan(trD[300:]).all()
# the header promising more frames than the file holds is still the END of the video
trE, _, evE = run_segment(580, GT[580], refine=False, n_frames=610)
assert evE["finished"] == (599, False) and not evE["error"], evE
assert np.isfinite(trE[580:600, :, 0]).all()
# the segment-only loop (no point tracker) tells the two apart the same way
wA = TrackingWorker(VID, 250, GT[250][:1], [0], FrameCache(64 * 1024 * 1024), 600)
wA._frame_wh = (640, 480)
vs.VideoSource.read_next = _fails_at_300
try:
    srcA = vs.VideoSource(VID, FrameCache(64 * 1024 * 1024))
    lastA = wA._run_animal_only(srcA, 250)
    srcA.close()
finally:
    vs.VideoSource.read_next = _read_next
assert wA.decode_failed_at == 300 and lastA == 299, (wA.decode_failed_at, lastA)
wB = TrackingWorker(VID, 580, GT[580][:1], [0], FrameCache(64 * 1024 * 1024), 610)
wB._frame_wh = (640, 480)
srcB = vs.VideoSource(VID, FrameCache(64 * 1024 * 1024))
lastB = wB._run_animal_only(srcB, 580)
srcB.close()
assert wB.decode_failed_at is None and lastB == 599, (wB.decode_failed_at, lastB)
print("decode failure reported distinctly; a short header is still the end OK")

# ---- 6. points + balls without a segment: a restart that drops every point hands the
# balls on instead of ending the run short of the video's end (I120) ----
import kinetrace.balls as balls_mod  # noqa: E402
import kinetrace.tracker as trk  # noqa: E402
from kinetrace.tracker import BallSpec, PointSpec  # noqa: E402

calls = {}
saved = (trk.get_model, trk.get_segmenter, balls_mod.BallTracker)
trk.get_model = lambda: (None, "cpu")               # no model: the segment itself is stubbed
trk.get_segmenter = lambda backend: None
balls_mod.BallTracker = lambda seg, wh: object()
try:
    wH = TrackingWorker(VID, 0, None, None, FrameCache(64 * 1024 * 1024), 600, autopause=False,
                        specs=[PointSpec(0, GT[0, 0].astype(np.float32))],
                        balls=[BallSpec(9, {}, seed=GT[0, 1], backend="stub")])
    # the first segment ends in an ROI restart at which the only point has no valid position
    # (the run's loops are step generators since I141: the stubs return at once)
    def _segment_stub(src, *a):
        return "restart", 40, [None]
        yield                                           # noqa: unreachable -- makes it a generator

    def _animal_stub(src, start):
        calls["start"] = start
        return 599
        yield                                           # noqa: unreachable -- makes it a generator

    wH._segment_steps = _segment_stub
    wH._animal_only_steps = _animal_stub
    wH.finished_ok.connect(lambda last_, paused_: calls.__setitem__("finished", (last_, paused_)))
    wH.error.connect(lambda m: calls.__setitem__("error", m))
    wH.run()
finally:
    trk.get_model, trk.get_segmenter, balls_mod.BallTracker = saved
assert "error" not in calls, calls.get("error")
assert calls.get("start") == 41 and calls.get("finished") == (599, False), calls
print("balls go on after every point was dropped at a restart OK")

# ---- 7. appearance re-anchor snaps to the CLICK, not to its rounded pixel (I121) ----
from collections import deque  # noqa: E402


def blob(cx, cy, h=400, w=600, s=3.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 40 + 180 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    return np.clip(g + np.random.default_rng(0).normal(0, 1.0, g.shape), 0, 255).astype(np.uint8)


seed = np.array([300.4, 200.3], np.float32)          # the blob's true centre, clicked exactly
wR = TrackingWorker(VID, 0, seed[None], [0], FrameCache(1 << 20), 10)
wR._gate = 2.0 * 3840 / 512                          # a 4K run's search radius
wR._do_refine = False                                # the anchor snap alone
wR._frame_wh = (600, 400)
wR._templates, wR._template_frac, wR._refined = {}, {}, {}
spR = PointSpec(0, seed.copy(), anchor=True)
g0, g1 = blob(*seed), blob(seed[0] + 20.0, seed[1] + 10.0)
wR._capture_templates(g0, [spR], [("point", 0)])
pred = np.array([[seed], [seed + [21.5, 9.0]], [seed + [1.5, -1.0]]], np.float32)   # model guesses ~1.5 px off
outR = wR._refine_window(0, pred, np.ones((3, 1), bool), deque([(0, g0), (1, g1), (2, g0)]), 0, {0: spR})
e_moved = outR[1, 0] - (seed + [20.0, 10.0])
e_same = outR[2, 0] - seed
print(f"re-anchor error: after a (+20, +10) move {e_moved.round(3)}, same picture {e_same.round(3)} px")
assert np.abs(e_moved).max() < 0.05 and np.abs(e_same).max() < 0.05, "re-anchor biased by the seed's sub-pixel part"
print("re-anchor keeps the click's sub-pixel position OK")

print("ALL TRACKER CHECKS PASSED")
