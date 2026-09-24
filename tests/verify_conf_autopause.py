"""Auto-pause detector: must fire on real tracking loss, must NOT fire on
ordinary occlusion.

Video 1 (sabotage): two dots; at frame 150 dot A vanishes AND the background
texture is swapped — everything the model knew about A is gone. Dot B keeps
existing on the new background. Auto-pause must trigger, for A, near 150.

Video 2 (occlusion): a dot passes under an opaque bar for ~40 frames and
re-emerges. Occlusion drops *visibility*, not (much) confidence — the run
must complete with no auto-pause."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])
from kinetrace.tracker import CONF_PAUSE_THRESHOLD, PointSpec, TrackingWorker
from kinetrace.video_source import FrameCache

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
W, H = 640, 480


def make_bg(seed):
    r = np.random.default_rng(seed)
    bg = r.integers(30, 90, size=(H, W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(bg, (0, 0), 1.5)


def dot(frame, x, y, color):
    cv2.circle(frame, (round(x), round(y)), 9, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (round(x), round(y)), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)


# ---- 0. lost-point blanking survives the overlap rewrite (I118), bare worker ----
def blank_sim(gone_frames, T=264):   # 264: the last window ends on the last frame
    """Drive _check_autopause with the online windows (16 rows, 8 of them new)
    and a session that keeps the LAST write of each frame, as write_segment
    does. 'Gone' = in frame, confidence 0.05, not visible."""
    w = TrackingWorker("none.mp4", 0, np.array([[50, 50]], np.float32), [0],
                       FrameCache(1 << 20), T, autopause=False)
    w._frame_wh = (W, H)
    w._low_run, w._low_start, w._gone_run, w._gone_start = {0: 0}, {0: 0}, {0: 0}, {0: 0}
    w._back_run, w._lost, w._counted_until = {0: 0}, set(), -1
    sp = [PointSpec(0, np.array([50, 50], np.float32))]
    sess = np.full((T, 2), np.nan, np.float32)
    for e in range(15, T, 8):
        w0 = max(0, e - 15)
        L = e - w0 + 1
        tr = np.full((L, 1, 2), 50.0, np.float32)
        vi = np.ones((L, 1), bool)
        cf = np.ones((L, 1), np.float32)
        for i in range(L):
            if w0 + i in gone_frames:
                vi[i, 0] = False
                cf[i, 0] = 0.05
        w._check_autopause(w0, tr, vi, cf, sp, [0])
        sess[w0:w0 + L] = tr[:, 0]
    return sess


# gone 100..140 then back; an 8-frame dip at 60..67 is occlusion-length and never blanked.
# The loss is declared at 115; frames 100..103 were emitted a window earlier, too late to
# blank (the kept head of a dip), everything from 104 on must stay blank -- including the
# frames just before the recovery, which the next window re-emits (they came back once)
s1 = blank_sim(set(range(60, 68)) | set(range(100, 141)))
assert np.isfinite(s1[:100, 0]).all(), "a short dip must not be blanked"
kept = [f for f in range(104, 141) if np.isfinite(s1[f, 0])]
assert not kept, f"gone frames that got their coordinates back: {kept}"
assert np.isfinite(s1[141:, 0]).all(), "a recovered point must keep its data"
# a single spurious 'visible' frame (150) inside a lost stretch 100..200
s2 = blank_sim(set(range(100, 201)) - {150})
kept = [f for f in range(104, 201) if f != 150 and np.isfinite(s2[f, 0])]
assert not kept, f"gone frames around a one-frame flicker that carry coordinates: {kept}"
assert np.isfinite(s2[150, 0]) and np.isfinite(s2[201:, 0]).all()
print("lost-point blanking holds through overlap rewrites and a flicker OK")


def run(vid, specs, n_frames, autopause=True):
    K = len(specs)
    tracks = np.full((n_frames, K, 2), np.nan, np.float32)
    conf = np.zeros((n_frames, K), np.float32)
    state = {"auto": None, "end": None}
    w = TrackingWorker(vid, 0, None, None, FrameCache(200 * 1024 * 1024),
                       n_frames, refine=True, specs=specs, autopause=autopause)
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: (
        tracks.__setitem__(slice(w0, w0 + len(tr)), tr),
        conf.__setitem__(slice(w0, w0 + len(tr)), cf)))
    w.autopaused.connect(lambda f, pid: state.update(auto=(f, pid)))
    w.finished_ok.connect(lambda last, paused: state.update(end=(last, paused)))
    w.error.connect(lambda m: (print(m), sys.exit("worker error")))
    w.run()
    return tracks, conf, state


# ---- 1. sabotage: dot A vanishes + background swap at frame 150 ----
VID1 = os.path.join(SCRATCH, "sabotage.mp4")
T1 = 300
SAB = 150
bg_a, bg_b = make_bg(3), make_bg(77)
ax = 200 + 60 * np.sin(np.arange(T1) / 40)
ay = 200 + 40 * np.cos(np.arange(T1) / 55)
bx = 460 + 50 * np.sin(np.arange(T1) / 47)
by = 320 + 45 * np.cos(np.arange(T1) / 38)
vw = cv2.VideoWriter(VID1, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in range(T1):
    frame = (bg_a if f < SAB else bg_b).copy()
    if f < SAB:
        dot(frame, ax[f], ay[f], (60, 60, 230))
    dot(frame, bx[f], by[f], (60, 230, 60))
    vw.write(frame)
vw.release()

specs = [PointSpec(0, np.array([ax[0], ay[0]], np.float32)),
         PointSpec(1, np.array([bx[0], by[0]], np.float32))]
tracks, conf, state = run(VID1, specs, T1, autopause=True)
in_a = np.isfinite(tracks[:, 0, 0])
print(f"sabotage: A conf median pre-150 {np.median(conf[20:SAB, 0][in_a[20:SAB]]):.3f}, "
      f"post-150 {np.median(conf[SAB:SAB + 40, 0][in_a[SAB:SAB + 40]] if in_a[SAB:SAB + 40].any() else [0]):.3f}")
print(f"sabotage: autopaused={state['auto']}, finished={state['end']}")
assert state["auto"] is not None, "auto-pause must fire when the target vanishes"
fail_frame, fail_pid = state["auto"]
assert fail_pid == 0, f"the vanished dot is point 0, not {fail_pid}"
assert SAB - 8 <= fail_frame <= SAB + 60, \
    f"pause should localize the failure near {SAB}, got {fail_frame}"
assert state["end"][1] is True, "auto-pause must finish as a paused run"
# without auto-pause the same run completes (flag off = old behavior)
_, conf2, state2 = run(VID1, specs, T1, autopause=False)
assert state2["auto"] is None and state2["end"] == (T1 - 1, False)
print(f"sabotage OK: fired at frame {fail_frame} for the vanished dot; "
      "flag-off run completed")

# ---- 2. occlusion: dot passes under an opaque bar, must NOT auto-pause ----
VID2 = os.path.join(SCRATCH, "occlusion.mp4")
T2 = 240
ox = np.linspace(60, 580, T2)              # steady left-to-right
oy = 240 + 30 * np.sin(np.arange(T2) / 30)
bg = make_bg(5)
BAR = (280, 140, 360, 340)                  # x0, y0, x1, y1 — the dot passes behind
vw = cv2.VideoWriter(VID2, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in range(T2):
    frame = bg.copy()
    dot(frame, ox[f], oy[f], (60, 60, 230))
    cv2.rectangle(frame, BAR[:2], BAR[2:], (90, 90, 100), -1)
    cv2.rectangle(frame, BAR[:2], BAR[2:], (140, 140, 150), 2)
    vw.write(frame)
vw.release()

occ = (ox > BAR[0] - 9) & (ox < BAR[2] + 9)
print(f"occlusion window: frames {np.nonzero(occ)[0][0]}–{np.nonzero(occ)[0][-1]} "
      f"({occ.sum()} frames)")
specs2 = [PointSpec(0, np.array([ox[0], oy[0]], np.float32))]
tracks3, conf3, state3 = run(VID2, specs2, T2, autopause=True)
occ_conf = conf3[occ, 0]
print(f"occlusion: conf during occlusion median {np.median(occ_conf):.3f} "
      f"(min {occ_conf.min():.3f}), threshold {CONF_PAUSE_THRESHOLD}")
assert state3["auto"] is None, \
    f"auto-pause fired on ordinary occlusion at {state3['auto']} — must not"
assert state3["end"] == (T2 - 1, False)
# and the point must still be tracked correctly after re-emerging
post = np.nonzero(~occ)[0]
post = post[post > np.nonzero(occ)[0][-1]]
err_post = np.abs(tracks3[post, 0, 0] - ox[post])
print(f"occlusion: |x err| after re-emerge mean {np.nanmean(err_post):.2f} px")
assert np.nanmean(err_post) < 6.0, "track must survive the occlusion"

os.remove(VID1)
os.remove(VID2)
print("CONF/AUTOPAUSE PASSED")
