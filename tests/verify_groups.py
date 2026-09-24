"""Region groups vs the motivating failure case: three balls on a stick that
ROTATES, with the BACKGROUND swapped mid-video (the exact scenario that broke
plain point tracking for the user).

Builds a synthetic video with analytic ground truth: a rigid triad of colored
balls (red, blue, white) rotating about a translating center; at frame 200 the
background texture is replaced wholesale. Tracks every ball twice — as a plain
point AND as a region group — and requires the groups to survive with small
error. Also covers: group re-seed semantics (seed row exact), fit fallback
(members half-occluded), degenerate tiny-radius groups, and a group at the
frame edge (clamped member sampling)."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])
from kinetrace.tracker import PointSpec, TrackingWorker, sample_members, fit_group
from kinetrace.video_source import FrameCache

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
VID = os.path.join(SCRATCH, "stick.mp4")

# ---- build the rotating-stick video ----
W, H, T = 1280, 720, 400
R_BALL = 14
ARM = 110            # ball distance from stick center
rng = np.random.default_rng(2)


def make_bg(seed):
    r = np.random.default_rng(seed)
    bg = r.integers(30, 110, size=(H, W, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
    grid = np.zeros((H, W), np.uint8)
    grid[::40, :] = 35
    grid[:, ::40] = 35
    return cv2.add(bg, cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR))


bg_a, bg_b = make_bg(10), make_bg(99)   # background swaps at BG_SWAP
BG_SWAP = 200

t = np.arange(T, dtype=np.float64)
cx = W / 2 + 180 * np.sin(2 * np.pi * t / 500)          # stick center translates
cy = H / 2 + 90 * np.sin(2 * np.pi * t / 350 + 1.0)
theta = 2 * np.pi * t / 160                              # ~2.5 revolutions
# balls at arm offsets -1, 0, +1 along the rotating stick
offsets = np.array([-1.0, 0.0, 1.0])
gt = np.zeros((T, 3, 2))
gt[:, :, 0] = cx[:, None] + ARM * offsets[None, :] * np.cos(theta)[:, None]
gt[:, :, 1] = cy[:, None] + ARM * offsets[None, :] * np.sin(theta)[:, None]

BALL_COLORS = [(60, 60, 230), (230, 100, 60), (245, 245, 245)]  # red, blue, white (BGR)
vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
assert vw.isOpened()
for f in range(T):
    frame = (bg_a if f < BG_SWAP else bg_b).copy()
    x0, y0 = gt[f, 0]
    x2, y2 = gt[f, 2]
    cv2.line(frame, (round(x0), round(y0)), (round(x2), round(y2)), (70, 50, 40), 7,
             lineType=cv2.LINE_AA)
    for d in range(3):
        x, y = gt[f, d]
        cv2.circle(frame, (round(x), round(y)), R_BALL, BALL_COLORS[d], -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(frame, (round(x), round(y)), R_BALL, (25, 25, 25), 2,
                   lineType=cv2.LINE_AA)
        # off-center highlight gives each ball internal texture, like real shading
        cv2.circle(frame, (round(x - 4), round(y - 4)), 4, (255, 255, 255), -1,
                   lineType=cv2.LINE_AA)
    vw.write(frame)
vw.release()


def run(specs, n_frames=T, **kw):
    K = len(specs)
    tracks = np.full((n_frames, K, 2), np.nan, np.float32)
    conf = np.zeros((n_frames, K), np.float32)
    state = {"auto": None, "end": None}
    w = TrackingWorker(VID, 0, None, None, FrameCache(400 * 1024 * 1024),
                       n_frames, refine=True, specs=specs, **kw)

    def on_chunk(w0, tr, vi, cf, members, frames):
        tracks[w0:w0 + tr.shape[0]] = tr
        conf[w0:w0 + tr.shape[0]] = cf

    w.chunk_ready.connect(on_chunk)
    w.autopaused.connect(lambda f, pid: state.update(auto=(f, pid)))
    w.finished_ok.connect(lambda last, paused: state.update(end=(last, paused)))
    w.error.connect(lambda m: (print(m), sys.exit("worker error")))
    w.run()
    return tracks, conf, state


# ---- 1. rotation + background swap: groups on all three balls ----
specs = ([PointSpec(i, gt[0, i].astype(np.float32)) for i in range(3)]
         + [PointSpec(3 + i, gt[0, i].astype(np.float32), kind="group",
                      radius=float(R_BALL)) for i in range(3)])
tracks, conf, state = run(specs, autopause=False)  # measure the whole video
assert state["end"] == (T - 1, False), state

names = ["red", "blue", "white"]
print("rotating stick + background swap @200 (px mean / max):")
group_errs = []
for i in range(3):
    ep = np.linalg.norm(tracks[:, i] - gt[:, i], axis=1)
    eg = np.linalg.norm(tracks[:, 3 + i] - gt[:, i], axis=1)
    group_errs.append(eg)
    print(f"  {names[i]:5s}: point {np.nanmean(ep):5.2f} / {np.nanmax(ep):6.2f}   "
          f"group {np.nanmean(eg):5.2f} / {np.nanmax(eg):6.2f}")
for i, eg in enumerate(group_errs):
    assert np.isfinite(tracks[:, 3 + i]).all(), f"group {names[i]} lost coverage"
    assert np.nanmean(eg) < 5.0, f"group {names[i]} err too high: {np.nanmean(eg):.2f}"
# the background swap must not break the groups
post = slice(BG_SWAP, BG_SWAP + 100)
for i, eg in enumerate(group_errs):
    assert np.nanmean(eg[post]) < 6.0, \
        f"group {names[i]} broke at background swap: {np.nanmean(eg[post]):.2f} px"
print("rotation + background-swap survival OK")

# seed row must be the exact user click for groups too
assert np.allclose(tracks[0, 3:], gt[0], atol=1e-3), "group seed rows not exact"

# ---- 2. fit fallback: occlude half the members -> center must not jump ----
seed_c = np.array([300.0, 300.0], np.float32)
mem = sample_members(seed_c, 20.0, W, H)
cur = mem + np.array([5.0, -3.0], np.float32)       # clean translation
c, v, cf_ = fit_group(mem, seed_c, cur, np.full(len(mem), 0.9, np.float32),
                      np.ones(len(mem), bool), 20.0, (W, H))
assert np.allclose(c, seed_c + (5, -3), atol=0.5) and cf_ > 0.7
cur2 = cur.copy()
cur2[7:] = np.nan                                    # over half the members gone
c2, v2, cf2 = fit_group(mem, seed_c, cur2, np.full(len(mem), 0.9, np.float32),
                        np.ones(len(mem), bool), 20.0, (W, H))
assert np.isfinite(c2).all() and np.linalg.norm(c2 - (seed_c + (5, -3))) < 3.0, \
    "fallback center drifted"
assert cf2 < cf_, "losing members must reduce confidence"
none_valid = np.full_like(cur, np.nan)
c3, v3, cf3 = fit_group(mem, seed_c, none_valid, np.full(len(mem), 0.9, np.float32),
                        np.ones(len(mem), bool), 20.0, (W, H))
assert not np.isfinite(c3).any() and cf3 == 0.0, "all-invalid must yield NaN/0"
print("fit fallback OK")

# ---- 3. degenerate radii + frame-edge group ----
m_tiny = sample_members(np.array([50.0, 50.0], np.float32), 2.0, W, H)
assert np.isfinite(m_tiny).all() and (np.abs(m_tiny - (50, 50)) <= 2.01).all()
m_edge = sample_members(np.array([2.0, 718.0], np.float32), 30.0, W, H)
assert (m_edge[:, 0] >= 1).all() and (m_edge[:, 1] <= H - 2).all(), \
    "edge members must clamp in-frame"
rot = cv2.getRotationMatrix2D((300.0, 300.0), 30, 1.0)
cur_rot = (rot[:, :2] @ mem.T).T + rot[:, 2]        # pure 30-degree rotation
c4, _, cf4 = fit_group(mem, seed_c, cur_rot.astype(np.float32),
                       np.full(len(mem), 0.9, np.float32),
                       np.ones(len(mem), bool), 20.0, (W, H))
assert np.allclose(c4, seed_c, atol=0.5), f"rotation about center must keep center: {c4}"
print("degenerate/edge/rotation fit OK")

os.remove(VID)
print("GROUPS PASSED")
