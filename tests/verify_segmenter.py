"""Segmentation layer verification: SAM 2.1 streaming sessions (bounded memory,
multi-object, mid-stream prompts) against a synthetic two-animal video with
exact ground-truth silhouettes, plus silhouette geometry (midline / tail tip /
feet) and MaskTrack persistence. GPU recommended (~1 min); model weights must be
cached in models/hf (first run downloads ~620 MB)."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from kinetrace import silhouette as S
from kinetrace.silhouette import extremity_roles, midline
from kinetrace.segmenter import (FrameMasks, MaskTrack, Prompt, SegSession, get_segmenter,
                                     model_is_cached, summarize_mask, working_size, MEMORY_WINDOW)

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
W, H, T = 1280, 720, 160
VID = os.path.join(OUT, "synth_animals.mp4")


# ------------------------------------------------------------ synthetic animals
def pose(t: float, k: int):
    """Body centre + heading of animal k at time t (two non-overlapping orbits)."""
    cx0 = W * (0.28 if k == 0 else 0.72)
    ph = k * 2.1
    cx = cx0 + 0.13 * W * np.cos(0.025 * t + ph)
    cy = H * 0.5 + 0.24 * H * np.sin(0.02 * t + ph)
    cx1 = cx0 + 0.13 * W * np.cos(0.025 * (t + 1) + ph)
    cy1 = H * 0.5 + 0.24 * H * np.sin(0.02 * (t + 1) + ph)
    heading = np.arctan2(cy1 - cy, cx1 - cx)
    return np.array([cx, cy]), heading


def draw_animal(canvas, mask, t: int, k: int):
    """Ellipse body, round head, tapering undulating tail, four swinging legs.
    Returns ground truth: head, tail tip, feet (4), body centre."""
    c, h = pose(t, k)
    u = np.array([np.cos(h), np.sin(h)])
    n = np.array([-np.sin(h), np.cos(h)])
    col = (38, 44, 58)

    def P(v):
        return (int(round(v[0])), int(round(v[1])))

    for img, colour in ((canvas, col), (mask, 255)):
        cv2.ellipse(img, P(c), (60, 26), float(np.degrees(h)), 0, 360, colour, -1, cv2.LINE_AA)
        cv2.circle(img, P(c + 62 * u), 18, colour, -1, cv2.LINE_AA)
    head = c + 66 * u
    tail_pts = []
    for i in range(25):
        amp = 3.0 + 0.9 * i
        p = c - (50 + 6 * i) * u + amp * np.sin(0.25 * t + 0.35 * i + k) * n
        tail_pts.append(p)
    for i in range(24):
        th = int(round(9 - 7 * i / 24))
        for img, colour in ((canvas, col), (mask, 255)):
            cv2.line(img, P(tail_pts[i]), P(tail_pts[i + 1]), colour, max(2, th), cv2.LINE_AA)
    feet = []
    for si, side in enumerate((-1, 1)):
        for fi, fore in enumerate((1, -1)):
            attach = c + fore * 25 * u + side * 20 * n
            swing = 16 * np.sin(0.3 * t + 1.6 * fi + 0.8 * si + k)
            foot = attach + side * 38 * n + swing * u
            for img, colour in ((canvas, col), (mask, 255)):
                cv2.line(img, P(attach), P(foot), colour, 7, cv2.LINE_AA)
                cv2.circle(img, P(foot), 5, colour, -1, cv2.LINE_AA)
            feet.append(foot)
    return {"head": head, "tip": tail_pts[-1], "feet": np.array(feet), "centre": c}


def gt_frame(t: int):
    """(bgr frame, per-animal dict of gt + bool mask)."""
    rng = np.random.default_rng(7)
    bg = rng.integers(60, 140, (H // 8, W // 8, 3), np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    bg[..., 1] = np.clip(bg[..., 1].astype(int) + 40, 0, 255)  # greenish, textured
    canvas = bg.copy()
    out = {}
    for k in (0, 1):
        m = np.zeros((H, W), np.uint8)
        gt = draw_animal(canvas, m, t, k)
        gt["mask"] = m > 127
        out[k] = gt
    return canvas, out


def build_video():
    vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for t in range(T):
        frame, _ = gt_frame(t)
        vw.write(frame)
    vw.release()


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


# ------------------------------------------------------- 1. silhouette geometry
t0 = time.perf_counter()
tip_err, feet_hits, lengths = [], [], []
for t in range(0, T, 8):
    _, gts = gt_frame(t)
    for k in (0, 1):
        g = gts[k]
        ml = midline(g["mask"], anchor=tuple(g["head"]))
        assert ml is not None, f"no midline at t={t} k={k}"
        tip_err.append(np.linalg.norm(ml.tip - g["tip"]))
        lengths.append(ml.length)
        d = np.linalg.norm(ml.extremities[:, None, :] - g["feet"][None], axis=2) if len(ml.extremities) else np.full((1, 4), 1e9)
        feet_hits.append(int((d.min(axis=0) < 10).sum()))
        s = ml.sample([0.0, 0.5, 1.0])
        assert np.allclose(s[0], ml.head) and np.allclose(s[-1], ml.tip)

# without an anchor (no head point yet) a legless body+tail must still be
# oriented thick-end-first; with legs the longest path may legitimately start
# at a foot, which is why the app anchors on the tracked head point
fish = np.zeros((400, 600), np.uint8)
cv2.ellipse(fish, (420, 200), (70, 32), 0, 0, 360, 255, -1)
cv2.circle(fish, (496, 200), 22, 255, -1)
for i in range(30):
    cv2.line(fish, (350 - 8 * i, 200 + int(25 * np.sin(i / 4))), (342 - 8 * i, 200 + int(25 * np.sin((i + 1) / 4))),
             255, max(2, 10 - i // 3))
mf = midline(fish > 0)
assert mf is not None and np.linalg.norm(mf.head - (518, 200)) < 25, mf.head
assert np.linalg.norm(mf.tip - (102, 200 + 25 * np.sin(30 / 4))) < 25, mf.tip
print(f"unanchored fish: head {mf.head.round()} tip {mf.tip.round()} length {mf.length:.0f} px")
dt_geo = (time.perf_counter() - t0) / (2 * len(range(0, T, 8)))
print(f"silhouette on GT masks: tail-tip error mean {np.mean(tip_err):.1f} px max {np.max(tip_err):.1f} px, "
      f"feet found (of 4) mean {np.mean(feet_hits):.2f}, midline length {np.mean(lengths):.0f} px, {dt_geo*1000:.0f} ms/mask")
assert np.mean(tip_err) < 6 and np.max(tip_err) < 12, "midline does not reach the tail tip"
assert np.mean(feet_hits) >= 3.0, "extremities miss the feet"

# a 4K-sized blob must be handled through the analysis-scale path quickly
big = np.zeros((2160, 3840), np.uint8)
cv2.ellipse(big, (1900, 1000), (500, 180), 20, 0, 360, 255, -1)
cv2.line(big, (1450, 900), (700, 1500), 255, 30)
t0 = time.perf_counter(); mlb = midline(big > 0, anchor=(2350, 1150)); dt_big = time.perf_counter() - t0
assert mlb is not None and mlb.scale < 1.0 and np.linalg.norm(mlb.tip - (700, 1500)) < 40, mlb.tip
print(f"4K silhouette: scale {mlb.scale:.2f}, {dt_big*1000:.0f} ms, tip err {np.linalg.norm(mlb.tip - (700, 1500)):.0f} px")
assert dt_big < 1.0
assert midline(np.zeros((50, 50), bool)) is None

# ------------------------------------------- 1b. extremity sides and roles (I54, I55)
# gt_frame's feet: [0] side -1 fore, [1] side -1 hind, [2] side +1 fore, [3] side +1 hind, and
# +n is the heading turned clockwise on screen = the animal's RIGHT seen from above (dorsal)
DORSAL = ["FL", "HL", "FR", "HR"]
good = tot = 0
for t in range(0, T, 4):
    _, gts = gt_frame(t)
    for k in (0, 1):
        roles = extremity_roles(midline(gts[k]["mask"], anchor=tuple(gts[k]["head"])))
        for j, nm in enumerate(DORSAL):
            if nm in roles:
                tot += 1
                d = np.linalg.norm(gts[k]["feet"] - roles[nm], axis=1)
                good += int(np.argmin(d) == j and d[j] < 12)
print(f"extremity roles (dorsal convention): {good}/{tot} on the right foot")
assert tot >= 250 and good >= 0.95 * tot, "ext:FL/FR/HL/HR name the wrong foot (I54)"
for flip in (False, True):          # one foot on the animal's right, head up and head down
    m = np.zeros((600, 400), np.uint8)
    cv2.ellipse(m, (200, 80), (18, 30), 0, 0, 360, 1, -1)
    cv2.ellipse(m, (200, 200), (30, 100), 0, 0, 360, 1, -1)
    cv2.line(m, (200, 290), (200, 560), 1, 10)
    cv2.line(m, (225, 150), (300, 130), 1, 8)
    anc = (200.0, 52.0)
    if flip:
        m, anc = m[::-1, ::-1].copy(), (399 - anc[0], 599 - anc[1])
    roles = extremity_roles(midline(m, anchor=anc))
    assert set(roles) >= {"R", "FR"} and not {"L", "FL", "HL"} & set(roles), (flip, sorted(roles))


def lizard(limb):
    """Dorsal lizard, head up (the animal's right = screen right), 673-px body, four limbs."""
    m = np.zeros((1000, 900), np.uint8)
    cv2.ellipse(m, (450, 100), (22, 40), 0, 0, 360, 1, -1)
    cv2.ellipse(m, (450, 250), (38, 130), 0, 0, 360, 1, -1)
    cv2.line(m, (450, 370), (450, 730), 1, 12)
    feet = {}
    for side, sx in (("R", +1), ("L", -1)):
        for fh, y in (("F", 175), ("H", 335)):
            b = (450 + sx * (30 + limb), y - 25 if fh == "F" else y + 25)
            cv2.line(m, (450 + sx * 30, y), b, 1, 9)
            cv2.circle(m, b, 7, 1, -1)
            feet[fh + side] = np.array(b, float)
    return m, feet


for limb in (55, 80, 105):          # 8, 12 and 16 % of the body: long limbs kept 2-3 points each (I55)
    m, feet = lizard(limb)
    ml = midline(m, anchor=(450.0, 62.0))
    roles = extremity_roles(ml)
    errs = {k: float(np.linalg.norm(roles[k] - feet[k])) if k in roles else 1e9 for k in feet}
    print(f"lizard, limbs {limb} px: {len(ml.extremities)} extremities, worst role error {max(errs.values()):.0f} px")
    assert len(ml.extremities) == 4 and max(errs.values()) < 10, (limb, len(ml.extremities), errs)

# ------------------------- 1c. downscaled midline convention (I57), specks by the snout (I63)
def blob(dx, dy):
    m = np.zeros((576, 1024), np.uint8)
    cv2.ellipse(m, (330 + dx, 280 + dy), (230, 60), 0, 0, 360, 1, -1)
    cv2.fillPoly(m, [np.array([[520 + dx, 262 + dy], [960 + dx, 278 + dy], [960 + dx, 282 + dy],
                               [520 + dx, 298 + dy]], np.int32)], 1)
    return m


errs, keep_nodes = [], S.MAX_NODES
try:
    for dx, dy in [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (3, 3), (5, 7), (11, -9)]:
        m, anc = blob(dx, dy), (101.0 + dx, 280.0 + dy)
        S.MAX_NODES = 30_000
        ml = midline(m, anchor=anc)                  # ~52k px: analysed downscaled
        S.MAX_NODES = 10 ** 7
        ref = midline(m, anchor=anc)                 # the full-resolution answer
        errs.append(ml.sample([0.25, 0.5, 0.75]) - ref.sample([0.25, 0.5, 0.75]))
finally:
    S.MAX_NODES = keep_nodes
mean_y = float(np.array(errs)[..., 1].mean())
print(f"downscaled midline vs full resolution, across the body: mean {mean_y:+.2f} px (was +0.91)")
assert abs(mean_y) < 0.25, "the downscaled midline is biased (pixel-centre convention, I57)"
m = np.zeros((600, 1000), np.uint8)
cv2.ellipse(m, (400, 300), (220, 50), 0, 0, 360, 1, -1)
cv2.line(m, (600, 300), (900, 300), 1, 8)
clean = midline(m, anchor=(182.0, 300.0))
for isl, anc in (("9-px", (151.0, 291.0)), ("29-px", (150.0, 297.0))):
    mm = m.copy()
    if isl == "9-px":
        cv2.rectangle(mm, (150, 290), (152, 292), 1, -1)
    else:
        cv2.circle(mm, (150, 297), 3, 1, -1)
    ml = midline(mm, anchor=anc)
    print(f"head anchor on a {isl} speck: midline length {ml.length if ml else 0:.0f} (clean {clean.length:.0f})")
    assert ml is not None and abs(ml.length - clean.length) < 0.05 * clean.length, "a speck took the midline (I63)"

# ----------------------------------------------------------- 2. MaskTrack I/O
_, gts = gt_frame(3)
mt = MaskTrack(T)
for f in (3, 10, 11):
    mt.set(f, gts[0]["mask"], score=0.5 * f)
assert mt.has(3) and not mt.has(4) and mt.area[3] == gts[0]["mask"].sum()
r = mt.rasterize(3, H, W)
assert iou(r, gts[0]["mask"]) > 0.97, iou(r, gts[0]["mask"])
arrs = mt.to_arrays("obj0_")
np.savez(os.path.join(OUT, "masktrack.npz"), **arrs)
back = MaskTrack.from_arrays("obj0_", np.load(os.path.join(OUT, "masktrack.npz")))
assert back.has(3) and back.has(11) and not back.has(4)
assert np.array_equal(back.bbox, mt.bbox) and iou(back.rasterize(3, H, W), r) > 0.999
assert np.isclose(back.score[10], 5.0)
mt.clear(10, 11)
assert not mt.has(10) and not mt.has(11) and mt.has(3)
print("MaskTrack round-trip OK")

# the native size survives the undo snapshot, and the seed rebuilt from the stored outline
# at WORKING size lands on the mask (I53: after Ctrl+Z it was empty or elsewhere at 4K)
from kinetrace.session import TrackingSession  # noqa: E402
NW, NH = 3840, 2160
ww4, wh4 = working_size(NW, NH)
s4 = TrackingSession("none.mp4", 50, 25.0, NW, NH)
s4.ensure_animal()
right = np.zeros((wh4, ww4), np.uint8)
cv2.ellipse(right, (700, 300), (120, 40), 15, 0, 360, 1, -1)       # right of native x = 1024
topleft = np.zeros((wh4, ww4), np.uint8)
cv2.ellipse(topleft, (200, 120), (60, 20), 0, 0, 360, 1, -1)
s4.write_mask(10, right.astype(bool), NW / ww4, 6.0)
s4.write_mask(11, topleft.astype(bool), NW / ww4, 6.0)
s4.restore(s4.snapshot())                                          # what Ctrl+Z does
assert (s4.masks.native_w, s4.masks.native_h) == (NW, NH), "undo lost the silhouette's native size (I53)"
for f, truth in ((10, right), (11, topleft)):
    seed = s4.masks.rasterize(f, wh4, ww4)                         # what a resumed run seeds SAM with
    ys, xs = np.nonzero(seed)
    ty, tx = np.nonzero(truth)
    off = np.hypot(xs.mean() - tx.mean(), ys.mean() - ty.mean())
    print(f"seed after undo, frame {f}: IoU {iou(seed, truth > 0):.3f}, centroid off by {off:.2f} working px")
    assert iou(seed, truth > 0) > 0.95 and off < 0.2
assert MaskTrack(5, NW, NH).copy().native_w == NW

# per-axis working scale: 2704x1520 -> 1024x576 rounds y differently from x (I61)
NW, NH = 2704, 1520
ww2, wh2 = working_size(NW, NH)
fm2 = FrameMasks(0, [1], np.zeros((1, wh2, ww2), bool), np.ones(1, np.float32), (ww2, wh2), (NW, NH))
sx2, sy2 = fm2.scale_xy
assert abs(sx2 - NW / ww2) < 1e-9 and abs(sy2 - NH / wh2) < 1e-9 and sx2 != sy2
low = np.zeros((wh2, ww2), bool)
low[560:572, 400:430] = True                                       # near the bottom, where the drift peaks
d2 = summarize_mask(low, fm2.scale_xy, 5.0)
ys, xs = np.nonzero(low)
want = ((xs.mean() + 0.5) * sx2 - 0.5, (ys.mean() + 0.5) * sy2 - 0.5)
print(f"2704x1520 centroid y {d2['centroid'][1]:.3f} (per-axis truth {want[1]:.3f}; the x scale gave "
      f"{(ys.mean() + 0.5) * sx2 - 0.5:.3f})")
assert np.allclose(d2["centroid"], want, atol=0.05)
assert d2["bbox"][3] == round(572 * sy2) - 1 and d2["bbox"][2] == round(430 * sx2) - 1
ses = SegSession.__new__(SegSession)                               # the prompt mapping, no model needed
ses.native_size, ses.work = (NW, NH), (ww2, wh2)
assert np.allclose(ses._to_work((1000.0, 1500.0)), (1000.0 * ww2 / NW, 1500.0 * wh2 / NH))
print("native size through undo + per-axis scale OK")

# ------------------------------------------- 2b. skeleton specs are checked (I62)
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from kinetrace import skeletons as K  # noqa: E402

for spec in ("tip", "centroid", "midline:0", "midline:0.25", "midline:1", "ext:FL", "ext:R"):
    assert K.validate_spec(spec)[0], spec
for spec, word in (("midline:50", "midline:0.5"), ("ext:LF", "extremity"), ("midline:abc", "number"),
                   ("tail", "rule"), ("midline:-0.1", "number")):
    ok, why = K.validate_spec(spec)
    assert not ok and word in why, (spec, why)
for t in K.BUILTIN:
    assert not K.validate_template(t)[1], t["name"]
tmp = Path(tempfile.mkdtemp(prefix="kskel_", dir=OUT))
(tmp / "bad_json.json").write_text('{"name": "broken", "landmarks": ["a", "b"],}', encoding="utf-8")
(tmp / "bad_spec.json").write_text(json.dumps(
    {"name": "fish", "landmarks": ["snout", "body_mid", "foot", "tail_tip"],
     "derived": {"body_mid": "midline:50", "foot": "ext:LF", "tail_tip": "tip"},
     "bones": [["snout", "body_mid"], ["body_mid", "tail_tip", "x"], "snout"]}), encoding="utf-8")
keep_dir, K.SKELETON_DIR = K.SKELETON_DIR, tmp
try:
    problems = []
    user = K.load_user_templates(problems)
finally:
    K.SKELETON_DIR = keep_dir
    import shutil  # noqa: E402
    shutil.rmtree(tmp, ignore_errors=True)
for p in problems:
    print("  skeleton file problem: " + p)
assert any("bad_json.json" in p for p in problems), "an unreadable skeleton file must be reported"
assert any("midline:0.5" in p for p in problems) and any("ext:LF" in p for p in problems)
assert sum("bone" in p for p in problems) == 2
fish = next(t for t in user if t["name"] == "fish")
assert fish["derived"] == {"tail_tip": "tip"} and fish["bones"] == [["snout", "body_mid"]]
sk = TrackingSession("none.mp4", 10, 25.0, 640, 480)
sk.apply_skeleton(fish)
sk.bones()                                                         # a 3-element bone raised here
ml = midline(lizard(60)[0], anchor=(450.0, 62.0))
assert np.isnan(ml.sample([50.0])).all() and np.isfinite(ml.sample([0.0, 0.5, 1.0])).all(), \
    "a fraction outside 0..1 must be no data, never the tail tip"
print("skeleton spec validation OK")

# --------------------------------------------- 3. SAM 2 streaming on synthetic
if not model_is_cached():
    print("SAM 2 weights not cached in models/hf — first run downloads ~620 MB")
build_video()
t0 = time.perf_counter()
seg = get_segmenter()
print(f"segmenter loaded on {seg.device} in {time.perf_counter()-t0:.1f}s")
import torch  # noqa: E402  (installed with the app; only for the memory guard)

cap = cv2.VideoCapture(VID)
sess = seg.new_session(0, (W, H))
ious = {0: [], 1: []}; cerr = {0: [], 1: []}; present = {0: [], 1: []}
mem_at = {}
t_model = 0.0
for t in range(T):
    ok, bgr = cap.read(); assert ok
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, gts = gt_frame(t)
    prompts = None
    if t == 0:   # animal 0: one click on the body, one on the tail, one negative on the background
        c0 = gts[0]["centre"]; tail0 = (gts[0]["tip"] + c0) / 2
        prompts = [Prompt(0, np.array([c0, tail0, [W * 0.5, H * 0.5]]), np.array([1, 1, 0]))]
    if t == 40:  # animal 1 joins mid-stream with a box prompt
        m1 = gts[1]["mask"]; ys, xs = np.nonzero(m1)
        prompts = [Prompt(1, box=(xs.min(), ys.min(), xs.max(), ys.max()))]
    t1 = time.perf_counter()
    fm = sess.step(rgb, t, prompts)
    t_model += time.perf_counter() - t1
    for k in fm.obj_ids:
        m = fm.native_mask(k); g = gts[k]   # masks come at working res; compare at native
        ious[k].append(iou(m, g["mask"]))
        present[k].append(fm.present(k))
        if m.any():
            ys, xs = np.nonzero(m)
            gy, gx = np.nonzero(g["mask"])
            cerr[k].append(np.linalg.norm([xs.mean() - gx.mean(), ys.mean() - gy.mean()]))
    if t in (100, T - 1):  # both objects' memory windows are full by frame 100
        mem_at[t] = torch.cuda.memory_allocated() if seg.device == "cuda" else 0
    n_pf, n_nc = sess.memory_footprint()
    assert n_pf <= 1 and n_nc <= MEMORY_WINDOW + 1, (t, n_pf, n_nc)
fps = T / t_model
print(f"streaming: {T} frames at {W}x{H} -> {fps:.1f} fps (model+post)")
for k in (0, 1):
    io = np.array(ious[k]); ce = np.array(cerr[k])
    print(f"  animal {k}: IoU mean {io.mean():.3f} min {io.min():.3f} | centroid err mean {ce.mean():.1f} px max {ce.max():.1f} px "
          f"| present {np.mean(present[k]):.0%} of {len(io)} frames")
    assert len(io) == (T if k == 0 else T - 40)
    assert io.mean() > 0.80 and io.min() > 0.55, f"animal {k} mask quality"
    assert ce.mean() < 4.0 and ce.max() < 12.0, f"animal {k} centroid"
    assert np.mean(present[k]) > 0.95
if seg.device == "cuda":
    growth = (mem_at[T - 1] - mem_at[100]) / 2 ** 20
    print(f"  VRAM growth frames 100->{T-1}: {growth:+.0f} MB (bounded session; unpruned would be ~{9*(T-101)} MB)")
    assert growth < 16, f"session memory not bounded: {growth:.0f} MB over {T-101} frames"

# silhouette landmarks on the *predicted* masks must still find the tail tip
tips = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
sess2 = seg.new_session(0, (W, H))
for t in range(24):
    ok, bgr = cap.read(); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, gts = gt_frame(t)
    pr = [Prompt(0, np.array([gts[0]["centre"]]), np.array([1]))] if t == 0 else None
    fm = sess2.step(rgb, t, pr)
    ml = midline(fm.native_mask(0), anchor=tuple(gts[0]["head"]))
    tips.append(np.linalg.norm(ml.tip - gts[0]["tip"]) if ml is not None else 1e9)
print(f"  tail tip from predicted masks: mean err {np.mean(tips):.1f} px, max {np.max(tips):.1f} px")
assert np.mean(tips) < 12, "tail tip lost on predicted masks"

print("ALL SEGMENTER CHECKS PASSED")
