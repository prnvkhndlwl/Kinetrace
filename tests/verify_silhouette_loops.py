"""Silhouette self-contact loops (I293-I296), on synthetic masks with a KNOWN true midline and
known feet, through the real `silhouette.midline` / `extremity_roles` and a bare TrackingWorker's
`_fill_derived` (no model, no GPU, no video decode; the verify_onbody_rules.py pattern).

A silhouette that touches itself can close a loop (tail end over the flank, a foot on an upturned
tail, a closed C-curl). The anchored midline then cuts through the contact, and the derived tail
tip / midline points / feet are wrong for as long as the loop lasts.

Part A measures every case and prints one row each: the largest enclosed hole (share of the area),
the midline length against the true one, the tail-tip and midline:0.5 / 0.75 errors, and which
foot roles land on the right foot. Cases (h)-(j) are foot-role cases recorded for a second work item;
nothing is asserted on them. The controls must be right (else the synthetic truth is wrong).

Part B runs 20 frames through the worker: 5 before the loop, 10 with it, 5 after. Every frame of
the loop must carry a LOW confidence on the derived tip / midline points, not only the first one.

Golden output (coordinates and confidences of every sequence) goes to tests/out/loops/ for the
byte-identity proof against main.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from kinetrace import silhouette as sil
from kinetrace import tracker as trk
from kinetrace.segmenter import summarize_mask
from kinetrace.silhouette import extremity_roles, midline
from kinetrace.tracker import AnimalSpec, DerivedSpec, PointSpec, TrackingWorker
from kinetrace.video_source import FrameCache

OUT = os.path.join(ROOT, "tests", "out", "loops")
os.makedirs(OUT, exist_ok=True)
FOOT_TOL = 30.0          # a role is on its foot when within this many px of the true foot tip
# the low-confidence bar on a loop frame: LOOP_CONF once it exists, else today's jump demotion
LOW = getattr(trk, "LOOP_CONF", trk.DERIVED_JUMP_CONF)
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


# ------------------------------------------------------------------ drawing with a known truth
def polyline(pts, step: float = 2.0) -> np.ndarray:
    """Dense, evenly spaced centreline through the given vertices."""
    p = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, arc[-1], max(2, int(arc[-1] // step) + 2))
    return np.stack([np.interp(want, arc, p[:, 0]), np.interp(want, arc, p[:, 1])], axis=1)


def stroke(m: np.ndarray, pts, t0: float, t1: float) -> np.ndarray:
    """Draw a limb / tail along `pts`, thickness t0 -> t1; returns its dense centreline."""
    xy = polyline(pts)
    th = np.linspace(t0, t1, len(xy))
    for i in range(len(xy) - 1):
        a = (int(round(xy[i, 0])), int(round(xy[i, 1])))
        b = (int(round(xy[i + 1, 0])), int(round(xy[i + 1, 1])))
        cv2.line(m, a, b, 1, max(1, int(round(th[i]))))
    return xy


def arc_of(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def at_frac(xy: np.ndarray, f: float) -> np.ndarray:
    a = arc_of(xy)
    return np.array([np.interp(f * a[-1], a, xy[:, 0]), np.interp(f * a[-1], a, xy[:, 1])])


# a lizard facing screen-left (head at x ~ 105), y down. Seen from above with the head pointing
# left, the animal's RIGHT is screen UP (verify_segmenter: the heading turned clockwise on screen)
HEAD = (115.0, 400.0)
LEGS = {"FR": [(220, 360), (200, 260), (240, 230)], "FL": [(220, 440), (200, 540), (240, 570)],
        "HR": [(400, 370), (440, 270), (480, 240)], "HL": [(400, 430), (440, 530), (480, 560)]}
STRAIGHT_TAIL = [(460, 400), (650, 420), (850, 380), (1000, 420)]


def lizard(tail, legs=LEGS, size=(800, 1100), extra=()):
    """mask, true centreline (anchor -> tail tip), true feet {role: xy}, named extra strokes."""
    m = np.zeros(size, np.uint8)
    cv2.ellipse(m, (300, 400), (160, 55), 0, 0, 360, 1, -1)
    cv2.circle(m, (150, 400), 45, 1, -1)
    feet = {}
    for role, pts in legs.items():
        stroke(m, pts, 18, 12)
        feet[role] = np.asarray(pts[-1], np.float64)
    tail_xy = stroke(m, tail, 40, 10)
    for pts, t0, t1 in extra:
        stroke(m, pts, t0, t1)
    centre = np.vstack([polyline([HEAD, (460.0, 400.0)]), tail_xy[1:]])
    return m, centre, feet, tail_xy


def holes(m: np.ndarray) -> list[int]:
    """Areas of the background regions the silhouette encloses (4-connected background)."""
    n, _, st, _ = cv2.connectedComponentsWithStats((m == 0).astype(np.uint8), connectivity=4)
    H, W = m.shape
    return sorted((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                   if st[i, 0] > 0 and st[i, 1] > 0 and st[i, 0] + st[i, 2] < W and st[i, 1] + st[i, 3] < H),
                  reverse=True)


def label_of(p, feet: dict, tail_xy: np.ndarray, blobs=()) -> str:
    """What a role's point really sits on: a foot's name, 'tail', 'blob' or '?'."""
    best, name = FOOT_TOL, "?"
    for role, q in feet.items():
        d = float(np.linalg.norm(p - q))
        if d < best:
            best, name = d, role
    if name == "?" and len(tail_xy) and float(np.min(np.linalg.norm(tail_xy - p, axis=1))) < 25:
        name = "tail"
    for b in blobs:
        if name == "?" and float(np.linalg.norm(p - np.asarray(b))) < 40:
            name = "blob"
    return name


# --------------------------------------------------------------------------- Part A: geometry
CASES = {}
# controls: nothing touching / a near miss / an open C (the midline must be right on these)
CASES["base lizard (control)"] = dict(build=lambda: lizard(STRAIGHT_TAIL), control=True)
NEAR = [(460, 400), (570, 470), (580, 620), (470, 690), (340, 640), (310, 505)]
CASES["near miss, curled tail (control)"] = dict(build=lambda: lizard(NEAR, legs={}), control=True)
OVER = [(460, 400), (570, 470), (580, 620), (470, 690), (340, 640), (300, 470), (290, 330), (300, 240)]
CASES["(a) tail over the flank, tip out"] = dict(build=lambda: lizard(OVER), loop=True)
CASES["(a') same, no legs"] = dict(build=lambda: lizard(OVER, legs={}), loop=True)
UPTAIL = [(460, 400), (590, 390), (640, 310), (590, 240), (520, 230), (470, 250)]
HR_ON_TAIL = dict(LEGS, HR=[(400, 370), (430, 300), (470, 262)])
CASES["(b) hind foot on an upturned tail"] = dict(build=lambda: lizard(UPTAIL, legs=HR_ON_TAIL), loop=True)


def fish(turn: float, size=(800, 800)):
    m = np.zeros(size, np.uint8)
    a = np.linspace(0.0, turn * np.pi, 120)
    pts = np.stack([400 + 200 * np.cos(a), 400 + 200 * np.sin(a)], axis=1)
    xy = stroke(m, pts, 70, 14)
    return m, xy, {}, xy[len(xy) // 3:]


CASES["(c) closed C-curl fish"] = dict(build=lambda: fish(1.95), loop=True, anchor="start")
CASES["open C fish (control)"] = dict(build=lambda: fish(1.6), control=True, anchor="start")
HAIRPIN = [(460, 400), (600, 410), (640, 440), (600, 470), (420, 470)]
CASES["(d) tail folded tight along the flank"] = dict(build=lambda: lizard(HAIRPIN, legs={}), limit=True)
FOOT_FLANK = [((400, 370), (450, 280), (360, 300), (330, 350)), 18, 12]
CASES["(e) hind foot on its own flank"] = dict(
    build=lambda: lizard(STRAIGHT_TAIL, legs={k: v for k, v in LEGS.items() if k != "HR"},
                         extra=[FOOT_FLANK]), control=True)
FOOT_TAIL = [((400, 430), (470, 520), (600, 470), (640, 425)), 18, 12]
CASES["(f) hind foot on a straight tail"] = dict(
    build=lambda: lizard(STRAIGHT_TAIL, legs={k: v for k, v in LEGS.items() if k != "HL"},
                         extra=[FOOT_TAIL]), control=True)


def specks():
    m, c, f, t = lizard(STRAIGHT_TAIL)
    for p in ((300, 400), (250, 390), (350, 410)):
        cv2.circle(m, p, 3, 0, -1)
    return m, c, f, t


CASES["(g) SAM-like specks in the body"] = dict(build=specks, control=True)
LONG_HIND = {"FR": [(220, 360), (215, 290)], "FL": [(220, 440), (215, 510)],
             "HR": [(400, 370), (500, 272)], "HL": [(400, 430), (500, 528)]}
CASES["(h) 140-px hind legs, 70-px fore legs"] = dict(build=lambda: lizard(STRAIGHT_TAIL, legs=LONG_HIND), roles=True)
LONG_BACK = {"FR": [(220, 360), (215, 290)], "FL": [(220, 440), (215, 510)],
             "HR": [(420, 370), (540, 300)], "HL": [(420, 430), (540, 500)]}
CASES["(h') 140-px hind legs swept back"] = dict(build=lambda: lizard(STRAIGHT_TAIL, legs=LONG_BACK), roles=True)
BLOB = (330, 482)


def with_blob():
    m, c, f, t = lizard(STRAIGHT_TAIL)
    cv2.circle(m, BLOB, 30, 1, -1)
    return m, c, f, t


CASES["(i) an extra blob merged on the flank"] = dict(build=with_blob, roles=True, blobs=[BLOB])


def belly_up():
    m, c, f, t = lizard(STRAIGHT_TAIL)
    H = m.shape[0]
    flip = lambda p: np.array([p[0], H - 1 - p[1]], np.float64)
    return (m[::-1].copy(), np.array([flip(p) for p in c]), {k: flip(v) for k, v in f.items()},
            np.array([flip(p) for p in t]))


CASES["(j) belly-up (mirror of the base)"] = dict(build=belly_up, roles=True, anchor_flip=True)
BIG = 3                        # x3: well above MAX_NODES, so the analysis runs downscaled


def gap_big():
    """(a)'s curl at x3 with the tail end stopping 2 px short of the flank: no loop in the mask."""
    m = np.zeros((800 * BIG, 1100 * BIG), np.uint8)
    cv2.ellipse(m, (300 * BIG, 400 * BIG), (160 * BIG, 55 * BIG), 0, 0, 360, 1, -1)
    cv2.circle(m, (150 * BIG, 400 * BIG), 45 * BIG, 1, -1)
    flank_y = 400 * BIG + 55 * BIG                  # lower edge of the trunk below x = 300
    tail = [(460, 400), (570, 470), (580, 620), (470, 690), (340, 640), (300, 460)]
    xy = stroke(m, [(x * BIG, y * BIG) for x, y in tail], 40 * BIG, 10 * BIG)
    # the tail end ran into the trunk: cut it flat 2 px below the flank (rows flank_y+1, +2), drop
    # what it drew inside the trunk's band, and redraw the trunk -> a 2-px gap, no contact
    band = slice(250 * BIG, 350 * BIG)
    m[flank_y + 1:flank_y + 3, band] = 0
    m[:flank_y + 1, band] = 0
    cv2.ellipse(m, (300 * BIG, 400 * BIG), (160 * BIG, 55 * BIG), 0, 0, 360, 1, -1)
    cv2.circle(m, (150 * BIG, 400 * BIG), 45 * BIG, 1, -1)
    centre = np.vstack([polyline([(HEAD[0] * BIG, HEAD[1] * BIG), (460.0 * BIG, 400.0 * BIG)]), xy[1:]])
    return m, centre, {}, xy


CASES["(k) 2-px gap, x3 size (downscaled)"] = dict(build=gap_big, big=True)

print("Part A: geometry on synthetic masks (real silhouette.midline / extremity_roles)")
print(f"  {'case':40s} {'hole %':>7s} {'analysed':>8s} {'len/true':>8s} {'tip px':>7s} {'m50 px':>7s} {'m75 px':>7s} {'ms':>5s}  roles")
golden_a = {}
results = {}
for name, cfg in CASES.items():
    m, centre, feet, tail_xy = cfg["build"]()
    area = int(m.sum())
    hs = holes(m)
    hole_frac = (hs[0] / area) if hs else 0.0
    if cfg.get("anchor") == "start":
        anchor = (float(centre[0, 0]), float(centre[0, 1]))
    elif cfg.get("anchor_flip"):
        anchor = (HEAD[0], m.shape[0] - 1 - HEAD[1])
    elif cfg.get("big"):
        anchor = (HEAD[0] * BIG, HEAD[1] * BIG)
    else:
        anchor = HEAD
    t0 = time.perf_counter()
    ml = midline(m, anchor=anchor)
    ms = 1000 * (time.perf_counter() - t0)
    true_len = float(arc_of(centre)[-1])
    tip_err = float(np.linalg.norm(ml.tip - centre[-1]))
    m50 = float(np.linalg.norm(ml.sample([0.5])[0] - at_frac(centre, 0.5)))
    m75 = float(np.linalg.norm(ml.sample([0.75])[0] - at_frac(centre, 0.75)))
    roles = extremity_roles(ml)
    got = {r: label_of(p, feet, tail_xy if cfg.get("loop") else np.zeros((0, 2)), cfg.get("blobs", ()))
           for r, p in sorted(roles.items()) if r in ("FL", "FR", "HL", "HR")}
    role_txt = " ".join(f"{r}={'ok' if g == r else g}" for r, g in got.items()) or "-"
    analysed = 100 * getattr(ml, "hole_frac", float("nan"))      # (I293) the hole the midline saw
    print(f"  {name:40s} {100 * hole_frac:7.2f} {analysed:8.2f} {ml.length / true_len:8.2f} {tip_err:7.0f} {m50:7.0f} {m75:7.0f} "
          f"{ms:5.0f}  {role_txt}")
    results[name] = dict(hole_frac=hole_frac, ratio=ml.length / true_len, tip=tip_err, m50=m50, m75=m75,
                         roles=got, scale=ml.scale)
    golden_a[name] = ml
    if cfg.get("big"):
        # the same mask analysed at FULL resolution: does the downscale alone close the gap (I296)?
        keep = sil.MAX_NODES
        sil.MAX_NODES = 10 ** 9
        try:
            full = midline(m, anchor=anchor, want_extremities=False)
        finally:
            sil.MAX_NODES = keep
        print(f"  {'':40s} full-resolution analysis: len/true {full.length / true_len:.2f}, "
              f"tip {np.linalg.norm(full.tip - centre[-1]):.0f} px (analysis scale was {ml.scale:.2f})")
        results[name]["full_ratio"] = full.length / true_len

# the synthetic truth must hold where nothing touches (else every other number is meaningless)
for name, cfg in CASES.items():
    if cfg.get("control"):
        r = results[name]
        check(0.93 <= r["ratio"] <= 1.07 and r["tip"] < 20,
              f"control '{name}': length {r['ratio']:.2f} of the truth, tip {r['tip']:.0f} px")

# --------------------------------------------------------------- Part B: through the worker
SCALE = 2.0                       # native / working, as in a real run


def native(p):
    return np.array([(p[0] + 0.5) * SCALE - 0.5, (p[1] + 0.5) * SCALE - 0.5], np.float32)


def shifted(m, dx):
    out = np.zeros_like(m)
    out[:, dx:] = m[:, :m.shape[1] - dx] if dx else m
    return out


SPECS = ["tip", "midline:0.5", "midline:0.75", "ext:HR", "centroid"]


def run_sequence(frames):
    """frames = [(mask_working, head_working_xy)]; returns (tracks, conf, worker)."""
    derived = [DerivedSpec(1 + j, s) for j, s in enumerate(SPECS)]
    w = TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100,
                       specs=[PointSpec(0, native(frames[0][1]))], animal=AnimalSpec({0: []}, {}, None),
                       derived=derived, head_pid=0)
    L = len(frames)
    out_tr = np.full((L, w.n_cols, 2), np.nan, np.float32)
    out_vi = np.zeros((L, w.n_cols), bool)
    out_cf = np.zeros((L, w.n_cols), np.float32)
    for f, (mw, head) in enumerate(frames):
        w._mask_hist[f] = (mw.astype(bool), SCALE)
        w._summ[f] = summarize_mask(mw.astype(np.uint8), SCALE, 8.0)
        out_tr[f, 0] = native(head)
        out_cf[f, 0] = 1.0
    w._fill_derived(0, out_tr, out_vi, out_cf)
    return out_tr, out_cf, w


def sequence(before, during, after, head=HEAD, n=(5, 10, 5)):
    """Masks of three phases, the animal drifting right by 1 working px per frame."""
    frames = []
    for k, (m, cnt) in enumerate(((before, n[0]), (during, n[1]), (after, n[2]))):
        for i in range(cnt):
            f = len(frames)
            frames.append((shifted(m, f), (head[0] + f, head[1])))
    return frames


print("\nPart B: 20-frame runs through TrackingWorker._fill_derived (frames 5-14 = the loop)")
near_m = lizard(NEAR)[0]
no_hr = {k: v for k, v in LEGS.items() if k != "HR"}
fish_head = tuple(float(v) for v in fish(1.6)[1][0])
LOOPS = {   # name: (before / after: mask + centreline, loop: mask + centreline, head)
    "loop_a": (lizard(NEAR), lizard(OVER), HEAD),                                   # length 0.41
    "loop_b": (lizard(UPTAIL, legs=no_hr), lizard(UPTAIL, legs=HR_ON_TAIL), HEAD),  # 0.66
    "loop_c": (fish(1.6), fish(1.95), fish_head),                                   # 0.69
}
golden_b = {}
loop = range(5, 15)
for key, (calm, curl, head) in LOOPS.items():
    tr, cf, w = run_sequence(sequence(calm[0], curl[0], calm[0], head=head))
    golden_b[key] = (tr, cf)
    # the derived tail tip against the TRUE tip of each frame (native px), per phase
    err = [float(np.linalg.norm(tr[f, 1] - native((c[1][-1][0] + f, c[1][-1][1]))))
           for f, c in enumerate([calm] * 5 + [curl] * 10 + [calm] * 5)]
    print(f"  {key}: true tail-tip error (native px) mean before {np.mean(err[:5]):.0f}, "
          f"during the loop {np.mean(err[5:15]):.0f}, after {np.mean(err[15:]):.0f}")
    for j, spec in enumerate(SPECS[:3]):
        c = cf[:, 1 + j]
        low = [f for f in loop if c[f] <= LOW + 1e-6]
        print(f"    {spec:13s} conf per frame: " + " ".join(f"{v:.2f}" for v in c))
        check(len(low) == len(loop), f"{key} {spec}: low confidence (<= {LOW}) on {len(low)}/{len(loop)} "
                                     f"loop frames (frames {low})")
    anch = sorted(w._anchor_lost)
    print(f"    anchor distrusted on frames {anch} (the anchor-check interplay; not asserted)")
    tip_jump = [float(np.linalg.norm(tr[f, 1] - tr[f - 1, 1])) for f in range(1, len(tr))]
    print(f"    tail-tip step per frame (native px): " + " ".join(f"{v:.0f}" for v in tip_jump))
    if hasattr(w, "_loop_frames"):
        check(sorted(w._loop_frames) == list(loop), f"{key}: loop flagged on exactly frames 5-14 "
                                                    f"(got {sorted(w._loop_frames)})")
        check(bool((cf[:, 1 + SPECS.index("centroid")] > LOW).all()),
              f"{key}: the centroid (from the mask, not the midline) is never demoted as a loop")

if hasattr(trk, "LOOP_HOLE_FRAC"):
    # (I293 / I296) every loop the midline really cut short is seen as an enclosed hole where the path
    # was found -- (k) only after the downscale closed its 2-px gap
    for name in ("(a) tail over the flank, tip out", "(a') same, no legs", "(b) hind foot on an upturned tail",
                 "(c) closed C-curl fish", "(k) 2-px gap, x3 size (downscaled)"):
        hf = golden_a[name].hole_frac
        check(hf >= trk.LOOP_HOLE_FRAC, f"'{name}': analysed hole {100 * hf:.1f} % >= {100 * trk.LOOP_HOLE_FRAC:.0f} %")
    # a run that STARTS inside a loop: no length to compare with, so it counts as one (owner, gate)
    calm, curl, head = LOOPS["loop_a"]
    tr_s, cf_s, w_s = run_sequence(sequence(curl[0], curl[0], calm[0], head=head))
    check(sorted(w_s._loop_frames) == list(range(15)) and bool((cf_s[:15, 1] <= LOW + 1e-6).all()),
          f"a run starting in the loop: frames 0-14 flagged and low (got {sorted(w_s._loop_frames)})")



def run_windowed(frames, win=16, step=8, nudge=0.3):
    """As run_sequence, but the way the worker emits: windows of `win` rows every `step`, so each
    window's overlap rows are sent AGAIN, with the head point moved by `nudge` native px (a new cache
    key: the midline is recomputed). The last write of a row wins, as in the session."""
    derived = [DerivedSpec(1 + j, s) for j, s in enumerate(SPECS)]
    w = TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100,
                       specs=[PointSpec(0, native(frames[0][1]))], animal=AnimalSpec({0: []}, {}, None),
                       derived=derived, head_pid=0)
    L = len(frames)
    for f, (mw, _) in enumerate(frames):
        w._mask_hist[f] = (mw.astype(bool), SCALE)
        w._summ[f] = summarize_mask(mw.astype(np.uint8), SCALE, 8.0)
    final_cf = np.zeros((L, w.n_cols), np.float32)
    for k, s0 in enumerate(range(0, L, step)):
        if s0 and s0 >= L - step:
            break
        rows = range(s0, min(s0 + win, L))
        tr = np.full((len(rows), w.n_cols, 2), np.nan, np.float32)
        vi = np.zeros((len(rows), w.n_cols), bool)
        cf = np.zeros((len(rows), w.n_cols), np.float32)
        for i, f in enumerate(rows):
            tr[i, 0] = native(frames[f][1]) + np.float32(nudge * k)
            cf[i, 0] = 1.0
        w._fill_derived(s0, tr, vi, cf)
        final_cf[s0:s0 + len(rows)] = cf
    return final_cf, w


if hasattr(trk, "LOOP_HOLE_FRAC"):
    # (I293, review) the worker re-sends every window's 8 overlap rows: the verdict of a re-sent row
    # must not depend on the window's LAST frame (a running state flagged frame 8 of loop (b) off,
    # and carried a loop back onto harmless foot-on-flank frames before a loop)
    def phases(*parts, head=HEAD):
        out = []
        for m, n in parts:
            for _ in range(n):
                f = len(out)
                out.append((shifted(m, f), (head[0] + f, head[1])))
        return out

    cases = {k: sequence(c[0], l[0], c[0], head=h) for k, (c, l, h) in LOOPS.items()}
    cases["flank_then_loop"] = phases((base_m := lizard(STRAIGHT_TAIL)[0], 5),
                                      (CASES["(e) hind foot on its own flank"]["build"]()[0], 5),
                                      (lizard(OVER)[0], 14), (base_m, 4))
    for key, frames in cases.items():
        _, cf1, w1 = run_sequence(frames)
        cfw, ww = run_windowed(frames)
        one, win = sorted(w1._loop_frames), sorted(ww._loop_frames)
        saved_low = all(cfw[f, 1] <= LOW + 1e-6 for f in one)        # what the session keeps (last write)
        check(win == one and saved_low, f"{key}: re-sent overlap rows give the one-pass loop frames {one} "
                                        f"(windowed: {win}; saved tip low on all of them: {saved_low})")

# the loop-free bar: sequences that never close a harmful loop (saved for the byte-identity proof);
# foot_flank / foot_tail DO enclose a hole (4.6 % / 15.9 %) but their midline is right: the foot
# comes down onto the flank / tail at frame 5 and lifts at 15, as in a real run
base_m = lizard(STRAIGHT_TAIL)[0]
flank_m = CASES["(e) hind foot on its own flank"]["build"]()[0]
tailc_m = CASES["(f) hind foot on a straight tail"]["build"]()[0]
for key, seq in (("base", sequence(base_m, base_m, base_m)), ("near", sequence(near_m, near_m, near_m)),
                 ("foot_flank", sequence(base_m, flank_m, base_m)),
                 ("foot_tail", sequence(base_m, tailc_m, base_m)),
                 ("specks", sequence(specks()[0], specks()[0], specks()[0]))):
    tr_k, cf_k, w_k = run_sequence(seq)
    golden_b[key] = (tr_k, cf_k)
    if hasattr(w_k, "_loop_frames"):
        check(not w_k._loop_frames, f"no loop flagged on the loop-free sequence '{key}' "
                                    f"(got {sorted(w_k._loop_frames)})")
if hasattr(trk, "LOOP_HOLE_FRAC"):
    # the gate's rule has a price, recorded here: a run that STARTS with a harmless contact (no
    # length yet to compare with) is demoted until the contact ends -- then judged normally
    _, cf_h, w_h = run_sequence(sequence(flank_m, flank_m, base_m))
    check(sorted(w_h._loop_frames) == list(range(15)) and bool((cf_h[15:, 1] > LOW).all()),
          f"a run starting with a foot on the flank: demoted on frames 0-14 only "
          f"(got {sorted(w_h._loop_frames)})")

try:
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, timeout=10).stdout.strip() or "nogit"
    if subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "kinetrace"], capture_output=True,
                      text=True, timeout=10).stdout.strip():
        sha += "_dirty"                     # uncommitted engine code: not that commit's output
except Exception:
    sha = "nogit"
dump = {}
for name, ml in golden_a.items():
    k = "".join(ch if ch.isalnum() else "_" for ch in name)
    dump[f"A_{k}_path"], dump[f"A_{k}_arc"], dump[f"A_{k}_width"], dump[f"A_{k}_ext"] = (
        ml.path, ml.arc, ml.width, ml.extremities)
for key, (tr_k, cf_k) in golden_b.items():
    dump[f"B_{key}_tracks"], dump[f"B_{key}_conf"] = tr_k, cf_k
gpath = os.path.join(OUT, f"golden_{sha}.npz")
np.savez(gpath, **dump)
print(f"\ngolden output: {gpath}")

if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\nALL SILHOUETTE LOOP CHECKS PASSED")
