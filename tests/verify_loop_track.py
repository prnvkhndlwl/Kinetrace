"""(I293) A real Track run over a self-contact loop: the fused TrackingWorker (CoTracker3 head
point + SAM silhouette + derived tail tip / midline:0.5 / centroid) on a deterministic synthetic
clip of a lizard whose tail sweeps over its own flank, stays there, and sweeps back:

    frames  0-14  tail straight          (no contact)
    frames 15-24  tail sweeping round
    frames 25-36  tail over the flank    (a loop: the midline cuts through the contact)
    frames 37-47  tail sweeping back     (the loop opens INSIDE a window's re-sent overlap rows)
    frames 48-59  tail straight          (no contact)

On EVERY frame whose drawn silhouette encloses a hole the derived tip / midline points must be
LOW confidence -- before I293 only the first frame of the contact was (the one-frame jump rule).
Where the tail is straight nothing may be flagged. The tracks and confidences go to
tests/out/loops/track_<sha>.npz for the byte-identity proof against main. GPU, ~1-2 min.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])

from kinetrace import tracker as trk  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402
from kinetrace.tracker import AnimalSpec, DerivedSpec, PointSpec, TrackingWorker  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "loops")
os.makedirs(OUT, exist_ok=True)
W, H, T = 1280, 720, 60
LOW = getattr(trk, "LOOP_CONF", trk.DERIVED_JUMP_CONF)
S, OX, OY = 0.5, 250.0, 60.0                  # the suite's lizard geometry, scaled into 1280 x 720


def P(x, y):
    return (OX + S * x, OY + S * y)


def resample(pts, n=80):
    p = np.asarray(pts, np.float64)
    a = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    w = np.linspace(0.0, a[-1], n)
    return np.stack([np.interp(w, a, p[:, 0]), np.interp(w, a, p[:, 1])], axis=1)


# the tail BENDS at constant length: the curled shape's turning angles scaled by 0 (straight) .. 1
# (blending the two shapes point by point shortens the tail mid-sweep, which no animal does)
CURLED = resample([P(460, 400), P(570, 470), P(580, 620), P(470, 690), P(340, 640), P(300, 470), P(290, 330),
                   P(300, 240)], 120)
_seg = np.linalg.norm(np.diff(CURLED, axis=0), axis=1)
_turn = np.unwrap(np.arctan2(*np.diff(CURLED, axis=0)[:, ::-1].T))


def tail_at(a: float) -> np.ndarray:
    th = a * _turn
    return np.vstack([CURLED[:1], CURLED[0] + np.cumsum(np.stack([_seg * np.cos(th), _seg * np.sin(th)], 1), 0)])
LEGS = [[P(220, 360), P(200, 260), P(240, 230)], [P(220, 440), P(200, 540), P(240, 570)],
        [P(400, 370), P(440, 270), P(480, 240)], [P(400, 430), P(440, 530), P(480, 560)]]
HEAD = P(115, 400)
EYE = P(140, 385)


def curl(t: int) -> float:
    """0 = straight, 1 = over the flank."""
    if t < 15 or t >= 48:
        return 0.0
    if 25 <= t < 37:
        return 1.0
    return (t - 14) / 11.0 if t < 25 else (48 - t) / 11.0


def stroke(imgs, pts, t0, t1):
    th = np.linspace(t0, t1, len(pts))
    for i in range(len(pts) - 1):
        a = tuple(int(round(v)) for v in pts[i])
        b = tuple(int(round(v)) for v in pts[i + 1])
        for img, col in imgs:
            cv2.line(img, a, b, col, max(1, int(round(th[i]))), cv2.LINE_AA)


def frame(t: int):
    rng = np.random.default_rng(7)
    bg = rng.integers(60, 140, (H // 8, W // 8, 3), np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    bg[..., 1] = np.clip(bg[..., 1].astype(int) + 40, 0, 255)
    img, mask, col = bg.copy(), np.zeros((H, W), np.uint8), (38, 44, 58)
    imgs = ((img, col), (mask, 255))
    for im, c in imgs:
        cv2.ellipse(im, tuple(int(round(v)) for v in P(300, 400)), (int(160 * S), int(55 * S)), 0, 0, 360, c, -1,
                    cv2.LINE_AA)
        cv2.circle(im, tuple(int(round(v)) for v in P(150, 400)), int(45 * S), c, -1, cv2.LINE_AA)
    for leg in LEGS:
        stroke(imgs, resample(leg, 20), 18 * S, 12 * S)
    stroke(imgs, tail_at(curl(t)), 40 * S, 10 * S)
    cv2.circle(img, tuple(int(round(v)) for v in EYE), 5, (230, 230, 240), -1, cv2.LINE_AA)   # a feature to track
    return img, mask > 127


def holes(m) -> float:
    n, _, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    inner = [int(st[i, 4]) for i in range(1, n)
             if st[i, 0] > 0 and st[i, 1] > 0 and st[i, 0] + st[i, 2] < W and st[i, 1] + st[i, 3] < H]
    return max(inner) / max(int(m.sum()), 1) if inner else 0.0


VID = os.path.join(OUT, "loop_track.mp4")
truth_hole = []
vw = None
if not os.path.exists(VID) or int(cv2.VideoCapture(VID).get(cv2.CAP_PROP_FRAME_COUNT)) != T:
    vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
for t in range(T):
    img, m = frame(t)
    truth_hole.append(holes(m))
    if vw is not None:
        vw.write(img)
if vw is not None:
    vw.release()
print("drawn hole share per frame (%): " + " ".join(f"{100 * h:.0f}" for h in truth_hole))

s = TrackingSession(VID, T, 25.0, W, H)
head = s.add_point(0, float(EYE[0]), float(EYE[1]), name="snout")
s.ensure_animal()
s.animal.add_click(0, *P(300, 400), True)
tip = s.add_landmark("tail_tip", "silhouette", "tip")
mid = s.add_landmark("mid50", "silhouette", "midline:0.5")
cen = s.add_landmark("centre", "silhouette", "centroid")
w = TrackingWorker(VID, 0, None, None, FrameCache(256 * 1024 ** 2), T, refine=True,
                   specs=[PointSpec(head, s.tracks[0, head].astype(np.float32).copy())], roi=False,
                   autopause=False, animal=AnimalSpec({0: list(s.animal.prompts[0])}, {}, None),
                   derived=[DerivedSpec(tip, "tip"), DerivedSpec(mid, "midline:0.5"), DerivedSpec(cen, "centroid")],
                   head_pid=head)
ev = {"error": None}
w.masks_ready.connect(lambda summ: s.write_mask_summaries(summ))
w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: s.write_segment(w0, tr, vi, w.point_ids, cf))
w.error.connect(lambda msg: ev.update(error=msg))
t0 = time.perf_counter()
w.run()
print(f"tracked {int(s.tracked.any(axis=1).sum())} frames in {time.perf_counter() - t0:.0f}s")
if ev["error"]:
    sys.exit("worker error: " + ev["error"])

failures = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


for nm, pid in (("tail_tip", tip), ("mid50", mid), ("centre", cen)):
    print(f"  {nm:8s} conf: " + " ".join(f"{v:.2f}" for v in s.confidence[:T, pid]))
print(f"  loop frames (worker): {sorted(getattr(w, '_loop_frames', []))}")
holed = [f for f in range(T) if truth_hole[f] >= 0.02]
print(f"  frames whose drawn silhouette encloses a hole: {holed}")
for nm, pid in (("tail_tip", tip), ("mid50", mid)):
    missed = [f for f in holed if not (s.tracked[f, pid] and s.confidence[f, pid] <= LOW + 1e-6)]
    check(len(holed) >= 12 and not missed,
          f"{nm}: low (<= {LOW}) on every one of the {len(holed)} frames with a loop (missed: {missed})")
calm = [f for f in range(T) if f < 15 or f >= 50]
for nm, pid in (("tail_tip", tip), ("mid50", mid), ("centre", cen)):
    bad = [f for f in calm if s.tracked[f, pid] and s.confidence[f, pid] <= LOW + 1e-6]
    check(not bad, f"{nm}: never low where the tail is straight (got {bad})")
if hasattr(w, "_loop_frames"):
    check(not [f for f in w._loop_frames if truth_hole[f] < 0.01],
          f"no loop flagged on a frame whose drawn silhouette encloses no hole")

try:
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         timeout=10).stdout.strip() or "nogit"
    if subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "kinetrace"], capture_output=True,
                      text=True, timeout=10).stdout.strip():
        sha += "_dirty"
except Exception:
    sha = "nogit"
np.savez(os.path.join(OUT, f"track_{sha}.npz"), tracks=s.tracks[:T], conf=s.confidence[:T],
         tracked=s.tracked[:T])
if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED")
    sys.exit(1)
print("\nverify_loop_track PASSED")
