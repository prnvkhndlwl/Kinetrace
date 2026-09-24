"""The on-body rules of the worker, on a bare TrackingWorker with a synthetic
mask (no model, no GPU, no video decode):

* a landmark predicted OUTSIDE the segment beyond the jitter band stops the run
  at that frame (snapping it back does not guarantee the same spot) and its data from that frame on is blank;
* a landmark a few pixels outside (outline jitter) is nudged onto the
  silhouette and the run continues;
* group members are always nudged (they are internal samples);
* two constrained landmarks that were clearly apart when clicked and now sit
  on the same pixel are flagged as merged, and demoted in THEIR OWN output
  columns after an ROI restart dropped a point (I116), with or without a
  skeleton (I117);
* a free (unconstrained) point is never touched.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from cotracker_app.tracker import (AnimalSpec, PointSpec, TrackingWorker, EXIT_BAND_MIN_PX,
                                   EXIT_BAND_FRAC, OFF_BODY_CONF)
from cotracker_app.video_source import FrameCache

SCALE = 2.0                                   # native / working
mask = np.zeros((150, 200), bool)             # working res
cv2.ellipse(mask.view(np.uint8), (100, 75), (60, 30), 0, 0, 360, 1, -1)
ys, xs = np.nonzero(mask)
bbox_native = (int(xs.min() * SCALE), int(ys.min() * SCALE), int((xs.max() + 1) * SCALE) - 1,
               int((ys.max() + 1) * SCALE) - 1)
diag_w = float(np.hypot(bbox_native[2] - bbox_native[0] + 1, bbox_native[3] - bbox_native[1] + 1)) / SCALE
band_w = max(EXIT_BAND_MIN_PX / SCALE, EXIT_BAND_FRAC * diag_w)
print(f"mask {mask.sum()} px, band {band_w:.1f} working px")


def native(xw, yw):
    return np.array([(xw + 0.5) * SCALE - 0.5, (yw + 0.5) * SCALE - 0.5], np.float32)


def make_worker(seeds, constrain, group=None, on_body=None):
    specs = [PointSpec(i, np.asarray(s, np.float32)) for i, s in enumerate(seeds)]
    if group is not None:
        specs.append(group)
    w = TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100, specs=specs,
                       animal=AnimalSpec({0: []}, {}, None), constrain_pids=constrain,
                       on_body_pids=on_body)
    w._refined = {}
    for f in range(0, 8):
        w._mask_hist[f] = (mask, SCALE)
        w._summ[f] = {"bbox": bbox_native, "area": int(mask.sum() * SCALE * SCALE), "score": 8.0}
    return w


# ---- 1. exit stops the run at the frame it happens; jitter is nudged ----------------
seeds = [native(100, 75), native(160, 75), native(40, 75), native(100, 40)]
w = make_worker(seeds, constrain=[0, 1, 2])          # 3 is free
out_map = [("point", 0), ("point", 1), ("point", 2), ("point", 3)]
L = 4
tr = np.zeros((L, 4, 2), np.float32)
for i in range(L):
    tr[i, 0] = native(100, 75)                       # inside
    tr[i, 1] = native(160 + 2, 75)                   # 2 px outside the right edge: jitter
    tr[i, 2] = native(40, 75) if i == 0 else native(40 - 30, 75)   # leaves at row 1, by 30 px
    tr[i, 3] = native(100, 10)                       # free, far outside: untouched
w._constrain_to_mask(2, tr, w.specs, out_map, first_seg=False)
assert w._exit_hit == (3, 2), w._exit_hit
assert w._autopause_hit == (3, 2) and w._autopause_reason == "exit"
assert np.isfinite(tr[0, 2]).all() and np.isnan(tr[1:, 2]).all(), "exited landmark must be blank from the exit frame"
xw, yw = (tr[1, 1] + 0.5) / SCALE - 0.5
assert mask[int(round(yw)), int(round(xw))], "a jittering point is nudged onto the silhouette"
assert 1 in w._snapped[3] and 2 not in w._snapped[3]
assert np.allclose(tr[:, 3], native(100, 10)), "free point touched"
assert np.allclose(tr[:, 0], native(100, 75)), "inside point touched"
print("exit stops at the frame, jitter nudged, free point untouched OK")

# ---- 2. no exit: nothing stops --------------------------------------------------------
w2 = make_worker(seeds[:2], constrain=[0, 1])
tr2 = np.zeros((L, 2, 2), np.float32)
tr2[:, 0] = native(100, 75)
tr2[:, 1] = native(161, 75)
w2._constrain_to_mask(0, tr2, w2.specs, [("point", 0), ("point", 1)], first_seg=False)
assert w2._exit_hit is None and w2._autopause_hit is None
print("in-band jitter never stops the run OK")

# ---- 3. merged landmarks ----------------------------------------------------------------
w3 = make_worker([native(60, 75), native(140, 75)], constrain=[0, 1])   # 80 working px apart when clicked
tr3 = np.zeros((L, 2, 2), np.float32)
tr3[:, 0] = native(100, 75)
tr3[:, 1] = native(101, 75)                             # now on the same spot
w3._constrain_to_mask(0, tr3, w3.specs, [("point", 0), ("point", 1)], first_seg=False)
assert all(w3._collapsed[f] == {0, 1} for f in range(0, L)), w3._collapsed
w4 = make_worker([native(100, 75), native(101, 75)], constrain=[0, 1])  # clicked on the same spot: not a collapse
w4._constrain_to_mask(0, tr3.copy(), w4.specs, [("point", 0), ("point", 1)], first_seg=False)
assert not any(w4._collapsed.get(f) for f in range(0, L))
print("merged landmarks flagged only when they were apart at the click OK")

# ---- 3b. after an ROI restart that dropped a point, the merged pair is demoted in its
# OWN columns and judged by its OWN clicks (I116); with a skeleton and without (I117:
# points placed with N on a segment are constrained too, and the cap sat behind the
# skeleton-only guard). pid0 is a free point dropped at the restart, pids 1..3 are
# constrained; after the restart specs = [1, 2, 3] fill columns [1, 2, 3].
col_idx = [1, 2, 3]
out_map_r = [("point", 0), ("point", 1), ("point", 2)]
for on_body in ([1, 2, 3], None):
    # pid1 and pid2 were clicked on ONE spot, pid3 far away: indexing the clicks by
    # spec index compares pid1 with pid2 ("not apart") instead of pid2 with pid3
    w6 = make_worker([native(10, 10), native(90, 75), native(91, 75), native(150, 75)],
                     constrain=[1, 2, 3], on_body=on_body)
    specs_r = [PointSpec(1, native(50, 75)), PointSpec(2, native(120, 75)), PointSpec(3, native(121, 75))]
    tr6 = np.zeros((L, 3, 2), np.float32)
    tr6[:, 0] = native(50, 75)                  # pid1: far from the others, innocent
    tr6[:, 1] = native(120, 75)                 # pid2 and pid3 now on one spot: merged
    tr6[:, 2] = native(120.5, 75)
    w6._constrain_to_mask(2, tr6, specs_r, out_map_r, first_seg=False, col_idx=col_idx)
    assert all(w6._collapsed.get(f) == {1, 2} for f in range(2, 2 + L)), w6._collapsed
    out_tr6 = np.full((L, 4, 2), np.nan, np.float32)
    for om, k in zip(out_map_r, col_idx):
        out_tr6[:, k] = tr6[:, om[1]]
    out_cf6 = np.ones((L, 4), np.float32)
    out_cf6[:, 0] = 0.0                         # the dropped column
    w6._demote_off_body(2, out_tr6, out_cf6)
    w6._demote_merged(2, out_cf6)
    tag = "skeleton" if on_body else "no skeleton"
    assert (out_cf6[:, 1] == 1.0).all(), f"{tag}: the innocent landmark was demoted {out_cf6[0]}"
    assert (out_cf6[:, 2:] <= OFF_BODY_CONF).all(), f"{tag}: a merged landmark kept its confidence {out_cf6[0]}"
print("merged landmarks demoted in their own columns after a restart, skeleton or not OK")

# ---- 4. group members are nudged, never an exit ---------------------------------------------
from cotracker_app.tracker import PointSpec as _PS
grp = _PS(5, native(100, 75), kind="group", radius=20.0)
w5 = make_worker([native(100, 75)], constrain=[0, 5], group=grp)
sl = slice(1, 4)
tr5 = np.zeros((L, 4, 2), np.float32)
tr5[:, 0] = native(100, 75)
tr5[:, 1] = native(100, 75)
tr5[:, 2] = native(100, 10)                              # member far outside
tr5[:, 3] = native(100, 75)
w5._constrain_to_mask(0, tr5, w5.specs, [("point", 0), ("group", sl, None, None)], first_seg=False)
assert w5._exit_hit is None
xw, yw = (tr5[0, 2] + 0.5) / SCALE - 0.5
assert mask[int(round(yw)), int(round(xw))], "member not nudged onto the silhouette"
print("group members nudged, never an exit OK")

# ---- the worker's spec types build the way the app builds them (I2, 2026-09-22):
# a decorator displaced onto BallSpec left DerivedSpec without a constructor and
# every Track with silhouette-derived landmarks raised TypeError in the app
from cotracker_app.tracker import BallSpec, DerivedSpec  # noqa: E402
d = DerivedSpec(3, "midline:0.5")
assert (d.pid, d.spec) == (3, "midline:0.5")
assert DerivedSpec(pid=4, spec="tip").spec == "tip"
b1 = BallSpec(1, {0: [[10.0, 12.0, 1]]})
b2 = BallSpec(2, {5: [[30.0, 32.0, 1]]}, seed=(30.0, 32.0), radius=6.0)
assert b1.pid == 1 and b2.radius == 6.0 and b1 != b2, "two different balls must not compare equal"
wd = TrackingWorker("none.mp4", 0, None, None, FrameCache(1 << 20), 100,
                    specs=[PointSpec(0, native(100, 75))], animal=AnimalSpec({0: []}, {}, None),
                    derived=[DerivedSpec(1, "tip")], balls=[b1])
assert wd.derived[0].spec == "tip" and wd.balls[0].pid == 1
print("DerivedSpec / BallSpec construct as the app builds them OK")
print("verify_onbody_rules PASSED")
