"""Animal layer, headless: the fused TrackingWorker (CoTracker3 points + per-
frame SAM masks + silhouette-derived landmarks) against the synthetic two-
animal video with exact ground truth, then session v3 persistence and the new
export formats. GPU, ~3 min (mostly model loads)."""
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication

app = QCoreApplication([])

import _synth  # noqa: E402
from cotracker_app.segmenter import working_size  # noqa: E402
from cotracker_app.session import TrackingSession  # noqa: E402
from cotracker_app.skeletons import template_by_name  # noqa: E402
from cotracker_app.tracker import AnimalSpec, DerivedSpec, PointSpec, TrackingWorker  # noqa: E402
from cotracker_app.video_source import FrameCache  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
W, H, T = _synth.W, _synth.H, _synth.T
VID = _synth.build_video(os.path.join(OUT, "synth_animals.mp4"))
GT = [_synth.gt_frame(t)[1][0] for t in range(T)]   # animal 0 ground truth per frame


def run_worker(s: TrackingSession, start: int, stop: int, specs, animal, derived,
               head_pid=None, on_body=(), autopause=False, roi=True, constrain=()):
    w = TrackingWorker(VID, start, None, None, FrameCache(256 * 1024 ** 2), stop, refine=True,
                       specs=specs, roi=roi, autopause=autopause, animal=animal, derived=derived,
                       head_pid=head_pid, on_body_pids=list(on_body), constrain_pids=list(constrain))
    ev = {"error": None, "finished": None, "autopaused": None, "n_masks": 0, "chunks": 0}
    w.masks_ready.connect(lambda summ: (s.write_mask_summaries(summ), ev.__setitem__("n_masks", ev["n_masks"] + len(summ))))
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: (s.write_segment(w0, tr, vi, w.point_ids, cf),
                                                          ev.__setitem__("chunks", ev["chunks"] + 1)))
    w.finished_ok.connect(lambda last, p: ev.update(finished=(last, p)))
    w.autopaused.connect(lambda f, pid: ev.update(autopaused=(f, pid)))
    w.error.connect(lambda m: ev.update(error=m))
    t0 = time.perf_counter()
    w.run()
    ev["dt"] = time.perf_counter() - t0
    if ev["error"]:
        print(ev["error"])
        sys.exit("worker error")
    return w, ev


def err_stats(s, pid, key, frames):
    e = []
    for t in frames:
        if s.tracked[t, pid]:
            e.append(np.linalg.norm(s.tracks[t, pid] - GT[t][key]))
    e = np.array(e)
    return (e.mean() if len(e) else np.inf, e.max() if len(e) else np.inf, len(e))


def mask_centroid(t):
    ys, xs = np.nonzero(GT[t]["mask"])
    return np.array([xs.mean(), ys.mean()])


for t in range(T):
    GT[t]["mask_centroid"] = mask_centroid(t)

# ---- 0. baseline: the same two tracked points WITHOUT an animal ---------------
g0 = GT[0]
sb = TrackingSession(VID, T, 25.0, W, H)
pb_head = sb.add_point(0, float(g0["eye"][0]), float(g0["eye"][1]), name="snout")
pb_body = sb.add_point(0, float(g0["centre"][0]), float(g0["centre"][1]), name="body")
specs_b = [PointSpec(p, sb.tracks[0, p].astype(np.float32).copy()) for p in (pb_head, pb_body)]
wb, evb = run_worker(sb, 0, T, specs_b, None, None)
base_head = err_stats(sb, pb_head, "eye", range(T))
base_body = err_stats(sb, pb_body, "centre", range(T))
print(f"baseline (no animal): snout err mean {base_head[0]:.2f} px, body {base_body[0]:.2f} px "
      f"(the synthetic eye is a 4 px dot on a flat head — a hard feature by design)")

# ---- 1. fused run: tracked head/body + animal + derived landmarks --------------
s = TrackingSession(VID, T, 25.0, W, H)
pid_head = s.add_point(0, float(g0["eye"][0]), float(g0["eye"][1]), name="snout")
pid_body = s.add_point(0, float(g0["centre"][0]), float(g0["centre"][1]), name="body")
pid_rock = s.add_point(0, W * 0.5, H * 0.08, name="foot_FL")  # a "foot" seeded on the background
s.ensure_animal()
s.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
mid_tail = (np.asarray(g0["tip"]) + np.asarray(g0["centre"])) / 2
s.animal.add_click(0, float(mid_tail[0]), float(mid_tail[1]), True)
tpl = {"name": "test lizard", "head": "snout",
       "landmarks": ["snout", "body", "tail_tip", "mid50", "centre", "foot_FL", "foot_FR", "foot_HL", "foot_HR"],
       "bones": [["snout", "body"], ["body", "tail_tip"]],
       "derived": {"tail_tip": "tip", "mid50": "midline:0.5", "centre": "centroid",
                   "foot_FR": "ext:FR", "foot_HL": "ext:HL", "foot_HR": "ext:HR"}}
new = s.apply_skeleton(tpl)
assert s.head_pid() == pid_head and len(new) == 6, (s.head_pid(), new)
assert not s.points[pid_rock].derived  # existing tracked point keeps its source
derived = [DerivedSpec(p, s.points[p].spec) for p in s.derived_pids()]
specs = [PointSpec(p, s.tracks[0, p].astype(np.float32).copy()) for p in s.seedable_at(0)]
animal = AnimalSpec({0: list(s.animal.prompts[0])}, {}, None)
on_body = [p for p in s.seedable_at(0) if s.points[p].name in tpl["landmarks"]]
w, ev = run_worker(s, 0, T, specs, animal, derived, head_pid=pid_head, on_body=on_body)
assert ev["finished"] == (T - 1, False), ev["finished"]
frames = range(T)
n_masked = s.masks.n_masked()
print(f"fused run: {T} frames in {ev['dt']:.1f}s (includes the cold model loads), masks on "
      f"{n_masked}/{T} frames, {ev['chunks']} chunks, {ev['n_masks']} mask summaries")
assert n_masked >= 0.97 * T
pid = {s.points[i].name: i for i in range(s.n_points)}
head_e = err_stats(s, pid_head, "eye", frames)
body_e = err_stats(s, pid_body, "centre", frames)
tip_e = err_stats(s, pid["tail_tip"], "tip", frames)
cen_e = err_stats(s, pid["centre"], "mask_centroid", frames)
print(f"  snout (tracked) err mean {head_e[0]:.2f} max {head_e[1]:.2f} px on {head_e[2]} frames "
      f"(baseline {base_head[0]:.2f})")
print(f"  body  (tracked) err mean {body_e[0]:.2f} max {body_e[1]:.2f} px (baseline {base_body[0]:.2f})")
print(f"  tail tip (derived) err mean {tip_e[0]:.2f} max {tip_e[1]:.2f} px on {tip_e[2]} frames")
print(f"  centroid (derived) vs GT mask centroid: err mean {cen_e[0]:.2f} max {cen_e[1]:.2f} px")
# the animal layer (support points, mask crop) must not make the tracked points worse
assert head_e[0] <= base_head[0] + 2.0 and body_e[0] <= base_body[0] + 2.0, "tracked points degraded by the animal layer"
assert head_e[0] < 12.0 and body_e[0] < 12.0
assert tip_e[2] >= 0.95 * T and tip_e[0] < 8.0 and tip_e[1] < 25.0, "tail tip from the silhouette"
assert cen_e[0] < 6.0
# warm throughput of the fused run (models already loaded)
sw = TrackingSession(VID, T, 25.0, W, H)
pw = [sw.add_point(0, float(g0["eye"][0]), float(g0["eye"][1]), name="snout"),
      sw.add_point(0, float(g0["centre"][0]), float(g0["centre"][1]), name="body")]
sw.ensure_animal()
sw.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
pt_w = sw.add_landmark("tail_tip", "silhouette", "tip")
ww_, evw = run_worker(sw, 0, 64, [PointSpec(p, sw.tracks[0, p].astype(np.float32).copy()) for p in pw],
                      AnimalSpec({0: list(sw.animal.prompts[0])}, {}, None), [DerivedSpec(pt_w, "tip")],
                      head_pid=pw[0])
print(f"  warm fused throughput at {W}x{H}: {64/evw['dt']:.1f} fps (points + masks + landmarks)")
# midline:0.5 must lie on the body (inside the dilated GT mask)
inside = 0
for t in frames:
    if s.tracked[t, pid["mid50"]]:
        x, y = np.round(s.tracks[t, pid["mid50"]]).astype(int)
        m = cv2.dilate(GT[t]["mask"].astype(np.uint8), np.ones((7, 7), np.uint8))
        inside += int(0 <= x < W and 0 <= y < H and m[y, x] > 0)
print(f"  midline:0.5 inside the body on {inside}/{T} frames")
assert inside >= 0.9 * T
# feet: derived extremities near the GT feet (which swing) on most frames
hits = []
for t in frames:
    d = []
    for nm in ("foot_FR", "foot_HL", "foot_HR"):
        p = s.tracks[t, pid[nm]]
        if np.isfinite(p).all():
            d.append(np.min(np.linalg.norm(GT[t]["feet"] - p, axis=1)))
    hits.append(sum(1 for v in d if v < 14))
hits = np.array(hits)
print(f"  feet within 14 px: mean {hits.mean():.2f} of 3, >=2 on {(hits >= 2).mean():.0%} of frames")
assert (hits >= 2).mean() >= 0.6, "extremity landmarks do not find the feet"
# off-body demotion: the "foot" seeded on the background must read as low confidence
rock_conf = s.confidence[1:, pid_rock]
print(f"  background 'foot' confidence: median {np.median(rock_conf):.2f}, frames <= 0.2: {(rock_conf <= 0.2).mean():.0%}")
assert (rock_conf <= 0.2).mean() > 0.8, "off-body point not demoted"
# midline stored for display/export
assert len(s.masks.midline) >= 0.9 * T and s.masks.midline[10].shape == (32, 2)
# a real skeleton template applies on top without disturbing data
n_before = s.n_points
s.apply_skeleton(template_by_name("Lizard / iguana"))
assert s.n_points > n_before and s.tracked[:, pid_head].sum() == head_e[2]

# ---- 1b. landmark identity + derived continuity (lessons from real footage) -----
from cotracker_app.tracker import ANCHOR_LOST_CONF, EXT_CONF_CAP, OFF_BODY_CONF  # noqa: E402
# silhouette feet are never as certain as a tracked point
for nm in ("foot_FR", "foot_HL", "foot_HR"):
    c = s.confidence[s.tracked[:, pid[nm]], pid[nm]]
    assert len(c) and (c <= EXT_CONF_CAP + 1e-6).all(), f"{nm}: extremity confidence above the cap ({c.max():.2f})"
# a head anchor clicked MID-BODY is detected (the anchored midline is a half-midline):
# the run falls back to the body's own diameter and caps the derived confidence
sa = TrackingSession(VID, T, 25.0, W, H)
pa_head = sa.add_point(0, float(g0["centre"][0]), float(g0["centre"][1]), name="snout")   # wrong: mid-body
sa.ensure_animal()
sa.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
pa_tip = sa.add_landmark("tail_tip", "silhouette", "tip")
pa_mid = sa.add_landmark("mid50", "silhouette", "midline:0.5")
wa, eva = run_worker(sa, 0, 40, [PointSpec(pa_head, sa.tracks[0, pa_head].astype(np.float32).copy())],
                     AnimalSpec({0: list(sa.animal.prompts[0])}, {}, None),
                     [DerivedSpec(pa_tip, "tip"), DerivedSpec(pa_mid, "midline:0.5")], head_pid=pa_head)
flagged = sorted(f for f in wa._anchor_lost if f < 40)
tip_conf = sa.confidence[1:40, pa_tip][sa.tracked[1:40, pa_tip]]
tip_err = err_stats(sa, pa_tip, "tip", range(1, 40))
print(f"  mid-body anchor: flagged on {len(flagged)}/40 frames, tail-tip conf max {tip_conf.max():.2f}, "
      f"tail tip err mean {tip_err[0]:.1f} px (still the real tail)")
assert len(flagged) >= 30, "a mid-body head anchor must be detected as lost"
assert (tip_conf <= ANCHOR_LOST_CONF + 1e-6).all(), "derived confidence not capped on anchor-lost frames"
assert tip_err[0] < 15.0, "with the anchor distrusted the tail tip must still come from the whole body"
# a constrained landmark that LEAVES the segment stops the run at that frame:
# nothing is invented for it, the other landmarks keep their data, and the
# worker names the frame and the point (the merged-landmark rule itself is unit-tested
# in verify_onbody_rules.py, where positions can be dictated)
sc = TrackingSession(VID, T, 25.0, W, H)
pc_body = sc.add_point(0, float(g0["centre"][0]), float(g0["centre"][1]), name="body")
far = np.array([W * 0.5, H * 0.08])                         # background, far from animal 0
pc_a = sc.add_point(0, float(far[0]), float(far[1]), name="eye")
sc.ensure_animal()
sc.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
wc, evc = run_worker(sc, 0, 60, [PointSpec(p, sc.tracks[0, p].astype(np.float32).copy()) for p in (pc_body, pc_a)],
                     AnimalSpec({0: list(sc.animal.prompts[0])}, {}, None), [],
                     on_body=[pc_body, pc_a], constrain=[pc_body, pc_a], autopause=True)
print(f"  exit: autopaused={evc['autopaused']} reason={wc._autopause_reason!r}, stray tracked on "
      f"{int(sc.tracked[:, pc_a].sum())} frames, body tracked on {int(sc.tracked[:, pc_body].sum())} frames")
assert evc["autopaused"] is not None and evc["autopaused"][1] == pc_a, "the stray landmark must stop the run"
f_exit = evc["autopaused"][0]
assert 1 <= f_exit <= 3 and wc._autopause_reason == "exit", (f_exit, wc._autopause_reason)
assert not sc.tracked[f_exit:, pc_a].any(), "nothing may be invented for the landmark after it left"
assert sc.tracked[:f_exit, pc_a].all()
assert sc.tracked[:, pc_body].sum() >= f_exit + 1, "the other landmark keeps its data to the end of the window"
assert sc.tracked[:, pc_body].sum() <= 24, "the run must stop within the window of the exit"
print("landmark identity + derived continuity OK")

# ---- 2. resume from a stored mask (no prompt on the start frame) ---------------
s2 = TrackingSession(VID, T, 25.0, W, H)
s2.ensure_animal()
ww, wh = working_size(W, H)
s2.masks.set_from_work(80, cv2.resize(GT[80]["mask"].astype(np.uint8), (ww, wh),
                                      interpolation=cv2.INTER_NEAREST), W / ww, 9.0)
assert s2.animal_seedable_at(80) and not s2.animal_seedable_at(81)
pid_tip2 = s2.add_landmark("tail_tip", "silhouette", "tip")
seed_mask = s2.masks.rasterize(80, wh, ww)
w2, ev2 = run_worker(s2, 80, 140, [], AnimalSpec({}, {}, seed_mask), [DerivedSpec(pid_tip2, "tip")])
assert ev2["finished"] == (139, False), ev2["finished"]
tip2 = err_stats(s2, pid_tip2, "tip", range(80, 140))
print(f"resume from mask @80 (animal only, no points): masks {s2.masks.n_masked()} frames, "
      f"tail tip err mean {tip2[0]:.2f} px on {tip2[2]} frames, {60/ev2['dt']:.1f} fps")
assert s2.masks.n_masked() >= 58 and tip2[0] < 10.0 and tip2[2] >= 55

# ---- 2a. on-body constraint: a point cannot leave the silhouette ---------------
s2a = TrackingSession(VID, T, 25.0, W, H)
s2a.ensure_animal()
s2a.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
p_edge = s2a.add_point(0, float(g0["centre"][0]) + 20.0, float(g0["centre"][1]) - 24.0, name="edge")  # on the body edge
p_free = s2a.add_point(0, W * 0.5, H * 0.12, name="rock")  # exempt: stays where it is tracked
specs2a = [PointSpec(p, s2a.tracks[0, p].astype(np.float32).copy()) for p in (p_edge, p_free)]
w2a, ev2a = run_worker(s2a, 0, 120, specs2a, AnimalSpec({0: list(s2a.animal.prompts[0])}, {}, None), [],
                       constrain=[p_edge], autopause=True)
inside = {p: 0 for p in (p_edge, p_free)}; n = {p: 0 for p in inside}
for t in range(1, 120):
    m = cv2.dilate(GT[t]["mask"].astype(np.uint8), np.ones((5, 5), np.uint8))
    for p in inside:
        if s2a.tracked[t, p]:
            x, y = np.round(s2a.tracks[t, p]).astype(int)
            n[p] += 1
            inside[p] += int(0 <= x < W and 0 <= y < H and m[y, x] > 0)
print(f"on-body constraint: edge point inside {inside[p_edge]}/{n[p_edge]}, free point inside "
      f"{inside[p_free]}/{n[p_free]} frames; autopaused={ev2a['autopaused']}")
# an edge point jitters a few pixels across the outline every frame: that is nudged, never an exit
assert ev2a["autopaused"] is None, "outline jitter must not stop the run"
assert n[p_edge] >= 110 and inside[p_edge] >= 0.98 * n[p_edge], "constrained edge point left the body"
assert inside[p_free] <= 0.2 * max(n[p_free], 1), "free point must not be constrained"

# ---- 2b. animal only, NO landmark points at all (regression: zero-column chunks) ----
s2b = TrackingSession(VID, T, 25.0, W, H)
s2b.ensure_animal()
s2b.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
w2b, ev2b = run_worker(s2b, 0, 40, [], AnimalSpec({0: list(s2b.animal.prompts[0])}, {}, None), [])
assert ev2b["finished"] == (39, False) and ev2b["chunks"] >= 5, ev2b
assert s2b.masks.n_masked() >= 38 and s2b.n_points == 0
print(f"animal only, no landmarks: {ev2b['chunks']} zero-column chunks written, masks on {s2b.masks.n_masked()}/40 frames")

# ---- 3. animal-lost auto-pause: the animal leaves the scene at frame 40 --------
VID3 = _synth.build_video(os.path.join(OUT, "synth_vanish.mp4"), 100, hide0_after=40)
s3 = TrackingSession(VID3, 100, 25.0, W, H)
s3.ensure_animal()
s3.animal.add_click(0, float(g0["centre"][0]), float(g0["centre"][1]), True)
pid_c3 = s3.add_landmark("centre", "silhouette", "centroid")
w3_ = TrackingWorker(VID3, 0, None, None, FrameCache(128 * 1024 ** 2), 100, specs=[], autopause=True,
                     animal=AnimalSpec({0: list(s3.animal.prompts[0])}, {}, None),
                     derived=[DerivedSpec(pid_c3, "centroid")])
ev3 = {"error": None, "finished": None, "autopaused": None}
w3_.masks_ready.connect(lambda summ: s3.write_mask_summaries(summ))
w3_.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: s3.write_segment(w0, tr, vi, w3_.point_ids, cf))
w3_.finished_ok.connect(lambda last, p: ev3.update(finished=(last, p)))
w3_.autopaused.connect(lambda f, pid: ev3.update(autopaused=(f, pid)))
w3_.error.connect(lambda m: ev3.update(error=m))
w3_.run()
assert ev3["error"] is None, ev3["error"]
present = s3.masks.area > 0
print(f"vanishing animal: autopaused={ev3['autopaused']} finished={ev3['finished']}, present frames "
      f"{int(present[:40].sum())}/40 before, {int(present[40:60].sum())}/20 after")
assert present[:40].mean() > 0.9, "animal must be present before it vanishes"
assert ev3["autopaused"] is not None and ev3["autopaused"][1] == -1 and ev3["finished"][1] is True, \
    "auto-pause must report the animal lost after it leaves"
assert 38 <= ev3["autopaused"][0] <= 70, ev3["autopaused"]

# ---- 4. persistence (schema v3) and exports -----------------------------------
p = os.path.join(OUT, "animal_session.cotrk")
s.save_npz(p)
r = TrackingSession.load_npz(p)
assert r.animal is not None and r.animal.prompts == s.animal.prompts
assert r.masks.n_masked() == s.masks.n_masked() and r.skeleton["name"] == "Lizard / iguana"
assert [pt.source for pt in r.points] == [pt.source for pt in s.points]
assert np.array_equal(r.masks.bbox, s.masks.bbox) and len(r.masks.midline) == len(s.masks.midline)
assert np.allclose(r.masks.midline[10], s.masks.midline[10])
assert r.masks.rasterize(10, H, W).sum() > 0 and _synth.iou(r.masks.rasterize(10, H, W), GT[10]["mask"]) > 0.8
print("session v3 round-trip OK")
# undo snapshot restores masks too
snap = s.snapshot()
s.clear_masks(0, 50)
assert s.masks.n_masked() < snap.masks.n_masked()
s.restore(snap)
assert s.masks.n_masked() == snap.masks.n_masked()

base = os.path.join(OUT, "animal_export")
s.export_csv(base + ".csv")
s.export_dlc_csv(base + "_dlc.csv")
s.export_dltdv_csv(base + "_dltdv.csv")
s.export_animal_csv(base + "_animal.csv")
s.export_mat(base + ".mat")
with open(base + "_dlc.csv", newline="") as f:
    rows = list(csv.reader(f))
assert rows[0][0] == "scorer" and rows[1][0] == "bodyparts" and rows[2][0] == "coords"
assert rows[1][1] == s.points[0].name and rows[2][1:4] == ["x", "y", "likelihood"]
assert len(rows) == 3 + T and len(rows[3]) == 1 + 3 * s.n_points
with open(base + "_dltdv.csv", newline="") as f:
    rows = list(csv.reader(f))
assert rows[0][0] == "pt1_cam1_X" and len(rows) == 1 + T and len(rows[1]) == 2 * s.n_points
xy = rows[1][2 * pid_head:2 * pid_head + 2]
# DLTdv8 convention: top-left origin, first pixel = 1
assert abs(float(xy[0]) - (s.tracks[0, pid_head, 0] + 1)) < 1e-2 and abs(float(xy[1]) - (s.tracks[0, pid_head, 1] + 1)) < 1e-2
assert os.path.exists(base + "_dltdv_pointnames.csv")
with open(base + "_animal.csv", newline="") as f:
    rows = list(csv.reader(f))
assert rows[0][:3] == ["frame", "present", "score"] and len(rows) == 1 + T
assert sum(int(r_[1]) for r_ in rows[1:]) == s.masks.n_masked()
assert len(rows[0]) == 10 + 64
from scipy.io import loadmat
m = loadmat(base + ".mat")
assert m["segment_midline"].shape == (T, 32, 2) and m["segment_present"].sum() == s.masks.n_masked()
assert "skeleton_bones" in m and m["point_source"].shape[-1] == s.n_points
print("exports OK: wide/DLC/DLTdv/animal CSV + .mat")
print("ALL ANIMAL CHECKS PASSED")
