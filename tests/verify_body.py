"""Body layer, core: joint angles against ground truth, storage, exports, the
backend registry and the side-by-side drawing. No GPU, no weights, no network.

The spine of this suite is `tests/_synth_human.py`, which BUILDS a walking
pose out of chosen joint angles. Recovering those same numbers from the
resulting 3D joints is a real check of the measurement code, not a comparison
of one estimate against another.

Run:  .venv\\Scripts\\python.exe tests\\verify_body.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "tests" / "out"
OUT.mkdir(parents=True, exist_ok=True)

import _synth_human as sh                                   # noqa: E402
from kinetrace import body                              # noqa: E402

fails: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + what + ((" -- " + detail) if detail else ""))
    if not ok:
        fails.append(what)


def rodrigues(v, k, deg):
    k = np.asarray(k, float)
    k = k / np.linalg.norm(k)
    th = np.radians(deg)
    return (v * np.cos(th) + np.cross(k, v) * np.sin(th)
            + k * np.dot(k, v) * (1 - np.cos(th)))


def fill(rig, names, arr, n_frames):
    """(T, J, D) for `rig` from the synthetic truth's own name list."""
    out = np.full((n_frames, rig.n_joints, arr.shape[-1]), np.nan)
    for k, nm in enumerate(names):
        i = rig.index(nm)
        if i is not None:
            out[:, i] = arr[:, k]
    return out


# ------------------------------------------------------------------ 1. rigs
print("\n[1] rigs and joint names")
mhr = body.rig_of("mhr70")
coco = body.rig_of("coco17")
check(mhr.n_joints == 70, "MHR rig has 70 joints", str(mhr.n_joints))
check(coco.n_joints == 17, "COCO rig has 17 joints", str(coco.n_joints))
check(body.canon("left-big-toe-tip") == "left_big_toe", "hyphen/alias spelling folded")
check(body.canon("L_Shoulder") == "left_shoulder", "ViTPose L_/R_ spelling folded")
check(mhr.index("left_wrist") == 62 and mhr.index("neck") == 69,
      "MHR emission order preserved (wrist 62, neck 69)")
check(len(mhr.bones()) >= 20 and len(coco.bones()) >= 15, "both rigs resolve bones")
check(all(0 <= i < mhr.n_joints and 0 <= j < mhr.n_joints for i, j in mhr.bones()),
      "every bone index is inside the rig")
# The synthetic truth must be expressible in the MHR rig, or the suite below
# would be testing a subset without saying so.
gtn = sh.truth(1)["names"]
check(all(mhr.index(n) is not None for n in gtn), "every synthetic joint exists in MHR",
      str([n for n in gtn if mhr.index(n) is None]))

# -------------------------------------------------- 2. angles vs ground truth
print("\n[2] joint angles recovered from the poses they built")
gt = sh.truth()
X = fill(mhr, gt["names"], gt["xyz"], sh.T)
defs, ang = body.joint_angles(X, mhr, have_3d=True)
lut = {d.name: k for k, d in enumerate(defs)}
worst = 0.0
for name, truth_vals in gt["angles"].items():
    # "trunk lean" is measured against the subject's own median upright, not
    # against a fixed axis, so the number to match is the generator's lean
    # with its own median taken out. Everything else is absolute.
    key = "trunk lean" if name == "trunk lean from vertical" else name
    want = truth_vals - np.median(truth_vals) if key == "trunk lean" else truth_vals
    if key not in lut:
        check(False, f"{key} is computed")
        continue
    err = float(np.nanmax(np.abs(ang[:, lut[key]] - want)))
    worst = max(worst, err)
    check(err < 1e-6, f"{key} within 1e-6 deg", f"max err {err:.2e}")
check(worst < 1e-6, "every ground-truth angle exact", f"worst {worst:.2e} deg")

# (I87) The zero each angle claims must be where the angle reads zero. The
# walker's head is built on the trunk axis (ears over the shoulders), so an
# upright head must read 0 -- the nose-based definition read +19.7. And the
# stride angle must not include the hip width: it read ~32 deg with the thighs
# side by side; it now equals left minus right hip flexion exactly.
neck = ang[:, lut["neck flexion"]]
check(float(np.nanmax(np.abs(neck))) < 1e-6, "neck flexion reads 0 for a head in line with the trunk",
      f"max {float(np.nanmax(np.abs(neck))):.2e} deg")
lr_hip = gt["angles"]["left hip flexion"] - gt["angles"]["right hip flexion"]
check("thigh separation (stride)" in lut and "thigh-shank separation (stride)" not in lut,
      "the stride angle is named for what it measures (thigh to thigh)")
st = ang[:, lut["thigh separation (stride)"]]
check(float(np.nanmax(np.abs(st - lr_hip))) < 1e-6,
      "stride = left minus right hip flexion, so thighs side by side read 0",
      f"max err {float(np.nanmax(np.abs(st - lr_hip))):.2e} deg")
check(st[np.argmax(lr_hip)] > 30, "and it is positive with the left knee ahead",
      f"{st[np.argmax(lr_hip)]:.1f} deg")

# The ankle uses the foot's REAL axis (heel to toe). Measured from the ankle
# to the toe tip instead, a neutral standing foot lands on atan2's branch cut
# and the value jumps by 360 -- that is what the real MHR rig did.
ankle = ang[:, lut["left ankle dorsiflexion"]]
check(np.abs(ankle).max() < 45, "ankle dorsiflexion stays in an anatomical range",
      f"{ankle.min():.1f} .. {ankle.max():.1f} deg")
check(np.abs(np.diff(ankle)).max() < 20, "and never jumps by 360 between frames",
      f"biggest step {np.abs(np.diff(ankle)).max():.1f} deg")
# trunk lean must be camera-independent: rotate the whole clip and it must not move
Rz = np.array([[np.cos(0.7), -np.sin(0.7), 0], [np.sin(0.7), np.cos(0.7), 0], [0, 0, 1]])
Rx = np.array([[1, 0, 0], [0, np.cos(0.9), -np.sin(0.9)], [0, np.sin(0.9), np.cos(0.9)]])
_, ang_rot = body.joint_angles(X @ (Rx @ Rz).T, mhr, have_3d=True)
lean_i = lut["trunk lean"]
check(float(np.nanmax(np.abs(ang_rot[:, lean_i] - ang[:, lean_i]))) < 1e-6,
      "trunk lean is unchanged by tilting the camera",
      f"{float(np.nanmax(np.abs(ang_rot[:, lean_i] - ang[:, lean_i]))):.2e} deg")
check(float(np.nanmax(np.abs(ang_rot[:, lut['left knee flexion']]
                            - ang[:, lut['left knee flexion']]))) < 1e-6,
      "and so is every limb angle")

# the signed angles must actually go negative -- an unsigned measurement would
# pass a "close to truth" test only if the truth never changed sign
hipL = ang[:, lut["left hip flexion"]]
check(hipL.min() < -15 and hipL.max() > 15,
      "hip flexion is signed (flexion AND extension)", f"{hipL.min():.1f} .. {hipL.max():.1f}")

# twist sign convention, stated in body.AngleDef, checked with an explicit rotation
for want in (0.0, 30.0, -25.0, 60.0):
    x = np.full((1, coco.n_joints, 3), np.nan)
    x[0, coco.index("left_hip")] = [-0.1, 0, 0]
    x[0, coco.index("right_hip")] = [0.1, 0, 0]
    n = np.array([0.0, -1.0, 0.0])                # pelvis -> shoulders
    s = rodrigues(np.array([-0.2, 0.0, 0.0]), n, want)
    x[0, coco.index("left_shoulder")] = n + s / 2
    x[0, coco.index("right_shoulder")] = n - s / 2
    d2, a2 = body.joint_angles(x, coco, have_3d=True)
    got = a2[0, [q.name for q in d2].index("shoulder-hip twist")]
    check(abs(got - want) < 1e-6, f"twist {want:+.0f} deg by the documented rule",
          f"got {got:.4f}")

print("\n[3] angles in the image plane, and what they cost")
U = fill(mhr, gt["names"], gt["uv"], sh.T)
d2d, ang2d = body.joint_angles(U, mhr, have_3d=False)
lut2 = {d.name: k for k, d in enumerate(d2d)}
check("shoulder-hip twist" not in lut2, "a 3D-only angle is not offered in 2D")
neck2 = ang2d[:, lut2["neck flexion"]]
check(float(np.nanmax(np.abs(neck2))) < 1.0, "neck flexion reads ~0 upright in the image too (I87)",
      f"max {float(np.nanmax(np.abs(neck2))):.2f} deg")
st2 = ang2d[:, lut2["thigh separation (stride)"]]
check(float(np.nanmax(np.abs(st2 - lr_hip))) < 1.0,
      "stride matches left minus right hip flexion in the image (I87)",
      f"max err {float(np.nanmax(np.abs(st2 - lr_hip))):.3f} deg")
for name in ("left knee flexion", "right hip flexion", "left elbow flexion"):
    diff = float(np.nanmean(np.abs(ang2d[:, lut2[name]] - gt["angles"][name])))
    check(diff < 1.0, f"{name} in the image is within 1 deg of the truth (side-on)",
          f"mean {diff:.3f} deg")
# a rig that lacks a joint drops the angle rather than emitting NaN columns
dc, _ = body.joint_angles(np.full((3, coco.n_joints, 3), np.nan), coco, have_3d=True)
check(not any("ankle" in d.name for d in dc), "ankle angles dropped on a rig with no toes")
check(any("knee" in d.name for d in dc), "knee angles kept on COCO")

print("\n[4] degenerate input")
blank = np.full((5, mhr.n_joints, 3), np.nan)
_, ab = body.joint_angles(blank, mhr, have_3d=True)
check(not np.isfinite(ab).any(), "all-missing joints give all-NaN angles")
collapsed = np.zeros((1, mhr.n_joints, 3))
_, ac = body.joint_angles(collapsed, mhr, have_3d=True)
check(not np.isfinite(ac).any(), "a collapsed skeleton gives NaN, not 0 deg")
rom = body.range_of_motion(ab)
check(rom["range"].shape == (ab.shape[1],) and not np.isfinite(rom["range"]).any(),
      "range of motion of nothing is NaN, and does not warn")
v = body.angular_velocity(np.arange(6.0).reshape(6, 1), 10.0)
check(np.allclose(v, 10.0), "angular velocity of a 1 deg/frame ramp at 10 fps is 10 deg/s")
# (I90) a run sampled every 4th frame: its neighbours are 4 frames apart, and
# differencing adjacent video frames gave it no velocity at all
ramp = np.full((40, 1), np.nan)
ramp[::4, 0] = np.arange(0, 40, 4, dtype=float)              # 1 deg per video frame
v4 = body.angular_velocity(ramp, 10.0, step=4)
check(int(np.isfinite(v4).sum()) == 10 and np.allclose(v4[::4], 10.0),
      "a 1-in-4 sampled ramp gives 10 deg/s on every sampled frame",
      f"{int(np.isfinite(v4).sum())} finite")
check(not np.isfinite(body.angular_velocity(ramp, 10.0)).any(),
      "(without the step it would have none -- the bug)")
vg = body.angular_velocity(np.array([[0.0], [np.nan], [2.0]]), 10.0)
check(not np.isfinite(vg[1, 0]), "no velocity on a frame that has no angle")

# ----------------------------------------------------------- 5. BodyTrack
print("\n[5] BodyTrack storage")
track = body.BodyTrack(sh.T, mhr, 2)
track.backend = "unit test"
for t in range(sh.T):
    track.set_person(t, 0, joints3d=X[t], joints2d=U[t],
                     conf=np.ones(mhr.n_joints), score=0.9, bbox=[1, 2, 3, 4], focal=900.0)
for t in range(40, 60):
    track.set_person(t, 1, joints2d=U[t] + 30, conf=np.ones(mhr.n_joints) * 0.6, score=0.5)
check(track.n_posed() == sh.T, "frames with a person counted", str(track.n_posed()))
check(track.people_at(10) == [0] and track.people_at(50) == [0, 1],
      "people present per frame")
check(track.has_3d and track.pose3d(5, 0) is not None and track.pose3d(50, 1) is None,
      "3D is returned only where it exists")
check(len(track.frames(1)) == 20, "per-person frame list", str(len(track.frames(1))))

npz = OUT / "body_track.npz"
np.savez_compressed(npz, **track.to_arrays("b_"))
with np.load(npz, allow_pickle=True) as z:
    back = body.BodyTrack.from_arrays("b_", z)
check(back.n_people == 2 and back.rig.name == "mhr70" and back.backend == "unit test",
      "npz round trip keeps rig, people and backend")
check(np.allclose(back.joints3d, track.joints3d, equal_nan=True)
      and np.allclose(back.joints2d, track.joints2d, equal_nan=True)
      and np.allclose(back.conf, track.conf), "npz round trip is bit-faithful")
d3, a3 = back.angles(0)
check(float(np.nanmax(np.abs(a3[:, [d.name for d in d3].index("left knee flexion")]
                            - gt["angles"]["left knee flexion"]))) < 1e-4,
      "angles survive the round trip")
n_cleared = track.clear(0, 9)
check(n_cleared == 10 and track.n_posed() == sh.T - 10 and not track.has(3),
      "clearing a window blanks exactly that window")
cp = back.copy()
cp.clear(0, 200)
check(back.n_posed() == sh.T, "copy() is deep")

# ------------------------------------------------------------- 6. exports
print("\n[6] exports")
jcsv, acsv = OUT / "body_joints.csv", OUT / "body_angles.csv"
body.export_joints_csv(back, jcsv)
body.export_angles_csv(back, acsv, sh.FPS)
jt_all = jcsv.read_text(encoding="utf-8").splitlines()
at = acsv.read_text(encoding="utf-8").splitlines()
jcom = [l for l in jt_all if l.startswith("#")]
jt = [l for l in jt_all if not l.startswith("#")]
check(jt_all[:len(jcom)] == jcom, "the joint CSV's '#' notes all come before the header")
check(jt[0].startswith("frame,person,nose_x,nose_y,nose_conf"), "joint CSV header", jt[0][:60])
check("nose_X" in jt[0], "joint CSV carries 3D columns when the backend has 3D")
check(len(jt) - 1 == sh.T + 20, "one joint row per frame and person", str(len(jt) - 1))
check(any("top-left" in l for l in jcom) and any("CAMERA's frame" in l for l in jcom),
      "the joint CSV states the pixel origin and the 3D frame")
head = [l for l in at if l.startswith("#")]
check(any("left knee flexion" in l and "straight" in l for l in head),
      "angle CSV states the convention for each column")
hdr = next(l for l in at if l.startswith("frame,"))
# (I92) csv.writer quoted every comment containing a comma, so it began '"#'
# and comment-aware readers took it for data or for the header
pre = at[:at.index(hdr)]
check(len(pre) >= 5 and all(l.startswith("# ") for l in pre),
      "every line above the angle CSV header is a plain '#' comment",
      f"{sum(not l.startswith('# ') for l in pre)} of {len(pre)} are not")
import csv as _csv                                           # noqa: E402
rows_ = list(_csv.reader(l for l in at if not l.startswith("#")))
check(rows_[0][0] == "frame" and len({len(r) for r in rows_}) == 1,
      "skipping '#' lines leaves one header and equal-width rows",
      f"widths {sorted({len(r) for r in rows_})}")
check("time_s" in hdr and "deg/s" in hdr, "angle CSV has time and rate columns")
check(all(c in hdr for c in ("left hip flexion", "shoulder-hip twist")),
      "angle CSV lists the measured angles")
rep = body.angle_report(back, sh.FPS)
check("Verdict" in rep and "good" in rep, "report ends in a plain-language verdict")
check("range of motion" in rep and "What 0 means" in rep,
      "report gives range of motion and the conventions")
# the reported ROM must agree with the generator
check(abs(float(np.nanmax(a3[:, [d.name for d in d3].index("left shoulder flexion")])) - 72.0) < 1e-4,
      "reported peak shoulder flexion matches the generator", "72 deg")
# a 2D-only track must SAY the angles are image-plane
flat = body.BodyTrack(sh.T, coco, 1)
for t in range(sh.T):
    flat.set_person(t, 0, joints2d=fill(coco, gt["names"], gt["uv"], sh.T)[t],
                    conf=np.ones(coco.n_joints), score=0.8)
r2 = body.angle_report(flat, sh.FPS)
check("not in space" in r2 and "2D (image plane)" in r2,
      "a 2D result warns that its angles are image-plane")

# --------------------------------------------------- 7. backend registry
print("\n[7] backend registry (no network)")
from kinetrace import bodypose                          # noqa: E402
states = {k: bodypose.backend_status(k)[0] for k in bodypose.BACKENDS}
check(set(bodypose.BACKENDS) >= {"sam-3d-body-dinov3", "sam-3d-body-vith", "vitpose-base"},
      "SAM 3D Body and the 2D fallback are both registered")
check(all(s in ("ready", "download", "needs-code", "needs-weights") for s in states.values()),
      "every backend reports a known state", str(states))
for k, spec in bodypose.BACKENDS.items():
    why = bodypose.backend_status(k)[1]
    check(len(why) > 30 and (states[k] in ("ready", "download") or "http" in why),
          f"{k} explains itself and says where to get it", why[:70])
check(bodypose.BACKENDS["sam-3d-body-dinov3"].rig == "mhr70"
      and bodypose.BACKENDS["vitpose-base"].rig == "coco17", "backends name their rig")
check(bodypose.preferred_backend() in bodypose.BACKENDS, "a preferred backend is choosable")
try:
    bodypose.make_estimator("no-such-backend")
    check(False, "an unknown backend raises")
except RuntimeError:
    check(True, "an unknown backend raises RuntimeError")

print("\n[8] person identity over time")
def person(box):
    return bodypose.PersonPose(joints2d=np.zeros((17, 2)), conf=np.ones(17),
                               bbox=np.asarray(box, np.float32))
m = bodypose.PersonMatcher(2)
check(m.assign([person([0, 0, 10, 20]), person([100, 0, 110, 20])]) == [0, 1],
      "two people take two columns")
check(m.assign([person([101, 0, 111, 20]), person([1, 0, 11, 20])]) == [1, 0],
      "columns follow the person, not the detection order")
m.assign([person([102, 0, 112, 20])])
check(m.assign([person([2, 0, 12, 20]), person([103, 0, 113, 20])]) == [0, 1],
      "a person who vanishes for a frame returns to their own column")
# (I84) the column of a person missed for a frame is RESERVED: a newcomer goes
# to the unused column, and a stranger is never written in as the subject
Pb, Rb = [0, 0, 100, 200], [600, 0, 700, 200]
m = bodypose.PersonMatcher(2)
seq = [m.assign([person(Pb)]), m.assign([person(Rb)]), m.assign([person(Pb), person(Rb)])]
check(seq == [[0], [1], [0, 1]], "a newcomer takes the empty column, not the missed person's",
      str(seq))
m1 = bodypose.PersonMatcher(1)
m1.assign([person(Pb)])
check(m1.assign([person(Rb)]) == [-1],
      "with one column, a far-away box the frame the subject is missed goes nowhere")
check(m1.assign([person([5, 0, 105, 200])]) == [0], "and the subject comes back to their column")
m2 = bodypose.PersonMatcher(2)
m2.assign([person([0, 0, 100, 200])], frame=0)
check(m2.assign([person([400, 0, 500, 200])], frame=16) == [0],
      "a sampled run (16 frames apart) keeps a fast mover in their own column")
m3 = bodypose.PersonMatcher(1, patience=3)
m3.assign([person(Pb)])
for _ in range(4):
    m3.assign([])
check(m3.assign([person(Rb)]) == [0], "a column empty for longer than its patience is free again")
check(bodypose.mask_bbox(np.zeros((10, 10), bool)) is None, "an empty mask has no box")
mk = np.zeros((10, 10), bool)
mk[2:5, 3:8] = True
check(list(bodypose.mask_bbox(mk)) == [3, 2, 7, 4], "mask box is tight")

# ------------------------------------------------------------ 9. drawing
print("\n[9] side-by-side drawing")
import cv2                                                  # noqa: E402
from kinetrace import bodyview                          # noqa: E402

frame = sh.render_frame(70)
opts = bodyview.PoseDrawOptions(angles_shown=("left knee flexion", "right knee flexion"))
drawn = bodyview.draw_pose(frame, back, 70, opts)
check(drawn.shape == frame.shape, "draw_pose keeps the frame size")
check(int(np.abs(drawn.astype(int) - frame.astype(int)).sum()) > 100000,
      "draw_pose actually marks the frame")
blank_track = body.BodyTrack(sh.T, mhr, 1)                # never given a pose
untouched = bodyview.draw_pose(frame, blank_track, 5, opts)
check(np.array_equal(untouched, frame), "a frame with no pose is left alone")
check(np.array_equal(frame, sh.render_frame(70)),
      "draw_pose does not scribble on the caller's frame")

panel = bodyview.render_pose_panel(back, 70, 0, (400, 380))
check(panel.shape == (380, 400, 3), "pose panel size")
# the figure must be the right way up and inside the panel
R = bodyview._view_rotation(25.0, 12.0, up=1) @ bodyview._up_basis(mhr.up)
pts = back.joints3d[70, 0].astype(float)
ok = np.isfinite(pts).all(axis=1)
seen = pts[ok] @ R.T
c = (seen.min(0) + seen.max(0)) / 2
span_x, span_y = bodyview._stable_span(back.joints3d[:, 0], R)
sc = 0.78 * min(400 / span_x, 380 / span_y)
scr_y = 190 - (pts[ok] @ R.T - c)[:, 1] * sc
head_i = list(np.nonzero(ok)[0]).index(mhr.index("nose"))
ank_i = list(np.nonzero(ok)[0]).index(mhr.index("left_ankle"))
check(scr_y[head_i] < scr_y[ank_i], "the 3D panel draws the head above the ankles")
check(scr_y.min() > -1 and scr_y.max() < 381, "the whole figure fits inside the panel",
      f"rows {scr_y.min():.0f}..{scr_y.max():.0f}")


def drawn_extent(img):
    """(rows, cols) of the panel the skeleton actually covers, measured from
    the pixels. Checked directly rather than re-derived, because the failure
    this guards against -- a figure filling half a tall panel on portrait
    footage -- is a property of the picture, not of the arithmetic."""
    bg = np.array(bodyview.BG_PANEL, np.int16)
    ink = (np.abs(img.astype(np.int16) - bg).max(axis=2) > 24)
    ink[:28] = False            # the caption line at the top
    ink[-34:] = False           # the left/right legend at the bottom
    rows = np.nonzero(ink.any(axis=1))[0]
    cols = np.nonzero(ink.any(axis=0))[0]
    return ((rows[-1] - rows[0] + 1) if len(rows) else 0,
            (cols[-1] - cols[0] + 1) if len(cols) else 0)


ph, pw = drawn_extent(panel)
check(ph > 0.55 * 380, "the figure fills the panel's height", f"{ph} of 380 px")
# a tall (portrait) panel is where a single min(W, H) scale under-filled
tall = bodyview.render_pose_panel(back, 70, 0, (300, 700))
th_, tw_ = drawn_extent(tall)
check(th_ > 0.55 * 700, "and still fills a portrait panel", f"{th_} of 700 px")
check(tw_ <= 300, "without spilling out of it sideways", f"{tw_} of 300 px")

plot = bodyview.angle_plot(back, 0, 70, (900, 200), ["left knee flexion"], sh.FPS)
check(plot.shape == (200, 900, 3), "angle plot size")
check(len(np.unique(plot.reshape(-1, 3), axis=0)) > 4, "angle plot draws something")
sbs = bodyview.compose_side_by_side(frame, back, 70, 0, opts, True, 25, 12, sh.FPS)
check(sbs.shape[1] == frame.shape[1] * 2 and sbs.shape[0] > frame.shape[0],
      "side-by-side is two panels wide with the plot underneath", str(sbs.shape))
cv2.imwrite(str(OUT / "body_side_by_side.png"), sbs)
noplot = bodyview.compose_side_by_side(frame, back, 70, 0, opts, False)
check(noplot.shape[0] == frame.shape[0], "the plot can be turned off")
# a 2D-only track must still compose
sbs2 = bodyview.compose_side_by_side(frame, flat, 40, 0, bodyview.PoseDrawOptions(), True)
check(sbs2.shape[1] == frame.shape[1] * 2, "a 2D backend still gets a side-by-side")
empty = body.BodyTrack(sh.T, mhr, 1)
check(bodyview.compose_side_by_side(frame, empty, 3, 0).shape[1] == frame.shape[1] * 2,
      "an empty track composes without crashing")

# --------------------------------------------------- 10. session and undo
print("\n[10] session integration")
from kinetrace.session import SCHEMA_VERSION, TrackingSession      # noqa: E402
check(SCHEMA_VERSION >= 4, "session schema bumped for the body layer", str(SCHEMA_VERSION))
s = TrackingSession("walker.mp4", sh.T, sh.FPS, sh.W, sh.H)
check(not s.has_body() and s.body is None, "a fresh session has no body track")
bt = s.ensure_body(mhr, 1, "unit test")
for t in range(sh.T):
    bt.set_person(t, 0, joints3d=X[t], joints2d=U[t], conf=np.ones(mhr.n_joints), score=0.9)
check(s.has_body() and len(s.body_frames()) == sh.T, "session reports its body track")
again = s.ensure_body(mhr, 1)
check(again is bt, "ensure_body reuses a matching track")
swapped = s.ensure_body(coco, 1)
check(swapped is not bt and swapped.rig.name == "coco17",
      "a different rig replaces the track rather than mixing joint sets")
s.body = bt

snap = s.snapshot()
s.clear_body_window(0, 49)
check(s.body.n_posed() == sh.T - 50, "clearing a window through the session")
s.restore(snap)
check(s.body.n_posed() == sh.T, "undo restores the cleared window")
s.clear_body()
check(s.body is None, "clear_body removes it")
s.restore(snap)
check(s.body is not None and s.body.n_posed() == sh.T,
      "undo brings back a track that had been removed entirely")

p = OUT / "body_session.npz"
s.save_npz(p)
s2 = TrackingSession.load_npz(p)
check(s2.body is not None and s2.body.n_posed() == sh.T
      and s2.body.rig.name == "mhr70" and s2.body.has_3d,
      "session npz round trip keeps the body track")
check(np.allclose(s2.body.joints3d, s.body.joints3d, equal_nan=True),
      "session round trip is bit-faithful")
s3 = TrackingSession("other.mp4", sh.T, sh.FPS, sh.W, sh.H)
s3.save_npz(OUT / "body_none.npz")
check(TrackingSession.load_npz(OUT / "body_none.npz").body is None,
      "a session with no body track loads as None")
s.export_body_joints_csv(OUT / "sess_joints.csv")
s.export_body_angles_csv(OUT / "sess_angles.csv")
check((OUT / "sess_joints.csv").stat().st_size > 1000
      and (OUT / "sess_angles.csv").stat().st_size > 1000, "session exports write")

# ------------------------------------- 11. the SAM 3D Body adapter, on a stand-in
print("\n[11] SAM 3D Body adapter against a stand-in of Meta's API")
# The real checkpoints are gated and cannot be downloaded here, so the ADAPTER
# -- our code: how the release's output dict is turned into a PersonPose -- is
# checked against a module that reproduces the published API of
# sam_3d_body/sam_3d_body_estimator.py. This does NOT validate Meta's model;
# it validates the 60 lines of ours that sit in front of it.
import shutil                                                 # noqa: E402
import textwrap                                               # noqa: E402

STAND = OUT / "_s3db_standin"
if STAND.exists():
    shutil.rmtree(STAND)
(STAND / "sam_3d_body").mkdir(parents=True)
(STAND / "sam_3d_body" / "__init__.py").write_text(textwrap.dedent('''
    """Stand-in for Meta's sam_3d_body, reproducing only its public shape."""
    import numpy as np

    N_MHR = 308          # the release emits the full keypoint set
    N_VERT = 64
    SEEN = {}            # what the adapter handed us, for the test to inspect


    class _Head:
        faces = np.zeros((2, 3), np.int32)


    class _Model:
        device = "cpu"
        head_pose = _Head()


    def load_sam_3d_body(ckpt, device=None, mhr_path=""):
        SEEN["ckpt"], SEEN["mhr"], SEEN["device"] = str(ckpt), str(mhr_path), str(device)
        return _Model(), {"MODEL": {"IMAGE_SIZE": (256, 192)}}


    class SAM3DBodyEstimator:
        def __init__(self, sam_3d_body_model=None, model_cfg=None, human_detector=None,
                     human_segmentor=None, fov_estimator=None):
            self.faces = sam_3d_body_model.head_pose.faces

        def process_one_image(self, img, bboxes=None, masks=None, cam_int=None,
                              det_cat_id=0, bbox_thr=0.5, nms_thr=0.3, use_mask=False,
                              inference_type="full"):
                # upstream asserts this, so the adapter must always honour it
                if masks is not None:
                    assert bboxes is not None, "mask-conditioned inference needs bboxes"
                SEEN["img"] = np.asarray(img).copy()
                SEEN["bboxes"] = None if bboxes is None else np.asarray(bboxes).copy()
                SEEN["masks"] = None if masks is None else np.asarray(masks).copy()
                SEEN["cam_int"] = cam_int
                SEEN["bbox_thr"] = bbox_thr
                n = 1 if bboxes is None else len(np.asarray(bboxes).reshape(-1, 4))
                if cam_int is not None:
                    # exactly what upstream does with it (base_model.py:124,
                    # sam3d_body.py:1047): an unbatched (3, 3) raises here
                    import torch
                    ci = torch.as_tensor(cam_int)
                    ci.unsqueeze(1).expand(-1, n, -1, -1)
                    ci[:, None, None, None, [0, 1], [2, 2]]
                out = []
                for i in range(n):
                    kp3 = np.arange(N_MHR * 3, dtype=np.float32).reshape(N_MHR, 3) / 1000.0
                    kp2 = np.arange(N_MHR * 2, dtype=np.float32).reshape(N_MHR, 2) + 7.0
                    out.append({
                        "bbox": np.array([10.0, 20.0, 110.0, 220.0], np.float32),
                        "focal_length": np.array([1234.5], np.float32),
                        "pred_keypoints_3d": kp3,
                        "pred_keypoints_2d": kp2,
                        "pred_vertices": np.zeros((N_VERT, 3), np.float32),
                        "pred_cam_t": np.array([0.1, 0.2, 3.0], np.float32),
                        "pred_pose_raw": np.zeros(3, np.float32),
                        "global_rot": np.eye(3, dtype=np.float32),
                        "body_pose": np.zeros(3, np.float32),
                        "hand": np.zeros(3, np.float32),
                        "scale": np.zeros(1, np.float32),
                        "shape": np.zeros(3, np.float32),
                        "face": np.zeros(3, np.float32),
                        "mask": None,
                        "pred_joint_coords": np.zeros((70, 3), np.float32),
                        "pred_global_rots": np.zeros((70, 3, 3), np.float32),
                        "mhr_model_params": {},
                    })
                return out
    ''').lstrip(), encoding="utf-8")
# the marker `code_available` looks for, as in the real checkout
(STAND / "sam_3d_body" / "sam_3d_body_estimator.py").write_text(
    "from sam_3d_body import SAM3DBodyEstimator  # noqa: F401\n", encoding="utf-8")

WEIGHTS = OUT / "_s3db_weights"
if WEIGHTS.exists():
    shutil.rmtree(WEIGHTS)
(WEIGHTS / "sam-3d-body-dinov3" / "assets").mkdir(parents=True)
(WEIGHTS / "sam-3d-body-dinov3" / "model.ckpt").write_bytes(b"not a real checkpoint")
(WEIGHTS / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt").write_bytes(b"not a real asset")

_real_models, _real_repo = bodypose.MODELS_DIR, bodypose.S3DB_REPO

# With nothing installed the message must name the model and where to get it.
# Pointed at empty folders, so this holds whether or not the real SAM 3D Body
# happens to be present on this machine.
EMPTY = OUT / "_s3db_empty"
(EMPTY / "nothing").mkdir(parents=True, exist_ok=True)
try:
    bodypose.MODELS_DIR, bodypose.S3DB_REPO = EMPTY, EMPTY / "no-code"
    missing = bodypose.backend_status("sam-3d-body-dinov3")
    check(missing[0] in ("needs-code", "needs-weights"),
          "with nothing installed SAM 3D Body says so", missing[0])
    try:
        bodypose.make_estimator("sam-3d-body-dinov3")
        check(False, "make_estimator refuses when SAM 3D Body is not installed")
    except RuntimeError as exc:
        check("sam-3d-body" in str(exc) or "huggingface" in str(exc),
              "and the refusal says where to get it", str(exc)[:70])
    # the half-installed case: code present, checkpoint present, rig absent
    (EMPTY / "sam-3d-body-dinov3").mkdir(exist_ok=True)
    (EMPTY / "sam-3d-body-dinov3" / "model.ckpt").write_bytes(b"x")
    (EMPTY / "no-code" / "sam_3d_body").mkdir(parents=True, exist_ok=True)
    (EMPTY / "no-code" / "sam_3d_body" / "sam_3d_body_estimator.py").write_text("", encoding="utf-8")
    st, why = bodypose.backend_status("sam-3d-body-dinov3")
    check(st == "needs-weights" and "mhr_model.pt" in why and "model.ckpt is here" in why,
          "a checkpoint with no rig asset is reported as exactly that", why[:80])
finally:
    bodypose.MODELS_DIR, bodypose.S3DB_REPO = _real_models, _real_repo
    shutil.rmtree(EMPTY, ignore_errors=True)

_saved_path = list(sys.path)
try:
    bodypose.MODELS_DIR, bodypose.S3DB_REPO = WEIGHTS, STAND
    check(bodypose.backend_status("sam-3d-body-dinov3")[0] == "ready",
          "with code and weights present it reports ready")
    check(bodypose.preferred_backend() == "sam-3d-body-dinov3",
          "and becomes the preferred backend (it is the only 3D one)")
    est = bodypose.make_estimator("sam-3d-body-dinov3", device="cpu", use_detector=False)
    import sam_3d_body as stand                                # noqa: E402
    check(est.gives_3d and est.rig.name == "mhr70",
          "the adapter reports a 3D backend on the MHR rig")
    check(stand.SEEN["ckpt"].endswith("model.ckpt") and stand.SEEN["mhr"].endswith("mhr_model.pt"),
          "the checkpoint and the MHR asset are both passed through")

    bgr = np.zeros((240, 320, 3), np.uint8)
    bgr[:, :, 0] = 255                                         # pure BLUE in BGR
    mask = np.zeros((240, 320), bool)
    mask[50:150, 60:160] = True
    people = est.step(bgr, masks=[mask])
    check(len(people) == 1, "one person out for one mask in")
    p = people[0]
    check(p.joints3d is not None and p.joints3d.shape == (70, 3)
          and p.joints2d.shape == (70, 2),
          "the 308-keypoint output is cut to the rig's 70 joints",
          "" if p.joints3d is None else str(p.joints3d.shape))
    check(np.allclose(p.joints3d[0], [0.0, 0.001, 0.002]) and np.allclose(p.joints2d[0], [7.0, 8.0]),
          "the first joints are the model's first joints (no reordering)")
    check(abs(p.focal - 1234.5) < 1e-3, "the focal length is carried through", str(p.focal))
    check(p.cam_t is not None and p.vertices is not None and len(p.vertices) == 64,
          "the camera translation and the mesh come through")
    check(list(p.bbox) == [10.0, 20.0, 110.0, 220.0], "the box comes through", str(p.bbox))
    seen_img = stand.SEEN["img"]
    check(seen_img[0, 0, 2] == 255 and seen_img[0, 0, 0] == 0,
          "the frame is handed over as RGB, as the release requires",
          f"first pixel {tuple(int(v) for v in seen_img[0, 0])}")
    check(stand.SEEN["masks"] is not None and stand.SEEN["bboxes"] is not None,
          "a mask prompt always travels with its box (upstream asserts it)")
    mb = stand.SEEN["bboxes"].reshape(-1, 4)[0]
    check(mb[0] < 60 and mb[2] > 159, "the box is the mask's, padded a little", str(mb))
    check(np.isnan(p.conf).all(),
          "no per-joint confidence is invented (NaN, not a constant 1.0) (I85)")
    est2 = bodypose.make_estimator("sam-3d-body-dinov3", device="cpu", use_detector=False,
                                   intrinsics=np.eye(3) * 900.0)
    try:
        est2.step(bgr, boxes=[[1, 2, 100, 200]])
        ok_step = True
    except Exception as exc:                                   # noqa: BLE001
        ok_step, err = False, str(exc)
    check(ok_step, "a measured focal length does not make the model fail (I88)",
          "" if ok_step else err[:80])
    ci = stand.SEEN["cam_int"]
    check(ci is not None, "supplied intrinsics reach the model")
    check(ci is not None and tuple(ci.shape) == (1, 3, 3),
          "they go in batched, (1, 3, 3), as upstream indexes them",
          "" if ci is None else str(tuple(ci.shape)))
    check(ci is not None and abs(float(ci[0, 0, 2]) - 160) < 1 and abs(float(ci[0, 1, 2]) - 120) < 1
          and abs(float(ci[0, 0, 0]) - 900) < 1e-3 and abs(float(ci[0, 2, 2]) - 1) < 1e-9,
          "focal length kept, principal point at the frame centre (not the corner)",
          "" if ci is None else str(np.round(np.asarray(ci)[0], 1).tolist()))
    check(stand.SEEN["masks"] is None, "a box prompt alone sends no mask")
    # what the run dialog builds from a lens profile: principal point left for
    # the estimator to put at each frame's centre
    from kinetrace import bodyview as _bv                 # noqa: E402
    est3 = bodypose.make_estimator("sam-3d-body-dinov3", device="cpu", use_detector=False,
                                   intrinsics=_bv.lens_intrinsics(1000.0))
    est3.step(np.zeros((480, 640, 3), np.uint8), boxes=[[1, 2, 100, 200]])
    ci = np.asarray(stand.SEEN["cam_int"])
    check(ci.shape == (1, 3, 3) and np.isfinite(ci).all() and np.allclose(ci[0, :2, 2], [320, 240])
          and abs(ci[0, 0, 0] - 1000) < 1e-3, "the dialog's lens matrix arrives complete and centred",
          str(np.round(ci[0], 1).tolist()))
finally:
    bodypose.MODELS_DIR, bodypose.S3DB_REPO = _real_models, _real_repo
    sys.path[:] = _saved_path
    sys.modules.pop("sam_3d_body", None)
    shutil.rmtree(STAND, ignore_errors=True)
    shutil.rmtree(WEIGHTS, ignore_errors=True)

# ------------------------- 12. standing the subject up, and the body shape
print("\n[12] upright orientation and the 3D body shape")


def rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tilted_track(pitch, roll, with_mesh=False):
    """The synthetic walker as a camera pitched/rolled by that much would
    report it -- which is the situation on any rig-mounted action camera."""
    M = rot_x(pitch) @ rot_z(roll)
    bt = body.BodyTrack(sh.T, mhr, 1)
    bt.backend = "tilt"
    for t in range(sh.T):
        Xi = np.full((mhr.n_joints, 3), np.nan)
        for k, nm in enumerate(gt["names"]):
            i = mhr.index(nm)
            if i is not None:
                Xi[i] = gt["xyz"][t, k] @ M.T
        kw = {}
        if with_mesh:
            mv, mf = sh.mesh_3d(t)
            kw = dict(vertices=mv @ M.T, faces=mf)
        bt.set_person(t, 0, joints3d=Xi, joints2d=sh.project(Xi),
                      conf=np.ones(mhr.n_joints), score=0.9, **kw)
    return bt


def axis_from_vertical(bt, frame, upright):
    """How far the DRAWN pelvis-to-shoulders axis is from straight up, in
    degrees, in the projection the panel actually uses."""
    stack = bt.joints3d[:, 0]
    base = bodyview.body_basis(stack, mhr) if upright else bodyview._up_basis(mhr.up)
    Rv = bodyview._view_rotation(0.0, 0.0, up=1) @ base
    p = bt.joints3d[frame, 0].astype(float)
    hip = (p[mhr.index("left_hip")] + p[mhr.index("right_hip")]) / 2
    sho = (p[mhr.index("left_shoulder")] + p[mhr.index("right_shoulder")]) / 2
    v = (sho - hip) @ Rv.T
    return abs(float(np.degrees(np.arctan2(v[0], v[1]))))


for pitch, roll in ((0, 0), (45, 0), (70, 0), (30, 25), (90, 0)):
    bt = tilted_track(pitch, roll)
    off = axis_from_vertical(bt, 60, upright=False)
    on = axis_from_vertical(bt, 60, upright=True)
    check(on < 0.01, f"pitch {pitch} roll {roll}: the body's long axis is vertical",
          f"{on:.4f} deg (camera frame would be {off:.1f} deg)")
bt90 = tilted_track(90, 0)
check(axis_from_vertical(bt90, 60, upright=False) > 45,
      "and the raw camera frame really was tilted over", "90 deg camera reads 90 deg")
# it must not rock frame to frame as the subject walks
tilt = tilted_track(45, 20)
spread = [axis_from_vertical(tilt, t, True) for t in range(0, sh.T, 7)]
check(max(spread) < 0.01, "and stays vertical through the whole stride",
      f"worst {max(spread):.4f} deg")
# a track with no hips at all must not crash, just fall back
lonely = body.BodyTrack(5, mhr, 1)
lonely.set_person(0, 0, joints3d=np.full((mhr.n_joints, 3), np.nan),
                  conf=np.zeros(mhr.n_joints), score=0.5)
check(bodyview.body_basis(lonely.joints3d[:, 0], mhr) is None,
      "no usable joints -> no body basis (the rig's own up is used)")
check(bodyview.render_pose_panel(lonely, 0, 0, (200, 200)).shape == (200, 200, 3),
      "and the panel still renders")

# --- the mesh -----------------------------------------------------------
meshed = tilted_track(40, 10, with_mesh=True)
check(meshed.has_mesh() and meshed.has_mesh(60, 0), "the body shape is stored")
mv = meshed.mesh_at(60, 0)
check(mv is not None and mv[0].ndim == 2 and mv[0].shape[1] == 3 and mv[1].shape[1] == 3,
      "mesh_at gives vertices and faces", "" if mv is None else f"{mv[0].shape} {mv[1].shape}")
check(meshed.mesh[(60, 0)].dtype == np.float16,
      "vertices are stored as float16 (half the memory, finer than the model)")
check(len(meshed.mesh_frames()) == sh.T, "one mesh per posed frame")
check(meshed.mesh_bytes() > 0 and "body shape" in meshed.summary(),
      "the summary says the shape is there", meshed.summary()[-46:])

mnpz = OUT / "body_mesh.npz"
np.savez_compressed(mnpz, **meshed.to_arrays("m_"))
with np.load(mnpz, allow_pickle=True) as z:
    mback = body.BodyTrack.from_arrays("m_", z)
r0, r1 = meshed.mesh_at(60, 0), mback.mesh_at(60, 0)
check(r1 is not None and np.array_equal(r0[0], r1[0]) and np.array_equal(r0[1], r1[1]),
      "the mesh survives the npz round trip unchanged")
check(len(mback.mesh_frames()) == len(meshed.mesh_frames()), "every frame's mesh comes back")
cp = meshed.copy()
cp.clear(0, sh.T - 1)
check(meshed.has_mesh(60, 0) and not cp.has_mesh(), "copy is deep and clear drops meshes")

with_shape = bodyview.render_pose_panel(meshed, 60, 0, (360, 520), 0, 0, mesh=True)
without = bodyview.render_pose_panel(meshed, 60, 0, (360, 520), 0, 0, mesh=False)
check(not np.array_equal(with_shape, without), "the shape is actually drawn")
bgp = np.array(bodyview.BG_PANEL, np.int16)
painted = (np.abs(with_shape.astype(np.int16) - bgp).max(axis=2) > 24).sum()
bare = (np.abs(without.astype(np.int16) - bgp).max(axis=2) > 24).sum()
check(painted > 3 * bare, "the shape covers far more of the panel than the skeleton",
      f"{painted} vs {bare} px")
joints_only = tilted_track(40, 10, with_mesh=False)
check(np.array_equal(bodyview.render_pose_panel(joints_only, 60, 0, (360, 520), 0, 0, mesh=True),
                     bodyview.render_pose_panel(joints_only, 60, 0, (360, 520), 0, 0, mesh=False)),
      "asking for a shape that was never stored changes nothing")
sbs_mesh = bodyview.compose_side_by_side(sh.render_frame(60), meshed, 60, 0,
                                         bodyview.PoseDrawOptions(), True, 0, 0, sh.FPS)
check(sbs_mesh.shape[1] == sh.W * 2, "the side-by-side carries the shape through")

# ------------------------------------------- 13. release-sweep fixes (I82-I93)
print("\n[13] release-sweep fixes")
import time                                                   # noqa: E402

cu17 = fill(coco, gt["names"], gt["uv"], sh.T)


def coco_track(frames, shift=0.0, step=1):
    bt_ = body.BodyTrack(sh.T, coco, 1)
    bt_.backend = "unit"
    for t in frames:
        bt_.set_person(t, 0, joints2d=cu17[t] + shift, conf=np.full(17, 0.9), score=0.8)
    bt_.runs, bt_.step = [(min(frames), max(frames), step)], step
    bt_.n_requested = len(list(frames))
    return bt_


# --- I82: a re-run is merged, never swapped in whole
old = coco_track(range(24))
new = coco_track([5], shift=100.0)
new.examined = np.array([5])
merged, note = body.merge_run(old, new)
check(sorted(merged.frames().tolist()) == list(range(24)),
      "a one-frame re-run keeps every other posed frame (I82)", f"{merged.n_posed()} posed")
check(np.allclose(merged.joints2d[5, 0], cu17[5] + 100) and np.allclose(merged.joints2d[6, 0], cu17[6]),
      "only the re-run frame changed")
check("posed again" in note and "kept" in note, "and the message says what was kept", note)
check(old.n_posed() == 24 and np.allclose(old.joints2d[5, 0], cu17[5]),
      "the earlier track object itself is untouched (undo still has it)")
stopped = coco_track(range(3), shift=50.0)
stopped.examined, stopped.stopped_early = np.array([0, 1, 2]), True
stopped.runs = [(0, 23, 1)]
m2_, note2 = body.merge_run(old, stopped)
check(m2_.n_posed() == 24 and np.allclose(m2_.joints2d[2, 0], cu17[2] + 50)
      and np.allclose(m2_.joints2d[3, 0], cu17[3]),
      "a stopped run replaces what it reached and keeps the rest", note2)
check("Stopped early" in note2, "and says it was stopped")
nobody = body.BodyTrack(sh.T, coco, 1)
nobody.examined, nobody.runs = np.array([7]), [(7, 7, 1)]
m3_, _ = body.merge_run(old, nobody)
check(not m3_.has(7) and m3_.n_posed() == 23,
      "a frame the run looked at and found nobody on becomes blank")
other = body.BodyTrack(sh.T, mhr, 1)
other.set_person(4, 0, joints3d=X[4], joints2d=U[4], conf=np.ones(70), score=0.9)
other.examined = np.array([4])
m4_, note4 = body.merge_run(old, other)
check(m4_ is other and "REPLACED" in note4, "another joint set replaces the track and says so", note4)
wide = coco_track(range(0, 24, 4), step=4)
wide.examined = np.arange(0, 24, 4)
wide.runs = [(0, 23, 4)]
m5_, _ = body.merge_run(coco_track(range(40, 60)), wide)
check(m5_.n_requested == 20 + 6, "the frames asked for are the union of the runs",
      str(m5_.n_requested))
check(body.merge_run(None, new)[0] is new, "with nothing to merge into, the run is the track")

# --- I83: a finished run knows which view it belongs to
from kinetrace.session import TrackingSession as _TS    # noqa: E402
sa_, sb_ = _TS("a.mp4", sh.T, sh.FPS, sh.W, sh.H), _TS("b.mp4", sh.T, sh.FPS, sh.W, sh.H)
wk = bodyview.BodyPoseWorker("a.mp4", sh.T, bodyview.BodyRunOptions(), target=sa_)
check(wk.target is sa_ and wk.result_fits(sa_) and not wk.result_fits(sb_),
      "the worker carries its target view and refuses another one (I83)")
check(not wk.result_fits(_TS("a.mp4", sh.T + 1, sh.FPS, sh.W, sh.H)),
      "and a session of another length")
wk2 = bodyview.BodyPoseWorker("a.mp4", sh.T, bodyview.BodyRunOptions())
check(wk2.result_fits(sa_) and not wk2.result_fits(sb_),
      "without a target the video path decides")

# --- I85: the camera translation is kept and exported
walk = body.BodyTrack(30, mhr, 1)
for t in range(30):
    walk.set_person(t, 0, joints3d=X[t] - X[t, mhr.index("left_hip")], joints2d=U[t],
                    conf=np.full(70, np.nan), score=0.9, focal=1100.0,
                    cam_t=[t / 29.0, 0.0, 4.0])              # the body moves 1 m right
wp = OUT / "body_walk_joints.csv"
body.export_joints_csv(walk, wp)
lines = wp.read_text(encoding="utf-8").splitlines()
com = [l for l in lines if l.startswith("#")]
rows_w = list(_csv.reader(l for l in lines if not l.startswith("#")))
hd = rows_w[0]
ix = hd.index("left_hip_X")
xs = [float(r[ix]) for r in rows_w[1:]]
check(abs((xs[-1] - xs[0]) - 1.0) < 1e-3, "exported pelvis X moves the metre the body moved (I85)",
      f"{xs[-1] - xs[0]:.4f} m")
check(all(r[hd.index("xyz_frame")] == "camera" for r in rows_w[1:]),
      "every row says its X/Y/Z are in the camera's frame")
check(any("origin = the camera" in l for l in com) and any("focal length used" in l for l in com),
      "the header states the origin, the axes and the focal length")
check(rows_w[1][hd.index("nose_conf")] == "" and any("no per-joint confidence" in l for l in com),
      "a model with no joint scores writes a blank conf and says why")
old_way = walk.copy()
old_way.cam_t[:] = np.nan                                     # a run from before the fix
body.export_joints_csv(old_way, wp)
rows_o = list(_csv.reader(l for l in wp.read_text(encoding="utf-8").splitlines()
                          if not l.startswith("#")))
check(all(r[rows_o[0].index("xyz_frame")] == "body" for r in rows_o[1:]),
      "a track without the translation is labelled 'body' on every row")
np.savez_compressed(OUT / "body_walk.npz", **walk.to_arrays("w_"))
with np.load(OUT / "body_walk.npz", allow_pickle=True) as z:
    wb = body.BodyTrack.from_arrays("w_", z)
check(np.allclose(wb.cam_t, walk.cam_t, equal_nan=True), "cam_t survives the npz round trip")
legacy = {k: v for k, v in walk.to_arrays("l_").items() if k != "l_cam_t"}
check(np.isnan(body.BodyTrack.from_arrays("l_", legacy).cam_t).all(),
      "an older file without cam_t loads with it unknown")
check(np.isfinite(walk.angles(0)[1]).any(), "NaN confidence never hides joints from the angles")

# --- I86: a joint the model scored as unseen gives no angle
blind = coco_track(range(sh.T))
blind.conf[10, 0, coco.index("left_ankle")] = 0.02
blind.joints2d[10, 0, coco.index("left_ankle")] = cu17[10, coco.index("left_knee")] + [15, 2]
blind.touch()
dB, aB = blind.angles(0)
lk, rk = [d.name for d in dB].index("left knee flexion"), [d.name for d in dB].index("right knee flexion")
ref2 = coco_track(range(sh.T)).angles(0)[1]
check(np.isnan(aB[10, lk]), "an ankle scored 0.02 gives no left knee angle (I86)", f"{aB[10, lk]}")
check(abs(aB[10, rk] - ref2[10, rk]) < 1e-9 and abs(aB[11, lk] - ref2[11, lk]) < 1e-9,
      "the other knee, and the next frame, are unchanged")
rep_b = body.angle_report(blind, sh.FPS)
check("treated as unseen" in rep_b, "the report states the rule")

# --- I89: redrawing the side-by-side must not recompute the whole clip
big = body.BodyTrack(40000, mhr, 1)
reps = np.arange(40000) % sh.T
big.joints3d[:, 0] = X[reps].astype(np.float32)
big.joints2d[:, 0] = U[reps].astype(np.float32)
big.conf[:, 0] = 1.0
big.score[:, 0] = 0.9
big.has_3d = True
big.touch()
fr10 = sh.render_frame(10)
o13 = bodyview.PoseDrawOptions(angles_shown=("left knee flexion",), person=0)
bodyview.compose_side_by_side(fr10, big, 10, 0, o13, True, 25, 12, sh.FPS, width=1170)
t0 = time.time()
bodyview.compose_side_by_side(fr10, big, 11, 0, o13, True, 25, 12, sh.FPS, width=1170)
dt = time.time() - t0
check(dt < 0.1, "a 40k-frame track redraws in under 0.1 s once cached (I89)", f"{dt * 1000:.0f} ms")
big.set_person(500, 0, joints3d=X[3], joints2d=U[3], conf=np.ones(70), score=0.9)
dK, aK = big.angles(0)
kn_i = [d.name for d in dK].index("left knee flexion")
check(abs(aK[500, kn_i] - gt["angles"]["left knee flexion"][3]) < 1e-4
      and abs(aK[501, kn_i] - gt["angles"]["left knee flexion"][501 % sh.T]) < 1e-4,
      "and a change to the track is seen (the cache follows the data)")

# --- I90: a sampled run exports velocities and its video only its posed frames
samp = body.BodyTrack(sh.T, mhr, 1)
samp.step = 4
for t in range(0, sh.T, 4):
    samp.set_person(t, 0, joints3d=X[t], joints2d=U[t], conf=np.ones(70), score=0.9)
sp = OUT / "body_sampled_angles.csv"
body.export_angles_csv(samp, sp, sh.FPS)
rows_s = list(_csv.reader(l for l in sp.read_text(encoding="utf-8").splitlines()
                          if not l.startswith("#")))
kv = rows_s[0].index("left knee flexion (deg/s)")
vals_s = [r[kv] for r in rows_s[1:]]
check(sum(v != "" for v in vals_s) >= len(vals_s) - 1,
      "a 1-in-4 run has a rate on its sampled frames (I90)",
      f"{sum(v != '' for v in vals_s)} of {len(vals_s)}")
_, sa_ang = samp.angles(0)
kk = [d.name for d in samp.angles(0)[0]].index("left knee flexion")
want = (sa_ang[44, kk] - sa_ang[36, kk]) * sh.FPS / 8.0
got = float(rows_s[1 + 10][kv])                               # row of frame 40
check(abs(got - want) < 1e-3, "and it is the central difference over the sampled frames",
      f"{got:.3f} vs {want:.3f} deg/s")
rnd = bodyview.SideBySideRenderer("x.mp4", samp, "y.mp4", 0, sh.T - 1, 0,
                                  bodyview.PoseDrawOptions(), sh.FPS)
check(rnd.frames == list(range(0, sh.T, 4)) and abs(rnd.out_fps - sh.FPS / 4) < 1e-9
      and "posed frames" in rnd.note,
      "its video writes only the posed frames, at fps / step", f"{len(rnd.frames)} frames")

# --- I91: a 4K frame composed for a 1600-px video keeps its text legible
f4k = cv2.resize(sh.render_frame(10), (3840, 2160))
t4k = body.BodyTrack(sh.T, mhr, 1)
for t in range(sh.T):
    t4k.set_person(t, 0, joints3d=X[t], joints2d=U[t] * 4.0, conf=np.ones(70), score=0.9)


def legend_ink(img):
    """Grey legend text in the plot's right margin: pixels with every channel
    above 150 (the traces are coloured, so their darkest channel is lower)."""
    strip = img[-int(img.shape[0] * 0.2):, -150:-50]
    return int((strip.min(axis=2) > 150).sum())


sbs4 = bodyview.compose_side_by_side(f4k, t4k, 10, 0, o13, True, 25, 12, sh.FPS, width=1600)
shrunk = cv2.resize(bodyview.compose_side_by_side(f4k, t4k, 10, 0, o13, True, 25, 12, sh.FPS),
                    (1600, sbs4.shape[0]), interpolation=cv2.INTER_AREA)
ink_new, ink_old = legend_ink(sbs4), legend_ink(shrunk)
check(sbs4.shape[1] == 1600, "the 4K composite comes out at the width asked for", str(sbs4.shape))
check(ink_new > 40 and ink_new > 4 * ink_old,
      "its legend text is drawn at size, not shrunk 5x afterwards (I91)",
      f"{ink_new} legible px vs {ink_old} when shrunk")
cv2.imwrite(str(OUT / "body_sbs_4k_to_1600.png"), sbs4)
fit = bodyview.compose_side_by_side(f4k, t4k, 10, 0, o13, True, 25, 12, sh.FPS, width=1170,
                                    max_height=500)
check(fit.shape[0] <= 500 and fit.shape[1] <= 1170, "and fits a window both ways", str(fit.shape))

# --- I93: copies share the (large, read-only) mesh instead of duplicating it
mcopy = meshed.copy()
k0 = (60, 0)
check(mcopy.mesh[k0] is meshed.mesh[k0] and not meshed.mesh[k0].flags.writeable,
      "a copy (an undo snapshot) shares the mesh arrays, which are read-only (I93)")
mcopy.clear(60, 60)
check(meshed.has_mesh(60, 0) and not mcopy.has_mesh(60, 0), "and clearing the copy leaves the original")
v0 = meshed.mesh_version
meshed.set_person(61, 0, joints3d=meshed.joints3d[61, 0], conf=np.ones(70), score=0.9,
                  vertices=meshed.mesh[(61, 0)].astype(np.float32))
check(meshed.mesh_version > v0, "a new mesh bumps mesh_version (a save can skip an unchanged one)")
check(not any("mesh" in k for k in meshed.to_arrays("q_", mesh=False))
      and "q_mesh_verts" in meshed.mesh_arrays("q_"),
      "the mesh block can be written apart from the rest of the track")

print("\n" + "=" * 62)
if fails:
    print(f"verify_body FAILED ({len(fails)}):")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("verify_body PASSED")
