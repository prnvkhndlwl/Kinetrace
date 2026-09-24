"""Human body layer, model-free half: joint sets, joint angles, per-frame
storage and exports.

Why this is a separate module from `bodypose.py` (which owns the networks):
everything here is pure numpy, so the tests can drive it against analytic
ground truth without a GPU, without weights and without torch, and
`session.py` can persist a body track without importing the model stack.
That mirrors the split the animal layer already uses (`silhouette.py` is the
geometry, `segmenter.py` is the network).

Three ideas hold the layer together:

* a **rig** is a named list of joints plus the bones between them. Different
  backends emit different joint sets (SAM 3D Body speaks MHR-70, a 2D
  keypoint model speaks COCO-17), so nothing downstream may hard-code an
  index. Joints are referred to by CANONICAL name and resolved per rig.
* an **angle** is defined once, in canonical names, and is silently skipped on
  a rig that lacks one of its joints. This is what lets the same angle table
  serve a 70-joint mesh model and a 17-joint keypoint model.
* a **BodyTrack** is (frames x people x joints), NaN where there is no data,
  in exactly the same spirit as `TrackingSession.tracks` — blank means blank,
  it is never a guess parked inside the frame.

Angle conventions are stated per angle in `AngleDef.zero_means` and travel all
the way to the CSV header and the tooltip, because "hip angle = 43 deg" is
worthless to the reader who does not know which way 0 points.
"""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- names

# Backends spell the same joint several ways ("left-shoulder", "left_shoulder",
# "L_Shoulder"); everything inside Kinetrace uses the canonical spelling on the
# right. Only aliases that actually occur in a supported backend are listed --
# a fuzzy matcher here would silently mis-map a joint, which is far worse than
# an unresolved angle.
_ALIASES = {
    "left_big_toe_tip": "left_big_toe",
    "right_big_toe_tip": "right_big_toe",
    "left_small_toe_tip": "left_small_toe",
    "right_small_toe_tip": "right_small_toe",
}


def canon(name: str) -> str:
    """Canonical spelling of a joint name: lower case, underscores, aliases
    applied. 'left-big-toe-tip' -> 'left_big_toe', 'L_Shoulder' ->
    'left_shoulder' (that second spelling is what ViTPose's `id2label` uses)."""
    n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in n:
        n = n.replace("__", "_")
    if n.startswith("l_"):
        n = "left_" + n[2:]
    elif n.startswith("r_"):
        n = "right_" + n[2:]
    return _ALIASES.get(n, n)


@dataclass
class BodyRig:
    """One backend's joint set. `joints` is in the backend's own emission
    order -- the arrays are stored in that order, so the rig is what makes a
    stored track readable again."""
    name: str
    label: str
    joints: list[str]
    bone_names: list[tuple[str, str]]
    # Which world axis points up in this backend's 3D output, as a unit vector.
    # Only the handful of angles measured against gravity use it; every
    # limb angle is intrinsic and needs no convention at all.
    up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    units: str = "m"

    def __post_init__(self):
        self.joints = [canon(j) for j in self.joints]
        self._index = {j: i for i, j in enumerate(self.joints)}

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    def index(self, name: str) -> int | None:
        return self._index.get(canon(name))

    def has(self, *names: str) -> bool:
        return all(canon(n) in self._index for n in names)

    def bones(self) -> list[tuple[int, int]]:
        """Bone list as index pairs, skipping bones this rig cannot draw."""
        out = []
        for a, b in self.bone_names:
            i, j = self.index(a), self.index(b)
            if i is not None and j is not None:
                out.append((i, j))
        return out

    def up_vector(self) -> np.ndarray:
        v = np.asarray(self.up, np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([0.0, -1.0, 0.0])


# The body bones every rig shares (fingers are deliberately left out: they are
# 40 of MHR's 70 joints, they clutter every view, and no gait or reaching
# measure uses them). Names that a rig lacks are dropped by `BodyRig.bones`.
_BODY_BONES: list[tuple[str, str]] = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_heel"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"), ("neck", "nose"),
    ("left_ankle", "left_small_toe"), ("right_ankle", "right_small_toe"),
]

# MHR-70 carries a full hand: 40 of its 70 joints are finger joints. Without
# these chains the hands draw as a cloud of dots, which is exactly how they
# looked before. A rig without fingers (COCO) drops them all in
# `BodyRig.bones`, so the list can simply be appended.
#
# Naming runs proximal to distal: the "third joint" is the knuckle at the
# wrist end and "tip" is the fingertip -- that is the order SAM 3D Body's own
# skeleton metadata links them in.
_FINGER_BONES: list[tuple[str, str]] = []
for _side in ("left", "right"):
    for _f in ("thumb", "index", "middle", "ring", "pinky"):
        _chain = [f"{_side}_wrist", f"{_side}_{_f}_third_joint",
                  f"{_side}_{_f}_second_joint", f"{_side}_{_f}_first_joint",
                  f"{_side}_{_f}_tip"]
        _FINGER_BONES += list(zip(_chain, _chain[1:]))
_BODY_BONES += _FINGER_BONES


def is_finger(name: str) -> bool:
    """True for a finger or thumb joint -- they are drawn smaller, or they
    swamp the other thirty joints."""
    n = canon(name)
    return any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))

# SAM 3D Body emits 308 MHR keypoints and its own metadata keeps the first 70
# (the rest are face detail). Order copied from sam_3d_body/metadata/mhr70.py
# -- it is the emission order of `pred_keypoints_3d`, so it must not be
# re-sorted.
MHR70_JOINTS = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-hip", "right-hip", "left-knee", "right-knee",
    "left-ankle", "right-ankle",
    "left-big-toe-tip", "left-small-toe-tip", "left-heel",
    "right-big-toe-tip", "right-small-toe-tip", "right-heel",
    "right-thumb-tip", "right-thumb-first-joint", "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip", "right-index-first-joint", "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip", "right-middle-first-joint", "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip", "right-ring-first-joint", "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip", "right-pinky-first-joint", "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip", "left-thumb-first-joint", "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip", "left-index-first-joint", "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip", "left-middle-first-joint", "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip", "left-ring-first-joint", "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip", "left-pinky-first-joint", "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon", "right-olecranon",
    "left-cubital-fossa", "right-cubital-fossa",
    "left-acromion", "right-acromion", "neck",
]

COCO17_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

RIGS: dict[str, BodyRig] = {
    # SAM 3D Body works in camera coordinates: x right, y DOWN, z forward
    # (the OpenCV convention its focal length and cam_t are expressed in), so
    # "up" is -y.
    "mhr70": BodyRig("mhr70", "SAM 3D Body (MHR, 70 joints)", MHR70_JOINTS,
                     _BODY_BONES, up=(0.0, -1.0, 0.0), units="m"),
    # A 2D keypoint rig has no 3D at all; "up" is the image's up, which is -y
    # in pixel coordinates. Angles are then image-plane angles -- flagged as
    # such everywhere, because a limb pointing at the camera foreshortens.
    "coco17": BodyRig("coco17", "2D keypoints (COCO, 17 joints)", COCO17_JOINTS,
                      _BODY_BONES, up=(0.0, -1.0, 0.0), units="px"),
}
DEFAULT_RIG = "coco17"


def rig_of(name: str) -> BodyRig:
    return RIGS.get(str(name), RIGS[DEFAULT_RIG])


# ------------------------------------------------------------------- angles

@dataclass
class AngleDef:
    """One measured angle.

    `kind`:
      * "joint"    -- the UNSIGNED angle at `b` in the chain a-b-c. Right for a
                      hinge that only folds one way (knee, elbow).
      * "sagittal" -- the SIGNED angle at `b` from segment b->a to segment
                      b->c, measured about the subject's own left-right axis,
                      positive towards the direction they face. A hip swings
                      both ways and an unsigned angle cannot tell flexion from
                      extension -- it reports +22 deg for both, which turns a
                      gait cycle into nonsense.
      * "axis"     -- the signed lean of the segment a->b away from the rig's
                      up axis, in the same sagittal sense.
      * "segments" -- the signed angle from segment a->b to segment c->d about
                      the subject's left-right axis. Needed wherever the two
                      limbs do not share a joint: the foot's axis is heel to
                      toe, not ankle to toe.
      * "lean"     -- the signed angle between segment a->b and that same
                      segment's MEDIAN direction over the clip, i.e. how far
                      the subject is leaning out of their own upright. A
                      monocular model reports camera coordinates, so measuring
                      lean against a fixed axis measures the camera's tilt.
      * "twist"    -- the signed rotation between segment a->b and segment
                      c->d about the trunk axis (3D only).

    The reported value is `scale * raw + offset`, which is how one piece of
    geometry serves several clinical conventions (a knee's raw interior angle
    is 180 deg when straight; physios write that as 0 deg of flexion).
    """
    name: str
    kind: str
    joints: tuple[str, ...]
    scale: float = 1.0
    offset: float = 0.0
    zero_means: str = ""
    needs_3d: bool = False
    group: str = ""


# Virtual joints: a reference of the form "mid(a,b)" is the midpoint of two
# real joints. COCO-17 has no neck or pelvis, and inventing them this way is
# what lets the trunk angles work on both rigs.
def _resolve(ref: str, xyz: np.ndarray, rig: BodyRig) -> np.ndarray | None:
    """(..., D) coordinates for a joint reference, or None if the rig lacks it."""
    ref = ref.strip()
    if ref.startswith("mid(") and ref.endswith(")"):
        a, b = (s.strip() for s in ref[4:-1].split(","))
        pa, pb = _resolve(a, xyz, rig), _resolve(b, xyz, rig)
        if pa is None or pb is None:
            return None
        return (pa + pb) / 2.0
    i = rig.index(ref)
    return None if i is None else xyz[..., i, :]


ANGLE_DEFS: list[AngleDef] = [
    AngleDef("left elbow flexion", "joint", ("left_shoulder", "left_elbow", "left_wrist"),
             -1.0, 180.0, "0 deg = arm straight; larger = more bent", group="arm"),
    AngleDef("right elbow flexion", "joint", ("right_shoulder", "right_elbow", "right_wrist"),
             -1.0, 180.0, "0 deg = arm straight; larger = more bent", group="arm"),
    AngleDef("left shoulder flexion", "sagittal", ("left_hip", "left_shoulder", "left_elbow"),
             -1.0, 0.0, "0 deg = upper arm alongside the trunk; positive = arm swung "
                        "forwards, negative = behind the body", group="arm"),
    AngleDef("right shoulder flexion", "sagittal", ("right_hip", "right_shoulder", "right_elbow"),
             -1.0, 0.0, "0 deg = upper arm alongside the trunk; positive = arm swung "
                        "forwards, negative = behind the body", group="arm"),
    AngleDef("left knee flexion", "joint", ("left_hip", "left_knee", "left_ankle"),
             -1.0, 180.0, "0 deg = leg straight; larger = more bent", group="leg"),
    AngleDef("right knee flexion", "joint", ("right_hip", "right_knee", "right_ankle"),
             -1.0, 180.0, "0 deg = leg straight; larger = more bent", group="leg"),
    AngleDef("left hip flexion", "sagittal", ("left_shoulder", "left_hip", "left_knee"),
             -1.0, 180.0, "0 deg = thigh in line with the trunk; positive = knee drawn "
                          "forwards, negative = leg trailing behind", group="leg"),
    AngleDef("right hip flexion", "sagittal", ("right_shoulder", "right_hip", "right_knee"),
             -1.0, 180.0, "0 deg = thigh in line with the trunk; positive = knee drawn "
                          "forwards, negative = leg trailing behind", group="leg"),
    # heel->toe is the foot's real axis. Measured from the ankle to the TOE TIP
    # instead, a neutral standing foot reads 113 deg on the real MHR rig, which
    # sits on atan2's branch cut and made the value jump by 360.
    AngleDef("left ankle dorsiflexion", "segments",
             ("left_ankle", "left_knee", "left_heel", "left_big_toe"),
             -1.0, 90.0, "0 deg = foot square to the shin; positive = toes pulled up, "
                         "negative = toes pointed down", group="leg"),
    AngleDef("right ankle dorsiflexion", "segments",
             ("right_ankle", "right_knee", "right_heel", "right_big_toe"),
             -1.0, 90.0, "0 deg = foot square to the shin; positive = toes pulled up, "
                         "negative = toes pointed down", group="leg"),
    # (I87) Measured to the EARS, not the nose. The nose sits ~10 cm in front
    # of the neck, so a nose-based angle read +20 deg on an upright synthetic
    # head and +34..58 deg on real standing people. Ear over
    # shoulder over hip is the posture plumb line, so 0 really is "in line".
    AngleDef("neck flexion", "sagittal", ("mid(left_ear,right_ear)",
                                          "mid(left_shoulder,right_shoulder)",
                                          "mid(left_hip,right_hip)"),
             -1.0, 180.0, "0 deg = ears straight above the shoulders, in line with the "
                          "trunk; positive = head carried forwards. Measured to the ears, "
                          "so a nod of the head alone barely changes it", group="trunk"),
    AngleDef("trunk lean", "lean", ("mid(left_hip,right_hip)",
                                    "mid(left_shoulder,right_shoulder)"),
             1.0, 0.0, "0 deg = this subject's own average upright over the clip; "
                       "positive = leaning forwards. It is measured against the "
                       "subject, not against the camera, so tilting the camera "
                       "cannot change it", group="trunk"),
    # (I87) Thigh to thigh, seen from the side (about the subject's left-right
    # axis). The old unsigned angle at the mid-hip between the two KNEES let
    # the hip width leak in: parallel legs read ~32 deg in 3D.
    AngleDef("thigh separation (stride)", "segments",
             ("left_hip", "left_knee", "right_hip", "right_knee"),
             1.0, 0.0, "angle between the two thighs seen from the side; 0 deg = thighs "
                       "side by side; positive = left knee ahead of the right, negative = "
                       "right knee ahead", group="leg"),
    AngleDef("shoulder-hip twist", "twist",
             ("right_shoulder", "left_shoulder", "right_hip", "left_hip"),
             1.0, 0.0, "0 deg = shoulders square over the hips; signed by the right-hand "
                       "rule about the pelvis-to-shoulders axis",
             needs_3d=True, group="trunk"),
]


def _interior(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b in the chain a-b-c, degrees, over leading axes.
    A degenerate (zero-length) segment gives NaN rather than an arbitrary 0 --
    a collapsed limb is missing data, not a straight one."""
    u = a - b
    v = c - b
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = nu * nv
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.sum(u * v, axis=-1) / denom
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return np.where((denom > 1e-12) & np.isfinite(denom), ang, np.nan)


def _axis_angle(a: np.ndarray, b: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Angle between the segment a->b and `up`, degrees."""
    v = b - a
    n = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (v @ up) / n
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return np.where((n > 1e-12) & np.isfinite(n), ang, np.nan)


def median_axis(xyz: np.ndarray, rig: BodyRig, a: str, b: str) -> np.ndarray | None:
    """Median unit vector from joint reference `a` to `b` over every frame
    that has both. Median, not mean, so one frame with a swapped limb cannot
    tip the whole clip over. Shared by the lean angle and by the pose panel's
    upright basis, so the two can never disagree about which way is up."""
    pa, pb = _resolve(a, xyz, rig), _resolve(b, xyz, rig)
    if pa is None or pb is None:
        return None
    v = pb - pa
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = (n[..., 0] > 1e-9) & np.isfinite(v).all(axis=-1)
    if not ok.any():
        return None
    m = np.median(v[ok] / n[ok], axis=0)
    nm = np.linalg.norm(m)
    return m / nm if nm > 1e-9 else None


def _lateral_axis(xyz: np.ndarray, rig: BodyRig) -> np.ndarray | None:
    """The subject's own left-pointing axis, per frame: hips for preference
    (they twist least), shoulders as the fallback. (..., 3) unit vectors, NaN
    where neither pair is available."""
    for a, b in (("right_hip", "left_hip"), ("right_shoulder", "left_shoulder")):
        pa, pb = _resolve(a, xyz, rig), _resolve(b, xyz, rig)
        if pa is None or pb is None:
            continue
        v = pb - pa
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = v / n
        return np.where(n > 1e-9, u, np.nan)
    return None


def _facing_sign_2d(xyz: np.ndarray, rig: BodyRig) -> np.ndarray:
    """Which way the subject faces in the picture: +1 when they face towards
    increasing x, -1 towards decreasing x, from the nose (or the toes) against
    the pelvis. In an image there is no out-of-plane axis to sign an angle
    with, so this is the honest substitute -- and it is only meaningful for a
    subject filmed side on, which is stated wherever 2D angles are shown."""
    hips = _resolve("mid(left_hip,right_hip)", xyz, rig)
    if hips is None:
        return np.ones(xyz.shape[:-2])
    ahead = None
    for ref in ("nose", "mid(left_big_toe,right_big_toe)", "neck"):
        cand = _resolve(ref, xyz, rig)
        if cand is not None:
            ahead = cand if ahead is None else np.where(np.isnan(ahead), cand, ahead)
    if ahead is None:
        return np.ones(xyz.shape[:-2])
    dx = ahead[..., 0] - hips[..., 0]
    return np.where(np.isfinite(dx) & (np.abs(dx) > 1e-9), np.sign(dx), 1.0)


def _signed(u: np.ndarray, v: np.ndarray, axis: np.ndarray | None,
            flat_sign: np.ndarray | None = None) -> np.ndarray:
    """Signed angle from u to v, degrees in (-180, 180].

    In 3D the sign is the right-hand rule about `axis` (the subject's left),
    so a limb swinging towards the face reads positive on both sides of the
    body. In 2D there is no such axis: the scalar cross product is used and
    multiplied by `flat_sign` (which way the subject faces) so the result
    still means "forwards is positive".
    """
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    dot = np.sum(u * v, axis=-1)
    if u.shape[-1] >= 3:
        cr = np.cross(u, v)
        if axis is None:
            return np.full(dot.shape, np.nan)
        sin = np.sum(cr * axis, axis=-1)
    else:
        sin = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        if flat_sign is not None:
            sin = sin * flat_sign
    ang = np.degrees(np.arctan2(sin, dot))
    ok = (nu > 1e-12) & (nv > 1e-12) & np.isfinite(dot) & np.isfinite(sin)
    return np.where(ok, ang, np.nan)


def _twist(sa: np.ndarray, sb: np.ndarray, ha: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """Signed angle (degrees) from the hip axis to the shoulder axis, measured
    about the trunk axis -- i.e. how far the shoulders are rotated out of
    square with the pelvis. Both axes are projected onto the plane whose
    normal is the trunk direction, so a forward lean does not leak in.

    Sign: the right-hand rule about the axis pointing from the pelvis to the
    shoulders. Equivalently, rotating the hip axis by +theta about that axis
    produces a shoulder axis that reads +theta. Stating it this way (rather
    than "left" or "clockwise") is what makes it checkable: the verification
    suite builds the shoulder axis with an explicit Rodrigues rotation and
    asserts the same number comes back.
    """
    trunk = (sa + sb) / 2.0 - (ha + hb) / 2.0
    tn = np.linalg.norm(trunk, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = trunk / tn
    s = sb - sa
    h = hb - ha
    s = s - np.sum(s * n, axis=-1, keepdims=True) * n
    h = h - np.sum(h * n, axis=-1, keepdims=True) * n
    cross = np.cross(h, s)
    sin = np.sum(cross * n, axis=-1)
    cos = np.sum(h * s, axis=-1)
    ang = np.degrees(np.arctan2(sin, cos))
    bad = (tn[..., 0] <= 1e-12) | (np.linalg.norm(s, axis=-1) <= 1e-12) | \
          (np.linalg.norm(h, axis=-1) <= 1e-12)
    return np.where(bad, np.nan, ang)


def applicable_angles(rig: BodyRig, have_3d: bool) -> list[AngleDef]:
    """The angles this rig can actually produce. Anything referring to a joint
    the rig does not carry is dropped here rather than emitting a column of
    NaN that looks like lost data."""
    out = []
    for d in ANGLE_DEFS:
        if d.needs_3d and not have_3d:
            continue
        refs = []
        for r in d.joints:
            refs.extend(r[4:-1].split(",") if r.startswith("mid(") else [r])
        if rig.has(*[r.strip() for r in refs]):
            out.append(d)
    return out


def joint_angles(xyz: np.ndarray, rig: BodyRig,
                 have_3d: bool | None = None) -> tuple[list[AngleDef], np.ndarray]:
    """Angles for a stack of poses.

    `xyz` is (..., J, D) with D = 2 (image plane) or 3. Returns the angle
    definitions actually computed and an (..., A) array in degrees, NaN
    wherever an input joint was missing.
    """
    xyz = np.asarray(xyz, np.float64)
    dim = xyz.shape[-1]
    if have_3d is None:
        have_3d = dim >= 3
    defs = applicable_angles(rig, bool(have_3d))
    lead = xyz.shape[:-2]
    out = np.full(lead + (len(defs),), np.nan)
    up = rig.up_vector()
    if dim < 3:
        # Dropping the third component can leave a degenerate axis (a rig whose
        # up is +Z has no image-plane up at all). Image rows grow downwards, so
        # the honest fallback for 2D is "up the picture".
        up = up[:dim]
        if np.linalg.norm(up) <= 1e-9:
            up = np.array([0.0, -1.0][:dim])
        up = up / np.linalg.norm(up)
    lateral = _lateral_axis(xyz, rig) if dim >= 3 else None
    flat = _facing_sign_2d(xyz, rig) if dim < 3 else None
    for k, d in enumerate(defs):
        pts = [_resolve(r, xyz, rig) for r in d.joints]
        if any(p is None for p in pts):
            continue
        if d.kind == "joint":
            raw = _interior(pts[0], pts[1], pts[2])
        elif d.kind == "sagittal":
            raw = _signed(pts[0] - pts[1], pts[2] - pts[1], lateral, flat)
        elif d.kind == "axis":
            raw = _signed(np.broadcast_to(up, pts[1].shape), pts[1] - pts[0], lateral, flat)
        elif d.kind == "segments":
            raw = _signed(pts[1] - pts[0], pts[3] - pts[2], lateral, flat)
        elif d.kind == "lean":
            ref = median_axis(xyz, rig, d.joints[0], d.joints[1])
            if ref is None:
                continue
            raw = _signed(np.broadcast_to(ref[:dim], pts[1].shape), pts[1] - pts[0],
                          lateral, flat)
        elif d.kind == "twist":
            if dim < 3:
                continue
            raw = _twist(pts[0], pts[1], pts[2], pts[3])
        else:                                    # pragma: no cover - guarded by ANGLE_DEFS
            continue
        val = d.scale * raw + d.offset
        if d.kind != "joint":
            # A signed angle comes out of atan2 in (-180, 180]. The hip's two
            # segments point nearly opposite ways, so the raw value sits right
            # on that branch cut and a leg trailing behind reads +338 instead
            # of -22 -- a 360 deg jump in the middle of every stride. Fold the
            # transformed value back into (-180, 180], where every anatomical
            # angle belongs.
            val = (val + 180.0) % 360.0 - 180.0
        out[..., k] = val
    return defs, out


def angular_velocity(angles: np.ndarray, fps: float, step: int = 1) -> np.ndarray:
    """deg/s over the frame axis (axis 0), by central differences between each
    value's nearest measured neighbours, one-sided where only one exists.

    `step` is how far apart the run sampled its frames (`BodyTrack.step`). A
    neighbour further away than that is across a real gap and is not used.
    (I90) Differencing ADJACENT frames only gave a run sampled every Nth
    frame no velocity at all, because its neighbours are always blank. With
    step 1 and no gaps this is the plain central difference. A frame with no
    angle has no velocity, even when both its neighbours do."""
    a = np.asarray(angles, np.float64)
    v = np.full(a.shape, np.nan)          # C-contiguous, so reshape below is a view
    if a.shape[0] < 2 or fps <= 0:
        return v
    gap = max(1, int(step or 1))
    flat_a = a.reshape(a.shape[0], -1)
    flat_v = v.reshape(a.shape[0], -1)
    for c in range(flat_a.shape[1]):
        idx = np.nonzero(np.isfinite(flat_a[:, c]))[0]
        if len(idx) < 2:
            continue
        val = flat_a[idx, c]
        near = np.diff(idx) <= gap                    # neighbour i -> i+1 usable
        has_prev = np.r_[False, near]
        has_next = np.r_[near, False]
        prev = np.r_[0, np.arange(len(idx) - 1)]
        nxt = np.r_[np.arange(1, len(idx)), len(idx) - 1]
        lo = np.where(has_prev, prev, np.arange(len(idx)))
        hi = np.where(has_next, nxt, np.arange(len(idx)))
        dt = (idx[hi] - idx[lo]) / float(fps)
        ok = dt > 0
        out = np.full(len(idx), np.nan)
        out[ok] = (val[hi[ok]] - val[lo[ok]]) / dt[ok]
        flat_v[idx, c] = out
    return v


def range_of_motion(angles: np.ndarray) -> dict[str, np.ndarray]:
    """Per-angle min / max / range / median over the frames present."""
    a = np.asarray(angles, np.float64)
    if not len(a):
        blank = np.full(a.shape[1:], np.nan)
        return {"min": blank, "max": blank, "range": blank.copy(), "median": blank}
    # An angle that is NaN on every frame is normal (a joint the backend never
    # saw); numpy's all-NaN warning would then fire once per column.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
        md = np.nanmedian(a, axis=0)
    return {"min": lo, "max": hi, "range": hi - lo, "median": md}


# -------------------------------------------------------------------- track

# A joint the pose model scored below this is treated as UNSEEN: it is not
# drawn and no angle is computed from it. A 2D keypoint model returns
# coordinates for every joint, even one outside the picture, and signals that
# only through its score (an ankle cut off by the frame edge comes back parked
# beside the knee at ~0.02). NaN means "the backend gives no per-joint score"
# and is never treated as low. (I86)
MIN_JOINT_CONF = 0.15


class BodyTrack:
    """Every body pose in one video: (frames x people x joints).

    People are columns, like points are columns in `TrackingSession` -- a
    person who leaves and comes back keeps their column, and a frame with no
    detection is NaN rather than a repeat of the last pose. `n_people` is
    fixed when the track is made; the estimator run decides it.
    """

    def __init__(self, n_frames: int, rig: BodyRig | str = DEFAULT_RIG, n_people: int = 1):
        self.n_frames = int(n_frames)
        self.rig = rig if isinstance(rig, BodyRig) else rig_of(rig)
        self.n_people = max(1, int(n_people))
        J, P, T = self.rig.n_joints, self.n_people, self.n_frames
        self.joints3d = np.full((T, P, J, 3), np.nan, np.float32)
        self.joints2d = np.full((T, P, J, 2), np.nan, np.float32)
        self.conf = np.zeros((T, P, J), np.float32)
        self.score = np.full((T, P), np.nan, np.float32)       # detection / person score
        self.bbox = np.full((T, P, 4), -1, np.float32)         # native px x0,y0,x1,y1
        self.focal = np.full(T, np.nan, np.float32)            # px, the model's own estimate
        # Where the body's own origin sits in the CAMERA frame (metres, x right,
        # y down, z away from the camera), per frame and person. SAM 3D Body
        # reports `joints3d` centred on the body (zero global translation);
        # camera-frame position = joints3d + cam_t. NaN = not known (a 2D
        # backend, or a run made before this was stored). (I85)
        self.cam_t = np.full((T, P, 3), np.nan, np.float32)
        self.names: list[str] = [f"person {i + 1}" for i in range(P)]
        self.backend = ""
        self.has_3d = False
        self.notes = ""            # free text shown in the report ("no calibration used")
        # What the run asked for. A run over a selected window, or one that
        # sampled every Nth frame, must be judged against the frames it was
        # asked to do -- not against the whole video, which would call a
        # perfect 1-in-8 survey of a 40k-frame clip a failure.
        self.n_requested = 0       # 0 = "the whole video"
        self.step = 1              # 1 = every frame
        # Every run that went into this track, as (first, last, step). A later
        # run over part of the video is MERGED in (`merge_run`), so the frames
        # asked for are the union of these; `n_requested` is kept equal to it.
        self.runs: list[tuple[int, int, int]] = []
        # The body MESH, when the backend makes one (SAM 3D Body does). Stored
        # sparsely, per (frame, person), because it is ~18k vertices: dense
        # float32 storage would be 220 kB a frame, or 9 GB over a 40k-frame
        # clip. float16 halves that and is far finer than the model's own
        # accuracy, and the face list is shared by every frame.
        self.faces: np.ndarray | None = None          # (F, 3) int32
        self.mesh: dict[tuple[int, int], np.ndarray] = {}   # (frame, person) -> (V, 3) f16
        # Bumped whenever the mesh dict changes, so a save can tell whether
        # the (large) mesh block needs writing again. (I93)
        self.mesh_version = 0
        # Bumped on every write; keys the per-person caches below. Anything
        # that writes into the arrays directly must call `touch()`.
        self._version = 0
        self._cache: dict = {}
        # Transient, set by `BodyPoseWorker` and read by `merge_run`: the frames
        # the run actually put through the model, and whether it was stopped.
        # Not persisted.
        self.examined: np.ndarray | None = None
        self.stopped_early = False

    def touch(self) -> None:
        """Say the arrays changed (the cached angles / plot data are stale)."""
        self._version += 1

    def _cache_key(self, *extra):
        # the array identities too, so re-assigning an array (a loader, a test)
        # can never serve stale angles
        return (self._version, id(self.joints3d), id(self.joints2d), id(self.conf),
                bool(self.has_3d)) + tuple(extra)

    def cached(self, name: str, key: tuple, make):
        """Per-track memo for derived whole-clip data (angles, the upright
        basis, plot traces). (I89) The side-by-side view redraws on every seek,
        and recomputing whole-video angles each time cost 0.5-1.2 s a frame on
        a 40k-frame video."""
        full = self._cache_key(*key)
        hit = self._cache.get(name)
        if hit is not None and hit[0] == full:
            return hit[1]
        val = make()
        self._cache[name] = (full, val)
        return val

    @staticmethod
    def _frozen_mesh(vertices) -> np.ndarray:
        # Mesh arrays are never written in place (a new run replaces the dict
        # entry), so copies of the track can SHARE them. Read-only enforces it.
        v = np.array(vertices, dtype=np.float16).reshape(-1, 3)   # always our own
        v.flags.writeable = False
        return v

    # ----------------------------------------------------------------- write
    def set_person(self, frame: int, person: int, *, joints2d=None, joints3d=None,
                   conf=None, score: float = np.nan, bbox=None, focal: float = np.nan,
                   vertices=None, faces=None, cam_t=None) -> None:
        if not (0 <= frame < self.n_frames and 0 <= person < self.n_people):
            return
        if faces is not None and self.faces is None:
            self.faces = np.asarray(faces, np.int32).reshape(-1, 3)
        if vertices is not None:
            self.mesh[(int(frame), int(person))] = self._frozen_mesh(vertices)
            self.mesh_version += 1
        if joints2d is not None:
            self.joints2d[frame, person] = np.asarray(joints2d, np.float32).reshape(-1, 2)
        if joints3d is not None:
            self.joints3d[frame, person] = np.asarray(joints3d, np.float32).reshape(-1, 3)
            self.has_3d = True
        if conf is not None:
            self.conf[frame, person] = np.asarray(conf, np.float32).reshape(-1)
        elif joints2d is not None or joints3d is not None:
            # no score given = no score known, which is NOT the same as zero:
            # zero would hide every joint and every angle (I86)
            self.conf[frame, person] = np.nan
        self.score[frame, person] = float(score)
        if bbox is not None:
            self.bbox[frame, person] = np.asarray(bbox, np.float32).reshape(4)
        if np.isfinite(focal):
            self.focal[frame] = float(focal)
        if cam_t is not None:
            self.cam_t[frame, person] = np.asarray(cam_t, np.float32).reshape(3)
        self._version += 1

    def clear(self, start: int, end: int) -> int:
        """Blank a frame window. Returns how many (frame, person) cells went."""
        a, b = max(0, int(start)), min(self.n_frames - 1, int(end))
        if b < a:
            return 0
        n = int(np.isfinite(self.score[a:b + 1]).sum())
        gone = [k for k in self.mesh if a <= k[0] <= b]
        for key in gone:
            del self.mesh[key]
        if gone:
            self.mesh_version += 1
        self.joints3d[a:b + 1] = np.nan
        self.joints2d[a:b + 1] = np.nan
        self.conf[a:b + 1] = 0.0
        self.score[a:b + 1] = np.nan
        self.bbox[a:b + 1] = -1
        self.focal[a:b + 1] = np.nan
        self.cam_t[a:b + 1] = np.nan
        self._version += 1
        return n

    def clear_frames(self, frames) -> None:
        """Blank a set of individual frames (every person)."""
        f = np.asarray(frames, np.int64).reshape(-1)
        f = f[(f >= 0) & (f < self.n_frames)]
        if not len(f):
            return
        fs = set(int(x) for x in f)
        gone = [k for k in self.mesh if k[0] in fs]
        for key in gone:
            del self.mesh[key]
        if gone:
            self.mesh_version += 1
        self.joints3d[f] = np.nan
        self.joints2d[f] = np.nan
        self.conf[f] = 0.0
        self.score[f] = np.nan
        self.bbox[f] = -1
        self.focal[f] = np.nan
        self.cam_t[f] = np.nan
        self._version += 1

    def copy(self) -> "BodyTrack":
        bt = BodyTrack(self.n_frames, self.rig, self.n_people)
        bt.joints3d = self.joints3d.copy()
        bt.joints2d = self.joints2d.copy()
        bt.conf = self.conf.copy()
        bt.score = self.score.copy()
        bt.bbox = self.bbox.copy()
        bt.focal = self.focal.copy()
        bt.cam_t = self.cam_t.copy()
        bt.names = list(self.names)
        bt.backend = self.backend
        bt.has_3d = self.has_3d
        bt.notes = self.notes
        bt.n_requested = self.n_requested
        bt.step = self.step
        bt.runs = list(self.runs)
        bt.faces = None if self.faces is None else self.faces.copy()
        # (I93) The mesh arrays are read-only and never written in place, so
        # the copy SHARES them: an undo snapshot of a long meshed run used to
        # duplicate ~110 kB per frame (4.5 GB over 40k frames). Only the dict
        # is new, so removing or replacing an entry in one copy never touches
        # the other.
        bt.mesh = dict(self.mesh)
        bt.mesh_version = self.mesh_version
        return bt

    # ------------------------------------------------------------------ read
    def has(self, frame: int, person: int = 0) -> bool:
        return (0 <= frame < self.n_frames and 0 <= person < self.n_people
                and bool(np.isfinite(self.score[frame, person])))

    def frames(self, person: int | None = None) -> np.ndarray:
        """Frames with at least one person (or with this person)."""
        ok = np.isfinite(self.score)
        return np.nonzero(ok[:, person] if person is not None else ok.any(axis=1))[0]

    def n_posed(self) -> int:
        return int(np.isfinite(self.score).any(axis=1).sum())

    def person_score(self, person: int) -> float:
        """Mean detector confidence for one person over the frames they appear
        on. A weak box still yields a confident-looking body -- a top-down
        model always returns one -- so this is the number that says whether a
        column is a real person or a coat stand."""
        s = self.score[:, person]
        s = s[np.isfinite(s)]
        return float(s.mean()) if len(s) else float("nan")

    def person_label(self, person: int) -> str:
        nm = self.names[person] if person < len(self.names) else f"person {person + 1}"
        sc = self.person_score(person)
        n = len(self.frames(person))
        return nm if not np.isfinite(sc) else f"{nm}  ({n} frames, found {sc:.0%})"

    def people_at(self, frame: int) -> list[int]:
        if not (0 <= frame < self.n_frames):
            return []
        return [p for p in range(self.n_people) if np.isfinite(self.score[frame, p])]

    def pose2d(self, frame: int, person: int = 0) -> np.ndarray | None:
        return self.joints2d[frame, person] if self.has(frame, person) else None

    def has_mesh(self, frame: int | None = None, person: int = 0) -> bool:
        if self.faces is None or not self.mesh:
            return False
        return True if frame is None else (int(frame), int(person)) in self.mesh

    def mesh_at(self, frame: int, person: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
        """(vertices float32, faces int32) for one person on one frame."""
        v = self.mesh.get((int(frame), int(person)))
        if v is None or self.faces is None:
            return None
        return v.astype(np.float32), self.faces

    def mesh_frames(self) -> np.ndarray:
        return np.array(sorted({k[0] for k in self.mesh}), np.int64)

    def mesh_bytes(self) -> int:
        return sum(v.nbytes for v in self.mesh.values()) + (
            0 if self.faces is None else self.faces.nbytes)

    def pose3d(self, frame: int, person: int = 0) -> np.ndarray | None:
        """3D joints, or None when THIS person on THIS frame has none.
        `has_3d` is a property of the track as a whole, so it is not enough:
        a 3D backend that fell back to 2D for one person would otherwise hand
        back a row of NaN that reads as a pose."""
        if not (self.has_3d and self.has(frame, person)):
            return None
        row = self.joints3d[frame, person]
        return row if np.isfinite(row).any() else None

    def bones(self) -> list[tuple[int, int]]:
        return self.rig.bones()

    # ---------------------------------------------------------------- angles
    def joints_for_angles(self, person: int = 0) -> np.ndarray:
        """(T, J, D) float64 joints the angles are measured from: 3D when the
        backend produced it, else the image-plane keypoints, with every joint
        the model scored below MIN_JOINT_CONF blanked. (I86) A 2D keypoint
        model hands back coordinates for a joint it did not see; an angle
        built from one is a confident-looking number about nothing."""
        src = self.joints3d if self.has_3d else self.joints2d
        xyz = src[:, person].astype(np.float64)
        low = self.conf[:, person] < MIN_JOINT_CONF          # NaN (no score) -> False
        if low.any():
            xyz[low] = np.nan
        return xyz

    def angles(self, person: int = 0) -> tuple[list[AngleDef], np.ndarray]:
        """(defs, (T, A)) for one person, from 3D when the backend produced it,
        otherwise from the image-plane keypoints. Cached until the track
        changes (I89); the array is read-only because it is shared."""
        def make():
            defs, vals = joint_angles(self.joints_for_angles(person), self.rig,
                                      have_3d=bool(self.has_3d))
            vals.flags.writeable = False
            return defs, vals
        return self.cached(f"angles{int(person)}", (int(person),), make)

    def angle_source(self) -> str:
        return "3D" if self.has_3d else "2D (image plane)"

    def summary(self) -> str:
        """One plain-English line for the status bar and the report."""
        n = self.n_posed()
        if not n:
            return "no body pose yet"
        who = "1 person" if self.n_people == 1 else f"{self.n_people} people"
        extra = ""
        if self.has_mesh():
            extra = (f", 3D body shape on {len(self.mesh_frames())} frames "
                     f"({self.mesh_bytes() / 1e6:.0f} MB)")
        return (f"{n} frame{'s' if n != 1 else ''} posed, {who}, "
                f"{self.rig.label}, angles from {self.angle_source()}{extra}")

    # ----------------------------------------------------------- persistence
    def mesh_arrays(self, prefix: str) -> dict:
        """The mesh block alone. Meshes are sparse: one concatenated vertex
        block plus offsets, the same shape of storage MaskTrack uses for its
        contours. Separate from `to_arrays(mesh=False)` so a save can write
        this large, barely compressible block (float16 vertices compress to
        ~93 %) uncompressed, or only when `mesh_version` changed. (I93)"""
        keys = sorted(self.mesh)
        verts, off = [], [0]
        for k in keys:
            verts.append(self.mesh[k])
            off.append(off[-1] + len(self.mesh[k]))
        return {
            f"{prefix}mesh_faces": (self.faces if self.faces is not None
                                    else np.zeros((0, 3), np.int32)).astype(np.int32),
            f"{prefix}mesh_verts": (np.concatenate(verts) if verts
                                    else np.zeros((0, 3), np.float16)).astype(np.float16),
            f"{prefix}mesh_off": np.asarray(off, np.int64),
            f"{prefix}mesh_key": np.asarray(keys, np.int64).reshape(-1, 2),
        }

    def to_arrays(self, prefix: str, mesh: bool = True) -> dict:
        return {
            **(self.mesh_arrays(prefix) if mesh else {}),
            f"{prefix}cam_t": self.cam_t,
            f"{prefix}joints3d": self.joints3d,
            f"{prefix}joints2d": self.joints2d,
            f"{prefix}conf": self.conf,
            f"{prefix}score": self.score,
            f"{prefix}bbox": self.bbox,
            f"{prefix}focal": self.focal,
            f"{prefix}meta": json.dumps({
                "rig": self.rig.name, "names": self.names, "backend": self.backend,
                "has_3d": bool(self.has_3d), "notes": self.notes,
                "up": list(self.rig.up), "units": self.rig.units,
                "n_requested": int(self.n_requested), "step": int(self.step),
                "runs": [[int(a), int(b), int(c)] for a, b, c in self.runs],
            }),
        }

    @classmethod
    def from_arrays(cls, prefix: str, arrays) -> "BodyTrack":
        meta = json.loads(str(arrays[f"{prefix}meta"]))
        j3 = np.asarray(arrays[f"{prefix}joints3d"], np.float32)
        bt = cls(j3.shape[0], rig_of(meta.get("rig", DEFAULT_RIG)), j3.shape[1])
        bt.joints3d = j3
        bt.joints2d = np.asarray(arrays[f"{prefix}joints2d"], np.float32)
        bt.conf = np.asarray(arrays[f"{prefix}conf"], np.float32)
        bt.score = np.asarray(arrays[f"{prefix}score"], np.float32)
        bt.bbox = np.asarray(arrays[f"{prefix}bbox"], np.float32)
        bt.focal = np.asarray(arrays[f"{prefix}focal"], np.float32)
        bt.names = list(meta.get("names") or bt.names)
        bt.backend = str(meta.get("backend", ""))
        bt.has_3d = bool(meta.get("has_3d", False))
        bt.notes = str(meta.get("notes", ""))
        bt.n_requested = int(meta.get("n_requested", 0) or 0)
        bt.step = max(1, int(meta.get("step", 1) or 1))
        try:
            bt.runs = [(int(a), int(b), max(1, int(c))) for a, b, c in meta.get("runs") or []]
        except (TypeError, ValueError):
            bt.runs = []
        have = set(getattr(arrays, "files", arrays))
        if f"{prefix}cam_t" in have:                  # older files: not stored (NaN)
            ct = np.asarray(arrays[f"{prefix}cam_t"], np.float32)
            if ct.shape == bt.cam_t.shape:
                bt.cam_t = ct
        if f"{prefix}mesh_faces" in have:
            faces = np.asarray(arrays[f"{prefix}mesh_faces"], np.int32)
            bt.faces = faces.reshape(-1, 3) if len(faces) else None
            vs = np.asarray(arrays[f"{prefix}mesh_verts"], np.float16)
            off = np.asarray(arrays[f"{prefix}mesh_off"], np.int64)
            keys = np.asarray(arrays[f"{prefix}mesh_key"], np.int64).reshape(-1, 2)
            for i, (f, p) in enumerate(keys):
                v = vs[off[i]:off[i + 1]]
                v.flags.writeable = False             # shared by copies (see copy())
                bt.mesh[(int(f), int(p))] = v
            bt.mesh_version = 1 if len(keys) else 0
        return bt


def _runs_mask(runs, n_frames: int) -> np.ndarray:
    m = np.zeros(max(0, int(n_frames)), bool)
    for a, b, c in runs:
        a, b = max(0, int(a)), min(int(n_frames) - 1, int(b))
        if b >= a:
            m[a:b + 1:max(1, int(c))] = True
    return m


def merge_run(old: BodyTrack | None, new: BodyTrack) -> tuple[BodyTrack, str]:
    """Fold a finished (or stopped) pose run into the track the view already
    has. Returns (the track to store, one plain sentence for the user, or ""
    when there was nothing to merge with).

    A run replaces exactly the frames it put through the model
    (`new.examined`) and leaves every other frame as it was. (I82) Swapping
    the new track in whole threw away every pose outside a re-run's range,
    and a stopped run kept only what it had reached -- hours of SAM 3D Body
    work gone to fix 100 frames. When the two cannot be merged (another joint
    set, number of people or video length) the new run replaces the old one
    and the sentence says so; the caller's undo snapshot can take it back."""
    ex = new.examined
    if ex is None:
        ex = new.frames()
    ex = np.unique(np.asarray(ex, np.int64).reshape(-1))
    ex = ex[(ex >= 0) & (ex < new.n_frames)]
    new.examined, new.stopped_early, stopped = None, False, bool(new.stopped_early)
    if old is None or old.n_posed() == 0:
        return new, ""
    if (old.n_frames != new.n_frames or old.rig.name != new.rig.name
            or old.n_people != new.n_people):
        why = ("a different joint set" if old.rig.name != new.rig.name else
               "a different number of people" if old.n_people != new.n_people else
               "a different video length")
        return new, (f"This run used {why} from the earlier one, so it REPLACED the earlier "
                     f"poses ({old.n_posed()} frames). Ctrl+Z brings them back.")
    out = old.copy()
    out.clear_frames(ex)
    for name in ("joints3d", "joints2d", "conf", "score", "bbox", "cam_t"):
        getattr(out, name)[ex] = getattr(new, name)[ex]
    got = np.isfinite(new.focal[ex])
    out.focal[ex[got]] = new.focal[ex[got]]
    ex_set = set(int(f) for f in ex)
    for key, v in new.mesh.items():
        if key[0] in ex_set:
            out.mesh[key] = v
            out.mesh_version += 1
    if out.faces is None and new.faces is not None:
        out.faces = new.faces
    out.has_3d = bool(out.has_3d or new.has_3d)
    if new.backend and new.backend not in out.backend:
        out.backend = f"{out.backend} + {new.backend}" if out.backend else new.backend
    parts = [p for p in (out.notes.split("  ") + new.notes.split("  ")) if p.strip()]
    out.notes = "  ".join(dict.fromkeys(parts))           # unique, in order
    # what was asked for is the union of the runs
    old_runs = out.runs or ([(0, out.n_frames - 1, 1)] if not out.n_requested else
                            [(int(old.frames()[0]), int(old.frames()[-1]), old.step)])
    out.runs = old_runs + (new.runs or [])
    out.n_requested = int(_runs_mask(out.runs, out.n_frames).sum())
    kept = int(np.isfinite(out.score).any(axis=1).sum()) - int(
        np.isfinite(out.score[ex]).any(axis=1).sum())
    out.step = max(out.step, new.step) if kept else new.step
    out.touch()
    if not len(ex):
        return out, "The run stopped before it finished a frame; the poses are unchanged."
    span = (f"frame {int(ex[0])} was" if len(ex) == 1 else
            f"frames {int(ex[0])}-{int(ex[-1])} were")
    if stopped:
        return out, (f"Stopped early: {span} posed again; every other frame keeps its "
                     f"earlier pose ({kept} frames).")
    return out, (f"{span[0].upper()}{span[1:]} posed again; the earlier poses on the other "
                 f"{kept} frames were kept.")


# ------------------------------------------------------------------ exports

def _fmt(v) -> str:
    return "" if v is None or not np.isfinite(v) else f"{float(v):.4f}"


def has_joint_scores(track: BodyTrack) -> bool:
    """True when the per-joint confidence is a real score. SAM 3D Body gives
    none: new runs store NaN, runs made before 2026-09-22 stored a constant
    1.0, and neither may be written out as if the model were sure. (I85)"""
    vals = track.conf[np.isfinite(track.score)]
    fin = vals[np.isfinite(vals)]
    return bool(len(fin)) and not bool(np.all(fin == 1.0))


def _comment(f, text: str) -> None:
    """One '#' comment line, written raw. (I92) Through csv.writer any comma in
    the text made it quote the whole line, so it started with '"#' and every
    comment-aware reader (pandas comment='#', numpy, R, MATLAB) took it for
    data or for the header."""
    f.write("# " + " ".join(str(text).split()) + "\n")


def export_joints_csv(track: BodyTrack, path: str | Path) -> None:
    """One row per frame and person; x/y in native video pixels, plus X/Y/Z in
    metres when the backend is a 3D one. Blank means no data -- never a stale
    or interpolated value. The '#' lines at the top say what every column's
    origin, axes and units are, and whether the confidence is a real score."""
    rig = track.rig
    scored = has_joint_scores(track)
    with open(path, "w", newline="", encoding="utf-8") as f:
        _comment(f, f"Kinetrace joint positions - rig {rig.label}"
                    + (f" - backend {track.backend}" if track.backend else ""))
        _comment(f, "x, y: native video pixels, origin at the top-left corner of the "
                    "picture, y growing downwards")
        _comment(f, (f"conf: the model's score for each joint (0-1); joints under "
                     f"{MIN_JOINT_CONF:g} are treated as unseen and give no angle")
                 if scored else
                 "conf: blank - this model gives no per-joint confidence, so a hidden "
                 "joint is a guess that looks as sure as a visible one")
        if track.has_3d:
            # (I85) SAM 3D Body reports the pose centred on the body; only its
            # camera translation places it in the scene. Without it, pelvis X/Z
            # never move and speed / stride / distances between people are ~0.
            _comment(f, "X, Y, Z: metres in the CAMERA's frame when xyz_frame is 'camera' "
                        "(origin = the camera, X right, Y down, Z away from the camera); "
                        "'body' rows are centred on the body itself, so they give angles "
                        "and segment lengths but NOT movement across frames")
            fl = track.focal[np.isfinite(track.focal)]
            if len(fl):
                _comment(f, f"focal length used: median {float(np.median(fl)):.0f} px. Unless "
                            f"the notes say a lens calibration was used, the model guessed "
                            f"it, and absolute distances scale with that guess (angles do not)")
        if track.notes:
            _comment(f, track.notes)
        w = csv.writer(f)
        head = ["frame", "person"]
        for j in rig.joints:
            head += [f"{j}_x", f"{j}_y", f"{j}_conf"]
        if track.has_3d:
            head += ["xyz_frame"]
            for j in rig.joints:
                head += [f"{j}_X", f"{j}_Y", f"{j}_Z"]
        w.writerow(head)
        for t in range(track.n_frames):
            for p in track.people_at(t):
                row = [t, track.names[p] if p < len(track.names) else f"person {p + 1}"]
                for k in range(rig.n_joints):
                    row += [_fmt(track.joints2d[t, p, k, 0]), _fmt(track.joints2d[t, p, k, 1]),
                            _fmt(track.conf[t, p, k]) if scored else ""]
                if track.has_3d:
                    ct = track.cam_t[t, p].astype(np.float64)
                    cam = bool(np.isfinite(ct).all())
                    xyz = track.joints3d[t, p].astype(np.float64) + (ct if cam else 0.0)
                    row += ["camera" if cam else "body"]
                    for k in range(rig.n_joints):
                        row += [_fmt(v) for v in xyz[k]]
                w.writerow(row)


def export_angles_csv(track: BodyTrack, path: str | Path, fps: float = 0.0) -> None:
    """Joint angles in degrees, one row per frame and person, with the
    convention for every column written into the header block so the file
    still explains itself a year later."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        defs, _ = track.angles(0)
        _comment(f, f"Kinetrace joint angles - degrees - measured in "
                    f"{track.angle_source()} - rig {track.rig.label}"
                    + (f" - backend {track.backend}" if track.backend else ""))
        for d in defs:
            _comment(f, f"{d.name}: {d.zero_means}")
        if has_joint_scores(track):
            _comment(f, f"a joint the model scored under {MIN_JOINT_CONF:g} is treated as "
                        f"unseen: any angle that needs it is blank on that frame")
        if fps > 0 and track.step > 1:
            _comment(f, f"deg/s: differences between the sampled frames ({track.step} "
                        f"frames apart), not between neighbouring video frames")
        if track.notes:
            _comment(f, track.notes)
        w = csv.writer(f)
        head = ["frame"] + (["time_s"] if fps > 0 else []) + ["person"] + [d.name for d in defs]
        if fps > 0:
            head += [d.name + " (deg/s)" for d in defs]
        w.writerow(head)
        per_person = {}
        for p in range(track.n_people):
            _, a = track.angles(p)
            per_person[p] = (a, angular_velocity(a, fps, track.step) if fps > 0 else None)
        for t in range(track.n_frames):
            for p in track.people_at(t):
                a, v = per_person[p]
                row = [t] + ([f"{t / fps:.6f}"] if fps > 0 else [])
                row += [track.names[p] if p < len(track.names) else f"person {p + 1}"]
                row += [_fmt(x) for x in a[t]]
                if v is not None:
                    row += [_fmt(x) for x in v[t]]
                w.writerow(row)


def angle_report(track: BodyTrack, fps: float = 0.0) -> str:
    """Plain-language verdict + range of motion, for the dialog and a sidecar
    .txt. Written for a reader who has never done motion capture."""
    lines = []
    n = track.n_posed()
    total = track.n_requested or track.n_frames
    lines.append("Kinetrace body pose report")
    lines.append("=" * 26)
    lines.append("")
    lines.append(f"Model            : {track.backend or 'unknown'}")
    lines.append(f"Joint set        : {track.rig.label}")
    asked = ("frames asked for" if track.n_requested else "frames in the video")
    lines.append(f"Frames with a person: {n} of {total} {asked}")
    if track.step > 1:
        lines.append(f"Sampled           : every {track.step} frames"
                     f" (the frames in between were never looked at)")
    lines.append(f"People           : {track.n_people}")
    lines.append(f"Angles measured in  : {track.angle_source()}")
    lines.append("")
    if not track.has_3d:
        lines.append("These angles are measured in the image, not in space. A limb that")
        lines.append("points towards or away from the camera looks shorter than it is, so")
        lines.append("its angle reads smaller than the real one. Treat them as a guide and")
        lines.append("compare like with like (the same camera, the same direction of travel).")
        lines.append("")
    if has_joint_scores(track):
        lines.append(f"A joint the model scored under {MIN_JOINT_CONF:.0%} (a foot cut off by the")
        lines.append("edge of the picture, say) is treated as unseen: it is not drawn and")
        lines.append("no angle is computed from it, so that angle is blank on that frame.")
        lines.append("")
    cover = n / total if total else 0.0
    if cover >= 0.9:
        verdict = "good - a person was found on almost every frame"
    elif cover >= 0.5:
        verdict = "ok - a person was found on most frames; check the gaps on the timeline"
    elif n:
        verdict = "poor - the person was found on a minority of frames"
    else:
        verdict = "nothing found - no person was detected anywhere in the range"
    lines.append(f"Verdict          : {verdict}")
    lines.append("")
    for p in range(track.n_people):
        defs, a = track.angles(p)
        if not np.isfinite(a).any():
            continue
        rom = range_of_motion(a)
        nm = track.names[p] if p < len(track.names) else f"person {p + 1}"
        sc = track.person_score(p)
        lines.append(f"{nm} - range of motion (degrees)")
        if np.isfinite(sc):
            lines.append(f"  found on {len(track.frames(p))} frames, "
                         f"mean detector confidence {sc:.0%}"
                         + ("  <-- LOW: check this is really a person" if sc < 0.6 else ""))
        lines.append(f"  {'angle':<34}{'min':>9}{'max':>9}{'range':>9}{'median':>9}")
        for k, d in enumerate(defs):
            if not np.isfinite(rom['range'][k]):
                continue
            lines.append(f"  {d.name:<34}{rom['min'][k]:>9.1f}{rom['max'][k]:>9.1f}"
                         f"{rom['range'][k]:>9.1f}{rom['median'][k]:>9.1f}")
        lines.append("")
        lines.append("  What 0 means for each angle:")
        for d in defs:
            lines.append(f"    {d.name}: {d.zero_means}")
        lines.append("")
    return "\n".join(lines)
