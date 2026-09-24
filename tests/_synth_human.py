"""Synthetic walking human with EXACT ground-truth joint angles, shared by the
body-layer tests.

The pose is built by forward kinematics *from* the angles: a knee is placed by
rotating the shank away from the thigh by a chosen number of degrees, so the
angle the measurement code recovers can be compared with the number that
generated the pose rather than with another estimate of it. That makes the
whole chain -- joints to angles to storage to export -- checkable without a
GPU, without weights and without any footage of a real person.

Coordinates are in the CAMERA convention the monocular body models use:
x right, y DOWN, z away from the camera, metres. That is also the convention
`body.RIGS["mhr70"].up` describes, so the gravity-referenced angles (trunk
lean) are exercised against a real convention rather than a convenient one.

`render_frame` draws a shaded figure good enough to look at and to run a real
pose model over; it is deliberately plain (no texture, no background clutter)
because its job is ground truth, not photorealism.
"""
from __future__ import annotations

import numpy as np

try:                                   # cv2 is only needed for the renderer
    import cv2
except ImportError:                    # pragma: no cover
    cv2 = None

W, H, T = 960, 720, 120
FPS = 30.0
FOCAL = 900.0
DEPTH = 3.6                 # metres from the camera; sets how big the figure is
CX, CY = W / 2.0, H / 2.0

# Segment lengths of a ~1.75 m adult, metres.
SEG = dict(trunk=0.52, neck=0.11, head=0.115, shoulder_w=0.38, hip_w=0.25,
           upper_arm=0.30, forearm=0.26, thigh=0.44, shank=0.42, foot=0.19,
           heel=0.07)

# The angles the gait is built from. Every one of these is recovered by
# `body.joint_angles` and asserted in tests/verify_body.py.
GT_ANGLES = ("left hip flexion", "right hip flexion",
             "left knee flexion", "right knee flexion",
             "left elbow flexion", "right elbow flexion",
             "left shoulder flexion", "right shoulder flexion",
             "left ankle dorsiflexion", "right ankle dorsiflexion",
             "trunk lean from vertical")


def _rot(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation of `v` about a unit `axis` by `deg` degrees."""
    th = np.radians(deg)
    k = axis / np.linalg.norm(axis)
    return (v * np.cos(th) + np.cross(k, v) * np.sin(th)
            + k * np.dot(k, v) * (1.0 - np.cos(th)))


def gait(t: int) -> dict:
    """The angles that define frame `t`, in degrees. A plain walking cycle
    plus a slow arm raise, so every angle sweeps a decent range."""
    ph = 2.0 * np.pi * t / 40.0            # 40-frame stride
    raise_ = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / float(T)))   # 0 -> 1 -> 0
    return {
        "left hip flexion": 22.0 * np.sin(ph),
        "right hip flexion": 22.0 * np.sin(ph + np.pi),
        "left knee flexion": 12.0 + 28.0 * max(0.0, np.sin(ph + 0.9)),
        "right knee flexion": 12.0 + 28.0 * max(0.0, np.sin(ph + 0.9 + np.pi)),
        "left elbow flexion": 25.0 + 15.0 * np.sin(ph + np.pi),
        "right elbow flexion": 25.0 + 15.0 * np.sin(ph),
        "left shoulder flexion": 12.0 + 60.0 * raise_,
        "right shoulder flexion": 12.0 + 60.0 * raise_,
        "left ankle dorsiflexion": 8.0 * np.sin(ph + 1.5),
        "right ankle dorsiflexion": 8.0 * np.sin(ph + 1.5 + np.pi),
        "trunk lean from vertical": 6.0 + 7.0 * np.sin(2.0 * np.pi * t / 90.0),
    }


def pose_3d(t: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """(joint name -> (3,) camera-frame metres, angle name -> degrees).

    Signs follow the anatomy: the knee folds backwards, the hip and shoulder
    swing in the sagittal plane, and the subject walks across the view so the
    limbs are never hidden behind the trunk.
    """
    a = gait(t)
    up = np.array([0.0, -1.0, 0.0])              # camera y points down
    # The subject is filmed SIDE ON -- the classic gait view. Their left hand
    # side points away from the camera (+z), so the sagittal plane in which
    # hips, knees, shoulders and ankles swing is the image plane, and the 2D
    # measurement can be compared with the true 3D one on equal terms.
    side = np.array([0.0, 0.0, 1.0])             # subject's left
    lateral = side                               # rotation axis of the sagittal swing
    facing = np.array([1.0, 0.0, 0.0])           # cross(side, up): they walk to the right

    pelvis = np.array([-0.45 + 0.9 * t / max(1, T - 1), 0.08 * np.sin(4.0 * np.pi * t / 40.0),
                       DEPTH])

    trunk_dir = _rot(up, lateral, a["trunk lean from vertical"])
    down = -trunk_dir
    j: dict[str, np.ndarray] = {}
    mid_sh = pelvis + SEG["trunk"] * trunk_dir
    neck = mid_sh + 0.35 * SEG["neck"] * trunk_dir
    head_c = mid_sh + (SEG["neck"] + SEG["head"]) * trunk_dir
    j["neck"] = neck
    j["nose"] = head_c + 0.09 * facing + 0.01 * trunk_dir
    j["left_eye"] = head_c + 0.035 * side + 0.075 * facing + 0.03 * trunk_dir
    j["right_eye"] = head_c - 0.035 * side + 0.075 * facing + 0.03 * trunk_dir
    j["left_ear"] = head_c + 0.085 * side + 0.02 * trunk_dir
    j["right_ear"] = head_c - 0.085 * side + 0.02 * trunk_dir

    for s, sgn in (("left", 1.0), ("right", -1.0)):
        sh = mid_sh + sgn * SEG["shoulder_w"] / 2.0 * side
        hip = pelvis + sgn * SEG["hip_w"] / 2.0 * side
        j[f"{s}_shoulder"] = sh
        j[f"{s}_acromion"] = sh + 0.02 * trunk_dir
        j[f"{s}_hip"] = hip

        # arm: 0 deg elevation = upper arm straight down alongside the trunk
        ua = _rot(down, lateral, -a[f"{s} shoulder flexion"])
        elbow = sh + SEG["upper_arm"] * ua
        fa = _rot(ua, lateral, -a[f"{s} elbow flexion"])
        wrist = elbow + SEG["forearm"] * fa
        j[f"{s}_elbow"] = elbow
        j[f"{s}_wrist"] = wrist

        # leg: 0 deg hip flexion = thigh in line with the trunk
        th = _rot(down, lateral, -a[f"{s} hip flexion"])
        knee = hip + SEG["thigh"] * th
        sh_dir = _rot(th, lateral, a[f"{s} knee flexion"])     # knee folds backwards
        ankle = knee + SEG["shank"] * sh_dir
        j[f"{s}_knee"] = knee
        j[f"{s}_ankle"] = ankle
        # 0 deg dorsiflexion = foot square to the shin, toes FORWARD (+x, the
        # way the subject walks). The sign here matches the real MHR rig: a
        # neutral standing foot is +90 deg from the shin about the subject's
        # left axis, measured heel-to-toe.
        foot_dir = _rot(-sh_dir, lateral, (90.0 - a[f"{s} ankle dorsiflexion"]))
        j[f"{s}_big_toe"] = ankle + SEG["foot"] * foot_dir
        j[f"{s}_small_toe"] = ankle + SEG["foot"] * foot_dir + 0.04 * sgn * side
        j[f"{s}_heel"] = ankle - SEG["heel"] * foot_dir
    return j, a


def project(p3: np.ndarray) -> np.ndarray:
    """Pinhole projection into pixels; (..., 3) -> (..., 2)."""
    p = np.asarray(p3, np.float64)
    z = np.maximum(p[..., 2], 1e-6)
    return np.stack([FOCAL * p[..., 0] / z + CX, FOCAL * p[..., 1] / z + CY], axis=-1)


def truth(n_frames: int = T) -> dict:
    """Ground truth for the whole clip: joint names, (T, J, 3) camera-frame
    metres, (T, J, 2) pixels, and the angles each frame was built from."""
    names = sorted(pose_3d(0)[0])
    xyz = np.full((n_frames, len(names), 3), np.nan)
    for t in range(n_frames):
        j, _ = pose_3d(t)
        for k, nm in enumerate(names):
            xyz[t, k] = j[nm]
    ang = {a: np.array([gait(t)[a] for t in range(n_frames)]) for a in GT_ANGLES}
    return {"names": names, "xyz": xyz, "uv": project(xyz), "angles": ang,
            "fps": FPS, "size": (W, H)}


# ----------------------------------------------------------------- rendering

_SKIN = (140, 168, 198)
_SHIRT = (150, 96, 60)
_TROUSER = (72, 62, 54)
_SHOE = (40, 40, 44)


def _limb(img, a, b, r0, r1, colour):
    """A tapered capsule between two projected points, softly shaded."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return
    u = d / n
    p = np.array([-u[1], u[0]])
    quad = np.array([a + p * r0, b + p * r1, b - p * r1, a - p * r0])
    cv2.fillConvexPoly(img, np.round(quad).astype(np.int32), colour, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(a).astype(int)), int(round(r0)), colour, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(b).astype(int)), int(round(r1)), colour, -1, cv2.LINE_AA)
    hi = tuple(min(255, int(c * 1.18)) for c in colour)
    cv2.line(img, tuple(np.round(a - p * r0 * 0.45).astype(int)),
             tuple(np.round(b - p * r1 * 0.45).astype(int)), hi,
             max(1, int(round(min(r0, r1) * 0.5))), cv2.LINE_AA)


def background(t: int) -> np.ndarray:
    """The scene with nobody in it. Kept separate so `person_mask` can be an
    exact silhouette rather than an approximation of one."""
    if cv2 is None:                                    # pragma: no cover
        raise RuntimeError("cv2 is required to render the synthetic human")
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = (96, 92, 86)
    grad = np.linspace(0.75, 1.3, H).reshape(H, 1, 1)
    img = np.clip(img * grad, 0, 255).astype(np.uint8)
    j3, _ = pose_3d(t)
    uv = {k: project(v) for k, v in j3.items()}
    horizon = int(max(uv["left_heel"][1], uv["right_heel"][1]))
    cv2.rectangle(img, (0, horizon), (W, H), (108, 116, 130), -1)
    cv2.line(img, (0, horizon), (W, horizon), (126, 134, 148), 2, cv2.LINE_AA)
    return img


def person_mask(t: int) -> np.ndarray:
    """Exact bool silhouette of the figure: what the segment tool would give
    on this frame, used to drive a pose model the way a user would.

    The soft ground shadow is part of the figure's drawing, so it is excluded
    by requiring a colour change rather than any change at all."""
    fg = render_frame(t).astype(np.int16)
    bg = background(t).astype(np.int16)
    d = np.abs(fg - bg)
    # the shadow darkens all three channels about equally; the body does not
    spread = d.max(axis=2) - d.min(axis=2)
    m = (d.max(axis=2) > 18) & (spread > 6)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return m.astype(bool)


def render_frame(t: int) -> np.ndarray:
    """BGR frame of the synthetic walker on a plain graded background."""
    if cv2 is None:                                    # pragma: no cover
        raise RuntimeError("cv2 is required to render the synthetic human")
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = (96, 92, 86)
    grad = np.linspace(0.75, 1.3, H).reshape(H, 1, 1)
    img = np.clip(img * grad, 0, 255).astype(np.uint8)

    j3, _ = pose_3d(t)
    uv = {k: project(v) for k, v in j3.items()}
    scale = FOCAL / DEPTH                               # px per metre at the subject
    # the floor sits at the feet, so the figure is never standing in mid-air
    horizon = int(max(uv["left_heel"][1], uv["right_heel"][1]))
    cv2.rectangle(img, (0, horizon), (W, H), (108, 116, 130), -1)
    cv2.line(img, (0, horizon), (W, horizon), (126, 134, 148), 2, cv2.LINE_AA)

    def R(m):
        return max(2.0, m * scale)

    # ground shadow
    feet = (uv["left_ankle"] + uv["right_ankle"]) / 2
    cv2.ellipse(img, (int(feet[0]), int(feet[1] + R(0.03))),
                (int(R(0.35)), int(R(0.07))), 0, 0, 360, (54, 58, 66), -1, cv2.LINE_AA)

    # Painter's algorithm on the subject's own depth: the far side of the body
    # is drawn first and dimmed, so a side-on figure still reads as a figure.
    order = sorted(("right", "left"), key=lambda s: -j3[f"{s}_shoulder"][2])

    def dim(colour, s):
        f = 0.72 if s == order[0] else 1.0
        return tuple(int(c * f) for c in colour)

    for s in order:
        _limb(img, uv[f"{s}_hip"], uv[f"{s}_knee"], R(0.085), R(0.062), dim(_TROUSER, s))
        _limb(img, uv[f"{s}_knee"], uv[f"{s}_ankle"], R(0.062), R(0.040), dim(_TROUSER, s))
        _limb(img, uv[f"{s}_heel"], uv[f"{s}_big_toe"], R(0.036), R(0.028), dim(_SHOE, s))
    # the far arm goes behind the torso, the near arm in front of it
    far, near = order
    _limb(img, uv[f"{far}_shoulder"], uv[f"{far}_elbow"], R(0.055), R(0.042), dim(_SHIRT, far))
    _limb(img, uv[f"{far}_elbow"], uv[f"{far}_wrist"], R(0.040), R(0.030), dim(_SKIN, far))
    torso = np.array([uv["left_shoulder"], uv["right_shoulder"],
                      uv["right_hip"], uv["left_hip"]])
    cv2.fillConvexPoly(img, np.round(torso).astype(np.int32), _SHIRT, cv2.LINE_AA)
    _limb(img, (uv["left_shoulder"] + uv["right_shoulder"]) / 2,
          (uv["left_hip"] + uv["right_hip"]) / 2, R(0.19), R(0.15), _SHIRT)
    # neck before the head, or it is drawn across the face
    hc = (uv["left_ear"] + uv["right_ear"]) / 2
    _limb(img, hc, (uv["left_shoulder"] + uv["right_shoulder"]) / 2, R(0.055), R(0.075), _SKIN)
    _limb(img, uv[f"{near}_shoulder"], uv[f"{near}_elbow"], R(0.055), R(0.042), _SHIRT)
    _limb(img, uv[f"{near}_elbow"], uv[f"{near}_wrist"], R(0.040), R(0.030), _SKIN)
    # head
    cv2.circle(img, tuple(np.round(hc).astype(int)), int(R(0.105)), _SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(img, tuple(np.round(hc - [0, R(0.035)]).astype(int)),
                (int(R(0.108)), int(R(0.072))), 0, 175, 365, (52, 46, 44), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(uv["nose"]).astype(int)), max(1, int(R(0.016))),
               tuple(int(c * 0.88) for c in _SKIN), -1, cv2.LINE_AA)
    for e in ("left_eye", "right_eye"):
        cv2.circle(img, tuple(np.round(uv[e]).astype(int)), max(1, int(R(0.013))),
                   (40, 38, 40), -1, cv2.LINE_AA)
    return img


def write_video(path: str, n_frames: int = T) -> str:
    """Render the clip to an mp4 (mp4v: every OpenCV build can write it)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not vw.isOpened():                              # pragma: no cover
        raise RuntimeError("could not open a writer for " + str(path))
    for t in range(n_frames):
        vw.write(render_frame(t))
    vw.release()
    return str(path)


# ------------------------------------------------------------- body mesh

_MESH_BONES = [
    ("left_hip", "left_knee", 0.085, 0.062), ("left_knee", "left_ankle", 0.062, 0.042),
    ("right_hip", "right_knee", 0.085, 0.062), ("right_knee", "right_ankle", 0.062, 0.042),
    ("left_shoulder", "left_elbow", 0.055, 0.042), ("left_elbow", "left_wrist", 0.042, 0.032),
    ("right_shoulder", "right_elbow", 0.055, 0.042), ("right_elbow", "right_wrist", 0.042, 0.032),
    ("left_heel", "left_big_toe", 0.036, 0.028), ("right_heel", "right_big_toe", 0.036, 0.028),
]
_TUBE_SIDES = 10


def _tube(a, b, r0, r1, sides=_TUBE_SIDES):
    """Vertices + faces of a tapered tube from a to b."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    w = d / n
    seed = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(w, seed)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
    verts = np.vstack([a + r0 * ring, b + r1 * ring])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, sides + i], [j, sides + j, sides + i]]
    return verts, np.asarray(faces, np.int32)


def _sphere(c, r, rings=8, sides=12):
    c = np.asarray(c, np.float64)
    vs, fs = [], []
    for i in range(rings + 1):
        lat = np.pi * i / rings
        for k in range(sides):
            lon = 2 * np.pi * k / sides
            vs.append(c + r * np.array([np.sin(lat) * np.cos(lon),
                                        np.cos(lat),
                                        np.sin(lat) * np.sin(lon)]))
    for i in range(rings):
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            fs += [[a, b, a + sides], [b, b + sides, a + sides]]
    return np.asarray(vs), np.asarray(fs, np.int32)


def mesh_3d(t: int) -> tuple[np.ndarray, np.ndarray]:
    """A crude but honest body mesh for frame `t`: a tapered tube per limb, a
    box-ish trunk and a sphere for the head, in the same camera-frame metres
    as `pose_3d`. Stands in for SAM 3D Body's own mesh so the storage,
    round-trip and drawing of a real one can be tested without it."""
    j, _ = pose_3d(t)
    verts, faces, base = [], [], 0
    pieces = [_tube(j[a], j[b], r0, r1) for a, b, r0, r1 in _MESH_BONES]
    mid_sh = (j["left_shoulder"] + j["right_shoulder"]) / 2
    mid_hip = (j["left_hip"] + j["right_hip"]) / 2
    pieces.append(_tube(mid_hip, mid_sh, 0.17, 0.15, sides=14))
    head_c = (j["left_ear"] + j["right_ear"]) / 2
    pieces.append(_tube(mid_sh, head_c, 0.055, 0.07))
    pieces.append(_sphere(head_c, 0.105))
    for v, f in pieces:
        if not len(v):
            continue
        verts.append(v)
        faces.append(f + base)
        base += len(v)
    return np.concatenate(verts), np.concatenate(faces).astype(np.int32)
