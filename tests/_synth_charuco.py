"""Synthetic ChArUco pictures through a known camera, for verify_charuco*.py (I321, I322).

`render(spec, K, dist, fisheye, rvec, tvec, size)` draws the printed board seen by a camera
(pinhole + OpenCV radial / tangential, or the fisheye model) by ray casting every output
pixel onto the board plane - so the lens bend, perspective and a board running out of the
picture are all exact - at 2x and area-averaged down, plus noise and a slight blur.
`truth(spec, K, dist, fisheye, rvec, tvec)` = the (N, 2) projected inner corners (0-based
pixel centres), the numbers a detection is compared with. Board coordinates are
`charuco.object_points`' (corner 0 at the origin). Deterministic.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from kinetrace import charuco

PX_PER_SQUARE = 120              # the board texture's resolution


def texture(spec):
    """(board image, margin px, px per square) of the printed board."""
    m = PX_PER_SQUARE
    return charuco.board_image(spec, PX_PER_SQUARE, m), m, PX_PER_SQUARE


def truth(spec, K, dist, fisheye, rvec, tvec) -> np.ndarray:
    obj = charuco.object_points(spec, np.arange(spec.n_corners))
    if fisheye:
        p, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), np.asarray(rvec, float), np.asarray(tvec, float),
                                         K, np.asarray(dist, float).reshape(-1, 1)[:4])
    else:
        p, _ = cv2.projectPoints(obj, np.asarray(rvec, float), np.asarray(tvec, float), K, np.asarray(dist, float))
    return p.reshape(-1, 2)


def render(spec, K, dist, fisheye, rvec, tvec, size, noise=2.0, blur=0.6, seed=0, background=140) -> np.ndarray:
    """A BGR picture of the board (see the module docstring)."""
    W, H = size
    tex, margin, s = texture(spec)
    ss = 2
    ys, xs = np.mgrid[0:H * ss, 0:W * ss].astype(np.float64)
    # supersample centres in output pixel coordinates (pixel centre at integer i)
    px = np.stack([(xs + 0.5) / ss - 0.5, (ys + 0.5) / ss - 0.5], -1).reshape(-1, 1, 2)
    if fisheye:
        und = cv2.fisheye.undistortPoints(px, K, np.asarray(dist, float).reshape(-1, 1)[:4]).reshape(-1, 2)
    else:
        und = cv2.undistortPoints(px, K, np.asarray(dist, float)).reshape(-1, 2)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    Hm = np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, float).ravel()])
    Hi = np.linalg.inv(Hm)
    ray = np.column_stack([und, np.ones(len(und))])
    q = ray @ Hi.T
    front = q[:, 2] > 0
    X = q[:, 0] / q[:, 2]
    Y = q[:, 1] / q[:, 2]
    # board metres (corner 0 at the origin) -> texture pixels: corner 0 is one square in from the
    # board's outer top-left corner, which sits at the margin
    u = margin + (X / spec.square_m + 1.0) * s - 0.5
    v = margin + (Y / spec.square_m + 1.0) * s - 0.5
    u[~front] = -1e6
    v[~front] = -1e6
    mapx = u.reshape(H * ss, W * ss).astype(np.float32)
    mapy = v.reshape(H * ss, W * ss).astype(np.float32)
    big = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    r = np.random.RandomState(seed)
    img = np.clip(img.astype(np.float64) + r.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def random_pose(K, size, r, spread=0.8, tilt_deg=40.0, z=(0.45, 1.0), board_centre=(0.09, 0.06)):
    """A pose putting the board centre somewhere over the picture, tilted."""
    W, H = size
    zz = r.uniform(*z)
    u = r.uniform(0.5 - 0.5 * spread, 0.5 + 0.5 * spread) * W
    v = r.uniform(0.5 - 0.5 * spread, 0.5 + 0.5 * spread) * H
    rvec = np.deg2rad(tilt_deg) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.5])
    R, _ = cv2.Rodrigues(rvec)
    c = np.array([(u - K[0, 2]) / K[0, 0] * zz, (v - K[1, 2]) / K[1, 1] * zz, zz])
    tvec = c - R @ np.array([board_centre[0], board_centre[1], 0.0])
    return rvec, tvec


RIG_SPEC = charuco.BoardSpec(9, 7, 0.06, 0.045)        # 540 x 420 mm of squares: big enough for a rig
RIG_SIZE = (640, 480)
RIG_OFFSETS = [0.0, 0.5, -0.25]                        # fractional: the scan interpolates between frames
RIG_FOCAL = [600.0, 640.0, 580.0]
RIG_DIST = [np.array([-0.22, 0.06, 0, 0, 0]), np.array([-0.18, 0.04, 0, 0, 0]), np.array([-0.25, 0.08, 0, 0, 0])]


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


def rig_cameras():
    """Three cameras 1.4 m from the origin, 30 degrees apart, looking at it: [(K, dist, R, t, centre)]."""
    W, H = RIG_SIZE
    out = []
    for k, az in enumerate((-30.0, 0.0, 30.0)):
        a = np.deg2rad(az - 90.0)
        pos = np.array([1.4 * np.cos(a), 1.4 * np.sin(a), 0.2 + 0.15 * (k % 2)])
        R, t = look_at(pos, np.zeros(3))
        K = np.array([[RIG_FOCAL[k], 0, (W - 1) / 2 + 4 * (k - 1)], [0, RIG_FOCAL[k], (H - 1) / 2 - 3 * (k - 1)],
                      [0, 0, 1]])
        out.append((K, RIG_DIST[k], R, t, pos))
    return out


def board_at(t):
    """The board's pose at reference instant t (frames): (R board -> world, corner 0 in world). Its
    centre sweeps a 1.1 x 0.3 x 0.8 m box (into every picture's corners, often half out of a
    picture); its printed face looks towards the cameras (-y), tilted up to ~25 degrees."""
    s = float(t) / 30.0
    c = np.array([0.55 * np.sin(1.7 * s), 0.15 * np.sin(0.7 * s + 1.0), 0.4 * np.sin(2.3 * s + 0.4)])
    tilt_x, tilt_z = 0.45 * np.sin(1.3 * s), 0.45 * np.sin(0.8 * s + 0.7)
    n = np.array([np.sin(tilt_z), -np.cos(tilt_z) * np.cos(tilt_x), np.sin(tilt_x)])   # towards the cameras
    zb = -n                                                                           # board z: away from them
    xb = np.cross(np.array([0, 0, 1.0]), zb)
    xb /= np.linalg.norm(xb)
    yb = np.cross(zb, xb)
    Rb = np.column_stack([xb, yb, zb])
    cols, rows = RIG_SPEC.pattern
    centre_b = np.array([(cols - 1) / 2.0 * RIG_SPEC.square_m, (rows - 1) / 2.0 * RIG_SPEC.square_m, 0.0])
    return Rb, c - Rb @ centre_b


def rig_videos(folder, n_frames=90):
    """One board video per camera (written once); camera c's frame f shows reference instant
    f - RIG_OFFSETS[c]. -> (paths, cameras)."""
    cams = rig_cameras()
    paths = []
    for k, (K, D, R, t, _pos) in enumerate(cams):
        path = os.path.join(folder, f"charuco_rig_cam{k + 1}.mp4")
        paths.append(path)
        if os.path.exists(path):
            continue
        pics = []
        for f in range(n_frames):
            Rb, ob = board_at(f - RIG_OFFSETS[k])
            rvec, _ = cv2.Rodrigues(R @ Rb)
            tvec = R @ ob + t
            pics.append(render(RIG_SPEC, K, D, False, rvec.ravel(), tvec, RIG_SIZE, seed=5000 + 100 * k + f))
        write_video(path, pics)
    return paths, cams


def write_video(path, frames_bgr, fps=30.0):
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()
