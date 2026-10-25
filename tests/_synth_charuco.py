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


def write_video(path, frames_bgr, fps=30.0):
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()
