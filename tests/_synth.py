"""Synthetic two-animal video with exact ground truth (silhouette mask, head,
tail tip, feet, body centre) shared by the segmentation and animal-layer
tests. Deterministic; renders in a second."""
from __future__ import annotations

import os

import cv2
import numpy as np

W, H, T = 1280, 720, 160


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
    """Ellipse body, round head with a pale eye spot, tapering undulating tail,
    four swinging legs. Returns ground truth: head, eye, tail tip, feet (4),
    body centre."""
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
    eye = c + 62 * u + 8 * n
    cv2.circle(canvas, P(eye), 4, (230, 230, 240), -1, cv2.LINE_AA)  # textured feature
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
    return {"head": head, "eye": eye, "tip": tail_pts[-1], "feet": np.array(feet), "centre": c,
            "heading": h}


def gt_frame(t: int, hide0_after: int | None = None):
    """(bgr frame, per-animal dict of gt + bool mask). With `hide0_after`,
    animal 0 is not drawn from that frame on (it "leaves the scene")."""
    rng = np.random.default_rng(7)
    bg = rng.integers(60, 140, (H // 8, W // 8, 3), np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    bg[..., 1] = np.clip(bg[..., 1].astype(int) + 40, 0, 255)  # greenish, textured
    canvas = bg.copy()
    out = {}
    for k in (0, 1):
        m = np.zeros((H, W), np.uint8)
        if k == 0 and hide0_after is not None and t >= hide0_after:
            c, h = pose(t, k)
            out[k] = {"head": None, "eye": None, "tip": None, "feet": np.zeros((0, 2)),
                      "centre": c, "heading": h, "mask": m > 127}
            continue
        gt = draw_animal(canvas, m, t, k)
        gt["mask"] = m > 127
        out[k] = gt
    return canvas, out


def build_video(path: str, n_frames: int = T, hide0_after: int | None = None) -> str:
    if os.path.exists(path):
        cap = cv2.VideoCapture(path)
        ok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n_frames
        cap.release()
        if ok:
            return path
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for t in range(n_frames):
        frame, _ = gt_frame(t, hide0_after)
        vw.write(frame)
    vw.release()
    return path


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)
