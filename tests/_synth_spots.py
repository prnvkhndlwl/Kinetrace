"""Synthetic clips for the Moving spot point model (verify_spots.py), with
exact ground truth. Every scene is the hard case it is named after:

  sea    a flapping white spot (5 px, 13-14 px/frame, a faint streak behind
         it) over moving water whose fine ripples and white glints are as
         strong as the spot or stronger -- glints never come within 9 px of
         the spot, so a tight search follows it and a wide one grabs glints
  sky    a dark bird (7 px) on a curved path against a bright sky with drifting
         clouds and a few far-away dark specks
  river  a dark "bat" (11 px, wingbeat) turning over water: a STILL texture
         full of rock spots as dark as the bat, plus per-pixel flicker -- a
         dark-spot search finds rocks, the unusual-change cue finds the bat

Options: `vanish_at` (the spot is gone from that frame on), `fade` = (first,
last) frame over which the sea spot fades to nothing (as on real footage: it
fades into the ripples), `twin` (a second identical spot crosses the first), `zigzag` (the sky bird dodges
up and down), `bird_sigma` (its size),
`exit_right` (the spot flies out of the picture). Frames are RGB uint8.
"""
from __future__ import annotations

import cv2
import numpy as np


def _blob(img, x, y, sigma, amp, rgb=(1.0, 1.0, 1.0)):
    """Add a Gaussian blob (amp > 0 brighter, < 0 darker) in place."""
    h, w = img.shape[:2]
    r = int(4 * sigma + 2)
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 2)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 2)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)) * amp
    for c in range(3):
        img[y0:y1, x0:x1, c] += g * rgb[c]


def make_scene(kind: str, n: int = 60, w: int = 960, h: int = 540, seed: int = 0,
               vanish_at: int | None = None, twin: bool = False, exit_right: bool = False, fade=None,
               zigzag: bool = False, bird_sigma: float = 2.2):
    """-> (frames [RGB uint8], gt (n, 2) float64 with NaN where the spot is not in the picture)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames, gt = [], np.full((n, 2), np.nan)
    if kind == "sea":
        p0, v = np.array([260.0, 400.0]), np.array([11.0, -8.0])
        if exit_right:
            p0, v = np.array([w - 300.0, 300.0]), np.array([13.0, -2.0])
    elif kind == "sky":
        c0, rad = np.array([w / 2, h / 2 + 60]), 230.0
    elif kind == "river":
        base = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
        base = 140 + 30 * base / base.std()        # never black: a dark bat must have room to show
        heading0 = 0.3
        pos = np.array([150.0, 270.0])
        path = [pos.copy()]
        for t in range(1, n):
            hd = heading0 + 0.6 * np.sin(t / 9.0)
            path.append(path[-1] + 10.0 * np.array([np.cos(hd), np.sin(hd)]))
        path = np.array(path)
        rocks = np.zeros((h, w, 3), np.float32)
        for x, y in rng.uniform([0, 0], [w, h], (700, 2)):    # rocks as dark as the bat, beside its path
            if np.hypot(path[:, 0] - x, path[:, 1] - y).min() < 14.0:
                continue                                       # (a dark bat over a black rock cannot be seen)
            _blob(rocks, x, y, 4.0, -rng.uniform(63.0, 90.0))
        still = np.stack([base * 0.9, base * 0.85, base * 0.75], -1) + rocks
    else:
        raise ValueError(kind)
    for t in range(n):
        if kind == "sea":
            img = np.empty((h, w, 3), np.float32)
            wave = (25 * np.sin(0.09 * xx + 0.05 * yy - 0.6 * t) + 15 * np.sin(0.05 * xx - 0.11 * yy - 0.45 * t)
                    + 10 * np.sin(0.55 * xx + 0.21 * yy - 1.7 * t) * np.sin(0.13 * xx - 0.47 * yy + 1.1 * t))
            for c, (b, k) in enumerate(((30, 0.6), (70, 0.8), (120, 1.0))):
                img[..., c] = b + k * wave
            p = p0 + v * t + np.array([0.0, 0.04 * t * t])
            if exit_right:
                p = p0 + v * t
            for _ in range(70):                                # glints: white, 1 frame, never on the spot
                g = rng.uniform([0, 0], [w, h]) if rng.random() < 0.6 else p + rng.normal(0, 14, 2)
                if np.hypot(*(g - p)) < 9.0:
                    continue
                _blob(img, g[0], g[1], 1.0, rng.uniform(150, 230))
            amp = 140 * (0.75 + 0.25 * np.sin(2 * np.pi * t / 6))
            if fade is not None:
                amp *= float(np.clip((fade[1] - t) / float(fade[1] - fade[0]), 0.0, 1.0))
            if vanish_at is None or t < vanish_at:
                for k_, a_ in ((0.3, 0.25), (0.6, 0.18), (0.9, 0.1)):
                    q = p - k_ * v
                    _blob(img, q[0], q[1], 1.2, a_ * amp)
                _blob(img, p[0], p[1], 1.6, amp)
                gt[t] = p
            if twin:
                q = p + np.array([60.0 - 4.0 * t, 0.0])          # crosses the spot at t = 15
                _blob(img, q[0], q[1], 1.6, amp)
            img += rng.normal(0, 3.0, img.shape).astype(np.float32)
        elif kind == "sky":
            img = np.empty((h, w, 3), np.float32)
            grad = yy / h
            cloud = 20 * np.sin(0.012 * xx + 0.4 * t * 0.05) * np.sin(0.017 * yy - 0.03 * t)
            for c, (top, bot) in enumerate(((150, 200), (180, 220), (220, 250))):
                img[..., c] = top + (bot - top) * grad + cloud
            for sx, sy in ((80, 60), (870, 90), (120, 470), (820, 480), (480, 40)):
                _blob(img, sx, sy, 1.5, -90.0)
            ang = -2.6 + 0.045 * t
            p = c0 + rad * np.array([np.cos(ang), np.sin(ang)])
            if zigzag:                  # dodging: up and down 45 px every ~14 frames at 11 px / frame across
                p = np.array([150.0 + 11.0 * t, 270.0 + 45.0 * np.sin(t / 2.2)])
            if vanish_at is None or t < vanish_at:
                _blob(img, p[0], p[1], bird_sigma, -120.0)
                gt[t] = p
            img += rng.normal(0, 2.0, img.shape).astype(np.float32)
        else:   # river
            img = still.copy()
            ripple = 10 * np.sin(0.1 * xx - 0.07 * yy - 0.5 * t)
            img += ripple[..., None]
            p = path[t].copy()
            if vanish_at is None or t < vanish_at:
                beat = 0.7 + 0.3 * abs(np.sin(np.pi * t / 5))
                _blob(img, p[0], p[1], 4.0, -90.0 * beat)          # 63-90 grey levels deep
                gt[t] = p
            img += rng.normal(0, 6.0, img.shape).astype(np.float32)
        if not (0 <= gt[t, 0] < w and 0 <= gt[t, 1] < h):
            gt[t] = np.nan
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames, gt


def write_video(path: str, frames, fps: float = 60.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cannot write {path}")
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
