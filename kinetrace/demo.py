"""The demo video the interface tour opens on an empty window (G183, owner 2026-10-12: a tour of a blank
window teaches nothing -- the timeline, LAYERS and the video only explain themselves with something in them).

Kinetrace MAKES it, the first time it is needed: one synthetic lizard-like animal (an ellipse body, a round
head with a pale eye spot, an undulating tail, four swinging legs) walking an ellipse on a textured
background. Deterministic, so `truth(t)` gives the exact position of each part and the silhouette on every
frame; the tour fills the first `N_FILLED` frames from it (no model run, nothing downloaded) and leaves the
rest for Track. Nothing is committed to git (the repo ignores *.mp4) and nothing is written outside the
Kinetrace folder: `demo/` inside it, the per-user data folder when it cannot be written, the temp folder
for the offscreen test suites. No Qt here; cv2 + numpy only.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
N_FRAMES = 150
FPS = 25.0
N_FILLED = 100              # frames 0..N_FILLED-1 come tracked and outlined; the rest is Track's to do
ANIMAL = "lizard"
PARTS = ("snout", "eye", "tail tip", "left fore foot", "right fore foot")
FILE_NAME = "kinetrace-demo.mp4"
_INSTALL = Path(__file__).resolve().parent.parent / "demo"


def demo_folder() -> Path:
    """Where the demo is kept: `demo/` in the Kinetrace folder (deleting the folder removes it; updates keep
    it, `update.PROTECTED`); the per-user data folder when that cannot be written; a temporary folder for the
    offscreen test suites (they must not write into the user's Kinetrace folder)."""
    from kinetrace import recovery
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        import tempfile
        return Path(tempfile.gettempdir()) / "kinetrace-test-demo"
    if recovery._writable(_INSTALL.parent):        # the Kinetrace folder: asking never creates demo/
        return _INSTALL
    return recovery._user_dir().parent / "demo"


def video_path() -> Path:
    return demo_folder() / FILE_NAME


def _pose(t: float):
    """Body centre and heading at time t: one slow ellipse around the middle of the picture."""
    def at(s):
        return np.array([W * 0.5 + 0.2 * W * np.cos(0.025 * s), H * 0.5 + 0.2 * H * np.sin(0.025 * s)])
    c, ahead = at(t), at(t + 1)
    d = ahead - c
    return c, float(np.arctan2(d[1], d[0]))


def _background() -> np.ndarray:
    rng = np.random.default_rng(7)
    bg = rng.integers(60, 140, (H // 8, W // 8, 3), np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    bg[..., 1] = np.clip(bg[..., 1].astype(int) + 40, 0, 255)    # greenish, textured: something to track
    return bg


def _draw(t: int, canvas: np.ndarray | None):
    """Draw frame t's animal into `canvas` (if given) and return (parts {name: (x, y)}, mask bool HxW).
    Positions are in Kinetrace's pixel convention (OpenCV pixel centres from 0), the same as cv2's
    drawing coordinates."""
    c, h = _pose(t)
    u = np.array([np.cos(h), np.sin(h)])          # forward
    n = np.array([-np.sin(h), np.cos(h)])         # the animal's right (y points down)
    col = (38, 44, 58)
    mask = np.zeros((H, W), np.uint8)
    targets = [(mask, 255)] + ([(canvas, col)] if canvas is not None else [])

    def P(v):
        return (int(round(v[0])), int(round(v[1])))

    for img, colour in targets:
        cv2.ellipse(img, P(c), (60, 26), float(np.degrees(h)), 0, 360, colour, -1, cv2.LINE_AA)
        cv2.circle(img, P(c + 62 * u), 18, colour, -1, cv2.LINE_AA)
    eye = c + 62 * u - 8 * n
    if canvas is not None:
        cv2.circle(canvas, P(eye), 4, (230, 230, 240), -1, cv2.LINE_AA)     # a textured feature
    tail = [c - (50 + 6 * i) * u + (3.0 + 0.9 * i) * np.sin(0.25 * t + 0.35 * i) * n for i in range(25)]
    for i in range(24):
        th = max(3, int(round(9 - 7 * i / 24)))
        for img, colour in targets:
            cv2.line(img, P(tail[i]), P(tail[i + 1]), colour, th, cv2.LINE_AA)
    feet = {}
    for side, side_name in ((-1, "left"), (1, "right")):
        for fore, fore_name in ((1, "fore"), (-1, "hind")):
            attach = c + fore * 25 * u + side * 20 * n
            swing = 16 * np.sin(0.3 * t + (0 if fore == 1 else 1.6) + (0 if side == -1 else 0.8))
            foot = attach + side * 38 * n + swing * u
            for img, colour in targets:
                cv2.line(img, P(attach), P(foot), colour, 7, cv2.LINE_AA)
                cv2.circle(img, P(foot), 5, colour, -1, cv2.LINE_AA)
            feet[f"{side_name} {fore_name} foot"] = foot
    parts = {"snout": c + 66 * u, "eye": eye, "tail tip": tail[-1],
             "left fore foot": feet["left fore foot"], "right fore foot": feet["right fore foot"]}
    return {k: (float(v[0]), float(v[1])) for k, v in parts.items()}, mask > 127


def truth(t: int):
    """(parts {name: (x, y)}, silhouette bool HxW) of frame t."""
    return _draw(int(t), None)


def _frame_count(p: Path) -> int:
    cap = cv2.VideoCapture(str(p))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()


def ensure_video() -> Path:
    """The demo video, made now if it is missing or incomplete (about a second). Written to a temporary name
    and renamed, so an interrupted write never leaves a short demo behind. Runs on a worker thread (it opens
    a VideoCapture to count the frames). Raises OSError with a sentence when it cannot be written."""
    p = video_path()
    if p.exists() and _frame_count(p) == N_FRAMES:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}-{os.getpid()}.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not vw.isOpened():
        raise OSError(f"Kinetrace could not write its demo video in {p.parent}.")
    try:
        bg = _background()
        for t in range(N_FRAMES):
            frame = bg.copy()
            _draw(t, frame)
            vw.write(frame)
    finally:
        vw.release()
    if _frame_count(tmp) != N_FRAMES:
        tmp.unlink(missing_ok=True)
        raise OSError(f"Kinetrace's demo video in {p.parent} came out incomplete.")
    os.replace(tmp, p)
    return p
