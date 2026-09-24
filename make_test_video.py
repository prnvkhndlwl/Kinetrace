"""Synthetic test video generator + export checker for the CoTracker3 GUI.

Generate a video of moving dots with known ground-truth trajectories:

    python make_test_video.py test600.mp4 --frames 600 --size 640x480 --dots 4

Ground truth is saved next to the video as <video>.gt.npz.

Validate an exported CSV/TSV against the ground truth:

    python make_test_video.py test600.mp4 --check tracks.csv [--tol 6.0]

The checker matches each exported point to the nearest ground-truth dot at its
first tracked frame, then reports mean/max pixel error over tracked frames.
Exits non-zero if any matched point's mean error exceeds the tolerance.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def dot_positions(t: np.ndarray, w: int, h: int, n_dots: int, rng: np.random.Generator) -> np.ndarray:
    """Ground-truth positions, shape (len(t), n_dots, 2). Smooth per-dot sinusoids."""
    margin = 60
    cx = rng.uniform(margin, w - margin, n_dots)
    cy = rng.uniform(margin, h - margin, n_dots)
    ax = rng.uniform(0.15, 0.35, n_dots) * (w / 2 - margin)
    ay = rng.uniform(0.15, 0.35, n_dots) * (h / 2 - margin)
    fx = rng.uniform(0.3, 1.2, n_dots)
    fy = rng.uniform(0.3, 1.2, n_dots)
    px = rng.uniform(0, 2 * np.pi, n_dots)
    py = rng.uniform(0, 2 * np.pi, n_dots)

    tt = t[:, None]  # (T, 1)
    x = cx + ax * np.sin(2 * np.pi * fx * tt / 600 + px)
    y = cy + ay * np.sin(2 * np.pi * fy * tt / 600 + py)
    x = np.clip(x, 10, w - 10)
    y = np.clip(y, 10, h - 10)
    return np.stack([x, y], axis=-1).astype(np.float64)


def generate(video_path: Path, frames: int, w: int, h: int, n_dots: int, seed: int, fps: float) -> None:
    rng = np.random.default_rng(seed)
    gt = dot_positions(np.arange(frames, dtype=np.float64), w, h, n_dots, rng)

    # Static textured background: plain backgrounds starve feature trackers.
    bg = rng.integers(30, 90, size=(h, w, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
    grid = np.zeros((h, w), np.uint8)
    grid[::32, :] = 40
    grid[:, ::32] = 40
    bg = cv2.add(bg, cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR))

    colors = [(60, 60, 230), (60, 230, 60), (230, 160, 40), (200, 60, 200),
              (40, 200, 230), (230, 230, 60), (150, 90, 40), (240, 240, 240)]

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        sys.exit(f"ERROR: could not open VideoWriter for {video_path}")
    radius = max(6, round(min(w, h) / 80))
    for f in range(frames):
        frame = bg.copy()
        for d in range(n_dots):
            x, y = gt[f, d]
            c = colors[d % len(colors)]
            cv2.circle(frame, (round(x), round(y)), radius, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (round(x), round(y)), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    gt_path = Path(str(video_path) + ".gt.npz")
    np.savez_compressed(gt_path, gt=gt.astype(np.float32), fps=fps, size=(w, h))
    print(f"wrote {video_path} ({frames} frames, {w}x{h}, {n_dots} dots) and {gt_path}")


def load_export(path: Path) -> dict[str, np.ndarray]:
    """Parse a wide CSV or sparse TSV export into {point_name: (T, 2) array with NaNs}."""
    text = path.read_text(encoding="utf-8").strip().splitlines()
    if not text:
        sys.exit("ERROR: export file is empty")
    header = text[0]
    if "\t" in header:  # sparse TSV: frame, point, x, y, visible
        rows = list(csv.reader(text, delimiter="\t"))
        body = rows[1:]
        max_frame = max(int(r[0]) for r in body)
        points: dict[str, np.ndarray] = {}
        for r in body:
            name = r[1]
            if name not in points:
                points[name] = np.full((max_frame + 1, 2), np.nan, np.float64)
            points[name][int(r[0])] = (float(r[2]), float(r[3]))
        return points
    # wide CSV: frame, {name}_x, {name}_y, {name}_visible, ...
    rows = list(csv.reader(text))
    cols = rows[0]
    names = [c[:-2] for c in cols if c.endswith("_x")]
    body = rows[1:]
    t_count = len(body)
    points = {n: np.full((t_count, 2), np.nan, np.float64) for n in names}
    for r in body:
        f = int(r[0])
        for i, n in enumerate(names):
            xs, ys = r[1 + 3 * i], r[2 + 3 * i]
            if xs != "" and ys != "":
                points[n][f] = (float(xs), float(ys))
    return points


def check(video_path: Path, export_path: Path, tol: float) -> None:
    gt_path = Path(str(video_path) + ".gt.npz")
    if not gt_path.exists():
        sys.exit(f"ERROR: ground truth {gt_path} not found (generate the video first)")
    gt = np.load(gt_path)["gt"]  # (T, D, 2)
    points = load_export(export_path)
    if not points:
        sys.exit("ERROR: no points found in export")

    failed = False
    for name, track in points.items():
        valid = ~np.isnan(track[:, 0])
        if not valid.any():
            print(f"{name}: no tracked frames — SKIP")
            continue
        first = int(np.argmax(valid))
        T = min(len(track), len(gt))
        # match to nearest gt dot at the first tracked frame
        d0 = np.linalg.norm(gt[first] - track[first], axis=-1)
        dot = int(np.argmin(d0))
        v = valid[:T]
        err = np.linalg.norm(track[:T][v] - gt[:T, dot][v], axis=-1)
        status = "OK" if err.mean() <= tol else "FAIL"
        failed |= status == "FAIL"
        print(f"{name}: matched gt dot {dot}, {v.sum()} frames, "
              f"mean err {err.mean():.2f} px, max {err.max():.2f} px  [{status}]")
    if failed:
        sys.exit(1)
    print(f"all points within tolerance ({tol} px mean)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--size", default="640x480", help="WxH, e.g. 3840x2160")
    ap.add_argument("--dots", type=int, default=4)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=Path, default=None, metavar="EXPORT",
                    help="validate an exported CSV/TSV instead of generating")
    ap.add_argument("--tol", type=float, default=6.0, help="mean-error tolerance in px for --check")
    args = ap.parse_args()

    if args.check is not None:
        check(args.video, args.check, args.tol)
    else:
        w, h = (int(v) for v in args.size.lower().split("x"))
        generate(args.video, args.frames, w, h, args.dots, args.seed, args.fps)


if __name__ == "__main__":
    main()
