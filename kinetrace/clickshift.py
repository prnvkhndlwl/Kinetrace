"""Do a camera's imported clicks sit on the right pictures? (I310)

A DLTdv project numbers frames the way the MATLAB that opened it decoded the video, and MATLAB 2024b numbered
GoPro files differently from earlier versions and from ffmpeg (which Kinetrace uses: frame N = the N-th
picture). A project can therefore put every click of a camera whole frames away from the picture it was made
on. This module measures that per camera, from the pictures alone:

* up to 40 clicks from the camera's busiest stretch (300 of its frames);
* for each, the background at that spot: the per-pixel median of 11 frames spread around the stretch (a click
  is on the animal, which moves; the background does not);
* for every whole-frame shift s in a window of +-max(60, one second of the camera's frames), how much the
  patch under the click at frame F + s differs from that background. Median over the clicks = the score of s.

At the right shift the patches show the animal (a high score); at a wrong one, background (a low score). One
sequential decode of the stretch; only small patches are kept, never whole frames. A worker runs it (the GUI
thread never opens a VideoCapture).

Verdicts: "fits" (the best shift is 0, or DLTdv's frames score at least 0.8 of the best), "better" (another
shift scores at least twice DLTdv's AND stands out 3x from the typical shift: offered to the user, never
applied by itself), "cannot" (fewer than 5 clicks, or no shift stands out 2x: the subject barely moves),
"unclear" (anything else: DLTdv's frames are kept).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from kinetrace.project import half_up
from kinetrace.video_source import FrameReader

MAX_CLICKS = 40
STRETCH = 300               # frames of the camera the sampled clicks come from
N_BACKGROUND = 11
BG_MARGIN = 300             # background frames are spread over the stretch +- this
MIN_CLICKS = 5
FITS_FRACTION = 0.8         # DLTdv's frames score >= this x the best: they fit
BETTER_GAIN = 2.0           # another shift scores >= this x DLTdv's frames ...
BETTER_CONTRAST = 3.0       # ... and >= this x the median shift: offered
CANNOT_CONTRAST = 2.0       # no shift stands out this much: the check cannot tell


@dataclass
class ClickFit:
    verdict: str                          # "fits" | "better" | "cannot" | "unclear"
    best: int = 0                         # the best whole-frame shift (0 = DLTdv's frames)
    offer: int | None = None              # the shift offered to the user (only for "better")
    score0: float = float("nan")          # the score at DLTdv's frames
    score_best: float = float("nan")
    contrast: float = float("nan")        # best / median over the window
    n_clicks: int = 0                     # clicks used
    window: int = 0
    shifts: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    crops: tuple | None = None            # (at DLTdv's frame, at the best) BGR, for the hand check
    crop_xy: tuple | None = None          # the click inside the crops
    crop_frames: tuple | None = None      # the two frame numbers shown
    reason: str = ""

    @property
    def sentence(self) -> str:
        """The verdict in plain words."""
        if self.verdict == "cannot":
            return f"Cannot tell: {self.reason}. DLTdv's frames are kept; check a few clicks by eye."
        nums = (f"score {self.score_best:.1f} at the best fit, {self.score0:.1f} at DLTdv's frames, "
                f"{self.score_best / self.contrast:.1f} typical")
        if self.verdict == "fits":
            return f"Fits: the clicks sit on what moves at DLTdv's own frame numbers ({nums})."
        way = "EARLIER" if self.best < 0 else "LATER"
        if self.verdict == "better":
            return (f"Another frame fits clearly better: the clicks sit on what moves {abs(self.best)} frame(s) "
                    f"{way} than DLTdv's numbers say ({nums}). A newer MATLAB numbers some videos' frames "
                    f"differently; see the two pictures.")
        return (f"Unclear: the best fit is {abs(self.best)} frame(s) {way}, but not by enough to be sure "
                f"({nums}). DLTdv's frames are kept; check a few clicks by eye.")


def _busiest(frames: np.ndarray, span: int) -> np.ndarray:
    """Indices (into `frames`) of the clicks in the `span`-frame stretch holding the most clicks."""
    order = np.argsort(frames, kind="stable")
    f = frames[order]
    best_i, best_n, j = 0, 0, 0
    for i in range(len(f)):
        while f[i] - f[j] >= span:
            j += 1
        if i - j + 1 > best_n:
            best_n, best_i = i - j + 1, j
    return order[best_i:best_i + best_n]


def _patch(grey: np.ndarray, x: int, y: int, h: int) -> np.ndarray | None:
    H, W = grey.shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return None
    return grey[max(0, y - h):y + h + 1, max(0, x - h):x + h + 1]


def check_camera(path: str, xy: np.ndarray, frames: np.ndarray, fps: float, n_frames: int | None = None,
                 should_cancel=None, progress=None) -> ClickFit | None:
    """Score whole-frame shifts of camera `path`'s clicks. `xy` (n, 2) Kinetrace pixels (0-based), `frames`
    (n,) the frame each click is on by the project, `n_frames` the video's length (what decodes; None = the
    file's header). None = cancelled."""
    should_cancel = should_cancel or (lambda: False)
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    frames = np.asarray(frames, np.int64).ravel()
    window = int(max(60, half_up(fps))) if np.isfinite(fps) and fps > 0 else 60
    if len(frames) < MIN_CLICKS:
        return ClickFit("cannot", window=window, n_clicks=len(frames),
                        reason=f"only {len(frames)} click(s) in this camera")
    sel = _busiest(frames, STRETCH)
    sel = sel[np.argsort(frames[sel], kind="stable")]
    if len(sel) > MAX_CLICKS:
        sel = sel[np.linspace(0, len(sel) - 1, MAX_CLICKS).round().astype(int)]
    f = frames[sel]
    px = np.atleast_1d(half_up(xy[sel, 0])).astype(int)
    py = np.atleast_1d(half_up(xy[sel, 1])).astype(int)
    rd = FrameReader(path, max(0, int(f.min()) - window))         # VideoOpenError: the caller says it in words
    try:
        W_ = int(rd.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        H_ = int(rd.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        h = int(max(4, half_up(10 * max(W_, H_) / 1920)))
        n = len(f)
        end = int(n_frames) if n_frames else int(rd.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(f.max()) + window + 1
        # the background at each click: the median of N_BACKGROUND patches spread around the stretch
        lo_b, hi_b = max(0, int(f.min()) - BG_MARGIN), min(end - 1, int(f.max()) + BG_MARGIN)
        bg: list[list] = [[] for _ in range(n)]
        for g in np.unique(np.linspace(lo_b, hi_b, N_BACKGROUND).round().astype(int)):
            if should_cancel():
                return None
            rd.reset(int(g))
            b = rd.read(int(g))
            if b is None:
                continue
            grey = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for j in range(n):
                p = _patch(grey, px[j], py[j], h)
                if p is not None:
                    bg[j].append(p)
        bgm = [np.median(np.stack(b), 0) if len(b) >= 5 else None for b in bg]
        if sum(b is not None for b in bgm) < MIN_CLICKS:
            return ClickFit("cannot", window=window, n_clicks=n,
                            reason="too few background pictures could be read around the clicks")
        shifts = np.arange(-window, window + 1)
        sc = np.full((len(shifts), n), np.nan, np.float32)
        first, last = max(0, int(f.min()) - window), int(f.max()) + window
        rd.reset(first)
        for g in range(first, last + 1):
            if should_cancel():
                return None
            if progress is not None:
                progress(g - first, last - first + 1)
            b = rd.read(g)
            if b is None:
                break                                      # the video ends: later shifts stay unknown
            grey = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for j in np.flatnonzero(np.abs(g - f) <= window):
                if bgm[j] is None:
                    continue
                p = _patch(grey, px[j], py[j], h)
                if p is not None and p.shape == bgm[j].shape:
                    sc[g - f[j] + window, j] = float(np.abs(p - bgm[j]).mean())
    finally:
        rd.release()
    valid = np.isfinite(sc).sum(axis=1) >= min(MIN_CLICKS, n)
    score = np.full(len(shifts), np.nan)
    score[valid] = np.nanmedian(sc[valid], axis=1)
    if not np.isfinite(score[window]):
        return ClickFit("cannot", window=window, n_clicks=n, shifts=shifts, scores=score,
                        reason="the frames at the clicks could not be read")
    top = np.nanmax(score)
    ties = np.flatnonzero(score == top)
    best = int(shifts[ties[np.argmin(np.abs(shifts[ties]))]])          # a tie goes to the smallest shift
    typical = float(np.nanmedian(score))
    contrast = top / typical if typical > 0 else np.inf
    s0 = float(score[window])
    fit = ClickFit("unclear", best=best, score0=s0, score_best=float(top), contrast=float(contrast), n_clicks=n,
                   window=window, shifts=shifts, scores=score)
    if contrast < CANNOT_CONTRAST:
        fit.verdict, fit.reason = "cannot", "no frame shift stands out (the clicked points barely move)"
    elif best == 0 or s0 >= FITS_FRACTION * top:
        fit.verdict = "fits"
    elif top >= BETTER_GAIN * s0 and contrast >= BETTER_CONTRAST:
        fit.verdict, fit.offer = "better", best
    _crops(fit, path, f, px, py, sc, h)
    return fit


def _crops(fit: ClickFit, path: str, f, px, py, sc, h: int) -> None:
    """The hand check: one click (the one whose score at the best shift is the median) at DLTdv's frame and at
    the best one, cut out around it."""
    col = sc[fit.best + fit.window]
    ok = np.flatnonzero(np.isfinite(col))
    if not len(ok):
        return
    j = int(ok[np.argsort(col[ok])[len(ok) // 2]])
    r = 6 * h
    out = []
    try:
        rd = FrameReader(path, int(f[j]))
    except Exception:  # noqa: BLE001 - the crops are a help, not the verdict
        return
    try:
        for g in sorted({int(f[j]), int(f[j]) + fit.best}):
            rd.reset(g)
            b = rd.read(g)
            if b is None:
                return
            H, W = b.shape[:2]
            x0, y0 = max(0, px[j] - r), max(0, py[j] - r)
            out.append((g, b[y0:min(H, py[j] + r + 1), x0:min(W, px[j] + r + 1)].copy(), (px[j] - x0, py[j] - y0)))
    finally:
        rd.release()
    by = {g: (c, xy) for g, c, xy in out}
    a, b = by[int(f[j])], by[int(f[j]) + fit.best]
    fit.crops, fit.crop_xy, fit.crop_frames = (a[0], b[0]), (a[1], b[1]), (int(f[j]), int(f[j]) + fit.best)
