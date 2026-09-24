"""Whole-frame camera sync from the pictures themselves.

Hand-started cameras (a GoPro rig without a genlock or a sync cable) begin
recording seconds apart, and the offsets that align them must be found
before any 3D. The sub-frame estimator in `calib.py` refines an offset by a
frame or so from tracked landmarks; it cannot find one that is off by
hundreds of frames, and it needs tracks. This module needs neither: it
compares how much the PICTURE changes from frame to frame in every camera
and slides the curves against each other until they line up.

Why that works: every camera sees the same event at the same instant. A wand
swung into view, a clap, a person walking through, a flash -- each makes a
bump in "how much changed since the previous frame" in every camera that
sees it, and the bumps line up at the true offset. Camera shake correlates
too when the cameras share a rig. Cameras with different frame rates are put
on the reference camera's clock before comparing.

`motion_signal` decodes a stretch of one video once, sequentially, at a tiny
size (64 px wide): a 2704-pixel GoPro frame costs about as much as reading
it from disk. `align` cross-correlates two signals over a range of lags and
reports the best lag with a plain verdict: how far the best match stands
above the rest. Pure numpy + cv2; runs off the GUI thread in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

THUMB_W = 160                # analysis width: motion energy, not detail
HIGHPASS = 31                # frames: slow drifts (exposure, tide) are removed before correlating
MOTION_PCT = 99.0            # the measure: the 99th percentile of |frame - previous| over the thumbnail


def motion_signal(path: str, f0: int, f1: int, progress=None, should_cancel=None,
                  thumb_w: int = THUMB_W) -> np.ndarray:
    """Frame-to-frame change of a downscaled grey copy, for frames f0 .. f1 - 1
    (the first value is 0): the 99th percentile of |difference| over the
    160-px-wide thumbnail. A high percentile, not the mean: a wand a few
    pixels across barely moves the mean of a 2704-px GoPro frame, but it is
    the brightest change in the picture. Measured on a real 8-camera rig (two
    cameras, 40 s): mean |d| correlation 0.53, 99th percentile 0.84, same
    lag. Own VideoCapture, one sequential read after a single seek. NaN where
    a frame did not decode."""
    from kinetrace.video_source import open_capture
    cap = open_capture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open {path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f0 = max(0, int(f0))
        f1 = min(int(f1), n) if n > 0 else int(f1)
        out = np.full(max(0, f1 - f0), np.nan)
        if f1 <= f0:
            return out
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        prev = None
        for k in range(f1 - f0):
            if should_cancel is not None and should_cancel():
                break
            ok, bgr = cap.read()
            if not ok:
                break
            h = max(2, int(round(bgr.shape[0] * thumb_w / max(1, bgr.shape[1]))))
            g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (thumb_w, h),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
            out[k] = 0.0 if prev is None else float(np.percentile(np.abs(g - prev), MOTION_PCT))
            prev = g
            if progress is not None and (k % 200 == 0 or k == f1 - f0 - 1):
                progress((k + 1) / (f1 - f0))
        return out
    finally:
        cap.release()


def _clean(x: np.ndarray) -> np.ndarray:
    """High-pass, NaN-safe, z-scored copy of a motion signal."""
    x = np.asarray(x, np.float64).copy()
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.zeros_like(x)
    x[~ok] = np.nanmedian(x)
    k = int(HIGHPASS) | 1
    pad = np.pad(x, k // 2, mode="edge")
    slow = np.convolve(pad, np.ones(k) / k, mode="valid")
    y = x - slow
    s = y.std()
    return (y - y.mean()) / s if s > 1e-12 else np.zeros_like(y)


def resample_to_rate(x: np.ndarray, rate: float) -> np.ndarray:
    """A signal sampled at `rate` frames per reference frame, resampled onto
    reference frames (linear). rate 2.0 = a 240 fps camera in a 120 fps rig."""
    x = np.asarray(x, np.float64)
    if abs(rate - 1.0) < 1e-9 or len(x) < 2:
        return x.copy()
    n_ref = int(np.floor((len(x) - 1) / rate)) + 1
    t = np.arange(n_ref) * rate
    return np.interp(t, np.arange(len(x)), x)


@dataclass
class AlignResult:
    lag: int                    # frames of the REFERENCE signal by which `other` must be shifted
    score: float                # normalised correlation at the best lag, in [-1, 1]
    margin: float               # best minus the best competing peak (>= 3 frames away)
    verdict: str                # "clear" | "weak" | "none"
    why: str
    lags: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    curve: np.ndarray = field(default_factory=lambda: np.zeros(0))


def align(ref: np.ndarray, other: np.ndarray, max_lag: int) -> AlignResult:
    """Best lag such that other[i] lines up with ref[i + lag]... precisely:
    the lag L maximising the normalised correlation of ref[t] with
    other[t - L] over their overlap. Positive L = `other` shows the same
    event L frames LATER in its own numbering than the reference does."""
    a = _clean(ref)
    b = _clean(other)
    max_lag = int(max(1, max_lag))
    lags = np.arange(-max_lag, max_lag + 1)
    curve = np.full(len(lags), np.nan)
    min_overlap = max(16, int(0.1 * min(len(a), len(b))))
    for k, L in enumerate(lags.tolist()):
        # ref index t, other index t - L
        t0 = max(0, L)
        t1 = min(len(a), len(b) + L)
        if t1 - t0 < min_overlap:
            continue
        sa = a[t0:t1]
        sb = b[t0 - L:t1 - L]
        na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
        if na < 1e-12 or nb < 1e-12:
            continue
        curve[k] = float(np.dot(sa, sb) / (na * nb))
    if not np.isfinite(curve).any():
        return AlignResult(0, float("nan"), float("nan"), "none",
                           "The two stretches do not overlap enough to compare.", lags, curve)
    best = int(np.nanargmax(curve))
    score = float(curve[best])
    far = np.abs(lags - lags[best]) >= 3
    rival = float(np.nanmax(curve[far])) if np.isfinite(curve[far]).any() else float("-inf")
    margin = score - rival if np.isfinite(rival) else score
    if score >= 0.45 and margin >= 0.12:
        verdict = "clear"
        why = (f"The two cameras' motion curves match best at one offset (correlation {score:.2f}, the "
               f"next-best offset is {margin:.2f} lower): a clear alignment.")
    elif score >= 0.25 and margin >= 0.05:
        verdict = "weak"
        why = (f"A likely offset, but not a sharp one (correlation {score:.2f}, next-best only {margin:.2f} "
               "lower). Check it by stepping through a moment both cameras see; a clap or a hand wave "
               "in front of every camera makes this unambiguous.")
    else:
        verdict = "none"
        why = (f"No offset stands out (best correlation {score:.2f}). The cameras probably do not see the "
               "same movement in this stretch, or it lies outside the search range. Pick a stretch with "
               "an event every camera saw, or widen the search.")
    return AlignResult(int(lags[best]), score, margin, verdict, why, lags, curve)


@dataclass
class CameraSync:
    view: int
    offset: float               # the offset to store for this view (its own frames)
    lag_ref: int                # frames of the reference
    result: AlignResult


_STAMP = None


def offsets_from_filenames(paths: list[str], fps: list[float] | float, reference: int = 0
                           ) -> list[float | None]:
    """A coarse prior from the recording clocks in the file names: GoPro and
    most action cameras write `..._YYYYMMDD_HHMMSS_...`. Camera i that started
    D seconds AFTER the reference is at frame -D*fps_i when the reference is
    at its frame 0, so its offset is about -D*fps_i. Whole seconds only (a
    prior of +-1 s, which is why the motion search still runs around it).
    None for a name without a stamp."""
    import re
    from datetime import datetime
    global _STAMP
    if _STAMP is None:
        _STAMP = re.compile(r"(\d{8})_(\d{6})")
    stamps = []
    for p in paths:
        m = _STAMP.search(str(p).replace("\\", "/").split("/")[-1])
        try:
            stamps.append(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None)
        except ValueError:
            stamps.append(None)
    fps_list = [float(fps)] * len(paths) if np.isscalar(fps) else [float(v) for v in fps]
    out: list[float | None] = []
    ref = stamps[reference] if reference < len(stamps) else None
    for i, st in enumerate(stamps):
        if st is None or ref is None:
            out.append(None)
        else:
            out.append(-(st - ref).total_seconds() * fps_list[i])
    return out


def estimate_offsets_from_motion(paths: list[str], rates: list[float], ref_range: tuple[int, int],
                                 search: int, reference: int = 0, progress=None,
                                 should_cancel=None, prior: list[float | None] | None = None
                                 ) -> list[CameraSync]:
    """For every non-reference camera, the whole-frame offset that lines its
    picture changes up with the reference camera's over reference frames
    `ref_range` = (f0, f1), searching ±`search` reference frames around a
    `prior` offset per camera (0 without one; `offsets_from_filenames` gives
    one from the recording clocks). Offsets are in the convention of
    `project.Project`: local_i = rate_i * t + offset_i, with the reference at
    offset 0. Returns one `CameraSync` per other view."""
    f0, f1 = int(ref_range[0]), int(ref_range[1])
    n_views = len(paths)
    steps = max(1, n_views)
    ref_sig = motion_signal(paths[reference], f0, f1,
                            progress=(lambda p: progress(p / steps)) if progress else None,
                            should_cancel=should_cancel)
    out: list[CameraSync] = []
    done = 1
    for i, path in enumerate(paths):
        if i == reference:
            continue
        r = float(rates[i]) if rates and i < len(rates) and rates[i] > 0 else 1.0
        pri = float(prior[i]) if (prior is not None and i < len(prior) and prior[i] is not None) else 0.0
        # the other camera's own frames covering the reference stretch (through the
        # prior offset), widened by the search on both sides
        g0 = int(np.floor(pri + r * (f0 - search)))
        g1 = int(np.ceil(pri + r * (f1 + search)))
        sig = motion_signal(path, max(0, g0), g1,
                            progress=(lambda p, d=done: progress((d + p) / steps)) if progress else None,
                            should_cancel=should_cancel)
        done += 1
        sig_ref_clock = resample_to_rate(sig, r)
        g0p = max(0, g0)
        # ref[t] is the reference at frame f0 + t; sig_ref_clock[j] is this camera at
        # ITS frame g0p + r*j. `align` pairs ref[t] with sig_ref_clock[t - L], i.e.
        # reference frame f0 + t with this camera's frame g0p + r*(t - L). With
        # local_i = r*T + offset that gives offset = g0p - r*(f0 + L). Because the
        # stretch was widened by `search` on both sides, a perfectly aligned pair
        # sits at L = -search, so the lags to try run from about -2*search to 0.
        res = align(ref_sig, sig_ref_clock, max_lag=2 * int(np.ceil(search)) + 2)
        offset = float(g0p - r * (f0 + res.lag))
        out.append(CameraSync(i, offset, res.lag, res))
    return out
