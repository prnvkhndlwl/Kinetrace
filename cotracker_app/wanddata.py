"""Turning a Kinetrace project into wand-calibration observations.

`wand.py` wants plain arrays: the two wand ends per frame per camera, extra
points, a falling object, reference points. This module builds them from a
`Project` (one `TrackingSession` per camera, with rates and offsets): the
landmarks are matched BY NAME across cameras, every view is sampled at the
reference instants through its own rate / offset (linear interpolation at
fractional frames, never across a gap — exactly what the 3D layer does), and
cells the user marked hidden are dropped. Pure numpy, no Qt.
"""

from __future__ import annotations

import warnings

import numpy as np

from cotracker_app.calib import sample_tracks_at


def raw_tracks(session) -> np.ndarray:
    """(T, N, 2) float64 pixels, NaN where not exportable (untracked or hidden)."""
    out = np.asarray(session.tracks, np.float64).copy()
    ok = session.exportable & np.isfinite(session.tracks).all(axis=2)
    out[~ok] = np.nan
    return out


def reference_window(project) -> tuple[int, int]:
    """Integer reference instants every camera covers."""
    lo, hi = -np.inf, np.inf
    for i, s in enumerate(project.sessions):
        lo = max(lo, project.reference_time(i, 0))
        hi = min(hi, project.reference_time(i, s.n_frames - 1))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return (0, -1)
    return int(np.ceil(lo)), int(np.floor(hi))


def sampling_window(project) -> tuple[int, int]:
    """Integer reference instants ANY camera covers - what the wand, the extra
    points and the reference frames are sampled over (I30). Only >= 2 cameras
    are ever needed at an instant, and `sample_tracks_at` is NaN outside a
    camera's own frames, so the per-instant view count does the filtering;
    the all-camera intersection (`reference_window`) threw away every wand
    frame of a hand-started rig recorded before its last camera started.
    May start below 0 (a camera that started before the reference)."""
    lo, hi = np.inf, -np.inf
    for i, s in enumerate(project.sessions):
        if s.n_frames <= 0:
            continue
        a = project.reference_time(i, 0)
        b = project.reference_time(i, s.n_frames - 1)
        lo, hi = min(lo, a, b), max(hi, a, b)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return (0, -1)
    return int(np.ceil(lo)), int(np.floor(hi))


def common_point_names(project, min_views: int = 2) -> list[str]:
    """Landmark names tracked (with data) in at least `min_views` cameras, in
    first-seen order."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for s in project.sessions:
        seen = set()
        for j, q in enumerate(s.points):
            if q.name in seen or not s.exportable[:, j].any():
                continue
            seen.add(q.name)
            if q.name not in counts:
                order.append(q.name)
            counts[q.name] = counts.get(q.name, 0) + 1
    return [n for n in order if counts[n] >= min_views]


def _sample_name(project, name: str, t: np.ndarray) -> np.ndarray:
    """(M, C, 2) pixels of landmark `name` in every camera at reference
    instants `t` (NaN where a camera lacks it)."""
    C = project.n_views
    out = np.full((len(t), C, 2), np.nan)
    for c, s in enumerate(project.sessions):
        j = s.pid_by_name(name)
        if j is None:
            continue
        rt = raw_tracks(s)[:, j:j + 1]                 # (T, 1, 2)
        # vectorised Project.local_frame: rate * t + offset for every instant
        local = project.rates[c] * np.asarray(t, np.float64) + project.offsets[c]
        out[:, c] = sample_tracks_at(rt, local)[:, 0]
    return out


def _thin(idx: np.ndarray, n_max: int) -> np.ndarray:
    if len(idx) <= n_max:
        return idx
    return idx[np.linspace(0, len(idx) - 1, n_max).round().astype(int)]


def collect_wand(project, name_a: str, name_b: str, max_frames: int = 400,
                 min_views: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Wand observations: (F, C, 2, 2) [frame, camera, end, xy] over the
    reference instants where BOTH ends are seen by >= `min_views` cameras,
    thinned evenly to `max_frames`. Also returns the reference frames used."""
    t0, t1 = sampling_window(project)
    if t1 < t0:
        return np.zeros((0, project.n_views, 2, 2)), np.zeros(0, np.int64)
    t = np.arange(t0, t1 + 1)
    a = _sample_name(project, name_a, t)
    b = _sample_name(project, name_b, t)
    both = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)   # (M, C)
    keep = np.nonzero(both.sum(axis=1) >= min_views)[0]
    keep = _thin(keep, max_frames)
    uv = np.stack([a[keep], b[keep]], axis=2)                          # (F, C, 2, 2)
    # a camera that sees only ONE end on a frame contributes nothing there
    one = np.isfinite(uv).all(axis=3)                                   # (F, C, 2)
    uv[~(one[..., 0] & one[..., 1])] = np.nan
    return uv, t[keep].astype(np.int64)


def collect_extra_points(project, names: list[str], max_points: int = 3000,
                         min_views: int = 2, stride_hint: int = 1) -> np.ndarray:
    """Every (name, reference frame) seen by >= `min_views` cameras is one
    independent 3D point: (B, C, 2). Thinned evenly to `max_points`."""
    t0, t1 = sampling_window(project)
    rows = []
    if t1 >= t0 and names:
        t = np.arange(t0, t1 + 1, max(1, int(stride_hint)))
        for nm in names:
            p = _sample_name(project, nm, t)                          # (M, C, 2)
            ok = np.isfinite(p).all(axis=2).sum(axis=1) >= min_views
            if ok.any():
                rows.append(p[ok])
    if not rows:
        return np.zeros((0, project.n_views, 2))
    bg = np.concatenate(rows, axis=0)
    idx = _thin(np.arange(len(bg)), max_points)
    return bg[idx]


def collect_drop(project, name: str, t0: int, t1: int) -> np.ndarray:
    """A falling object at CONSECUTIVE reference frames t0..t1: (N, C, 2)."""
    if t1 < t0:
        t0, t1 = t1, t0
    t = np.arange(int(t0), int(t1) + 1)
    return _sample_name(project, name, t)


def collect_at(project, name: str, t: int) -> np.ndarray:
    """One landmark at one reference instant: (C, 2)."""
    return _sample_name(project, name, np.asarray([int(t)]))[0]


def best_common_frame(project, names: list[str], min_views: int = 2) -> int | None:
    """A reference instant where EVERY name in `names` is seen by >= min_views
    cameras (the earliest), or None."""
    t0, t1 = sampling_window(project)
    if t1 < t0 or not names:
        return None
    t = np.arange(t0, t1 + 1)
    ok = np.ones(len(t), bool)
    for nm in names:
        p = _sample_name(project, nm, t)
        ok &= np.isfinite(p).all(axis=2).sum(axis=1) >= min_views
    hits = np.nonzero(ok)[0]
    return int(t[hits[0]]) if len(hits) else None


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) index of every run of True."""
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def _fall_in_run(uv: np.ndarray, seen: np.ndarray, still: np.ndarray, min_frames: int):
    """(first, last, trimmed_start, trimmed_end) indices of the free fall in one
    stretch where >= 2 cameras see the object, or None. `still` = per camera
    the image speed (px/frame) below which it counts as not moving."""
    n = len(uv)
    if n < min_frames:
        return None
    v = uv[1:] - uv[:-1]                                         # (n-1, C, 2) per-camera velocity
    both = seen[1:] & seen[:-1]
    with np.errstate(invalid="ignore"):
        spd = np.linalg.norm(v, axis=2) / still[None, :]         # in units of "still"
    spd[~both] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows: a transition no camera saw
        s = np.nanmedian(spd, axis=1)                            # (n-1,) per transition
        dots = (v[1:] * v[:-1]).sum(axis=2)
        nrm = np.linalg.norm(v[1:], axis=2) * np.linalg.norm(v[:-1], axis=2)
        cos = np.where(both[1:] & both[:-1] & (nrm > 0), dots / np.where(nrm > 0, nrm, 1.0), np.nan)
        cmed = np.nanmedian(cos, axis=1)                         # (n-2,) turn between transitions k, k+1
    moving = np.nan_to_num(s, nan=0.0) >= 1.0
    # motion starts at the first of three consecutive moving transitions (one noisy
    # transition during a hold is not a release)
    m0 = next((k for k in range(len(s) - 2) if moving[k] and moving[k + 1] and moving[k + 2]), None)
    if m0 is None:
        return None
    # a rest before it (>= 4 still transitions) is left out, keeping the last two: a fall
    # from rest begins slowly, and two frames of rest cost the parabola nothing
    trimmed_start = m0 >= 4
    first = m0 - 2 if trimmed_start else 0
    last = n - 1
    trimmed_end = False
    for k in range(m0 + 1, len(s)):
        prev, cur = s[k - 1], s[k]
        if not np.isfinite(cur):
            last, trimmed_end = k, True
            break
        if prev >= 2.0 and (cur < 1.0 or cur < 0.5 * prev):          # lands, or is caught
            last, trimmed_end = k, True
            break
        if prev >= 2.0 and cur >= 1.0 and np.isfinite(cmed[k - 1]) and cmed[k - 1] < 0.0:   # bounces
            last, trimmed_end = k, True
            break
    if last - first + 1 < min_frames:
        return None
    seg = s[first:last]
    if not np.isfinite(seg).any() or np.nanmedian(seg) < 2.0:      # a jittering mark, not a fall
        return None
    return first, last, trimmed_start, trimmed_end


def find_fall(project, name: str, t0: int | None = None, t1: int | None = None,
              min_frames: int = 6) -> tuple[int, int, str] | None:
    """The reference frames where landmark `name` falls freely, found from the
    pictures alone - there is no calibration yet (I26). Within each stretch
    seen by >= 2 cameras: the time it rests before being let go (>= 4 still
    frames) is left out, and so is everything from the first bounce (its
    image velocity turns back), landing or catch (the speed collapses) on;
    the stretch with the most image travel wins. Returns (first, last, a
    note for the page) or None when nothing moves steadily for `min_frames`
    frames - a floor mark or a resting ball is not a dropped object."""
    if t0 is None or t1 is None:
        t0, t1 = sampling_window(project)
    if t1 < t0:
        return None
    t = np.arange(int(t0), int(t1) + 1)
    uv = _sample_name(project, name, t)                              # (M, C, 2)
    seen = np.isfinite(uv).all(axis=2)
    ok = seen.sum(axis=1) >= 2
    # "not moving": 1 px per frame at 640 px wide, 3 px at 4K (a hand-digitized jitter)
    still = np.array([max(1.0, 0.0008 * float(s.width or 0)) for s in project.sessions])
    best = None
    for a, b in _runs(ok):
        r = _fall_in_run(uv[a:b + 1], seen[a:b + 1], still, min_frames)
        if r is None:
            continue
        i0, i1, ts, te = r
        seg = uv[a + i0:a + i1 + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            travel = float(np.nansum(np.nanmedian(np.linalg.norm(seg[1:] - seg[:-1], axis=2)
                                                  / still[None, :], axis=1)))
        if best is None or travel > best[0]:
            best = (travel, a + i0, a + i1, ts, te)
    if best is None:
        return None
    _, i0, i1, ts, te = best
    left = [w for w, on in (("the rest before it", ts), ("what follows it (landing, bounce or catch)", te)) if on]
    note = (" - " + " and ".join(left) + " left out") if left else ""
    return int(t[i0]), int(t[i1]), note


def coverage_summary(project, name_a: str, name_b: str) -> list[tuple[str, int, int]]:
    """Per camera: (name, frames with both ends, frames with at least one)."""
    out = []
    for c, s in enumerate(project.sessions):
        ja, jb = s.pid_by_name(name_a), s.pid_by_name(name_b)
        ea = s.exportable[:, ja] if ja is not None else np.zeros(s.n_frames, bool)
        eb = s.exportable[:, jb] if jb is not None else np.zeros(s.n_frames, bool)
        out.append((project.name(c), int((ea & eb).sum()), int((ea | eb).sum())))
    return out
