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

from dataclasses import dataclass, field

import numpy as np

from kinetrace.calib import even_indices, residual_scale, sample_tracks_at
from kinetrace.kinematics import contiguous_runs


def raw_track(session, j: int) -> np.ndarray:
    """(T, 2) float64 pixels of landmark column `j`, NaN where not exportable
    (untracked or hidden). One column, not the whole (T, N) array copied per
    (landmark, camera) as it was (R20: 17 ms against 0.4 ms at 40k x 30; the
    values are the same)."""
    col = np.asarray(session.tracks[:, j], np.float64)
    ok = session.exportable_at(slice(None), j) & np.isfinite(col).all(axis=1)
    return np.where(ok[:, None], col, np.nan)


class RunView:
    """(I271) The cameras `views` of a project, in that order, at other `offsets` (one per PROJECT
    camera; None = the project's own) - what the wand timing checks and the wand wizard sample the wand
    through, without ever changing the project (the wizard's offsets are pending until Finish, I303).

    It offers ONLY what the collect_* functions, `sampling_window`, `timing_check`, `calib`'s offset
    estimator and `wandsync` read, in run order: `sessions`, `rates`, `offsets`, `n_views`, `names` /
    `name(k)`, `reference_time`, `local_frame` - plus `views` (the project's camera numbers). Anything
    else (`map_frame`, `local_index`, ...) is an AttributeError, so a method that would silently use
    the project's own offsets is never reached. Run camera 0 is the run's clock."""

    def __init__(self, project, views=None, offsets=None):
        n = project.n_views
        self.views = list(range(n)) if views is None else [int(v) for v in views]
        offs = list(project.offsets) if offsets is None else [float(o) for o in offsets]
        if len(offs) != n:
            raise ValueError(f"{len(offs)} offsets for a project of {n} cameras")
        self.sessions = [project.sessions[v] for v in self.views]
        self.rates = [float(project.rates[v]) for v in self.views]
        self.offsets = [float(offs[v]) for v in self.views]
        self.names = [project.name(v) for v in self.views]
        self.n_views = len(self.views)

    def name(self, k: int) -> str:
        return self.names[k] if 0 <= k < len(self.names) else ""

    def reference_time(self, k: int, frame):
        """Run camera k's local `frame` -> reference instant (what `Project.reference_time` does)."""
        return (np.asarray(frame, np.float64) - self.offsets[k]) / self.rates[k]

    def local_frame(self, k: int, t):
        """Reference instant -> run camera k's local frame, fractional (`Project.local_frame`)."""
        return self.rates[k] * np.asarray(t, np.float64) + self.offsets[k]


def retimed(project, offsets=None) -> RunView:
    """Every camera of `project`, in project order, at `offsets` (None = its own): a `RunView`."""
    return RunView(project, None, offsets)


def reference_window(project) -> tuple[int, int]:
    """Integer reference instants every camera covers (kept for the tests: the
    app samples over `sampling_window`, I30)."""
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
            if q.name in seen or not s.exportable_at(slice(None), j).any():
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
        rt = raw_track(s, j)[:, None]                  # (T, 1, 2)
        # vectorised Project.local_frame: rate * t + offset for every instant
        local = project.rates[c] * np.asarray(t, np.float64) + project.offsets[c]
        out[:, c] = sample_tracks_at(rt, local)[:, 0]
    return out


def _thin(idx: np.ndarray, n_max: int) -> np.ndarray:
    if len(idx) <= n_max:
        return idx
    return idx[even_indices(len(idx), n_max)]


def collect_wand(project, name_a: str, name_b: str, max_frames: int = 400,
                 min_views: int = 2, views: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Wand observations: (F, C, 2, 2) [frame, camera, end, xy] over the
    reference instants where BOTH ends are seen by >= `min_views` cameras,
    thinned evenly to `max_frames`. Also returns the reference frames used.
    `views` (I272): only these cameras, in this order (C = len(views)); None = every camera."""
    t0, t1 = sampling_window(project)
    n_c = project.n_views if views is None else len(views)
    if t1 < t0:
        return np.zeros((0, n_c, 2, 2)), np.zeros(0, np.int64)
    t = np.arange(t0, t1 + 1)
    a = _sample_name(project, name_a, t)
    b = _sample_name(project, name_b, t)
    if views is not None:
        a, b = a[:, list(views)], b[:, list(views)]
    both = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)   # (M, C)
    keep = np.nonzero(both.sum(axis=1) >= min_views)[0]
    keep = _thin(keep, max_frames)
    uv = np.stack([a[keep], b[keep]], axis=2)                          # (F, C, 2, 2)
    # a camera that sees only ONE end on a frame contributes nothing there
    one = np.isfinite(uv).all(axis=3)                                   # (F, C, 2)
    uv[~(one[..., 0] & one[..., 1])] = np.nan
    return uv, t[keep].astype(np.int64)


def collect_extra_points(project, names: list[str], max_points: int = 3000,
                         min_views: int = 2, stride_hint: int = 1, views: list[int] | None = None) -> np.ndarray:
    """Every (name, reference frame) seen by >= `min_views` cameras is one
    independent 3D point: (B, C, 2). Thinned evenly to `max_points`. `views` as in `collect_wand`."""
    t0, t1 = sampling_window(project)
    rows = []
    if t1 >= t0 and names:
        t = np.arange(t0, t1 + 1, max(1, int(stride_hint)))
        for nm in names:
            p = _sample_name(project, nm, t)                          # (M, C, 2)
            if views is not None:
                p = p[:, list(views)]
            ok = np.isfinite(p).all(axis=2).sum(axis=1) >= min_views
            if ok.any():
                rows.append(p[ok])
    if not rows:
        return np.zeros((0, project.n_views if views is None else len(views), 2))
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


STATIC_WINDOW = 3               # frames: a mark clicked once is looked for this far either side (I304)


def collect_static_at(project, name: str, t, window: int = STATIC_WINDOW) -> np.ndarray:
    """(I304) A STATIC mark - the axes' floor marks, clicked once on one frame per camera - at reference
    instant `t`: (C, 2). Per camera `collect_at` (interpolated) where it finds the mark - so wherever
    the old code found the axes they are bit-identical (review) - else its position on the PLACED frame
    nearest to the local frame of `t` (a tie goes to the later frame: `Project.local_index`'s half-up
    rule), within `window` frames; NaN when there is none. `collect_at` interpolates, and at a
    fractional offset that needs two placed frames, so a refined (fractional) offset lost the axes."""
    out = collect_at(project, name, t)
    for c, s in enumerate(project.sessions):
        j = s.pid_by_name(name)
        if j is None or np.isfinite(out[c]).all():
            continue
        tr = raw_track(s, j)
        local = float(project.rates[c]) * float(t) + float(project.offsets[c])
        lo = max(int(np.ceil(local - window - 1e-9)), 0)
        hi = min(int(np.floor(local + window + 1e-9)), len(tr) - 1)
        if hi < lo:
            continue
        idx = np.arange(lo, hi + 1)
        idx = idx[np.isfinite(tr[idx]).all(axis=1)]
        if not len(idx):
            continue
        best = idx[np.lexsort((-idx, np.abs(idx - local)))[0]]     # nearest; a tie: the later frame
        out[c] = tr[best]
    return out


def best_common_frame(project, names: list[str], min_views: int = 2, static: bool = False) -> int | None:
    """A reference instant where EVERY name in `names` is seen by >= min_views
    cameras (the earliest), or None. `static` (I304): the names are marks clicked
    once (the axes): the interpolating answer when there is one (the same frame as
    before), else the earliest instant a camera's placed mark stands for, inside the
    sampled instants, where `collect_static_at` sees every name."""
    t0, t1 = sampling_window(project)
    if t1 < t0 or not names:
        return None
    t = np.arange(t0, t1 + 1)
    ok = np.ones(len(t), bool)
    for nm in names:
        p = _sample_name(project, nm, t)
        ok &= np.isfinite(p).all(axis=2).sum(axis=1) >= min_views
    hits = np.nonzero(ok)[0]
    if len(hits) or not static:
        return int(t[hits[0]]) if len(hits) else None
    # (review) each track read ONCE per camera and name; every candidate judged by one searchsorted -
    # the per-candidate collect_static_at re-read every track (57 s at 20k frames x 4 cameras)
    placed = {}                                                      # (camera, name) -> sorted frames
    for c, s in enumerate(project.sessions):
        for nm in names:
            j = s.pid_by_name(nm)
            if j is not None:
                placed[c, nm] = np.nonzero(np.isfinite(raw_track(s, j)).all(axis=1))[0]
    rate = [float(r) for r in project.rates]
    off = [float(o) for o in project.offsets]
    cand = set()
    for (c, nm), f in placed.items():
        tt = np.floor((f - off[c]) / rate[c] + 0.5).astype(np.int64)  # the instant each placed frame shows
        cand.update(int(x) for x in tt[(tt >= t0) & (tt <= t1)])
    if not cand:
        return None
    tc = np.asarray(sorted(cand), np.int64)
    ok = np.ones(len(tc), bool)
    for nm in names:
        seen = np.zeros(len(tc), np.int64)
        for c, s in enumerate(project.sessions):
            f = placed.get((c, nm))
            if f is None or not len(f):
                continue
            # collect_static_at's rule: a placed frame within STATIC_WINDOW of the local frame (interpolation
            # needs one within a frame, so this also covers every camera collect_at finds)
            local = rate[c] * tc.astype(np.float64) + off[c]
            lo = np.maximum(np.ceil(local - STATIC_WINDOW - 1e-9), 0)
            hi = np.minimum(np.floor(local + STATIC_WINDOW + 1e-9), s.n_frames - 1)
            n = np.searchsorted(f, hi, side="right") - np.searchsorted(f, lo, side="left")
            seen += (hi >= lo) & (n > 0)
        ok &= seen >= min_views
    hits = np.nonzero(ok)[0]
    return int(tc[hits[0]]) if len(hits) else None


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
    # "not moving": 1 px per frame at 640 px, 3 px at 4K (a hand-digitized jitter), by the LONG side (I354: the
    # width made a portrait 4K camera 1.7 px where the same camera in landscape is 3.1; landscape unchanged)
    still = np.array([max(1.0, 0.0008 * float(max(s.width or 0, s.height or 0))) for s in project.sessions])
    best = None
    for a, b in contiguous_runs(ok):
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


def wand_visibility(project, name_a: str, name_b: str) -> np.ndarray:
    """(M, C) bool over the sampling window: camera c sees BOTH wand ends at that reference instant
    (what `collect_wand` keeps, before thinning) - computed once, read by the camera ticks (I272)."""
    t0, t1 = sampling_window(project)
    if t1 < t0:
        return np.zeros((0, project.n_views), bool)
    t = np.arange(t0, t1 + 1)
    a = _sample_name(project, name_a, t)
    b = _sample_name(project, name_b, t)
    return np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)


def shared_frames(both: np.ndarray, views: list[int]) -> dict[int, int]:
    """Per camera of `views`: the instants it sees both wand ends while at least one OTHER camera of
    `views` does too - what that camera can be calibrated from in a run on `views`."""
    v = list(views)
    if not v or both.shape[0] == 0:
        return {c: 0 for c in v}
    sub = both[:, v]
    n = sub.sum(axis=1, keepdims=True)
    return {c: int((sub[:, k] & (n[:, 0] >= 2)).sum()) for k, c in enumerate(v)}


def camera_groups(both: np.ndarray, views: list[int], min_frames: int) -> list[list[int]]:
    """The cameras of `views` joined into groups: two cameras are joined when they see both wand
    ends together on at least `min_frames` instants (directly or through other cameras). One group
    = one calibration can hold them all; several = no wand frame ties one group to another."""
    v = list(views)
    parent = {c: c for c in v}

    def root(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    if both.shape[0]:
        for i, a in enumerate(v):
            for b in v[i + 1:]:
                if int((both[:, a] & both[:, b]).sum()) >= min_frames:
                    parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for c in v:
        groups.setdefault(root(c), []).append(c)
    return sorted(groups.values(), key=lambda g: g[0])


def camera_ticks(both: np.ndarray, min_frames: int) -> list[tuple[bool, int, int, str]]:
    """(I272) The wizard's default choice of cameras: per camera (ticked, instants with both wand
    ends, instants shared with another camera that has them, why it is left out in words or "").
    A camera is ticked when it shares at least `min_frames` such instants with another camera."""
    C = both.shape[1]
    n_both = both.sum(axis=0) if both.shape[0] else np.zeros(C, int)
    shared = shared_frames(both, list(range(C)))
    out = []
    for c in range(C):
        nb, sh = int(n_both[c]), shared[c]
        if nb == 0:
            out.append((False, 0, 0, "no wand clicks (no frame with both ends): left out, not calibrated"))
        elif sh < min_frames:
            out.append((False, nb, sh, f"only {sh} of its frames are shared with another camera (at least "
                                       f"{min_frames} needed): left out, not calibrated"))
        else:
            out.append((True, nb, sh, ""))
    return out


def coverage_summary(project, name_a: str, name_b: str) -> list[tuple[str, int, int]]:
    """Per camera: (name, frames with both ends, frames with at least one)."""
    out = []
    for c, s in enumerate(project.sessions):
        ja, jb = s.pid_by_name(name_a), s.pid_by_name(name_b)
        ea = s.exportable_at(slice(None), ja) if ja is not None else np.zeros(s.n_frames, bool)
        eb = s.exportable_at(slice(None), jb) if jb is not None else np.zeros(s.n_frames, bool)
        out.append((project.name(c), int((ea & eb).sum()), int((ea | eb).sum())))
    return out


# ------------------------------------------------------------- the cameras' timing from the wand (I274)
TIMING_MAX_SHIFT = 30          # frames tried each way
TIMING_MAX_ROWS = 300          # wand instants per camera pair and shift (thinned evenly)
TIMING_MIN_ROWS = 30           # a pair with fewer shared wand instants says nothing about timing
TIMING_THR_PX = 1.5            # epipolar inlier band at 1080p (scaled by the picture's long side): 3 px
                               # at 4K hid a 2-frame error between two 4K cameras at 240 fps (owner's rig)
TIMING_GOOD_SHARE = 0.6        # a pair's best shift counts when this share of its clicks fit one geometry ...
TIMING_MARGIN = 0.15           # ... and that is this much more than at the offsets as they are
WHOLE_MIN = 0.75               # (I300) a fitted shift this far from none (own frames) is a whole-frame error ...
FRACTION_MIN = 0.25            # ... between this and WHOLE_MIN it is a FRACTION of a frame: named, never rounded


@dataclass
class TimingCheck:
    """What `timing_check` found. `shift[v]`: whole frames to ADD to camera v's offset so its wand
    clicks agree with the other cameras' (None = the wand cannot time it); `suggestions`: the
    cameras whose shift is a whole-frame error (|fitted| >= WHOLE_MIN), only when every camera pair
    agrees with them; `fitted`: each camera's shift before rounding (own frames); `fraction`: the
    cameras off by a fraction of a frame (FRACTION_MIN..WHOLE_MIN, I300) - what Refine offsets with
    the wand measures; `lines`: what to tell the user; `pairs`: the evidence."""
    shift: dict = field(default_factory=dict)
    suggestions: dict = field(default_factory=dict)
    lines: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    verdict: str = "fine"          # fine | fraction | suggest | unclear
    fitted: dict = field(default_factory=dict)
    fraction: dict = field(default_factory=dict)


@dataclass
class ShiftFit:
    """(I300) `_fit_shifts`'s answer for one set of ticked cameras (see `TimingCheck` for the fields)."""
    shift: dict
    fitted: dict
    suggestions: dict
    fraction: dict
    consistent: bool
    used: list
    reach: set


def _centered(fitted: dict, clock_ticked: bool) -> dict:
    """With the project's clock (camera 1) not the fit's reference, the MIDDLE camera's unrounded fit is
    "none" (I300: the most common ROUNDED shift flipped with half-frame fits, and made the run's
    reference read a whole frame off). With an even count the one of the two middle fits nearer 0 (the
    run's reference): an actual camera's fit, never the mean of two, so a whole-frame error is never
    split half onto the right camera (review: two ticked cameras, +1 read -0.5 / +0.5)."""
    f = {c: float(v) for c, v in fitted.items()}
    if not clock_ticked and f:
        v = sorted(f.values())
        mid = v[len(v) // 2] if len(v) % 2 else min(v[len(v) // 2 - 1], v[len(v) // 2], key=abs)
        f = {c: x - mid for c, x in f.items()}
    return f


def _classify_shifts(fitted: dict, clock_ticked: bool) -> tuple[dict, dict, dict]:
    """{camera: fitted shift, own frames, unrounded} -> (suggestions {camera: whole frames}, fraction
    {camera: fitted}, shift {camera: whole frames}): a whole-frame suggestion only from WHOLE_MIN on;
    FRACTION_MIN..WHOLE_MIN is a fraction of a frame, never rounded into a suggestion (I300)."""
    f = _centered(fitted, clock_ticked)
    shift = {c: int(np.floor(v + 0.5)) for c, v in f.items()}          # (no banker's rounding)
    sugg = {c: shift[c] for c in f if shift[c] != 0 and abs(f[c]) >= WHOLE_MIN}
    frac = {c: f[c] for c in f if FRACTION_MIN <= abs(f[c]) < WHOLE_MIN}
    return sugg, frac, shift


def _fit_shifts(pairs: list, views: list[int], rates) -> ShiftFit:
    """One shift per camera of `views` from the camera pairs' best whole-frame shifts (the `pairs`
    records of `timing_check`; a pair counts only when both its cameras are in `views`). Fitted in
    REFERENCE frames (review of I274: a pair's shift d is in the second camera's OWN frames, and cameras
    may run at other rates): u_j - u_i = d_ij / r_j, weighted by the shared instants; camera v's shift
    is u_v * r_v of its own frames. The first camera of `views` - camera 1 when it is ticked - is the
    reference; a pair off the rounded fit by more than a frame makes the fit inconsistent. Pure: the
    pair evidence does not depend on which other cameras are ticked, only this step does."""
    views = list(views)
    used = [r for r in pairs if r["used"] and r["cams"][0] in views and r["cams"][1] in views]
    out = ShiftFit(shift={v: None for v in views}, fitted={v: None for v in views}, suggestions={},
                   fraction={}, consistent=True, used=used, reach=set())
    if not used:
        return out
    timed = sorted({c for r in used for c in r["cams"]})
    rate = {c: float(rates[c]) or 1.0 for c in views}
    same_rate = all(abs(rate[c] - rate[views[0]]) <= 1e-9 for c in views)
    ref = views[0] if views[0] in timed else timed[0]
    col = {c: k for k, c in enumerate(c for c in timed if c != ref)}
    A = np.zeros((len(used), len(col)))
    y = np.zeros(len(used))
    w = np.zeros(len(used))
    for r_, rec in enumerate(used):
        i, j = rec["cams"]
        if j in col:
            A[r_, col[j]] += 1.0
        if i in col:
            A[r_, col[i]] -= 1.0
        y[r_] = rec["best"] / rate[j]
        w[r_] = np.sqrt(max(rec["rows"], 1))
    sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0] if len(col) else np.zeros(0)
    u = {ref: 0.0}
    for c, k in col.items():
        u[c] = float(sol[k])
    # connected to the reference through used pairs? (a camera only in pairs among others is still
    # fixed relative to them; lstsq gives the minimum-norm answer for a separate group: say unclear)
    reach, frontier = {ref}, [ref]
    while frontier:
        c = frontier.pop()
        for rec in used:
            a_, b_ = rec["cams"]
            for x, y_ in ((a_, b_), (b_, a_)):
                if x == c and y_ not in reach:
                    reach.add(y_)
                    frontier.append(y_)
    # camera 1 (the project's clock) not timed in this run: the median of the fits is "none" (I300)
    clock_ticked = not (ref != 0 and same_rate)
    f = _centered({c: u[c] * rate[c] for c in reach}, clock_ticked)
    sugg, frac, s = _classify_shifts(f, True)
    # every pair within one of the second camera's frames of the fitted shifts
    resid = [abs(rec["best"] - rate[rec["cams"][1]] * (s[rec["cams"][1]] / rate[rec["cams"][1]]
                                                        - s[rec["cams"][0]] / rate[rec["cams"][0]]))
             for rec in used if rec["cams"][0] in reach and rec["cams"][1] in reach]
    out.consistent = all(r_ <= 1.0 + 1e-9 for r_ in resid)
    out.reach = reach
    out.shift = {c: (s[c] if c in reach else None) for c in views}
    out.fitted = {c: (f[c] if c in reach else None) for c in views}
    if out.consistent:
        out.suggestions, out.fraction = sugg, frac
    return out


def _undistorted_track(session, j: int, model) -> np.ndarray:
    """(T, 1, 2) a landmark's track in this camera's lens-corrected pixels (NaN where not tracked)."""
    tr = raw_track(session, j)
    if model is not None:
        ok = np.isfinite(tr).all(axis=1)
        if ok.any():
            tr = tr.copy()
            tr[ok] = model.undistort(tr[ok])
    return tr[:, None]


def _inlier_share(xa: np.ndarray, xb: np.ndarray, thr: float) -> float:
    """Share of point pairs that agree with ONE epipolar geometry (fundamental matrix, RANSAC with
    a fixed seed: the same project always gives the same answer)."""
    import cv2
    if len(xa) < 2 * TIMING_MIN_ROWS:
        return float("nan")
    cv2.setRNGSeed(0)
    F, m = cv2.findFundamentalMat(xa, xb, cv2.FM_RANSAC, float(thr), 0.99, 1000)
    if F is None or m is None:
        return 0.0
    return float(m.ravel().astype(bool).mean())


def _share_word(x: float) -> str:
    """A verdict for a share of wand clicks that agree with one camera layout."""
    if not np.isfinite(x):
        return "not measured"
    return "good" if x >= TIMING_GOOD_SHARE else ("ok" if x >= TIMING_GOOD_SHARE - TIMING_MARGIN else "poor")


def timing_check(project, name_a: str, name_b: str, views: list[int], lens_models: list | None = None,
                 max_shift: int = TIMING_MAX_SHIFT, progress=None) -> TimingCheck:
    """(I274) Are the cameras' offsets right, to the whole frame, by the wand clicks alone? For every
    pair of `views` with enough shared wand instants, the share of their clicks that fit one
    epipolar geometry is measured with the second camera's offset moved by -max_shift..+max_shift
    frames: a wrong offset shows as a share that is clearly higher a few frames away. The pairs'
    best shifts are then fitted with one shift per camera (`_fit_shifts`: the first camera of
    `views` - camera 1 when it is ticked - keeps its offset; with camera 1 left out, the median
    shift is taken as none) and suggested only when every pair agrees within a frame and the fit
    is at least WHOLE_MIN frames: a fraction of a frame is named, never rounded (I300 - rounding
    made the answer flip with the cameras ticked). A camera it cannot time is told why: too few
    shared wand frames, or no clear peak (I301).
    `lens_models`: per project camera, the undistortion of its lens profile (or None)."""
    views = list(views)
    out = TimingCheck(shift={v: None for v in views})
    if len(views) < 2:
        return out
    models = list(lens_models) if lens_models is not None else [None] * project.n_views
    t0, t1 = sampling_window(project)
    if t1 < t0:
        return out
    t = np.arange(t0, t1 + 1, dtype=np.float64)
    tracks = {}
    for v in views:
        s = project.sessions[v]
        ja, jb = s.pid_by_name(name_a), s.pid_by_name(name_b)
        if ja is None or jb is None:
            continue
        tracks[v] = (_undistorted_track(s, ja, models[v]), _undistorted_track(s, jb, models[v]))

    def sample(v, shift):
        local = project.rates[v] * t + project.offsets[v] + shift
        a = sample_tracks_at(tracks[v][0], local)[:, 0]
        b = sample_tracks_at(tracks[v][1], local)[:, 0]
        return a, b

    # (I280) the one pixel scale (long side / 1920, never below 1) every residual verdict uses
    scale = {v: residual_scale(project.sessions[v].width, project.sessions[v].height) for v in views}
    have = [v for v in views if v in tracks]
    pairs = [(i, j) for k, i in enumerate(have) for j in have[k + 1:]]
    for n_done, (i, j) in enumerate(pairs):
        if progress is not None:
            progress(n_done / max(len(pairs), 1), f"Checking the timing of {project.name(i)} and {project.name(j)}")
        ai, bi = sample(i, 0)
        thr = TIMING_THR_PX * max(scale[i], scale[j])
        shares, rows = {}, 0

        def score(d):
            nonlocal rows
            if d in shares or abs(d) > max_shift:
                return
            aj, bj = sample(j, d)
            ok = (np.isfinite(ai).all(axis=1) & np.isfinite(bi).all(axis=1)
                  & np.isfinite(aj).all(axis=1) & np.isfinite(bj).all(axis=1))
            idx = np.nonzero(ok)[0]
            if d == 0:
                rows = len(idx)
            if len(idx) < TIMING_MIN_ROWS:
                shares[d] = float("nan")
                return
            idx = _thin(idx, TIMING_MAX_ROWS)
            xa = np.vstack([ai[idx], bi[idx]]).astype(np.float64)
            xb = np.vstack([aj[idx], bj[idx]]).astype(np.float64)
            shares[d] = _inlier_share(xa, xb, thr)

        # every other shift, then the neighbours of the best three (a peak is a few frames wide)
        for d in range(-int(max_shift), int(max_shift) + 1, 2):
            score(d)
        score(0)
        top = sorted((d for d in shares if np.isfinite(shares[d])), key=lambda d: -shares[d])[:3]
        for d0 in top:
            for d in (d0 - 1, d0 + 1):
                score(d)
        rec = {"cams": (i, j), "rows": rows, "share0": shares.get(0, float("nan")), "best": None,
               "share_best": float("nan"), "used": False}
        fin = {d: v for d, v in shares.items() if np.isfinite(v)}
        if 0 in fin and rows >= TIMING_MIN_ROWS:
            best = max(fin, key=lambda d: (fin[d], -abs(d)))
            rec["best"], rec["share_best"] = int(best), float(fin[best])
            # evidence only from a clear peak: at the offsets as they are (the best IS 0), or clearly
            # better elsewhere; a pair whose best is elsewhere by a hair says nothing either way
            if fin[best] >= TIMING_GOOD_SHARE and (best == 0 or fin[best] - fin[0] >= TIMING_MARGIN):
                rec["used"] = True
        out.pairs.append(rec)
    fit = _fit_shifts(out.pairs, views, project.rates)
    used = fit.used
    names = {c: project.name(c) for c in views}
    if not used:
        out.verdict = "unclear"
        out.lines = ["The wand clicks cannot check the cameras' timing here (too few frames that two "
                     "cameras share, or no clear agreement at any offset)."]
        return out
    out.shift, out.fitted = dict(fit.shift), dict(fit.fitted)
    if not fit.consistent:
        out.verdict = "unclear"
        out.lines = ["The camera pairs disagree about the timing (no single offset per camera fits all of them): "
                     "check the offsets by hand at a sharp event (CAMERAS panel), or a wand end clicked as the "
                     "other end in one camera."]
        return out
    out.suggestions, out.fraction = dict(fit.suggestions), dict(fit.fraction)
    for c, d in sorted(out.suggestions.items()):
        ev = [rec for rec in used if c in rec["cams"] and rec["best"] != 0]
        now = float(np.mean([rec["share0"] for rec in ev])) if ev else float("nan")
        then = float(np.mean([rec["share_best"] for rec in ev])) if ev else float("nan")
        old = float(project.offsets[c])
        out.lines.append(
            f"{names[c]}: its wand clicks agree with the other cameras when its offset is {old + d:g} instead of "
            f"{old:g} ({abs(d)} frame{'s' if abs(d) != 1 else ''} {'more' if d > 0 else 'less'}): "
            f"{100 * then:.0f} % of its clicks agree with one camera layout then ({_share_word(then)}), "
            f"{100 * now:.0f} % now ({_share_word(now)}).")
    if out.suggestions:
        out.verdict = "suggest"
    elif out.fraction:
        out.verdict = "fraction"
    if out.fraction:
        frac = ", ".join(f"{names[c]} (about {v:+.2f})" for c, v in sorted(out.fraction.items()))
        out.lines.append(
            f"Off by a FRACTION of a frame by the camera pairs: {frac}. This check sets whole frames only, and a "
            "fraction must not be rounded into one: after calibrating, press Refine offsets with the wand, "
            "which measures it.")
    if not out.suggestions and not out.fraction:
        out.lines.append("No camera is a whole frame off by the wand clicks (a fraction of a frame is what Refine "
                         "offsets with the wand measures).")
    for c in views:
        if out.shift[c] is not None:
            continue
        mine = [r for r in out.pairs if c in r["cams"]]
        rows = max((int(r["rows"]) for r in mine), default=0)
        if rows < TIMING_MIN_ROWS:
            why = (f"too few wand frames shared with another camera: at most {rows}, {TIMING_MIN_ROWS} needed")
        elif any(r["used"] for r in mine):
            # (review) its pairs DID give a clear shift, only within a group of cameras that never reaches the
            # first ticked one - "not a clear peak" would be false
            why = ("its wand clicks agree with "
                   + ", ".join(sorted({names[x] for r in mine if r["used"] for x in r["cams"] if x != c}))
                   + ", but no clear shift links that group to the other ticked cameras")
        else:
            best = max((r for r in mine if np.isfinite(r["share_best"])), key=lambda r: r["share_best"],
                       default=None)
            if best is None:
                why = "no clear agreement with any other camera at any whole-frame shift"
            else:
                other = best["cams"][0] if best["cams"][1] == c else best["cams"][1]
                d = best["best"] if best["cams"][1] == c else -best["best"]
                sb, s0 = best["share_best"], best["share0"]
                at = f"with {names[other]} at {d:+d} frame{'s' if abs(d) != 1 else ''}"
                if sb < TIMING_GOOD_SHARE:
                    how = (f"at best {100 * sb:.0f} % of its wand clicks fit one camera layout ({at}), and "
                           f"{100 * TIMING_GOOD_SHARE:.0f} % is needed")
                else:
                    how = (f"{100 * sb:.0f} % of its wand clicks fit one camera layout {at}, but that is only "
                           f"{100 * (sb - s0):.0f} points more than at its offset as it is "
                           f"({100 * TIMING_MARGIN:.0f} are needed for a clear peak)")
                why = (f"no clear agreement at any whole-frame shift: {how}. Refine offsets with the wand can "
                       "still time it after calibrating; if it reads poor there, check its offset by hand at a "
                       "sharp event (CAMERAS panel)")
        out.lines.append(f"The wand cannot check the timing of {names[c]} ({why}).")
    return out
