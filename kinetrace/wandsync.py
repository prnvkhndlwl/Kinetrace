"""Camera timing from the wand (I271): refine the cameras' offsets with the wand - to a fraction of a
frame, whole frames too - calibrating again at them, and the checks that say whether the timing can be
trusted at all.

Why a loop
----------
3D -> Sync -> Estimate Sub-frame Offsets holds the calibration FIXED and moves the offsets to where
the cameras agree best. But a wand calibration fitted with wrong offsets has already bent itself a
little towards them (the bundle adjustment absorbs part of the timing error into the camera poses and
focal lengths), so an estimate made against it moves the offsets only part of the way: it
under-corrects. `refine` alternates the two steps instead -

    calibrate at the offsets -> estimate the offsets with that calibration fixed -> calibrate again
    at the new offsets -> ...

until no offset moves by more than `ROUND_TOL` frame, or `MAX_ROUNDS` rounds; the calibration it
offers is fitted at exactly the offsets it suggests (the owner, 2026-10-15: refining the offsets must
not leave the calibration behind). Each round's estimate searches +-1 frame twice over, so a camera
whole frames off is walked to its place too. This is deliberately not a joint bundle adjustment of
cameras and offsets: the alternation reuses the two solvers the app already trusts and reports each
step.

The estimate is ROBUST (`calib.estimate_offsets(robust=True)`): a first search on a capped cost gets
near, the observations behind the cells that are still far off there (a wand end swapped in one
camera, a click on the wrong ball) are removed, and the ordinary search runs on the rest - so a few bad
frames cannot drag the minimum (on the synthetic rig 40 swapped frames moved the plain estimate 0.3
frame, the robust one not at all). The sharpness verdict (how much the disagreement rises half a frame
either side, on each camera's OWN cells, I302) is estimate_offsets' own. Because a calibration can
absorb a timing error the fixed-calibration sharpness cannot see, `joint_sharpness` also moves each
camera half a frame and calibrates AGAIN.

Which cameras
-------------
Everything runs on the cameras of one wand run (the wizard's ticks, I272), in that order, through a
`wanddata.RunView`, with the run's own calibration (its cameras in the same order). Run camera 0 - the
first ticked camera, camera 1 when it is ticked - is the run's CLOCK: it never moves and has no row.
Every result is handed out by PROJECT camera number.

The checks
----------
* `split_half`: the offsets estimated again on the first and on the second half of the wand frames,
  the calibration fixed. Two halves of one steady recording must agree within `agreement_tolerance`;
  when they do not, the timing CHANGED during the recording: a frame-rate drift or a dropped frame,
  and no single offset is right.
* `speed_check`: if a camera's timing is off, a moving wand end is seen where it was a fraction of a
  frame earlier or later, and the triangulated wand length is wrong in proportion to the wand's SPEED:
  the fastest quarter of the wand frames is clearly worse than the slowest quarter ("sync suspect");
  the camera whose removal takes that away (leave-one-camera-out) is the one most implicated. A
  screen, not a measurement.
* `window_check`: the offsets estimated on the start, the middle and the end of the wand frames. A
  steady drift (start < middle < end) is a frame-rate mismatch; one step between two windows is a
  dropped (or repeated) frame. Detected and said, never corrected (the owner, 2026-10-15).
* `prediction_error`: each camera's wand ends triangulated from the OTHER cameras and projected into
  it - how far from where it saw them (what the app's diamond does). A camera whose timing is off
  stands out at once.

The agreement tolerance
-----------------------
For one camera, a timing error of d frames moves each of its wand-end observations by v * d pixels (v
= the end's image velocity in that camera's own pixels per frame). With picture noise s px per axis,
the least-squares error of d over cells with velocities v_i is

    sigma = s / sqrt(sum |v_i|^2)

(`timing_sigma`; s from the per-camera triangulation errors, the sum over the instants the estimator
really scores). Two estimates agree when they differ by at most `AGREE_K` times their combined sigma,
and never by less than `AGREE_FLOOR` frame. `AGREE_K` was set on the synthetic rig: steady recordings
with 0.3 px noise and swapped ends stay inside it, a drift of 0.5 frame over the recording does not.

Hand check (what the report tells the user to do): step to the frame where the wand moves fastest
(`speed_check`'s `fastest_frame`) and turn on the epipolar guides: the line from a wand end in one
camera must pass through that end in every other camera. A line that misses ALONG the wand's direction
of travel is a timing error, not a calibration error.

Offsets found with the wand describe the wand recording. They carry to the trials only if the cameras
kept recording between the two (one continuous recording per camera); a camera restarted between them
has a new offset.

Pure numpy / scipy, no Qt. Everything that goes into a report is `json_safe` (strict JSON: the report
is written into the .kcal.json).
"""
from __future__ import annotations

import warnings

import numpy as np

from kinetrace import calib as _calib
from kinetrace import wanddata as wd

ROUND_TOL = 0.01            # frames: the loop stops once no offset moves more than this ...
MAX_ROUNDS = 4              # ... or after this many calibrate -> estimate rounds
SETTLED_TOL = 0.05          # a row stays "good" when the last estimate moved less than this (half AGREE_FLOOR)
SEARCH = 1.0                # frames either side of the current offsets (the estimator's own default)
WINDOW_SEARCH = 1.5         # the window / half checks look wider: a dropped frame is a whole-frame step
AGREE_K = 4.0               # two estimates agree within AGREE_K x their combined sigma ...
AGREE_FLOOR = 0.1           # ... and never closer than this (frames; steady synthetic halves / thirds
                            # scatter by up to ~0.06: interpolation between frames, what the calibration
                            # absorbed)
FINE_WINDOWS = 6            # a drift or a jump seen in three windows is looked at again in this many
EVEN_SHARE = 0.33           # window steps of one sign within this share of each other = a drift; one
                            # step under this share of the other = a jump
JUMP_MIN_FRAMES = 0.5       # (I353) a dropped / repeated frame is a whole-frame step: a smaller step between
                            # windows is no "jump" (the 3-window check and the fine one alike)
MIN_CHECK_FRAMES = 30       # wand frames a half / window needs before its estimate counts
SPEED_SUSPECT_RATIO = 2.0   # fastest-quarter / slowest-quarter wand-length error: sync suspect (speed_check)
SPEED_OK_RATIO = 1.5        # ... above this "may be slightly off"; clean synthetic rigs read 1.2-1.3
SPEED_P = 1e-4              # ... and only when the difference is this clear (one-sided Mann-Whitney)
SPEED_MAX_POINTS = 1500     # wand frames the speed check looks at (evenly spread)
SPEED_RANGE_MIN = 1.5       # the fast quarter must move this many times faster than the slow one to see anything
LOO_SHARE = 0.5             # leaving one camera out must remove at least half the excess to name it
PROFILE_STEP = 0.5          # frames: the joint check shifts one camera this much and calibrates again
JOINT_SHARP, JOINT_FLAT = 0.25, 0.08  # share of a half-frame shift the recalibrated fit still shows
                            # (joint_sharpness); synthetic rig: steady cameras 0.40-0.45, a drifting or
                            # frame-dropping one ~0.2, a wand always moved one way at one speed 0.0-0.15
OFF_BY_CAVEAT = 0.1         # frames: the window check is read only once the wand agrees with the offsets
PREDICT_MIN = 10            # wand ends the other cameras must predict before a camera gets a verdict
CONTINUOUS_NOTE = ("Offsets measured with the wand describe the wand recording: they carry over to a trial only "
                   "if every camera kept recording in between (one continuous recording per camera). A camera "
                   "restarted in between has a new offset (sync that trial on its own: a clap or a flash).")


class _Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------------------- strict JSON
def json_safe(obj):
    """`obj` with numpy types as Python ones, tuples as lists, NaN / inf as None - strict JSON (the
    wand report is written into the .kcal.json). Dict keys keep their (int) camera numbers."""
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------------- the run's cameras
def _run(project, views, offsets) -> wd.RunView:
    return wd.RunView(project, views, offsets)


def _full(base, views, run_offsets) -> list[float]:
    """The project's offsets `base` with the run's cameras set to `run_offsets` (run order)."""
    out = [float(o) for o in base]
    for k, v in enumerate(views):
        out[v] = float(run_offsets[k])
    return out


def _by_project(d: dict, views) -> dict:
    """{run camera: value} -> {project camera: value}."""
    return {int(views[int(k)]): v for k, v in d.items()}


def _cal(res):
    return res.to_calibration() if hasattr(res, "to_calibration") else res


def _frames(rv, a: str, b: str) -> np.ndarray:
    """Reference instants where both wand ends are seen by >= 2 of the run's cameras."""
    _, frames = wd.collect_wand(rv, a, b, max_frames=10 ** 9)
    return np.asarray(frames, np.int64)


# --------------------------------------------------------------------------------------- the estimate
def _estimate(rv, cal, a: str, b: str, t_range, search: float = SEARCH, progress=None,
              cancelled=None) -> tuple[list[float], dict]:
    rep: dict = {}
    offs, before, after = _calib.estimate_offsets(
        rv.sessions, cal, list(rv.rates), list(rv.offsets), (int(t_range[0]), int(t_range[1])),
        names=[a, b], search=search, report=rep, progress=progress, cancelled=cancelled, robust=True)
    rep.setdefault("before_px", before)
    rep.setdefault("after_px", after)
    return [float(o) for o in offs], rep


def estimate(project, cal, a: str, b: str, offsets, t_range, search: float = SEARCH, progress=None,
             cancelled=None, views=None) -> tuple[list[float], dict]:
    """The offsets that make the wand ends agree best, the run's calibration `cal` held fixed:
    `calib.estimate_offsets` on the two wand landmarks only, with the robust search. Returns (the
    project's offsets with the run's cameras moved, its report by project camera)."""
    full = [float(o) for o in offsets]
    rv = _run(project, views, full)
    est, rep = _estimate(rv, cal, a, b, t_range, search, progress, cancelled)
    for key in ("sharpness", "per_view", "own_cells"):
        if isinstance(rep.get(key), dict):
            rep[key] = _by_project(rep[key], rv.views)
    return _full(full, rv.views, est), rep


def _sigma(rv, cal, a: str, b: str, offsets, t_range) -> np.ndarray:
    """(C,) the best-case standard error, in frames, of each run camera's offset estimated over
    `t_range` (module docstring); NaN for the run's clock and for a camera with no shared moving end."""
    C = rv.n_views
    prep = _calib._Prepared(rv.sessions, cal, [a, b])
    t_all = np.arange(int(t_range[0]), int(t_range[1]) + 1)
    out = np.full(C, np.nan)
    if not len(t_all):
        return out
    u0 = prep.samples(rv.rates, offsets, t_all)                            # (T, 2, C, 2)
    u1 = prep.samples(rv.rates, offsets, t_all + 1)
    shared = np.isfinite(u0).all(axis=3).sum(axis=2) >= 2                  # (T, 2)
    _, _, _, err = prep.triangulate(u0, 2)                                 # (T, 2, C) px
    # the estimator scores at most MAX_SCORE_INSTANTS instants: the information is that share
    n_inst = int(shared.any(axis=1).sum())
    share = min(1.0, _calib.MAX_SCORE_INSTANTS / max(n_inst, 1))
    for c in range(1, C):
        v = (u1[:, :, c] - u0[:, :, c]) / float(rv.rates[c])              # px per LOCAL frame
        ok = shared & np.isfinite(v).all(axis=2)
        e = err[:, :, c][ok & np.isfinite(err[:, :, c])]
        if not ok.any() or not len(e):
            continue
        # the median 2D error of a round Gaussian is 1.177 sigma per axis
        s_px = max(float(np.median(e)) / 1.177, 0.05)
        info = float((v[ok] ** 2).sum()) * share
        if info > 0:
            out[c] = s_px / np.sqrt(info)
    return out


def agreement_tolerance(sig1: float, sig2: float) -> float:
    """Frames two estimates of one camera's offset may differ by and still agree (module docstring)."""
    s = np.sqrt(np.nan_to_num(sig1, nan=0.0) ** 2 + np.nan_to_num(sig2, nan=0.0) ** 2)
    return float(max(AGREE_FLOOR, AGREE_K * s))


def _split(frames: np.ndarray, n: int) -> list[tuple[int, int]]:
    """`n` consecutive stretches of the wand frames, each with the same number of wand frames (not of
    reference instants: a pause in the waving does not leave a window empty)."""
    frames = np.sort(np.asarray(frames, np.int64))
    parts = np.array_split(frames, n)
    return [(int(p[0]), int(p[-1])) for p in parts if len(p)]


def _part_estimates(rv, cal, a, b, frames, n, progress=None, cancelled=None):
    """Offsets, reports and sigmas on `n` consecutive stretches of the wand frames (run order)."""
    out = []
    parts = _split(frames, n)
    for k, rng in enumerate(parts):
        n_fr = int(((frames >= rng[0]) & (frames <= rng[1])).sum())
        if n_fr < MIN_CHECK_FRAMES:
            out.append({"range": rng, "n_frames": n_fr, "offsets": None, "report": {}, "sigma": None})
            continue
        sub = None if progress is None else (lambda d, t, k=k: progress((k + d / max(t, 1)) / len(parts)))
        offs, rep = _estimate(rv, cal, a, b, rng, WINDOW_SEARCH, sub, cancelled)
        if rep.get("verdict") == "cancelled":
            raise _Cancelled()
        out.append({"range": rng, "n_frames": n_fr, "offsets": offs, "report": rep,
                    "sigma": _sigma(rv, cal, a, b, offs, rng)})
    return out


def _usable(part, c) -> bool:
    return (part["offsets"] is not None
            and part["report"].get("per_view", {}).get(c) in ("sharp", "weak"))


def _split_half(rv, cal, a: str, b: str, frames, progress=None, cancelled=None) -> dict:
    """The offsets estimated on each half of the wand frames, the calibration fixed (run order). Per
    camera: both values, the difference, the tolerance and "agree" / "disagree" / "unknown", plus a
    sentence."""
    parts = _part_estimates(rv, cal, a, b, frames, 2, progress, cancelled)
    per = {}
    for c in range(1, rv.n_views):
        row = {"halves": [None if p["offsets"] is None else p["offsets"][c] for p in parts]}
        if len(parts) < 2 or not all(_usable(p, c) for p in parts):
            row.update(status="unknown", diff=float("nan"), tol=float("nan"),
                       why="one half of the wand frames is too short, or the wand barely moves in it")
        else:
            o1, o2 = parts[0]["offsets"][c], parts[1]["offsets"][c]
            tol = agreement_tolerance(parts[0]["sigma"][c], parts[1]["sigma"][c])
            d = o2 - o1
            ok = abs(d) <= tol
            row.update(status="agree" if ok else "disagree", diff=float(d), tol=tol,
                       why=(f"the two halves agree ({o1:+.3f} and {o2:+.3f}, within {tol:.3f})" if ok else
                            f"the first half says {o1:+.3f}, the second {o2:+.3f}: {abs(d):.2f} frame apart "
                            f"(more than {tol:.3f} can be noise). The timing changed during the recording: "
                            "a frame-rate drift or a dropped frame"))
        per[c] = row
    return {"parts": [{"range": p["range"], "n_frames": p["n_frames"]} for p in parts], "per_camera": per}


def _line_or_step(x: np.ndarray, y: np.ndarray):
    """Which explains offsets `y` at window centres `x` better: a straight line (a drift) or one step (a
    dropped / repeated frame)? -> ("drift", rate, None) / ("jump", step, (x before, x after)) / None when
    neither is clearly better (each must leave under half the other's squared misfit)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    k1, k0 = np.polyfit(x, y, 1)
    rss_line = float(((y - (k1 * x + k0)) ** 2).sum())
    best = None
    for k in range(1, len(y)):
        rss = float(((y[:k] - y[:k].mean()) ** 2).sum() + ((y[k:] - y[k:].mean()) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, k)
    rss_step, k = best
    step = float(y[k:].mean() - y[:k].mean())
    if rss_step < 0.5 * rss_line and abs(step) >= JUMP_MIN_FRAMES:
        return "jump", step, (float(x[k - 1]), float(x[k]))
    if rss_line < 0.5 * rss_step:
        return "drift", float(k1), None
    return None


def _window_check(rv, cal, a: str, b: str, frames, progress=None, cancelled=None) -> dict:
    """The offsets estimated on the start, middle and end of the wand frames (run order). Per camera a
    pattern: "steady", "drift" (a frame-rate mismatch: the offset changes steadily), "jump" (a step: a
    dropped or repeated frame), "unsteady" or "unknown", with a sentence; and a verdict. Three windows
    cannot tell a step INSIDE the middle window from a drift, so a camera that reads drift or jump is
    looked at again on `FINE_WINDOWS` windows, where a straight line and a single step are told apart."""
    def part(lo, hi):
        return None if progress is None else (lambda f: progress(lo + (hi - lo) * f))

    parts = _part_estimates(rv, cal, a, b, frames, 3, part(0.0, 0.5), cancelled)
    fine = None
    per = {}
    for c in range(1, rv.n_views):
        vals = [None if p["offsets"] is None else float(p["offsets"][c]) for p in parts]
        row = {"windows": vals, "ranges": [p["range"] for p in parts]}
        if len(parts) < 3 or not all(_usable(p, c) for p in parts):
            row.update(pattern="unknown", verdict="ok",
                       why="a window is too short or the wand barely moves in it: no timing check")
            per[c] = row
            continue
        s = [p["sigma"][c] for p in parts]
        o_s, o_m, o_e = vals
        tol = max(agreement_tolerance(s[0], s[1]), agreement_tolerance(s[1], s[2]),
                  agreement_tolerance(s[0], s[2]))
        d1, d2 = o_m - o_s, o_e - o_m
        big, small = max(abs(d1), abs(d2)), min(abs(d1), abs(d2))
        centres = [0.5 * (p["range"][0] + p["range"][1]) for p in parts]
        three = f"{o_s:+.3f} / {o_m:+.3f} / {o_e:+.3f} at the start / middle / end"
        if max(vals) - min(vals) <= tol:
            row.update(pattern="steady", verdict="good", why=f"steady: {three}")
            per[c] = row
            continue
        if np.sign(d1) == np.sign(d2) and small >= EVEN_SHARE * big:
            shape = ("drift", (o_e - o_s) / max(centres[2] - centres[0], 1.0), None)
        elif small < EVEN_SHARE * big and big >= JUMP_MIN_FRAMES:   # (I353) a 0.13-frame step is no dropped frame
            k = 0 if abs(d1) >= abs(d2) else 1
            shape = ("jump", d1 if k == 0 else d2, (centres[k], centres[k + 1]))
        else:
            shape = None
        if shape is not None:
            if fine is None:
                fine = _part_estimates(rv, cal, a, b, frames, FINE_WINDOWS, part(0.5, 1.0), cancelled)
            pts = [(0.5 * (p["range"][0] + p["range"][1]), float(p["offsets"][c])) for p in fine if _usable(p, c)]
            if len(pts) >= FINE_WINDOWS - 1:
                again = _line_or_step([q[0] for q in pts], [q[1] for q in pts])
                if again is not None:
                    shape = again
                    row["fine"] = pts
        if shape is None:
            row.update(pattern="unsteady", verdict="ok",
                       why=f"changes between the windows ({three}) without a clear drift or step: noisy "
                           "tracks or a wand that barely moves in one window")
        elif shape[0] == "drift":
            rate = float(shape[1])                                   # frames per reference frame
            span = max(centres[2] - centres[0], 1.0)
            every = abs(1.0 / rate) if rate else float("inf")
            row.update(pattern="drift", verdict="poor", drift_per_frame=rate,
                       why=(f"drifts steadily ({three}): {rate * span:+.2f} frame over {span:.0f} frames, one "
                            f"frame every {every:,.0f}. Its real frame rate differs from the clock camera's: "
                            "no one offset fits the whole recording; set its true rate, or sync each trial on "
                            "its own"))
        else:
            step, (x0, x1) = float(shape[1]), shape[2]
            row.update(pattern="jump", verdict="poor", jump=step, jump_between=(int(x0), int(x1)),
                       why=(f"jumps {step:+.2f} frame ({three}): a dropped or repeated frame somewhere between "
                            f"reference frames {x0:.0f} and {x1:.0f}. No one offset fits the whole recording"))
        per[c] = row
    worst = [r["verdict"] for r in per.values()]
    verdict = "poor" if "poor" in worst else ("ok" if "ok" in worst or not worst else "good")
    return {"windows": [{"range": p["range"], "n_frames": p["n_frames"]} for p in parts],
            "per_camera": per, "verdict": verdict}


def _wand_lengths(prep, uv):
    xyz, _, _, _ = prep.triangulate(uv, 2)                                 # (T, 2, 3)
    return np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1), xyz


def _speed_check(rv, cal, a: str, b: str, frames) -> dict:
    """Wand-length error against wand speed, frame by frame (module docstring), run order: the median
    length error in the fastest quarter of the wand frames against the slowest quarter. A timing error
    makes the fast frames worse; noise does not care. Returns both medians (% of the wand length), their
    "ratio", a one-sided Mann-Whitney p-value, a verdict good / ok / poor ("sync suspect"), the camera
    most implicated (leave-one-camera-out, three cameras or more), the frame where the wand moves
    fastest, and a sentence. A SCREEN, not a measurement (the measurement is `refine`)."""
    names = list(rv.names)
    C = rv.n_views
    frames = np.sort(np.asarray(frames, np.int64))
    if len(frames) > SPEED_MAX_POINTS:
        frames = frames[np.round(np.linspace(0, len(frames) - 1, SPEED_MAX_POINTS)).astype(int)]
    prep = _calib._Prepared(rv.sessions, cal, [a, b])
    uv = prep.samples(rv.rates, rv.offsets, frames)                        # (T, 2, C, 2)
    ln, _ = _wand_lengths(prep, uv)
    _, xm = _wand_lengths(prep, prep.samples(rv.rates, rv.offsets, frames - 1))
    _, xp = _wand_lengths(prep, prep.samples(rv.rates, rv.offsets, frames + 1))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sp = np.nanmax(np.linalg.norm(xp - xm, axis=2) / 2.0, axis=1)     # fastest end, units / frame
    out = {"n_frames": 0, "verdict": "ok", "ratio": float("nan"), "p": float("nan"), "implicated": None,
           "loo_ratio": {}, "fastest_frame": None, "unit": cal.unit}
    ok = np.isfinite(ln) & np.isfinite(sp)
    if ok.sum() < MIN_CHECK_FRAMES:
        out["why"] = (f"Only {int(ok.sum())} wand frames with a speed: too few to see whether the wand length "
                      "depends on its speed.")
        return out
    x = sp[ok]
    p25, p75 = np.percentile(x, [25, 75])
    slow, fast = x <= p25, x >= p75

    def compare(lengths):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            L = float(np.nanmedian(lengths))
            dev = 100.0 * np.abs(lengths - L) / L
            ms, mf = float(np.nanmedian(dev[slow])), float(np.nanmedian(dev[fast]))
        return ms, mf, (mf / ms if ms > 0 else float("inf"))

    ms, mf, ratio = compare(ln[ok])
    from scipy.stats import mannwhitneyu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dev = 100.0 * np.abs(ln[ok] - np.median(ln[ok])) / np.median(ln[ok])
        pval = float(mannwhitneyu(dev[fast], dev[slow], alternative="greater").pvalue)
    fi = np.nonzero(ok)[0][int(np.argmax(x))]
    out.update(n_frames=int(ok.sum()), slow_pct=ms, fast_pct=mf, ratio=float(ratio), p=pval,
               speed_slow=float(p25), speed_fast=float(p75), fastest_frame=int(frames[fi]))
    narrow = not (p75 >= SPEED_RANGE_MIN * max(p25, 1e-12))
    clear = pval < SPEED_P
    if narrow:
        verdict = "ok"
    elif clear and ratio >= SPEED_SUSPECT_RATIO:
        verdict = "poor"
    elif clear and ratio >= SPEED_OK_RATIO:
        verdict = "ok"
    else:
        verdict = "good"
    out["verdict"] = verdict
    if C >= 3 and verdict != "good" and not narrow:
        for c in range(C):
            u = uv.copy()
            u[:, :, c] = np.nan
            lc, _ = _wand_lengths(prep, u)
            if np.isfinite(lc[ok]).sum() < MIN_CHECK_FRAMES:
                continue
            out["loo_ratio"][c] = float(compare(lc[ok])[2])
        if out["loo_ratio"]:
            c_best = min(out["loo_ratio"], key=out["loo_ratio"].get)
            if out["loo_ratio"][c_best] - 1.0 <= LOO_SHARE * (ratio - 1.0):
                out["implicated"] = int(c_best)
    unit = "wand lengths" if cal.unit == "wand" else cal.unit
    if narrow:
        why = (f"The wand moved at nearly one speed all the time ({p25:.3g}-{p75:.3g} {unit} per frame): this "
               "check needs slow AND fast moments to see a timing error, so it cannot say anything here.")
    else:
        why = (f"In the fastest quarter of the wand frames its length is off by {mf:.2f} % (median), in the "
               f"slowest quarter by {ms:.2f} %: {ratio:.1f} times as much. ")
        if verdict == "good":
            why += "No sign of a timing error (a wrong offset makes the fast frames clearly worse)."
        else:
            why += ("That is what a camera whose timing is off does: SYNC SUSPECT. " if verdict == "poor" else
                    "The timing may be slightly off (or a few wand frames are bad). ")
            if out["implicated"] is not None:
                c = out["implicated"]
                why += (f"Leaving {names[c]} out brings it to {out['loo_ratio'][c]:.1f} times: {names[c]} is the "
                        "camera most implicated.")
            elif C >= 3:
                why += "No single camera accounts for it (leaving each out in turn)."
            else:
                why += "With two cameras the check cannot tell which one."
    out["why"] = why
    return out


# ------------------------------------------------------------------- the other cameras' prediction
def _prediction(rv, cal, a: str, b: str) -> dict:
    """Run order: each camera's wand ends triangulated from the OTHER cameras of the run - where they
    agree, the diamond's own rule (residual within the 'ok' band of their pictures) - and projected into
    it: how far from where it saw them (its raw pixels). {per_camera: {median_px, p90_px, n, verdict},
    verdict, why}."""
    C = rv.n_views
    per = {k: {"median_px": float("nan"), "p90_px": float("nan"), "n": 0, "verdict": "none"} for k in range(C)}
    out = {"per_camera": per, "verdict": "none"}
    if C < 3:
        out["why"] = ("The other cameras can predict a camera's wand ends only with 3 cameras or more (two to "
                      "triangulate, one to check).")
        return out
    uv, _frames_ = wd.collect_wand(rv, a, b, max_frames=10 ** 9)          # (F, C, 2, 2) raw px
    if not len(uv):
        out["why"] = "No wand frame is seen by two cameras: nothing to predict."
        return out
    raw = np.concatenate([uv[:, :, 0], uv[:, :, 1]], axis=0)              # (2F, C, 2): both ends
    cams = [cal.camera(k) for k in range(C)]
    und = np.full_like(raw, np.nan)
    for k, cam in enumerate(cams):
        okk = np.isfinite(raw[:, k]).all(axis=1)
        if cam is not None and okk.any():
            und[okk, k] = cam.to_calib_frame(raw[okk, k])
    coefs = np.array([cam.coefs for cam in cams])
    sizes = [(s.width or 1920, s.height or 0) for s in rv.sessions]
    for k, cam in enumerate(cams):
        own = np.isfinite(raw[:, k]).all(axis=1)
        if cam is None or not own.any():
            continue
        u = und.copy()
        u[:, k] = np.nan
        xyz, res, ncams, _ = _calib.triangulate_batch(coefs, u, 2)
        others = [j for j in range(C) if j != k]
        ok_band = max(_calib.residual_bands(*sizes[j])[1] for j in others)
        use = own & (ncams >= 2) & np.isfinite(res) & (res <= ok_band)
        if not use.any():
            continue
        pred = cam.project(xyz[use])
        e = np.hypot(*(pred - raw[use, k]).T)
        e = e[np.isfinite(e)]
        if not len(e):
            continue
        good, okb = _calib.residual_bands(*sizes[k])
        med = float(np.median(e))
        per[k].update(median_px=med, p90_px=float(np.percentile(e, 90)), n=int(len(e)),
                      verdict=("none" if len(e) < PREDICT_MIN else
                               "good" if med <= good else "ok" if med <= okb else "poor"))
    words = [r["verdict"] for r in per.values() if r["verdict"] != "none"]
    out["verdict"] = "poor" if "poor" in words else "ok" if "ok" in words else ("good" if words else "none")
    lst = ", ".join(f"{rv.names[k]} {r['median_px']:.1f} px ({r['verdict']})" for k, r in per.items()
                    if r["verdict"] != "none")
    out["why"] = (f"Each camera's wand ends as the other cameras put them (median distance from where the camera "
                  f"saw them, in its own pixels): {lst}. A camera clearly worse than the others is usually one "
                  "whose offset is off - Refine offsets with the wand." if lst else
                  "Too few wand ends seen by three cameras to predict any camera.")
    return out


def prediction_error(project, cal, a: str, b: str, views=None, offsets=None) -> dict:
    """`_prediction` for the run's cameras `views` at `offsets`, by PROJECT camera, strict JSON."""
    rv = _run(project, views, offsets)
    p = _prediction(rv, cal, a, b)
    p["per_camera"] = _by_project(p["per_camera"], rv.views)
    return json_safe(p)


# ------------------------------------------------------------------------------- the checks together
def _assemble(rv, speed, est, rep, windows, prediction, offsets_run) -> dict:
    """The timing checks of one calibration, run order: the speed screen, how far one fixed estimate
    would move each offset ("off_by", an UNDER-estimate), the window check and the prediction; a window
    check made at offsets the wand says are off by more than OFF_BY_CAVEAT frame is capped at ok with a
    caveat (a calibration fitted at wrong offsets bends differently in each stretch: a drift that is not
    there)."""
    pv = rep.get("per_view", {})
    windows = dict(windows)                 # the caveat below is this report's, never the caller's dict
    out = {"speed": speed, "windows": windows, "prediction": prediction, "estimate": rep,
           "off_by": {c: float(est[c] - offsets_run[c]) for c in range(1, rv.n_views)
                      if pv.get(c) in ("sharp", "weak")}}
    worst_off = max((abs(v) for v in out["off_by"].values()), default=0.0)
    if worst_off > OFF_BY_CAVEAT and windows["verdict"] == "poor":
        windows["verdict"] = "ok"
        windows["caveat"] = (
            f"Measured at offsets the wand says are off by up to {worst_off:.2f} frame: a calibration made "
            "with wrong offsets bends differently in each stretch of the recording, which can look like a "
            "drift or a jump. Refine the offsets first, then read this check again.")
    vs = [speed["verdict"], windows["verdict"]]
    out["verdict"] = "poor" if "poor" in vs else ("ok" if "ok" in vs else "good")
    return out


def _to_project(d: dict, views) -> dict:
    """A run-order checks dict (`_assemble`'s) by PROJECT camera."""
    d = dict(d)
    if d.get("off_by"):
        d["off_by"] = _by_project(d["off_by"], views)
    if d.get("windows"):
        w = dict(d["windows"])
        w["per_camera"] = _by_project(w.get("per_camera", {}), views)
        d["windows"] = w
    if d.get("speed"):
        s = dict(d["speed"])
        if s.get("implicated") is not None:
            s["implicated"] = int(views[s["implicated"]])
        s["loo_ratio"] = _by_project(s.get("loo_ratio", {}), views)
        d["speed"] = s
    if d.get("prediction"):
        p = dict(d["prediction"])
        p["per_camera"] = _by_project(p.get("per_camera", {}), views)
        d["prediction"] = p
    if d.get("estimate"):
        e = dict(d["estimate"])
        for key in ("sharpness", "per_view", "own_cells"):
            if isinstance(e.get(key), dict):
                e[key] = _by_project(e[key], views)
        d["estimate"] = e
    d["views"] = list(views)
    return d


def _diagnose(rv, cal, a: str, b: str, progress=None, cancelled=None) -> dict:
    frames = _frames(rv, a, b)
    out = {"n_frames": int(len(frames)), "offsets": list(rv.offsets), "speed": None, "windows": None,
           "off_by": {}, "prediction": _prediction(rv, cal, a, b)}
    if len(frames) < MIN_CHECK_FRAMES:
        out.update(verdict="ok", why=f"only {len(frames)} wand frames: too few for the timing checks")
        return out

    def part(lo, hi):
        return None if progress is None else (lambda f: progress(lo + (hi - lo) * f))

    speed = _speed_check(rv, cal, a, b, frames)
    est, rep = _estimate(rv, cal, a, b, (frames.min(), frames.max()),
                         progress=None if progress is None else (lambda d, t: progress(0.25 * d / max(t, 1))),
                         cancelled=cancelled)
    if rep.get("verdict") == "cancelled":
        out.update(cancelled=True, verdict="ok", why="cancelled")
        return out
    try:
        windows = _window_check(rv, cal, a, b, frames, part(0.25, 1.0), cancelled)
    except _Cancelled:
        out.update(cancelled=True, verdict="ok", why="cancelled")
        return out
    out.update(_assemble(rv, speed, est, rep, windows, out["prediction"], rv.offsets))
    out["n_frames"] = int(len(frames))
    return out


def diagnose(project, cal, a: str, b: str, offsets=None, views=None, progress=None, cancelled=None) -> dict:
    """The timing checks shown with every wand calibration, without refining, the run's calibration
    `cal` fixed, at `offsets` (one per project camera; None = the project's): the prediction, the speed
    screen, one `estimate` over all the wand frames and the window check. None of it changes anything.
    By PROJECT camera, strict JSON."""
    rv = _run(project, views, offsets)
    return json_safe(_to_project(_diagnose(rv, cal, a, b, progress, cancelled), rv.views))


# -------------------------------------------------------------------------------------- the loop
def _camera_fit(res, c: int) -> float:
    """Median reprojection error (px) of run camera `c`'s wand observations, the rejected ones included
    (a median: a few bad clicks do not move it)."""
    e = np.asarray(res.reproj_obs["wand"], np.float64)[:, c]
    e = e[np.isfinite(e)]
    return float(np.median(e)) if len(e) else float("nan")


def _image_speed(rv, a: str, b: str) -> np.ndarray:
    """(C,) each run camera's median image speed of the wand ends, px per LOCAL frame, over the wand
    frames (NaN without any)."""
    frames = _frames(rv, a, b)
    out = np.full(rv.n_views, np.nan)
    if not len(frames):
        return out
    t = frames.astype(np.float64)
    v = []
    for nm in (a, b):
        p0, p1 = wd._sample_name(rv, nm, t), wd._sample_name(rv, nm, t + 1)      # (M, C, 2)
        v.append(np.linalg.norm(p1 - p0, axis=2))
    v = np.concatenate(v, axis=0) / np.asarray(rv.rates, np.float64)[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out[:] = np.nanmedian(v, axis=0)
    return out


def _joint(calibrate, full, views, res, cams, speed, names, note=None, stop=None) -> dict:
    """{run camera: share}: moving run camera c's offset `PROFILE_STEP` frame either way shifts its wand
    ends by about PROFILE_STEP x `speed[c]` px. If nothing absorbed that, its own fit (`_camera_fit`,
    after calibrating AGAIN) would get worse by about as much; the share is how much of it really shows
    (the smaller of the two sides). The fixed-calibration sharpness cannot see a timing the calibration
    absorbs whole - a wand always moved the same way at the same speed shifts a camera's points by one
    constant vector, which a camera moved a little explains exactly; this one can. A shifted rig that no
    longer solves counts as sharp."""
    out = {}
    cams = list(cams)
    for i, c in enumerate(cams):
        base = _camera_fit(res, c)
        expect = PROFILE_STEP * float(speed[c]) if np.isfinite(speed[c]) else float("nan")
        shares = []
        for j, d in enumerate((-PROFILE_STEP, PROFILE_STEP)):
            if stop is not None and stop():
                raise _Cancelled()
            if note is not None:
                note((2 * i + j) / max(2 * len(cams), 1),
                     f"Joint check: {names[c]} shifted {d:+.1f} frame, calibrating again")
            o = list(full)
            o[views[c]] = o[views[c]] + d
            try:
                r = calibrate(o, hint=res)
            except Exception:       # noqa: BLE001 -- a shifted rig that no longer solves: a sharp minimum
                if stop is not None and stop():
                    raise _Cancelled()
                shares.append(float("inf"))
                continue
            if stop is not None and stop():
                raise _Cancelled()
            fit = _camera_fit(r, c)
            if np.isfinite(fit) and np.isfinite(base) and expect > 0:
                shares.append((fit - base) / expect)
        out[c] = float(min(shares)) if shares else float("nan")
    return out


def _sharp_word(share: float) -> str:
    """The joint check's bands (`JOINT_SHARP` / `JOINT_FLAT`); NaN = nothing to judge = flat."""
    if share != share:
        return "flat"
    return "sharp" if share >= JOINT_SHARP else ("weak" if share >= JOINT_FLAT else "flat")


def refine(project, a: str, b: str, calibrate, offsets=None, views=None, progress=None, cancelled=None) -> dict:
    """Alternate `calibrate(offsets) -> WandResult` and the robust `estimate` until no offset moves by
    more than `ROUND_TOL` frame or `MAX_ROUNDS` rounds; then, on the final calibration (fitted at exactly
    the suggested offsets), the joint sharpness, the split-half check, the window check, the speed screen
    and the other cameras' prediction before and after. Never changes the project: returns the
    suggestion.

    `calibrate(offsets, hint=None)`: `offsets` one per PROJECT camera; the run's cameras `views` (the
    ticks; None = all) in that order; `hint` a WandResult to start from (the joint check only). Run
    camera 0 is the clock: it never moves and has no row.

    Result (project camera numbers): "before" / "after" offsets (whole project lists), "res_before" /
    "res_after" (the run's WandResults at each; "res_after" is fitted at exactly "after"), "views",
    "rounds" [{offsets, estimate, moved, wand_score_pct, rmse_px}], "converged", "settled",
    "last_move", "estimate" (the final estimate's report), "joint" {camera: share}, "split",
    "windows", "rows" {camera: {old, new, sharpness, joint, joint_rise, split, window, verdict, why,
    tick, reproj_before, reproj_after, pred_before, pred_after}}, "prediction_before" / "_after",
    "timing_after" (the checks of the final calibration, `diagnose`'s shape, strict JSON), an overall
    "verdict" + "why". `cancelled()` -> {"cancelled": True}."""
    views = list(range(project.n_views)) if views is None else [int(v) for v in views]
    C = len(views)
    full = list(project.offsets) if offsets is None else [float(o) for o in offsets]
    before = list(full)

    def note(frac, msg):
        if progress is not None:
            progress(float(frac), str(msg))

    def stop():
        return cancelled is not None and cancelled()

    span = 0.50 / (MAX_ROUNDS + 1)
    rounds, res_before, res, rep, est = [], None, None, {}, None
    converged = False
    try:
        for k in range(MAX_ROUNDS):
            note(k * span, f"Round {k + 1}: calibrating at the current offsets")
            res = calibrate(full)
            if stop():
                return {"cancelled": True}
            if res_before is None:
                res_before = res
            rv = _run(project, views, full)
            frames = _frames(rv, a, b)
            if len(frames) < 2:
                break
            note(k * span + 0.6 * span, f"Round {k + 1}: estimating the offsets with that calibration fixed")
            est, rep = _estimate(rv, _cal(res), a, b, (frames.min(), frames.max()), cancelled=cancelled)
            if rep.get("verdict") == "cancelled":
                return {"cancelled": True}
            moved = float(max((abs(est[c] - rv.offsets[c]) for c in range(1, C)), default=0.0))
            rounds.append({"offsets": list(full), "estimate": _full(full, views, est), "moved": moved,
                           "wand_score_pct": float(res.report.get("wand_score_pct", float("nan"))),
                           "rmse_px": float(res.report.get("reproj_rmse_all", float("nan")))})
            if moved < ROUND_TOL:
                converged = True
                break
            full = _full(full, views, [rv.offsets[0]] + [float(est[c]) for c in range(1, C)])   # the clock stays
        if res is None:
            return {"cancelled": True}
        last_move = rounds[-1]["moved"] if rounds else float("nan")
        rv = _run(project, views, full)
        if not converged and rounds:
            note(MAX_ROUNDS * span, "Calibrating at the last offsets found")
            res = calibrate(full)               # the result must be fitted at exactly the suggested offsets
            if stop():
                return {"cancelled": True}
            # the rows' sharpness and the settled test from the FINAL calibration, not the one before it
            frames = _frames(rv, a, b)
            if len(frames) >= 2:
                note(MAX_ROUNDS * span + 0.5 * span, "Estimating the offsets once more with the final calibration")
                est, rep = _estimate(rv, _cal(res), a, b, (frames.min(), frames.max()), cancelled=cancelled)
                if rep.get("verdict") == "cancelled":
                    return {"cancelled": True}
                last_move = float(max((abs(est[c] - rv.offsets[c]) for c in range(1, C)), default=0.0))
        per_view = rep.get("per_view", {})
        timed = [c for c in range(1, C) if per_view.get(c, "none") != "none"]
        speed_px = _image_speed(rv, a, b)
        joint = _joint(calibrate, full, views, res, timed, speed_px, rv.names,
                       note=lambda f, m: note(0.50 + 0.25 * f, m), stop=stop)
        frames = _frames(rv, a, b)
        note(0.75, "Checking the two halves of the wand frames")
        split = _split_half(rv, _cal(res), a, b, frames,
                            lambda f: note(0.75 + 0.08 * f, "Checking the two halves of the wand frames"), cancelled)
        note(0.83, "Checking the start, middle and end of the wand frames")
        windows = _window_check(rv, _cal(res), a, b, frames,
                                lambda f: note(0.83 + 0.12 * f, "Checking the start, middle and end"), cancelled)
        note(0.95, "Comparing each camera with the other cameras' prediction")
        rv_before = _run(project, views, before)
        pred_before = _prediction(rv_before, _cal(res_before), a, b)
        pred_after = _prediction(rv, _cal(res), a, b)
        speed_after = _speed_check(rv, _cal(res), a, b, frames) if len(frames) else {"verdict": "ok", "why": ""}
    except _Cancelled:
        return {"cancelled": True}
    # the loop stops at ROUND_TOL; a last move under SETTLED_TOL is the estimator's own jitter (the
    # calibration changes a little every round), far below what the checks can resolve
    settled = converged or (np.isfinite(last_move) and last_move < SETTLED_TOL)
    names = rv.names
    rows = {}
    for c in range(1, C):
        sharp = per_view.get(c, "none")
        jw = _sharp_word(joint[c]) if c in joint else "none"
        sp = split["per_camera"].get(c, {"status": "unknown", "why": ""})
        win = windows["per_camera"].get(c, {"pattern": "unknown", "why": ""})
        if sharp == "none":
            v, why = "poor", "shares no wand frame with another camera: its offset is left as it was"
        elif sharp == "flat":
            v, why = "poor", "the wand does not move enough in it to time it: do not apply"
        elif jw == "flat":
            v, why = "poor", ("shifting it half a frame and calibrating again fits the wand just as well: the "
                              "wand's motion cannot tell this camera's timing from where it stands (a wand moved "
                              "the same way at the same speed). Do not apply; wave the wand in every direction, "
                              "slow and fast")
        elif sp["status"] == "disagree":
            v, why = "poor", sp["why"] + " (see the window check): do not apply one offset"
        elif win["pattern"] in ("drift", "jump"):
            v, why = "poor", "the window check: it " + win["why"] + ". Do not apply one offset"
        elif sharp == "sharp" and jw == "sharp" and sp["status"] == "agree":
            v, why = "good", "a clear minimum, and " + sp["why"]
        elif sharp == "sharp" and jw == "sharp":
            v, why = "ok", "a clear minimum, but " + sp["why"]
        else:
            v, why = "ok", "a shallow minimum (the wand's motion barely times this camera); " + sp["why"]
        if v == "good" and not settled:
            v, why = "ok", why + (f"; the rounds had not settled after {MAX_ROUNDS} (the last estimate still moved "
                                  f"an offset by {last_move:.2f} frame)")
        pb, pa = pred_before["per_camera"][c], pred_after["per_camera"][c]
        rows[views[c]] = {
            "old": float(before[views[c]]), "new": float(full[views[c]]), "sharpness": sharp, "joint": jw,
            "joint_rise": float(joint.get(c, float("nan"))), "split": sp["status"], "window": win["pattern"],
            "verdict": v, "why": why, "tick": v == "good",
            "reproj_before": float(res_before.reproj_cam[c]), "reproj_after": float(res.reproj_cam[c]),
            "pred_before": float(pb["median_px"]), "pred_after": float(pa["median_px"]),
            "pred_verdict_before": pb["verdict"], "pred_verdict_after": pa["verdict"]}
    sb = float(res_before.report.get("wand_score_pct", float("nan")))
    sa = float(res.report.get("wand_score_pct", float("nan")))
    vs = [r["verdict"] for r in rows.values()]
    verdict = "poor" if "poor" in vs else ("ok" if "ok" in vs else "good")
    # a refinement that fits the wand clearly WORSE is not one (10 % and 0.02 points: noise moves it less)
    worse = bool(np.isfinite(sb) and np.isfinite(sa) and sa > 1.10 * sb + 0.02)
    if worse:
        verdict = "poor"
        for r in rows.values():
            r["tick"] = False
    if verdict == "good":
        why = (f"Every camera's offset is clearly determined by the wand, and the two halves of the recording "
               f"agree. The wand score goes from {sb:.2f} % to {sa:.2f} %.")
    elif worse:
        why = (f"The refined offsets fit the wand WORSE (wand score {sb:.2f} % -> {sa:.2f} %): keep the current "
               "offsets and read the rows and the window check.")
    else:
        bad = [names[c] for c in range(1, C) if rows[views[c]]["verdict"] != "good"]
        why = (f"Not every camera is clearly timed by the wand ({', '.join(bad)}: see its row). The wand score "
               f"goes from {sb:.2f} % to {sa:.2f} %. Apply only the rows marked good.")
    timing_after = _to_project(_assemble(rv, speed_after, est if est is not None else list(rv.offsets), rep,
                                         windows, pred_after, rv.offsets), views)
    timing_after.update(n_frames=int(len(frames)), offsets=list(rv.offsets))
    split_p = {"parts": split["parts"], "per_camera": _by_project(split["per_camera"], views)}
    windows_p = dict(windows)
    windows_p["per_camera"] = _by_project(windows["per_camera"], views)
    est_rep = dict(rep)
    for key in ("sharpness", "per_view", "own_cells"):
        if isinstance(est_rep.get(key), dict):
            est_rep[key] = _by_project(est_rep[key], views)
    return {"cancelled": False, "views": views, "clock": int(views[0]), "before": before, "after": list(full),
            "res_before": res_before, "res_after": res, "rounds": rounds, "converged": converged,
            "settled": bool(settled), "last_move": float(last_move), "estimate": est_rep,
            "joint": _by_project(joint, views), "split": split_p, "windows": windows_p, "rows": rows,
            "prediction_before": json_safe(_to_project({"prediction": pred_before}, views)["prediction"]),
            "prediction_after": json_safe(_to_project({"prediction": pred_after}, views)["prediction"]),
            "timing_after": json_safe(timing_after), "verdict": verdict, "why": why,
            "wand_score_before": sb, "wand_score_after": sa}
