"""Automatic epipolar-constrained re-tracking: when a camera's disagreement
band lights up on the timeline, offer to re-seed that camera's landmark on the
other cameras' rays and re-track from there, with a verdict before and after.

The pure half, no Qt: from the last reconstruction's per-camera reprojection
errors (`Reconstruction.per_cam`, (T, N, C)), find the STRETCHES where one
camera disagrees with the rest (the magenta band the timeline paints), and
for each stretch compute where the other cameras' rays say the landmark is
on its first frame - the same construction as the hand "Snap to the other
cameras' rays". The app then moves the landmark there (hand-placed, one undo
step per camera) and re-tracks it through the stretch with the ordinary
worker, reconstructs again and reports before / after.

Why the first frame of the stretch: the tracker is forward-only, and a
landmark that slid keeps sliding; re-seeding it where the other cameras
agree it must be and letting the appearance model follow from there is the
user-driven fix automated. Why only stretches (>= `MIN_RUN` frames): a
single disagreeing frame is a flicker, not a slide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinetrace.calib import epipolar_polyline, intersect_polylines, working_probe

MIN_RUN = 3             # frames: shorter runs are noise, not a slide
GAP_JOIN = 2            # runs separated by <= this many clean frames are one stretch


@dataclass
class Stretch:
    view: int                   # the camera that disagrees
    name: str                   # landmark
    t0: int                     # reference frames, inclusive
    t1: int
    local0: int                 # the camera's own frames
    local1: int
    median_px: float            # this camera's reprojection error over the stretch
    n: int = 0                  # cells over the threshold
    target: np.ndarray | None = None     # where the rays put it on local0 (native px), or None
    n_rays: int = 0             # other cameras whose ray was used
    full_px: float = float("nan")        # triangulation residual of ALL cameras over the stretch (median)
    rest_px: float = float("nan")        # ... of the OTHER cameras alone: small = they agree without this one
    reason: str = ""            # why a stretch is skipped (target None)
    note: str = ""              # what was left out of a stretch that is re-tracked (its start moved)


def disagreeing_stretches(project, threshold_px: list[float] | float, min_run: int = MIN_RUN) -> list[Stretch]:
    """Per camera and landmark, the runs of reference frames where that
    camera's reprojection error exceeds its threshold (px, one per view or one
    for all), joined across gaps of <= GAP_JOIN frames, at least `min_run` long."""
    r = project.reconstruction
    if r is None or r.per_cam is None:
        return []
    T, N, C = r.per_cam.shape
    thr = list(threshold_px) if not np.isscalar(threshold_px) else [float(threshold_px)] * C
    out: list[Stretch] = []
    for c in range(min(C, project.n_views)):
        s = project.sessions[c]
        for j, nm in enumerate(r.names):
            if s.pid_by_name(nm) is None:
                continue
            e = r.per_cam[:, j, c]
            bad = np.isfinite(e) & (e > thr[c])
            if not bad.any():
                continue
            # runs of bad frames, small gaps joined
            idx = np.nonzero(bad)[0]
            groups: list[list[int]] = [[int(idx[0])]]
            for k in idx[1:]:
                if int(k) - groups[-1][-1] <= GAP_JOIN + 1:
                    groups[-1].append(int(k))
                else:
                    groups.append([int(k)])
            for g in groups:
                a, b = g[0], g[-1]
                if b - a + 1 < min_run:
                    continue
                seg = e[a:b + 1]
                l0 = project.local_index(c, r.t0 + a)          # (I258) map_frame's tie rule, not round()
                l1 = project.local_index(c, r.t0 + b)
                if not (0 <= l0 < s.n_frames):
                    continue
                l1 = int(min(max(l1, l0), s.n_frames - 1))
                out.append(Stretch(c, nm, r.t0 + a, r.t0 + b, l0, l1,
                                   float(np.nanmedian(seg)), int(np.isfinite(seg).sum() and (seg > thr[c]).sum())))
    out.sort(key=lambda st: (st.view, st.local0, st.name))
    return out


def edge_tolerance(width: float) -> float:
    """How far outside the picture (px) a ray crossing may land and still be
    taken as the border pixel: calibration noise on a landmark at the edge."""
    return max(3.0, 0.005 * float(width or 0))


def outside_by(xy, width: float, height: float) -> float:
    """Distance (px) from `xy` to the picture [0, w-1] x [0, h-1]; 0 inside."""
    dx = max(0.0, -float(xy[0]), float(xy[0]) - (float(width) - 1.0))
    dy = max(0.0, -float(xy[1]), float(xy[1]) - (float(height) - 1.0))
    return float(np.hypot(dx, dy))


def ray_target(project, view: int, name: str, local_frame: int, probe=None,
               why: list | None = None, *, margin: float | None = None, observations=None,
               info: dict | None = None) -> tuple[np.ndarray, int] | None:
    """Where the OTHER cameras' rays put landmark `name` in `view` at that
    view's `local_frame`: the nearest point on one camera's epipolar polyline,
    the least-squares crossing of several. None when no other camera has it
    there or there is no calibration - or when the crossing lies OUTSIDE this
    camera's picture by more than `edge_tolerance` (I9: out of the picture
    means no data; clipping it onto the border stored a hand placement there
    and tracked a border patch). `why`, when given, receives the reason for a
    None. `probe` = a 3D point in front of the cameras (the reconstruction's
    centroid at that instant, else `working_probe`).

    The hand "Snap to the other cameras' rays" is this function too (R14), with its own
    rules passed in: `margin` (the polylines' extra reach past the picture; None =
    `epipolar_polyline`'s default, 0 = cut exactly at the edges, what the guides draw),
    `observations` ([(camera, its calibration, raw (x, y)), ...], gathered by the caller at
    the fractional frame, I253, among the calibrated cameras only -- then two calibrated cameras
    are enough, where the default needs a calibration for EVERY camera of the project) and
    `info`, which receives `intersect_polylines`'s {"mode", ...} plus "n_lines" and, when the
    crossing is refused, "outside_px"."""
    p = project
    cal = p.calibration
    if cal is None:
        return None
    if observations is None and len(cal.cameras) != p.n_views:
        return None
    if observations is not None and not 0 <= view < min(len(cal.cameras), p.n_views):
        return None
    s = p.sessions[view]
    pid = s.pid_by_name(name)
    if pid is None:
        return None
    if probe is None:
        r = p.reconstruction
        t = p.reference_index(view, int(local_frame))        # (I258)
        if r is not None and r.t0 <= t < r.t0 + r.n_frames:
            row = r.xyz[t - r.t0]
            row = row[np.isfinite(row).all(axis=1)]
            probe = row.mean(axis=0) if len(row) else None
        if probe is None:
            probe = working_probe(cal)
    dst = cal.cameras[view]
    if observations is None:
        observations = []
        for c in range(p.n_views):
            if c == view:
                continue
            sc = p.sessions[c]
            j = sc.pid_by_name(name)
            if j is None:
                continue
            fc = p.map_frame(view, c, int(local_frame))
            if fc is None or not (0 <= fc < sc.n_frames) or not sc.exportable_at(fc, j):
                continue
            uv = sc.tracks[fc, j]
            if not np.isfinite(uv).all():
                continue
            observations.append((c, cal.cameras[c], uv))
    kw = {} if margin is None else {"margin": float(margin)}
    lines = []
    for _c, cam, uv in observations:
        try:
            pts = epipolar_polyline(cam, dst, uv, probe, **kw)
        except Exception:       # noqa: BLE001 - a degenerate pair must not stop the others
            continue
        if len(pts) >= 2:
            lines.append(np.asarray(pts, np.float64))
    if info is not None:
        info["n_lines"] = len(lines)
    if not lines:
        return None
    cur = s.tracks[int(local_frame), pid]
    near = cur if np.isfinite(cur).all() else np.array([s.width / 2.0, s.height / 2.0])
    target = intersect_polylines(lines, near, info)
    if target is None or not np.isfinite(target).all():
        return None
    off = outside_by(target, s.width, s.height)
    if off > edge_tolerance(s.width):
        if info is not None:
            info["outside_px"] = float(off)
        if why is not None:
            why.append(f"the other cameras put it {off:.0f} px outside this camera's picture there (out of the "
                       "picture = no data): clear its track on those frames instead of re-tracking")
        return None
    x = float(np.clip(target[0], 0, s.width - 1))
    y = float(np.clip(target[1], 0, s.height - 1))
    return np.array([x, y]), len(lines)


def _blame(project, stretches: list[Stretch], threshold_px) -> list[Stretch]:
    """Which camera slid? With three cameras a slide in ONE spreads the
    residual over all three (measured: 8-10 px in each), so the per-camera
    error alone cannot name it. Leave-one-out can: the culprit is the camera
    whose removal makes the REST agree. Stretches of one landmark over
    overlapping frames are grouped; in each group the camera with the
    smallest rest-residual is blamed if the rest agree (under the band and
    under half the full residual) and no other camera comes close; otherwise
    every stretch of the group is skipped with the reason."""
    from kinetrace.calib import _Prepared, triangulate_batch
    p = project
    r = p.reconstruction
    if r is None or p.calibration is None or not stretches:
        return stretches
    thr = list(threshold_px) if not np.isscalar(threshold_px) else [float(threshold_px)] * p.n_views
    try:
        prep = _Prepared(p.sessions, p.calibration, list(r.names))
    except ValueError:
        return stretches
    # group: same landmark, frame ranges overlapping by half the shorter one
    groups: list[list[Stretch]] = []
    for st in stretches:
        for g in groups:
            g0 = g[0]
            ov = min(st.t1, g0.t1) - max(st.t0, g0.t0) + 1
            if g0.name == st.name and ov >= 0.5 * min(st.t1 - st.t0 + 1, g0.t1 - g0.t0 + 1):
                g.append(st)
                break
        else:
            groups.append([st])
    out: list[Stretch] = []
    for g in groups:
        j = r.names.index(g[0].name)
        for st in g:
            t = np.arange(st.t0, st.t1 + 1)
            uv = prep.samples(p.rates, p.offsets, t)[:, j]              # (T, C, 2)
            _, res_full, _, _ = triangulate_batch(prep.coefs, uv)
            rest = uv.copy()
            rest[:, st.view] = np.nan
            _, res_rest, _, _ = triangulate_batch(prep.coefs, rest)
            st.full_px = float(np.nanmedian(res_full)) if np.isfinite(res_full).any() else float("nan")
            st.rest_px = float(np.nanmedian(res_rest)) if np.isfinite(res_rest).any() else float("nan")
        # (I8) rank CAMERAS, not stretches: a camera whose error dipped under the
        # band mid-slide has two stretches in the group, with near-equal rest
        # residuals - ranked per stretch it out-voted itself ("no single camera
        # stands out") or had its own second stretch skipped as "X slid here".
        by_view: dict[int, list[Stretch]] = {}
        for st in g:
            if np.isfinite(st.rest_px):
                by_view.setdefault(st.view, []).append(st)
        if not by_view:
            for st in g:
                st.reason = ("only two cameras see it: cannot tell which one slid" if p.n_views <= 2
                             else "the other cameras have no 3D there without this one")
            out.extend(g)
            continue
        vrest, vfull = {}, {}
        for v, sts in by_view.items():
            full = np.array([q.full_px for q in sts])
            vrest[v] = float(np.median([q.rest_px for q in sts]))
            vfull[v] = float(np.nanmedian(full)) if np.isfinite(full).any() else float("nan")
        order = sorted(vrest, key=vrest.get)
        bv = order[0]
        # (I228) as documented above: under the band AND under half the full residual
        # (it used to accept whichever of "half the full residual" and "half the band" was LARGER,
        # so the others could still disagree beyond the band and be called "agreeing")
        agree = vrest[bv] <= thr[bv] and (not np.isfinite(vfull[bv]) or vrest[bv] <= 0.5 * vfull[bv])
        clear = len(order) == 1 or vrest[order[1]] > 1.5 * vrest[bv]
        for st in g:
            if st.view == bv and agree and clear:
                continue
            if not agree:
                st.reason = (f"the other cameras still disagree without {p.name(bv)} ({vrest[bv]:.1f} px): "
                             "the calibration or the offsets, not one camera")
            elif not clear:
                st.reason = "no single camera stands out as the one that slid"
            else:
                st.reason = f"{p.name(bv)} is the camera that slid here"
        out.extend(g)
    out.sort(key=lambda st: (st.view, st.local0, st.name))
    return out


def _n_other_views(project, view: int, name: str, local_frame: int) -> int:
    """How many OTHER cameras have landmark `name` (exportable, finite) at the
    instant of `view`'s `local_frame` - the rays `ray_target` could cross."""
    p = project
    n = 0
    for c in range(p.n_views):
        if c == view:
            continue
        sc = p.sessions[c]
        j = sc.pid_by_name(name)
        fc = p.map_frame(view, c, int(local_frame)) if j is not None else None
        if fc is not None and 0 <= fc < sc.n_frames and sc.exportable_at(fc, j) and np.isfinite(sc.tracks[fc, j]).all():
            n += 1
    return n


def _start_on_two_rays(project, st: Stretch, threshold_px, min_run: int) -> bool:
    """Move the start of `st` to its first frame that TWO other cameras see.
    With one, its ray is a line and the 'target' is merely the point of it
    nearest the slid position (21 px off the truth in verify_3d_gui, where the
    third camera starts later): a re-seed there tracks the wrong spot. False
    (and `st.reason` set) when no such frame leaves `min_run` frames."""
    p = project
    r = p.reconstruction
    s = p.sessions[st.view]
    for t in range(st.t0, st.t1 + 1):
        lf = p.local_index(st.view, t)                     # (I258)
        if 0 <= lf < s.n_frames and _n_other_views(p, st.view, st.name, lf) >= 2:
            break
    else:
        st.reason = ("fewer than two other cameras see it in this stretch: one camera's ray is a line, not a point, "
                     "so there is no reliable spot to re-seed it - place it by hand")
        return False
    if t == st.t0:
        return True
    if st.t1 - t + 1 < min_run:
        st.reason = (f"two other cameras see it only on the last {st.t1 - t + 1} frame(s) of this stretch: one "
                     "camera's ray is a line, not a point - place it by hand")
        return False
    st.note = (f"frames {st.local0}-{max(st.local0, lf - 1)} left as they are: only one other camera sees it there, "
               "so there is no crossing of rays to re-seed on")
    st.t0, st.local0 = t, lf
    st.local1 = max(st.local1, lf)
    j = r.names.index(st.name) if st.name in r.names else None
    if j is not None and r.per_cam is not None:
        seg = r.per_cam[t - r.t0:st.t1 - r.t0 + 1, j, st.view]
        thr = float(threshold_px) if np.isscalar(threshold_px) else float(list(threshold_px)[st.view])
        if np.isfinite(seg).any():
            st.median_px = float(np.nanmedian(seg))
            st.n = int((seg[np.isfinite(seg)] > thr).sum())
    return True


def plan(project, threshold_px, min_run: int = MIN_RUN) -> list[Stretch]:
    """The stretches to re-track with their ray targets filled in. Stretches
    that cannot or should not be re-tracked are kept with `target None` and a
    `reason` (no other camera at the first frame; the leave-one-out test does
    not blame this camera; the other cameras put the landmark outside this
    camera's picture on the first frame or inside the stretch; fewer than two
    other cameras see it), so the report can say why. A stretch whose first
    frames only ONE other camera sees starts where two do (`note` says so)."""
    out = _blame(project, disagreeing_stretches(project, threshold_px, min_run), threshold_px)
    for st in out:
        if st.reason:
            continue
        if project.n_views >= 3 and not _start_on_two_rays(project, st, threshold_px, min_run):
            continue
        why: list[str] = []
        got = ray_target(project, st.view, st.name, st.local0, why=why)
        if got is None:
            st.reason = why[0] if why else "no other camera has it at the start of the stretch"
            continue
        # (I9) a landmark that LEAVES the picture during the stretch (its first frame on
        # the border) would be re-tracked as a border patch through frames that must be
        # blank: probe a few frames inside the stretch too
        for f in sorted({int(round(st.local0 + q * (st.local1 - st.local0))) for q in (0.25, 0.5, 0.75, 1.0)}
                        - {st.local0}):
            out_why: list[str] = []
            if ray_target(project, st.view, st.name, f, why=out_why) is None and out_why:
                st.reason = out_why[0].replace(" there ", f" at frame {f} ")
                break
        else:
            st.target, st.n_rays = got
    return out


def cells_summary(project, stretches: list[Stretch]) -> dict:
    """The disagreement of exactly the flagged cells in the CURRENT
    reconstruction: {n_cells, median_px, max_px, n_over} over the cells with a
    value, plus EVERY cell of the stretches in a fixed order (`cells`, NaN
    where there is no value now; `has2d`, the camera still has a position
    there; `views`, `local`, `lm` + `names` to name them) - call before and
    after so the verdict compares like with like, lost cells included (I7)."""
    p = project
    r = p.reconstruction
    cells, has2d, views, local, lm, names = [], [], [], [], [], []
    for st in stretches:
        t = np.arange(st.t0, st.t1 + 1)
        v = np.full(len(t), np.nan)
        j = r.names.index(st.name) if (r is not None and st.name in r.names) else None
        if j is not None and r.per_cam is not None and st.view < r.per_cam.shape[2]:
            k = t - r.t0
            inside = (k >= 0) & (k < r.n_frames)
            v[inside] = r.per_cam[k[inside], j, st.view]
        lf = np.asarray(p.local_index(st.view, t), dtype=int)       # (I258)
        s = p.sessions[st.view] if st.view < p.n_views else None
        pid = s.pid_by_name(st.name) if s is not None else None
        h = np.zeros(len(t), bool)
        if pid is not None:
            inb = (lf >= 0) & (lf < s.n_frames)
            h[inb] = s.exportable_at(lf[inb], pid) & np.isfinite(s.tracks[lf[inb], pid]).all(axis=1)
        if st.name not in names:
            names.append(st.name)
        cells.append(v)
        has2d.append(h)
        views.append(np.full(len(t), st.view))
        local.append(lf)
        lm.append(np.full(len(t), names.index(st.name)))
    out = {"n_cells": 0, "median_px": float("nan"), "max_px": float("nan"), "n_over": 0,
           "cells": np.zeros(0), "has2d": np.zeros(0, bool), "views": np.zeros(0, int), "local": np.zeros(0, int),
           "lm": np.zeros(0, int), "names": names}
    if not cells:
        return out
    c = np.concatenate(cells)
    out.update(cells=c, has2d=np.concatenate(has2d), views=np.concatenate(views), local=np.concatenate(local),
               lm=np.concatenate(lm))
    v = c[np.isfinite(c)]
    out.update(n_cells=int(len(v)), median_px=float(np.median(v)) if len(v) else float("nan"),
               max_px=float(v.max()) if len(v) else float("nan"))
    return out


def _where(summary: dict, mask: np.ndarray, project=None, limit: int = 4) -> str:
    """'head in camC: frames 40-69, ...' for the cells in `mask`."""
    views, local, lm, names = summary["views"][mask], summary["local"][mask], summary["lm"][mask], summary["names"]
    parts = []
    for key in sorted(set(zip(views.tolist(), lm.tolist()))):
        fr = np.sort(local[(views == key[0]) & (lm == key[1])])
        runs, a = [], int(fr[0])
        for x0, x1 in zip(fr[:-1], fr[1:]):
            if x1 - x0 > 1:
                runs.append((a, int(x0)))
                a = int(x1)
        runs.append((a, int(fr[-1])))
        cam = project.name(key[0]) if project is not None else f"camera {key[0] + 1}"
        parts.append(f"{names[key[1]]} in {cam}: frames " + ", ".join(f"{x}-{y}" if y > x else f"{x}" for x, y in runs))
    if len(parts) > limit:
        parts = parts[:limit] + [f"and {len(parts) - limit} more"]
    return "; ".join(parts)


def verdict(before: dict, after: dict, threshold_px, project=None) -> tuple[str, str]:
    """('better' | 'same' | 'worse', a sentence) from the flagged cells.

    (I7) Cells that had a value before and have none now are counted FIRST: a
    re-track job that auto-paused has its track cut from the fail frame, the
    cut cells dropped out of the median, and the few good frames left read as
    a big improvement - 30 of 40 frames deleted still said "BETTER ... Keep
    it". Any loss is 'worse', named, with the default on Undo. The medians
    compare only the cells with a value both times. `threshold_px` is one band
    for all, or one per view (each camera's own band, not camera 0's)."""
    thr = None if np.isscalar(threshold_px) else [float(x) for x in threshold_px]
    band = float(threshold_px) if thr is None else None
    cb, ca = before.get("cells"), after.get("cells")
    if cb is not None and ca is not None and len(cb) == len(ca) and len(cb):
        fb, fa = np.isfinite(cb), np.isfinite(ca)
        lost = fb & ~fa
        both = fb & fa
        if lost.any():
            gone = lost & ~np.asarray(after.get("has2d", np.zeros(len(ca), bool)), bool)
            what = (f"{int(lost.sum())} of {int(fb.sum())} frame(s) that had a position before have NONE now "
                    f"({_where(after, lost, project)})" if gone.sum() == lost.sum() else
                    f"{int(lost.sum())} of {int(fb.sum())} frame(s) no longer have 3D in this camera "
                    f"({_where(after, lost, project)}; {int(gone.sum())} of them lost their position)")
            rest = (f" The {int(both.sum())} frame(s) still tracked go from {np.median(cb[both]):.1f} px to "
                    f"{np.median(ca[both]):.1f} px (median)." if both.any() else "")
            return "worse", (f"Data lost: {what} - the tracker stopped or lost the landmark inside the stretch."
                             f"{rest} Undo, then click the landmark by hand on those frames (or re-track them).")
        if not both.any():
            return "same", "The re-tracked stretches could not be compared (no 3D there afterwards)."
        b, a = float(np.median(cb[both])), float(np.median(ca[both]))
        if thr is not None:
            views = np.asarray(before["views"])[both]
            per = np.array([thr[v] if v < len(thr) else thr[0] for v in views])
            under = float(np.median(ca[both] / per)) <= 1.0
            lo, hi = float(per.min()), float(per.max())
            band_txt = f"{lo:.1f} px band" if hi - lo < 1e-9 else f"cameras' own bands ({lo:.1f}-{hi:.1f} px)"
        else:
            under, band_txt = a <= band, f"{band:.1f} px band"
    else:
        b, a = before.get("median_px", float("nan")), after.get("median_px", float("nan"))
        if not np.isfinite(b) or not np.isfinite(a):
            return "same", "The re-tracked stretches could not be compared (no 3D there afterwards)."
        if after.get("n_cells", 0) < before.get("n_cells", 0):
            return "worse", (f"Data lost: {before.get('n_cells', 0) - after.get('n_cells', 0)} frame(s) of the "
                             "re-tracked stretches have no position any more. Undo, then click them by hand.")
        band = band if band is not None else float(min(thr))
        under, band_txt = a <= band, f"{band:.1f} px band"
    if under and a < 0.7 * b:
        return "better", (f"The re-tracked stretches now agree with the other cameras: {b:.1f} px -> {a:.1f} px "
                          f"(median), under the {band_txt}. Keep it.")
    if a < 0.7 * b:
        return "better", (f"Improved, {b:.1f} px -> {a:.1f} px (median), but still over the {band_txt}: "
                          "the landmark may need a hand click in this camera, or the calibration / sync is off.")
    if a > 1.3 * b:
        return "worse", (f"Worse: {b:.1f} px -> {a:.1f} px (median). The rays put the landmark on the wrong spot "
                         "(a bad calibration, a wrong offset, or the OTHER cameras are the ones that slid). Undo.")
    return "same", (f"No real change: {b:.1f} px -> {a:.1f} px (median). The disagreement is not a slide in this "
                    "camera; check the other cameras, the offsets and the calibration.")
