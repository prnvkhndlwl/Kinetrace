"""What the 3D picture of a wand calibration shows (G185, G186) - pure numpy, no Qt.

`CalibPicture` holds everything the 3D window draws for a calibration:

* each calibrated camera: its centre, a small pyramid pointing where it looks (the "pose") and its whole
  field of view out to a depth past the wand (`fov`): the picture's border pixels, lens-corrected, turned
  into rays (`calib.dlt_ray`). Only the border points the lens model round-trips are used
  (`calib.calib_frame_ok`): a fisheye model that runs away past its boards is cut, never extrapolated. A
  camera without a lens correction is drawn as a straight pinhole pyramid and said to be one;
* the wand: both ends of every frame, with the frame, the worse camera's error and a state (good / ok /
  poor / rejected). From a wand run (`from_wand_result`: the fit's own errors and rejected clicks) or
  recomputed from the project's two wand landmarks (`from_project`: a camera's error past the residual
  band counts as "disagrees");
* the space seen by at least N cameras (`coverage`): voxel centres projected through each camera (its lens
  model, in front of it, inside its picture) - the count a whole-picture silhouette would give
  `hull.carve`, without its per-pixel maps; its surface by `hull.surface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinetrace.calib import (Calibration, CameraCalibration, calib_frame_ok, dlt_project, dlt_ray, front_sign,
                             in_front, reconstruct, residual_bands, residual_scale, round_trip_tol, working_probe)

GOOD, OK, POOR, REJECTED = 0, 1, 2, 3
STATE_WORDS = {GOOD: "good", OK: "ok", POOR: "poor", REJECTED: "rejected by the fit"}
COVER_GOOD_PCT, COVER_OK_PCT = 90.0, 70.0      # share of the wand cloud inside the covered space
POSE_FRAC = 0.03                               # the pose pyramid's depth, of the scene (wand + cameras) size (G187)
FOV_REACH = 1.15                               # the field of view reaches this far past the farthest wand end


def _cover_word(pct: float) -> str:
    return "good" if pct >= COVER_GOOD_PCT else ("ok" if pct >= COVER_OK_PCT else "poor")


@dataclass
class Fov:
    """One camera's viewing pyramid cut by a flat far plane `depth` in front of it along its viewing axis (as
    Blender draws a camera): `border` (M, 3) world points of the picture's border loop,
    `raw` (M, 2) the border pixels they come from, `rays` the four corners as (centre, point) segments,
    `pinhole` = the camera has no lens correction, `side` (M,) the picture side each border point lies on
    (0 top, 1 right, 2 bottom, 3 left; G187: one translucent face per side)."""
    centre: np.ndarray
    border: np.ndarray
    raw: np.ndarray
    rays: list
    pinhole: bool
    axis: np.ndarray | None = None       # the viewing axis (unit) the far plane is perpendicular to
    depth: float = 0.0
    side: np.ndarray | None = None


def _border_loop(w: int, h: int, n: int) -> tuple[np.ndarray, list[int]]:
    """Raw pixels around the picture's edge (pixel edges at -0.5 / w - 0.5), clockwise from the top-left
    corner, `n` per side; and the indices of the four corners."""
    x0, y0, x1, y1 = -0.5, -0.5, w - 0.5, h - 0.5
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    sides = [np.column_stack([x0 + t * (x1 - x0), np.full(n, y0)]),
             np.column_stack([np.full(n, x1), y0 + t * (y1 - y0)]),
             np.column_stack([x1 - t * (x1 - x0), np.full(n, y1)]),
             np.column_stack([np.full(n, x0), y1 - t * (y1 - y0)])]
    return np.vstack(sides), [0, n, 2 * n, 3 * n]


def fov(cam: CameraCalibration, depth: float, probe, n: int = 24) -> Fov:
    """The field of view of `cam` out to `depth` (world units) along each border ray."""
    raw, corners = _border_loop(int(cam.width), int(cam.height), n)
    q, ok = calib_frame_ok(cam, raw)
    c, d = dlt_ray(cam.coefs, q[ok], probe) if ok.any() else (cam.center(), np.zeros((0, 3)))
    centre_px = cam.to_calib_frame(np.array([[(cam.width - 1) / 2.0, (cam.height - 1) / 2.0]]))
    _, axis = dlt_ray(cam.coefs, centre_px, probe)
    along = d @ axis[0]
    pts = np.full((len(raw), 3), np.nan)
    good = along > 1e-6                                   # rays at or past 90 degrees never reach the plane
    sub = np.nonzero(ok)[0]
    ok[sub[~good]] = False
    pts[sub[good]] = c + d[good] * (float(depth) / along[good])[:, None]
    rays = [(c, pts[k]) for k in corners if ok[k]]
    keep = ok
    side = np.repeat(np.arange(4), n)                     # _border_loop's order: n points per side
    return Fov(np.asarray(c, np.float64), pts[keep], raw[keep], rays,
               not cam.has_lens_correction, axis[0], float(depth), side[keep])


def seen_by(cal: Calibration, X: np.ndarray, probe=None, reach: dict | None = None) -> np.ndarray:
    """(N,) how many calibrated cameras see each world point: in front of the camera, inside its picture
    through its lens model, and where that model round-trips (no fisheye extrapolation). `reach`: {view:
    (centre, axis, depth)} - a camera counts only up to that far plane (the field of view drawn)."""
    X = np.asarray(X, np.float64).reshape(-1, 3)
    probe = working_probe(cal) if probe is None else probe
    out = np.zeros(len(X), np.int16)
    for v in cal.views():
        out += camera_sees(cal.camera(v), X, probe, None if reach is None else reach.get(v))
    return out


def camera_sees(cam: CameraCalibration, X: np.ndarray, probe, reach: tuple | None = None) -> np.ndarray:
    """(N,) bool: `cam` sees each world point - in front of it, inside its picture through its lens model, where
    that model round-trips (`seen_by`'s rule for one camera; G209: the box walls a field of view covers).
    `reach` (centre, axis, depth): only up to that far plane."""
    X = np.asarray(X, np.float64).reshape(-1, 3)
    out = np.zeros(len(X), bool)
    front = in_front(cam.coefs, X, front_sign(cam.coefs, probe))
    if reach is not None:
        c0, ax, dep = reach
        front &= (X - c0) @ ax <= dep
    idx = np.nonzero(front)[0]
    if not len(idx):
        return out
    uv = cam.project(X[idx])
    inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -0.5) & (uv[:, 0] <= cam.width - 0.5)
              & (uv[:, 1] >= -0.5) & (uv[:, 1] <= cam.height - 0.5))
    if cam.has_lens_correction and inside.any():
        j = np.nonzero(inside)[0]
        q = cam.to_calib_frame(uv[j])
        tol = round_trip_tol(cam.width)
        good = np.isfinite(q).all(axis=1)
        good[good] &= np.linalg.norm(q[good] - dlt_project(cam.coefs, X[idx[j[good]]]), axis=1) < tol
        inside[j] = good
    out[idx[inside]] = True
    return out


@dataclass
class Coverage:
    """How many cameras see each voxel of a box: `counts` (nx, ny, nz), voxel centres at origin + i * voxel."""
    counts: np.ndarray
    origin: np.ndarray
    voxel: float
    n_cams: int
    _surf: dict = field(default_factory=dict, repr=False)

    def count_at(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64).reshape(-1, 3)
        i = np.floor((X - self.origin) / self.voxel + 0.5).astype(np.int64)
        shape = np.array(self.counts.shape)
        ok = np.isfinite(X).all(axis=1) & (i >= 0).all(axis=1) & (i < shape).all(axis=1)
        out = np.zeros(len(X), np.int16)
        out[ok] = self.counts[i[ok, 0], i[ok, 1], i[ok, 2]]
        return out

    def pct(self, X: np.ndarray, n: int) -> float:
        X = np.asarray(X, np.float64).reshape(-1, 3)
        X = X[np.isfinite(X).all(axis=1)]
        return float(100.0 * np.mean(self.count_at(X) >= n)) if len(X) else float("nan")

    def surface(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The boundary of the space seen by >= n cameras (closed: the box is padded with zeros)."""
        if n not in self._surf:
            from scipy.ndimage import gaussian_filter
            from kinetrace.hull import surface
            # the inside-or-not field, softened by one voxel so the drawn surface has no voxel steps
            # (display only: `pct` counts the cameras exactly)
            f = gaussian_filter(np.pad((self.counts >= n).astype(np.float64), 1), 1.0)
            self._surf[n] = surface(f, self.origin - self.voxel, self.voxel, iso=0.5)
        return self._surf[n]

    def sentences(self, X: np.ndarray) -> list[str]:
        """The verdict in words, for the wand cloud X (none without a wand)."""
        X = np.asarray(X, np.float64).reshape(-1, 3)
        if not np.isfinite(X).all(axis=1).any():
            return []
        p2 = self.pct(X, 2)
        out = [f"The wand cloud lies {p2:.0f} % inside the space 2+ cameras see ({_cover_word(p2)})"]
        if self.n_cams >= 3:
            p3 = self.pct(X, 3)
            out[0] += f"; 3+ cameras see {p3:.0f} % of it ({_cover_word(p3)})."
        else:
            out[0] += "."
        return out


def coverage(cal: Calibration, lo: np.ndarray, hi: np.ndarray, n_max: int = 64, probe=None,
             reach: dict | None = None) -> Coverage:
    """Count the cameras that see each voxel of the box [lo, hi] (at most `n_max` voxels per side)."""
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    voxel = float(max(np.max(hi - lo), 1e-9) / max(n_max - 1, 1))
    n = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 1, 2)
    from kinetrace.hull import voxel_centres
    centres = voxel_centres(lo, n, voxel)
    counts = seen_by(cal, centres, probe, reach).reshape(tuple(n)).astype(np.int8)
    return Coverage(counts, lo, voxel, cal.n_calibrated)


@dataclass
class CamDraw:
    view: int
    name: str
    centre: np.ndarray
    pose: Fov | None
    fov: Fov | None
    pinhole: bool


@dataclass
class CalibPicture:
    """Everything the 3D window draws for one calibration (see the module docstring)."""
    cams: list
    ends: np.ndarray                 # (P, 3) wand ends, P = 2 x frames
    end_frame: np.ndarray            # (P,) reference frame
    end_which: np.ndarray            # (P,) 0 = end 1, 1 = end 2
    end_err: np.ndarray              # (P,) px, the worse camera
    end_state: np.ndarray            # (P,) GOOD / OK / POOR / REJECTED
    pairs: np.ndarray                # (F, 2) indices of the two ends of a frame (both known)
    unit: str = ""
    end_names: tuple = ("end 1", "end 2")
    source: str = "fit"              # "fit" = a wand run's own errors; "tracks" = recomputed from the landmarks
    lines: list = field(default_factory=list)
    cal: Calibration | None = None
    cover: dict = field(default_factory=dict)       # n -> (verts, faces), filled by `use_coverage`
    coverage: Coverage | None = None

    def finite_ends(self) -> np.ndarray:
        return self.ends[np.isfinite(self.ends).all(axis=1)]

    def hover_text(self, k: int) -> str:
        nm = self.end_names[int(self.end_which[k])]
        st = int(self.end_state[k])
        word = STATE_WORDS[st] if self.source == "fit" or st != REJECTED else "disagrees with the other cameras"
        err = self.end_err[k]
        e = f"{err:.1f} px" if np.isfinite(err) else "no error"
        return f"frame {int(self.end_frame[k])}, {nm}: {e} - {word}"

    def box(self, margin: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """The wand cloud's box grown by `margin` of its size on every side (where coverage is counted)."""
        p = self.finite_ends()
        if not len(p):
            c = np.array([cd.centre for cd in self.cams])
            p = c if len(c) else np.zeros((1, 3))
        lo, hi = p.min(axis=0), p.max(axis=0)
        ext = np.maximum(hi - lo, 1e-3)
        return lo - margin * ext, hi + margin * ext

    def coverage_for(self, views=None, n_max: int = 40) -> tuple:
        """(Coverage, {n: surface}, sentences) of the space the cameras `views` (CamDraw.view ids; None =
        all) see - computed WITHOUT changing the picture, so a worker can run it while the window draws.
        The box is the whole space their drawn fields of view span; each camera counts up to its drawn far
        plane (G186; owner 2026-10-12: any choice of cameras)."""
        if self.cal is None:
            raise ValueError("no calibration to count cameras with")
        chosen = [cd for cd in self.cams if views is None or cd.view in views]
        cams = [None] * len(self.cal)
        reach, pts = {}, []
        for cd in chosen:
            cams[cd.view] = self.cal.camera(cd.view)
            if cd.fov is not None and cd.fov.axis is not None and len(cd.fov.border):
                reach[cd.view] = (cd.centre, cd.fov.axis, cd.fov.depth)
                pts += [cd.fov.border, cd.centre[None]]
        sub = Calibration(cams, self.cal.unit)
        if pts:
            allp = np.vstack(pts)
            lo, hi = allp.min(axis=0), allp.max(axis=0)
        else:
            lo, hi = self.box()
        cov = coverage(sub, lo, hi, n_max, probe=working_probe(self.cal), reach=reach or None)
        surfaces = {n: cov.surface(n) for n in range(2, cov.n_cams + 1)}
        return cov, surfaces, (cov.sentences(self.finite_ends()) if cov.n_cams >= 2 else [])

    def use_coverage(self, result: tuple) -> None:
        """Show a `coverage_for` result (on the window's thread)."""
        cov, surfaces, sentences = result
        self.coverage, self.cover = cov, dict(surfaces)
        self.lines = [ln for ln in self.lines if not ln.startswith("The wand cloud lies")]
        self.lines[:0] = sentences


def _cameras(cal: Calibration, names: list, ends: np.ndarray, probe) -> tuple[list, list[str]]:
    p = ends[np.isfinite(ends).all(axis=1)]
    centres = np.array([cal.camera(v).center() for v in cal.views()]).reshape(-1, 3)
    span = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) if len(p) else 1.0
    allp = np.vstack([p, centres]) if len(p) else centres
    scene = float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0))) if len(allp) > 1 else span
    look = np.asarray(probe, np.float64) if probe is not None else (centres.mean(axis=0) if len(centres) else None)
    cams, lines, flat = [], [], []
    for v in cal.views():
        cam = cal.camera(v)
        name = names[v] if names and v < len(names) else f"camera {v + 1}"
        c = cam.center()
        if len(p):
            far = float(np.max(np.linalg.norm(p - c, axis=1))) * FOV_REACH
        else:                       # no wand: past the point the cameras look at
            far = 1.5 * float(np.linalg.norm(look - c)) if look is not None else 2.0 * span
        # (the far plane at the farthest wand end's distance: every end lies before it)
        pinhole = not cam.has_lens_correction
        # the pose pyramid: the four corners only (a wide lens's bent edge is the field of view's job)
        cams.append(CamDraw(v, name, c, fov(cam, POSE_FRAC * max(scene, 1e-3), probe, n=1),
                            fov(cam, far, probe), pinhole))
        if pinhole:
            flat.append(name)
    if flat:
        lines.append(f"{', '.join(flat)}: no lens correction, so {'its' if len(flat) == 1 else 'their'} field of "
                     "view is drawn as a straight pinhole pyramid (with a wide lens it is wider near the corners).")
    return cams, lines


def _states(err_n: np.ndarray, rejected: np.ndarray, good: float, ok: float) -> np.ndarray:
    st = np.full(len(err_n), POOR, np.int8)
    st[err_n <= ok] = OK
    st[err_n <= good] = GOOD
    st[rejected] = REJECTED
    return st


def from_wand_result(res, frames, names: list, end_names: tuple = ("end 1", "end 2")) -> CalibPicture:
    """The picture of one wand run: `res` a `wand.WandResult` (its cameras in run order), `frames` the
    reference frame of each wand row, `names` the run's camera names."""
    F = len(res.wand_xyz)
    frames = np.asarray(frames if frames is not None else np.arange(F), np.int64).reshape(-1)[:F]
    ends = np.asarray(res.wand_xyz, np.float64).reshape(F * 2, 3)
    # every finite error: the fit's rows (their rejected clicks are in the report's outliers) and the rows
    # beyond the fit, triangulated afterwards with the final cameras (no "used" mask there)
    err = np.asarray(res.reproj_obs["wand"], np.float64)               # (F, C, 2)
    sizes = [(c.width, c.height) for c in res.cameras]
    scale = np.array([residual_scale(w, h) for w, h in sizes], np.float64)
    kept = err
    with np.errstate(all="ignore"):
        # an end's error = the RMS over the cameras that saw it: what the report's RMSE bands are for (the
        # worst single camera against them painted most of a GOOD calibration yellow - review of G185)
        worst_px = np.sqrt(np.nanmean(kept ** 2, axis=1))                  # (F, 2) px
        worst_n = np.sqrt(np.nanmean((kept / scale[None, :, None]) ** 2, axis=1))
    rej = np.zeros((F, 2), bool)
    for o in res.report.get("outliers", []):
        if o.get("kind") == "wand" and 0 <= int(o.get("row", -1)) < F:
            rej[int(o["row"]), int(o["end"])] = True
    from kinetrace.wand import RMSE_GOOD_PX, RMSE_OK_PX     # (R34) the wand report's own bands at 1080p
    state = _states(np.nan_to_num(worst_n.reshape(-1), nan=np.inf), rej.reshape(-1), RMSE_GOOD_PX, RMSE_OK_PX)
    both = np.isfinite(res.wand_xyz).all(axis=(1, 2))
    pairs = np.column_stack([2 * np.nonzero(both)[0], 2 * np.nonzero(both)[0] + 1])
    cal = Calibration(list(res.cameras), res.unit)
    probe = working_probe(cal)
    cams, lines = _cameras(cal, names, ends, probe)
    return CalibPicture(cams, ends, np.repeat(frames, 2), np.tile([0, 1], F), worst_px.reshape(-1), state, pairs,
                        res.unit, tuple(end_names), "fit", lines, cal)


def from_project(project, name_a: str | None, name_b: str | None, max_frames: int = 2000) -> CalibPicture:
    """The picture of the project's calibration (after reopening nothing of the wand run is saved): the
    cameras, their fields of view and the covered space from the calibration alone; the wand recomputed from
    its two landmarks when they are given (None: no wand points - an imported calibration, an animal project).
    An end is judged by its triangulation residual (`calib.residual_bands`); past the usable band it
    disagrees (red)."""
    from kinetrace.wanddata import _thin, sampling_window
    cal = project.calibration
    names = [project.name(c) for c in range(project.n_views)]
    if not name_a or not name_b:
        probe = working_probe(cal)
        cams, lines = _cameras(cal, names, np.zeros((0, 3)), probe)
        empty = np.zeros(0, np.int64)
        return CalibPicture(cams, np.zeros((0, 3)), empty, empty, np.zeros(0), np.zeros(0, np.int8),
                            np.zeros((0, 2), np.int64), cal.unit, ("end 1", "end 2"), "tracks",
                            lines + ["No wand landmarks: the cameras and the space they cover only (the wand "
                                     "points come with a wand calibration's two landmarks)."], cal)
    t0, t1 = sampling_window(project)
    rec = reconstruct(project.sessions, cal, project.rates, project.offsets, (t0, t1), [name_a, name_b])
    xyz = np.asarray(rec.xyz, np.float64)                                  # (T, 2, 3)
    have = np.nonzero(np.isfinite(xyz).all(axis=(1, 2)))[0]
    rows = _thin(have, max_frames)
    F = len(rows)
    ends = xyz[rows].reshape(F * 2, 3)
    res_px = np.asarray(rec.residual, np.float64)[rows].reshape(-1)       # DLTdv's residual per end
    W = max((s.width for s in project.sessions), default=1920) or 1920
    Hh = max((s.height for s in project.sessions), default=0) or 0
    good, ok = residual_bands(W, Hh)
    r = np.nan_to_num(res_px, nan=np.inf)
    state = _states(r, r > ok, good, ok)
    worst_px = res_px
    pairs = np.column_stack([2 * np.arange(F), 2 * np.arange(F) + 1])
    probe = working_probe(cal)
    cams, lines = _cameras(cal, names, ends, probe)
    frames = (rec.t0 + rows).astype(np.int64)
    return CalibPicture(cams, ends, np.repeat(frames, 2), np.tile([0, 1], F), worst_px, state, pairs,
                        cal.unit, (name_a, name_b), "tracks", lines, cal)


def wand_names(tracked: list, saved=None) -> tuple[str, str] | None:
    """The two wand landmarks for a picture rebuilt from the tracks: those of the last wand calibration
    (`saved`, display state) when both are tracked, else two tracked names that say "wand" (or "ball"); None
    otherwise - never just the first two names (an animal's snout and tail are no wand, review of G185)."""
    saved = list(saved or [])
    if len(saved) >= 2 and saved[0] in tracked and saved[1] in tracked and saved[0] != saved[1]:
        return str(saved[0]), str(saved[1])
    hits = [n for n in tracked if "wand" in n.lower()] or [n for n in tracked if "ball" in n.lower()]
    return (hits[0], hits[1]) if len(hits) >= 2 else None
