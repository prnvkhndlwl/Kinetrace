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
                             in_front, reconstruct, residual_bands, residual_scale, working_probe)

GOOD, OK, POOR, REJECTED = 0, 1, 2, 3
STATE_WORDS = {GOOD: "good", OK: "ok", POOR: "poor", REJECTED: "rejected by the fit"}
COVER_GOOD_PCT, COVER_OK_PCT = 90.0, 70.0      # share of the wand cloud inside the covered space
POSE_FRAC = 0.08                               # the pose pyramid's depth, of the cloud's size
FOV_REACH = 1.15                               # the field of view reaches this far past the farthest wand end
WAND_RMSE_GOOD_PX, WAND_RMSE_OK_PX = 1.0, 2.5  # the wand report's bands at 1080p (wand.RMSE_GOOD_PX / _OK_PX)


def _cover_word(pct: float) -> str:
    return "good" if pct >= COVER_GOOD_PCT else ("ok" if pct >= COVER_OK_PCT else "poor")


@dataclass
class Fov:
    """One camera's viewing pyramid at `depth`: `border` (M, 3) world points of the picture's border loop,
    `raw` (M, 2) the border pixels they come from, `rays` the four corners as (centre, point) segments,
    `pinhole` = the camera has no lens correction."""
    centre: np.ndarray
    border: np.ndarray
    raw: np.ndarray
    rays: list
    pinhole: bool


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
    pts = np.full((len(raw), 3), np.nan)
    pts[ok] = c + d * float(depth)
    rays = [(c, pts[k]) for k in corners if ok[k]]
    keep = ok
    return Fov(np.asarray(c, np.float64), pts[keep], raw[keep], rays,
               getattr(cam.undistort, "kind", "none") == "none")


def seen_by(cal: Calibration, X: np.ndarray, probe=None) -> np.ndarray:
    """(N,) how many calibrated cameras see each world point: in front of the camera, inside its picture
    through its lens model, and where that model round-trips (no fisheye extrapolation)."""
    X = np.asarray(X, np.float64).reshape(-1, 3)
    probe = working_probe(cal) if probe is None else probe
    out = np.zeros(len(X), np.int16)
    for v in cal.views():
        cam = cal.camera(v)
        front = in_front(cam.coefs, X, front_sign(cam.coefs, probe))
        idx = np.nonzero(front)[0]
        if not len(idx):
            continue
        uv = cam.project(X[idx])
        inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -0.5) & (uv[:, 0] <= cam.width - 0.5)
                  & (uv[:, 1] >= -0.5) & (uv[:, 1] <= cam.height - 0.5))
        if getattr(cam.undistort, "kind", "none") != "none" and inside.any():
            j = np.nonzero(inside)[0]
            q = cam.to_calib_frame(uv[j])
            tol = max(1.0, 1e-3 * cam.width)
            good = np.isfinite(q).all(axis=1)
            good[good] &= np.linalg.norm(q[good] - dlt_project(cam.coefs, X[idx[j[good]]]), axis=1) < tol
            inside[j] = good
        out[idx[inside]] += 1
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
            from kinetrace.hull import surface
            f = np.pad(self.counts.astype(np.float64), 1)
            self._surf[n] = surface(f, self.origin - self.voxel, self.voxel, iso=n - 0.5)
        return self._surf[n]

    def sentences(self, X: np.ndarray) -> list[str]:
        """The verdict in words, for the wand cloud X."""
        p2 = self.pct(X, 2)
        out = [f"The wand cloud lies {p2:.0f} % inside the space 2+ cameras see ({_cover_word(p2)})"]
        if self.n_cams >= 3:
            p3 = self.pct(X, 3)
            out[0] += f"; 3+ cameras see {p3:.0f} % of it ({_cover_word(p3)})."
        else:
            out[0] += "."
        return out


def coverage(cal: Calibration, lo: np.ndarray, hi: np.ndarray, n_max: int = 64, probe=None) -> Coverage:
    """Count the cameras that see each voxel of the box [lo, hi] (at most `n_max` voxels per side)."""
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    voxel = float(max(np.max(hi - lo), 1e-9) / max(n_max - 1, 1))
    n = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 1, 2)
    g = np.meshgrid(*(np.arange(k) for k in n), indexing="ij")
    centres = lo + np.column_stack([a.ravel() for a in g]) * voxel
    counts = seen_by(cal, centres, probe).reshape(tuple(n)).astype(np.int8)
    return Coverage(counts, lo, voxel, cal.n_calibrated)


@dataclass
class CamDraw:
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
    cover: dict = field(default_factory=dict)       # n -> (verts, faces), filled by `add_coverage`
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

    def add_coverage(self, n_max: int = 64) -> Coverage:
        """Count the cameras over the box and make the surfaces for 2, 3 and all cameras (call it on a
        worker: tens of milliseconds to a few seconds)."""
        if self.cal is None:
            raise ValueError("no calibration to count cameras with")
        lo, hi = self.box()
        cov = coverage(self.cal, lo, hi, n_max)
        for n in sorted({2, 3, cov.n_cams} - {0, 1}):
            if n <= cov.n_cams:
                self.cover[n] = cov.surface(n)
        self.coverage = cov
        self.lines = [ln for ln in self.lines if not ln.startswith("The wand cloud lies")]
        self.lines[:0] = cov.sentences(self.finite_ends())
        return cov


def _cameras(cal: Calibration, names: list, ends: np.ndarray, probe) -> tuple[list, list[str]]:
    p = ends[np.isfinite(ends).all(axis=1)]
    span = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) if len(p) else 1.0
    cams, lines = [], []
    for v in cal.views():
        cam = cal.camera(v)
        name = names[v] if names and v < len(names) else f"camera {v + 1}"
        c = cam.center()
        far = float(np.max(np.linalg.norm(p - c, axis=1))) * FOV_REACH if len(p) else 2.0 * span
        pinhole = getattr(cam.undistort, "kind", "none") == "none"
        cams.append(CamDraw(name, c, fov(cam, POSE_FRAC * max(span, 1e-3), probe, n=4),
                            fov(cam, far, probe), pinhole))
        if pinhole:
            lines.append(f"{name}: no lens correction - its field of view is drawn as a straight pinhole "
                         "pyramid (with a wide lens it is wider near the corners).")
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
        worst_px = np.nanmax(kept, axis=1)                                 # (F, 2)
        worst_n = np.nanmax(kept / scale[None, :, None], axis=1)
    rej = np.zeros((F, 2), bool)
    for o in res.report.get("outliers", []):
        if o.get("kind") == "wand" and 0 <= int(o.get("row", -1)) < F:
            rej[int(o["row"]), int(o["end"])] = True
    state = _states(np.nan_to_num(worst_n.reshape(-1), nan=np.inf), rej.reshape(-1),
                    WAND_RMSE_GOOD_PX, WAND_RMSE_OK_PX)
    both = np.isfinite(res.wand_xyz).all(axis=(1, 2))
    pairs = np.column_stack([2 * np.nonzero(both)[0], 2 * np.nonzero(both)[0] + 1])
    cal = Calibration(list(res.cameras), res.unit)
    probe = working_probe(cal)
    cams, lines = _cameras(cal, names, ends, probe)
    return CalibPicture(cams, ends, np.repeat(frames, 2), np.tile([0, 1], F), worst_px.reshape(-1), state, pairs,
                        res.unit, tuple(end_names), "fit", lines, cal)


def from_project(project, name_a: str, name_b: str, max_frames: int = 2000) -> CalibPicture:
    """The picture of the project's calibration, the wand recomputed from its two landmarks (after reopening:
    nothing of the wand run is saved). An end whose worse camera disagrees past the residual band is marked
    as disagreeing."""
    from kinetrace.wanddata import _thin, sampling_window
    cal = project.calibration
    t0, t1 = sampling_window(project)
    rec = reconstruct(project.sessions, cal, project.rates, project.offsets, (t0, t1), [name_a, name_b])
    xyz = np.asarray(rec.xyz, np.float64)                                  # (T, 2, 3)
    have = np.nonzero(np.isfinite(xyz).all(axis=(1, 2)))[0]
    rows = _thin(have, max_frames)
    F = len(rows)
    ends = xyz[rows].reshape(F * 2, 3)
    pc = np.asarray(rec.per_cam, np.float64)[rows]                         # (F, 2, C)
    C = pc.shape[2]
    scale = np.array([residual_scale(project.sessions[c].width, project.sessions[c].height) if c < project.n_views
                      else 1.0 for c in range(C)])
    with np.errstate(all="ignore"):
        worst_px = np.nanmax(pc, axis=2).reshape(-1)
        worst_n = np.nanmax(pc / scale[None, None, :], axis=2).reshape(-1)
    good, ok = residual_bands(1920.0)
    worst_n = np.nan_to_num(worst_n, nan=np.inf)
    state = _states(worst_n, worst_n > ok, good, ok)
    pairs = np.column_stack([2 * np.arange(F), 2 * np.arange(F) + 1])
    probe = working_probe(cal)
    cams, lines = _cameras(cal, [project.name(c) for c in range(project.n_views)], ends, probe)
    frames = (rec.t0 + rows).astype(np.int64)
    return CalibPicture(cams, ends, np.repeat(frames, 2), np.tile([0, 1], F), worst_px, state, pairs,
                        cal.unit, (name_a, name_b), "tracks", lines, cal)
