"""Camera calibration, lens undistortion and 3D triangulation.

Kinetrace's 2D layer digitizes each camera on its own; this module is what
turns those per-view tracks into 3D. Everything here is plain numpy/scipy —
no GUI, no torch — so it runs headless and is easy to test against ground
truth.

Camera model
------------
The 11-parameter **direct linear transform** (DLT), the format used by DLTdv,
easyWand and Argus. For camera coefficients L1..L11 a world point (X, Y, Z)
appears at::

    u = (L1 X + L2 Y + L3 Z + L4) / (L9 X + L10 Y + L11 Z + 1)
    v = (L5 X + L6 Y + L7 Z + L8) / (L9 X + L10 Y + L11 Z + 1)

`triangulate` solves the linear least-squares system of 2 rows per camera for
(X, Y, Z) and reports DLTdv's residual: the reprojection RMS normalised by the
degrees of freedom, ``sqrt(sum(e^2) / (2 n_cams - 3))``. Verified on a real
6-camera DLTdv project: 0.07 mm median distance from DLTdv's own xyz and a
residual ratio of 0.9999.

Image coordinate conventions (this is where 3D pipelines silently break)
-------------------------------------------------------------------------
The DLT coefficients are fitted to whatever 2D coordinates the calibration
software was given. (I278) Inside Kinetrace EVERY calibration is in Kinetrace's
own pixels -- 0-based, y down from the top -- like the tracks: a file's
convention is converted once where it enters (`Calibration.to_internal`: the
import dialog, opening a project, `convert`) and back only for the programs
that want theirs (dltCoefs.csv / MATLAB: +1). A file's convention, until then:

* ``pixel_origin`` — MATLAB (DLTdv, easyWand) numbers pixel centres from 1,
  OpenCV/Kinetrace from 0: a MATLAB-fitted DLT wants ``x + 1, y + 1``.
* ``y_flip`` — some exports measure y from the bottom edge (``height - y``).
  DLTdv8 projects use a top-left origin (checked on a real DLTdv8 project:
  the clicked coordinates land on the animal only unflipped); older
  DLTdv5-era files may need True. A file that does not say (a bare
  dltCoefs.csv, easyWand) is asked about.
* the undistortion model, fitted on undistorted points. `LWMUndistort` reproduces
  MATLAB's `fitgeotrans(..., 'lwm', N)` local weighted mean transform
  (Goshtasby 1988) from its control points, which is how DLTdv undistorts
  GoPro footage; it matches DLTdv to 0.55–1.7 px median on real clicks (the
  DLT's own residual). `OpenCVUndistort` wraps the usual K + distortion
  coefficients.

Sub-frame timing
----------------
Cameras that were not genlocked differ by a fractional frame. Each view has a
frame ``rate`` relative to the reference camera (2.0 for a 240 fps camera in
a 120 fps rig) and a fractional ``offset``: reference instant ``t`` (in
reference frames) is that view's local frame ``rate * t + offset``. A 2D
track is sampled at a fractional local frame by LINEAR interpolation between
consecutive tracked frames (never across a gap). That reproduces DLTdv's
``fs17_timeShift`` output to 0.08 mm median on a real project — a cubic
spline was measurably worse (0.25 mm), so linear it is. `estimate_offsets`
recovers the fractional part by minimising the triangulation residual, which
is exactly the quantity a sync error inflates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- DLT maths


def dlt_matrix(coefs: np.ndarray) -> np.ndarray:
    """The 3 x 4 projection matrix P of 11 DLT coefficients (P[2, 3] = 1)."""
    L = np.asarray(coefs, np.float64).reshape(11)
    return np.array([[L[0], L[1], L[2], L[3]], [L[4], L[5], L[6], L[7]], [L[8], L[9], L[10], 1.0]])


def dlt_denominator(coefs: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """The DLT denominator ``L9 X + L10 Y + L11 Z + 1`` of world points (N, 3):
    the depth up to a scale and a sign (see `front_sign`)."""
    L = np.asarray(coefs, np.float64).reshape(11)
    X = np.atleast_2d(np.asarray(xyz, np.float64))
    return L[8] * X[:, 0] + L[9] * X[:, 1] + L[10] * X[:, 2] + 1.0


def in_front(coefs: np.ndarray, xyz: np.ndarray, sign: float) -> np.ndarray:
    """(N,) bool: the world points in front of the camera, given its
    `front_sign` (the sign its denominator has for points in view)."""
    return sign * dlt_denominator(coefs, xyz) > 0


def dlt_project(coefs: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Project world points (N, 3) through 11 DLT coefficients -> (N, 2)."""
    L = np.asarray(coefs, np.float64).reshape(11)
    X = np.atleast_2d(np.asarray(xyz, np.float64))
    den = dlt_denominator(L, X)
    u = (L[0] * X[:, 0] + L[1] * X[:, 1] + L[2] * X[:, 2] + L[3]) / den
    v = (L[4] * X[:, 0] + L[5] * X[:, 1] + L[6] * X[:, 2] + L[7]) / den
    return np.column_stack([u, v])


def dlt_camera_center(coefs: np.ndarray) -> np.ndarray:
    """World position of the camera: the point every ray passes through."""
    L = np.asarray(coefs, np.float64).reshape(11)
    A = np.array([[L[0], L[1], L[2]], [L[4], L[5], L[6]], [L[8], L[9], L[10]]])
    b = np.array([-L[3], -L[7], -1.0])
    return np.linalg.solve(A, b)


def front_sign(coefs: np.ndarray, probe: np.ndarray | None = None) -> float:
    """+1 or -1: the sign the DLT denominator ``L9 X + L10 Y + L11 Z + 1`` has
    for world points IN FRONT of the camera. With a probe point known to be
    in view (a triangulated landmark, the centre of the working volume) it is
    simply that point's sign. Without one, fall back to sign(det M) with
    M = the 3x3 of L1-3 / L5-7 / L9-11 — right for a right-handed pinhole
    (det K > 0) and WRONG for calibrations fitted with a mirrored image axis
    (easyWand / DLTdv coefficients on a real rig: det M < 0 while the
    animal has a positive denominator in every camera). Pass a probe."""
    L = np.asarray(coefs, np.float64).reshape(11)
    if probe is not None:
        p = np.asarray(probe, np.float64).reshape(3)
        den = L[8] * p[0] + L[9] * p[1] + L[10] * p[2] + 1.0
        if abs(den) > 1e-12:
            return 1.0 if den > 0 else -1.0
    M = np.array([L[0:3], L[4:7], L[8:11]])
    return float(np.sign(np.linalg.det(M)) or 1.0)


def dlt_ray(coefs: np.ndarray, uv: np.ndarray, probe: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and unit direction(s) of the ray(s) through image points
    (N, 2). The direction is the null vector of the two DLT row equations,
    oriented forward (`front_sign`; give a `probe` point in view for
    calibrations with a mirrored image axis)."""
    L = np.asarray(coefs, np.float64).reshape(11)
    c = dlt_camera_center(L)
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    # rows: (u L9 - L1, u L10 - L2, u L11 - L3) . d = 0 and the v analogue
    r1 = np.column_stack([uv[:, 0] * L[8] - L[0], uv[:, 0] * L[9] - L[1], uv[:, 0] * L[10] - L[2]])
    r2 = np.column_stack([uv[:, 1] * L[8] - L[4], uv[:, 1] * L[9] - L[5], uv[:, 1] * L[10] - L[6]])
    d = np.cross(r1, r2)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
    # orient forward: the DLT denominator grows along the viewing direction
    # with the front sign of this calibration
    sgn = front_sign(L, probe)
    flip = sgn * (d @ L[8:11]) < 0
    d[flip] *= -1
    return c, d


def triangulate(coefs: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Linear DLT triangulation of ONE point seen by n >= 2 cameras.

    coefs: (11, n) or (n, 11); uv: (n, 2) undistorted image coordinates in the
    calibration's convention. Returns (xyz, residual, per-camera reprojection
    distances). residual = sqrt(sum(e^2) / (2 n - 3)) — DLTdv's definition,
    which is what its xyzres files hold (verified: ratio 0.9999)."""
    Ls = np.asarray(coefs, np.float64)
    if Ls.shape[0] != 11:
        Ls = Ls.T
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    n = uv.shape[0]
    if n < 2 or Ls.shape[1] != n:
        raise ValueError("triangulate needs the same >= 2 cameras in coefs and uv")
    # (R14) one triangulator: the batch one, so the single point and the
    # many-points paths cannot drift apart
    xyz, res, _, err = triangulate_batch(Ls.T, uv[None])
    return xyz[0], float(res[0]), err[0]


def dlt_from_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """11 DLT coefficients from an OpenCV-style pinhole camera P = K [R | t]
    (x_cam = R X + t). Handy for tests and for the OpenCV/MATLAB importers."""
    P = np.asarray(K, np.float64) @ np.hstack([np.asarray(R, np.float64),
                                               np.asarray(t, np.float64).reshape(3, 1)])
    P = P / P[2, 3]
    return np.concatenate([P[0], P[1], P[2, :3]])


# ------------------------------------------------------------ undistortion


class NoUndistort:
    """Identity: the calibration was fitted on raw pixels (or the lens is
    rectilinear enough not to matter)."""
    kind = "none"

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(pts, np.float64)).copy()

    def distort(self, pts: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(pts, np.float64)).copy()

    def to_json(self) -> dict:
        return {"kind": self.kind}


class LWMUndistort:
    """MATLAB's local weighted mean transform (`fitgeotrans(uv, xy, 'lwm', N)`,
    Goshtasby 1988), rebuilt from its control points.

    For each control point a degree-2 polynomial (1, x, y, xy, x², y²) is
    fitted to its N nearest control points; a query is the weighted mean of
    the polynomials whose radius of influence (distance to the N-th
    neighbour) covers it, weight ``1 - 3R² + 2R³``. Outside every radius the
    map is undefined (NaN) — exactly MATLAB's behaviour.

    `raw_pts` are the distorted (as-filmed) control points, `und_pts` their
    undistorted positions. `undistort` maps raw -> undistorted (what DLTdv's
    ``camud.inverse_fcn`` does to every click); `distort` is the same
    construction fitted the other way round, used to look silhouettes up in
    the raw frame from a DLT projection."""
    kind = "lwm"

    def __init__(self, raw_pts: np.ndarray, und_pts: np.ndarray, n_neighbours: int = 12):
        self.raw_pts = np.asarray(raw_pts, np.float64).reshape(-1, 2)
        self.und_pts = np.asarray(und_pts, np.float64).reshape(-1, 2)
        self.n_neighbours = int(n_neighbours)
        if len(self.raw_pts) != len(self.und_pts) or len(self.raw_pts) < 6:
            raise ValueError("LWM needs >= 6 matched control points")
        self._fwd = _LWMFit(self.raw_pts, self.und_pts, self.n_neighbours)
        self._inv = _LWMFit(self.und_pts, self.raw_pts, self.n_neighbours)

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        return self._fwd(pts)

    def distort(self, pts: np.ndarray) -> np.ndarray:
        return self._inv(pts)

    def to_json(self) -> dict:
        return {"kind": self.kind, "n": self.n_neighbours,
                "raw": self.raw_pts.tolist(), "und": self.und_pts.tolist()}


class _LWMFit:
    """One direction of the LWM map, vectorised over queries in chunks."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, n: int):
        self.src = src
        M = len(src)
        n = int(min(max(n, 6), M))
        d = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
        self.T = np.zeros((M, 6, 2))
        self.radius = np.zeros(M)
        for i in range(M):
            nn = np.argsort(d[i])[:n]
            self.radius[i] = d[i, nn].max()
            x, y = src[nn, 0], src[nn, 1]
            X = np.column_stack([np.ones(n), x, y, x * y, x * x, y * y])
            self.T[i] = np.linalg.lstsq(X, dst[nn], rcond=None)[0]
        self.radius = np.maximum(self.radius, 1e-9)

    def __call__(self, pts: np.ndarray, chunk: int = 2048) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, np.float64))
        out = np.full(p.shape, np.nan)
        for s in range(0, len(p), chunk):
            q = p[s:s + chunk]
            ok = np.isfinite(q).all(axis=1)
            if not ok.any():
                continue
            qq = q[ok]
            R = np.sqrt(((qq[:, None, :] - self.src[None, :, :]) ** 2).sum(2)) / self.radius[None, :]
            W = np.where(R < 1.0, 1.0 - 3.0 * R ** 2 + 2.0 * R ** 3, 0.0)     # (n, M)
            X = np.column_stack([np.ones(len(qq)), qq[:, 0], qq[:, 1], qq[:, 0] * qq[:, 1],
                                 qq[:, 0] ** 2, qq[:, 1] ** 2])              # (n, 6)
            vals = np.einsum("nj,mjk->nmk", X, self.T)                          # (n, M, 2)
            wsum = W.sum(1)
            res = np.einsum("nm,nmk->nk", W, vals) / np.where(wsum > 0, wsum, np.nan)[:, None]
            tmp = np.full(q.shape, np.nan)
            tmp[ok] = res
            out[s:s + chunk] = tmp
        return out


class OpenCVUndistort:
    """Brown–Conrady / OpenCV model: camera matrix K and distortion
    coefficients (k1, k2, p1, p2[, k3 ...]). Undistorted points are returned
    in the same pixel frame (P = K), which is what a DLT fitted on
    `cv2.undistortPoints(..., P=K)` output expects."""
    kind = "opencv"
    # (I73) OpenCV's standard-model point undistortion is a fixed-point
    # iteration that stops after 5 steps unless told otherwise: far from
    # converged at the edges of a wide lens (1.6 px off on a GoPro-like
    # f = 1300 / k1 = -0.25 lens, over 100 px on a stronger one, against a
    # ~0.3 px lens fit), which also made `lens.border_check` call a sound,
    # invertible model a runaway. Iterate until the point re-distorts to within
    # 1e-8 px (count | eps); a point where the model has no inverse still fails
    # the round trip, which is what the border check is for. Converging costs
    # ~0.3 s per 400k tracked cells on a strong lens. The fisheye solver is
    # already Newton on the angle (10 steps, eps 1e-8) and converges.
    CRITERIA = (3, 1000, 1e-8)             # cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS

    def __init__(self, K: np.ndarray, dist: np.ndarray, fisheye: bool = False,
                 P: np.ndarray | None = None):
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).ravel()
        self.fisheye = bool(fisheye)
        # the pixel frame the undistorted points land in: K itself by default,
        # or a "square" camera matrix (fx = fy) when a downstream solver with a
        # single focal length per camera (the wand core) consumes them
        self.P = self.K if P is None else np.asarray(P, np.float64).reshape(3, 3)

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        import cv2
        p = np.atleast_2d(np.asarray(pts, np.float64)).reshape(-1, 1, 2)
        ok = np.isfinite(p).all(axis=2).ravel()
        out = np.full((len(p), 2), np.nan)
        if ok.any():
            if self.fisheye:
                u = cv2.fisheye.undistortPoints(p[ok], self.K, self.dist.reshape(-1, 1)[:4], P=self.P)
            else:
                u = cv2.undistortPoints(p[ok], self.K, self.dist, None, None, self.P, self.CRITERIA)
            out[ok] = u.reshape(-1, 2)
        return out

    def distort(self, pts: np.ndarray) -> np.ndarray:
        import cv2
        p = np.atleast_2d(np.asarray(pts, np.float64))
        ok = np.isfinite(p).all(axis=1)
        out = np.full(p.shape, np.nan)
        if ok.any():
            Kinv = np.linalg.inv(self.P)
            hom = np.column_stack([p[ok], np.ones(ok.sum())]) @ Kinv.T   # normalised coords
            obj = np.column_stack([hom[:, :2], np.zeros(ok.sum())]).reshape(-1, 1, 3)
            rvec = np.zeros(3)
            tvec = np.array([0.0, 0.0, 1.0])
            if self.fisheye:
                d, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, self.K, self.dist.reshape(-1, 1)[:4])
            else:
                d, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
            out[ok] = d.reshape(-1, 2)
        return out

    def to_json(self) -> dict:
        d = {"kind": self.kind, "K": self.K.tolist(), "dist": self.dist.tolist(),
             "fisheye": self.fisheye}
        if not np.allclose(self.P, self.K):
            d["P"] = self.P.tolist()
        return d


def undistort_from_json(d: dict):
    kind = d.get("kind", "none")
    if kind == "lwm":
        return LWMUndistort(np.asarray(d["raw"]), np.asarray(d["und"]), int(d.get("n", 12)))
    if kind == "opencv":
        return OpenCVUndistort(np.asarray(d["K"]), np.asarray(d["dist"]), bool(d.get("fisheye", False)),
                               None if d.get("P") is None else np.asarray(d["P"]))
    return NoUndistort()


def _mcos_object_id(e) -> int | None:
    """The MCOS object id a MATLAB object reference points at (a scalar
    object's metadata is [0xDD000000, 2, 1, 1, id, class]), or None when `e`
    is not such a reference (e.g. a camera DLTdv left without undistortion)."""
    try:
        meta = np.asarray(np.asarray(e["_ObjectMetadata"]).ravel()[0]).astype(np.uint32).ravel()
    except Exception:      # noqa: BLE001 — not an object reference
        return None
    if len(meta) >= 6 and meta[0] == 0xDD000000 and meta[1] == 2 and meta[2] == 1 and meta[3] == 1:
        return int(meta[4])
    return None


def _assign_lwm_profiles(n: int, cells: list, ids: list) -> tuple[list, str]:
    """Which of a DLTdv project's LWM transforms (the property structs in the
    MCOS store, `cells`) belongs to which of its `n` cameras. (I102) Matched by
    the object id each camera's `camd` entry references (object k's
    properties are cells[1 + k]); a camera without a reference has no
    undistortion. Without usable ids, assigned in order ONLY when there is
    exactly one transform per camera -- with fewer, the ones after a missing
    camera would land on the wrong cameras, so none is applied. Returns
    (per-camera struct or None, a sentence for the user or "")."""
    def lwm(k):
        if k is None or not (1 <= k and 1 + k < len(cells)):
            return None
        e = cells[1 + k]
        return e if hasattr(e, "_fieldnames") and "uvPoints" in e._fieldnames else None

    tforms = [e for e in cells if hasattr(e, "_fieldnames") and "uvPoints" in e._fieldnames]
    if not tforms:
        return [None] * n, ""
    if len(ids) == n and any(k is not None for k in ids):
        out = [lwm(k) for k in ids]
        broken = [c + 1 for c, k in enumerate(ids) if k is not None and out[c] is None]
        if not broken:
            bare = [c + 1 for c in range(n) if out[c] is None]
            return out, ("" if not bare else
                         "Lens undistortion was found for every camera except camera(s) "
                         + ", ".join(map(str, bare)) + ", which the DLTdv project leaves as filmed.")
    if len(tforms) == n:
        return list(tforms), ""
    return [None] * n, (f"The project holds {len(tforms)} lens profile(s) for {n} cameras and does not say "
                        "which camera each belongs to, so NONE was applied: every camera would be used as "
                        "filmed. Check the project in DLTdv8 (every camera needs its undistortion profile).")


# ------------------------------------------------------------ calibration


@dataclass
class CameraCalibration:
    """One camera's DLT coefficients plus the coordinate convention they were
    fitted in (see the module docstring): 0 / False for every camera inside the
    app (`to_internal`); a file's own only between its reader and `to_internal`."""
    coefs: np.ndarray                      # (11,)
    width: int = 0
    height: int = 0
    undistort: object = field(default_factory=NoUndistort)
    pixel_origin: float = 0.0              # 1.0 = MATLAB-fitted (DLTdv/easyWand), 0.0 = OpenCV / Kinetrace
    y_flip: bool = False                   # True: y measured from the bottom edge
    rmse: float = float("nan")             # calibration residual, if the file carried one

    @property
    def has_lens_correction(self) -> bool:
        """(R30) The camera's raw pixels are undistorted before the DLT (an LWM or OpenCV model)."""
        return getattr(self.undistort, "kind", "none") != "none"

    def is_internal(self) -> bool:
        """In Kinetrace's own pixels (0-based, top-left): the only convention inside the app."""
        return float(self.pixel_origin) == 0.0 and not self.y_flip

    def to_internal(self) -> "CameraCalibration":
        """(I278) This camera re-expressed in Kinetrace's own pixels. Exact: the coefficients go
        through `pixel_matrix` (its last row is (0, 0, 1), so L9..L11 and the scale are kept),
        an LWM lens model's control points and an OpenCV model's principal point move the same
        way, and the world, the camera centre and every residual are unchanged. ValueError (a
        sentence) for a bottom-left file without its picture height, and for an OpenCV lens
        model measured from the bottom (no file makes one)."""
        if self.is_internal():
            return self
        if self.y_flip and self.height <= 0:
            raise ValueError("this calibration measures y from the bottom edge of the picture, so its picture "
                             "height is needed to turn it over; open it with the project's videos (or give "
                             "--size WIDTHxHEIGHT)")
        A = self.pixel_matrix()
        P = A @ dlt_matrix(self.coefs)
        und = self.undistort
        if isinstance(und, LWMUndistort):
            und = LWMUndistort(self.from_dlt_pixels(und.raw_pts), self.from_dlt_pixels(und.und_pts),
                               und.n_neighbours)
        elif isinstance(und, OpenCVUndistort):
            if self.y_flip:
                raise ValueError("an OpenCV lens model with y measured from the bottom edge cannot be used; "
                                 "OpenCV lens models are 0-based, y down from the top")
            und = OpenCVUndistort(A @ und.K, und.dist, und.fisheye, None if und.P is und.K else A @ und.P)
        return CameraCalibration(np.concatenate([P[0], P[1], P[2, :3]]), self.width, self.height, und,
                                 0.0, False, self.rmse)

    # (R14) THE pixel-convention transform, now used only to bring a file's
    # convention in (`to_internal`) and to write it out again.
    def to_dlt_pixels(self, pts_xy: np.ndarray) -> np.ndarray:
        """Kinetrace pixels (0-based, top-left) -> the DLT's pixel convention,
        WITHOUT undistortion (`pixel_origin`, then `y_flip`)."""
        p = np.atleast_2d(np.asarray(pts_xy, np.float64)).copy()
        p += self.pixel_origin
        if self.y_flip:
            p[:, 1] = self.height - p[:, 1] + 2 * self.pixel_origin - 1
        return p

    def from_dlt_pixels(self, pts_uv: np.ndarray) -> np.ndarray:
        """Inverse of `to_dlt_pixels` (the flip is its own inverse)."""
        p = np.atleast_2d(np.asarray(pts_uv, np.float64)).copy()
        if self.y_flip:
            p[:, 1] = self.height - p[:, 1] + 2 * self.pixel_origin - 1
        return p - self.pixel_origin

    def pixel_matrix(self) -> np.ndarray:
        """3 x 3 homogeneous map from the DLT's pixel convention to
        Kinetrace's (0-based, top-left): the matrix form of `from_dlt_pixels`."""
        po, H = float(self.pixel_origin), float(self.height)
        if self.y_flip:
            return np.array([[1.0, 0.0, -po], [0.0, -1.0, H + po - 1.0], [0.0, 0.0, 1.0]])
        return np.array([[1.0, 0.0, -po], [0.0, 1.0, -po], [0.0, 0.0, 1.0]])

    def coefs_for_origin(self, origin: float = 1.0) -> np.ndarray:
        """The 11 coefficients re-expressed for pixels counted from `origin`
        (u' = u + (origin - pixel_origin), the same for v: L1..L3 += shift *
        L9..L11, L4 += shift) -- what a MATLAB-side dltCoefs.csv holds (1)."""
        L = np.asarray(self.coefs, np.float64).copy()
        shift = float(origin) - float(self.pixel_origin)
        if abs(shift) > 1e-12:
            L[0:3] += shift * L[8:11]
            L[3] += shift
            L[4:7] += shift * L[8:11]
            L[7] += shift
        return L

    def to_calib_frame(self, pts_xy: np.ndarray) -> np.ndarray:
        """Kinetrace pixels (0-based, top-left) -> undistorted coordinates in
        the frame the DLT was fitted in (the same pixels: `to_internal`)."""
        return self.undistort.undistort(pts_xy)

    def from_calib_frame(self, pts_uv: np.ndarray) -> np.ndarray:
        """Inverse of `to_calib_frame`: calibration-frame coordinates (e.g. a DLT
        projection) -> raw Kinetrace pixels."""
        return self.undistort.distort(np.atleast_2d(np.asarray(pts_uv, np.float64)))

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """World -> raw Kinetrace pixels (distortion re-applied)."""
        return self.from_calib_frame(dlt_project(self.coefs, xyz))

    def center(self) -> np.ndarray:
        return dlt_camera_center(self.coefs)


class Calibration:
    """The cameras of a project, in VIEW order, plus the world unit.

    (I273) A camera the calibration does not cover is `None` in `cameras` (a wand calibration of
    some of the cameras), and so is every camera past the end of the list (a camera added after
    calibrating, G25). Read a camera with `camera(v)`, never `cameras[v]`; `views()` are the
    calibrated cameras. `len()` counts the slots, calibrated or not."""

    def __init__(self, cameras: list[CameraCalibration | None] | None = None, unit: str = "",
                 source: str = ""):
        self.cameras: list[CameraCalibration | None] = list(cameras or [])
        self.unit = unit            # "m", "mm", … purely a label for exports
        self.source = source        # where it came from (file name), for the UI
        # set by `from_krt`: the world origin was moved by this vector (old-frame
        # coordinates) so that no camera sits at the DLT origin; add it to 3D
        # results to return to the file's frame
        self.origin_shift: np.ndarray | None = None
        # plain sentences from the importer about what it could NOT read (a lens
        # store that failed to parse, distortion lines it did not apply), shown
        # by the import dialog instead of a generic "no undistortion" (I102)
        self.notes: list[str] = []
        # (I278) False: the file does not say how it counted pixels (a bare dltCoefs.csv, easyWand),
        # so the user is asked before `to_internal`
        self.convention_stated = True
        # (G201) the world the user chose with 3D -> Set World Axes: {"origin", "x", "y": landmark names,
        # "when": "frame N" | "median"}; None = the calibration's own world. The 3D window draws it.
        self.world: dict | None = None

    def with_cameras(self, cams: list, source: str | None = None) -> "Calibration":
        """The same calibration with other cameras: unit, source (or `source`), the moved
        origin, the importer's notes and the quality report travel with it."""
        out = Calibration(cams, self.unit, self.source if source is None else source)
        out.origin_shift = self.origin_shift
        out.notes = list(self.notes)
        out.world = None if self.world is None else dict(self.world)
        if hasattr(self, "report"):
            out.report = self.report
        return out

    def is_internal(self) -> bool:
        return all(c is None or c.is_internal() for c in self.cameras)

    def to_internal(self) -> "Calibration":
        """(I278) Every camera in Kinetrace's own pixels (`CameraCalibration.to_internal`); itself
        when they already are."""
        return self if self.is_internal() else self.with_cameras(
            [None if c is None else c.to_internal() for c in self.cameras])

    def __len__(self) -> int:
        return len(self.cameras)

    def camera(self, v: int) -> CameraCalibration | None:
        """Camera `v`'s calibration, or None when it is not calibrated."""
        return self.cameras[v] if 0 <= v < len(self.cameras) else None

    def views(self) -> list[int]:
        """The calibrated cameras' indices, in camera order."""
        return [k for k, c in enumerate(self.cameras) if c is not None]

    def calibrated_cameras(self) -> list[CameraCalibration]:
        """(R30) The calibrated cameras themselves, in camera order (`views()`'s cameras)."""
        return [c for c in self.cameras if c is not None]

    @property
    def n_calibrated(self) -> int:
        return sum(c is not None for c in self.cameras)

    def n_usable(self, n_views: int) -> int:
        """(I283) The cameras of an `n_views` project that 3D can use: the calibrated ones -- none
        when this calibration lists MORE cameras than the project (made for another camera set:
        Reconstruct refuses it). THE "two or more calibrated cameras" count of the window."""
        return self.n_calibrated if len(self.cameras) <= int(n_views) else 0

    def covers(self, n_views: int) -> bool:
        """Every one of `n_views` cameras is calibrated."""
        return len(self.cameras) >= n_views and all(c is not None for c in self.cameras[:n_views])

    def padded(self, n_views: int) -> list[CameraCalibration | None]:
        """One slot per camera of an `n_views` project (None = not calibrated)."""
        return [self.camera(v) for v in range(n_views)]

    # ---- files ----------------------------------------------------------
    @staticmethod
    def load_dlt_csv(path: str | Path, sizes: list[tuple[int, int]] | None = None,
                     pixel_origin: float = 1.0, y_flip: bool = False) -> "Calibration":
        """DLTdv / easyWand / Argus `*dltCoefs.csv`: 11 rows, one column per
        camera (a header row is tolerated). `sizes` = (width, height) per
        camera when known."""
        raw = np.genfromtxt(str(path), delimiter=",")
        raw = np.atleast_2d(raw)
        if raw.shape[0] != 11 and raw.shape[1] == 11:
            raw = raw.T
        if raw.shape[0] == 12:               # header row became NaN
            raw = raw[1:]
        if raw.shape[0] != 11 or not np.isfinite(raw).all():
            raise ValueError(f"{Path(path).name}: expected 11 DLT coefficients per camera")
        cams = []
        for c in range(raw.shape[1]):
            w, h = (sizes[c] if sizes and c < len(sizes) else (0, 0))
            cams.append(CameraCalibration(raw[:, c].copy(), int(w), int(h), NoUndistort(),
                                          pixel_origin, y_flip))
        cal = Calibration(cams, source=Path(path).name)
        cal.convention_stated = False          # DLTdv, easyWand and Argus all write this file: ask (I278)
        return cal

    @staticmethod
    def load_easywand_mat(path: str | Path, pixel_origin: float = 1.0,
                          y_flip: bool = False) -> "Calibration":
        """easyWand `*_easyWandData.mat` (MATLAB v5–v7): `coefs` (11 x n),
        image sizes, per-camera DLT RMSE and the wand length (its unit is the
        world unit). Cameras come out in easyWand's camera order."""
        from scipy.io import loadmat
        m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
        ew = m.get("easyWandData")
        if ew is None:
            raise ValueError(f"{Path(path).name}: no easyWandData variable")
        coefs = np.asarray(ew.coefs, np.float64)
        if coefs.shape[0] != 11:
            coefs = coefs.T
        n = coefs.shape[1]
        ws = np.atleast_1d(getattr(ew, "imageWidth", np.zeros(n))).astype(int)
        hs = np.atleast_1d(getattr(ew, "imageHeight", np.zeros(n))).astype(int)
        rm = np.atleast_1d(getattr(ew, "dltRMSE", np.full(n, np.nan))).astype(float)
        cams = [CameraCalibration(coefs[:, c].copy(), int(ws[c]) if c < len(ws) else 0,
                                  int(hs[c]) if c < len(hs) else 0, NoUndistort(),
                                  pixel_origin, y_flip, float(rm[c]) if c < len(rm) else np.nan)
                for c in range(n)]
        cal = Calibration(cams, source=Path(path).name)
        cal.convention_stated = False          # MATLAB's 1-based pixels, but y from the top or the bottom: ask
        wl = getattr(ew, "wandLen", None)
        if wl is not None:
            cal.unit = f"wand={float(wl):g}"
        return cal

    @staticmethod
    def load_dltdv_project(path: str | Path) -> "Calibration":
        """A DLTdv8 `*_dvProject.mat` saved as MATLAB v7 (not v7.3/HDF5): the
        DLT coefficients, image sizes and the per-camera LWM undistortion
        control points that DLTdv keeps in its MCOS object store."""
        from scipy.io import loadmat
        from scipy.io.matlab._mio5 import MatFile5Reader
        import io
        m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
        ud = m.get("udExport")
        if ud is None:
            raise ValueError(f"{Path(path).name}: not a DLTdv project (no udExport)")
        coefs = np.asarray(ud.data.dltcoef, np.float64)
        sizes = np.atleast_2d(np.asarray(ud.data.movsizes)).astype(int)   # rows: (h, w)
        n = coefs.shape[1]
        cams = [CameraCalibration(coefs[:, c].copy(), int(sizes[c, 1]), int(sizes[c, 0]), NoUndistort(),
                                  1.0, False)            # DLTdv8: MATLAB pixels, y down from the top
                for c in range(n)]
        notes: list[str] = []
        # undistortion: DLTdv stores one MATLAB geometric transform per camera in
        # the function-workspace subsystem; read it back as a nested MAT stream
        fw = m.get("__function_workspace__")
        if fw is not None:
            try:
                blob = np.asarray(fw, np.uint8).tobytes()
                buf = b" " * 116 + b"\x00" * 8 + b"\x00\x01IM" + blob[8:]
                rdr = MatFile5Reader(io.BytesIO(buf), struct_as_record=False, squeeze_me=True)
                rdr.initialize_read()
                rdr.read_file_header()
                hdr, _ = rdr.read_var_header()
                val = rdr.read_var_array(hdr, process=True)
                cells = list(np.asarray(val.MCOS[0][2]).ravel())
                camd = getattr(ud.data, "camd", None)
                ids = ([_mcos_object_id(e) for e in np.atleast_1d(np.asarray(camd, dtype=object)).ravel()]
                       if camd is not None else [])
                profiles, note = _assign_lwm_profiles(n, cells, ids)
                for c, tf in enumerate(profiles):
                    if tf is not None:
                        cams[c].undistort = LWMUndistort(np.asarray(tf.uvPoints), np.asarray(tf.xyPoints),
                                                         int(getattr(tf, "N", 12)))
                if note:
                    notes.append(note)
            except Exception as e:      # noqa: BLE001 — the DLT still loads; SAY the lenses did not (I102)
                for cam in cams:
                    cam.undistort = NoUndistort()
                notes.append(f"The project's lens undistortion could not be read ({type(e).__name__}: {e}), so "
                             "every camera would be used as filmed -- pixels of error at the edges of a "
                             "GoPro-style lens. Re-save the project from DLTdv8 as MATLAB v7, or import the "
                             "cameras' lens profiles another way before trusting the 3D.")
        cal = Calibration(cams, source=Path(path).name)
        cal.notes = notes
        return cal

    @staticmethod
    def from_krt(cams: list[dict], unit: str = "", source: str = "") -> "Calibration":
        """Cameras given the OpenCV way -- K (3x3), R (3x3), t (3), x_cam = R X + t,
        optional `dist` (OpenCV coefficients, or `fisheye`: True with 4), `width`,
        `height` -- turned into 11-parameter DLT cameras.

        The pitfall: a DLT is P scaled so that
        P[2, 3] = 1, and a camera AT the world origin has P[2, 3] = 0 -- the
        usual OpenCV stereo convention (camera 1 at the origin) has no DLT at
        all. So the world origin is moved to a point the cameras look at: the
        mean of the camera centres pushed one baseline length along the mean
        viewing direction. The shift is recorded in `origin_shift` (add it to
        any 3D result to get back to the file's frame) and named in `source`,
        so the report can say where the origin went. Pixels are OpenCV's:
        0-based, top-left, unflipped."""
        if len(cams) < 2:
            raise ValueError("a K + R/t calibration needs at least two cameras")
        Ks, Rs, ts = [], [], []
        for i, c in enumerate(cams):
            # (I227) a camera without a K / with a malformed one is a sentence naming it,
            # not a KeyError that reaches the user as "'K'"
            if not isinstance(c, dict) or "K" not in c:
                raise ValueError(f"camera {i + 1} has no K (its 3x3 camera matrix)")
            try:
                K = np.asarray(c["K"], np.float64).reshape(3, 3)
                R = np.asarray(c.get("R", np.eye(3)), np.float64).reshape(3, 3)
                t = np.asarray(c.get("t", np.zeros(3)), np.float64).reshape(3)
            except (TypeError, ValueError):
                raise ValueError(f"camera {i + 1}: K and R must be 3x3 and t 3 numbers") from None
            if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(t).all()):
                raise ValueError(f"camera {i + 1}: K, R and t must be finite numbers")
            Ks.append(K)
            Rs.append(R)
            ts.append(t)
        centres = np.array([-R.T @ t for R, t in zip(Rs, ts)])               # world positions
        axes = np.array([R.T @ np.array([0.0, 0.0, 1.0]) for R in Rs])       # viewing directions
        mean_axis = axes.mean(axis=0)
        mean_axis /= max(np.linalg.norm(mean_axis), 1e-12)
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        baseline = float(d.max()) if d.max() > 1e-9 else 1.0
        shift = centres.mean(axis=0) + baseline * mean_axis                   # new origin, old frame
        # X' = X - shift  =>  x_cam = R X + t = R X' + (t + R shift)
        out = []
        for K, R, t, c in zip(Ks, Rs, ts, cams):
            t2 = t + R @ shift
            if abs(t2[2]) < 1e-9:
                raise ValueError("a camera's principal plane passes through the chosen origin; "
                                 "move the cameras' frame slightly and try again")
            coefs = dlt_from_camera(K, R, t2)
            dist = c.get("dist")
            und: object = NoUndistort()
            if dist is not None and np.any(np.abs(np.asarray(dist, np.float64)) > 0):
                und = OpenCVUndistort(K, np.asarray(dist, np.float64), bool(c.get("fisheye", False)))
            out.append(CameraCalibration(coefs, int(c.get("width", 0)), int(c.get("height", 0)),
                                         und, pixel_origin=0.0, y_flip=False))
        cal = Calibration(out, unit=unit, source=source or "K + R/t cameras")
        cal.origin_shift = shift
        cal.source = (f"{cal.source} (origin moved to {shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f} "
                      f"{unit or 'units'} from the file's origin)")
        return cal

    @staticmethod
    def load_krt(path: str | Path) -> "Calibration":
        """A K + R/t calibration file. Two forms are read:

        * JSON: {"cameras": [{"K": [[..]], "R": [[..]], "t": [..], "dist": [..],
          "width": w, "height": h}, ...], "unit": "m"}. A camera without R / t is
          at the origin.
        * plain text as OpenCV stereo tools print it: `K1=[...]`, `K2=[...]`
          (3x3, rows on lines) and one 4x4 `T...=[...]` (camera 1 -> camera 2),
          or `R=[3x3]` and `t=[3]`. Unicode minus signs and arrows are fine.
          Optional `D1=[...]`, `D2=[...]` with 5 (or 8, 12, 14) OpenCV standard
          coefficients are applied; anything ambiguous is left out and named in
          `notes`.
        """
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        unit = ""
        cams: list[dict] = []
        notes: list[str] = []
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                d = json.loads(stripped)
                unit = str(d.get("unit", ""))
                for c in d.get("cameras", []):
                    cams.append(dict(c))
            except (ValueError, AttributeError, TypeError) as e:        # (I227)
                raise ValueError(f"{p.name}: not a K + R/t JSON file ({e}); expected "
                                 '{"cameras": [{"K": [[..]], "R": [[..]], "t": [..]}, ...]}') from None
        else:
            import re
            norm = (text.replace("−", "-").replace("→", "->").replace("\t", " ")
                    .replace("​", ""))
            blocks = re.findall(r"([A-Za-z][A-Za-z0-9_>\-]*)\s*=\s*\[([^\]]*)\]", norm)
            mats: dict[str, np.ndarray] = {}
            for name, body in blocks:
                nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", body)
                if nums:
                    mats[name] = np.asarray([float(v) for v in nums], np.float64)
            ks = sorted((k for k in mats if k.upper().startswith("K") and mats[k].size == 9),
                        key=lambda s: s)
            if len(ks) < 2:
                raise ValueError(f"{p.name}: expected at least two 3x3 K matrices (K1=[...], K2=[...])")
            T4 = next((mats[k].reshape(4, 4) for k in mats if k.upper().startswith("T") and mats[k].size == 16), None)
            R3 = next((mats[k].reshape(3, 3) for k in mats if k.upper().startswith("R") and mats[k].size == 9), None)
            t3 = next((mats[k].reshape(3) for k in mats if k.lower().startswith("t") and mats[k].size == 3), None)
            if T4 is None and (R3 is None or t3 is None):
                raise ValueError(f"{p.name}: expected a 4x4 T matrix or R (3x3) and t (3)")
            R2, t2 = (T4[:3, :3], T4[:3, 3]) if T4 is not None else (R3, t3)
            cams = [{"K": mats[ks[0]].reshape(3, 3)}, {"K": mats[ks[1]].reshape(3, 3), "R": R2, "t": t2}]
            m = re.search(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", norm)
            if m:
                for c in cams:
                    c["width"], c["height"] = int(m.group(1)), int(m.group(2))
            # (I96) distortion lines (`D1=[k1 k2 p1 p2 k3]`, `D2=[...]`) used to be
            # dropped in silence. 5 / 8 / 12 / 14 numbers can only be OpenCV's
            # standard model; 4 could be that model without k3 OR the fisheye
            # model, so those are not guessed at -- the note says so.
            ds = sorted(k for k in mats if k.upper().startswith("D") and mats[k].size in (4, 5, 8, 12, 14))
            if ds:
                if len(ds) >= 2 and all(mats[k].size != 4 for k in ds[:2]):
                    for c, k in zip(cams, ds[:2]):
                        c["dist"] = mats[k]
                else:
                    notes.append(f"The file's distortion line(s) ({', '.join(ds)}) were NOT applied: "
                                 + ("4 numbers could be OpenCV's standard model (k1 k2 p1 p2) or its fisheye "
                                    "model, and the file does not say which." if any(mats[k].size == 4 for k in ds)
                                    else "expected one line per camera.")
                                 + " The tracks are used as filmed; give the cameras as JSON with 'dist' "
                                 "(and 'fisheye': true where it applies) to include them.")
        try:
            cal = Calibration.from_krt(cams, unit=unit, source=p.name)
        except ValueError as e:                                           # (I227) name the file
            raise ValueError(f"{p.name}: {e}") from None
        cal.notes = notes
        return cal

    # ---- a world the user chose (I170) -----------------------------------
    def reframed(self, B: np.ndarray, o: np.ndarray, note: str = "", world: dict | None = None) -> "Calibration":
        """The same cameras in another world: X_old = B^T X_new + o, i.e.
        X_new = B (X_old - o) -- B's rows are the new axes in the old
        coordinates (`world_axes`), o the new origin in the old ones. Every
        DLT becomes P_new = P_old [[B^T, o], [0 0 0 1]], normalised so
        P[2, 3] = 1 again; the undistortion, the pixel convention, the picture
        sizes and the RMSEs are untouched, so every camera projects every
        point to the same pixel (to rounding) before and after. `origin_shift`
        is dropped (the new origin is the user's, not the file's). `note` is
        recorded in `source` (replacing an earlier world note), which the
        project file already saves; `world` (G201) = the landmarks that made it, kept as `self.world`."""
        B = np.asarray(B, np.float64).reshape(3, 3)
        o = np.asarray(o, np.float64).reshape(3)
        T = np.eye(4)
        T[:3, :3] = B.T
        T[:3, 3] = o
        cams = []
        for k, cam in enumerate(self.cameras):
            if cam is None:                     # (I273) not calibrated: stays so
                cams.append(None)
                continue
            P = dlt_matrix(cam.coefs) @ T
            den = float(P[2, 3])
            if abs(den) <= 1e-9 * float(np.linalg.norm(P[2, :3])) * max(1.0, float(np.linalg.norm(o))):
                raise ValueError(f"camera {k + 1}'s image plane passes through the chosen origin: choose an "
                                 "origin that is in front of every camera (on the floor of the working volume)")
            P = P / den
            cams.append(CameraCalibration(np.concatenate([P[0], P[1], P[2, :3]]), cam.width, cam.height,
                                          cam.undistort, cam.pixel_origin, cam.y_flip, cam.rmse))
        src = re.sub(r"\s*\(world: .*\)\s*$", "", self.source or "")
        out = self.with_cameras(cams, f"{src} ({note})".strip() if note else src)
        out.origin_shift = None
        out.world = None if world is None else {k: str(v) for k, v in world.items()}
        return out

    # ---- arrays form (the project file stores it as calibration.json) -----
    def to_arrays(self, prefix: str = "calib_") -> dict:
        """(I273) A camera that is not calibrated is a NaN row of coefficients and a null in "cams";
        not-calibrated cameras at the end are left out (a calibration of the first k cameras is
        written as it always was)."""
        cams = list(self.cameras)
        while cams and cams[-1] is None:
            cams.pop()
        if not cams:
            return {}
        meta = {"unit": self.unit, "source": self.source,
                "cams": [None if c is None else
                         {"width": c.width, "height": c.height, "pixel_origin": c.pixel_origin,
                          "y_flip": c.y_flip, "rmse": c.rmse, "undistort": c.undistort.to_json()}
                         for c in cams]}
        if self.origin_shift is not None:
            meta["origin_shift"] = np.asarray(self.origin_shift, np.float64).tolist()
        if self.world:
            meta["world"] = dict(self.world)                                   # (G201)
        return {f"{prefix}coefs": np.stack([np.full(11, np.nan) if c is None else c.coefs
                                            for c in cams]).astype(np.float64),
                f"{prefix}meta": json.dumps(meta)}

    @staticmethod
    def from_arrays(z, prefix: str = "calib_") -> "Calibration | None":
        have = set(getattr(z, "files", z))
        if f"{prefix}coefs" not in have:
            return None
        coefs = np.asarray(z[f"{prefix}coefs"], np.float64).reshape(-1, 11)
        meta = json.loads(str(z[f"{prefix}meta"])) if f"{prefix}meta" in have else {}
        cams = []
        for i, L in enumerate(coefs):
            d = (meta.get("cams") or [{}] * len(coefs))[i] if i < len(meta.get("cams", [])) else {}
            if d is None or not np.isfinite(L).any():      # (I273) a camera that is not calibrated
                cams.append(None)
                continue
            cams.append(CameraCalibration(L.copy(), int(d.get("width", 0)), int(d.get("height", 0)),
                                          undistort_from_json(d.get("undistort", {})),
                                          float(d.get("pixel_origin", 0.0)), bool(d.get("y_flip", False)),
                                          float(d.get("rmse", np.nan))))
        cal = Calibration(cams, str(meta.get("unit", "")), str(meta.get("source", "")))
        if meta.get("origin_shift") is not None:
            cal.origin_shift = np.asarray(meta["origin_shift"], np.float64).reshape(3)
        w = meta.get("world")
        if isinstance(w, dict) and all(isinstance(w.get(k), str) for k in ("origin", "x", "y")):
            cal.world = {k: str(v) for k, v in w.items()}                      # (G201)
        return cal


def spread_calibration(cal, views: list[int], n_views: int):
    """(I272) A calibration of the cameras `views` (in that order) as one of an `n_views` project:
    camera views[k] gets cal.cameras[k], every other camera None (not calibrated). Every camera
    calibrated = `cal` itself. (Here since I271: the wand timing checks spread a run's calibration
    without the wizard's Qt.)"""
    views = list(views)
    if views == list(range(n_views)):
        return cal
    cams = [None] * n_views
    for k, v in enumerate(views):
        cams[v] = cal.cameras[k]
    return cal.with_cameras(cams)


# -------------------------------------------------------- sub-frame timing


def local_frame(t: float | np.ndarray, rate: float, offset: float):
    """Reference instant (reference frames, may be fractional) -> this view's
    local frame number (fractional)."""
    return rate * np.asarray(t, np.float64) + offset


# ------------------------------------------------------------ project-level


@dataclass
class Reconstruction:
    """3D output over a stretch of reference frames."""
    t0: int                        # first reference frame
    names: list[str]               # landmark names (columns)
    xyz: np.ndarray                # (T, N, 3) NaN where fewer than min_cams saw it
    residual: np.ndarray           # (T, N) DLTdv-style residual, NaN where no solution
    n_cams: np.ndarray             # (T, N) int
    unit: str = ""
    # (T, N, C) reprojection error of EACH camera in its own calibration-frame
    # pixels, NaN where that camera did not see the landmark: which camera
    # disagrees, not just by how much the set disagrees
    per_cam: np.ndarray | None = None
    # (I305) "tracks" = triangulated from the project's 2D data (kept up to date with every change);
    # "imported" = read from a file (File > Import 3D points): left alone until Ctrl+3
    source: str = "tracks"

    @property
    def n_frames(self) -> int:
        return self.xyz.shape[0]

    def index(self, t) -> int | None:
        """(R31) The row of reference frame `t`, or None outside the result."""
        k = int(t) - int(self.t0)
        return k if 0 <= k < self.n_frames else None

    def points_at(self, t) -> np.ndarray:
        """(R31) The landmarks solved at reference frame `t`, (M, 3): rows without a position left
        out, none outside the result. (A row that must line up with `names` is `xyz[index(t)]`.)"""
        k = self.index(t)
        if k is None:
            return self.xyz[:0].reshape(0, 3)
        row = self.xyz[k]
        return row[np.isfinite(row).all(axis=1)]

    def reframed(self, B: np.ndarray, o: np.ndarray) -> "Reconstruction":
        """The same 3D result in the world `Calibration.reframed(B, o)` makes:
        X_new = B (X_old - o). Residuals, camera counts and the per-camera
        errors are pixel quantities and do not change."""
        B = np.asarray(B, np.float64).reshape(3, 3)
        o = np.asarray(o, np.float64).reshape(3)
        xyz = (np.asarray(self.xyz, np.float64) - o) @ B.T          # NaN stays NaN
        return Reconstruction(self.t0, list(self.names), xyz, self.residual.copy(), self.n_cams.copy(),
                              self.unit, None if self.per_cam is None else self.per_cam.copy(), self.source)

    def export_csv(self, path: str | Path) -> None:
        """DLTdv-style `xyzpts` CSV (`pt1_X,pt1_Y,pt1_Z,...`) with a leading
        `frame` column (reference frames), NaN where unsolved; plus a
        `*_xyzres.csv` sidecar with the residual and camera count."""
        from kinetrace.session import _sanitize
        p = Path(path)
        # (I95) header cells through the same sanitizer as the 2D exports: a
        # landmark named "wing tip, left" would otherwise add a header cell and
        # shift every later landmark's columns under the wrong names
        names = [_sanitize(nm) for nm in self.names]
        hdr = "frame," + ",".join(f"{nm}_{ax}" for nm in names for ax in ("X", "Y", "Z"))
        lines = [hdr]
        for i in range(self.n_frames):
            cells = [str(self.t0 + i)]
            for j in range(len(self.names)):
                v = self.xyz[i, j]
                cells += ([f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}"] if np.isfinite(v).all()
                          else ["NaN", "NaN", "NaN"])
            lines.append(",".join(cells))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        side = p.with_name(p.stem + "_xyzres.csv")
        pc = self.per_cam
        C = int(pc.shape[2]) if pc is not None and pc.ndim == 3 else 0
        hdr = "frame," + ",".join(f"{nm}_res,{nm}_ncams" + "".join(f",{nm}_cam{c + 1}_px" for c in range(C))
                                  for nm in names)
        lines = [hdr]
        for i in range(self.n_frames):
            cells = [str(self.t0 + i)]
            for j in range(len(self.names)):
                r = self.residual[i, j]
                cells += [f"{r:.4f}" if np.isfinite(r) else "NaN", str(int(self.n_cams[i, j]))]
                for c in range(C):
                    e = pc[i, j, c]
                    cells.append(f"{e:.3f}" if np.isfinite(e) else "NaN")
            lines.append(",".join(cells))
        side.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


# ------------------------------------------------- the world the user chooses (I170)

WORLD_AXES_MIN_DEG = 5.0     # the +Y point must leave the X axis by at least this angle


def world_is_left_handed(cal: "Calibration", probe: np.ndarray | None = None) -> bool:
    """True when the calibration's world is LEFT-handed (easyWand / DLTdv
    coefficients usually are): the majority of its cameras, in OpenCV's pixel
    convention with depth positive in front, have det(P[:, :3]) < 0. The one
    test `calibio.to_models` uses to export such a world mirrored in Z."""
    if not cal.views():
        return False
    probe = probe if probe is not None else working_probe(cal)
    dets = []
    for cam in cal.calibrated_cameras():
        P = opencv_projection(cam, probe)
        dets.append(np.linalg.det(P[:, :3]))
    return sum(d < 0 for d in dets) > len(dets) / 2


def opencv_projection(cam: CameraCalibration, probe) -> np.ndarray:
    """(R30) The camera's 3 x 4 projection matrix with depth (its third row) positive in front:
    what `world_is_left_handed` and `calibio.to_models` read K, R, t from."""
    return dlt_matrix(cam.coefs) * front_sign(cam.coefs, probe)


def world_axes(origin, px, py, left_handed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The user's world from three points of the CURRENT world: `origin`, a
    point on the +X axis and a point in the direction of +Y. Returns (B, o)
    for `Calibration.reframed` / `Reconstruction.reframed` -- B's rows are
    the new X, Y and Z axes in the old coordinates, o the origin. +X points
    from the origin to `px`, +Y is `py` made perpendicular to it, and +Z
    follows the RIGHT-HAND rule of the PHYSICAL world: ex x ey, negated when
    the calibration's world is left-handed (`world_is_left_handed`), because
    a mirrored world's cross product points the wrong way round. ValueError
    in a plain sentence when the points coincide or lie (nearly) on one
    line."""
    o = np.asarray(origin, np.float64).reshape(3)
    a = np.asarray(px, np.float64).reshape(3) - o
    b = np.asarray(py, np.float64).reshape(3) - o
    if not (np.isfinite(o).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("every one of the three points needs a position in 3D")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12:
        raise ValueError("the origin and the point on the +X axis are the same place: choose two different points")
    if nb < 1e-12:
        raise ValueError("the origin and the point in the +Y direction are the same place: choose two different points")
    ex = a / na
    rest = b - (b @ ex) * ex
    nr = float(np.linalg.norm(rest))
    if nr / nb < np.sin(np.radians(WORLD_AXES_MIN_DEG)):
        raise ValueError("the origin, the +X point and the +Y point lie on (nearly) one line, so they do not "
                         "say which way +Y is: choose a +Y point clearly off the X axis (more than "
                         f"{WORLD_AXES_MIN_DEG:g} degrees)")
    ey = rest / nr
    ez = np.cross(ex, ey)
    if left_handed:
        ez = -ez
    return np.stack([ex, ey, ez]), o


def _closest_approach(c1, d1, c2, d2):
    """Depths (s, t) along two rays c1 + s d1, c2 + t d2 at their closest approach,
    and the distance between those points."""
    w = c1 - c2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d, e = float(d1 @ w), float(d2 @ w)
    den = a * c - b * b
    if abs(den) < 1e-12:
        return 0.0, 0.0, float("inf")
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    gap = float(np.linalg.norm((c1 + s * d1) - (c2 + t * d2)))
    return s, t, gap


PROBE_MAX_BASELINES = 50.0   # a principal-ray crossing farther than this is "parallel", not a target


def working_probe(cal: "Calibration") -> np.ndarray | None:
    """A world point the cameras are looking at, from the calibration alone,
    used as the `probe` for `dlt_ray` when there is no reconstruction yet.

    (I98) The forward direction is taken from the WORLD ORIGIN: every
    calibration tool puts it in the working volume (a calibration frame's
    corner, the wand's centroid, easyWand's axes points; `from_krt` moves it
    to the cameras' look-at point), and the DLT denominator there is exactly 1
    for every camera, so "in front" is "denominator > 0". The probe is where
    the first two calibrated cameras' forward principal rays come closest -- when they
    converge IN FRONT and within `PROBE_MAX_BASELINES` baselines; otherwise
    (a parallel or toed-out stereo bar) the origin itself. The old rule tried
    all four ray orientations and kept whichever converged: on a toed-out pair
    that is a point BEHIND both cameras, on a near-parallel pair one hundreds
    of metres away, and the epipolar guides missed by ~200 px.

    The origin assumption is checked: sign(det M) = handedness x (origin in
    front), so it is the same for every camera exactly when the origin is in
    front of all of them (one calibration has one handedness). A mixed sign
    means the origin is behind some camera, and the old search -- which needs
    no forward direction but fails on parallel / toed-out pairs -- is used."""
    cams = [c for c in cal.cameras if c is not None] if cal is not None else []    # (I273) calibrated only
    if len(cams) < 2:
        return None
    cached = getattr(cal, "_probe_cache", None)
    if cached is not None:
        return cached
    origin = np.zeros(3)
    dets = {float(np.sign(np.linalg.det(np.array([L[0:3], L[4:7], L[8:11]]))))
            for L in (np.asarray(c.coefs, np.float64).reshape(11) for c in cams)}
    origin_ok = len(dets) == 1
    rays = []
    for cam in cams[:2]:
        w = cam.width or 1920
        h = cam.height or 1080
        pp = cam.to_calib_frame(np.array([[w / 2.0, h / 2.0]]))
        c, d = dlt_ray(cam.coefs, pp, probe=origin if origin_ok else None)
        rays.append((c, d[0]))
    baseline = float(np.linalg.norm(rays[0][0] - rays[1][0]))
    probe = origin if origin_ok else None
    if origin_ok:
        s, t, gap = _closest_approach(rays[0][0], rays[0][1], rays[1][0], rays[1][1])
        if (np.isfinite(gap) and s > 0 and t > 0 and baseline > 1e-12
                and max(s, t) <= PROBE_MAX_BASELINES * baseline):
            probe = np.asarray(rays[0][0] + s * rays[0][1], np.float64)
    else:
        best = None
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                s, t, gap = _closest_approach(rays[0][0], s1 * rays[0][1], rays[1][0], s2 * rays[1][1])
                if s > 0 and t > 0 and (best is None or gap < best[0]):
                    best = (gap, rays[0][0] + s * s1 * rays[0][1])
        probe = None if best is None else np.asarray(best[1], np.float64)
    try:
        cal._probe_cache = probe
    except Exception:       # noqa: BLE001
        pass
    return probe


def calib_frame_ok(cam: CameraCalibration, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw pixels -> (the undistorted calibration frame (N, 2), (N,) bool where the lens model round-trips
    them within max(1, 0.001 x width) px): past its boards a lens model can run away, and those points
    must not be used (the picture's box, the field of view of G186)."""
    raw = np.asarray(raw, np.float64).reshape(-1, 2)
    q = cam.to_calib_frame(raw)
    back = cam.from_calib_frame(q)
    tol = round_trip_tol(cam.width)
    ok = np.isfinite(q).all(axis=1) & np.isfinite(back).all(axis=1)
    ok[ok] &= np.linalg.norm(back[ok] - raw[ok], axis=1) < tol
    return q, ok


def _picture_in_calib_frame(cam: CameraCalibration, margin: float) -> tuple:
    """((x0, y0, x1, y1) of the picture plus `margin` in raw Kinetrace pixels,
    (x0, y0, x1, y1) of a box holding it in the undistorted calibration frame).
    With a lens model the picture's outline is bent there, so the box is that
    of its undistorted border -- the points of it the model round-trips (a
    model that runs away past its boards must not blow the box up); the exact
    cut is made afterwards in raw pixels."""
    key = (float(margin), cam.width, cam.height, id(cam.undistort))
    cached = getattr(cam, "_picture_box_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    mx, my = margin * cam.width, margin * cam.height
    raw = (-0.5 - mx, -0.5 - my, cam.width - 0.5 + mx, cam.height - 0.5 + my)
    x0, y0, x1, y1 = raw
    if not cam.has_lens_correction:
        q = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    else:
        t = np.linspace(0.0, 1.0, 64)
        border = np.vstack([np.column_stack([x0 + t * (x1 - x0), np.full_like(t, y0)]),
                            np.column_stack([x0 + t * (x1 - x0), np.full_like(t, y1)]),
                            np.column_stack([np.full_like(t, x0), y0 + t * (y1 - y0)]),
                            np.column_stack([np.full_like(t, x1), y0 + t * (y1 - y0)])])
        q, ok = calib_frame_ok(cam, border)
        q = q[ok] if ok.sum() >= 8 else border
    box = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
    out = (raw, box)
    try:
        cam._picture_box_cache = (key, out)
    except Exception:       # noqa: BLE001
        pass
    return out


def _clip_s(lo: float, hi: float, a: float, b: float) -> tuple[float, float]:
    """Narrow the ray parameter interval [lo, hi] to where a + s * b >= 0."""
    if abs(b) < 1e-300:
        return (lo, hi) if a >= 0 else (1.0, 0.0)
    r = -a / b
    return (max(lo, r), hi) if b > 0 else (lo, min(hi, r))


def _clip_polyline_to_box(pts: np.ndarray, flags: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """The longest run of polyline `pts` inside `box` (x0, y0, x1, y1), cut
    exactly where it crosses the box's edges; `flags` travel with the points."""
    x0, y0, x1, y1 = box
    eps = 1e-6
    ins = ((pts[:, 0] >= x0 - eps) & (pts[:, 0] <= x1 + eps)
           & (pts[:, 1] >= y0 - eps) & (pts[:, 1] <= y1 + eps))

    def exit_point(a, b):
        # from `a` (inside) towards `b` (outside): where the segment leaves the box
        d = b - a
        t = 1.0
        for k, lo_, hi_ in ((0, x0, x1), (1, y0, y1)):
            if d[k] > 1e-12:
                t = min(t, (hi_ - a[k]) / d[k])
            elif d[k] < -1e-12:
                t = min(t, (lo_ - a[k]) / d[k])
        return a + max(t, 0.0) * d

    runs, cur, curf = [], [], []
    for i in range(len(pts)):
        if ins[i]:
            if not cur and i > 0:
                cur.append(exit_point(pts[i], pts[i - 1]))
                curf.append(flags[i])
            cur.append(pts[i])
            curf.append(flags[i])
        elif cur:
            cur.append(exit_point(pts[i - 1], pts[i]))
            curf.append(flags[i - 1])
            runs.append((cur, curf))
            cur, curf = [], []
    if cur:
        runs.append((cur, curf))
    if not runs:
        return np.zeros((0, 2)), np.zeros(0, bool)
    arc = [float(np.linalg.norm(np.diff(np.asarray(r), axis=0), axis=1).sum()) if len(r) > 1 else 0.0
           for r, _f in runs]
    r, f = runs[int(np.argmax(arc))]
    return np.asarray(r, np.float64), np.asarray(f, bool)


def epipolar_polyline(src: CameraCalibration, dst: CameraCalibration, uv_raw_src, probe=None,
                      n: int = 240, margin: float = 0.25, info: dict | None = None) -> np.ndarray:
    """Where a landmark seen at raw pixel `uv_raw_src` of camera `src` can be
    in camera `dst`: its viewing ray, projected into `dst` as a polyline in
    DST RAW pixels (M, 2), ordered from the near end (the epipole side) to the
    far end (the vanishing point side) -- a polyline, not a line, because
    `dst`'s lens undistortion bends it. Only the part in front of BOTH cameras
    is kept, cut at `dst`'s picture plus `margin` of its size (0 = exactly at
    the picture's edges, what the guides draw). Empty when none of it lands
    there. `probe` is a point the cameras look at (the forward direction of
    each camera; I98).

    (I132) Computed exactly: in `dst`'s undistorted plane the ray's image is
    the straight segment from the epipole (depth 0) to the vanishing point
    (infinite depth); the depths in front of `dst` and inside the picture are
    each a linear condition on the ray parameter, so the kept part is found
    in closed form, then sampled evenly ALONG THE PICTURE and bent through
    the lens. The old version sampled 80 depths from 1/50 to 50 times the
    probe distance and joined them, so it stopped at its last sample: short
    of the picture edge, or of the vanishing point on a long baseline.
    `info`, if given, receives {"trusted": (M,) bool} -- False where the lens
    model does not invert consistently (it is guessing there)."""
    p = src.to_calib_frame(np.asarray(uv_raw_src, np.float64).reshape(1, 2))
    if info is not None:
        info["trusted"] = np.zeros(0, bool)
    if not np.isfinite(p).all():
        return np.zeros((0, 2))
    c, d = dlt_ray(src.coefs, p, probe)
    d = d[0]
    if dst.width and dst.height:
        L = np.asarray(dst.coefs, np.float64).reshape(11)
        P = dlt_matrix(L)
        xc, xd = P @ np.append(c, 1.0), P @ np.append(d, 0.0)       # x(s) = xc + s xd, s >= 0
        sgn = front_sign(L, probe)
        lo, hi = _clip_s(0.0, np.inf, sgn * xc[2], sgn * xd[2])      # in front of dst
        raw_box, (bx0, by0, bx1, by1) = _picture_in_calib_frame(dst, margin)
        # inside the box: sgn * (h . x(s)) >= 0 per edge (w keeps the sign sgn in front)
        for h in ((1.0, 0.0, -bx0), (-1.0, 0.0, bx1), (0.0, 1.0, -by0), (0.0, -1.0, by1)):
            h = np.asarray(h)
            lo, hi = _clip_s(lo, hi, sgn * float(h @ xc), sgn * float(h @ xd))
        if not lo < hi:
            return np.zeros((0, 2))
        q0 = (xc + lo * xd)[:2] / (xc + lo * xd)[2]
        q1 = xd[:2] / xd[2] if not np.isfinite(hi) else (xc + hi * xd)[:2] / (xc + hi * xd)[2]
        kind = getattr(dst.undistort, "kind", "none")
        # a bent line is smooth: a quarter of the samples draws it to a fraction of a
        # pixel, and an LWM map costs O(samples x control points) -- 25 ms a line at 240
        m = max(int(n), 2) if kind == "none" else max(int(n) // 4, 16)
        t = np.linspace(0.0, 1.0, m)
        und = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
        uv = dst.from_calib_frame(und)
        ok = np.isfinite(uv).all(axis=1)         # an LWM map is undefined (NaN) off its control points
        uv, und = uv[ok], und[ok]
        trusted = np.ones(len(uv), bool)
        if kind == "opencv" and len(uv):
            # a polynomial lens model extrapolates past its boards and can run away;
            # where it no longer inverts consistently it is guessing
            back = dst.to_calib_frame(uv)
            tol = max(0.5, PROJECTION_TRUST_TOL * dst.width)
            trusted = np.isfinite(back).all(axis=1)
            trusted[trusted] &= np.linalg.norm(back[trusted] - und[trusted], axis=1) < tol
        uv, trusted = _clip_polyline_to_box(uv, trusted, raw_box)
        if info is not None:
            info["trusted"] = trusted
        return uv
    # no picture size recorded: sample depths as before (a few decades around the probe)
    ref = float(np.linalg.norm(np.asarray(probe, np.float64) - c)) if probe is not None else 1.0
    ref = ref if np.isfinite(ref) and ref > 1e-9 else 1.0
    depths = ref * np.logspace(np.log10(0.02), np.log10(50.0), int(n))
    xyz = c[None, :] + depths[:, None] * d[None, :]
    if probe is not None:
        sgn = front_sign(dst.coefs, probe)
        L = np.asarray(dst.coefs, np.float64)
        den = xyz @ L[8:11] + 1.0
        xyz = xyz[np.sign(den) == sgn]              # only points in front of dst
        if len(xyz) == 0:
            return np.zeros((0, 2))
    uv = dst.project(xyz)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if info is not None:
        info["trusted"] = np.ones(len(uv), bool)
    return uv


def closest_on_polyline(pts: np.ndarray, p) -> np.ndarray | None:
    """The point of polyline `pts` (M, 2) nearest to `p`."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    p = np.asarray(p, np.float64).reshape(2)
    if len(pts) == 0:
        return None
    if len(pts) == 1:
        return pts[0]
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    q = a + t[:, None] * ab
    return q[np.argmin(np.linalg.norm(q - p, axis=1))]


def even_indices(n: int, k: int) -> np.ndarray:
    """(R33) `k` indices spread evenly over 0 .. n - 1, ends included (k <= n): the frames, boards
    or instants a long list is thinned to."""
    return np.linspace(0, n - 1, k).round().astype(int)


def widest_angle_deg(U: np.ndarray) -> float:
    """(R32) The widest angle in degrees between unit vectors (rows of U), sign-free: two lines'
    normals or two cameras' rays. Under `PARALLEL_LINES_DEG` they do not "cross"."""
    return float(np.degrees(np.arccos(np.abs(np.clip(U @ U.T, -1.0, 1.0)).min())))


def round_trip_tol(width: float) -> float:
    """(R32) How far (px) a raw pixel may come back from the lens model's undistort -> distort and
    still count as described by it: max(1, 0.001 x the picture width)."""
    return max(1.0, 1e-3 * width)


PROJECTION_TRUST_TOL = 5e-4       # (R32) x width, at least 0.5 px: a projected point the lens still inverts
PIXEL_CONVENTIONS = {"matlab": (1.0, False), "bottom-left": (1.0, True), "opencv": (0.0, False)}   # (R32)
PARALLEL_LINES_DEG = 5.0    # epipolar lines closer than this in angle do not "cross" (THE 5-degree rule:
                            # intersect_polylines, the app's prediction diamond and the guides' "parallel" wording)


def intersect_polylines(lines: list[np.ndarray], near, info: dict | None = None) -> np.ndarray | None:
    """Least-squares intersection of several epipolar polylines, each taken as
    the straight line through its segment nearest to `near`. With one line
    the closest point on it is returned; with none, None.

    (I97) When the lines are (nearly) parallel -- every pair within
    `PARALLEL_LINES_DEG` -- their crossing is not defined by the geometry but
    by noise: on a rig whose camera centres are collinear (three cameras on
    one bar) every other camera's line in the middle view is the SAME line,
    and the old least-squares answer landed 400-1000 px away, at the picture
    edge, as a hand-placed point. Then the point is only moved ONTO the lines
    (the mean of the closest points to `near`), never along them. `info`, if
    given, receives {"mode": "one" | "crossing" | "parallel", "angle_deg":
    the widest angle between two of the lines}."""
    near = np.asarray(near, np.float64).reshape(2)
    cands = [np.asarray(l, np.float64).reshape(-1, 2) for l in lines if l is not None and len(l) >= 1]
    if info is not None:
        info.update({"mode": "one", "angle_deg": 0.0})
    if not cands:
        return None
    if len(cands) == 1:
        return closest_on_polyline(cands[0], near)
    A, b, feet = [], [], []
    for pts in cands:
        if len(pts) < 2:
            continue
        q = closest_on_polyline(pts, near)
        k = int(np.argmin(np.linalg.norm(pts - q, axis=1)))
        k0, k1 = (k - 1, k) if k == len(pts) - 1 else (k, k + 1)
        d = pts[k1] - pts[k0]
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            continue
        nvec = np.array([-d[1], d[0]]) / nrm             # unit normal: n . x = n . q
        A.append(nvec)
        b.append(float(nvec @ q))
        feet.append(q)
    if len(A) < 2:
        return closest_on_polyline(cands[0], near)
    N = np.asarray(A)
    widest = widest_angle_deg(N)                   # between two lines = between their normals
    if info is not None:
        info.update({"angle_deg": widest})
    if widest < PARALLEL_LINES_DEG:
        if info is not None:
            info["mode"] = "parallel"
        return np.mean(np.asarray(feet), axis=0)
    if info is not None:
        info["mode"] = "crossing"
    x, *_ = np.linalg.lstsq(N, np.asarray(b), rcond=None)
    return x


TWO_CAMERA_CAP_FRAC = 0.9   # share of two-camera 3D points above which "good" is capped at "ok"


def residual_scale(width_px: float, height_px: float = 0.0) -> float:
    """How many 1080p pixels one pixel of this picture is worth for the
    verdict bands: max(w, h) / 1920, never below 1 (I250: the LONG side, so a
    portrait 4K clip is judged like a landscape one)."""
    return max(1.0, max(float(width_px), float(height_px or 0.0)) / 1920.0)


def residual_bands(width_px: float, height_px: float = 0.0) -> tuple[float, float]:
    """(good, ok) limits in pixels of a triangulation residual for a picture
    of this size: good <= 1.5 px, usable <= 5 px, poor above, at 1920 on the
    long side (R14: the one place those numbers live)."""
    s = residual_scale(width_px, height_px)
    return 1.5 * s, 5.0 * s


def cell_verdict(residual: float, n_cams: int, bands: tuple[float, float]) -> str:
    """good / ok / poor of one triangulated point from its residual (px) against `bands` = (good, ok)
    limits (`residual_bands`). Two rays are blind to a mistake along the line, so two cameras cap at
    "ok" as the Reconstruct report does (G83, I101). "" for no point."""
    if not np.isfinite(residual) or n_cams < 2:
        return ""
    good, usable = bands
    verdict = "good" if residual <= good else "ok" if residual <= usable else "poor"
    return "ok" if (n_cams == 2 and verdict == "good") else verdict


def reconstruction_report(rec: Reconstruction, n_views: int, width_px: float = 1920.0,
                          height_px: float = 0.0) -> dict:
    """A verdict a first-time user can act on, from the residuals alone.

    The residual is DLTdv's: the RMS distance, in pixels, between where each
    camera saw the landmark and where the 3D point projects back -- how much
    the cameras DISAGREE. Thresholds scale with the picture's long side (a
    pixel of 4K is half a pixel of 1080p; `residual_bands`): good <= 1.5 px,
    usable <= 5 px, poor above, at 1920 on the long side. A point made from only two rays is blind to one whole
    direction (a mistake along the epipolar line triangulates perfectly): with
    two cameras, or when at least `TWO_CAMERA_CAP_FRAC` of the points were
    seen by only two, the report says so and "good" becomes "ok". Returns
    {"verdict": good / ok / poor / empty, "reasons": [...], "median_px",
    "solved_frac", "two_camera_frac", "worst": (name, median_px),
    "per_landmark": {name: median_px}}."""
    res = np.asarray(rec.residual, np.float64)
    ok = np.isfinite(res)
    n_cells = int(res.size)
    solved = int(ok.sum())
    out = {"verdict": "empty", "reasons": [], "median_px": float("nan"),
           "solved_frac": (solved / n_cells) if n_cells else 0.0, "worst": None, "per_landmark": {}}
    if solved == 0:
        out["reasons"].append("No landmark was seen by two cameras at the same instant, so nothing could "
                              "be placed in 3D. Track the SAME named landmarks in at least two cameras "
                              "over the same stretch of frames.")
        return out
    med = float(np.nanmedian(res))
    out["median_px"] = med
    per = {}
    for j, nm in enumerate(rec.names):
        col = res[:, j]
        if np.isfinite(col).any():
            per[nm] = float(np.nanmedian(col))
    out["per_landmark"] = per
    worst = max(per.items(), key=lambda kv: kv[1]) if per else None
    out["worst"] = worst
    good_lim, ok_lim = residual_bands(width_px, height_px)
    if med <= good_lim:
        verdict = "good"
        out["reasons"].append(f"The cameras agree to {med:.2f} px (median): the 3D positions are "
                              "consistent with every camera that saw them.")
    elif med <= ok_lim:
        verdict = "ok"
        out["reasons"].append(f"The cameras disagree by {med:.2f} px (median). Usable, but check the "
                              "worst landmark below and the camera offsets (3D -> Sync -> Estimate "
                              "Sub-frame Offsets) before trusting fine detail.")
    else:
        verdict = "poor"
        out["reasons"].append(f"The cameras disagree by {med:.2f} px (median): the 3D is NOT trustworthy. "
                              "Usual causes, in order: a landmark tracked onto different body parts in "
                              "different cameras, a wrong frame offset between cameras, the wrong pixel "
                              "convention or camera order in the calibration, or a bad calibration.")
    frac = out["solved_frac"]
    out["reasons"].append(f"{solved} of {n_cells} landmark-frames were placed in 3D ({100 * frac:.0f}%)."
                          + ("" if frac >= 0.5 else " Most cells are empty: the cameras do not share "
                             "enough tracked frames, or the landmark names differ between cameras."))
    if worst is not None and len(per) > 1:
        out["reasons"].append(f"Worst landmark: {worst[0]} at {worst[1]:.2f} px median"
                              + (" -- look at its track in each camera." if worst[1] > ok_lim else "."))
    # (I101) The blind spot is a property of each 3D POINT (how many rays made
    # it), not of the project: a 3-camera project whose third camera tracked
    # nothing is a two-camera reconstruction. rec.n_cams holds the evidence.
    nc = np.asarray(rec.n_cams)[ok] if np.shape(rec.n_cams) == res.shape else np.zeros(0)
    frac2 = float(np.mean(nc <= 2)) if len(nc) else (1.0 if n_views <= 2 else 0.0)
    out["two_camera_frac"] = frac2
    if n_views <= 2 or frac2 >= TWO_CAMERA_CAP_FRAC:
        lead = ("Only two cameras" if n_views <= 2 else
                f"{100 * frac2:.0f}% of the 3D positions were seen by only two of the {n_views} cameras")
        out["reasons"].append(f"{lead}: a residual cannot see a mistake that lies along the line "
                              "joining the two cameras' views (it triangulates perfectly), so a low number "
                              "here is necessary but not sufficient. A third camera tracking the same "
                              "landmarks makes such mistakes visible.")
        if verdict == "good":
            verdict = "ok"
    elif frac2 > 0.5:
        # most points still come from two rays, but a third camera cross-checks
        # every landmark in part of the clip (a real 6-camera rig: 65 % two-camera);
        # a landmark NEVER seen by three cameras has no such check -- name it
        nc_all = np.asarray(rec.n_cams)
        unchecked = [nm for j, nm in enumerate(rec.names)
                     if np.isfinite(res[:, j]).any() and not (nc_all[:, j] >= 3).any()]
        out["reasons"].append(
            f"{100 * frac2:.0f}% of the 3D positions come from only two cameras, which cannot catch a "
            "mistake along the line joining their views; the frames where a third camera also saw the "
            "landmark are the cross-check."
            + (f" Never seen by three cameras (no cross-check at all): {', '.join(unchecked)}."
               if unchecked else ""))
    out["verdict"] = verdict
    return out


def calib_tracks(session, cal: CameraCalibration, cells: np.ndarray | None = None) -> np.ndarray:
    """One view's tracks converted to the calibration frame ONCE (vectorised
    undistortion of every tracked cell), NaN where untracked. Interpolating
    these at fractional frames is what DLTdv's time shift does too. `cells`
    (T, N) bool (I305): only those cells (the rest NaN) - the undistortion is
    per cell, so a cell comes out the same bits either way."""
    T, N = session.tracks.shape[:2]
    out = np.full((T, N, 2), np.nan)
    # (R36) the cells that leave the app (hand-marked hidden: never triangulated), with a position
    ok = session.exportable & np.isfinite(session.tracks).all(axis=2)
    if cells is not None:
        ok &= cells
    if ok.any():
        out[ok] = cal.to_calib_frame(session.tracks[ok].astype(np.float64))
    return out


def sample_tracks_at(ct: np.ndarray, local_t: np.ndarray) -> np.ndarray:
    """Calibration-frame tracks (T, N, 2) sampled at fractional local frames
    (M,) -> (M, N, 2): linear between the two neighbouring frames, NaN unless
    both hold data (a gap is never bridged)."""
    T = ct.shape[0]
    lt = np.asarray(local_t, np.float64)
    f0 = np.floor(lt)
    a = lt - f0
    f0i = f0.astype(np.int64)
    out = np.full((len(lt),) + ct.shape[1:], np.nan)
    exact = (a < 1e-9) & (f0i >= 0) & (f0i < T)
    if exact.any():
        out[exact] = ct[f0i[exact]]
    mid = (~exact) & (f0i >= 0) & (f0i + 1 < T)
    if mid.any():
        p0 = ct[f0i[mid]]
        p1 = ct[f0i[mid] + 1]
        w = a[mid][:, None, None]
        out[mid] = (1.0 - w) * p0 + w * p1        # NaN propagates when either is missing
    return out


def triangulate_batch(coefs: np.ndarray, uv: np.ndarray, min_cams: int = 2,
                      max_residual: float = float("inf")):
    """Vectorised `triangulate` over many cells. coefs (C, 11); uv (..., C, 2)
    with NaN where a camera did not see the point. Returns xyz (..., 3),
    residual (...), n_cams (...), per-camera errors (..., C). Cells with fewer
    than `min_cams` views are NaN. With `max_residual` the worst camera is
    dropped (once per pass, up to C - min_cams passes) while a cell's
    residual exceeds it — a cheap guard against one bad view."""
    Ls = np.asarray(coefs, np.float64)
    if Ls.ndim == 2 and Ls.shape[0] == 11 and Ls.shape[1] != 11:
        Ls = Ls.T
    C = Ls.shape[0]
    uv = np.asarray(uv, np.float64)
    lead = uv.shape[:-2]
    uvf = uv.reshape(-1, C, 2)
    use = np.isfinite(uvf).all(axis=2)                         # (M, C)
    M = uvf.shape[0]
    u = np.nan_to_num(uvf[..., 0])
    v = np.nan_to_num(uvf[..., 1])
    # rows of A and b for every camera, zero-weighted where unused
    A1 = np.stack([u * Ls[:, 8] - Ls[:, 0], u * Ls[:, 9] - Ls[:, 1], u * Ls[:, 10] - Ls[:, 2]], -1)
    A2 = np.stack([v * Ls[:, 8] - Ls[:, 4], v * Ls[:, 9] - Ls[:, 5], v * Ls[:, 10] - Ls[:, 6]], -1)
    b1 = Ls[:, 3] - u
    b2 = Ls[:, 7] - v
    xyz = np.full((M, 3), np.nan)
    resid = np.full(M, np.nan)
    err = np.full((M, C), np.nan)
    for _ in range(max(C - min_cams, 0) + 1):
        w = use.astype(np.float64)[..., None]                    # (M, C, 1)
        AtA = (np.einsum("mci,mcj->mij", A1 * w, A1) + np.einsum("mci,mcj->mij", A2 * w, A2))
        Atb = (np.einsum("mci,mc->mi", A1 * w, b1) + np.einsum("mci,mc->mi", A2 * w, b2))
        n = use.sum(1)
        ok = n >= min_cams
        X = np.full((M, 3), np.nan)
        if ok.any():
            # (I305) a cell's answer depends on that cell alone: a singular cell goes to `lstsq` by itself
            # (it used to send EVERY cell of the batch there), so a few cells solved on their own give the
            # same bits as the whole result
            try:
                X[ok] = np.linalg.solve(AtA[ok], Atb[ok][..., None])[..., 0]
            except np.linalg.LinAlgError:
                for m in np.nonzero(ok)[0]:
                    try:
                        X[m] = np.linalg.solve(AtA[m], Atb[m][..., None])[..., 0]
                    except np.linalg.LinAlgError:
                        X[m] = np.linalg.lstsq(AtA[m], Atb[m], rcond=None)[0]
        # (I305) element by element, not `X @ Ls.T`: a one-row matrix product takes another BLAS path and
        # changed a cell's error in the last bit, so a cell recomputed alone must not depend on the batch
        x0, x1, x2 = X[:, 0, None], X[:, 1, None], X[:, 2, None]
        den = x0 * Ls[:, 8] + x1 * Ls[:, 9] + x2 * Ls[:, 10] + 1.0          # (M, C)
        pu = (x0 * Ls[:, 0] + x1 * Ls[:, 1] + x2 * Ls[:, 2] + Ls[:, 3]) / den
        pv = (x0 * Ls[:, 4] + x1 * Ls[:, 5] + x2 * Ls[:, 6] + Ls[:, 7]) / den
        e = np.hypot(pu - u, pv - v)
        e[~use] = np.nan
        r = np.sqrt(np.nansum(e ** 2, axis=1) / np.maximum(2 * n - 3, 1))
        r[~ok] = np.nan
        xyz, resid, err = X, r, e
        bad = ok & (r > max_residual) & (n > min_cams)
        if not bad.any():
            break
        worst = np.nanargmax(np.where(use, e, -1.0), axis=1)
        use[bad, worst[bad]] = False
    n_cams = use.sum(1)
    n_cams[~np.isfinite(resid)] = 0
    return (xyz.reshape(lead + (3,)), resid.reshape(lead), n_cams.reshape(lead),
            err.reshape(lead + (C,)))


def landmark_names(sessions) -> list[str]:
    """The 3D result's columns: every landmark name of every view, in order of first appearance."""
    names: list[str] = []
    for s in sessions:
        for q in s.points:
            if q.name not in names:
                names.append(q.name)
    return names


class _Prepared:
    """Per-view calibration-frame tracks + name index, computed once per
    reconstruction so repeated calls (offset search) stay cheap."""

    def __init__(self, sessions, calib: Calibration, names: list[str] | None, cells: list | None = None):
        """`cells` (I305): per view None or a (T, N_view) bool of the cells to read (`calib_tracks`)."""
        if len(calib) > len(sessions):
            raise ValueError("calibration has a different number of cameras than the project")
        # (I273) a camera that is not calibrated (a gap, or past the end of the list) has no data
        # here: its tracks are all NaN and it never enters a triangulation
        cams = calib.padded(len(sessions))
        self.used = [c for c, cam in enumerate(cams) if cam is not None]
        self.n_views = len(sessions)
        self.names = list(landmark_names(sessions) if names is None else names)
        self.coefs = (np.stack([cams[c].coefs for c in self.used]) if self.used
                      else np.zeros((0, 11)))                          # the CALIBRATED cameras' rows
        self.ct = []                    # per view (T, N_names, 2) in calibration frame
        for v, (s, cal) in enumerate(zip(sessions, cams)):
            if cal is None:
                self.ct.append(np.full((s.n_frames, len(self.names), 2), np.nan))
                continue
            full = calib_tracks(s, cal, None if cells is None else cells[v])
            idx = {q.name: j for j, q in enumerate(s.points)}
            arr = np.full((full.shape[0], len(self.names), 2), np.nan)
            for j, nm in enumerate(self.names):
                if nm in idx:
                    arr[:, j] = full[:, idx[nm]]
            self.ct.append(arr)

    def triangulate(self, uv: np.ndarray, min_cams: int = 2, max_residual: float = float("inf")):
        """`triangulate_batch` over the calibrated cameras of `uv` (..., C views, 2); the per-camera
        errors come back with one column per view (NaN for a camera that is not calibrated). With
        every camera calibrated this IS `triangulate_batch(self.coefs, uv, ...)`."""
        if len(self.used) == self.n_views:
            return triangulate_batch(self.coefs, uv, min_cams, max_residual)
        xyz, resid, ncams, err_u = triangulate_batch(self.coefs, uv[..., self.used, :], min_cams, max_residual)
        err = np.full(err_u.shape[:-1] + (self.n_views,), np.nan, err_u.dtype)
        err[..., self.used] = err_u
        return xyz, resid, ncams, err

    def samples(self, rates, offsets, t: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
        """(T, N, C, 2) calibration-frame 2D at reference instants t. `keep`
        (T, N, C) bool restricts which view/cell pairs may contribute."""
        cols = [sample_tracks_at(ct, local_frame(t, rates[c], offsets[c]))
                for c, ct in enumerate(self.ct)]
        uv = np.stack(cols, axis=2)
        if keep is not None:
            uv[~keep] = np.nan
        return uv

    def stable_cells(self, rates, offsets, t: np.ndarray, search: float) -> np.ndarray:
        """(T, N, C) bool: view/cell pairs with data at the instant AND at
        ±`search` local frames around it — the set an offset search must be
        scored on, so that shifting an offset cannot change WHICH cells are
        counted (sparse or gappy tracks would otherwise let the optimiser
        lower the mean by dropping cells instead of improving them)."""
        ok = None
        for d in (-search, 0.0, search):
            o = [offsets[c] + (d if c else 0.0) for c in range(len(offsets))]
            cur = np.isfinite(self.samples(rates, o, t)).all(axis=3)
            ok = cur if ok is None else (ok & cur)
        return ok


def reconstruct(sessions: list, calib: Calibration, rates: list[float], offsets: list[float],
                t_range: tuple[int, int], names: list[str] | None = None,
                min_cams: int = 2, max_residual: float = float("inf"),
                _prep: "_Prepared | None" = None) -> Reconstruction:
    """Triangulate every landmark (matched BY NAME across views) at every
    integer reference frame in t_range (inclusive). Views without the
    landmark, or without data at that instant, simply drop out; a point needs
    `min_cams` views. `max_residual`: drop the worst camera while the residual
    exceeds it and >= min_cams remain (a cheap outlier guard)."""
    prep = _prep or _Prepared(sessions, calib, names)
    t0, t1 = int(t_range[0]), int(t_range[1])
    t = np.arange(t0, max(t1 + 1, t0))
    uv = prep.samples(rates, offsets, t)
    xyz, resid, ncams, err = prep.triangulate(uv, min_cams, max_residual)
    return Reconstruction(t0, prep.names, xyz, resid, ncams.astype(np.int32), calib.unit,
                          err.astype(np.float32))


MAX_SCAN_INSTANTS = 6000     # estimate_offsets looks at no more than this many instants at all ...
MAX_SCORE_INSTANTS = 500     # ... and scores the search on at most this many MOVING ones (G76)
ROBUST_CAP_K = 4.0           # estimate_offsets(robust=True): a cell counts at most this x the starting median (I271)
MIN_OWN_CELLS = 30           # (I302) a camera judged on fewer of its own scored cells reads at most "weak"


def reference_span(f_lo: int, f_hi: int, rate: float, offset: float) -> tuple[int, int]:
    """(I305) The reference instants whose sample (`sample_tracks_at` at `local_frame(t)`) can read a local
    frame in f_lo..f_hi: floor(rate t + offset) in f_lo - 1 .. f_hi, widened by one instant each side.
    Over-including only recomputes a cell to the same value."""
    lo = int(np.floor((f_lo - 1 - offset) / rate)) - 1
    hi = int(np.ceil((f_hi + 1 - offset) / rate)) + 1
    return lo, hi


def update_reconstruction(rec: "Reconstruction | None", sessions: list, calib: Calibration, rates: list[float],
                          offsets: list[float], t_range: tuple[int, int],
                          changed: dict[str, list[tuple[int, int]]]) -> Reconstruction:
    """(I305) `rec` brought up to date with the 2D data: the result `reconstruct(sessions, calib, rates,
    offsets, t_range)` would give, computed only where it can differ. Columns = `landmark_names(sessions)`;
    a column whose name `rec` has keeps its cells inside `rec`'s range except the reference instants in
    `changed[name]` (inclusive spans); every other cell (a new name, rows outside `rec`'s range) is
    computed, through the same `_Prepared.samples` / `triangulate` as `reconstruct` on the needed cells
    only. Each cell depends on that cell's data alone, so the result is the same bits (verify_3d,
    verify_auto3d). `rec` None or of another camera count: everything is computed."""
    names = landmark_names(sessions)
    t0, t1 = int(t_range[0]), int(t_range[1])
    T = max(t1 + 1, t0) - t0
    n_views = len(sessions)
    if (rec is None or rec.per_cam is None or rec.per_cam.ndim != 3 or rec.per_cam.shape[2] != n_views):
        return reconstruct(sessions, calib, rates, offsets, (t0, t1))
    N = len(names)
    xyz = np.full((T, N, 3), np.nan)
    resid = np.full((T, N), np.nan)
    ncams = np.zeros((T, N), np.int32)
    err = np.full((T, N, n_views), np.nan, np.float32)
    lo, hi = max(t0, rec.t0), min(t1, rec.t0 + rec.n_frames - 1)
    old_col = {nm: j for j, nm in enumerate(rec.names)}
    for k, nm in enumerate(names):
        need = np.ones(T, bool)
        if nm in old_col and lo <= hi:
            jo = old_col[nm]
            a, b, ao = lo - t0, hi - t0 + 1, lo - rec.t0
            xyz[a:b, k] = rec.xyz[ao:ao + b - a, jo]
            resid[a:b, k] = rec.residual[ao:ao + b - a, jo]
            ncams[a:b, k] = rec.n_cams[ao:ao + b - a, jo]
            err[a:b, k] = rec.per_cam[ao:ao + b - a, jo]
            need[a:b] = False
            for c_lo, c_hi in changed.get(nm, ()):
                need[min(max(c_lo - t0, 0), T):min(max(c_hi - t0 + 1, 0), T)] = True
        if not need.any():
            continue
        t = t0 + np.nonzero(need)[0]
        cells = []
        for v, s in enumerate(sessions):
            m = np.zeros(s.tracks.shape[:2], bool)
            jv = s.pid_by_name(nm)
            if jv is not None:
                f0 = np.floor(local_frame(t, rates[v], offsets[v])).astype(np.int64)
                rows = np.concatenate([f0, f0 + 1])
                m[rows[(rows >= 0) & (rows < m.shape[0])], jv] = True
            cells.append(m)
        prep = _Prepared(sessions, calib, [nm], cells)
        x, r, n, e = prep.triangulate(prep.samples(rates, offsets, t))
        rows = t - t0
        xyz[rows, k] = x[:, 0]
        resid[rows, k] = r[:, 0]
        ncams[rows, k] = n[:, 0]
        err[rows, k] = e[:, 0]
    return Reconstruction(t0, names, xyz, resid, ncams, calib.unit, err)


class _SearchCancelled(Exception):
    pass


def _scoring_instants(prep: "_Prepared", rates, offs, t_all: np.ndarray, search: float):
    """(t, keep, n_cells): the reference instants the offset search is scored
    on and their stable view / cell sets (G76). A long project has tens of
    thousands of instants and ~650 cost evaluations are made: scoring each
    on every instant is minutes. Only instants with data in >= 2 views count,
    only those where the landmarks MOVE carry timing information (a landmark
    at rest scores the same at every offset), and a few hundred of them,
    spread evenly over the window, give the same minimum. A short window
    (<= MAX_SCORE_INSTANTS usable instants) is used whole, exactly as before.
    `n_cells` = landmark-frames seen by two views over the whole window."""
    stride = max(1, int(np.ceil(len(t_all) / MAX_SCAN_INSTANTS)))
    t_scan = t_all[::stride]
    keep = prep.stable_cells(rates, offs, t_scan, search)
    shared = keep.sum(axis=2) >= 2                                   # (T, N)
    n_cells = int(np.count_nonzero(shared)) * stride
    usable = np.nonzero(shared.any(axis=1))[0]
    if len(usable) > MAX_SCORE_INSTANTS:
        a = prep.samples(rates, offs, t_scan, keep)
        b = prep.samples(rates, offs, t_scan + 1, keep)
        with np.errstate(all="ignore"):
            move = np.nanmax(np.nan_to_num(np.linalg.norm(b - a, axis=3), nan=-1.0).reshape(len(t_scan), -1), axis=1)
        mv = move[usable]
        pos = mv[mv > 0]
        if len(pos) >= 10:
            moving = usable[mv >= 0.25 * float(np.median(pos))]
            if len(moving) >= 10:
                usable = moving
        if len(usable) > MAX_SCORE_INSTANTS:
            usable = usable[even_indices(len(usable), MAX_SCORE_INSTANTS)]
    if len(usable) == 0:
        usable = np.arange(len(t_scan))
    return t_scan[usable], keep[usable], n_cells


def estimate_offsets(sessions: list, calib: Calibration, rates: list[float], offsets: list[float],
                     t_range: tuple[int, int], names: list[str] | None = None,
                     search: float = 1.0, step: float = 0.05, refine: float = 0.005,
                     passes: int = 2, report: dict | None = None,
                     progress=None, cancelled=None, robust: bool = False) -> tuple[list[float], float, float]:
    """Sub-frame sync from the tracks themselves: for each non-reference view,
    the fractional offset that minimises the mean triangulation residual, by
    coordinate descent (coarse grid of `step` over ±`search` local frames,
    then a fine grid of `refine` around the best). Returns (offsets, residual
    before, residual after). The reference view (view 0) keeps its offset.

    (G76) The search is scored on a subsample of the MOVING instants of a long
    window (`_scoring_instants`), and `progress(done, total)` is called after
    every cost evaluation (total is an estimate that includes the final joint
    refinement's budget: it can finish early) and `cancelled()` asked before
    each one: when it returns True the search stops, the offsets come back
    UNCHANGED and `report["verdict"]` is "cancelled".

    `report` (a dict, filled in place) says how much to trust the result:
    the minimum's SHARPNESS per view (how much the residual rises half a
    frame either side of the answer, relative to the answer -- a flat surface
    means the tracks carry no timing information, and a real stereo test
    showed the estimator returning -1.26 then -0.32 frames on exactly such a
    surface), the number of cells scored, the improvement, and a verdict
    "reliable" / "weak" / "flat" with a sentence.

    (I302) Each camera's sharpness is measured on its OWN scored cells (those it shares with another
    camera): the mean over every cell diluted a camera that sees the landmarks in a few % of the
    instants into "flat" however clear its own minimum. A camera with fewer than `MIN_OWN_CELLS` of
    its own reads at most "weak". `report["own_cells"]` / `["n_scored_cells"]` say how many. The
    verdict is computed after the search and never changes it.

    `robust` (I271): a minority of bad cells (a wand whose two ends are swapped in one camera on a few
    frames: pixels to tens of pixels each) must not drag the minimum. A first search on a CAPPED mean
    (each cell counts at most `ROBUST_CAP_K` x the median residual) gets near; the observations behind
    the cells whose residual THERE exceeds that are removed (`_drop_bad_observations`;
    `report["n_set_aside"]` = how many cells) and the ordinary search runs again from the given offsets
    on the rest, so the answer, its before / after and its sharpness are the plain estimator's on clean
    data. (A plain median cost was tried first: not moved by bad cells either, but its surface is grainy,
    0.08 frame off on the synthetic rig.) `robust=False` is exactly the estimator it always was."""
    offs = [float(o) for o in offsets]
    reference = 0                       # stable_cells holds view 0 still, as does the project (REFERENCE_VIEW)
    prep = _Prepared(sessions, calib, names)
    t_all = np.arange(int(t_range[0]), int(t_range[1]) + 1)
    t, keep, n_cells = _scoring_instants(prep, rates, offs, t_all, search)
    n_free = max(len(sessions) - 1, 0)
    n_grid = int(np.floor(2 * search / step + 1e-9)) + 1
    n_fine = int(np.floor(2 * step / refine + 1e-9)) + 1
    total = 1 + passes * n_free * (n_grid + n_fine) + 400 + 3 * n_free + 1
    if robust:
        total *= 2                      # two searches (below)
    done = [0]
    cap = [float("inf")]
    use = [keep]                        # the view / cell pairs scored (the robust search sets some aside)

    def residuals(o):
        _, r, _, _ = prep.triangulate(prep.samples(rates, o, t, use[0]), 2)
        return r

    def set_cap(o):
        r0 = residuals(o)
        r0 = r0[np.isfinite(r0)]
        cap[0] = ROBUST_CAP_K * float(np.median(r0)) if len(r0) and np.median(r0) > 0 else float("inf")

    def cost(o, mask=None):
        """The mean residual over the scored cells (only those of `mask`, (T, N), when given: I302)."""
        if cancelled is not None and cancelled():
            raise _SearchCancelled()
        r = residuals(o)
        r = r[np.isfinite(r) if mask is None else (mask & np.isfinite(r))]
        done[0] += 1
        if progress is not None:
            progress(min(done[0], total - 1), total)
        if not len(r):
            return float("inf")
        return float(np.minimum(r, cap[0]).mean()) if np.isfinite(cap[0]) else float(r.mean())

    try:
        if robust:
            # 1) a capped search gets NEAR; 2) the OBSERVATIONS behind the cells whose residual there is
            # more than ROBUST_CAP_K x the median are removed (`_drop_bad_observations`) and the ordinary
            # search runs again from the given offsets on what is left. Neither a capped search alone nor
            # setting the bad CELLS aside is enough: a bad observation, interpolated at a fractional
            # offset, spoils the two instants around it, so offsets that put its camera on whole frames
            # halve the spoiled cells -- on the synthetic rig with swapped ends both stopped 0.3 frame
            # off, every camera moved together.
            set_cap(offs)
            near, _, _ = _search_offsets(sessions, list(offs), cost, keep, n_cells, search, step, refine, passes,
                                         reference, None)
            n_bad = _drop_bad_observations(prep, rates, near, t_all)
            t, keep, n_cells = _scoring_instants(prep, rates, offs, t_all, search)
            use[0] = keep
            cap[0] = float("inf")
            out = _search_offsets(sessions, list(offs), cost, keep, n_cells, search, step, refine, passes,
                                  reference, report)
            if report is not None:
                report["n_set_aside"] = n_bad
            return out
        return _search_offsets(sessions, offs, cost, keep, n_cells, search, step, refine, passes, reference,
                               report)
    except _SearchCancelled:
        if report is not None:
            report.update({"verdict": "cancelled", "why": "Cancelled: the offsets were not changed.",
                           "sharpness": {}, "per_view": {}, "min_sharpness": float("nan"),
                           "improvement": float("nan"), "n_cells": n_cells, "own_cells": {},
                           "n_scored_cells": 0, "before_px": float("nan"), "after_px": float("nan")})
        return [float(o) for o in offsets], float("nan"), float("nan")
    finally:
        if progress is not None:
            progress(total, total)


def _drop_bad_observations(prep: "_Prepared", rates, offs, t_all: np.ndarray, chunk: int = 2000) -> int:
    """(I271) The raw observations behind the outlying cells at `offs` -- residual over
    ROBUST_CAP_K x the median: a wand end swapped in one camera, a click on the wrong ball -- set to
    NaN in `prep` (its own copy of the tracks; both frames each one is interpolated from). EVERY
    instant of `t_all` is looked at, not only the scored ones: a bad frame next to a scored instant
    spoils it at other offsets. The camera to blame is the one whose removal lowers the cell's
    residual most (leave-one-camera-out; a cell seen by two cameras loses both). A camera that is
    not calibrated never sees a cell (I273). Returns how many cells were outliers."""
    C = len(prep.ct)
    rs = []
    for k in range(0, len(t_all), chunk):
        _, r, _, _ = prep.triangulate(prep.samples(rates, offs, t_all[k:k + chunk]), 2)
        rs.append(r)
    if not rs:
        return 0
    r = np.concatenate(rs, axis=0)                                              # (T, N)
    fin = np.isfinite(r)
    if not fin.any():
        return 0
    bad = fin & (r > ROBUST_CAP_K * float(np.median(r[fin])))
    ti, ni = np.nonzero(bad)
    if not len(ti):
        return 0
    cells = prep.samples(rates, offs, t_all[ti])[np.arange(len(ti)), ni]       # (M, C, 2)
    seen = np.isfinite(cells).all(axis=2)
    loo = np.full((len(ti), C), np.inf)
    for c in range(C):
        if not seen[:, c].any():
            continue
        u = cells.copy()
        u[:, c] = np.nan
        _, rc, _, _ = prep.triangulate(u, 2)
        loo[:, c] = np.where(seen[:, c] & np.isfinite(rc), rc, np.inf)
    for m in range(len(ti)):
        if seen[m].sum() >= 3 and np.isfinite(loo[m]).any():
            blame = [int(np.argmin(loo[m]))]
        else:
            blame = [int(c) for c in np.nonzero(seen[m])[0]]
        for c in blame:
            f0 = int(np.floor(float(local_frame(t_all[ti[m]], rates[c], offs[c]))))
            for f in (f0, f0 + 1):
                if 0 <= f < prep.ct[c].shape[0]:
                    prep.ct[c][f, ni[m]] = np.nan
    return int(len(ti))


def _search_offsets(sessions, offs, cost, keep, n_cells, search, step, refine, passes, reference, report):
    """The body of `estimate_offsets` (split off so a cancel can unwind it)."""
    before = cost(offs)
    for _ in range(passes):
        for c in range(len(sessions)):
            if c == reference:
                continue
            base = offs[c]
            best = (cost(offs), base)
            # a view with no stable data leaves the cost flat: only a REAL
            # improvement may move it, never a tie (which would drift to the
            # edge of the search window)
            eps = 1e-9

            def better(cand, cur):
                return cand[0] < cur[0] - eps

            for d in np.arange(-search, search + 1e-9, step):
                offs[c] = base + float(d)
                cand = (cost(offs), offs[c])
                if better(cand, best):
                    best = cand
            centre = best[1]
            for d in np.arange(-step, step + 1e-9, refine):
                offs[c] = centre + float(d)
                cand = (cost(offs), offs[c])
                if better(cand, best):
                    best = cand
            offs[c] = best[1]
    # the offsets are coupled (every camera's best shift depends on the
    # others'), so coordinate descent alone creeps: finish with a joint
    # local minimisation from the grid result (Powell — the cost is smooth
    # but has no analytic gradient). Measured on the synthetic helix: grid
    # 0.055 frames off the truth, grid + Powell 0.007.
    free = [c for c in range(len(sessions)) if c != reference]
    if free and np.isfinite(cost(offs)):
        from scipy.optimize import minimize

        def joint(x):
            o = list(offs)
            for k, c in enumerate(free):
                o[c] = float(x[k])
            return cost(o)

        x0 = np.array([offs[c] for c in free])
        try:
            r = minimize(joint, x0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-8, "maxfev": 4000})
            if np.isfinite(r.fun) and r.fun <= joint(x0) and np.all(np.abs(r.x - x0) <= step + search):
                for k, c in enumerate(free):
                    offs[c] = float(r.x[k])
        except Exception:      # noqa: BLE001 — the grid result stands
            pass
    after = cost(offs)
    if report is not None:
        # (I302) each camera judged on its OWN scored cells: the cells it shares with another camera
        shared = keep.sum(axis=2) >= 2
        sharp, own_cells = {}, {}
        for c in range(len(sessions)):
            if c == reference:
                continue
            own = shared & keep[:, :, c]
            own_cells[c] = int(own.sum())
            base = cost(offs, own) if own_cells[c] else float("nan")
            rise = []
            for d in (-0.5, 0.5):
                o = list(offs)
                o[c] = offs[c] + d
                cd = cost(o, own) if own_cells[c] else float("nan")
                rise.append((cd - base) / max(base, 1e-9) if np.isfinite(cd) and np.isfinite(base) else np.nan)
            sharp[c] = float(np.nanmin(rise)) if np.isfinite(rise).any() else float("nan")
        improvement = ((before - after) / before) if (np.isfinite(before) and before > 0
                                                    and np.isfinite(after)) else float("nan")
        # Per view: a camera whose surface is flat has NO shared tracks in the
        # window (e.g. a 240 fps camera where nothing it saw was tracked)
        # -- the grid leaves such an offset where it was, and the verdict must
        # judge only the cameras that were actually determined, not call the
        # whole result flat because one camera had nothing to say.
        # (G75) "none" is a camera with NO cell shared with another one; a camera
        # that shares cells but whose landmarks hardly move reads "flat" (the
        # surface is flat because there is nothing to time, which is not the
        # same as having no shared tracks); (I302) a clear minimum on fewer than
        # MIN_OWN_CELLS cells of its own is only "weak"
        per_view = {}
        for c, v in sharp.items():
            if not own_cells[c]:
                per_view[c] = "none"          # unconstrained: offset kept
            elif not np.isfinite(v) or v < 0.03:
                per_view[c] = "flat"
            elif v >= 0.10 and own_cells[c] >= MIN_OWN_CELLS:
                per_view[c] = "sharp"
            else:
                per_view[c] = "weak"
        determined = [c for c, w in per_view.items() if w != "none"]
        sharps = [sharp[c] for c in determined]
        min_sharp = min(sharps) if sharps else float("nan")
        n_none = sum(1 for w in per_view.values() if w == "none")
        static = bool(sharps) and all((not np.isfinite(s)) or s < 0.005 for s in sharps)
        if not determined or n_cells < 10:
            verdict = "flat"
            why = (f"Only {n_cells} landmark-frames are seen by two cameras across the whole search "
                   "window: too few to time the cameras from. Keep the current offsets.")
        elif static:
            verdict = "flat"
            why = ("flat: the landmarks do not move enough to time it. The disagreement does not change with "
                   "the offsets because nothing in this window moves (a landmark at rest looks the same at "
                   "every offset). Do NOT apply these offsets; pick a stretch where the animal moves, or sync "
                   "the cameras from a flash or clap.")
        elif all(per_view[c] == "sharp" for c in determined):
            # (I100) reliability is the SHARPNESS of the minimum. Requiring an
            # improvement as well called offsets that were already right (a
            # re-check after applying, a flash sync) "weak", and the weak
            # sentence then quoted this minimum's sharpness: "by only 969%".
            verdict = "reliable"
            if np.isfinite(improvement) and improvement < 0.02 and before >= 1e-6:
                why = (f"The current offsets are already at a clear minimum: shifting any of the timed "
                       f"cameras by half a frame raises the disagreement by at least {100 * min_sharp:.0f}%, "
                       f"and the search changed it by less than 2%. Nothing needs to change (applying the "
                       "numbers below is harmless).")
            else:
                why = (f"A clear minimum: shifting any of the timed cameras by half a frame raises the "
                       f"disagreement by at least {100 * min_sharp:.0f}%. The offsets are supported by the tracks.")
        elif all(per_view[c] in ("sharp", "weak") for c in determined):
            verdict = "weak"
            why = (f"A shallow minimum for at least one camera: half a frame either way changes the "
                   f"disagreement by only {100 * min_sharp:.0f}%. The tracks barely constrain its timing "
                   "(slow motion, few cells, or noisy tracks); apply only if the numbers look plausible.")
        else:
            verdict = "flat"
            why = (f"The disagreement hardly changes with the offsets ({100 * max(min_sharp, 0):.0f}% over "
                   "half a frame) for at least one camera: the tracks carry no timing "
                   "information for it. Do NOT apply these offsets; sync the cameras from a flash or "
                   "clap instead.")
        if n_none:
            why += (f" {n_none} camera(s) share no tracked landmark with the others in this window, so "
                    "their offsets are left exactly as they were.")
        report.update({"verdict": verdict, "why": why, "sharpness": sharp, "per_view": per_view,
                       "min_sharpness": min_sharp, "improvement": improvement, "n_cells": n_cells,
                       "own_cells": own_cells, "n_scored_cells": int(np.count_nonzero(shared)),
                       "before_px": before, "after_px": after})
    return offs, before, after
