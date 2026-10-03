"""Wand calibration of a multi-camera rig, natively (no MATLAB, no easyWand).

What a wand calibration is
--------------------------
A rigid wand with two markers a known distance L apart is waved through the
working volume while every camera films it; the two ends are digitized in
each camera on F frames. Any extra point that is co-visible in >= 2 cameras
("background" points: a corner of the enclosure, a static marker, anything)
adds constraints. This is the calibration of Theriault et al. (2014,
J. Exp. Biol. 217:1843 - easyWand): the cameras and every digitized 3D point
are solved together by sparse bundle adjustment on the reprojection error;
the wand length only sets the SCALE afterwards, so the frame-to-frame
variation of the reconstructed wand length (the "wand score",
100 * sd / mean, in %) is an honest measure of the calibration's quality -
the wand was never forced to be rigid inside the fit.

Pipeline (`calibrate_wand`)
---------------------------
1. Pick the camera pair with the most co-visible points; normalise the pixels
   with the intrinsic guesses; essential matrix (5-point RANSAC) + the
   cheirality test over its four decompositions -> relative pose with a unit
   baseline; linear triangulation.
2. Register the remaining cameras one at a time by PnP-RANSAC on the points
   already triangulated (the camera that sees the most of them first, at
   least `MIN_PNP_POINTS`), re-triangulating everything after each addition.
3. Bundle adjustment (`scipy.optimize.least_squares`, trust-region reflective,
   finite-difference Jacobian with an explicit sparsity pattern so that 6
   cameras x 800 points solve in well under a second per pass): per camera
   rvec(3) + t(3) + f (+ k1, k2), 3 per 3D point (the principal point stays at
   the image centre or the lens profile's). Camera 0's pose
   is the gauge; the scale is left free here and fixed in step 5.
4. Outlier rejection on the observations (3 x the median error but never
   below `OUTLIER_FLOOR_PX` at 1080p, two passes; or a fixed threshold with
   `outlier_px`) and a refit after each pass. A bad click drags its own 3D
   point, so which observation of an offending point is the bad one is
   decided by leave-one-out re-triangulation, not by the largest residual.
5. Scale so that the MEAN reconstructed wand length equals `wand_length`.
6. Unknown focal lengths: steps 1-3 run for every entry of `FOCAL_GRID`
   (x image width, applied to every camera; each PnP camera also tries the
   grid for itself) and the lowest bundle-adjustment cost wins; the focal
   lengths are then free in the final adjustment. A single camera PAIR
   cannot determine focal lengths when the optical axes intersect - which is
   the usual rig, all cameras aimed at the same volume - so the whole rig is
   used to score each guess.

Conventions
-----------
* Pixels are Kinetrace's: 0-based, top-left origin, pixel centres at integer
  coordinates (`pixel_origin=0.0, y_flip=False` in `calib.CameraCalibration`).
  MATLAB / DLTdv / easyWand pixels are 1-based: subtract 1 before calling and
  use `export_dlt_csv(..., pixel_origin_out=1.0)` to hand a result back.
* Camera model: x_cam = R X + t; u = f (x/z) d + cx, v = f (y/z) d + cy with
  the OpenCV radial polynomial d = 1 + k1 r^2 + k2 r^4 (zero unless
  `estimate_distortion`). The DLT coefficients in the result are
  `calib.dlt_from_camera(K, R, t)`, valid for UNDISTORTED pixels, so a
  `CameraCalibration` carrying `OpenCVUndistort(K, dist)` reproduces the
  model exactly (`NoUndistort` when no distortion was fitted).
* World frame after `calibrate_wand`: camera 0's axes (R_0 = I) with the
  origin at the centroid of the reconstructed wand ends - NOT at camera 0
  itself: a camera sitting at the world origin has no 11-parameter DLT (the
  coefficients are normalised by L12 = t_z, which would be zero).
  `align_gravity` (Z up from a dropped object, an independent scale check
  through g) and `align_axes` (three reference points) re-express the whole
  result through `transform_result`. The world is right-handed (a proper
  rotation from camera 0's frame); easyWand's DLT output is left-handed, so
  comparing the two needs a similarity WITH reflection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .calib import (Calibration, CameraCalibration, NoUndistort, OpenCVUndistort,
                    dlt_from_camera)

FOCAL_GRID = (0.25, 0.4, 0.55, 0.75, 1.0, 1.3, 1.7, 2.2)   # initial f as a multiple of image width
OUTLIER_FLOOR_PX = 2.0      # automatic rejection threshold floor at 1080p (scaled by max(w, h) / 1920):
                            # a hand-digitized wand end is not reliable below ~2 px, and 3 x median
                            # alone threw away 13% of a real GoPro calibration's clicks and made its
                            # wand score WORSE (0.95% vs 0.68% with nothing rejected)
MIN_PAIR_POINTS = 8         # co-visible points the initial camera pair needs
MIN_PNP_POINTS = 6          # reconstructed points a camera needs before it can be registered
_NCP = 11                   # parameters per camera: rvec(3) t(3) f k1 k2 cx cy
# The named checks behind the verdict (I176): the reasons AND the verdict come from the
# same list, so a verdict can never say GOOD beside a reason that says "unreliable".
SCORE_GOOD_PCT, SCORE_OK_PCT = 1.0, 2.5        # wand score (sd / mean of the wand length, %)
RMSE_GOOD_PX, RMSE_OK_PX = 1.0, 2.5            # reprojection error at 1080p
MIN_COVERAGE_PCT = 25.0                        # of the picture the wand visited, per camera
MIN_OBS_PER_CAMERA = 30                        # usable observations a camera should keep
MIN_OBS_KEPT_FRAC = 0.5                        # ... and the share of its own observations
MAX_REJECTED_PCT = 5.0                         # outliers set aside, of every observation
MIN_FRAMES_USED = 20                           # wand frames; fewer is a note, not a failure
# the gravity check's bands (G137): consistent / suspect, in % away from g
G_CONSISTENT_PCT, G_SUSPECT_PCT = 2.0, 5.0
# g in the world's unit (I27): the world is in whatever unit the wand length was
# typed in, so a drop measured in mm/s^2 must be compared with 9810, not 9.81.
# "wand" (length not measured yet) has no entry: there the fall MEASURES the unit.
G_BY_UNIT = {"m": 9.81, "cm": 981.0, "mm": 9810.0, "in": 9.81 / 0.0254}
# a drop whose acceleration is further than this from g does not set the vertical (I25):
# either it was not falling freely on those frames or the scale inputs are badly off
G_RATIO_RANGE = (0.8, 1.25)


class WandError(ValueError):
    """The calibration cannot proceed; the message names the cause and, where
    possible, the camera."""


def _cam(c: int, names=None, cap: bool = False) -> str:
    """How a message names camera index `c` (I34): the project's own name for
    it when the caller passed the names (cam1, GoPro2 ...), else "camera c+1" -
    never the 0-based index, because the wizard, the camera panel and the
    report's table all count from 1."""
    if names is not None and 0 <= c < len(names) and str(names[c]).strip():
        return str(names[c])
    return f"{'Camera' if cap else 'camera'} {c + 1}"


def _px_scale(sizes) -> np.ndarray:
    """Per camera: how many times a 1080p pixel one of its pixels is, by the
    LONGER side (I250: a portrait 4K clip is as sharp as a landscape one, and
    max(1, width / 1920) judged it 1.8 x stricter). The one rule for the
    outlier floor, the soft-L1 knee and the reprojection verdict."""
    s = np.asarray([[w, h] for w, h in sizes], np.float64)
    return np.maximum(1.0, s.max(axis=1) / 1920.0)


def _per_camera(value, C: int, what: str, dtype=bool) -> np.ndarray:
    """One value, or one per camera, as a (C,) array: the rule of `focal`,
    `estimate_focal` and `estimate_distortion`."""
    arr = np.asarray(value, dtype)
    if arr.ndim == 0:
        return np.full(C, arr)
    arr = arr.ravel()
    if len(arr) != C:
        raise WandError(f"{what} has {len(arr)} entries for {C} cameras")
    return arr.copy()


# ------------------------------------------------------------------ result


@dataclass
class WandResult:
    """Everything `calibrate_wand` produced. Arrays are in the result's world
    frame (see the module docstring); pixels are 0-based."""
    cameras: list          # CameraCalibration per camera (pixel_origin=0.0, y_flip=False, rmse = px)
    K: np.ndarray          # (C, 3, 3)
    R: np.ndarray          # (C, 3, 3)   x_cam = R X + t
    t: np.ndarray          # (C, 3)
    dist: np.ndarray       # (C, 5) OpenCV (k1, k2, p1, p2, k3); zeros when not estimated
    wand_xyz: np.ndarray   # (F, 2, 3) world coordinates of the wand ends, NaN where unknown
    wand_len: np.ndarray   # (F,) reconstructed wand length per frame, NaN where unknown
    bg_xyz: np.ndarray     # (B, 3) background points, NaN where not triangulable
    frame_used: np.ndarray # (F,) bool: frame took part in the fit and both ends survived
    reproj_cam: np.ndarray # (C,) RMSE px per camera over the kept observations
    reproj_obs: dict       # "wand" (F, C, 2) px, "bg" (B, C) px, plus "wand_used" / "bg_used" masks
    unit: str              # the wand length's unit label
    report: dict           # plain-language diagnostics (JSON-serialisable)

    def to_calibration(self) -> Calibration:
        return Calibration(list(self.cameras), self.unit, "wand (Kinetrace)")


# ------------------------------------------------------------- observations


class _Obs:
    """Every digitized 2D observation as flat arrays: point index, camera,
    pixel, provenance (kind 0 = wand end / 1 = background; frame or
    background index; camera; end or -1) and an inlier flag. Points are
    numbered wand ends of the SELECTED frames first (2 k + end for the k-th
    selected frame), then background points."""

    def __init__(self, wand_uv: np.ndarray, bg_uv: np.ndarray | None, frames: np.ndarray):
        F, C = wand_uv.shape[:2]
        self.C = C
        self.frames = np.asarray(frames, int)
        Fs = len(self.frames)
        self.n_wand = 2 * Fs
        w = wand_uv[self.frames]                                    # (Fs, C, 2, 2)
        ok = np.isfinite(w).all(axis=3)
        k, c, e = np.nonzero(ok)
        pts = [2 * k + e]
        cams = [c]
        uvs = [w[k, c, e].reshape(-1, 2)]
        srcs = [np.column_stack([np.zeros_like(k), self.frames[k], c, e]).reshape(-1, 4)]
        B = 0 if bg_uv is None else int(bg_uv.shape[0])
        if B:
            okb = np.isfinite(bg_uv).all(axis=2)
            b, cb = np.nonzero(okb)
            pts.append(2 * Fs + b)
            cams.append(cb)
            uvs.append(bg_uv[b, cb].reshape(-1, 2))
            srcs.append(np.column_stack([np.ones_like(b), b, cb, np.full_like(b, -1)]).reshape(-1, 4))
        self.P = 2 * Fs + B
        self.n_bg = B
        self.pt = np.concatenate(pts).astype(int)
        self.cam = np.concatenate(cams).astype(int)
        self.uv = np.concatenate(uvs).astype(np.float64)
        self.src = np.concatenate(srcs).astype(int)
        self.used = np.ones(len(self.pt), bool)
        self.reject_err = np.full(len(self.pt), np.nan)     # px error at the moment of rejection

    @property
    def N(self) -> int:
        return len(self.pt)


# ------------------------------------------------------------ camera maths


def _rodrigues(rv: np.ndarray) -> np.ndarray:
    """Rotation vectors (n, 3) -> matrices (n, 3, 3), vectorised."""
    rv = np.atleast_2d(np.asarray(rv, np.float64))
    th = np.linalg.norm(rv, axis=1)
    k = rv / np.where(th > 1e-12, th, 1.0)[:, None]
    Kx = np.zeros((len(rv), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    c = np.cos(th)[:, None, None]
    s = np.sin(th)[:, None, None]
    R = c * np.eye(3)[None] + (1.0 - c) * k[:, :, None] * k[:, None, :] + s * Kx
    R[th <= 1e-12] = np.eye(3)
    return R


def _rvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def _Ps(cams: np.ndarray) -> np.ndarray:
    """(C, 3, 4) normalised projection matrices [R | t]."""
    return np.concatenate([_rodrigues(cams[:, :3]), cams[:, 3:6, None]], axis=2)


def _K_of(cams: np.ndarray, c: int) -> np.ndarray:
    return np.array([[cams[c, 6], 0.0, cams[c, 9]], [0.0, cams[c, 6], cams[c, 10]], [0.0, 0.0, 1.0]])


def _project(cams: np.ndarray, X: np.ndarray, pt_idx: np.ndarray, cam_idx: np.ndarray) -> np.ndarray:
    """Pixel projection of X[pt_idx] through camera cam_idx, vectorised over
    observations (the residual function's core: no Python loop)."""
    Rs = _rodrigues(cams[:, :3])
    Xc = np.einsum("nij,nj->ni", Rs[cam_idx], X[pt_idx]) + cams[cam_idx, 3:6]
    z = Xc[:, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)          # never divide by zero; a bad init just costs
    xn = Xc[:, 0] / z
    yn = Xc[:, 1] / z
    r2 = xn * xn + yn * yn
    d = 1.0 + cams[cam_idx, 7] * r2 + cams[cam_idx, 8] * r2 * r2
    f = cams[cam_idx, 6]
    return np.column_stack([f * xn * d + cams[cam_idx, 9], f * yn * d + cams[cam_idx, 10]])


def _normalise(uv: np.ndarray, cams: np.ndarray, cam_idx: np.ndarray) -> np.ndarray:
    """Finite pixels -> normalised camera coordinates (x/z, y/z); cameras
    carrying k1/k2 are undistorted through OpenCV (iterative inversion)."""
    f = cams[cam_idx, 6]
    xn = (uv - cams[cam_idx, 9:11]) / f[:, None]
    has_dist = np.any(cams[:, 7:9] != 0.0, axis=1)
    for c in np.nonzero(has_dist)[0]:
        m = cam_idx == c
        if not m.any():
            continue
        dist = np.array([cams[c, 7], cams[c, 8], 0.0, 0.0, 0.0])
        # (I177) converge the inverse (calib.OpenCVUndistort's rule, I73): OpenCV's default
        # 5 steps left a GoPro-like k1 / k2 corner point 4.8 mm (median) off in 3D
        und = cv2.undistortPoints(uv[m].reshape(-1, 1, 2).astype(np.float64), _K_of(cams, c), dist,
                                  None, None, None, OpenCVUndistort.CRITERIA)
        xn[m] = und.reshape(-1, 2)
    return xn


def _triangulate_normalised(Ps: np.ndarray, xn: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of M points from normalised coordinates
    (M, C, 2) with `mask` (M, C) saying which cameras saw each point: the
    two rows x P3 - P1 / y P3 - P2 per camera, the null vector by a batched
    SVD (unseen cameras contribute zero rows). NaN with < 2 cameras."""
    M, C = mask.shape
    x = np.nan_to_num(xn)
    Ps = np.nan_to_num(Ps)
    A = np.zeros((M, 2 * C, 4))
    for c in range(C):
        A[:, 2 * c] = x[:, c, 0, None] * Ps[c, 2][None] - Ps[c, 0][None]
        A[:, 2 * c + 1] = x[:, c, 1, None] * Ps[c, 2][None] - Ps[c, 1][None]
        A[~mask[:, c], 2 * c:2 * c + 2] = 0.0
    out = np.full((M, 3), np.nan)
    ok = mask.sum(1) >= 2
    if ok.any():
        _, _, Vt = np.linalg.svd(A[ok])
        X = Vt[:, -1, :]
        w = X[:, 3]
        good = np.abs(w) > 1e-12
        res = np.full((int(ok.sum()), 3), np.nan)
        res[good] = X[good, :3] / w[good, None]
        out[ok] = res
    return out


def _triangulate_uv(cams: np.ndarray, uv: np.ndarray):
    """Triangulate pixel observations (M, C, 2) (NaN = unseen) with the
    current cameras -> xyz (M, 3) and per-camera reprojection error (M, C)."""
    uv = np.asarray(uv, np.float64)
    M, C = uv.shape[:2]
    mask = np.isfinite(uv).all(axis=2)
    xn = np.full((M, C, 2), np.nan)
    m, c = np.nonzero(mask)
    if len(m):
        xn[m, c] = _normalise(uv[m, c], cams, c)
    xyz = _triangulate_normalised(_Ps(cams), xn, mask)
    err = np.full((M, C), np.nan)
    okp = np.isfinite(xyz).all(axis=1)
    m2, c2 = np.nonzero(mask & okp[:, None])
    if len(m2):
        proj = _project(cams, xyz, m2, c2)
        err[m2, c2] = np.linalg.norm(proj - uv[m2, c2], axis=1)
    return xyz, err


def _apply_similarity(cams: np.ndarray, pts: np.ndarray, R_w: np.ndarray, t_w: np.ndarray,
                      s: float):
    """World X' = s R_w X + t_w. A camera then sees x_cam = R X + t =
    (R R_w^T) X' + (s t - R R_w^T t_w) up to the positive factor s, which the
    perspective division removes - so R' = R R_w^T, t' = s t - R' t_w."""
    Rn = _rodrigues(cams[:, :3]) @ R_w.T
    tn = s * cams[:, 3:6] - np.einsum("cij,j->ci", Rn, t_w)
    out = cams.copy()
    out[:, :3] = np.stack([_rvec(R) for R in Rn])
    out[:, 3:6] = tn
    return out, s * (pts @ R_w.T) + t_w


def _errors(obs: _Obs, cams: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    """Reprojection error (px) of EVERY observation, NaN where its point has
    no 3D position."""
    e = np.full(obs.N, np.nan)
    ok = np.isfinite(pts3[obs.pt]).all(axis=1)
    if ok.any():
        proj = _project(cams, pts3, obs.pt[ok], obs.cam[ok])
        e[ok] = np.linalg.norm(proj - obs.uv[ok], axis=1)
    return e


# ------------------------------------------------------------ initialisation


def _retriangulate(obs: _Obs, cams: np.ndarray, registered) -> np.ndarray:
    reg = np.zeros(obs.C, bool)
    reg[list(registered)] = True
    sel = obs.used & reg[obs.cam]
    xn = np.full((obs.P, obs.C, 2), np.nan)
    if sel.any():
        xn[obs.pt[sel], obs.cam[sel]] = _normalise(obs.uv[sel], cams, obs.cam[sel])
    mask = np.isfinite(xn).all(axis=2)
    return _triangulate_normalised(_Ps(cams), xn, mask)


def _init_pair(obs: _Obs, cams: np.ndarray, i: int, j: int, thr_px: float, names=None) -> None:
    """Relative pose of camera j w.r.t. camera i (which becomes [I | 0]) from
    the essential matrix; the four decompositions are ranked by how many
    inlier points land in front of BOTH cameras."""
    uv_i = np.full((obs.P, 2), np.nan)
    uv_j = np.full((obs.P, 2), np.nan)
    si = obs.used & (obs.cam == i)
    sj = obs.used & (obs.cam == j)
    uv_i[obs.pt[si]] = obs.uv[si]
    uv_j[obs.pt[sj]] = obs.uv[sj]
    common = np.nonzero(np.isfinite(uv_i).all(1) & np.isfinite(uv_j).all(1))[0]
    xi = (uv_i[common] - cams[i, 9:11]) / cams[i, 6]
    xj = (uv_j[common] - cams[j, 9:11]) / cams[j, 6]
    thr = thr_px / float(np.mean([cams[i, 6], cams[j, 6]]))
    E, mask = cv2.findEssentialMat(xi, xj, np.eye(3), method=cv2.RANSAC, prob=0.9999,
                                   threshold=thr, maxIters=5000)
    pair = f"{_cam(i, names)} and {_cam(j, names)}"
    if E is None or E.shape[0] < 3 or mask is None:
        raise WandError(f"{pair}: the essential matrix could not be estimated "
                        f"from {len(common)} shared points")
    E = np.asarray(E, np.float64)[:3, :3]     # several 3x3 candidates may come back stacked
    inl = mask.ravel().astype(bool)
    if inl.sum() < MIN_PAIR_POINTS:
        raise WandError(f"{pair}: only {int(inl.sum())} of {len(common)} shared points "
                        f"agree on a relative pose - check that the same wand end is digitized "
                        f"as end 1 in both cameras and that the videos are frame-synchronised")
    R1, R2, tt = cv2.decomposeEssentialMat(E)
    best = None
    xn = np.stack([xi[inl], xj[inl]], axis=1)
    ones = np.ones((int(inl.sum()), 2), bool)
    for R in (R1, R2):
        for t in (tt.ravel(), -tt.ravel()):
            Ps = np.stack([np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, t[:, None]])])
            X = _triangulate_normalised(Ps, xn, ones)
            z0 = X[:, 2]
            z1 = (X @ R.T + t)[:, 2]
            n_front = int(np.sum(np.isfinite(z0) & (z0 > 0) & (z1 > 0)))
            if best is None or n_front > best[0]:
                best = (n_front, R, t)
    n_front, R, t = best
    if n_front < MIN_PAIR_POINTS:
        raise WandError(f"{pair}: no relative pose puts the shared points in front "
                        f"of both cameras ({n_front} of {int(inl.sum())})")
    cams[i, :6] = 0.0
    cams[j, :3] = _rvec(R)
    cams[j, 3:6] = t


def _register_camera(obs: _Obs, cams: np.ndarray, pts3: np.ndarray, c: int, f_candidates,
                     thr_px: float, refine_f: bool, names=None) -> None:
    """Pose of camera c by PnP-RANSAC on the points already reconstructed;
    with several focal candidates the one with the most inliers (then the
    lowest error) wins and pose + focal are refined together."""
    sel = obs.used & (obs.cam == c) & np.isfinite(pts3[obs.pt]).all(axis=1)
    obj = np.ascontiguousarray(pts3[obs.pt[sel]], np.float64)
    img = np.ascontiguousarray(obs.uv[sel], np.float64)
    nm = _cam(c, names)
    if len(obj) < MIN_PNP_POINTS:
        raise WandError(f"{nm} sees only {len(obj)} of the points that are already "
                        f"reconstructed (needs >= {MIN_PNP_POINTS}): digitize the wand in more "
                        f"frames where {nm} and an already calibrated camera both see it, "
                        f"or add background points visible in {nm}")
    best = None
    for f in f_candidates:
        K = np.array([[f, 0.0, cams[c, 9]], [0.0, f, cams[c, 10]], [0.0, 0.0, 1.0]])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, None,
                                                 iterationsCount=2000, reprojectionError=float(thr_px),
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < MIN_PNP_POINTS:
            continue
        inl = inl.ravel()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl].reshape(-1, 1, 3), img[inl].reshape(-1, 1, 2),
                                              K, None, rvec, tvec)
        except cv2.error:
            pass
        row = cams[c].copy()
        row[:3] = np.ravel(rvec)
        row[3:6] = np.ravel(tvec)
        row[6] = float(f)
        row[7:9] = 0.0
        n = len(inl)
        e = np.linalg.norm(_project(row[None], obj[inl], np.arange(n), np.zeros(n, int)) - img[inl], axis=1)
        key = (-n, float(np.sqrt(np.mean(e ** 2))))
        if best is None or key < best[0]:
            best = (key, row, inl)
    if best is None:
        raise WandError(f"{nm}: no pose is consistent with its {len(obj)} reconstructed points "
                        f"- check that the same wand end is digitized as end 1 in every camera")
    _, row, inl = best
    if refine_f:
        o, im = obj[inl], img[inl]
        n = len(o)
        idx, zero = np.arange(n), np.zeros(n, int)

        def fun(x):
            r = row.copy()
            r[:7] = x
            return (_project(r[None], o, idx, zero) - im).ravel()

        try:
            r = least_squares(fun, row[:7], method="trf", x_scale="jac", max_nfev=100)
            if np.isfinite(r.x).all() and r.x[6] > 0:
                row[:7] = r.x
        except Exception:      # noqa: BLE001 - the PnP pose stands
            pass
    cams[c] = row


def _initialise(obs: _Obs, f0: np.ndarray, pp: np.ndarray, sizes, search_focal, names=None):
    """Steps 1-2 of the pipeline: pair pose + incremental PnP; returns the
    camera parameters (C, 11) in camera 0's frame and the 3D points (P, 3).
    `search_focal`: one bool, or one per camera (I221: a camera with a known
    focal length keeps it while the others try the grid for themselves)."""
    C = obs.C
    widths = np.array([s[0] for s in sizes], np.float64)
    scale = _px_scale(sizes)
    search_c = _per_camera(search_focal, C, "search_focal")
    cams = np.full((C, _NCP), np.nan)
    cams[:, 6] = f0
    cams[:, 7:9] = 0.0
    cams[:, 9:11] = pp
    vis = np.zeros((obs.P, C), bool)
    vis[obs.pt[obs.used], obs.cam[obs.used]] = True
    co = vis.T.astype(np.int64) @ vis.astype(np.int64)
    np.fill_diagonal(co, 0)
    i, j = np.unravel_index(int(np.argmax(co)), co.shape)
    i, j = sorted((int(i), int(j)))
    if co[i, j] < MIN_PAIR_POINTS:
        raise WandError(f"no two cameras share {MIN_PAIR_POINTS} or more digitized points (the best "
                        f"pair, {_cam(i, names)} and {_cam(j, names)}, shares {int(co[i, j])}): digitize "
                        f"the wand in frames where at least two cameras see both ends")
    _init_pair(obs, cams, i, j, 3.0 * float(max(scale[i], scale[j])), names)
    registered = [i, j]
    pts3 = _retriangulate(obs, cams, registered)
    while len(registered) < C:
        have = np.isfinite(pts3).all(axis=1)
        counts = np.zeros(C, int)
        m = obs.used & have[obs.pt]
        np.add.at(counts, obs.cam[m], 1)
        counts[registered] = -1
        c = int(np.argmax(counts))
        cands = [mult * widths[c] for mult in FOCAL_GRID] if search_c[c] else [float(f0[c])]
        _register_camera(obs, cams, pts3, c, cands, 4.0 * float(scale[c]), refine_f=bool(search_c[c]),
                         names=names)
        registered.append(c)
        pts3 = _retriangulate(obs, cams, registered)
    # camera 0 becomes the reference frame: world' = R0 X + t0
    return _apply_similarity(cams, pts3, _rodrigues(cams[0, :3])[0], cams[0, 3:6].copy(), 1.0)


# --------------------------------------------------------- bundle adjustment


def _bundle_adjust(obs: _Obs, cams: np.ndarray, pts3: np.ndarray, *, free_f, free_dist,
                   loss: str = "linear", f_scale: float = 1.0, max_nfev: int = 200):
    """Joint refinement of cameras and points on the kept observations of
    every point that has >= 2 of them. Parameter vector = the free entries of
    [cams (C x 11) | active points (x 3)]: camera 0's pose is never free (the
    gauge), f only with `free_f` (one bool, or one per camera, I144), k1/k2
    only with `free_dist` (one bool, or
    one per camera: a camera whose points were already undistorted by a lens
    profile keeps k1 = k2 = 0 while the others fit theirs); the principal
    point stays where it was put (the image centre or the lens profile's).
    The Jacobian sparsity (each residual pair touches one
    camera block and one point) lets scipy estimate it with ~14 function
    evaluations per iteration."""
    C = cams.shape[0]
    P = pts3.shape[0]
    fd = _per_camera(free_dist, C, "free_dist")
    ff = _per_camera(free_f, C, "free_f")
    ok_pt = np.isfinite(pts3).all(axis=1) & (np.bincount(obs.pt[obs.used], minlength=P) >= 2)
    sel = obs.used & ok_pt[obs.pt]
    pt_g, cam_g, uv = obs.pt[sel], obs.cam[sel], obs.uv[sel]
    act = np.nonzero(ok_pt)[0]
    local = np.full(P, -1)
    local[act] = np.arange(len(act))
    pl = local[pt_g]
    n_cam = C * _NCP
    template = np.concatenate([cams.ravel(), pts3[act].ravel()])
    free = np.zeros(len(template), bool)
    for c in range(C):
        b = c * _NCP
        if c > 0:
            free[b:b + 6] = True
        free[b + 6] = bool(ff[c])
        free[b + 7:b + 9] = bool(fd[c])
    free[n_cam:] = True
    free_idx = np.nonzero(free)[0]
    pos = np.full(len(template), -1)
    pos[free_idx] = np.arange(len(free_idx))
    N = len(pt_g)
    if N < 8 or len(act) == 0:
        raise WandError("too few observations remain for a bundle adjustment")

    def fun(x):
        full = template.copy()
        full[free_idx] = x
        cp = full[:n_cam].reshape(C, _NCP)
        X = full[n_cam:].reshape(-1, 3)
        return (_project(cp, X, pl, cam_g) - uv).ravel()

    # sparsity: residuals 2i, 2i+1 <- free columns of camera cam_g[i] + the 3 of point pl[i]
    cam_cols = pos[cam_g[:, None] * _NCP + np.arange(_NCP)[None, :]]
    pt_cols = pos[n_cam + pl[:, None] * 3 + np.arange(3)[None, :]]
    cols = np.concatenate([cam_cols, pt_cols], axis=1)                     # (N, 14)
    rows = np.repeat(2 * np.arange(N)[:, None], cols.shape[1], axis=1)
    keep = cols >= 0
    r = np.concatenate([rows[keep], rows[keep] + 1])
    cidx = np.concatenate([cols[keep], cols[keep]])
    S = coo_matrix((np.ones(len(r)), (r, cidx)), shape=(2 * N, len(free_idx)))
    x0 = template[free_idx]
    if not np.isfinite(fun(x0)).all():
        raise WandError("the initial reconstruction is not finite (a point at zero depth)")
    res = least_squares(fun, x0, jac_sparsity=S, method="trf", x_scale="jac", loss=loss,
                        f_scale=f_scale, max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    full = template.copy()
    full[free_idx] = res.x
    cams_new = full[:n_cam].reshape(C, _NCP).copy()
    pts_new = np.full_like(pts3, np.nan)
    pts_new[act] = full[n_cam:].reshape(-1, 3)
    return cams_new, pts_new, res


def _reject(obs: _Obs, cams: np.ndarray, pts3: np.ndarray, outlier_px, floor_px: float,
            max_rounds: int = 8):
    """Drop observations whose reprojection error exceeds the threshold
    (`outlier_px`, or 3 x the median of the kept ones, never below `floor_px`).

    One bad click drags its 3D point towards itself, so the point's GOOD
    observations also look wrong while the bad one is still in the fit -
    neither "everything above the threshold" nor "the largest residual" is
    the bad click. Each round therefore removes ONE observation per
    offending point, chosen by leave-one-out: the observation whose removal
    leaves the smallest worst-case error among the rest after
    re-triangulation (a point with only two observations loses the larger
    residual and dies). Rounds stop when nothing exceeds the threshold.
    Points are updated in place (`pts3`)."""
    e = _errors(obs, cams, pts3)
    cur = obs.used & np.isfinite(e)
    if not cur.any():
        return float("nan"), 0
    if outlier_px is None:
        thr = max(3.0 * float(np.median(e[cur])), float(floor_px))
    else:
        thr = float(outlier_px)
    n_new = 0
    for _ in range(max_rounds):
        cand = obs.used & np.isfinite(e) & (e > thr)
        if not cand.any():
            break
        drop = []
        variants, owner, left_out = [], [], []
        for p in np.unique(obs.pt[cand]):
            idx = np.nonzero(obs.used & (obs.pt == p))[0]
            if len(idx) <= 2:
                drop.append(int(idx[np.argmax(e[idx])]))
                continue
            base = np.full((obs.C, 2), np.nan)
            base[obs.cam[idx]] = obs.uv[idx]
            for k in idx:
                v = base.copy()
                v[obs.cam[k]] = np.nan
                variants.append(v)
                owner.append(p)
                left_out.append(int(k))
        if variants:
            xyz, er = _triangulate_uv(cams, np.stack(variants))
            worst = np.where(np.isfinite(er), er, -1.0).max(axis=1)
            worst[~np.isfinite(xyz).all(axis=1)] = np.inf
            owner = np.asarray(owner)
            left_out = np.asarray(left_out)
            for p in np.unique(owner):
                m = owner == p
                drop.append(int(left_out[m][np.argmin(worst[m])]))
        drop = np.asarray(drop, int)
        obs.used[drop] = False
        obs.reject_err[drop] = e[drop]
        n_new += len(drop)
        affected = np.unique(obs.pt[drop])
        fresh = _retriangulate(obs, cams, range(obs.C))
        pts3[affected] = fresh[affected]                  # NaN once fewer than 2 remain
        e = _errors(obs, cams, pts3)
    return thr, n_new


# ---------------------------------------------------------- the public entry


@dataclass
class _Run:
    """What one `calibrate_wand` call was asked and what it found out on the
    way, as `_finish` and the report need it."""
    sizes: list
    wand_length: float
    unit: str
    free_f: np.ndarray             # (C,) focal length refined in the final adjustment
    free_d: np.ndarray             # (C,) k1 / k2 fitted
    searched: np.ndarray           # (C,) focal length found by the grid search (I221)
    best_mult: float | None        # the grid multiple that won (None without a search)
    names: list | None
    frame_ids: np.ndarray | None   # reference frame of each wand_uv row (I248)
    coverage: list                 # % of each picture the wand visited, on the RAW points (I247)


def _prepare_flags(C: int, focal, estimate_focal, estimate_distortion):
    """The per-camera settings of a run -> (free_f, free_d, f0, unknown).
    `focal`: None (no camera's focal length is known), one value, or one per
    camera with NaN = unknown (I221: a rig where only SOME cameras carry a lens
    profile searches the grid for the others and holds the profiled ones)."""
    free_f = _per_camera(estimate_focal, C, "estimate_focal")
    free_d = _per_camera(estimate_distortion, C, "estimate_distortion")
    f0 = np.full(C, np.nan) if focal is None else _per_camera(focal, C, "focal", np.float64)
    if np.isinf(f0).any() or (np.isfinite(f0) & (f0 <= 0)).any():
        raise WandError("focal lengths must be positive numbers (px), or NaN where unknown")
    return free_f, free_d, f0, ~np.isfinite(f0)


def _search_focal(obs: _Obs, f0: np.ndarray, unknown: np.ndarray, pp: np.ndarray, sizes,
                  free_f: np.ndarray, f_scale: float, names, note):
    """Step 6: steps 1-3 for every `FOCAL_GRID` entry (applied to the cameras
    whose focal length is `unknown`; the others keep theirs) and the lowest
    bundle-adjustment cost wins -> (cams, pts3, the winning multiple)."""
    widths = np.array([s[0] for s in sizes], np.float64)
    ff = unknown | free_f                  # a known focal length stays put unless it is to be refined
    cands, errs = [], []
    for gi, mult in enumerate(FOCAL_GRID):
        try:
            cams, pts3 = _initialise(obs, np.where(unknown, mult * widths, f0), pp, sizes, unknown, names)
            cams, pts3, r = _bundle_adjust(obs, cams, pts3, free_f=ff, free_dist=False,
                                           loss="soft_l1", f_scale=f_scale, max_nfev=40)
        except WandError as ex:
            errs.append(f"f = {mult:g} x width: {ex}")
            continue
        cands.append((float(r.cost) / max(r.fun.size, 1), mult, cams, pts3))
        note(0.02 + 0.5 * (gi + 1) / len(FOCAL_GRID),
             f"focal guess {mult:g} x width: {np.sqrt(np.mean(r.fun ** 2)):.2f} px")
    if not cands:
        raise WandError("the cameras could not be initialised with any focal-length guess:\n"
                        + "\n".join(errs))
    cands.sort(key=lambda cnd: cnd[0])
    _, best_mult, cams, pts3 = cands[0]
    return cams, pts3, best_mult


def _refine_passes(obs: _Obs, cams: np.ndarray, pts3: np.ndarray, free_f: np.ndarray, free_d: np.ndarray,
                   f_scale: float, outlier_px, scale_all: float, note):
    """Bundle adjustment: robust pass -> reject -> linear pass -> reject ->
    final linear pass -> (cams, pts3, rejection threshold, observations set aside)."""
    thr, n_out = float("nan"), 0
    passes = (("soft_l1", True), ("linear", True), ("linear", False))
    for k, (loss, reject) in enumerate(passes):
        note(0.55 + 0.12 * k, f"bundle adjustment pass {k + 1}/{len(passes)}")
        cams, pts3, r = _bundle_adjust(obs, cams, pts3, free_f=free_f, free_dist=free_d,
                                       loss=loss, f_scale=f_scale, max_nfev=300)
        if reject:
            thr, n_new = _reject(obs, cams, pts3, outlier_px, OUTLIER_FLOOR_PX * scale_all)
            n_out += n_new
    if not np.isfinite(thr):
        thr = float(outlier_px) if outlier_px is not None else float("nan")
    return cams, pts3, thr, n_out


def _scale_and_centre(obs: _Obs, cams: np.ndarray, pts3: np.ndarray, wand_length: float):
    """Scale so the MEAN reconstructed wand length is the true length; then the
    origin moves to the wand centroid (camera 0's axes are kept)."""
    w = pts3[:obs.n_wand].reshape(-1, 2, 3)
    ln = np.linalg.norm(w[:, 0] - w[:, 1], axis=1)
    okl = np.isfinite(ln)
    if okl.sum() < 2:
        raise WandError("fewer than two frames have both wand ends reconstructed: the scale cannot "
                        "be set - digitize both ends in at least two cameras per frame")
    s = wand_length / float(ln[okl].mean())
    cams, pts3 = _apply_similarity(cams, pts3, np.eye(3), np.zeros(3), s)
    centre = w[okl].reshape(-1, 3).mean(axis=0) * s
    return _apply_similarity(cams, pts3, np.eye(3), -centre, 1.0)


def calibrate_wand(wand_uv, wand_length, sizes, *, focal=None, principal=None, bg_uv=None,
                   estimate_focal=True, estimate_distortion=False, unit="m", max_frames=400,
                   outlier_px=None, progress=None, names=None, frame_ids=None,
                   coverage_uv=None, coverage_bg_uv=None) -> WandResult:
    """Calibrate C cameras from a waved wand (see the module docstring).

    wand_uv: (F, C, 2, 2) float - [frame, camera, end, xy], 0-based pixels,
        NaN where an end is not digitized in that camera.
    wand_length: distance between the two wand markers, in `unit`.
    sizes: [(width, height)] per camera.
    focal: None (unknown: searched over `FOCAL_GRID`) or the initial focal
        length in px, one value or one per camera; NaN for a camera = unknown
        (I221: searched / refined for those cameras only).
    principal: None (image centre) or (cx, cy) per camera.
    bg_uv: None or (B, C, 2) background points (NaN where unseen).
    estimate_focal: leave the focal lengths free in the final adjustment. One
        bool, or one per camera (I144: cameras sharing one lens profile refine
        theirs while a camera with its own profile keeps its focal length).
        A camera whose `focal` is unknown is always refined.
    estimate_distortion: also fit k1, k2 per camera (needs good corner
        coverage; leave off for pre-undistorted points). One bool, or one per
        camera (I35: a rig where some cameras carry a lens profile fits k1/k2
        only for the others).
    max_frames: evenly subsample the wand frames beyond this many.
    outlier_px: fixed rejection threshold, or None for 3 x median (two passes).
    progress: optional callable(fraction, message).
    names: optional camera names for the messages (the project's cam1,
        GoPro2 ...); without them cameras are "camera 1", "camera 2" ...
    frame_ids: optional (F,) reference frame of each wand_uv row; the report's
        outlier list then names the frame, not the array row (I248; "row"
        always holds the row).
    coverage_uv / coverage_bg_uv: the RAW (not lens-straightened) wand / extra
        points the coverage figure is measured on (I247: a straightened box
        divided by the raw picture area read 33-52 % for a 25 % box). Default:
        `wand_uv` / `bg_uv` themselves.
    """
    wand_uv = np.asarray(wand_uv, np.float64)
    if wand_uv.ndim != 4 or wand_uv.shape[2:] != (2, 2):
        raise WandError(f"wand_uv must be (frames, cameras, 2 ends, xy), got {wand_uv.shape}")
    F, C = wand_uv.shape[:2]
    sizes = [(int(w), int(h)) for w, h in sizes]
    if len(sizes) != C:
        raise WandError(f"{len(sizes)} image sizes for {C} cameras")
    if C < 2:
        raise WandError("a wand calibration needs at least two cameras")
    widths = np.array([s[0] for s in sizes], np.float64)
    heights = np.array([s[1] for s in sizes], np.float64)
    if principal is None:
        pp = np.column_stack([(widths - 1.0) / 2.0, (heights - 1.0) / 2.0])
    else:
        pp = np.asarray(principal, np.float64).reshape(C, 2).copy()
    if bg_uv is not None:
        bg_uv = np.asarray(bg_uv, np.float64)
        if bg_uv.ndim != 3 or bg_uv.shape[1:] != (C, 2):
            raise WandError(f"bg_uv must be (points, {C} cameras, xy), got {bg_uv.shape}")
        if bg_uv.shape[0] == 0:
            bg_uv = None
    wand_length = float(wand_length)
    if not (wand_length > 0) or not np.isfinite(wand_length):
        raise WandError("the wand length must be a positive number")
    if frame_ids is not None:
        frame_ids = np.asarray(frame_ids, np.int64).ravel()
        if len(frame_ids) != F:
            raise WandError(f"frame_ids has {len(frame_ids)} entries for {F} wand frames")
    cov_w = wand_uv if coverage_uv is None else np.asarray(coverage_uv, np.float64)
    if cov_w.shape != wand_uv.shape:
        raise WandError(f"coverage_uv must have the shape of wand_uv {wand_uv.shape}, got {cov_w.shape}")
    cov_b = bg_uv if coverage_bg_uv is None else np.asarray(coverage_bg_uv, np.float64)
    names = list(names) if names is not None else None

    def note(frac, msg):
        if progress is not None:
            progress(float(frac), str(msg))

    if F > int(max_frames) > 0:
        sel = np.unique(np.round(np.linspace(0, F - 1, int(max_frames))).astype(int))
    else:
        sel = np.arange(F)
    obs = _Obs(wand_uv, bg_uv, sel)
    if obs.N < 2 * MIN_PAIR_POINTS:
        raise WandError(f"only {obs.N} digitized wand ends in total; digitize the wand in more frames")
    cv2.setRNGSeed(20260914)          # thread-local in OpenCV: deterministic RANSAC, no side effects
    scale_all = float(_px_scale(sizes).max())
    f_scale = 2.0 * scale_all          # soft-L1 knee, px
    free_f, free_d, f0, unknown = _prepare_flags(C, focal, estimate_focal, estimate_distortion)

    if unknown.any():
        note(0.02, "searching focal lengths")
        cams, pts3, best_mult = _search_focal(obs, f0, unknown, pp, sizes, free_f, f_scale, names, note)
    else:
        note(0.05, "initial camera poses")
        cams, pts3 = _initialise(obs, f0, pp, sizes, False, names)
        best_mult = None
    free_f = free_f | unknown                 # a focal length found by the search is refined too
    cams, pts3, thr, n_out = _refine_passes(obs, cams, pts3, free_f, free_d, f_scale, outlier_px, scale_all, note)
    cams, pts3 = _scale_and_centre(obs, cams, pts3, wand_length)
    note(0.95, "assembling the result")
    run = _Run(sizes, wand_length, str(unit), free_f, free_d, unknown, best_mult, names, frame_ids,
               _coverage(cov_w, cov_b, sizes))
    res = _finish(obs, cams, pts3, wand_uv, bg_uv, run, thr, n_out)
    note(1.0, f"done: wand score {res.report['wand_score_pct']:.2f}%, "
              f"{res.report['reproj_rmse_all']:.2f} px")
    return res


def _finish(obs, cams, pts3, wand_uv, bg_uv, run: _Run, thr, n_out) -> WandResult:
    F, C = wand_uv.shape[:2]
    sizes = run.sizes
    Fs = len(obs.frames)
    B = obs.n_bg
    e = _errors(obs, cams, pts3)
    Rm = _rodrigues(cams[:, :3])
    tv = cams[:, 3:6].copy()
    Km = np.stack([_K_of(cams, c) for c in range(C)])
    dist = np.zeros((C, 5))
    dist[:, 0] = cams[:, 7]
    dist[:, 1] = cams[:, 8]

    # ---- per-observation errors and masks
    err_w = np.full((F, C, 2), np.nan)
    used_w = np.zeros((F, C, 2), bool)
    err_b = np.full((B, C), np.nan)
    used_b = np.zeros((B, C), bool)
    iw = obs.src[:, 0] == 0
    err_w[obs.src[iw, 1], obs.src[iw, 2], obs.src[iw, 3]] = e[iw]
    used_w[obs.src[iw, 1], obs.src[iw, 2], obs.src[iw, 3]] = obs.used[iw] & np.isfinite(e[iw])
    ib = ~iw
    if ib.any():
        err_b[obs.src[ib, 1], obs.src[ib, 2]] = e[ib]
        used_b[obs.src[ib, 1], obs.src[ib, 2]] = obs.used[ib] & np.isfinite(e[ib])

    # ---- 3D wand ends: fitted frames from the adjustment, the rest (frames
    # beyond max_frames) triangulated afterwards with the final cameras
    wand_xyz = np.full((F, 2, 3), np.nan)
    wand_xyz[obs.frames] = pts3[:obs.n_wand].reshape(Fs, 2, 3)
    unsel = np.setdiff1d(np.arange(F), obs.frames)
    if len(unsel):
        # rows = (frame, end) pairs: transpose the end axis in front of the camera axis first
        uv = wand_uv[unsel].transpose(0, 2, 1, 3).reshape(-1, C, 2)
        xyz, er = _triangulate_uv(cams, uv)
        if np.isfinite(thr):
            drop = er > thr
            if drop.any():
                uv = uv.copy()
                uv[drop] = np.nan
                xyz, er = _triangulate_uv(cams, uv)
        wand_xyz[unsel] = xyz.reshape(len(unsel), 2, 3)
        err_w[unsel] = er.reshape(len(unsel), 2, C).transpose(0, 2, 1)
    wand_len = np.linalg.norm(wand_xyz[:, 0] - wand_xyz[:, 1], axis=1)
    frame_used = np.zeros(F, bool)
    frame_used[obs.frames] = np.isfinite(wand_len[obs.frames])
    bg_xyz = pts3[obs.n_wand:].copy() if B else np.zeros((0, 3))

    # ---- easyWand's wand score: every wand observation, no rejection,
    # triangulated with the final cameras (what its DLT-based score measures)
    all_xyz, _ = _triangulate_uv(cams, wand_uv.transpose(0, 2, 1, 3).reshape(-1, C, 2))
    all_xyz = all_xyz.reshape(F, 2, 3)
    all_len = np.linalg.norm(all_xyz[:, 0] - all_xyz[:, 1], axis=1)
    all_len = all_len[np.isfinite(all_len)]
    score_all = float(100.0 * all_len.std(ddof=1) / all_len.mean()) if len(all_len) > 1 else float("nan")

    # ---- reprojection statistics over the kept observations
    kept = obs.used & np.isfinite(e)
    reproj_cam = np.full(C, np.nan)
    for c in range(C):
        m = kept & (obs.cam == c)
        if m.any():
            reproj_cam[c] = float(np.sqrt(np.mean(e[m] ** 2)))
    rmse_all = float(np.sqrt(np.mean(e[kept] ** 2))) if kept.any() else float("nan")
    scale_c = _px_scale(sizes)
    rmse_eq = float(np.sqrt(np.mean((e[kept] / scale_c[obs.cam[kept]]) ** 2))) if kept.any() else float("nan")

    cameras = []
    for c in range(C):
        und = OpenCVUndistort(Km[c], dist[c]) if np.any(dist[c] != 0) else NoUndistort()
        cameras.append(CameraCalibration(dlt_from_camera(Km[c], Rm[c], tv[c]), sizes[c][0], sizes[c][1],
                                         und, 0.0, False, float(reproj_cam[c])))

    lens = wand_len[frame_used]
    wand_mean = float(lens.mean()) if len(lens) else float("nan")
    wand_sd = float(lens.std(ddof=1)) if len(lens) > 1 else float("nan")
    outliers = []
    for k in np.nonzero(~obs.used)[0]:
        s = obs.src[k]
        px = obs.reject_err[k] if np.isfinite(obs.reject_err[k]) else e[k]
        # (I248) "frame" is the REFERENCE frame when the caller said which rows they are; "row"
        # is always the array row (the report used to call the row "frame": row 37 = frame 370)
        fid = int(run.frame_ids[s[1]]) if (run.frame_ids is not None and s[0] == 0) else int(s[1])
        entry = {"kind": "wand", "frame": fid, "row": int(s[1]), "cam": int(s[2]), "end": int(s[3])} \
            if s[0] == 0 else {"kind": "bg", "index": int(s[1]), "cam": int(s[2])}
        entry["px"] = float(px) if np.isfinite(px) else None
        outliers.append(entry)
    report = _build_report(obs, e, kept, wand_uv, run, Rm, tv, cams, wand_mean, wand_sd, score_all,
                           reproj_cam, rmse_all, rmse_eq, frame_used, thr, n_out, outliers)
    return WandResult(cameras, Km, Rm, tv, dist, wand_xyz, wand_len, bg_xyz, frame_used, reproj_cam,
                      {"wand": err_w, "bg": err_b, "wand_used": used_w, "bg_used": used_b}, run.unit, report)


def _camera_centres(Rm: np.ndarray, tv: np.ndarray) -> np.ndarray:
    return -np.einsum("cji,cj->ci", Rm, tv)


def _camera_distances(Rm: np.ndarray, tv: np.ndarray) -> list:
    cen = _camera_centres(Rm, tv)
    C = len(cen)
    return [[i, j, float(np.linalg.norm(cen[i] - cen[j]))] for i in range(C) for j in range(i + 1, C)]


def _coverage(wand_uv: np.ndarray, bg_uv, sizes) -> list:
    """Convex-hull area of each camera's digitized points as % of the image."""
    F, C = wand_uv.shape[:2]
    out = []
    for c in range(C):
        pts = [wand_uv[:, c].reshape(-1, 2)]
        if bg_uv is not None:
            pts.append(bg_uv[:, c].reshape(-1, 2))
        p = np.concatenate(pts)
        p = p[np.isfinite(p).all(axis=1)]
        w, h = sizes[c]
        if len(p) < 3 or w <= 0 or h <= 0:
            out.append(0.0)
            continue
        hull = cv2.convexHull(p.astype(np.float32))
        out.append(float(100.0 * cv2.contourArea(hull) / (w * h)))
    return out


def _n_bg_reconstructed(obs: _Obs) -> int:
    """Background points with >= 2 kept observations (= reconstructed)."""
    if not obs.n_bg:
        return 0
    cnt = np.bincount(obs.pt[obs.used], minlength=obs.P)[obs.n_wand:]
    return int(np.sum(cnt >= 2))


class _Check(NamedTuple):
    """One named check behind the verdict (I176). `level` 0 = passes, 1 = the
    result can be at best USABLE, 2 = the result is poor; `text` is its reason
    (empty for a check that passes quietly)."""
    name: str
    level: int
    text: str


_ALLOWS = ("good", "ok", "poor")


def _verdict_checks(score, rmse_all, rmse_eq, rmse_ok, coverage, n_obs_cam, n_obs_all, n_used, n_out, n_tot,
                    thr, names, dist_fitted: bool) -> list:
    """The checks, in the order their reasons are shown. The reasons AND the
    verdict are derived from this one list (I176): the verdict used to look at
    the pooled score / error / coverage only and said GOOD for a rig in which
    one camera had lost nearly all its observations."""
    checks: list[_Check] = []
    add = lambda name, level, text="": checks.append(_Check(name, level, text))   # noqa: E731
    if not np.isfinite(score):
        add("wand_score", 2, "The wand length could not be measured in enough frames to judge the calibration.")
    elif score <= SCORE_GOOD_PCT:
        add("wand_score", 0, f"Wand length is consistent between frames (varies by {score:.2f}%).")
    elif score <= SCORE_OK_PCT:
        add("wand_score", 1, f"Wand length varies by {score:.1f}% between frames: wave the wand more slowly, "
                             f"keep both ends sharp in every camera, or check the length you typed.")
    else:
        add("wand_score", 2, f"Wand length varies by {score:.1f}% between frames - the calibration is unreliable: "
                             f"check the wand length you typed, that end 1 / end 2 are the same physical end in "
                             f"every camera, and that the videos are frame-synchronised.")
    if np.isfinite(rmse_eq):
        px_note = "" if rmse_ok <= 1.0 else f" ({rmse_all:.2f} px at this resolution)"
        if rmse_eq <= RMSE_GOOD_PX:
            add("reprojection", 0, f"Reprojection error is {rmse_all:.2f} px: the cameras agree well{px_note}.")
        elif rmse_eq <= RMSE_OK_PX:
            add("reprojection", 1, f"Reprojection error is {rmse_all:.2f} px{px_note}: acceptable; digitize the "
                                   f"marker centres more carefully or add background points for a tighter fit.")
        else:
            # (I35) do not tell someone who already fitted k1/k2 to "enable distortion estimation"
            lens_hint = ("lens distortion that the fitted k1/k2 could not describe (calibrate the lens with a "
                         "checkerboard instead)" if dist_fitted else
                         "lens distortion (undistort the points or enable distortion estimation)")
            add("reprojection", 2, f"Reprojection error is {rmse_all:.2f} px{px_note}: too high - look for "
                                   f"mis-clicked or swapped wand ends, a sync error between videos, or {lens_hint}.")
    else:
        add("reprojection", 2, "No observation could be reprojected with the solved cameras.")
    for c, cov in enumerate(coverage):
        low = cov < MIN_COVERAGE_PCT
        add(f"coverage_{c}", int(low),
            f"{_cam(c, names, cap=True)}: the wand covered only {cov:.0f}% of the image; wave it through more "
            f"of this camera's view (corners included) so the whole lens is calibrated." if low else "")
    for c, n in enumerate(n_obs_cam):
        if n < MIN_PNP_POINTS:
            add(f"observations_{c}", 2,
                f"{_cam(c, names, cap=True)} kept only {n} usable observation{'s' if n != 1 else ''}, fewer than "
                f"the {MIN_PNP_POINTS} a camera needs to be placed: its part of this calibration cannot be "
                f"trusted. Digitize more wand frames visible in this camera, and check its frame offset.")
        elif n < MIN_OBS_PER_CAMERA:
            add(f"observations_{c}", 1,
                f"{_cam(c, names, cap=True)} contributed only {n} usable observations; digitize more "
                f"wand frames visible in this camera.")
        b = int(n_obs_all[c])
        if b > 0 and n < MIN_OBS_KEPT_FRAC * b:
            add(f"kept_{c}", 1,
                f"{_cam(c, names, cap=True)} lost {b - n} of its {b} observations ({100.0 * (b - n) / b:.0f}%) to "
                f"outlier rejection: look for a sync error (frame offset) between this camera and the others, "
                f"or wand ends swapped in it.")
    if n_used < MIN_FRAMES_USED:
        add("frames", 0, f"Only {n_used} wand frames were used; 20-50 well-spread wand positions give a stable "
                         f"calibration.")
    if n_out:
        pct = 100.0 * n_out / max(n_tot, 1)
        msg = f"{n_out} of {n_tot} observations ({pct:.1f}%) were rejected as outliers (> {thr:.2f} px)."
        if pct > MAX_REJECTED_PCT:
            msg += " That is a lot: look for mis-clicked wand ends, swapped ends or a sync error."
        add("rejected", int(pct > MAX_REJECTED_PCT), msg)
    return checks


def _focal_reasons(cams: np.ndarray, run: _Run) -> list:
    """How the focal lengths came about, in words (informational)."""
    C = len(cams)
    names, sr, ff = run.names, run.searched, run.free_f
    if sr.all():
        return ["Focal lengths were estimated from the data (f = "
                + ", ".join(f"{f:.0f}" for f in cams[:, 6]) + " px); if you know them from the lens "
                "specification, entering them makes the result more stable."]
    if sr.any():
        found = [c for c in range(C) if sr[c]]
        rest = [c for c in range(C) if not sr[c]]
        return ["Focal lengths were GUESSED from the wand alone for " + ", ".join(_cam(c, names) for c in found)
                + " (f = " + ", ".join(f"{cams[c, 6]:.0f}" for c in found) + " px): a lens profile or a known "
                "focal length makes those cameras more certain. " + ", ".join(_cam(c, names) for c in rest)
                + (" started from the values you gave and were refined." if ff[rest].any()
                   else " kept the values you gave.")]
    if ff.all():
        return ["Focal lengths were refined from your initial values."]
    if ff.any():
        return ["Focal lengths were refined for " + ", ".join(_cam(c, names) for c in range(C) if ff[c])
                + " (from their starting values) and kept as given for "
                + ", ".join(_cam(c, names) for c in range(C) if not ff[c]) + "."]
    return []


def _build_report(obs, e, kept, wand_uv, run: _Run, Rm, tv, cams, wand_mean, wand_sd, score_all, reproj_cam,
                  rmse_all, rmse_eq, frame_used, thr, n_out, outliers) -> dict:
    F, C = wand_uv.shape[:2]
    sizes, names, est_dist, free_f = run.sizes, run.names, run.free_d, run.free_f
    score = float(100.0 * wand_sd / wand_mean) if np.isfinite(wand_sd) and wand_mean > 0 else float("nan")
    coverage = run.coverage
    n_obs_cam = [int(np.sum(kept & (obs.cam == c))) for c in range(C)]
    n_obs_all = np.bincount(obs.cam, minlength=C)
    n_in = int(np.sum(np.isfinite(wand_uv).all(axis=3).any(axis=(1, 2))))
    n_used = int(frame_used.sum())
    n_bg = _n_bg_reconstructed(obs)
    rmse_ok = float(_px_scale(sizes).max())
    n_tot = int(obs.N)

    checks = _verdict_checks(score, rmse_all, rmse_eq, rmse_ok, coverage, n_obs_cam, n_obs_all, n_used,
                             n_out, n_tot, thr, names, bool(est_dist.any()))
    reasons = [c.text for c in checks if c.text]
    reasons += _focal_reasons(cams, run)
    weak_dist = [c for c in range(C) if est_dist[c] and coverage[c] < MIN_COVERAGE_PCT]
    if weak_dist:
        reasons.append("Lens distortion was fitted, but the wand covered less than a quarter of the image "
                       f"in {', '.join(_cam(c, names) for c in weak_dist)}: the distortion terms are poorly "
                       "constrained there and may misbehave near the edges - cover the corners, or switch "
                       "distortion off.")
    e_all = np.where(np.isfinite(e), e, obs.reject_err)
    e_all = e_all[np.isfinite(e_all)]
    rmse_incl = float(np.sqrt(np.mean(e_all ** 2))) if len(e_all) else float("nan")

    verdict = _ALLOWS[max(c.level for c in checks)]
    return {
        "n_cameras": int(C),
        "n_frames_in": n_in,
        "n_frames_used": n_used,
        "n_bg_points": n_bg,
        "wand_length": float(run.wand_length),
        "wand_mean": float(wand_mean),
        "wand_sd": float(wand_sd),
        "wand_score_pct": score,
        "wand_score_all_obs_pct": score_all,
        "reproj_rmse_px": [float(v) for v in reproj_cam],
        "reproj_rmse_all": float(rmse_all),
        "reproj_rmse_incl_outliers": rmse_incl,
        "reproj_rmse_1080p_equiv": float(rmse_eq),
        "focal_px": [float(f) for f in cams[:, 6]],
        "principal_px": [[float(a), float(b)] for a, b in cams[:, 9:11]],
        "focal_estimated": bool(free_f.any()),
        "focal_refined": [bool(v) for v in free_f],
        "focal_searched": bool(run.searched.any()),
        "focal_searched_per_camera": [bool(v) for v in run.searched],
        "focal_grid_best": (None if run.best_mult is None else float(run.best_mult)),
        "principal_estimated": False,
        "distortion_estimated": bool(est_dist.any()),
        "distortion_estimated_per_camera": [bool(v) for v in est_dist],
        "distortion": [[float(cams[c, 7]), float(cams[c, 8])] for c in range(C)],
        "image_sizes": [[int(w), int(h)] for w, h in sizes],
        "coverage_pct": [float(v) for v in coverage],
        "camera_distances": _camera_distances(Rm, tv),
        "n_observations": n_tot,
        "n_observations_per_camera": n_obs_cam,
        "n_observations_total_per_camera": [int(v) for v in n_obs_all],
        "outlier_threshold_px": (float(thr) if np.isfinite(thr) else None),
        "outliers_removed": int(n_out),
        "outliers": outliers,
        "camera_names": [_cam(c, names) for c in range(C)],
        "frame": f"{_cam(0, names)}'s axes, origin at the centroid of the wand ends",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "verdict_checks": [{"name": c.name, "allows": _ALLOWS[c.level]} for c in checks],
    }


# ------------------------------------------------------------ re-expression


def _cams_from_result(res: WandResult) -> np.ndarray:
    C = len(res.R)
    cams = np.zeros((C, _NCP))
    cams[:, :3] = np.stack([_rvec(R) for R in res.R])
    cams[:, 3:6] = res.t
    cams[:, 6] = res.K[:, 0, 0]
    cams[:, 7] = res.dist[:, 0]
    cams[:, 8] = res.dist[:, 1]
    cams[:, 9:11] = res.K[:, :2, 2]
    return cams


def transform_result(res: WandResult, R_w: np.ndarray, t_w: np.ndarray, scale: float = 1.0) -> WandResult:
    """Apply the similarity X' = scale * R_w X + t_w to the world: cameras
    (R' = R R_w^T, t' = scale t - R' t_w - see `_apply_similarity`), every 3D
    point, the wand lengths and the DLT coefficients. Reprojection errors are
    unchanged. Returns a new result."""
    R_w = np.asarray(R_w, np.float64).reshape(3, 3)
    t_w = np.asarray(t_w, np.float64).reshape(3)
    s = float(scale)
    if not np.allclose(R_w @ R_w.T, np.eye(3), atol=1e-6) or np.linalg.det(R_w) < 0:
        raise WandError("R_w must be a proper rotation matrix")
    if not (s > 0) or not np.isfinite(s):
        raise WandError("scale must be positive")
    Rn = res.R @ R_w.T
    tn = s * res.t - np.einsum("cij,j->ci", Rn, t_w)

    def tr(X):
        return s * (X @ R_w.T) + t_w

    wand_xyz = tr(res.wand_xyz.reshape(-1, 3)).reshape(res.wand_xyz.shape)
    bg_xyz = tr(res.bg_xyz) if len(res.bg_xyz) else res.bg_xyz.copy()
    cameras = []
    names = res.report.get("camera_names")
    for c, cam in enumerate(res.cameras):
        if abs(tn[c, 2]) < 1e-9:
            raise WandError(f"the new world origin lies in {_cam(c, names)}'s principal plane, where the DLT "
                            f"form is undefined; choose a different origin")
        cameras.append(CameraCalibration(dlt_from_camera(res.K[c], Rn[c], tn[c]), cam.width, cam.height,
                                         cam.undistort, 0.0, False, cam.rmse))
    report = copy.deepcopy(res.report)
    report["camera_distances"] = _camera_distances(Rn, tn)
    if s != 1.0:
        for key in ("wand_mean", "wand_sd"):
            if key in report and report[key] is not None and np.isfinite(report[key]):
                report[key] = float(report[key] * s)
    return WandResult(cameras, res.K.copy(), Rn, tn, res.dist.copy(), wand_xyz, res.wand_len * s,
                      bg_xyz, res.frame_used.copy(), res.reproj_cam.copy(),
                      {k: v.copy() for k, v in res.reproj_obs.items()}, res.unit, report)


def keep_camera_frame(res: WandResult, reason: str, what: str = "the chosen world frame") -> WandResult:
    """The result left in camera 1's axes because a world-frame step could not
    be applied (I25, I29): a copy whose report says so in `frame`, carries
    `reason` among the verdict reasons, and whose verdict is capped at 'ok'.
    The calibration itself is sound, but a headline saying "trust this" about
    a world frame the user did not get would send them off without reading
    why."""
    rep = copy.deepcopy(res.report)
    rep["frame"] = (f"{_cam(0, rep.get('camera_names'))}'s axes, origin at the centroid of the wand ends "
                    f"({what} was NOT applied)")
    rep["frame_applied"] = False
    rep.setdefault("verdict_reasons", []).append(reason)
    if rep.get("verdict") == "good":
        rep["verdict"] = "ok"
    return replace(res, report=rep)


def _path_noise(xyz: np.ndarray, ok: np.ndarray) -> float:
    """Frame-to-frame jitter of a 3D path, in its own units, from the third
    differences over consecutive reconstructed frames: a free fall is a
    quadratic in time, so its third difference is pure noise (20 sigma^2 per
    axis), and a bounce or a catch spoils only the few differences around it,
    which the median ignores. Returned as a 3D rms (sqrt(3) sigma, the same
    measure as a fit's rms). NaN without 4 consecutive frames."""
    if len(xyz) < 4:
        return float("nan")
    d3 = xyz[3:] - 3.0 * xyz[2:-1] + 3.0 * xyz[1:-2] - xyz[:-3]
    good = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
    if not good.any():
        return float("nan")
    m = float(np.median(np.linalg.norm(d3[good], axis=1)))
    return m * np.sqrt(3.0) / (1.538 * np.sqrt(20.0))      # median of a chi(3) variable = 1.538


def align_gravity(res: WandResult, drop_uv, fps: float, g: float | None = None,
                  frame0: int | None = None) -> tuple[WandResult, dict]:
    """World frame from a dropped object: `drop_uv` (N, C, 2) are consecutive
    frames of the falling object at `fps`, digitized in >= 2 cameras (NaN
    elsewhere). Its triangulated path is fitted with a quadratic in time
    (with one 3 x median rejection pass for stray frames); the acceleration
    vector points DOWN (-Z) and its magnitude is compared with g - an
    INDEPENDENT check of the wand length and frame rate (g_ratio = 1.00 when
    both are right). The new frame has Z up, the origin at the first
    reconstructed drop point and X along camera 1's viewing direction
    projected onto the horizontal plane. Returns (new result, check dict).

    g: gravity in the WORLD's unit. None (the default) takes it from
    `res.unit` (`G_BY_UNIT`, I27); for "wand" (length not measured) there is
    no ratio and the fall instead says how long the wand is.
    frame0: the reference frame of the first row of `drop_uv`, for messages.

    The vertical is set only when the frames look like ONE free fall (I25):
    the object moved well beyond its jitter, one parabola describes its path
    to within that jitter (or 3 % of the path), it moved the way it
    accelerated (not rising after a bounce), and - in a real unit - the
    acceleration is within `G_RATIO_RANGE` of g. Otherwise the result keeps
    camera 1's axes (`keep_camera_frame`: verdict capped at ok, a reason
    naming the frames) and the check dict says why (`applied` False): a
    resting mark once set the vertical from noise and a bounce turned the
    world upside down, both under a GOOD headline."""
    drop_uv = np.asarray(drop_uv, np.float64)
    if drop_uv.ndim != 3 or drop_uv.shape[1] != len(res.R):
        raise WandError(f"drop_uv must be (frames, {len(res.R)} cameras, xy), got {drop_uv.shape}")
    if not (float(fps) > 0):
        raise WandError("fps must be positive")
    names = res.report.get("camera_names")
    unit = str(res.unit)
    if g is None:
        g = G_BY_UNIT.get(unit)
    cams = _cams_from_result(res)
    xyz, err = _triangulate_uv(cams, drop_uv)
    ok = np.isfinite(xyz).all(axis=1)
    if ok.sum() < 4:
        raise WandError(f"the drop is reconstructed in only {int(ok.sum())} frames (>= 2 cameras each); "
                        f"at least 4 are needed")
    ok_all = ok.copy()
    N = len(drop_uv)
    tt = np.arange(N) / float(fps)

    def fit(sel):
        A = np.column_stack([np.ones(int(sel.sum())), tt[sel], 0.5 * tt[sel] ** 2])
        coef = np.linalg.lstsq(A, xyz[sel], rcond=None)[0]
        return coef, np.linalg.norm(A @ coef - xyz[sel], axis=1)

    coef, resid = fit(ok)
    if ok.sum() >= 6:
        keep = ok.copy()
        keep[ok] = resid <= max(3.0 * float(np.median(resid)), 1e-9)
        if 4 <= keep.sum() < ok.sum():
            ok = keep
            coef, resid = fit(ok)
    a = coef[2]
    g_meas = float(np.linalg.norm(a))

    # ---- is this ONE free fall? (unit-free tests first, then g itself)
    idx = np.nonzero(ok)[0]
    base = int(frame0) if frame0 is not None else 0
    f_lo, f_hi = base + int(idx[0]), base + int(idx[-1])
    where = f"frames {f_lo}-{f_hi}" if frame0 is not None else f"the {len(idx)} drop frames"
    P = xyz[ok]
    fit_rms = float(np.sqrt(np.mean(resid ** 2)))
    noise = _path_noise(xyz, ok_all)
    noise0 = noise if np.isfinite(noise) else 0.0
    span = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    dur = float(tt[idx[-1]] - tt[idx[0]])
    fall = 0.5 * g_meas * dur * dur                     # how far the fitted acceleration moved it
    ratio = (g_meas / float(g)) if g else None
    why = None
    if not (g_meas > 0) or fall < 5.0 * max(fit_rms, noise0):
        why = "it hardly moved - its fall is lost in the tracking jitter"
    elif fit_rms > max(4.0 * noise0, 0.03 * span):
        why = "one parabola does not describe its path: it rested, bounced or was caught on some of them"
    elif float((P[-1] - P[0]) @ a) <= 0.0:
        why = "it moved against its own acceleration (rising after a bounce, or thrown upwards)"
    elif ratio is not None and not (G_RATIO_RANGE[0] <= ratio <= G_RATIO_RANGE[1]):
        why = (f"it accelerated at {g_meas:.2f} {unit}/s^2, {100.0 * (ratio - 1.0):+.0f}% away from g "
               f"({float(g):.2f} {unit}/s^2): either it was not falling freely, or the wand length, its unit "
               f"or the frame rate ({fps:g} fps) is wrong")
    info = {"g_measured": g_meas, "g_expected": (float(g) if g else None), "g_ratio": ratio, "unit": unit,
            "residual_px": float(np.nanmean(err[ok])), "n_frames_used": int(ok.sum()),
            "fit_rms": fit_rms, "path_noise": (float(noise) if np.isfinite(noise) else None),
            "path_extent": span, "frames": [f_lo, f_hi]}
    if why is not None:
        info["applied"] = False
        info["why"] = why
        out = keep_camera_frame(
            res, f"The vertical was NOT set from the dropped object: on {where} {why}. Pick only the frames "
                 f"between its release and the moment it lands, bounces or is caught (the 'Which way is up' "
                 f"page finds them), or choose another way to set the frame. The calibration itself is not "
                 f"affected; its 3D keeps {_cam(0, names)}'s axes.", "the vertical from the dropped object")
        out.report["gravity_check"] = dict(info)
        return out, info

    ez = -a / g_meas
    d = res.R[0].T @ np.array([0.0, 0.0, 1.0])            # camera 0's optical axis in the world
    ex = d - (d @ ez) * ez
    if np.linalg.norm(ex) < 1e-6:                          # looking straight down: use its x axis
        d = res.R[0].T @ np.array([1.0, 0.0, 0.0])
        ex = d - (d @ ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    R_w = np.stack([ex, ey, ez])
    origin = xyz[np.nonzero(ok)[0][0]]
    out = transform_result(res, R_w, -R_w @ origin, 1.0)
    info["applied"] = True
    info["accel_world"] = (R_w @ a).tolist()
    out.report["frame"] = (f"gravity: Z up, origin = first drop point, X = {_cam(0, names)}'s horizontal "
                           f"viewing direction")
    if ratio is None:
        # "wand lengths" (I27): no g to compare with - the fall MEASURES the unit instead
        L = float(res.report.get("wand_length") or 1.0)
        m_per_unit = 9.81 / g_meas
        info["implied_wand_length_m"] = L * m_per_unit
        word = "wand lengths" if unit == "wand" else unit
        out.report["verdict_reasons"].append(
            f"The dropped object accelerates at {g_meas:.3f} {word}/s^2. If the frame rate ({fps:g} fps) is "
            f"right, the wand is about {L * m_per_unit:.3f} m long centre to centre: measure it to confirm, "
            f"then calibrate again with that length in metres to get every result in metres.")
    else:
        ratio_pct = 100.0 * (ratio - 1.0)
        if abs(ratio_pct) <= G_CONSISTENT_PCT:
            out.report["verdict_reasons"].append(
                f"The dropped object accelerates at {g_meas:.2f} {unit}/s^2 ({ratio_pct:+.1f}% of g): "
                f"wand length and frame rate are consistent.")
        else:
            msg = (f"The dropped object accelerates at {g_meas:.2f} {unit}/s^2 ({ratio_pct:+.1f}% of g, "
                   f"expected {float(g):.2f} {unit}/s^2): check the wand length, its unit and the frame rate "
                   f"({fps:g} fps) - one of them is off.")
            if abs(ratio_pct) > G_SUSPECT_PCT and out.report.get("verdict") == "good":
                # an independent scale check that disagrees by > 5 % is not "trust this"
                out.report["verdict"] = "ok"
                msg += " Until it agrees, every distance in this world is suspect."
            out.report["verdict_reasons"].append(msg)
    out.report["gravity_check"] = dict(info)
    return out, info


def align_axes(res: WandResult, origin_uv, x_uv, xy_uv) -> WandResult:
    """World frame from three reference points digitized in >= 2 cameras
    each ((C, 2) arrays, NaN where unseen): the origin, a point on the +X
    axis and a point in the +XY half-plane. Right-handed: Z = X x (XY)."""
    C = len(res.R)
    uv = np.stack([np.asarray(p, np.float64).reshape(C, 2) for p in (origin_uv, x_uv, xy_uv)])
    cams = _cams_from_result(res)
    xyz, _ = _triangulate_uv(cams, uv)
    names = ("origin", "+X", "XY-plane")
    missing = [names[k] for k in range(3) if not np.isfinite(xyz[k]).all()]
    if missing:
        raise WandError("reference point(s) not seen by two cameras: " + ", ".join(missing))
    O, A, Bp = xyz
    ex = A - O
    if np.linalg.norm(ex) < 1e-9:
        raise WandError("the +X point coincides with the origin")
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, Bp - O)
    if np.linalg.norm(ez) < 1e-9:
        raise WandError("the three reference points are collinear")
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    R_w = np.stack([ex, ey, ez])
    out = transform_result(res, R_w, -R_w @ O, 1.0)
    out.report["frame"] = "axes: origin / +X / XY-plane reference points"
    return out


# -------------------------------------------------------------------- files


def export_dlt_csv(res_or_cal, path, pixel_origin_out: float = 1.0) -> None:
    """Write a DLTdv / easyWand style `dltCoefs.csv`: 11 rows, one column per
    camera, no header. The coefficients are converted to `pixel_origin_out`
    (1.0 = MATLAB pixels): with u' = u + d the numerator gains d x the
    denominator, i.e. L1..3 += d L9..11, L4 += d (and the v row likewise)."""
    cams = res_or_cal.cameras
    cols = []
    for cam in cams:
        L = np.asarray(cam.coefs, np.float64).reshape(11).copy()
        d = float(pixel_origin_out) - float(getattr(cam, "pixel_origin", 0.0))
        if d != 0.0:
            L[0:3] += d * L[8:11]
            L[3] += d
            L[4:7] += d * L[8:11]
            L[7] += d
        cols.append(L)
    M = np.stack(cols, axis=1)
    lines = [",".join(f"{v:.12g}" for v in row) for row in M]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


