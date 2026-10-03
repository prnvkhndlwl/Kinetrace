"""Camera calibrations, lens profiles, offsets and 3D points in the formats of
other programs (no Qt). The menus and the command-line converter call these;
there is no second implementation.

Everything goes through ONE intermediate form, `CamModel`: per camera K, R, t,
distortion and picture size in OpenCV's conventions (x_cam = R X + t, pixel
centres at whole numbers counted from 0, distortion (k1, k2, p1, p2, k3) or 4
fisheye coefficients). A Kinetrace calibration becomes CamModels with
`to_models`:

  * the DLT is decomposed (RQ) into K [R | t]. easyWand / DLTdv coefficients
    are LEFT-handed (det < 0 while the animal is in front), which no proper
    rotation can express: the exported world is then mirrored in Z and the
    report says so. 3D points are in the SAME world only when they are
    written with these models (`write_points3d(..., models=...)`, which
    applies `Models.world_to_export`); the exports (Ctrl+E, exports/, the
    converter) must pass them. A world chosen with 3D -> Set World Axes
    (`calib.world_axes`) is right-handed by construction: nothing is mirrored
    and the points need no change;
  * a lens model other than OpenCV's (DLTdv's LWM) is fitted with an OpenCV
    model over the whole picture and its worst error reported, never dropped
    silently;
  * a calibration imported from K + R/t had its origin moved (`origin_shift`);
    the export moves it back, so the file's own world comes out again;
  * every model is checked by projecting points of the working volume through
    both Kinetrace's calibration and the exported model (`check_px`).

Writers: Anipose calibration.toml, OpenCV YAML / JSON (cv2.FileStorage),
MATLAB .mat, a Blender camera script. Readers: Anipose, OpenCV YAML / JSON,
MATLAB .mat. Also: lens profiles as OpenCV YAML / JSON and Argus lines, camera
offsets as CSV, 3D points as Anipose / DLTdv xyzpts / Kinetrace CSV.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kinetrace.calib import (Calibration, CameraCalibration, NoUndistort, OpenCVUndistort, Reconstruction,
                             dlt_matrix, front_sign, working_probe, world_is_left_handed)


class CalibFormatError(Exception):
    """A file that cannot be read or written; the message says why in plain words."""


def _read_text(path, encoding: str = "utf-8-sig") -> str:
    """A text file, or a CalibFormatError naming it (missing, locked, not text)."""
    try:
        return Path(path).read_text(encoding=encoding)
    except UnicodeDecodeError:
        raise CalibFormatError(f"{Path(path).name}: not a text file") from None
    except OSError as e:
        raise CalibFormatError(f"{Path(path).name}: cannot be read ({e.strerror or e})") from None


@dataclass
class CamModel:
    name: str
    width: int
    height: int
    K: np.ndarray                       # 3x3, pixel centres at whole numbers from 0
    dist: np.ndarray                    # (5,) k1 k2 p1 p2 k3, or (4,) fisheye
    R: np.ndarray                       # 3x3 world -> camera, det +1
    t: np.ndarray                       # (3,)
    fisheye: bool = False
    skew: float = 0.0                   # K[0, 1] the DLT implied (dropped: OpenCV has none)
    lens_fit_px: float = 0.0            # worst error of a fitted lens model (0 = exact)
    check_px: float = 0.0               # worst difference to Kinetrace's own projection

    @property
    def rvec(self) -> np.ndarray:
        import cv2
        return cv2.Rodrigues(self.R)[0].ravel()

    def project(self, X: np.ndarray) -> np.ndarray:
        import cv2
        X = np.asarray(X, np.float64).reshape(-1, 1, 3)
        if self.fisheye:
            uv, _ = cv2.fisheye.projectPoints(X, self.rvec, self.t, self.K, self.dist.reshape(-1, 1)[:4])
        else:
            uv, _ = cv2.projectPoints(X, self.rvec, self.t, self.K, self.dist)
        return uv.reshape(-1, 2)

    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Models:
    cameras: list[CamModel]
    unit: str = ""
    mirrored: bool = False              # the exported world is Kinetrace's mirrored in Z
    shift: np.ndarray = field(default_factory=lambda: np.zeros(3))   # added before the mirror
    notes: list[str] = field(default_factory=list)

    def world_to_export(self, X: np.ndarray) -> np.ndarray:
        """Kinetrace world points -> the exported cameras' world (for 3D points
        written beside these cameras)."""
        X = np.asarray(X, np.float64) + self.shift
        if self.mirrored:
            X = X * np.array([1.0, 1.0, -1.0])
        return X


# ------------------------------------------------------------------ DLT -> K [R | t]
def _rq(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M = K R with K upper-triangular (positive diagonal) and R orthogonal."""
    P = np.flipud(np.eye(3))
    Q, U = np.linalg.qr((P @ M).T)
    K = P @ U.T @ P
    R = P @ Q.T
    S = np.diag(np.sign(np.diag(K)))
    return K @ S, S @ R


def _volume_points(cam: CameraCalibration, probe: np.ndarray | None, n: int = 12):
    """World points spread over this camera's picture at the working distance."""
    c = cam.center()
    depth = float(np.linalg.norm(probe - c)) if probe is not None else 1.0
    from kinetrace.calib import dlt_ray
    W, H = max(cam.width, 2), max(cam.height, 2)
    gx, gy = np.meshgrid(np.linspace(0.05, 0.95, n) * (W - 1), np.linspace(0.05, 0.95, n) * (H - 1))
    raw = np.column_stack([gx.ravel(), gy.ravel()])
    und = cam.to_calib_frame(raw)
    ok = np.isfinite(und).all(axis=1)
    _, d = dlt_ray(cam.coefs, und[ok], probe)
    pts = []
    for s in (0.8, 1.0, 1.25):
        pts.append(c + d * depth * s)
    return np.vstack(pts)


def to_models(cal: Calibration, names: list[str] | None = None, probe: np.ndarray | None = None) -> Models:
    """Kinetrace calibration -> OpenCV-convention cameras (see the module
    docstring). `probe`: a world point in view (a reconstruction's centroid);
    the calibration's own working point otherwise."""
    if not cal.cameras:
        raise CalibFormatError("the calibration has no cameras")
    if any(c.width <= 0 or c.height <= 0 for c in cal.cameras):
        raise CalibFormatError("the calibration does not record every camera's picture size; open it with "
                               "the project's videos so the sizes are known")
    probe = probe if probe is not None else working_probe(cal)
    Ps = []
    for cam in cal.cameras:
        P = cam.pixel_matrix() @ dlt_matrix(cam.coefs)
        P = P * front_sign(cam.coefs, probe)           # third row = depth, positive in front
        Ps.append(P)
    mirrored = world_is_left_handed(cal, probe)        # the one handedness test (calib)
    F = np.diag([1.0, 1.0, -1.0]) if mirrored else np.eye(3)
    shift = (np.asarray(cal.origin_shift, np.float64).reshape(3) if cal.origin_shift is not None
             else np.zeros(3))
    out: list[CamModel] = []
    notes: list[str] = []
    for i, (cam, P) in enumerate(zip(cal.cameras, Ps)):
        name = names[i] if names and i < len(names) else f"cam{i + 1}"
        M = P[:, :3] @ F
        if np.linalg.det(M) <= 0:
            raise CalibFormatError(f"{name}: its coefficients are mirrored differently from the other cameras'; "
                                   "they cannot share one world in a K + R/t format")
        K0, R = _rq(M)
        lam = K0[2, 2]
        K0 = K0 / lam
        t = np.linalg.solve(K0, P[:, 3]) / lam
        # so far x_cam ~ R F X + t (R proper, taken from M = P F). The exported
        # world is X_e = F (X + shift), i.e. X = F X_e - shift:
        # x_cam ~ R X_e + (t - R F shift)
        t = t - R @ (F @ shift)
        skew = float(K0[0, 1])
        K = K0.copy()
        K[0, 1] = 0.0
        und = cam.undistort
        fisheye = isinstance(und, OpenCVUndistort) and und.fisheye
        dist = np.zeros(4 if fisheye else 5)
        fit = 0.0
        if not isinstance(und, NoUndistort):
            K, dist, fit = _fit_lens(cam, K0, fisheye)
        m = CamModel(name, int(cam.width), int(cam.height), K, dist, R, t, fisheye, skew, fit)
        X = _volume_points(cam, probe)
        ref = cam.project(X)
        Xe = X + shift
        if mirrored:
            Xe = Xe * np.array([1.0, 1.0, -1.0])
        got = m.project(Xe)
        ok = np.isfinite(ref).all(axis=1) & np.isfinite(got).all(axis=1)
        m.check_px = float(np.max(np.linalg.norm(ref[ok] - got[ok], axis=1))) if ok.any() else float("nan")
        out.append(m)
        if abs(skew) > 1e-3 * K[0, 0]:
            notes.append(f"{name}: the DLT implied a skew of {skew:.2f} px that OpenCV-style formats "
                         "cannot hold; it was left out")
        if fit > 0.5:
            notes.append(f"{name}: its lens correction is not OpenCV's; the OpenCV model fitted to it is off "
                         f"by up to {fit:.2f} px")
    if mirrored:
        notes.append("These coefficients describe a mirrored (left-handed) world, as easyWand / DLTdv "
                     "coefficients often do. The exported cameras' world is Kinetrace's mirrored in Z "
                     "(Z -> -Z). 3D points are in that same world only when they are exported together "
                     "with these cameras; to get a right-handed world in Kinetrace itself (nothing mirrored "
                     "anywhere), choose the axes with 3D -> Set World Axes.")
    if np.any(shift):
        notes.append("The world origin is the calibration file's own again (Kinetrace had moved it to the "
                     "point the cameras look at).")
    worst = max((m.check_px for m in out if np.isfinite(m.check_px)), default=0.0)
    notes.append(f"Check: the exported cameras put points of the working volume within {worst:.3f} px of "
                 "where Kinetrace's calibration puts them.")
    return Models(out, cal.unit, mirrored, shift, notes)


def _fit_lens(cam: CameraCalibration, K0: np.ndarray, fisheye: bool):
    """An OpenCV lens model reproducing this camera's own lens correction:
    raw pixels vs the normalised rays K0^-1 (undistorted pixels), sampled over
    the whole picture; K and the distortion coefficients are fitted."""
    import cv2
    from scipy.optimize import least_squares
    W, H = cam.width, cam.height
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 41), np.linspace(0, H - 1, 31))
    raw = np.column_stack([gx.ravel(), gy.ravel()])
    und_file = cam.to_calib_frame(raw)                  # the calibration's own undistorted pixels
    A = cam.pixel_matrix()
    und = und_file @ A[:2, :2].T + A[:2, 2]
    ok = np.isfinite(und).all(axis=1)
    raw, und = raw[ok], und[ok]
    n = np.column_stack([und, np.ones(len(und))]) @ np.linalg.inv(K0).T
    obj = np.column_stack([n[:, :2], np.ones(len(n))]).reshape(-1, 1, 3)
    zero = np.zeros(3)
    d0 = cam.undistort.dist[:4 if fisheye else 5] if isinstance(cam.undistort, OpenCVUndistort) else None
    ndist = 4 if fisheye else 5
    x0 = np.concatenate([[K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]],
                         np.pad(d0, (0, ndist - len(d0))) if d0 is not None else np.zeros(ndist)])

    def proj(x):
        K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1.0]])
        if fisheye:
            uv, _ = cv2.fisheye.projectPoints(obj, zero, zero, K, x[4:8].reshape(-1, 1))
        else:
            uv, _ = cv2.projectPoints(obj, zero, zero, K, x[4:9])
        return uv.reshape(-1, 2)

    res = least_squares(lambda x: (proj(x) - raw).ravel(), x0, method="lm", max_nfev=4000)
    x = res.x
    err = np.linalg.norm(proj(x) - raw, axis=1)
    K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1.0]])
    return K, np.asarray(x[4:], np.float64), float(err.max()) if len(err) else float("nan")


def from_models(models: list[CamModel], unit: str = "", source: str = "") -> Calibration:
    """OpenCV-convention cameras -> a Kinetrace calibration (via Calibration.from_krt)."""
    cams = [{"K": m.K, "R": m.R, "t": m.t, "dist": m.dist, "fisheye": m.fisheye,
             "width": m.width, "height": m.height} for m in models]
    return Calibration.from_krt(cams, unit=unit, source=source)


def _num(v: float) -> str:
    return repr(float(v))


# ------------------------------------------------------------------ Anipose calibration.toml
def write_anipose(models: Models, path) -> None:
    """aniposelib's CameraGroup layout: [cam_0] name, size = [w, h], matrix,
    distortions (OpenCV order; 4 + `fisheye = true` for a fisheye), rotation
    (Rodrigues vector), translation; world -> camera, OpenCV pixels."""
    out = []
    for i, m in enumerate(models.cameras):
        out.append(f"[cam_{i}]")
        out.append(f"name = {json.dumps(m.name)}")
        out.append(f"size = [{m.width}, {m.height}]")
        out.append("matrix = [" + ", ".join("[" + ", ".join(_num(v) for v in row) + "]" for row in m.K) + "]")
        out.append("distortions = [" + ", ".join(_num(v) for v in m.dist) + "]")
        out.append("rotation = [" + ", ".join(_num(v) for v in m.rvec) + "]")
        out.append("translation = [" + ", ".join(_num(v) for v in m.t) + "]")
        if m.fisheye:
            out.append("fisheye = true")
        out.append("")
    out += ["[metadata]", "adjusted = false", "error = 0.0",
            f"# written by Kinetrace; unit of translation: {models.unit or 'as calibrated'}", ""]
    Path(path).write_text("\n".join(out), encoding="utf-8", newline="\n")


def read_anipose(path) -> list[CamModel]:
    import cv2
    try:
        import tomllib
    except ModuleNotFoundError:          # Python 3.10 (Ubuntu 22.04): the tomli backport
        import tomli as tomllib
    try:
        d = tomllib.loads(_read_text(path))
    except (OSError, ValueError) as e:
        raise CalibFormatError(f"{Path(path).name}: not a readable TOML file ({e})") from None
    keys = sorted((k for k in d if k.startswith("cam_") and isinstance(d[k], dict)),
                  key=lambda k: int(k[4:]) if k[4:].isdigit() else 1 << 30)
    if not keys:
        raise CalibFormatError(f"{Path(path).name}: no [cam_0] ... sections - not an Anipose calibration")
    out = []
    for k in keys:
        c = d[k]
        try:
            w, h = (int(v) for v in c["size"])
            K = np.asarray(c["matrix"], np.float64).reshape(3, 3)
            dist = np.asarray(c.get("distortions", [0] * 5), np.float64).ravel()
            R = cv2.Rodrigues(np.asarray(c["rotation"], np.float64).reshape(3))[0]
            t = np.asarray(c["translation"], np.float64).reshape(3)
        except (KeyError, TypeError, ValueError) as e:
            raise CalibFormatError(f"{Path(path).name} [{k}]: needs size, matrix, rotation and translation "
                                   f"({e})") from None
        fish = bool(c.get("fisheye", False))
        out.append(CamModel(str(c.get("name", k)), w, h, K, dist[:4] if fish else np.pad(dist, (0, max(0, 5 - len(dist)))),
                            R, t, fish))
    return out


# ------------------------------------------------------------------ OpenCV YAML / JSON
def write_opencv(models: Models, path) -> None:
    """cv2.FileStorage (.yml / .yaml / .json by the file name): per camera a
    map camera_<i> with the keys of OpenCV's calibration sample
    (camera_matrix, distortion_coefficients, image_width, image_height) plus
    rvec, tvec and R (x_cam = R X + t)."""
    import cv2
    fs = _fs_new(path)
    fs.write("format", "kinetrace-cameras")
    fs.write("convention", "x_cam = R X + t; pixel centres at whole numbers from 0 (OpenCV)")
    fs.write("unit", models.unit or "")
    fs.write("n_cameras", len(models.cameras))
    for i, m in enumerate(models.cameras):
        fs.startWriteStruct(f"camera_{i}", cv2.FileNode_MAP)
        fs.write("name", m.name)
        fs.write("image_width", int(m.width))
        fs.write("image_height", int(m.height))
        fs.write("camera_matrix", np.asarray(m.K, np.float64))
        fs.write("distortion_coefficients", np.asarray(m.dist, np.float64).reshape(1, -1))
        fs.write("fisheye", int(m.fisheye))
        fs.write("rvec", m.rvec.reshape(3, 1))
        fs.write("tvec", np.asarray(m.t, np.float64).reshape(3, 1))
        fs.write("R", np.asarray(m.R, np.float64))
        fs.endWriteStruct()
    _fs_save(fs, path)


# (I222) cv2.FileStorage opens its file with the ANSI API: a path with a character outside the
# system code page ("Jose" with an accent) cannot be opened, and the failure read as "no such file".
# The files are read and written by Python and handed to OpenCV as text in memory.
_FS_FORMATS = {".yml": ".yml", ".yaml": ".yml", ".json": ".json", ".xml": ".xml"}


def _fs_new(path=None):
    """A cv2.FileStorage that writes into memory (format from `path`'s extension; YAML by default)."""
    import cv2
    ext = _FS_FORMATS.get(Path(path).suffix.lower(), ".yml") if path is not None else ".yml"
    fs = cv2.FileStorage(ext, cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    if not fs.isOpened():
        raise CalibFormatError("OpenCV cannot write this format here")
    return fs


def _fs_save(fs, path) -> None:
    """Finish a `_fs_new` storage into `path` as UTF-8."""
    p = Path(path)
    text = fs.releaseAndGetString()
    try:
        p.write_bytes(text.encode("utf-8"))
    except OSError as e:
        raise CalibFormatError(f"{p.name}: cannot be written ({e.strerror or e})") from None


def _fs_read(path):
    """An opened cv2.FileStorage on the CONTENTS of `path` (any of YAML / JSON / XML), or a
    CalibFormatError that says which: missing, unreadable, or not an OpenCV storage file."""
    import cv2
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        raise CalibFormatError(f"{p.name}: cannot be read (no such file)") from None
    except OSError as e:
        raise CalibFormatError(f"{p.name}: cannot be read ({e.strerror or e})") from None
    try:
        fs = cv2.FileStorage(data.decode("utf-8-sig", errors="replace"),
                             cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    except Exception as e:      # noqa: BLE001 - cv2.error / SystemError on text OpenCV cannot parse
        raise CalibFormatError(f"{p.name}: not an OpenCV YAML / JSON file ({str(e).splitlines()[0][:120]})") from None
    if not fs.isOpened():
        raise CalibFormatError(f"{p.name}: not an OpenCV YAML / JSON file")
    return fs


def read_opencv(path) -> list[CamModel]:
    import cv2
    fs = _fs_read(path)
    out = []
    try:
        root = fs.root()
        names = [k for k in root.keys() if k.startswith("camera_") and k[7:].isdigit()]
        nodes = [root.getNode(k) for k in sorted(names, key=lambda k: int(k[7:]))] or [root]
        for i, nd in enumerate(nodes):
            K = nd.getNode("camera_matrix").mat()
            if K is None:
                raise CalibFormatError(f"{Path(path).name}: camera {i + 1} has no camera_matrix")
            dist = nd.getNode("distortion_coefficients").mat()
            dist = np.zeros(5) if dist is None else dist.ravel().astype(np.float64)
            R = nd.getNode("R").mat()
            if R is None:
                rv = nd.getNode("rvec").mat()
                R = np.eye(3) if rv is None else cv2.Rodrigues(rv.reshape(3))[0]
            tv = nd.getNode("tvec").mat()
            t = np.zeros(3) if tv is None else tv.ravel().astype(np.float64)
            fish = bool(nd.getNode("fisheye").real()) if not nd.getNode("fisheye").empty() else False
            name = nd.getNode("name").string() or f"cam{i + 1}"
            out.append(CamModel(name, int(nd.getNode("image_width").real()), int(nd.getNode("image_height").real()),
                                np.asarray(K, np.float64), dist[:4] if fish else np.pad(dist, (0, max(0, 5 - len(dist)))),
                                np.asarray(R, np.float64), t, fish))
    finally:
        fs.release()
    return out


# ------------------------------------------------------------------ MATLAB
def write_matlab(models: Models, path) -> None:
    """A .mat with a struct array `cameras`, in MATLAB's conventions AND
    OpenCV's, so neither needs converting by hand:
    K (MATLAB, R2022b+: 1-based principal point), IntrinsicMatrix (= K',
    older releases), ImageSize [rows cols], RadialDistortion [k1 k2 k3],
    TangentialDistortion [p1 p2], RotationMatrix (= R', MATLAB's row-vector
    form) and TranslationVector (1x3); and K_opencv / R / t (x_cam = R X + t,
    0-based pixels)."""
    from scipy.io import savemat
    cams = []
    for m in models.cameras:
        K1 = m.K.copy()
        K1[0, 2] += 1.0
        K1[1, 2] += 1.0
        d = np.asarray(m.dist, np.float64)
        cams.append({"name": m.name, "K": K1, "IntrinsicMatrix": K1.T,
                     "ImageSize": np.array([m.height, m.width], np.float64),
                     "RadialDistortion": (d[[0, 1, 4]] if not m.fisheye else d[:4]).astype(np.float64),
                     "TangentialDistortion": (d[[2, 3]] if not m.fisheye else np.zeros(2)),
                     "RotationMatrix": m.R.T, "TranslationVector": m.t.reshape(1, 3),
                     "K_opencv": m.K, "R": m.R, "t": m.t.reshape(3, 1), "fisheye": float(m.fisheye)})
    rec = np.empty((1, len(cams)), dtype=[(k, object) for k in cams[0]])
    for i, c in enumerate(cams):
        for k, v in c.items():
            rec[0, i][k] = v
    savemat(str(path), {"cameras": rec, "unit": models.unit or "",
                        "convention": "K / IntrinsicMatrix / RotationMatrix / TranslationVector: MATLAB "
                                      "(1-based pixels, row vectors); K_opencv / R / t: OpenCV (0-based, "
                                      "x_cam = R X + t)"},
            do_compression=True)


def read_matlab(path) -> list[CamModel]:
    """A .mat with `cameras` as written by `write_matlab`, or MATLAB-style
    fields (K or IntrinsicMatrix, RadialDistortion, TangentialDistortion,
    RotationMatrix / TranslationVector, ImageSize). 1-based pixels are
    converted; K is told from IntrinsicMatrix (its transpose)."""
    from scipy.io import loadmat
    try:
        z = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception as e:  # noqa: BLE001 - scipy raises several kinds for a non-.mat file
        raise CalibFormatError(f"{Path(path).name}: not a readable .mat file ({e})") from None
    cams = z.get("cameras")
    if cams is None:
        raise CalibFormatError(f"{Path(path).name}: no 'cameras' variable (a struct array, one per camera)")
    cams = np.atleast_1d(cams)
    out = []
    for i, c in enumerate(cams):
        g = lambda k: getattr(c, k, None)  # noqa: E731
        if g("K_opencv") is not None and g("R") is not None:
            K = np.asarray(g("K_opencv"), np.float64)
            R = np.asarray(g("R"), np.float64)
            t = np.asarray(g("t"), np.float64).ravel()
        else:
            if g("K") is not None:
                K = np.asarray(g("K"), np.float64).copy()
            elif g("IntrinsicMatrix") is not None:
                K = np.asarray(g("IntrinsicMatrix"), np.float64).T.copy()
            else:
                raise CalibFormatError(f"{Path(path).name}: camera {i + 1} has neither K nor IntrinsicMatrix")
            K[0, 2] -= 1.0
            K[1, 2] -= 1.0
            R = np.asarray(g("RotationMatrix"), np.float64).T
            t = np.asarray(g("TranslationVector"), np.float64).ravel()
        size = np.asarray(g("ImageSize"), np.float64).ravel()
        rad = np.atleast_1d(np.asarray(g("RadialDistortion") if g("RadialDistortion") is not None else [0, 0],
                                       np.float64))
        tan = np.atleast_1d(np.asarray(g("TangentialDistortion") if g("TangentialDistortion") is not None
                                       else [0, 0], np.float64))
        fish = bool(g("fisheye")) if g("fisheye") is not None else False
        if fish:
            dist = np.pad(rad, (0, max(0, 4 - len(rad))))[:4]
        else:
            k = np.pad(rad, (0, max(0, 3 - len(rad))))
            dist = np.array([k[0], k[1], tan[0] if len(tan) else 0.0, tan[1] if len(tan) > 1 else 0.0, k[2]])
        name = str(g("name")) if g("name") is not None else f"cam{i + 1}"
        out.append(CamModel(name, int(size[1]), int(size[0]), K, dist, R, t, fish))
    return out


# ------------------------------------------------------------------ Blender
def write_blender(models: Models, path, sensor_mm: float = 36.0) -> None:
    """A Python script that makes one Blender camera per calibrated camera
    (run it in Blender's Scripting tab). Horizontal sensor fit:
    lens = fx * sensor / width; the principal point becomes the shift,
    measured from the picture's CENTRE PIXEL (w - 1) / 2; the camera looks
    down its -Z with +Y up, so its matrix is [R' | -R' t] diag(1, -1, -1).
    Skew and lens distortion have no Blender equivalent and are left out."""
    lines = ["# Kinetrace cameras for Blender - run in Scripting (Blender 2.8+)",
             f"# world unit: {models.unit or 'as calibrated'}; lens distortion is not applied", "",
             "import bpy", "from mathutils import Matrix", "", "CAMERAS = ["]
    for m in models.cameras:
        fx, fy, cx, cy = (float(v) for v in (m.K[0, 0], m.K[1, 1], m.K[0, 2], m.K[1, 2]))   # plain floats:
        # numpy 2 would write np.float64(...) into the script
        Rt = m.R.T
        C = -Rt @ m.t
        B = np.diag([1.0, -1.0, -1.0])
        rot = Rt @ B
        mw = [[*rot[r], C[r]] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
        if abs(fy / fx - 1.0) > 0.005:
            models.notes.append(f"{m.name}: fx and fy differ by {abs(fy / fx - 1) * 100:.1f} %; the Blender "
                                "camera uses fx (Blender cameras have square pixels here)")
        lines.append("    {" + f"'name': {m.name!r}, 'w': {m.width}, 'h': {m.height}, "
                     f"'lens': {fx * sensor_mm / m.width!r}, "
                     f"'shift_x': {-(cx - (m.width - 1) / 2.0) / m.width!r}, "
                     f"'shift_y': {(cy - (m.height - 1) / 2.0) / m.width!r}, "
                     f"'matrix': {[[float(v) for v in row] for row in mw]!r}" + "},")
    lines += ["]", "",
              "scene = bpy.context.scene",
              "for c in CAMERAS:",
              "    data = bpy.data.cameras.new(c['name'])",
              "    data.sensor_fit = 'HORIZONTAL'",
              f"    data.sensor_width = {sensor_mm!r}",
              "    data.lens = c['lens']",
              "    data.shift_x = c['shift_x']",
              "    data.shift_y = c['shift_y']",
              "    ob = bpy.data.objects.new(c['name'], data)",
              "    ob.matrix_world = Matrix(c['matrix'])",
              "    scene.collection.objects.link(ob)",
              "scene.render.resolution_x = CAMERAS[0]['w']",
              "scene.render.resolution_y = CAMERAS[0]['h']",
              ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ reading any of them
def read_cameras(path) -> list[CamModel]:
    """Anipose .toml, OpenCV .yml / .yaml / .json, MATLAB .mat, by extension."""
    suf = Path(path).suffix.lower()
    if suf == ".toml":
        return read_anipose(path)
    if suf in (".yml", ".yaml", ".json", ".xml"):
        return read_opencv(path)
    if suf == ".mat":
        return read_matlab(path)
    raise CalibFormatError(f"{Path(path).name}: Kinetrace reads camera files as Anipose .toml, OpenCV "
                           ".yml / .json or MATLAB .mat")


def load_calibration(path, sizes: list[tuple[int, int]] | None = None) -> Calibration:
    """Pick the importer from the file name / contents. `sizes` only labels the
    columns of a size-less dltCoefs.csv; the import dialog does NOT pass the
    videos' sizes, because a size the file never recorded must not be shown
    or compared as if it were the file's (I99)."""
    p = Path(path)
    if not p.is_file():
        raise CalibFormatError(f"{p.name}: cannot be read (no such file)")
    try:
        return _load_calibration(p, sizes)
    except CalibFormatError:
        raise
    except ValueError as e:                         # the importers' own sentences: make sure the FILE is named
        raise CalibFormatError(str(e) if p.name in str(e) else f"{p.name}: {e}") from None
    except OSError as e:
        raise CalibFormatError(f"{p.name}: cannot be read ({e.strerror or e})") from None
    except Exception as e:    # noqa: BLE001 - (I227) a KeyError / IndexError / cv2.error from the LAST loader
        raise CalibFormatError(f"{p.name}: not a calibration file Kinetrace can read "
                               f"({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else 'no detail'})"
                               ) from None


def _load_calibration(p: Path, sizes) -> Calibration:
    low = p.name.lower()

    def cameras_file():
        return from_models(read_cameras(p), source=p.name)
    if low.endswith((".toml", ".yml", ".yaml", ".xml")):          # (I255) .xml = OpenCV's cameras file
        return cameras_file()                       # Anipose / OpenCV FileStorage
    if low.endswith(".json"):
        try:
            return load_kcal(p)
        except (ValueError, KeyError, TypeError):
            pass
        if "opencv-matrix" in p.read_text(encoding="utf-8", errors="replace")[:20000]:
            return cameras_file()                   # an OpenCV FileStorage JSON
        return Calibration.load_krt(p)              # an OpenCV-style K + R/t JSON
    if low.endswith(".txt"):
        return Calibration.load_krt(p)
    if low.endswith(".mat"):
        from scipy.io import whosmat
        try:
            if "cameras" in {w[0] for w in whosmat(str(p))}:
                return cameras_file()               # MATLAB cameras (K / IntrinsicMatrix ...)
        except Exception:      # noqa: BLE001 - a v7.3 / odd file: the importers below say what is wrong
            pass
        if "easywand" in low:
            return Calibration.load_easywand_mat(p)
        if "dvproject" in low:
            return Calibration.load_dltdv_project(p)
        try:
            return Calibration.load_easywand_mat(p)
        except Exception:      # noqa: BLE001
            return Calibration.load_dltdv_project(p)
    return Calibration.load_dlt_csv(p, sizes)

# ------------------------------------------------------------------ Kinetrace's own files
KCAL_VERSION = 1


def calibration_to_kcal(cal, report: dict | None = None, grav: dict | None = None) -> dict:
    """The Kinetrace calibration file: every camera's DLT coefficients WITH the
    pixel convention and undistortion they were fitted in, the unit, and the
    wand report when there is one. Self-describing, unlike a bare dltCoefs.csv."""
    out = {"kinetrace_calibration": KCAL_VERSION, "unit": cal.unit, "source": cal.source,
           "cameras": [{"width": int(c.width), "height": int(c.height),
                        "pixel_origin": float(c.pixel_origin), "y_flip": bool(c.y_flip),
                        "rmse": (None if not np.isfinite(c.rmse) else float(c.rmse)),
                        "coefs": [float(v) for v in c.coefs],
                        "undistort": c.undistort.to_json()} for c in cal.cameras],
           "report": report or {}, "gravity": grav or {}}
    if getattr(cal, "origin_shift", None) is not None:          # (I229) a K + R/t rig's moved origin
        out["origin_shift"] = [float(v) for v in np.asarray(cal.origin_shift, np.float64).reshape(3)]
    return out


def load_kcal(path):
    """`*.kcal.json` -> `calib.Calibration` (cameras in file order)."""
    from kinetrace.calib import undistort_from_json
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "kinetrace_calibration" not in d:
        raise ValueError(f"{Path(path).name}: not a Kinetrace calibration file")
    cams = []
    for c in d.get("cameras", []):
        rmse = c.get("rmse")
        cams.append(CameraCalibration(np.asarray(c["coefs"], np.float64).reshape(11),
                                      int(c.get("width", 0)), int(c.get("height", 0)),
                                      undistort_from_json(c.get("undistort") or {}),
                                      float(c.get("pixel_origin", 0.0)), bool(c.get("y_flip", False)),
                                      float("nan") if rmse is None else float(rmse)))
    cal = Calibration(cams, str(d.get("unit", "")), str(d.get("source", "")) or Path(path).name)
    cal.report = d.get("report") or {}
    if d.get("origin_shift") is not None:                       # (I229)
        cal.origin_shift = np.asarray(d["origin_shift"], np.float64).reshape(3)
    return cal


def dlt_csv_matlab(cal, path) -> None:
    """dltCoefs.csv for DLTdv / easyWand users: 11 rows, one column per camera,
    converted to MATLAB's 1-based pixels when the calibration was fitted on
    0-based ones (u' = u + 1: L1..L3 += L9..L11, L4 += 1; same for v)."""
    cols = [c.coefs_for_origin(1.0) for c in cal.cameras]       # (R14) the one pixel-convention shift
    np.savetxt(str(path), np.stack(cols, axis=1), delimiter=",", fmt="%.12g")

# ------------------------------------------------------------------ lens profiles
def write_lens(profile, path) -> None:
    """One lens (lens.LensProfile): OpenCV .yml / .json (camera_matrix,
    distortion_coefficients, image_width / height, fisheye), or an Argus
    profile line (.txt: `1 f w h cx cy AR k1 k2 t1 t2 k3`, OpenCV pixels)."""
    import cv2
    suf = Path(path).suffix.lower()
    if suf == ".txt":
        if profile.fisheye:
            raise CalibFormatError("an Argus profile holds the standard (pinhole) model, not a fisheye one: "
                                   "save this lens as .yml or .json")
        K, d = profile.K, np.pad(np.asarray(profile.dist, np.float64), (0, 5))[:5]
        vals = [1, K[0, 0], profile.width, profile.height, K[0, 2], K[1, 2], K[1, 1] / K[0, 0], *d]
        Path(path).write_text(" ".join(_num(v) if isinstance(v, float) else str(v) for v in vals) + "\n",
                              encoding="utf-8", newline="\n")
        return
    fs = _fs_new(path)
    fs.write("image_width", int(profile.width))
    fs.write("image_height", int(profile.height))
    fs.write("camera_matrix", np.asarray(profile.K, np.float64))
    fs.write("distortion_coefficients", np.asarray(profile.dist, np.float64).reshape(1, -1))
    fs.write("fisheye", int(profile.fisheye))
    if np.isfinite(profile.rms):
        fs.write("rms", float(profile.rms))
    _fs_save(fs, path)


def read_lens(path, view: int = 0):
    """A lens profile: Kinetrace .klens.json, OpenCV .yml / .json, or an Argus
    profile (.txt; the line for camera `view`)."""
    import cv2
    from kinetrace import lens
    name = Path(path).name
    if not Path(path).is_file():
        raise CalibFormatError(f"{name}: cannot be read (no such file)")
    if name.lower().endswith(".json"):
        try:
            return lens.LensProfile.load(path)          # Kinetrace's own (.klens.json)
        except (ValueError, KeyError, TypeError):
            pass                                        # else an OpenCV FileStorage JSON
    if Path(path).suffix.lower() == ".txt":
        prof, why = lens.argus_profile_for(path, view)
        if prof is None:
            raise CalibFormatError(why)
        return prof
    try:
        fs = _fs_read(path)                          # (I222) by content: non-ASCII paths, a missing file said as missing
    except CalibFormatError as e:
        if "no such file" in str(e) or "cannot be read" in str(e):
            raise
        raise CalibFormatError(f"{name}: not a lens file Kinetrace reads (.klens.json, OpenCV .yml / .json, "
                               "Argus .txt)") from None
    try:
        K = fs.getNode("camera_matrix").mat()
        if K is None:
            raise CalibFormatError(f"{name}: no camera_matrix")
        d = fs.getNode("distortion_coefficients").mat()
        fish = not fs.getNode("fisheye").empty() and bool(fs.getNode("fisheye").real())
        return lens.LensProfile(int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()),
                                np.asarray(K, np.float64),
                                np.zeros(4 if fish else 5) if d is None else d.ravel().astype(np.float64), fish,
                                float(fs.getNode("rms").real()) if not fs.getNode("rms").empty() else float("nan"),
                                0, f"OpenCV lens file {name}")
    finally:
        fs.release()


# ------------------------------------------------------------------ camera offsets
OFFSET_COLS = ("camera", "name", "offset", "rate")


def write_offsets(project, path) -> None:
    """camera,name,offset,rate: offset = the frame this camera shows when the
    reference camera is at its frame 0; rate = its frame rate / the reference's."""
    rows = [[i + 1, project.name(i), _num(project.offsets[i]), _num(project.rates[i])]
            for i in range(project.n_views)]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(OFFSET_COLS)
    w.writerows(rows)
    Path(path).write_text(buf.getvalue(), encoding="utf-8", newline="")


class OffsetRows(list):
    """The rows `read_offsets` returns -- a plain list of (view, offset, rate) the
    app can apply directly -- plus what it did: `rebased` (the file's row for
    the project's first camera was used to re-base the others) and
    `reference_missing` (the file has no row for the project's first camera, so
    the offsets could not be re-based and are the file's own numbers: the app
    asks before applying them)."""
    rebased: bool = False
    reference_missing: bool = False


def read_offsets(path, project) -> OffsetRows:
    """-> [(view, offset, rate)] for the project's cameras, matched by name,
    else by the camera number.

    (I169) The numbers in a file are measured against ITS reference camera, but
    the project's reference is its first camera, whose offset is always 0 and
    whose rate 1 (`Project._normalize`). So the file is re-based on the file's
    row for the project's camera 0: rate_i' = rate_i / rate_0 and
    offset_i' = offset_i - rate_i' * offset_0 -- a clap table 'camA 120, camB
    97, camC 130' gives camB -23 and camC +10 frames against camA, and an offsets
    file from a project with another reference camera lands where it belongs.
    The first camera's own row comes back as (0, 0.0, 1.0). Without a row for
    the first camera the file's numbers are returned unchanged and
    `reference_missing` is set."""
    from kinetrace.projectfile import ProjectFileError, parse_table
    try:
        cols, n = parse_table(_read_text(path), Path(path).name, ("offset",),
                              ("camera", "name", "rate"))
    except ProjectFileError as e:
        raise CalibFormatError(str(e)) from None
    out = OffsetRows()
    by_name = {project.name(i): i for i in range(project.n_views)}
    for k in range(n):
        v = None
        if cols.get("name") is not None and cols["name"][k] in by_name:
            v = by_name[cols["name"][k]]
        elif cols.get("camera") is not None and cols["camera"][k].strip().isdigit():
            v = int(cols["camera"][k]) - 1
        if v is None or not 0 <= v < project.n_views:
            continue
        try:
            off = float(cols["offset"][k])
            rate = float(cols["rate"][k]) if cols.get("rate") is not None and cols["rate"][k].strip() else None
        except ValueError:
            raise CalibFormatError(f"{Path(path).name}: row {k + 2}: offset / rate must be numbers") from None
        rate = project.rates[v] if rate is None else rate
        if not (np.isfinite(off) and np.isfinite(rate) and rate > 0):
            raise CalibFormatError(f"{Path(path).name}: row {k + 2}: the offset must be a number and the rate "
                                   "a number above 0")
        out.append((v, off, rate))
    if not out:
        raise CalibFormatError(f"{Path(path).name}: no row names a camera of this project")
    ref = next(((o, r) for v, o, r in out if v == 0), None)
    if ref is None:
        out.reference_missing = True
        return out
    o0, r0 = ref
    out[:] = [(0, 0.0, 1.0) if v == 0 else (v, o - (r / r0) * o0, r / r0) for v, o, r in out]
    out.rebased = abs(o0) > 1e-12 or abs(r0 - 1.0) > 1e-12
    return out


# ------------------------------------------------------------------ 3D points
def write_points3d(rec: Reconstruction, path, kind: str = "kinetrace", models: Models | None = None) -> int:
    """kind: "kinetrace" (frame + name_X/_Y/_Z, the xyz export), "dltdv"
    (xyzpts: pt{i}_X/_Y/_Z, row k = reference frame k, NaN = none) or
    "anipose" (name_x/_y/_z/_error/_ncams/_score + fnum). With `models`, the
    points are put in those exported cameras' world (mirror / origin): ALWAYS
    pass the models of the cameras exported with these points (I170), or the
    points reproject 30-100 px off through them. A DLTdv xyzpts file starts at
    reference frame 0, so a result that starts BEFORE 0 (other cameras running
    before the reference started) is written from frame 0 and the instants
    before it are left out; the return value is how many instants were (0
    for every other case, I215)."""
    left_out = 0
    xyz = rec.xyz.astype(np.float64)
    if models is not None:
        xyz = models.world_to_export(xyz.reshape(-1, 3)).reshape(xyz.shape)
    if kind == "kinetrace":
        r2 = Reconstruction(rec.t0, rec.names, xyz, rec.residual, rec.n_cams, rec.unit, rec.per_cam)
        r2.export_csv(path)
        return 0
    from kinetrace.session import _sanitize
    names = [_sanitize(n) for n in rec.names]
    T, N = xyz.shape[:2]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if kind == "dltdv":
        w.writerow([f"pt{j + 1}_{a}" for j in range(N) for a in "XYZ"])
        skip = max(0, -int(rec.t0))                       # (I215) instants before reference frame 0
        left_out = min(skip, T)
        xyz = xyz[skip:]
        t0 = max(int(rec.t0), 0)
        full = np.full((t0 + len(xyz), N, 3), np.nan)
        full[t0:] = xyz
        for row in full:
            w.writerow(["NaN" if not np.isfinite(v) else f"{v:.6f}" for v in row.ravel()])
        side = Path(path).with_name(Path(path).stem + "_pointnames.csv")
        side.write_text("index,name\n" + "".join(f"pt{j + 1},{nm}\n" for j, nm in enumerate(names)),
                        encoding="utf-8", newline="")
    elif kind == "anipose":
        w.writerow([f"{nm}_{a}" for nm in names for a in ("x", "y", "z", "error", "ncams", "score")]
                   + [f"M_{i}{j}" for i in range(3) for j in range(3)] + ["center_0", "center_1", "center_2",
                                                                          "fnum"])
        eye = [1.0 if i == j else 0.0 for i in range(3) for j in range(3)]
        for k in range(T):
            cells = []
            for j in range(N):
                v, ok = xyz[k, j], np.isfinite(xyz[k, j]).all()
                cells += ([f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}",
                           f"{rec.residual[k, j]:.4f}" if np.isfinite(rec.residual[k, j]) else "",
                           str(int(rec.n_cams[k, j])), "1.0"] if ok else ["", "", "", "", "0", ""])
            w.writerow(cells + eye + [0.0, 0.0, 0.0, rec.t0 + k])
    else:
        raise CalibFormatError(f"unknown 3D points format {kind!r}")
    Path(path).write_text(buf.getvalue(), encoding="utf-8", newline="")
    return left_out


def read_points3d(path, unit: str = "") -> tuple[Reconstruction, list[str]]:
    """Kinetrace xyz CSV, DLTdv xyzpts or Anipose points_3d CSV (by header) ->
    (Reconstruction, notes). Frames are the file's (reference frames)."""
    text = _read_text(path)
    rows = list(csv.reader(io.StringIO(text)))
    blank = lambda r: not any(c.strip() for c in r)  # noqa: E731
    while rows and blank(rows[0]):                    # blank lines before the header
        rows.pop(0)
    while rows and blank(rows[-1]):                   # (I173) only TRAILING blank rows are dropped
        rows.pop()
    if not rows:
        raise CalibFormatError(f"{Path(path).name}: empty")
    head = [h.strip() for h in rows[0]]
    data = rows[1:]
    notes: list[str] = []
    positional = bool(head) and all(re.fullmatch(r"pt\d+_[XYZ]", h) for h in head)    # DLTdv xyzpts: row k = frame k
    if positional:
        # a frame the writer left as empty cells (pandas' NaN) is still a frame:
        # keep it as a row of NaN so every later frame stays on its own row
        data = [r if not blank(r) else [""] * len(head) for r in data]
    else:
        data = [r for r in data if not blank(r)]      # these formats name their frame in a column

    def col(k):
        try:
            return np.array([float(r[k]) if r[k].strip() else np.nan for r in data], np.float64)
        except (ValueError, IndexError):
            raise CalibFormatError(f"{Path(path).name}: column {head[k]!r} is not all numbers") from None
    if head and head[0].lower() == "frame":                       # Kinetrace
        names = [h[:-2] for h in head[1:] if h.endswith("_X")]
        frames = col(0).astype(np.int64)
        idx = {h: k for k, h in enumerate(head)}
        get = lambda nm, a: col(idx[f"{nm}_{a}"])  # noqa: E731
        kind = "Kinetrace"
    elif positional:                                            # DLTdv xyzpts
        n = max(int(re.match(r"pt(\d+)", h).group(1)) for h in head)
        side = Path(path).with_name(Path(path).stem + "_pointnames.csv")
        names = [f"pt{j + 1}" for j in range(n)]
        if side.is_file():
            for r in csv.reader(io.StringIO(side.read_text(encoding="utf-8-sig"))):
                if len(r) > 1 and re.fullmatch(r"pt\d+", r[0].strip()) and int(r[0].strip()[2:]) <= n:
                    names[int(r[0].strip()[2:]) - 1] = r[1].strip()
        frames = np.arange(len(data))
        idx = {h: k for k, h in enumerate(head)}
        get = lambda nm, a: col(idx[f"pt{names.index(nm) + 1}_{a}"])  # noqa: E731
        kind = "DLTdv xyzpts"
    elif any(h.endswith("_error") for h in head):                 # Anipose
        names = [h[:-2] for h in head if h.endswith("_x") and f"{h[:-2]}_error" in head]
        idx = {h: k for k, h in enumerate(head)}
        frames = col(idx["fnum"]).astype(np.int64) if "fnum" in idx else np.arange(len(data))
        get = lambda nm, a: col(idx[f"{nm}_{a.lower()}"])  # noqa: E731
        kind = "Anipose"
        if "M_00" in idx:
            M = np.array([[col(idx[f"M_{i}{j}"])[0] for j in range(3)] for i in range(3)])
            if not np.allclose(M, np.eye(3), atol=1e-9):
                notes.append("Anipose aligned these points to its own axes (the M_ / center_ columns); they "
                             "are imported as written, in those axes")
    else:
        raise CalibFormatError(f"{Path(path).name}: not a 3D points file Kinetrace reads (Kinetrace xyz CSV, "
                               "DLTdv xyzpts, Anipose points_3d CSV)")
    if not len(frames):
        raise CalibFormatError(f"{Path(path).name}: no rows")
    t0, t1 = int(frames.min()), int(frames.max())
    T, N = t1 - t0 + 1, len(names)
    xyz = np.full((T, N, 3), np.nan)
    res = np.full((T, N), np.nan)
    ncams = np.zeros((T, N), np.int32)           # not known unless the file says
    for j, nm in enumerate(names):
        for a, ax in enumerate("XYZ"):
            xyz[frames - t0, j, a] = get(nm, ax)
        if kind == "Anipose":
            res[frames - t0, j] = col(idx[f"{nm}_error"])
            if f"{nm}_ncams" in idx:
                ncams[frames - t0, j] = np.nan_to_num(col(idx[f"{nm}_ncams"])).astype(np.int32)
    rec = Reconstruction(t0, names, xyz, res, ncams, unit)
    notes.insert(0, f"{kind}: {N} landmark(s), frames {t0} - {t1}")
    return rec, notes
