"""A synthetic DLTdv8 project with KNOWN answers, for verify_dltdv_project.py (I310, G211).

Three cameras film two bright dots ("head", "tail") moving fast in front of a still, textured background:
camera 1 (the clock) and camera 2 at 100 fps, camera 3 at 200 fps (DLTdv's frame-rate ratio 2). DLTdv's offsets
are [0, -4, 3]; by DLTdv's rule (1-based: video frame = ratio x row + offset) row r is frame r, r - 4 and 2r + 4
(0-based) of the three videos. Camera 2's VIDEO is drawn 9 frames later than the project says (`PLANTED`): what a
newer MATLAB's frame numbering does to an old project, so the frame check must offer +9 there and nothing elsewhere.

The project is written twice, with the same content:
* MATLAB v7 (scipy `savemat`): no lens profile (scipy cannot write MATLAB objects);
* MATLAB v7.3 (HDF5, h5py), in the layout MATLAB itself writes: a 512-byte MAT header in the userblock,
  `MATLAB_class` attributes, cells as object references into `#refs#`, sparse matrices as CSC groups (data / ir /
  jc + `MATLAB_sparse`), and camera 1's lens as an LWM object in the `#subsystem#/MCOS` store (a near-identity
  profile, so the 3D stays exact).

Clicks are the true projections (DLTdv pixels: first pixel 1, y down); DLTdv's stored 3D is the truth. Every
number here is made up. Deterministic; everything goes into the folder given.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from kinetrace.calib import CameraCalibration, NoUndistort, dlt_from_camera

W, H = 640, 360
FPS = [100.0, 100.0, 200.0]
RATIO = [1, 1, 2]
DV_OFFSETS = [0, -4, 3]                 # DLTdv's video.offset (frames)
PLANTED = [0, 9, 0]                     # where each video REALLY is, relative to what the project says
N_REF = 140                             # camera 1's frames
ROWS = np.arange(20, 120)               # rows (clock frames) with clicks
N_ROWS = 400                            # the project's row count (DLTdv: the clock video's length or more)
POINTS = ["head", "tail"]
VIDEOS = ["cam1_side.mp4", "cam2_top.mp4", "cam3_fast.mp4"]
STORED = ["C:\\Lab\\rig\\cam1_side.mp4", "/home/someone/rig/cam2_top.mp4", "C:/Lab/rig/cam3_fast.mp4"]  # another machine
DOT_R = 7


def local_frame(c: int, r, shift: int = 0):
    """DLTdv's rule, 0-based: camera c's frame of 0-based row r."""
    return RATIO[c] * (np.asarray(r) + 1) + DV_OFFSETS[c] - 1 + shift


def cameras() -> list[CameraCalibration]:
    """Three 0-based cameras around a 1 m volume (the truth)."""
    out = []
    for k, (a, z) in enumerate([(0.2, 1.1), (1.5, 1.6), (2.7, 0.9)]):
        pos = np.array([3.0 * np.cos(a), 3.0 * np.sin(a), z])
        zc = np.array([0.0, 0.0, 1.0]) - pos
        zc /= np.linalg.norm(zc)
        x = np.cross(zc, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(zc, x), zc])
        K = np.array([[600.0 + 20 * k, 0, (W - 1) / 2], [0, 600.0 + 20 * k, (H - 1) / 2], [0, 0, 1]])
        out.append(CameraCalibration(dlt_from_camera(K, R, -R @ pos), W, H, NoUndistort(), pixel_origin=0.0))
    return out


def body(t) -> np.ndarray:
    """(len(t), 2, 3): head and tail at (fractional) reference frame t (camera 1's frames)."""
    t = np.atleast_1d(np.asarray(t, np.float64))
    c = np.stack([0.5 * np.sin(0.06 * t), 0.35 * np.cos(0.045 * t), 1.0 + 0.15 * np.sin(0.08 * t)], -1)
    d = np.stack([np.cos(0.03 * t), np.sin(0.03 * t), 0.2 * np.ones_like(t)], -1) * 0.12
    return np.stack([c, c - d], 1)


def _background(c: int) -> np.ndarray:
    rng = np.random.default_rng(100 + c)
    small = rng.integers(40, 140, (H // 8, W // 8, 3)).astype(np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def n_frames(c: int) -> int:
    return int(local_frame(c, N_REF - 1, PLANTED[c])) + 12


def build_videos(folder) -> list[str]:
    """Camera c's frame F shows the body at reference time t with F = local_frame(c, t, PLANTED[c])."""
    cams = cameras()
    paths = []
    for c in range(3):
        p = os.path.join(folder, VIDEOS[c])
        paths.append(p)
        if os.path.exists(p):
            cap = cv2.VideoCapture(p)
            ok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n_frames(c)
            cap.release()
            if ok:
                continue
        bg = _background(c)
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS[c], (W, H))
        for F in range(n_frames(c)):
            img = bg.copy()
            t = (F - DV_OFFSETS[c] + 1 - PLANTED[c]) / RATIO[c] - 1     # inverse of local_frame
            for k, xyz in enumerate(body(t)[0]):
                u, v = cams[c].project(xyz[None])[0]
                cv2.circle(img, (int(np.floor(u + 0.5)), int(np.floor(v + 0.5))), DOT_R,
                           (250, 250, 250) if k == 0 else (60, 230, 250), -1, cv2.LINE_AA)
            vw.write(img)
        vw.release()
    return paths


def clicks() -> np.ndarray:
    """(N_ROWS, P, V, 2) in DLTdv pixels (1-based), NaN = not clicked."""
    cams = cameras()
    xy = np.full((N_ROWS, len(POINTS), 3, 2), np.nan)
    truth = body(ROWS)
    for c in range(3):
        uv = cams[c].project(truth.reshape(-1, 3)).reshape(len(ROWS), len(POINTS), 2) + 1.0
        xy[ROWS, :, c] = uv
    xy[ROWS[ROWS % 7 == 0], :, 2] = np.nan            # camera 3 skipped some rows
    xy[ROWS[ROWS % 5 == 0], 1, 1] = np.nan            # camera 2 lost the tail now and then
    return xy


def content(sizes: dict | None = None, names: list | None = None, files: list | None = None,
            drop: tuple = ()) -> dict:
    """Everything the project holds, in MATLAB's shapes (before the file layout). `names` / `files` replace the
    point names / stored video paths; `drop` = (camera, point) pairs with no clicks at all."""
    cams = cameras()
    xy = clicks()
    for c, k in drop:
        xy[:, k, c] = np.nan
    R, P, V = N_ROWS, len(POINTS), 3
    xyz = np.full((R, P, 3), np.nan)
    res = np.full((R, P), np.nan)
    n_seen = np.isfinite(xy[..., 0]).sum(axis=2)
    both = n_seen >= 2
    truth = np.full((R, P, 3), np.nan)
    truth[ROWS] = body(ROWS)
    xyz[both] = truth[both]
    res[both] = 0.25
    offs = np.zeros((R, V))
    clicked = np.isfinite(xy[..., 0]).any(axis=1)                       # (R, V)
    offs[clicked] = np.broadcast_to(np.array(DV_OFFSETS, float), (R, V))[clicked]
    hw = [(H, W)] * V
    for c, (w, h) in (sizes or {}).items():
        hw[c] = (h, w)
    return {"xypts": xy.reshape(R, P * V * 2),           # point-major, then video, then (x, y)
            "dltpts": xyz.reshape(R, P * 3), "dltres": res, "offset": offs,
            "dltcoef": np.stack([cam.coefs_for_origin(1.0) for cam in cams], 1),       # (11, V), 1-based
            "movsizes": np.array(hw, float), "nvid": V, "numpts": P,
            "movFrameRates": np.array(FPS), "movFrameRateRatios": np.array(RATIO, float),
            "pointNames": list(names or POINTS), "files": list(files or STORED),
            "video_offset": [str(o) for o in DV_OFFSETS],
            "undistort": ["Undistortion file: lens_a.mat", "Undistortion file: none", "Undistortion file: none"]}


def lens_points() -> tuple[np.ndarray, np.ndarray]:
    """Camera 1's LWM control points (DLTdv pixels): a grid, 'undistorted' by a tiny fixed bend (< 0.05 px), so
    the lens is real enough to be applied yet changes no verdict."""
    gx, gy = np.meshgrid(np.linspace(1, W, 16), np.linspace(1, H, 10))
    uv = np.stack([gx.ravel(), gy.ravel()], 1)
    xy = uv + 0.04 * np.stack([np.sin(uv[:, 1] / 90.0), np.cos(uv[:, 0] / 120.0)], 1)
    return uv, xy


# ---------------------------------------------------------------- DLTdv's own undistortions (I313)
# DLTdv undistorts every click with the camera's `camud`, NOT with the LWM it stores (`camd`, undistorted -> raw,
# for drawing only). The two camud functions in use, written here from their MATLAB sources (independently of
# kinetrace.calib): DLTdv8a_internal/create_Tforms.m `undistort_Tform` (a camera profile .txt: Bouguet, 3 fixed-
# point iterations) and the GoPro fisheye `fisheyeUndistort_v2` (Scaramuzza's OCamCalib cam2world, then a pinhole
# at focal width/4 about the picture centre). Every constant below is made up.

BROWN = dict(f=500.0, c=(320.5, 180.5), k=(-0.2, 0.05, 3e-4, -2e-4, 0.0))
OCAM = dict(ss=(-300.0, 0.0, 8e-4, 1e-7, 5e-10), xc=182.3, yc=318.9, c=1.002, d=-0.001, e=0.0012)


def brown3(uv) -> np.ndarray:
    """create_Tforms.m undistort_Tform with T.tdata = [f f c k 3], line for line (1-based pixels)."""
    f, c, k = BROWN["f"], np.array(BROWN["c"]), BROWN["k"]
    uvn = (np.asarray(uv, np.float64) - c) / f
    uvnd = uvn.copy()
    for _ in range(3):
        r2 = (uvnd ** 2).sum(1)
        rad = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3
        tx = 2 * k[2] * uvnd[:, 0] * uvnd[:, 1] + k[3] * (r2 + 2 * uvnd[:, 0] ** 2)
        ty = k[2] * (r2 + 2 * uvnd[:, 1] ** 2) + 2 * k[3] * uvnd[:, 0] * uvnd[:, 1]
        uvnd = (uvn - np.column_stack([tx, ty])) / rad[:, None]
    return uvnd * f + c


def ocam(uv) -> np.ndarray:
    """fisheyeUndistort_v2's arithmetic, line for line (1-based pixels; OCamCalib works in (row, col))."""
    p = np.asarray(uv, np.float64)
    m = np.vstack([p[:, 1], p[:, 0]])
    m = np.linalg.inv(np.array([[OCAM["c"], OCAM["d"]], [OCAM["e"], 1.0]])) @ (m - np.array([[OCAM["xc"]], [OCAM["yc"]]]))
    M = np.vstack([m[0], m[1], np.polyval(np.array(OCAM["ss"])[::-1], np.sqrt(m[0] ** 2 + m[1] ** 2))])
    M = M / np.sqrt((M ** 2).sum(0))
    M = M / M[2] * (-W / 4)
    return np.column_stack([M[1] + W / 2, M[0] + H / 2])


LENS_MODELS = {1: brown3, 2: ocam}               # camera -> its camud in the lens project (camera 1: none)
LENS_FUNCS = {1: "undistort_Tform", 2: "fisheyeUndistort_v2"}


def dltdv_control_points(fn, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """camd's control points the way create_Tforms makes them: a 10-pixel grid, ~1,000 random picks, camud run on
    them (MATLAB's ceil(n * rand) then unique)."""
    mx, my = np.meshgrid(np.arange(1, W + 1, 10), np.arange(1, H + 1, 10))
    cp = np.column_stack([mx.T.ravel(), my.T.ravel()]).astype(np.float64)
    pick = np.unique(np.ceil(len(cp) * np.random.default_rng(seed).random(1000)).astype(int)) - 1
    return cp[pick], fn(cp[pick])


def _raw_for(fn, target) -> np.ndarray:
    """The raw pixel whose camud is `target` (Newton, numerical Jacobian), rounded to 1/256 px so a float32 track
    keeps it exactly."""
    r = np.asarray(target, np.float64).copy()
    h = 1e-4
    for _ in range(40):
        e = fn(r) - target
        jx = (fn(r + [h, 0]) - fn(r - [h, 0])) / (2 * h)
        jy = (fn(r + [0, h]) - fn(r - [0, h])) / (2 * h)
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        r -= np.column_stack([(jy[:, 1] * e[:, 0] - jy[:, 0] * e[:, 1]) / det,
                              (-jx[:, 1] * e[:, 0] + jx[:, 0] * e[:, 1]) / det])
    assert np.abs(fn(r) - target).max() < 1e-8
    return np.round(r * 256.0) / 256.0


def lens_content() -> dict:
    """The lens project: cameras 2 and 3 carry DLTdv's two camud models; their clicks are what was filmed (raw);
    DLTdv's stored 3D is DLTdv's rule (camud, then the DLT), not the truth."""
    from kinetrace.calib import triangulate_batch
    k = content()
    R, P, V = N_ROWS, len(POINTS), 3
    xy = k["xypts"].reshape(R, P, V, 2).copy()
    xy = np.round(xy * 256.0) / 256.0                  # every click float32-exact (a track is float32)
    for c, fn in LENS_MODELS.items():
        ok = np.isfinite(xy[:, :, c, 0])
        xy[ok, c] = _raw_for(fn, xy[ok, c])
    und = xy.copy()
    for c, fn in LENS_MODELS.items():
        ok = np.isfinite(xy[:, :, c, 0])
        und[ok, c] = fn(xy[ok, c])
    xyz = triangulate_batch(k["dltcoef"].T, und.reshape(R * P, V, 2))[0].reshape(R, P, 3)
    both = np.isfinite(xy[..., 0]).sum(axis=2) >= 2
    xyz[~both] = np.nan
    k.update(xypts=xy.reshape(R, P * V * 2), dltpts=xyz.reshape(R, P * 3),
             undistort=["Undistortion file: none", "Undistortion file: lens_b.mat", "Undistortion file: lens_c.txt"])
    return k


def _sparse(dense: np.ndarray):
    """MATLAB sparse: only the finite, non-zero cells are stored (DLTdv's 'not clicked' is an empty cell)."""
    from scipy.sparse import csc_matrix
    r, c = np.nonzero(np.isfinite(dense) & (dense != 0))
    return csc_matrix((dense[r, c], (r, c)), shape=dense.shape)


def write_v7(path, sizes: dict | None = None, **kw) -> str:
    from scipy.io import savemat
    k = content(sizes, **kw)
    obj = lambda xs: np.array(xs, dtype=object)          # noqa: E731 - a MATLAB cell array
    savemat(str(path), {"udExport": {
        "DLTdv_version": 8.5, "saveTime": "2026-01-01 12:00:00",
        "data": {"xypts": _sparse(k["xypts"]), "dltpts": _sparse(k["dltpts"]), "dltres": _sparse(k["dltres"]),
                 "offset": _sparse(k["offset"]), "dltcoef": k["dltcoef"], "movsizes": k["movsizes"],
                 "nvid": k["nvid"], "numpts": k["numpts"], "movFrameRates": k["movFrameRates"],
                 "movFrameRateRatios": k["movFrameRateRatios"], "pointNames": obj(k["pointNames"])},
        "video": {"files": obj(k["files"]), "offset": obj(k["video_offset"]),
                  "undistortString": obj(k["undistort"])}}}, oned_as="column")
    return str(path)


# ---------------------------------------------------------------- MATLAB v7.3 (HDF5) writer

class _V73:
    """The few MATLAB-on-HDF5 forms a DLTdv8 project uses. MATLAB is column-major, so a MATLAB (m, n) array is
    an HDF5 (n, m) dataset."""

    def __init__(self, f):
        import h5py
        self.h5py, self.f, self.n = h5py, f, 0
        self.refs = f.create_group("#refs#")

    def _name(self) -> str:
        self.n += 1
        return f"r{self.n}"

    def double(self, g, name, a):
        ds = g.create_dataset(name, data=np.atleast_2d(np.asarray(a, np.float64)).T)
        ds.attrs["MATLAB_class"] = np.bytes_(b"double")
        return ds

    def char(self, g, name, s: str):
        ds = g.create_dataset(name, data=np.array([[ord(ch)] for ch in s], np.uint16))     # MATLAB (1, len)
        ds.attrs["MATLAB_class"] = np.bytes_(b"char")
        ds.attrs["MATLAB_int_decode"] = np.int32(2)
        return ds

    def sparse(self, g, name, dense):
        m = _sparse(np.asarray(dense, np.float64))
        sg = g.create_group(name)
        sg.attrs["MATLAB_class"] = np.bytes_(b"double")
        sg.attrs["MATLAB_sparse"] = np.uint64(dense.shape[0])
        sg.create_dataset("data", data=m.data.astype(np.float64))
        sg.create_dataset("ir", data=m.indices.astype(np.uint64))
        sg.create_dataset("jc", data=m.indptr.astype(np.uint64))

    def cell(self, g, name, items):
        """items: callables that write one element into #refs# and return the dataset / group."""
        refs = [make(self.refs, self._name()).ref for make in items]
        ds = g.create_dataset(name, data=np.array(refs, dtype=self.h5py.ref_dtype).reshape(len(refs), 1))
        ds.attrs["MATLAB_class"] = np.bytes_(b"cell")
        return ds

    def struct(self, g, name):
        sg = g.create_group(name)
        sg.attrs["MATLAB_class"] = np.bytes_(b"struct")
        return sg


def write_v73(path, sizes: dict | None = None, lens: bool = True, models: bool = False) -> str:
    """`lens`: camera 1's near-identity LWM (no camud model fits it). `models`: the lens project instead
    (`lens_content`: cameras 2 and 3 with DLTdv's two camud models, their LWMs made from them, N 15)."""
    import h5py
    k = lens_content() if models else content(sizes)
    if models:
        lwms = {c: (dltdv_control_points(fn, 7 + c), 15.0) for c, fn in LENS_MODELS.items()}
    elif lens:
        lwms = {0: (lens_points(), 12.0)}
    else:
        lwms = {}
    path = str(path)
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w", userblock_size=512, libver="earliest") as f:
        w = _V73(f)
        ud = w.struct(f, "udExport")
        w.double(ud, "DLTdv_version", 8.5)
        w.char(ud, "saveTime", "2026-01-01 12:00:00")
        d = w.struct(ud, "data")
        for nm in ("xypts", "dltpts", "dltres", "offset"):
            w.sparse(d, nm, k[nm])
        w.double(d, "dltcoef", k["dltcoef"])
        w.double(d, "movsizes", k["movsizes"])
        w.double(d, "nvid", k["nvid"])
        w.double(d, "numpts", k["numpts"])
        w.double(d, "movFrameRates", k["movFrameRates"][:, None])
        w.double(d, "movFrameRateRatios", k["movFrameRateRatios"][:, None])
        w.cell(d, "pointNames", [lambda g, n, s=s: w.char(g, n, s) for s in k["pointNames"]])
        # lenses: the j-th camera with a lens references MCOS object j (its properties are MCOS cell 1 + j); the
        # others hold an empty double, as DLTdv leaves a camera without undistortion
        oid = {c: j + 1 for j, c in enumerate(sorted(lwms))}

        def obj_ref(g, n, c):
            ds = g.create_dataset(n, data=np.array([[0xDD000000, 2, 1, 1, oid[c], 1]], np.uint32))
            ds.attrs["MATLAB_class"] = np.bytes_(b"images.geotrans.LocalWeightedMeanTransformation2D")
            ds.attrs["MATLAB_object_decode"] = np.int32(3)
            return ds

        def empty(g, n):
            ds = g.create_dataset(n, data=np.zeros(2, np.uint64))
            ds.attrs["MATLAB_class"] = np.bytes_(b"double")
            ds.attrs["MATLAB_empty"] = np.uint8(1)
            return ds

        def camud(g, n, c):
            """maketform('custom', ...): the struct DLTdv saves, its inverse function named as MATLAB saves a
            function handle."""
            sg = w.struct(g, n)
            fh = w.struct(w.struct(sg, "inverse_fcn"), "function_handle")
            w.char(fh, "function", LENS_FUNCS[c])
            w.char(fh, "type", "simple")
            w.double(sg, "ndims_in", 2.0)
            w.double(sg, "ndims_out", 2.0)
            return sg
        w.cell(d, "camd", [(lambda g, n, c=c: obj_ref(g, n, c)) if c in lwms else empty for c in range(3)])
        if models:
            w.cell(d, "camud", [(lambda g, n, c=c: camud(g, n, c)) if c in LENS_MODELS else empty
                                for c in range(3)])
        v = w.struct(ud, "video")
        w.cell(v, "files", [lambda g, n, s=s: w.char(g, n, s) for s in k["files"]])
        w.cell(v, "offset", [lambda g, n, s=s: w.char(g, n, s) for s in k["video_offset"]])
        w.cell(v, "undistortString", [lambda g, n, s=s: w.char(g, n, s) for s in k["undistort"]])
        if lwms:
            def blob(g, n):
                ds = g.create_dataset(n, data=np.zeros((1, 64), np.uint8))
                ds.attrs["MATLAB_class"] = np.bytes_(b"uint8")
                return ds

            def lwm(g, n, c):
                (uv, xy), nn = lwms[c]
                sg = w.struct(g, n)
                w.double(sg, "N", nn)
                w.double(sg, "uvPoints", uv)
                w.double(sg, "xyPoints", xy)
                return sg
            ss = f.create_group("#subsystem#")
            w.cell(ss, "MCOS", [blob, empty] + [(lambda g, n, c=c: lwm(g, n, c)) for c in sorted(lwms)])
            ss["MCOS"].attrs["MATLAB_class"] = np.bytes_(b"FileWrapper__")
    head = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Thu Jan  1 12:00:00 2026 HDF5 schema 1.00 ."
    with open(path, "r+b") as fh:
        fh.write(head.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM")
    return path


def build(folder) -> dict:
    """Videos + both copies of the project, in `folder`: the v7 copy beside the videos, the v7.3 copy in a
    sub-folder (its videos are found in the parent folder)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "click").mkdir(exist_ok=True)
    vids = build_videos(str(folder))
    return {"videos": vids,
            "v7": write_v7(folder / "synth_v7_dvProject.mat"),
            "v73": write_v73(folder / "click" / "synth_dvProject.mat"),
            "lens": write_v73(folder / "click" / "synth_lens_dvProject.mat", models=True)}
