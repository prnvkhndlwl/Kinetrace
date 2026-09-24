"""3D layer on synthetic ground truth: DLT cameras built from known K/R/t,
triangulation and its residual, the LWM and OpenCV undistortion models,
sub-frame timing (fractional offsets + a 2x frame-rate camera) recovered by
the residual search, the v5 project round-trip with calibration and 3D, the
xyz export, and the visual hull (carving, marching tetrahedra, volumes).

No GPU, no video files. Run: .venv\\Scripts\\python.exe tests\\verify_3d.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from kinetrace.calib import (Calibration, CameraCalibration, LWMUndistort, NoUndistort,
                                 OpenCVUndistort, dlt_camera_center, dlt_from_camera, dlt_project,
                                 dlt_ray, estimate_offsets, front_sign, reconstruct, sample_track, triangulate,
                                 triangulate_batch)
from kinetrace.hull import (bounds_from_points, carve, hull_mesh, mesh_volume,
                                save_obj, save_ply, signed_distance, surface)
from kinetrace.project import Project
from kinetrace.session import TrackingSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(3)


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


W, H = 1280, 720
K = np.array([[1100.0, 0, W / 2], [0, 1100.0, H / 2], [0, 0, 1]])
positions = [np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.6 + 0.5 * (k % 3)])
             for k, a in enumerate(np.linspace(0, 2 * np.pi, 7)[:-1])]
cams = []
for pos in positions:
    R, t = look_at(pos)
    L = dlt_from_camera(K, R, t)
    cams.append(CameraCalibration(L, W, H, NoUndistort(), pixel_origin=0.0))
    assert np.allclose(dlt_camera_center(L), pos, atol=1e-6), "camera centre from DLT"
    # DLT projection must equal the pinhole projection
    X = rng.uniform(-0.3, 0.3, (50, 3))
    xc = (R @ X.T + t[:, None]).T
    uv_ref = (K @ (xc / xc[:, 2:3]).T).T[:, :2]
    assert np.allclose(dlt_project(L, X), uv_ref, atol=1e-6), "DLT projection"
    # rays through projected points pass through the points, pointing forward
    c, d = dlt_ray(L, uv_ref)
    assert np.allclose(np.cross(X - c, d), 0, atol=1e-6) and np.all(np.einsum("ij,ij->i", X - c, d) > 0)
print("DLT cameras from K/R/t OK (projection, centre, rays)")
# a calibration fitted with a MIRRORED image axis (det K < 0 — what easyWand /
# DLTdv coefficients look like on real data): the front-of-camera sign must
# come from a probe point in view, not from det(M)
Km = K.copy()
Km[1, 1] *= -1
R0, t0m = look_at(positions[0])
Lm = dlt_from_camera(Km, R0, t0m)
Mm = np.array([Lm[0:3], Lm[4:7], Lm[8:11]])
Xf = rng.uniform(-0.3, 0.3, (20, 3))
den = Lm[8] * Xf[:, 0] + Lm[9] * Xf[:, 1] + Lm[10] * Xf[:, 2] + 1.0
assert np.sign(np.linalg.det(Mm)) != np.sign(den[0]), "the mirrored case must contradict the det rule"
assert front_sign(Lm, Xf[0]) == np.sign(den[0])
cm, dm = dlt_ray(Lm, dlt_project(Lm, Xf), probe=Xf[0])
assert np.all(np.einsum("ij,ij->i", Xf - cm, dm) > 0), "mirrored camera: rays must point forward"
print("mirrored-axis calibration OK (front sign from a probe, not det M)")

# ---- triangulation + residual --------------------------------------------
X = rng.uniform(-0.3, 0.3, (200, 3))
uv = np.stack([dlt_project(c.coefs, X) for c in cams], axis=1)          # (200, 6, 2)
noise = rng.normal(0, 0.5, uv.shape)
xyz, res, n, e = triangulate_batch(np.stack([c.coefs for c in cams]), uv + noise)
err = np.linalg.norm(xyz - X, axis=1)
assert n.min() == 6 and err.mean() < 1.5e-3, (err.mean(), n.min())
one = [triangulate(np.stack([c.coefs for c in cams], 1), uv[i] + noise[i]) for i in range(20)]
assert np.allclose([o[0] for o in one], xyz[:20], atol=1e-9), "batch == single triangulation"
assert np.allclose([o[1] for o in one], res[:20], atol=1e-9)
# residual definition: sqrt(sum e^2 / (2n - 3))
assert np.allclose(res, np.sqrt(np.nansum(e ** 2, 1) / (2 * 6 - 3)), atol=1e-9)
# a missing camera (NaN) simply drops out; fewer than two -> NaN
uv2 = (uv + noise).copy()
uv2[:, 3] = np.nan
uv2[0, :5] = np.nan
xyz2, res2, n2, _ = triangulate_batch(np.stack([c.coefs for c in cams]), uv2)
assert n2[1] == 5 and n2[0] == 0 and np.isnan(xyz2[0]).all() and np.isfinite(xyz2[1]).all()
# outlier guard: a wrecked camera is dropped when max_residual is set
uv3 = (uv + noise).copy()
uv3[:, 2] += 40.0
_, r3, n3, _ = triangulate_batch(np.stack([c.coefs for c in cams]), uv3, 2, max_residual=3.0)
assert (n3 == 5).all() and r3.mean() < 1.5, (n3.mean(), r3.mean())
print(f"triangulation OK: 0.5 px noise -> {err.mean() * 1e3:.2f} mm mean error, residual formula, "
      "missing views, outlier guard")

# ---- undistortion models ---------------------------------------------------
# a synthetic fisheye-ish radial distortion, sampled on a grid -> LWM control points
def distort_radial(p, k1=-0.25, k2=0.06):
    q = (p - [W / 2, H / 2]) / 700.0
    r2 = (q ** 2).sum(1, keepdims=True)
    return (q * (1 + k1 * r2 + k2 * r2 ** 2)) * 700.0 + [W / 2, H / 2]


gx, gy = np.meshgrid(np.linspace(20, W - 20, 30), np.linspace(20, H - 20, 18))
und_ctrl = np.column_stack([gx.ravel(), gy.ravel()])
raw_ctrl = distort_radial(und_ctrl)
t0 = time.time()
lwm = LWMUndistort(raw_ctrl, und_ctrl, 12)
build = time.time() - t0
test_und = rng.uniform([60, 60], [W - 60, H - 60], (500, 2))
test_raw = distort_radial(test_und)
u = lwm.undistort(test_raw)
rt = lwm.distort(u)
e_u = np.linalg.norm(u - test_und, axis=1)
e_rt = np.linalg.norm(rt - test_raw, axis=1)
assert np.isfinite(u).all() and e_u.max() < 0.3 and e_rt.max() < 0.3, (e_u.max(), e_rt.max())
far = lwm.undistort(np.array([[-5000.0, -5000.0]]))
assert np.isnan(far).all(), "outside every control radius the LWM is undefined"
print(f"LWM undistortion OK: {len(raw_ctrl)} control points built in {build:.2f} s, "
      f"undistort max {e_u.max():.3f} px, round trip max {e_rt.max():.3f} px")
dist = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
ocv = OpenCVUndistort(K, dist)
raw = ocv.distort(test_und)
back = ocv.undistort(raw)
assert np.linalg.norm(back - test_und, axis=1).max() < 0.05
print("OpenCV undistortion round trip OK")
# (I73) OpenCV's default 5-step point undistortion is far from converged at the
# edges of a wide lens (1.8 px on a GoPro-like lens, 100+ px on a stronger one);
# the model must round-trip over the WHOLE picture, and a sound, invertible
# standard model must not be called a runaway by the lens border check
from kinetrace import lens as _lens  # noqa: E402
for _f, _k in ((1300.0, (-0.25, 0.08)), (1100.0, (-0.30, 0.10))):
    _Wl, _Hl = 2704, 1520
    _Kl = np.array([[_f, 0, _Wl / 2], [0, _f, _Hl / 2], [0, 0, 1.0]])
    _dl = np.array([_k[0], _k[1], 0.0, 0.0, 0.0])
    _gx, _gy = np.meshgrid(np.linspace(0, _Wl - 1, 120), np.linspace(0, _Hl - 1, 68))
    _raw = np.column_stack([_gx.ravel(), _gy.ravel()])
    _prof = _lens.LensProfile(_Wl, _Hl, _Kl, _dl, False, 0.3, 30)
    for _m in (OpenCVUndistort(_Kl, _dl), _prof.undistort_model()):
        _rt = np.linalg.norm(_m.distort(_m.undistort(_raw)) - _raw, axis=1)
        assert np.isfinite(_rt).all() and _rt.max() < 0.01, f"f={_f}: round trip {_rt.max():.3f} px at the edges"
    _chk = _prof.border_check()
    assert not _chk["runaway"] and _chk["valid_frac"] == 1.0, (_f, _chk)
print("OpenCV undistortion converges at the edges of wide lenses OK (I73)")

# a calibration in MATLAB convention (1-based, y up) must map Kinetrace pixels correctly
calm = CameraCalibration(cams[0].coefs, W, H, NoUndistort(), pixel_origin=1.0, y_flip=True)
p = np.array([[100.0, 50.0]])
q = calm.to_calib_frame(p)
assert np.allclose(q, [[101.0, H - 50.0]]), q         # x + 1; y from the bottom, 1-based
assert np.allclose(calm.from_calib_frame(q), p), "convention round trip"
print("coordinate conventions OK (pixel origin, y flip)")

# ---- files: dltCoefs.csv + easyWand .mat + npz persistence -----------------
csv = os.path.join(OUT, "test_dltCoefs.csv")
np.savetxt(csv, np.stack([c.coefs for c in cams], 1), delimiter=",", fmt="%.8g")
cal_csv = Calibration.load_dlt_csv(csv, [(W, H)] * 6, pixel_origin=0.0)
assert len(cal_csv) == 6 and np.allclose(cal_csv.cameras[2].coefs, cams[2].coefs, rtol=1e-6)
from scipy.io import savemat
matp = os.path.join(OUT, "test_easyWandData.mat")
savemat(matp, {"easyWandData": {"coefs": np.stack([c.coefs for c in cams], 1), "wandLen": 1.04,
                                "imageWidth": np.array([W] * 6), "imageHeight": np.array([H] * 6),
                                "dltRMSE": np.full(6, 0.7)}})
cal_ew = Calibration.load_easywand_mat(matp, pixel_origin=0.0)
assert len(cal_ew) == 6 and cal_ew.cameras[1].width == W and abs(cal_ew.cameras[1].rmse - 0.7) < 1e-9
assert np.allclose(cal_ew.cameras[4].coefs, cams[4].coefs, rtol=1e-6) and cal_ew.unit.startswith("wand=")
print("calibration importers OK (dltCoefs.csv, easyWandData.mat)")

# ---- sub-frame timing ------------------------------------------------------
# a point flying along a helix, filmed at 120 fps by five cameras and at 240
# fps by the sixth, each started at its own (fractional) time
T_REF = 200
fps_ref = 120.0
true_rates = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
true_off = [0.0, -14.35, 3.6, -2.15, 0.7, 63.4]      # local frames at reference instant 0
N_PTS = 4


def world_at(t_ref):
    """(N_PTS, 3) at reference time t_ref (frames)."""
    s = np.asarray(t_ref, np.float64)[..., None]
    base = np.stack([0.4 * np.cos(s[..., 0] * 0.05), 0.4 * np.sin(s[..., 0] * 0.05), 0.002 * s[..., 0]], -1)
    offs = np.array([[0, 0, 0], [0.05, 0.01, 0.0], [-0.03, 0.04, 0.02], [0.0, -0.05, 0.03]])
    return base[..., None, :] + offs


sessions = []
for c, cal in enumerate(cams):
    n_local = int(T_REF * true_rates[c]) + 80
    s = TrackingSession(f"cam{c}", n_local, fps_ref * true_rates[c], W, H)
    for j in range(N_PTS):
        s.add_landmark(f"p{j}")
    lf = np.arange(n_local)
    t_ref = (lf - true_off[c]) / true_rates[c]
    Xw = world_at(t_ref)                                    # (n_local, N_PTS, 3)
    for f in range(n_local):
        if 0 <= t_ref[f] <= T_REF:
            uvf = cal.project(Xw[f]) + rng.normal(0, 0.3, (N_PTS, 2))
            for j in range(N_PTS):
                s.set_position(f, j, float(uvf[j][0]), float(uvf[j][1]))
    sessions.append(s)
proj = Project(sessions, [f"cam{c}" for c in range(6)], [round(o) for o in true_off])
assert proj.rates == true_rates, proj.rates
cal_set = Calibration(cams, "m")
proj.calibration = cal_set
# whole-frame offsets: the error is the motion within half a frame
rec_int = reconstruct(sessions, cal_set, proj.rates, proj.offsets, (10, 180))
e_int = np.linalg.norm(rec_int.xyz - world_at(np.arange(10, 181)), axis=2)
# exact offsets: only the 0.3 px noise remains
rec_true = reconstruct(sessions, cal_set, proj.rates, true_off, (10, 180))
e_true = np.linalg.norm(rec_true.xyz - world_at(np.arange(10, 181)), axis=2)
assert e_true.mean() < 1.0e-3 and e_int.mean() > 3 * e_true.mean(), (e_true.mean(), e_int.mean())
t0 = time.time()
est, before, after = estimate_offsets(sessions, cal_set, proj.rates, proj.offsets, (10, 180))
dt = time.time() - t0
d_off = np.abs(np.array(est) - np.array(true_off))
# 0.3 px of click noise bounds how well a sub-frame shift can be seen; what
# matters is that the 3D error collapses to the noise floor
assert d_off.max() < 0.08, f"offsets not recovered: {est} vs {true_off}"
assert after < before
rec_est = reconstruct(sessions, cal_set, proj.rates, est, (10, 180))
e_est = np.linalg.norm(rec_est.xyz - world_at(np.arange(10, 181)), axis=2)
assert e_est.mean() < 1.3 * e_true.mean(), (e_est.mean(), e_true.mean())
print(f"sub-frame sync OK: whole-frame offsets {e_int.mean() * 1e3:.2f} mm, recovered offsets "
      f"{e_est.mean() * 1e3:.2f} mm (true {e_true.mean() * 1e3:.2f} mm); max offset error "
      f"{d_off.max():.3f} frames incl. the 240 fps camera; residual {before:.3f} -> {after:.3f} px in {dt:.1f} s")
# (I100) re-checking offsets that are already right: a sharp minimum with nothing
# to improve is RELIABLE (it used to say "weak ... changes it by only 969%")
_rep = {}
estimate_offsets(sessions, cal_set, proj.rates, list(est), (10, 180), report=_rep)
assert all(v == "sharp" for v in _rep["per_view"].values()), _rep["per_view"]
assert _rep["verdict"] == "reliable" and "shallow" not in _rep["why"] and "already" in _rep["why"], _rep
print("re-checked offsets are RELIABLE, not weak OK (I100)")
# sample_track: never bridges a gap
s0 = sessions[0]
s0.tracked[50, 0] = False
assert np.isnan(sample_track(s0.tracks, s0.tracked, 49.5)[0]).all()
assert np.isfinite(sample_track(s0.tracks, s0.tracked, 48.5)[0]).all()
s0.tracked[50, 0] = True

# ---- project v5 round trip with calibration + reconstruction ----------------
proj.offsets = list(est)
proj.reconstruction = rec_est
path = os.path.join(OUT, "test3d.cotrk")
proj.save_npz(path)
back = Project.load_npz(path)
assert back.rates == true_rates and np.allclose(back.offsets, est)
assert back.calibration is not None and len(back.calibration) == 6
assert np.allclose(back.calibration.cameras[3].coefs, cams[3].coefs) and back.calibration.unit == "m"
assert back.reconstruction is not None and np.allclose(back.reconstruction.xyz, rec_est.xyz, equal_nan=True)
assert back.reconstruction.names == rec_est.names and back.reconstruction.t0 == 10
# an LWM-undistorted camera survives the round trip too
proj.calibration.cameras[0].undistort = lwm
proj.save_npz(path)
back = Project.load_npz(path)
assert back.calibration.cameras[0].undistort.kind == "lwm"
assert np.allclose(back.calibration.cameras[0].undistort.undistort(test_raw[:5]), u[:5])
proj.calibration.cameras[0].undistort = NoUndistort()
# v4-style file (integer offsets, no rates) still loads with rates from fps
with np.load(path, allow_pickle=True) as z:
    old = {k: z[k] for k in z.files if not k.startswith(("calib_", "xyz_")) and k != "view_rates"}
old["view_offsets"] = np.array([0, -14, 4, -2, 1, 63], np.int64)
old["schema"] = 4
with open(os.path.join(OUT, "test3d_v4.cotrk"), "wb") as fh:
    np.savez_compressed(fh, **old)
v4 = Project.load_npz(os.path.join(OUT, "test3d_v4.cotrk"))
assert v4.rates == true_rates and v4.offsets == [0.0, -14.0, 4.0, -2.0, 1.0, 63.0] and v4.calibration is None
print("project v5 round trip OK (rates, fractional offsets, calibration incl. LWM, 3D; v4 loads)")

# ---- exports -----------------------------------------------------------------
xp = os.path.join(OUT, "test3d_xyz.csv")
rec_est.export_csv(xp)
rows = open(xp, encoding="utf-8").read().splitlines()
assert rows[0].startswith("frame,p0_X,p0_Y,p0_Z,") and len(rows) == rec_est.n_frames + 1
assert os.path.exists(os.path.join(OUT, "test3d_xyz_xyzres.csv"))
vals = np.array(rows[1].split(",")[1:4], float)
assert np.allclose(vals, rec_est.xyz[0, 0], atol=1e-6)
# (I95) a comma in a landmark name ("wing tip, left" is a natural one) must not
# add header cells and shift every later landmark's columns
import csv as _csv  # noqa: E402
from kinetrace.calib import Reconstruction as _Rec  # noqa: E402
_rc = _Rec(5, ["wing tip, left", "tail"], np.arange(12, dtype=float).reshape(2, 2, 3), np.full((2, 2), 0.5),
           np.full((2, 2), 2, np.int32), per_cam=np.full((2, 2, 3), 0.25))
_xpc = os.path.join(OUT, "test3d_xyz_comma.csv")
_rc.export_csv(_xpc)
for _fn in (_xpc, os.path.join(OUT, "test3d_xyz_comma_xyzres.csv")):
    with open(_fn, encoding="utf-8", newline="") as _fh:
        _rows = list(_csv.reader(_fh))
    assert all(len(_r) == len(_rows[0]) for _r in _rows), (_fn, [len(_r) for _r in _rows])
with open(_xpc, encoding="utf-8", newline="") as _fh:
    _rows = list(_csv.reader(_fh))
assert _rows[0][4:7] == ["tail_X", "tail_Y", "tail_Z"], _rows[0]
assert [float(v) for v in _rows[1][4:7]] == list(_rc.xyz[0, 1]), _rows[1]
print("xyz export OK (a comma in a landmark name keeps the columns aligned, I95)")

# ---- visual hull -------------------------------------------------------------
axes = np.array([0.12, 0.05, 0.03])
centre = np.array([0.05, -0.02, 0.01])
pts = centre + axes * rng.uniform(-1, 1, (300000, 3))
pts = pts[(((pts - centre) / axes) ** 2).sum(1) <= 1]
masks = []
for c in cams:
    uvp = np.round(c.project(pts)).astype(int)
    m = np.zeros((H, W), np.uint8)
    ok = (uvp[:, 0] >= 0) & (uvp[:, 0] < W) & (uvp[:, 1] >= 0) & (uvp[:, 1] < H)
    m[uvp[ok, 1], uvp[ok, 0]] = 1
    masks.append(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool))
lo, hi = bounds_from_points(pts[:2000], 0.1)
t0 = time.time()
coarse = carve(cams, masks, lo, hi, 0.01, dilate_px=1)
ext = coarse.extent()
fine = carve(cams, masks, ext[0] - 0.01, ext[1] + 0.01, 0.002, dilate_px=0)
dt = time.time() - t0
V = 4 / 3 * np.pi * axes.prod()
ratio = fine.volume() / V
assert 0.95 < ratio < 1.6, f"hull volume {ratio:.2f} x analytic"
assert np.linalg.norm(fine.centroid() - centre) < 0.002
# a hull is the intersection: every true body point is inside it
inside_idx = np.round((pts[:5000] - fine.origin) / fine.voxel).astype(int)
inb = np.all((inside_idx >= 0) & (inside_idx < np.array(fine.occupancy.shape)), axis=1)
hit = fine.occupancy[tuple(inside_idx[inb].T)]
assert hit.mean() > 0.97, hit.mean()
# a camera behind the object with no silhouette (None) does not vote
fine2 = carve(cams, masks[:3] + [None, None, None], ext[0] - 0.01, ext[1] + 0.01, 0.002, dilate_px=0)
assert fine2.n_views == 3 and fine2.volume() >= fine.volume()
# the same hull with camera 0 mirrored (image flipped vertically): the front
# test must still admit the body — this is the real-data failure mode
cam_m = CameraCalibration(Lm, W, H, NoUndistort(), pixel_origin=0.0)
mask_m = masks[0][::-1].copy()
fine3 = carve([cam_m] + cams[1:], [mask_m] + masks[1:], ext[0] - 0.01, ext[1] + 0.01, 0.002, dilate_px=1)
assert fine3.n_voxels > 0.5 * fine.n_voxels, (fine3.n_voxels, fine.n_voxels)
assert np.linalg.norm(fine3.centroid() - centre) < 0.003
# (I94) camera 0's lens map (an identity LWM) covers only the left part of its
# picture, ending mid-silhouette: where it has no map the camera saw nothing and
# must not vote (it used to count as "seen, outside the silhouette" and carve the
# body away along the coverage edge)
_ys, _xs = np.nonzero(masks[0])
_xcut = float(np.median(_xs))
_gx, _gy = np.meshgrid(np.arange(_xs.min() - 60.0, _xcut, 6.0), np.arange(_ys.min() - 60.0, _ys.max() + 66.0, 6.0))
_ctrl = np.column_stack([_gx.ravel(), _gy.ravel()])          # a dense map, so it ends close to _xcut
cam_part = CameraCalibration(cams[0].coefs, W, H, LWMUndistort(_ctrl, _ctrl, 12), pixel_origin=0.0)
part = carve([cam_part] + cams[1:], masks, ext[0] - 0.01, ext[1] + 0.01, 0.002, dilate_px=0)
assert part.volume() >= fine.volume() and part.volume() >= V, (part.volume(), fine.volume(), V)
_hit_p = part.occupancy[tuple(inside_idx[inb].T)]
assert _hit_p.mean() >= hit.mean(), (_hit_p.mean(), hit.mean())
print(f"  partial lens map: hull {part.volume() / V * 100:.0f}% of analytic (full map {ratio * 100:.0f}%) OK (I94)")
t0 = time.time()
verts, faces = hull_mesh(fine)
mt = time.time() - t0
vm = mesh_volume(verts, faces)
assert abs(vm - fine.volume()) / fine.volume() < 0.08, (vm, fine.volume())
save_obj(os.path.join(OUT, "test_hull.obj"), verts, faces, "test")
save_ply(os.path.join(OUT, "test_hull.ply"), verts, faces)
# marching tetrahedra on an analytic sphere
n = 40
g = np.mgrid[0:n, 0:n, 0:n]
sdf = 8.0 - np.sqrt(((g - n / 2) ** 2).sum(0))
v, f = surface(sdf, np.zeros(3), 1.0, 0.0)
vs = mesh_volume(v, f)
assert abs(vs - 4 / 3 * np.pi * 512) / (4 / 3 * np.pi * 512) < 0.02
assert signed_distance(np.zeros((3, 3, 3), bool)).max() < 0
print(f"visual hull OK: ellipsoid hull {ratio * 100:.0f}% of the analytic volume from 6 views "
      f"({fine.n_voxels} voxels in {dt:.1f} s), {hit.mean() * 100:.1f}% of body points inside, "
      f"mesh {len(faces)} triangles in {mt:.1f} s, mesh volume within {abs(vm - fine.volume()) / fine.volume() * 100:.1f}% "
      f"of the voxel count; sphere mesh volume within {abs(vs - 4 / 3 * np.pi * 512) / (4 / 3 * np.pi * 512) * 100:.1f}%")

# ---- K + R/t (OpenCV) cameras -> DLT, with camera 1 at the file's origin --------------
# (the usual stereo convention has no DLT until the origin moves)
from kinetrace.calib import Calibration as _Cal, dlt_project as _proj
_K1 = np.array([[2783.0, 0, 1920.0], [0, 2784.0, 1080.0], [0, 0, 1.0]])
_K2 = np.array([[2801.0, 0, 1920.0], [0, 2802.0, 1080.0], [0, 0, 1.0]])
_ang = np.deg2rad(4.4)
_R2 = np.array([[np.cos(_ang), 0, np.sin(_ang)], [0, 1, 0], [-np.sin(_ang), 0, np.cos(_ang)]])
_t2 = np.array([-2.88, -0.003, 0.09])
_cal = _Cal.from_krt([{"K": _K1, "width": 3840, "height": 2160},
                      {"K": _K2, "R": _R2, "t": _t2, "width": 3840, "height": 2160}], unit="m", source="test")
assert len(_cal) == 2 and _cal.origin_shift is not None and _cal.cameras[0].pixel_origin == 0.0
_X = np.array([[0.3, -0.2, 4.5], [-1.0, 0.4, 5.2], [1.2, 0.1, 6.0], [0.0, 0.0, 4.0]])   # file frame
for _c, (_K, _R, _t) in enumerate(((_K1, np.eye(3), np.zeros(3)), (_K2, _R2, _t2))):
    _xc = (_R @ _X.T).T + _t
    _uv_true = (_K @ (_xc / _xc[:, 2:3]).T).T[:, :2]
    _uv_dlt = _proj(_cal.cameras[_c].coefs, _X - _cal.origin_shift)          # DLT frame = file frame - shift
    assert np.abs(_uv_dlt - _uv_true).max() < 1e-6, f"camera {_c}: DLT does not reproduce K[R|t]"
    _uv_app = _cal.cameras[_c].project(_X - _cal.origin_shift)                # through the convention layer too
    assert np.abs(_uv_app - _uv_true).max() < 1e-6
# and back: triangulating the two DLT views recovers the (shifted) points
from kinetrace.calib import triangulate as _tri
for _i in range(len(_X)):
    _uv = np.stack([_proj(_cal.cameras[c].coefs, _X[_i:_i + 1] - _cal.origin_shift)[0] for c in range(2)])
    _xyz, _res, _ = _tri(np.stack([c.coefs for c in _cal.cameras]), _uv)
    assert np.abs(_xyz + _cal.origin_shift - _X[_i]).max() < 1e-6 and _res < 1e-6
# npz round trip keeps the shift
_arr = _cal.to_arrays("calib_")
_back = _Cal.from_arrays(_arr, "calib_")
assert np.allclose(_back.origin_shift, _cal.origin_shift) and np.allclose(_back.cameras[1].coefs, _cal.cameras[1].coefs)
# JSON and the plain-text form OpenCV stereo tools print (unicode minus / arrow / tabs)
import json as _json
SCRATCH = os.path.join(ROOT, "tests", "out")
os.makedirs(SCRATCH, exist_ok=True)
_jp = os.path.join(SCRATCH, "krt.json")
open(_jp, "w", encoding="utf-8").write(_json.dumps({"unit": "m", "cameras": [
    {"K": _K1.tolist(), "width": 3840, "height": 2160},
    {"K": _K2.tolist(), "R": _R2.tolist(), "t": _t2.tolist(), "width": 3840, "height": 2160}]}))
_cj = _Cal.load_krt(_jp)
assert np.allclose(_cj.cameras[1].coefs, _cal.cameras[1].coefs) and _cj.unit == "m"
_tp = os.path.join(SCRATCH, "krt.txt")
_T = np.eye(4); _T[:3, :3] = _R2; _T[:3, 3] = _t2
_rows = lambda M: "\n".join("\t".join(f"{v:.7f}".replace("-", "−") for v in row) for row in M)
open(_tp, "w", encoding="utf-8").write(
    f"Intrinsics at 3840×2160\n\nC1 K\n\nK1=[{_rows(_K1)}]\n\nC2 K\n\nK2=[{_rows(_K2)}]\n\t​\n\n"
    f"Stereo extrinsic\n\nTC1→C2=[{_rows(_T)}]\n")
_ct = _Cal.load_krt(_tp)
assert np.allclose(_ct.cameras[1].coefs, _cal.cameras[1].coefs, atol=1e-6) and _ct.cameras[0].width == 3840
print("K + R/t importer OK (camera at the file's origin handled by moving the DLT origin)")
# (I96) D1 / D2 distortion lines in the plain-text form: 5 numbers are applied,
# 4 (standard without k3, or fisheye?) are not guessed at and the notes say so
_D = "D1=[-0.21 0.05 0.0005 -0.0003 0.01]\nD2=[-0.2 0.04 0.0 0.0 0.0]\n"
open(_tp, "a", encoding="utf-8").write("\n" + _D)
_cd = _Cal.load_krt(_tp)
assert [c.undistort.kind for c in _cd.cameras] == ["opencv", "opencv"] and not _cd.notes, _cd.notes
assert np.allclose(_cd.cameras[0].undistort.dist, [-0.21, 0.05, 0.0005, -0.0003, 0.01])
_tp4 = os.path.join(SCRATCH, "krt4.txt")
open(_tp4, "w", encoding="utf-8").write(open(_tp, encoding="utf-8").read()
                                         .replace(_D, "D1=[-0.21 0.05 0.001 0.002]\nD2=[-0.2 0.04 0 0]\n"))
_c4 = _Cal.load_krt(_tp4)
assert [c.undistort.kind for c in _c4.cameras] == ["none", "none"] and _c4.notes and "NOT applied" in _c4.notes[0]
print("K + R/t distortion lines applied when unambiguous, named when not OK (I96)")

# ---- epipolar geometry: collinear rigs, probes without a reconstruction ---------------
from kinetrace.calib import (closest_on_polyline, epipolar_polyline, intersect_polylines,  # noqa: E402
                                 reconstruction_report, working_probe)
Kb = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
# (I97) three cameras on one bar: the other two cameras' epipolar lines in the
# middle view are the SAME line, so they do not "cross"; the old least-squares
# answer landed hundreds of px away. Snap may move the point onto the line only.
for _raise in (0.0, 0.6):
    bar = []
    for _xb in (-0.6, 0.0, 0.6):
        _pos = np.array([_xb, -3.0, _raise if _xb == 0.0 else 0.0])
        _R, _t = look_at(_pos)
        bar.append(CameraCalibration(dlt_from_camera(Kb, _R, _t), 1280, 720, NoUndistort(), pixel_origin=0.0))
    _X = np.array([0.05, 0.1, 0.08])
    _uv = [c.project(_X[None])[0] for c in bar]
    _pr = working_probe(Calibration(bar))
    _lines = [epipolar_polyline(bar[k], bar[1], _uv[k], _pr) for k in (0, 2)]
    _disp = np.array([15.0, -9.0])
    _info = {}
    _snap = intersect_polylines(_lines, _uv[1] + _disp, _info)
    if _raise == 0.0:          # collinear: onto the shared line, never farther than the displacement
        assert _info["mode"] == "parallel", _info
        assert np.linalg.norm(_snap - _uv[1]) <= np.linalg.norm(_disp) + 0.5, (_snap, _uv[1])
        assert np.linalg.norm(closest_on_polyline(_lines[0], _snap) - _snap) < 0.1
    else:                      # the middle camera raised: a real crossing, back on the truth
        assert _info["mode"] == "crossing" and np.linalg.norm(_snap - _uv[1]) < 0.5, (_info, _snap, _uv[1])
print(f"snap to rays on a collinear rig moves onto the line only ({np.linalg.norm(_disp):.0f} px "
      "displacement bound); a raised middle camera still crosses OK (I97)")
# (I98) no reconstruction yet: the probe comes from the calibration alone. On a
# near-parallel or toed-out stereo bar the old probe sat hundreds of metres away
# or BEHIND the cameras and the guide missed by ~200 px -- also for a mirrored
# (easyWand-like) calibration, where the det rule is wrong
Kbm = Kb.copy()
Kbm[1, 1] *= -1
for _Kp in (Kb, Kbm):
    for _toe in (0.05, -0.05, -0.3, -1.0, 0.3, 2.0):     # degrees, + = toed in
        pair = []
        for _sx in (-1, 1):
            _pos = np.array([0.25 * _sx, -3.0, 0.0])
            _yaw = np.radians(_toe) * -_sx
            _R, _t = look_at(_pos, _pos + np.array([np.sin(_yaw), np.cos(_yaw), 0.0]))
            pair.append(CameraCalibration(dlt_from_camera(_Kp, _R, _t), 1280, 720, NoUndistort(), pixel_origin=0.0))
        _pr = working_probe(Calibration(pair))
        for _X in (np.array([0.05, 0.2, -0.05]), np.array([-0.3, 1.5, 0.2])):
            _u0, _u1 = pair[0].project(_X[None])[0], pair[1].project(_X[None])[0]
            _line = epipolar_polyline(pair[0], pair[1], _u0, _pr)
            _miss = np.linalg.norm(closest_on_polyline(_line, _u1) - _u1) if len(_line) >= 2 else np.inf
            assert _miss < 1.5, (_toe, _X, _miss, _pr)
# a world origin BEHIND one camera (the assumption the rule rests on is broken):
# detected from the mixed sign of det M, and the converging-rays search still works
_tgt = np.array([0.0, 5.0, 0.0])
pair = []
for _pos in (np.array([-1.0, 2.0, 0.0]), np.array([3.0, 6.0, 0.0])):      # the origin is behind the first
    _R, _t = look_at(_pos, _tgt)
    pair.append(CameraCalibration(dlt_from_camera(Kb, _R, _t), 1280, 720, NoUndistort(), pixel_origin=0.0))
_pr = working_probe(Calibration(pair))
assert _pr is not None and np.linalg.norm(_pr - _tgt) < 0.5, _pr
for _X in (_tgt + [0.1, 0.2, -0.05], _tgt + [-0.3, -0.4, 0.2]):
    _u0, _u1 = pair[0].project(_X[None])[0], pair[1].project(_X[None])[0]
    _line = epipolar_polyline(pair[0], pair[1], _u0, _pr)
    assert len(_line) >= 2 and np.linalg.norm(closest_on_polyline(_line, _u1) - _u1) < 1.5
print("epipolar guides without a reconstruction hit on near-parallel / toed-out / mirrored pairs OK (I98)")

# ---- reconstruction verdict: the two-camera blind spot from the evidence (I101) -------
_T = 50
_z = np.zeros((_T, 2, 3))
_r2 = reconstruction_report(_Rec(0, ["a", "b"], _z, np.full((_T, 2), 0.4), np.full((_T, 2), 2, np.int32)), 3, 1920)
assert _r2["verdict"] == "ok" and any("only two" in s.lower() for s in _r2["reasons"]), _r2
_nc = np.full((_T, 2), 2, np.int32)
_nc[:20] = 3                                   # a third camera cross-checks part of the clip
_r3 = reconstruction_report(_Rec(0, ["a", "b"], _z, np.full((_T, 2), 0.4), _nc), 6, 1920)
assert _r3["verdict"] == "good" and any("cross-check" in s for s in _r3["reasons"]), _r3
_nc[:, 1] = 2
_r4 = reconstruction_report(_Rec(0, ["a", "b"], _z, np.full((_T, 2), 0.4), _nc), 6, 1920)
assert any("Never seen by three cameras" in s and ": b." in s for s in _r4["reasons"]), _r4["reasons"]
assert reconstruction_report(_Rec(0, ["a"], _z[:, :1], np.full((_T, 1), 0.4), np.full((_T, 1), 3, np.int32)),
                             3, 1920)["verdict"] == "good"
print("two-camera caveat follows the points, not the project's camera count OK (I101)")

# ---- DLTdv project whose lens store cannot be read / does not match (I102) ------------
from scipy.io import savemat as _savemat  # noqa: E402
import scipy.io as _sio  # noqa: E402
from kinetrace.calib import _assign_lwm_profiles  # noqa: E402
_dvp = os.path.join(OUT, "test_synth_dvProject.mat")
_savemat(_dvp, {"udExport": {"data": {"dltcoef": np.stack([c.coefs for c in cams[:3]], 1),
                                      "movsizes": np.array([[H, W]] * 3)}}})
_plain = Calibration.load_dltdv_project(_dvp)
assert len(_plain) == 3 and not _plain.notes and _plain.cameras[1].width == W
_orig_loadmat = _sio.loadmat


def _garbage_workspace(*a, **k):
    d = _orig_loadmat(*a, **k)
    d["__function_workspace__"] = np.frombuffer(b"not a MAT stream" * 8, np.uint8)
    return d


_sio.loadmat = _garbage_workspace
try:
    _broken = Calibration.load_dltdv_project(_dvp)
finally:
    _sio.loadmat = _orig_loadmat
assert all(c.undistort.kind == "none" for c in _broken.cameras)
assert _broken.notes and "could not be read" in _broken.notes[0], _broken.notes


class _TF:
    _fieldnames = ["uvPoints", "xyPoints", "N"]


_cells = [np.zeros(4), np.zeros(0), _TF(), _TF()]            # 2 profiles for 3 cameras
_prof3, _note3 = _assign_lwm_profiles(3, _cells, [None, None, None])
assert _prof3 == [None, None, None] and "2 lens profile(s) for 3 cameras" in _note3
_prof3, _note3 = _assign_lwm_profiles(3, _cells, [1, None, 2])      # camera 2 left as filmed in DLTdv
assert _prof3[0] is _cells[2] and _prof3[1] is None and _prof3[2] is _cells[3] and "camera(s) 2" in _note3
_prof3, _note3 = _assign_lwm_profiles(2, _cells, [])
assert _prof3 == [_cells[2], _cells[3]] and _note3 == ""
print("DLTdv lens store: a failure or a count mismatch is SAID, profiles follow their cameras OK (I102)")

# ---- kinematics: smoothing + derivatives against a known trajectory ------------------
from kinetrace import kinematics as kin  # noqa: E402

fps_k = 120.0
T_k = 480
tt = np.arange(T_k) / fps_k
A_, w_ = 0.20, 2 * np.pi * 1.5                 # a 1.5 Hz oscillation of 20 cm amplitude
pos = np.stack([A_ * np.sin(w_ * tt), 0.5 * A_ * np.cos(w_ * tt), 0.02 * tt], axis=1)
vel_true = np.stack([A_ * w_ * np.cos(w_ * tt), -0.5 * A_ * w_ * np.sin(w_ * tt), np.full(T_k, 0.02)], axis=1)
acc_true = np.stack([-A_ * w_ * w_ * np.sin(w_ * tt), -0.5 * A_ * w_ * w_ * np.cos(w_ * tt), np.zeros(T_k)], axis=1)
noise = np.random.RandomState(9).normal(0, 0.0005, pos.shape)          # 0.5 mm of tracking jitter
xyz_k = (pos + noise)[:, None, :]
xyz_k[200:215] = np.nan                                                  # a gap
sm, fc, why = kin.smooth_xyz(xyz_k, fps_k, "auto")
v_s, a_s = kin.derivatives(sm, fps_k)
v_r, a_r = kin.derivatives(xyz_k, fps_k)
mid = np.r_[30:190, 230:450]                                             # away from the edges and the gap
ev_s = np.nanmedian(np.linalg.norm(v_s[mid, 0] - vel_true[mid], axis=1))
ev_r = np.nanmedian(np.linalg.norm(v_r[mid, 0] - vel_true[mid], axis=1))
ea_s = np.nanmedian(np.linalg.norm(a_s[mid, 0] - acc_true[mid], axis=1))
ea_r = np.nanmedian(np.linalg.norm(a_r[mid, 0] - acc_true[mid], axis=1))
print(f"kinematics: auto cutoff {fc:.1f} Hz ({why[:60]}...); velocity error {ev_r:.3f} -> {ev_s:.3f} m/s, "
      f"acceleration error {ea_r:.1f} -> {ea_s:.2f} m/s^2 (peak true {np.abs(acc_true).max():.1f})")
assert 3.0 <= fc <= 30.0, fc
assert ev_s < 0.3 * ev_r and ev_s < 0.02, (ev_s, ev_r)
assert ea_s < 0.2 * ea_r and ea_s < 1.5, (ea_s, ea_r)
assert np.isnan(sm[205, 0]).all() and np.isnan(v_s[205, 0]).all(), "gaps stay gaps"
assert np.isfinite(sm[199, 0]).all() and np.isfinite(v_s[216, 0]).all()
sm0, fc0, why0 = kin.smooth_xyz(xyz_k, fps_k, None)
assert fc0 is None and np.allclose(sm0, xyz_k, equal_nan=True)
sm5, fc5, _ = kin.smooth_xyz(xyz_k, fps_k, 5.0)
assert fc5 == 5.0
from kinetrace.calib import Reconstruction  # noqa: E402
rec_k = Reconstruction(100, ["pt"], xyz_k, np.zeros((T_k, 1)), np.full((T_k, 1), 2), unit="m")
out_k = os.path.join(OUT, "kin_test.csv")
files = kin.export_kinematics(out_k, rec_k, fps_k, "m", "auto")
assert len(files) == 2 and all(os.path.exists(f) for f in files)
rows = open(out_k, encoding="utf-8").read().splitlines()
assert rows[0].startswith("# unit m") and rows[1].split(",")[:2] == ["frame", "time_s"] and len(rows) == T_k + 2
assert rows[1].count("pt_") == 11
rep_txt = open(files[1], encoding="utf-8").read()
assert "9.81" in rep_txt and "residual analysis" in rep_txt
# (I4) the automatic cutoff must not filter away a fast movement (a wingbeat), must not be
# dragged down by landmarks that stand still, and must say so when it cannot decide
def _amp_at(sig, fps, f):
    """Amplitude of the f-Hz component of a 1-D signal (least squares, ends trimmed)."""
    t = np.arange(len(sig)) / fps
    m = slice(len(sig) // 10, 9 * len(sig) // 10)
    B = np.stack([np.sin(2 * np.pi * f * t[m]), np.cos(2 * np.pi * f * t[m]), np.ones(len(t[m]))], 1)
    c = np.linalg.lstsq(B, sig[m], rcond=None)[0]
    return float(np.hypot(c[0], c[1]))


def _moving(fps, comps, seed, T=480, noise=0.0005):
    t = np.arange(T) / fps
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in comps)
    p_ = np.stack([x, 0.01 * np.cos(2 * np.pi * 0.5 * t), 0.02 * t], axis=1)
    return p_ + np.random.RandomState(seed).normal(0, noise, p_.shape)


wing = _moving(120.0, [(3.0, 0.03), (16.0, 0.04)], 1)                  # 3 Hz sway + a 16 Hz wingbeat
fc_w, info_w = kin.residual_analysis(wing, 120.0)
kept_w = _amp_at(kin.lowpass(wing, 120.0, fc_w)[:, 0], 120.0, 16.0) / 0.04
print(f"  wingbeat 16 Hz at 120 fps: auto cutoff {fc_w:.1f} Hz, keeps {100 * kept_w:.1f} % of the wingbeat, "
      f"noise estimate {1000 * info_w['noise_rms']:.2f} mm (true 0.50)")
assert fc_w > 20.0 and kept_w > 0.95, (fc_w, kept_w, info_w["reason"])
assert info_w["noise_rms"] < 0.002, info_w["noise_rms"]
fast = _moving(120.0, [(20.0, 0.03)], 2)
fc_f, info_f = kin.residual_analysis(fast, 120.0)
print(f"  20 Hz at 120 fps: auto cutoff {fc_f:.1f} Hz; {info_f['reason'][-80:]}")
assert fc_f > 20.0, "the cutoff must never land below the movement's own frequency"
assert "careful" in info_f["reason"] and "20.0 Hz" in info_f["reason"], info_f["reason"]
too_fast = _moving(120.0, [(30.0, 0.03)], 3)                           # a quarter of the frame rate
fc_t, info_t = kin.residual_analysis(too_fast, 120.0)
assert info_t["unreliable"] and "cannot be trusted" in info_t["reason"], info_t["reason"]
assert _amp_at(kin.lowpass(too_fast, 120.0, fc_t)[:, 0], 120.0, 30.0) / 0.03 > 0.9, fc_t
still = np.random.RandomState(4).normal(0, 0.0005, (480, 3)) + [0.3, 0.1, 0.0]
fc_s, info_s = kin.residual_analysis(still, 120.0)
assert info_s["still"] and "barely moves" in info_s["reason"], info_s["reason"]
body = _moving(120.0, [(1.5, 0.2)], 5)
refs = np.random.RandomState(6).normal(0, 0.0005, (480, 2, 3)) + np.array([[0.3, 0.1, 0.0], [-0.2, 0.4, 0.0]])
_, fc_b, _ = kin.smooth_xyz(body[:, None, :], 120.0, "auto")
_, fc_br, why_br = kin.smooth_xyz(np.concatenate([body[:, None, :], refs], axis=1), 120.0, "auto")
print(f"  a moving landmark alone: {fc_b:.1f} Hz; with two still reference markers: {fc_br:.1f} Hz")
assert abs(fc_br - fc_b) < 0.05 * fc_b, (fc_b, fc_br, why_br)
assert "left out" in why_br, why_br
# a wing tip beside the body: the shared cutoff is lower than the wing asks for - the export names it
_bw = np.stack([body, body + np.stack([0.04 * np.sin(2 * np.pi * 16 * np.arange(480) / 120.0),
                                       np.zeros(480), np.zeros(480)], axis=1)], axis=1)
rec_bw = Reconstruction(0, ["body", "wingtip"], _bw, np.zeros((480, 2)), np.full((480, 2), 3), unit="m")
files_bw = kin.export_kinematics(os.path.join(OUT, "kin_wing.csv"), rec_bw, 120.0, "m", "auto")
rep_bw = open(files_bw[1], encoding="utf-8").read()
head_bw = open(files_bw[0], encoding="utf-8").readline()
assert "faster than this cutoff keeps: wingtip" in head_bw, head_bw
_wline = rep_bw.split("  wingtip:")[1].split("\n\n")[0]
assert "on its own it asks for" in _wline and rep_bw.count("    ! ") == 1, rep_bw   # the wing, not the body
print("automatic cutoff keeps fast movement, ignores still markers, flags what it cannot decide (I4) OK")

# (I5) a perfect free fall reads 9.81 m/s^2 on EVERY frame, the ends of the run included,
# and the report's peak acceleration is 9.81 - the report's own hand check
fps_d, T_d = 240.0, 120
t_d = np.arange(T_d) / fps_d
drop = np.stack([0.3 * t_d, np.zeros(T_d), 1.0 - 0.5 * kin.G * t_d ** 2], axis=1)[:, None, :]
rec_d = Reconstruction(0, ["ball"], drop, np.zeros((T_d, 1)), np.full((T_d, 1), 3), unit="m")
out_d = os.path.join(OUT, "kin_drop.csv")
for cut in ("auto", 20.0, None):
    files_d = kin.export_kinematics(out_d, rec_d, fps_d, "m", cut)
    rows_d = [r.split(",") for r in open(files_d[0], encoding="utf-8").read().splitlines()[2:]]
    az = np.array([float(r[2 + 9]) for r in rows_d])                  # ball_Az
    worst = float(np.max(np.abs(az + kin.G)))
    rep_d = open(files_d[1], encoding="utf-8").read()
    peak = float(rep_d.split("acceleration up to ")[1].split(" ")[0])
    print(f"  free fall, smoothing {cut}: Az {az[0]:.2f} .. {az[-1]:.2f}, worst frame off by {worst:.3f}, "
          f"report peak {peak:.2f} m/s^2")
    assert worst < 0.05 * kin.G, (cut, az[:4], az[-4:])
    assert abs(peak - kin.G) < 0.05 * kin.G, (cut, peak)
print("free fall reads 9.81 m/s^2 at the ends of the run too, and in the report (I5) OK")

# (I6) a stretch too short to smooth between two gaps: its derivatives are left blank and
# said so, instead of raw jitter setting the landmark's peak in a file labelled smoothed
fps_i, T_i = 240.0, 1200
t_i = np.arange(T_i) / fps_i
w_i = 2 * np.pi * 1.5
isl = np.stack([0.2 * np.sin(w_i * t_i), np.zeros(T_i), np.zeros(T_i)], axis=1)
isl = (isl + np.random.RandomState(9).normal(0, 0.0005, isl.shape))[:, None, :]
isl[380:410] = np.nan
isl[418:451] = np.nan                                                # frames 410-417: an 8-frame island
rec_i = Reconstruction(0, ["pt"], isl, np.zeros((T_i, 1)), np.full((T_i, 1), 2), unit="m")
out_i = os.path.join(OUT, "kin_island.csv")
files_i = kin.export_kinematics(out_i, rec_i, fps_i, "m", "auto")
lines_i = open(files_i[0], encoding="utf-8").read().splitlines()
acc_i = np.array([float(r.split(",")[2 + 10]) for r in lines_i[2:]])  # pt_acc
rep_i = open(files_i[1], encoding="utf-8").read()
peak_i = float(rep_i.split("acceleration up to ")[1].split(" ")[0])
true_peak = 0.2 * w_i * w_i
print(f"  8-frame island: acceleration there {acc_i[410:418]}, report peak {peak_i:.2f} (true {true_peak:.2f})")
assert np.isnan(acc_i[410:418]).all(), acc_i[410:418]
assert "could not be smoothed" in lines_i[0] and "too short to smooth" in rep_i
assert peak_i < 1.3 * true_peak, (peak_i, true_peak)
print("stretches too short to smooth are not differentiated, and say so (I6) OK")
# a comma in a landmark name must not split its column (the CSV sanitizer, as in I95)
rec_c = Reconstruction(0, ["left, wrist", "tail"], np.concatenate([isl, isl], axis=1), np.zeros((T_i, 2)),
                       np.full((T_i, 2), 2), unit="m")
lines_c = open(kin.export_kinematics(os.path.join(OUT, "kin_comma.csv"), rec_c, fps_i, "m", "auto")[0],
               encoding="utf-8").read().splitlines()
hdr_c = lines_c[1].split(",")
assert len(hdr_c) == 2 + 2 * 11 and all(len(r.split(",")) == len(hdr_c) for r in lines_c[2:]), len(hdr_c)
assert hdr_c[2] == "left_ wrist_X" and hdr_c[13] == "tail_X", hdr_c[:14]
print("landmark names are CSV-safe in the kinematics header OK")
print("kinematics OK")

# ---- re-track planner and verdict (pure, no video, no GPU) -----------------------------
# The verify_retrack rig without its videos: three 640x480 cameras round a landmark set.
from kinetrace import retrack  # noqa: E402
from kinetrace.calib import dlt_from_camera as _dlt_rt, reconstruct as _recon_rt  # noqa: E402

_W, _H, _NF = 640, 480, 120
_K_rt = np.array([[700.0, 0, _W / 2], [0, 700.0, _H / 2], [0, 0, 1]])


def _look_at_rt(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z]), None


_cams_rt, _Rt_rt = [], []
for _k, _a in enumerate((0.3, 2.4, 4.4)):
    _pos = np.array([2.0 * np.cos(_a), 2.0 * np.sin(_a), 0.5 + 0.4 * _k])
    _R = _look_at_rt(_pos)[0]
    _Rt_rt.append((_R, -_R @ _pos))
    _cams_rt.append(CameraCalibration(_dlt_rt(_K_rt, _R, -_R @ _pos), _W, _H, NoUndistort(), pixel_origin=0.0))


def _world_rt(f):
    hip = np.array([0.3 * np.cos(f * 0.05), 0.3 * np.sin(f * 0.05), 0.003 * f])
    return np.stack([hip + [0.08, 0.0, 0.03], hip, hip - [0.09, 0.0, 0.01]])


_truth_rt = [np.stack([np.round(cal.project(_world_rt(f))) for f in range(_NF)]) for cal in _cams_rt]


def _rig_rt(slides=None, truth=None):
    """slides {camera: {frame: (dx, dy)}} applied to the head; NaN truth cells stay empty."""
    truth = truth if truth is not None else _truth_rt
    sess = []
    for c in range(3):
        s = TrackingSession(f"cam{c}.mp4", _NF, 60.0, _W, _H)
        for nm in ("head", "hip", "tail"):
            s.add_landmark(nm)
        for f in range(_NF):
            for j in range(3):
                if np.isfinite(truth[c][f, j]).all():
                    s.set_position(f, j, float(truth[c][f, j, 0]), float(truth[c][f, j, 1]))
            s.manual[f] = False
        for f, d in (slides or {}).get(c, {}).items():
            s.tracks[f, 0] = truth[c][f, 0] + np.asarray(d, float)
        s._touch()
        sess.append(s)
    p_ = Project(sess, ["camA", "camB", "camC"], [0.0, 0.0, 0.0])
    p_.calibration = Calibration(_cams_rt)
    p_.reconstruction = _recon_rt(p_.sessions, p_.calibration, p_.rates, p_.offsets, (0, _NF - 1))
    return p_


# (I8) the slid camera's error dips under its band mid-slide, so it has TWO stretches; the
# blame must rank cameras, not stretches, and re-track both of its stretches
_slide = {f: (30.0, 0.0) for f in range(30, 71)}
_slide.update({f: (20.0, 0.0) for f in range(45, 51)})
_bands = [5.0, 5.0, 10.0]                     # camC twice as wide as the others would get this
_plan8 = retrack.plan(_rig_rt({2: _slide}), _bands)
_c8 = [st for st in _plan8 if st.view == 2 and st.name == "head"]
_o8 = [st for st in _plan8 if st.view != 2 and st.name == "head"]
for st in _plan8:
    print(f"  plan: {['camA', 'camB', 'camC'][st.view]} {st.name} {st.local0}-{st.local1} rest {st.rest_px:.2f} "
          f"-> {'RE-TRACK' if st.target is not None else 'skip: ' + st.reason}")
assert len(_c8) == 2, "the scenario needs camC split into two stretches"
assert all(st.target is not None and not st.reason for st in _c8), [st.reason for st in _c8]
assert _o8 and all(st.target is None and "camC" in st.reason for st in _o8), [st.reason for st in _o8]
print("leave-one-out blames the camera, not the stretch: both camC stretches re-trackable (I8) OK")

# (I9) the other cameras put the landmark OUTSIDE camC's picture (its tracker parked a guess
# inside): no border hand placement, the stretch is skipped with the reason
_R2, _t2 = _Rt_rt[2]


def _back_rt(u, v, depth=2.0):
    ray = np.linalg.inv(_K_rt) @ np.array([u, v, 1.0])
    return _R2.T @ (depth * ray / ray[2] - _t2)


_head_w = np.stack([_back_rt(320.0, 360.0 + 5.0 * f if f < 30 else (510.0 if f <= 60 else 510.0 - 5.0 * (f - 60)))
                    for f in range(_NF)])
_truth9 = []
for c, cal in enumerate(_cams_rt):
    tr = np.stack([cal.project(np.stack([_head_w[f], _head_w[f] + [0.03, 0, 0], _head_w[f] + [0, 0.03, 0]]))
                   for f in range(_NF)])
    tr[(tr[..., 0] < 0) | (tr[..., 0] >= _W) | (tr[..., 1] < 0) | (tr[..., 1] >= _H)] = np.nan
    _truth9.append(tr)
_p9 = _rig_rt(truth=_truth9)
_s9 = _p9.sessions[2]
for f in range(_NF):
    if not np.isfinite(_truth9[2][f, 0]).all():            # the head is out of camC: a parked guess inside
        _s9.set_position(f, 0, 320.0, 310.0)
        _s9.manual[f, 0] = False
_p9.reconstruction = _recon_rt(_p9.sessions, _p9.calibration, _p9.rates, _p9.offsets, (0, _NF - 1))
_plan9 = [st for st in retrack.plan(_p9, [5.0] * 3) if st.view == 2 and st.name == "head"]
_why9: list = []
_got9 = retrack.ray_target(_p9, 2, "head", 40, why=_why9)
print(f"  landmark out of camC's picture: plan {[(st.local0, st.local1, st.reason[:60]) for st in _plan9]}")
assert _got9 is None and _why9 and "outside this camera's picture" in _why9[0], (_got9, _why9)
assert _plan9 and all(st.target is None and "outside this camera's picture" in st.reason for st in _plan9)
assert retrack.ray_target(_p9, 2, "head", 10) is not None, "inside the picture the ray target still works"
print("a ray target outside the picture is refused, not clipped onto the border (I9) OK")
# one other camera's ray is a line, not a point: where only camB sees the head (camA lost it
# for the first 10 frames of camC's slide), the stretch starts where BOTH others see it
_truth1 = [tr.astype(float).copy() for tr in _truth_rt]
_truth1[0][25:40, 0] = np.nan
_plan1 = [st for st in retrack.plan(_rig_rt({2: {f: (25.0, 0.0) for f in range(30, 70)}}, truth=_truth1), [5.0] * 3)
          if st.view == 2 and st.name == "head"]
assert len(_plan1) == 1 and _plan1[0].target is not None, [(st.local0, st.reason) for st in _plan1]
_st1 = _plan1[0]
_d1 = float(np.linalg.norm(_st1.target - _truth_rt[2][_st1.local0, 0]))
print(f"  camA blind on 25-39: camC stretch re-seeded at frame {_st1.local0} on {_st1.n_rays} rays, "
      f"{_d1:.2f} px from the truth; {_st1.note}")
assert _st1.local0 == 40 and _st1.n_rays == 2 and _d1 < 1.5, (_st1.local0, _st1.n_rays, _d1)
assert "30-39 left as they are" in _st1.note, _st1.note
_truth1[0][25:75, 0] = np.nan                                       # camA never sees the slide at all
_plan1b = [st for st in retrack.plan(_rig_rt({2: {f: (25.0, 0.0) for f in range(30, 70)}}, truth=_truth1),
                                     [5.0] * 3) if st.view == 2 and st.name == "head"]
assert all(st.target is None for st in _plan1b), [(st.local0, st.target, st.reason) for st in _plan1b]
print("a re-seed needs two other cameras' rays: the stretch starts where they cross, or is skipped OK")

# (I7) a re-track job that auto-paused has its track cut from the fail frame: the verdict must
# count the lost cells (not only the median of the survivors) and name them
def _retrack_sim(n_good):
    p_ = _rig_rt({2: {f: (25.0, 0.0) for f in range(30, 70)}})
    doable = [st for st in retrack.plan(p_, [5.0] * 3) if st.target is not None]
    assert len(doable) == 1 and doable[0].view == 2, doable
    st = doable[0]
    before = retrack.cells_summary(p_, doable)
    s = p_.sessions[2]
    pid = s.pid_by_name("head")
    s.set_position(st.local0, pid, float(st.target[0]), float(st.target[1]))       # app._retrack_next
    for f in range(st.local0 + 1, st.local0 + n_good):                             # the worker's good frames
        s.set_position(f, pid, float(_truth_rt[2][f, 0, 0]), float(_truth_rt[2][f, 0, 1]))
    if st.local0 + n_good <= st.local1:                                            # auto-pause: the cut
        s.clear_window([pid], st.local0 + n_good, st.local1)
    p_.reconstruction = _recon_rt(p_.sessions, p_.calibration, p_.rates, p_.offsets, (0, _NF - 1))
    return st, before, retrack.cells_summary(p_, doable), p_


_st7, _b7, _a7, _p7 = _retrack_sim(10)
_v7 = retrack.verdict(_b7, _a7, [5.0, 5.0, 5.0], _p7)
print(f"  cut after 10 of {_st7.local1 - _st7.local0 + 1} frames: before n={_b7['n_cells']} median "
      f"{_b7['median_px']:.1f}, after n={_a7['n_cells']} median {_a7['median_px']:.1f} -> {_v7[0]}: {_v7[1][:140]}")
assert _v7[0] == "worse", _v7
assert "Data lost" in _v7[1] and f"{_st7.local1 - _st7.local0 + 1 - 10} of {_b7['n_cells']}" in _v7[1], _v7[1]
assert f"head in camC: frames {_st7.local0 + 10}-{_st7.local1}" in _v7[1], _v7[1]
assert retrack.verdict(_b7, _a7, 5.0)[0] == "worse"                   # a single band works the same
_st7b, _b7b, _a7b, _p7b = _retrack_sim(_st7.local1 - _st7.local0 + 1)                 # nothing cut
_v7b = retrack.verdict(_b7b, _a7b, [5.0, 5.0, 5.0], _p7b)
assert _v7b[0] == "better" and "under the 5.0 px band" in _v7b[1], _v7b
assert retrack.verdict(_b7b, _a7b, [5.0, 5.0, 10.0], _p7b)[1].count("10.0 px") == 1   # camC's own band
print("the re-track verdict counts and names lost frames; a clean fix is still BETTER (I7) OK")

print("VERIFY 3D PASSED")
