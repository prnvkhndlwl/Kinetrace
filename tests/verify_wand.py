"""Native wand calibration (kinetrace/wand.py) on synthetic ground truth.

Synthetic: 5 cameras (four 1920x1080 f=1400, one 1280x720 f=900) around a
   1 m^3 volume, a 0.5 m wand in 150 poses, 40 background points, 0.3 px
   noise, 15% of the observations dropped, 5 gross outliers of 30 px. Focal
   lengths UNKNOWN. Then the two world-frame alignments (three reference
   points; a 60 fps drop -> gravity, g as an independent scale check), the
   DLTdv CSV export (1-based pixels) and the .kcal.json round trip, the
   max_frames path, distortion estimation, transform consistency, error paths.

No GPU, no video files. Run: .venv\\Scripts\\python.exe tests\\verify_wand.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from kinetrace import wand
from kinetrace.calib import Calibration, NoUndistort, OpenCVUndistort, dlt_project, triangulate_batch
from kinetrace.wand import (WandError, align_axes, align_gravity, calibrate_wand, export_dlt_csv,
                                transform_result)


def kcal_round_trip(res, path):
    """The one calibration file the app writes and reads (.kcal.json)."""
    import json
    from kinetrace.calibwizard import calibration_to_kcal, load_kcal
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calibration_to_kcal(res.to_calibration()), fh)
    return load_kcal(path)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(11)
T0 = time.time()
summary = []


def ascii_(s):
    return str(s).encode("ascii", "replace").decode()


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


# =========================================================== 1. synthetic rig
sizes = [(1920, 1080), (1920, 1080), (1280, 720), (1920, 1080), (1920, 1080)]
focals = [1400.0, 1400.0, 900.0, 1400.0, 1400.0]
Ks, Rs, ts = [], [], []
for k, (az, h) in enumerate(zip([0.1, 1.3, 2.4, 3.6, 5.0], [0.3, 1.0, 0.6, 1.4, 0.8])):
    pos = np.array([2.4 * np.cos(az), 2.4 * np.sin(az), h])
    R, t = look_at(pos, rng.uniform(-0.1, 0.1, 3))
    w, hh = sizes[k]
    Ks.append(np.array([[focals[k], 0, w / 2], [0, focals[k], hh / 2], [0, 0, 1]]))
    Rs.append(R)
    ts.append(t)
centres_true = [-Rs[c].T @ ts[c] for c in range(5)]


def project(X, c, noise=0.3):
    """Truth projection into camera c with gaussian pixel noise; NaN when the
    point is outside the image or behind the camera."""
    xc = (Rs[c] @ X.T + ts[c][:, None]).T
    uv = (Ks[c] @ (xc / xc[:, 2:3]).T).T[:, :2] + rng.normal(0, noise, (len(X), 2))
    w, h = sizes[c]
    out = (uv[:, 0] < 0) | (uv[:, 0] > w - 1) | (uv[:, 1] < 0) | (uv[:, 1] > h - 1) | (xc[:, 2] <= 0)
    uv[out] = np.nan
    return uv


F = 150
cen = rng.uniform(-0.4, 0.4, (F, 3))
d = rng.normal(size=(F, 3))
d /= np.linalg.norm(d, axis=1, keepdims=True)
ends = np.stack([cen - 0.25 * d, cen + 0.25 * d], axis=1)             # (F, 2, 3), 0.5 m apart
bg = rng.uniform(-0.5, 0.5, (40, 3))
wand_uv = np.full((F, 5, 2, 2), np.nan)
bg_uv = np.full((40, 5, 2), np.nan)
for c in range(5):
    wand_uv[:, c] = project(ends.reshape(-1, 3), c).reshape(F, 2, 2)
    bg_uv[:, c] = project(bg, c)
wand_uv[rng.rand(F, 5, 2) < 0.15] = np.nan
bg_uv[rng.rand(40, 5) < 0.15] = np.nan
finite = np.argwhere(np.isfinite(wand_uv).all(-1))
injected = finite[rng.choice(len(finite), 5, replace=False)]
for f_, c_, e_ in injected:
    ang = rng.uniform(0, 2 * np.pi)
    wand_uv[f_, c_, e_] += 30.0 * np.array([np.cos(ang), np.sin(ang)])
n_obs = int(np.isfinite(wand_uv).all(-1).sum() + np.isfinite(bg_uv).all(-1).sum())
print(f"synthetic rig: 5 cameras, {F} wand frames, 40 background points, {n_obs} observations, "
      f"5 outliers of 30 px injected at (frame, cam, end) = {injected.tolist()}")

msgs = []
t0 = time.time()
res = calibrate_wand(wand_uv, 0.5, sizes, bg_uv=bg_uv, focal=None, estimate_focal=True, unit="m",
                     progress=lambda f, m: msgs.append((f, m)))
dt_syn = time.time() - t0
r = res.report
assert msgs and msgs[-1][0] == 1.0 and all(0.0 <= f <= 1.0 for f, _ in msgs), "progress callback"
print(f"calibrate_wand (focal unknown) in {dt_syn:.1f} s: wand score {r['wand_score_pct']:.3f}%, "
      f"reprojection {r['reproj_rmse_all']:.3f} px, {r['outliers_removed']} outliers removed, "
      f"grid pick {r['focal_grid_best']} x width, verdict '{r['verdict']}'")
assert r["wand_score_pct"] < 0.5, r["wand_score_pct"]
assert r["reproj_rmse_all"] < 0.6, r["reproj_rmse_all"]
assert r["n_frames_used"] >= 130 and r["n_bg_points"] >= 35, (r["n_frames_used"], r["n_bg_points"])
f_err = [abs(fe - ft) / ft for fe, ft in zip(r["focal_px"], focals)]
print("  focal px: fitted", [round(f, 1) for f in r["focal_px"]], "truth", focals,
      f"(max {100 * max(f_err):.2f}% off)")
assert max(f_err) < 0.03, f_err
d_err = []
for i, j, dd in r["camera_distances"]:
    dt_ = np.linalg.norm(centres_true[i] - centres_true[j])
    d_err.append(abs(dd - dt_) / dt_)
print(f"  camera-to-camera distances: max {100 * max(d_err):.3f}% off over {len(d_err)} pairs")
assert max(d_err) < 0.01, d_err
flagged = {(o["frame"], o["cam"], o["end"]) for o in r["outliers"] if o["kind"] == "wand"}
for f_, c_, e_ in injected:
    assert (int(f_), int(c_), int(e_)) in flagged, ("injected outlier not rejected", (f_, c_, e_))
    assert not res.reproj_obs["wand_used"][f_, c_, e_]
    px = [o["px"] for o in r["outliers"] if o["kind"] == "wand" and (o["frame"], o["cam"], o["end"]) == (f_, c_, e_)][0]
    assert px is not None and px > 10.0, px
assert r["outliers_removed"] <= 5 + 0.02 * n_obs, ("too many rejections", r["outliers_removed"])
print(f"  all 5 injected outliers flagged (rejected at "
      f"{[round(o['px'], 1) for o in r['outliers'] if o['kind'] == 'wand' and (o['frame'], o['cam'], o['end']) in {tuple(int(v) for v in q) for q in injected}]} px); "
      f"{r['outliers_removed'] - 5} additional noise-tail rejections")
assert set(r) >= {"n_cameras", "n_frames_in", "n_frames_used", "n_bg_points", "wand_length", "wand_mean",
                  "wand_sd", "wand_score_pct", "reproj_rmse_px", "reproj_rmse_all", "focal_px",
                  "focal_estimated", "coverage_pct", "camera_distances", "outliers_removed", "verdict",
                  "verdict_reasons"}
assert r["verdict"] in ("good", "ok") and r["verdict_reasons"] and r["focal_estimated"] is True
assert abs(r["wand_mean"] - 0.5) < 1e-9, r["wand_mean"]            # scale fixed to the mean length
assert res.wand_xyz.shape == (F, 2, 3) and res.bg_xyz.shape == (40, 3) and res.K.shape == (5, 3, 3)
assert res.reproj_obs["wand"].shape == (F, 5, 2) and res.reproj_obs["bg"].shape == (40, 5)
assert all(c.pixel_origin == 0.0 and not c.y_flip and np.isfinite(c.rmse) for c in res.cameras)
cal0 = res.to_calibration()
assert len(cal0) == 5 and cal0.unit == "m" and cal0.source.startswith("wand")
import json as _json
_json.dumps(r)                                                       # report must be JSON-serialisable
for reason in r["verdict_reasons"]:
    print("   -", ascii_(reason))
summary.append(("synthetic, focal unknown", r["wand_score_pct"], r["reproj_rmse_all"], r["outliers_removed"],
                r["verdict"], dt_syn))

# ---- world frame from three reference points --------------------------------
O = np.array([0.1, -0.2, -0.3])
ref = np.stack([O, O + [0.6, 0, 0], O + [0.2, 0.5, 0]])          # origin, +X, XY-plane in the TRUTH frame
ref_uv = np.stack([project(ref, c) for c in range(5)], axis=1)   # (3, C, 2)
res_ax = align_axes(res, ref_uv[0], ref_uv[1], ref_uv[2])
e_bg = np.linalg.norm(res_ax.bg_xyz - (bg - O), axis=1)          # aligned frame = truth frame shifted to O
e_w = np.linalg.norm(res_ax.wand_xyz - (ends - O), axis=2)
print(f"align_axes: background points vs truth median {np.nanmedian(e_bg) * 1000:.2f} mm, max "
      f"{np.nanmax(e_bg) * 1000:.2f} mm; wand ends max {np.nanmax(e_w) * 1000:.2f} mm")
assert np.nanmax(e_bg) < 0.005 and np.nanmax(e_w) < 0.005
X = np.array([[0.1, 0.2, -0.1], [-0.3, 0.1, 0.25], [0.45, -0.4, 0.3]])
for c in range(5):
    pe = np.abs(res_ax.cameras[c].project(X - O) - project(X, c, 0.0)).max()
    assert pe < 1.0, (c, pe)                                     # DLT coefs in the aligned frame
assert res_ax.report["frame"].startswith("axes")

# ---- world frame from a dropped object -----------------------------------------
fps, N = 60.0, 30
tt = np.arange(N) / fps
drop_xyz = np.array([0.05, -0.05, 0.55])[None] + 0.5 * np.array([0, 0, -9.81])[None] * tt[:, None] ** 2
drop_uv = np.stack([project(drop_xyz, c) for c in range(5)], axis=1)
drop_uv[3:6, 1] = np.nan                                         # a few frames lost in one camera
drop_uv[12, :] = np.nan                                          # one frame lost everywhere
assert (np.isfinite(drop_uv).all(-1).sum(1) >= 2).sum() >= N - 1
res_g, ginfo = align_gravity(res, drop_uv, fps)
print(f"align_gravity: g measured {ginfo['g_measured']:.3f} m/s^2 (ratio {ginfo['g_ratio']:.4f}), "
      f"{ginfo['n_frames_used']} frames, fit rms {ginfo['fit_rms'] * 1000:.2f} mm")
assert abs(ginfo["g_ratio"] - 1.0) < 0.02, ginfo
coefs_g = np.stack([c.coefs for c in res_g.cameras])
xyz_g, _, _, _ = triangulate_batch(coefs_g, drop_uv)
ok = np.isfinite(xyz_g).all(1)
A = np.column_stack([np.ones(N), tt, 0.5 * tt ** 2])
acc = np.linalg.lstsq(A[ok], xyz_g[ok], rcond=None)[0][2]
print(f"  drop acceleration in the aligned frame: ({acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f}) m/s^2")
assert np.linalg.norm(acc - [0, 0, -9.81]) < 0.02 * 9.81, acc
assert np.linalg.norm(xyz_g[0]) < 0.003                          # origin = first drop point
z_off = res_g.bg_xyz[:, 2] - bg[:, 2]                            # Z up in both frames -> constant offset
assert np.nanstd(z_off) < 0.005, np.nanstd(z_off)
assert "gravity_check" in res_g.report and res_g.report["frame"].startswith("gravity")
assert ginfo["applied"] is True and ginfo["g_expected"] == 9.81
summary.append(("synthetic gravity check", ginfo["g_ratio"], ginfo["residual_px"], ginfo["n_frames_used"],
                "g ratio", 0.0))

# ---- (I25) a "drop" that is not one free fall must NOT set the vertical ------------
# A resting mark (noise only), a ball that bounces, and a ball held, dropped and left
# lying: each used to rotate the world (random vertical / upside down) under a GOOD
# headline. Now: no rotation, camera 1's axes, verdict capped at ok, frames named.
res_good = transform_result(res, np.eye(3), np.zeros(3), 1.0)
res_good.report["verdict"] = "good"                              # so the cap is visible


def _bounce_path(n=90, z0=0.5, floor=-0.4):
    out, z, v = [], z0, 0.0
    for _ in range(n):
        out.append([0.05, -0.05, z])
        v -= 9.81 / fps
        z += v / fps
        if z < floor:
            z, v = floor, -0.7 * v
    return np.array(out)


def _hold_drop_rest(n=120, hold=50, z0=0.5, floor=-0.4):
    tt_ = (np.arange(n) - hold).clip(0) / fps
    z = np.maximum(z0 - 0.5 * 9.81 * tt_ ** 2, floor)
    return np.column_stack([np.full(n, 0.05), np.full(n, -0.05), z])


bad_drops = {"static mark": np.tile([0.1, 0.2, -0.3], (60, 1)), "drop + bounces": _bounce_path(),
             "hold, drop, rest": _hold_drop_rest()}
for label, path in bad_drops.items():
    duv = np.stack([project(path, c) for c in range(5)], axis=1)
    out_b, info_b = align_gravity(res_good, duv, fps, frame0=100)
    moved = max(np.abs(out_b.cameras[c].coefs - res_good.cameras[c].coefs).max() for c in range(5))
    print(f"  bad drop '{label}': applied={info_b['applied']}, g ratio {info_b['g_ratio']:.3f}, "
          f"verdict {out_b.report['verdict']} | {ascii_(info_b['why'])[:70]}")
    assert info_b["applied"] is False and info_b["why"], (label, info_b)
    assert moved == 0.0, (label, "the world was rotated anyway", moved)
    assert out_b.report["verdict"] == "ok", (label, out_b.report["verdict"])
    assert "NOT applied" in out_b.report["frame"] and out_b.report["frame"].startswith("camera 1"), out_b.report["frame"]
    assert any("NOT set" in r and "frames 100-" in r for r in out_b.report["verdict_reasons"]), \
        out_b.report["verdict_reasons"][-1]
    assert res_good.report["verdict"] == "good"                  # the input is left alone
# and a clean drop still is used, with the verdict untouched
out_ok, info_ok = align_gravity(res_good, drop_uv, fps)
assert info_ok["applied"] and out_ok.report["verdict"] == "good" and out_ok.report["frame"].startswith("gravity")
print("bad drops refused (no rotation, camera 1's axes, verdict capped), clean drop still applied OK")

# ---- (I27) the gravity check in the world's own unit ------------------------------
# The same rig in mm / cm / in must give g ratio ~1 (it read 1000 / 100 / 39 and told
# the user the wand length was wrong); in "wand lengths" the fall measures the wand.
for unit_, L_ in (("mm", 500.0), ("cm", 50.0), ("in", 0.5 / 0.0254), ("wand", 1.0)):
    r_u = calibrate_wand(wand_uv, L_, sizes, bg_uv=bg_uv, focal=focals, estimate_focal=False, unit=unit_)
    out_u, gi_u = align_gravity(r_u, drop_uv, fps)
    assert gi_u["applied"], (unit_, gi_u)
    if unit_ == "wand":
        assert gi_u["g_ratio"] is None and gi_u["g_expected"] is None, gi_u
        assert abs(gi_u["implied_wand_length_m"] - 0.5) < 0.01, gi_u
        assert "0.50" in out_u.report["verdict_reasons"][-1] and "metres" in out_u.report["verdict_reasons"][-1]
        print(f"  unit wand: the fall says the wand is {gi_u['implied_wand_length_m']:.4f} m (truth 0.5)")
    else:
        assert abs(gi_u["g_ratio"] - 1.0) < 0.02, (unit_, gi_u["g_ratio"])
        assert "consistent" in out_u.report["verdict_reasons"][-1], out_u.report["verdict_reasons"][-1]
        print(f"  unit {unit_}: g {gi_u['g_measured']:.2f} {unit_}/s^2, ratio {gi_u['g_ratio']:.4f}")
print("gravity check is unit-aware OK")

# ---- DLTdv csv export (1-based pixels) and .kcal.json round trip --------------
csv_path = os.path.join(OUT, "wand_dltCoefs.csv")
export_dlt_csv(res_ax, csv_path, pixel_origin_out=1.0)
cal1 = Calibration.load_dlt_csv(csv_path, sizes, pixel_origin=1.0)
assert len(cal1) == 5
for c in range(5):
    a = res_ax.cameras[c].project(X)
    b = cal1.cameras[c].project(X)                               # 1-based coefs, origin fix applied back
    assert np.abs(a - b).max() < 1e-6, (c, np.abs(a - b).max())
    raw = np.genfromtxt(csv_path, delimiter=",")
    assert raw.shape == (11, 5)
    # the 1-based coefficients project to u + 1, v + 1 directly
    assert np.abs(dlt_project(raw[:, c], X) - (a + 1.0)).max() < 1e-6
export_dlt_csv(res_ax, csv_path, pixel_origin_out=0.0)
raw0 = np.genfromtxt(csv_path, delimiter=",")
assert np.allclose(raw0[:, 2], res_ax.cameras[2].coefs, rtol=1e-9, atol=1e-9)
json_path = os.path.join(OUT, "wand_test.kcal.json")
cal2 = kcal_round_trip(res_ax, json_path)
assert len(cal2) == 5 and cal2.unit == "m"
for c in range(5):
    assert np.allclose(cal2.cameras[c].coefs, res_ax.cameras[c].coefs)
    assert cal2.cameras[c].width == sizes[c][0] and cal2.cameras[c].pixel_origin == 0.0
    assert isinstance(cal2.cameras[c].undistort, NoUndistort)
    assert np.abs(cal2.cameras[c].project(X) - res_ax.cameras[c].project(X)).max() < 1e-9
print("export_dlt_csv (pixel_origin 1 and 0) and .kcal.json round trips OK")

# ---- transform_result consistency ------------------------------------------------
Rw, _ = look_at(np.array([1.0, 2.0, 0.5]), np.zeros(3))
res_t = transform_result(res, Rw, np.array([0.3, -0.2, 0.9]), 2.5)
for c in range(5):
    p0 = res.cameras[c].project(res.bg_xyz[:5])
    p1 = res_t.cameras[c].project(res_t.bg_xyz[:5])
    assert np.abs(p0 - p1).max() < 1e-6                           # projections invariant under the similarity
assert abs(res_t.report["wand_mean"] - 2.5 * 0.5) < 1e-9 and np.allclose(res_t.wand_len, 2.5 * res.wand_len, equal_nan=True)
try:
    transform_result(res, np.eye(3) * 2, np.zeros(3))
    raise AssertionError("a non-rotation must be refused")
except WandError:
    pass

# ---- max_frames: frames beyond the budget are triangulated afterwards ---------
res_m = calibrate_wand(wand_uv, 0.5, sizes, bg_uv=bg_uv, focal=focals, estimate_focal=False, max_frames=60)
assert res_m.frame_used.sum() <= 60
unsel = ~res_m.frame_used
assert np.isfinite(res_m.wand_len[unsel]).sum() >= 0.8 * unsel.sum()
med = np.nanmedian(res_m.wand_len[unsel])
print(f"max_frames=60: {int(res_m.frame_used.sum())} frames fitted, {int(np.isfinite(res_m.wand_len[unsel]).sum())} "
      f"more triangulated afterwards (median length {med:.4f} m)")
assert abs(med - 0.5) < 0.005 and res_m.report["wand_score_pct"] < 0.6

# ---- distortion estimation (none in the data: must stay harmless) ---------------
t0 = time.time()
res_d = calibrate_wand(wand_uv, 0.5, sizes, bg_uv=bg_uv, focal=focals, estimate_focal=True, estimate_distortion=True)
rd = res_d.report
print(f"estimate_distortion on undistorted data ({time.time() - t0:.1f} s): reprojection {rd['reproj_rmse_all']:.3f} px, "
      f"k1 = {[round(k[0], 4) for k in rd['distortion']]}")
assert rd["reproj_rmse_all"] < 0.6 and rd["distortion_estimated"]
assert all(isinstance(c.undistort, OpenCVUndistort) for c in res_d.cameras)
res_dg, _ = align_gravity(res_d, drop_uv, fps)
for c in range(5):
    pe = np.abs(res_dg.cameras[c].project(drop_xyz[:5] - drop_xyz[0]) - project(drop_xyz[:5], c, 0.0))
    assert np.nanmax(pe) < 1.5, (c, np.nanmax(pe))               # K + dist + DLT agree with the truth
cal_d = kcal_round_trip(res_d, json_path)
assert isinstance(cal_d.cameras[0].undistort, OpenCVUndistort)
assert rd["distortion_estimated_per_camera"] == [True] * 5

# ---- (I35) distortion per camera: fitted where asked, held at zero elsewhere ----------
# (a rig where some cameras carry a lens profile fits k1/k2 only for the others; it used
# to switch distortion off for EVERY camera as soon as one had a profile)
per = [True, False, True, False, False]
res_p = calibrate_wand(wand_uv, 0.5, sizes, bg_uv=bg_uv, focal=focals, estimate_focal=True, estimate_distortion=per)
rp_ = res_p.report
assert rp_["distortion_estimated"] is True and rp_["distortion_estimated_per_camera"] == per, rp_
for c in range(5):
    k12 = rp_["distortion"][c]
    if per[c]:
        assert any(k != 0.0 for k in k12), (c, k12)
    else:
        assert k12 == [0.0, 0.0] and isinstance(res_p.cameras[c].undistort, NoUndistort), (c, k12)
assert rp_["reproj_rmse_all"] < 0.6
print(f"per-camera distortion: fitted on cameras 1 and 3 only, k1 = {[round(k[0], 4) for k in rp_['distortion']]} OK")

# ---- error paths ----------------------------------------------------------------
blind = wand_uv.copy()
blind[:, 3] = np.nan
bg_blind = bg_uv.copy()
bg_blind[:, 3] = np.nan
try:
    calibrate_wand(blind, 0.5, sizes, bg_uv=bg_blind, focal=focals)
    raise AssertionError("a camera without observations must be an error")
except WandError as ex:
    # (I34) counted from 1 like every other place in the app: index 3 is "camera 4"
    assert "camera 4" in str(ex) and "camera 3 " not in str(ex), str(ex)
    print("  blind camera ->", ascii_(str(ex))[:90], "...")
NAMES = ["cam1", "CAM2", "GX04", "side", "top"]
try:
    calibrate_wand(blind, 0.5, sizes, bg_uv=bg_blind, focal=focals, names=NAMES)
    raise AssertionError("a camera without observations must be an error")
except WandError as ex:
    assert "side sees only" in str(ex), str(ex)                 # the project's own name
# every per-camera reason names the camera the way the report's table does
assert any(r.startswith("Camera 1:") for r in res.report["verdict_reasons"]), res.report["verdict_reasons"]
assert not any(r.startswith("Camera 0") for r in res.report["verdict_reasons"])
assert res.report["camera_names"] == ["camera 1", "camera 2", "camera 3", "camera 4", "camera 5"]
res_n = calibrate_wand(wand_uv, 0.5, sizes, bg_uv=bg_uv, focal=focals, estimate_focal=False, names=NAMES)
assert res_n.report["camera_names"] == NAMES and res_n.report["frame"].startswith("cam1's axes")
cov_reasons = [r for r in res_n.report["verdict_reasons"] if "the wand covered only" in r]
assert cov_reasons and all(r.split(":")[0] in NAMES for r in cov_reasons), cov_reasons
print("camera names in messages (1-based / project names) OK")
for bad in ({"wand_uv": wand_uv[:, :, 0]}, {"sizes": sizes[:3]}, {"wand_length": 0.0}):
    kw = {"wand_uv": wand_uv, "wand_length": 0.5, "sizes": sizes}
    kw.update(bad)
    try:
        calibrate_wand(kw["wand_uv"], kw["wand_length"], kw["sizes"], focal=focals)
        raise AssertionError(f"bad input accepted: {list(bad)}")
    except WandError:
        pass
try:
    align_axes(res, np.full((5, 2), np.nan), ref_uv[1], ref_uv[2])
    raise AssertionError("an unseen reference point must be an error")
except WandError as ex:
    assert "origin" in str(ex)
print("error paths OK")

# =========================================================== 2. summary
for p in (csv_path, json_path):
    if os.path.exists(p):
        os.remove(p)
print()
print(f"{'calibration':<36} {'wand score %':>12} {'reproj px':>10} {'rejected':>9} {'verdict':>8} {'time s':>7}")
print("-" * 88)
for name, score, rmse, rej, verdict, dt in summary:
    if verdict == "g ratio":
        continue
    print(f"{name:<36} {score:>12.3f} {rmse:>10.3f} {rej:>9} {verdict:>8} {dt:>7.1f}")
print()
print(f"{'gravity check (dropped object)':<36} {'g ratio':>12} {'resid px':>10} {'frames':>9}")
print("-" * 70)
for name, ratio, resid, n, verdict, _ in summary:
    if verdict == "g ratio":
        print(f"{name:<36} {ratio:>12.4f} {resid:>10.3f} {n:>9}")
print(f"total {time.time() - T0:.1f} s")
print("verify_wand PASSED")
