"""Golden check of the EXISTING calibration paths (ChArUco work, plan step 0).

The ChArUco branch adds a per-view-ids path to lens.py and a board source to the wand
wizard; the checkerboard lens fits and the wand solve must stay bit-identical. This
script computes their outputs on fixed inputs (projected checkerboards for the standard,
fisheye, auto and GoPro-curve fits, one rendered checkerboard video through
`scan_video`, and the synthetic wand rig) and either SAVES them (`--save NAME`) or
COMPARES them with `np.array_equal` to a saved file (`--check NAME`).

  python tests/golden_calib_paths.py --save charuco_golden_main     (on main, before any edit)
  python tests/golden_calib_paths.py --check charuco_golden_main    (after every step)
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _synth_gopro import H12_POLY, H12_ZFOV, H12_ZMPL  # noqa: E402
from kinetrace import gpmf, lens  # noqa: E402

# calibrateCamera's threaded reductions differ by ~1e-8 px between two runs of the SAME code (measured
# 2026-10-24); one thread makes it repeat exactly, so a difference here is a code-path difference
cv2.setNumThreads(1)

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
PATTERN = (9, 6)
SQUARE = 0.024
W, H = 960, 720


def views_std(K, D, n, seed, fisheye=False):
    """Projected checkerboard views (deterministic; no video codec involved)."""
    r = np.random.RandomState(seed)
    obj = lens._object_points(PATTERN, SQUARE)
    out = []
    while len(out) < n:
        rv = np.deg2rad(35.0) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        z = r.uniform(0.45, 0.9)
        tv = np.array([r.uniform(-0.25, 0.25) * z, r.uniform(-0.2, 0.2) * z, z])
        if fisheye:
            p, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D.reshape(-1, 1))
        else:
            p, _ = cv2.projectPoints(obj, rv, tv, K, D)
        p = p.reshape(-1, 2) + r.normal(0, 0.1, (len(obj), 2))
        if (p > 2).all() and (p[:, 0] < W - 2).all() and (p[:, 1] < H - 2).all():
            out.append(p.astype(np.float32))
    return out


def board_video(path):
    """One rendered checkerboard video for `scan_video` (written once, then reused, so the
    main and branch runs read the SAME file)."""
    if os.path.exists(path):
        return
    K = np.array([[820.0, 0, W / 2 + 6], [0, 815.0, H / 2 - 4], [0, 0, 1]])
    D = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
    r = np.random.RandomState(0)
    nx, ny = PATTERN[0] + 1, PATTERN[1] + 1
    SUB = 3
    gx, gy = np.meshgrid(np.arange(-1, nx * SUB + 1) / SUB, np.arange(-1, ny * SUB + 1) / SUB)
    board = np.stack([gx.ravel() * SQUARE, gy.ravel() * SQUARE, np.zeros(gx.size)], axis=1)
    board -= np.array([(nx - 2) / 2.0 * SQUARE, (ny - 2) / 2.0 * SQUARE, 0.0])
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for _ in range(40):
        z = r.uniform(0.45, 0.9)
        u, v = r.uniform(0.08, 0.92) * W, r.uniform(0.08, 0.92) * H
        tvec = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        rvec = np.deg2rad(35.0) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        pts, _ = cv2.projectPoints(board.reshape(-1, 1, 3), rvec, tvec, K, D)
        pts = pts.reshape(ny * SUB + 2, nx * SUB + 2, 2)
        img = np.full((H, W, 3), 140, np.uint8)
        for yy in range(ny * SUB):
            for xx in range(nx * SUB):
                q = np.array([pts[yy + 1, xx + 1], pts[yy + 1, xx + 2], pts[yy + 2, xx + 2], pts[yy + 2, xx + 1]],
                             np.float32)
                if not np.isfinite(q).all() or np.abs(q).max() > 4 * max(W, H):
                    continue
                col = 15 if ((xx // SUB) + (yy // SUB)) % 2 == 0 else 245
                cv2.fillConvexPoly(img, np.round(q * 16).astype(np.int32), (col, col, col), cv2.LINE_AA, 4)
        vw.write(cv2.GaussianBlur(img, (0, 0), 0.6))
    vw.release()


def prof_arrays(tag, prof, res):
    res[f"{tag}_K"] = np.asarray(prof.K, np.float64)
    res[f"{tag}_dist"] = np.asarray(prof.dist, np.float64)
    res[f"{tag}_rms"] = np.array([prof.rms])
    res[f"{tag}_fisheye"] = np.array([bool(prof.fisheye)])
    res[f"{tag}_used"] = np.asarray(prof.report.get("views_used", []), np.int64)


def compute():
    res = {}
    # standard / auto
    K = np.array([[820.0, 0, W / 2 + 6], [0, 815.0, H / 2 - 4], [0, 0, 1]])
    D = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
    vs = views_std(K, D, 40, 1)
    prof_arrays("std", lens.calibrate_lens(vs, PATTERN, SQUARE, (W, H), model="standard"), res)
    prof_arrays("auto", lens.calibrate_lens(vs, PATTERN, SQUARE, (W, H), model="auto"), res)
    res["std_per_view"] = np.asarray(lens.per_view_errors(vs, PATTERN, SQUARE,
                                                          lens.calibrate_lens(vs, PATTERN, SQUARE, (W, H),
                                                                              model="standard")), np.float64)
    idx, err, _why = lens.auto_select(vs, PATTERN, SQUARE, (W, H), "auto", keep=20, base=None)
    res["autosel_idx"] = np.asarray(idx, np.int64)
    res["autosel_err"] = np.asarray(err, np.float64)
    # fisheye
    Kf = np.array([[520.0, 0, W / 2 - 3], [0, 520.0, H / 2 + 2], [0, 0, 1]])
    Df = np.array([0.08, -0.02, 0.005, 0.0])
    vf = views_std(Kf, Df, 40, 2, fisheye=True)
    prof_arrays("fish", lens.calibrate_lens(vf, PATTERN, SQUARE, (W, H), model="fisheye"), res)
    # GoPro curve
    h12 = gpmf.GoProInfo("h12", model="HERO12 Black", width=2704, height=1520, fps=239.76, lens_mode="Wide",
                         poly=list(H12_POLY), zmpl=H12_ZMPL, zfov_deg=H12_ZFOV)
    base = gpmf.lens_profile(h12)
    rng = np.random.default_rng(4)
    Kg = base.K_square().copy()
    Kg[0, 0] = Kg[1, 1] = base.K[0, 0] * 1.012
    Kg[0, 2] += 8.0
    Kg[1, 2] -= 6.0
    Dg = np.asarray(base.dist, float).reshape(4, 1)
    obj = lens._object_points(PATTERN, 25.0)
    vg = []
    while len(vg) < 40:
        rv = rng.normal(0, 0.5, 3)
        tv = np.array([rng.uniform(-200, 200), rng.uniform(-120, 120), rng.uniform(450, 900)])
        pr, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, Kg, Dg)
        pr = pr.reshape(-1, 2) + rng.normal(0, 0.15, (len(obj), 2))
        r = np.hypot(pr[:, 0] - 1351.5, pr[:, 1] - 759.5) / np.hypot(1351.5, 759.5)
        if r.max() < 0.6 and (pr[:, 0] > 0).all():
            vg.append(pr.astype(np.float32))
    prof_arrays("gopro", lens.calibrate_lens(vg, PATTERN, 25.0, (2704, 1520), "gopro",
                                             max_views=lens.MAX_VIEWS, base=base), res)
    # the scan of a real (rendered) video
    vid = os.path.join(OUT, "golden_calib_board.mp4")
    board_video(vid)
    scan = lens.scan_video(vid, PATTERN, max_candidates=40)
    res["scan_frames"] = np.asarray(scan.frames, np.int64)
    res["scan_corners"] = np.asarray(scan.corners, np.float64)
    # the wand solve
    import _synth_wand as sw
    proj, truth = sw.build(error=(0.0, 0.0, 0.0, 0.0), n_ref=500)
    wr = sw.calibrate_fn(proj, max_frames=200)(list(truth["offsets"]))
    res["wand_K"] = np.asarray(wr.K, np.float64)
    res["wand_R"] = np.asarray(wr.R, np.float64)
    res["wand_t"] = np.asarray(wr.t, np.float64)
    res["wand_dist"] = np.asarray(wr.dist, np.float64)
    return res


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("--save", "--check"):
        print(__doc__)
        sys.exit(2)
    path = os.path.join(OUT, sys.argv[2] + ".npz")
    res = compute()
    if sys.argv[1] == "--save":
        np.savez(path, **res)
        print(f"saved {len(res)} arrays to {path}")
        return
    ref = np.load(path)
    bad = [k for k in ref.files if k not in res or not np.array_equal(ref[k], res[k])]
    bad += [k for k in res if k not in ref.files]
    for k in bad:
        print(f"DIFFERENT: {k}")
    if bad:
        print(f"FAILED: {len(bad)} of {len(ref.files)} arrays differ")
        sys.exit(1)
    print(f"PASSED: all {len(ref.files)} arrays identical (np.array_equal)")


if __name__ == "__main__":
    main()
