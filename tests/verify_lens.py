"""Lens (intrinsic) calibration core on synthetic ground truth (no GPU).

A 9x6-corner checkerboard is rendered through a known camera (K + radial
distortion, then a fisheye one) at many poses into a small video; the scan
must find the boards, `calibrate_lens` must recover K within 0.5 %, the
undistortion must straighten the board rows to < 0.3 px, and the report must
say GOOD. Also: the auto model choice, a poor-coverage case that must NOT be
called good, the printable checkerboard (with print resolution), the lens
file round trip, the Argus profile importer, and the square-K undistortion
frame the wand core consumes.
"""
import os
import struct
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from kinetrace import lens
from kinetrace.calib import OpenCVUndistort

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.RandomState(3)

W, H = 960, 720
PATTERN = (9, 6)
SQUARE = 0.024                     # metres


def render_video(path, K, dist, fisheye, n_frames, spread=1.0, tilt_deg=35.0, seed=0):
    """Board poses: translated over the picture, tilted, at two distances."""
    r = np.random.RandomState(seed)
    cols, rows = PATTERN
    # board corners INCLUDING the outer squares so the squares can be drawn
    nx, ny = cols + 1, rows + 1
    SUB = 3                                       # sub-squares per square: curved edges render faithfully
    gx, gy = np.meshgrid(np.arange(-1, nx * SUB + 1) / SUB, np.arange(-1, ny * SUB + 1) / SUB)
    board = np.stack([gx.ravel() * SQUARE, gy.ravel() * SQUARE, np.zeros(gx.size)], axis=1)
    centre = np.array([(nx - 2) / 2.0 * SQUARE, (ny - 2) / 2.0 * SQUARE, 0.0])
    board -= centre
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    truth_frames = 0
    for i in range(n_frames):
        z = r.uniform(0.45, 0.9)
        # place the board centre so that it lands across the picture
        u = r.uniform(0.5 - 0.42 * spread, 0.5 + 0.42 * spread) * W
        v = r.uniform(0.5 - 0.42 * spread, 0.5 + 0.42 * spread) * H
        xc = (u - K[0, 2]) / K[0, 0] * z
        yc = (v - K[1, 2]) / K[1, 1] * z
        rvec = np.deg2rad(tilt_deg) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        tvec = np.array([xc, yc, z])
        obj = board.reshape(-1, 1, 3)
        if fisheye:
            pts, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, dist.reshape(-1, 1)[:4])
        else:
            pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pts = pts.reshape(ny * SUB + 2, nx * SUB + 2, 2)
        img = np.full((H, W, 3), 140, np.uint8)
        noise = r.normal(0, 3, (H, W, 1)).astype(np.int16)
        for yy in range(ny * SUB):
            for xx in range(nx * SUB):
                quad = np.array([pts[yy + 1, xx + 1], pts[yy + 1, xx + 2], pts[yy + 2, xx + 2], pts[yy + 2, xx + 1]],
                                np.float32)
                if not np.isfinite(quad).all() or np.abs(quad).max() > 4 * max(W, H):
                    continue
                col = 15 if ((xx // SUB) + (yy // SUB)) % 2 == 0 else 245
                cv2.fillConvexPoly(img, np.round(quad * 16).astype(np.int32), (col, col, col),
                                   cv2.LINE_AA, 4)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        vw.write(img)
        truth_frames += 1
    vw.release()
    return truth_frames


def row_residuals(pts_list, prof=None):
    """Residual (px) of every board row to a straight line, raw or after
    undistortion with `prof`: one value per row over all boards."""
    cols, rows = PATTERN
    out = []
    for corners in pts_list:
        p = (OpenCVUndistort(prof.K, prof.dist, prof.fisheye).undistort(corners)
             if prof is not None else corners)
        for r_ in range(rows):
            row = p[r_ * cols:(r_ + 1) * cols]
            A = np.column_stack([row[:, 0], np.ones(cols)])
            coef, *_ = np.linalg.lstsq(A, row[:, 1], rcond=None)
            out.append(float(np.abs(row[:, 1] - A @ coef).max()))
    return np.asarray(out)


def straightness(prof, corners):
    return float(row_residuals([corners], prof).max())


# ---- standard lens with strong radial distortion -----------------------------
K_true = np.array([[820.0, 0, W / 2 + 6], [0, 815.0, H / 2 - 4], [0, 0, 1]])
D_true = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
vid = os.path.join(SCRATCH, "lens_std.mp4")
render_video(vid, K_true, D_true, False, 70)
scan = lens.scan_video(vid, PATTERN, max_candidates=70)
print(f"standard: {len(scan.frames)} boards in {scan.n_scanned} frames scanned")
assert len(scan.frames) >= 50, len(scan.frames)
prof = lens.calibrate_lens(scan.corners, PATTERN, SQUARE, scan.size, model="auto")
rep = prof.report
print(f"  model {rep['model']}, rms {prof.rms:.3f} px, f {prof.K[0,0]:.1f}/{prof.K[1,1]:.1f} "
      f"(true 820/815), pp ({prof.K[0,2]:.1f}, {prof.K[1,2]:.1f}), k1 {prof.dist[0]:.4f} k2 {prof.dist[1]:.4f}, "
      f"coverage {rep['coverage_pct']:.0f}%, reach {rep['edge_reach_pct']:.0f}%, bend {rep['distortion_border_px']:.1f} px, "
      f"verdict {rep['verdict']}")
assert not prof.fisheye, "standard lens must be recognised as such"
assert abs(prof.K[0, 0] - 820) / 820 < 0.005 and abs(prof.K[1, 1] - 815) / 815 < 0.005
assert abs(prof.K[0, 2] - K_true[0, 2]) < 8 and abs(prof.K[1, 2] - K_true[1, 2]) < 8   # the principal point is the loosest parameter
assert abs(prof.dist[0] - D_true[0]) < 0.02, prof.dist
assert prof.rms < 0.5
assert rep["verdict"] == "good", rep["verdict_reasons"]
raw = float(np.percentile(row_residuals(scan.corners), 95))
s = float(np.percentile(row_residuals(scan.corners, prof), 95))
print(f"  board rows (95th pct): {raw:.2f} px bent raw -> {s:.2f} px after undistortion")
assert s < 0.6 and s < 0.5 * raw, (raw, s)
# the square frame for the wand core: undistort maps into K_square, and distort inverts it
m = prof.undistort_model()
p = np.array([[100.0, 80.0], [W - 50.0, H - 60.0], [W / 2, H / 2]])
back = m.distort(m.undistort(p))
assert np.allclose(back, p, atol=0.05), back - p
assert np.allclose(m.P[0, 0], m.P[1, 1]), "square camera matrix"
print("standard lens OK")

# ---- fisheye lens -------------------------------------------------------------
Kf = np.array([[520.0, 0, W / 2 - 3], [0, 520.0, H / 2 + 2], [0, 0, 1]])
Df = np.array([0.08, -0.02, 0.005, 0.0])
vidf = os.path.join(SCRATCH, "lens_fish.mp4")
render_video(vidf, Kf, Df, True, 90, spread=1.7, seed=1)
scanf = lens.scan_video(vidf, PATTERN, max_candidates=70)
proff = lens.calibrate_lens(scanf.corners, PATTERN, SQUARE, scanf.size, model="fisheye")
print(f"fisheye: {len(scanf.frames)} boards, rms {proff.rms:.3f} px, f {proff.K[0,0]:.1f} (true 520), "
      f"k {np.round(proff.dist, 4)}, verdict {proff.report['verdict']}, coverage {proff.report['coverage_pct']:.0f}%, "
      f"reach {proff.report['edge_reach_pct']:.0f}%")
for r_ in proff.report["verdict_reasons"]:
    print("   -", r_)
assert proff.fisheye and abs(proff.K[0, 0] - 520) / 520 < 0.01 and proff.rms < 0.5
assert proff.report["verdict"] in ("good", "ok"), proff.report["verdict_reasons"]
rawf = float(np.percentile(row_residuals(scanf.corners), 95))
sf = float(np.percentile(row_residuals(scanf.corners, proff), 95))
print(f"  board rows (95th pct): {rawf:.2f} px bent raw -> {sf:.2f} px after undistortion")
assert sf < 0.5 * rawf and sf < 1.0, (rawf, sf)
auto = lens.calibrate_lens(scanf.corners, PATTERN, SQUARE, scanf.size, model="auto")
print(f"  auto choice on the fisheye video: {auto.report['model']} (rms by model {auto.report['rms_by_model']})")
print("fisheye lens OK")

# ---- poor coverage must not be called good --------------------------------------
vidp = os.path.join(SCRATCH, "lens_poor.mp4")
render_video(vidp, K_true, D_true, False, 30, spread=0.25, tilt_deg=6.0, seed=2)
scanp = lens.scan_video(vidp, PATTERN, max_candidates=30)
profp = lens.calibrate_lens(scanp.corners, PATTERN, SQUARE, scanp.size, model="standard")
print(f"poor case: verdict {profp.report['verdict']}; reasons: {profp.report['verdict_reasons'][:3]}")
assert profp.report["verdict"] != "good"
assert any("edge" in r or "middle" in r or "corner" in r for r in profp.report["verdict_reasons"])
print("poor coverage flagged OK")

# ---- printable board, files, Argus import ----------------------------------------
png = os.path.join(SCRATCH, "checkerboard.png")
wmm, hmm = lens.save_checkerboard_png(png)
data = open(png, "rb").read()
assert b"pHYs" in data[:80]
i = data.index(b"pHYs")
ppm = struct.unpack(">I", data[i + 4:i + 8])[0]
assert abs(ppm / 39.3701 - 300) < 1, ppm
assert wmm < 297 and hmm < 210, (wmm, hmm)          # fits A4 / Letter landscape
img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
c = lens.detect_board(cv2.resize(img, None, fx=0.4, fy=0.4), PATTERN)
assert c is not None and len(c) == 54, "the printed board must itself be detectable"
lp = os.path.join(SCRATCH, "lens_std.klens.json")
saved = prof.save(lp)
back = lens.LensProfile.load(saved)
assert np.allclose(back.K, prof.K) and np.allclose(back.dist, prof.dist) and back.report["verdict"] == "good"
argus = os.path.join(SCRATCH, "argus_profile.txt")
open(argus, "w").write("1 485 1920 1080 960 540 1 0 0 0 0 0\n2 1400.5 1920 1080 962.2 541.1 1 -0.2 0.05 0 0 0\n")
profs = lens.load_argus_profile(argus)
# (I75) Argus profiles are OpenCV's 0-based values: the principal point is taken as written
assert len(profs) == 2 and profs[1].K[0, 0] == 1400.5 and abs(profs[1].K[0, 2] - 962.2) < 1e-9
assert abs(profs[1].K[1, 2] - 541.1) < 1e-9
assert profs[1].dist[0] == -0.2 and profs[0].width == 1920
print(f"checkerboard {wmm:.0f} x {hmm:.0f} mm at 300 dpi, lens file + Argus import OK")

# ---- Argus convention (I75): the database's own 2.7K line, undistorted as Argus does it ----
# argus_gui builds K = [[f, 0, cx], [0, f, cy], [0, 0, 1]] from the columns and hands it to
# cv2.undistortPoints / initUndistortRectifyMap on the raw 0-based picture (undistort.py
# calibParse, tools.undistort_pts); its database centres are (w - 1) / 2.
argus_db = os.path.join(SCRATCH, "argus_db_line.txt")
open(argus_db, "w").write("1 1788 2704 1520 1351.5 759.5 1 -0.2583 0.077 0 0 0\n")
pa = lens.load_argus_profile(argus_db)[0]
assert pa.principal == (1351.5, 759.5), f"the principal point must be used as written: {pa.principal}"
K_argus = np.array([[1788.0, 0, 1351.5], [0, 1788.0, 759.5], [0, 0, 1.0]])
d_argus = np.array([-0.2583, 0.077, 0, 0, 0])
# central pixels, so the comparison does not depend on how many solver iterations either side runs
test_px = np.array([[1500.0, 850.0], [1200.0, 650.0], [1351.5, 759.5], [1700.0, 1000.0]])
ours = OpenCVUndistort(pa.K, pa.dist).undistort(test_px)
theirs = cv2.undistortPoints(test_px.reshape(-1, 1, 2), K_argus, d_argus, None, K_argus,
                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 1e-10)).reshape(-1, 2)
assert np.abs(ours - theirs).max() < 1e-4, f"Argus undistortion reproduced: {np.abs(ours - theirs).max()}"
print(f"Argus 2.7K database line: principal (1351.5, 759.5) as written, undistortion matches Argus's "
      f"to {np.abs(ours - theirs).max():.1e} px OK")

# ---- Argus lines belong to cameras by the file's camera column (I80) ----------------
two = os.path.join(SCRATCH, "argus_two.txt")
open(two, "w").write("1 485 1920 1080 959.5 539.5 1 0 0 0 0 0\n2 1400.5 1920 1080 962.2 541.1 1 -0.2 0.05 0 0 0\n")
got, why = lens.argus_profile_for(two, 1, "cam2")
assert got is not None and got.f_square == 1400.5 and "camera 2" in why, why
got, why = lens.argus_profile_for(two, 2, "cam3")
assert got is None and "camera 3" in why and "cam3" in why and "1, 2" in why, \
    f"a camera past the file's lines is refused, not given the last line: {why}"
swapped = os.path.join(SCRATCH, "argus_swapped.txt")
open(swapped, "w").write("2 1400.5 1920 1080 962.2 541.1 1 -0.2 0.05 0 0 0\n1 485 1920 1080 959.5 539.5 1 0 0 0 0 0\n")
got, _ = lens.argus_profile_for(swapped, 0, "cam1")
assert got is not None and got.f_square == 485, "lines listed 2, 1: camera 1 still gets the line numbered 1"
one = os.path.join(SCRATCH, "argus_one.txt")
open(one, "w").write("1 1788 2704 1520 1351.5 759.5 1 -0.2583 0.077 0 0 0\n")
got, why = lens.argus_profile_for(one, 4, "cam5")
assert got is not None and got.f_square == 1788 and "only line" in why, "a one-line file is one lens for any camera"
rep_nums = os.path.join(SCRATCH, "argus_repeat.txt")
open(rep_nums, "w").write("1 485 1920 1080 959.5 539.5 1 0 0 0 0 0\n1 900 1920 1080 959.5 539.5 1 0 0 0 0 0\n")
got, why = lens.argus_profile_for(rep_nums, 1, "cam2")
assert got is not None and got.f_square == 900 and "line order" in why, why
assert lens.argus_profile_for(rep_nums, 2, "cam3")[0] is None
assert "camera 2" in lens.load_argus_profile(two)[1].source, "the source names the line used"
# a profile for another picture size is named as such (lens#2 / I31)
assert lens.size_mismatch((1920, 1080), (1920, 1080), "cam1") is None
msg = lens.size_mismatch((1280, 960), (640, 480), "cam1")
assert msg and "1280 x 960" in msg and "640 x 480" in msg and "cam1" in msg, msg
print("Argus lines matched by camera number (refused when missing), picture-size check OK")

# ---- one corner 0, however the board is turned and whichever detector found it -------
# On a real hand-held GoPro clip the review board ringed frame 3471 at the
# bottom-right corner of the board and frame 3578 at the top-left: SB orders
# corners from the picture's top-left, the classic fallback by square colour.
# Canonical rule: corner 0 is the inner corner whose first inner square is black.
src = cv2.resize(lens.checkerboard_image(9, 6, 100, 100), None, fx=0.5, fy=0.5,
                 interpolation=cv2.INTER_AREA)                # 600 x 450, squares of 50 px
sh, sw = src.shape
p0 = np.array([100.0, 100.0])                                 # physical corner 0 (square (1,1) is black)
p_row = p0 + np.array([8 * 50.0, 0.0])                        # last corner of the first row
p_col = p0 + np.array([0.0, 50.0])                            # first corner of the second row
CAN = 820
n_pairs = n_disagree = 0
for k, ang in enumerate((0, 33, 90, 150, 180, 214, 270, 305)):
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    T0 = np.array([[1, 0, -sw / 2], [0, 1, -sh / 2], [0, 0, 1.0]])
    T1 = np.array([[1, 0, CAN / 2], [0, 1, CAN / 2], [0, 0, 1.0]])
    P = np.eye(3)
    P[2, 0], P[2, 1] = 0.00035 * np.cos(1.3 * k), 0.00030 * np.sin(0.7 * k + 1)   # a tilt
    Hm = T1 @ P @ R @ T0
    warped = cv2.warpPerspective(src, Hm, (CAN, CAN), borderValue=255)
    def _h(p):
        q = Hm @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]
    info = lens.detect_board_info(warped, PATTERN)
    assert info is not None, f"board not found at {ang} deg"
    c_can, how = info
    assert how == "colour", (ang, how)
    for got, want, what in ((c_can[0], _h(p0), "corner 0"), (c_can[8], _h(p_row), "end of row 0"),
                            (c_can[9], _h(p_col), "start of row 1")):
        assert np.linalg.norm(got - want) < 1.5, f"{what} at {ang} deg: {got} vs {want}"
    # the two raw detectors, each through orient_board, must agree with each other
    raws = []
    ok_sb, c_sb = cv2.findChessboardCornersSB(warped, PATTERN, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    ok_cl, c_cl = cv2.findChessboardCorners(warped, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok_sb and ok_cl:
        r_sb, r_cl = c_sb.reshape(-1, 2).astype(float), c_cl.reshape(-1, 2).astype(float)
        n_pairs += 1
        if np.linalg.norm(r_sb[0] - r_cl[0]) > 5:
            n_disagree += 1
        o_sb, _ = lens.orient_board(warped, r_sb, PATTERN)
        o_cl, _ = lens.orient_board(warped, r_cl, PATTERN)
        assert np.abs(o_sb - o_cl).max() < 1.5, f"detectors disagree after orient_board at {ang} deg"
assert n_pairs >= 3, n_pairs
assert n_disagree >= 1, "the raw detectors were expected to disagree somewhere (the bug being pinned)"
print(f"corner 0 is the same physical corner at 8 rotations + tilts; raw detectors disagreed on "
      f"{n_disagree} of {n_pairs} boards and agree after orient_board OK")
# a symmetric board (8 x 6 inner corners: 9 x 7 squares) cannot be oriented by colour
sym = cv2.resize(lens.checkerboard_image(8, 6, 100, 100), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
assert not lens.board_is_asymmetric((8, 6)) and lens.board_is_asymmetric((9, 6))
info_a = lens.detect_board_info(sym, (8, 6))
info_b = lens.detect_board_info(cv2.rotate(sym, cv2.ROTATE_180), (8, 6))
assert info_a is not None and info_b is not None
assert info_a[1] == "image" and info_b[1] == "image"
assert np.linalg.norm(info_a[0][0] - info_b[0][0]) < 1.5, "symmetric board: corner 0 = the picture's top-left both times"
assert lens.orientation_note((9, 6)) == "" and "upside down" in lens.orientation_note((8, 6))
assert "all 5" in lens.orientation_summary(["colour"] * 5, (9, 6))
assert "1 of 2" in lens.orientation_summary(["colour", "image"], (9, 6))
assert "180" in lens.orientation_summary(["image"] * 3, (8, 6))
print("symmetric board falls back to the picture's top-left and says so OK")

# ---- a model that RUNS AWAY in the corners must say so, not print a sentinel -------
# A real GoPro clip (2704 x 1520, board reaching 69 % of the way to the corners),
# fitted with the fisheye model, reported "the edges are bent by up to 1417199 px":
# OpenCV's fisheye inverse does not converge at the corners and returns
# (-1000000, -1000000). These are that fit's exact numbers.
Kh = np.array([[1316.4, 0, 1357.73], [0, 1315.9, 769.30], [0, 0, 1.0]])
runaway = lens.LensProfile(2704, 1520, Kh, np.array([0.07591, -0.12884, 0.30894, -0.21558]), True, 0.90, 36)
chk = runaway.border_check()
assert chk["runaway"] and chk["valid_frac"] < 1.0, chk
assert np.isfinite(chk["bend_px"]) and 100 < chk["bend_px"] < 3000, chk       # over the valid part only
assert not np.isfinite(chk["fov_diag_deg"]), "no field of view can be quoted when a corner is lost"
assert "RUNS AWAY" in runaway.summary() and "1417" not in runaway.summary(), runaway.summary()
# the same boards under the standard model hold everywhere and imply a GoPro-like field of view
Ks = np.array([[1316.3, 0, 1360.24], [0, 1315.6, 768.81], [0, 0, 1.0]])
sane = lens.LensProfile(2704, 1520, Ks, np.array([-0.2595, 0.07982, 0, 0, 0]), False, 0.91, 36)
chk_s = sane.border_check()
assert not chk_s["runaway"] and 300 < chk_s["bend_px"] < 450, chk_s
assert 100 < chk_s["fov_diag_deg"] < 125, chk_s
# the report: never "good", "poor" when most of the border is lost, and it says what to do
rep = lens.lens_report(runaway, [np.zeros((54, 2))] * 20, [0.9] * 20, [25.0] * 20,
                       {"standard": 0.91, "fisheye": 0.90}, {}, "fisheye")
assert rep["verdict"] != "good" and rep["border_runaway"] and rep["border_valid_frac"] < 1
assert any("RUNS AWAY" in r and "standard model" in r for r in rep["verdict_reasons"]), rep["verdict_reasons"]
rep_s = lens.lens_report(sane, [np.zeros((54, 2))] * 20, [0.9] * 20, [25.0] * 20, {"standard": 0.91}, {}, "standard")
assert not rep_s["border_runaway"] and any("field of view" in r for r in rep_s["verdict_reasons"])
# the synthetic fits above hold along the whole border
for _p in (prof, proff):
    _c = _p.border_check()
    assert not _c["runaway"] and np.isfinite(_c["fov_diag_deg"]), _c
print(f"runaway fisheye model flagged (border {100 * chk['valid_frac']:.0f}% valid, bend >= {chk['bend_px']:.0f} px, "
      f"verdict {rep['verdict']}); sane model implies {chk_s['fov_diag_deg']:.0f} deg corner to corner OK")
# ---- the review board's core: thumbnails, per-view error, axes, auto-select ----
scan_r = lens.scan_video(vid, PATTERN)
assert len(scan_r.thumbs) == len(scan_r.corners), "one thumbnail per board found"
assert scan_r.thumbs[0].shape[1] <= lens.THUMB_MAX_W, "thumbnails are small"
assert scan_r.video == vid, "the scan remembers its video, for the corner editor"
assert len(scan_r.orient) == len(scan_r.corners), "one orientation tag per board"
assert all(o == "colour" for o in scan_r.orient), "a 9 x 6 board is oriented by colour on every frame"
_sc = scan_r.thumb_scale(0)
assert 0 < _sc <= 1.0 and abs(_sc - scan_r.thumbs[0].shape[1] / scan_r.size[0]) < 1e-9

_idx, _err, _why = lens.auto_select(scan_r.corners, PATTERN, SQUARE, scan_r.size, "standard")
assert 3 <= len(_idx) <= len(scan_r.corners), f"auto_select chose {len(_idx)}"
assert np.isfinite(_err).sum() == len(scan_r.corners), "every view is scored, not just the chosen"
assert len(_why) > 30 and "boards" in _why, "and it says in words what it did"

_prof = lens.calibrate_lens([scan_r.corners[i] for i in _idx], PATTERN, SQUARE,
                            scan_r.size, "standard")
_pv = lens.per_view_errors(scan_r.corners, PATTERN, SQUARE, _prof)
assert np.nanmedian(_pv) < 0.5, f"clean synthetic boards reproject well: {np.nanmedian(_pv):.3f} px"

_pose = lens.view_pose(scan_r.corners[0], PATTERN, SQUARE, _prof)
assert _pose is not None and 0.05 < abs(float(np.ravel(_pose[1])[2])) < 50, "a sane board distance"

# a corner knocked out of place must show up as a much worse view -- that is
# the signal the review board ranks by and the user then fixes by dragging
_bad = scan_r.corners[0].copy()
_bad[5] += [40.0, -25.0]
_e0 = lens.per_view_errors([scan_r.corners[0]], PATTERN, SQUARE, _prof)[0]
_e1 = lens.per_view_errors([_bad], PATTERN, SQUARE, _prof)[0]
assert _e1 > 10 * max(_e0, 0.02) and _e1 > 1.0, f"a bad corner must stand out: {_e0:.3f} -> {_e1:.3f}"

_marked = lens.draw_board_review(scan_r.thumbs[0], scan_r.corners[0], PATTERN, SQUARE,
                                 _prof, scale=scan_r.thumb_scale(0))
assert _marked.shape == scan_r.thumbs[0].shape and not np.array_equal(_marked, scan_r.thumbs[0])
_plain = lens.draw_board_review(scan_r.thumbs[0], scan_r.corners[0], PATTERN, SQUARE,
                                None, scale=scan_r.thumb_scale(0), axes=False)
assert not np.array_equal(_marked, _plain), "the axes are drawn only when a lens is known"
print(f"review board core OK ({len(scan_r.corners)} boards, {len(_idx)} auto-chosen, "
      f"median {np.nanmedian(_pv):.3f} px, bad corner {_e0:.2f} -> {_e1:.2f} px)")

# ---- the report describes the views the final fit used (I77) -----------------------
# Three of twenty views blurred into nonsense: the robust pass sets them aside, and the
# report's view count, coverage and reach must be about the seventeen it kept.
_twenty = [c.copy() for c in scan_r.corners[:20]]
_bad_idx = (3, 9, 15)
for _j in _bad_idx:
    _twenty[_j] = _twenty[_j] + np.random.RandomState(_j).normal(0, 6.0, _twenty[_j].shape)
_p20 = lens.calibrate_lens(_twenty, PATTERN, SQUARE, scan_r.size, "standard")
_r20 = _p20.report
assert _r20.get("views_set_aside") == 3, _r20.get("views_set_aside")
assert _r20["n_views"] == _p20.n_views == 17, (_r20["n_views"], _p20.n_views)
assert sorted(_r20["views_used"]) == [i for i in range(20) if i not in _bad_idx], _r20["views_used"]
_kept = [_twenty[i] for i in _r20["views_used"]]
assert abs(_r20["coverage_pct"] - lens.coverage_pct(_kept, scan_r.size)) < 1e-9
assert abs(_r20["edge_reach_pct"] - lens.edge_reach_pct(_kept, scan_r.size)) < 1e-9
assert any(r.startswith("17 views") for r in _r20["verdict_reasons"]), _r20["verdict_reasons"]
print(f"report counts the {_p20.n_views} fitted views, not the 20 given (3 set aside) OK")

print("verify_lens PASSED")
