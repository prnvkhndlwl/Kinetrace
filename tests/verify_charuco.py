"""ChArUco calibration core on synthetic ground truth (no GPU; I321, I322, I323).

[1] the board: spec checks, the printable page (print resolution and size), detection on
    boards rendered through a known lens (every corner near its projected truth, the right
    id, a board half out of the picture still read), the older OpenCV layout recognised,
    Argus's frame rules (quality, best frames, long-edge pairs, wand length).
"""
import os
import struct
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import _synth_charuco as sc  # noqa: E402
from kinetrace import charuco  # noqa: E402

SCRATCH = os.path.join(HERE, "out")
os.makedirs(SCRATCH, exist_ok=True)
FAILS = []


def check(ok, what, detail=None):
    print(("ok    " if ok else "FAIL  ") + what + ("" if ok or detail is None else f"  -- {detail}"))
    if not ok:
        FAILS.append(what)


W, H = 960, 720
K_STD = np.array([[820.0, 0, W / 2 + 6], [0, 815.0, H / 2 - 4], [0, 0, 1]])
D_STD = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
SPEC = charuco.BoardSpec()

# ------------------------------------------------------------------ [1] the board
print("[1] the board")
check(SPEC.pattern == (6, 4) and SPEC.n_corners == 24, "7 x 5 squares = 6 x 4 inner corners")
check(SPEC.problem() == "", "the default board is valid", SPEC.problem())
check("smaller" in charuco.BoardSpec(marker_m=0.04).problem(), "a marker bigger than its square is refused")
check("fewer" in charuco.BoardSpec(12, 10, dictionary="DICT_4X4_50").problem(),
      "a board needing more markers than its dictionary has is refused")
check("6 inner" in charuco.BoardSpec(3, 3).problem() and charuco.BoardSpec(4, 3).problem() == "",
      "a 3 x 3 board (4 corners: no lens pose) is refused; 4 x 3 is the smallest")
check(charuco.BoardSpec.from_json(SPEC.to_json()) == SPEC, "the board spec survives a JSON round trip")

png = os.path.join(SCRATCH, "charuco_print.png")
wmm, hmm = charuco.save_board_png(png, SPEC, dpi=300)
raw = open(png, "rb").read()
i = raw.find(b"pHYs")
ppm = struct.unpack(">I", raw[i + 4:i + 8])[0] if i > 0 else 0
img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
d = charuco.detect(img, SPEC)
check(abs(ppm - 300 / 0.0254) < 2, "the printable board carries its print resolution (300 dpi)", ppm)
check(abs(wmm - (210 + 24)) < 1.0 and abs(hmm - (150 + 24)) < 1.0,
      "printed at 100 %: 210 x 150 mm of squares + 12 mm margins", (wmm, hmm))
check(d is not None and len(d[1]) == 24, "every corner is found on the printable page",
      None if d is None else len(d[1]))
uni = os.path.join(SCRATCH, "charuco_print_José.png")
charuco.save_board_png(uni, SPEC, dpi=150)
check(os.path.exists(uni), "I222: a path with a non-ASCII character is written")

# rendered through a strongly bending lens, many poses
r = np.random.RandomState(1)
errs, ids_ok, n_found = [], True, 0
for k in range(16):
    rv, tv = sc.random_pose(K_STD, (W, H), r)
    pic = sc.render(SPEC, K_STD, D_STD, False, rv, tv, (W, H), seed=k)
    det = charuco.detect(cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY), SPEC)
    tr = sc.truth(SPEC, K_STD, D_STD, False, rv, tv)
    if det is None:
        continue
    n_found += 1
    e = np.linalg.norm(det[0] - tr[det[1]], axis=1)
    ids_ok &= bool(e.max() < 1.0)                    # a wrong id would land a square (tens of px) away
    errs += e.tolist()
errs = np.asarray(errs)
print(f"      {n_found} of 16 boards found, {len(errs)} corners: mean {errs.mean():.3f} px, "
      f"95th pct {np.percentile(errs, 95):.3f} px, worst {errs.max():.3f} px")
check(n_found >= 13, "the board is found in most pictures", n_found)
check(ids_ok, "every corner carries its own id (none a square away from its truth)")
check(errs.mean() <= 0.10 and np.percentile(errs, 95) <= 0.25 and errs.max() <= 0.7,
      "sub-pixel corners: mean <= 0.10 px, 95th pct <= 0.25 px, worst <= 0.7 px",
      (errs.mean(), np.percentile(errs, 95), errs.max()))

# half out of the picture: what is inside is still read, by id
rv = np.array([0.15, -0.2, 0.05])
tv = np.array([-0.09 + (W - 1 - K_STD[0, 2]) / K_STD[0, 0] * 0.5, -0.06, 0.5])     # board centre at the right edge
pic = sc.render(SPEC, K_STD, D_STD, False, rv, tv, (W, H), seed=99)
det = charuco.detect(cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY), SPEC)
tr = sc.truth(SPEC, K_STD, D_STD, False, rv, tv)
inside = (tr[:, 0] > 5) & (tr[:, 0] < W - 5)
check(det is not None and 6 <= len(det[1]) < 24 and np.linalg.norm(det[0] - tr[det[1]], axis=1).max() < 0.7,
      "a board half out of the picture: the corners inside are read with their ids",
      None if det is None else (len(det[1]), int(inside.sum())))
view = charuco.padded(det[0], det[1], SPEC) if det is not None else np.full((24, 2), np.nan)
check(view.shape == (24, 2) and np.isnan(view).any() and charuco.usable_for_lens(view, SPEC),
      "padded view: the whole grid, NaN where unseen, usable for a lens fit")
row_only = np.full((24, 2), np.nan)
row_only[:6] = np.arange(12, dtype=float).reshape(6, 2)
check(not charuco.usable_for_lens(row_only, SPEC), "one row of corners alone has no pose: not a lens board")
check(np.array_equal(charuco.object_points(SPEC, [0, 5, 6, 23]),
                     np.array([[0, 0, 0], [0.15, 0, 0], [0, 0.03, 0], [0.15, 0.09, 0]])),
      "object points: corner 0 at the origin, row-major like lens._object_points")

# the older OpenCV layout (an even number of rows), typed as the current one
spec6 = charuco.BoardSpec(7, 6)
legacy6 = charuco.BoardSpec(7, 6, legacy=True)
pics = []
for k in range(3):
    rv, tv = sc.random_pose(K_STD, (W, H), r, z=(0.5, 0.7))
    pics.append(cv2.cvtColor(sc.render(legacy6, K_STD, D_STD, False, rv, tv, (W, H), seed=200 + k),
                             cv2.COLOR_BGR2GRAY))
got, note = charuco.guess_legacy(pics, spec6)
check(got.legacy and "OLDER" in note, "a board printed with the older layout is recognised and said", note)
got2, note2 = charuco.guess_legacy(pics, legacy6)
check(got2.legacy and note2 == "", "the right layout typed: nothing to say")

# Argus's frame rules
a = np.full((24, 2), np.nan)
a[:12] = np.column_stack([np.linspace(100, 200, 12), np.linspace(100, 110, 12)])
b = a.copy()
b[:12, 0] = np.linspace(100, 800, 12)
check(abs(charuco.quality(a, (W, H)) - (12 + 10 * (a[:12, 0].std() / W + a[:12, 1].std() / H) / 2)) < 1e-12,
      "R3: quality = n + 10 x mean normalised spread")
check(charuco.quality(b, (W, H)) > charuco.quality(a, (W, H)), "R3: the same corners spread wider rank higher")
vs = [a] * 5 + [b] + [a] * 5
check(charuco.best_frames(vs, (W, H), keep=3)[:1] == [0] and 5 in charuco.best_frames(vs, (W, H), keep=3)
      and charuco.best_frames(vs, (W, H), keep=3) == sorted(charuco.best_frames(vs, (W, H), keep=3)),
      "R3: the best frames are kept, back in time order")
check(charuco.long_edge_pairs(SPEC) == [(0, 5), (18, 23)], "R4: a wide board pairs the ends of its top / bottom rows")
check(charuco.long_edge_pairs(charuco.BoardSpec(5, 8)) == [(0, 24), (3, 27)],
      "R4: a tall board pairs the ends of its left / right columns")
check(abs(charuco.wand_length_m(SPEC) - 0.15) < 1e-12, "R4: the pair is 5 squares = 150 mm apart")

# ------------------------------------------------------------------ [2] the lens from a ChArUco video
print("[2] the lens from a ChArUco video")
from kinetrace import gpmf, lens  # noqa: E402
from _synth_gopro import H12_POLY, H12_ZFOV, H12_ZMPL  # noqa: E402


def board_video(path, K, D, fisheye, n, seed, spread=1.1, z=(0.28, 0.6)):
    if os.path.exists(path):
        return
    rr = np.random.RandomState(seed)
    pics = []
    for k in range(n):
        rv_, tv_ = sc.random_pose(K, (W, H), rr, spread=spread, z=z)
        pics.append(sc.render(SPEC, K, D, fisheye, rv_, tv_, (W, H), seed=seed * 1000 + k))
    sc.write_video(path, pics)


vid = os.path.join(SCRATCH, "charuco_lens_std.mp4")
board_video(vid, K_STD, D_STD, False, 70, 5)
scan = lens.scan_video(vid, SPEC.pattern, max_candidates=70, board=SPEC)
partial = sum(1 for v in scan.corners if np.isnan(v).any())
print(f"      {len(scan.frames)} boards in {scan.n_scanned} frames, {partial} of them partly seen")
check(len(scan.frames) >= 45 and scan.board == SPEC and set(scan.orient) == {"id"},
      "the scan finds the ChArUco board (corners named by id)", len(scan.frames))
check(partial >= 5, "boards partly out of the picture are kept", partial)
prof = lens.calibrate_lens(scan.corners, SPEC.pattern, SPEC.square_m, scan.size, model="auto")
rep = prof.report
print(f"      model {rep['model']}, rms {prof.rms:.3f} px, f {prof.K[0, 0]:.1f}/{prof.K[1, 1]:.1f} (true 820/815), "
      f"pp ({prof.K[0, 2]:.1f}, {prof.K[1, 2]:.1f}) true ({K_STD[0, 2]:.1f}, {K_STD[1, 2]:.1f}), k1 {prof.dist[0]:.4f}, "
      f"coverage {rep['coverage_pct']:.0f}%, reach {rep['edge_reach_pct']:.0f}%, verdict {rep['verdict']}")
check(not prof.fisheye, "a standard lens is recognised as such")
check(abs(prof.K[0, 0] - 820) / 820 < 0.005 and abs(prof.K[1, 1] - 815) / 815 < 0.005,
      "focal length within 0.5 %", prof.K)
check(abs(prof.K[0, 2] - K_STD[0, 2]) < 2.0 and abs(prof.K[1, 2] - K_STD[1, 2]) < 2.0,
      "centre within 2 px (the centre is fitted, not pinned as in Argus)", prof.K)
check(abs(prof.dist[0] - D_STD[0]) < 0.02, "k1 recovered", prof.dist)
check(prof.rms < 0.3, "fit error under 0.3 px", prof.rms)
check(rep["verdict"] in ("good", "ok"), "the report gives a verdict a user can act on", rep["verdict_reasons"])
err = lens.per_view_errors(scan.corners, SPEC.pattern, SPEC.square_m, prof)
check(np.isfinite(err).all() and float(np.max(err)) < 1.0, "every view (partial ones too) scores", err)
used = rep["views_used"]
rec = lens.board_record(scan, used, SPEC.pattern, SPEC.square_m)
check(rec.get("kind") == "charuco" and charuco.BoardSpec.from_json(rec["charuco"]) == SPEC,
      "the profile's board record names the ChArUco board")
check("ChArUco" in lens.board_sentence(rec), "the report sentence names the board", lens.board_sentence(rec))
prof.board = rec
p2 = lens.LensProfile.from_json(prof.to_json())
check(p2.board.get("charuco") == SPEC.to_json() and np.array_equal(p2.K, prof.K),
      "the board survives the lens file round trip")
tile = lens.draw_board_review(scan.thumbs[0], scan.corners[0], SPEC.pattern, SPEC.square_m, prof,
                              scale=scan.thumb_scale(0))
k_part = next(i for i, v in enumerate(scan.corners) if np.isnan(v).any())
tile2 = lens.draw_board_review(scan.thumbs[k_part], scan.corners[k_part], SPEC.pattern, SPEC.square_m, prof,
                               scale=scan.thumb_scale(k_part))
check(tile.shape == scan.thumbs[0].shape and tile2.shape == scan.thumbs[k_part].shape,
      "a partly seen board is drawn for review")
cov = lens.coverage_image(scan, used)
check(cov.ndim == 3, "the coverage picture skips unseen corners")
check("markers" in lens.orientation_summary(scan.orient, SPEC.pattern), "the review board says how corner 0 is known")

# (review) 2.7K: a board with ~35 px squares in a far corner - where the lens is measured - must be found
# whole (a scan on a 1280-px copy lost half its corners); unit-sensitive code is tested at real sizes
W4, H4 = 2704, 1520
K4 = np.array([[1300.0, 0, (W4 - 1) / 2], [0, 1300.0, (H4 - 1) / 2], [0, 0, 1]])
D4 = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
vid4 = os.path.join(SCRATCH, "charuco_lens_27k.mp4")
poses4 = [(np.array([0.2, -0.25, 0.05]), np.array([-0.88, -0.48, 1.0])),        # top-left corner
          (np.array([-0.15, 0.3, 0.0]), np.array([0.85, 0.40, 1.10])),         # bottom-right corner
          (np.array([0.1, 0.1, 0.1]), np.array([-0.09, -0.06, 1.12]))]         # middle
if not os.path.exists(vid4):
    sc.write_video(vid4, [sc.render(SPEC, K4, D4, False, rv_, tv_, (W4, H4), seed=900 + k)
                          for k, (rv_, tv_) in enumerate(poses4)])
scan4 = lens.scan_video(vid4, SPEC.pattern, max_candidates=3, board=SPEC)
n4 = [int(np.isfinite(v).all(axis=1).sum()) for v in scan4.corners]
want4 = []
for rv_, tv_ in poses4:
    tr = sc.truth(SPEC, K4, D4, False, rv_, tv_)
    want4.append(int(((tr[:, 0] > 8) & (tr[:, 0] < W4 - 8) & (tr[:, 1] > 8) & (tr[:, 1] < H4 - 8)).sum()))
print(f"      2.7K, ~35 px squares: {len(scan4.corners)} of 3 boards, corners {n4} (inside the picture {want4})")
check(len(scan4.corners) == 3 and sum(n4) >= sum(want4) - 2,
      "2.7K: small boards in the corners are found whole (detected at full resolution)", (n4, want4))

# fisheye
Kf = np.array([[520.0, 0, W / 2 - 3], [0, 520.0, H / 2 + 2], [0, 0, 1]])
Df = np.array([0.08, -0.02, 0.005, 0.0])
vidf = os.path.join(SCRATCH, "charuco_lens_fish.mp4")
board_video(vidf, Kf, Df, True, 80, 6, spread=1.4, z=(0.2, 0.45))
scanf = lens.scan_video(vidf, SPEC.pattern, max_candidates=80, board=SPEC)
proff = lens.calibrate_lens(scanf.corners, SPEC.pattern, SPEC.square_m, scanf.size, model="fisheye")
print(f"      fisheye: {len(scanf.frames)} boards, rms {proff.rms:.3f} px, f {proff.K[0, 0]:.1f} (true 520), "
      f"pp ({proff.K[0, 2]:.1f}, {proff.K[1, 2]:.1f}), verdict {proff.report['verdict']}")
check(proff.fisheye and abs(proff.K[0, 0] - 520) / 520 < 0.005 and abs(proff.K[1, 1] - 520) / 520 < 0.005,
      "fisheye: focal length within 0.5 %", proff.K)
check(abs(proff.K[0, 2] - Kf[0, 2]) < 2.0 and abs(proff.K[1, 2] - Kf[1, 2]) < 2.0, "fisheye: centre within 2 px",
      proff.K)

# the GoPro curve: projected partial views (the curve fixed, focal + centre fitted)
h12 = gpmf.GoProInfo("h12", model="HERO12 Black", width=2704, height=1520, fps=239.76, lens_mode="Wide",
                     poly=list(H12_POLY), zmpl=H12_ZMPL, zfov_deg=H12_ZFOV)
base = gpmf.lens_profile(h12)
Kg = base.K_square().copy()
Kg[0, 0] = Kg[1, 1] = base.K[0, 0] * 1.012
Kg[0, 2] += 8.0
Kg[1, 2] -= 6.0
Dg = np.asarray(base.dist, float).reshape(4, 1)
big = charuco.BoardSpec(9, 7, 0.06, 0.045)
obj = charuco.object_points(big, np.arange(big.n_corners))
rg = np.random.default_rng(4)
vg = []
while len(vg) < 40:
    rv_ = rg.normal(0, 0.5, 3)
    tv_ = np.array([rg.uniform(-0.4, 0.2), rg.uniform(-0.3, 0.1), rg.uniform(0.5, 1.0)])
    pr, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv_, tv_, Kg, Dg)
    pr = pr.reshape(-1, 2) + rg.normal(0, 0.15, (len(obj), 2))
    r_ = np.hypot(pr[:, 0] - 1351.5, pr[:, 1] - 759.5) / np.hypot(1351.5, 759.5)
    ok_ = (r_ < 0.6) & (pr[:, 0] > 0)
    v = np.where(ok_[:, None], pr, np.nan)
    if charuco.usable_for_lens(v, big):
        vg.append(v)
n_part = sum(1 for v in vg if np.isnan(v).any())
profg = lens.calibrate_lens(vg, big.pattern, big.square_m, (2704, 1520), "gopro", max_views=lens.MAX_VIEWS, base=base)
dfg = profg.K[0, 0] / Kg[0, 0] - 1
print(f"      GoPro curve: {len(vg)} views ({n_part} partial), focal {100 * dfg:+.2f} %, centre "
      f"({profg.K[0, 2] - Kg[0, 2]:+.1f}, {profg.K[1, 2] - Kg[1, 2]:+.1f}) px")
check(n_part >= 5 and abs(dfg) < 0.005, "GoPro curve with partly seen boards: focal length within 0.5 %", dfg)

# ------------------------------------------------------------------ [3] the rig: lenses and camera positions
print("[3] the rig: three synced cameras film the board")
import dataclasses  # noqa: E402
import itertools  # noqa: E402

from kinetrace import charucowizard as cwz  # noqa: E402
from kinetrace import lens as _lens  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

RIG_N = 150
paths, cams = sc.rig_videos(SCRATCH, RIG_N)
RW, RH = sc.RIG_SIZE
rig = Project([TrackingSession(pth, RIG_N, 30.0, RW, RH) for pth in paths], ["cam1", "cam2", "cam3"],
              list(sc.RIG_OFFSETS))
scan = charuco.scan_rig(rig, sc.RIG_SPEC, n=100)
seen = [scan.seen(c) for c in range(3)]
print(f"      board seen at {seen} of {len(scan.ref_frames)} instants; failed frames {scan.failed}")
check(min(seen) >= 70 and scan.failed == [None, None, None] and scan.cameras == [0, 1, 2],
      "every camera sees the board at most instants; every frame decodes", (seen, scan.failed))
# cam2 runs half a frame late: its corners at an instant are interpolated between two frames
errs2 = []
for f, t in enumerate(scan.ref_frames):
    v = scan.views[f][1]
    K, D, R, tt, _pos = cams[1]
    Rb, ob = sc.board_at(t)
    rvec, _ = cv2.Rodrigues(R @ Rb)
    tr = sc.truth(sc.RIG_SPEC, K, D, False, rvec.ravel(), R @ ob + tt)
    ok_ = np.isfinite(v).all(axis=1)
    errs2 += np.linalg.norm(v[ok_] - tr[ok_], axis=1).tolist()
errs2 = np.asarray(errs2)
print(f"      cam2 (offset +0.5 frame): corners at the reference instants within {np.median(errs2):.2f} px "
      f"(median), {np.percentile(errs2, 95):.2f} px (95th pct) of the truth")
check(np.median(errs2) < 0.25 and np.percentile(errs2, 95) < 0.8,
      "a fractional offset: the corners are interpolated to the instant, not read half a frame off")
wand_uv, L, bg_uv, frames, _bgf = charuco.pairs_and_extras(scan)
check(len(wand_uv) >= 30 and abs(L - 0.42) < 1e-12 and len(bg_uv) > 500,
      "R2 / R4: board instants give wand pairs (7 squares = 420 mm) and extra points", (len(wand_uv), len(bg_uv)))
check(np.all((np.isfinite(wand_uv[:, :, 0]).all(axis=2)).sum(axis=1) >= 2),
      "every wand row is seen whole by two or more cameras")


def spacing(res):
    C = np.array([-R.T @ t for R, t in zip(res.R, res.t)])
    return np.array([np.linalg.norm(C[i] - C[j]) for i, j in itertools.combinations(range(3), 2)])


TRUE_SPACING = np.array([np.linalg.norm(cams[i][4] - cams[j][4]) for i, j in itertools.combinations(range(3), 2)])
out = cwz.calibrate_rig(rig, scan, "both")
res = out["result"]
for v, prof in sorted(out["lenses"].items()):
    K = cams[v][0]
    print(f"      cam{v + 1} lens: f {prof.K[0, 0]:.1f}/{prof.K[1, 1]:.1f} (true {K[0, 0]:.0f}), centre "
          f"({prof.K[0, 2] - K[0, 2]:+.1f}, {prof.K[1, 2] - K[1, 2]:+.1f}) px, rms {prof.rms:.2f}, "
          f"{prof.report['verdict']}")
check(sorted(out["lenses"]) == [0, 1, 2], "lenses and positions: a lens for every camera")
check(all(abs(out["lenses"][v].K[0, 0] - cams[v][0][0, 0]) / cams[v][0][0, 0] < 0.005 for v in range(3)),
      "every lens: focal length within 0.5 %")
check(all(abs(out["lenses"][v].K[0, 2] - cams[v][0][0, 2]) < 3 and abs(out["lenses"][v].K[1, 2] - cams[v][0][1, 2]) < 3
          for v in range(3)), "every lens: centre within 3 px")
check(all(out["lenses"][v].board.get("kind") == "charuco" and out["lenses"][v].rotation == 0 for v in range(3)),
      "each lens names the board and its video's turn")
# (review) a lens is fitted on corners ONE picture shows, and the frames it records hold exactly those corners:
# cam2 runs half a frame late, so its instants are interpolated - its lens must not be
fr2 = out["lenses"][1].board["frames"]
cap = cv2.VideoCapture(paths[1])
redet = []
for f in fr2[:4]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok_, bgr = cap.read()
    d_ = charuco.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), sc.RIG_SPEC) if ok_ else None
    redet.append(d_ is not None and np.array_equal(charuco.padded(d_[0], d_[1], sc.RIG_SPEC), scan.frame_views[1][f],
                                                   equal_nan=True))
cap.release()
check(len(fr2) >= 10 and set(fr2) <= set(scan.frame_views[1]) and all(redet),
      "a fractional offset: the lens boards are the pictures read, and their recorded frames re-detect the same "
      "corners", (len(fr2), redet))
sp = spacing(res)
print(f"      camera spacing {np.round(sp, 4)} m, true {np.round(TRUE_SPACING, 4)} m; verdict {res.report['verdict']}, "
      f"reprojection {res.report.get('reproj_rmse_all', float('nan')):.2f} px")
check(np.all(np.abs(sp / TRUE_SPACING - 1) < 0.01), "camera spacing within 1 % (the scale from the square size)",
      sp / TRUE_SPACING)
check(res.report["verdict"] in ("good", "ok"), "the calibration's verdict", res.report["verdict_reasons"])
check(all(res.cameras[c].undistort is not None and not isinstance(res.cameras[c].undistort, type(None))
          for c in range(3)), "the calibrated cameras straighten points with the fitted lenses")
check("board" in res.report and "420 mm" in cwz.board_html(res.report, out["lenses"], rig),
      "the report says the wand is the board's long edge and gives its length")

only = cwz.calibrate_rig(rig, scan, "lenses")
check(only["result"] is None and sorted(only["lenses"]) == [0, 1, 2]
      # (calibrateCamera's threaded sums differ by ~1e-8 px between two runs of the same input)
      and all(np.allclose(only["lenses"][v].K, out["lenses"][v].K, rtol=0, atol=1e-5) for v in range(3)),
      "lenses only: the same lenses, no calibration")
existing = {v: out["lenses"][v] for v in range(3)}
keys = [_lens.profile_key(existing[v]) for v in range(3)]
pos = cwz.calibrate_rig(rig, scan, "positions", existing)
check(pos["lenses"] == {} and [_lens.profile_key(existing[v]) for v in range(3)] == keys,
      "positions only: no lens fitted, the cameras' own profiles untouched")
check(np.all(np.abs(spacing(pos["result"]) / TRUE_SPACING - 1) < 0.01), "positions only: spacing within 1 %")
big = dataclasses.replace(scan, spec=dataclasses.replace(sc.RIG_SPEC, square_m=0.12, marker_m=0.09))
pos2 = cwz.calibrate_rig(rig, big, "positions", existing)
ratio = spacing(pos2["result"]) / spacing(pos["result"])
check(np.all(np.abs(ratio - 2.0) < 0.002), "a square typed twice as big doubles every length", ratio)
try:
    cwz.calibrate_rig(rig, dataclasses.replace(scan, views=[row[:1] + [np.full_like(row[1], np.nan)] * 2
                                                            for row in scan.views]), "positions", existing)
    said = ""
except ValueError as e:
    said = str(e)
check("at least 10" in said, "a rig seen by one camera only is refused in words", said)

# ------------------------------------------------------------------ [4] Argus's files
print("[4] Argus's files")
from kinetrace import calibio  # noqa: E402

prof_path = os.path.join(SCRATCH, "charuco_camera_profile.txt")
lens_list = [(v + 1, out["lenses"][v]) for v in range(3)]
calibio.write_argus_profiles(prof_path, lens_list)
lines = open(prof_path, encoding="utf-8").read().splitlines()
cols = [ln.split("\t") for ln in lines]
check(len(lines) == 3 and all(len(c) == 12 for c in cols) and [c[0] for c in cols] == ["1", "2", "3"],
      "camera_profile.txt: a tab-separated line per camera, 12 columns, numbered 1..3", lines[:1])
p0 = out["lenses"][0]
c0 = cols[0]
check(c0[2:4] == [str(RW), str(RH)] and all(len(x.split(".")[1]) == 6 for x in (c0[1], *c0[4:])),
      "integers plain, every other number to 6 decimals (as Argus)", c0)
check(abs(float(c0[5]) - (RH - p0.K[1, 2])) < 1e-6 and abs(float(c0[6]) + p0.K[1, 1] / p0.K[0, 0]) < 1e-6
      and abs(float(c0[4]) - p0.K[0, 2]) < 1e-6,
      "y up as Argus's ChArUco script: cy' = h - cy, the aspect ratio negative", c0[4:7])
check(np.allclose([float(x) for x in c0[7:]], np.pad(p0.dist, (0, 5))[:5], atol=1e-6),
      "distortion k1 k2 p1 p2 k3 as fitted")
back = _lens.load_argus_profile(prof_path)
check(len(back) == 3 and all(np.allclose(back[k].K, out["lenses"][k].K, atol=2e-6)
                             and np.allclose(back[k].dist, np.pad(out["lenses"][k].dist, (0, 5))[:5], atol=2e-6)
                             for k in range(3)),
      "Kinetrace reads the y-up file back to the same lenses (to the file's 6 decimals)")
one, which = _lens.argus_profile_for(prof_path, 1, "cam2")
check(one is not None and "camera 2" in which and np.allclose(one.K, out["lenses"][1].K, atol=2e-6),
      "camera 2's line goes to camera 2")
calibio.write_argus_profiles(prof_path, lens_list, flip_y=False)
c0n = open(prof_path, encoding="utf-8").read().splitlines()[0].split("\t")
check(abs(float(c0n[5]) - p0.K[1, 2]) < 1e-6 and float(c0n[6]) > 0, "flip_y=False: top-left, as written")
fish = _lens.LensProfile(RW, RH, np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]]), np.zeros(4), True)
try:
    calibio.write_argus_profiles(prof_path, [(1, fish)])
    said = ""
except calibio.CalibFormatError as e:
    said = str(e)
check("fisheye" in said, "a fisheye lens is refused in words", said)

files = calibio.write_argus_xypts(SCRATCH, wand_uv, bg_uv, scan.sizes)
paired = open(files[0], encoding="utf-8").read().splitlines()
unp = open(files[1], encoding="utf-8").read().splitlines()
check(os.path.basename(files[0]) == "paired_pts-xypts.csv" and os.path.basename(files[1]) == "unpaired_pts-xypts.csv",
      "Argus's file names")
check(paired[0].split(",")[:4] == ["pt1_cam1_x", "pt1_cam1_y", "pt1_cam2_x", "pt1_cam2_y"]
      and paired[0].split(",")[6:8] == ["pt2_cam1_x", "pt2_cam1_y"] and len(paired[0].split(",")) == 12,
      "paired header: pt1 in every camera, then pt2", paired[0])
check(unp[0].split(",") == [f"pt1_cam{c}_{a}" for c in (1, 2, 3) for a in ("x", "y")], "unpaired header", unp[0])
check(len(paired) - 1 == len(wand_uv) and len(unp) - 1 == len(bg_uv), "a row per board instant / per extra corner")
P = np.genfromtxt(files[0], delimiter=",", skip_header=1).reshape(-1, 2, 3, 2)      # [row, end, camera, xy]
P[..., 1] = np.array([h for _w, h in scan.sizes])[None, None, :] - P[..., 1]
raw = np.transpose(wand_uv, (0, 2, 1, 3))
same = np.isnan(P) == np.isnan(raw)
check(same.all() and np.allclose(P[~np.isnan(P)], raw[~np.isnan(raw)], atol=1e-6),
      "paired points: raw pixels, y' = h - y, nan where unseen")
B = np.genfromtxt(files[1], delimiter=",", skip_header=1).reshape(-1, 3, 2)
B[..., 1] = np.array([h for _w, h in scan.sizes])[None, :] - B[..., 1]
check((np.isnan(B) == np.isnan(bg_uv)).all() and np.allclose(B[~np.isnan(B)], bg_uv[~np.isnan(bg_uv)], atol=1e-6),
      "unpaired points the same")

if FAILS:
    print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
    sys.exit(1)
print("PASSED")
