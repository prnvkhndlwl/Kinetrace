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

if FAILS:
    print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
    sys.exit(1)
print("PASSED")
