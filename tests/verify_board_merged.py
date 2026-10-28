"""(I363) A checkerboard view with two neighbouring corners found on ONE spot is flagged and never auto-picked.

On a strongly bent (fisheye) lens the squares near the picture's edge can be 9-12 px wide. The corner search there
(the 19 px full-size refine window, and once OpenCV's own detector) is wider than a square, so two neighbouring
corners slide onto the same spot and the board is 2-8 px wrong. The board review now says so on the board's tile
("poor: corners merged", with the reason and a hand check), *Best spread* and *Drop the worst* leave such a board
out (even when few boards are left), *All* and a hand tick still use it (the user decides), and dragging the
corners apart clears the flag. No corner value changes anywhere.

Sections: the rule itself on made-up boards; the board review's buttons and a tile click, as the user does them;
the rendered fisheye video (verify_lens's renderer) where the flag must pick exactly the boards that are wrong.

Offscreen; no GPU. Run: .venv\\Scripts\\python.exe tests\\verify_board_merged.py
"""
import contextlib
import os
import sys
import time
import warnings

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from kinetrace import boardreview as brv
from kinetrace import lens

OUT = os.path.join(HERE, "out", "board_merged")
os.makedirs(OUT, exist_ok=True)
FAILS = []
app = QApplication.instance() or QApplication(sys.argv)

PATTERN = (9, 6)
SQUARE = 0.024
W, H = 960, 720
K_STD = np.array([[820.0, 0, 480.0], [0, 815.0, 360.0], [0, 0, 1.0]])
D_STD = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])


def check(label):
    """Run the function now; a failure is recorded and printed, the rest still run."""
    def deco(fn):
        t0 = time.time()
        try:
            fn()
            print(f"  ok   {label} ({time.time() - t0:.1f} s)")
        except BaseException as e:      # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            FAILS.append(label)
            print(f"  FAIL {label}: {type(e).__name__}: {str(e)[:300]}")
        return fn
    return deco


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def pump(seconds=0.2):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def synth_boards(n, seed):
    """n checkerboard poses projected through a standard lens (no video): corner arrays (54, 2)."""
    r = np.random.RandomState(seed)
    obj = lens._object_points(PATTERN, SQUARE)
    obj = obj - np.array([(PATTERN[0] - 1) / 2.0 * SQUARE, (PATTERN[1] - 1) / 2.0 * SQUARE, 0.0])
    out = []
    while len(out) < n:
        z = r.uniform(0.45, 0.9)
        u, v = r.uniform(0.08, 0.92) * W, r.uniform(0.08, 0.92) * H
        rvec = np.deg2rad(35.0) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        tvec = np.array([(u - K_STD[0, 2]) / K_STD[0, 0] * z, (v - K_STD[1, 2]) / K_STD[1, 1] * z, z])
        p, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K_STD, D_STD)
        p = p.reshape(-1, 2)
        if (p < 2).any() or (p[:, 0] > W - 3).any() or (p[:, 1] > H - 3).any():
            continue
        out.append(p + r.normal(0, 0.08, p.shape))
    return out


def fake_scan(corners, video=""):
    n = len(corners)
    thumbs = [np.full((360, 480, 3), 120, np.uint8) for _ in range(n)]
    return lens.ScanResult((W, H), 30.0, 10 * n, n, list(range(0, 10 * n, 10)), [c.copy() for c in corners],
                           thumbs[0], thumbs, video, ["colour"] * n)


def merge(c, a=7, b=8):
    """Board `c` with corner `b` found on corner `a`'s spot (the failure, made by hand)."""
    c = c.copy()
    c[b] = c[a]
    return c


def button(rv, text):
    return next(b for b in rv.findChildren(QPushButton) if b.text() == text)


print("verify_board_merged: two corners on one spot (I363)")

# =================================================================== the rule
print("the rule")


@check("a sound board has no merged corners; a board with corner 8 on corner 7 has exactly that pair")
def _():
    b = synth_boards(1, seed=3)[0]
    assert lens.merged_corners(b, PATTERN) == [], lens.merged_corners(b, PATTERN)
    assert lens.merged_reason(b, PATTERN) == ""
    assert lens.merged_corners(merge(b), PATTERN) == [(7, 8)], lens.merged_corners(merge(b), PATTERN)
    why = lens.merged_reason(merge(b), PATTERN)
    for word in ("same spot", "0.0 px", "row 1", "corners 8 and 9", "squares are about", "Best spread",
                 "Check by hand", "Untick", "drag"):
        assert word in why, (word, why)
    why = lens.merged_reason(merge(b), PATTERN, editable=False)          # a ChArUco view cannot be dragged
    assert "drag" not in why and "Leave it unticked" in why, why


@check("a strongly foreshortened column (0.36 x the median spacing, the tightest sound board measured) is not merged")
def _():
    gx, gy = np.meshgrid(np.arange(9) * 20.0, np.arange(6) * 20.0)
    g = np.stack([gx, gy], axis=2)
    g[:, 8, 0] = g[:, 7, 0] + 0.36 * 20.0
    assert lens.merged_corners(g.reshape(-1, 2), PATTERN) == []
    g[:, 8, 0] = g[:, 7, 0] + 0.05 * 20.0
    assert len(lens.merged_corners(g.reshape(-1, 2), PATTERN)) == 6


@check("a view with unseen (NaN) corners, as a ChArUco board gives, is never flagged by them and warns nothing")
def _():
    b = synth_boards(1, seed=4)[0]
    b[[3, 4, 12, 13]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert lens.merged_corners(b, PATTERN) == []
        b[[0, 1, 2, 5, 6, 7, 8] + list(range(9, 54))] = np.nan          # one corner left: nothing to compare
        assert lens.merged_corners(b, PATTERN) == []


@check("auto_select never picks a merged board, even when dropping it leaves fewer than 8, and says why")
def _():
    boards = synth_boards(8, seed=11)
    boards[5] = merge(boards[5])
    idx, err, why = lens.auto_select(boards, PATTERN, SQUARE, (W, H), "standard")
    assert 5 not in idx, (idx, why)
    assert sorted(idx) == [0, 1, 2, 3, 4, 6, 7], idx
    assert np.isfinite(err[5]), "the merged board is still scored, so its tile can show its number"
    assert "two corners found on one spot" in why, why
    idx, err, why = lens.auto_select(boards[3:6], PATTERN, SQUARE, (W, H), "standard")    # board 5 is the last
    assert sorted(idx) == [0, 1] and why.startswith("only 2 of 3 boards usable -- using those"), (idx, why)
    every = [merge(b) for b in boards]
    idx, err, why = lens.auto_select(every, PATTERN, SQUARE, (W, H), "standard")
    assert idx == [] and "every board" in why, (idx, why)


# =================================================================== the board review, by its buttons
print("the board review")


@check("Best spread unticks the merged board and its tile says so; Drop the worst keeps it out; All ticks it, said")
def _():
    boards = synth_boards(8, seed=11)
    boards[5] = merge(boards[5])
    scan = fake_scan(boards)
    prof = lens.calibrate_lens([b for i, b in enumerate(boards) if i != 5], PATTERN, SQUARE, (W, H), "standard")
    rv = brv.BoardReview()
    rv.set_scan(scan, PATTERN, SQUARE, prof, list(range(8)), lens.per_view_errors(boards, PATTERN, SQUARE, prof),
                model="standard")
    rv.show()
    pump(0.1)
    try:
        QTest.mouseClick(button(rv, "Best spread"), Qt.LeftButton)
        pump(0.1)
        assert not rv.used[5], "Best spread left the merged board ticked (the 'too few would be left' fallback)"
        assert all(rv.used[i] for i in range(8) if i != 5), rv.used
        assert rv._tiles[5].err.text() == "poor: corners merged", rv._tiles[5].err.text()
        assert "same spot" in rv._tiles[5].err.toolTip(), rv._tiles[5].err.toolTip()
        assert "px" in rv._tiles[0].err.text(), "a sound board's tile keeps its number"
        assert "1 with two corners on one spot left out" in rv.summary.text(), rv.summary.text()
        QTest.mouseClick(button(rv, "All"), Qt.LeftButton)
        pump(0.05)
        assert rv.used[5], "All is the user's choice: every board, the merged one too"
        assert "1 ticked board has two corners on one spot" in rv.summary.text(), rv.summary.text()
        QTest.mouseClick(button(rv, "Drop the worst"), Qt.LeftButton)
        pump(0.05)
        assert not rv.used[5], "Drop the worst unticks a merged board"
        assert "two corners on one spot" in rv.summary.text(), rv.summary.text()
        QTest.mouseClick(rv._tiles[5].chk, Qt.LeftButton)
        pump(0.05)
        assert rv.used[5], "a hand tick uses the merged board"
        assert "1 ticked board has two corners on one spot" in rv.summary.text(), rv.summary.text()
    finally:
        rv.close()
        rv.deleteLater()


@check("clicking the board and dragging the corner apart (Keep the change) clears the flag")
def _():
    boards = synth_boards(8, seed=11)
    sound = boards[5].copy()
    boards[5] = merge(boards[5])
    vid = os.path.join(OUT, "frames80.mp4")
    if not os.path.exists(vid):
        vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        for _ in range(80):
            vw.write(np.full((H, W, 3), 120, np.uint8))
        vw.release()
    scan = fake_scan(boards, video=vid)
    rv = brv.BoardReview()
    rv.set_scan(scan, PATTERN, SQUARE, None, list(range(8)), None, model="standard")
    rv.show()
    pump(0.1)
    opened = []

    def keep(dlg):                     # the user drags corner 8 back onto its own corner and keeps the change
        opened.append(dlg)
        dlg.corners = sound.copy()
        dlg._dirty = True
        return QDialog.Accepted
    try:
        assert rv._tiles[5].err.text() == "poor: corners merged", rv._tiles[5].err.text()
        with patched(brv.CornerEditor, "exec", keep):
            QTest.mouseClick(rv._tiles[5].pic, Qt.LeftButton)
            t0 = time.time()
            while not opened and time.time() - t0 < 30:
                pump(0.05)
        assert opened, "the editor opened"
        assert rv._tiles[5].err.text() != "poor: corners merged", rv._tiles[5].err.text()
        assert "same spot" not in rv._tiles[5].err.toolTip()
        assert np.array_equal(scan.corners[5], sound)
    finally:
        rv.stop_reader()
        rv.close()
        rv.deleteLater()


# =================================================================== rendered boards with known corners
print("rendered boards (verify_lens's renderer, true corners known)")


def render(path, K, dist, fisheye, n_frames, spread=1.0, tilt_deg=35.0, seed=0):
    """verify_lens.render_video, returning the true inner corners of every frame."""
    r = np.random.RandomState(seed)
    cols, rows = PATTERN
    nx, ny = cols + 1, rows + 1
    SUB = 3
    gx, gy = np.meshgrid(np.arange(-1, nx * SUB + 1) / SUB, np.arange(-1, ny * SUB + 1) / SUB)
    board = np.stack([gx.ravel() * SQUARE, gy.ravel() * SQUARE, np.zeros(gx.size)], axis=1)
    centre = np.array([(nx - 2) / 2.0 * SQUARE, (ny - 2) / 2.0 * SQUARE, 0.0])
    board -= centre
    ix, iy = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    inner = np.stack([ix.ravel() * SQUARE, iy.ravel() * SQUARE, np.zeros(ix.size)], 1) - centre
    write = not os.path.exists(path)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H)) if write else None
    truth = []
    for _ in range(n_frames):
        z = r.uniform(0.45, 0.9)
        u = r.uniform(0.5 - 0.42 * spread, 0.5 + 0.42 * spread) * W
        v = r.uniform(0.5 - 0.42 * spread, 0.5 + 0.42 * spread) * H
        rvec = np.deg2rad(tilt_deg) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        tvec = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])

        def project(o):
            if fisheye:
                return cv2.fisheye.projectPoints(o.reshape(-1, 1, 3), rvec, tvec, K, dist.reshape(-1, 1)[:4])[0]
            return cv2.projectPoints(o.reshape(-1, 1, 3), rvec, tvec, K, dist)[0]
        truth.append(project(inner).reshape(-1, 2))
        noise = r.normal(0, 3, (H, W, 1)).astype(np.int16)
        if not write:
            continue
        pts = project(board).reshape(ny * SUB + 2, nx * SUB + 2, 2)
        img = np.full((H, W, 3), 140, np.uint8)
        for yy in range(ny * SUB):
            for xx in range(nx * SUB):
                quad = np.array([pts[yy + 1, xx + 1], pts[yy + 1, xx + 2], pts[yy + 2, xx + 2], pts[yy + 2, xx + 1]],
                                np.float32)
                if not np.isfinite(quad).all() or np.abs(quad).max() > 4 * max(W, H):
                    continue
                col = 15 if ((xx // SUB) + (yy // SUB)) % 2 == 0 else 245
                cv2.fillConvexPoly(img, np.round(quad * 16).astype(np.int32), (col, col, col), cv2.LINE_AA, 4)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vw.write(cv2.GaussianBlur(img, (0, 0), 0.6))
    if vw is not None:
        vw.release()
    return truth


def off_truth(c, t):
    """Mean distance of a found board's corners to the true ones, corner by corner (whichever of the four
    grid orientations matches)."""
    g, tg = c.reshape(PATTERN[1], PATTERN[0], 2), t.reshape(PATTERN[1], PATTERN[0], 2)
    return min(float(np.linalg.norm(g - o, axis=2).mean()) for o in (tg, tg[::-1, ::-1], tg[::-1, :], tg[:, ::-1]))


@check("fisheye video shrunk by 2: the flagged boards are exactly the ones more than 1.5 px off the true corners")
def _():
    Kf = np.array([[520.0, 0, W / 2 - 3], [0, 520.0, H / 2 + 2], [0, 0, 1]])
    Df = np.array([0.08, -0.02, 0.005, 0.0])
    vid = os.path.join(OUT, "lens_fish.mp4")
    truth = render(vid, Kf, Df, True, 90, spread=1.7, seed=1)
    scan = lens.scan_video(vid, PATTERN, max_candidates=90, downscale_max=480)
    off = {f: off_truth(c, truth[f]) for f, c in zip(scan.frames, scan.corners)}
    flagged = {f for f, c in zip(scan.frames, scan.corners) if lens.merged_corners(c, PATTERN)}
    wrong = {f for f, e in off.items() if e > 1.5}
    print(f"     {len(scan.frames)} boards; flagged {sorted(flagged)}; > 1.5 px off {sorted(wrong)}; "
          f"worst unflagged {max(e for f, e in off.items() if f not in flagged):.2f} px")
    assert len(wrong) >= 4, "the video no longer shows the failure: the test proves nothing"
    assert flagged == wrong, (sorted(flagged), sorted(wrong))


@check("standard lens video shrunk by 2: no board is flagged")
def _():
    K = np.array([[820.0, 0, W / 2 + 6], [0, 815.0, H / 2 - 4], [0, 0, 1]])
    D = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
    vid = os.path.join(OUT, "lens_std.mp4")
    render(vid, K, D, False, 70)
    scan = lens.scan_video(vid, PATTERN, max_candidates=70, downscale_max=480)
    flagged = [f for f, c in zip(scan.frames, scan.corners) if lens.merged_corners(c, PATTERN)]
    assert len(scan.frames) >= 50 and flagged == [], (len(scan.frames), flagged)


print("\nverify_board_merged: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
