"""Code review 2026-10-02: the calibration half (wand, lens, their wizards, the board review).

One check per finding, each written so that it FAILS on the code before the fix:
I176 (wand verdict from named checks), I177 (converged lens inverse in the wand core),
I178 / G117 (auto lens model scored on the same views, a rejected fisheye said),
I179 (held poses are one pose), I200 (wizard close waits), I221 (focal NaN = unknown),
I222 (non-ASCII save paths, errors in words), I247 (coverage on raw clicks), I248 (outlier
frames), I249 (editor / wizard released), I250 (portrait bands), G112-G116 (wizard
errors, stale fits, the video page, "All" is fitted whole), G137 / G138, R19 / R20 (one rule
each, one column sampled).

Run: .venv\\Scripts\\python.exe tests\\verify_review_calib.py
To see which checks FAIL on the code before the fixes, point KINETRACE_TEST_ROOT at an
export of the base commit (git archive HEAD kinetrace) and run the same file.
No GPU, no network, no video footage (a one-frame clip is written for the corner editor).
"""
import contextlib
import copy
import logging
import os
import sys
import threading
import time
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("KINETRACE_TEST_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2
import numpy as np

SCRATCH = os.path.join(HERE, "out", "review_calib")
os.makedirs(SCRATCH, exist_ok=True)
FAILS: list[str] = []
T_ALL = time.time()


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
            print(f"  FAIL {label}: {type(e).__name__}: {str(e)[:220]}")
        return fn
    return deco


@contextlib.contextmanager
def patched(obj, name, value):
    had = name in getattr(obj, "__dict__", {})
    old = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had or not hasattr(obj, "__dict__"):
            setattr(obj, name, old)
        else:
            try:
                delattr(obj, name)
            except AttributeError:
                setattr(obj, name, old)


# ======================================================================= helpers
def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


def wand_rig(seed, sizes, focals, F=100, noise=0.4, nb=20, radius=2.4, vol=0.4, drop=0.15):
    """A synthetic wand rig: (wand_uv (F, C, 2, 2), bg_uv, rng)."""
    rng = np.random.RandomState(seed)
    C = len(sizes)
    Ks, Rs, ts = [], [], []
    for k in range(C):
        az = 0.3 + k * 2 * np.pi / C + rng.uniform(-0.3, 0.3)
        pos = np.array([radius * np.cos(az), radius * np.sin(az), rng.uniform(0.3, 1.6)])
        R, t = look_at(pos, rng.uniform(-0.1, 0.1, 3))
        w, h = sizes[k]
        Ks.append(np.array([[focals[k], 0, (w - 1) / 2], [0, focals[k], (h - 1) / 2], [0, 0, 1]]))
        Rs.append(R)
        ts.append(t)
    cen = rng.uniform(-vol, vol, (F, 3))
    d = rng.normal(size=(F, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ends = np.stack([cen - 0.25 * d, cen + 0.25 * d], axis=1)
    bg = rng.uniform(-0.5, 0.5, (nb, 3))

    def project(X, c):
        xc = (Rs[c] @ X.T + ts[c][:, None]).T
        uv = (Ks[c] @ (xc / xc[:, 2:3]).T).T[:, :2] + rng.normal(0, noise, (len(X), 2))
        w, h = sizes[c]
        out = (uv[:, 0] < 0) | (uv[:, 0] > w - 1) | (uv[:, 1] < 0) | (uv[:, 1] > h - 1) | (xc[:, 2] <= 0)
        uv[out] = np.nan
        return uv
    wand_uv = np.full((F, C, 2, 2), np.nan)
    bg_uv = np.full((nb, C, 2), np.nan)
    for c in range(C):
        wand_uv[:, c] = project(ends.reshape(-1, 3), c).reshape(F, 2, 2)
        bg_uv[:, c] = project(bg, c)
    wand_uv[rng.rand(F, C, 2) < drop] = np.nan
    return wand_uv, bg_uv, rng


PATTERN = (9, 6)
SQUARE = 0.024


def synth_boards(K, dist, n, seed, fisheye=False, tilt=35.0, spread=1.0, noise=0.08, copies=1, size=None):
    """n checkerboard poses projected through a camera (no video): corner arrays (54, 2)."""
    from kinetrace import lens
    r = np.random.RandomState(seed)
    obj = lens._object_points(PATTERN, SQUARE)
    obj = obj - np.array([(PATTERN[0] - 1) / 2.0 * SQUARE, (PATTERN[1] - 1) / 2.0 * SQUARE, 0.0])
    w, h = size if size else (int(K[0, 2] * 2), int(K[1, 2] * 2))
    out = []
    while len(out) < n:
        z = r.uniform(0.45, 0.9)
        u = r.uniform(0.5 - 0.42 * spread, 0.5 + 0.42 * spread) * w
        v = r.uniform(0.5 - 0.42 * spread, 0.5 + 0.42 * spread) * h
        rvec = np.deg2rad(tilt) * r.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.6])
        tvec = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        o = obj.reshape(-1, 1, 3)
        if fisheye:
            p, _ = cv2.fisheye.projectPoints(o, rvec, tvec, K, np.asarray(dist, np.float64).reshape(-1, 1)[:4])
        else:
            p, _ = cv2.projectPoints(o, rvec, tvec, K, np.asarray(dist, np.float64))
        p = p.reshape(-1, 2)
        if not np.isfinite(p).all() or (p < 2).any() or (p[:, 0] > w - 3).any() or (p[:, 1] > h - 3).any():
            continue
        out.append(p + r.normal(0, noise, p.shape))
    return [c.copy() for c in out for _ in range(copies)]


def fake_scan(corners, size, video="", orient=None):
    from kinetrace import lens
    n = len(corners)
    thumbs = [np.full((int(size[1] * 480 / size[0]), 480, 3), 120, np.uint8) for _ in range(n)]
    return lens.ScanResult(size, 30.0, 10 * n, n, list(range(0, 10 * n, 10)), [c.copy() for c in corners],
                           thumbs[0] if thumbs else None, thumbs, video, list(orient or ["colour"] * n))


print("verify_review_calib: code review fixes, calibration half")
from kinetrace import lens, wand, wanddata as wd  # noqa: E402

# =================================================================== wand (no Qt)
print("-- wand")


@check("I177 wand normalise: the lens inverse is converged (k1 -0.30 / k2 0.10, f 0.45 x width)")
def _():
    C = 2
    cams = np.zeros((C, 11))
    cams[:, 6] = 0.45 * 1920
    cams[:, 7], cams[:, 8] = -0.30, 0.10
    cams[:, 9], cams[:, 10] = 959.5, 539.5
    cams[1, :3] = [0.0, 0.25, 0.0]
    cams[1, 3:6] = [-0.9, 0.0, 0.2]
    # points that land near the picture CORNERS of both cameras (the lens bends them most)
    X = np.array([[1.0 * z, 0.55 * z, z] for z in (2.2, 2.6, 3.0)] +
                 [[-1.0 * z, -0.55 * z, z] for z in (2.2, 2.6, 3.0)] +
                 [[1.0 * z, -0.55 * z, z] for z in (2.4, 2.8)] + [[-1.0 * z, 0.55 * z, z] for z in (2.4, 2.8)])
    pts = np.arange(len(X))
    uv = np.full((len(X), C, 2), np.nan)
    for c in range(C):
        uv[:, c] = wand._project(cams, X, pts, np.full(len(X), c))
    xyz, err = wand._triangulate_uv(cams, uv)
    e = np.linalg.norm(xyz - X, axis=1)
    print(f"      corner points: triangulation error median {np.median(e) * 1000:.4f} mm, max {e.max() * 1000:.4f} mm; "
          f"reprojection {np.nanmax(err):.5f} px")
    assert e.max() < 1e-6, e.max()


@check("I176 wand verdict: a camera that lost 3/4 of its observations is not GOOD (named checks)")
def _():
    sizes = [(1280, 720)] * 4
    foc = [760.0] * 4
    n_good_old = 0
    for seed in (1, 2, 3):
        wuv, buv, rng = wand_rig(seed, sizes, foc, F=120, vol=0.6, radius=2.0, drop=0.1)
        bad = wuv.copy()
        m = rng.rand(bad.shape[0]) < 0.85
        bad[m, 3] += rng.normal(0, 25.0, bad[m, 3].shape)          # camera 4: a sync slip, almost all wrong
        rep = wand.calibrate_wand(bad, 0.5, sizes, bg_uv=buv, focal=foc, estimate_focal=False).report
        kept, total = rep["n_observations_per_camera"], rep.get("n_observations_total_per_camera")
        print(f"      seed {seed}: verdict {rep['verdict']}, rejected {rep['outliers_removed']} of "
              f"{rep['n_observations']}, camera 4 kept {kept[3]} of {total[3] if total else '?'}")
        assert rep["verdict"] != "good", (seed, rep["verdict"], rep["verdict_reasons"])
        names = {c["name"] for c in rep["verdict_checks"]}
        assert "kept_3" in names and "rejected" in names, names
        # the verdict is the worst check: no check allows better than the verdict
        order = ["good", "ok", "poor"]
        assert order[max(order.index(c["allows"]) for c in rep["verdict_checks"])] == rep["verdict"]
        assert any("lost" in r and "camera 4" in r.lower() for r in rep["verdict_reasons"]), rep["verdict_reasons"]
    # and a clean rig is still GOOD: the new checks do not turn a sound calibration away
    wuv, buv, rng = wand_rig(11, [(1280, 720)] * 4, foc, F=120, vol=0.6, radius=2.0, drop=0.1)
    rep = wand.calibrate_wand(wuv, 0.5, [(1280, 720)] * 4, bg_uv=buv, focal=foc, estimate_focal=False).report
    assert rep["verdict"] == "good", (rep["verdict"], rep["verdict_reasons"])
    # a camera under the PnP minimum is poor
    chk = wand._verdict_checks(0.1, 0.3, 0.3, 1.0, [50.0] * 3, [200, 4, 200], np.array([200, 200, 200]), 60, 0, 600,
                               2.0, None, False)
    assert max(c.level for c in chk) == 2 and any("kept only 4" in c.text for c in chk)


@check("I221 wand: focal NaN = unknown for that camera; the mixed rig calibrates (6 rigs)")
def _():
    sizes = [(1920, 1080)] * 4
    foc = [864.0, 672.0, 1400.0, 1400.0]          # two wide cameras, two with a lens profile
    bad = 0
    for seed in range(1, 7):
        wuv, buv, rng = wand_rig(seed, sizes, foc, F=100, radius=1.8, vol=0.6, drop=0.3)
        res = wand.calibrate_wand(wuv, 0.5, sizes, bg_uv=buv, focal=[np.nan, np.nan, 1400.0, 1400.0],
                                  estimate_focal=True)
        rep = res.report
        errs = [abs(f - t) / t for f, t in zip(rep["focal_px"], foc)]
        print(f"      rig {seed}: {rep['verdict']}, focal {[round(f) for f in rep['focal_px']]}, "
              f"max {100 * max(errs):.2f} % off")
        if rep["verdict"] == "poor" or max(errs) > 0.05:
            bad += 1
        assert rep["focal_searched_per_camera"] == [True, True, False, False]
        assert any("GUESSED" in r and "camera 1" in r for r in rep["verdict_reasons"]), rep["verdict_reasons"]
    assert bad == 0, f"{bad} of 6 rigs failed"
    # a NaN-free list and an all-NaN list behave as before
    wuv, buv, _ = wand_rig(1, sizes, foc, F=100, radius=1.8, vol=0.6, drop=0.3)
    try:
        wand.calibrate_wand(wuv, 0.5, sizes, bg_uv=buv, focal=[0.0, 600.0, 600.0, 600.0])
        raise AssertionError("a zero focal length must be refused")
    except wand.WandError:
        pass


@check("I247 wand: coverage is measured on the RAW clicks (coverage_uv)")
def _():
    sizes = [(1280, 720)] * 3
    wuv, buv, rng = wand_rig(5, sizes, [900.0] * 3, F=90, vol=0.6, radius=2.0, drop=0.05)
    centre = np.array([639.5, 359.5])
    raw = wuv.copy()
    raw = centre + (raw - centre) * 0.5                       # the raw clicks fill a quarter of the area
    foc = [900.0] * 3
    full = wand.calibrate_wand(wuv, 0.5, sizes, focal=foc, estimate_focal=False).report["coverage_pct"]
    held = wand.calibrate_wand(wuv, 0.5, sizes, focal=foc, estimate_focal=False, coverage_uv=raw)
    cov = held.report["coverage_pct"]
    expect = wand._coverage(raw, None, sizes)
    print(f"      coverage on the straightened points {np.round(full, 1)} %, on the raw clicks {np.round(cov, 1)} %")
    assert np.allclose(cov, expect) and all(c < 0.4 * f for c, f in zip(cov, full)), (cov, full)
    assert any("covered only" in r for r in held.report["verdict_reasons"])
    try:
        wand.calibrate_wand(wuv, 0.5, sizes, focal=foc, coverage_uv=raw[:10])
        raise AssertionError("a coverage array of the wrong shape must be refused")
    except wand.WandError:
        pass


@check("I248 wand: the outlier list names the REFERENCE frame and keeps the row")
def _():
    sizes = [(1280, 720)] * 4
    wuv, buv, rng = wand_rig(7, sizes, [760.0] * 4, F=100, vol=0.6, radius=2.0, drop=0.05)
    fin = np.argwhere(np.isfinite(wuv).all(-1))
    inj = fin[rng.choice(len(fin), 4, replace=False)]
    for f_, c_, e_ in inj:
        wuv[f_, c_, e_] += 30.0
    frames = 10 * np.arange(100) + 7                     # every 10th reference frame was clicked
    rep = wand.calibrate_wand(wuv, 0.5, sizes, bg_uv=buv, focal=[760.0] * 4, estimate_focal=False,
                              frame_ids=frames).report
    got = {(o["row"], o["cam"], o["end"]): o["frame"] for o in rep["outliers"] if o["kind"] == "wand"}
    for f_, c_, e_ in inj:
        assert got.get((int(f_), int(c_), int(e_))) == int(frames[f_]), (f_, got.get((int(f_), int(c_), int(e_))))
    rep0 = wand.calibrate_wand(wuv, 0.5, sizes, bg_uv=buv, focal=[760.0] * 4, estimate_focal=False).report
    assert all(o["frame"] == o["row"] for o in rep0["outliers"] if o["kind"] == "wand"), "no frame_ids: row = frame"


@check("I250 wand: pixel bands scale with the LONGER side (portrait 4K)")
def _():
    s = wand._px_scale([(2160, 3840), (3840, 2160), (1920, 1080), (640, 480)])
    assert np.allclose(s, [2.0, 2.0, 1.0, 1.0]), s


# ==================================================================== lens (no Qt)
print("-- lens")
K_STD = np.array([[820.0, 0, 480.0], [0, 815.0, 360.0], [0, 0, 1.0]])
D_STD = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])


@check("I179 lens: held poses are ONE pose (no repeated index, no inflated view / tilt count)")
def _():
    poses = synth_boards(K_STD, D_STD, 10, seed=21, size=(960, 720), tilt=40.0, noise=0.0)
    views = [c.copy() for c in poses for _ in range(8)]                 # 10 poses, 8 identical frames each
    idx = lens.select_diverse(views, (960, 720), 40)
    assert len(set(idx)) == len(idx), f"the same index twice: {sorted(idx)}"
    assert len(idx) == 10, len(idx)
    lab = lens.pose_clusters(views, (960, 720))
    assert len(set(lab.tolist())) == 10
    prof = lens.calibrate_lens(views, PATTERN, SQUARE, (960, 720), "standard")
    rep = prof.report
    print(f"      80 views of 10 poses: verdict {rep['verdict']}, n_views {rep['n_views']}, poses {rep.get('n_poses')}, "
          f"tilted {rep['tilted_views']}")
    assert rep["verdict"] != "good", rep["verdict_reasons"]
    assert rep["n_poses"] == 10 and rep["tilted_views"] <= 10
    # random views stay distinct: nothing is merged that should not be
    rnd = synth_boards(K_STD, D_STD, 30, seed=22, size=(960, 720))
    assert len(set(lens.pose_clusters(rnd, (960, 720)).tolist())) == 30


def _fits_stub(K_f, D_f, K_s, D_s, drop_f=(3, 17), drop_s=()):
    def fake_fit(cl, pattern, square, size_, fisheye):
        n = len(cl)
        if fisheye:
            pv = [0.14] * n
            if n == 40:
                for k in drop_f:
                    pv[k] = 4.0
            return 0.145, K_f, D_f, pv, [30.0] * n
        pv = [0.138] * n
        if n == 40:
            for k in drop_s:
                pv[k] = 3.0
        return 0.138, K_s, D_s, pv, [30.0] * n
    return fake_fit


@check("I178 lens auto: both models are scored on the SAME views (and the dropped views are named neutrally)")
def _():
    K_f = np.array([[520.0, 0, 480], [0, 520.0, 360], [0, 0, 1.0]])
    D_f = np.array([0.05, 0.01, 0.0, 0.0])
    boards = synth_boards(K_STD, D_STD, 44, seed=4, size=(960, 720))
    # the standard model drops five views it cannot follow and "wins" 0.138 vs 0.145 on its own
    # fit; on the same views its typical error is 0.28 against the fisheye's 0.14
    fake_pve = lambda views, pattern, square, prof: np.full(len(views), 0.14 if prof.fisheye else 0.28)   # noqa: E731
    with patched(lens, "_fit", _fits_stub(K_f, D_f, K_STD, D_STD, drop_f=(3, 17), drop_s=(1, 5, 9, 13, 21))), \
            patched(lens, "per_view_errors", fake_pve):
        prof = lens.calibrate_lens(boards, PATTERN, SQUARE, (960, 720), "auto")
    rep = prof.report
    print(f"      chosen {rep['model']}, scores on the same views {rep['rms_by_model']}")
    assert prof.fisheye and rep["model"] == "fisheye", rep["rms_by_model"]
    assert abs(rep["rms_by_model"]["standard"] - 0.28) < 1e-9 and abs(rep["rms_by_model"]["fisheye"] - 0.14) < 1e-9
    assert any("cannot follow" in r for r in rep["verdict_reasons"]), rep["verdict_reasons"]
    assert not any("blur or a mis-detected board)" in r for r in rep["verdict_reasons"])


@check("G117 lens auto: a better-fitting fisheye that RUNS AWAY is rejected out loud")
def _():
    size = (2704, 1520)
    Kh = np.array([[1316.4, 0, 1357.73], [0, 1315.9, 769.30], [0, 0, 1.0]])
    Dh = np.array([0.07591, -0.12884, 0.30894, -0.21558])
    Ks = np.array([[1316.3, 0, 1360.24], [0, 1315.6, 768.81], [0, 0, 1.0]])
    Ds = np.array([-0.2595, 0.07982, 0, 0, 0])
    boards = synth_boards(Ks, Ds, 20, seed=5, size=size)
    fake_pve = lambda views, pattern, square, prof: np.full(len(views), 0.10 if prof.fisheye else 0.20)   # noqa: E731
    with patched(lens, "_fit", _fits_stub(Kh, Dh, Ks, Ds, drop_f=())), patched(lens, "per_view_errors", fake_pve):
        prof = lens.calibrate_lens(boards, PATTERN, SQUARE, size, "auto")
    rep = prof.report
    assert not prof.fisheye and rep["model"] == "standard"
    info = rep["fisheye_runaway_rejected"]
    assert info and 0.0 < info["border_valid_frac"] < 1.0, info
    txt = " ".join(rep["verdict_reasons"])
    assert "fisheye model fitted these boards better" in txt and "runs away" in txt, rep["verdict_reasons"]
    print(f"      rejected: fisheye {info['fisheye_score_px']:.2f} px vs standard {info['standard_score_px']:.2f} px, "
          f"{100 * (1 - info['border_valid_frac']):.0f} % of the border lost")


@check("I250 lens: verdict bands scale with the longer side; one rms_limits for the report and the tiles")
def _():
    assert lens.rms_limits(2160, 3840) == lens.rms_limits(3840, 2160) == (1.2, 2.4)
    assert lens.rms_limits(1920, 1080) == (0.6, 1.2) and lens.rms_limits(640, 480) == (0.6, 1.2)
    K = np.array([[3000.0, 0, 1079.5], [0, 3000.0, 1919.5], [0, 0, 1.0]])
    views = synth_boards(K, np.zeros(5), 20, seed=6, size=(2160, 3840))
    prof = lens.LensProfile(2160, 3840, K, np.zeros(5), False, 0.9, 20)
    rep = lens.lens_report(prof, views, [0.9] * 20, [25.0] * 20, {"standard": 0.9}, {}, "standard")
    assert "clean fit" in rep["verdict_reasons"][0], rep["verdict_reasons"][0]
    from kinetrace import boardreview as brv
    from kinetrace import theme
    assert brv.quality(0.9, (3840, 2160))[1] == theme.GREEN and brv.quality(0.9, (1920, 1080))[1] != theme.GREEN
    assert brv.quality(1.1, (1920, 1080))[1] != theme.RED and brv.quality(1.3, (1920, 1080))[1] == theme.RED


@check("I222 lens: the printable board saves under a non-ASCII path")
def _():
    d = os.path.join(SCRATCH, "José été")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "board.png")
    if os.path.exists(p):
        os.remove(p)
    wmm, hmm = lens.save_checkerboard_png(p)
    data = open(p, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"pHYs" in data[:80] and wmm < 297 and hmm < 210
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None and lens.detect_board(cv2.resize(img, None, fx=0.4, fy=0.4), PATTERN) is not None


@check("R19 lens: one lens-file dispatch, one content key, one outlier rule, flags by name")
def _():
    k = lens.profile_key
    a = lens.LensProfile(640, 480, K_STD.copy(), D_STD.copy(), False, 0.1, 5)
    b = copy.deepcopy(a)
    assert k(a) == k(b) and k(None) is None and hash(k(a)) == hash(k(b))
    b.K[0, 0] += 0.5
    assert k(a) != k(b)
    argus = os.path.join(SCRATCH, "two_lines.txt")
    open(argus, "w").write("1 600 640 480 319.5 239.5 1 0 0 0 0 0\n2 620 640 480 322.5 237.5 1 0 0 0 0 0\n")
    prof, which = lens.read_lens_for(argus, 1, "cam2")
    assert prof.f_square == 620 and "camera 2" in which
    none, why = lens.read_lens_for(argus, 5, "cam6")
    assert none is None and "no line for camera 6" in why
    jf = os.path.join(SCRATCH, "k_probe.klens.json")
    a.save(jf)
    p2, w2 = lens.read_lens_for(jf, 0)
    assert w2 == "" and np.allclose(p2.K, a.K)
    e = np.array([0.1, 0.1, 0.12, 0.11, 5.0, np.nan])
    assert abs(lens.outlier_limit(e, 0.5) - 0.5) < 1e-12 and lens.outlier_limit(e * 10, 0.5) > 1.0
    assert lens._fisheye_flags() > 0 and lens.MAX_VIEWS == 40
    # a size mismatch is ONE sentence from lens.size_mismatch
    assert lens.size_mismatch((640, 480), (1280, 960), "cam1") and not lens.size_mismatch((640, 480), (640, 480))


# ============================================================ wanddata (no Qt)
print("-- wanddata")


@check("R20 wanddata: one column sampled, bit-identical to copying the whole arrays, and faster")
def _():
    from kinetrace.calib import sample_tracks_at
    from kinetrace.project import Project
    from kinetrace.session import TrackingSession
    T, N = 40000, 30
    sess = []
    rng = np.random.RandomState(2)
    for c in range(2):
        s = TrackingSession(os.path.join(SCRATCH, f"col_{c}.mp4"), T, 30.0, 1280, 720)
        for j in range(N):
            s.add_landmark(f"pt{j}")
        s.tracks[:] = rng.uniform(0, 1000, s.tracks.shape)
        s.tracked[:, :] = rng.rand(T, N) < 0.9
        s.occluded[:, :] = rng.rand(T, N) < 0.02
        s.tracks[rng.rand(T, N) < 0.001] = np.nan
        sess.append(s)
    proj = Project(sess, ["a", "b"], [0, 3])
    t = np.arange(0, T - 10, 7)

    def reference(name):
        out = np.full((len(t), 2, 2), np.nan)
        for c, s in enumerate(proj.sessions):
            j = s.pid_by_name(name)
            rt = np.asarray(s.tracks, np.float64).copy()
            ok = s.exportable & np.isfinite(s.tracks).all(axis=2)
            rt[~ok] = np.nan
            rt = rt[:, j:j + 1]
            local = proj.rates[c] * np.asarray(t, np.float64) + proj.offsets[c]
            out[:, c] = sample_tracks_at(rt, local)[:, 0]
        return out
    t0 = time.time()
    ref = [reference(f"pt{j}") for j in range(6)]
    t_ref = (time.time() - t0) / 6
    t0 = time.time()
    new = [wd._sample_name(proj, f"pt{j}", t) for j in range(6)]
    t_new = (time.time() - t0) / 6
    for r_, n_ in zip(ref, new):
        assert np.array_equal(r_, n_, equal_nan=True)
    print(f"      40k x 30, 2 cameras: whole-array copy {t_ref * 1000:.1f} ms per landmark, one column "
          f"{t_new * 1000:.1f} ms ({t_ref / max(t_new, 1e-9):.1f}x)")
    assert t_new < t_ref, (t_new, t_ref)
    assert wd.raw_track(sess[0], 3).shape == (T, 2)


# ===================================================================== Qt checks
print("-- wizards (offscreen)")
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QMessageBox, QTextBrowser, QWizard  # noqa: E402
import shiboken6  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication.instance() or QApplication([])

from kinetrace import boardreview as brv  # noqa: E402
from kinetrace import calibwizard as cw  # noqa: E402
from kinetrace import lenswizard as lw  # noqa: E402
from kinetrace import theme  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def gone(obj, timeout=2.0):
    """True once the C++ object behind `obj` has been deleted (deleteLater delivered)."""
    end = time.time() + timeout
    while time.time() < end:
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        if not shiboken6.isValid(obj):
            return True
        time.sleep(0.02)
    return not shiboken6.isValid(obj)


def mem_project(names, sizes, T=200, tag="rc", landmarks=("wand A", "wand B")):
    ss = []
    for c, (w, h) in enumerate(sizes):
        s = TrackingSession(os.path.join(SCRATCH, f"{tag}_{c}.mp4"), T, 60.0, w, h)
        for nm in landmarks:
            s.add_landmark(nm)
        for f in range(120):
            for k, nm in enumerate(landmarks):
                j = s.pid_by_name(nm)
                s.tracks[f, j] = (w * 0.3 + 2.0 * f + 60 * k + 11 * c, h * 0.4 + 1.5 * f + 20 * k)
                s.tracked[f, j] = True
        ss.append(s)
    return Project(ss, list(names), [0] * len(sizes))


class FakeThread:
    """What a page's `_thread` looks like to a wizard's done(): records how it was stopped."""

    def __init__(self):
        self.calls = []
        self.running = True

    def cancel(self):
        self.calls.append("cancel")

    def isRunning(self):
        return self.running

    def wait(self, *a):
        self.calls.append(("wait",) + a)
        self.running = False
        return True


@check("I200 wizards: close = cancel, then wait with NO cap (lens wizard, wand wizard, frame reader)")
def _():
    wiz = lw.LensWizard(None, None, "", None, SCRATCH)
    th = FakeThread()
    wiz.page_video._thread = th
    rd = FakeThread()
    wiz.page_review.review._reader = rd
    wiz.done(0)
    assert th.calls == ["cancel", ("wait",)], th.calls               # old: ("wait", 15000)
    assert rd.calls == [("wait",)], rd.calls
    wiz.deleteLater()
    proj = mem_project(["c1", "c2"], [(640, 480)] * 2)
    wz = cw.WandWizard(None, proj, SCRATCH)
    th2 = FakeThread()
    wz.page_run._thread = th2
    wz.done(0)
    assert th2.calls == ["cancel", ("wait",)], th2.calls
    wz.deleteLater()


@check("G112 wand wizard: a program fault is a sentence + a log line; a WandError is shown as is")
def _():
    import kinetrace.wand as wmod
    got = []
    recs = []

    class H(logging.Handler):
        def emit(self, r):
            recs.append(r.getMessage())
    h = H()
    lg = logging.getLogger("kinetrace.errors")
    lg.addHandler(h)
    try:
        th = cw._CalibThread({}, "none", None, None)
        th.error.connect(lambda m, g: got.append(m))
        with patched(wmod, "calibrate_wand", lambda **kw: (_ for _ in ()).throw(IndexError("index 3 is out of bounds"))):
            th.run()
        with patched(wmod, "calibrate_wand", lambda **kw: (_ for _ in ()).throw(wmod.WandError("camera 2 sees too few points"))):
            th.run()
    finally:
        lg.removeHandler(h)
    assert len(got) == 2, got
    assert got[0].startswith("The calibration could not be completed") and "IndexError" in got[0], got[0]
    assert got[1] == "camera 2 sees too few points", got[1]
    assert any("The calibration could not be completed" in r for r in recs), recs


@check("G113 wand wizard: saving files into a blocked place says what happened and what was written")
def _():
    proj = mem_project(["alpha", "beta"], [(640, 480)] * 2)
    wz = cw.WandWizard(None, proj, SCRATCH)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1.0]])
    R, t = look_at(np.array([2.0, 0.5, 1.0]), np.zeros(3))
    cal = Calibration([CameraCalibration(dlt_from_camera(K, R, t), 640, 480, NoUndistort(), 0.0) for _ in range(2)],
                      "m", "test")
    wz.result = types.SimpleNamespace(to_calibration=lambda: cal, report={"verdict": "good", "verdict_reasons": [],
                                                                          "camera_names": ["alpha", "beta"]})
    wz.gravity = None
    d = os.path.join(SCRATCH, "blocked")
    os.makedirs(os.path.join(d, "x_dltCoefs.csv"), exist_ok=True)      # a FOLDER where the csv must go
    for f in ("x.kcal.json",):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    page = wz.page_run
    with patched(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (os.path.join(d, "x.kcal.json"), ""))):
        page._save()                                                    # old: the error escapes the slot
    txt = page.status.text()
    print("      " + txt[:150].replace("\n", " "))
    assert "could not be saved" in txt and "x.kcal.json" in txt and "incomplete" in txt, txt
    wz.deleteLater()


@check("G137 wand wizard: gravity bands = the sentences' (2 % / 5 %); the first camera is named; one frame wording")
def _():
    assert cw.gravity_colour(1.5) == theme.GREEN and cw.gravity_colour(2.0) == theme.GREEN
    assert cw.gravity_colour(3.0) not in (theme.GREEN, theme.RED) and cw.gravity_colour(5.0) != theme.RED
    assert cw.gravity_colour(6.0) == theme.RED
    proj = mem_project(["alpha", "beta"], [(640, 480)] * 2)
    wz = cw.WandWizard(None, proj, SCRATCH)
    assert "alpha" in wz.page_frame.r_none.text() and "camera 1" not in wz.page_frame.r_none.text()
    html = cw.report_html({"verdict": "ok", "verdict_reasons": []}, {"applied": False, "why": "it rested"}, proj, "m")
    assert "alpha's axes" in html and "camera 1" not in html
    intro = wz.page_intro.findChild(QTextBrowser).toPlainText()
    assert "30 is the minimum" not in intro and "10 frames" in intro and "30" in intro, intro[:50]
    wz.show()
    wz.restart()
    pump(0.05)
    wz.next()
    pump(0.05)
    assert wz.currentPage() is wz.page_wand
    wz.page_wand.length.setValue(0.5)
    assert wz.page_wand._n_frames >= 100
    wz.page_run.initializePage()
    assert "alpha's axes" in wz.page_run.summary.text(), wz.page_run.summary.text()
    wz.deleteLater()


def lens_profile_for(K, size, dist=None):
    return lens.LensProfile(size[0], size[1], K, np.zeros(5) if dist is None else dist, False, 0.2, 30, "test")


@check("I221 / I247 / I248 / R20 wand wizard: NaN focal for the unprofiled, raw coverage, frame ids, content key")
def _():
    import kinetrace.wand as wmod
    sizes = [(1920, 1080)] * 4
    proj = mem_project(["c1", "c2", "c3", "c4"], sizes)
    Kp = np.array([[1400.0, 0, 959.5], [0, 1400.0, 539.5], [0, 0, 1.0]])
    proj.lenses[0] = lens_profile_for(Kp, sizes[0], np.array([-0.25, 0.07, 0, 0, 0]))   # a strong lens
    proj.lenses[1] = lens_profile_for(Kp, sizes[1])
    wz = cw.WandWizard(None, proj, SCRATCH)
    wz.show()
    wz.restart()
    pump(0.05)
    for pg in (wz.page_wand, wz.page_cams, wz.page_frame, wz.page_run):
        while wz.currentPage() is not pg:
            wz.next()
            pump(0.03)
        if pg is wz.page_wand:
            wz.page_wand.length.setValue(0.5)
    pc = wz.page_cams
    f = pc.focal()
    assert pc.r_auto.isChecked() and f is not None
    assert np.isfinite(f[0]) and abs(f[0] - 1400.0) < 1e-6 and np.isnan(f[2]) and np.isnan(f[3]), f   # old: 1920.0
    assert pc.focal_free_per_camera() is True
    seen = {}

    def capture(**kw):
        seen.update(kw)
        raise wmod.WandError("stub: kwargs captured")
    with patched(wmod, "calibrate_wand", capture):
        wz.page_run._run()
        t0 = time.time()
        while not wz.page_run.btn_run.isEnabled() and time.time() - t0 < 30:
            pump(0.05)
    assert seen, "the run never reached calibrate_wand"
    uv_raw, frames = wd.collect_wand(proj, "wand A", "wand B", max_frames=400)
    assert np.array_equal(seen["frame_ids"], frames) and len(frames) == 120
    assert np.allclose(seen["coverage_uv"], uv_raw, equal_nan=True), "coverage is measured on the raw clicks"
    assert not np.allclose(seen["wand_uv"][:, 0], uv_raw[:, 0], equal_nan=True), "camera 1 was straightened"
    assert np.allclose(seen["wand_uv"][:, 2], uv_raw[:, 2], equal_nan=True), "camera 3 has no lens"
    assert np.isnan(seen["focal"][2]) and np.isfinite(seen["focal"][0])
    # the run summary names the cameras whose focal length is guessed
    wz.page_run.initializePage()
    assert "found automatically for c3, c4" in wz.page_run.summary.text(), wz.page_run.summary.text()
    # settings keyed by CONTENT: NaN compares equal to itself, a reloaded profile copy is the same lens
    k1 = wz.page_run._settings_key()
    assert k1 == wz.page_run._settings_key()
    proj.lenses[0] = copy.deepcopy(proj.lenses[0])
    assert wz.page_run._settings_key() == k1, "a copy of the same profile is not a new setting"
    proj.lenses[0].K = proj.lenses[0].K * np.array([[1.01], [1.01], [1.0]])
    assert wz.page_run._settings_key() != k1, "an edited profile is"
    wz.reject()
    wz.deleteLater()


@check("I249 wand wizard: the lens wizard opened from the Cameras page is released after use")
def _():
    proj = mem_project(["c1", "c2"], [(640, 480)] * 2)
    wz = cw.WandWizard(None, proj, SCRATCH)
    made = []
    real_init = lw.LensWizard.__init__

    def spy(self, *a, **k):
        real_init(self, *a, **k)
        made.append(self)
    with patched(lw.LensWizard, "__init__", spy), patched(lw.LensWizard, "exec", lambda self: QDialog.Rejected):
        wz.page_cams._calib_lens(0)
    assert len(made) == 1
    assert gone(made[0]), "the lens wizard (and its scan) stays alive for the session"
    wz.deleteLater()


@check("I222 lens wizard: the board and the lens file save under a non-ASCII path; errors are sentences")
def _():
    wiz = lw.LensWizard(None, None, "", None, SCRATCH)
    d = os.path.join(SCRATCH, "José board")
    os.makedirs(d, exist_ok=True)
    target = os.path.join(d, "kinetrace_checkerboard.png")
    if os.path.exists(target):
        os.remove(target)
    with patched(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target, ""))):
        wiz.page_intro._save_board()
    assert os.path.exists(target) and "Saved" in wiz.page_intro.note.text(), wiz.page_intro.note.text()
    with patched(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target, ""))), \
            patched(lens, "save_checkerboard_png", lambda p: (_ for _ in ()).throw(PermissionError(13, "denied", p))):
        wiz.page_intro._save_board()                                # old: the error escapes the slot
    assert "may not write there" in wiz.page_intro.note.text(), wiz.page_intro.note.text()
    wiz.wiz = wiz
    wiz.result_profile = lens_profile_for(K_STD, (960, 720))
    with patched(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target + ".klens.json", ""))), \
            patched(lens.LensProfile, "save", lambda self, p: (_ for _ in ()).throw(PermissionError(13, "denied", p))):
        wiz.page_result._save()
    assert "may not write there" in wiz.page_result.saved.text(), wiz.page_result.saved.text()
    wiz.deleteLater()


def make_clip(path, n=3, size=(64, 48)):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), 40 + 30 * i, np.uint8))
    vw.release()
    return path


def editor_read(double_click):
    """Click a tile (twice if asked) with a SLOW frame read; the corner editor's exec is stubbed to
    Rejected. -> (summary text right after the clicks, the editor dialogs that opened, the review, the scan)"""
    clip = make_clip(os.path.join(SCRATCH, "rc_clip.mp4"))
    corners = [np.column_stack([np.linspace(6, 56, 54), np.linspace(6, 40, 54)])]
    scan = fake_scan(corners, (64, 48), video=clip)
    rv = brv.BoardReview()
    rv.resize(900, 600)
    rv.show()
    rv.set_scan(scan, PATTERN, SQUARE, None, [0], np.array([0.1]))
    pump(0.05)
    import kinetrace.video_source as vsrc
    opened = []
    real_oc = vsrc.open_capture

    def slow_oc(pth):
        time.sleep(0.5)
        return real_oc(pth)
    with patched(vsrc, "open_capture", slow_oc), \
            patched(brv.CornerEditor, "exec", lambda dlg: (opened.append(dlg), QDialog.Rejected)[1]):
        rv._edit(0)
        if double_click:
            rv._edit(0)                                             # a second click while the first frame is read
        said = rv.summary.text()
        t0 = time.time()
        while not opened and time.time() - t0 < 20:
            pump(0.05)
    rv.stop_reader()
    return said, opened, rv, scan


@check("I249 board review: the corner editor (and its full-resolution frame) is released after use")
def _():
    said, opened, rv, scan = editor_read(False)
    assert opened, "the editor never opened"
    assert gone(opened[0]), "the editor (with its full frame) stays alive for the session"        # old: alive
    rv.deleteLater()


@check("G138 board review: a second tile click during a frame read says it is still reading")
def _():
    said, opened, rv, scan = editor_read(True)
    assert "still reading" in said, said                                                             # old: silent
    assert len(opened) == 1, "one frame at a time"
    rv.deleteLater()


@check("G138 corner editor: the hint follows how THIS board's corner 0 was chosen")
def _():
    bgr = np.full((48, 64, 3), 90, np.uint8)

    def hint(pattern, orient):
        n = pattern[0] * pattern[1]
        ed = brv.CornerEditor(None, bgr, np.zeros((n, 2)) + np.arange(n)[:, None], pattern, SQUARE, None, 3, orient)
        txt = ed.findChildren(QLabel)[0].text()
        ed.deleteLater()
        return txt
    assert "NOT the same physical corner" in hint((8, 6), "image")
    assert "beside the black square" in hint((9, 6), "colour") and "NOT" not in hint((9, 6), "colour")
    assert "too little contrast" in hint((9, 6), "image")


@check("G114 board review: a frame that arrives for a hidden review opens no editor")
def _():
    said, opened, rv, scan = editor_read(False)
    rv.hide()
    n_before = len(opened)
    bgr = np.full((48, 64, 3), 90, np.uint8)
    with patched(brv.CornerEditor, "exec", lambda dlg: (opened.append(dlg), QDialog.Rejected)[1]):
        rv._reader_scan = scan
        rv._on_frame(0, bgr, "")
    assert len(opened) == n_before, "an editor opened for a hidden review"
    rv.deleteLater()


@check("R19 corner editor: the frame is a cached pixmap, the markup is painted with QPainter (axes included)")
def _():
    board = synth_boards(K_STD, D_STD, 1, seed=61, size=(960, 720))[0]
    prof = lens.LensProfile(960, 720, K_STD, D_STD, False, 0.1, 10, "t")
    bgr = np.full((720, 960, 3), 110, np.uint8)
    calls = []
    real = lens.draw_board_review
    with patched(lens, "draw_board_review", lambda *a, **k: (calls.append(1), real(*a, **k))[1]):
        ed = brv.CornerEditor(None, bgr, board, PATTERN, SQUARE, prof, 5)
        ed.resize(1000, 760)
        ed.show()
        pump(0.1)
        t0 = time.time()
        for _ in range(5):
            img = ed.view.grab().toImage()
        dt = (time.time() - t0) / 5
        ed.view.update()
        pump(0.05)
    assert not calls, "every repaint re-marked the whole frame with OpenCV"                    # old: called
    assert img.width() > 100 and img.height() > 100
    im = img.convertToFormat(QImage.Format_RGB888)
    px = np.frombuffer(im.constBits(), np.uint8)[:im.height() * im.bytesPerLine()]
    px = px.reshape(im.height(), im.bytesPerLine())[:, :im.width() * 3].reshape(im.height(), im.width(), 3)
    red = ((np.abs(px[..., 0].astype(int) - 240) < 40) & (px[..., 1] < 100) & (px[..., 2] < 100)).sum()
    assert red > 20, f"the board's X axis was not painted ({red})"
    print(f"      repaint {dt * 1000:.1f} ms at {bgr.shape[1]}x{bgr.shape[0]}")
    # a drag changes the corner and the picture repaints without error
    ed.corners[4] += [15.0, -9.0]
    ed._dirty = True
    ed._refresh()
    ed.view.grab()
    assert ed.moved()
    ed.deleteLater()


@check("R19 board review: set_fit API, 'Best spread' through the runner, errors of the profile shown")
def _():
    boards = synth_boards(K_STD, D_STD, 24, seed=31, size=(960, 720))
    scan = fake_scan(boards, (960, 720))
    prof = lens.calibrate_lens(boards, PATTERN, SQUARE, scan.size, "standard")
    rv = brv.BoardReview()
    rv.set_scan(scan, PATTERN, SQUARE, prof, list(range(24)), lens.per_view_errors(boards, PATTERN, SQUARE, prof))
    calls = []
    rv.runner = lambda fn: (calls.append(1), fn())[1]
    rv._auto()
    assert calls == [1], "Best spread must go through the runner (the wizard's off-thread one)"
    assert np.allclose(rv.errors, lens.per_view_errors(boards, PATTERN, SQUARE, prof), equal_nan=True), \
        "the tiles show the errors of the profile they draw"
    prof2 = lens.calibrate_lens(boards[:12], PATTERN, SQUARE, scan.size, "standard")
    errs2 = lens.per_view_errors(boards, PATTERN, SQUARE, prof2)
    rv.set_fit(prof2, errs2, "refitted")
    assert rv.prof is prof2 and np.allclose(rv.errors, errs2, equal_nan=True)
    assert "refitted" in rv.summary.text()
    rv.deleteLater()


def review_wizard(n=60, seed=41, model="standard"):
    boards = synth_boards(K_STD, D_STD, n, seed=seed, size=(960, 720))
    scan = fake_scan(boards, (960, 720))
    wiz = lw.LensWizard(None, None, "", None, SCRATCH)
    wiz.scan, wiz.pattern, wiz.square, wiz.model = scan, PATTERN, SQUARE, model
    wiz.show()
    wiz.page_review.initializePage()
    pump(0.1)
    return wiz, scan


@check("R19 review page: the errors beside the profile are the errors OF that profile")
def _():
    real = lens.auto_select
    # auto_select's own provisional fit scores the views against ITS profile: make that visibly
    # another number than the displayed profile's, as it is whenever the two fits used other views
    skewed = lambda *a, **k: (lambda r: (r[0], r[1] * 2.0, r[2]))(real(*a, **k))      # noqa: E731
    with patched(lens, "auto_select", skewed):
        wiz, scan = review_wizard()
    pg = wiz.page_review
    rv = pg.review
    assert pg.isComplete() and wiz.result_profile is not None
    assert np.allclose(rv.errors, lens.per_view_errors(scan.corners, PATTERN, SQUARE, rv.prof), atol=1e-12,
                       equal_nan=True), "the errors beside the profile are another fit's"
    wiz.reject()
    wiz.deleteLater()


@check("G116 review page: every board ticked by hand is fitted, not thinned to a 40-view spread")
def _():
    wiz, scan = review_wizard()
    pg = wiz.page_review
    rv = pg.review
    rv._set_all(True)
    pg._fit()
    pump(0.1)
    n_used = wiz.result_profile.n_views
    print(f"      all 60 boards ticked: the fit used {n_used}; status: {pg.status.text()[:70]}")
    assert n_used > 40 and len(wiz.used_boards) == n_used, n_used            # old: 40
    assert pg.isComplete()
    wiz.reject()
    wiz.deleteLater()


@check("G114 review page: a corner edit that lands DURING the fit leaves the profile stale")
def _():
    wiz, scan = review_wizard()
    pg = wiz.page_review
    rv = pg.review
    assert pg.isComplete()
    real_off = lw._off_thread

    def sneaky(owner, fn):
        r = real_off(owner, fn)
        rv.corner_rev += 1
        rv.edited.add(0)
        rv.changed.emit()
        return r
    with patched(lw, "_off_thread", sneaky):
        pg._fit()
    assert not pg.isComplete(), "an edit that landed during the fit counted as fitted"
    assert "changed" in pg.status.text()
    pg._fit()                                                                 # fitting again makes it current
    assert pg.isComplete()
    wiz.reject()
    wiz.deleteLater()


@check("G115 lens video page: the scan's board size is used and locked; the lens type is read at Next; a size "
       "change after the scan asks for a new scan")
def _():
    wiz = lw.LensWizard(None, None, "", None, SCRATCH)
    wiz.show()
    wiz.restart()
    pump(0.05)
    wiz.next()
    pump(0.05)
    vp = wiz.page_video
    assert wiz.currentPage() is vp
    vp.path.setText(os.path.join(SCRATCH, "rc_clip.mp4"))
    boards = synth_boards(K_STD, D_STD, 12, seed=51, size=(960, 720))
    ev = threading.Event()

    def slow_scan(path, pattern, max_candidates=240, progress=None, should_cancel=None, downscale_max=1280):
        ev.wait(15)
        return fake_scan(boards, (960, 720))
    with patched(lens, "scan_video", slow_scan):
        vp._run()
        pump(0.2)
        locked = (not vp.cols.isEnabled()) and (not vp.rows.isEnabled())
        vp.cols.setValue(7)                                                  # changed while the scan runs
        ev.set()
        t0 = time.time()
        while not vp.btn_run.isEnabled() and time.time() - t0 < 20:
            pump(0.05)
    assert locked, "the board-size boxes stay enabled while the scan runs"
    assert wiz.scan is not None and tuple(wiz.pattern) == (9, 6), wiz.pattern          # old: (7, 6)
    assert vp.cols.value() == 9, "the box shows the size the boards were found for"
    vp.r_fish.setChecked(True)                                               # chosen AFTER the scan
    assert vp.validatePage()
    assert wiz.model == "fisheye", wiz.model                                 # old: "auto"
    vp.cols.setValue(8)                                                      # a different board: the scan is stale
    assert wiz.scan is None and not vp.isComplete(), "boards found for another size were kept"
    wiz.reject()
    wiz.deleteLater()


print()
print(f"verify_review_calib: {len(FAILS)} failed, total {time.time() - T_ALL:.1f} s")
if FAILS:
    for f in FAILS:
        print("  FAILED:", f)
    sys.exit(1)
print("VERIFY_REVIEW_CALIB PASSED")
