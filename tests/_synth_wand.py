"""A synthetic wand rig with KNOWN camera timing, for the wand timing tests
(verify_wand_sync.py, verify_wand_sync_gui.py; I271, I300-I304, G199, G200).

Four (up to seven) 1280 x 720 cameras at 100 fps around a 1 m volume film a 0.5 m
wand. Each camera's true offset is fractional; the project is built with offsets
that are WRONG by `error` frames, as after a whole-frame sound sync. Options:
picture noise (one value, or one per camera), wand ends swapped in one camera on a
few frames, a camera whose real frame rate differs a little from the reference's
(a drift), a dropped frame, a camera that films the wand only in a window of
reference frames, a "straight-line" wand that only ever moves the same way at the
same speed (the timing of such a wand cannot be told from a camera's position),
and static floor marks clicked on ONE frame per camera (the axes of the world).
Deterministic.
"""
from __future__ import annotations

import numpy as np

from kinetrace.calib import CameraCalibration, NoUndistort, dlt_from_camera
from kinetrace.project import Project, half_up
from kinetrace.session import TrackingSession

W, H = 1280, 720
FPS = 100.0
T = 1200                            # reference frames of wand waving
WAND_L = 0.5
FOCAL = [1000.0, 1050.0, 980.0, 1020.0, 990.0, 1010.0, 1030.0]
TRUE_OFFSETS = [0.0, 3.30, -2.60, 1.15]
# three floor marks (the world's origin, +X and a point of the XY plane), all inside every camera's view
MARKS = {"origin": np.array([-0.2, -0.2, 0.55]), "X": np.array([0.3, -0.2, 0.55]),
         "Y": np.array([-0.2, 0.3, 0.55])}


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


def cameras(n=4):
    out, pos = [], []
    for k in range(n):
        a = 0.3 + k * min(2 * np.pi / n, np.pi / 2)        # two cameras: 90 degrees apart, not facing
        p = np.array([2.8 * np.cos(a), 2.8 * np.sin(a), 1.2 + 0.35 * (k % 2)])
        R, t = look_at(p, np.array([0.0, 0.0, 1.0]))
        K = np.array([[FOCAL[k], 0, (W - 1) / 2], [0, FOCAL[k], (H - 1) / 2], [0, 0, 1]])
        out.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))
        pos.append(p)
    return out, np.array(pos)


def wand_varied(t):
    """Both ends at (fractional) reference frame t: a Lissajous centre at changing
    speeds, the wand tumbling - motion in every direction, fast and slow."""
    s = np.asarray(t, np.float64) / FPS
    c = np.stack([0.55 * np.sin(1.9 * s) + 0.15 * np.sin(5.3 * s),
                  0.55 * np.sin(1.3 * s + 1.0) + 0.12 * np.cos(4.1 * s),
                  1.0 + 0.4 * np.sin(1.1 * s) + 0.08 * np.sin(6.7 * s)], -1)
    th, ph = 2.6 * s, 1.7 * s + 0.5
    d = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], -1)
    return c + 0.5 * WAND_L * d, c - 0.5 * WAND_L * d


SWEEP = 120                         # frames per straight sweep (+ a gap with no data before the next)


def wand_straight(t):
    """Every sweep carries the wand the SAME way at the SAME speed (+X at
    0.6 m/s), at a new height / depth / held orientation each time; between
    sweeps it is not tracked. A timing error then shifts every point of a
    camera by one constant vector - exactly what moving that camera does."""
    t = np.asarray(t, np.float64)
    k = np.floor(t / (SWEEP + 30))
    u = (t - k * (SWEEP + 30)) / SWEEP                    # 0..1 along the sweep, > 1 in the gap
    rng = np.random.RandomState(7)
    table = rng.uniform(-1, 1, (64, 5))
    row = table[k.astype(int) % 64]
    c = np.stack([-0.36 + 0.72 * u, 0.4 * row[:, 0], 1.0 + 0.35 * row[:, 1]], -1)
    th, ph = 1.5 * row[:, 2], 0.6 * row[:, 3]
    d = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], -1)
    a, b = c + 0.5 * WAND_L * d, c - 0.5 * WAND_L * d
    gap = u > 1.0
    a[gap] = np.nan
    b[gap] = np.nan
    return a, b


def build(error=(0.0, 0.4, -0.7, 0.0), noise=0.3, swaps=0, swap_cam=1, rate_true=None, straight=False,
          seed=3, true_offsets=TRUE_OFFSETS, video_paths=None, dropped=None, n_ref=T, window=None,
          marks_at=None):
    """(project, truth dict).

    `error` = what is ADDED to the true offsets in the project; `noise` = px of picture noise (one value
    or one per camera); `swaps` = how many frames of camera `swap_cam` have the two ends swapped;
    `rate_true` = per camera the REAL frame-rate ratio (the project assumes 1.0: a drift); `dropped` =
    (camera, local frame): that camera skipped one picture there (every later frame shows one frame
    later); `n_ref` = reference frames of waving; `window` = (camera, t0, t1): that camera sees the wand
    only at reference instants t0..t1; `marks_at` = a reference frame: the three floor marks of `MARKS`
    are clicked in every camera on ONE frame each - the frame the playhead shows at that instant with
    the project's (told) offsets, as a user clicks them; `video_paths` = files to name in the sessions
    (the GUI test writes real videos)."""
    rng = np.random.RandomState(seed)
    C = len(true_offsets)
    cams, pos = cameras(C)
    wand = wand_straight if straight else wand_varied
    rate_true = list(rate_true) if rate_true is not None else [1.0] * C
    noise = list(noise) if np.ndim(noise) else [float(noise)] * C
    offs = [float(o + e) for o, e in zip(true_offsets, error)]
    offs[0] = 0.0
    sessions = []
    n_frames = int(n_ref) + 8
    for c, cal in enumerate(cams):
        path = video_paths[c] if video_paths else f"synthetic_wand_cam{c + 1}.mp4"
        s = TrackingSession(path, n_frames, FPS, W, H)
        ja, jb = s.add_landmark("wand A"), s.add_landmark("wand B")
        f = np.arange(n_frames, dtype=np.float64)
        t = (f - true_offsets[c]) / rate_true[c]          # the reference instant local frame f shows
        if dropped is not None and dropped[0] == c:
            t[f >= dropped[1]] += 1.0
        A, B = wand(t)
        inside = (t >= 0) & (t <= n_ref - 1)
        if window is not None and window[0] == c:
            inside &= (t >= window[1]) & (t <= window[2])
        uva = cal.project(np.nan_to_num(A))
        uvb = cal.project(np.nan_to_num(B))
        ok = inside & np.isfinite(A).all(axis=1)
        if swaps and c == swap_cam:
            idx = rng.choice(np.nonzero(ok)[0], swaps, replace=False)
            uva[idx], uvb[idx] = uvb[idx].copy(), uva[idx].copy()
        for j, uv in ((ja, uva), (jb, uvb)):
            uv = uv + rng.normal(0, noise[c], uv.shape)
            vis = ok & (uv[:, 0] >= 2) & (uv[:, 0] < W - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < H - 2)
            s.tracks[vis, j] = uv[vis]
            s.tracked[vis, j] = True
        s.manual[:] = False
        if marks_at is not None:
            # a static mark clicked once: on the frame shown at reference instant `marks_at` (told offsets)
            fm = int(half_up(float(marks_at) + offs[c]))
            for nm, X in MARKS.items():
                jm = s.add_landmark(nm)
                uv = cal.project(X[None])[0] + rng.normal(0, noise[c], 2)
                s.manual[:, jm] = False
                if 0 <= fm < n_frames:
                    s.tracks[fm, jm] = uv
                    s.tracked[fm, jm] = True
                    s.manual[fm, jm] = True
        s._touch()
        sessions.append(s)
    proj = Project(sessions, [f"cam{c + 1}" for c in range(C)], offs)
    return proj, {"offsets": list(true_offsets), "cams": cams, "pos": pos}


def calibrate_fn(project, focal=None, max_frames=400, views=None):
    """`calibrate(offsets, hint=None) -> WandResult` the way the wizard calls the core: the wand sampled
    at those offsets (one per PROJECT camera), the cameras `views` (None = all; the result's cameras in
    that order), auto focal unless `focal` (per project camera) is given."""
    from kinetrace import wand as _wand
    from kinetrace import wanddata as wd
    vs = list(range(project.n_views)) if views is None else [int(v) for v in views]

    def calibrate(offsets, hint=None):
        uv, frames = wd.collect_wand(wd.retimed(project, offsets), "wand A", "wand B", max_frames=max_frames,
                                     views=None if views is None else vs)
        if hint is not None and focal is None:
            f = hint.K[:, 0, 0].copy()
        else:
            f = None if focal is None else [float(focal[v]) for v in vs]
        return _wand.calibrate_wand(uv, WAND_L, [(W, H)] * len(vs), focal=f, frame_ids=frames,
                                    names=[project.name(v) for v in vs])
    return calibrate
