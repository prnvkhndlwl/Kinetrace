"""The 3D core's code-review fixes (2026-10-03): every check here FAILS on the code
as it was before the review and passes now (run against the base modules by
copying this file into a checkout of the base commit's tests/ folder).

  I170  the world the user chooses: world_axes / Calibration.reframed /
        Reconstruction.reframed, right-handed whatever the calibration's
        handedness, cameras and exported points in ONE world
  I169  Import camera offsets re-bases on the project's first camera
  I171  the import dialog resets the pixel convention for a file that is not
        self-described
  I172  the hull: a camera that frames the whole animal votes outside its picture
  I173  3D points: interior blank rows keep their frame
  I174  convert tracks --camera K of an all-cameras file is refused
  I175  the all-cameras xypts sidecar is a real CSV
  I214  rates / offsets of a Project: re-base on the reference, refuse nonsense
  I215  an xyzpts file of a result that starts before frame 0
  I222  lens / camera files under a path with a non-ASCII character
  I227  importer errors are sentences (and the converter exits 2)
  I228  re-track: "the others agree" means under the band AND under half
  I229  .kcal.json keeps origin_shift
  I230  convert import --out gives the new project a new id
  I250  reconstruction verdict bands scale with the long side
  I255  OpenCV .xml camera files
  I258  one frame-rounding rule (Project.local_index)
  G74   import dialog: columns matched by picture size
  G75   estimate_offsets: static landmarks read "flat", not "no shared tracks"
  G76   estimate_offsets: progress, cancel, a long window is subsampled
  R14   one pixel-convention transform, one triangulator, dead code gone

Run: .venv\\Scripts\\python.exe tests\\verify_review_3d.py
"""
import csv
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from kinetrace import calib, calibio, hull, projectfile, retrack  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, NoUndistort, OpenCVUndistort,  # noqa: E402
                             Reconstruction, dlt_from_camera, dlt_project, triangulate, triangulate_batch)
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "review3d")
if os.path.isdir(OUT):
    shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(5)
fails = []


def check(tag, what, fn):
    """Run fn(); an exception (AssertionError, AttributeError, ...) is a FAIL."""
    try:
        fn()
        print(f"  ok    {tag:5s} {what}", flush=True)
    except Exception as e:      # noqa: BLE001
        print(f"  FAIL  {tag:5s} {what}  ({type(e).__name__}: {str(e)[:160]})", flush=True)
        fails.append(f"{tag} {what}")


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


W, H = 1280, 720
K = np.array([[1100.0, 0, W / 2], [0, 1100.0, H / 2], [0, 0, 1]])


def rig(n=3, po=0.0, flip=False, und=None, mirror=False):
    cams = []
    for k in range(n):
        a = 0.6 + 2.0 * k
        pos = np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.7 + 0.35 * k])
        R, t = look_at(pos)
        cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, und or NoUndistort(), po, flip))
    cal = Calibration(cams, "m", "synthetic")
    return cal.reframed(np.diag([1.0, 1.0, -1.0]), np.zeros(3), "mirrored") if mirror else cal


def pairs(cal, X):
    return np.stack([c.project(X) for c in cal.cameras], axis=1)          # (N, C, 2) raw pixels


def tri(cal, uv_raw):
    """3D from raw pixels through a calibration (what Reconstruct does)."""
    cu = np.stack([c.to_calib_frame(uv_raw[:, k]) for k, c in enumerate(cal.cameras)], axis=1)
    return triangulate_batch(np.stack([c.coefs for c in cal.cameras]), cu)[0]


# ============================================================================ I170
print("\nI170 the world the user chooses")
F = np.diag([1.0, 1.0, -1.0])
Xt = rng.uniform(-0.4, 0.4, (30, 3))


def t_axes_basic():
    B, o = calib.world_axes([1, 2, 3], [2, 2, 3], [1, 5, 3], left_handed=False)
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(B) - 1) < 1e-12
    assert np.allclose(B[0], [1, 0, 0]) and np.allclose(B[1], [0, 1, 0]) and np.allclose(B[2], [0, 0, 1])
    # a +Y point that is not perpendicular: made perpendicular to X
    B, o = calib.world_axes([0, 0, 0], [1, 0, 0], [2, 1, 0], left_handed=False)
    assert np.allclose(B[1], [0, 1, 0]) and np.allclose(B[2], [0, 0, 1])
    B, o = calib.world_axes([0, 0, 0], [1, 0, 0], [0, 1, 0], left_handed=True)
    assert abs(np.linalg.det(B) + 1) < 1e-12 and np.allclose(B[2], [0, 0, -1])


def t_axes_errors():
    for args, words in (((np.zeros(3), np.zeros(3), [0, 1, 0]), "same place"),
                        ((np.zeros(3), [1, 0, 0], np.zeros(3)), "same place"),
                        ((np.zeros(3), [1, 0, 0], [2, 0, 0]), "one line"),
                        ((np.zeros(3), [1, 0, 0], [-3, 0.01, 0]), "one line"),
                        ((np.zeros(3), [1, 0, 0], [1, np.nan, 0]), "3D")):
        try:
            calib.world_axes(*args)
        except ValueError as e:
            assert words in str(e), (words, str(e))
        else:
            raise AssertionError(f"not refused: {args}")


def t_reframed_same_pixels():
    for kw in ({}, {"po": 1.0}, {"po": 1.0, "flip": True},
               {"po": 0.0, "und": OpenCVUndistort(K, np.array([-0.2, 0.05, 0.001, -0.001, 0.0]))},
               {"po": 1.0, "flip": True, "und": OpenCVUndistort(K, np.array([-0.1, 0.02, 0, 0, 0.0]))}):
        cal = rig(**kw)
        B, o = calib.world_axes([0.1, 0.2, 0.0], [0.9, 0.3, 0.1], [0.0, 1.5, 0.2], False)
        c2 = cal.reframed(B, o, "world: origin = A, +X toward B, +Y toward C (frame 12)")
        Xn = (Xt - o) @ B.T
        for k in range(len(cal.cameras)):
            d = np.abs(cal.cameras[k].project(Xt) - c2.cameras[k].project(Xn)).max()
            assert d < 1e-6, (kw, k, d)
            assert c2.cameras[k].pixel_origin == cal.cameras[k].pixel_origin and c2.cameras[k].y_flip == cal.cameras[k].y_flip
            assert c2.cameras[k].undistort is cal.cameras[k].undistort
        assert c2.origin_shift is None and "world: origin = A" in c2.source
        # a second reframe replaces the note instead of stacking it
        c3 = c2.reframed(np.eye(3), np.zeros(3), "world: origin = D, +X toward E, +Y toward F (frame 3)")
        assert c3.source.count("world:") == 1 and "origin = D" in c3.source, c3.source


def t_reframed_right_handed():
    left = rig(mirror=True)
    assert calib.world_is_left_handed(left) and calibio.to_models(left).mirrored
    B, o = calib.world_axes([0, 0, 0], [1, 0, 0], [0, 1, 0], left_handed=True)
    c2 = left.reframed(B, o, "world: x")
    assert not calib.world_is_left_handed(c2)
    assert not calibio.to_models(c2).mirrored
    right = rig()
    assert not calib.world_is_left_handed(right)
    assert not calib.world_is_left_handed(right.reframed(*calib.world_axes([0, 0, 0], [1, 0, 0], [0, 1, 0]), "w"))


def t_physical_right_hand():
    """A mirrored (left-handed) world: marks chosen on the floor, +Z must come out as the PHYSICAL
    up (ex x ey in the real world), not the mirror image the coordinates' cross product gives."""
    phys = rig()
    left = phys.reframed(F, np.zeros(3), "mirrored")                 # the same cameras in X_M = F X_W
    A_w, Bx_w, Cy_w = np.array([0.1, -0.1, 0.0]), np.array([0.5, -0.1, 0.0]), np.array([0.1, 0.4, 0.0])
    T_w = np.array([[0.3, 0.2, 0.5]])
    uv = pairs(phys, T_w)                                            # what every camera saw
    assert np.abs(tri(left, uv) - T_w @ F.T).max() < 1e-6            # the mirrored world holds F * truth
    B, o = calib.world_axes(F @ A_w, F @ Bx_w, F @ Cy_w, left_handed=calib.world_is_left_handed(left))
    new = left.reframed(B, o, "world")
    got = tri(new, uv)
    assert np.abs(got - (T_w - A_w)).max() < 1e-6, got                # (0.2, 0.3, +0.5): up is +Z
    # and the unchanged right-handed calibration agrees when given the same marks
    B2, o2 = calib.world_axes(A_w, Bx_w, Cy_w, left_handed=False)
    got2 = tri(phys.reframed(B2, o2, "world"), uv)
    assert np.abs(got2 - got).max() < 1e-6
    # the old cross product without the handedness flag would have made Z point DOWN
    Bw, ow = calib.world_axes(F @ A_w, F @ Bx_w, F @ Cy_w, left_handed=False)
    assert tri(left.reframed(Bw, ow, "w"), uv)[0, 2] < 0


def t_reconstruction_reframed():
    cal = rig(4)
    uv = pairs(cal, Xt)
    xyz, res, n, err = triangulate_batch(np.stack([c.coefs for c in cal.cameras]),
                                         np.stack([c.to_calib_frame(uv[:, k]) for k, c in enumerate(cal.cameras)], 1))
    xyz[3] = np.nan
    rec = Reconstruction(5, ["a"] * 1, xyz.reshape(30, 1, 3), res.reshape(30, 1), n.reshape(30, 1).astype(np.int32),
                         "m", err.reshape(30, 1, 4).astype(np.float32))
    B, o = calib.world_axes([0.1, 0.2, 0.0], [0.9, 0.3, 0.1], [0.0, 1.5, 0.2])
    r2 = rec.reframed(B, o)
    assert r2.t0 == 5 and np.array_equal(r2.residual, rec.residual, equal_nan=True) and np.array_equal(r2.n_cams, rec.n_cams)
    assert np.array_equal(r2.per_cam, rec.per_cam, equal_nan=True) and np.isnan(r2.xyz[3]).all()
    ok = np.isfinite(xyz).all(axis=1)
    assert np.abs(r2.xyz[ok, 0] - (xyz[ok] - o) @ B.T).max() < 1e-12
    # consistent with triangulating through the reframed calibration
    c2 = cal.reframed(B, o, "w")
    again = tri(c2, uv)
    assert np.abs(again[ok] - r2.xyz[ok, 0]).max() < 1e-6
    assert rec.xyz[ok, 0].shape == r2.xyz[ok, 0].shape and not np.shares_memory(rec.xyz, r2.xyz)


def t_origin_on_image_plane():
    cal = rig()
    centre = cal.cameras[1].center()
    try:
        cal.reframed(np.eye(3), centre, "w")
    except ValueError as e:
        assert "camera 2" in str(e), str(e)
    else:
        raise AssertionError("an origin on a camera's image plane was accepted")


def t_same_world_export():
    left = rig(mirror=True)
    for c in left.cameras:
        c.width, c.height = W, H
    models = calibio.to_models(left)
    assert models.mirrored
    P = rng.uniform(-0.3, 0.3, (20, 3))
    rec = Reconstruction(0, [f"p{j}" for j in range(20)], P[None], np.zeros((1, 20)), np.full((1, 20), 3, np.int32))
    for mode in ("with", "without"):
        path = os.path.join(OUT, f"pts_{mode}.csv")
        calibio.write_points3d(rec, path, "kinetrace", models=models if mode == "with" else None)
        back, _ = calibio.read_points3d(path)
        Xe = back.xyz[0]
        d = max(np.abs(m.project(Xe) - c.project(P)).max() for m, c in zip(models.cameras, left.cameras))
        if mode == "with":
            assert d < 0.05, d
        else:
            assert d > 5, d
    # the converter does the same when given the calibration
    kc = os.path.join(OUT, "left.kcal.json")
    with open(kc, "w", encoding="utf-8") as fh:
        json.dump(calibio.calibration_to_kcal(left), fh)
    src = os.path.join(OUT, "pts_src.csv")
    rec.export_csv(src)
    out = os.path.join(OUT, "pts_conv.csv")
    r = subprocess.run([sys.executable, "-m", "kinetrace.convert", "points3d", src, out, "--calibration", kc],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    back, _ = calibio.read_points3d(out)
    d = max(np.abs(m.project(back.xyz[0]) - c.project(P)).max() for m, c in zip(models.cameras, left.cameras))
    assert d < 0.05, d


def t_models_note_true():
    left = rig(mirror=True)
    models = calibio.to_models(left)
    note = " ".join(models.notes)
    assert "mirrored the same way" not in note and "Set World Axes" in note, note
    doc = calibio.__doc__
    assert "write_points3d" in doc and "Set World Axes" in doc


check("I170", "world_axes: orthonormal, right-handed, a +Y point made perpendicular", t_axes_basic)
check("I170", "world_axes refuses coincident / collinear / non-finite points in a sentence", t_axes_errors)
check("I170", "Calibration.reframed: every camera projects every point to the same pixel (to 1e-6 px), "
      "conventions and lens untouched, the note replaces an older one", t_reframed_same_pixels)
check("I170", "a left-handed calibration is right-handed after reframing (to_models.mirrored False)",
      t_reframed_right_handed)
check("I170", "+Z follows the PHYSICAL right-hand rule in a left-handed world", t_physical_right_hand)
check("I170", "Reconstruction.reframed = B (x - o), residuals and counts unchanged, = triangulating reframed",
      t_reconstruction_reframed)
check("I170", "an origin on a camera's image plane is refused naming the camera", t_origin_on_image_plane)
check("I170", "exported points are in the exported cameras' world (write_points3d(models=), convert --calibration)",
      t_same_world_export)
check("I170", "the mirroring note and the module docstring say what really happens", t_models_note_true)

# ============================================================================ I169
print("\nI169 camera offsets")


def stub_project(names, fps=(30.0, 30.0, 30.0)):
    ss = [TrackingSession(os.path.join(OUT, f"v{i}.mp4"), 600, f, 64, 48) for i, f in enumerate(fps)]
    return Project(ss, list(names))


def t_offsets_rebase():
    p = stub_project(["camA", "camB", "camC"])
    f = os.path.join(OUT, "clap.csv")
    open(f, "w", newline="").write("camera,name,offset,rate\n1,camA,120,1\n2,camB,97,1\n3,camC,130,1\n")
    rows = calibio.read_offsets(f, p)
    assert [(v, round(o, 9), r) for v, o, r in rows] == [(0, 0.0, 1.0), (1, -23.0, 1.0), (2, 10.0, 1.0)], list(rows)
    assert rows.rebased and not rows.reference_missing
    # another reference camera and rates: the file's camA runs at half the file's reference's rate
    f2 = os.path.join(OUT, "other_ref.csv")
    open(f2, "w", newline="").write("camera,name,offset,rate\n1,camA,20,0.5\n2,camB,50,1\n3,camC,40,2\n")
    rows = calibio.read_offsets(f2, p)
    assert [(v, round(o, 9), round(r, 9)) for v, o, r in rows] == [(0, 0.0, 1.0), (1, 10.0, 2.0), (2, -40.0, 4.0)], list(rows)
    # the instants agree: file instant t -> camA frame 0.5 t + 20 -> camB frame t + 50
    t_file = 37.0
    fa = 0.5 * t_file + 20
    assert abs((rows[1][2] * fa + rows[1][1]) - (t_file + 50)) < 1e-9
    # a project's own file comes back unchanged
    q = stub_project(["camA", "camB", "camC"])
    q.set_offset(1, 4.25)
    q.set_rate(2, 2.0)
    f3 = os.path.join(OUT, "own.csv")
    calibio.write_offsets(q, f3)
    rows = calibio.read_offsets(f3, q)
    assert [(v, o, r) for v, o, r in rows] == [(0, 0.0, 1.0), (1, 4.25, 1.0), (2, 0.0, 2.0)] and not rows.rebased


def t_offsets_no_reference_row():
    p = stub_project(["camA", "camB", "camC"])
    f = os.path.join(OUT, "no_ref.csv")
    open(f, "w", newline="").write("camera,name,offset\n2,camB,97\n3,camC,130\n")
    rows = calibio.read_offsets(f, p)
    assert rows.reference_missing and [(v, o) for v, o, _ in rows] == [(1, 97.0), (2, 130.0)]


check("I169", "a clap table 'camA 120, camB 97, camC 130' gives [0, -23, +10]; other reference camera / rates "
      "re-based; the project's own file unchanged", t_offsets_rebase)
check("I169", "a file without a row for the first camera says so (reference_missing)", t_offsets_no_reference_row)

# ============================================================================ I171, G74
print("\nI171 / G74 the import dialog")


def dialog_files():
    cams_a = rig(2)
    for c, (w, h) in zip(cams_a.cameras, ((1920, 1080), (848, 480))):
        c.width, c.height = w, h
        c.pixel_origin = 0.0
    kc = os.path.join(OUT, "dlg.kcal.json")
    with open(kc, "w", encoding="utf-8") as fh:
        json.dump(calibio.calibration_to_kcal(cams_a), fh)
    dl = os.path.join(OUT, "dlg_dltCoefs.csv")
    np.savetxt(dl, np.stack([c.coefs for c in cams_a.cameras], 1), delimiter=",", fmt="%.10g")
    return kc, dl


def t_dialog_convention_reset():
    from PySide6.QtWidgets import QApplication
    from kinetrace.view3d import CalibrationDialog
    app = QApplication.instance() or QApplication([])
    kc, dl = dialog_files()
    dlg = CalibrationDialog(None, ["a", "b"], [(1920, 1080), (848, 480)])
    assert dlg.load(kc) and dlg.conv.currentIndex() == 2 and not dlg.conv.isEnabled()
    assert dlg.load(dl)
    assert dlg.conv.currentIndex() == 0 and dlg.conv.isEnabled(), (dlg.conv.currentIndex(), dlg.conv.isEnabled())
    dlg.deleteLater()


def t_dialog_swapped_columns():
    from PySide6.QtWidgets import QApplication
    from kinetrace.view3d import CalibrationDialog
    app = QApplication.instance() or QApplication([])
    kc, _ = dialog_files()
    # the project lists the 848x480 camera FIRST: the file has it second
    dlg = CalibrationDialog(None, ["small", "big"], [(848, 480), (1920, 1080)])
    assert dlg.load(kc)
    assert [cb.currentIndex() for cb in dlg._combos] == [1, 0], [cb.currentIndex() for cb in dlg._combos]
    dlg.deleteLater()


def t_match_columns_rules():
    from kinetrace.view3d import match_columns

    class C:
        def __init__(self, w, h):
            self.width, self.height = w, h
    five = [C(1920, 1080)] * 2 + [C(848, 480)] + [C(1920, 1080)] * 3
    sizes = [(1920, 1080), (1920, 1080), (1920, 1080), (1920, 1080), (1920, 1080), (848, 480)]
    assert match_columns(sizes, five, 6) == [0, 1, 3, 4, 5, 2]
    assert match_columns([(1, 1)] * 3, [C(0, 0)] * 3, 3) == [0, 1, 2]                 # no sizes: identity
    assert match_columns([(10, 10), (20, 20)], [C(20, 20), C(10, 10)], 2) == [1, 0]  # a swap
    assert match_columns([(10, 10)] * 3, [C(10, 10)] * 2, 3)[:2] == [0, 1]           # more cameras than columns


check("I171", "choosing a dltCoefs.csv after a self-described file resets the pixel convention", t_dialog_convention_reset)
check("G74", "a swapped / lone camera's column is matched by picture size", t_dialog_swapped_columns)
check("G74", "match_columns: unique sizes first, shared sizes in order, the rest in order", t_match_columns_rules)

# ============================================================================ I172
print("\nI172 the hull")
R_SPHERE = 0.1


def sphere_mask(cal, n=120000):
    p = rng.uniform(-1, 1, (n, 3)) * R_SPHERE
    p = p[(p ** 2).sum(1) <= R_SPHERE ** 2]
    uv = np.round(cal.project(p)).astype(int)
    m = np.zeros((cal.height, cal.width), np.uint8)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < cal.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cal.height)
    m[uv[ok, 1], uv[ok, 0]] = 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def make_cam(pos, w, h, f, up=np.array([0, 0, 1.0])):
    Km = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    R, t = look_at(np.asarray(pos, float), up=up)
    return CameraCalibration(dlt_from_camera(Km, R, t), w, h, NoUndistort(), 0.0)


def t_hull_tight_top_camera():
    az = np.radians(15)
    side = [make_cam([2 * np.cos(s * az), 2 * np.sin(s * az), 0.0], 640, 480, 800) for s in (-1, 1)]
    top_tight = make_cam([0, 0, 2.0], 160, 160, 1000, up=np.array([0, 1, 0.0]))     # frames the sphere tightly
    top_loose = make_cam([0, 0, 2.0], 640, 480, 1000, up=np.array([0, 1, 0.0]))
    lo, hi = np.full(3, -0.3), np.full(3, 0.3)
    Vs = 4 / 3 * np.pi * R_SPHERE ** 3
    m_side = [sphere_mask(c) for c in side]
    m_tight, m_loose = sphere_mask(top_tight), sphere_mask(top_loose)
    assert not (m_tight[0].any() or m_tight[-1].any() or m_tight[:, 0].any() or m_tight[:, -1].any()), "framed"
    three_loose = hull.carve(side + [top_loose], m_side + [m_loose], lo, hi, 0.01, dilate_px=1)
    three_tight = hull.carve(side + [top_tight], m_side + [m_tight], lo, hi, 0.01, dilate_px=1)
    two = hull.carve(side, m_side, lo, hi, 0.01, dilate_px=1)
    r_loose, r_tight, r_two = (h.volume() / Vs for h in (three_loose, three_tight, two))
    assert r_two > 1.7 * r_loose, (r_two, r_loose)                                   # the sliver is real
    assert abs(r_tight - r_loose) < 0.08 * r_loose, f"tight top camera {r_tight:.2f}x vs loose {r_loose:.2f}x"
    assert r_tight < 1.7, r_tight
    assert three_tight.thin_fraction() < hull.THIN_WARN_FRAC, three_tight.thin_fraction()
    # a camera that is cut by its picture's edge says nothing beyond it: no vote there, and the
    # hull says that fewer than three cameras checked part of its volume
    top_cut = make_cam([0, 0, 2.0], 90, 90, 1000, up=np.array([0, 1, 0.0]))
    m_cut = sphere_mask(top_cut)
    assert m_cut[0].any() or m_cut[-1].any() or m_cut[:, 0].any() or m_cut[:, -1].any(), "cut by the border"
    cut = hull.carve(side + [top_cut], m_side + [m_cut], lo, hi, 0.01, dilate_px=1)
    assert cut.thin_fraction() > hull.THIN_WARN_FRAC, cut.thin_fraction()
    assert cut.volume() > 1.3 * three_tight.volume()
    # an empty mask has no vote outside its picture either
    empty = hull.carve(side + [top_tight], m_side + [np.zeros_like(m_tight)], lo, hi, 0.01, dilate_px=1)
    assert empty.n_views == 3
    # a camera that records no picture size (0 x 0) takes the mask's: the same hull
    blank = CameraCalibration(top_tight.coefs, 0, 0, NoUndistort(), 0.0)
    nosize = hull.carve(side + [blank], m_side + [m_tight], lo, hi, 0.01, dilate_px=1)
    assert nosize.n_voxels == three_tight.n_voxels, (nosize.n_voxels, three_tight.n_voxels)


def t_hull_lens_camera_unaffected():
    """With a lens model the voxels beyond the UNDISTORTED CANVAS but inside the picture are not
    misses: a camera that frames the animal in a barrel-distorted picture gives the same hull as
    before wherever the voxel is on the picture."""
    az = np.radians(30)
    side = [make_cam([2 * np.cos(s * az), 2 * np.sin(s * az), 0.0], 640, 480, 800) for s in (-1, 1, 0)]
    masks = [sphere_mask(c) for c in side]
    lo, hi = np.full(3, -0.25), np.full(3, 0.25)
    h0 = hull.carve(side, masks, lo, hi, 0.01, dilate_px=1)
    Kc = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    lensed = [CameraCalibration(c.coefs, c.width, c.height, OpenCVUndistort(Kc, np.array([-0.05, 0.0, 0, 0, 0])), 0.0)
              for c in side]
    h1 = hull.carve(lensed, masks, lo, hi, 0.01, dilate_px=1)
    assert h1.n_voxels > 0.5 * h0.n_voxels


check("I172", "a tightly framing top camera: the volume comes near the 3-camera value (was ~2x), "
      "thin_fraction low; an edge-cut camera is reported thin; two cameras alone are the sliver",
      t_hull_tight_top_camera)
check("I172", "a lens-model camera does not turn its cropped-canvas voxels into misses", t_hull_lens_camera_unaffected)

# ============================================================================ I173
print("\nI173 3D points")


def t_points_blank_rows():
    head = "pt1_X,pt1_Y,pt1_Z,pt2_X,pt2_Y,pt2_Z"
    rows = ["1,2,3,4,5,6", "2,3,4,5,6,7", ",,,,,", "4,5,6,7,8,9", "", "6,7,8,9,10,11", ",,,,,", "", ""]
    f = os.path.join(OUT, "blank_xyzpts.csv")
    open(f, "w", newline="").write(head + "\n" + "\n".join(rows) + "\n")
    rec, notes = calibio.read_points3d(f)
    assert rec.t0 == 0 and rec.n_frames == 6, (rec.t0, rec.n_frames)                 # trailing blanks only dropped
    assert np.isnan(rec.xyz[2]).all() and np.isnan(rec.xyz[4]).all()
    assert np.allclose(rec.xyz[3, 0], [4, 5, 6]) and np.allclose(rec.xyz[5, 1], [9, 10, 11])
    # a Kinetrace CSV names its frame in a column: a blank row there is just dropped
    g = os.path.join(OUT, "blank_kinetrace.csv")
    open(g, "w", newline="").write("frame,a_X,a_Y,a_Z\n10,1,2,3\n\n12,4,5,6\n")
    rec2, _ = calibio.read_points3d(g)
    assert rec2.t0 == 10 and rec2.n_frames == 3 and np.isnan(rec2.xyz[1]).all() and np.allclose(rec2.xyz[2, 0], [4, 5, 6])


check("I173", "an xyzpts frame written as empty cells keeps its row (later frames do not shift)", t_points_blank_rows)

# ============================================================================ I174 / I230
print("\nI174 / I230 the converter")


def convert(*args):
    r = subprocess.run([sys.executable, "-m", "kinetrace.convert", *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr).strip()


def t_convert_all_cameras_refused():
    f = os.path.join(OUT, "allcams_xypts.csv")
    lines = ["pt1_cam1_X,pt1_cam1_Y,pt1_cam2_X,pt1_cam2_Y"] + [f"{10 + k},{20 + k},{30 + k},{40 + k}" for k in range(8)]
    open(f, "w", newline="").write("\n".join(lines) + "\n")
    c1, o1 = convert("tracks", f, os.path.join(OUT, "ac1.csv"), "--to", "dlc", "--size", "640x480", "--frames", 8,
                     "--camera", 1)
    assert c1 == 0, o1                                                               # the reference camera is fine
    c2, o2 = convert("tracks", f, os.path.join(OUT, "ac2.csv"), "--to", "dlc", "--size", "640x480", "--frames", 8,
                     "--camera", 2)
    assert c2 == 2 and "convert import" in o2 and o2.startswith("error:"), (c2, o2)


def tiny_video(path, n=12, w=64, h=48):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for k in range(n):
        vw.write(np.full((h, w, 3), 20 + k, np.uint8))
    vw.release()
    return path


def t_convert_import_new_id():
    vid = tiny_video(os.path.join(OUT, "tiny.mp4"))
    s = TrackingSession(vid, 12, 30.0, 64, 48)
    proj = Project([s], ["cam1"])
    src = os.path.join(OUT, "src_proj.kinetrace")
    projectfile.save(proj, src)
    sid = projectfile.read_meta(src).get("project_id")
    dlc = os.path.join(OUT, "tracks_dlc.csv")
    rows = [["scorer", "x", "x", "x"], ["bodyparts", "nose", "nose", "nose"], ["coords", "x", "y", "likelihood"]]
    rows += [[k, 10.0 + k, 20.0, 0.9] for k in range(12)]
    with open(dlc, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    new = os.path.join(OUT, "new_proj.kinetrace")
    code, out = convert("import", src, "--tracks", dlc, "--out", new)
    assert code == 0, out
    nid = projectfile.read_meta(new).get("project_id")
    assert sid and nid and sid != nid, (sid, nid)
    # importing in place keeps the project's identity
    code, out = convert("import", src, "--tracks", dlc)
    assert code == 0 and projectfile.read_meta(src).get("project_id") == sid, out


check("I174", "convert tracks --camera 2 of an all-cameras file is refused and points at `convert import`",
      t_convert_all_cameras_refused)
check("I230", "convert import --out NEW gives the new project a new id (in place keeps it)", t_convert_import_new_id)

# ============================================================================ I175 / I214
print("\nI175 / I214 the project")


def t_sidecar_quoted():
    ss = [TrackingSession(os.path.join(OUT, f"sc{i}.mp4"), 30, 30.0, 64, 48) for i in range(2)]
    for s in ss:
        s.add_landmark("tail, tip")
        s.add_landmark("cam1")
    p = Project(ss, ["left", "right"])
    path = os.path.join(OUT, "multi_xypts.csv")
    p.export_multi_dltdv(path)
    side = os.path.join(OUT, "multi_xypts_pointnames.csv")
    with open(side, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["name", "cameras"]
    assert rows[1] == ["tail, tip", "2"] and rows[2] == ["cam1", "2"], rows[:4]
    assert rows[3][0] == "convention" and len(rows[3]) == 2 and rows[4][0] == "rows" and len(rows[4]) == 2
    assert [r[0] for r in rows[5:]] == ["cam1", "cam2"] and all(len(r) == 2 for r in rows[5:]), rows[5:]


def t_project_rates():
    ss = [TrackingSession(os.path.join(OUT, f"pr{i}.mp4"), 600, 30.0, 64, 48) for i in range(2)]
    # a file whose first camera carries rate 0.5: re-based, the mapping between the cameras unchanged
    p = Project(ss, ["a", "b"], offsets=[7.0, 10.0], rates=[0.5, 1.0])
    assert p.rates == [1.0, 2.0] and p.offsets[0] == 0.0, (p.rates, p.offsets)
    # old: local_0 = 0.5 t + 7, local_1 = t + 10  ->  local_1 = 2 (local_0 - 7) + 10
    for f in (10, 33, 100):
        assert abs(p.map_frame_exact(0, 1, f) - (2.0 * (f - 7) + 10.0)) < 1e-9, f
    for bad in ({"rates": [1.0, 0.0]}, {"rates": [1.0, float("nan")]}, {"rates": [1.0, -2.0]},
                {"offsets": [0.0, float("nan")]}, {"offsets": [float("inf"), 1.0]}):
        try:
            Project(ss, ["a", "b"], **bad)
        except ValueError as e:
            assert "camera" in str(e), str(e)
        else:
            raise AssertionError(f"accepted {bad}")
    q = Project(ss, ["a", "b"])
    q.set_rate(1, float("nan"))
    q.set_offset(1, float("nan"))
    assert q.rates[1] == 1.0 and q.offsets[1] == 0.0


check("I175", "the all-cameras sidecar is csv-quoted: 'tail, tip' stays one cell, the rows keep their meaning",
      t_sidecar_quoted)
check("I214", "a reference rate other than 1 re-bases the rates; 0 / NaN / negative rates and NaN offsets refused",
      t_project_rates)

# ============================================================================ I215
print("\nI215 xyzpts before frame 0")


def t_points_before_zero():
    names = ["a", "b"]
    xyz = rng.uniform(-1, 1, (10, 2, 3))
    rec = Reconstruction(-3, names, xyz, np.ones((10, 2)), np.full((10, 2), 3, np.int32))
    f = os.path.join(OUT, "neg_xyzpts.csv")
    left = calibio.write_points3d(rec, f, "dltdv")
    assert left == 3, left
    back, _ = calibio.read_points3d(f)
    assert back.t0 == 0 and back.n_frames == 7 and np.allclose(back.xyz, xyz[3:], atol=1e-5)
    assert calibio.write_points3d(Reconstruction(2, names, xyz, np.ones((10, 2)), np.full((10, 2), 3, np.int32)), f,
                                  "dltdv") == 0
    back, _ = calibio.read_points3d(f)
    assert back.n_frames == 12 and np.isnan(back.xyz[:2]).all()
    g = os.path.join(OUT, "neg_anipose.csv")
    calibio.write_points3d(rec, g, "anipose")
    back, _ = calibio.read_points3d(g)
    assert back.t0 == -3 and back.n_frames == 10


check("I215", "a 3D result that starts before reference frame 0 is written from frame 0 and says how many were left out",
      t_points_before_zero)

# ============================================================================ I222 / I227 / I255
print("\nI222 / I227 / I255 calibration and lens files")


def fake_lens():
    from kinetrace import lens
    Kc = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
    return lens.LensProfile(1920, 1080, Kc, np.array([-0.2, 0.05, 0.001, 0.0, 0.0]), False, 0.31, 30, "test")


def t_lens_non_ascii():
    d = os.path.join(OUT, "José Muñoz")
    os.makedirs(d, exist_ok=True)
    prof = fake_lens()
    for name in ("lens.yml", "lens.json"):
        p = os.path.join(d, name)
        calibio.write_lens(prof, p)
        assert os.path.getsize(p) > 50
        back = calibio.read_lens(p)
        assert np.allclose(back.K, prof.K) and np.allclose(back.dist[:5], prof.dist) and back.width == 1920, name
    try:
        calibio.read_lens(os.path.join(d, "missing.yml"))
    except calibio.CalibFormatError as e:
        assert "no such file" in str(e), str(e)
    else:
        raise AssertionError("a missing lens file was read")
    # camera files as well
    cal = rig(2)
    for c in cal.cameras:
        c.width, c.height = W, H
    models = calibio.to_models(cal)
    p = os.path.join(d, "cams.yml")
    calibio.write_opencv(models, p)
    back = calibio.read_cameras(p)
    assert len(back) == 2 and np.allclose(back[0].K, models.cameras[0].K)


def bad_calibration(name, text, words):
    p = os.path.join(OUT, name)
    open(p, "w", newline="").write(text)
    try:
        calibio.load_calibration(p)
    except calibio.CalibFormatError as e:
        assert name in str(e) or all(w in str(e) for w in words), str(e)
        assert all(w in str(e) for w in words), (words, str(e))
        return
    raise AssertionError(f"{name} was read")


def t_importer_sentences():
    bad_calibration("nok.json", json.dumps({"cameras": [{"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, {"t": [1, 0, 0]}]}),
                    ["camera 1", "K"])
    bad_calibration("trunc.json", '{"cameras": [{"K": [[1,0', ["JSON"])
    bad_calibration("junk.yml", "this is: [not\nvalid", ["OpenCV"])
    bad_calibration("empty.mat", "", [])
    bad_calibration("nope.txt", "hello", ["K"])
    # and the converter: one sentence, exit 2, no traceback
    for name in ("nok.json", "trunc.json", "junk.yml", "empty.mat"):
        code, out = convert("info", os.path.join(OUT, name))
        assert code == 2 and "Traceback" not in out and out.startswith("error:"), (name, code, out[:200])


def t_xml_cameras():
    cal = rig(3)
    for c in cal.cameras:
        c.width, c.height = W, H
    models = calibio.to_models(cal)
    p = os.path.join(OUT, "cameras.xml")
    calibio.write_opencv(models, p)
    assert open(p, encoding="utf-8").read().lstrip().startswith("<?xml")
    got = calibio.load_calibration(p)
    assert len(got.cameras) == 3 and got.cameras[0].width == W
    from kinetrace.view3d import CALIB_FILTER
    assert "*.xml" in CALIB_FILTER


def t_kcal_origin_shift():
    cams = [{"K": K, "R": np.eye(3), "t": np.zeros(3), "width": W, "height": H},
            {"K": K, "R": cv2.Rodrigues(np.array([0.0, 0.3, 0.0]))[0], "t": np.array([-1.0, 0.0, 0.2]),
             "width": W, "height": H}]
    cal = Calibration.from_krt(cams, "m", "rig")
    assert cal.origin_shift is not None
    f = os.path.join(OUT, "krt.kcal.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(calibio.calibration_to_kcal(cal), fh)
    back = calibio.load_kcal(f)
    assert back.origin_shift is not None and np.allclose(back.origin_shift, cal.origin_shift)
    m1, m2 = calibio.to_models(cal), calibio.to_models(back)
    assert np.allclose(m1.cameras[1].t, m2.cameras[1].t, atol=1e-9) and np.allclose(m1.shift, m2.shift)
    # an older file without the key still loads
    d = json.load(open(f, encoding="utf-8"))
    d.pop("origin_shift")
    json.dump(d, open(f, "w", encoding="utf-8"))
    assert calibio.load_kcal(f).origin_shift is None


check("I222", "lens and camera files are written / read under a path with a non-ASCII character; a missing "
      "file is said to be missing", t_lens_non_ascii)
check("I227", "a JSON without K, a truncated JSON, a malformed .yml, an empty .mat: a sentence naming what is wrong; "
      "the converter exits 2 with no traceback", t_importer_sentences)
check("I255", "an OpenCV .xml camera file goes to the cameras reader; *.xml is in the file filter", t_xml_cameras)
check("I229", "a K + R/t rig's origin_shift survives .kcal.json", t_kcal_origin_shift)

# ============================================================================ I228, I258
print("\nI228 / I258 re-tracking")


def retrack_rig(noise_ab, slide_px, offsets=(0.0, 0.0, 0.0), n=120):
    rr = np.random.RandomState(77)
    cams = []
    Kc = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    for k, a in enumerate((0.3, 2.4, 4.4)):
        pos = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.5 + 0.4 * k])
        R, t = look_at(pos)
        cams.append(CameraCalibration(dlt_from_camera(Kc, R, t), 640, 480, NoUndistort(), 0.0))
    ss = []
    for c, cal in enumerate(cams):
        s = TrackingSession(os.path.join(OUT, f"rt{c}.mp4"), n, 60.0, 640, 480)
        s.add_landmark("head")
        for f in range(n):
            X = np.array([[0.3 * np.cos(f * 0.05), 0.3 * np.sin(f * 0.05), 0.003 * f]])
            uv = cal.project(X)[0]
            if c < 2:
                uv = uv + rr.normal(0, noise_ab, 2)
            if c == 2 and 30 <= f <= 70:
                uv = uv + [slide_px, 0.0]
            s.tracks[f, 0] = uv
            s.tracked[f, 0] = True
        ss.append(s)
    p = Project(ss, ["camA", "camB", "camC"], list(offsets))
    p.calibration = Calibration(cams)
    p.reconstruction = calib.reconstruct(ss, p.calibration, p.rates, p.offsets, (0, n - 1))
    return p


def t_blame_documented_rule():
    p = retrack_rig(noise_ab=2.0, slide_px=40.0)
    thr = [1.5, 1.5, 1.5]
    sts = retrack.disagreeing_stretches(p, thr)
    c_stretches = [s for s in sts if s.view == 2]
    assert c_stretches, "the slid camera was not flagged"
    out = retrack._blame(p, sts, thr)
    cs = [s for s in out if s.view == 2 and s.full_px > 5.0]                       # the slid stretch itself
    assert cs and all(s.rest_px > 1.5 for s in cs), [(s.rest_px, s.full_px) for s in out]   # the others disagree beyond the band ...
    assert all(s.rest_px <= 0.5 * s.full_px for s in cs), [(s.rest_px, s.full_px) for s in cs]   # ... yet under half the full residual
    assert all(s.reason and "still disagree" in s.reason for s in cs), [s.reason for s in cs]
    # and a clean rig still blames the camera that slid
    q = retrack_rig(noise_ab=0.1, slide_px=40.0)
    out = retrack._blame(q, retrack.disagreeing_stretches(q, thr), thr)
    cs = [s for s in out if s.view == 2]
    assert cs and all(not s.reason for s in cs), [s.reason for s in cs]


def t_half_frame_rounding():
    p = retrack_rig(noise_ab=0.0, slide_px=30.0, offsets=(0.0, 0.0, 0.5))
    # sessions are long enough; with a .5 offset map_frame rounds the tie UP
    assert p.map_frame(0, 2, 30) == 31 == p.local_index(2, 30)
    assert list(p.local_index(2, np.arange(5))) == [p.map_frame(0, 2, t) for t in range(5)] == [1, 2, 3, 4, 5]
    assert [p.reference_index(2, f) for f in (1, 2, 3)] == [0, 1, 2]
    r = p.reconstruction
    sts = retrack.disagreeing_stretches(p, [1.5, 1.5, 1.5])
    c = [s for s in sts if s.view == 2 and s.name == "head"]
    assert c, "no stretch"
    st = c[0]
    assert st.local0 == p.map_frame(0, 2, st.t0) and st.local1 == p.map_frame(0, 2, st.t1), (st.t0, st.local0)
    summ = retrack.cells_summary(p, [st])
    assert list(summ["local"]) == [p.map_frame(0, 2, t) for t in range(st.t0, st.t1 + 1)]


check("I228", "_blame: the others 'agree' only under the band AND under half the full residual", t_blame_documented_rule)
check("I258", "Project.local_index / reference_index use map_frame's tie rule; retrack follows it at .5 offsets",
      t_half_frame_rounding)

# ============================================================================ I250, R14
print("\nI250 / R14")


def t_report_long_side():
    res = np.full((10, 2), 2.5)
    rec = Reconstruction(0, ["a", "b"], np.zeros((10, 2, 3)), res, np.full((10, 2), 3, np.int32))
    portrait = calib.reconstruction_report(rec, 3, 2160.0, 3840.0)
    landscape = calib.reconstruction_report(rec, 3, 3840.0, 2160.0)
    assert portrait["verdict"] == landscape["verdict"] == "good", (portrait["verdict"], landscape["verdict"])
    assert calib.reconstruction_report(rec, 3, 2160.0)["verdict"] == "ok"             # width alone: stricter, as before
    assert calib.residual_bands(1920) == (1.5, 5.0) and calib.residual_bands(1280, 720) == (1.5, 5.0)
    assert calib.residual_bands(3840, 2160) == (3.0, 10.0) and calib.residual_bands(2160, 3840) == (3.0, 10.0)


def t_pixel_convention_one_place():
    for po, flip in ((0.0, False), (1.0, False), (1.0, True), (0.0, True)):
        cam = CameraCalibration(rig(1).cameras[0].coefs, W, H, NoUndistort(), po, flip)
        pts = rng.uniform([0, 0], [W, H], (20, 2))
        q = cam.to_dlt_pixels(pts)
        assert np.allclose(cam.from_dlt_pixels(q), pts, atol=1e-9)
        hom = np.column_stack([q, np.ones(20)]) @ cam.pixel_matrix().T
        assert np.allclose(hom[:, :2], pts, atol=1e-9), (po, flip)
        # the MATLAB-side coefficients project the 1-based pixels of the same points
        L1 = cam.coefs_for_origin(1.0)
        X = rng.uniform(-0.3, 0.3, (10, 3))
        want = cam.to_dlt_pixels(cam.project(X))
        got = dlt_project(L1, X)
        assert np.allclose(got - (1.0 - po), want - 0.0, atol=1e-6) or np.allclose(got, want + (1.0 - po), atol=1e-6)
        # one DLT matrix / denominator / front rule
        assert np.allclose(calib.dlt_denominator(cam.coefs, X), calib.dlt_matrix(cam.coefs)[2] @ np.vstack([X.T, np.ones(10)]))


def t_single_equals_batch():
    cal = rig(5)
    co = np.stack([c.coefs for c in cal.cameras])
    X = rng.uniform(-0.3, 0.3, (15, 3))
    uv = np.stack([dlt_project(c.coefs, X) for c in cal.cameras], 1) + rng.normal(0, 0.5, (15, 5, 2))
    xb, rb, nb, eb = triangulate_batch(co, uv)
    for i in range(15):
        x1, r1, e1 = triangulate(co.T, uv[i])
        assert np.abs(x1 - xb[i]).max() < 1e-9 and abs(r1 - rb[i]) < 1e-9 and np.abs(e1 - eb[i]).max() < 1e-9
    try:
        triangulate(co.T[:, :1], uv[0, :1])
    except ValueError:
        pass
    else:
        raise AssertionError("one camera accepted")


def t_dead_code_gone():
    assert not hasattr(calib, "reference_time") and not hasattr(calib, "sample_track")
    assert not hasattr(Reconstruction, "valid") and not hasattr(Reconstruction, "to_mat_dict")
    assert "reference" not in inspect.signature(calib.estimate_offsets).parameters
    assert calib.PARALLEL_LINES_DEG == 5.0 and calib.WORLD_AXES_MIN_DEG == 5.0


check("I250", "the verdict bands scale with the LONG side: portrait 4K judged like landscape 4K", t_report_long_side)
check("R14", "one pixel-convention transform (matrix / affine / coefficient shift agree), one DLT matrix",
      t_pixel_convention_one_place)
check("R14", "triangulate() is triangulate_batch() for one point", t_single_equals_batch)
check("R14", "dead code removed (reference_time, sample_track, Reconstruction.valid / to_mat_dict, reference=)",
      t_dead_code_gone)

# ============================================================================ G75 / G76
print("\nG75 / G76 estimate_offsets")


def helix_sessions(T, true_off, static=False, empty_cam=None, npts=2):
    cams = rig(3).cameras
    cal = Calibration(cams, "m")
    ss = []
    for c, cam in enumerate(cams):
        n_local = T + 60
        s = TrackingSession(os.path.join(OUT, f"he{c}.mp4"), n_local, 60.0, W, H)
        for j in range(npts):
            s.add_landmark(f"p{j}")
        if c != empty_cam:
            lf = np.arange(n_local)
            t_ref = lf - true_off[c]
            ok = (t_ref >= 0) & (t_ref <= T)
            sc = np.zeros_like(t_ref) if static else t_ref
            for j in range(npts):
                X = np.stack([0.4 * np.cos(sc * 0.05) + 0.05 * j, 0.4 * np.sin(sc * 0.05), 0.002 * sc + 0.03 * j], -1)
                uv = cam.project(X)
                s.tracks[ok, j] = uv[ok]
                s.tracked[ok, j] = True
        ss.append(s)
    return ss, cal


def t_static_is_flat():
    ss, cal = helix_sessions(150, [0.0, 0.0, 0.0], static=True, empty_cam=2)
    rep = {}
    calib.estimate_offsets(ss, cal, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (10, 120), report=rep)
    assert rep["per_view"][1] == "flat" and rep["per_view"][2] == "none", rep["per_view"]
    assert rep["verdict"] == "flat" and "do not move enough to time it" in rep["why"], rep["why"]
    # moving landmarks are still timed (sharp), and a camera with nothing stays "none"
    ss, cal = helix_sessions(150, [0.0, -3.0, 1.0], empty_cam=2)
    rep = {}
    calib.estimate_offsets(ss, cal, [1.0] * 3, [0.0, -3.0, 1.0], (10, 120), report=rep)
    assert rep["per_view"][1] == "sharp" and rep["per_view"][2] == "none", rep["per_view"]
    assert "share no tracked landmark" in rep["why"]


def t_progress_and_cancel():
    ss, cal = helix_sessions(150, [0.0, -3.35, 1.6])
    calls = []
    est, before, after = calib.estimate_offsets(ss, cal, [1.0] * 3, [0.0, -3.0, 2.0], (10, 130),
                                                progress=lambda d, t: calls.append((d, t)))
    assert calls and calls[-1][0] == calls[-1][1] and all(0 <= d <= t for d, t in calls)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)
    assert abs(est[1] + 3.35) < 0.08 and abs(est[2] - 1.6) < 0.08, est
    # cancel after a few evaluations: nothing changes, the verdict says so
    n = [0]

    def stop():
        n[0] += 1
        return n[0] > 5
    rep = {}
    est2, b2, a2 = calib.estimate_offsets(ss, cal, [1.0] * 3, [0.0, -3.0, 2.0], (10, 130), report=rep, cancelled=stop)
    assert est2 == [0.0, -3.0, 2.0] and rep["verdict"] == "cancelled", (est2, rep.get("verdict"))


def t_long_window_subsampled():
    T = 12000
    ss, cal = helix_sessions(T, [0.0, -3.35, 1.6])
    t0 = time.time()
    rep = {}
    est, before, after = calib.estimate_offsets(ss, cal, [1.0] * 3, [0.0, -3.0, 2.0], (10, T - 10), report=rep)
    dt = time.time() - t0
    assert abs(est[1] + 3.35) < 0.08 and abs(est[2] - 1.6) < 0.08, est
    assert rep["n_cells"] > 10 * 1000, rep["n_cells"]
    assert dt < 40, f"{dt:.0f} s"
    prep = calib._Prepared(ss, cal, None)
    t, keep, n_cells = calib._scoring_instants(prep, [1.0] * 3, [0.0, -3.0, 2.0], np.arange(10, T - 10), 1.0)
    assert len(t) <= calib.MAX_SCORE_INSTANTS and keep.shape[0] == len(t), (len(t), keep.shape)
    print(f"        (12000-instant window: {dt:.1f} s, scored on {len(t)} instants)")


check("G75", "static landmarks read 'flat: the landmarks do not move enough to time it'; 'none' only without "
      "shared cells", t_static_is_flat)
check("G76", "estimate_offsets reports progress, cancels with the offsets unchanged", t_progress_and_cancel)
check("G76", "a 12000-instant window is scored on a few hundred moving instants and still recovers the offsets",
      t_long_window_subsampled)

print()
if fails:
    print(f"VERIFY_REVIEW_3D FAILED ({len(fails)}):")
    for f_ in fails:
        print("   -", f_)
    sys.exit(1)
print("VERIFY_REVIEW_3D PASSED")
