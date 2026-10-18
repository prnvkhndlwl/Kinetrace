"""Offscreen GUI: the 3D window while a Track run goes (I309), why a landmark has no 3D at an instant (G206), and
the camera rays of a 3D point (G207), each through the user's gestures.

The project: 3 synthetic cameras (camC at a fractional offset), 600 frames, five landmarks:
- "hip": camB has it on every frame, camA only on frames 0-10, camC never - so beyond frame 10 it has 3D only
  once a run tracks it in camA (the white dot in camA's video; Moving spot, CPU);
- "nose": camA only (no 3D anywhere: "only camA has it");
- "tail": camA on frames 0-100, camB on frames 200-300 - never at the same instant (the owner's case);
- "head": all three cameras, camC 15 px off (one camera that disagrees: its ray is the poor one);
- "neck": all three cameras, exact.

Run: .venv\\Scripts\\python.exe tests\\verify_live3d.py
"""
import copy
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, NoUndistort, dlt_from_camera, reconstruct,
                             residual_bands)
from kinetrace.project import Project
from kinetrace.session import TrackingSession

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.RandomState(9)

W, H = 320, 240
N_FR = 600
FPS = 60.0
OFF = [0.0, 3.0, -2.4]
CAMS = ["camA", "camB", "camC"]
NAMES = ["hip", "nose", "tail", "head", "neck"]
FAILS = []


def check(ok, what, detail=""):
    detail = str(detail or "").encode("ascii", "replace").decode("ascii")      # the console is cp1252
    print(("PASS " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILS.append(what)


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[420.0, 0, W / 2], [0, 420.0, H / 2], [0, 0, 1]])
cams = []
for k, a in enumerate((0.3, 2.4, 4.4)):
    R, t = look_at(np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.5 + 0.4 * k]))
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))


def world_at(t_ref):
    s = float(t_ref)
    hip = np.array([0.25 * np.cos(s * 0.02), 0.25 * np.sin(s * 0.02), 0.0005 * s])
    return {"hip": hip, "nose": hip + [0.08, 0.0, 0.03], "tail": hip - [0.08, 0.0, 0.01],
            "head": hip + [0.05, 0.02, 0.05], "neck": hip + [0.03, 0.0, 0.04]}


def has(name, c, f):
    """Which camera / local frame holds a landmark (the module docstring)."""
    if name == "hip":
        return c == 1 or (c == 0 and f <= 10)
    if name == "nose":
        return c == 0
    if name == "tail":
        return (c == 0 and f <= 100) or (c == 1 and 200 <= f <= 300)
    return True


sessions, paths = [], []
for c, cal in enumerate(cams):
    p = os.path.join(SCRATCH, f"live3d_{c}.mp4")
    write_video = not os.path.exists(p)
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)) if write_video else None
    s = TrackingSession(p, N_FR, FPS, W, H)
    for nm in NAMES:
        s.add_landmark(nm)
    for f in range(N_FR):
        t_ref = f - OFF[c]
        pts = world_at(t_ref)
        if vw is not None:
            img = np.full((H, W, 3), 30, np.uint8)
            u = cal.project(pts["hip"][None])[0]
            cv2.circle(img, (int(round(u[0])), int(round(u[1]))), 6, (230, 230, 230), -1)
            vw.write(img)
        for j, nm in enumerate(NAMES):
            if not has(nm, c, f):
                continue
            u = cal.project(pts[nm][None])[0]
            if nm == "head" and c == 2:
                u = u + [15.0, 0.0]
            noise = 0.0 if nm == "neck" else 0.15
            s.set_position(f, j, float(u[0] + rng.normal(0, noise)), float(u[1] + rng.normal(0, noise)))
    if vw is not None:
        vw.release()
    sessions.append(s)
    paths.append(p)
proj = Project(sessions, CAMS, OFF)
proj.calibration = Calibration(cams, unit="m", source="synthetic")
PROJ = os.path.join(SCRATCH, "live3d.kinetrace")
proj.save(PROJ)
for p in paths:
    forget_recovery(p)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(cond, timeout):
    t0 = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t0 > timeout:
            return False
    return True


def select_point(name):
    s = win.session
    win.layers.setCurrentItem(win.layers.point_item(s.pid_by_name(name)))
    pump(0.1)
    return s.pid_by_name(name)


def goto_ref(t_ref):
    """The working camera (camA, offset 0) to reference instant t_ref."""
    win._goto(int(t_ref), force=True)
    pump(0.3)


def pick_combo(cmb, index):
    """Choose an entry the way the user does: click the box, arrow to the entry, Enter."""
    QTest.mouseClick(cmb, Qt.LeftButton)
    pump(0.1)
    view = cmb.view()
    cur = cmb.currentIndex()
    for _ in range(abs(index - cur)):
        QTest.keyClick(view, Qt.Key_Down if index > cur else Qt.Key_Up)
    QTest.keyClick(view, Qt.Key_Return)
    pump(0.2)


win._open_project_from_path(PROJ)
assert wait(lambda: win.state == READY and win.project is not None and win.project.n_views == 3
            and not win._loading, 30), "the project opens"
p = win.project
assert p.calibration is not None and p.calibration.n_usable(3) == 3
pump(1.0)                                    # the automatic 3D result of what was opened
win._set_active_view(0)
pump(0.3)
QTest.keyClick(win, Qt.Key_5, Qt.ControlModifier)          # 3D > 3D View
pump(0.5)
v = win.view3d
assert v is not None and v.isVisible(), "(setup) Ctrl+5 opened the 3D window"
hip_j = lambda sc: sc.names.index("hip") if "hip" in sc.names else None      # noqa: E731

# ---- [I] the 3D window follows a Track run (I309) --------------------------------------------------
print("[I] the 3D window follows a Track run as it goes (I309)")
seen = []                 # (state, instant, hip finite in the scene at the instant)
ticks = []                # (copies of the 2D data, the live overlay) at each live update
_orig_set_scene = v.set_scene


def _probe_set_scene(scene, info=""):
    j = hip_j(scene)
    fin = (j is not None and scene.points is not None and j < len(scene.points)
           and bool(np.isfinite(scene.points[j]).all()))
    seen.append((win.state, win._reference_instant(), fin))
    return _orig_set_scene(scene, info)


v.set_scene = _probe_set_scene
_orig_live = getattr(win, "_live_3d", None)
if _orig_live is not None:
    def _probe_live(*a, **k):
        out = _orig_live(*a, **k)
        lv = getattr(win, "_live3d", None)
        if lv is not None and win.state == TRACKING:
            ticks.append(([copy.deepcopy(s) for s in win.project.sessions], copy.deepcopy(lv)))
        return out
    win._live_3d = _probe_live
r0 = p.reconstruction
k40 = r0.index(40) if r0 is not None else None
check(r0 is None or k40 is None or "hip" not in r0.names or not np.isfinite(r0.xyz[k40, r0.names.index("hip")]).all(),
      "(setup) before the run the hip has no 3D at reference 40")
win.act_pm_spot.trigger()
select_point("hip")
goto_ref(10)
win.act_track_all.setChecked(False)
QTest.keyClick(win, Qt.Key_T)
check(wait(lambda: win.state != READY, 10) and wait(lambda: win.state == READY, 180), "(setup) a tracking run ran")
pump(0.3)
check(bool(win.session.tracked[300, win.session.pid_by_name("hip")]), "(setup) the run tracked the hip past frame 300")
during = [(t, fin) for st, t, fin in seen if st == TRACKING]
live_dots = [t for t, fin in during if fin and t > 12]
check(len(live_dots) >= 3, "during the run the 3D window shows the hip's dot at the instant being tracked",
      f"{len(during)} redraws during the run, {len(live_dots)} with the hip's dot past frame 12")
v.set_scene = _orig_set_scene

# I2: every live update = what Ctrl+3 computes from the same 2D data, at those instants, bit for bit
bad = []
for copies, lv in ticks:
    t0, L = int(lv["t0"]), lv["xyz"].shape[0]
    ref = reconstruct(copies, p.calibration, p.rates, p.offsets, (t0, t0 + L - 1), names=list(lv["names"]))
    for key, a in (("xyz", ref.xyz), ("resid", ref.residual), ("n", ref.n_cams), ("err", ref.per_cam)):
        if not np.array_equal(np.asarray(lv[key]), a, equal_nan=True):
            bad.append(f"t0={t0} {key}")
check(len(ticks) >= 3 and not bad, "every live update equals reconstruct on the data of that moment (same bits)",
      f"{len(ticks)} live updates; differ: {bad[:5]}")
# I4: after the run the overlay is gone and the saved result is the full one
pump(1.0)
f_ = reconstruct(p.sessions, p.calibration, p.rates, p.offsets, win._t_range_3d())
r = p.reconstruction
check(getattr(win, "_live3d", "missing") is None, "after the run the live overlay is dropped")
check(r is not None and list(r.names) == list(f_.names) and r.t0 == f_.t0
      and np.array_equal(r.xyz, f_.xyz, equal_nan=True) and np.array_equal(r.per_cam, f_.per_cam, equal_nan=True),
      "after the run the project's 3D result equals the full reconstruct")
if _orig_live is not None:
    win._live_3d = _orig_live

# I5: X pauses a run with the 3D window open in < 1 s
goto_ref(300)
select_point("hip")
QTest.keyClick(win, Qt.Key_T)
wait(lambda: win.state == TRACKING and win._live3d is not None, 10)
running = win.state == TRACKING and win._live3d is not None
f_x = win.current
t_x = time.time()
QTest.keyClick(win, Qt.Key_X)
paused = wait(lambda: win.state == READY, 5)
dt_x = time.time() - t_x
check(running and paused and dt_x < 1.0 and win.current < N_FR - 1,
      f"X pauses a live run with the 3D window open in {dt_x:.2f} s (< 1 s)",
      f"running {running}, paused {paused}, X at frame {f_x}, stopped at {win.current}")
pump(0.5)
print(f"     ({len(ticks)} live updates during the first run)")

# I3: what one live update costs on a big rig: 10 cameras x 40,000 frames of 4K x 20 landmarks, a run tracking
# 5 of them, the trail 1 s at 240 fps behind the instant - the same calls `_live_3d` / `_live_window` make
from kinetrace.calib import Reconstruction, triangulate_cells  # noqa: E402
BW, BH, BT, BN, BC = 3840, 2160, 40_000, 20, 10
Kb = np.array([[2800.0, 0, BW / 2], [0, 2800.0, BH / 2], [0, 0, 1]])
bcams = []
for k in range(BC):
    a = 2 * np.pi * k / BC
    Rb, tb = look_at(np.array([3.0 * np.cos(a), 3.0 * np.sin(a), 1.0 + 0.1 * k]))
    bcams.append(CameraCalibration(dlt_from_camera(Kb, Rb, tb), BW, BH, NoUndistort(), pixel_origin=0.0))
bcal = Calibration(bcams, unit="m", source="synthetic")
bsess = []
tf = np.arange(BT, dtype=np.float64)
for c in range(BC):
    bs = TrackingSession(os.path.join(SCRATCH, f"live3d_big_{c}.mp4"), BT, 240.0, BW, BH)
    for j in range(BN):
        bs.add_landmark(f"m{j}")
    for j in range(BN):
        X = np.stack([0.2 * np.cos(tf * 0.001 + j), 0.2 * np.sin(tf * 0.001 + j), 0.00001 * tf], 1)
        bs.tracks[:, j] = bcams[c].project(X).astype(np.float32)
    bs.tracked[:] = True
    bsess.append(bs)
brates, boffs = [1.0] * BC, [0.0] + [0.3 * c for c in range(1, BC)]
t_now, back = 20_000, 240
run_names = [f"m{j}" for j in range(5)]
ts = np.arange(t_now - back, t_now + 1)
best = float("inf")
for _ in range(3):
    t_ = time.perf_counter()
    for nm in run_names:
        triangulate_cells(bsess, bcal, brates, boffs, nm, ts)
    best = min(best, time.perf_counter() - t_)
# the window merge, on a whole-length result of the 20 landmarks
brec = Reconstruction(0, [f"m{j}" for j in range(BN)], np.zeros((BT, BN, 3)), np.zeros((BT, BN)),
                      np.full((BT, BN), BC, np.int32), "m", np.zeros((BT, BN, BC), np.float32))
_keep = (win._live3d, win.project)
win._live3d = {"t0": int(ts[0]), "names": run_names, "xyz": np.zeros((len(ts), 5, 3)),
               "resid": np.zeros((len(ts), 5)), "n": np.zeros((len(ts), 5), np.int32),
               "err": np.zeros((len(ts), 5, BC), np.float32)}
win.project = Project(bsess, [f"B{c}" for c in range(BC)], boffs)
t_ = time.perf_counter()
win._live_window(brec, t_now, 240)
merge = time.perf_counter() - t_
win.project.calibration = bcal
t_ = time.perf_counter()
miss_big = win._missing_3d(None, None, t_now)           # the worst case: all 20 landmarks without 3D
dt_miss = time.perf_counter() - t_
win._live3d, win.project = _keep
check(len(miss_big) == BN and dt_miss < 0.01,
      f"the reasons for 20 landmarks x 10 cameras take {dt_miss * 1000:.1f} ms per frame change (< 10 ms)",
      f"{len(miss_big)} reasons")
check(best + merge < 0.015, f"one live update on 10 cameras x 40,000 frames x 20 landmarks (5 tracked, 1 s trail): "
      f"{(best + merge) * 1000:.1f} ms (< 15 ms)", f"triangulate {best * 1000:.1f} ms, merge {merge * 1000:.1f} ms")
del bsess, brec

# ---- [J] why a landmark has no 3D here (G206) ------------------------------------------------------
print("[J] a landmark without 3D at the instant: which, and why (G206)")
goto_ref(50)
txt = v.info.text() + " " + v.info.toolTip()
check("nose" in txt and "camA" in txt, "the info line names the nose and the one camera that has it", txt)
check("tail" in txt and "camA" in txt, "the tail at reference 50: only camA has it", txt)
goto_ref(250)                                   # camB's tail stretch (local 253) - camA's frame 250 has none
txt = v.info.text() + " " + v.info.toolTip()
check("tail" in txt and "camB" in txt, "the tail at reference 250: only camB has it", txt)
# J3: where every landmark that is anywhere has 3D, no reason is given for it
goto_ref(5)
txt = v.info.text() + " " + v.info.toolTip()
check("head" not in txt and "neck" not in txt, "landmarks with 3D are not listed as missing", txt)
# J4: ONE click of the nose in camC (offset -2.4: reference 400 falls between its frames 397 and 398) - the
# known pitfall of a single click at a fractional offset, said in words
win._set_active_view(2)
pump(0.3)
select_point("nose")
win._goto(397, force=True)
pump(0.3)
u = cams[2].project(world_at(397 - OFF[2])["nose"][None])[0]
QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier,
                 win.canvas.mapFromScene(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(float(u[0]), float(u[1]))))
pump(0.5)
win._set_active_view(0)
pump(0.3)
goto_ref(400)
txt = v.info.text() + " " + v.info.toolTip()
check("camC has it on its frame 397 but not on 398" in txt and "only camA" in txt,
      "a single click at a fractional offset: the reason names the frame it lacks", txt)

# ---- [R] the camera rays (G207) --------------------------------------------------------------------
print("[R] Rays: each 3D point with the rays of the cameras that made it (G207)")
goto_ref(150)
cmb = getattr(v, "cmb_rays", None)
check(cmb is not None, "the 3D window has a Rays control")
if cmb is not None:
    check(cmb.currentIndex() == 0 and v._hits.get("rays") in (None, []), "no rays by default")
    pick_combo(cmb, 2)                                           # Rays: all
    pump(0.3)
    rays = v._hits.get("rays") or []
    rec = p.reconstruction
    k = rec.index(win._reference_instant())
    want = {(nm, c) for j, nm in enumerate(rec.names) if np.isfinite(rec.xyz[k, j]).all()
            for c in range(3) if np.isfinite(rec.per_cam[k, j, c])}
    got = {(ry["name"], ry["view"]) for ry in rays}
    check(got == want and len(want) >= 6, "Rays: all draws one ray per camera that made each point",
          f"want {sorted(want)}, got {sorted(got)}")
    dots = v._hits.get("dots")
    ends_ok = dots is not None and all(
        np.hypot(*(np.asarray(ry["end"]) - dots[rec.names.index(ry["name"])])) < 1.0 for ry in rays)
    check(ends_ok, "each ray ends on its point's dot (< 1 px)")
    # R2: the camera that disagrees is the poor one, an exact camera is good
    by = {(ry["name"], ry["view"]): ry for ry in rays}
    bands = residual_bands(W, H)
    hc = by.get(("head", 2))
    check(hc is not None and hc["verdict"] == "poor" and hc["err"] > bands[1],
          "the head's ray from camC (15 px off) is poor", None if hc is None else f"{hc['verdict']} {hc['err']:.2f}")
    nk = [by.get(("neck", c)) for c in range(3)]
    check(all(r_ is not None and r_["verdict"] == "good" for r_ in nk), "the exact neck's rays are good",
          str([None if r_ is None else (r_["verdict"], round(r_["err"], 3)) for r_ in nk]))
    labs = [t_ for _, _, t_ in v._hits.get("labels", [])]
    check(any(t_.startswith("camC") and "px" in t_ for t_ in labs), "a ray is labelled with its camera and px error",
          str(labs[:12]))
    # R3: zoomed far in, the cameras are off the picture: rays are clipped, labelled at the edge, no error
    for z in (50.0, 0.05):
        v.zoom = z
        v._render()
        rays = v._hits.get("rays") or []
        hh, ww = v.last_image.shape[:2]
        inside = all(-1 <= ry["start"][0] <= ww + 1 and -1 <= ry["start"][1] <= hh + 1 for ry in rays)
        check(inside, f"at zoom {z:g} every ray's camera end is on the picture (clipped)")
    QTest.mouseDClick(v.canvas, Qt.LeftButton)                  # the view back to its start
    pump(0.2)
    # R4: Rays: selected draws only the selected landmark's rays
    pick_combo(cmb, 1)
    win.activateWindow()
    select_point("neck")
    pump(0.3)
    names_drawn = {ry["name"] for ry in (v._hits.get("rays") or [])}
    check(names_drawn == {"neck"}, "Rays: selected draws the selected landmark's rays only", str(names_drawn))
    select_point("head")
    pump(0.3)
    names_drawn = {ry["name"] for ry in (v._hits.get("rays") or [])}
    check(names_drawn == {"head"}, "selecting another landmark in LAYERS moves the rays to it", str(names_drawn))
    # R5: the saved picture has them, in both backgrounds
    for light in (False, True):
        if v.chk_light.isChecked() != light:
            QTest.mouseClick(v.chk_light, Qt.LeftButton)
            pump(0.2)
        png = os.path.join(SCRATCH, f"live3d_rays_{'light' if light else 'dark'}.png")
        v.save_png(png, scale=2.0)
        img = cv2.imread(png)
        cmb.blockSignals(True)
        cmb.setCurrentIndex(0)
        cmb.blockSignals(False)
        v.save_png(png + ".norays.png", scale=2.0)
        img0 = cv2.imread(png + ".norays.png")
        cmb.blockSignals(True)
        cmb.setCurrentIndex(1)
        cmb.blockSignals(False)
        check(img is not None and img0 is not None and int((np.abs(img.astype(int) - img0).sum(axis=2) > 30).sum()) > 200,
              f"Save picture keeps the rays ({'light' if light else 'dark'} background)")

# ---- [S] one selection: a click on a dot in 3D selects that landmark in the main window --------------
print("[S] a click on a 3D dot selects that landmark in LAYERS")
v._render()
dots = v._hits.get("dots")
rec = p.reconstruction
j = list(v.scene.names).index("neck") if "neck" in v.scene.names else None
if dots is not None and j is not None and np.isfinite(dots[j]).all():
    dpr = v._last_dpr
    hh, ww = v.last_image.shape[:2]
    ox, oy = (v.canvas.width() - ww / dpr) / 2.0, (v.canvas.height() - hh / dpr) / 2.0
    at = QPoint(int(round(dots[j][0] / dpr + ox)), int(round(dots[j][1] / dpr + oy)))
    QTest.mouseClick(v.canvas, Qt.LeftButton, Qt.NoModifier, at)
    pump(0.3)
    check(win.selected == win.session.pid_by_name("neck") and win._selected_pids() == [win.session.pid_by_name("neck")],
          "the click on the neck's dot selected the neck in LAYERS", f"selected {win.selected}")
else:
    check(False, "(setup) the neck's dot is on the picture")

win._dev_probe.wait(15000)
win.project.dirty = False
win.close()
for pth in paths:
    forget_recovery(pth)
if FAILS:
    print(f"FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("PASSED")
