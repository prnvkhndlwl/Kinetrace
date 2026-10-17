"""Offscreen GUI: the automatic 3D result (I305). With a calibration, every landmark that has data in two
or more cameras is in the project's 3D result at once, after every gesture that changes the 2D data: a click
on a canvas, Ctrl+Z, a rename in the LAYERS panel, a tracking run, Delete. After each one the result must
EQUAL a fresh full `calib.reconstruct` over `_t_range_3d()` (what Ctrl+3 computes) bit for bit.

The project: 3 synthetic cameras (one with an OpenCV lens), camera 2 at twice the rate, fractional offsets
(so every reference instant interpolates between two local frames), three landmarks flying a known path.

Run: .venv\\Scripts\\python.exe tests\\verify_auto3d.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, NoUndistort, OpenCVUndistort, dlt_from_camera,
                             reconstruct)
from kinetrace.project import Project
from kinetrace.session import TrackingSession

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
rng = np.random.RandomState(5)

W, H = 320, 240
N_FR = [90, 180, 90]                  # camera 2 runs at 2x
FPS = [60.0, 120.0, 60.0]
OFF = [0.0, 20.4, -5.3]               # fractional: every instant interpolates (I305 must cover it)
RATES = [1.0, 2.0, 1.0]
NAMES = ["head", "hip", "tail"]
FAILS = []


def check(ok, what, detail=""):
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
    pos = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.5 + 0.4 * k])
    R, t = look_at(pos)
    lens = OpenCVUndistort(K, np.array([-0.08, 0.01, 0.0, 0.0])) if k == 1 else NoUndistort()
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, lens, pixel_origin=0.0))


def world_at(t_ref):
    s = float(t_ref)
    hip = np.array([0.3 * np.cos(s * 0.06), 0.3 * np.sin(s * 0.06), 0.003 * s])
    return np.stack([hip + [0.06, 0.0, 0.02], hip, hip - [0.07, 0.0, 0.01]])


sessions, paths = [], []
for c, cal in enumerate(cams):
    p = os.path.join(SCRATCH, f"auto3d_{c}.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS[c], (W, H))
    s = TrackingSession(p, N_FR[c], FPS[c], W, H)
    for nm in NAMES:
        s.add_landmark(nm)
    for f in range(N_FR[c]):
        t_ref = (f - OFF[c]) / RATES[c]
        img = np.full((H, W, 3), 30, np.uint8)
        uv = cal.project(world_at(t_ref))
        if 0 <= t_ref <= 60:
            cv2.circle(img, (int(round(uv[1][0])), int(round(uv[1][1]))), 6, (230, 230, 230), -1)
            for j in range(3):
                if j == 2 and c != 0:          # the tail: camera 1 only, so it has no 3D yet
                    continue
                s.set_position(f, j, float(uv[j][0] + rng.normal(0, 0.2)), float(uv[j][1] + rng.normal(0, 0.2)))
        vw.write(img)
    vw.release()
    sessions.append(s)
    paths.append(p)
proj = Project(sessions, ["camA", "camB", "camC"], OFF)
assert proj.rates == RATES, proj.rates
proj.calibration = Calibration(cams, unit="m", source="synthetic")
PROJ = os.path.join(SCRATCH, "auto3d.kinetrace")
proj.save(PROJ)
for p in paths:
    forget_recovery(p)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from kinetrace.app import READY, MainWindow  # noqa: E402

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


def full():
    """What Ctrl+3 computes now: every landmark, the whole overlap."""
    p = win.project
    return reconstruct(p.sessions, p.calibration, p.rates, p.offsets, win._t_range_3d())


def same(what):
    """The project's 3D result equals a fresh full reconstruct, bit for bit (NaN = NaN)."""
    pump(1.0)                                   # the automatic update runs on a timer
    r, f = win.project.reconstruction, full()
    if r is None:
        check(False, what, "no 3D result")
        return
    diffs = []
    if r.t0 != f.t0 or r.n_frames != f.n_frames:
        diffs.append(f"range {r.t0}+{r.n_frames} vs {f.t0}+{f.n_frames}")
    if list(r.names) != list(f.names):
        diffs.append(f"names {r.names} vs {f.names}")
    if not diffs:
        for k in ("xyz", "residual", "n_cams", "per_cam"):
            a, b = getattr(r, k), getattr(f, k)
            if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype \
                    or not np.array_equal(a, b, equal_nan=True):
                diffs.append(k)
    check(not diffs, what, "; ".join(diffs))


def click(cv, x, y):
    QTest.mouseClick(cv.viewport(), Qt.LeftButton, Qt.NoModifier, cv.mapFromScene(QPointF(float(x), float(y))))
    pump(0.1)


def select_point(name):
    s = win.session
    win.layers.setCurrentItem(win.layers.point_item(s.pid_by_name(name)))
    pump(0.1)
    return s.pid_by_name(name)


def rename_row(pid, text):
    """Rename the way the user does: double-click the point's LAYERS row, type the name, Enter."""
    win.activateWindow()                     # the user's click on the main window (Ctrl+3 opened the 3D window)
    pump(0.1)
    item = win.layers.point_item(pid)
    par = item.parent()
    while par is not None:
        par.setExpanded(True)
        par = par.parent()
    win.layers.scrollToItem(item)
    pump(0.05)
    at = win.layers.visualItemRect(item).center()
    QTest.mouseClick(win.layers.viewport(), Qt.LeftButton, Qt.NoModifier, at)
    pump(0.2 + QApplication.doubleClickInterval() / 1000.0)   # not read as part of the double-click
    item = win.layers.point_item(pid)
    at = win.layers.visualItemRect(item).center()
    QTest.mouseDClick(win.layers.viewport(), Qt.LeftButton, Qt.NoModifier, at)
    pump(0.2)
    ed = QApplication.focusWidget()
    assert type(ed).__name__ == "QLineEdit", f"(setup) the double-click opened no name editor: {type(ed).__name__}"
    QTest.keyClick(ed, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(ed, text)
    QTest.keyClick(ed, Qt.Key_Return)
    pump(0.2)


win._open_project_from_path(PROJ)
assert wait(lambda: win.state == READY and win.project is not None and win.project.n_views == 3
            and not win._loading, 20), "the project opens"
p = win.project
assert p.calibration is not None and p.calibration.n_usable(3) == 3

# ---- [A] every gesture leaves the 3D result equal to Ctrl+3 ---------------------------------------
print("[A] the 3D result follows every 2D gesture (I305)")
win.act_recon.trigger()                     # a first full result, as a user would have
pump(0.5)
check(p.reconstruction is not None and "tail" in p.reconstruction.names, "(setup) Ctrl+3 made a 3D result")
same("right after Ctrl+3 the result is the full reconstruct")

# 1. a click: the tail at camera 2's frame 60 (reference ~19.8) -> the tail now has two cameras there
win._set_active_view(1)
pump(0.3)
f_b = 60
t_ref = (f_b - OFF[1]) / RATES[1]
pid = select_point("tail")
for f in (f_b, f_b + 1):                   # an instant at a fractional offset reads TWO local frames
    win._goto(f, force=True)
    pump(0.2)
    uv = cams[1].project(world_at((f - OFF[1]) / RATES[1]))[2]
    click(win.canvas, uv[0], uv[1])
check(bool(win.session.tracked[f_b:f_b + 2, pid].all()), "(setup) two clicks placed the tail in camera 2")
same("a click on a canvas: the tail's new 3D cells are in the result")
k = p.reconstruction.index(int(round(t_ref))) if p.reconstruction is not None else None
check(k is not None and np.isfinite(p.reconstruction.xyz[k:k + 2, p.reconstruction.names.index("tail")]).any(),
      "the clicked tail has a 3D position near reference frame %d" % int(round(t_ref)))

# 2. Ctrl+Z takes it back
QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
pump(0.3)
check(bool(win.session.tracked[f_b, pid]) and not win.session.tracked[f_b + 1, pid],
      "(setup) Ctrl+Z removed the second click")
same("Ctrl+Z: the result follows the undo")

# 3. a rename in the LAYERS panel (every camera)
rename_row(win.session.pid_by_name("hip"), "pelvis")
check(win.session.pid_by_name("pelvis") is not None, "(setup) the rename landed")
same("a rename: the column carries the new name, the old one is gone")

# 4. a tracking run (Moving spot, CPU) on the pelvis in camera 1
win._set_active_view(0)
pump(0.3)
win.act_pm_spot.trigger()
pid = select_point("pelvis")
win._goto(30, force=True)
pump(0.2)
before = win.session.tracks[31:60, pid].copy()
win.act_track_all.setChecked(False)
QTest.keyClick(win, Qt.Key_T)
check(wait(lambda: win.state != READY, 10) and wait(lambda: win.state == READY, 120), "(setup) a tracking run ran")
pump(0.3)
check(not np.array_equal(before, win.session.tracks[31:60, pid], equal_nan=True), "(setup) the run changed tracks")
same("a tracking run: the result follows the run's frames")

# 5. Delete the head (every camera)
select_point("head")
QTest.keyClick(win, Qt.Key_Delete)          # the question "Delete head ...?" is answered Yes (stub)
pump(0.3)
check(win.session.pid_by_name("head") is None, "(setup) Delete removed the head")
same("Delete: the head's column is gone")

# ---- [B] a retime in the camera panel rebuilds the result once, after the box is still ---------------
print("[B] a retime rebuilds the whole result once it is still (I305)")
from PySide6.QtWidgets import QFileDialog  # noqa: E402
from kinetrace import calib as _calib  # noqa: E402

full_calls = []
_orig_run = win._sync3d_run


def _counting_run(changed, *a, **k):
    full_calls.append(changed is None)
    return _orig_run(changed, *a, **k)


win._sync3d_run = _counting_run
win._refresh_cameras()
pump(0.2)
row = win.cameras._rows[1]
if not row.expanded():
    QTest.mouseClick(row.btn_expand, Qt.LeftButton)
    pump(0.2)
off0 = p.offsets[1]
row.spin.setFocus()
for _ in range(3):                                # three steps in a row, faster than a sync tick
    QTest.keyClick(row.spin, Qt.Key_Up)
    app.processEvents()
check(p.offsets[1] != off0, "(setup) the arrows moved camera 2's offset", f"{off0} -> {p.offsets[1]}")
pump(1.5)
check(sum(full_calls) == 1, "three quick offset steps rebuild the 3D result ONCE", f"rebuilds: {full_calls}")
same("after the retime the result is the full reconstruct at the new offset")
win._sync3d_run = _orig_run

# ---- [C] an imported 3D result is left alone until Ctrl+3 ----------------------------------------
print("[C] an imported result is not overwritten by the automatic updates")
csv3 = os.path.join(SCRATCH, "auto3d_import_xyz.csv")
p.reconstruction.export_csv(csv3)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (csv3, ""))
win.act_import_xyz.trigger()
pump(0.5)
imp = p.reconstruction
check(imp is not None and imp.source == "imported", "(setup) the 3D points were imported",
      None if imp is None else imp.source)
win._set_active_view(0)
pump(0.3)
pid = select_point("tail")
win._goto(12, force=True)
pump(0.2)
click(win.canvas, 100.0, 90.0)
pump(1.0)
check(p.reconstruction is imp, "a click leaves the imported result as it is")
win.act_recon.trigger()
pump(0.5)
check(p.reconstruction.source == "tracks", "Ctrl+3 makes a result from the tracks again")
same("... and the updates follow it from there")

# ---- [D] the cost on a long 4K project: a click, a run's end ---------------------------------------
print("[D] what an update costs on 40,000 frames of 4K, 3 cameras, 10 landmarks")
from kinetrace.app import block_digests  # noqa: E402

BW, BH, BT, BN = 3840, 2160, 40_000, 10
bcams = []
Kb = np.array([[2800.0, 0, BW / 2], [0, 2800.0, BH / 2], [0, 0, 1]])
for k, a in enumerate((0.3, 2.4, 4.4)):
    Rb, tb = look_at(np.array([3.0 * np.cos(a), 3.0 * np.sin(a), 1.0]))
    lens = OpenCVUndistort(Kb, np.array([-0.2, 0.05, 0.0, 0.0])) if k == 1 else NoUndistort()
    bcams.append(CameraCalibration(dlt_from_camera(Kb, Rb, tb), BW, BH, lens, pixel_origin=0.0))
bcal = Calibration(bcams, "m")
bses = []
brng = np.random.RandomState(9)
for k in range(3):
    sb = TrackingSession(f"big{k}", BT, 120.0, BW, BH)
    for j in range(BN):
        sb.add_landmark(f"m{j}")
    sb.tracks[:] = brng.uniform(500, 3000, (BT, BN, 2)).astype(np.float32)
    sb.tracked[:] = True
    bses.append(sb)
brates, boffs = [1.0, 1.0, 1.0], [0.0, 0.4, -0.3]
btr = (1, BT - 2)
brec = _calib.reconstruct(bses, bcal, brates, boffs, btr)
bdig = block_digests(bses[1])
bses[1].set_position(20_000, 3, 1234.5, 987.25)          # one click
t_ = time.perf_counter()
ndig = block_digests(bses[1])
blk = [nm for nm in ndig if (ndig[nm] != bdig[nm]).any()]
lo, hi = _calib.reference_span(20_000 - 20_000 % 256, 20_000 - 20_000 % 256 + 255, brates[1], boffs[1])
brec2 = _calib.update_reconstruction(brec, bses, bcal, brates, boffs, btr, {"m3": [(lo, hi)]})
dt_click = (time.perf_counter() - t_) * 1000
check(blk == ["m3"], "(setup) the digest finds the one changed landmark", str(blk))
check(dt_click < 50, f"one click on 40,000 frames x 10 landmarks updates in {dt_click:.0f} ms (< 50 ms)")
for f in range(10_000, 11_000):                           # a run's end: 1,000 frames of every landmark
    bses[0].tracks[f] += np.float32(0.5)
t_ = time.perf_counter()
lo, hi = _calib.reference_span(10_000 - 10_000 % 256, 11_263, brates[0], boffs[0])
brec3 = _calib.update_reconstruction(brec2, bses, bcal, brates, boffs, btr, {f"m{j}": [(lo, hi)] for j in range(BN)})
dt_run = time.perf_counter() - t_
check(dt_run < 1.0, f"a 10-landmark 1,000-frame run's end updates in {dt_run:.2f} s (< 1 s)")
_full = _calib.reconstruct(bses, bcal, brates, boffs, btr)
check(all(np.array_equal(getattr(brec3, k), getattr(_full, k), equal_nan=True)
          for k in ("xyz", "residual", "n_cams", "per_cam")), "... and equals the full reconstruct bit for bit")

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
