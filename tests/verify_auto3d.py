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
