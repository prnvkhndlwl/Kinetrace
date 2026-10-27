"""Calibration, 3D and sync fixes through the window (docs/AUDIT.md G222-G224 and I351's app half; the
2026-10-25 full sweep's PR 3). Offscreen, no GPU, every dialog stubbed; each section FAILS on main 393f21e.

  G222  File > Import > Calibration over a calibration, then Ctrl+Z: the earlier calibration is back
  G223  the wand wizard's lens rows: Remove + Cancel leaves the lens; Remove + Finish, then Ctrl+Z, brings
        the lens and the earlier calibration back
  G224  3D > Show 3D View of a mirror-image (easyWand / DLTdv-like) world: drawn as the room looks (the same
        screen orientation as its right-handed twin) and the window says the world is mirrored
  I351  3D > Sync > Estimate Sub-frame Offsets searches a camera at twice the rate over +/-2 of its frames

.venv\\Scripts\\python.exe tests\\verify_calib_sync_gui.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(errors="replace")
OUT = os.path.join(HERE, "out", "calib_sync_gui")
os.makedirs(OUT, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWizard  # noqa: E402

import _synth_wand as sw  # noqa: E402
from _clean import forget_recovery  # noqa: E402
from kinetrace import calib  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, Reconstruction, dlt_from_camera, dlt_project  # noqa: E402
from kinetrace.lens import LensProfile  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))

from kinetrace import app as appmod  # noqa: E402
from kinetrace import calibwizard as cw  # noqa: E402
from kinetrace import view3d  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_for(cond, what, limit=600):
    t0 = time.time()
    while not cond():
        pump(0.1)
        if time.time() - t0 > limit:
            raise TimeoutError(what)


def click(w):
    assert w.isEnabled(), f"{w.text() if hasattr(w, 'text') else w} is disabled"
    QTest.mouseClick(w, Qt.LeftButton)
    pump(0.05)


def ctrl_z():
    win.activateWindow()                     # a closed wizard leaves no active window (offscreen)
    pump(0.1)
    QTest.keyClick(win.canvas.viewport(), Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)


def write_video(path, n, w, h, fps):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    img = np.full((h, w, 3), 40, np.uint8)
    for _ in range(n):
        vw.write(img)
    vw.release()
    forget_recovery(path)


def open_project(path, n_views):
    win._open_project_from_path(path)
    wait_for(lambda: win.state == READY and win.project is not None and win.project.n_views == n_views, "open")
    return win.project


def look_at(C):
    C = np.asarray(C, float)
    z = -C / np.linalg.norm(C)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    return R, -R @ C


# ------------------------------------------------------------------ a small 2-camera project (G222, G224, I351)
W2, H2 = 320, 240
K2 = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])
CENTRES = [(6, 0, 2), (0, 6, 2.5)]


def rig(mirror_y=False, shift=0.0):
    coefs = []
    for c in CENTRES:
        R, t = look_at(np.array(c) + shift)
        L = dlt_from_camera(K2, R, t)
        if mirror_y:
            L[[1, 5, 9]] *= -1.0               # the same cameras, the world's Y reversed: a mirror image
        coefs.append(L)
    return Calibration([CameraCalibration(L, W2, H2) for L in coefs], unit="m",
                       source=f"rig mirror={mirror_y} shift={shift}")


PTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
paths = [os.path.join(OUT, "small_cam1.mp4"), os.path.join(OUT, "small_cam2.mp4")]
write_video(paths[0], 30, W2, H2, 30.0)
write_video(paths[1], 60, W2, H2, 60.0)                 # twice the rate (I351)
cal_r = rig()
sessions = [TrackingSession(paths[0], 30, 30.0, W2, H2), TrackingSession(paths[1], 60, 60.0, W2, H2)]
for k, s in enumerate(sessions):
    for j in range(len(PTS)):
        s.add_landmark(f"p{j}")
    uv = dlt_project(cal_r.camera(k).coefs, PTS)
    for f in range(s.n_frames):
        for j in range(len(PTS)):
            s.set_position(f, j, float(uv[j, 0]) + 0.01 * f, float(uv[j, 1]))
SMALL = os.path.join(OUT, "small.kinetrace")
Project(sessions, ["cam1", "cam2"], [0.0, 0.0]).save(SMALL)

p = open_project(SMALL, 2)
check(p.rates == [1.0, 2.0] or list(p.rates) == [1.0, 2.0], "(the second camera runs at twice the rate)",
      str(p.rates))

# ------------------------------------------------------------------ G222
print("G222 File > Import > Calibration, then Ctrl+Z")
first = rig(shift=0.0)
p.calibration = first
second = rig(shift=0.3)


class FakeImportDialog:
    def __init__(self, *a, **k):
        self.result_calibration = second

    def exec(self):
        return QDialog.Accepted


orig_dialog = appmod.CalibrationDialog
appmod.CalibrationDialog = FakeImportDialog
try:
    win.act_calib.trigger()                        # File -> Import -> Calibration...
    pump(0.3)
finally:
    appmod.CalibrationDialog = orig_dialog
check(p.calibration is second, "the imported calibration is the project's")
check(win.act_undo.isEnabled(), "Edit > Undo is available after the import")
ctrl_z()
check(p.calibration is first, "Ctrl+Z puts the earlier calibration back",
      getattr(p.calibration, "source", None))

# ------------------------------------------------------------------ I351 (app half)
print("I351 Estimate Sub-frame Offsets: the search width of a camera at twice the rate")
p.calibration = cal_r
seen = {}


def fake_estimate(sessions_, cal_, rates, offsets, t_range, names=None, search=1.0, *a, **k):
    seen["search"] = search
    return list(offsets), 1.0, 1.0


orig_est = calib.estimate_offsets
calib.estimate_offsets = fake_estimate
try:
    win.act_offsets3d.trigger()                    # 3D -> Sync -> Estimate Sub-frame Offsets...
    wait_for(lambda: "search" in seen, "the estimate was asked for", 60)
    pump(0.3)
finally:
    calib.estimate_offsets = orig_est
s_ = seen.get("search")
check(np.ndim(s_) == 1 and len(s_) == 2 and s_[0] == 1.0 and s_[1] == 2.0,
      "the app asks for +/-1 frame for the reference and +/-2 for the camera at twice the rate", str(s_))

# ------------------------------------------------------------------ G224
print("G224 the 3D view of a mirror-image world")
cal_m = rig(mirror_y=True)
check(calib.world_is_left_handed(cal_m) and not calib.world_is_left_handed(cal_r),
      "(the twin calibration is a mirror image)")
Xm = PTS * np.array([1.0, -1.0, 1.0])               # the same four points in the mirrored world
p.calibration = cal_m
p.reconstruction = Reconstruction(0, [f"p{j}" for j in range(4)], np.repeat(Xm[None], 30, 0),
                                  np.zeros((30, 4)), np.full((30, 4), 2), "m")
if win.act_view3d.isChecked():                     # (the sub-frame estimate above showed its 3D result)
    win.act_view3d.trigger()
    pump(0.2)
win.act_view3d.trigger()                           # 3D -> Show 3D View
pump(0.5)
v3 = win.view3d
check(v3 is not None and v3.isVisible(), "the 3D window is open")
if v3 is not None:
    sc = v3.scene
    check(bool(getattr(sc, "mirrored", False)), "the window knows the world is mirrored")
    check("mirror" in v3.info.text().lower(), "and says so", v3.info.text()[:160])

    def on_screen(scene, pts):
        f, _, _ = view3d.screen_transform(scene, (800, 800), 35, 25)
        return np.asarray(f(pts)[0], float)
    twin = view3d.Scene3D()
    twin.points, twin.points_all = PTS, PTS[None]
    # the window reflects the world's Y back (Z stays up, X keeps its direction): the same physical points
    # land where their right-handed twin's do
    d = float(np.max(np.abs(on_screen(sc, Xm) - on_screen(twin, PTS))))
    check(d < 0.5, "the mirrored world is drawn as the room looks (where its right-handed twin is)",
          f"{d:.3g} px apart")
    # a closed mesh (a hull, an animal model) is lit the same in the mirrored world, not turned inside out
    vtx = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    fcs = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])            # outward normals (right-handed)

    def lit(mirror):
        sc_ = view3d.Scene3D()
        F = np.array([1.0, -1.0, 1.0]) if mirror else np.ones(3)
        sc_.mesh = (vtx * F, fcs)
        sc_.points_all = (vtx * F)[None]
        sc_.mirrored = mirror
        img = view3d.render_scene(sc_, (400, 400), 35, 25, show_points=False, show_cameras=False)
        bg = img[0, 0].astype(int)
        m = np.abs(img.astype(int) - bg).sum(axis=2) > 30
        return float(img[m].mean()), int(m.sum())
    (b0, n0), (b1, n1) = lit(False), lit(True)
    check(abs(b1 - b0) < 2.0 and abs(n1 - n0) < 0.02 * n0,
          "a mesh in the mirrored world is lit as in the right-handed one (not inside out)",
          f"brightness {b0:.1f} vs {b1:.1f}, area {n0} vs {n1}")
if win.act_view3d.isChecked():                     # closed again: its window would take the next Ctrl+Z
    win.act_view3d.trigger()
    pump(0.2)

# ------------------------------------------------------------------ G223 (the wand rig)
print("G223 the wand wizard's lens rows")
wpaths = [os.path.join(OUT, f"wand_cam{c + 1}.mp4") for c in range(4)]
proj, truth = sw.build(video_paths=wpaths, n_ref=300)
for pth, s in zip(wpaths, proj.sessions):
    write_video(pth, s.n_frames, sw.W, sw.H, sw.FPS)
WPROJ = os.path.join(OUT, "wand.kinetrace")
proj.save(WPROJ)
p = open_project(WPROJ, 4)
TRUE_CAL = Calibration(truth["cams"], "m", "truth")
p.calibration = TRUE_CAL
prof = LensProfile(sw.W, sw.H, np.array([[1000.0, 0, 639.5], [0, 1000.0, 359.5], [0, 0, 1]]),
                   np.array([-0.05, 0.01, 0, 0, 0.0]), rotation=0, source="test.klens.json")
win._set_lenses({0: prof})
pump(0.1)
assert p.lenses[0] is prof
state = {}


def to_cams_page(wiz):
    wiz.show()
    wiz.restart()
    pump(0.1)
    nxt = wiz.button(QWizard.NextButton)
    click(nxt)
    ln = wiz.page_wand.length
    ln.lineEdit().selectAll()
    QTest.keyClicks(ln.lineEdit(), f"{sw.WAND_L}")
    QTest.keyClick(ln.lineEdit(), Qt.Key_Tab)
    click(nxt)
    assert wiz.currentPage() is wiz.page_cams
    return nxt


def remove_lens(wiz):
    rm = wiz.page_cams.lens_buttons[0][2]
    click(rm)
    state["during"] = p.lenses[0]


def drive_cancel(wiz):
    to_cams_page(wiz)
    remove_lens(wiz)
    wiz.reject()
    pump(0.1)
    return QDialog.Rejected


def drive_finish(wiz):
    nxt = to_cams_page(wiz)
    remove_lens(wiz)
    click(wiz.page_cams.r_known)
    for sp, f in zip(wiz.page_cams._spins, sw.FOCAL):
        sp.lineEdit().selectAll()
        QTest.keyClicks(sp.lineEdit(), str(int(f)))
        QTest.keyClick(sp.lineEdit(), Qt.Key_Tab)
    ex = wiz.page_cams.extras
    for i in range(ex.count()):
        ex.item(i).setCheckState(Qt.Unchecked)
    click(nxt)
    click(wiz.page_frame.r_none)
    click(nxt)
    page = wiz.page_run
    click(page.btn_run)
    wait_for(lambda: (page._thread is None or not page._thread.isRunning()) and page.btn_run.isEnabled()
             and (wiz.result_calibration is not None or page.timing_box.isVisible()), "calibration")
    if wiz.result_calibration is None and page.timing_box.isVisible():
        click(page.btn_keep_offsets)
        wait_for(lambda: wiz.result_calibration is not None and page.btn_run.isEnabled(), "calibration (kept)")
    state["made"] = wiz.result_calibration
    click(wiz.button(QWizard.FinishButton))
    pump(0.1)
    return QDialog.Accepted


orig_exec = cw.WandWizard.exec
cw.WandWizard.exec = drive_cancel
try:
    win._wand_wizard()
    pump(0.3)
finally:
    cw.WandWizard.exec = orig_exec
check(state.get("during") is prof, "Remove on the Cameras page does not change the project yet")
check(p.lenses[0] is prof and p.calibration is TRUE_CAL, "Cancel leaves the lens and the calibration as they were")
if p.lenses[0] is not prof:                      # (today's Cancel removed it: put it back for the next half)
    win._set_lenses({0: prof})
    pump(0.1)

cw.WandWizard.exec = drive_finish
try:
    win._wand_wizard()
    pump(0.3)
finally:
    cw.WandWizard.exec = orig_exec
check(p.lenses[0] is None and p.calibration is state.get("made"),
      "Finish applies the removal with the new calibration")
ctrl_z()
check(p.lenses[0] is prof, "one Ctrl+Z brings the lens back")
check(p.calibration is TRUE_CAL, "and the calibration it was made with goes with it",
      getattr(p.calibration, "source", None))

win._dev_probe.wait(60000)
p.dirty = False
for s in p.sessions:
    s.dirty = False if hasattr(s, "dirty") else None
win.close()
pump(0.2)
for pth in paths + wpaths:
    forget_recovery(pth)
print("\nverify_calib_sync_gui: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
