"""Offscreen GUI: several cameras share ONE landmark list, and with a calibration
a point clicked in one camera shows its epipolar line in the others (G19).

A calibrated 2-camera synthetic rig (known DLT cameras, a dot at a known 3D
path) is opened through the real MainWindow with no points at all, and the
digitizing workflow is driven with real key and mouse events:
  import the calibration -> the guides switch on;
  N + click in camera A -> the point exists in camera B too (no data there), and
  B's picture shows the line it must lie on, through the true position;
  click camera B -> B becomes the working camera with the SAME point selected
  and the guide from A drawn; a plain click places it in B;
  rename / delete / Ctrl+Z act on every camera at once; a click with nothing
  selected shows where that spot can be in the other camera, Esc clears it;
  a skeleton, a new camera and an older project with different point lists all
  end with the same names in every camera.

Run: .venv\\Scripts\\python.exe tests\\verify_shared_landmarks.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import CameraCalibration, NoUndistort, closest_on_polyline, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
W, H, N, FPS = 640, 480, 60, 30.0


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])
cams = []
for a in (0.2, 1.4):
    pos = np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.6])
    R, t = look_at(pos)
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))


def world(f):
    return np.array([[0.15 * np.cos(f * 0.05), 0.15 * np.sin(f * 0.05), 0.05]])


paths, sessions = [], []
for c, cal in enumerate(cams):
    path = os.path.join(OUT, f"shared_cam{c}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 40, np.uint8)
        u, v = cal.project(world(f))[0]
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (230, 230, 230), -1)
        vw.write(img)
    vw.release()
    paths.append(path)
    sessions.append(TrackingSession(path, N, FPS, W, H))
    forget_recovery(path)
PROJ = os.path.join(OUT, "shared_landmarks.kinetrace")
Project(sessions, ["camA", "camB"], [0, 0]).save(PROJ)
CSV = os.path.join(OUT, "shared_landmarks_dltCoefs.csv")
np.savetxt(CSV, np.stack([c.coefs for c in cams], 1), delimiter=",", fmt="%.10g")

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace import view3d as v3  # noqa: E402
from kinetrace.skeletons import template_by_name  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1600, 1000)
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def click(canvas, x, y):
    QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier,
                     canvas.mapFromScene(QPointF(float(x), float(y))))
    pump(0.1)


def truth(cam, f):
    return cams[cam].project(world(f))[0]


def dist_to_guides(canvas, uv):
    return min(float(np.linalg.norm(closest_on_polyline(np.asarray(pts), uv) - uv))
               for pts, _c, _n in canvas._guides.lines)


win._open_project_from_path(PROJ)
for _ in range(200):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == 2:
        break
assert win.state == READY and win.project.n_views == 2
p = win.project
A, B = win.grid.canvas(0), win.grid.canvas(1)
app.setActiveWindow(win)

# ---- 1. importing the calibration switches the guides on ------------------------------
win.act_epipolar.setChecked(False)                 # even if the user had turned them off


def _fake_exec(self):
    assert self.load(CSV)
    self.conv.setCurrentIndex(2)                   # OpenCV, 0-based: how the cameras were built
    self._accept()
    return QDialog.Accepted


v3.CalibrationDialog.exec = _fake_exec
win._import_calibration()
pump()
assert p.calibration is not None and len(p.calibration) == 2
assert win.act_epipolar.isChecked(), "a new calibration switches the epipolar guides on"
print("calibration import switches the guides on OK")

# ---- 2. N + click in camera A: the point is in camera B too, with its line there ----------
F = 20
win._goto(F)
pump(0.3)
QTest.keyClick(win, Qt.Key_N)
ua = truth(0, F)
click(A, *ua)
sa, sb = p.sessions
assert sa.n_points == 1 and sb.n_points == 1, (sa.n_points, sb.n_points)
name = sa.points[0].name
jb = sb.pid_by_name(name)
assert jb is not None and not sb.tracked[:, jb].any(), "camera B has the landmark, with no data"
assert sb.points[jb].color == sa.points[0].color, "the same landmark has the same colour"
assert win.selected == 0
assert len(B._guides.lines) == 1, f"camera B shows the line from camera A: {len(B._guides.lines)}"
d = dist_to_guides(B, truth(1, F))
assert d < 3.0, f"the line in camera B misses the true position by {d:.2f} px"
assert not A._guides.lines, "camera A: no other camera has the point yet"
print(f"a point clicked in camera A is in camera B's list and B shows its line ({d:.2f} px from truth) OK")

# ---- 3. click camera B: same point selected there, placed with a plain click -------------
click(B, W / 2, H / 2)
assert p.active == 1, "a click on camera B makes it the working camera"
assert win.selected == jb, f"the same landmark stays selected: {win.selected} vs {jb}"
assert win.point_list.currentRow() == jb
assert len(win.canvas._guides.lines) == 1, "the working camera B shows the guide from A"
assert "not placed" in win.statusBar().currentMessage(), win.statusBar().currentMessage()
ub = truth(1, F)
click(win.canvas, *ub)                              # a plain click places the SELECTED point
assert sb.tracked[F, jb] and sb.manual[F, jb], "placed in camera B"
assert np.linalg.norm(sb.tracks[F, jb] - ub) < 1.5
assert len(A._guides.lines) == 1, "now camera A (a companion) shows the line from B"
print("click camera B: same landmark selected, guide drawn, a plain click places it OK")

# ---- 4. rename acts on every camera -----------------------------------------------------
got = win._apply_rename(jb, "snout")
assert got == "snout" and sa.pid_by_name("snout") is not None and sb.pid_by_name("snout") is not None
assert sa.pid_by_name(name) is None and sb.pid_by_name(name) is None
print("rename applies to every camera OK")

# ---- 5. a new point is one undo step in every camera ---------------------------------------
QTest.keyClick(win, Qt.Key_Escape)                  # deselect (nothing armed)
pump(0.05)
QTest.keyClick(win, Qt.Key_N)
click(win.canvas, 100, 100)
new = sb.points[-1].name
assert new not in ("snout", name) and sa.pid_by_name(new) is not None, (new, [q.name for q in sa.points])
win._undo_run()
pump(0.05)
assert sa.pid_by_name(new) is None and sb.pid_by_name(new) is None, "Ctrl+Z removes it from every camera"
print(f"adding {new} and Ctrl+Z act on every camera OK")

# ---- 6. delete acts on every camera; Ctrl+Z brings back each camera's data ----------------
asked = []
QMessageBox.question = staticmethod(lambda *a, **k: (asked.append(str(a[2])), QMessageBox.Yes)[1])
win._on_delete(sb.pid_by_name("snout"))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
assert sa.pid_by_name("snout") is None and sb.pid_by_name("snout") is None
assert asked and "camA" in asked[0], f"the question names the other camera: {asked}"
win._undo_run()
pump(0.05)
ja2, jb2 = sa.pid_by_name("snout"), sb.pid_by_name("snout")
assert ja2 is not None and jb2 is not None, "Ctrl+Z restores it in every camera"
assert sa.tracked[F, ja2] and sb.tracked[F, jb2], "with each camera's data"
print("delete asks once, removes it from every camera, Ctrl+Z restores both OK")

# ---- 7. nothing selected: a plain click places nothing and SAYS so; Alt+click looks ------
win._deselect()
pump(0.05)
n_before = (sa.n_points, sb.n_points)
spot = world(F) + [0.12, -0.05, 0.08]               # an empty spot, away from every marker
click(win.canvas, *cams[1].project(spot)[0])
assert (sa.n_points, sb.n_points) == n_before, "nothing edited"
# (G21) the look-here cross this click used to draw was taken for a point that
# "changed into a crosshair" and never reached the POINTS list
assert win._epi_probe is None and not win.canvas._guides.lines, "a plain click draws no look-here cross"
assert win.toast.isVisible() and "Nothing was placed" in win.toast.text(), "the notice is on the video"
QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.AltModifier,
                 win.canvas.mapFromScene(QPointF(*cams[1].project(spot)[0])))
pump(0.1)
assert (sa.n_points, sb.n_points) == n_before, "Alt+click edits nothing"
assert win._epi_probe is not None, "Alt+click asked where the spot is"
ring = win.canvas._guides.lines
assert len(A._guides.lines) == 1 and len(ring) == 1 and ring[0][2] == "?", "a line in A, a '?' ring in B"
d = dist_to_guides(A, cams[0].project(spot)[0])
assert d < 3.0, f"the look-here line in camera A misses the truth by {d:.2f} px"
QTest.keyClick(win, Qt.Key_Escape)
pump(0.05)
assert win._epi_probe is None and not A._guides.lines, "Esc clears the look-here line"
print(f"a plain click with nothing selected places nothing and says so; Alt+click shows the spot's line "
      f"({d:.2f} px), Esc clears it OK")

# ---- 8. a skeleton reaches every camera -----------------------------------------------------
win._apply_skeleton_template(template_by_name("Lizard / iguana"))
pump(0.05)
names_a, names_b = [q.name for q in sa.points], [q.name for q in sb.points]
assert set(names_a) == set(names_b) and len(names_a) > 5, (len(names_a), len(names_b))
assert sa.skeleton and sb.skeleton, "the camera without a skeleton adopts it"
print(f"a skeleton applied in one camera gives every camera its {len(names_a)} landmarks OK")

# ---- 9. a camera added later gets every landmark --------------------------------------------
cam3 = os.path.join(OUT, "shared_cam2.mp4")
shutil.copyfile(paths[1], cam3)
forget_recovery(cam3)
assert win._add_view(cam3)
pump(0.2)
sc = p.sessions[2]
assert [q.name for q in sc.points] == names_a, "the new camera has the shared list, in the same order"
# (G25) the two calibrated cameras keep their guides; the new one has none and 3D still waits for it
assert win._guides_ready() and win._cal_cam(2) is None and win._cal_cam(1) is not None
assert win._need_calibration("test") is False, "3D still needs every camera calibrated"
win._remove_view(2)
pump(0.2)
assert win._guides_ready()
print("a camera added later receives every landmark, in the same order; guides stay among the calibrated OK")

# ---- 10. an older project with different lists opens with one list, unchanged on disk -----
s1 = TrackingSession(paths[0], N, FPS, W, H)
s2 = TrackingSession(paths[1], N, FPS, W, H)
s1.add_point(5, 100, 100, name="eye")
s2.add_point(5, 200, 200, name="tail")
OLD = os.path.join(OUT, "shared_landmarks_old.kinetrace")
if os.path.isdir(OLD):
    __import__("shutil").rmtree(OLD)
from kinetrace import projectfile  # noqa: E402
# an older project: saved as one file (the form before I145), which opening must not rewrite
projectfile.save(Project([s1, s2], ["camA", "camB"], [0, 0]), OLD, single_file=True)
before = open(OLD, "rb").read()
win._open_project_from_path(OLD)
for _ in range(200):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == 2 \
            and win.project.sessions[0].pid_by_name("eye") is not None:
        break
q = win.project
assert [x.name for x in q.sessions[0].points] == [x.name for x in q.sessions[1].points] == ["eye", "tail"], \
    "one list, in one order (G26)"
assert q.sessions[1].tracked[5, q.sessions[1].pid_by_name("tail")]
assert not q.sessions[0].tracked[:, q.sessions[0].pid_by_name("tail")].any(), "the filled-in point has no data"
assert not q.dirty, "filling in the shared list does not make the project unsaved"
assert open(OLD, "rb").read() == before
print("an older project opens with one shared list, not marked changed OK")

win._dev_probe.wait(15000)
win.close()
for path in paths + [cam3]:
    forget_recovery(path)
print("SHARED LANDMARKS PASSED")
