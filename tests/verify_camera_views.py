"""(G168-G172) Many cameras on screen, and their order.

  [1] G168  the VIDEO's zoom buttons (lower right) stay in place, visible, while the view is zoomed and
            panned -- they used to scroll away with the picture and came back only on Fit
  [2] G169  CAMERAS: a compact row per camera with an eye (show / hide its view; hidden views are not
            decoded) and a disclosure that opens the row's controls (Align here, offset, fps, remove)
  [3] G170  the views are arranged by dragging a view's title bar onto another view (display only:
            the cameras keep their numbers); the arrangement and the hidden cameras come back with
            the project and never make it unsaved
  [4] G171  File -> Open Folder of Videos: the cameras' ORDER is chosen (Move up / Move down); camera 1,
            the reference, is the first ticked video
  [5] G172  Add videos (several) asks their order; CAMERAS -> Camera order... reorders an open project:
            each camera keeps its data, offsets are re-based on the new reference, a calibration
            follows its cameras, the 3D result is dropped

.venv\\Scripts\\python.exe tests\\verify_camera_views.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "camera_views")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

ANSWER = {"q": QMessageBox.Yes}
QMessageBox.question = staticmethod(lambda *a, **k: ANSWER["q"])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])

from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

FAILS = []
W, H, N = 320, 240, 30


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.15):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def clip(name, shade, n=N):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for f in range(n):
        im = np.full((H, W, 3), shade, np.uint8)
        cv2.circle(im, (40 + 3 * f, 120), 12, (230, 230, 230), -1)
        cv2.putText(im, name[:4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(im)
    vw.release()
    return path


def wait_ready(win, n_views=None, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(0.05)
        if (win.state == READY and win.project is not None and not win._loading
                and (n_views is None or win.project.n_views == n_views)):
            return True
    return False


def wait(cond, timeout):
    t0 = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t0 > timeout:
            return False
    return True


VIDS = [clip(f"cam{k}.mp4", 40 + 12 * k) for k in range(1, 5)]
win = MainWindow()
win.resize(1500, 950)
win.show()
win._open_video(VIDS[0])
check(wait_ready(win), "(setup) the first video opens")

# ------------------------------------------------------------------ [1] the video's zoom buttons
print("[1] the video's zoom buttons stay put while zoomed (G168)")


def bar_in_corner(cv):
    """The zoom bar is shown, inside the picture area, in its lower-right corner."""
    bar = cv._zoom_bar
    r = QRect(bar.mapTo(cv, QPoint(0, 0)), bar.size())
    vp = cv.viewport().geometry()
    return (bar.isVisible() and vp.contains(r) and abs(r.right() - (vp.right() - 6)) <= 2
            and abs(r.bottom() - (vp.bottom() - 6)) <= 2), (r, vp, bar.isVisible())


cv = win.canvas
pump(0.3)
ok, d = bar_in_corner(cv)
check(ok, "fitted: the zoom buttons are in the picture's lower-right corner", d)
for _ in range(4):
    QTest.mouseClick(cv.btn_zoom_in, Qt.LeftButton)
    pump(0.05)
ok, d = bar_in_corner(cv)
check(cv.transform().m11() > 2.0 and ok, "zoomed in 4 times with its own button: still there, same corner",
      (cv.transform().m11(), d))
vp_c = cv.viewport().rect().center()
QTest.mousePress(cv.viewport(), Qt.MiddleButton, Qt.NoModifier, vp_c)
for k in range(1, 6):
    QTest.mouseMove(cv.viewport(), vp_c + QPoint(-25 * k, -15 * k))
    pump(0.02)
QTest.mouseRelease(cv.viewport(), Qt.MiddleButton, Qt.NoModifier, vp_c + QPoint(-125, -75))
pump(0.1)
ok, d = bar_in_corner(cv)
check(ok, "panned while zoomed: still there", d)
QTest.mouseClick(cv.btn_zoom_out, Qt.LeftButton)
pump(0.05)
ok, d = bar_in_corner(cv)
check(ok, "its zoom-out button is still clickable there and the bar stays", d)
QTest.mouseClick(cv.btn_zoom_fit, Qt.LeftButton)
pump(0.05)

# ------------------------------------------------------------------ [2] CAMERAS: eyes, open / shut rows
print("[2] CAMERAS: a compact row per camera, its eye, its controls behind its disclosure (G169)")
for v in VIDS[1:]:
    check(win._add_view(v), f"(setup) {os.path.basename(v)} added")
check(wait_ready(win, 4), "(setup) four cameras")
p = win.project
grid, panel = win.grid, win.cameras
rows = panel._rows
pump(0.3)
lst = panel.list
check(len(rows) == 4 and not any(r.expanded() for r in rows) and not any(r.details.isVisible() for r in rows),
      "every row starts shut: one compact line each")
check(not lst.verticalScrollBar().isVisible() and all(lst.visualItemRect(lst.item(k)).bottom() <= lst.viewport().height()
                                                      for k in range(4)),
      "the four cameras fit the list at once, without scrolling", lst.height())
check(rows[0].name.text().startswith("1  ") and "reference" in rows[0].name.text() and rows[2].name.text().startswith("3  "),
      "each row says its number in the camera order; camera 1 says it is the reference", rows[0].name.text())
check(rows[1].summary.isVisible() and rows[1].summary.text().startswith("offset"),
      "a shut row still shows its offset", rows[1].summary.text())
check(not rows[p.active].btn_eye.isEnabled() and rows[1].btn_eye.isEnabled(),
      "the working camera's eye is off-limits (it is always shown); the others' can be clicked")
win._goto(10, force=True)
pump(0.6)
p.dirty = False
QTest.mouseClick(rows[1].btn_eye, Qt.LeftButton)
pump(0.4)
check(1 not in grid.visible_indices() and not grid._cells[1].isVisible() and grid._cells[2].isVisible(),
      "its eye hides camera 2's view; the others stay", grid.visible_indices())
check(win._views[1].seek is None and win._views[1].want_frame is None,
      "a hidden view is not decoded (its decoder is given back)")
check(not p.dirty, "hiding a view is display state: the project is not made unsaved")
check(panel.btn_show_all.isVisible() and "1 hidden" in panel.btn_show_all.text() and win.act_show_all_views.isEnabled(),
      "Show all says how many are hidden", panel.btn_show_all.text())
win._goto(14, force=True)
pump(0.4)
check(win._views[1].want_frame is None, "moving the playhead asks nothing of a hidden camera")
QTest.mouseClick(panel.btn_show_all, Qt.LeftButton)
pump(0.4)
check(grid.visible_indices() == [0, 1, 2, 3] and rows[1].btn_eye.isChecked() and not panel.btn_show_all.isVisible(),
      "Show all brings it back (its eye too)")
check(win._views[1].want_frame is not None, "... and it decodes again")
h0 = lst.visualItemRect(lst.item(1)).height()
QTest.mouseClick(rows[1].btn_expand, Qt.LeftButton)
pump(0.2)
check(rows[1].expanded() and rows[1].details.isVisible() and rows[1].btn_align.isVisible() and rows[1].spin.isVisible()
      and rows[1].btn_fps.isVisible() and lst.visualItemRect(lst.item(1)).height() > h0 + 20,
      "its disclosure opens the row: Align here, the offset box, the frame rate (the row grows)",
      (h0, lst.visualItemRect(lst.item(1)).height()))
check(not rows[1].summary.isVisible(), "an open row shows its offset in the box, not twice")
QTest.mouseClick(rows[1].btn_expand, Qt.LeftButton)
pump(0.2)
check(not rows[1].details.isVisible() and lst.visualItemRect(lst.item(1)).height() == h0, "and shuts it again")
QTest.mouseClick(rows[2].btn_eye, Qt.LeftButton)
pump(0.3)
check(2 not in grid.visible_indices(), "(camera 3 hidden)")
QTest.mouseClick(rows[2].name, Qt.LeftButton, Qt.NoModifier, QPoint(4, rows[2].name.height() // 2))
pump(0.6)
check(p.active == 2 and 2 in grid.visible_indices() and rows[2].btn_eye.isChecked() and not rows[2].btn_eye.isEnabled(),
      "clicking a hidden camera's row works in it: it is shown while it is the working camera",
      (p.active, grid.visible_indices()))
win._set_active_view(0)
pump(0.4)
check(2 not in grid.visible_indices() and not rows[2].btn_eye.isChecked() and 2 in grid.hidden(),
      "back in camera 1, camera 3 is hidden again: working in a camera never changes its eye (G174)",
      grid.visible_indices())
QTest.mouseClick(rows[2].btn_eye, Qt.LeftButton)
pump(0.3)
win.act_solo.setChecked(True)
pump(0.3)
check(grid.visible_indices() == [0] and not rows[1].btn_eye.isEnabled() and "Only the working camera" in rows[1].btn_eye.toolTip(),
      "Only the working camera (Ctrl+2): the eyes rest and say why")
win.act_solo.setChecked(False)
pump(0.3)
check(grid.visible_indices() == [0, 1, 2, 3] and rows[1].btn_eye.isEnabled(), "... and come back with the views")

# ------------------------------------------------------------------ [3] arranging the views
print("[3] views arranged by their title bars; display only; kept with the project (G170)")
import kinetrace.viewgrid as vg  # noqa: E402
from PySide6.QtCore import QMimeData  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402


class FakeDrag:
    made = []

    def __init__(self, src):
        FakeDrag.made.append(self)
        self.md = None

    def setMimeData(self, md):            # noqa: N802
        self.md = md

    def setPixmap(self, pm):              # noqa: N802
        pass

    def setHotSpot(self, pt):             # noqa: N802
        pass

    def exec(self, *a):
        return Qt.MoveAction


real_drag = vg.QDrag
vg.QDrag = FakeDrag
try:
    cap = grid._cells[0].caption
    c0 = QPoint(20, cap.height() // 2)
    QTest.mousePress(cap, Qt.LeftButton, Qt.NoModifier, c0)
    QTest.mouseMove(cap, c0 + QPoint(40, 2))
    QTest.mouseRelease(cap, Qt.LeftButton, Qt.NoModifier, c0 + QPoint(40, 2))
    pump(0.1)
finally:
    vg.QDrag = real_drag
md = FakeDrag.made[-1].md if FakeDrag.made else None
check(md is not None and bytes(md.data(vg.VIEW_MIME)).decode() == "0",
      "dragging camera 1's title bar starts a drag of camera 1's view")
names_before = list(p.names)


def drop(md, onto: int):
    """The drop as Qt delivers it: onto the target view's picture, which passes it up to its view."""
    target = grid._cells[onto].canvas.viewport()
    pos = target.rect().center()
    enter = QDragEnterEvent(pos, Qt.MoveAction, md, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, enter)
    ev = QDropEvent(QPointF(pos), Qt.MoveAction, md, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, ev)
    pump(0.3)
    return enter.isAccepted()


p.dirty = False
accepted = drop(md, 2)
check(accepted and grid.display_order() == [1, 2, 0, 3],
      "dropped on camera 3's view: camera 1 takes its place, cameras 2 and 3 shift", grid.display_order())
lay_ = grid._lay


def cell_at(i):
    idx = lay_.indexOf(grid._cells[i])
    r, c, _rs, _cs = lay_.getItemPosition(idx)
    return r, c


check(cell_at(1) == (0, 0) and cell_at(0) == (1, 0), "the grid shows them so: camera 2 top-left, camera 1 below",
      (cell_at(1), cell_at(0)))
check(list(p.names) == names_before and p.active == 0 and not p.dirty,
      "display only: the cameras keep their numbers, the working camera stays, nothing unsaved")
md2 = QMimeData()
md2.setData(vg.VIEW_MIME, b"3")
drop(md2, 1)
check(grid.display_order() == [3, 1, 2, 0], "camera 4 dropped on camera 2 takes the first place",
      grid.display_order())
menu, acts = grid.build_view_menu(0)
check(acts["camera_order"].isEnabled() and not acts["hide"].isEnabled(),
      "a view's title bar menu: back to camera order; the working camera cannot be hidden")
menu.deleteLater()
menu, acts = grid.build_view_menu(3)
grid.view_menu_action(acts["hide"], acts, 3)
menu.deleteLater()
pump(0.3)
check(3 not in grid.visible_indices() and not rows[3].btn_eye.isChecked(), "Hide from the title bar menu: its eye follows")
target = os.path.join(OUT, "views.kinetrace")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
win.project_path = None
check(win._save_project(), "(setup) saved")
p.dirty = False
order_names = [p.name(i) for i in grid.display_order()]
win._open_project_from_path(target)
check(wait_ready(win, 4), "(setup) the project opened again")
pump(0.6)
p = win.project
grid, panel = win.grid, win.cameras
rows = panel._rows
check([p.name(i) for i in grid.display_order()] == order_names and {p.name(i) for i in grid.hidden()} == {"cam4"},
      "the arrangement and the hidden camera come back with the project", ([p.name(i) for i in grid.display_order()],
                                                                          grid.hidden()))
win._views_in_camera_order()
win._show_all_views()
pump(0.3)
check(grid.display_order() == [0, 1, 2, 3] and not grid.hidden(), "View -> Other cameras: back to camera order, all shown")

# ------------------------------------------------------------------ [5] camera order: add with order, reorder
print("[5] the cameras' order: a project reordered, videos added in order (G172)")
from kinetrace import cameraorder  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, Reconstruction  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

# the model: every mapping between two cameras is the same in any order (fractional offsets, rates != 1)
ss = [TrackingSession(f"m{k}.mp4", 400, fps, W, H) for k, fps in enumerate((60.0, 120.0, 30.0, 240.0))]
pm = Project(ss, ["a", "b", "c", "d"], [0.0, -7.25, 3.5, 12.0])
pairs = [(x, y) for x in "abcd" for y in "abcd"]


def mapping(pr):
    at = {n: i for i, n in enumerate(pr.names)}
    return {(x, y, f): pr.map_frame_exact(at[x], at[y], f) for x, y in pairs for f in (0, 17, 133)}


m0 = mapping(pm)
pm.calibration = Calibration([CameraCalibration(np.arange(11.0) + 100 * k, W, H, NoUndistort(), 0.0, False)
                              for k in range(4)])
pm.reconstruction = object()
pm.active = 1
check(pm.reorder_views([2, 0, 3, 1]) and pm.names == ["c", "a", "d", "b"] and pm.offsets[0] == 0.0
      and pm.rates[0] == 1.0 and pm.active == 3 and pm.reconstruction is None,
      "reordered: camera c is camera 1 (offset 0, rate 1), the working camera kept, the 3D result dropped",
      (pm.names, pm.offsets, pm.rates))
m1 = mapping(pm)
check(max(abs(m0[k] - m1[k]) for k in m0) < 1e-9, "every mapping between two cameras is unchanged (re-based)",
      max(abs(m0[k] - m1[k]) for k in m0))
check([c.coefs[0] for c in pm.calibration.cameras] == [200.0, 0.0, 300.0, 100.0] and pm.sessions[0] is ss[2],
      "each camera keeps its calibration and its data")
pm.calibration.cameras = pm.calibration.cameras[:2]
check(pm.order_problem([2, 0, 1, 3]) and pm.order_problem([1, 0, 3, 2]) is None,
      "a calibration of the first two cameras: they must stay first (in any order)")

# the app: CAMERAS -> Camera order... with its dialog's own buttons
win._on_view_offset(1, -3)
win._on_view_offset(2, 5)
win._on_view_offset(3, 2)
pump(0.3)
p.calibration = Calibration([CameraCalibration(np.arange(11.0) + 100 * k, W, H, NoUndistort(), 0.0, False)
                             for k in range(4)])
win._set_active_view(1)
win._goto(12, force=True)
pump(0.5)
s_cam2 = p.sessions[1]
s_cam2.add_point(12, 50.0, 60.0, name="nose")
win._refresh_point_list()
p.reconstruction = Reconstruction(0, [], np.zeros((1, 0, 3)), np.zeros((1, 0)), np.zeros((1, 0), int))
grid.move_view(0, 3)                              # a custom arrangement: it must follow its cameras
grid.set_hidden(3, True)
pump(0.3)
shown_names = [p.name(i) for i in grid.display_order()]
hidden_names = {p.name(i) for i in grid.hidden()}
m_app = {(x, y): p.map_frame_exact(p.names.index(x), p.names.index(y), 20) for x in p.names for y in p.names}
seen = {}


def run_dialog(dlg):
    seen["entries"] = [dlg.list.item(r).text() for r in range(dlg.list.count())]
    dlg.list.setCurrentRow(2)                      # cam3 ...
    QTest.mouseClick(dlg.btn_up, Qt.LeftButton)
    QTest.mouseClick(dlg.btn_up, Qt.LeftButton)    # ... to the top: camera 1
    seen["first"] = dlg.list.item(0).text()
    QTest.mouseClick(dlg.buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)
    return dlg.result()


real_exec = cameraorder.CameraOrderDialog.exec
cameraorder.CameraOrderDialog.exec = run_dialog
try:
    QTest.mouseClick(panel.btn_order, Qt.LeftButton)
    pump(0.6)
finally:
    cameraorder.CameraOrderDialog.exec = real_exec
check(seen.get("entries", [""])[0].startswith("1   cam1") and "reference" in seen.get("entries", [""])[0]
      and seen.get("first", "").startswith("1   cam3") and "reference" in seen.get("first", ""),
      "the dialog numbers the cameras and says which is the reference, as rows move", seen)
check(p.names == ["cam3", "cam1", "cam2", "cam4"] and p.offsets[0] == 0.0, "Camera order: cam3 is camera 1 now",
      (p.names, p.offsets))
check(all(abs(p.map_frame_exact(p.names.index(x), p.names.index(y), 20) - v) < 1e-9 for (x, y), v in m_app.items()),
      "the cameras stay in sync (every offset re-based on the new reference)")
check(p.session is s_cam2 and p.name(p.active) == "cam2" and win.current == 12 and p.sessions[2].pid_by_name("nose") is not None,
      "the working camera is still cam2 at frame 12, with its point")
check([round(c.coefs[0]) for c in p.calibration.cameras] == [200, 0, 100, 300], "the calibration follows its cameras")
check(p.reconstruction is None, "the 3D result is dropped (made in the old order)")
check(all(os.path.basename(win._views[k].info.path) == os.path.basename(p.sessions[k].video_path) for k in range(4))
      and all(grid.canvas(k)._native_size == (W, H) for k in range(4)),
      "each view shows its own camera at its new number")
check([p.name(i) for i in grid.display_order()] == shown_names and {p.name(i) for i in grid.hidden()} == hidden_names,
      "the arrangement on screen follows the cameras (each stays where it was, hidden stays hidden)")
check(rows[0].name.text().startswith("1  cam3") and "reference" in rows[0].name.text(), "CAMERAS numbers them anew",
      rows[0].name.text())
check(p.dirty, "the order is part of the project: unsaved until saved")

# adding videos: the dialog lists the project's cameras and the new ones; the user sets the order
NEW = [clip("cam5.mp4", 120), clip("cam6.mp4", 140)]
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (list(reversed(NEW)), ""))   # the file dialog's own order


def add_dialog(dlg):
    seen["add"] = [dlg.list.item(r).text() for r in range(dlg.list.count())]
    dlg.list.setCurrentRow(dlg.list.count() - 1)  # cam6 ...
    for _ in range(dlg.list.count() - 1):
        QTest.mouseClick(dlg.btn_up, Qt.LeftButton)   # ... to camera 1
    ok_btn = dlg.buttons.button(QDialogButtonBox.Ok)
    seen["ok_enabled"], seen["note"] = ok_btn.isEnabled(), dlg.note.text()
    if ok_btn.isEnabled():
        QTest.mouseClick(ok_btn, Qt.LeftButton)
    else:
        QTest.mouseClick(dlg.buttons.button(QDialogButtonBox.Cancel), Qt.LeftButton)
    return dlg.result()


cameraorder.CameraOrderDialog.exec = add_dialog
try:
    # the calibration covers cameras 1-4: a new video cannot go before them (it would take a calibrated
    # camera's number) -- OK greys out and the dialog says why
    QTest.mouseClick(panel.btn_add, Qt.LeftButton)
    pump(0.3)
    check(not seen.get("ok_enabled", True) and "calibration covers cameras 1-4" in seen.get("note", "")
          and p.n_views == 4, "with a calibration of cameras 1-4 a new video cannot become camera 1: refused, and why",
          seen.get("note"))
    p.calibration = None
    m_old = {(x, y): p.map_frame_exact(p.names.index(x), p.names.index(y), 20) for x in p.names for y in p.names}
    QTest.mouseClick(panel.btn_add, Qt.LeftButton)
    check(wait_ready(win, 6), "(setup) the two videos are added")
finally:
    cameraorder.CameraOrderDialog.exec = real_exec
pump(0.5)
add_rows = seen.get("add", [])
check(len(add_rows) == 6 and add_rows[4].startswith("5   cam5") and "new" in add_rows[4] and add_rows[5].startswith("6   cam6"),
      "Add video: the project's cameras, then the new ones in name order (not the file dialog's), marked new", add_rows)
check(p.names == ["cam6", "cam3", "cam1", "cam2", "cam4", "cam5"],
      "they go where the user put them: cam6 is camera 1", p.names)
check(all(abs(p.map_frame_exact(p.names.index(x), p.names.index(y), 20) - v) < 1e-9 for (x, y), v in m_old.items()),
      "the cameras already there stay in sync with each other")

# ------------------------------------------------------------------ [4] Open Folder of Videos: camera order
print("[4] Open Folder of Videos: the cameras' order (G171)")
from kinetrace.folderimport import VideoFolderDialog  # noqa: E402

FOLD = os.path.join(OUT, "folder")
os.makedirs(FOLD, exist_ok=True)
for src, dst in ((VIDS[0], "cam1.mp4"), (VIDS[1], "cam2.mp4"), (VIDS[2], "cam10.mp4")):
    shutil.copyfile(src, os.path.join(FOLD, dst))
dlg = VideoFolderDialog(win, FOLD)
for _ in range(100):
    pump(0.05)
    if len(dlg._headers) == 3:
        break
check(dlg.order_names() == ["cam1.mp4", "cam2.mp4", "cam10.mp4"] and dlg.table.item(0, 1).text() == "1 · reference",
      "the folder's order to start with (cam2 before cam10); camera 1 is the reference")
dlg.table.selectRow(2)
QTest.mouseClick(dlg.btn_up, Qt.LeftButton)
QTest.mouseClick(dlg.btn_up, Qt.LeftButton)
check(dlg.order_names() == ["cam10.mp4", "cam1.mp4", "cam2.mp4"] and dlg.table.item(0, 2).text() == "cam10.mp4"
      and dlg.table.item(0, 1).text() == "1 · reference" and dlg.table.item(1, 1).text() == "2",
      "Move up twice: cam10 is camera 1, the numbers follow", dlg.order_names())
dlg._ticks[2].setChecked(False)                  # leave cam2 out
check(dlg.order_names() == ["cam10.mp4", "cam1.mp4"] and dlg.table.item(2, 1).text() == "",
      "an unticked video has no number")
check(dlg._headers.get(2) is not None and dlg.table.item(0, 3).text() == f"{W} × {H}",
      "each row keeps its own header when rows move")
QTest.mouseClick(dlg.btn_sort, Qt.LeftButton)
check(dlg.order_names() == ["cam1.mp4", "cam10.mp4"], "Sort by name: back to the folder's order", dlg.order_names())
dlg.table.selectRow(2)
QTest.mouseClick(dlg.btn_up, Qt.LeftButton)
QTest.mouseClick(dlg.btn_up, Qt.LeftButton)
dlg.chk_save.setChecked(False)
QTest.mouseClick(dlg.buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)
check([os.path.basename(x) for x in dlg.result_paths] == ["cam10.mp4", "cam1.mp4"], "OK: the ticked videos in that order",
      dlg.result_paths)
paths_ = list(dlg.result_paths)
dlg.deleteLater()
win.project.dirty = False
win._import_folder(paths_, None)
check(wait_ready(win, 2), "(setup) imported")
q = win.project
check(q.names == ["cam10", "cam1"] and os.path.basename(q.sessions[0].video_path) == "cam10.mp4",
      "the project's cameras are in that order: cam10 is camera 1, the reference", q.names)

# ------------------------------------------------------------------ [6] the timeline shows the other cameras
print("[6] a landmark tracked in ANOTHER camera shows in the working camera's lane (G173)")
pump(0.3)
QTest.mouseClick(win.btn_new_point, Qt.LeftButton)          # + Point: in every camera, no data
pump(0.3)
sA, sB = q.sessions[0], q.sessions[1]
name = sA.points[-1].name
pB = sB.pid_by_name(name)
win._on_view_offset(1, 3)                                  # cam1's frame = cam10's + 3
for fb in range(10, 20):                                   # tracked in cam1 only, frames 10-19
    sB.tracks[fb, pB] = (100.0 + fb, 100.0)
    sB.tracked[fb, pB] = True
    sB.visibility[fb, pB] = True
    sB.confidence[fb, pB] = 1.0
sB._touch()
pA = sA.pid_by_name(name)
win._goto(25, force=True)
tl = win.timeline
tl.update()
pump(0.4)
check(not sA.tracked[:, pA].any() and q.active == 0, "(setup) the working camera (cam10) has no data for it")
y0, _y1 = tl._row_rect_y(pA)
img = tl.grab().toImage()


def lit(frame):
    x = int(round((tl._x_edge(frame) + tl._x_edge(frame + 1)) / 2))
    c = img.pixelColor(x, y0 + 6)
    bgc = img.pixelColor(x, y0 + 2)                         # the lane's own background, above the line
    return abs(c.red() - bgc.red()) + abs(c.green() - bgc.green()) + abs(c.blue() - bgc.blue()) > 60


on = [f for f in range(0, 30) if lit(f)]
check(on == list(range(7, 17)), "its lane draws a thin line exactly where cam1 has it: cam1 frames 10-19 = this "
      "camera's 7-16 (through the offset)", on)
xm = int(round((tl._x_edge(10) + tl._x_edge(11)) / 2))
QTest.mouseMove(tl, QPoint(xm, y0 + 6))
pump(0.1)
tip = tl.toolTip()
check("not tracked in this camera here" in tip and "tracked in cam1" in tip, "hovering it names the camera that has it",
      tip[:160])
QTest.mouseMove(tl, QPoint(int(tl._x_edge(25)), y0 + 6))
pump(0.1)
check("tracked in cam1" not in tl.toolTip(), "... and only where it has it")
win._set_active_view(1)
pump(0.4)
check(tl._trk_col is not None and tl._trk_col[:, sB.pid_by_name(name)].any(),
      "working in cam1: its own track is the lane's full bar")

# ------------------------------------------------------------------ [7] every camera, one hidden
print("[7] an every-camera run with a hidden camera: tracked too, and said clearly (G174)")
rng = np.random.default_rng(3)
bg = cv2.GaussianBlur((rng.random((H, W)) * 90 + 40).astype(np.uint8), (0, 0), 1.2)
ND = 160
DOT, GT = [], []
for k in range(2):
    path_ = np.stack([np.linspace(30 + 10 * k, 290, ND), 120 + 25 * np.sin(np.linspace(0, 4, ND) + k)], axis=1)
    vid = os.path.join(OUT, f"dot{k}.mp4")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for f in range(ND):
        im = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        cv2.circle(im, (int(round(path_[f, 0])), int(round(path_[f, 1]))), 2, (255, 255, 255), -1)
        vw.write(im)
    vw.release()
    DOT.append(vid)
    GT.append(path_)
win.project.dirty = False
win._open_video(DOT[0])
check(wait_ready(win, 1), "(setup) dot0 opens")
check(win._add_view(DOT[1]) and wait_ready(win, 2), "(setup) dot1 added")
p = win.project
grid, panel = win.grid, win.cameras
win.act_pm_spot.trigger()                         # Moving spot: no model, a run on the CPU
win._goto(0, force=True)
pump(0.3)
win.btn_add.setChecked(True)
win.canvas.add_requested.emit(float(GT[0][0, 0]), float(GT[0][0, 1]))
pump(0.1)
pname = p.sessions[0].points[0].name
win._goto(1, force=True)
win._on_annotate(*GT[0][1])
win._set_active_view(1)
pump(0.3)
win.layers.setCurrentItem(win.layers.point_item(p.sessions[1].pid_by_name(pname)))
for f in (0, 1):
    win._goto(f, force=True)
    win._on_annotate(*GT[1][f])
win._set_active_view(0)
win._goto(1, force=True)
pump(0.3)
QTest.mouseClick(panel._rows[1].btn_eye, Qt.LeftButton)      # dot1 hidden
pump(0.3)
win.act_track_all.setChecked(True)
win.act_select_all.trigger()
pump(0.1)
win._update_track_button()
check(1 not in grid.visible_indices() and "2 cams (1 hidden)" in win.btn_track.text()
      and "HIDDEN: dot1 is hidden" in win.btn_track.toolTip(),
      "before the press, Track says it: '· 2 cams (1 hidden)', and why in its tooltip", win.btn_track.text())
QTest.keyClick(win, Qt.Key_T)
check(wait(lambda: win._multi is not None or win.state == TRACKING, 20), "T starts the run in both cameras")
toast = win.toast.text()
check("dot1 is hidden" in toast and "tracked too" in toast, "a clear warning: the hidden camera is tracked too, not drawn",
      toast[:200])
check(1 not in grid.visible_indices(), "starting the run did not show the hidden camera (its eye is the user's)")
check(wait(lambda: win._multi is None and win.state == READY, 120), "(setup) the run ends")
pump(0.3)
sb_ = p.sessions[1]
check(int(sb_.tracked[:, sb_.pid_by_name(pname)].sum()) > 20, "the hidden camera was tracked",
      int(sb_.tracked[:, sb_.pid_by_name(pname)].sum()))
check(1 in grid.hidden() and not panel._rows[1].btn_eye.isChecked(), "... and is still hidden after the run")
win._show_tracked_views()                      # what clicking the notice does
pump(0.3)
check(not grid.hidden() and 1 in grid.visible_indices(), "clicking the notice shows every camera")
win.act_track_all.setChecked(False)

# ------------------------------------------------------------------ end
win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_camera_views: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
