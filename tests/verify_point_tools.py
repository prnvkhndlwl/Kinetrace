"""(G147) Point tools: swap two points, move a stretch of one point's data to another, fill one point's
empty frames from another, split a point in two at a frame.

  [1] the session operations: every per-point array travels (positions, flags, confidence, hidden,
      radius), only inside the range; a ball's SAM clicks follow its data; silhouette landmarks and a
      ball with a plain point are refused; nothing to do = 0 and nothing changed.
  [2] through the window (offscreen, two cameras, real clicks and keys): two selected points ->
      right-click menu "Swap these two points…" -> the dialog (real clicks on its range and OK) ->
      swapped; Ctrl+Z puts them back; Edit -> Point Tools… "Move A's data to B" over typed frames;
      "Fill B's empty frames from A"; the point menu's "Split it into a new point" -> the new point in
      BOTH cameras' lists, one Ctrl+Z removes it everywhere; a refusal disables OK and says why.

.venv\\Scripts\\python.exe tests\\verify_point_tools.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "point_tools")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMenu, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication.instance() or QApplication([])
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.pointedit import PointToolsDialog  # noqa: E402
from kinetrace.session import POINT_ARRAYS, TrackingSession  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def arrays(s, pid, f0=0, f1=None):
    f1 = s.n_frames - 1 if f1 is None else f1
    return {a.name: getattr(s, a.name)[f0:f1 + 1, pid].copy() for a in POINT_ARRAYS}


def same(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


# ------------------------------------------------------------------ [1] the session
print("[1] the session operations")
s = TrackingSession("x.mp4", 100, 30.0, 640, 480)
a = s.add_point(0, 10, 10, name="left foot")
b = s.add_point(0, 50, 50, name="right foot")
for f in range(100):
    s.set_position(f, a, 10 + f, 10)
    s.set_position(f, b, 50 + f, 50)
s.confidence[:, a] = 0.9
s.confidence[:, b] = 0.4
s.occluded[30, a] = True
s.manual[:, :] = False
s.manual[40, b] = True
A0, B0 = arrays(s, a), arrays(s, b)
n = s.swap_points(a, b, 20, 59)
check(n == 40, "swap counts the frames with data in the range", n)
inside = slice(20, 60)
check(all(np.array_equal(getattr(s, k.name)[inside, a], B0[k.name][inside], equal_nan=True) for k in POINT_ARRAYS)
      and all(np.array_equal(getattr(s, k.name)[inside, b], A0[k.name][inside], equal_nan=True) for k in POINT_ARRAYS),
      "inside the range every array is exchanged (positions, flags, confidence, hidden)")
check(same({k: v[:20] for k, v in arrays(s, a).items()}, {k: v[:20] for k, v in A0.items()})
      and same({k: v[60:] for k, v in arrays(s, b).items()}, {k: v[60:] for k, v in B0.items()}),
      "outside the range nothing moves")
check(s.occluded[30, b] and not s.occluded[30, a] and s.manual[40, a], "the hidden mark and the hand-placed flag travel")
s.swap_points(a, b, 20, 59)
check(same(arrays(s, a), A0) and same(arrays(s, b), B0), "swapping twice gives back the original")

s.clear_window([b], 70, 99)
n = s.move_point_data(a, b, 60, 99)
check(n == 40 and not s.tracked[60:100, a].any() and np.allclose(s.tracks[70:100, b, 0], 10 + np.arange(70, 100)),
      "move: B takes A's frames in the range (also where B had none), A is cleared there", n)
s2 = TrackingSession("x.mp4", 50, 30.0, 640, 480)
c = s2.add_point(0, 5, 5, name="tail a")
d = s2.add_point(0, 5, 5, name="tail b")
s2.clear_window([c, d], 0, 49)
for f in range(0, 20):
    s2.set_position(f, c, f, 1)
for f in range(15, 50):
    s2.set_position(f, d, 100 + f, 2)
n = s2.fill_point_gaps(d, c, 0, 49)
check(n == 30 and s2.tracked[:, c].all() and np.allclose(s2.tracks[20:, c, 0], 100 + np.arange(20, 50))
      and np.allclose(s2.tracks[15:20, c, 0], np.arange(15, 20)) and s2.tracked[15:, d].all(),
      "fill: only B's empty frames take A's data; B's own frames and A stay", n)
check(s2.fill_point_gaps(d, c, 0, 49) == 0, "fill again: nothing left to fill")
new, n = s2.split_point(c, 30)
check(new == 2 and n == 20 and s2.points[new].name == "tail a (2)" and not s2.tracked[30:, c].any()
      and s2.tracked[30:, new].all() and not s2.tracked[:30, new].any(),
      "split: from the frame on the data is a new point, the old one ends the frame before", (new, n))
check(s2.split_point(c, 45) == (-1, 0), "split where the point has no data: nothing")

s3 = TrackingSession("x.mp4", 30, 30.0, 640, 480)
p = s3.add_point(0, 5, 5)
q = s3.add_landmark("tail tip", source="silhouette", spec="tip")
b1 = s3.add_ball(0, 100, 100)
b2 = s3.add_ball(5, 200, 200)
s3.add_ball_prompt(b1, 10, 101, 101)
check(s3.point_tool_problem(p, q) is not None and "silhouette" in s3.point_tool_problem(p, q),
      "a silhouette landmark is refused, and the reason says so")
check(s3.point_tool_problem(p, b1) is not None and s3.swap_points(p, b1, 0, 29) == 0,
      "a plain point and a ball marker are refused")
check(s3.point_tool_problem(p, p) is not None, "the same point twice is refused")
s3.swap_points(b1, b2, 0, 29)
check(set(s3.points[b2].ball_prompts) == {0, 10} and set(s3.points[b1].ball_prompts) == {5},
      "a ball's SAM clicks follow its data", (s3.points[b1].ball_prompts, s3.points[b2].ball_prompts))
v0 = s3.data_version
check(s3.swap_points(p, b1, 0, 29) == 0 and s3.data_version == v0, "a refused tool changes nothing")


# ------------------------------------------------------------------ [2] the window
print("[2] through the window")


def clip(name, n=60):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    rng = np.random.RandomState(3)
    base = rng.randint(0, 255, (280, 320 + 4 * n, 3)).astype(np.uint8)
    for f in range(n):
        vw.write(np.ascontiguousarray(base[20:260, 4 * f:4 * f + 320]))
    vw.release()
    return path


VA, VB = clip("a.mp4"), clip("b.mp4")
w = MainWindow()
w.resize(1400, 900)
w.show()
app.setActiveWindow(w)
w._open_video(VA)
for _ in range(300):
    pump(0.05)
    if w.state == READY and w.project is not None and not w._loading:
        break
pump(0.3)
check(w._add_view(VB), "a second camera")
pump(0.3)
s = w.session
w._begin_edit()
pa = s.add_point(0, 20, 20, name="left")
pb = s.add_point(0, 200, 200, name="right")
for f in range(60):
    s.set_position(f, pa, 20 + f, 20)
    s.set_position(f, pb, 200 - f, 200)
w._share_landmarks()
w._refresh_point_list()
w._undo_snap = None
pump(0.2)
A0, B0 = arrays(s, pa), arrays(s, pb)
check(w.act_point_tools.isEnabled(), "Edit -> Point Tools… is enabled with points")

# select both rows in POINTS with real clicks (click, then Ctrl+click)
lw = w.layers
r0 = lw.visualItemRect(lw.point_item(pa)).center()
r1 = lw.visualItemRect(lw.point_item(pb)).center()
QTest.mouseClick(lw.viewport(), Qt.LeftButton, Qt.NoModifier, r0)
QTest.mouseClick(lw.viewport(), Qt.LeftButton, Qt.ControlModifier, r1)
pump(0.2)
sel = sorted(w._selected_pids())
check(sel == [pa, pb], "both points selected by real clicks", sel)

DRIVE = {"log": []}


def drive(steps):
    """Run `steps(dialog)` on the Point Tools dialog once it is up (the modal exec runs meanwhile)."""
    def tick():
        dlg = QApplication.activeModalWidget()
        if not isinstance(dlg, PointToolsDialog):
            QTimer.singleShot(30, tick)
            return
        try:
            steps(dlg)
        except Exception as e:      # noqa: BLE001
            DRIVE["log"].append(repr(e))
            dlg.reject()
    QTimer.singleShot(30, tick)


def click(widget):
    QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, QPoint(8, widget.height() // 2))
    pump(0.05)


def ok_button(dlg):
    return dlg.buttons.button(QDialogButtonBox.Ok)


menu, acts = w._build_multi_menu(sel)
check(acts["swap_two"].isEnabled() and acts["swap_two"].text() == "Swap these two points…",
      "the two-point menu offers Swap these two points…")


def swap_steps(dlg):
    DRIVE["swap_op"] = dlg.op()
    click(dlg.r_all)
    DRIVE["summary"] = dlg.summary.text()
    QTest.mouseClick(ok_button(dlg), Qt.LeftButton)


drive(swap_steps)
w._multi_menu_action(acts["swap_two"], acts, sel)
menu.deleteLater()
pump(0.3)
check(DRIVE.get("swap_op") == "swap" and "they exchange it" in DRIVE.get("summary", ""), "the dialog opened on Swap",
      DRIVE)
check(same(arrays(s, pa), B0) and same(arrays(s, pb), A0), "OK swapped the two points over the whole video")
app.setActiveWindow(w)
QTest.keyClick(w, Qt.Key_Z, Qt.ControlModifier)
pump(0.3)
check(same(arrays(s, pa), A0) and same(arrays(s, pb), B0), "Ctrl+Z (a real key press) put them back")


def move_steps(dlg):
    click(dlg.op_buttons["move"])
    dlg.combo_a.setCurrentIndex(pa)
    dlg.combo_b.setCurrentIndex(pb)
    dlg.spin0.selectAll()
    QTest.keyClicks(dlg.spin0, "10")
    dlg.spin1.selectAll()
    QTest.keyClicks(dlg.spin1, "19")
    pump(0.05)
    DRIVE["range"] = dlg.span()
    QTest.mouseClick(ok_button(dlg), Qt.LeftButton)


drive(move_steps)
w.act_point_tools.trigger()
pump(0.3)
check(DRIVE.get("range") == (10, 19), "typed frames 10-19 became the range", DRIVE.get("range"))
check(not s.tracked[10:20, pa].any() and np.allclose(s.tracks[10:20, pb, 0], 20 + np.arange(10, 20))
      and np.allclose(s.tracks[20:, pb, 0], 200 - np.arange(20, 60)),
      "Move: B took A's frames 10-19 and A is empty there, B's other frames are its own")


def fill_steps(dlg):
    click(dlg.op_buttons["fill"])
    dlg.combo_a.setCurrentIndex(pb)
    dlg.combo_b.setCurrentIndex(pa)
    click(dlg.r_all)
    DRIVE["fill_summary"] = dlg.summary.text()
    QTest.mouseClick(ok_button(dlg), Qt.LeftButton)


drive(fill_steps)
w.act_point_tools.trigger()
pump(0.3)
check("10 empty frame(s)" in DRIVE.get("fill_summary", ""), "the dialog counts the empty frames it will fill",
      DRIVE.get("fill_summary"))
check(s.tracked[:, pa].all() and np.allclose(s.tracks[10:20, pa, 0], 20 + np.arange(10, 20)),
      "Fill: A's empty frames 10-19 came back from B")
QTest.keyClick(w, Qt.Key_Z, Qt.ControlModifier)
pump(0.2)

# the point menu: Split it into a new point from this frame
w._goto(40)
pump(0.3)
n_other = w.project.sessions[1].n_points
m = QMenu()
acts1: dict = {}
w._extend_point_menu(m, acts1, pa)
check(acts1["split_here"].isEnabled() and "from frame 40" in acts1["split_here"].text(),
      "the point menu offers Split it into a new point from frame 40", acts1["split_here"].text())
w._point_menu_extra_action(acts1["split_here"], acts1, pa)
m.deleteLater()
pump(0.3)
newp = s.pid_by_name("left (2)")
check(newp is not None and s.tracked[40:, newp].all() and not s.tracked[40:, pa].any() and s.tracked[:40, pa].all(),
      "split: left (2) has frames 40-59, left ends at 39")
check(w.project.sessions[1].pid_by_name("left (2)") is not None and w.project.sessions[1].n_points == n_other + 1,
      "the new point is in the other camera's list too")
app.setActiveWindow(w)
QTest.keyClick(w, Qt.Key_Z, Qt.ControlModifier)
pump(0.3)
check(s.pid_by_name("left (2)") is None and w.project.sessions[1].pid_by_name("left (2)") is None
      and s.tracked[:, pa].all(), "one Ctrl+Z removes it from every camera and gives left its frames back")

# a refusal: a silhouette landmark cannot be swapped
lm = s.add_landmark("tail tip", source="silhouette", spec="tip")
w._refresh_point_list()


def refuse_steps(dlg):
    dlg.combo_a.setCurrentIndex(pa)
    dlg.combo_b.setCurrentIndex(lm)
    pump(0.05)
    DRIVE["ok_enabled"] = ok_button(dlg).isEnabled()
    DRIVE["why"] = dlg.summary.text()
    dlg.reject()


drive(refuse_steps)
w.act_point_tools.trigger()
pump(0.3)
check(DRIVE.get("ok_enabled") is False and "silhouette" in DRIVE.get("why", ""),
      "a silhouette landmark disables OK and the dialog says why", DRIVE.get("why"))
check(not DRIVE["log"], "the dialog drivers ran without errors", DRIVE["log"])

w.project.dirty = False
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_point_tools: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
