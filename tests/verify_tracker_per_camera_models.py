"""A tracker per camera (D8, G216) and "Hand only" (D10, G219) with the real point MODELS: the GPU half of
verify_tracker_per_camera (plan dev/work/2026-10-20-silhouette-hand-mix, PR D). Offscreen; AllTracker and CoTracker3
from models/.

Three cameras (80 frames of test600: as filmed, mirrored, as filmed), four landmarks clicked in each, each
camera's trackers chosen through LAYERS -> Tracker -> Only in this camera:
    camera 1: snout AllTracker, tail CoTracker3, ear AllTracker, fin Hand only (clicked on 0, 20, 40)
    camera 2: snout CoTracker3, tail CoTracker3, ear AllTracker, fin AllTracker
    camera 3: CoTracker3 for all four (a camera holding only CoTracker3 points)
One Track press with Every camera -> every point in every camera is tracked to the end by ITS camera's tracker
(checked against the clip's ground truth), the fin is never tracked in camera 1 (its clicks stay), camera 3 runs
CoTracker3 only. FAILS on the code before the change: the menu has no camera scope (the trackers are then set
straight on each camera's points), camera 2's snout is dropped as "another tracker" and Hand only does not exist.
Run: .venv\\Scripts\\python.exe tests\\verify_tracker_per_camera_models.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "tracker_per_camera_models")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402
from _synth_spots import write_video  # noqa: E402

import kinetrace.app as appmod  # noqa: E402
from kinetrace import alltracker_backend  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.session import POINT_ARRAYS  # noqa: E402

POP = stub_track_choice("keep")
FAILS: list = []


def check(label, cond, detail=None):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001
            cond, detail = False, f"raised {e!r}"
    cond = bool(cond)
    line = ("  ok    " if cond else "  FAIL  ") + label + ("" if cond or detail is None else f"   [{detail}]")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not cond:
        FAILS.append(label)
    return cond


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.003)


def wait(cond, timeout):
    t0 = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t0 > timeout:
            return False
    return True


if not alltracker_backend.available():
    print("AllTracker is not installed here: this suite needs both point models")
    sys.exit(1)

# ---- every model worker the app builds: (video, backend, point ids)
BUILT: list = []
_orig_launch = MainWindow._launch_run


def _spy_launch(self, w, driver=None):
    for x in (driver.workers if driver is not None else [w]):
        BUILT.append({"video": os.path.basename(x.video_path), "backend": x.point_backend,
                      "pids": [sp.pid for sp in x.specs]})
    return _orig_launch(self, w, driver)


MainWindow._launch_run = _spy_launch

# ---- three cameras cut from test600
V600 = os.path.join(ROOT, "test600.mp4")
GT600 = np.load(V600 + ".gt.npz")["gt"]
N = 80
cap = cv2.VideoCapture(V600)
FR = []
for _ in range(N):
    ok, bgr = cap.read()
    FR.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
cap.release()
W = FR[0].shape[1]
MIRROR = GT600[:N].copy()
MIRROR[..., 0] = (W - 1) - MIRROR[..., 0]
CAMS, GT = [], []
for k, (frames, gt) in enumerate(((FR, GT600[:N]), ([f[:, ::-1].copy() for f in FR], MIRROR), (FR, GT600[:N]))):
    path = os.path.join(OUT, f"cam{k}.mp4")
    write_video(path, frames, fps=30.0)
    CAMS.append(path)
    GT.append(gt)
forget_recovery(*CAMS)
NAMES = ["snout", "tail", "ear", "fin"]
PLAN = {0: {"snout": "alltracker", "tail": "cotracker3", "ear": "alltracker", "fin": "hand"},
        1: {"snout": "cotracker3", "tail": "cotracker3", "ear": "alltracker", "fin": "alltracker"},
        2: {"snout": "cotracker3", "tail": "cotracker3", "ear": "cotracker3", "fin": "cotracker3"}}
LABEL = {"alltracker": "AllTracker", "cotracker3": "CoTracker3", "hand": "Hand only"}


def canvas_click(win, x, y):
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    pump(0.05)


def select_rows(win, *pids):
    win.layers.clearSelection()
    for q in pids:
        win.layers.point_item(q).setSelected(True)
    pump(0.05)


def place(win, v, name, f, xy):
    win._set_active_view(v)
    pump(0.1)
    q = win.session.pid_by_name(name)
    select_rows(win, q)
    win._on_select(q)
    win._goto(f, force=True)
    pump(0.05)
    canvas_click(win, *xy)


def set_tracker_here(win, v, name, key):
    """Camera v: LAYERS right-click -> Tracker -> Only in this camera -> the tracker (else set on the point)."""
    win._set_active_view(v)
    pump(0.1)
    s = win.session
    q = s.pid_by_name(name)
    got = {"hit": False}

    def pick(menu, pos):
        menu.deleteLater()              # as `pop_menu` does
        m = menu
        for want in ("Tracker", "Only in this camera", LABEL[key]):
            hit = next((a for a in m.actions() if a.text().startswith(want)), None)
            if hit is None:
                return None
            m = hit.menu() if hit.menu() is not None else m
        got["hit"] = hit.isEnabled()
        return hit if hit.isEnabled() else None
    select_rows(win, q)
    win._on_select(q)
    orig = appmod.pop_menu
    appmod.pop_menu = pick
    try:
        lw = win.layers
        r = lw.visualItemRect(lw.point_item(q)).center()
        QApplication.sendEvent(lw.viewport(), QContextMenuEvent(QContextMenuEvent.Mouse, r,
                                                                lw.viewport().mapToGlobal(r)))
        pump(0.2)
    finally:
        appmod.pop_menu = orig
    if not got["hit"]:
        s.points[q].tracker = key
    return got["hit"]


def main():
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    win._open_video(CAMS[0])
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    for path in CAMS[1:]:
        win._add_view(path)
        wait(lambda: win.state == READY and not win._loading, 60)
    pump(0.3)
    p = win.project
    check("(setup) three cameras", p is not None and p.n_views == 3)
    win._set_active_view(0)
    pump(0.1)
    win._goto(0, force=True)
    for j, nm in enumerate(NAMES):
        win.btn_add.setChecked(True)
        canvas_click(win, *GT[0][0, j])
        win._apply_rename(win.session.n_points - 1, nm)
    for v in (1, 2):
        for j, nm in enumerate(NAMES):
            place(win, v, nm, 0, GT[v][0, j])
    for f in (20, 40):                               # camera 1's fin: clicks only
        place(win, 0, "fin", f, GT[0][f, 3])
    ok = True
    for v, d in PLAN.items():
        for nm, key in d.items():
            ok &= set_tracker_here(win, v, nm, key)
    check("(setup) each camera's trackers chosen by LAYERS -> Tracker -> Only in this camera", ok)
    fin0 = p.sessions[0].pid_by_name("fin")
    before = {a.name: getattr(p.sessions[0], a.name)[:, fin0].copy() for a in POINT_ARRAYS}
    win._set_active_view(0)
    pump(0.1)
    win.act_track_all.setChecked(True)
    win.act_select_all.trigger()
    win._goto(0, force=True)
    pump(0.05)
    BUILT.clear()
    t0 = time.time()
    QTest.keyClick(win, Qt.Key_T)
    wait(lambda: win.state != READY or win._passes is not None, 30)
    done = wait(lambda: win.state == READY and win.worker is None and win._passes is None and win._multi is None,
                900)
    pump(0.5)
    print(f"  (the press took {time.time() - t0:.0f} s)")
    check("(setup) the every-camera press ended", done)
    vid = {os.path.basename(CAMS[v]): v for v in range(3)}
    have = {v: set() for v in range(3)}
    for b in BUILT:
        v = vid[b["video"]]
        sv = p.sessions[v]
        have[v] |= {(sv.points[q].name, b["backend"]) for q in b["pids"]}
    want = {v: {(nm, key) for nm, key in d.items() if key != "hand"} for v, d in PLAN.items()}
    check("each camera tracked each point with ITS OWN tracker", have == want, have)
    check("the camera holding only CoTracker3 points ran CoTracker3 and no AllTracker",
          {b["backend"] for b in BUILT if vid[b["video"]] == 2} == {"cotracker3"},
          [b["backend"] for b in BUILT if vid[b["video"]] == 2])
    errs = {}
    for v in range(3):
        sv = p.sessions[v]
        for j, nm in enumerate(NAMES):
            if PLAN[v][nm] == "hand":
                continue
            q = sv.pid_by_name(nm)
            ok_rows = sv.tracked[1:N, q]
            d = np.linalg.norm(sv.tracks[1:N, q] - GT[v][1:N, j], axis=1)
            errs[(v, nm)] = (int(ok_rows.sum()), float(np.nanmedian(d)) if ok_rows.any() else np.inf)
    print("  (tracked frames, median px vs truth): " + "; ".join(
        f"cam{v + 1} {nm} {n} {e:.2f}" for (v, nm), (n, e) in errs.items()))
    check("every tracked point in every camera reaches the last frame and follows its landmark (median < 3 px)",
          all(n == N - 1 and e < 3.0 for n, e in errs.values()), errs)
    check("camera 1's Hand-only fin is never tracked: its clicks on 0, 20, 40 are exactly as they were",
          all(np.array_equal(getattr(p.sessions[0], a.name)[:, fin0], before[a.name], equal_nan=True)
              for a in POINT_ARRAYS), np.flatnonzero(p.sessions[0].tracked[:, fin0]))
    win.act_track_all.setChecked(False)
    p.dirty = False
    for sv in p.sessions:
        sv.dirty = False
    win.close()
    pump(0.3)
    win._dev_probe.wait(30000)


main()
print()
if FAILS:
    print(f"FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("  - " + f.encode("ascii", "backslashreplace").decode("ascii"))
    sys.exit(1)
print("verify_tracker_per_camera_models PASSED")
