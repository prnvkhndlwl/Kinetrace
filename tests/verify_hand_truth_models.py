"""Hand-placed frames are the truth, with the point MODELS (PR A of dev/work/2026-10-20-silhouette-hand-mix): the
GPU half of verify_hand_truth. Offscreen; CoTracker3 and AllTracker are used (the models in models/).
  [1] a CoTracker3 run over a later click keeps the click exactly (still hand-placed) and writes every other
      frame; until the re-seed sessions (PR B) the model point does not restart from it, and the run says so;
  [2] a lost model point with a click ahead still stops the run where it was lost (PR A: cut-and-resume is
      Moving spot only), and the click ahead is kept;
  [3] Fresh run (the Track pop-up, G217) on a two-pass press (AllTracker points, then CoTracker3 points whose
      dot vanishes): the passes are trimmed to the earliest end FIRST, then everything after it is erased -
      nothing old comes back, and one Ctrl+Z restores the old data and the clicks.
[1] and [3] FAIL on the code before the change (the run writes over the click; there is no pop-up); [2] guards
today's stop. Run: .venv\\Scripts\\python.exe tests\\verify_hand_truth_models.py
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
OUT = os.path.join(HERE, "out", "hand_truth_models")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402

import kinetrace.app as appmod  # noqa: E402
from kinetrace import alltracker_backend  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.session import POINT_ARRAYS  # noqa: E402

FAILS: list = []
POP = {"seen": [], "pick": "keep"}
if hasattr(appmod, "TrackChoiceDialog"):
    def _pop_exec(self):
        POP["seen"].append(self.message())
        self.pick(POP["pick"])
        return QDialog.Accepted
    appmod.TrackChoiceDialog.exec = _pop_exec


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


def new_window(video):
    forget_recovery(video)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(video)
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    pump(0.3)
    return win


def close(win):
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


def press_t(win, start=0, timeout=600):
    win.act_select_all.trigger()
    win._goto(start, force=True)
    pump(0.05)
    QTest.keyClick(win, Qt.Key_T)
    wait(lambda: win.state != READY or win._passes is not None, 15)
    return wait(lambda: win.state == READY and win.worker is None and win._passes is None and win._multi is None,
                timeout)


V600 = os.path.join(ROOT, "test600.mp4")
GT = np.load(V600 + ".gt.npz")["gt"]


# ---- [1] a CoTracker3 run keeps a later click
def section1():
    print("[1] CoTracker3 over a later click")
    win = new_window(V600)
    s = win.session
    for k in range(4):
        win._on_add(*[float(v) for v in GT[0, k]])
        s.points[k].tracker = "cotracker3"
    click = (float(GT[200, 0, 0]) + 6.0, float(GT[200, 0, 1]))
    s.set_position(200, 0, *click)
    win._refresh_point_list()
    ok = press_t(win)
    check("(setup) the run ended", ok)
    check("[1] the click at 200 is kept exactly and still hand-placed",
          lambda: s.manual[200, 0] and s.tracked[200, 0] and tuple(float(v) for v in s.tracks[200, 0]) == click,
          tuple(s.tracks[200, 0]))
    check("[1] ... and the run wrote the frames round it", lambda: s.tracked[150:250, 0].sum() == 100
          and not s.manual[201:250, 0].any())
    close(win)


# ---- the sabotage clip: dot A vanishes (and the background changes) at 150
W, H, T1, SAB = 640, 480, 300, 150


def make_bg(seed):
    r = np.random.default_rng(seed)
    bg = r.integers(30, 90, size=(H, W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(bg, (0, 0), 1.5)


def dot(frame, x, y, color):
    cv2.circle(frame, (round(x), round(y)), 9, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (round(x), round(y)), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)


SABO = os.path.join(OUT, "sabotage.mp4")
AX = 200 + 60 * np.sin(np.arange(T1) / 40)
AY = 200 + 40 * np.cos(np.arange(T1) / 55)
BX = 460 + 50 * np.sin(np.arange(T1) / 47)
BY = 320 + 45 * np.cos(np.arange(T1) / 38)
bg_a, bg_b = make_bg(3), make_bg(77)
vw = cv2.VideoWriter(SABO, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in range(T1):
    fr = (bg_a if f < SAB else bg_b).copy()
    if f < SAB:
        dot(fr, AX[f], AY[f], (60, 60, 230))
    dot(fr, BX[f], BY[f], (60, 230, 60))
    vw.write(fr)
vw.release()


def section2():
    print("[2] a lost model point with a click ahead still stops (PR A)")
    win = new_window(SABO)
    s = win.session
    win._on_add(float(AX[0]), float(AY[0]))
    win._on_add(float(BX[0]), float(BY[0]))
    for k in range(2):
        s.points[k].tracker = "cotracker3"
    s.set_position(250, 0, 300.0, 200.0)
    win._refresh_point_list()
    press_t(win)
    check("[2] the run stops where dot A was lost, as today", lambda: s.tracked[:140, 1].all()
          and not s.tracked[230:, 1].any(), np.flatnonzero(s.tracked[:, 1])[-1:])
    check("[2] ... and the click ahead is kept", lambda: s.manual[250, 0] and tuple(s.tracks[250, 0]) == (300.0, 200.0))
    close(win)


def section3():
    print("[3] Fresh run on a two-pass press")
    if not alltracker_backend.available():
        check("(setup) AllTracker is installed (needed for two passes)", False)
        return
    win = new_window(SABO)
    s = win.session
    win._on_add(float(BX[0]), float(BY[0]))          # B on AllTracker: pass 1 runs to the end
    win._on_add(float(AX[0]), float(AY[0]))          # A on CoTracker3: pass 2 stops near 150
    s.points[0].tracker = "alltracker"
    s.points[1].tracker = "cotracker3"
    for f in range(160, T1):                         # old data ahead, and a click at 280
        for k in range(2):
            s.tracks[f, k] = (100.0, 100.0)
            s.tracked[f, k] = s.visibility[f, k] = True
            s.confidence[f, k] = 0.8
    s.set_position(280, 0, 120.0, 120.0)
    win._refresh_point_list()
    pre = s.snapshot()
    POP["pick"] = "fresh"
    n_seen = len(POP["seen"])
    press_t(win, timeout=900)
    s = win.session
    check("[3] the press asked first (the Track pop-up)", len(POP["seen"]) > n_seen)
    # the run's own last frame for dot A: its old data ahead is the sentinel (100, 100)
    ran_a = s.tracked[:, 1] & ~np.all(s.tracks[:, 1] == 100.0, axis=1)
    last_a = int(np.flatnonzero(ran_a).max()) if ran_a.any() else -1
    check("(setup) pass 2 (dot A) stopped near the loss", 140 <= last_a <= 220, last_a)
    # the passes end on one frame (the trim): the last frame either point has run data on; the lost dot A was cut
    # at its loss, as today. Fresh run: nothing old (the (100, 100) data, the click at 280) is left after it
    old = s.tracked[:, :2] & np.all(s.tracks[:, :2] == 100.0, axis=2)
    end = int(max(np.flatnonzero(s.tracked[:, k] & ~old[:, k]).max() for k in range(2)))
    check("[3] Fresh run: after the trim and the erase nothing old is left (the old data, the click at 280)",
          lambda: not old.any() and not s.manual[280, 0], (int(old.sum()), end))
    check("[3] ... and no point has data after the frame the passes ended on", lambda: not s.tracked[end + 1:, :2].any()
          and end <= 230, [np.flatnonzero(s.tracked[:, k])[-1:] for k in range(2)])
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)
    s = win.session
    check("[3] one Ctrl+Z restores the old data and the click",
          lambda: all(np.array_equal(getattr(s, a.name), getattr(pre, a.name), equal_nan=True) for a in POINT_ARRAYS))
    close(win)


for fn in (section1, section2, section3):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_hand_truth_models: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
