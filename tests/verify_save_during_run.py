"""(I265) Save pressed while a run is going (owner, 2026-10-07: "I tracked points P1 and hit save but the
project did not save the points ... it got saved but on the second try").

Save (Ctrl+S) was greyed out during a run, so pressing it while Track was still going did nothing at all
and said nothing: the work looked saved and was not. Now it is kept and done the moment the run (all of a
Track press: every pass, camera and re-track stretch) is over, and the app says so.

A real run (Moving spot, CPU) on a synthetic dot clip, a saved project, the user's own keys: T to track,
Ctrl+S while it runs.

.venv\\Scripts\\python.exe tests\\verify_save_during_run.py
"""
import json
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "save_during_run")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])

from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

FAILS = []
W, H, N = 320, 240, 900


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not ok:
        FAILS.append(what)


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


# a small bright dot crossing a textured picture slowly, in view the whole time
rng = np.random.default_rng(7)
bg = cv2.GaussianBlur((rng.random((H, W)) * 90 + 40).astype(np.uint8), (0, 0), 1.2)
path = np.stack([np.linspace(30, 290, N), 120 + 30 * np.sin(np.linspace(0, 6, N))], axis=1)
VID = os.path.join(OUT, "dot.mp4")
vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
for f in range(N):
    im = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    cv2.circle(im, (int(round(path[f, 0])), int(round(path[f, 1]))), 2, (255, 255, 255), -1)
    vw.write(im)
vw.release()

win = MainWindow()
win.resize(1300, 850)
win.show()
win._open_video(VID)
check(wait(lambda: win.state == READY and win.project is not None and not win._loading, 30), "(setup) the video opens")
pump(0.3)
s = win.session
win._goto(0, force=True)
check(wait(lambda: win.cache.get(0) is not None, 10), "(setup) frame 0 shown")
win.canvas.fit()
pump(0.1)
win.btn_add.setChecked(True)
vp = win.canvas.mapFromScene(QPointF(*path[0]))
QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
pump(0.1)
check(s.n_points == 1, "(setup) a point placed on the dot with N + a click")
win.act_pm_spot.trigger()                       # Moving spot: no model, a run on the CPU
win._goto(1, force=True)
pump(0.1)
win._on_annotate(*path[1])                      # its speed: the dot on frame 1 too
win._goto(0, force=True)
pump(0.1)
target = os.path.join(OUT, "run.kinetrace")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
check(win._save_project(), "(setup) the project is saved once (it has a folder)")
saved_at0 = json.load(open(os.path.join(target, "kinetrace.json"), encoding="utf-8"))["saved_at"]
pump(0.2)

print("[1] Ctrl+S while the run is going (I265)")
win.act_select_all.trigger()                    # only what is selected is tracked (G61)
pump(0.1)
QTest.keyClick(win, Qt.Key_T)                   # Track
check(wait(lambda: win.state == TRACKING, 20), "T starts the run")
QTest.keyClick(win, Qt.Key_S, Qt.ControlModifier)   # Save, while it runs
pump(0.05)
still_running = win.state == TRACKING
check(still_running, "(setup) the run was still going when Ctrl+S was pressed")
check(win.act_save.isEnabled(), "Save is not greyed out during a run")
check("saved as soon as it stops" in win.toast.text(), "the app says the save happens when the run stops",
      win.toast.text())
now_at = json.load(open(os.path.join(target, "kinetrace.json"), encoding="utf-8"))["saved_at"]
check(now_at == saved_at0, "nothing is written while the run is still writing its data")
check(wait(lambda: win.state == READY and win.worker is None, 120), "(setup) the run ends")
check(wait(lambda: not win._save_after_run and not win.project.dirty, 20),
      "the moment the run is over the project is saved (nothing left unsaved)", (win._save_after_run, win.project.dirty))
meta = json.load(open(os.path.join(target, "kinetrace.json"), encoding="utf-8"))
csv_path = os.path.join(target, "cameras", "cam1", "tracks", f"{s.points[0].name}.csv")
rows = open(csv_path, encoding="utf-8").read().strip().splitlines()[1:] if os.path.exists(csv_path) else []
n_tracked = int(s.tracked[:, 0].sum())
check(meta["saved_at"] != saved_at0 and n_tracked > 2 and len(rows) == n_tracked,
      "the saved file has every tracked frame of the run", (len(rows), n_tracked))

print("[2] the same Save after the run: saved at once")
s.add_point(5, 50.0, 60.0, name="extra")
pump(0.1)
check(win.project.dirty, "(setup) an edit")
QTest.keyClick(win, Qt.Key_S, Qt.ControlModifier)
pump(0.3)
check(not win.project.dirty and "Project saved" in win.statusBar().currentMessage(),
      "Ctrl+S with no run going saves at once, as before", win.statusBar().currentMessage())

win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_save_during_run: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
