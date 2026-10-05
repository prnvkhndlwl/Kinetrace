"""(G149) Two segments tracked by the REAL segmentation model in one pass, through the window.

A clip with two moving shapes of different colours (exact ground truth): S + click the first, + New
segment, S + click the second, select both rows, press T -> ONE run segments both on every frame;
each silhouette overlaps its own shape (IoU > 0.8) and not the other; saved and reopened, both are
there. GPU (SAM, cached in models/hf).

.venv\\Scripts\\python.exe tests\\verify_segments_sam.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "segments_sam")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication.instance() or QApplication([])
from kinetrace import projectfile  # noqa: E402
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

FAILS = []
W, H, N = 640, 480, 48


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def shapes(f):
    """(ellipse A centre, rectangle B centre) at frame f."""
    return (140 + 4 * f, 160 + 1.5 * f), (480 - 3 * f, 330)


def truth(f):
    a, b = shapes(f)
    ma = np.zeros((H, W), np.uint8)
    cv2.ellipse(ma, (int(a[0]), int(a[1])), (55, 30), 15, 0, 360, 1, -1)
    mb = np.zeros((H, W), np.uint8)
    cv2.rectangle(mb, (int(b[0]) - 35, int(b[1]) - 35), (int(b[0]) + 35, int(b[1]) + 35), 1, -1)
    return ma.astype(bool), mb.astype(bool)


VIDEO = os.path.join(OUT, "two.mp4")
vw = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
rng = np.random.default_rng(2)
bg = (rng.integers(30, 70, (H, W, 3))).astype(np.uint8)
for f in range(N):
    im = bg.copy()
    ma, mb = truth(f)
    im[ma] = (60, 160, 240)          # orange-ish ellipse (BGR)
    im[mb] = (200, 120, 40)          # blue-ish square
    vw.write(im)
vw.release()

win = MainWindow()
win.resize(1400, 900)
win.show()
app.setActiveWindow(win)
win._open_video(VIDEO)
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and not win._loading:
        break
pump(0.5)
s = win.session


def wait_preview():
    for _ in range(1200):
        pump(0.1)
        if win._preview is None:
            break
    pump(0.3)


a0, b0 = shapes(0)
win._on_animal_click(a0[0], a0[1], True)
wait_preview()
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump(0.2)
win._on_animal_click(b0[0], b0[1], True)
wait_preview()
check(s.n_segments == 2 and s.seg_masks[0].has(0) and s.seg_masks[1].has(0),
      "two segments, each with its silhouette on frame 0 from its click")
win._select_segments([0, 1])
win._update_track_button()
check(win.btn_track.text().startswith("Track") and "2 silhouettes" in win.btn_track.text(), "Track names both segments",
      ascii(win.btn_track.text()))
win.activateWindow()
QTest.keyClick(win, Qt.Key_T)
for _ in range(3000):
    pump(0.1)
    if win.state == READY and win.worker is None:
        break
pump(0.5)
check(win.state == READY, "the run ended")
iou_a, iou_b, cross = [], [], []
for f in range(N):
    ta, tb = truth(f)
    ra = s.seg_masks[0].rasterize(f, H, W) if s.seg_masks[0].has(f) else np.zeros((H, W), bool)
    rb = s.seg_masks[1].rasterize(f, H, W) if s.seg_masks[1].has(f) else np.zeros((H, W), bool)
    iou_a.append((ra & ta).sum() / max((ra | ta).sum(), 1))
    iou_b.append((rb & tb).sum() / max((rb | tb).sum(), 1))
    cross.append(max((ra & tb).sum() / max(tb.sum(), 1), (rb & ta).sum() / max(ta.sum(), 1)))
check(min(iou_a) > 0.8 and min(iou_b) > 0.8,
      f"each silhouette follows its own shape on every frame (IoU min {min(iou_a):.2f} / {min(iou_b):.2f})")
check(max(cross) < 0.05, f"and never the other one (overlap max {max(cross):.3f})")
target = os.path.join(OUT, "two.kinetrace")
from PySide6.QtWidgets import QFileDialog  # noqa: E402
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
win._save_project()
pump(0.5)
rp, _st, _m = projectfile.read(target)
check(rp.sessions[0].n_segments == 2 and all(rp.sessions[0].seg_masks[k].n_masked() == N for k in (0, 1)),
      "saved and read back: both segments on every frame")
win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_segments_sam: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
