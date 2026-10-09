"""(G177) The cameras' properties and lenses at a glance: a lens badge on every CAMERAS row and
3D -> Cameras Overview... (which took over the GoPro-only table, G145).

Pure: `camera_facts` judges each lens by the attach rule (used / attached but not used / none) and
says whether the calibration covers the camera. Through the window (offscreen, three cameras, no GPU,
real clicks): the badges read "lens ✓" / "no lens" / "lens ✗" with a tooltip that says what the
camera records (a video that plays turned included), its lens and its calibration; attaching a lens
through 3D -> Load a Lens Profile for Cameras... updates the badge at once; a click on a badge opens the
overview, whose table and notes say the same, and whose "Load a lens profile for cameras..." button
attaches a lens and refreshes the table.

.venv\\Scripts\\python.exe tests\\verify_camera_overview.py
"""
import os
import shutil
import subprocess
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "camera_overview")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

ASK = {"open": ""}
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
app = QApplication.instance() or QApplication([])
import _lens_attach  # noqa: E402

_lens_attach.install()
from kinetrace import cameraoverview, lens  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)      # the console is cp1252
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def video(path, w, h, seed):
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for _ in range(10):
        vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    vw.release()
    return path


# ------------------------------------------------------------------ 1. the facts
print("[1] camera_facts")
K = np.array([[300.0, 0, 163.5], [0, 302.0, 117.0], [0, 0, 1]])
lensA = lens.LensProfile(320, 240, K, np.array([-0.2, 0.04, 0.001, -0.002, 0.0]), False, 0.3, 20,
                         "checkerboard (Kinetrace)", {"verdict": "good", "verdict_reasons": ["clean"]}, rotation=0)


class _S:
    def __init__(self, w, h):
        self.width, self.height, self.fps, self.file_fps, self.video_path = w, h, 30.0, 30.0, ""


class _P:
    def __init__(self, sessions, lenses, calibration):
        self.sessions, self.lenses, self.calibration = sessions, lenses, calibration
        self.n_views = len(sessions)

    def name(self, v):
        return f"cam{v + 1}"


facts = cameraoverview.camera_facts(_P([_S(320, 240), _S(640, 480), _S(320, 240)], [lensA, lensA, None],
                                       ["calibrated camera", "calibrated camera"]), [])
check([f.lens_state for f in facts] == ["ok", "unused", "none"], "lens: used / attached but not used / none",
      [f.lens_state for f in facts])
check("640" in facts[1].lens_note and facts[1].lens_note[0].isupper(), "why it is not used, as a sentence",
      facts[1].lens_note)
check([f.calibrated for f in facts] == [True, True, False] and "2 cameras" in facts[2].calibration_text,
      "a calibration of two cameras covers the first two", [f.calibration_text for f in facts])
check(cameraoverview.camera_facts(_P([_S(320, 240)], [None], None), [])[0].calibration_text == "no calibration yet",
      "no calibration: said")

# ------------------------------------------------------------------ 2. through the window
print("\n[2] CAMERAS badges and 3D -> Cameras Overview")
A = video(os.path.join(OUT, "A.mp4"), 320, 240, 1)
SIDE = os.path.join(OUT, "side.mp4")
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-display_rotation", "90",
                "-i", video(os.path.join(OUT, "side_level.mp4"), 320, 240, 2), "-c", "copy", SIDE], check=True)
B = video(os.path.join(OUT, "B.mp4"), 640, 480, 3)
w = MainWindow()
w.resize(1400, 900)
w.show()
w._open_video(A)
for _ in range(300):
    pump(0.05)
    if w.state == READY and w.project is not None and not w._loading:
        break
pump(0.3)
check(w._add_view(SIDE) and w._add_view(B), "three cameras: A, one filmed on its side, B (another size)")
pump(0.3)
p = w.project
w._set_lenses({0: lensA, 2: lensA})
pump(0.1)
rows = w.cameras._rows
texts = [r.lens_badge.text() for r in rows]
check(texts == ["lens ✓", "no lens", "lens ✗"] and all(r.lens_badge.isVisible() for r in rows),
      "the badges: lens used / none / attached but not used", texts)
tip1, tip2 = rows[1].lens_badge.toolTip(), rows[2].lens_badge.toolTip()
check("Records 240 x 320" in tip1 and "turned 90° counter-clockwise" in tip1 and "No lens profile" in tip1
      and "no calibration yet" in tip1, "the tooltip: what it records, its turn, its lens, its calibration", tip1)
check("NOT USED" in tip2 and "640" in tip2, "an unused lens's tooltip says why", tip2)

# attach to the side camera through the menu (the camera list): its badge changes at once
pathA = lensA.save(os.path.join(OUT, "A.klens.json"))
w._set_active_view(1)
pump(0.3)
ASK["open"] = pathA
w.act_load_lens.trigger()
pump(0.1)
check(rows[1].lens_badge.text() == "lens ✓" and (p.lenses[1].width, p.lenses[1].height) == (240, 320),
      "attaching a lens turns the side camera's badge to 'lens ✓' at once", rows[1].lens_badge.text())

# a click on a badge opens the overview
SEEN = {}


def overview_exec(dlg):
    SEEN["cols"] = [dlg.table.horizontalHeaderItem(c).text() for c in range(dlg.table.columnCount())]
    SEEN["records"] = [dlg.table.item(r, 1).text() for r in range(dlg.table.rowCount())]
    SEEN["lens"] = [dlg.table.item(r, 2).text() for r in range(dlg.table.rowCount())]
    SEEN["calib"] = [dlg.table.item(r, 3).text() for r in range(dlg.table.rowCount())]
    SEEN["notes"] = dlg.notes.text()
    p.lenses[0] = None                       # so the overview's Load button has something to do
    dlg._refresh()
    ASK["open"] = pathA                       # the working camera (cam 0) starts ticked in the camera list
    QTest.mouseClick(dlg.btn_load, Qt.LeftButton)
    pump(0.1)
    SEEN["lens_after"] = [dlg.table.item(r, 2).text() for r in range(dlg.table.rowCount())]
    dlg.reject()
    return dlg.result()


cameraoverview.CamerasOverview.exec = overview_exec
w._set_active_view(0)
pump(0.3)
QTest.mouseClick(rows[2].lens_badge, Qt.LeftButton)
pump(0.3)
check(SEEN.get("cols") == list(cameraoverview.BASE_COLUMNS), "a badge click opened Cameras Overview (no GoPro columns)",
      SEEN.get("cols"))
rec = SEEN.get("records", [""] * 3)
check(rec[0].startswith("320 x 240") and "240 x 320" in rec[1] and "turned" in rec[1] and rec[2].startswith("640 x 480"),
      "Records: sizes, fps, the turned video", rec)
lz = SEEN.get("lens", [""] * 3)
check("NOT USED" in lz[2] and "NOT USED" not in lz[0] and "NOT USED" not in lz[1], "Lens profile: the unused one is marked",
      lz)
check(all(c == "no calibration yet" for c in SEEN.get("calib", [])), "3D calibration: none yet", SEEN.get("calib"))
notes = SEEN.get("notes", "")
check("NOT used" in notes and p.name(2) in notes.split("NOT used")[1], "the notes name the camera whose lens is not used",
      notes)
check(SEEN.get("lens_after", ["none"])[0] != "none" and p.lenses[0] is not None,
      "the overview's Load button attached the lens and the table shows it", SEEN.get("lens_after"))
check(w.act_cameras.isEnabled() and w.act_cameras.text().startswith("Cameras &Overview"),
      "3D -> Cameras Overview… is there for any footage")

w.project.dirty = False
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_camera_overview: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
