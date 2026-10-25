"""ChArUco calibration through the app, by the user's own gestures (offscreen, no GPU; G220, G221).

[1] the lens wizard: 3D -> Calibrate -> Calibrate a Lens (checkerboard or ChArUco)... ; the intro's
    "Save a ChArUco board to print..." writes the board; on the video page the ChArUco radio shows the
    board's boxes (and hides the checkerboard's), a board that cannot exist is refused in words, Find
    the boards scans a rendered ChArUco video, a dictionary changed afterwards discards the scan, the
    review board says a ChArUco corner cannot be dragged, Finish attaches a profile matching the truth
    that names the board.

Run: .venv\\Scripts\\python.exe tests\\verify_charuco_gui.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWizard  # noqa: E402

import _synth_charuco as sc  # noqa: E402
from _clean import forget_recovery  # noqa: E402
from kinetrace import charuco  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

SCRATCH = os.path.join(HERE, "out")
os.makedirs(SCRATCH, exist_ok=True)
FAILS = []


def check(ok, what, detail=None):
    print(("ok    " if ok else "FAIL  ") + what + ("" if ok or detail is None else f"  -- {detail}"))
    if not ok:
        FAILS.append(what)


W, H = 960, 720
K1 = np.array([[820.0, 0, W / 2 + 6], [0, 815.0, H / 2 - 4], [0, 0, 1]])
D1 = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
SPEC = charuco.BoardSpec()

# the board video of camera 1, and a short video for the camera itself
board_vid = os.path.join(SCRATCH, "charuco_gui_board.mp4")
if not os.path.exists(board_vid):
    r = np.random.RandomState(31)
    pics = []
    for k in range(60):
        rv, tv = sc.random_pose(K1, (W, H), r, spread=1.1, z=(0.28, 0.6))
        pics.append(sc.render(SPEC, K1, D1, False, rv, tv, (W, H), seed=3100 + k))
    sc.write_video(board_vid, pics)
cam_vid = os.path.join(SCRATCH, "charuco_gui_cam1.mp4")
sc.write_video(cam_vid, [np.full((H, W, 3), 40, np.uint8)] * 12)
proj = Project([TrackingSession(cam_vid, 12, 30.0, W, H)], ["cam1"], [0.0])
PROJ = os.path.join(SCRATCH, "charuco_gui.kinetrace")
proj.save(PROJ)
forget_recovery(cam_vid)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))

from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace import lenswizard as lw  # noqa: E402
from kinetrace.boardreview import CHARUCO_NO_EDIT  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def click(w):
    QTest.mouseClick(w, Qt.LeftButton)
    app.processEvents()


def type_into(spin, text):
    """Select the box's text and type `text`, as a user does."""
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key_Tab)
    app.processEvents()


def wait_scan(page, limit=300):
    t0 = time.time()
    while not page.btn_run.isEnabled():
        pump(0.1)
        if time.time() - t0 > limit:
            raise TimeoutError("the board scan did not finish")


win._open_project_from_path(PROJ)
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None:
        break
p = win.project
check(win.state == READY and p is not None and p.n_views == 1, "a one-camera project opened")

# ------------------------------------------------------------------ [1] the lens wizard
print("[1] the lens wizard")
check("ChArUco" in win.act_lens.text(), "the menu entry names both boards", win.act_lens.text())
printed = os.path.join(SCRATCH, "charuco_gui_print.png")
if os.path.exists(printed):
    os.remove(printed)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (printed, ""))
SEEN = {}


def drive(self):
    self.show()
    self.restart()
    pump(0.1)
    click(self.page_intro.btn_charuco)
    SEEN["printed"] = os.path.exists(printed) and "ChArUco" in self.page_intro.note.text()
    click(self.button(QWizard.NextButton))
    pump(0.1)
    vp = self.page_video
    SEEN["checker_first"] = vp.r_checker.isChecked() and not vp.charuco.isVisibleTo(vp) and vp.cols.isVisibleTo(vp)
    click(vp.r_charuco)
    SEEN["charuco_shown"] = vp.charuco.isVisibleTo(vp) and not vp.cols.isVisibleTo(vp) and not vp.square.isVisibleTo(vp)
    # a marker bigger than its square: refused in words, nothing scanned
    type_into(vp.charuco.marker, "40")
    SEEN["bad_note"] = vp.charuco.note.text()
    vp.path.setText(board_vid)
    click(vp.btn_run)
    SEEN["bad_refused"] = "cannot exist" in vp.status.text() and self.scan is None
    type_into(vp.charuco.marker, "22")
    click(vp.btn_run)
    wait_scan(vp)
    SEEN["scan1"] = (self.scan is not None and self.scan.board == SPEC, vp.status.text())
    # a dictionary changed after the scan: those boards are not this board's
    vp.charuco.dictionary.setFocus()
    QTest.keyClick(vp.charuco.dictionary, Qt.Key_Down)
    app.processEvents()
    SEEN["discarded"] = self.scan is None and not vp.isComplete() and "changed" in vp.status.text()
    QTest.keyClick(vp.charuco.dictionary, Qt.Key_Up)
    app.processEvents()
    click(vp.btn_run)
    wait_scan(vp)
    SEEN["scan2"] = self.scan is not None and len(self.scan.corners) >= 30 and vp.isComplete()
    click(self.button(QWizard.NextButton))
    pump(0.3)
    rv = self.page_review.review
    SEEN["review"] = (self.currentPage() is self.page_review and len(rv._tiles) == len(self.scan.corners)
                      and self.page_review.isComplete())
    SEEN["tile_tip"] = rv._tiles[0].pic.toolTip() == CHARUCO_NO_EDIT
    QTest.mouseClick(rv._tiles[0].pic, Qt.LeftButton)
    app.processEvents()
    SEEN["no_edit_said"] = "nothing to drag" in rv.summary.text() and rv._reader is None
    SEEN["orient"] = rv.orient_label.text()
    click(self.button(QWizard.NextButton))
    pump(0.2)
    SEEN["result"] = self.currentPage() is self.page_result and "Attach to cam1" in self.buttonText(
        QWizard.FinishButton)
    click(self.button(QWizard.FinishButton))
    pump(0.1)
    return self.result()


lw.LensWizard.exec = drive
win.act_lens.trigger()
pump(0.3)
check(SEEN.get("printed"), "the intro page saves the ChArUco board to print")
check(SEEN.get("checker_first"), "the video page starts on the checkerboard, its ChArUco boxes hidden")
check(SEEN.get("charuco_shown"), "the ChArUco radio shows the board's boxes and hides the checkerboard's")
check("smaller" in SEEN.get("bad_note", ""), "a marker bigger than its square is said at once", SEEN.get("bad_note"))
check(SEEN.get("bad_refused"), "Find the boards refuses a board that cannot exist, in words")
check(bool(SEEN.get("scan1", (False,))[0]), "Find the boards scans for the typed ChArUco board", SEEN.get("scan1"))
check(SEEN.get("discarded"), "a dictionary changed after the scan discards the boards")
check(SEEN.get("scan2"), "the board scanned again: Next is available")
check(SEEN.get("review"), "the review board shows every board, fitted")
check(SEEN.get("tile_tip"), "a ChArUco tile says its corners cannot be dragged")
check(SEEN.get("no_edit_said"), "clicking a ChArUco tile says so, opens no editor")
check("markers" in SEEN.get("orient", ""), "the review board says corners are named by markers", SEEN.get("orient"))
check(SEEN.get("result"), "the result page attaches to cam1")
prof = p.lenses[0]
check(prof is not None, "Finish attached the profile to cam1")
if prof is not None:
    print(f"      f {prof.K[0, 0]:.1f}/{prof.K[1, 1]:.1f} (true 820/815), centre ({prof.K[0, 2]:.1f}, {prof.K[1, 2]:.1f}), "
          f"rms {prof.rms:.3f} px, verdict {prof.report['verdict']}")
    check(abs(prof.K[0, 0] - 820) / 820 < 0.005 and abs(prof.K[1, 1] - 815) / 815 < 0.005,
          "the attached lens: focal length within 0.5 %", prof.K)
    check(prof.board.get("kind") == "charuco" and prof.board.get("charuco") == SPEC.to_json(),
          "the attached profile names its ChArUco board")


def finish():
    p.dirty = False
    win.close()
    pump(0.3)
    win._dev_probe.wait(10000)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
        sys.exit(1)
    print("PASSED")


finish()
