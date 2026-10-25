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


# ------------------------------------------------------------------ [2] the camera wizard
print("[2] the camera wizard: three synced cameras film the board")
from kinetrace import charucowizard as cwz  # noqa: E402

RIG_N = 150
paths, cams = sc.rig_videos(SCRATCH, RIG_N)
RW, RH = sc.RIG_SIZE
rig = Project([TrackingSession(pth, RIG_N, 30.0, RW, RH) for pth in paths], ["cam1", "cam2", "cam3"],
              list(sc.RIG_OFFSETS))
PROJ2 = os.path.join(SCRATCH, "charuco_gui_rig.kinetrace")
rig.save(PROJ2)
for pth in paths:
    forget_recovery(pth)
p.dirty = False
win._open_project_from_path(PROJ2)
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == 3:
        break
p = win.project
check(p is not None and p.n_views == 3 and win.state == READY and all(x is None for x in p.lenses)
      and p.calibration is None, "the three-camera board project opened, no lens, no calibration")
check(any(a is win.act_charuco for a in win.findChildren(type(win.act_charuco)))
      and "ChArUco" in win.act_charuco.text() and win.act_charuco.isEnabled(), "3D > Calibrate has the ChArUco entry")
RIG = {"choice": "both", "cancel": False}


def wait_thread(page, limit=600):
    t0 = time.time()
    pump(0.1)
    while page._thread is not None and page._thread.isRunning():
        pump(0.1)
        if time.time() - t0 > limit:
            raise TimeoutError("the worker did not finish")
    pump(0.2)


def drive_rig(self):
    self.show()
    self.restart()
    pump(0.1)
    bp = self.page_board
    RIG["ticks"] = [cb.isChecked() for cb in bp.ticks]
    RIG["window"] = (bp.first.value(), bp.last.value())
    click(bp.r_choice[RIG["choice"]])
    # the rig's board: 9 x 7 squares of 60 mm, markers 45 mm
    type_into(bp.fields.sx, "9")
    type_into(bp.fields.sy, "7")
    type_into(bp.fields.square, "60")
    type_into(bp.fields.marker, "45")
    type_into(bp.count, "100")
    RIG["next_before"] = self.button(QWizard.NextButton).isEnabled()
    click(bp.btn_scan)
    wait_thread(bp)
    RIG["scan_status"] = bp.status.text()
    RIG["scan_ok"] = self.scan is not None and bp.isComplete()
    click(self.button(QWizard.NextButton))
    pump(0.2)
    rp = self.page_run
    RIG["finish_text"] = self.buttonText(QWizard.FinishButton)
    click(rp.btn_run)
    wait_thread(rp)
    RIG["report"] = rp.report.toPlainText()
    RIG["complete"] = rp.isComplete()
    if RIG.get("argus_dir"):
        click(rp.btn_argus)
        RIG["argus_status"] = rp.status.text()
    if RIG["cancel"]:
        click(self.button(QWizard.CancelButton))
    else:
        click(self.button(QWizard.FinishButton))
    pump(0.1)
    return QDialog.result(self)          # the wizard's own `result` is its WandResult (as the wand wizard's)


cwz.CharucoWizard.exec = drive_rig
win.act_charuco.trigger()
pump(0.5)
check(RIG.get("ticks") == [True, True, True] and RIG.get("window", (0, 0))[1] > 100,
      "every camera ticked, the stretch they share filled in", (RIG.get("ticks"), RIG.get("window")))
check(not RIG.get("next_before"), "Next waits for the board to be found")
check(RIG.get("scan_ok") and "good enough for its lens" in RIG.get("scan_status", ""),
      "Find the board: per camera what it saw, and Next", RIG.get("scan_status"))
check(RIG.get("finish_text") == "Use these lenses and this calibration", "Finish says what it will do",
      RIG.get("finish_text"))
check(RIG.get("complete") and "GOOD" in RIG.get("report", "") and "420 mm" in RIG.get("report", ""),
      "Calibrate now: the report gives verdicts and the board's edge length", RIG.get("report", "")[:300])
check(all(x is not None and x.board.get("kind") == "charuco" for x in p.lenses),
      "Finish attached a ChArUco lens to every camera")
check(p.calibration is not None and len(p.calibration.views()) == 3, "Finish made the three-camera calibration")
cal_made, lenses_made = p.calibration, list(p.lenses)
if p.calibration is not None:
    import itertools  # noqa: E402
    C = [p.calibration.camera(v).center() for v in range(3)]
    sp = np.array([np.linalg.norm(C[i] - C[j]) for i, j in itertools.combinations(range(3), 2)])
    true = np.array([np.linalg.norm(cams[i][4] - cams[j][4]) for i, j in itertools.combinations(range(3), 2)])
    check(np.all(np.abs(sp / true - 1) < 0.01), "the project's calibration: camera spacing within 1 %", sp / true)
# offscreen Qt leaves NO focus window after a modal wizard closes (a desktop gives it back to the main
# window), and a shortcut fires only in the focus window: the window is activated as the desktop would
win.windowHandle().requestActivate()
pump(0.2)
QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
pump(0.3)
check(all(x is None for x in p.lenses) and p.calibration is None,
      "one Ctrl+Z takes back the lenses and the calibration together", win.statusBar().currentMessage())
check("Restored the lens profiles and the calibration" in win.statusBar().currentMessage()
      or "Restored the lens profiles and the calibration" in str(getattr(win.toast, "text", lambda: "")()),
      "Ctrl+Z says what it put back", win.statusBar().currentMessage())

# (review) a Cancel stops "Calibrate now" at its next step: the worker ends without a result
done_ = []
th_ = cwz._RunThread(p, cwz.charuco.scan_rig(p, sc.RIG_SPEC, n=30), "both", {})
th_.finished_ok.connect(lambda out: done_.append(out))
th_.cancel()
t0_ = time.time()
th_.start()
th_.wait(30000)
pump(0.1)
check(th_.isFinished() and not done_ and time.time() - t0_ < 5,
      "a cancelled Calibrate now ends at once, with no result", (th_.isFinished(), len(done_), time.time() - t0_))

RIG.update(choice="lenses")
win.act_charuco.trigger()
pump(0.5)
check(RIG.get("finish_text") == "Use these lenses", "lenses only: Finish says so", RIG.get("finish_text"))
check(all(x is not None for x in p.lenses) and p.calibration is None,
      "lenses only: lenses attached, the calibration left as it was")

# (I323) File -> Export -> Lens Profile... -> every camera: one Argus camera_profile.txt, lenses alone
from PySide6.QtWidgets import QInputDialog  # noqa: E402
from kinetrace import lens as _lens  # noqa: E402

argus_prof = os.path.join(SCRATCH, "charuco_gui_camera_profile.txt")
if os.path.exists(argus_prof):
    os.remove(argus_prof)
ITEMS = []
QInputDialog.getItem = staticmethod(lambda *a, **k: (ITEMS.append(list(a[3])), (a[3][-1], True))[1])
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (argus_prof, ""))
win.act_export_lens.trigger()
pump(0.2)
check(ITEMS and ITEMS[-1][-1].startswith("Every camera with a lens (3)"),
      "Export > Lens Profile offers every camera as one Argus file", ITEMS[-1:] or None)
back = _lens.load_argus_profile(argus_prof) if os.path.exists(argus_prof) else []
check(len(back) == 3 and all(np.allclose(back[k].K, p.lenses[k].K, atol=2e-6) for k in range(3)),
      "the Argus file holds the three lenses and reads back to them")

# "Save for Argus..." in the wizard, then Cancel: the files are written, the project is not changed
argus_dir = os.path.join(SCRATCH, "charuco_gui_argus")
os.makedirs(argus_dir, exist_ok=True)
for fn in os.listdir(argus_dir):
    os.remove(os.path.join(argus_dir, fn))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: argus_dir)
before_l, before_c = list(p.lenses), p.calibration
RIG.update(choice="both", cancel=True, argus_dir=argus_dir)
win.act_charuco.trigger()
pump(0.5)
check(sorted(os.listdir(argus_dir)) == ["camera_profile.txt", "paired_pts-xypts.csv", "unpaired_pts-xypts.csv"]
      and "1 = cam1" in RIG.get("argus_status", ""),
      "Save for Argus writes Argus's three files and says which camera is which", RIG.get("argus_status"))
check(all(a is b for a, b in zip(before_l, p.lenses)) and p.calibration is before_c,
      "Cancel after a run changes nothing")


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
