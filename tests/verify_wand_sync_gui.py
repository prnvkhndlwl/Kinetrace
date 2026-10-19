"""Refine offsets with the wand, through the app (I271, I303, I304, G199, G200; offscreen, no GPU).

The synthetic rig of `_synth_wand.py` (four 1280 x 720 cameras, true fractional offsets, the project's
offsets wrong by +0.4 / -0.7 frame, three floor marks clicked on one frame per camera) is opened in the
real MainWindow with a 3D result made with the true cameras. The wand wizard is driven with clicks:

1. Calibrate now: the 3D picture of the wand and the cameras opens by itself (G200); the report carries
   the camera-timing checks (strict JSON); the project's offsets are untouched.
2. Refine offsets with the wand: the button reads Stop refining while it runs; one row per camera.
3. Apply the ticked offsets: the page says what the calibration is MADE AT; the PROJECT is unchanged
   (pending until Finish); the calibration shown is the refine's own; the 3D picture shows it.
4. Calibrate now again at the pending offsets: bit-identical coefficients (made at exactly those offsets).
5. Save calibration files: the .kcal.json is strict JSON.
6. Use this calibration and the new offsets: offsets AND calibration in the project, the CAMERAS panel and
   the toast say old -> new;  7. ONE Ctrl+Z restores both.
8. One row unticked: calibrated again at the offsets as ticked; that camera keeps its offset.
9. Apply, then Cancel: nothing changes (I303);  12. "Use the project's offsets again" drops them.
10. A camera unticked on the wand page: each calibrated camera gets ITS coefficients.
11. The axes frame from marks clicked on one frame survives the fractional offsets (I304).
13. A second Finish without new offsets does not fold into the first Ctrl+Z.
14. (G200) the picture: no click needed; Show in 3D brings it back after closing it.

Run: .venv\\Scripts\\python.exe tests\\verify_wand_sync_gui.py
"""
import json
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QMessageBox, QStyle, QStyleOptionButton,
                               QWizard)

import _synth_wand as sw
from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import Calibration

SCRATCH = os.path.join(ROOT, "tests", "out")
os.makedirs(SCRATCH, exist_ok=True)
N_REF = 700
MARKS_AT = 350
paths = [os.path.join(SCRATCH, f"wandsync_cam{c + 1}.mp4") for c in range(4)]
proj, truth = sw.build(video_paths=paths, n_ref=N_REF, marks_at=MARKS_AT)
img = np.full((sw.H, sw.W, 3), 30, np.uint8)
for pth, s in zip(paths, proj.sessions):
    vw = cv2.VideoWriter(pth, cv2.VideoWriter_fourcc(*"mp4v"), sw.FPS, (sw.W, sw.H))
    for _ in range(s.n_frames):
        vw.write(img)
    vw.release()
PROJ = os.path.join(SCRATCH, "wandsync_gui.kinetrace")
proj.save(PROJ)
for pth in paths:
    forget_recovery(pth)
WRONG = list(proj.offsets)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))
KCAL = os.path.join(SCRATCH, "wandsync_gui.kcal.json")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (KCAL, ""))

from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace import calibwizard as cw  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_for(cond, what, limit=900):
    t0 = time.time()
    while not cond():
        pump(0.1)
        if time.time() - t0 > limit:
            raise TimeoutError(what)


def click(widget):
    assert widget.isEnabled(), f"{widget.text() if hasattr(widget, 'text') else widget} is disabled"
    QTest.mouseClick(widget, Qt.LeftButton)
    pump(0.05)


def click_box(cb):
    """A click on a check box's own indicator (the user's click, not setChecked)."""
    opt = QStyleOptionButton()
    cb.initStyleOption(opt)
    box = cb.style().subElementRect(QStyle.SE_CheckBoxIndicator, opt, cb)
    QTest.mouseClick(cb, Qt.LeftButton, Qt.NoModifier, box.center())
    pump(0.05)


def strict_json(text):
    def bad(name):
        raise ValueError(f"not strict JSON: {name}")
    return json.loads(text, parse_constant=bad)


win._open_project_from_path(PROJ)
wait_for(lambda: win.state == READY and win.project is not None and win.project.n_views == 4, "open")
p = win.project
assert np.allclose(p.offsets, WRONG)
TRUE_CAL = Calibration(truth["cams"], "m", "truth")       # a calibration and a 3D result made before
p.calibration = TRUE_CAL
win._reconstruct_3d(quiet=True)
pump(0.3)
assert p.reconstruction is not None
print("project with a 3D result opened OK")


def to_run_page(wiz, untick=None, axes=False):
    """Intro -> wand (camera `untick` unticked) -> cameras (focal lengths typed, no extra points) ->
    frame (neither, or the three floor marks) -> run, by clicks."""
    wiz.show()
    wiz.restart()
    pump(0.1)
    nxt = wiz.button(QWizard.NextButton)
    click(nxt)
    assert wiz.currentPage() is wiz.page_wand
    assert wiz.page_wand.end_a.currentText() == "wand A" and wiz.page_wand.end_b.currentText() == "wand B"
    ln = wiz.page_wand.length
    ln.lineEdit().selectAll()
    QTest.keyClicks(ln.lineEdit(), f"{sw.WAND_L}")
    QTest.keyClick(ln.lineEdit(), Qt.Key_Tab)
    assert abs(ln.value() - sw.WAND_L) < 1e-9, ln.value()
    if untick is not None:
        click_box(wiz.page_wand.ticks[untick])
        assert not wiz.page_wand.ticks[untick].isChecked()
    click(nxt)
    assert wiz.currentPage() is wiz.page_cams
    click(wiz.page_cams.r_known)
    for sp, f in zip(wiz.page_cams._spins, sw.FOCAL):
        sp.lineEdit().selectAll()
        QTest.keyClicks(sp.lineEdit(), str(int(f)))
        QTest.keyClick(sp.lineEdit(), Qt.Key_Tab)
    ex = wiz.page_cams.extras                     # setup: no extra points (the floor marks are listed there)
    for i in range(ex.count()):
        ex.item(i).setCheckState(Qt.Unchecked)
    click(nxt)
    assert wiz.currentPage() is wiz.page_frame
    fp = wiz.page_frame
    if axes:
        click(fp.r_axes)
        for cb, nm in zip((fp.ax_o, fp.ax_x, fp.ax_y), ("origin", "X", "Y")):
            cb.setCurrentText(nm)
        fp.ax_t.setValue(MARKS_AT)
        pump(0.1)
        assert fp.isComplete(), "the floor marks are clicked at the chosen instant"
    else:
        click(fp.r_none)
    click(nxt)
    assert wiz.currentPage() is wiz.page_run
    return wiz.page_run


def idle(page):
    return page._thread is None or not page._thread.isRunning()


def wait_result(page, timing="keep"):
    """Wait for the run to end. The whole-frame check (I274) may stop it first with a suggestion (this rig's
    cam3 is 0.7 frame off: every camera pair peaks a whole frame away); the user then clicks "Calibrate with
    the offsets as they are" (timing="keep") or "Apply these offsets and calibrate" ("apply")."""
    wait_for(lambda: idle(page) and page.btn_run.isEnabled()
             and (page.wiz.result_calibration is not None or page.timing_box.isVisible()), "calibration")
    pump(0.1)
    if page.wiz.result_calibration is None and page.timing_box.isVisible():
        click(page.btn_keep_offsets if timing == "keep" else page.btn_apply_offsets)
        wait_for(lambda: page.wiz.result_calibration is not None and idle(page) and page.btn_run.isEnabled(),
                 "calibration after the timing box")
        pump(0.1)
    assert page.wiz.result_calibration is not None, page.report.toPlainText()[:400]


def calibrate(page, timing="keep"):
    click(page.btn_run)
    wait_result(page, timing)


def refine(page):
    click(page.btn_refine)
    assert "Stop" in page.btn_refine.text() and not page.btn_run.isEnabled(), "a run while refining"
    wait_for(lambda: page.refined is not None and idle(page) and page.btn_refine.isEnabled(), "refinement")
    pump(0.1)
    assert "Stop" not in page.btn_refine.text()


def finish(wiz):
    click(wiz.button(QWizard.FinishButton))
    pump(0.1)
    assert not wiz.isVisible()
    return QDialog.Accepted


def cancel(wiz):
    wiz.reject()
    pump(0.1)
    return QDialog.Rejected


def undo():
    win.activateWindow()                     # the closed wizard left no active window (offscreen)
    pump(0.1)
    QTest.keyClick(win.canvas.viewport(), Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)


def coefs(cal):
    return [None if cal.camera(c) is None else np.asarray(cal.camera(c).coefs) for c in range(len(cal.cameras))]


state = {}


def drive_all(wiz):
    page = to_run_page(wiz)
    assert not page.btn_refine.isEnabled(), "Refine before any calibration"
    calibrate(page)
    # (G200) the picture opened by itself, with the wand and the cameras of THIS run
    v = page.view3d
    assert v is not None and v.isVisible(), "the 3D picture must open after the solve"
    # (G209) Show in 3D asks for the fields of view by name: every camera's is picked
    assert v.chk_calib.isChecked() and v.chk_pyr.isChecked() and v._fov_views() == {int(a.data()) for a in v.fov_actions}
    assert len(v.scene.calib.end_frame) == 2 * len(page.wiz.result.wand_xyz)
    v.close()
    pump(0.1)
    click(page.btn_3d)                                       # Show in 3D brings it back
    assert page.view3d is not None and page.view3d.isVisible()
    rep = page.wiz.result.report
    json.dumps(rep, allow_nan=False)
    tm = rep["timing"]
    html = page.report.toHtml()
    print(f"timing checks: {tm['verdict']}; speed ratio {tm['speed']['ratio']:.2f}; off by {tm['off_by']}; "
          f"prediction {({c: r['verdict'] for c, r in tm['prediction']['per_camera'].items()})}")
    assert tm["verdict"] in ("ok", "poor"), tm["verdict"]
    assert tm["off_by"][1] < -0.2 and tm["off_by"][2] > 0.4, tm["off_by"]    # told too late / too early
    # (the whole-frame "Timing check" lines are absent here: the user chose "Calibrate with the offsets as
    # they are" in its box, which skips that check - verify_wand_subset H pins its heading)
    for words in ("Camera timing (checked with the wand)", "Check it by hand",
                  "Epipolar guides from the other cameras", "one continuous", "UNDER-estimate", "made at"):
        assert words in html, words
    assert p.offsets == WRONG, "a check changed the offsets"
    # Stop refining: the page says it stopped (review: it kept "Stopping the refinement…"), nothing changed
    shown = page.wiz.result
    click(page.btn_refine)
    pump(0.3)
    click(page.btn_refine)
    wait_for(lambda: idle(page) and page.btn_refine.isEnabled(), "the stopped refinement")
    pump(0.2)
    assert "stopped" in page.status.text() and "Stopping" not in page.status.text(), page.status.text()
    assert page.refined is None and page.wiz.result is shown and p.offsets == WRONG
    refine(page)
    out = page.refined
    rows = page.refine_table.rowCount()
    assert rows == 3 and page.refine_box.isVisible(), rows
    err = np.abs(np.array(out["after"]) - truth["offsets"])
    print(f"refined: {out['verdict']}, left {np.round(err, 3)} frame of the injected 0.4 / 0.7")
    assert out["verdict"] == "good" and err.max() < 0.05, (out["why"], err)
    for r in range(rows):
        assert page.refine_table.item(r, 1).text() == f"cam{r + 2}", page.refine_table.item(r, 1).text()
        assert page.refine_table.item(r, 6).text() == "GOOD"
    ticks = {c: t.isChecked() for c, t in page._ticks}
    assert all(ticks.values()), ticks                        # every good row is ticked
    assert p.offsets == WRONG and p.reconstruction is not None, "the suggestion changed the project"
    click(page.btn_apply_refined)
    pump(0.2)
    # pending: the PROJECT is unchanged until Finish; the page says what the calibration is made at
    assert wiz.result_offsets == out["after"], (wiz.result_offsets, out["after"])
    assert p.offsets == WRONG and p.reconstruction is not None, "Apply must not touch the project"
    assert page.wiz.result is out["res_after"] and idle(page)
    note = page.offsets_note.text()
    assert "made at" in note and "cam2" in note and "→" in note, note
    assert "new offsets" in wiz.button(QWizard.FinishButton).text()
    v = page.view3d
    assert v is not None and v.isVisible() and len(v.scene.calib.end_frame) == 2 * len(out["res_after"].wand_xyz)
    adopted = coefs(page.wiz.result_calibration)
    # Calibrate now again at the pending offsets: exactly the same calibration
    calibrate(page)
    again = coefs(page.wiz.result_calibration)
    assert all(np.array_equal(a, b) for a, b in zip(adopted, again)), "not the calibration made at these offsets"
    assert p.offsets == WRONG
    click(page.btn_save)
    with open(KCAL, encoding="utf-8") as fh:
        saved = strict_json(fh.read())
    assert "timing" in json.dumps(saved), "the saved report carries the timing checks"
    state["after"] = list(out["after"])
    state["cal"] = again
    return finish(wiz)


cw.WandWizard.exec = drive_all
win._wand_wizard()
pump(0.3)
assert p.offsets == state["after"], (p.offsets, state["after"])
assert all(np.array_equal(a, b) for a, b in zip(coefs(p.calibration), state["cal"]))
assert p.reconstruction is None, "the 3D result made with the old offsets must go"
for c in range(1, 4):
    assert abs(win.cameras._rows[c].spin.value() - state["after"][c]) < 1e-3, c
t = win.toast.text()
assert "→" in t and "cam2" in t, t
print("Use this calibration: offsets and calibration in the project, the panel and the toast say so OK")
undo()
assert p.offsets == WRONG, p.offsets
assert p.calibration is TRUE_CAL, "the calibration made at the applied offsets must go back with them"
assert win._wand_result == (None, None)
assert "wand calibration" in win.statusBar().currentMessage().lower(), win.statusBar().currentMessage()
print("one Ctrl+Z restored the offsets and the calibration OK")


def drive_partial(wiz):
    page = to_run_page(wiz)
    calibrate(page)
    refine(page)
    out = page.refined
    click_box(dict(page._ticks)[1])                          # untick cam2: only cam3 / cam4 change
    assert not dict(page._ticks)[1].isChecked()
    click(page.btn_apply_refined)
    assert "Calibrating again" in page.report.toPlainText(), page.report.toPlainText()[:300]
    assert page.wiz.result_calibration is None and not page.isComplete()
    wait_result(page)
    assert page.wiz.result is not out["res_after"]
    assert wiz.result_offsets[1] == WRONG[1] and wiz.result_offsets[2] == out["after"][2]
    state["partial"] = (list(wiz.result_offsets), list(out["after"]))
    return finish(wiz)


cw.WandWizard.exec = drive_partial
win._wand_wizard()
pump(0.3)
now, sugg = state["partial"]
assert p.offsets == now and p.offsets[1] == WRONG[1], (p.offsets, now)
assert abs(p.offsets[2] - sugg[2]) < 1e-12 and abs(p.offsets[2] - WRONG[2]) > 0.3, (p.offsets, sugg)
print("partial apply: calibrated again at the ticked offsets OK")
undo()
assert p.offsets == WRONG and p.calibration is TRUE_CAL


def drive_cancel(wiz):
    page = to_run_page(wiz)
    calibrate(page)
    first = coefs(page.wiz.result_calibration)
    refine(page)
    click(page.btn_apply_refined)
    assert wiz.result_offsets is not None and p.offsets == WRONG
    # (12) back to the project's own offsets: nothing pending, a calibration made at them again
    click(page.btn_project_offsets)
    assert wiz.result_offsets is None and "made at" not in page.offsets_note.text()
    wait_result(page)
    assert all(np.array_equal(a, b) for a, b in zip(first, coefs(page.wiz.result_calibration)))
    assert wiz.button(QWizard.FinishButton).text() == "Use this calibration"
    assert page.refine_box.isVisible(), "back at the offsets it started from: the refinement fits again"
    # (9) the whole-frame suggestion's Apply (I274) is pending too; then Cancel
    calibrate(page, timing="apply")
    assert wiz.result_offsets is not None and wiz.result_offsets[2] == WRONG[2] + 1, wiz.result_offsets
    assert p.offsets == WRONG, "the whole-frame Apply must not touch the project either"
    return cancel(wiz)


cw.WandWizard.exec = drive_cancel
pump(1.0)                         # (I305) the 3D result follows the Ctrl+Z above (a rebuild one quiet tick later)
before_cancel = (list(p.offsets), p.calibration, p.reconstruction)
win._wand_wizard()
pump(0.3)
assert p.offsets == WRONG and p.calibration is TRUE_CAL, "Cancel must leave the offsets and the calibration as they were"
assert (list(p.offsets), p.calibration, p.reconstruction) == before_cancel or (
    p.offsets == before_cancel[0] and p.calibration is before_cancel[1] and p.reconstruction is before_cancel[2])
print("Apply then Cancel: nothing changed; 'Use the project's offsets again' dropped the pending ones OK")


def drive_axes(wiz):
    page = to_run_page(wiz, axes=True)
    calibrate(page)
    rep = page.wiz.result.report
    assert "axes" in str(rep.get("frame", "")) and rep.get("frame_applied", True) is not False, rep.get("frame")
    refine(page)
    click(page.btn_apply_refined)
    wait_result(page)
    rep = page.wiz.result.report
    print(f"axes after the refine: {rep.get('frame')} (applied: {rep.get('frame_applied', True)})")
    assert any(abs(o - round(o)) > 0.05 for o in wiz.result_offsets), "the refine made the offsets fractional"
    assert "axes" in str(rep.get("frame", "")) and rep.get("frame_applied", True) is not False, rep.get("frame")
    return finish(wiz)


cw.WandWizard.exec = drive_axes
win._wand_wizard()
pump(0.3)
print("the axes from marks clicked on one frame survive fractional offsets OK")
undo()
assert p.offsets == WRONG and p.calibration is TRUE_CAL


def drive_subset(wiz):
    page = to_run_page(wiz, untick=2)                        # cam3 left out of the calibration
    calibrate(page)
    refine(page)
    assert sorted(c for c, _ in page._ticks) == [1, 3], page._ticks
    click(page.btn_apply_refined)
    wait_result(page)
    res = page.wiz.result
    state["subset"] = [np.asarray(c.coefs) for c in res.cameras]
    state["subset_offsets"] = list(wiz.result_offsets)
    assert wiz.result_offsets[2] == WRONG[2], "a camera left out keeps its offset"
    return finish(wiz)


cw.WandWizard.exec = drive_subset
win._wand_wizard()
pump(0.3)
cal = p.calibration
assert cal.views() == [0, 1, 3], cal.views()
for k, c in enumerate([0, 1, 3]):
    assert np.array_equal(np.asarray(cal.camera(c).coefs), state["subset"][k]), c
assert p.offsets == state["subset_offsets"]
print("a camera unticked on the wand page: each calibrated camera got its own coefficients OK")


def drive_again(wiz):
    page = to_run_page(wiz)
    calibrate(page)
    state["second"] = page.wiz.result_calibration
    assert wiz.result_offsets is None
    return finish(wiz)


cw.WandWizard.exec = drive_again
win._wand_wizard()
pump(0.3)
second = p.calibration
assert second is state["second"]
undo()
assert p.offsets == state["subset_offsets"] and p.calibration is second, \
    "a later wizard Finish must not be folded into the earlier Ctrl+Z step"
print("a second Finish without new offsets kept its own calibration under Ctrl+Z OK")

win._dev_probe.wait(60000)
win.project.dirty = False
for s in win.project.sessions:
    s.dirty = False
win.close()
pump(0.3)
for pth in paths:
    forget_recovery(pth)
print("verify_wand_sync_gui PASSED")
