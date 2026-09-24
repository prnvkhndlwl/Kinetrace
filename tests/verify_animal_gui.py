"""Offscreen GUI E2E for the animal layer: open -> A + click the animal (canvas
gesture) -> mask preview -> apply a skeleton -> place the head -> Track (fused)
-> overlays + timeline lane -> export everything -> project round-trip -> undo
-> clear animal. GPU, ~4 min."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest

import _synth  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.critical = staticmethod(lambda *a, **k: print("CRITICAL:", a[2][:400]) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace.app import IDLE, READY, TRACKING, MainWindow, apply_theme  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)
VID = _synth.build_video(os.path.join(OUT, "synth_animals.mp4"))
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID)
W, H, T = _synth.W, _synth.H, _synth.T
GT = [_synth.gt_frame(t)[1][0] for t in range(T)]

app = QApplication([])
apply_theme(app)
win = MainWindow()
win.resize(1500, 900)
win.show()


def pump(seconds: float):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def wait_until(pred, timeout: float, what: str):
    t0 = time.time()
    while not pred():
        app.processEvents()
        time.sleep(0.02)
        assert time.time() - t0 < timeout, f"timeout waiting for {what}"


win._open_video(VID)
wait_until(lambda: win.state == READY, 30, "video open")
pump(0.5)
s = win.session
assert s is not None and win.n_frames == T

# ---- Segment ▾ model dropdown (parity with the Track ▾ point model) --------
from kinetrace.segmenter import BACKENDS

menu = win.btn_animal.menu()
assert menu is not None, "the Segment button must carry a model dropdown"
from PySide6.QtWidgets import QToolButton

assert win.btn_animal.popupMode() == QToolButton.MenuButtonPopup, \
    "the body must still toggle the tool; only the arrow opens the menu"
menu.aboutToShow.emit()                       # labels are rebuilt on open
app.processEvents()
assert set(win._seg_acts) == set(BACKENDS), "every backend must be listed"
assert win._seg_acts[win._seg_backend].isChecked(), "the live model must be ticked"
for k, a in win._seg_acts.items():
    assert BACKENDS[k][2] in a.text() and "[" in a.text(), \
        f"each entry needs its label + status badge: {a.text()!r}"
assert win.act_settings in menu.actions(), "the menu must still reach the full Settings dialog"
_before = win._seg_backend
_other = next(k for k in BACKENDS if k != _before)
win._set_seg_backend(_other)
assert win._seg_backend == _other and win._seg_acts[_other].isChecked() \
    and not win._seg_acts[_before].isChecked(), "picking a model must move the tick"
win._set_seg_backend(_before)                 # back to what the run below needs
assert win._seg_backend == _before
print("segment model dropdown OK")

# ---- animal tool via the real canvas gesture -------------------------------
QTest.keyClick(win, Qt.Key_S)
pump(0.1)
assert win.btn_animal.isChecked() and win.canvas._animal_mode, "S must arm the segment tool"
c0 = GT[0]["centre"]
vp = win.canvas.mapFromScene(QPointF(float(c0[0]), float(c0[1])))
QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
pump(0.1)
assert s.animal is not None and s.animal.has_prompt(0), "click must create a prompt"
wait_until(lambda: win._preview is None, 240, "mask preview")   # first use loads the model
pump(0.2)
assert s.masks is not None and s.masks.has(0), "preview mask missing"
assert _synth.iou(s.masks.rasterize(0, H, W), GT[0]["mask"]) > 0.75
assert win.canvas._mask_item.isVisible(), "mask overlay not shown"
# negative click on the other animal must not break anything; then remove it again
c1 = _synth.gt_frame(0)[1][1]["centre"]
vp1 = win.canvas.mapFromScene(QPointF(float(c1[0]), float(c1[1])))
QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.ShiftModifier, vp1)
pump(0.1)
assert len(s.animal.prompts[0]) == 2 and s.animal.prompts[0][1][2] == 0
wait_until(lambda: win._preview is None, 60, "second preview")
win._on_prompt_remove(1)
wait_until(lambda: win._preview is None, 60, "third preview")
assert len(s.animal.prompts[0]) == 1
QTest.keyClick(win, Qt.Key_Escape)
assert not win.btn_animal.isChecked(), "Esc must disarm the segment tool first"
print("animal tool + preview OK")

# ---- skeleton template, place the head by the click-continue rule ------------
tpl = next(t for t in __import__("kinetrace.skeletons", fromlist=["all_templates"]).all_templates()
           if t["name"].startswith("Undulating"))
win._apply_skeleton_template(tpl)
pump(0.1)
assert s.skeleton and s.n_points == len(tpl["landmarks"]) and len(s.derived_pids()) == 4
snout = s.pid_by_name("snout")
win._on_select(snout)
QTest.keyClick(win, Qt.Key_N)
assert win.btn_add.isChecked()
win._on_add(float(GT[0]["eye"][0]), float(GT[0]["eye"][1]))   # continues the selected unplaced landmark
assert s.tracked[0, snout] and s.n_points == len(tpl["landmarks"]), "click-continue must place the landmark"
assert win.btn_track.isEnabled()
assert win.btn_onbody.isChecked(), "Body constraint must default to on"
print("skeleton OK")

# ---- fused tracking run --------------------------------------------------------
win.point_list.clearSelection()   # a single selected landmark would scope the run to itself
assert win.btn_track.text().startswith("Track ") and "sel." not in win.btn_track.text()
win._start_tracking()
wait_until(lambda: win.state == TRACKING, 10, "tracking start")
wait_until(lambda: win.state == READY, 400, "tracking end")
pump(0.3)
assert s.masks.n_masked() >= 0.95 * T, s.masks.n_masked()
tip = s.pid_by_name("tail_tip")
errs = [np.linalg.norm(s.tracks[t, tip] - GT[t]["tip"]) for t in range(T) if s.tracked[t, tip]]
print(f"tracked: masks {s.masks.n_masked()}/{T}, tail tip err mean {np.mean(errs):.2f} px on {len(errs)} frames, "
      f"snout tracked {int(s.tracked[:, snout].sum())} frames")
assert len(errs) >= 0.9 * T and np.mean(errs) < 10
assert s.tracked[:, snout].sum() >= 0.95 * T
win._goto(60)
pump(0.2)
assert win.canvas._mask_item.isVisible() and win.canvas._midline_item.isVisible()
assert win.canvas._bones_item.isVisible(), "bones between placed landmarks"
assert win.timeline._animal_h() > 0 and win.timeline._ani_col is not None and win.timeline._ani_col.any()
# follow mode frames the segment when no point is selected: zoom in far away, then step a frame
win._deselect()
win.btn_follow.setChecked(True)
win.canvas.set_view_state(3.0, 30.0, 30.0, True)          # a corner, nowhere near the animal
win._goto(61)
pump(0.2)
assert win.canvas._mask_rect is not None
view = win.canvas.mapToScene(win.canvas.viewport().rect()).boundingRect()
c = win.canvas._mask_rect.center()
assert view.contains(c), f"follow did not frame the segment: view {view}, segment centre {c}"
print("follow frames the segment OK")
win.timeline.grab().save(os.path.join(OUT, "animal_timeline.png"))
win.grab().save(os.path.join(OUT, "animal_gui.png"))
# mask toggle hides the overlay
win.btn_mask.setChecked(False)
pump(0.1)
assert not win.canvas._mask_item.isVisible()
win.btn_mask.setChecked(True)

# ---- exports (all formats) -----------------------------------------------------
base = os.path.join(OUT, "gui_export")
written = []
for lab, suf, key in win.EXPORT_FORMATS:
    if key != "all":
        written += win._export_one(key, base + "_" + key + suf)
assert all(os.path.exists(p) for p in written) and any(p.endswith("_segment.csv") for p in written)
print("exports OK:", len(written), "files")

# ---- project round-trip restores animal + skeleton + masks ---------------------
proj = os.path.join(OUT, "gui_animal.kinetrace")
win.project_path = __import__("pathlib").Path(proj)
win._save_project()
n_masked = s.masks.n_masked()
win._open_project_from_path(proj)
wait_until(lambda: win.state == READY and win.session is not s, 30, "project reopen")
pump(0.3)
r = win.session
assert r.animal is not None and r.masks.n_masked() == n_masked and r.skeleton["name"] == tpl["name"]
# the segment's own panel row survives the round trip; the status line carries the counts
assert win.animal_list.isVisible() and win.animal_list.count() == 1
assert win.animal_list.item(0).text() == "segment", win.animal_list.item(0).text()
assert f"silhouette on {n_masked:,} of" in win.animal_label.text(), win.animal_label.text()
print("project round-trip OK")

# ---- undo restores the pre-run masks; clear animal ------------------------------
assert win._undo_snap is None   # a fresh load has nothing to undo
win._clear_masks_window(0, 40)
assert r.masks.n_masked() < n_masked and win.act_undo.isEnabled()
win._undo_run()
assert r.masks.n_masked() == n_masked
win._clear_animal()
assert r.animal is None and r.masks is None and not win.btn_clear_animal.isEnabled()
assert win.timeline._animal_h() == 0
print("undo + clear OK")

win.close()
pump(0.2)
win._dev_probe.wait(20000)
forget_recovery(VID)
print("ANIMAL GUI E2E PASSED")
