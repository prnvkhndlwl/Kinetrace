"""An animal's row in LAYERS (G154; was the SEGMENT row).

Offscreen, no GPU: a synthetic silhouette is written into a real session and
the SEGMENT section of the panel is driven the way a user would — the row
appears as soon as a segment exists, its checkbox shows / hides the
silhouette, editing its text renames it (the timeline lane follows), and the
right-click menu jumps to the silhouette, clears this frame / a selected
window / everything, toggles the overlays, and removes the segment. Undo,
the state gate during tracking, and a project round trip of the name are
covered too.
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

VID = os.path.join(ROOT, "test600.mp4")
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
_typed = {"v": ""}
QInputDialog.getText = staticmethod(lambda *a, **k: (_typed["v"], True))

app = QApplication([])
from kinetrace.app import MainWindow, READY, TRACKING

win = MainWindow()
win.resize(1280, 860)
win.show()


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


win._open_video(VID)
pump(lambda: win.state == READY, 20, "open")
s = win.session

# ---- no animal yet: no row ---------------------------------------------------
assert win._animal_item(0) is None and s.n_segments == 0
assert "No animal yet" in win.animal_label.text()
print("no segment: the row stays out of the way OK")

# ---- a segment with silhouettes on frames 10..40 -------------------------------
s.ensure_animal()
s.animal.add_click(10, 320.0, 240.0, True)
bitmap = np.zeros((s.height, s.width), bool)
bitmap[200:300, 280:380] = True
for f in range(10, 41):
    s.masks.set(f, bitmap, 9.0)
win._refresh_animal_panel()
app.processEvents()
item = win._animal_item(0)
assert item is not None and s.n_segments == 1
assert item.text(0) == "animal", item.text(0)
assert item.checkState(0) == Qt.Checked
assert not item.icon(0).isNull(), "the animal row needs its colour swatch"
assert "31 of 600" in item.toolTip(0), item.toolTip(0)
# the line under LAYERS: the numbers of the selected (here: the only) animal, by name
assert "silhouette on 31 of 600 frames" in win.animal_label.text(), win.animal_label.text()
print("the row appears with the swatch, name, checkbox and counts OK")

# ---- checkbox shows / hides the silhouette -------------------------------------
win._goto(20)
assert win.canvas._mask_item.isVisible(), "the silhouette should be drawn at frame 20"
# (G154) the row's checkbox is THIS animal's silhouette; the toolbar's Mask hides them all
item.setCheckState(0, Qt.Unchecked)
app.processEvents()
assert not s.animal.shown and not win.canvas._mask_item.isVisible() and win.btn_mask.isChecked()
win._animal_item(0).setCheckState(0, Qt.Checked)
app.processEvents()
assert s.animal.shown and win.canvas._mask_item.isVisible()
win.btn_mask.setChecked(False)
app.processEvents()
assert not win.canvas._mask_item.isVisible() and win._animal_item(0).checkState(0) == Qt.Checked
win.btn_mask.setChecked(True)
app.processEvents()
print("the row's checkbox hides its own silhouette, Mask hides all OK")

# ---- rename by editing the row ---------------------------------------------------
win._animal_item(0).setText(0, "iguana")
app.processEvents()
assert s.animal.name == "iguana", s.animal.name
assert win._animal_item(0).text(0) == "iguana"
win.timeline.repaint()
app.processEvents()
win._animal_item(0).setText(0, "   ")           # blank falls back, never empties the name
app.processEvents()
assert s.animal.name == "iguana", s.animal.name
print("inline rename OK")

# ---- context menu: jumps ----------------------------------------------------------
menu, acts = win._build_animal_menu()
assert acts["show_mask"].isChecked() and acts["clear_all"].isEnabled()
win._goto(300)
win._animal_menu_action(acts["first"], acts)
assert win.current == 10, win.current
win._animal_menu_action(acts["last"], acts)
assert win.current == 40, win.current
menu.deleteLater()

# ---- context menu: overlays -------------------------------------------------------
menu, acts = win._build_animal_menu()
acts["show_midline"].setChecked(False)
win._animal_menu_action(acts["show_midline"], acts)
assert not win.act_show_midline.isChecked()
win.act_show_midline.setChecked(True)
menu, acts = win._build_animal_menu()
acts["show_mask"].setChecked(False)
win._animal_menu_action(acts["show_mask"], acts)
assert not s.animal.shown and win._animal_item(0).checkState(0) == Qt.Unchecked
menu, acts = win._build_animal_menu()
acts["show_mask"].setChecked(True)
win._animal_menu_action(acts["show_mask"], acts)
assert s.animal.shown
print("menu overlay toggles OK")

# ---- context menu: rename through the dialog ---------------------------------------
_typed["v"] = "lizard"
menu, acts = win._build_animal_menu()
win._animal_menu_action(acts["rename"], acts)
assert s.animal.name == "lizard" and win._animal_item(0).text(0) == "lizard"

# ---- context menu: clear this frame, undo ------------------------------------------
win._goto(20)
menu, acts = win._build_animal_menu()
assert acts["clear_here"].isEnabled()
win._animal_menu_action(acts["clear_here"], acts)
assert not s.masks.has(20) and s.masks.has(21) and s.masks.n_masked() == 30
assert "30 of 600" in win._animal_item(0).toolTip(0)
win._undo_run()
win._refresh_animal_panel()
assert s.masks.has(20) and s.masks.n_masked() == 31, "Ctrl+Z restores the silhouette"
win._goto(500)
menu, acts = win._build_animal_menu()
assert not acts["clear_here"].isEnabled(), "no silhouette here: nothing to clear"
print("clear-this-frame + undo OK")

# ---- context menu: clear a selected window -------------------------------------------
menu, acts = win._build_animal_menu()
assert not acts["clear_window"].isEnabled(), "no timeline selection yet"
win.timeline.sel_range = (30, 40)
win.timeline.sel_rows = []
win.timeline.sel_seg = True
menu, acts = win._build_animal_menu()
assert acts["clear_window"].isEnabled() and "30" in acts["clear_window"].text()
win._animal_menu_action(acts["clear_window"], acts)
assert s.masks.n_masked() == 20 and s.masks.has(29) and not s.masks.has(35)
print("clear-selected-window OK")

# ---- context menu: clear everything, the segment survives -------------------------------
menu, acts = win._build_animal_menu()
win._animal_menu_action(acts["clear_all"], acts)
assert s.masks.n_masked() == 0 and s.animal is not None, "the clicks must survive"
assert s.animal.n_prompts() == 1
assert win._animal_item(0) is not None and s.n_segments == 1
menu, acts = win._build_animal_menu()
assert not acts["clear_all"].isEnabled() and not acts["first"].isEnabled()
print("clear-all keeps the segment and its clicks OK")

# ---- the row is dead while tracking -----------------------------------------------------
win.state = TRACKING
win._apply_state()
assert not win.layers.isEnabled()
win.state = READY
win._apply_state()
assert win.layers.isEnabled()

# ---- project round trip of the name -------------------------------------------------------
proj = os.path.join(SCRATCH, "segment_panel.kinetrace")
win._sync_ui_state()
win.project.save(proj)
from kinetrace.project import Project
back = Project.load(proj)
assert back.session.animal is not None and back.session.animal.name == "lizard"

# ---- context menu: remove the segment ---------------------------------------------------
menu, acts = win._build_animal_menu()
win._animal_menu_action(acts["remove"], acts)
app.processEvents()
assert s.animal is None and s.masks is None
assert win._animal_item(0) is None and s.n_segments == 0
assert "No animal yet" in win.animal_label.text()
print("remove-the-segment from the row OK")

win._dev_probe.wait(30000)
win.close()
app.processEvents()
forget_recovery(VID)
print("verify_segment_panel PASSED")
