"""Offscreen GUI: retiming a camera moves ONLY that camera's picture (I380), and the one rig-wide
"Align all cameras to their current visible frame" button (G233). No tracking, no GPU.

Three clips cut from one synthetic world at different starts (camA 0, camB 10, camC 25), so the true
offsets are [0, -10, -25]. "What a camera shows" = the frame it asks its decoder for: the playhead for the
working camera, `want_frame` for the others.

  [1] working in camB under Sync all, its offset box's up arrow (a real click) -> camB one frame later,
      camA and camC on the SAME frames as before (I380: before the fix camB stayed and the others moved)
  [2] the same for a number typed into the box
  [3] the button under Sync all changes nothing, switches to Active view only and says the steps
  [4] Active view only: camA parked at 60, camB at 50, camC never visited -> one press gives [0, -10, kept];
      every parked camera keeps its picture, the status names the kept camera; one Ctrl+Z puts it back
  [5] the button never widens the dock (its long name, the G3 pitfall)
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QDialogButtonBox, QFileDialog, QMessageBox, QSizePolicy, QStyle,
                               QStyleOptionSpinBox)

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)

W, H, WORLD, LEN = 320, 240, 160, 120
STARTS = {"alignA.mp4": 0, "alignB.mp4": 10, "alignC.mp4": 25}


def build_clips():
    frames = []
    for t in range(WORLD):
        img = np.full((H, W, 3), 18, np.uint8)
        cv2.circle(img, (20 + int(1.5 * t) % (W - 40), 120), 9, (250, 250, 250), -1)
        cv2.putText(img, str(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 60), 2)
        frames.append(img)
    paths = []
    for name, start in STARTS.items():
        p = os.path.join(SCRATCH, name)
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        for t in range(start, min(WORLD, start + LEN)):
            vw.write(frames[t])
        vw.release()
        paths.append(p)
    return paths


PATHS = build_clips()

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
_pick = {"files": []}
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (list(_pick["files"]), ""))

app = QApplication([])
from _clean import forget_recovery  # noqa: E402
forget_recovery(*PATHS)
from kinetrace import cameraorder  # noqa: E402
from kinetrace.app import MainWindow, READY  # noqa: E402

cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(), self.result())[1]

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


def settle(seconds=0.35):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.005)


def shown():
    """The frame each camera shows now: the playhead in the working camera, `want_frame` elsewhere."""
    p = win.project
    return [win.current if v == p.active else win._views[v].want_frame for v in range(p.n_views)]


def click_row(i):
    """Click camera i's line in CAMERAS (switches the working camera)."""
    lst = win.cameras.list
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, pos=lst.visualItemRect(lst.item(i)).center())
    settle()
    assert win.project.active == i, f"clicking camera {i + 1}'s line made camera {win.project.active + 1} working"


def open_row(i):
    row = win.cameras._rows[i]
    if not row.expanded():
        QTest.mouseClick(row.btn_expand, Qt.LeftButton)
    settle(0.1)
    return row


def click_up_arrow(spin):
    opt = QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    r = spin.style().subControlRect(QStyle.CC_SpinBox, opt, QStyle.SC_SpinBoxUp, spin)
    QTest.mouseClick(spin, Qt.LeftButton, pos=r.center())


win._open_video(PATHS[0])
pump(lambda: win.state == READY, 30, "open camA")
_pick["files"] = PATHS[1:]
win._add_video_dialog()
settle()
p = win.project
assert p.n_views == 3, p.n_views
p.set_offset(1, 3.0)               # wrong on purpose: the alignment below has something to fix
p.set_offset(2, 7.0)
win._refresh_cameras()
settle()

# ---- [1] the working camera's up arrow moves that camera alone (I380) -------------------------------
click_row(1)
win._goto(40, force=True)
settle()
before = shown()
row = open_row(1)
click_up_arrow(row.spin)
settle()
after = shown()
assert p.offsets[1] == 4.0, f"the up arrow must raise camB's offset by one: {p.offsets}"
assert after[1] == before[1] + 1, f"camB (the one retimed) must show one frame later: {before} -> {after}"
assert after[0] == before[0] and after[2] == before[2], \
    f"the other cameras must stay on their pictures: {before} -> {after}"
print(f"[1] up arrow in the working camera: frames {before} -> {after}, only camB moved OK")

# ---- [2] the same for a typed number ----------------------------------------------------------------
before = shown()
QTest.mouseClick(row.spin, Qt.LeftButton)
row.spin.selectAll()
QTest.keyClicks(row.spin, "9")
QTest.keyClick(row.spin, Qt.Key_Return)
settle()
after = shown()
assert p.offsets[1] == 9.0, p.offsets
assert after[1] == before[1] + 5 and after[0] == before[0] and after[2] == before[2], \
    f"typed 4 -> 9: only camB moves, by 5 frames: {before} -> {after}"
print(f"[2] typed offset: frames {before} -> {after}, only camB moved OK")
p.set_offset(1, 3.0)
win._refresh_cameras()
settle()

# ---- [3] under Sync all the button explains and switches to Active view only (G233) -----------------
btn = win.cameras.btn_align_all
assert btn.isVisible() and btn.isEnabled(), "the rig-wide Align button shows with two or more cameras"
assert btn.text() == "Align all cameras to their current visible frame", btn.text()
assert not any(hasattr(r, "btn_align") for r in win.cameras._rows), "no per-row Align here any more"
assert win._sync_mode == "all"
off0 = list(p.offsets)
QTest.mouseClick(btn, Qt.LeftButton)
settle()
msg = win.statusBar().currentMessage()
assert p.offsets == off0, f"under Sync all the press must change nothing: {off0} -> {p.offsets}"
assert win._sync_mode == "active", "the press switches to Active view only"
assert "park" in msg.lower(), f"the status must say the steps: {msg!r}"
print(f"[3] Sync all: nothing changed, Active view only now, the steps said: {msg[:90]!r} OK")

# ---- [4] park camA at 60 and camB at 50 (both world frame 60), camC never visited -------------------
click_row(0)
win._goto(60, force=True)
settle()
click_row(1)
win._goto(50, force=True)
settle()
parked = shown()
QTest.mouseClick(btn, Qt.LeftButton)
settle()
msg = win.statusBar().currentMessage()
assert p.offsets == [0.0, -10.0, 7.0], f"camB aligned to camA, camC (not parked) kept: {p.offsets}"
assert shown()[:2] == parked[:2], f"the parked cameras keep their pictures: {parked} -> {shown()}"
assert p.map_frame(0, 1, 60) == 50, "camA's 60 and camB's 50 are now the same instant"
assert p.name(2) in msg and "kept" in msg, f"the status names the camera kept: {msg!r}"
print(f"[4] aligned: offsets {p.offsets}; {msg[:110]!r} OK")
assert win.act_undo.isEnabled(), "the alignment is one undo step"
win.act_undo.trigger()
settle()
assert p.offsets == off0, f"Ctrl+Z puts every offset back: {p.offsets} vs {off0}"
print("[4] one Ctrl+Z puts the offsets back OK")

# ---- [5] its long name never widens the dock (G3 pitfall) -------------------------------------------
assert btn.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored, "the button must not set the panel's width"
assert win.cameras.minimumSizeHint().width() < btn.fontMetrics().horizontalAdvance(btn.text()), \
    "the panel's minimum width must not come from the button's text"
from PySide6.QtWidgets import QToolButton  # noqa: E402
_shown = QToolButton.text(btn)            # what it draws (offscreen fonts are much wider than real ones)
assert _shown == btn.text() or (_shown.endswith("…") and btn.text() in btn.toolTip()), \
    f"a narrow panel shortens the name with ... and the tooltip keeps it whole: {_shown!r}"
print(f"[5] the button ({btn.width()} px) shows {_shown!r} and never widens the dock OK")

win._dev_probe.wait(180_000)
win.close()
app.processEvents()
forget_recovery(*PATHS)
print("CAMERA OFFSET ALIGN PASSED")
