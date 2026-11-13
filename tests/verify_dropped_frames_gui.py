"""Offscreen GUI: a video whose camera dropped frames, from the moment it is open (I382, I383).

The user's gestures: open the video, jump about and step with the keys RIGHT AWAY (before any
background check can have finished). The picture on screen at frame N must be the N-th picture
reading forward (every picture of the clip carries its own number), and the window says once, in
plain words, what kind of file this is.

Run: .venv\\Scripts\\python.exe tests\\verify_dropped_frames_gui.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import _synth_frames as sf  # noqa: E402
from _clean import forget_recovery  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "dropped_frames")
VID = sf.make_clips(OUT, ["dropped_late"])["dropped_late"]
REF = sf.forward(VID)
forget_recovery(VID)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace import video_source as vs  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402
stub_track_choice()

app = QApplication.instance() or QApplication([])
win = MainWindow()
# the background frame check is held back: what is tested is the state BEFORE it has run
win._frames_check_soon = lambda *a, **k: None
win.resize(1400, 900)
win.show()
FAILED = []
SAID = []
_show = win.toast.show_message


def _spy(text, *a, **k):
    SAID.append(str(text))
    return _show(text, *a, **k)


win.toast.show_message = _spy


def pump(seconds=0.1):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def check(ok, what, detail=""):
    print(("OK   " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAILED.append(what)


def shown():
    """The number drawn in the picture the window holds for the frame it is on."""
    for _ in range(200):
        pump(0.03)
        if not win._picture_pending():
            break
    rgb = win.cache.get(win.current) if win.cache is not None else None
    return sf.barcode(rgb)


app.setActiveWindow(win)
vs.set_seek_plan(VID, None)
win._open_video(VID)
for _ in range(400):
    pump(0.05)
    if win.state == READY:
        break
assert win.state == READY, "the video did not open"
check(win.n_frames == len(REF), "the timeline has as many frames as the video has pictures",
      f"{win.n_frames} vs {len(REF)}")
plan = vs.seek_plan(VID)
check(plan is not None and not plan.exact and plan.provisional,
      "the window has the file's plan before any background check ran", repr(plan)[:120])
note = [t for t in SAID if "dropped" in t]
check(len(note) == 1 and "exact" in note[0] and "re-encode" in note[0],
      "the window says once, at open, that the camera dropped frames and what Kinetrace does", str(SAID)[:200])

# jumps (the timeline's click ends in _goto), then steps with the keys across the frames that were wrong
wrong = []
for f in (452, 152, 598, 399, 400, 401, 30, 153, 451):
    win._goto(f)
    got = shown()
    if got != REF[f]:
        wrong.append((f, REF[f], got))
check(not wrong, "a jump shows the forward picture of that frame", f"(frame, forward, shown) {wrong[:5]}")
win._goto(146)
shown()
wrong = []
for _ in range(10):                                   # 147 ... 156, over 151 / 152
    QTest.keyClick(win, Qt.Key_Right)
    got = shown()
    if got != REF[win.current]:
        wrong.append((win.current, REF[win.current], got))
check(win.current == 156, "the step key moved ten frames on", str(win.current))
check(not wrong, "stepping forward shows each frame's own picture", str(wrong[:5]))
wrong = []
for _ in range(8):
    QTest.keyClick(win, Qt.Key_Left)
    got = shown()
    if got != REF[win.current]:
        wrong.append((win.current, REF[win.current], got))
check(win.current == 148 and not wrong, "stepping back too", f"at {win.current}: {wrong[:5]}")

# when the background check does report, the same sentence is not said a second time
n_said = len([t for t in SAID if "dropped" in t])
checked = vs.check_seeks(VID, win.n_frames)
vs.set_seek_plan(VID, checked)
win._say_seek_note(VID, "cam")
check(len([t for t in SAID if "dropped" in t]) == n_said, "the check's result is not said again")
win._goto(452)
check(shown() == REF[452], "and the picture is still the right one after the check")

win._dev_probe.wait(15000)
if win.project is not None:
    win.project.dirty = False
win.close()
forget_recovery(VID)
vs.set_seek_plan(VID, None)
if FAILED:
    print(f"DROPPED FRAMES GUI FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("DROPPED FRAMES GUI PASSED")
