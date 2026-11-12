"""Offscreen GUI: the first minutes of a new user (G236-G239).

- a video dropped on the window opens (G238); a file that is not a video is refused in words;
- the Getting started strip is Open video > Click a point > Track (G237);
- on a video with no point yet, one plain click on the picture places the first point, and Ctrl+Z takes it
  back (G236); with points present and none selected a click still places nothing and says so;
- a run's stop notice uses plain words and stays until the user acts (G239).

Run: .venv\\Scripts\\python.exe tests\\verify_first_click.py   (needs test600.mp4)
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from _clean import forget_recovery  # noqa: E402

VID = os.path.join(ROOT, "test600.mp4")
if not os.path.exists(VID):
    sys.exit("test600.mp4 missing - python make_test_video.py test600.mp4 --seed 0")
forget_recovery(VID)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace import app as kapp  # noqa: E402
from kinetrace.app import IDLE, READY, MainWindow  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402
stub_track_choice()

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1500, 950)
win.show()
FAILED = []


def pump(seconds=0.15):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def check(ok, what, detail=""):
    print(("OK   " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(what)


def drop(paths):
    """What the desktop sends when files are let go over the window: an enter, then the drop."""
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(p) for p in paths])
    pos = QPoint(win.width() // 2, win.height() // 2)
    enter = QDragEnterEvent(pos, Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(win, enter)
    ev = QDropEvent(QPointF(pos), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(win, ev)
    pump(0.1)
    return enter.isAccepted()


def click_video(x, y):
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, win.canvas.mapFromScene(QPointF(x, y)))
    pump(0.1)


app.setActiveWindow(win)
pump()

# ---- G237: the strip's steps ---------------------------------------------------------------
chips = [b.text() for b in win.onboarding._chips]
check(len(chips) == 3 and "Open video" in chips[0] and "Click a point" in chips[1] and "Track" in chips[2],
      "the strip reads Open video > Click a point > Track", str(chips))
check(not any("Segment" in c or "Skeleton" in c for c in chips), "no optional step among the numbered ones", str(chips))

# ---- G238: a dropped file ------------------------------------------------------------------
note = os.path.join(ROOT, "tests", "out", "first_click_note.txt")
os.makedirs(os.path.dirname(note), exist_ok=True)
with open(note, "w", encoding="utf-8") as fh:
    fh.write("not a video")
win.toast.hide()
drop([note])
check(win.state == IDLE and win.session is None, "a dropped text file opens nothing")
check(win.toast.isVisible() and "video" in win.toast.text().lower(), "... and the window says what can be dropped",
      win.toast.text()[:120])
win.toast.hide()
accepted = drop([VID])
for _ in range(300):
    pump(0.05)
    if win.state == READY:
        break
check(accepted and win.state == READY and win.info is not None
      and os.path.samefile(win.info.path, VID), "a video dropped on the window opens")
if win.state != READY:
    win._open_video(VID)
    for _ in range(300):
        pump(0.05)
        if win.state == READY:
            break
assert win.state == READY, "the video did not open at all"
for _ in range(100):                       # the first picture is on screen
    pump(0.05)
    if not win._picture_pending():
        break
s = win.session

# ---- G236: the first click places the first point ----------------------------------------
check(s.n_points == 0, "a fresh video has no point")
click_video(200.0, 150.0)
ok = s.n_points == 1 and bool(s.tracked[win.current, 0])
check(ok, "one plain click on a video without points places the first point", f"n_points={s.n_points}")
if ok:
    x, y = s.tracks[win.current, 0]
    check(abs(x - 200.0) < 1.0 and abs(y - 150.0) < 1.0, "... exactly where clicked", f"{x:.2f}, {y:.2f}")
    check(win.selected == 0, "... and it is selected, ready for Track")
    check(not win.btn_add.isChecked(), "... with the Point tool not left armed")
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.1)
    check(win.session.n_points == 0, "Ctrl+Z takes the first point back", f"n_points={win.session.n_points}")
    s = win.session
    click_video(210.0, 160.0)
    check(s.n_points == 1, "the click after the undo places it again")
    # (review) a double-click is ONE first click: its second press must not hand-place the new point again
    # (a second undo step: one Ctrl+Z then left the point there)
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.1)
    s = win.session
    win.toast.hide()                      # a notice over the picture would take the click
    # what a real double-click delivers: press, release, double-click, release (QTest.mouseDClick sends a
    # different sequence that places nothing here)
    _vp, _at = win.canvas.viewport(), win.canvas.mapFromScene(QPointF(205.0, 155.0))
    QTest.mousePress(_vp, Qt.LeftButton, Qt.NoModifier, _at)
    QTest.mouseRelease(_vp, Qt.LeftButton, Qt.NoModifier, _at)
    QApplication.sendEvent(_vp, QMouseEvent(QEvent.MouseButtonDblClick, QPointF(_at), QPointF(_vp.mapToGlobal(_at)),
                                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    QTest.mouseRelease(_vp, Qt.LeftButton, Qt.NoModifier, _at)
    pump(0.15)
    check(s.n_points == 1, "a double-click on the empty video places one point", f"n_points={s.n_points}")
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.1)
    check(win.session.n_points == 0, "... and ONE Ctrl+Z takes it back", f"n_points={win.session.n_points}")
    s = win.session
    click_video(210.0, 160.0)
# the strip's second step
win._refresh_onboarding()
done = win.onboarding._last_state[0] if win.onboarding._last_state else None
check(bool(done) and list(done)[:2] == [True, True], "the strip ticks Click a point once a point exists", str(done))

# with points present and nothing selected, a click still places nothing (G21 stays)
if s.n_points:
    n0 = s.n_points
    win._deselect()
    pump(0.05)
    win.toast.hide()
    click_video(400.0, 300.0)
    check(s.n_points == n0 and win.toast.isVisible() and "Nothing was placed" in win.toast.text(),
          "points present, none selected: a click places nothing and says so")
    win.toast.hide()

# the strip's Click a point chip arms the Point tool
if len(win.onboarding._chips) == 3:
    QTest.mouseClick(win.onboarding._chips[1], Qt.LeftButton)
    pump(0.1)
    check(win.btn_add.isChecked(), "the Click a point step arms the Point tool")
    QTest.keyClick(win, Qt.Key_Escape)
    pump(0.05)

# ---- G239: stop notices ---------------------------------------------------------------------
words = []
for kind, t in kapp._STOP_TEXTS.items():
    for key, v in t.items():
        if isinstance(v, str):
            words.append((kind, key, v))
bad = [(k, key) for k, key, v in words
       if "SAM" in v or "segment" in v.lower() or "confidence" in v.lower()]
check(not bad, "no stop notice says SAM, segment or confidence", str(bad))
if s.n_points:
    win._on_select(0)
    win.toast.hide()
    win._say_stop("unreliable", s.points[0].name, win.current + 5)
    pump(0.05)
    check(win.toast.isVisible() and not win.toast._timer.isActive(),
          "a stop notice stays on screen (no timer hides it)")
    # (review) stickiness belongs to ITS notice. On a Toast of its own: the window's may get a notice of
    # the program's at any moment (a recovery reminder did, and pushed the stop notice out)
    from PySide6.QtWidgets import QWidget
    from kinetrace.widgets import Toast
    host = QWidget()
    host.resize(900, 300)
    host.show()
    tt = Toast(host)
    tt.show_message("the run stopped", "warn", 0)
    tt.show_message("saved", "info", 1500)
    pump(2.2)
    check(tt.isVisible() and "the run stopped" in tt.text() and "saved" not in tt.text(),
          "a timed notice beside a waiting one times out; the waiting one stays", tt.text()[:80])
    tt.show_message("one", "info", 1500)
    tt.show_message("two", "info", 1500)       # the waiting notice scrolls out: nothing waits any more
    pump(2.2)
    check(not tt.isVisible(), "once the waiting notice has scrolled out, the later notices time out")
    tt.show_message("the run stopped", "warn", 0)
    tt.show_message("saved", "info", 1500)
    tt.dismiss_sticky()
    check(tt.isVisible() and "saved" in tt.text() and "the run stopped" not in tt.text(),
          "the user acting takes the waiting notice, not the timed one beside it")
    host.close()
    # a drop while something is opening is held back like the keyboard and the mouse (review)
    win._busy_stack.append({"title": "test"})
    before = win.info.path
    n_busy = len(win._busy_stack)
    drop([VID])
    pump(0.2)
    check(len(win._busy_stack) == n_busy and win.info.path == before,
          "a drop while the loading card is up opens nothing")
    win._busy_stack.pop()
    win.toast.hide()
    win._say_stop("unreliable", s.points[0].name, win.current + 5)
    pump(0.05)
    click_video(220.0, 170.0)
    check(not win.toast.isVisible() or "Auto-paused" not in win.toast.text(),
          "... until the user clicks on the video")

win._dev_probe.wait(15000)
if win.project is not None:
    win.project.dirty = False
win.close()
forget_recovery(VID)
if FAILED:
    print(f"FIRST CLICK FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("FIRST CLICK PASSED")
