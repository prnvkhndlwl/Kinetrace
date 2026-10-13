"""Offscreen GUI: the interface tour (G183, owner 2026-10-12: "the interface is very feature rich and the
user is lost on how to start. A quick interactive tour ... with the option to skip it").

Checked with the user's gestures (Help -> Take the Interface Tour, keys, button clicks):
  [1] the demo (kinetrace/demo.py): made by Kinetrace, deterministic, its truth inside its silhouette,
      remade when deleted;
  [2] on an EMPTY window the tour opens the demo pre-filled (one animal, five points tracked on frames
      0-99, a silhouette on 0-99), unchanged = nothing to save, no recovery copy;
  [3] Next / Right / Enter walk every stop, Left goes back; each stop's lit area is its widget; a resize
      mid-tour moves it; Done -> the demo pop-up, and the offer is never made again;
  [4] while the tour is up, N, S, Space, Right, Ctrl+O, F1 and a click on the video do nothing;
  [5] over the user's OWN video ("take tour as is but don't change any options selected by the user"):
      frame, zoom, selection, every button / menu entry, the panel (floated), the strip (hidden), the
      window layout, the unsaved state and settings.ini are exactly as before after Esc; the hidden /
      floated stops say how to show them instead of showing them; no demo pop-up; N works again after;
  [6] the first-start offer (`tour.offer_tour`): Take / Not now / Don't ask again, and when it is due;
  [7] Help -> Take the Tour while tracking is refused (X / Space must stay live).

Run: .venv\\Scripts\\python.exe tests\\verify_tour.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QMessageBox

from _clean import forget_recovery  # noqa: E402

FAILS: list[str] = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  [{detail}]" if not ok and str(detail) else ""))
    if not ok:
        FAILS.append(what)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# every message box is recorded, never shown (an unstubbed modal blocks the test)
SHOWN: list[tuple[str, str]] = []
ANSWER = {"text": None}


def fake_exec(self):
    SHOWN.append((self.windowTitle(), self.text()))
    want = ANSWER["text"]
    for b in self.buttons():
        if want is not None and b.text().replace("&", "") == want:
            b.click()
            break
    return 0


QMessageBox.exec = fake_exec
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: SHOWN.append(("critical", str(a[1:2]))) or QMessageBox.Ok)

from kinetrace import demo, recovery, tour  # noqa: E402
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t > timeout:
            check(False, what, "timed out")
            return False
    return True


def rect_of(win, w) -> QRect:
    return QRect(w.mapTo(win, QPoint(0, 0)), w.size())


# ---------------------------------------------------------------- [1] the demo itself
print("[1] the demo video")
folder = demo.demo_folder()
import tempfile  # noqa: E402
check(os.path.commonpath([os.path.realpath(folder), os.path.realpath(tempfile.gettempdir())])
      == os.path.realpath(tempfile.gettempdir()),
      "the offscreen tests make the demo in the temp folder, not the user's Kinetrace folder", str(folder))
in_frame = True
for t in range(0, demo.N_FRAMES, 5):
    _, m = demo.truth(t)
    if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
        in_frame = False
check(in_frame, "the whole animal stays inside the picture")
vpath = demo.video_path()
if vpath.exists():
    vpath.unlink()
made = demo.ensure_video()
cap = cv2.VideoCapture(str(made))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, f0 = cap.read()
cap.release()
check(made == vpath and n == demo.N_FRAMES and ok and f0.shape[:2] == (demo.H, demo.W),
      f"ensure_video writes {demo.N_FRAMES} frames of {demo.W}x{demo.H}", (n, None if f0 is None else f0.shape))
inside = True
for t in (0, 50, 99, 149):
    pts, mask = demo.truth(t)
    check(set(pts) == set(demo.PARTS), f"frame {t}: every part has a position", sorted(pts))
    for part, (x, y) in pts.items():
        if part != "eye" and not mask[int(round(y)), int(round(x))]:
            inside = False
check(inside, "the parts sit on the animal's silhouette")
a, _ = demo.truth(17)
b, _ = demo.truth(17)
check(a == b, "the truth is deterministic")
mtime = made.stat().st_mtime
check(demo.ensure_video() == made and made.stat().st_mtime == mtime, "an existing demo is kept, not rewritten")

# ---------------------------------------------------------------- [2] the tour on an empty window
print("[2] Help -> Take the Interface Tour on an empty window opens the demo")
forget_recovery(str(made))
win = MainWindow()
win.resize(1400, 900)
win.show()
app.setActiveWindow(win)
pump(0.2)
s = win._settings()
s.remove("tour")
s.sync()
check(hasattr(win, "act_tour") and win.act_tour in win.findChildren(QAction), "Help has Take the Interface Tour")
help_menu = [m for m in win.menuBar().findChildren(type(win.menuBar().actions()[0].menu())) if m.title() == "&Help"]
check(bool(help_menu) and win.act_tour in help_menu[0].actions(), "... in the Help menu")
win.act_tour.trigger()
wait(lambda: win._tour is not None, 60, "the tour starts once the demo is open")
sess = win.session
check(win.state == READY and sess is not None and os.path.samefile(sess.video_path, made), "the demo video is open")
if sess is not None:
    check(sess.n_segments == 1 and sess.has_silhouette(0), "one animal with a silhouette", sess.n_segments)
    check(sess.n_points == len(demo.PARTS) and all(p.segment == sess.segments[0].name for p in sess.points),
          f"{len(demo.PARTS)} points, all on the animal", [p.name for p in sess.points])
    tr = sess.tracked
    check(bool(tr[:demo.N_FILLED].all()) and not tr[demo.N_FILLED:].any(),
          f"tracked on frames 0-{demo.N_FILLED - 1} only (Track has the rest to do)")
    check(sess.seg_masks[0].n_masked() == demo.N_FILLED, "silhouette on the same frames", sess.seg_masks[0].n_masked())
    pts, _ = demo.truth(42)
    k = [p.name for p in sess.points].index(sess.segments[0].name + " snout")
    check(np.allclose(sess.tracks[42, k], pts["snout"], atol=1e-3), "a point sits on its truth", sess.tracks[42, k])
    check(not win.project.dirty, "the demo opens with nothing to save")
    win._autosave()
    pump(0.3)
    check(recovery.for_video(str(made)) is None, "no recovery copy of the untouched demo")
    check(win.layers.selected_pids() == list(range(sess.n_points)) and not win.layers.selected_animals(),
          "the demo's five points come selected, its animal row not (no SAM weights needed)",
          (win.layers.selected_pids(), win.layers.selected_animals()))
    check(win._track_blocked is None and win.btn_track.isEnabled(), "Track works at the first press on the demo",
          win._track_blocked)

# ---------------------------------------------------------------- [3] walking the stops
print("[3] walking the stops")
ov = win._tour
if ov is not None:
    stops = ov.stops
    check(8 <= len(stops) <= 12, "8-12 stops", len(stops))
    check(ov.index == 0 and ov.isVisible() and ov.geometry() == win.rect(), "the overlay covers the whole window")
    seen = []
    for i in range(len(stops)):
        st = stops[ov.index]
        target = st.widget(win)
        if target is not None:
            check(ov.target_rect() == rect_of(win, target), f"stop {ov.index + 1} '{st.key}' lights its widget",
                  (ov.target_rect(), rect_of(win, target)))
        else:
            check(ov.target_rect() is None or st.key in ("menus", "help"), f"stop '{st.key}' has no hole")
        check(ov.counter.text() == f"{ov.index + 1} / {len(stops)}", "the card counts the stops", ov.counter.text())
        seen.append(st.key)
        if i == 2:
            QTest.keyClick(win, Qt.Key_Left)
            check(ov.index == 1, "Left goes back", ov.index)
            QTest.keyClick(win, Qt.Key_Right)
            check(ov.index == 2, "Right goes on", ov.index)
        if st.key == "timeline":
            win.resize(1200, 800)
            pump(0.2)
            check(ov.geometry() == win.rect() and ov.target_rect() == rect_of(win, win.timeline),
                  "a resize mid-tour moves the lit area with the timeline")
            win.resize(1400, 900)
            pump(0.2)
        if i == len(stops) - 1:
            check(ov.btn_next.text() == "Done", "the last stop says Done", ov.btn_next.text())
            SHOWN.clear()
            QTest.mouseClick(ov.btn_next, Qt.LeftButton)
        elif i % 2:
            QTest.mouseClick(ov.btn_next, Qt.LeftButton)
        else:
            QTest.keyClick(win, Qt.Key_Return)
        pump(0.05)
    for key in ("menus", "video", "timeline", "controls", "track", "layers", "help"):
        check(key in seen, f"a stop for '{key}'", seen)
    check(win._tour is None and not ov.isVisible(), "Done closes the tour")
    check(any("demo" in (t + x).lower() for t, x in SHOWN), "the demo pop-up after a demo tour", SHOWN)
    txt = " ".join(x for _, x in SHOWN)
    check("Open Video" in txt and "keep" in txt.lower(), "it says how to open your own video or keep playing")
    check(str(win._settings().value("tour/offered", "")) == "done", "a finished tour is never offered again")
    check(not tour.offer_due(win._settings()), "offer_due is False after it")

# ---------------------------------------------------------------- [4] input held back
print("[4] keys and clicks while the tour is up")
win.act_tour.trigger()
wait(lambda: win._tour is not None, 10, "the tour starts over the open demo at once")
check(win._tour is not None and os.path.samefile(win.session.video_path, made), "no second demo is made")
before = (win.current, win.btn_add.isChecked(), win.btn_animal.isChecked(), win.btn_play.isChecked())
SHOWN.clear()
for key, mod in ((Qt.Key_N, Qt.NoModifier), (Qt.Key_S, Qt.NoModifier), (Qt.Key_Space, Qt.NoModifier),
                 (Qt.Key_O, Qt.ControlModifier), (Qt.Key_F1, Qt.NoModifier)):
    QTest.keyClick(win, key, mod)
    pump(0.05)
QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, win.canvas.viewport().rect().center())
pump(0.1)
after = (win.current, win.btn_add.isChecked(), win.btn_animal.isChecked(), win.btn_play.isChecked())
check(before == after, "N, S, Space, a click on the video: nothing changed", (before, after))
manual = getattr(win, "_manual_dlg", None)
check(manual is None or not manual.isVisible(), "F1 did not open the manual")
check(not SHOWN and win.state == READY, "Ctrl+O opened nothing", SHOWN)
check(win._tour is not None and win._tour.index == 1, "Space moved the TOUR on, not the video", win._tour.index)
QTest.keyClick(win, Qt.Key_Escape)
pump(0.1)
check(win._tour is None, "Esc leaves the tour")
QTest.keyClick(win, Qt.Key_N)
pump(0.05)
check(win.btn_add.isChecked(), "after the tour, N arms the point tool again")
QTest.keyClick(win, Qt.Key_N)
pump(0.05)
win.close()
pump(0.2)
win._dev_probe.wait(15000)

# ---------------------------------------------------------------- [5] the user's own video, as is
print("[5] over the user's own video: nothing changes")
own = os.path.join(OUT, "tour_own.mp4")
vw = cv2.VideoWriter(own, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
for f in range(60):
    img = np.full((360, 640, 3), 70, np.uint8)
    cv2.circle(img, (100 + 5 * f, 180), 10, (230, 230, 230), -1)
    vw.write(img)
vw.release()
forget_recovery(own)
win = MainWindow()
win.resize(1400, 900)
win.show()
app.setActiveWindow(win)
win._open_video(own)
wait(lambda: win.state == READY and not win._loading, 30, "the user's video opens")
pump(0.3)
s = win.session
pid = s.add_landmark("knee")
win.selected = pid
win._refresh_point_list()
win._goto(37)
win.btn_follow.setChecked(True)
win.canvas.set_view_state(2.5, 300.0, 170.0, True)
win.dock.setFloating(True)
win.act_onboarding.setChecked(False)
win._settings().remove("tour")
pump(0.3)


def snapshot(w):
    st = w._settings()
    keys = {k: str(st.value(k)) for k in st.allKeys() if not k.startswith("tour")}
    return {"frame": w.current, "view": w.canvas.view_state(), "selected": w.selected,
            "actions": [(a.text(), a.isChecked(), a.isEnabled()) for a in w.findChildren(QAction)],
            "buttons": [(b.objectName() or b.text(), b.isChecked(), b.isEnabled())
                        for b in w.findChildren(QAbstractButton) if not w._tour_owns(b)],
            "dock": (w.dock.isFloating(), w.dock.isVisible()), "strip": w.onboarding.isVisible(),
            "layout": bytes(w.saveState()), "dirty": w.project.dirty, "points": s.n_points,
            "menubar": w.menuBar().isEnabled(), "settings": keys, "title": w.windowTitle()}


snap = snapshot(win)
SHOWN.clear()
win.act_tour.trigger()
wait(lambda: win._tour is not None, 10, "the tour starts at once over the user's video")
check(os.path.samefile(win.session.video_path, own), "no demo over the user's video")
check(win.menuBar().isEnabled(), "the menus are shown as they are, not greyed")
ov = win._tour
if ov is not None:
    texts = {}
    for _ in range(len(ov.stops)):
        st = ov.stops[ov.index]
        texts[st.key] = (ov.target_rect(), ov.body.text())
        if ov.index < len(ov.stops) - 1:
            QTest.keyClick(win, Qt.Key_Right)
    for key in ("layers", "cameras"):
        if key in texts:
            check(texts[key][0] is None and "View" in texts[key][1], f"floated panel: '{key}' says how to find it",
                  texts[key][1][:120])
    if "strip" in texts:
        check(texts["strip"][0] is None and "Getting started strip" in texts["strip"][1],
              "hidden strip: the stop says View -> Getting started strip", texts["strip"][1][:120])
    QTest.keyClick(win, Qt.Key_Escape)
    pump(0.2)
after = snapshot(win)
for k in snap:
    check(snap[k] == after[k], f"unchanged after the tour: {k}",
          "" if k in ("layout", "actions", "buttons") else (snap[k], after[k]))
check(not SHOWN, "no demo pop-up after a tour over the user's own video", SHOWN)
check(str(win._settings().value("tour/offered", "")) == "done", "a skipped tour is not offered again either")

# ---------------------------------------------------------------- [6] the first-start offer
print("[6] the offer")
st = win._settings()
for text, want, due in (("Not now", "later", True), ("Take the tour", "take", True),
                        ("Don't ask again", "never", False)):
    st.remove("tour")
    ANSWER["text"] = text
    SHOWN.clear()
    got = tour.offer_tour(win)
    check(got == want, f"'{text}' -> {want}", got)
    check(SHOWN and "tour" in SHOWN[0][1].lower(), "the offer explains itself", SHOWN)
    if want == "never":
        win._set_setting("tour/offered", "never")
    check(tour.offer_due(win._settings()) is due, f"after '{text}' the offer is {'still' if due else 'no longer'} due")
ANSWER["text"] = None
st.remove("tour")
check(tour.offer_tour(win) == "later", "closing the offer (Esc / x) = Not now")

# ---------------------------------------------------------------- [7] refused while tracking
print("[7] while tracking")
real = win.state
win.state = TRACKING
win.act_tour.trigger()
pump(0.1)
notes = " ".join(str(n[0]) for n in getattr(win.toast, "_notices", []))
check(win._tour is None and ("tracking" in notes.lower() or "tracking" in win.toast.text().lower()),
      "refused with a note while tracking", notes[:160])
win.state = real
win.project.dirty = False
win.close()
pump(0.2)
win._dev_probe.wait(15000)

print("\nverify_tour: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
