"""Manual mode (G215; plan dev/work/2026-10-20-silhouette-hand-mix D6, PR C). The owner: "add a full manual mode where
the user can turn off all model assist and just track points manually across frames".

Offscreen; no GPU. Track > Manual - no tracker: nothing model-based runs (no worker, no SAM, no point-model test, no
Body), a click lands exactly where it was made (no snap to the other cameras' prediction) and the playhead advances by
Track > After a click (the next frame - default -, the step box, the next point, or stay); F / B still step frames;
Ctrl+Z takes the click back AND goes back to its frame; silhouette landmarks are clicked the same way; the corrections
and small-spot hints stay silent; the mode and the after-click choice are saved with the project.
Every check fails on main (there is no Manual mode), except F / B and the silhouette-landmark click (G218, PR A): those
guard that Manual mode keeps them. Run: .venv\\Scripts\\python.exe tests\\verify_manual_mode.py
"""
import os
import shutil
import sys
import time
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "manual_mode")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
SAVE_TO = {"path": ""}
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (SAVE_TO["path"], ""))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402

from kinetrace.app import READY, MainWindow  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402
TRACK_CHOICE = stub_track_choice()     # (G217) the Track pop-up answers "keep": no press waits on it

FAILS: list = []


def check(label, cond, detail=None):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001 - the old code raises where the new one answers
            cond, detail = False, f"raised {e!r}"
    cond = bool(cond)
    line = ("  ok    " if cond else "  FAIL  ") + label + ("" if cond or detail is None else f"   [{detail}]")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not cond:
        FAILS.append(label)
    return cond


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.003)


def wait(cond, timeout):
    t0 = time.time()
    while True:
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > timeout:
            return False
        pump(0.02)


# ---- a small clip: one bright spot on a static texture
W, H, N = 320, 240, 60
_rng = np.random.RandomState(3)
BG = cv2.GaussianBlur((_rng.rand(H, W) * 90 + 40).astype(np.float32), (0, 0), 3)
VID = os.path.join(OUT, "manual.mp4")
vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
for f in range(N):
    img = BG.copy()
    cv2.circle(img, (40 + 3 * f, 120), 3, 230.0, -1)
    g = np.clip(img, 0, 255).astype(np.uint8)
    vw.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
vw.release()
forget_recovery(VID)


def new_window(path=None, project=None):
    if path:
        forget_recovery(path)          # an earlier section's recovery copy is not offered to this window
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    if project:
        win._open_project_from_path(project)
    else:
        win._open_video(path)
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    pump(0.3)
    win._goto(0, force=True)
    wait(lambda: win.cache.get(0) is not None, 10)
    win.canvas.fit()
    pump(0.1)
    return win


def close(win):
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


class Toasts:
    def __init__(self, win):
        self.msgs = []
        win.toast.show_message = lambda text, *a, **k: self.msgs.append(str(text))

    def said(self, *words):
        return any(all(w.lower() in m.lower() for w in words) for m in self.msgs)


class _NoAction:
    """The old code has no such menu entry: its checks fail one by one instead of the suite stopping at the first."""

    def trigger(self):
        pass

    def isChecked(self):
        return False

    def isEnabled(self):
        return True


def act(win, name):
    return getattr(win, name, None) or _NoAction()


def click(win, x, y, double=False, look=True):
    """A real click (or double-click) on the canvas; returns the scene position Qt gives that pixel (what must be
    stored). `look`: first wait for the frame's own picture, as a user looks before clicking."""
    if look:
        pump(0.05)
        wait(lambda: (lambda rgb: rgb is not None and win.canvas.shows(rgb) and not win.canvas.is_preview())(
            win.cache.get(win.current)), 5)
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    sp = win.canvas.mapToScene(vp)
    vpw = win.canvas.viewport()
    QTest.mouseClick(vpw, Qt.LeftButton, Qt.NoModifier, vp)
    if double:
        # what the system sends for a double-click: press, release, double-click, release (QTest.mouseDClick sends
        # the double-click event alone here)
        QTest.mouseDClick(vpw, Qt.LeftButton, Qt.NoModifier, vp)
        QTest.mouseRelease(vpw, Qt.LeftButton, Qt.NoModifier, vp)
    pump(0.08)
    return float(sp.x()), float(sp.y())


def section_main():
    win = new_window(VID)
    s = win.session
    toasts = Toasts(win)
    check("(setup) the clip opens", s is not None and s.n_frames == N)
    check("Track > Manual - no tracker exists", lambda: getattr(win, "act_mode_manual", None) is not None)
    after_off = not act(win, "act_after_step").isEnabled()
    QTest.keyClick(win, Qt.Key_S)                     # an armed Segment tool (Automatic mode)...
    pump(0.05)
    armed = win.btn_animal.isChecked()
    act(win, "act_mode_manual").trigger()
    pump(0.05)
    check("... and ticking it puts the window in Manual mode", lambda: win._track_mode == "manual")
    check("... it puts down the armed Segment tool", armed and not win.btn_animal.isChecked(), armed)
    check("After a click is offered only in Manual mode (greyed in Automatic)",
          after_off and act(win, "act_after_step").isEnabled(), after_off)
    # N + a click adds the point, the next 9 clicks digitise it frame by frame
    launched = []
    orig_launch, orig_ready = win._launch_run, win._tracking_engine_ready
    win._launch_run = lambda *a, **k: (launched.append(1), orig_launch(*a, **k))[1]
    ready_calls = []
    win._tracking_engine_ready = lambda: (ready_calls.append(1), orig_ready())[1]
    snaps = []
    orig_snap = win._snap_to_prediction
    win._snap_to_prediction = lambda pid, x, y: (snaps.append(1), orig_snap(pid, x, y))[1]
    win.btn_add.setChecked(True)
    xy0 = click(win, 40.5, 120.3)
    pid = s.n_points - 1
    check("(setup) N + a click added the point on frame 0", s.n_points == 1 and s.tracked[0, pid])
    check("the small-spot hint stays silent in Manual mode (the point sits on a 6 px dot)",
          not toasts.said("moving spot"), toasts.msgs[-1:])
    check("adding a point (N + a click) keeps the video on its frame (owner 2026-10-23: set up several there)",
          win.current == 0, win.current)
    QTest.keyClick(win, Qt.Key_F)
    pump(0.1)
    placed = {0: xy0, 1: click(win, 43.3, 119.8)}
    check("a click advances the playhead to the next frame (After a click: the next frame, the default)",
          win.current == 2, win.current)
    for f in range(2, 10):
        placed[f] = click(win, 40.0 + 3 * f + 0.3, 120.0 - 0.2)
    check("ten clicks: ten hand-placed frames, each exactly where it was clicked",
          lambda: all(s.manual[f, pid] and np.allclose(s.tracks[f, pid], placed[f], atol=1e-4) for f in placed),
          [(f, tuple(np.round(s.tracks[f, pid], 3)), tuple(np.round(placed[f], 3))) for f in (1, 9)])
    check("... the playhead followed them (frame 10)", win.current == 10, win.current)
    check("... and no click was snapped to a prediction", not snaps, len(snaps))
    # Ctrl+Z takes the last click back and goes back to its frame
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.session
    check("Ctrl+Z takes the last click back and goes back to its frame (9)",
          lambda: not s.tracked[9, pid] and win.current == 9, (bool(s.tracked[9, pid]), win.current))
    # F / B step frames
    f_before = win.current
    QTest.keyClick(win, Qt.Key_F)
    pump(0.1)
    check("F steps one frame on", win.current == f_before + 1, (f_before, win.current))
    QTest.keyClick(win, Qt.Key_B)
    pump(0.1)
    check("B steps one frame back", win.current == f_before, (f_before, win.current))
    # a double-click is ONE click: its second press would land on the next frame, unseen
    win._goto(12, force=True)
    click(win, 76.0, 120.0, double=True)
    check("a double-click places ONE frame and moves on once (12 placed, 13 not, the playhead on 13)",
          lambda: s.manual[12, pid] and not s.tracked[13, pid] and win.current == 13,
          (bool(s.manual[12, pid]), bool(s.tracked[13, pid]), win.current))
    check("... and the bottom bar says the double-click counted once",
          "double-click" in win.statusBar().currentMessage().lower(), win.statusBar().currentMessage()[:80])
    # a click on a stand-in picture (the nearest cached frame, dimmed, while this one decodes: G204) is not placed
    wait(lambda: not win.canvas.is_preview(), 5)
    win.canvas.set_preview(True)
    click(win, 79.0, 120.0, look=False)
    check("a click while the frame's picture is still loading is not placed, and the video stays",
          lambda: not s.tracked[13, pid] and win.current == 13, (bool(s.tracked[13, pid]), win.current))
    check("... and the bottom bar says why", "loading" in win.statusBar().currentMessage().lower(),
          win.statusBar().currentMessage()[:80])
    win.canvas.set_preview(False)
    # T / the Track button track nothing and say why
    toasts.msgs.clear()
    QTest.keyClick(win, Qt.Key_T)
    pump(0.3)
    QTest.keyClick(win, Qt.Key_T, Qt.ShiftModifier)
    pump(0.3)
    win.btn_track.click()
    pump(0.3)
    check("T, Shift+T and the Track button start no run in Manual mode", not launched and win.state == READY)
    check("... the model is never even prepared", not ready_calls)
    check("... and they say so (Manual mode, and how to track again)",
          toasts.said("manual") and toasts.said("automatic"), toasts.msgs[-2:])
    # S, the point-model test and Body refuse in words
    toasts.msgs.clear()
    QTest.keyClick(win, Qt.Key_S)
    pump(0.1)
    check("S does not start the segment tool in Manual mode, and says so",
          lambda: not win.btn_animal.isChecked() and toasts.said("manual"), toasts.msgs[-1:])
    toasts.msgs.clear()
    win.act_test_models.trigger()
    pump(0.2)
    check("Test the point models refuses in Manual mode, in words", toasts.said("manual"), toasts.msgs[-1:])
    toasts.msgs.clear()
    win.act_body_run.trigger()
    pump(0.2)
    check("Body refuses in Manual mode, in words", toasts.said("manual"), toasts.msgs[-1:])
    toasts.msgs.clear()
    win.act_add_ball.trigger()
    pump(0.1)
    check("Point > Ball marker refuses in Manual mode (SAM outlines a ball), in words",
          lambda: not win.btn_add.isChecked() and toasts.said("manual"), toasts.msgs[-1:])
    # After a click: the step box, the next point, stay
    win.step_spin.setValue(5)
    act(win, "act_after_step").trigger()
    win._goto(20, force=True)
    pump(0.1)
    click(win, 100.0, 120.0)
    check("After a click: the step box (5) -> the playhead moves 5 frames", win.current == 25, win.current)
    act(win, "act_after_stay").trigger()
    f_stay = win.current
    click(win, 115.0, 120.0)
    check("After a click: stay -> the playhead stays (a guard: main stays too)", win.current == f_stay,
          (f_stay, win.current))
    # a second point: the next point, then the next frame
    act(win, "act_after_next_frame").trigger()
    win._goto(30, force=True)
    win.layers.clearSelection()
    pump(0.05)
    win.selected = None
    win.btn_add.setChecked(True)
    click(win, 130.0, 100.0)
    q = s.n_points - 1
    act(win, "act_after_point").trigger()
    win._goto(31, force=True)
    win._on_select(pid)
    pump(0.05)
    click(win, 133.0, 120.0)
    check("After a click: the next point -> the next point is selected, the frame stays",
          win.selected == q and win.current == 31, (win.selected, win.current))
    click(win, 133.0, 100.0)
    check("... and after the last point: the first point again, on the next frame",
          win.selected == pid and win.current == 32, (win.selected, win.current))
    # the next point skips a hidden row (its LAYERS checkbox off) and a ball marker
    act(win, "act_after_stay").trigger()
    win.layers.clearSelection()
    win.selected = None
    win.btn_add.setChecked(True)
    click(win, 140.0, 140.0)
    r = s.n_points - 1
    s.add_ball(32, 170.0, 60.0)                        # (setup: no SAM runs to add one)
    s.points[q].display = False
    win._refresh_point_list()
    act(win, "act_after_point").trigger()
    win._on_select(pid)
    pump(0.05)
    click(win, 136.0, 120.0)
    check("the next point skips a hidden row: P1 -> P3 (P2 hidden), the frame stays",
          win.selected == r and win.current == 32, (win.selected, win.current))
    click(win, 140.0, 141.0)
    check("... and a ball marker: P3 -> P1 on the next frame", win.selected == pid and win.current == 33,
          (win.selected, win.current))
    s.points[q].display = True
    win._refresh_point_list()
    # hints stay silent: no "try Moving spot" for frame-by-frame corrections
    toasts.msgs.clear()
    act(win, "act_after_next_frame").trigger()
    win._goto(40, force=True)
    win._on_select(pid)
    for f in range(40, 46):
        s.tracks[f, pid] = (160.0, 120.0)
        s.tracked[f, pid] = s.visibility[f, pid] = True
        s.manual[f, pid] = False
    for f in range(40, 46):
        win._goto(f, force=True)          # each correction on its own frame (main: the hint fires at the 5th)
        click(win, 160.0 + 3 * (f - 40), 121.0)
    check("the corrections hint stays silent in Manual mode", not toasts.said("moving spot"), toasts.msgs[-1:])
    # silhouette landmarks are clicked the same way
    s.apply_skeleton({"name": "d", "landmarks": ["snoot", "tailtip"], "bones": [], "head": "snoot",
                      "derived": {"tailtip": "tip"}})
    win._refresh_point_list()
    dpid = next(k for k in range(s.n_points) if s.part_name(k) == "tailtip")
    win._goto(50, force=True)
    win._on_select(dpid)
    click(win, 200.0, 130.0)
    check("a silhouette landmark is clicked by hand the same way", lambda: s.manual[50, dpid] and s.tracked[50, dpid])
    # the mode and the after-click choice come back with the project
    act(win, "act_after_step").trigger()
    target = os.path.join(OUT, "manual.kinetrace")
    SAVE_TO["path"] = target
    saved = win._save_project_as()
    close(win)
    win = new_window(project=target)
    check("Manual mode and After a click come back with the project (save, close, open)",
          lambda: saved and win._track_mode == "manual" and act(win, "act_mode_manual").isChecked()
          and act(win, "act_after_step").isChecked())
    win.act_mode_auto.trigger()
    close(win)


def section_setup():
    """Setting up points on one frame (review R1-C1), the continue rule's warning, Ctrl+Z after the next point, the
    last frame, and the previous picture left up without a veil (review R1-I2)."""
    win = new_window(VID)
    s = win.session
    act(win, "act_mode_manual").trigger()
    pump(0.05)
    win.btn_add.setChecked(True)
    click(win, 80.0, 100.0)
    win.btn_add.setChecked(True)
    click(win, 200.0, 140.0)
    check("N + click, N + click on one frame: two points, both on frame 0, the video stays (review R1-C1)",
          lambda: s.n_points == 2 and s.tracked[0, 0] and s.tracked[0, 1] and win.current == 0
          and not s.tracked[1].any(), (s.n_points, win.current))
    # the continue rule (I69, kept by the owner 2026-10-23): N + click where the selected point has no position
    QTest.keyClick(win, Qt.Key_F)
    pump(0.1)
    sel = win.selected
    win.btn_add.setChecked(True)
    click(win, 60.0, 60.0)
    check("N + click where the selected point has no position continues it, and says so (no new point)",
          lambda: s.n_points == 2 and s.manual[1, sel] and "no new point" in win.statusBar().currentMessage(),
          win.statusBar().currentMessage()[:80])
    # Ctrl+Z after the next point: the same frame, the clicked point selected again
    act(win, "act_after_point").trigger()
    win._goto(5, force=True)
    win._on_select(0)
    click(win, 82.0, 100.0)
    moved_to = win.selected
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.session
    check("Ctrl+Z after the next point: the click is gone, P1 is selected again, on the same frame",
          lambda: moved_to == 1 and not s.tracked[5, 0] and win.selected == 0 and win.current == 5,
          (moved_to, bool(s.tracked[5, 0]), win.selected, win.current))
    # the last frame: nowhere to go
    act(win, "act_after_next_frame").trigger()
    win._goto(N - 1, force=True)
    win._on_select(0)
    click(win, 120.0, 100.0)
    check("on the last frame a click is placed and the video stays",
          lambda: s.manual[N - 1, 0] and win.current == N - 1, win.current)
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.2)
    s = win.session
    check("... and Ctrl+Z takes it back there", lambda: not s.tracked[N - 1, 0] and win.current == N - 1,
          win.current)
    # a quick move (G24) shows no stand-in and no veil: the previous picture stays up until the frame decodes
    win._goto(10, force=True)
    win._on_select(0)
    click(win, 90.0, 100.0)                       # placed on 10, the video on 11
    wait(lambda: win.cache.get(11) is not None and win.canvas.shows(win.cache.get(11)), 5)
    svc = win.seek
    svc.request = lambda *a, **k: None            # a decoder that has not answered yet
    win.cache.clear()
    win._last_goto_t = time.monotonic()
    QTest.keyClick(win, Qt.Key_F)                 # quick: frame 11's picture stays up, unveiled
    pump(0.05)
    old_up = win.current == 12 and not win.canvas.is_preview()
    click(win, 93.0, 100.0, look=False)
    check("a click while the previous picture is still up (a quick move, no veil) is not placed (review R1-I2)",
          lambda: old_up and not s.tracked[12, 0] and win.current == 12,
          (old_up, bool(s.tracked[12, 0]), win.current))
    del svc.request
    win._goto(12, force=True)
    click(win, 93.0, 100.0)
    check("... and once frame 12's picture shows, the click is placed", lambda: s.manual[12, 0])
    close(win)


def section_two_cameras():
    """(review R1-I1, owner 2026-10-23) several cameras on screen: each camera's click is the same instant, so the
    video stays; both cameras get the point on the same frame, through the real tile clicks."""
    from PySide6.QtWidgets import QDialogButtonBox
    from kinetrace import cameraorder
    vid_b = os.path.join(OUT, "manual_b.mp4")
    shutil.copyfile(VID, vid_b)
    forget_recovery(vid_b)
    cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(),
                                                       self.result())[1]
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([vid_b], ""))
    win = new_window(VID)
    win._add_video_dialog()
    wait(lambda: win.project.n_views == 2 and win.state == READY and not win._loading, 30)
    pump(0.5)
    check("(setup) two cameras on screen", lambda: win.project.n_views == 2 and len(win.grid.visible_indices()) == 2)
    act(win, "act_mode_manual").trigger()
    win._goto(5, force=True)
    win.btn_add.setChecked(True)
    click(win, 70.0, 120.0)
    QTest.keyClick(win, Qt.Key_F)
    pump(0.2)
    f = win.current
    s0 = win.project.sessions[0]
    click(win, 72.0, 120.0)
    check("camera 1's click: placed, and the video stays (several cameras on screen)",
          lambda: win.project.active == 0 and s0.manual[f, 0] and win.current == f, (f, win.current))
    tile = win.grid.canvas(1)
    QTest.mouseClick(tile.viewport(), Qt.LeftButton, Qt.NoModifier, tile.mapFromScene(QPointF(72.0, 120.0)))
    pump(0.3)
    check("(setup) the tile click made camera 2 the working camera", win.project.active == 1, win.project.active)
    click(win, 73.0, 121.0)
    s1 = win.project.sessions[1]
    j = s1.pid_by_name(s0.points[0].name)
    both = [g for g in range(N) if j is not None and s0.tracked[g, 0] and s1.tracked[g, j]]
    check("camera 2's click lands on the SAME frame: both cameras have the point there (3D can be made)",
          lambda: j is not None and s1.manual[f, j] and f in both and win.current == f, (f, both, win.current))
    close(win)


for section in (section_main, section_setup, section_two_cameras):
    try:
        section()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({section.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_manual_mode: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
