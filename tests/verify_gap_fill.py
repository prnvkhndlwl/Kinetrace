"""Only fill the gaps (D9 choice 3, G217; PR B of dev/work/2026-10-20-silhouette-hand-mix, plan step 19). GPU (CoTracker3).

The owner: "only run on gaps where there no points tracked"; 2026-10-23: stretches cut by Auto-pause or deleted by the
user are filled too, with Auto-pause on inside each gap. Nothing that exists is touched: each empty stretch after the
start is tracked from the frame before it, only when that frame is a usable position.
  [1] the Track pop-up offers it, with the number of stretches and frames
  [2] three deleted stretches of one point are filled, every cell that existed is exactly as before
  [3] a stretch after a frame marked hidden, and one after the point left the picture, are skipped - and named
  [4] the frames missing between two clicks (one stretch of clicks, D3) are left to Fill gaps
  [5] a dot that is gone for good: its gap is cut again where the tracker loses it, not guessed
  [6] 50 short gaps over a 5000-frame clip: only the frames around them are decoded
  [7] "Use this every time": the next press fills the gaps without asking; Track > When there is data ahead shows it
  [8] the first press with data ahead counts the gaps without loading the tracker (torch) on the window's thread
      (review #2: it froze the window with no "Starting the tracking engine" card, G46) - run first, in a fresh process
  [9] (I368, I376; sweep PR 7) between two gap runs the press is not over: Ctrl+S is kept for its end (saved once,
      after the last gap run), and the press's undo point takes no extra snapshot - FAILS on eaee38b
Every check FAILS on the code before step 19 (there is no third choice). Run: .venv\\Scripts\\python.exe
tests\\verify_gap_fill.py
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
OUT = os.path.join(HERE, "out", "gap_fill")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402

import kinetrace.video_source as vs  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.session import POINT_ARRAYS  # noqa: E402
from kinetrace.trackchoice import TrackChoiceDialog  # noqa: E402

REAL_ASK = MainWindow._ask_track_choice           # the window's own question, before the stub replaces it
TRACK_CHOICE = stub_track_choice("gaps")
STUB_ASK = MainWindow._ask_track_choice
FAILS: list = []


def check(label, cond, detail=None):
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001
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


class Toasts:
    def __init__(self, win):
        self.msgs = []
        real = win.toast.show_message
        win.toast.show_message = lambda text, *a, **k: (self.msgs.append(str(text)), real(text, *a, **k))[1]

    def said(self, *words):
        return any(all(w.lower() in m.lower() for w in words) for m in self.msgs)


def new_window(video):
    forget_recovery(video)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(video)
    wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    pump(0.3)
    return win


def close(win):
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


def busy(win):
    return win.state != READY or win._passes is not None or getattr(win, "_gaps", None) is not None


def press_t(win, start=0, timeout=900):
    win.act_select_all.trigger()
    win._goto(start, force=True)
    pump(0.05)
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_T)
    wait(lambda: busy(win), 15)
    return wait(lambda: not busy(win) and win.worker is None, timeout)


def seed_track(s, GT, n, pids):
    """Every frame of `pids` tracked at the ground truth (the data a run would have left)."""
    for k in pids:
        s.tracks[:n, k] = GT[:n, k]
        s.tracked[:n, k] = s.visibility[:n, k] = True
        s.confidence[:n, k] = 0.9
        s.manual[:n, k] = False
    s.manual[0, pids] = True


V600 = os.path.join(ROOT, "test600.mp4")
GT = np.load(V600 + ".gt.npz")["gt"]
GAPS = [(100, 129), (300, 309), (500, 519)]


def section_plan():
    print("[0] the plans, no GPU: a stretch marked hidden inside; clicks on the start frame and the next")
    from kinetrace.session import TrackingSession
    s = TrackingSession(V600, 600, 30.0, 640, 480)
    q = s.add_point(0, float(GT[0, 0, 0]), float(GT[0, 0, 1]))
    seed_track(s, GT, 200, [q])
    s.clear_window([q], 80, 94)
    s.set_occluded(85, q, True)                          # a frame of the empty stretch itself marked hidden
    fill, skip = s.gap_plan(q, 0, 200)
    check("[0] a stretch with a frame of its own marked hidden is skipped and named, not filled (review R1-I2)",
          not any(a == 80 for _b, a, _g, _xy in fill) and any(a == 80 and why == "marked hidden" for a, _g, why in skip),
          (fill, skip))
    s2 = TrackingSession(V600, 600, 30.0, 640, 480)
    p = s2.add_point(10, float(GT[10, 0, 0]), float(GT[10, 0, 1]))
    s2.set_position(11, p, float(GT[11, 0, 0]), float(GT[11, 0, 1]))
    own_last, restarts = s2.hand_plan(p, 10, 600)
    check("[0] a point model clicked on the start frame and the next one restarts at the next one (review R1-M1)",
          [r[0] for r in restarts] == [11] and own_last == 9, (own_last, restarts))


def section_main():
    print("[1]-[4], [7] test600: three points, gaps of every kind")
    win = new_window(V600)
    s = win.session
    for k in range(3):
        win._on_add(float(GT[0, k, 0]), float(GT[0, k, 1]))
        s.points[k].tracker = "cotracker3"
    win._refresh_point_list()
    seed_track(s, GT, 600, [0, 1, 2])
    for g0, g1 in GAPS:                                  # [2] deleted stretches of P1
        s.clear_window([0], g0, g1)
    s.set_occluded(199, 1, True)                         # [3] P2: the frame before its gap marked hidden
    s.clear_window([1], 200, 239)
    s.tracks[399, 2] = (2.0, 240.0)                      # [3] P3: at the picture's edge before its gap
    s.clear_window([2], 400, 420)
    s.clear_window([2], 250, 259)                        # [3] P3: an empty stretch with a frame of its own hidden
    s.set_occluded(254, 2, True)
    s.set_position(450, 0, float(GT[450, 0, 0]), float(GT[450, 0, 1]))     # [4] two clicks, two frames between
    s.set_position(453, 0, float(GT[453, 0, 0]), float(GT[453, 0, 1]))
    s.clear_window([0], 451, 452)
    win._refresh_point_list()
    pre = s.snapshot()
    toasts = Toasts(win)
    n_ask = len(TRACK_CHOICE.infos)
    ok = press_t(win)
    s = win.session
    check("(setup) the press ended", ok)
    info = TRACK_CHOICE.infos[-1] if len(TRACK_CHOICE.infos) > n_ask else {}
    check("[1] the Track pop-up offered 'Only fill the gaps' with the stretches and frames it would fill",
          info.get("gaps") == 3 and info.get("gap_frames") == sum(b - a + 1 for a, b in GAPS),
          (info.get("gaps"), info.get("gap_frames")))
    filled = all(s.tracked[a:b + 1, 0].all() for a, b in GAPS)
    check("[2] the three deleted stretches of P1 are filled", filled,
          [int(s.tracked[a:b + 1, 0].sum()) for a, b in GAPS])
    err = np.concatenate([np.linalg.norm(s.tracks[a:b + 1, 0] - GT[a:b + 1, 0], axis=1) for a, b in GAPS])
    check(f"[2] ... on the dot (median {np.nanmedian(err):.2f} px)", np.isfinite(err).all() and np.median(err) < 1.5)
    gap_cells = np.zeros(pre.tracked.shape, bool)
    for a, b in GAPS:
        gap_cells[a:b + 1, 0] = True
    same = []
    for arr in POINT_ARRAYS:
        old, new = getattr(pre, arr.name), getattr(s, arr.name)
        keep = ~gap_cells if old.ndim == 2 else ~gap_cells[..., None].repeat(old.shape[2], axis=2)
        same.append(np.array_equal(old[keep], new[keep], equal_nan=True))
    check("[2] every cell that existed before is exactly as it was (every point array)", all(same),
          [a.name for a, ok_ in zip(POINT_ARRAYS, same) if not ok_])
    check("[3] the stretch after a frame marked hidden is left empty", not s.tracked[200:240, 1].any())
    check("[3] the stretch after the point left the picture is left empty", not s.tracked[400:421, 2].any())
    check("[3] an empty stretch with a frame of its own marked hidden is left empty (review R1-I2)",
          not s.tracked[250:260, 2].any())
    check("[3] ... and the end says which were skipped and why",
          toasts.said("hidden") and toasts.said("left the picture"), toasts.msgs[-1:])
    check("[4] the two frames missing between two clicks are left (Fill gaps interpolates them)",
          not s.tracked[451:453, 0].any() and s.manual[450, 0] and s.manual[453, 0])
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)
    s = win.session
    check("[2] one Ctrl+Z takes every filled stretch back",
          lambda: not any(s.tracked[a:b + 1, 0].any() for a, b in GAPS))
    # [7] remembered: the pop-up's choice with "Use this every time" (the real dialog, answered by its own buttons)
    TrackChoiceDialog.exec = lambda self: (self.pick("gaps"), self.remember_box.setChecked(True), 1)[2]
    MainWindow._ask_track_choice = REAL_ASK
    pre7 = win.session.snapshot()
    press_t(win)
    s = win.session
    check("[7] picked with 'Use this every time', the gaps are filled and nothing else is touched",
          lambda: s.tracked[100:130, 0].all() and np.array_equal(s.tracks[:, 1:], pre7.tracks[:, 1:], equal_nan=True))
    check("[7] ... and Track > When there is data ahead shows 'only fill the gaps'",
          lambda: win.act_ahead_gaps.isChecked() and win._ahead_choice == "gaps")
    asked = []
    TrackChoiceDialog.exec = lambda self: (asked.append(1), 0)[1]
    for a, b in GAPS[1:]:
        s.clear_window([0], a, b)
    press_t(win)
    s = win.session
    check("[7] the next press fills the gaps without asking", lambda: not asked and s.tracked[300:310, 0].all())
    MainWindow._ask_track_choice = STUB_ASK
    win._set_ahead_choice("")
    close(win)


# ---- the vanishing dot: gone for good at 60
W, H, T1, GONE = 640, 480, 220, 60
VAN = os.path.join(OUT, "gone.mp4")
AX = 200 + 60 * np.sin(np.arange(T1) / 40)
AY = 200 + 40 * np.cos(np.arange(T1) / 55)
BX = 460 + 50 * np.sin(np.arange(T1) / 47)
BY = 320 + 45 * np.cos(np.arange(T1) / 38)


def _bg(seed):
    r = np.random.default_rng(seed)
    return cv2.GaussianBlur(r.integers(30, 90, size=(H, W, 3), dtype=np.uint8), (0, 0), 1.5)


_a, _b = _bg(3), _bg(77)
_vw = cv2.VideoWriter(VAN, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for _f in range(T1):
    _fr = (_a if _f < GONE else _b).copy()
    if _f < GONE:
        cv2.circle(_fr, (round(AX[_f]), round(AY[_f])), 9, (60, 60, 230), -1, lineType=cv2.LINE_AA)
    cv2.circle(_fr, (round(BX[_f]), round(BY[_f])), 9, (60, 230, 60), -1, lineType=cv2.LINE_AA)
    _vw.write(_fr)
_vw.release()


def section_gone():
    print("[5] a dot gone for good: its gap is cut again where it is lost")
    win = new_window(VAN)
    s = win.session
    win._on_add(float(AX[0]), float(AY[0]))
    win._on_add(float(BX[0]), float(BY[0]))
    for k in range(2):
        s.points[k].tracker = "cotracker3"
    win._refresh_point_list()
    for f in range(T1):
        s.tracks[f, 1] = (BX[f], BY[f])
        s.tracked[f, 1] = s.visibility[f, 1] = True
    for f in range(GONE):
        s.tracks[f, 0] = (AX[f], AY[f])
        s.tracked[f, 0] = s.visibility[f, 0] = True
    pre = s.snapshot()
    ok = press_t(win)
    s = win.session
    check("(setup) the press ended", ok)
    last = int(np.flatnonzero(s.tracked[:, 0])[-1])
    check(f"[5] dot A's gap was not guessed to the end: cut where the tracker reported it lost (data to {last}; a "
          "confident parked guess before that is the known limit of any run)", GONE <= last < T1 - 60, last)
    check("[5] dot B (no gap) is exactly as it was", lambda: np.array_equal(s.tracks[:, 1], pre.tracks[:, 1]))
    win.activateWindow()
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump(0.3)
    win.btn_autopause.setChecked(False)                  # the toolbar's switch off: inside a gap it is on anyway
    ok = press_t(win)
    s = win.session
    last = int(np.flatnonzero(s.tracked[:, 0])[-1])
    check(f"[5] with Auto-pause switched off the gap is still cut where the dot is lost (data to {last}; review R1-I1)",
          ok and GONE <= last < T1 - 60, last)
    win.btn_autopause.setChecked(True)
    close(win)


def section_long():
    print("[6] 50 short gaps over a 5000-frame clip: only the frames around them are decoded")
    v5 = os.path.join(ROOT, "test5000.mp4")
    if not os.path.isfile(v5):
        check("(setup) test5000.mp4 exists (tests/make_test_video.py)", False)
        return
    G5 = np.load(v5 + ".gt.npz")["gt"]
    win = new_window(v5)
    s = win.session
    win._on_add(float(G5[0, 1, 0]), float(G5[0, 1, 1]))         # dot 1: never near the picture's edge
    s.points[0].tracker = "cotracker3"
    win._refresh_point_list()
    n = s.n_frames
    s.tracks[:n, 0] = G5[:n, 1]
    s.tracked[:n, 0] = s.visibility[:n, 0] = True
    s.confidence[:n, 0] = 0.9
    gaps = [(100 * i + 40, 100 * i + 42) for i in range(50)]
    for a, b in gaps:
        s.clear_window([0], a, b)
    decoded = []
    real = vs.VideoSource.read_next

    def counting(self):
        out = real(self)
        if out is not None:
            decoded.append(out[0])
        return out
    vs.VideoSource.read_next = counting
    try:
        ok = press_t(win, timeout=1200)
    finally:
        vs.VideoSource.read_next = real
    s = win.session
    check("(setup) the press ended", ok)
    check("[6] all 50 gaps are filled", lambda: all(s.tracked[a:b + 1, 0].all() for a, b in gaps))
    check(f"[6] only the frames around them were decoded ({len(set(decoded))} of {n})", len(set(decoded)) < n / 3)
    close(win)


def section_torch():
    print("[8] the first press with data ahead: the gap counts never load the tracker (torch) on the window's thread")
    if "kinetrace.tracker" in sys.modules:
        check("(setup) nothing has loaded the tracker before this section", False)
        return
    win = new_window(V600)
    s = win.session
    win._on_add(float(GT[0, 0, 0]), float(GT[0, 0, 1]))
    seed_track(s, GT, 200, [0])
    s.clear_window([0], 100, 129)
    seen = []

    def ask(self, info):
        seen.append(("kinetrace.tracker" in sys.modules, int(info.get("gaps") or 0)))
        return None                                   # cancelled: nothing runs
    MainWindow._ask_track_choice = ask
    try:
        win.act_select_all.trigger()
        win._goto(0, force=True)
        pump(0.05)
        win.activateWindow()
        QTest.keyClick(win, Qt.Key_T)
        pump(0.3)
    finally:
        MainWindow._ask_track_choice = STUB_ASK
    check("(setup) the press asked, and its counts hold the gap", seen and seen[0][1] >= 1, seen)
    check("[8] ... and the tracker was not loaded on the window's thread before it asked (review #2, G46: the "
          "engine card loads it)", seen and not seen[0][0], seen)
    # answered "Only fill the gaps": the gap runs too load the tracker only after the engine card (re-check of
    # 441467c: `_gaps_next` imported it first). The card is refused here, so nothing runs.
    ready = []
    win._tracking_engine_ready = lambda: (ready.append("kinetrace.tracker" in sys.modules), False)[1]
    MainWindow._ask_track_choice = lambda self, info: "gaps"
    try:
        QTest.keyClick(win, Qt.Key_T)
        pump(0.5)
    finally:
        MainWindow._ask_track_choice = STUB_ASK
        del win._tracking_engine_ready
    check("[8] answered Only fill the gaps: the tracker was not loaded before the engine card", ready and not any(ready),
          ready)
    close(win)


def section_press():
    print("[9] between two gap runs the press is not over (I368): Ctrl+S waits for its end; one undo point, no "
          "extra snapshot (I376)")
    from kinetrace.session import TrackingSession
    from PySide6.QtWidgets import QFileDialog
    win = new_window(V600)
    s = win.session
    win._on_add(float(GT[0, 0, 0]), float(GT[0, 0, 1]))
    s.points[0].tracker = "cotracker3"
    win._refresh_point_list()
    seed_track(s, GT, 600, [0])
    s.clear_window([0], 100, 129)                        # two gaps far apart: two gap runs
    s.clear_window([0], 400, 409)
    win._refresh_point_list()
    target = os.path.join(OUT, "press9.kinetrace")
    real_dlg = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
    try:
        check("(setup) the project is saved, so Ctrl+S writes it with no question", win._save_project_as())
    finally:
        QFileDialog.getSaveFileName = real_dlg
    wait(lambda: not win._writer_busy(), 30)
    s = win.session
    s.clear_window([0], 100, 129)                        # (Save As kept them; dirty again for the press)
    s.set_position(0, 0, float(GT[0, 0, 0]), float(GT[0, 0, 1]))
    toasts = Toasts(win)
    writes, between, snaps = [], [], []
    real_writer, real_next, real_undo = win._start_writer, win._gaps_next, win._set_run_undo_point

    def writer(frozen, path, *a, **k):
        if str(path).lower().startswith(str(target).lower()):
            writes.append({"gaps_over": win._gaps is None, "state": win.state})
        return real_writer(frozen, path, *a, **k)

    def gaps_next():
        st = win._gaps
        if st is not None and st["runs"] >= 1 and st["queue"] and win.state == READY and not between:
            between.append(win._run_in_progress())
            win.activateWindow()
            QTest.keyClick(win, Qt.Key_S, Qt.ControlModifier)          # the user's Ctrl+S between two gap runs
        return real_next()

    def undo_point(sn):
        n0 = len(snaps)
        real_snap = TrackingSession.snapshot
        TrackingSession.snapshot = lambda self_: (snaps.append(1), real_snap(self_))[1]
        try:
            return real_undo(sn)
        finally:
            TrackingSession.snapshot = real_snap
            snaps.append(("calls", len(snaps) - n0))
    win._start_writer, win._gaps_next, win._set_run_undo_point = writer, gaps_next, undo_point
    try:
        ok = press_t(win)
        wait(lambda: not win._save_after_run and not win._writer_busy(), 30)
        pump(0.5)
    finally:
        del win._start_writer, win._gaps_next, win._set_run_undo_point
    s = win.session
    check("(setup) the press ended and both gaps are filled",
          ok and s.tracked[100:130, 0].all() and s.tracked[400:410, 0].all())
    check("(setup) Ctrl+S was pressed between the two gap runs", between, between)
    check("[9] between two gap runs the press counts as not over (`_run_in_progress`)", between and between[0], between)
    check("[9] ... so Ctrl+S says it saves once the run stops", toasts.said("saved as soon as it stops"), toasts.msgs[-3:])
    check("[9] ... and the project is written once, after the last gap run", len(writes) == 1 and writes[0]["gaps_over"],
          writes)
    calls = [c[1] for c in snaps if isinstance(c, tuple)]
    check("[9] the press's one undo point takes no extra snapshot of the working camera (I376)", calls == [0], calls)
    close(win)


ONLY = {x.strip() for x in os.environ.get("GAP_FILL_ONLY", "").split(",") if x.strip()}    # e.g. "torch,plan"
for fn in (section_torch, section_plan, section_main, section_gone, section_long, section_press):
    if ONLY and fn.__name__.replace("section_", "") not in ONLY:
        continue
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_gap_fill: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
