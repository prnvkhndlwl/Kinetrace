"""(I378, sweep FS-F1-1) The point models' map between the native picture and the shrunk picture they track on.

The worker shrinks a 4K frame for the point model with cv2 INTER_AREA (pixel-centre rule: small p is native
(p + 0.5) * s - 0.5). The model's answers must come back with the same rule. Before I378 they came back with the
align-corners scale (W - 1) / (w - 1): a point that moved D px from its click was reported D * (k / s - 1) off,
about 2 px for a point crossing a 4K picture at AllTracker's 1024 x 576.

CPU only, no model: a stand-in "perfect tracker" replaces AllTracker (`get_alltracker` / `AllTrackerStream`, the
same start / push / flush contract). It answers query + (the object's centroid now - its centroid at the start),
measured in the SMALL picture it is fed, so every error left is the worker's own native <-> small map.
KINETRACE_ALLTRACKER_MAX_DIM=1024 fixes the small size on any machine.

The clip (tests/out/point_model_scale/far4k.avi, lossless, 3840 x 2160, 48 frames, made here):
  D  a large flat disc (radius 340) moving (500.3, 400.7) -> (3300.3, 1700.7): flat inside, so the native LK
     refinement cannot grip it and the model's own answer is what is stored (as on a smooth, flat body patch)
  G  a small gaussian dot moving (3400.3, 300.4) -> (600.3, 300.4): LK grips it
  S  a still gaussian dot at (2000.6, 1900.2): the control (a point that does not move is exact either way)
  [a] bare TrackingWorker, LK off, ROI off: D and G < 0.1 px on the last frame (main: D (+2.01, +1.66), G -2.0 px)
  [b] the same with LK on: G < 0.1 px (LK grips it on main too: the fix must not make it worse); D as [a]
  [c] the user's gesture: Point > AllTracker, N + a click on each object on frame 0, T. The app's own settings
      (LK on, ROI on: three points across the picture, so the crop is declined). D's last frame = its click +
      the disc's move, < 0.1 px (main ~2.6 px)
  [d] a picture that is not shrunk (1024 x 576): the worker's tracks ARE the stand-in's answers, bit for bit
[a] and [c] FAIL on main 48ca2fc. Run: .venv\\Scripts\\python.exe tests\\verify_point_model_scale.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["KINETRACE_ALLTRACKER_MAX_DIM"] = "1024"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "point_model_scale")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from _clean import forget_recovery  # noqa: E402

import kinetrace.app as appmod  # noqa: E402
from kinetrace import alltracker_backend as at_backend  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.tracker import TrackingWorker  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

FAILS: list = []
if hasattr(appmod, "TrackChoiceDialog"):
    def _pop_exec(self):
        self.pick("fresh")
        return QDialog.Accepted
    appmod.TrackChoiceDialog.exec = _pop_exec


def check(label, cond, detail=None):
    cond = bool(cond)
    line = ("  ok    " if cond else "  FAIL  ") + label + ("" if detail is None else f"   [{detail}]")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not cond:
        FAILS.append(label)


# ------------------------------------------------------------------ the clip
T = 48
W4, H4 = 3840, 2160
DISC_R = 340.0
D0, D1 = np.array([500.3, 400.7]), np.array([3300.3, 1700.7])
G0, G1 = np.array([3400.3, 300.4]), np.array([600.3, 300.4])
S0 = np.array([2000.6, 1900.2])
SIGMA = 8.0


def path(a, b, f):
    return a + (b - a) * (f / (T - 1))


def paint(img, c, kind):
    """Add one object centred on `c` (pixel i's centre at i): a flat disc with a one-pixel ramp, or a gaussian."""
    h = DISC_R + 3 if kind == "disc" else 5 * SIGMA
    x0, x1 = int(np.floor(c[0] - h)), int(np.ceil(c[0] + h)) + 1
    y0, y1 = int(np.floor(c[1] - h)), int(np.ceil(c[1] + h)) + 1
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    r = np.hypot(xx - c[0], yy - c[1])
    v = np.clip(DISC_R + 0.5 - r, 0.0, 1.0) * 200.0 if kind == "disc" else 250.0 * np.exp(-r * r / (2 * SIGMA ** 2))
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], v)


def make_clip(name, w, h, objects):
    p = os.path.join(OUT, name)
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (w, h))
    assert vw.isOpened(), "no FFV1 writer"
    for f in range(T):
        img = np.zeros((h, w), np.float64)
        for a, b, kind in objects:
            paint(img, path(a, b, f), kind)
        g = np.round(img).astype(np.uint8)
        vw.write(cv2.merge([g, g, g]))
    vw.release()
    return p


# ------------------------------------------------------------------ the stand-in "perfect tracker"
BIG, SMALL = 110, 24        # small-picture px: the whole disc (radius 340 / 3.75 = 91); a dot (sigma 2.1)


def half_of(frame, q):
    """The window an object needs: a dot fades to 0 inside SMALL px, the disc does not."""
    g = frame.astype(np.float64).sum(axis=2)
    x, y = int(round(q[0])), int(round(q[1]))
    ring = np.concatenate([g[y - SMALL, x - SMALL:x + SMALL + 1], g[y + SMALL, x - SMALL:x + SMALL + 1],
                           g[y - SMALL:y + SMALL + 1, x - SMALL], g[y - SMALL:y + SMALL + 1, x + SMALL]])
    return SMALL if ring.max() == 0 else BIG


def centroid(frame, guess, half):
    """Intensity centroid near `guess` (small-picture px), re-centred twice."""
    g = frame.astype(np.float64).sum(axis=2)
    c = np.asarray(guess, np.float64)
    for _ in range(3):
        x0, y0 = max(0, int(c[0]) - half), max(0, int(c[1]) - half)
        x1, y1 = min(g.shape[1], int(c[0]) + half + 1), min(g.shape[0], int(c[1]) + half + 1)
        win = g[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        s = win.sum()
        c = np.array([(win * xx).sum() / s, (win * yy).sum() / s])
    return c


class PerfectStream:
    """AllTrackerStream's contract (WINDOW 16, STRIDE 8, the query frame = frame 0 of the stream)."""
    answers: list = []      # every (frames, tracks) it returned, for [d]

    def __init__(self, iters: int = 4):
        self._buf, self._n_fed, self._first = [], 0, True

    def start(self, frame0_rgb, query_xy):
        self._xy = np.asarray(query_xy, np.float32).reshape(-1, 2)
        self._half = [half_of(frame0_rgb, q) for q in self._xy]
        self._c0 = [centroid(frame0_rgb, q, h) for q, h in zip(self._xy, self._half)]
        self._seen = {0: list(self._c0)}      # stream frame -> centroids (windows overlap: frames come again)
        self._buf, self._n_fed, self._first = [frame0_rgb], 1, True

    def _rows(self, frames, start_idx):
        tr = np.zeros((len(frames), len(self._xy), 2), np.float32)
        for i, fr in enumerate(frames):
            f = start_idx + i
            if f not in self._seen:
                prev, before = self._seen[f - 1], self._seen.get(f - 2, self._seen[f - 1])
                # constant-speed prediction centres the window on where the object should be now
                self._seen[f] = [centroid(fr, 2 * p - b, h) for p, b, h in zip(prev, before, self._half)]
            for k in range(len(self._xy)):
                tr[i, k] = self._xy[k] + (self._seen[f][k] - self._c0[k])
        n = len(frames)
        PerfectStream.answers.append((start_idx, tr.copy()))
        return start_idx, tr, np.ones((n, len(self._xy)), np.float32), np.ones((n, len(self._xy)), np.float32)

    def push(self, frame_rgb):
        self._buf.append(frame_rgb)
        self._n_fed += 1
        if len(self._buf) == at_backend.WINDOW:
            out = self._rows(self._buf, self._n_fed - at_backend.WINDOW)
            self._first = False
            self._buf = self._buf[at_backend.STRIDE:]
            return out
        return None

    def flush(self):
        n_real = len(self._buf)
        if n_real == 0 or (not self._first and n_real <= at_backend.WINDOW // 2):
            return None
        if self._first and n_real < 2:
            return None
        return self._rows(self._buf, self._n_fed - n_real)


at_backend.get_alltracker = lambda progress=None, cancel=lambda: False: (None, "cpu")
at_backend.AllTrackerStream = PerfectStream


def worker_run(video, seeds, n, refine, roi):
    tracks = np.full((n, len(seeds), 2), np.nan, np.float32)
    err = {}
    w = TrackingWorker(video, 0, np.asarray(seeds, np.float32), list(range(len(seeds))), FrameCache(1024 ** 3), n,
                       refine=refine, roi=roi, autopause=False, point_backend="alltracker")
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mem, fr: tracks.__setitem__(slice(w0, w0 + len(tr)), tr))
    w.error.connect(lambda m: err.update(e=m))
    w.run()
    if "e" in err:
        raise RuntimeError(err["e"])
    return tracks


def fmt(v):
    return f"({v[0]:+.3f}, {v[1]:+.3f})"


V4 = make_clip("far4k.avi", W4, H4, [(D0, D1, "disc"), (G0, G1, "gauss"), (S0, S0, "gauss")])
TRUTH = np.stack([path(D0, D1, T - 1), path(G0, G1, T - 1), S0])
SEEDS = np.stack([D0, G0, S0])


def section_a():
    print("[a] bare worker, LK off, ROI off: the model's own map")
    tr = worker_run(V4, SEEDS, T, refine=False, roi=False)
    for k, name in enumerate(("D disc", "G dot", "S still")):
        e = tr[T - 1, k] - TRUTH[k]
        check(f"{name}: last-frame error {fmt(e)} < 0.1 px", np.all(np.abs(e) < 0.1))


def section_b():
    print("[b] bare worker, LK on, ROI off")
    tr = worker_run(V4, SEEDS, T, refine=True, roi=False)
    for k, name in enumerate(("D disc (LK cannot grip it)", "G dot (LK grips it)", "S still")):
        e = tr[T - 1, k] - TRUTH[k]
        check(f"{name}: last-frame error {fmt(e)} < 0.1 px", np.all(np.abs(e) < 0.1))


def pump(sec=0.1):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.003)


def wait(cond, timeout):
    t0 = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t0 > timeout:
            return False
    return True


def canvas_click(win, x, y):
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    pump(0.05)


def section_c():
    print("[c] the app: Point > AllTracker, N + a click on each object, T")
    forget_recovery(V4)
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    win._open_video(V4)
    check("the clip opens", wait(lambda: win.state == READY and win.project is not None and not win._loading, 60))
    pump(0.3)
    win.act_pm_alltracker.trigger()
    win._goto(0, force=True)
    pump(0.05)
    for c in SEEDS:
        win.btn_add.setChecked(True)
        canvas_click(win, *c)
    s = win.session
    check("three points placed", s.n_points == 3, s.n_points)
    clicks = s.tracks[0, :3].astype(np.float64).copy()
    win.act_select_all.trigger()
    QTest.keyClick(win, Qt.Key_T)
    wait(lambda: win.state != READY or win._passes is not None, 15)
    done = wait(lambda: win.state == READY and win.worker is None and win._passes is None and win._multi is None, 300)
    check("the run ends", done)
    moved = TRUTH[0] - D0
    e = s.tracks[T - 1, 0].astype(np.float64) - (clicks[0] + moved)
    check(f"D disc: last frame = its click + the disc's move, error {fmt(e)} < 0.1 px", np.all(np.abs(e) < 0.1))
    if win.project is not None:
        win.project.dirty = False
    win.close()
    pump(0.3)


def section_d():
    print("[d] a picture that is not shrunk: the worker's tracks are the stand-in's answers, bit for bit")
    v = make_clip("small1024.avi", 1024, 576, [(np.array([100.3, 120.7]), np.array([900.6, 450.2]), "gauss")])
    PerfectStream.answers = []
    tr = worker_run(v, [[100.3, 120.7]], T, refine=False, roi=False)
    ans = np.full_like(tr, np.nan)
    for w0, a in PerfectStream.answers:
        ans[w0:w0 + len(a)] = a
    check("every frame equal (np.array_equal)", np.array_equal(tr, ans),
          f"max |diff| {np.nanmax(np.abs(tr - ans)):.3g}")


for fn in (section_a, section_b, section_c, section_d):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    probe = getattr(w_, "_dev_probe", None)
    if probe is not None:
        probe.wait(10000)
print("\nverify_point_model_scale: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
