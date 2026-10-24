"""Bit-identity GOLDEN harness for the hand-truth change (the hand-truth plan, PR A, step 2).

The coming change keeps hand-placed cells (manual & tracked) through a run, lets Moving spot re-seed at
later clicks, adds an `imported` per-point array and a Track pop-up (Fresh run / Keep my clicks). It must
not change one coordinate of a run that has no usable click after its start. This harness records
today's outputs (--record, on the branch's base commit, before any code change) and later runs the
changed code against them (--compare).

  python tests/handmix_golden.py --record                -> tests/out/handmix/golden_<sha>.npz + .json
  python tests/handmix_golden.py --compare <golden.npz>  -> per scenario "same" or its first difference
  --only W1,A3   a subset (with --record: those scenarios are replaced in the existing golden of this sha)
  --cpu          only the scenarios that need no model (W9, A3)
sha = `git rev-parse --short HEAD` of the code under test, "-dirty" when kinetrace/ (tracked or untracked
files) or tests/_synth_spots.py differs from HEAD; other test files do not count (the full git status is
in the json). The json also keeps the code's own id (git's tree hash of kinetrace/ + the helper's blob):
a compare says whether the code under test is the golden's own, and a record refuses to write when it
changed while the scenarios ran. Every scenario runs in its
own python process (models, Qt and caches start clean each time, so a subset gives exactly what the full
run gives). KINETRACE_REVIEW_ROOT=<checkout> tests that checkout's kinetrace/ with this worktree's media
(the verify_review_i185 pattern). HANDMIX_CHOICE (default "keep") is what MainWindow._ask_track_choice
(the Track pop-up the change adds) answers; it is patched on the class only when the method exists, so a
compare run never blocks.

LAYER W - a bare TrackingWorker run on this thread. Recorded: every chunk_ready payload (window start,
tracks, visibility, confidence, the group-member arrays by key, the display frames' indices + an md5 of
their pixels), the count + a content hash of every masks_ready / balls_ready payload, the signals
(autopaused, finished_ok, error) and worker._autopause_reason, n_cols, point_ids.
  W1  CoTracker3, test600, 4 points seeded at GT[0], frames 0..240, LK refine, ROI, auto-pause on
  W2  the same with AllTracker
  W3  CoTracker3, test4k (3840x2160), frames 0..150 (ROI restarts happen there)
  W4  the same with AllTracker
  W5  a group point (radius 12) + 2 points, CoTracker3, test600, frames 0..120
  W6  an anchor point (anchor=True) + 3 points, CoTracker3, test600, frames 0..120
  W7  held points on stored silhouettes (verify_review_i185 [2]: rectangles round dot 0, constrain 0-1,
      on-body 0-2, auto-pause off), CoTracker3, test600, frames 0..99
  W8  ball markers (the verify_balls [3] clip, real SAM): ball A with an EXTRA prompt at frame 40, ball C
      with its radius hint, auto-pause on
  W9  Moving spot (no model): the _synth_spots "sky" scene; the bird (gone from frame 45) with its speed,
      a still speck with no speed; auto-pause on
  W10 the verify_conf_autopause sabotage clip, CoTracker3, auto-pause on (it auto-pauses)
LAYER A - the app (MainWindow, offscreen); points and clicks set through the session API, the user's key
(T, or F in semi-automatic mode). Recorded per camera: every POINT_ARRAYS array before and after the
run, the hand cells (manual & tracked) after the start frame before the run, a digest of the
silhouettes, the end state (each worker's start / end / auto-pause / reason, the playhead) and that one
Ctrl+Z (act_undo) gives the pre-run arrays back. The A1 / A2 / A5 variants each start from their (i).
  A1i   test600, 4 CoTracker3 points at GT[0] on frame 0, T from frame 0 to the end (600 frames)
  A1ii  + every point clicked at GT on frames 20 and 50; T from frame 50 (clicks only at / before it)
  A1iii + a 5th point clicked on frames 200 / 400, NOT selected (LAYERS select_only of points 0-3)
  A1iv  + a click ahead on point 0 at frame 300, marked hidden (occluded)
  A1v   + clicks ahead on point 0 at frames 200 and 400, at GT + 6 px in x
  A2i   semi-automatic (act_mode_semi), 4 points at GT[100] on frame 100, F once
  A2ii  + a click on point 0 at frame 101 (GT + 6 px in x): the frame the step writes
  A3    Moving spot app run: the _synth_spots "sea" scene (gone from 45), clicked on frames 0 AND 1
        (the app takes the speed from the click on start+1), T from 0
  A4    two passes on the sabotage clip: 2 AllTracker points on dot B, then 2 CoTracker3 points (the
        sabotaged dot A, and dot B): pass 2 stops earlier, pass 1 is trimmed to it; auto-pause on
  A5i   two cameras (test600 and its mirror image), 2 landmarks in both, Track > Every camera, T
  A5ii  + a click ahead in the second camera only (P1 at frame 300, GT + 6 px in x)
  A6    verify_review_i185 [3]: a two-pass run (the AllTracker head + the animal with a mock SAM, then a
        CoTracker3 point kept on the silhouettes) + 2 silhouette-derived landmarks (tip, centroid)
  A7    (PR D, review R1-M5) two cameras (test600 + mirror), 2 AllTracker + 2 CoTracker3 landmarks with the SAME
        trackers in both cameras, Track > Every camera, T: two passes, each in both cameras (recorded on main
        1d4c3d4, before the per-camera pass planning: golden_1d4c3d4.npz)

COMPARE RULE
  W: every chunk payload and every signal np.array_equal (NaN == NaN); the mask / ball payload hashes
     equal.
  A: the pre-run arrays equal (the set-up is the same); for each camera and array, a cell that was a
     hand cell after the start frame before the run may equal EITHER the old post-run value OR its
     pre-run value (reported per scenario: "hand cells kept: n / overwritten: m"); a hand cell that is
     partly kept and partly overwritten (its position kept but its hand flag cleared, say) is a
     difference; every other cell must equal the old post-run value; the silhouettes and the end state
     equal; one Ctrl+Z must give the pre-run arrays back. An array the golden does not have (`imported`)
     is named, not compared; one it has that is gone is a difference.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
ROOT = os.environ.get("KINETRACE_REVIEW_ROOT") or WT
OUT = os.path.join(WT, "tests", "out", "handmix")
SCRATCH = os.path.join(OUT, "scratch")
V600 = os.path.join(WT, "test600.mp4")
V4K = os.path.join(WT, "test4k.mp4")
SEP = "~"                      # npz key = "<scenario>~<array>"
CHILD_TIMEOUT = 3600


def say(*parts) -> None:
    s = " ".join(str(p) for p in parts)
    print(s.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


# ===================================================================== generic helpers
def _feed(h, o) -> None:
    """A stable content hash of nested payloads (dicts, lists, arrays, numbers, objects)."""
    if o is None:
        h.update(b"N")
    elif isinstance(o, (bool, np.bool_)):
        h.update(b"B1" if o else b"B0")
    elif isinstance(o, (int, np.integer)):
        h.update(b"I" + str(int(o)).encode())
    elif isinstance(o, (float, np.floating)):
        h.update(b"F" + np.float64(o).tobytes())
    elif isinstance(o, str):
        h.update(b"S" + o.encode("utf-8", "replace"))
    elif isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        h.update(b"A" + str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    elif isinstance(o, dict):
        h.update(b"D%d" % len(o))
        for k in sorted(o, key=repr):
            _feed(h, k)
            _feed(h, o[k])
    elif isinstance(o, (list, tuple)):
        h.update(b"L%d" % len(o))
        for x in o:
            _feed(h, x)
    elif isinstance(o, (set, frozenset)):
        _feed(h, sorted(o, key=repr))
    else:
        h.update(b"O" + type(o).__name__.encode())
        d = getattr(o, "__dict__", None)
        _feed(h, d if d is not None else repr(o))


def digest(o) -> str:
    h = hashlib.md5()
    _feed(h, o)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [jsonable(x) for x in sorted(o, key=repr)]
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else repr(f)
    if o is None or isinstance(o, (int, str)):
        return o
    return repr(o)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_eq(a, b) -> np.ndarray:
    """Per-cell equality over the first two axes (trailing axes must all agree), NaN == NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        af, bf = a.astype(np.float64), b.astype(np.float64)
        e = (af == bf) | (np.isnan(af) & np.isnan(bf))
    else:
        e = np.asarray(a == b)
    if e.ndim == 0:
        e = e.reshape(1, 1)
    elif e.ndim == 1:
        e = e[:, None]
    while e.ndim > 2:
        e = e.all(axis=-1)
    return e


def first_diff(a, b):
    """None (equal), "shape", or (row, col) of the first differing cell."""
    if np.shape(a) != np.shape(b):
        return "shape"
    e = cell_eq(a, b)
    if e.all():
        return None
    r, c = np.argwhere(~e)[0]
    return int(r), int(c)


def fmt(v) -> str:
    a = np.asarray(v)
    if a.dtype.kind == "f":
        vals = [f"{float(x):.7g}" for x in a.ravel()]
        return vals[0] if a.ndim == 0 else "[" + " ".join(vals) + "]"
    return str(a.tolist())


# ===================================================================== the scenario table
SCENARIOS: dict = {}


def scenario(name: str, layer: str, gpu: bool, desc: str):
    def deco(fn):
        SCENARIOS[name] = types.SimpleNamespace(name=name, layer=layer, gpu=gpu, desc=desc, fn=fn)
        return fn
    return deco


class Skip(Exception):
    """A scenario that cannot be built here (said in the json with the reason)."""


K = types.SimpleNamespace()      # the child's imported modules (filled by child_setup)


def child_setup(name: str) -> None:
    os.makedirs(SCRATCH, exist_ok=True)
    rec = os.path.join(SCRATCH, "recovery_" + name)
    shutil.rmtree(rec, ignore_errors=True)
    st = os.path.join(SCRATCH, "settings_" + name + ".ini")
    if os.path.exists(st):
        os.remove(st)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["KINETRACE_RECOVERY_DIR"] = rec
    os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"
    os.environ["KINETRACE_SETTINGS"] = st
    sys.path.insert(0, ROOT)
    sys.path.insert(1, HERE)
    sys.stdout.reconfigure(errors="replace")
    import cv2
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QProgressDialog

    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    app = QApplication.instance() or QApplication([])
    from kinetrace import tracker as trk
    from kinetrace.video_source import FrameCache
    K.__dict__.update(cv2=cv2, Qt=Qt, QTimer=QTimer, QTest=QTest, QApplication=QApplication,
                      QMessageBox=QMessageBox, QProgressDialog=QProgressDialog, app=app, trk=trk,
                      FrameCache=FrameCache)


def env_info() -> dict:
    d = {"python": sys.version.split()[0], "numpy": np.__version__}
    t = sys.modules.get("torch")
    if t is not None:
        d["torch"] = t.__version__
        try:
            d["cuda"] = bool(t.cuda.is_available())
            d["gpu"] = t.cuda.get_device_name(0) if d["cuda"] else ""
        except Exception:      # noqa: BLE001 - information only
            pass
    trk = sys.modules.get("kinetrace.tracker")
    if trk is not None:
        d["tracker_device"] = str(getattr(trk, "_device", None))
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        d["cv2"] = cv2.__version__
    return d


def _gt(video: str) -> np.ndarray:
    return np.load(video + ".gt.npz")["gt"]


def _need_alltracker() -> None:
    from kinetrace import alltracker_backend
    if not alltracker_backend.available():
        raise Skip("AllTracker is not installed here")


# ===================================================================== synthetic clips (rebuilt each run)
def _make_sabotage(path: str):
    """tests/verify_conf_autopause.py section 1, verbatim: dot A vanishes and the background is swapped
    at frame 150."""
    cv2 = K.cv2
    W, H = 640, 480

    def make_bg(seed):
        r = np.random.default_rng(seed)
        bg = r.integers(30, 90, size=(H, W, 3), dtype=np.uint8)
        return cv2.GaussianBlur(bg, (0, 0), 1.5)

    def dot(frame, x, y, color):
        cv2.circle(frame, (round(x), round(y)), 9, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (round(x), round(y)), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    T1, SAB = 300, 150
    bg_a, bg_b = make_bg(3), make_bg(77)
    ax = 200 + 60 * np.sin(np.arange(T1) / 40)
    ay = 200 + 40 * np.cos(np.arange(T1) / 55)
    bx = 460 + 50 * np.sin(np.arange(T1) / 47)
    by = 320 + 45 * np.cos(np.arange(T1) / 38)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in range(T1):
        frame = (bg_a if f < SAB else bg_b).copy()
        if f < SAB:
            dot(frame, ax[f], ay[f], (60, 60, 230))
        dot(frame, bx[f], by[f], (60, 230, 60))
        vw.write(frame)
    vw.release()
    return ax, ay, bx, by


def _make_ball_clip(path: str):
    """tests/verify_balls.py 'the clip', verbatim: three balls, B vanishes at 90, C leaves at the right."""
    cv2 = K.cv2
    W, H, N, VANISH = 1280, 720, 140, 90
    rng = np.random.RandomState(3)
    truth = np.zeros((N, 3, 2))
    R = np.array([21.0, 17.0, 19.0])
    pos = np.array([[300.0, 360.0], [420.0, 330.0], [560.0, 400.0]])
    vel = np.array([[4.2, -1.1], [3.6, 1.4], [7.0, -0.6]])
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for k in range(N):
        img = np.full((H, W, 3), 96, np.uint8)
        img[..., 1] = (96 + 20 * np.sin(np.arange(W) / 37.0)[None, :]
                       + 10 * np.cos(np.arange(H) / 23.0)[:, None]).astype(np.uint8)
        truth[k] = pos
        cv2.line(img, (int(pos[0, 0]), int(pos[0, 1])), (int(pos[0, 0] + 90), int(pos[0, 1] + 30)),
                 (235, 235, 235), 7, cv2.LINE_AA)
        cv2.circle(img, (int(round(pos[0, 0])), int(round(pos[0, 1]))), int(R[0]), (40, 40, 220), -1, cv2.LINE_AA)
        if k < VANISH:
            cv2.circle(img, (int(round(pos[1, 0])), int(round(pos[1, 1]))), int(R[1]), (60, 200, 60), -1,
                       cv2.LINE_AA)
        cv2.circle(img, (int(round(pos[2, 0])), int(round(pos[2, 1]))), int(R[2]), (230, 120, 40), -1, cv2.LINE_AA)
        vw.write(img)
        pos = pos + vel + rng.normal(0, 0.3, (3, 2))
        pos[:2] = np.clip(pos[:2], 40, [W - 40, H - 40])
    vw.release()
    return np.round(truth), R, N


def _make_clip100(path: str) -> None:
    """tests/verify_review_i185.py make_clip: the first 100 frames of test600."""
    cv2 = K.cv2
    cap = cv2.VideoCapture(V600)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for _ in range(100):
        ok, fr = cap.read()
        vw.write(fr)
    cap.release()
    vw.release()


def _make_mirror(path: str) -> None:
    """test600 mirrored left-right (the second camera of A5)."""
    cv2 = K.cv2
    cap = cv2.VideoCapture(V600)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(cv2.flip(fr, 1))
    cap.release()
    vw.release()


HALF = (100, 100)               # verify_review_i185: the mock silhouette = a rectangle round dot 0, +-HALF px


def _rect_mask(GT, f, w=640, h=480, half=HALF):
    cx, cy = GT[f, 0]
    x0, x1 = int(round(cx - half[0])), int(round(cx + half[0]))
    y0, y1 = int(round(cy - half[1])), int(round(cy + half[1]))
    m = np.zeros((h, w), bool)
    m[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
    return m


# ===================================================================== layer W
class WRec:
    """Everything a bare worker emits, copied as it arrives (the worker runs on this thread)."""

    def __init__(self, w):
        self.w = w
        self.chunks = []
        self.masks, self.balls = [], []
        self.autopaused, self.finished, self.errors = [], [], []
        w.chunk_ready.connect(self._on_chunk)
        w.masks_ready.connect(lambda summ: self.masks.append(digest(summ)))
        w.balls_ready.connect(lambda rows: self.balls.append(digest(rows)))
        w.autopaused.connect(lambda f, p: self.autopaused.append([int(f), int(p)]))
        w.finished_ok.connect(lambda last, p: self.finished.append([int(last), bool(p)]))
        w.error.connect(lambda m: self.errors.append(str(m)))

    def _on_chunk(self, w0, tr, vi, cf, mem, fr):
        members = {int(k): np.array(v, copy=True) for k, v in (mem or {}).items()}
        idx = [int(i) for i, _rgb in (fr or [])]
        h = hashlib.md5()
        for i, rgb in (fr or []):
            a = np.ascontiguousarray(rgb)
            h.update(str((int(i), a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        self.chunks.append((int(w0), np.array(tr, copy=True), np.array(vi, copy=True), np.array(cf, copy=True),
                            members, idx, h.hexdigest()))

    def result(self):
        w, ch = self.w, self.chunks
        K_ = int(w.n_cols)
        arrays = {
            "w0": np.array([c[0] for c in ch], np.int64),
            "L": np.array([c[1].shape[0] for c in ch], np.int64),
            "tracks": np.concatenate([c[1] for c in ch]) if ch else np.zeros((0, K_, 2), np.float32),
            "vis": np.concatenate([c[2] for c in ch]) if ch else np.zeros((0, K_), bool),
            "conf": np.concatenate([c[3] for c in ch]) if ch else np.zeros((0, K_), np.float32),
            "frames_idx": np.array([i for c in ch for i in c[5]], np.int64),
            "frames_n": np.array([len(c[5]) for c in ch], np.int64),
        }
        for i, c in enumerate(ch):
            for k, v in c[4].items():
                arrays[f"mem.{i:04d}.{k}"] = v
        tails = [m.strip().splitlines()[-1] if m.strip() else "" for m in self.errors]
        meta = {
            "signals": {"autopaused": self.autopaused, "finished": self.finished, "n_errors": len(self.errors),
                        "error_tails": tails, "reason": str(getattr(w, "_autopause_reason", "")),
                        "n_cols": int(w.n_cols), "point_ids": [int(q) for q in w.point_ids]},
            "masks": {"count": len(self.masks), "hashes": self.masks},
            "balls": {"count": len(self.balls), "hashes": self.balls},
            "members_keys": [sorted(c[4]) for c in ch],
            "frames_md5": [c[6] for c in ch],
            "dtypes": {k: str(v.dtype) for k, v in arrays.items() if k in ("tracks", "vis", "conf")},
            # worker internals, for diagnosis only (compared as a note, never a failure)
            "extra": jsonable({k: getattr(w, k, None) for k in ("_spot_ended", "_ball_ended", "_seg_ended",
                                                              "_exit_hit", "_autopause_hit", "decode_failed_at")}),
        }
        if self.errors:
            meta["errors_full"] = self.errors
        return arrays, meta


def run_w(w, inputs: dict):
    rec = WRec(w)
    t0 = time.time()
    w.run()
    arrays, meta = rec.result()
    meta["run_seconds"] = round(time.time() - t0, 1)
    meta["inputs"] = {os.path.basename(p): md5_file(p) for p in inputs}
    sig = meta["signals"]
    say(f"  {len(rec.chunks)} chunks, finished {sig['finished']}, auto-paused {sig['autopaused']} "
        f"({sig['reason'] or '-'}), masks {meta['masks']['count']}, balls {meta['balls']['count']}, "
        f"errors {sig['n_errors']}, {meta['run_seconds']} s")
    if sig["n_errors"]:
        say("  ERROR from the worker: " + sig["error_tails"][0])
    return arrays, meta


def _w_four(video: str, n: int, backend: str, cache: int):
    if backend == "alltracker":
        _need_alltracker()
    trk = K.trk
    GT = _gt(video)
    specs = [trk.PointSpec(i, GT[0, i].astype(np.float32).copy()) for i in range(4)]
    w = trk.TrackingWorker(video, 0, None, None, K.FrameCache(cache), n, refine=True, specs=specs, roi=True,
                           autopause=True, point_backend=backend)
    return run_w(w, [video])


@scenario("W1", "W", True, "CoTracker3, test600, 4 points at GT[0], frames 0..240, refine + ROI + auto-pause")
def w1():
    return _w_four(V600, 241, "cotracker3", 1 << 28)


@scenario("W2", "W", True, "AllTracker, test600, 4 points at GT[0], frames 0..240, refine + ROI + auto-pause")
def w2():
    return _w_four(V600, 241, "alltracker", 1 << 28)


@scenario("W3", "W", True, "CoTracker3, test4k, 4 points at GT[0], frames 0..150 (ROI restarts)")
def w3():
    return _w_four(V4K, 151, "cotracker3", 1 << 30)


@scenario("W4", "W", True, "AllTracker, test4k, 4 points at GT[0], frames 0..150 (ROI restarts)")
def w4():
    return _w_four(V4K, 151, "alltracker", 1 << 30)


@scenario("W5", "W", True, "CoTracker3, test600, a group point (radius 12) + 2 points, frames 0..120")
def w5():
    trk = K.trk
    GT = _gt(V600)
    specs = [trk.PointSpec(0, GT[0, 0].astype(np.float32).copy(), kind="group", radius=12.0),
             trk.PointSpec(1, GT[0, 1].astype(np.float32).copy()),
             trk.PointSpec(2, GT[0, 2].astype(np.float32).copy())]
    w = trk.TrackingWorker(V600, 0, None, None, K.FrameCache(1 << 28), 121, refine=True, specs=specs, roi=True,
                           autopause=True, point_backend="cotracker3")
    return run_w(w, [V600])


@scenario("W6", "W", True, "CoTracker3, test600, an anchor point (anchor=True) + 3 points, frames 0..120")
def w6():
    trk = K.trk
    GT = _gt(V600)
    specs = [trk.PointSpec(0, GT[0, 0].astype(np.float32).copy(), anchor=True)]
    specs += [trk.PointSpec(i, GT[0, i].astype(np.float32).copy()) for i in range(1, 4)]
    w = trk.TrackingWorker(V600, 0, None, None, K.FrameCache(1 << 28), 121, refine=True, specs=specs, roi=True,
                           autopause=True, point_backend="cotracker3")
    return run_w(w, [V600])


@scenario("W7", "W", True, "CoTracker3, test600, held points on stored silhouettes (verify_review_i185 [2])")
def w7():
    trk = K.trk
    from kinetrace.segmenter import MaskTrack
    GT = _gt(V600)
    N = 100
    mt = MaskTrack(N + 8, 640, 480)                       # stored_track(N + 8) of verify_review_i185
    for f in range(N + 8):
        mt.set_from_work(f, _rect_mask(GT, f), (1.0, 1.0), 8.0)
    specs = [trk.PointSpec(i, GT[0, i].astype(np.float32).copy()) for i in range(3)]
    w = trk.TrackingWorker(V600, 0, None, None, K.FrameCache(1 << 28), N, specs=specs, constrain_pids=[0, 1],
                           on_body_pids=[0, 1, 2], point_backend="cotracker3", autopause=False, stored_masks=mt)
    return run_w(w, [V600])


@scenario("W8", "W", True, "ball markers, the verify_balls clip, real SAM, ball A with an extra prompt at 40")
def w8():
    trk = K.trk
    from kinetrace import segmenter
    path = os.path.join(OUT, "balls_test.mp4")
    truth_r, R, N = _make_ball_clip(path)
    backend = segmenter.preferred_backend()
    pa = {0: [[float(truth_r[0, 0, 0]), float(truth_r[0, 0, 1]), 1]],
          40: [[float(truth_r[40, 0, 0]), float(truth_r[40, 0, 1]), 1]]}          # the EXTRA later prompt
    pc = {0: [[float(truth_r[0, 2, 0]), float(truth_r[0, 2, 1]), 1]]}
    balls = [trk.BallSpec(0, pa, truth_r[0, 0].astype(np.float32), None, backend),
             trk.BallSpec(1, pc, truth_r[0, 2].astype(np.float32), R[2], backend)]
    w = trk.TrackingWorker(path, 0, None, None, K.FrameCache(256 * 1024 ** 2), N, refine=True, specs=[], roi=True,
                           autopause=True, balls=balls, point_backend="cotracker3")
    arrays, meta = run_w(w, [path])
    meta["seg_backend"] = backend
    return arrays, meta


@scenario("W9", "W", False, "Moving spot, the 'sky' scene: the bird with its speed + a still speck without")
def w9():
    trk = K.trk
    from _synth_spots import make_scene, write_video
    frames, gt = make_scene("sky", n=60, vanish_at=45)
    path = os.path.join(OUT, "sky60.mp4")
    write_video(path, frames)
    specs = [trk.SpotSpec(0, gt[0], gt[1] - gt[0], {}),           # the bird, its speed from frames 0 -> 1
             trk.SpotSpec(1, (480.0, 40.0), None, {})]           # a still dark speck, speed unknown
    w = trk.TrackingWorker(path, 0, None, None, K.FrameCache(1 << 30), len(frames), specs=[], spots=specs,
                           autopause=True)
    return run_w(w, [path])


@scenario("W10", "W", True, "CoTracker3, the sabotage clip, auto-pause on (it auto-pauses near 150)")
def w10():
    trk = K.trk
    path = os.path.join(OUT, "sabotage.mp4")
    ax, ay, bx, by = _make_sabotage(path)
    specs = [trk.PointSpec(0, np.array([ax[0], ay[0]], np.float32)),
             trk.PointSpec(1, np.array([bx[0], by[0]], np.float32))]
    w = trk.TrackingWorker(path, 0, None, None, K.FrameCache(200 * 1024 * 1024), 300, refine=True, specs=specs,
                           autopause=True, point_backend="cotracker3")
    return run_w(w, [path])


# ===================================================================== layer A: the app
def a_setup() -> None:
    from kinetrace.app import READY, TRACKING, MainWindow
    from kinetrace.session import POINT_ARRAYS
    patched = hasattr(MainWindow, "_ask_track_choice")
    if patched:                    # the Track pop-up of the change: answered, never shown (HANDMIX_CHOICE)
        MainWindow._ask_track_choice = lambda self, *a, **k: os.environ.get("HANDMIX_CHOICE", "keep")
    K.__dict__.update(MainWindow=MainWindow, READY=READY, TRACKING=TRACKING, POINT_ARRAYS=POINT_ARRAYS,
                      choice_patched=patched)
    K.guard = ModalGuard()


class ModalGuard:
    """Never block: a modal dialog nothing here expected (open for 2 s) is recorded and rejected. The
    golden has none; one in a compare run is a difference."""

    def __init__(self):
        self.seen = []
        self._first = {}
        self.timer = K.QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def tick(self):
        m = K.QApplication.activeModalWidget()
        if m is None or isinstance(m, K.QProgressDialog):
            return
        t0 = self._first.setdefault(id(m), time.time())
        if time.time() - t0 < 2.0:
            return
        txt = m.text() if isinstance(m, K.QMessageBox) else ""
        self.seen.append({"class": type(m).__name__, "title": m.windowTitle(), "text": txt[:300]})
        say(f"  UNEXPECTED MODAL {type(m).__name__} {m.windowTitle()!r}: rejected")
        m.reject()


def forget_recovery(*videos) -> None:
    """tests/_clean.forget_recovery, inlined (importing _clean would put this worktree's kinetrace first
    on sys.path, against KINETRACE_REVIEW_ROOT)."""
    from kinetrace import recovery
    want = {os.path.normcase(os.path.abspath(str(v))) for v in videos}
    for info in recovery.scan():
        if any(os.path.normcase(os.path.abspath(v)) in want for v in info.get("videos", [])):
            recovery.discard(info["project_id"])


def pump(t: float = 0.1) -> None:
    t0 = time.time()
    while time.time() - t0 < t:
        K.app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout: float, what: str) -> None:
    t0 = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def new_win():
    win = K.MainWindow()
    win.resize(1500, 950)
    win.show()
    pump(0.2)
    return win


def close_win(win) -> None:
    try:
        if win.project is not None:
            win.project.dirty = False
    except Exception:      # noqa: BLE001 - closing is best effort
        pass
    win.close()
    pump(0.3)
    win._dev_probe.wait(30000)


def open_video(win, path: str) -> None:
    forget_recovery(path)
    win._open_video(path)
    wait(lambda: win.state == K.READY and win.project is not None and not win._loading, 120,
         "open " + os.path.basename(path))
    pump(0.3)


def add_pts(s, frame: int, xys, tracker: str, names=None) -> list:
    """Points placed by hand on `frame` through the session API, each with its own tracker."""
    pids = []
    for i, (x, y) in enumerate(np.asarray(xys, np.float64)):
        q = s.add_point(frame, float(x), float(y), name=None if names is None else names[i])
        s.points[q].tracker = tracker
        pids.append(q)
    return pids


def select_rows(win, *rows) -> None:
    """verify_review_i185 select_rows: LAYERS rows selected by hand."""
    win.layers.clearSelection()
    for r in rows:
        win.layers.point_item(r).setSelected(True)
    pump(0.05)


class AppSpy:
    """Every worker a Track press launches (one per pass / camera) and what it signalled."""

    def __init__(self, win):
        self.runs = []
        orig = win._launch_run

        def spy(w, *a, **k):
            driver = k.get("driver", a[0] if a else None)
            for x in (list(driver.workers) if driver is not None else [w]):
                r = {"worker": x, "driver": driver is not None, "autopaused": [], "finished": [], "errors": []}
                x.autopaused.connect(lambda f, p, r=r: r["autopaused"].append([int(f), int(p)]))
                x.finished_ok.connect(lambda last, p, r=r: r["finished"].append([int(last), bool(p)]))
                x.error.connect(lambda m, r=r: r["errors"].append(str(m)))
                self.runs.append(r)
            return orig(w, *a, **k)

        win._launch_run = spy

    def end_state(self, win) -> dict:
        runs = []
        for r in self.runs:
            x = r["worker"]
            runs.append({"video": os.path.basename(x.video_path), "start": int(x.start_frame),
                         "n_frames": int(x.n_frames), "point_ids": [int(q) for q in x.point_ids],
                         "backend": str(x.point_backend), "n_specs": len(x.specs), "n_balls": len(x.balls),
                         "n_spots": len(x.spots), "n_derived": len(x.derived), "driver": r["driver"],
                         "autopaused": r["autopaused"], "finished": r["finished"], "n_errors": len(r["errors"]),
                         "reason": str(getattr(x, "_autopause_reason", ""))})
        p = win.project
        return {"runs": runs, "current": int(win.current), "active": int(p.active) if p is not None else 0}


class ToastLog:
    def __init__(self, win):
        self.msgs = []
        orig = win.toast.show_message

        def show(text, *a, **k):
            self.msgs.append(str(text))
            return orig(text, *a, **k)

        win.toast.show_message = show


def mask_digest(mt):
    arrs = {"bbox": mt.bbox.copy(), "area": mt.area.copy(), "centroid": mt.centroid.copy(), "score": mt.score.copy()}
    h = digest([arrs, {int(f): list(c) for f, c in mt.contours.items()}, {int(f): m for f, m in mt.midline.items()}])
    return arrs, h


def capture(win) -> list:
    out = []
    for sv in win.project.sessions:
        out.append({"arrays": {a.name: getattr(sv, a.name).copy() for a in K.POINT_ARRAYS},
                    "masks": [mask_digest(mt) for mt in sv.seg_masks],
                    "names": [q.name for q in sv.points], "trackers": [q.tracker for q in sv.points]})
    return out


def start_frames(win) -> list:
    p = win.project
    out = []
    for v in range(p.n_views):
        if v == p.active:
            out.append(int(win.current))
        else:
            f = p.map_frame(p.active, v, win.current)
            out.append(None if f is None else int(f))
    return out


def press(win, key, spy: AppSpy, timeout: float) -> None:
    """The user's key; then wait until the whole press is over (every pass, every camera)."""
    n0 = len(spy.runs)
    K.QTest.keyClick(win, key)
    t0 = time.time()
    while len(spy.runs) == n0:
        pump(0.05)
        if time.time() - t0 > 30:
            raise RuntimeError("the key started no run: " + win.toast.text())
    wait(lambda: (win.state == K.READY and win.worker is None and win._passes is None and win._multi is None
                  and getattr(win, "_retrack", None) is None), timeout, "the Track press")
    pump(0.6)


def a_run(win, key, sel=None, timeout: float = 1800):
    """`sel` None = Ctrl+A (act_select_all); a list = LAYERS select_only of those points; "keep" = the
    selection as the scenario left it."""
    if sel is None:
        win.act_select_all.trigger()
    elif sel != "keep":
        win.layers.select_only(list(sel), [])
        win._on_point_selection_changed()
    pump(0.1)
    spy, toasts = AppSpy(win), ToastLog(win)
    starts = start_frames(win)
    selected = [int(q) for q in win._selected_pids()]
    pre = capture(win)
    t0 = time.time()
    press(win, key, spy, timeout)
    secs = round(time.time() - t0, 1)
    post = capture(win)
    end = spy.end_state(win)
    undo_enabled = bool(win.act_undo.isEnabled())
    win.act_undo.trigger()                          # one Ctrl+Z
    pump(0.3)
    after = capture(win)
    bad = []
    for v, (pc, ac) in enumerate(zip(pre, after)):
        for nm, a in pc["arrays"].items():
            b = ac["arrays"].get(nm)
            if b is None or a.shape != b.shape:
                bad.append([v, nm, "shape"])
            else:
                n_bad = int((~cell_eq(a, b)).sum())
                if n_bad:
                    bad.append([v, nm, n_bad])
        if [h for _a, h in pc["masks"]] != [h for _a, h in ac["masks"]]:
            bad.append([v, "silhouettes", "differ"])
    undo = {"enabled": undo_enabled, "ok": bool(undo_enabled and not bad), "bad": bad}
    arrays, cams = {}, []
    for v, (pc, qc) in enumerate(zip(pre, post)):
        st = starts[v]
        for nm, a in pc["arrays"].items():
            arrays[f"v{v}.pre.{nm}"] = a
        for nm, a in qc["arrays"].items():
            arrays[f"v{v}.post.{nm}"] = a
        hand = pc["arrays"]["manual"] & pc["arrays"]["tracked"]
        if st is None:
            hand[:] = False
        else:
            hand[:st + 1] = False                   # hand cells AFTER the start frame
        arrays[f"v{v}.hand"] = hand
        for k, (marr, _h) in enumerate(qc["masks"]):
            for f, a in marr.items():
                arrays[f"v{v}.mask{k}.{f}"] = a
        cams.append({"view": v, "start": st, "names": qc["names"], "trackers": pc["trackers"],
                     "arrays": list(pc["arrays"]), "n_hand": int(hand.sum()),
                     "hand_cells": [[int(f), int(c)] for f, c in np.argwhere(hand)][:100],
                     "mask_hash_pre": [h for _a, h in pc["masks"]], "mask_hash_post": [h for _a, h in qc["masks"]]})
    meta = {"cameras": cams, "end": end, "undo": undo, "modals": list(K.guard.seen), "selected": selected,
            "choice_patched": bool(K.choice_patched), "press_seconds": secs,
            "buttons": {"autopause": bool(win.btn_autopause.isChecked()), "roi": bool(win.btn_roi.isChecked()),
                        "track_mode": str(win._track_mode), "track_all": bool(win.act_track_all.isChecked()),
                        "point_backend": str(win._point_backend)},
            "toasts": toasts.msgs[-6:]}
    say(f"  {len(end['runs'])} worker(s): " + "; ".join(
        f"{r['backend'] if r['n_specs'] else ('spots' if r['n_spots'] else 'balls/segment')} {r['video']} "
        f"from {r['start']} finished {r['finished']} auto-paused {r['autopaused']} {r['reason'] or ''}"
        for r in end["runs"]) + f"; playhead {end['current']}; hand cells ahead {sum(c['n_hand'] for c in cams)}; "
        f"undo {'ok' if undo['ok'] else 'FAILED ' + str(bad)}; {secs} s")
    return arrays, meta


def _a_test600(start: int, extra=None, sel=None, key=None, place_frame: int = 0, semi: bool = False):
    """test600 with 4 CoTracker3 points at GT[place_frame] on that frame, `extra(s, GT, pids)` adds the
    variant's clicks; the run starts at `start`."""
    GT = _gt(V600)
    win = new_win()
    try:
        open_video(win, V600)
        s = win.session
        pids = add_pts(s, place_frame, GT[place_frame], "cotracker3")
        if extra is not None:
            extra(s, GT, pids)
        win._refresh_point_list()
        if semi:
            win.act_mode_semi.trigger()
        win._goto(start, force=True)
        pump(0.3)
        arrays, meta = a_run(win, key if key is not None else K.Qt.Key_T, sel(pids) if callable(sel) else sel)
        meta["inputs"] = {"test600.mp4": md5_file(V600)}
        return arrays, meta
    finally:
        close_win(win)


@scenario("A1i", "A", True, "app, test600, 4 CoTracker3 points at GT[0], T from 0 to the end")
def a1i():
    a_setup()
    return _a_test600(0)


@scenario("A1ii", "A", True, "app, A1i + every point clicked on frames 20 and 50, T from 50")
def a1ii():
    a_setup()

    def extra(s, GT, pids):
        for f in (20, 50):
            for i, q in enumerate(pids):
                s.set_position(f, q, float(GT[f, i, 0]), float(GT[f, i, 1]))
    return _a_test600(50, extra)


@scenario("A1iii", "A", True, "app, A1i + a 5th point clicked ahead (200, 400), not selected")
def a1iii():
    a_setup()

    def extra(s, GT, pids):
        q = s.add_point(0, 100.0, 100.0)
        s.points[q].tracker = "cotracker3"
        for f in (200, 400):
            s.set_position(f, q, 100.0, 100.0)
    return _a_test600(0, extra, sel=lambda pids: pids[:4])


@scenario("A1iv", "A", True, "app, A1i + a hidden click ahead on point 0 at frame 300")
def a1iv():
    a_setup()

    def extra(s, GT, pids):
        s.set_position(300, pids[0], float(GT[300, 0, 0]), float(GT[300, 0, 1]))
        s.set_occluded(300, pids[0], True)
    return _a_test600(0, extra)


@scenario("A1v", "A", True, "app, A1i + clicks ahead on point 0 at 200 and 400 (GT + 6 px in x)")
def a1v():
    a_setup()

    def extra(s, GT, pids):
        for f in (200, 400):
            s.set_position(f, pids[0], float(GT[f, 0, 0] + 6.0), float(GT[f, 0, 1]))
    return _a_test600(0, extra)


@scenario("A2i", "A", True, "app, semi-automatic, 4 points at GT[100] on frame 100, F once")
def a2i():
    a_setup()
    return _a_test600(100, place_frame=100, semi=True, key=K.Qt.Key_F)


@scenario("A2ii", "A", True, "app, A2i + a click on point 0 at frame 101 (GT + 6 px in x), F once")
def a2ii():
    a_setup()

    def extra(s, GT, pids):
        s.set_position(101, pids[0], float(GT[101, 0, 0] + 6.0), float(GT[101, 0, 1]))
    return _a_test600(100, extra, place_frame=100, semi=True, key=K.Qt.Key_F)


@scenario("A3", "A", False, "app, Moving spot, the 'sea' scene (gone at 45), clicks on frames 0 and 1, T from 0")
def a3():
    a_setup()
    from _synth_spots import make_scene, write_video
    frames, GT = make_scene("sea", n=60, vanish_at=45)
    path = os.path.join(OUT, "sea60_app.mp4")
    write_video(path, frames)
    win = new_win()
    try:
        open_video(win, path)
        s = win.session
        (pid,) = add_pts(s, 0, GT[0:1], "spot")
        s.set_position(1, pid, float(GT[1, 0]), float(GT[1, 1]))     # the speed: the click on start+1
        win._refresh_point_list()
        win._goto(0, force=True)
        pump(0.3)
        arrays, meta = a_run(win, K.Qt.Key_T)
        meta["inputs"] = {os.path.basename(path): md5_file(path)}
        return arrays, meta
    finally:
        close_win(win)


@scenario("A4", "A", True, "app, two passes on the sabotage clip (AllTracker x2, CoTracker3 x2), auto-pause on")
def a4():
    a_setup()
    _need_alltracker()
    path = os.path.join(OUT, "sabotage.mp4")
    ax, ay, bx, by = _make_sabotage(path)
    win = new_win()
    try:
        open_video(win, path)
        s = win.session
        add_pts(s, 0, [(bx[0], by[0]), (bx[0] + 4.0, by[0])], "alltracker")       # both on dot B
        add_pts(s, 0, [(ax[0], ay[0]), (bx[0] - 4.0, by[0])], "cotracker3")       # the sabotaged dot A, dot B
        win.btn_autopause.setChecked(True)
        win._refresh_point_list()
        win._goto(0, force=True)
        pump(0.3)
        arrays, meta = a_run(win, K.Qt.Key_T)
        meta["inputs"] = {os.path.basename(path): md5_file(path)}
        return arrays, meta
    finally:
        close_win(win)


def _a5(click_b: bool):
    from kinetrace.project import Project
    from kinetrace.session import TrackingSession
    GT = _gt(V600)
    mirror = os.path.join(OUT, "test600_mirror.mp4")
    _make_mirror(mirror)
    GTB = GT.copy()
    GTB[..., 0] = 639.0 - GT[..., 0]
    proj = os.path.join(SCRATCH, "a5_" + ("ii" if click_b else "i") + ".kinetrace")
    shutil.rmtree(proj, ignore_errors=True)
    Project([TrackingSession(V600, 600, 30.0, 640, 480), TrackingSession(mirror, 600, 30.0, 640, 480)],
            ["camA", "camB"], [0, 0]).save(proj)
    forget_recovery(V600, mirror)
    win = new_win()
    try:
        win._open_project_from_path(proj)
        wait(lambda: (win.state == K.READY and win.project is not None and win.project.n_views == 2
                      and not win._loading), 120, "open the two-camera project")
        pump(0.3)
        p = win.project
        sa, sb = p.sessions
        add_pts(sa, 0, GT[0, :2], "cotracker3", names=["P1", "P2"])
        pb = add_pts(sb, 0, GTB[0, :2], "cotracker3", names=["P1", "P2"])
        if click_b:                     # a click ahead in camera B only
            sb.set_position(300, pb[0], float(GTB[300, 0, 0] + 6.0), float(GTB[300, 0, 1]))
        win._set_active_view(0)
        pump(0.2)
        win._refresh_point_list()
        win._goto(0, force=True)
        pump(0.3)
        win.act_track_all.setChecked(True)
        pump(0.05)
        arrays, meta = a_run(win, K.Qt.Key_T)
        meta["inputs"] = {"test600.mp4": md5_file(V600), os.path.basename(mirror): md5_file(mirror)}
        return arrays, meta
    finally:
        close_win(win)


@scenario("A5i", "A", True, "app, two cameras (test600 + mirror), 2 landmarks in both, Every camera, T")
def a5i():
    a_setup()
    return _a5(False)


@scenario("A5ii", "A", True, "app, A5i + a click ahead in camera B only (P1 at 300, GT + 6 px in x)")
def a5ii():
    a_setup()
    return _a5(True)


@scenario("A7", "A", True, "app, two cameras, AllTracker x2 + CoTracker3 x2 in both, Every camera, T (two passes)")
def a7():
    a_setup()
    _need_alltracker()
    from kinetrace.project import Project
    from kinetrace.session import TrackingSession
    GT = _gt(V600)
    mirror = os.path.join(OUT, "test600_mirror.mp4")
    _make_mirror(mirror)
    GTB = GT.copy()
    GTB[..., 0] = 639.0 - GT[..., 0]
    proj = os.path.join(SCRATCH, "a7.kinetrace")
    shutil.rmtree(proj, ignore_errors=True)
    Project([TrackingSession(V600, 600, 30.0, 640, 480), TrackingSession(mirror, 600, 30.0, 640, 480)],
            ["camA", "camB"], [0, 0]).save(proj)
    forget_recovery(V600, mirror)
    win = new_win()
    try:
        win._open_project_from_path(proj)
        wait(lambda: (win.state == K.READY and win.project is not None and win.project.n_views == 2
                      and not win._loading), 120, "open the two-camera project")
        pump(0.3)
        p = win.project
        sa, sb = p.sessions
        for s, gt in ((sa, GT), (sb, GTB)):          # the same trackers in both cameras
            add_pts(s, 0, gt[0, :2], "alltracker", names=["P1", "P2"])
            add_pts(s, 0, gt[0, 2:4], "cotracker3", names=["P3", "P4"])
        win._set_active_view(0)
        pump(0.2)
        win._refresh_point_list()
        win._goto(0, force=True)
        pump(0.3)
        win.act_track_all.setChecked(True)
        pump(0.05)
        arrays, meta = a_run(win, K.Qt.Key_T)
        meta["inputs"] = {"test600.mp4": md5_file(V600), os.path.basename(mirror): md5_file(mirror)}
        return arrays, meta
    finally:
        close_win(win)


@scenario("A6", "A", True, "app, verify_review_i185 [3]: two passes with the head + mock SAM, + 2 derived landmarks")
def a6():
    a_setup()
    _need_alltracker()
    trk = K.trk
    from kinetrace import segmenter
    GT = _gt(V600)
    clip = os.path.join(OUT, "clip100.mp4")
    _make_clip100(clip)

    class MockAnimal:
        """SAM stand-in: the rectangle round dot 0 of test600 on every frame (verify_review_i185)."""

        def new_session(self, start, size):
            w, h = size

            class S:
                def step(self, native, idx, prompts):
                    m = _rect_mask(GT, idx, w, h)
                    return segmenter.FrameMasks(idx, [1], m[None], np.array([8.0], np.float32), (w, h), (w, h))
            return S()

    win = new_win()
    try:
        open_video(win, clip)
        s = win.session
        win._on_add(*[float(v) for v in GT[0, 0]])
        win._on_add(*[float(v) for v in GT[0, 1]])
        s.points[0].tracker = "alltracker"
        s.points[1].tracker = "cotracker3"
        s.ensure_animal()
        s.animal.add_click(0, float(GT[0, 0, 0]), float(GT[0, 0, 1]), True)
        tip = s.add_landmark("tip", "silhouette", "tip")              # silhouette-derived landmarks
        cen = s.add_landmark("centre", "silhouette", "centroid")
        s.move_points([0, 1, tip, cen], 0)
        s.animal.hold = True
        s.set_head(0)
        win._refresh_point_list()
        win._goto(0, force=True)
        select_rows(win, 0, 1)
        win._on_select(0)
        select_rows(win, 0, 1)
        win._animal_item(0).setSelected(True)          # (G180) the animal's row = its silhouette
        pump(0.05)
        win._update_track_button()
        saved = trk.get_segmenter
        trk.get_segmenter = lambda be, **k: MockAnimal()
        try:
            arrays, meta = a_run(win, K.Qt.Key_T, sel="keep")
        finally:
            trk.get_segmenter = saved
        meta["inputs"] = {os.path.basename(clip): md5_file(clip)}
        return arrays, meta
    finally:
        close_win(win)


# ===================================================================== the child process
def child_main(name: str, prefix: str) -> None:
    sc = SCENARIOS[name]
    meta = {"scenario": name, "layer": sc.layer, "gpu": sc.gpu, "desc": sc.desc}
    arrays = {}
    t0 = time.time()
    try:
        child_setup(name)
        import kinetrace
        meta["kinetrace"] = os.path.dirname(os.path.abspath(kinetrace.__file__))
        say(f"[{name}] {sc.desc}  (code: {meta['kinetrace']})")
        arrays, m = sc.fn()
        meta.update(m)
        meta["status"] = "ok"
    except Skip as e:
        meta["status"], meta["reason"] = "SKIPPED", str(e)
        say(f"  SKIPPED: {e}")
    except Exception:      # noqa: BLE001 - reported to the parent, which decides
        tb = traceback.format_exc()
        say(tb)
        meta["status"], meta["reason"], meta["traceback"] = "ERROR", tb.strip().splitlines()[-1], tb
    meta["seconds"] = round(time.time() - t0, 1)
    meta["env"] = env_info()
    np.savez(prefix + ".npz", **arrays)
    with open(prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump(jsonable(meta), fh, indent=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                 # no Qt / CUDA teardown at exit: the results are written


def run_child(name: str):
    os.makedirs(SCRATCH, exist_ok=True)
    prefix = os.path.join(SCRATCH, "res_" + name)
    for ext in (".npz", ".json"):
        if os.path.exists(prefix + ext):
            os.remove(prefix + ext)
    cmd = [sys.executable, os.path.abspath(__file__), "--child", name, "--dump", prefix]
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, timeout=CHILD_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    if not os.path.exists(prefix + ".json"):
        return {}, {"scenario": name, "status": "ERROR", "reason": f"the child process ended ({rc}) with no result",
                    "seconds": round(time.time() - t0, 1)}
    with open(prefix + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays = {}
    if os.path.exists(prefix + ".npz"):
        with np.load(prefix + ".npz") as z:
            arrays = {k: z[k] for k in z.files}
    return arrays, meta


# ===================================================================== compare
def cmp_w(ga, gm, na, nm):
    diffs, notes = [], []
    gs, ns = gm["signals"], nm["signals"]
    for key in ("point_ids", "n_cols", "autopaused", "finished", "reason", "n_errors", "error_tails"):
        if gs.get(key) != ns.get(key):
            diffs.append(f"signal {key}: golden {gs.get(key)} now {ns.get(key)}")
    for key in ("masks", "balls"):
        g, n = gm[key], nm[key]
        if g != n:
            i = next((j for j, (a, b) in enumerate(zip(g["hashes"], n["hashes"])) if a != b),
                     min(len(g["hashes"]), len(n["hashes"])))
            diffs.append(f"{key}_ready: golden {g['count']} payloads, now {n['count']}; first different payload #{i}")
    gw0, nw0, gL, nL = ga["w0"], na["w0"], ga["L"], na["L"]
    if len(gw0) != len(nw0):
        diffs.append(f"chunk_ready count: golden {len(gw0)} now {len(nw0)}")
    goff = np.concatenate([[0], np.cumsum(gL)]).astype(int)
    noff = np.concatenate([[0], np.cumsum(nL)]).astype(int)
    foff_g = np.concatenate([[0], np.cumsum(ga["frames_n"])]).astype(int)
    foff_n = np.concatenate([[0], np.cumsum(na["frames_n"])]).astype(int)
    pids = gs.get("point_ids", [])
    first, n_bad = None, 0
    for i in range(min(len(gw0), len(nw0))):
        msg = None
        w0 = int(gw0[i])
        if gw0[i] != nw0[i] or gL[i] != nL[i]:
            msg = f"chunk {i}: window start / rows golden ({int(gw0[i])}, {int(gL[i])}) now ({int(nw0[i])}, {int(nL[i])})"
        if msg is None:
            for arr in ("tracks", "vis", "conf"):
                a, b = ga[arr][goff[i]:goff[i + 1]], na[arr][noff[i]:noff[i + 1]]
                d = first_diff(a, b)
                if d == "shape":
                    msg = f"{arr} chunk {i}: shape golden {a.shape} now {b.shape}"
                elif d is not None:
                    r, c = d
                    msg = (f"{arr} chunk {i} frame {w0 + r} column {c} (pid {pids[c] if c < len(pids) else '?'}): "
                           f"golden {fmt(a[r, c])} now {fmt(b[r, c])}")
                if msg:
                    break
        if msg is None:
            gk, nk = gm["members_keys"][i], nm["members_keys"][i]
            if gk != nk:
                msg = f"members chunk {i}: keys golden {gk} now {nk}"
            else:
                for k in gk:
                    a, b = ga[f"mem.{i:04d}.{k}"], na[f"mem.{i:04d}.{k}"]
                    d = first_diff(a, b)
                    if d is not None:
                        msg = (f"members chunk {i} column {k}: shape golden {a.shape} now {b.shape}" if d == "shape"
                               else f"members chunk {i} frame {w0 + d[0]} column {k} member {d[1]}: "
                                    f"golden {fmt(a[d])} now {fmt(b[d])}")
                        break
        if msg is None:
            gi = ga["frames_idx"][foff_g[i]:foff_g[i + 1]]
            ni = na["frames_idx"][foff_n[i]:foff_n[i + 1]]
            if not np.array_equal(gi, ni):
                msg = f"display frames chunk {i}: golden {gi.tolist()} now {ni.tolist()}"
            elif gm["frames_md5"][i] != nm["frames_md5"][i]:
                msg = f"display frames chunk {i} (frames {gi.tolist()[:3]}...): pixels differ"
        if msg:
            n_bad += 1
            first = first or msg
    if first:
        diffs.append(first + f"  [{n_bad} chunk(s) differ]")
    if gm.get("extra") != nm.get("extra"):
        notes.append(f"worker internals differ (not compared): golden {gm.get('extra')} now {nm.get('extra')}")
    if gm.get("dtypes") != nm.get("dtypes"):
        notes.append(f"dtypes: golden {gm.get('dtypes')} now {nm.get('dtypes')}")
    return diffs, notes, ""


def cmp_a(ga, gm, na, nm):
    diffs, notes = [], []
    kept = over = both = mixed = 0
    ncams = {c["view"]: c for c in nm["cameras"]}
    if len(gm["cameras"]) != len(nm["cameras"]):
        diffs.append(f"cameras: golden {len(gm['cameras'])} now {len(nm['cameras'])}")
    for gc in gm["cameras"]:
        v = gc["view"]
        nc = ncams.get(v)
        if nc is None:
            continue
        if gc["start"] != nc["start"]:
            diffs.append(f"camera {v}: start frame golden {gc['start']} now {nc['start']}")
        new_only = [a for a in nc["arrays"] if a not in gc["arrays"]]
        if new_only:
            notes.append(f"camera {v}: array(s) not in the golden, not compared: {', '.join(new_only)}")
        names = gc["names"]
        hand = ga[f"v{v}.hand"]
        eo_all = ep_all = None
        for arr in gc["arrays"]:
            if arr not in nc["arrays"]:
                diffs.append(f"camera {v}: array {arr} is gone")
                continue
            gpre, npre = ga[f"v{v}.pre.{arr}"], na[f"v{v}.pre.{arr}"]
            d = first_diff(gpre, npre)
            if d is not None:
                diffs.append(f"set-up differs: {arr} camera {v} " + ("shape" if d == "shape" else
                             f"frame {d[0]} column {d[1]}: golden {fmt(gpre[d])} now {fmt(npre[d])}"))
                continue
            old, new = ga[f"v{v}.post.{arr}"], na[f"v{v}.post.{arr}"]
            if old.shape != new.shape:
                diffs.append(f"{arr} camera {v}: shape golden {old.shape} now {new.shape}")
                continue
            eo, ep = cell_eq(new, old), cell_eq(new, gpre)
            bad = ~eo & ~(hand & ep)
            if bad.any():
                f, c = (int(x) for x in np.argwhere(bad)[0])
                diffs.append(f"{arr} camera {v} frame {f} column {c} ({names[c] if c < len(names) else '?'}): "
                             f"golden {fmt(old[f, c])} now {fmt(new[f, c])} (pre-run {fmt(gpre[f, c])})"
                             f"  [{int(bad.sum())} cell(s)]")
            eo_all = eo if eo_all is None else eo_all & eo
            ep_all = ep if ep_all is None else ep_all & ep
        if eo_all is not None and hand.shape == eo_all.shape:
            kept += int((hand & ep_all & ~eo_all).sum())
            over += int((hand & eo_all & ~ep_all).sum())
            both += int((hand & eo_all & ep_all).sum())
            mix = hand & ~eo_all & ~ep_all
            mixed += int(mix.sum())
            if mix.any():
                f, c = (int(x) for x in np.argwhere(mix)[0])
                diffs.append(f"hand cell partly kept, partly overwritten: camera {v} frame {f} column {c} "
                             f"({names[c] if c < len(names) else '?'})  [{int(mix.sum())} cell(s)]")
        if gc["mask_hash_post"] != nc["mask_hash_post"]:
            diffs.append(f"camera {v}: silhouettes after the run differ")
    ge, ne = gm["end"], nm["end"]
    for key in ("current", "active"):
        if ge[key] != ne[key]:
            diffs.append(f"end state {key}: golden {ge[key]} now {ne[key]}")
    if len(ge["runs"]) != len(ne["runs"]):
        diffs.append(f"workers launched: golden {len(ge['runs'])} now {len(ne['runs'])}")
    for i, (a, b) in enumerate(zip(ge["runs"], ne["runs"])):
        for key in a:
            if a.get(key) != b.get(key):
                diffs.append(f"worker {i} {key}: golden {a.get(key)} now {b.get(key)}")
                break
    if not nm["undo"]["ok"]:
        diffs.append(f"one Ctrl+Z did not give the pre-run arrays back: enabled {nm['undo']['enabled']}, "
                     f"{nm['undo']['bad']}")
    if gm.get("modals") != nm.get("modals"):
        diffs.append(f"unexpected dialogs: golden {gm.get('modals')} now {nm.get('modals')}")
    if gm.get("buttons") != nm.get("buttons"):
        notes.append(f"window settings: golden {gm.get('buttons')} now {nm.get('buttons')}")
    if nm.get("choice_patched"):
        notes.append(f"Track pop-up answered {os.environ.get('HANDMIX_CHOICE', 'keep')!r}")
    summary = f"hand cells kept: {kept} / overwritten: {over}" + (f" / the same either way: {both}" if both else "")
    if mixed:
        summary += f" / mixed: {mixed}"
    return diffs, notes, summary


# ===================================================================== the parent
MATTERS = ("kinetrace/", "tests/_synth_spots.py")     # what the recorded outputs depend on (+ the media, hashed)


def code_sha():
    """(sha[-dirty], dirty, git status lines): dirty = a change, tracked or not, in the code under test
    (kinetrace/) or in the synthetic-scene helper this harness imports; other test files being edited
    beside it do not change what it records (the full status goes into the json)."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=60).stdout
        except Exception:      # noqa: BLE001
            return ""
    sha = git("rev-parse", "--short", "HEAD").strip() or "nogit"
    lines = [ln for ln in git("status", "--porcelain", "--untracked-files=all").splitlines() if ln.strip()]
    dirty = any(ln[3:].strip().strip('"').replace("\\", "/").startswith(MATTERS) for ln in lines)
    return sha + ("-dirty" if dirty else ""), dirty, lines


def code_id() -> str:
    """The code under test itself: git's tree hash of kinetrace/ + the helper's blob ("-dirty" with local
    changes). A commit that touches only other test files keeps it."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:      # noqa: BLE001
            return ""
    _sha, dirty, _lines = code_sha()
    return (git("rev-parse", "HEAD:kinetrace")[:12] + "/" + git("rev-parse", "HEAD:tests/_synth_spots.py")[:12]
            + ("-dirty" if dirty else ""))


def pick(only: str, cpu: bool) -> list:
    names = list(SCENARIOS)
    if only:
        want = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in want if n not in SCENARIOS]
        if unknown:
            sys.exit(f"unknown scenario(s): {', '.join(unknown)} (known: {', '.join(SCENARIOS)})")
        names = [n for n in names if n in want]
    if cpu:
        names = [n for n in names if not SCENARIOS[n].gpu]
    return names


def save_results(path_npz: str, results: dict, header: dict) -> None:
    arrays, scen = {}, {}
    for name, (arr, meta) in results.items():
        for k, v in arr.items():
            arrays[f"{name}{SEP}{k}"] = v
        scen[name] = meta
    np.savez_compressed(path_npz, **arrays)
    with open(path_npz[:-4] + ".json", "w", encoding="utf-8") as fh:
        json.dump(jsonable(dict(header, scenarios=scen)), fh, indent=1)


def load_results(path_npz: str) -> tuple[dict, dict]:
    with open(path_npz[:-4] + ".json", encoding="utf-8") as fh:
        js = json.load(fh)
    results = {}
    with np.load(path_npz) as z:
        by = {}
        for k in z.files:
            name, key = k.split(SEP, 1)
            by.setdefault(name, {})[key] = z[k]
    for name, meta in js["scenarios"].items():
        results[name] = (by.get(name, {}), meta)
    return results, js


def header(sha: str, dirty: bool, status: list, cid: str) -> dict:
    return {"sha": sha, "dirty": dirty, "code_id": cid, "git_status": status,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": ROOT,
            "harness_md5": md5_file(os.path.abspath(__file__)), "python": sys.executable}


def do_record(names: list, partial: bool) -> int:
    sha, dirty, status = code_sha()
    cid = code_id()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"golden_{sha}.npz")
    say(f"recording the golden for {sha} ({len(names)} scenario(s)) -> {path}")
    results = {}
    if partial and os.path.exists(path):
        results, _js = load_results(path)
        say(f"  (merging into the existing golden: {', '.join(results)})")
    t0 = time.time()
    for n in names:
        arr, meta = run_child(n)
        results[n] = (arr, meta)
        say(f"  -> {n}: {meta.get('status')} {meta.get('reason', '')} ({meta.get('seconds')} s)")
        if meta.get("status") == "ok" and SCENARIOS[n].layer == "A" and not meta["undo"]["ok"]:
            say(f"  WARNING {n}: Ctrl+Z did not restore the pre-run arrays in today's code: {meta['undo']}")
    order = [n for n in SCENARIOS if n in results] + [n for n in results if n not in SCENARIOS]
    cid2 = code_id()
    if cid2 != cid:                 # the code under test changed while the scenarios ran
        say(f"GOLDEN RECORD INVALID: the code under test changed during the record ({cid} -> {cid2}); "
            "nothing written")
        return 1
    if code_sha()[0] != sha:
        say(f"  note: HEAD moved during the record ({sha} -> {code_sha()[0]}), the code under test did not "
            f"({cid}); the file keeps the name of the start")
    save_results(path, {n: results[n] for n in order}, header(sha, dirty, status, cid))
    ok = [n for n in order if results[n][1].get("status") == "ok"]
    skipped = [n for n in order if results[n][1].get("status") == "SKIPPED"]
    errors = [n for n in order if results[n][1].get("status") not in ("ok", "SKIPPED")]
    for n in skipped:
        say(f"SKIPPED {n}: {results[n][1].get('reason')}")
    for n in errors:
        say(f"ERROR {n}: {results[n][1].get('reason')}")
    say(f"total {time.time() - t0:.0f} s; file {path}")
    if errors:
        say(f"GOLDEN RECORD INCOMPLETE: {len(ok)} recorded, {len(errors)} error(s): {', '.join(errors)}")
        return 1
    say(f"GOLDEN RECORDED {len(ok)} scenarios" + (f" ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""))
    return 0


def do_compare(path: str, names: list) -> int:
    if not os.path.exists(path):
        sys.exit(f"no such golden: {path}")
    golden, js = load_results(path)
    sha, dirty, status = code_sha()
    cid = code_id()
    say(f"comparing the code at {sha} with the golden of {js.get('sha')} ({path})")
    if js.get("code_id") == cid:
        say(f"  the code under test is the golden's own ({cid}): any difference is run-to-run non-determinism")
    else:
        say(f"  the code under test changed: golden {js.get('code_id')} now {cid}")
    if js.get("harness_md5") != md5_file(os.path.abspath(__file__)):
        say("  note: this harness file changed since the golden was recorded")
    fresh, same, differ, not_compared = {}, [], [], []
    t0 = time.time()
    for n in names:
        if n not in golden:
            say(f"  {n:6s} not in the golden: not compared")
            not_compared.append(n)
            continue
        gar, gm = golden[n]
        if gm.get("status") != "ok":
            say(f"  {n:6s} {gm.get('status')} in the golden ({gm.get('reason')}): not compared")
            not_compared.append(n)
            continue
        nar, nm = run_child(n)
        fresh[n] = (nar, nm)
        if nm.get("status") != "ok":
            differ.append(n)
            say(f"  {n:6s} DIFFERS: the run is {nm.get('status')}: {nm.get('reason')}")
            continue
        diffs, notes, summary = (cmp_w if SCENARIOS[n].layer == "W" else cmp_a)(gar, gm, nar, nm)
        if gm.get("inputs") != nm.get("inputs"):
            diffs.insert(0, f"the input media differ: golden {gm.get('inputs')} now {nm.get('inputs')}")
        genv, nenv = gm.get("env", {}), nm.get("env", {})
        if {k: genv.get(k) for k in ("torch", "gpu", "tracker_device")} != {k: nenv.get(k) for k in ("torch", "gpu", "tracker_device")}:
            notes.append(f"environment: golden {genv} now {nenv}")
        if diffs:
            differ.append(n)
            say(f"  {n:6s} DIFFERS: {diffs[0]}" + (f"   ({summary})" if summary else ""))
            for d in diffs[1:8]:
                say(f"           also: {d}")
            if len(diffs) > 8:
                say(f"           ... {len(diffs) - 8} more")
        else:
            same.append(n)
            say(f"  {n:6s} same" + (f"   ({summary})" if summary else ""))
        for t in notes:
            say(f"           note: {t}")
    save_results(os.path.join(OUT, f"last_compare_{sha}.npz"), fresh,
                 dict(header(sha, dirty, status, cid), golden=path))
    say(f"total {time.time() - t0:.0f} s")
    if differ:
        say(f"GOLDEN DIFFERS {len(differ)} of {len(same) + len(differ)} scenarios: {', '.join(differ)}"
            + (f" (not compared: {', '.join(not_compared)})" if not_compared else ""))
        return 1
    say(f"GOLDEN SAME {len(same)} scenarios" + (f" (not compared: {', '.join(not_compared)})" if not_compared else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="the hand-truth golden harness (see the docstring)")
    ap.add_argument("--record", action="store_true", help="record tests/out/handmix/golden_<sha>.npz")
    ap.add_argument("--compare", metavar="GOLDEN.npz", help="re-run every scenario and compare")
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    ap.add_argument("--cpu", action="store_true", help="only the scenarios that need no model")
    ap.add_argument("--list", action="store_true", help="list the scenarios")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--dump", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child_main(a.child, a.dump)
        return 0
    if a.list:
        for n, sc in SCENARIOS.items():
            say(f"{n:6s} {sc.layer} {'GPU' if sc.gpu else 'CPU'}  {sc.desc}")
        return 0
    names = pick(a.only, a.cpu)
    if a.record == bool(a.compare):
        ap.error("give --record or --compare FILE")
    if a.record:
        return do_record(names, partial=bool(a.only or a.cpu))
    return do_compare(os.path.abspath(a.compare), names)


if __name__ == "__main__":
    sys.exit(main())
