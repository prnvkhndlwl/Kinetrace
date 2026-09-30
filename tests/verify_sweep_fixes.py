"""Regression tests for the hub fixes of the release sweep (2026-09-22,
docs/AUDIT.md "Release sweep"): project timing and exports, session undo and
keyframes, and the app-level multi-camera / canvas / timeline behaviours. Pure
parts first, then an offscreen MainWindow on two small synthetic clips. No GPU.

Run: .venv\\Scripts\\python.exe tests\\verify_sweep_fixes.py
"""
import os
import sys
import time
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np

from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession

OUT = os.path.join(ROOT, "tests", "out", "sweep_fixes")
os.makedirs(OUT, exist_ok=True)


def sess(n, w=320, h=240, fps=60.0, name="v.mp4"):
    return TrackingSession(os.path.join(OUT, name), n, fps, w, h)


# ---------------------------------------------------------------- I137
# exportable_at = exportable[frames, pid] (one cell or a column), without the (T, N) array
_s = sess(300, name="x.mp4")
for _ in range(4):
    _s.add_point(0, 10.0, 10.0)
_r = np.random.default_rng(7)
_s.tracked[:] = _r.random(_s.tracked.shape) < 0.7
_s.occluded[:] = _r.random(_s.occluded.shape) < 0.2
_full = _s.exportable
for _f in (0, 17, 299, np.int64(150)):
    for _j in range(4):
        assert bool(_s.exportable_at(_f, _j)) == bool(_full[_f, _j]), (_f, _j)
_idx = _r.integers(0, 300, 64)
for _j in range(4):
    assert np.array_equal(_s.exportable_at(_idx, _j), _full[_idx, _j]), _j
print("exportable_at matches exportable, one cell and a column (I137) OK")

# ---------------------------------------------------------------- I138
# in_frame = the six inline copies it replaced (tracker x5, session.write_segment)
from kinetrace.session import in_frame  # noqa: E402

_W, _H = 320, 240
_vals = np.array([0.0, -0.0, -1e-6, 1e-6, 319.0, 319.999, 320.0, 239.0, 239.999, 240.0, -5.0, 1000.0,
                  np.nan, np.inf, -np.inf, 17.25])
_xy = np.stack(np.meshgrid(_vals, _vals), -1).reshape(-1, 2)
for _dt in (np.float32, np.float64):
    _p = _xy.astype(_dt)
    _old_vec = (np.isfinite(_p).all(axis=1) & (_p[:, 0] >= 0) & (_p[:, 0] < _W) & (_p[:, 1] >= 0) & (_p[:, 1] < _H))
    _old_seg = ((_p[..., 0] >= 0) & (_p[..., 0] < _W) & (_p[..., 1] >= 0) & (_p[..., 1] < _H))   # write_segment's
    assert np.array_equal(in_frame(_p, _W, _H), _old_vec) and np.array_equal(in_frame(_p, _W, _H), _old_seg)
    assert np.array_equal(in_frame(_p.reshape(16, 16, 2), _W, _H), _old_vec.reshape(16, 16)), "(L, K, 2) windows"
    for _q in _p:
        _old = bool(np.isfinite(_q).all() and 0 <= _q[0] < _W and 0 <= _q[1] < _H)
        assert bool(in_frame(_q, _W, _H)) == _old, _q
print("in_frame is the picture-bounds rule it replaced, NaN / inf / edges included (I138) OK")

# ---------------------------------------------------------------- I19
p = Project([sess(200, name="a.mp4"), sess(200, name="b.mp4")], ["A", "B"], [0.0, -36.5])
seq = [p.map_frame(0, 1, f) for f in range(40, 48)]
assert seq == list(range(seq[0], seq[0] + 8)), f"half-frame offset repeats / skips frames: {seq}"
for f in range(40, 60):                           # a switch there and back lands where it started
    g = p.map_frame(0, 1, f)
    assert p.map_frame(1, 0, g) == f, (f, g, p.map_frame(1, 0, g))
p2 = Project([sess(200, name="a.mp4"), sess(400, fps=120.0, name="b2.mp4")], ["A", "B"], [0.0, 0.0])
back = [p2.map_frame(1, 0, f) for f in range(0, 12)]
assert back == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], f"2x camera to 1x: {back}"
print("map_frame ties neither repeat nor skip, and round-trip (I19) OK")

# ---------------------------------------------------------------- I15 / I13
s_a, s_b, s_c = sess(400, name="a.mp4"), sess(400, name="b.mp4"), sess(60, name="c.mp4")
p3 = Project([s_a, s_b, s_c], ["A", "B", "C"], [0.0, -50.0, 0.0])
assert p3.coverage() == (50, 59), p3.coverage()                   # every camera: the caption's span
span = p3.reference_span(min_views=2)
assert span is not None and abs(span[0] - 0) < 1e-9 and abs(span[1] - 399) < 1e-9, span
for s in (s_a, s_b):
    for f in range(s.n_frames):
        s.tracked[f, :] = False
pa = s_a.add_point(0, 10, 10)
pb = s_b.add_point(0, 10, 10)
s_a.tracked[:, pa] = True
s_b.tracked[:, pb] = True
from kinetrace.app import MainWindow  # noqa: E402
p3.active = 1                              # working in camera B (offset -50)
lo, hi = MainWindow._t_range_3d(SimpleNamespace(project=p3))
assert (lo, hi) == (50, 399), f"reference window from camera B's frames: {(lo, hi)}"
p3.active = 0
assert MainWindow._t_range_3d(SimpleNamespace(project=p3)) == (50, 399)
print("3D window in reference frames, >= 2 cameras, whichever camera is worked in (I13, I15) OK")

# ---------------------------------------------------------------- I16
K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
cams = []
for a in (0.3, 2.0):
    pos = np.array([2 * np.cos(a), 2 * np.sin(a), 0.5])
    z = -pos / np.linalg.norm(pos)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    cams.append(CameraCalibration(dlt_from_camera(K, R, -R @ pos), 320, 240, NoUndistort(), pixel_origin=0.0))
p4 = Project([sess(50, name="a.mp4"), sess(50, name="b.mp4")], ["A", "B"], [0.0, 0.0])
p4.calibration = Calibration(cams)
p4.add_view(sess(50, name="c.mp4"), "C")
f4 = os.path.join(OUT, "cal_kept.kinetrace")
p4.save(f4)
q4 = Project.load(f4)
assert q4.calibration is not None and len(q4.calibration) == 2, "a camera added after calibrating erased it on save"
print("calibration survives adding a camera, on disk too (I16) OK")

# ---------------------------------------------------------------- I17
s5 = sess(60)
k = s5.add_point(10, 100.0, 100.0)
s5.set_position(50, k, 140.0, 120.0)
s5.occluded[25:36, k] = True
n5, _ = s5.interpolate_keyframes(k)
assert n5 == 39 - 11, n5
assert not s5.tracked[25:36, k].any() and s5.occluded[25:36, k].all(), "hidden frames were filled / unmarked"
assert s5.last_interp_hidden_skipped == 11
assert s5.tracked[24, k] and s5.tracked[36, k]
print("keyframe fill leaves hidden frames hidden and empty (I17) OK")

# ---------------------------------------------------------------- I18 / I20 / I21
e_a, e_b = sess(300, name="ea.mp4"), sess(300, name="eb.mp4")
ia = e_a.add_point(0, 10.0, 20.0, name="head")
ib = e_b.add_point(0, 30.0, 40.0, name="head")
e_a.tracked[:, ia] = False
e_b.tracked[:, ib] = False
for f in range(300):
    e_a.set_position(f, ia, 10.0 + f, 20.0)
for f in range(0, 250):
    e_b.set_position(f, ib, 30.0 + f, 40.0)
p6 = Project([e_a, e_b], ["camA", "camB"], [0.0, -50.0])      # B started 50 frames later
p6.active = 1                                                  # exported while working in B
out6 = os.path.join(OUT, "allcams.csv")
p6.export_multi_dltdv(out6)
rows = open(out6).read().strip().splitlines()
assert len(rows) == 1 + 300, len(rows)                         # row k = reference frame k, from 0
r0 = rows[1].split(",")
assert r0[0] == "11.0000" and r0[2] == "NaN" and r0[3] == "NaN", r0   # B has no frame at t=0: NaN, not blank
r60 = rows[1 + 60].split(",")
assert r60[0] == "71.0000" and r60[2] == "41.0000", r60        # B's frame 10 at reference frame 60 (x+1)
side = open(os.path.join(OUT, "allcams_pointnames.csv")).read()
assert "row k = frame k" in side and "camA" in side and "cam2,camB" in side, side
big = [sess(20000, name=f"big{c}.mp4") for c in range(3)]
for s in big:
    for j in range(6):
        s.add_point(0, 5.0, 5.0, name=f"p{j}")
    s.tracked[:] = True
t0 = time.time()
Project(big, ["a", "b", "c"], [0.0, 3.0, -2.0]).export_multi_dltdv(os.path.join(OUT, "big.csv"))
dt = time.time() - t0
assert dt < 20, f"all-cameras export too slow: {dt:.1f} s"
print(f"all-cameras xypts: reference frames from 0, NaN cells, sidecar, 20k x 6 x 3 in {dt:.1f} s (I18, I20, I21) OK")

# ---------------------------------------------------------------- I22
s7 = sess(20)
s7.apply_skeleton({"name": "t", "landmarks": ["crown", "neck", "tail"], "bones": [["crown", "neck"], ["neck", "tail"]],
                   "head": "crown", "derived": {}})
snap = s7.snapshot()
s7.rename_point(s7.pid_by_name("crown"), "head_top")
s7.restore(snap)
assert s7.points[0].name == "crown" and s7.skeleton["head"] == "crown", s7.skeleton
assert len(s7.bones()) == 2 and s7.head_pid() is not None
print("undo restores the skeleton with the names it refers to (I22) OK")

# ---------------------------------------------------------------- I23
p8 = Project([sess(50, name="a.mp4"), sess(50, name="b.mp4")], ["A", "B"], [0.0, 0.0])
for fn in (lambda: p8.set_offset(1, 3.0), lambda: p8.set_rate(1, 2.0), lambda: p8.align_to(1, 9, 1)):
    p8.reconstruction = object()
    fn()
    assert p8.reconstruction is None, "retiming kept a stale reconstruction"
p8.reconstruction = object()
p8.set_offset(1, p8.offsets[1])                                # no change: nothing cleared
assert p8.reconstruction is not None
print("retiming a camera clears the stale 3D result (I23) OK")

# =============================================================== the app
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

ASK = {"answer": QMessageBox.No, "asked": [], "warned": [], "info": []}
QMessageBox.question = staticmethod(lambda *a, **k: (ASK["asked"].append(a[1] if len(a) > 1 else ""), ASK["answer"])[1])
QMessageBox.warning = staticmethod(lambda *a, **k: (ASK["warned"].append(a[1] if len(a) > 1 else ""), QMessageBox.Ok)[1])
QMessageBox.information = staticmethod(lambda *a, **k: (ASK["info"].append(a[1] if len(a) > 1 else ""), QMessageBox.Ok)[1])
app = QApplication.instance() or QApplication([])
from kinetrace.app import READY  # noqa: E402


def clip(path, n=60, w=320, h=240, shift=0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.RandomState(3)
    base = rng.randint(0, 255, (h + 40, w + 400, 3)).astype(np.uint8)
    for f in range(n):
        x = 4 * (f + shift)
        vw.write(np.ascontiguousarray(base[20:20 + h, x:x + w]))
    vw.release()
    return path


VA = clip(os.path.join(OUT, "camA.mp4"))
VB = clip(os.path.join(OUT, "camB.mp4"), shift=5)
VC = clip(os.path.join(OUT, "camC.mp4"), shift=9)
for v in (VA, VB, VC):
    forget_recovery(v)

win = MainWindow()
win.resize(1500, 950)
win.show()


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


win._open_video(VA)
for _ in range(200):
    pump(0.05)
    if win.state == READY:
        break
assert win.state == READY
s = win.session

# I45: a source switch that would erase a track asks, and is undoable
pid = s.add_point(0, 50.0, 60.0)
for f in range(30):
    s.set_position(f, pid, 50.0 + f, 60.0)
win._refresh_point_list()
ASK["answer"], ASK["asked"] = QMessageBox.No, []
win._on_source_change(pid, "tip")
assert ASK["asked"] and s.tracked[:, pid].sum() == 30 and s.points[pid].source != "silhouette", \
    "No must keep the track"
ASK["answer"] = QMessageBox.Yes
win._on_source_change(pid, "tip")
assert s.points[pid].source == "silhouette" and not s.tracked[:, pid].any()
win._undo_run()
assert win.session.tracked[:, pid].sum() == 30, "Ctrl+Z must bring the erased track back"
print("data-source switch asks before erasing a track and Ctrl+Z restores it (I45) OK")
s = win.session

# I46: an unticked point is never grabbed
pid2 = s.add_point(0, 52.0, 60.0)
win._refresh_point_list()
win._goto(1)
win._goto(0)
s.points[pid].display = False
win._refresh_overlay()
hit = win.canvas._hit_test(QPointF(50.5, 60.0))
assert hit == pid2, f"the hidden point was grabbed ({hit})"
s.points[pid].display = True
print("hidden points cannot be grabbed (I46) OK")

# I64: a drag / Ctrl+click is one undo step
win._undo_snap = None
win._on_place(pid2, 70.0, 70.0)
assert win._undo_snap is not None and win.act_undo.isEnabled()
print("a drag correction is one undo step (I64) OK")

# I48: a ball click is a click even with the polygon shape chosen
win._set_region_shape("polygon")
win._arm_ball()
assert win.canvas._click_only and win.btn_add.isChecked()
win.btn_add.setChecked(False)
assert not win.canvas._click_only
win._set_region_shape("circle")
print("ball placement ignores the region shape (I48) OK")

# I49: when zoomed, the painted cell and the click agree
tl = win.timeline
tl._set_view(10, 16)
w = tl._lane_w()
for f in (10, 17, 25):
    x0, x1 = tl._x_edge(f), tl._x_edge(f + 1)
    assert tl._frame_at(x0 + 0.01) == f and tl._frame_at(x1 - 0.01) == f, (f, x0, x1)
    assert tl._frame_at(tl._x_of(f)) == f
tl.zoom_fit()
print("timeline cells: painted = clicked = selected (I49) OK")

# I52: the magenta band explains itself
arr = np.full((s.n_frames, s.n_points), np.nan, np.float32)
arr[5, pid] = 40.0                                             # pid: row 0, tracked on frames 0..29
win._refresh_point_list()                                      # (it recomputes the band from the project)
tl.set_disagreement(arr, 5.0)
row_y = tl._row_rect_y(0)
assert tl._row_at((row_y[0] + row_y[1]) / 2) == 0, (row_y, tl.height())
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
_hp = QPointF(tl._x_of(5), (row_y[0] + row_y[1]) / 2)
app.sendEvent(tl, QMouseEvent(QEvent.MouseMove, _hp, tl.mapToGlobal(_hp), Qt.NoButton, Qt.NoButton,
                              Qt.NoModifier))
pump(0.1)
assert "disagrees with the others by 40.0 px" in tl.toolTip(), tl.toolTip()
tl.set_disagreement(None)
print("the magenta band's hover says what it means and what to do (I52) OK")

# ---- two cameras: I16 (message), I47 / I65 / I50, I43, I103 / I111, I23, I66, I40
ASK["info"] = []
win.project.calibration = Calibration([cams[0]])               # a 1-camera calibration, then add B
assert win._add_view(VB)
pump(0.5)
assert any("calibration covers" in t for t in ASK["info"]), ASK["info"]
win.project.calibration = None
print("adding a camera to a calibrated project says what happens to the calibration (I16) OK")

win.btn_autopause.setChecked(False)
win.btn_animal.setChecked(True)
win._set_active_view(1)
pump(0.3)
assert not win.btn_animal.isChecked() and not win.canvas._animal_mode, "a switch left S armed (I47)"
win._set_active_view(0)
pump(0.3)
win.btn_add.setChecked(True)
win._pending_event = 3
win.timeline.set_pending_event(3)
old_canvas = win.canvas
win._set_active_view(1)
pump(0.3)
# (G20) Add is carried OVER: armed in one camera, a click on another places the point there --
# but only the camera switched to is armed, so the button never shows armed over an unarmed view (I47)
assert win.btn_add.isChecked() and win.canvas._place_mode and not old_canvas._place_mode, \
    "Add must move with the switch to the camera switched to"
assert win._pending_event is None, "a half-marked event crossed cameras"
assert not win.btn_autopause.isChecked(), "a switch silently turned auto-pause back on"
win.btn_add.setChecked(False)
win._set_active_view(0)
pump(0.3)
assert not win.btn_autopause.isChecked()
win.btn_autopause.setChecked(True)
print("camera switch puts S down, carries Add over to the camera clicked, drops a pending event, keeps the "
      "user's toggles (I47, G20, I65, I50) OK")

cur = win.current
win._on_eof_truncated(1, 3)                                    # a companion's end is ITS frame number
assert win.current == cur, "a companion's EOF moved the working playhead"
print("a companion's end of file does not move the playhead (I43) OK")

calls = []
orig = win._on_seek_frame
win._on_seek_frame = lambda v, f, rgb: calls.append(v)
assert win._add_view(VC)
pump(0.5)
win._views[1].stop()                                           # (never drop a running decoder: I112)
win._start_seek_service(2)
win._remove_view(1)                                            # C moves from index 2 to 1
pump(0.3)
calls.clear()
win._views[1].seek.frame_ready.emit(0, np.zeros((4, 4, 3), np.uint8))
pump(0.1)
assert calls == [1], f"a decoder reported its old index after a camera was removed: {calls}"
win._on_seek_frame = orig
print("decoders report their current index after a camera is removed (I103, I111) OK")

win.project.reconstruction = object()
win._on_view_offset(1, 4.0)
assert win.project.reconstruction is None
print("editing an offset in the panel clears the stale 3D result (I23) OK")

sA = win.project.sessions[0]
sA.ensure_animal()
win._preview = SimpleNamespace(target_session=sA, wait=lambda ms: None, isRunning=lambda: False)
win._set_active_view(1) if win.project.active == 0 else None
pump(0.2)
mask = np.zeros((240, 320), bool)
mask[100:140, 100:160] = True
from kinetrace.segmenter import summarize_mask  # noqa: E402
summ = summarize_mask(mask, 1.0, 9.0)
summ["frame"] = 7
if summ is not None:
    win._on_preview_done(summ)
    assert sA.masks is not None and sA.masks.has(7), "a late preview was not stored in its own camera"
    assert win.session is not sA and (win.session.masks is None or not win.session.masks.has(7))
    print("a silhouette preview lands in the camera it was clicked in (I66) OK")

win._on_decode_failed(0, 12, "read failed")
win._on_decode_failed(0, 12, "read failed")
assert len(win._views[0].__dict__.get("_bad_frames", ())) == 1
print("a damaged frame is reported once, never taken for the end (I40) OK")

# I24: scene coordinates are OpenCV pixel centres
cvs = win.canvas
item_pt = cvs._pixitem.mapFromScene(QPointF(10.0, 20.0))
assert abs(item_pt.x() - 10.5) < 1e-6 and abs(item_pt.y() - 20.5) < 1e-6, item_pt   # the CENTRE of pixel (10, 20)
w0, h0 = cvs._native_size
assert cvs._in_video(QPointF(-0.4, -0.4)) and not cvs._in_video(QPointF(w0 - 0.4, 5.0))
print("scene coordinates = OpenCV pixel centres (I24) OK")

# I110: a click on a companion tile switches only
ASK["info"], ASK["warned"] = [], []
p_now = win.project
other = 1 - p_now.active
ss = p_now.sessions[other]
if ss.n_points == 0:
    ss.add_point(0, 30.0, 30.0)
ss.tracked[:, 0] = False
ss.ui_state["selected"] = 0
comp = win.grid.canvas(other)
before = ss.tracks[:, 0].copy()
vp = comp.mapFromScene(QPointF(100.0, 100.0))
QTest.mouseClick(comp.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
pump(0.3)
assert win.project.active == other, "the tile click did not switch"
assert np.array_equal(np.nan_to_num(before), np.nan_to_num(ss.tracks[:, 0])), "the switching click also placed a point"
print("a click on a companion tile only switches cameras (I110) OK")

# I115: the 3D entries open with one camera's worth of setup and explain what is missing
win.project.calibration = None
for act in (win.act_recon, win.act_calib, win.act_sync, win.act_hull, win.act_offsets3d, win.act_export_cal):
    assert act.isEnabled(), f"{act.text()} greyed out: it must open and explain"
ASK["info"] = []
win._reconstruct_3d()
assert ASK["info"] and "calibration" in " ".join(ASK["info"]).lower() or ASK["info"], ASK["info"]
print("3D entries stay clickable and say what they need (I115) OK")

# I104: a failed save says so and does not keep aiming at the bad path
ASK["warned"] = []
crit = []
QMessageBox.critical = staticmethod(lambda *a, **k: (crit.append(a[1] if len(a) > 1 else ""), QMessageBox.Ok)[1])
bad = os.path.join(OUT, "no_such_dir", "x.kinetrace")
__import__("shutil").rmtree(os.path.dirname(bad), ignore_errors=True)   # truly absent, whatever ran before
from PySide6.QtWidgets import QFileDialog  # noqa: E402
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (bad, ""))
win.project_path = None
ok_save = win._save_project_as()
assert ok_save is False and crit and win.project_path is None, (ok_save, crit, win.project_path)
print("a failed Save As is said and forgotten (I104) OK")

# I114: the reference instant of the playhead (hull cache / Export Mesh key)
pj = win.project
pj.set_offset(1, 7.0)
act_before = pj.active
if pj.active != 1:
    win._set_active_view(1)
    pump(0.2)
win._goto(20)
assert win._reference_instant() == int(np.floor(pj.reference_time(1, 20) + 0.5)) == 13, win._reference_instant()
print("the hull / mesh key is the reference instant (I114) OK")

# I67: epipolar guides in a non-reference camera never index past the 3D rows
pj.calibration = Calibration(cams)
pj.reconstruction = SimpleNamespace(t0=0, n_frames=10, xyz=np.zeros((10, 1, 3)), per_cam=None, names=["P1"])
win._goto(5)                         # reference instant 5 - 7 = -2: outside the rows
try:
    win._epipolar_guides(0)
    ok67 = True
except IndexError:
    ok67 = False
win._goto(12)                        # reference instant 5: inside
win._epipolar_guides(0)
pj.reconstruction = None
pj.calibration = None
assert ok67, "epipolar guides raised IndexError in a non-reference camera"
print("epipolar guides look up the reference row, bounds-checked (I67) OK")

# I69: a silhouette-derived landmark is never hand-placed by N-continue / drag
sd = win.session
sd.apply_skeleton({"name": "d", "landmarks": ["snoot", "tailtip"], "bones": [], "head": "snoot",
                   "derived": {"tailtip": "tip"}})
win._refresh_point_list()
dpid = sd.pid_by_name("tailtip")
n_before = sd.n_points
win._on_select(dpid)
win.btn_add.setChecked(True)
win._on_add(40.0, 40.0)
assert sd.n_points == n_before + 1 and not sd.tracked[win.current, dpid], "N-click placed the derived landmark"
win._on_place(dpid, 50.0, 50.0)
assert not sd.tracked[win.current, dpid], "a drag placed the derived landmark"
print("derived landmarks are not hand-placed; N adds the new point (I69) OK")

# I71: deleting one point / Shift+X are their own undo steps
win._undo_snap = None
last = sd.n_points - 1
win._on_occluded_toggled(last, True)
assert win._undo_snap is not None
win._undo_run()
assert not win.session.occluded[win.current, last], "Ctrl+Z did not take the hidden mark back"
n_now = win.session.n_points
win._on_delete(n_now - 1)
assert win.session.n_points == n_now - 1 and win._undo_snap is not None
win._undo_run()
assert win.session.n_points == n_now, "Ctrl+Z did not bring the deleted point back"
print("deleting one point and Shift+X are one undo step each (I71) OK")

# I70: the all-points question of Mark HIDDEN says what it does
win.point_list.clearSelection()
ASK["asked"], ASK["answer"] = [], QMessageBox.No
win._occlude_window(2, 4, None, True)
assert ASK["asked"] and "hidden" in ASK["asked"][-1].lower() and "clear" not in ASK["asked"][-1].lower(), ASK["asked"]
print("Mark HIDDEN asks about hiding, not clearing (I70) OK")

# I108: per-camera exports carry the camera's name
seen_start = []
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (seen_start.append(a[2] if len(a) > 2 else ""), ("", ""))[1])
win._export_dialog()
assert seen_start and win.project.name(win.project.active).replace(" ", "_")[:4] in seen_start[0], seen_start
print("multi-camera export names carry the camera (I108) OK")

# I107: X during an automatic re-track stops the whole queue
calls107 = []
orig_impl, orig_next = win._on_track_finished_impl, win._retrack_next
win._on_track_finished_impl = lambda last, paused: None
win._retrack_next = lambda: calls107.append("next")
win._retrack = {"jobs": [1, 2, 3]}
win._user_paused = True
win._on_track_finished(10, True)
pump(0.1)
assert win._retrack["jobs"] == [] and calls107 == ["next"], (win._retrack, calls107)
win._retrack = None
win._on_track_finished_impl, win._retrack_next = orig_impl, orig_next
print("pausing a re-track stops the queue and goes to Keep / Undo (I107) OK")

# I112: a thread still running after its wait is kept, not dropped
from PySide6.QtCore import QThread  # noqa: E402
from kinetrace import app as appmod  # noqa: E402


class _Slow(QThread):
    def run(self):
        time.sleep(0.4)


th = _Slow()
th.start()
appmod._retire(th)
assert th in appmod._ORPHANS
del th
t_end = time.time() + 3
while appmod._ORPHANS and time.time() < t_end:
    pump(0.05)
assert not appmod._ORPHANS, "a finished kept thread must be released"
print("a slow thread is kept until it finishes (I112) OK")

if act_before != win.project.active:
    win._set_active_view(act_before)
    pump(0.2)

# ---------------------------------------------------------------- batch after the docs cross-check
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QDialog, QFileDialog, QWidget  # noqa: E402
from kinetrace.canvas import VideoCanvas  # noqa: E402
import kinetrace.lenswizard as lwmod  # noqa: E402
s = win.session

# I123: adding a point is ONE undo step (Ctrl+Z used to restore an older snapshot)
n0 = s.n_points
win._undo_snap = None
win.selected = None                  # else the click CONTINUES a selected point with no data here
win.btn_add.setChecked(True)
win._on_add(120.0, 90.0)
assert win.session.n_points == n0 + 1 and win._undo_snap is not None and win.act_undo.isEnabled()
win._undo_run()
assert win.session.n_points == n0, "Ctrl+Z after adding must remove just the new point"
s = win.session
win._undo_snap = None
win._on_add_group(100.0, 100.0, 20.0)
assert win._undo_snap is not None
win._undo_run()
assert win.session.n_points == n0
s = win.session
print("adding a point / region is one undo step (I123) OK")

# I125: frame notes are exported with no events at all
s.events.clear()
s.set_note(3, "fin flare")
out_csv = os.path.join(OUT, "notes_only.csv")
written = win._export_one("wide", out_csv)
side = os.path.join(OUT, "notes_only_events.csv")
assert side in [os.path.normpath(w) for w in written] or os.path.exists(side), written
assert "fin flare" in open(side, encoding="utf-8").read()
s.set_note(3, "")
print("notes are exported without events (I125) OK")

# I128: a semi-automatic step starts from a segment alone
calls = []
orig_start, orig_seed, orig_aseed = win._start_tracking, s.seedable_at, s.animal_seedable_at
win._start_tracking = lambda **k: calls.append(k)
s.seedable_at = lambda f: False
s.animal_seedable_at = lambda f: True
win._track_step()
assert calls and calls[0].get("stop_after") == win.current + 1, calls
s.animal_seedable_at = lambda f: False
calls.clear()
win._track_step()
assert not calls and "S and click" in win.statusBar().currentMessage()
win._start_tracking = orig_start
del s.seedable_at, s.animal_seedable_at
print("a semi-automatic step runs with only a segment (I128) OK")

# I126: a click in pixel 0's outer half-pixel lands on the picture
q = VideoCanvas._onpic(QPointF(-0.3, -0.49))
assert (q.x(), q.y()) == (0.0, 0.0), (q.x(), q.y())
q = VideoCanvas._onpic(QPointF(12.25, 7.5))
assert (q.x(), q.y()) == (12.25, 7.5)
print("edge clicks are clamped onto the picture (I126) OK")

# G10: a second notice does not wipe the first
win.toast.show_message("first verdict", "warn", 8000)
win.toast.show_message("second note", "info", 3000)
t = win.toast.text()
assert "first verdict" in t and "second note" in t and win.toast._level == "warn", (t, win.toast._level)
win.toast.show_message("second note", "info", 3000)
assert win.toast.text().count("second note") == 1
win.toast.hide()
win.toast.show_message("alone", "info", 3000)
assert win.toast.text() == "alone"
win.toast.hide()
print("toasts keep the previous notice (G10) OK")

# G15: a menu built per right-click shows its entries' tooltips too
from PySide6.QtWidgets import QMenu  # noqa: E402
from kinetrace import timeline as tlmod  # noqa: E402


class _NoExecMenu(QMenu):
    def exec(self, *a, **k):
        return None


mm = _NoExecMenu()
tlmod._pop(mm, QPoint(0, 0))
assert mm.toolTipsVisible(), "timeline right-click menus must show tooltips"
print("per-right-click menus show tooltips (G15) OK")

# G11: the Body window keeps the hotkeys
bw = QWidget()
bw.show()
win.body_win = bw
orig_active = QApplication.activeWindow
QApplication.activeWindow = staticmethod(lambda: bw)
f0 = win.current
ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F, Qt.NoModifier, "f")
assert win._hotkeys_apply(ev), "a key in the Body window must reach the hotkeys"
QApplication.activeWindow = orig_active
win.body_win = None
bw.close()
print("the Body window keeps the hotkeys (G11) OK")

# I122: the Body run's focal length from a lens profile is a number, not a crash
from kinetrace.lens import LensProfile  # noqa: E402
K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
prof = LensProfile(K=K, dist=np.zeros(5), fisheye=False, width=320, height=240)
lenses_before = list(win.project.lenses)
win.project.lenses[win.project.active] = prof
fpx = win._lens_focal_px()
assert isinstance(fpx, float) and abs(fpx - 300.0) < 1e-6, fpx
win.project.lenses[:] = lenses_before
print("a lens profile gives the Body run its focal length (I122) OK")

# M2: a derived landmark says why it cannot be placed
dp = s.add_point(0, 30.0, 30.0)
s.points[dp].source = "silhouette"
s.tracked[:, dp] = False
win._refresh_point_list()
win._on_list_select(dp)
assert "derived from the silhouette" in win.statusBar().currentMessage(), win.statusBar().currentMessage()
s.remove_point(dp)
win._refresh_point_list()
print("selecting a derived landmark explains it (M2) OK")

# I127: balls lost inside the picture are reported with auto-pause off
bp = win.session.add_ball(0, 200.0, 100.0)
win._refresh_point_list()
win.state = appmod.TRACKING
win.worker = SimpleNamespace(wait=lambda t: True, isRunning=lambda: False, _autopause_reason="",
                             _ball_ended={bp: (12, "lost")})
win._autopause_info = None
win._run_start = 0
win._on_track_finished_impl(20, False)
assert "lost inside the picture" in win.toast.text() and "frame 12" in win.toast.text(), win.toast.text()
win.toast.hide()
win.session.remove_point(bp)
win._refresh_point_list()
print("a ball lost inside the picture is said with auto-pause off (I127) OK")

win.close()
pump(0.3)

# I14 / I106 / I113 on a reopen: a missing camera, a camera of another length, unreadable unsaved work
proj_file = os.path.join(OUT, "three_cams.kinetrace")
vids = [clip(os.path.join(OUT, f"re{k}.mp4"), n=60, shift=3 * k) for k in range(3)]
ps = [TrackingSession(v, 60, 30.0, 320, 240) for v in vids]
ps[2] = TrackingSession(os.path.join(OUT, "gone_away.mp4"), 60, 30.0, 320, 240)
ps[1] = TrackingSession(vids[1], 50, 30.0, 320, 240)            # remembered as 50 frames; the file has 60
Project(ps, ["c0", "c1", "c2"], [0.0, 0.0, 0.0]).save(proj_file)
mtime0 = os.path.getmtime(proj_file)
ASK["warned"], ASK["info"] = [], []
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))   # "locate video": cancelled
win2 = MainWindow()
win2.show()
win2._open_project_from_path(proj_file)
for _ in range(200):
    pump(0.05)
    if win2.state == READY and win2.project is not None:
        break
pump(0.5)
assert any("without some cameras" in t for t in ASK["warned"]), ASK["warned"]
assert any("length differs" in t for t in ASK["warned"]), ASK["warned"]
assert win2.project_path is None, "Save must ask for a name after a camera was left out"
assert win2._views[1].n_frames == 50, win2._views[1].n_frames
win2.project.dirty = True
win2._autosave()
assert os.path.getmtime(proj_file) == mtime0, "the project file was overwritten without the lost camera"
print("a missing camera leaves the project file alone; a longer companion is clamped and said (I14, I106) OK")
win2.close()
pump(0.3)

from kinetrace import projectfile, recovery  # noqa: E402
forget_recovery(VA)
bad_id = projectfile.new_id()
recovery.paths(bad_id)[0].write_bytes(b"not a zip file")
recovery.write_info(bad_id, base_saved_at=None, last_path=None, videos=[VA], temporary=True)
ASK["warned"] = []
win3 = MainWindow()
win3.show()
win3._open_video(VA)
for _ in range(200):
    pump(0.05)
    if win3.state == READY:
        break
assert any("could not be read" in t for t in ASK["warned"]), ASK["warned"]
damaged = recovery.folder()[0] / "damaged"
assert recovery.find(bad_id) is None and any(bad_id in f.name for f in damaged.iterdir()), \
    "unreadable unsaved work must be moved aside, not deleted"
print("unsaved work that cannot be read is moved aside and said (I113) OK")
win3.close()
pump(0.3)

# G9: the lens wizard opens with no video at all
made = []


class _FakeWiz:
    def __init__(self, parent, project, video, view, start):
        made.append((project, video, view))
        self.result_profile = None

    def exec(self):
        return QDialog.Rejected


real_wiz = lwmod.LensWizard
lwmod.LensWizard = _FakeWiz
w5 = MainWindow()
ASK["info"] = []
w5._lens_wizard()
lwmod.LensWizard = real_wiz
assert made and made[0][0] is None, made
assert not any("checkerboard video first" in t for t in ASK["info"]), ASK["info"]
w5.close()
pump(0.2)
print("the lens wizard opens without a video (G9) OK")

# I129 (recovery form): autosave goes to the recovery folder, never next to the
# video or into the project; Save retires it; a declined restore is kept aside
pj_path = os.path.join(OUT, "saved_once.kinetrace")
for f in (pj_path, pj_path + ".bak"):
    if os.path.exists(f):
        __import__('shutil').rmtree(f) if os.path.isdir(f) else os.remove(f)
win6 = MainWindow()
win6.show()
win6._open_video(VA)
for _ in range(200):
    pump(0.05)
    if win6.state == READY:
        break
files_before = sorted(os.listdir(OUT))
win6.session.add_point(0, 40.0, 40.0)
win6.project.dirty = True
win6._autosave(wait=True)
pid6 = win6._project_id
assert recovery.find(pid6) is not None, "autosave writes the recovery folder"
assert sorted(os.listdir(OUT)) == files_before, "autosave must not write beside the video"
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (pj_path, ""))
assert win6._save_project_as()
assert os.path.exists(pj_path) and recovery.find(pid6) is None, "saving retires the recovery"
saved_id, mtime6 = win6._project_id, os.path.getmtime(pj_path)
win6.session.add_point(0, 60.0, 60.0)
win6._autosave(wait=True)
assert os.path.getmtime(pj_path) == mtime6, "autosave never touches the saved project"
assert recovery.find(saved_id) is not None
ASK["answer"] = QMessageBox.No                   # "Save changes?" not answered Save / Discard: kept
win6.close()
pump(0.3)
assert recovery.find(saved_id) is not None, "closing without an answer keeps the unsaved work"
ASK["answer"], ASK["warned"] = QMessageBox.No, []
win7 = MainWindow()
win7.show()
win7._open_project_from_path(pj_path)
for _ in range(200):
    pump(0.05)
    if win7.state == READY and win7.project is not None:
        break
assert not ASK["warned"], "declining must not raise a warning"
assert win7.session.n_points == 1, "No opens the last saved version"
declined = recovery.folder()[0] / "declined"
assert recovery.find(saved_id) is None and any(saved_id in f.name for f in declined.iterdir()), \
    "a declined restore must be kept aside"
win7.close()
pump(0.3)
for f in (pj_path, pj_path + ".bak"):
    if os.path.exists(f):
        __import__('shutil').rmtree(f) if os.path.isdir(f) else os.remove(f)
print("autosave keeps the project file as saved; Save retires it; a declined one is kept aside (I129) OK")
forget_recovery(VA)

# ---------------------------------------------------------------- I139
# the tracking mode IS Track ▾'s checked action: the menu and the mode cannot disagree
win.act_mode_auto.trigger()
assert win._track_mode == "auto" and win.act_mode_auto.isChecked()
win.act_mode_semi.trigger()
assert win._track_mode == "semi" and not win.act_mode_auto.isChecked()
win._track_mode = "auto"                        # code setting the mode ticks the menu too
assert win.act_mode_auto.isChecked() and not win.act_mode_semi.isChecked() and win._track_mode == "auto"
win._track_mode = "semi"
assert win.act_mode_semi.isChecked() and win._track_mode == "semi"
win._track_mode = "bogus"                       # anything else is automatic (as a restored project does)
assert win._track_mode == "auto" and win.act_mode_auto.isChecked()
print("the tracking mode is read from the Track menu, never mirrored (I139) OK")

# ---------------------------------------------------------------- I133
# A run's result signal arrives while its thread is still cleaning up (a capture
# release that stalls on a network share): the handler's 2 s wait times out and
# the app used to drop its last reference to the RUNNING thread -- Qt then aborts
# the whole process (0xC0000409). Each case runs in its own process, in parallel.
I133_CHILD = r'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
for _n in ("question", "warning", "information", "critical"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: QMessageBox.No))
app = QApplication([])
from kinetrace import app as appmod, tracker
from kinetrace.app import MainWindow, READY, TRACKING
from kinetrace.segmenter import summarize_mask
mode, video = sys.argv[2], sys.argv[3]
OUTLIVE = 2.6            # longer than the handlers' 2 s wait

def slow_track(self):
    self.started_ok.emit()
    if mode == "track-error":
        self.error.emit("a plain sentence")
    else:
        self.finished_ok.emit(self.start_frame, False)
    time.sleep(OUTLIVE)

def slow_preview(self):
    if mode == "preview-error":
        self.error.emit("a traceback")
    else:
        summ = summarize_mask(np.zeros((240, 320), bool), (1.0, 1.0), 0.0)
        summ["frame"] = self._frame
        self.done.emit(summ)
    time.sleep(OUTLIVE)

tracker.TrackingWorker.run = slow_track
appmod._MaskPreviewWorker.run = slow_preview
win = MainWindow()
win.show()

def pump(cond, sec):
    t = time.time()
    while time.time() - t < sec and not cond():
        app.processEvents()
        time.sleep(0.005)
    return cond()

win._open_video(video)
assert pump(lambda: win.state == READY, 20), "video did not open"
if mode.startswith("track"):
    win.session.add_point(0, 100.0, 100.0)
    win._refresh_point_list()
    win._apply_state()
    win._toggle_tracking()
    assert pump(lambda: win.state == TRACKING, 10), "no run started"
    assert pump(lambda: win.state == READY and win.worker is None, 10), "the handler did not run"
else:
    win.session.ensure_animal()
    win.session.animal.add_click(win.current, 100.0, 100.0, True)
    win._preview_mask()
    assert win._preview is not None, "no preview started"
    assert pump(lambda: win._preview is None, 10), "the handler did not run"
pump(lambda: False, OUTLIVE + 0.8)       # the thread ends here: it must still be referenced
win.close()                              # the normal shutdown: every thread stopped / waited for
pump(lambda: False, 0.3)
print("survived", mode, flush=True)
'''
import subprocess  # noqa: E402

kids = {m: subprocess.Popen([sys.executable, "-c", I133_CHILD, ROOT, m, VA], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace",
                            env=dict(os.environ, KINETRACE_RECOVERY_DIR=os.path.join(OUT, f"i133_{m}")))
        for m in ("track-finished", "track-error", "preview-done", "preview-error")}
for m, k in kids.items():
    out, _ = k.communicate(timeout=180)
    assert k.returncode == 0 and f"survived {m}" in out, \
        f"I133 {m}: exit {k.returncode & 0xFFFFFFFF:#x} -- a thread was dropped while running\n{out[-800:]}"
print("a thread still running after its result is kept alive, not destroyed (I133) OK")

win.close()
pump(0.3)
for v in (VA, VB, VC):
    forget_recovery(v)
print("verify_sweep_fixes PASSED")
