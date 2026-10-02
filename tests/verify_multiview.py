"""Offscreen GUI: digitizing across three calibrated cameras (G20-G26, I132).

A calibrated 3-camera synthetic rig (known DLT cameras, a dot on a known 3D path)
is opened through the real MainWindow and driven with real key and mouse events:
  Add armed in one camera, a click on ANOTHER camera's picture switches to it AND
  places the point there (G20; reported on a real stereo pair: the click was swallowed,
  the next one drew the look-here cross and no point reached the list);
  the lines in the other cameras run exactly to the picture's edges (I132);
  placed in two cameras, the third shows a diamond where the point is (G23): A
  places it exactly there, a click next to it snaps onto it, Ctrl+Z undoes; the
  cameras that have it show how far their own placement is from the others'; two
  cameras that disagree give no diamond and say why;
  the other cameras drop the selection ring on Esc, draw a ball's circle, and show
  a pointing hand (G22);
  Sync all views shows a step's pictures together; Active view only keeps the
  others veiled while the playhead moves and catches up when it stops, the
  working camera's guides still drawn; the choice is kept in the layout (G24);
  a camera added after calibrating leaves the guides among the calibrated ones (G25);
  POINTS -> New point makes an empty point in every camera, in one order (G26).

Run: .venv\\Scripts\\python.exe tests\\verify_multiview.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox

from _clean import forget_recovery  # noqa: E402
from kinetrace.calib import CameraCalibration, NoUndistort, closest_on_polyline, dlt_from_camera
from kinetrace.project import Project
from kinetrace.session import TrackingSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
W, H, N, FPS = 640, 480, 60, 30.0
PERF = float(os.environ.get("KINETRACE_PERF_SCALE", "1"))


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])
cams = []
for a in (0.2, 1.3, 2.4):
    pos = np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.6])
    R, t = look_at(pos)
    cams.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))


def world(f):
    return np.array([[0.15 * np.cos(f * 0.05), 0.15 * np.sin(f * 0.05), 0.05]])


paths, sessions = [], []
for c, cal in enumerate(cams):
    path = os.path.join(OUT, f"multiview_cam{c}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 40 + 10 * c, np.uint8)
        u, v = cal.project(world(f))[0]
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (230, 230, 230), -1)
        cv2.putText(img, str(f), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 0), 2)
        vw.write(img)
    vw.release()
    paths.append(path)
    sessions.append(TrackingSession(path, N, FPS, W, H))
    forget_recovery(path)
PROJ = os.path.join(OUT, "multiview.kinetrace")
Project(sessions, ["camA", "camB", "camC"], [0, 0, 0]).save(PROJ)
CSV = os.path.join(OUT, "multiview_dltCoefs.csv")
np.savetxt(CSV, np.stack([c.coefs for c in cams], 1), delimiter=",", fmt="%.10g")

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace import view3d as v3  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1600, 1000)
win.show()


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def click(canvas, x, y, mods=Qt.NoModifier):
    QTest.mouseClick(canvas.viewport(), Qt.LeftButton, mods, canvas.mapFromScene(QPointF(float(x), float(y))))
    pump(0.15)


def truth(cam, f):
    return cams[cam].project(world(f))[0]


def dist_to_lines(canvas, uv):
    return min(float(np.linalg.norm(closest_on_polyline(np.asarray(pts), uv) - uv))
               for pts, _c, _n in canvas._guides.lines)


win._open_project_from_path(PROJ)
for _ in range(200):
    pump(0.05)
    if win.state == READY and win.project is not None and win.project.n_views == 3:
        break
assert win.state == READY and win.project.n_views == 3
p = win.project
A, B, C = (win.grid.canvas(k) for k in range(3))
app.setActiveWindow(win)


def _fake_exec(self):
    assert self.load(CSV)
    self.conv.setCurrentIndex(2)                   # OpenCV, 0-based: how the cameras were built
    self._accept()
    return QDialog.Accepted


v3.CalibrationDialog.exec = _fake_exec
win._import_calibration()
pump()
assert p.calibration is not None and len(p.calibration) == 3 and win.act_epipolar.isChecked()
sa, sb, sc = p.sessions
F = 20
win._goto(F)
pump(0.3)

# ---- 1. (G20) Add armed in camA, one click on camB's picture places the point IN camB ----
assert p.active == 0
QTest.keyClick(win, Qt.Key_N)
assert win.btn_add.isChecked() and A._place_mode
ub = truth(1, F)
click(B, *ub)
assert p.active == 1, "the click switched to camB"
assert [s.n_points for s in p.sessions] == [1, 1, 1], \
    f"the armed click made ONE point, in every camera's list: {[s.n_points for s in p.sessions]}"
name = sb.points[0].name
assert sb.tracked[F, 0] and np.linalg.norm(sb.tracks[F, 0] - ub) < 1.5, "placed in camB, where clicked"
assert not sa.tracked.any() and not sc.tracked.any(), "nowhere else"
assert not win.btn_add.isChecked() and not A._place_mode and not B._place_mode, "one placement per arm"
assert win._epi_probe is None and not B._guides.lines, "no look-here cross"
assert win.point_list.count() == 1 and win.selected == 0
print("G20: Add armed in one camera, a click on another camera's picture switches AND places the point there OK")

# ---- 2. (I132) the lines in the other cameras run exactly to the picture's edges -------
for cv, k in ((A, 0), (C, 2)):
    assert len(cv._guides.lines) == 1, f"cam{k} shows the line from camB"
    pts = np.asarray(cv._guides.lines[0][0])
    assert ((pts[:, 0] >= -0.5 - 1e-6) & (pts[:, 0] <= W - 0.5 + 1e-6)
            & (pts[:, 1] >= -0.5 - 1e-6) & (pts[:, 1] <= H - 0.5 + 1e-6)).all(), "inside the picture"
    for q in (pts[0], pts[-1]):
        border = min(abs(q[0] + 0.5), abs(q[0] - (W - 0.5)), abs(q[1] + 0.5), abs(q[1] - (H - 0.5)))
        assert border < 0.02, f"cam{k}: the line ends {border:.2f} px short of the picture edge at {q}"
    d = dist_to_lines(cv, truth(k, F))
    assert d < 3.0, f"cam{k}: the line misses the truth by {d:.2f} px"
    assert not cv._guides.dim and not cv._guides.notes, "one camera has it: full-strength lines, no 3D yet"
print("I132: the epipolar lines run to the picture edges and pass through the truth OK")

# ---- 3. (G23) placed in two cameras: the third shows where it is ------------------------
click(A, W / 2, H / 2)                              # unarmed: only switches (I110)
assert p.active == 0 and win.selected == sa.pid_by_name(name)
assert not sa.tracked.any(), "the switching click placed nothing"
ua = truth(0, F)
click(A, *ua)                                        # a plain click places the selected point
assert sa.tracked[F, 0] and sa.manual[F, 0]
# (G28) placed in two cameras: the 3D rmse beside it in both, the lines faint everywhere
tri = win._triangulate(win._observations(name, F, exclude=-1))
assert tri is not None and tri["views"] == [0, 1] and tri["residual"] < 2.0 and tri["verdict"] == "good", tri
for cv in (A, B):
    nt = cv._guides.notes
    assert len(nt) == 1 and f"3D rmse {tri['residual']:.2f} px · 2 cams" in nt[0][2], nt
    assert cv._guides.dim, "the lines are faint once two cameras have the point"
assert C._guides.dim and not C._guides.notes, "camC has no position: faint lines + the diamond, no rmse"
assert "3D rmse" in win.statusBar().currentMessage(), win.statusBar().currentMessage()
print(f"G28: placed in two cameras, both show '3D rmse {tri['residual']:.2f} px · 2 cams' (good) and every "
      "camera's lines turn faint OK")
pr = C._guides.preds
assert len(pr) == 1 and C._guides.dim, "camC shows the diamond; its lines step back"
err = float(np.hypot(pr[0][0] - truth(2, F)[0], pr[0][1] - truth(2, F)[1]))
assert err < 1.5, f"the diamond in camC is {err:.2f} px from the truth"
assert pr[0][4] is None and "from 2 cameras" in pr[0][3], pr[0][3]
assert not A._guides.preds and not B._guides.preds, "camA / camB have only one other camera with the point"
click(C, W / 2, H / 2)                               # switch to camC
assert p.active == 2 and win.selected == sc.pid_by_name(name)
assert "◇" in win.statusBar().currentMessage(), win.statusBar().currentMessage()
j = sc.pid_by_name(name)
QTest.keyClick(win, Qt.Key_A)                        # A: place it at the diamond
pump(0.05)
assert sc.tracked[F, j] and sc.manual[F, j]
assert np.allclose(sc.tracks[F, j], [pr[0][0], pr[0][1]], atol=1e-3), "placed exactly at the diamond"
win._undo_run()
pump(0.05)
assert not sc.tracked[F, j], "Ctrl+Z takes it back"
assert len(win.canvas._guides.preds) == 1
px, py = win.canvas._guides.preds[0][:2]
off = 3.0 * win.canvas.scene_px_per_screen_px()       # 3 screen px beside the diamond
click(win.canvas, px + off, py - off)
assert sc.tracked[F, j] and np.allclose(sc.tracks[F, j], [px, py], atol=1e-3), \
    f"a click next to the diamond snaps onto it: {sc.tracks[F, j]} vs {(px, py)}"
far = 40.0 * win.canvas.scene_px_per_screen_px()
click(win.canvas, px + far, py)
assert np.hypot(sc.tracks[F, j][0] - (px + far), sc.tracks[F, j][1] - py) < 1.5, "a click elsewhere is taken as it is"
click(win.canvas, px + off, py - off)                # back onto the diamond
pa = A._guides.preds
assert len(pa) == 1 and pa[0][4] is not None and "px from the 2 other cameras" in pa[0][3], \
    "a camera that has the point shows how far its placement is from the others'"
print(f"G23: the diamond in the third camera is {err:.2f} px from the truth; A, a click beside it and Ctrl+Z; "
      "the placed cameras show their distance OK")

# ---- 4. (G23) two cameras that disagree: no diamond, and why ----------------------------
jb = sb.pid_by_name(name)
keep = sb.tracks[F, jb].copy()
# across camA's line in camB: two cameras cannot see a slip ALONG the line (the rays
# still meet, at another depth) -- that is geometry, not a missed check
line = np.asarray(win._guides_into(1, name, F)[0][0])
tang = line[-1] - line[0]
nrm = np.array([-tang[1], tang[0]]) / np.linalg.norm(tang)
sb.set_position(F, jb, float(keep[0] + 30 * nrm[0]), float(keep[1] + 30 * nrm[1]))
sc.clear_window([j], F, F)                           # camC's own placement out of the way
win._refresh_overlay()
pump(0.05)
assert not sc.tracked[F, j], "camC's own placement cleared for this check"
assert not win.canvas._guides.preds, "no diamond when the two cameras disagree"
assert any("no ◇" in lab and "disagree" in lab for _p, _c, lab in win.canvas._guides.lines), \
    [lab for _p, _c, lab in win.canvas._guides.lines]
QTest.keyClick(win, Qt.Key_A)
pump(0.05)
assert not sc.tracked[F, j] and "disagree" in win.statusBar().currentMessage(), win.statusBar().currentMessage()
sb.set_position(F, jb, float(keep[0]), float(keep[1]))
win._refresh_overlay()
assert len(win.canvas._guides.preds) == 1
print("G23: cameras that disagree by 30 px give no diamond and say so; A refuses with the reason OK")

# ---- 5. (G22) the other cameras: selection ring, ball circles, pointing hand -------------
assert A._selected == sa.pid_by_name(name), "camA rings the selected landmark"
QTest.keyClick(win, Qt.Key_Escape)
pump(0.05)
assert win.selected is None and A._selected is None and B._selected is None, "Esc drops the ring everywhere"
ball = sa.add_ball(F, 120.0, 120.0)
sa.radius[F, ball] = 14.0
win._share_landmarks(undoable=False)
win._refresh_companions()
assert A._regions[ball].isVisible(), "a ball's fitted circle is drawn in a companion camera"
assert A.cursor().shape() == Qt.PointingHandCursor and win.canvas.cursor().shape() == Qt.ArrowCursor
print("G22: Esc drops the ring in every camera; a companion draws a ball's circle and shows a pointing hand OK")

# ---- 5b. (G33) View -> Trails: Off / Last 10 frames / Custom..., and the other cameras
# draw THEIR trails only while Track -> Every camera is ticked ------------------------------
from PySide6.QtWidgets import QInputDialog  # noqa: E402
from kinetrace.canvas import TRAIL_FRAMES  # noqa: E402

texts = [a.text() for a in win._trail_group.actions()]
assert TRAIL_FRAMES == 10 and texts == ["Off", "Last 10 frames", "Custom…"], texts
assert win._trail_len == 10 and win._trail_acts[10].isChecked(), "the default is the last 10 frames"
comp = next(k for k in range(3) if k != p.active)
sv, cvc = p.sessions[comp], win.grid.canvas(comp)
jv = sv.pid_by_name(name)
fv = p.map_frame(p.active, comp, win.current)
L = 12
_before = sv.snapshot()                            # put back exactly afterwards
tw = np.stack([np.linspace(100, 160, L), np.linspace(90, 120, L)], 1).astype(np.float32)[:, None]
sv.write_segment(fv - L + 1, tw, np.ones((L, 1), bool), [jv], np.ones((L, 1), np.float32))
win.act_track_all.setChecked(False)
pump(0.05)
assert cvc._motion.past == [], "tracking in one camera: no trails in the others"
win.act_track_all.setChecked(True)                 # ticking it redraws the others at once
pump(0.05)
tr = cvc._motion.past
assert tr and len(tr[jv]) == 11 and np.isfinite(tr[jv]).all(), "Every camera: the others show their trails"
_getint = QInputDialog.getInt
try:
    QInputDialog.getInt = staticmethod(lambda *a, **k: (25, True))
    win.act_trail_custom.trigger()
    assert win._trail_len == 25 and win.act_trail_custom.isChecked(), "Custom... sets the length typed"
    assert win.act_trail_custom.text() == "Custom… (25 frames)", win.act_trail_custom.text()
    assert len(cvc._motion.past[jv]) == min(fv, 25) + 1, "the other cameras use the same length"
    win._trail_acts[10].trigger()
    QInputDialog.getInt = staticmethod(lambda *a, **k: (0, False))
    win.act_trail_custom.trigger()                 # Cancel: the choice before is ticked again
    assert win._trail_len == 10 and win._trail_acts[10].isChecked() and win.act_trail_custom.text() == "Custom…"
finally:
    QInputDialog.getInt = _getint
win._trail_acts[0].trigger()                       # Off: no trail anywhere, Every camera or not
pump(0.05)
assert win._trail_len == 0 and cvc._motion.past == [] and win.canvas._motion.past == []
win._trail_acts[10].trigger()
win.act_track_all.setChecked(False)
sv.restore(_before)
win._refresh_companions()
print("G33: Trails = Off / Last 10 frames / Custom...; the other cameras draw theirs only with Every camera "
      "ticked, and Off shows none anywhere OK")

# ---- 6. (G24) Sync all: a step's pictures go up together ----------------------------------
got = []
for k, cv in enumerate((A, B, C)):
    cv.set_frame = (lambda orig, k: lambda rgb: (got.append((k, time.monotonic())), orig(rgb))[1])(cv.set_frame, k)
flushes = []
_orig_flush = win._flush_lockstep
win._flush_lockstep = lambda: (flushes.append((time.monotonic(), dict(win._lock or {}))), _orig_flush())[1]
win._last_goto_t = 0.0
win._goto(F + 5)
assert win._lock is not None and win._lock["want"] >= {2}, "a lone step waits for every camera"
for _ in range(100):
    pump(0.02)
    if win._lock is None:
        break
assert win._lock is None and flushes, "the step's pictures went up"
t_flush, lk = flushes[-1]
assert set(lk.get("frames", {})) == {0, 1, 2}, f"every camera delivered in time: {set(lk.get('frames', {}))}"
early = [k for k, tt in got if tt < t_flush]
assert not early, f"camera(s) {early} showed their new picture before the others"
assert {k for k, _tt in got} == {0, 1, 2}
print("G24: a step shows every camera's new picture at once (lockstep) OK")

# ---- 7. (G27) Active view only: ONLY the working camera reads its video -----------------
# (G27: the other cameras still followed with Sync all off -- the first
# version let the others catch up when the playhead stopped, and a lone step updated all)
requests = {0: 0, 1: 0}
for k in (0, 1):
    rt_ = win._views[k]
    if rt_.seek is None:
        win._start_seek_service(k)
    rt_.seek.request = (lambda orig, k: lambda f: (requests.__setitem__(k, requests[k] + 1), orig(f))[1])(
        rt_.seek.request, k)
win.cameras.btn_sync.click()
pump(0.05)
assert win._sync_mode == "active" and win.act_sync_active.isChecked() and not win.cameras.btn_sync.isChecked()
assert not A.is_stale(), "the others are at this instant when the mode is chosen: not veiled yet"
shown = [win._views[k].want_frame for k in (0, 1)]
win._last_goto_t = 0.0
win._goto(F)                                         # a lone step
pump(0.1)
win._last_goto_t = time.monotonic()
win._goto(F + 3)                                     # continuous moves
win._last_goto_t = time.monotonic()
win._goto(F + 1)
pump(0.6)                                            # the playhead stops: still nothing
assert requests == {0: 0, 1: 0}, f"the other cameras were asked to decode: {requests}"
assert [win._views[k].want_frame for k in (0, 1)] == shown, "they keep the picture they showed"
assert win._companions_stale and A.is_stale() and B.is_stale() and not C.is_stale()
cap = win.grid._cells[0].caption.text()
assert "not following" in cap and f"frame {shown[0]}" in cap, cap
win._last_goto_t = 0.0
win._goto(F)                                         # back on the frame with data
win.point_list.setCurrentRow(sc.pid_by_name(name))
pump(0.05)
assert len(win.canvas._guides.lines) == 2, "the working camera keeps its guides (they need tracks, not pictures)"
assert not A._guides.lines and not A._guides.preds and not A._guides.notes, "a veiled camera draws none"
click(A, W / 2, H / 2)                               # work in camA: it reads ITS video now
assert p.active == 0 and not A.is_stale() and requests[1] == 0
assert not C.is_stale() and win._views[2].want_frame == F, "the camera left shows this instant: not veiled"
assert B.is_stale(), "camB is still where it was"
click(C, W / 2, H / 2)
assert p.active == 2
lay = win._ui_global_state()["layout"]
assert lay["sync"] == "active", lay
win.act_sync_all.trigger()
assert win._sync_mode == "all" and win.cameras.btn_sync.isChecked()
win._apply_layout(dict(lay))
assert win.act_sync_active.isChecked() and win._sync_mode == "active", "the layout restores Active view only"
win.act_solo.setChecked(True)
pump(0.05)
assert win.grid.solo and win.grid.visible_indices() == [2]
win.act_solo.setChecked(False)
pump(0.05)
assert not win.grid.solo and win.act_sync_active.isChecked(), "unhiding goes back to Active view only"
win.act_solo.trigger()                               # Ctrl+2 ...
win.act_solo.trigger()                               # ... and Ctrl+2 again: the shortcut toggles
assert not win.grid.solo and win.act_sync_active.isChecked(), "Ctrl+2 twice: hidden, then back"
win.act_sync_active.trigger()                        # Ctrl+Shift+2 again: back to Sync all views
assert win.act_sync_all.isChecked() and win._sync_mode == "all" and win.cameras.btn_sync.isChecked()
win.act_sync_all.trigger()                           # Sync all views cannot be ticked off by itself
assert win.act_sync_all.isChecked() and win._sync_mode == "all"
pump(0.3)
assert not A.is_stale() and not B.is_stale() and win._views[1].want_frame == F, "Sync all: back to the playhead"
print("G27: Active view only never decodes the other cameras (steps, moves, stops); they keep their picture "
      "veiled, the working camera keeps its guides, a click works in another; kept in the layout; Hidden, "
      "the shortcuts and Sync all bring them back OK")

# ---- 8. (G25) a camera added after calibrating: guides stay among the calibrated -------
cam4 = os.path.join(OUT, "multiview_cam3.mp4")
shutil.copyfile(paths[0], cam4)
forget_recovery(cam4)
assert win._add_view(cam4)
win._refresh_cameras()
win._refresh_companions()
pump(0.3)
D = win.grid.canvas(3)
assert win._guides_ready() and win._cal_cam(3) is None
win._goto(F)
win.point_list.setCurrentRow(win.session.pid_by_name(name))
pump(0.1)
assert win.canvas._guides.lines and not D._guides.lines and not D._guides.preds, \
    "the calibrated cameras keep their guides; the new one has none"
assert win._need_calibration("3D") is False, "3D still needs every camera"
win._remove_view(3)
pump(0.2)
print("G25: a camera added after calibrating leaves the guides and diamonds among the calibrated cameras OK")

# ---- 9. (G26) POINTS -> New point, one order in every camera ------------------------------
n0 = win.session.n_points
win.btn_new_point.click()
pump(0.05)
new = win.session.points[-1].name
assert [s.n_points for s in p.sessions] == [n0 + 1] * 3 and win.selected == n0
assert all(not s.tracked[:, s.pid_by_name(new)].any() for s in p.sessions), "no data yet"
orders = [[q.name for q in s.points] for s in p.sessions]
assert orders[0] == orders[1] == orders[2], orders
click(win.canvas, 300, 200)
assert win.session.tracked[F, n0], "a plain click places the new point"
win._undo_run()
win._undo_run()
s_ = TrackingSession(paths[0], N, FPS, W, H)
for nm, (x, y) in (("a", (10, 10)), ("b", (20, 20)), ("c", (30, 30))):
    s_.add_point(3, x, y, name=nm)
s_.ui_state["selected"] = 2
s_.reorder_points([2, 0, 1])
assert [q.name for q in s_.points] == ["c", "a", "b"] and s_.ui_state["selected"] == 0
assert np.allclose(s_.tracks[3], [[30, 30], [10, 10], [20, 20]]) and s_.manual[3].all()
print("G26: New point is in every camera with no data, one order everywhere; a reorder carries data and "
      "the selection OK")

# ---- 10. (G30) File -> Open Folder of Videos: tick, pick the base, import, saved ---------
from kinetrace.folderimport import VideoFolderDialog, default_project_path, list_videos  # noqa: E402

FOLD = os.path.join(OUT, "multiview_folder")
shutil.rmtree(FOLD, ignore_errors=True)
os.makedirs(os.path.join(FOLD, "sub"))
for src, dst in ((paths[0], "cam1.MP4"), (paths[1], "cam2.mp4"), (paths[2], "cam10.mp4"),
                 (paths[2], os.path.join("sub", "cam3.mp4"))):
    shutil.copyfile(src, os.path.join(FOLD, dst))
open(os.path.join(FOLD, "notes.txt"), "w").write("not a video")
open(os.path.join(FOLD, "._cam2.mp4"), "wb").write(b"\0" * 10)       # a macOS shadow file
got = [x.name for x in list_videos(FOLD)]
assert got == ["cam1.MP4", "cam2.mp4", "cam10.mp4"], f"natural order, any case, no shadows: {got}"
assert [x.name for x in list_videos(FOLD, recursive=True)][-1] == "cam3.mp4"
for x in list_videos(FOLD, recursive=True):
    forget_recovery(str(x))
dlg = VideoFolderDialog(win, FOLD)
for _ in range(100):
    pump(0.05)
    if len(dlg._headers) == 3:
        break
assert len(dlg._headers) == 3 and dlg._headers[0][:2] == (W, H), dlg._headers
assert dlg.table.item(0, 3).text() == f"{W} × {H}" and dlg.base_row() == 0
dlg._base.button(1).setChecked(True)                 # base: cam2
dlg._ticks[2].setChecked(False)                      # leave cam10 out
assert dlg.buttons.button(QDialogButtonBox.Ok).text() == "Import 2 videos"
dlg._ticks[1].setChecked(False)                      # unticking the base hands it on
assert dlg.base_row() == 0
dlg._ticks[1].setChecked(True)
dlg._base.button(1).setChecked(True)
dlg._accept()
assert [os.path.basename(x) for x in dlg.result_paths] == ["cam2.mp4", "cam1.MP4"], dlg.result_paths
assert dlg.result_project == str(default_project_path(FOLD)) and dlg.result_project.endswith("multiview_folder.kinetrace")
if os.path.exists(dlg.result_project):          # a project folder (I145)
    __import__('shutil').rmtree(dlg.result_project) if os.path.isdir(dlg.result_project) else os.remove(dlg.result_project)
win._import_folder(dlg.result_paths, dlg.result_project)
for _ in range(200):
    pump(0.05)
    if win.project is not None and win.project.n_views == 2 and os.path.exists(dlg.result_project):
        break
q = win.project
assert q.n_views == 2 and q.names == ["cam2", "cam1"], (q.n_views, q.names)
assert os.path.basename(q.sessions[0].video_path) == "cam2.mp4", "the base is camera 1, the reference"
assert os.path.exists(dlg.result_project) and not q.dirty, "saved at once"
back = Project.load(dlg.result_project)
assert back.names == ["cam2", "cam1"] and [os.path.basename(x.video_path) for x in back.sessions] == \
    ["cam2.mp4", "cam1.MP4"], "the project file keeps the chosen videos and the base"
print("G30: a folder's videos listed (natural order, headers read in the background), two ticked, the base "
      "chosen, imported as one project named by file, the base = camera 1, saved at once and reopened OK")
for x in list_videos(FOLD, recursive=True):
    forget_recovery(str(x))

win._dev_probe.wait(15000)
win.close()
for path in paths + [cam4]:
    forget_recovery(path)
print("MULTIVIEW PASSED")
