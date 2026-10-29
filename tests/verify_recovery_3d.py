"""Offscreen GUI, CPU: the 3D result at the end of a run and in recovery copies (I372; sweep FS-F6-4, owner 2026-10-28:
"background + fix Restore"). The end of a run no longer brings the 3D result up to date on the window's thread for
the recovery copy (Save still does: what is saved is the 3D of the data saved, I305); so a recovery copy may hold 3D
that lags its tracks, and every recovery copy opened - File > Recover Unsaved Work, or File > Open answered
"Restore them?" - makes its 3D again once. After each step the project's 3D result must EQUAL a fresh full
`calib.reconstruct` (what Ctrl+3 computes) bit for bit.
  [1] a real run's end (Moving spot, T key) forces no 3D update; the automatic update then catches up
  [2] Ctrl+S right after an edit: the SAVED 3D result is the full reconstruct of the saved tracks
  [3] a recovery copy whose 3D lags, opened by File > Recover Unsaved Work: equal after opening
  [4] the same, opened by File > Open -> "Unsaved changes found ... Restore them?" Yes: equal (12.9 mm off before)
  [5] a recovery copy written DURING a run (the 30 s tick; a crash after it), opened by Restore: equal
  [6] a never-saved session's copy whose 3D lags, opened with its video -> "Restore unsaved work?" Yes: equal
      (review: that path did not ask for the rebuild)
[1] and [5] FAIL on eaee38b. The project: 3 synthetic calibrated cameras (one with an OpenCV lens), 400 frames, a dot
at the hip landmark in every camera.
Run: .venv\\Scripts\\python.exe tests\\verify_recovery_3d.py
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
OUT = os.path.join(HERE, "out", "recovery_3d")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_SETTINGS"] = os.path.join(OUT, "settings.ini")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QUESTIONS: list = []
QMessageBox.question = staticmethod(lambda *a, **k: (QUESTIONS.append(str(a[1:3])), QMessageBox.Yes)[1])
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[2] if len(a) > 2 else a)))
app = QApplication.instance() or QApplication([])

from _stubs import stub_track_choice  # noqa: E402
from kinetrace import recovery, threads  # noqa: E402
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, NoUndistort, OpenCVUndistort, dlt_from_camera,  # noqa
                             reconstruct)
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

TRACK_CHOICE = stub_track_choice("fresh")       # a press over data ahead tracks it again (the run's own rows)
FAILS: list = []
W, H, N = 320, 240, 400
NAMES = ["head", "hip", "tail"]


def check(label, cond, detail=None):
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
        pump(0.01)


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[420.0, 0, W / 2], [0, 420.0, H / 2], [0, 0, 1]])
CAMS = []
for _k, _a in enumerate((0.3, 2.4, 4.4)):
    _pos = np.array([2.0 * np.cos(_a), 2.0 * np.sin(_a), 0.5 + 0.4 * _k])
    _R, _t = look_at(_pos)
    _lens = OpenCVUndistort(K, np.array([-0.08, 0.01, 0.0, 0.0])) if _k == 1 else NoUndistort()
    CAMS.append(CameraCalibration(dlt_from_camera(K, _R, _t), W, H, _lens, pixel_origin=0.0))


def world_at(f):
    s = float(f)
    hip = np.array([0.3 * np.cos(s * 0.02), 0.3 * np.sin(s * 0.02), 0.001 * s])
    return np.stack([hip + [0.06, 0.0, 0.02], hip, hip - [0.07, 0.0, 0.01]])


def build_project():
    rng = np.random.RandomState(5)
    sessions = []
    for c, cal in enumerate(CAMS):
        p = os.path.join(OUT, f"cam{c}.mp4")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        s = TrackingSession(p, N, 30.0, W, H)
        for nm in NAMES:
            s.add_landmark(nm)
        bg = cv2.GaussianBlur(np.random.default_rng(c).integers(20, 60, size=(H, W, 3), dtype=np.uint8), (0, 0), 2)
        for f in range(N):
            uv = cal.project(world_at(f))
            img = bg.copy()
            cv2.circle(img, (int(round(uv[1][0])), int(round(uv[1][1]))), 6, (230, 230, 230), -1, cv2.LINE_AA)
            vw.write(img)
            for j in range(3):
                s.set_position(f, j, float(uv[j][0] + rng.normal(0, 0.2)), float(uv[j][1] + rng.normal(0, 0.2)))
        vw.release()
        s.manual[1:] = False                          # tracked data, not clicks (a run may write over it)
        sessions.append(s)
    proj = Project(sessions, ["camA", "camB", "camC"], [0.0, 0.0, 0.0])
    proj.calibration = Calibration(CAMS, unit="m", source="synthetic")
    path = os.path.join(OUT, "rec3d.kinetrace")
    proj.save(path)
    return path


def full(win):
    p = win.project
    return reconstruct(p.sessions, p.calibration, p.rates, p.offsets, win._t_range_3d())


def diffs(r, f):
    if r is None:
        return ["no 3D result"]
    out = []
    if r.t0 != f.t0 or r.n_frames != f.n_frames:
        return [f"range {r.t0}+{r.n_frames} vs {f.t0}+{f.n_frames}"]
    if list(r.names) != list(f.names):
        return [f"names {r.names} vs {f.names}"]
    for k in ("xyz", "residual", "n_cams", "per_cam"):
        a, b = getattr(r, k), getattr(f, k)
        if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
            out.append(k + (f" (max {np.nanmax(np.abs(a - b)) * 1000:.2f} mm)" if k == "xyz" and a is not None
                            and b is not None and a.shape == b.shape else ""))
    return out


def same(win, what, settle=1.5):
    """The project's 3D result equals a fresh full reconstruct, bit for bit (NaN = NaN), once the automatic
    update has had its ticks."""
    pump(settle)
    d = diffs(win.project.reconstruction, full(win))
    check(what, not d, "; ".join(d))


def new_window():
    win = MainWindow()
    win.resize(1300, 850)
    win.show()
    pump(0.2)
    return win


def open_by_menu(win, path):
    """File > Open Project, the dialog answered with the project's kinetrace.json."""
    real = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (os.path.join(path, "kinetrace.json"), ""))
    try:
        win._open_project_dialog()
    finally:
        QFileDialog.getOpenFileName = real
    return wait(lambda: win.state == READY and win.project is not None and not win._loading
                and not win._sync3d_busy, 60)


def crash(win):
    """The window is gone without its close (power cut): no save, its recovery copy stays as last written."""
    for t in (win._autosave_timer, win._sync3d_timer):
        t.stop()
    win._autosave = lambda *a, **k: None              # a lost window writes nothing more
    win.hide()


def copy_lag(pid):
    """How far the recovery copy's own 3D result is from the full reconstruct of its own tracks (mm)."""
    from kinetrace import projectfile
    c = projectfile.read(recovery.paths(pid)[0])[0]
    r = c.reconstruction
    if r is None:
        return float("inf")
    f = reconstruct(c.sessions, c.calibration, c.rates, c.offsets, (r.t0, r.t0 + r.n_frames - 1))
    return float(np.nanmax(np.abs(r.xyz - f.xyz)) * 1000) if r.xyz.shape == f.xyz.shape else float("inf")


def click(win, x, y):
    cv = win.canvas
    QTest.mouseClick(cv.viewport(), Qt.LeftButton, Qt.NoModifier, cv.mapFromScene(QPointF(float(x), float(y))))
    pump(0.1)


def select(win, name):
    s = win.session
    win.layers.select_only([s.pid_by_name(name)])
    win._on_point_selection_changed()
    pump(0.1)


def key(win, k, mod=Qt.NoModifier):
    win.activateWindow()
    QTest.keyClick(win, k, mod)
    pump(0.1)


def lagging_copy(win, frame, dx):
    """An edit (a click moves the hip in camera A), then the 30 s tick writes the recovery copy before the
    automatic 3D update has had its tick; -> how far the copy's 3D is from its tracks (mm)."""
    win._sync3d_timer.stop()
    win._set_active_view(0)
    win._goto(frame, force=True)
    select(win, "hip")
    s = win.session
    x, y = s.tracks[frame, s.pid_by_name("hip")]
    click(win, x + dx, y)
    win._autosave_timer.timeout.emit()                # the 30 s tick
    wait(lambda: not win._writer_busy(), 30)
    pump(0.2)
    win._sync3d_timer.start()
    return copy_lag(win._project_id)


# ---------------------------------------------------------------------------------------------------------------
PROJ = build_project()
A = new_window()
assert open_by_menu(A, PROJ), "the project opens"
p = A.project
assert p.calibration is not None and p.calibration.n_usable(3) == 3
A.act_recon.trigger()                                    # a first full result, as a user would have
pump(0.5)
same(A, "(setup) Ctrl+3 made the full 3D result")


def section_run_end():
    print("[1] the end of a real run brings 3D up to date in the background, not forced on the window's thread")
    win = A
    win._set_active_view(0)
    s = win.session
    hip = s.pid_by_name("hip")
    s.tracks[:, hip] += 1.5                              # (setup) the stored hip 1.5 px off: the run changes it
    s._touch()
    pump(1.5)                                            # the automatic update catches up with the setup change
    s.points[hip].tracker = "spot"                       # Moving spot: a real run on the CPU
    win._goto(0, force=True)
    select(win, "hip")
    forced, calls = [], []
    real = win._sync3d_run

    def spy(changed, n_cells=0, digs=None, force=False):
        calls.append(force)
        if force:
            forced.append(win.state)
        return real(changed, n_cells, digs=digs, force=force)
    win._sync3d_run = spy
    try:
        key(win, Qt.Key_T)
        check("(setup) the run started", wait(lambda: win.state == TRACKING, 20))
        check("(setup) the run ended", wait(lambda: win.state == READY and win.worker is None, 120))
        n_end = len(forced)
        pump(0.05)
    finally:
        del win._sync3d_run
    check("(setup) the run changed camera A's hip", not np.allclose(s.tracks[:N, hip], s.tracks[:N, hip] * 0))
    check("[1] the end of the run (its recovery copy) made no forced 3D update on the window's thread",
          n_end == 0, f"{n_end} forced")
    same(A, "[1] ... and the automatic update then made the 3D result equal to Ctrl+3")


def section_save():
    print("[2] Ctrl+S right after an edit saves the 3D result of the data saved")
    win = A
    win._sync3d_timer.stop()                             # Ctrl+S before the automatic update's tick
    win._set_active_view(0)
    win._goto(30, force=True)
    select(win, "hip")
    s = win.session
    x, y = s.tracks[30, s.pid_by_name("hip")]
    click(win, x + 2.0, y)
    key(win, Qt.Key_S, Qt.ControlModifier)
    wait(lambda: not win._writer_busy() and not win.project.dirty, 30)
    win._sync3d_timer.start()
    from kinetrace import projectfile
    saved = projectfile.read(PROJ)[0]
    d = diffs(saved.reconstruction, reconstruct(saved.sessions, saved.calibration, saved.rates, saved.offsets,
                                                (saved.reconstruction.t0, saved.reconstruction.t0
                                                 + saved.reconstruction.n_frames - 1)))
    check("[2] the saved 3D result is the full reconstruct of the saved tracks", not d, "; ".join(d))


def _open_recovered(how):
    win = new_window()
    n_q = len(QUESTIONS)
    if how == "recover":
        real = QInputDialog.getItem
        QInputDialog.getItem = staticmethod(lambda _w, _t, _l, items, *a, **k: (items[0], True))
        try:
            win._recover_dialog()                        # File > Recover Unsaved Work, the newest copy
        finally:
            QInputDialog.getItem = real
        ok = wait(lambda: win.state == READY and win.project is not None and not win._loading, 60)
    else:
        ok = open_by_menu(win, PROJ)                     # File > Open; "Restore them?" is answered Yes
    asked = [q for q in QUESTIONS[n_q:] if "Unsaved changes" in q]
    return win, ok, asked


def section_recover():
    print("[3] a recovery copy whose 3D lags its tracks, opened by File > Recover Unsaved Work")
    lag = lagging_copy(A, 60, 3.0)
    check("(setup) the copy's 3D lags its tracks (the run's end and the 30 s tick no longer force it)", lag > 0.01,
          f"{lag:.3f} mm")
    win, ok, _ = _open_recovered("recover")
    check("(setup) the copy opens", ok)
    same(win, "[3] after opening, its 3D result equals Ctrl+3")
    crash(win)


def section_restore():
    print("[4] the same, opened by File > Open -> 'Unsaved changes found ... Restore them?' Yes")
    lag = lagging_copy(A, 90, -3.0)
    check("(setup) the copy's 3D lags its tracks", lag > 0.01, f"{lag:.3f} mm")
    win, ok, asked = _open_recovered("restore")
    check("(setup) the project opens and asked to restore the unsaved changes", ok and asked, asked)
    same(win, "[4] after opening, its 3D result equals Ctrl+3 (eaee38b: stayed off)")
    crash(win)


def section_midrun():
    print("[5] a recovery copy written DURING a run (the 30 s tick), the window then lost; opened by Restore")
    win = A
    win._set_active_view(0)
    s = win.session
    hip = s.pid_by_name("hip")
    s.tracks[:, hip] -= 2.0                              # (setup) the run will change every frame of it
    s._touch()
    pump(1.5)
    same(win, "(setup) the 3D result is up to date before the run")
    win._goto(0, force=True)
    select(win, "hip")
    key(win, Qt.Key_T)
    caught = wait(lambda: win.state == TRACKING and int(s.tracked[:, hip].sum()) > 0
                  and win._display_queue is not None and win.current > 60, 60)
    check("(setup) the run is going (past frame 60)", caught and win.state == TRACKING, win.current)
    win._autosave_timer.timeout.emit()                   # the 30 s tick, mid-run
    wait(lambda: not win._writer_busy(), 30)
    crash(win)                                           # the power goes: the run never ends here
    win._pause_tracking()
    wait(lambda: win.worker is None, 30)
    lag = copy_lag(win._project_id)
    check("(setup) the copy written mid-run holds 3D that lags its tracks", lag > 0.01, f"{lag:.3f} mm")
    w2, ok, asked = _open_recovered("restore")
    check("(setup) the project opens and asked to restore the unsaved changes", ok and asked, asked)
    same(w2, "[5] after opening a copy made mid-run, its 3D result equals Ctrl+3 (eaee38b: stayed stale)")
    crash(w2)


def section_video():
    print("[6] a never-saved session's copy whose 3D lags, opened with its video -> 'Restore unsaved work?' Yes")
    from kinetrace import cameraorder, projectfile
    from PySide6.QtWidgets import QDialogButtonBox
    loaded = projectfile.read(PROJ)[0]
    vids = [s_.video_path for s_ in loaded.sessions]
    win = new_window()
    win._open_video(vids[0])                             # File > Open Video: a session never saved as a project
    wait(lambda: win.state == READY and not win._loading, 30)
    real_files, real_exec = QFileDialog.getOpenFileNames, cameraorder.CameraOrderDialog.exec
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (vids[1:], ""))
    cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(),
                                                       self.result())[1]
    try:
        win._add_video_dialog()                          # File > Add Camera Videos
        wait(lambda: win.project.n_views == 3 and not win._loading, 30)
    finally:
        QFileDialog.getOpenFileNames, cameraorder.CameraOrderDialog.exec = real_files, real_exec
    p_ = win.project
    for v in range(3):                                   # (setup) the saved project's tracks and calibration
        sv, lv = p_.sessions[v], loaded.sessions[v]
        for nm in NAMES:
            if sv.pid_by_name(nm) is None:
                sv.add_landmark(nm)
        for nm in NAMES:
            a, b = sv.pid_by_name(nm), lv.pid_by_name(nm)
            for arr in ("tracks", "tracked", "visibility", "manual", "confidence", "occluded"):
                getattr(sv, arr)[:, a] = getattr(lv, arr)[:, b]
        sv._touch()
    p_.calibration = loaded.calibration
    win._refresh_point_list()
    win.act_recon.trigger()                              # Ctrl+3
    pump(0.5)
    same(win, "(setup) the never-saved session has its full 3D result")
    lag = lagging_copy(win, 120, 3.0)
    check("(setup) its recovery copy's 3D lags its tracks", lag > 0.01, f"{lag:.3f} mm")
    crash(win)
    w2 = new_window()
    n_q = len(QUESTIONS)
    w2._open_video(vids[0])                              # the same video again: "Restore unsaved work?" Yes
    ok = wait(lambda: w2.state == READY and w2.project is not None and w2.project.n_views == 3
              and not w2._loading and not w2._sync3d_busy, 60)
    asked = [q for q in QUESTIONS[n_q:] if "Restore unsaved work" in q]
    check("(setup) the video opens and offered the unsaved work, all three cameras", ok and asked, asked)
    same(w2, "[6] after restoring, its 3D result equals Ctrl+3")
    crash(w2)


for fn in (section_run_end, section_save, section_recover, section_restore, section_midrun, section_video):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check(f"({fn.__name__} ran to its end)", False, repr(e))
for w_ in QApplication.topLevelWidgets():
    if isinstance(w_, MainWindow):
        for t in (w_._autosave_timer, w_._sync3d_timer):
            t.stop()
        if w_.project is not None:
            w_.project.dirty = False
            for s_ in w_.project.sessions:
                s_.dirty = False
        probe = getattr(w_, "_dev_probe", None)
        if probe is not None:
            probe.wait(10000)
        w_.closeEvent(QCloseEvent())                  # a "crashed" window is only hidden: stop its threads
threads.wait_all()                                    # a running QThread dropped at exit aborts (0xC0000409)
pump(0.2)
recovery.discard(A._project_id)
print("\nverify_recovery_3d: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
