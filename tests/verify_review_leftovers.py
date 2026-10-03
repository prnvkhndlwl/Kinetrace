"""The code review's leftover items (docs/AUDIT.md "Code review 2026-10-02": G71 I223 I197 I206 I242 I258
G66 G103 I240 I195 R14 R15 R16 R17 R19 R20 + the small app.py clean-ups). Each bug fix is checked in a way
that FAILS on the code before it; the refactors (R14, R15, R17, R19, R20) get identity / equivalence
checks. Offscreen, no GPU: a 3-camera synthetic project (known DLT cameras) is opened through the real
MainWindow.

Run: .venv\\Scripts\\python.exe tests\\verify_review_leftovers.py
"""
import json
import os
import sys
import tempfile
import time
import traceback
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(errors="replace")

OUT = os.path.join(ROOT, "tests", "out", "review_leftovers")
import shutil  # noqa: E402

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "recovery"), exist_ok=True)
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_LOG_DIR"] = os.path.join(OUT, "logs")
os.environ.pop("HF_TOKEN", None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QObject, Qt, QThread, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from kinetrace import app as app_mod  # noqa: E402
from kinetrace import calibio, downloads, lens, projectfile, retrack, wand  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.body import BodyTrack, rig_of  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, NoUndistort, dlt_from_camera,  # noqa: E402
                             intersect_polylines)
from kinetrace.project import Project  # noqa: E402
from kinetrace.render import OverlayOptions, OverlayRenderer  # noqa: E402
from kinetrace.session import POINT_ARRAYS, TrackingSession  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def section(title):
    def deco(fn):
        print(f"\n[{title}]")
        try:
            fn()
        except Exception as e:      # noqa: BLE001 - on the old code most sections end here
            check(False, f"{title}: raised {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=4)
        return fn
    return deco


# ---------------------------------------------------------------- the synthetic rig
W, H, NF, FPS = 320, 240, 40, 30.0
NAMES = ["O", "PX", "PY", "M"]
STATIC = {"O": np.array([0.05, 0.10, 0.00]), "PX": np.array([0.45, 0.20, 0.10]),
          "PY": np.array([-0.05, 0.45, 0.05])}


def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


K = np.array([[420.0, 0, W / 2], [0, 420.0, H / 2], [0, 0, 1]])
CAMS = []
for k, a in enumerate((0.3, 2.4, 4.4)):
    pos = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.5 + 0.4 * k])
    R, t = look_at(pos)
    CAMS.append(CameraCalibration(dlt_from_camera(K, R, t), W, H, NoUndistort(), pixel_origin=0.0))
CAL = Calibration(CAMS, "m", "synthetic")


def world(name, f):
    if name == "M":
        return np.array([0.10 + 0.004 * f, -0.10, 0.30 + 0.002 * f])
    return STATIC[name]


def make_videos(tag, n_frames=NF):
    paths = []
    for c in range(3):
        p = os.path.join(OUT, f"{tag}_cam{c}.mp4")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in range(n_frames):
            vw.write(np.full((H, W, 3), 40 + f, np.uint8))
        vw.release()
        paths.append(p)
    return paths


def make_project(paths, names=("camA", "camB", "camC"), n_frames=NF):
    sessions = []
    for c, p in enumerate(paths):
        s = TrackingSession(p, n_frames, FPS, W, H)
        for nm in NAMES:
            s.add_landmark(nm)
        for f in range(n_frames):
            for j, nm in enumerate(NAMES):
                uv = np.asarray(CAMS[c].project(world(nm, f))).reshape(-1, 2)[0]
                s.set_position(f, j, float(uv[0]), float(uv[1]))
        sessions.append(s)
    proj = Project(sessions, list(names), [0, 0, 0])
    proj.calibration = CAL
    return proj


PATHS = make_videos("rig")
PROJ = os.path.join(OUT, "rig.kinetrace")
make_project(PATHS).save(PROJ)

# ---------------------------------------------------------------- dialogs stubbed
INFO, WARN, CRIT, TOASTS, TITLES = [], [], [], [], []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: INFO.append(str(a[2]) if len(a) > 2 else "") or QMessageBox.Ok)


def _warn(*a, **k):
    TITLES.append(str(a[1]) if len(a) > 1 else "")
    WARN.append(str(a[2]) if len(a) > 2 else "")
    return QMessageBox.Ok


QMessageBox.warning = staticmethod(_warn)
QMessageBox.critical = staticmethod(lambda *a, **k: CRIT.append(str(a[2]) if len(a) > 2 else "") or QMessageBox.Ok)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))

app = QApplication.instance() or QApplication([])


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def new_window():
    w = MainWindow()
    w.show()
    w.toast.show_message = lambda m, *a, **k: TOASTS.append(str(m))
    return w


def open_project(w, path, n=3):
    w._open_project_from_path(path)
    for _ in range(300):
        pump(0.05)
        if w.state == READY and w.project is not None and w.project.n_views == n:
            break
    return w.project is not None and w.project.n_views == n and w.state == READY


def clear_logs():
    for L in (INFO, WARN, CRIT, TOASTS, TITLES):
        L.clear()


class Key:
    """The bit of a QKeyEvent `_hotkey` reads."""

    def __init__(self, key, mods=Qt.NoModifier):
        self._k, self._m = key, mods

    def key(self):
        return self._k

    def modifiers(self):
        return self._m

    def accept(self):
        pass


win = new_window()
assert open_project(win, PROJ), "the synthetic project did not open"
p = win.project
print("project opened: 3 cameras")


# ================================================================ 1. G71
@section("G71 Track does nothing while a project / video is opening")
def _g71():
    win._on_select(0)
    win._goto(10, force=True)
    calls = {"pass": 0, "button": 0}
    real_pass, real_btn = win._start_pass, win._update_track_button
    win._start_pass = lambda *a, **k: calls.__setitem__("pass", calls["pass"] + 1)
    win._update_track_button = lambda *a, **k: (calls.__setitem__("button", calls["button"] + 1), real_btn())[1]
    try:
        win._busy_stack.append({"tok": 9999, "cancel": None})       # a busy level = the loading card is up
        check(win._loading, "a busy level is open")
        win._toggle_tracking()
        check(calls["pass"] == 0 and calls["button"] == 0,
              f"T while loading starts nothing and does not even re-judge the button {calls}")
        win._busy_stack[:] = [b for b in win._busy_stack if b["tok"] != 9999]
        check(not win._loading, "the level is closed")
        win._toggle_tracking()
        check(calls["pass"] >= 1, f"with the card gone T starts the run (sanity) {calls}")
    finally:
        win._start_pass, win._update_track_button = real_pass, real_btn
        win._busy_stack[:] = []


# ================================================================ 3. MANUAL_WHICH_MODEL
@section("the small-spot hint opens the manual at MANUAL_WHICH_MODEL")
def _hint():
    from kinetrace import spots
    got, clicks = {}, []
    win.toast.show_message = lambda m, *a, **k: got.update(on_click=k.get("on_click"))
    win._show_manual = lambda *a, **k: clicks.append(a)
    real = spots.looks_like_small_spot
    spots.looks_like_small_spot = lambda *a, **k: types.SimpleNamespace(diameter=5.0)
    real_cache = win._views[0].cache
    win._views[0].cache = types.SimpleNamespace(get=lambda f: np.zeros((H, W, 3), np.uint8))
    try:
        win._spot_hints.discard(("tiny", id(win.project)))
        s = win.session
        pid = s.pid_by_name("M")
        win._hint_small_spot(pid, 100.0, 100.0)
    finally:
        spots.looks_like_small_spot = real
        win._views[0].cache = real_cache
        win.toast.show_message = lambda m, *a, **k: TOASTS.append(str(m))
    check(got.get("on_click") is not None, "the hint has a click action")
    if got.get("on_click"):
        got["on_click"]()
    check(clicks == [(app_mod.MANUAL_WHICH_MODEL,)], f"it opens the manual at the shared heading {clicks}")
    del win._show_manual


# ================================================================ 4. I223
@section("I223 Import offsets: a name two cameras share goes by camera number")
def _i223():
    sess = [TrackingSession(os.path.join(OUT, "x.mp4"), 10, 30.0, W, H) for _ in range(3)]
    pj = Project(sess, ["GX010001", "GX010001", "other"], [0, 0, 0])
    f = os.path.join(OUT, "dup_offsets.csv")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("camera,name,offset,rate\n1,GX010001,5,1\n2,GX010001,9,1\n3,other,0,1\n")
    rows = calibio.read_offsets(f, pj)
    got = {v: (o, r) for v, o, r in rows}
    check(sorted(got) == [0, 1, 2], f"every camera gets its own row {sorted(got)}")
    check(abs(got[1][0] - 4.0) < 1e-9 and abs(got[2][0] + 5.0) < 1e-9,
          f"re-based on camera 1's row: {got}")
    # unique names still match by name, whatever the camera column says
    pj2 = Project(sess, ["a", "b", "c"], [0, 0, 0])
    f2 = os.path.join(OUT, "uniq_offsets.csv")
    with open(f2, "w", encoding="utf-8") as fh:
        fh.write("camera,name,offset,rate\n9,c,7,1\n9,a,0,1\n")
    rows2 = calibio.read_offsets(f2, pj2)
    check({v for v, _o, _r in rows2} == {0, 2}, f"unique names match by name {list(rows2)}")


# ================================================================ 5. I197
@section("I197 a video replaced while a run is live: nothing of the old run starts in the new one")
def _i197():
    w = new_window()
    try:
        assert open_project(w, PROJ)
        seen = {}

        class FakeWorker:
            def request_pause(self):
                seen["user_paused"] = w._user_paused
                seen["passes"] = w._passes
                seen["multi"] = w._multi
                seen["retrack"] = None if w._retrack is None else list(w._retrack["jobs"])

            def wait(self, ms=0):
                return True

            def isRunning(self):
                return False

        class FakePreview:
            cancelled = False

            def wait(self, ms=0):
                seen["waited_cancelled"] = self.cancelled
                return True

            def isRunning(self):
                return False

        w.worker = FakeWorker()
        w._preview = FakePreview()
        w._passes = {"queue": [["M"]], "groups": [["O"], ["M"]], "labels": [], "view": 0, "frame": 0,
                     "step": False, "every": False, "done": [], "snap": None, "msnaps": None}
        w._multi = {"results": []}
        w._retrack = {"jobs": [object(), object()], "done": [], "snaps": {}}
        w._user_paused = False
        w._teardown_video()
        check(seen.get("user_paused") is True, f"the pause is a user's pause (so no 'normal' end) {seen}")
        check(seen.get("passes") is None and seen.get("multi") is None, "no later pass / camera is waiting")
        check(seen.get("retrack") in ([], None), f"the re-track queue is empty ({seen.get('retrack')})")
        check(w._retrack is None, "the re-track is dropped with its project")
        check(seen.get("waited_cancelled") is True, "the mask preview is told to stop before it is waited for")
    finally:
        w.worker = None
        w._preview = None
        w.close()


# ================================================================ 6. I206 / I242
@section("I206 / I242 one helper drops the 3D layer, for retimes and camera changes as for imports")
def _i206():
    check(not hasattr(win, "_drop_3d_results"), "the second helper is gone")
    # (a) a retime (the project drops its own result; the helper follows up)
    win._reconstruct_3d(quiet=True)
    check(p.reconstruction is not None, "reconstructed")
    win._hull_cache[99] = ("v", "f", "h")
    win._on_view_offset(1, 3.0)
    msg = win.statusBar().currentMessage()
    check(p.reconstruction is None and not win._hull_cache, "a retime clears the result and the carved volumes")
    check("3D result" in msg and "carved volume" in msg, f"and says so: {msg!r}")
    p.set_offset(1, 0.0)
    # (b) the importers' kind of caller (the helper drops the result itself)
    win._reconstruct_3d(quiet=True)
    win._hull_cache[98] = ("v", "f", "h")
    dropped = win._drop_reconstruction("a new calibration was imported")
    msg = win.statusBar().currentMessage()
    check(dropped and p.reconstruction is None and not win._hull_cache,
          "an import clears the result and the carved volumes")
    check("3D result" in msg and "carved volume" in msg and "calibration was imported" in msg.lower().replace("a new", "a new"),
          f"and says so: {msg!r}")
    # nothing there, nothing said
    win.statusBar().clearMessage()
    check(win._drop_reconstruction("again") is False and win.statusBar().currentMessage() == "",
          "with nothing to drop nothing is said")
    # a replacement keeps a result and clears the volumes without a 'cleared' sentence
    win._reconstruct_3d(quiet=True)
    rec = p.reconstruction
    win._hull_cache[97] = ("v", "f", "h")
    win.statusBar().clearMessage()
    win._drop_reconstruction(replacement=rec)
    check(p.reconstruction is rec and not win._hull_cache and "cleared" not in win.statusBar().currentMessage(),
          "a replacement stays, its old volumes go, no 'cleared' sentence")
    win._drop_reconstruction()


# ================================================================ 7. act_set_axes
@section("Set World Axes is gated like the other 3D entries")
def _axes():
    w = new_window()
    try:
        w._apply_state()
        check(not w.act_set_axes.isEnabled() and not w.act_recon.isEnabled(),
              "with no video it is off, as Reconstruct is")
    finally:
        w.close()
    check(win.act_set_axes.isEnabled() == win.act_recon.isEnabled() is True, "with a project it is on")


# ================================================================ 8. I258
@section("I258 the silhouettes of an instant follow map_frame's half-frame rule")
def _i258():
    p.set_offset(1, 0.5)
    try:
        s1 = p.sessions[1]
        asked = []

        class FakeMasks:
            def has(self, f):
                asked.append(f)
                return False

            def rasterize(self, *a):
                return None
        old_masks = [x.masks for x in p.sessions]
        for x in p.sessions:
            x.masks = FakeMasks()
        try:
            win._masks_at_instant(2)
        finally:
            for x, m in zip(p.sessions, old_masks):
                x.masks = m
        check(p.local_index(1, 2) == 3, "local_index(1, 2) at offset 0.5 is 3 (a .5 tie goes up)")
        check(3 in asked and asked.count(2) == 2,
              f"camera B is asked for frame 3, not Python's round() = 2: {asked}")
    finally:
        p.set_offset(1, 0.0)


# ================================================================ 9. G66
@section("G66 F re-judges the button first; a camera switch selects the landmark alone first")
def _g66():
    win._on_select(0)
    win._goto(10, force=True)
    win._track_mode = "semi"
    called = []
    real = win._toggle_tracking
    win._toggle_tracking = lambda *a, **k: called.append(1)
    try:
        win._track_blocked = "a stale reason"
        win._hotkey(Key(Qt.Key_F))
        check(called == [1], f"a stale 'blocked' verdict does not turn F into plain navigation {called}")
    finally:
        win._toggle_tracking = real
        win._track_mode = "auto"
    # the selection survives a camera switch: ClearAndSelect on the landmark, the others added back
    win._on_select(1)
    win.point_list.item(2).setSelected(True)
    win.point_list.item(3).setSelected(True)
    before = sorted(win.session.points[q].name for q in win._selected_pids())
    win._set_active_view(1)
    after = sorted(win.session.points[q].name for q in win._selected_pids())
    check(before == after and win.session.points[win.selected].name == "PX",
          f"the whole selection and the current landmark travel with the user {before} -> {after}")
    win._set_active_view(0)


# ================================================================ 10. G103
@section("G103 the run scope is saved per camera and restored")
def _g103():
    # (a) the project file: view.json
    s = TrackingSession(os.path.join(OUT, "g103.mp4"), 10, 30.0, W, H)
    for nm in ("a", "b", "c"):
        s.add_landmark(nm)
    s.ui_state["selected"] = 1
    s.ui_state["selected_names"] = ["b", "c"]
    s.ui_state["segment_selected"] = True
    pj = Project([s], ["cam"], [0])
    path = os.path.join(OUT, "g103.kinetrace")
    projectfile.save(pj, path)
    view = json.load(open(os.path.join(path, "cameras", "cam", "view.json"), encoding="utf-8"))
    check(view.get("selected_points") == ["b", "c"] and view.get("segment_selected") is True,
          f"view.json holds the scope {view.get('selected_points')} {view.get('segment_selected')}")
    q = projectfile.load(path)
    ui = q.sessions[0].ui_state
    check(ui.get("selected_names") == ["b", "c"] and ui.get("segment_selected") is True,
          "and a read brings it back")
    check("selected_names" in projectfile.VIEW_KEYS and "segment_selected" in projectfile.VIEW_KEYS,
          "both are per-camera keys (not window-wide state.json tools)")
    # (b) through the app: select, save, reopen
    path2 = os.path.join(OUT, "g103app.kinetrace")
    make_project(PATHS).save(path2)
    w = new_window()
    try:
        assert open_project(w, path2)
        w._on_select(1)
        w.point_list.item(2).setSelected(True)
        want = sorted(w.session.points[q].name for q in w._selected_pids())
        w.project.dirty = True
        check(w._save_project(), "saved")
        w2 = new_window()
        try:
            assert open_project(w2, path2)
            got = sorted(w2.session.points[q].name for q in w2._selected_pids())
            check(got == want and len(got) == 2, f"a reopened project selects the same points {want} -> {got}")
            check("Track · 2 points" in w2.btn_track.text(), f"and the Track button says so: {w2.btn_track.text()!r}")
        finally:
            w2.close()
    finally:
        w.close()


# ================================================================ 11. I240
@section("I240 a frame that could be read at a second try is 'press Track again', not 'damaged'")
def _i240():
    class FakeW:
        decode_failed_at = 7

        def __init__(self, transient):
            self.decode_transient = transient

        def wait(self, ms=0):
            return True

        def isRunning(self):
            return False
    for transient in (True, False):
        clear_logs()
        win.worker = FakeW(transient)
        win.state = app_mod.TRACKING
        win._on_track_error("VideoDecodeError: frame 7")
        say = " | ".join(TOASTS) + " | " + win.statusBar().currentMessage() + " | " + " ".join(TITLES)
        if transient:
            toasts = " | ".join(TOASTS)
            check("could not be read just now" in toasts and "press Track again" in toasts
                  and "decoded" not in toasts and "press Track again" in win.statusBar().currentMessage(),
                  f"transient: {say[:200]!r}")
            check(any("read just now" in t for t in TITLES), f"the box title says so {TITLES}")
        else:
            check("could not be decoded" in " | ".join(TOASTS) and "just now" not in say,
                  f"a damaged frame keeps its wording: {say[:160]!r}")
    win.worker = None
    win._apply_state()


# ================================================================ 12. dead 'apart'
@section("the dead 'apart' ball-drop branches are gone")
def _apart():
    src = open(os.path.join(ROOT, "kinetrace", "app.py"), encoding="utf-8").read()
    check('reason == "apart"' not in src and 'why == "apart"' not in src and '"apart":' not in src,
          "no 'apart' reason is handled in app.py")
    check("too far from the other balls" not in src, "and none is worded")


# ================================================================ 13. I195
@section("I195 the Hugging Face token saved in Settings is passed to a gated download")
def _i195():
    import huggingface_hub
    repo = next(iter(downloads.HF_REVISIONS))
    got = {}
    real_snap, real_cached, real_dir = huggingface_hub.snapshot_download, downloads.hf_cached, downloads.MODELS_DIR
    huggingface_hub.snapshot_download = lambda *a, **k: got.update(k)
    downloads.hf_cached = lambda r: False
    tmp = tempfile.mkdtemp(dir=OUT)
    try:
        downloads.MODELS_DIR = type(real_dir)(tmp)
        os.makedirs(os.path.join(tmp, "hf"), exist_ok=True)
        with open(os.path.join(tmp, "hf", "token"), "w", encoding="utf-8") as fh:
            fh.write("hf_test_value\n")
        downloads.hf_snapshot(repo, "a model")
        check(got.get("token") == "hf_test_value", f"the saved token goes along: {sorted(got)}")
        got.clear()
        os.environ["HF_TOKEN"] = "from_env"
        downloads.hf_snapshot(repo, "a model")
        check("token" not in got, "HF_TOKEN in the environment is left to the Hub")
        del os.environ["HF_TOKEN"]
        os.remove(os.path.join(tmp, "hf", "token"))
        got.clear()
        downloads.hf_snapshot(repo, "a model")
        check("token" not in got, "no token file, no token")
    finally:
        huggingface_hub.snapshot_download, downloads.hf_cached, downloads.MODELS_DIR = real_snap, real_cached, real_dir
        os.environ.pop("HF_TOKEN", None)


# ================================================================ 14. R20
@section("R20 one dltCoefs writer: calibio.dlt_csv_matlab(pixel_origin_out), wand.export_dlt_csv is a wrapper")
def _r20():
    def old_calibio(cal, path):          # what dlt_csv_matlab wrote before (the default 1.0)
        cols = [c.coefs_for_origin(1.0) for c in cal.cameras]
        np.savetxt(str(path), np.stack(cols, axis=1), delimiter=",", fmt="%.12g")

    def old_wand(cal, path, out):        # what wand.export_dlt_csv computed before
        cols = []
        for cam in cal.cameras:
            L = np.asarray(cam.coefs, np.float64).reshape(11).copy()
            d = float(out) - float(getattr(cam, "pixel_origin", 0.0))
            if d != 0.0:
                L[0:3] += d * L[8:11]
                L[3] += d
                L[4:7] += d * L[8:11]
                L[7] += d
            cols.append(L)
        return [",".join(f"{v:.12g}" for v in row) for row in np.stack(cols, axis=1)]
    cams = [CameraCalibration(dlt_from_camera(K, *look_at(np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 0.7]))),
                              W, H, NoUndistort(), pixel_origin=float(i % 2)) for i, a in enumerate((0.3, 1.9, 3.5))]
    cal = Calibration(cams, "m", "t")
    a, b = os.path.join(OUT, "old.csv"), os.path.join(OUT, "new.csv")
    old_calibio(cal, a)
    calibio.dlt_csv_matlab(cal, b)
    check(open(a, "rb").read() == open(b, "rb").read(), "the default writes exactly the bytes it wrote before")
    for out in (1.0, 0.0):
        c = os.path.join(OUT, f"wand_{out}.csv")
        wand.export_dlt_csv(cal, c, pixel_origin_out=out)
        text = open(c, "rb").read().decode("ascii").replace("\r\n", "\n").split("\n")
        check(text[:-1] == old_wand(cal, None, out) and text[-1] == "",
              f"wand.export_dlt_csv(pixel_origin_out={out}) writes the same lines as before")
    check(wand.export_dlt_csv.__code__.co_names.count("dlt_csv_matlab") == 1, "and is the thin wrapper")


# ================================================================ 15. R19
@section("R19 calibio.read_lens goes through lens.read_lens_for; every file kind reads as before")
def _r19():
    prof = lens.LensProfile(W, H, K.copy(), np.array([-0.2, 0.05, 0.001, -0.002, 0.0]), False, 0.31, 12, "t")
    d = os.path.join(OUT, "lens")
    os.makedirs(d, exist_ok=True)
    kl = os.path.join(d, "a.klens.json")
    prof.save(kl)
    files = {"klens": kl}
    for ext in (".yml", ".json", ".xml"):
        f = os.path.join(d, "cv" + ext)
        calibio.write_lens(prof, f)
        files["opencv" + ext] = f
    argus = os.path.join(d, "argus.txt")
    calibio.write_lens(prof, argus)
    files["argus"] = argus
    multi = os.path.join(d, "multi.txt")
    with open(multi, "w", encoding="utf-8") as fh:
        fh.write("1 400 320 240 160 120 1 -0.1 0.01 0 0 0\n2 410 320 240 161 121 1 -0.2 0.02 0 0 0\n")
    files["argus2"] = multi
    for kind, f in files.items():
        view = 1 if kind == "argus2" else 0
        got = calibio.read_lens(f, view)
        via, why = lens.read_lens_for(f, view)
        check(np.allclose(got.K, via.K) and np.allclose(got.dist, via.dist),
              f"{kind}: read_lens == read_lens_for")
        if kind == "argus2":
            check(abs(got.K[0, 0] - 410) < 1e-9, "the line for camera 2")
        else:
            check(np.allclose(got.K, K) and np.allclose(np.asarray(got.dist)[:5], prof.dist), f"{kind}: the values")
    for bad, why in ((os.path.join(d, "nope.yml"), "no such file"),):
        try:
            calibio.read_lens(bad)
            check(False, f"{bad} was read")
        except calibio.CalibFormatError as e:
            check(why in str(e), f"a missing file is said: {e}")
    junk = os.path.join(d, "junk.txt")
    open(junk, "w").write("this is not a lens\n")
    try:
        calibio.read_lens(junk)
        check(False, "junk read as a lens")
    except calibio.CalibFormatError as e:
        check("not a lens file" in str(e), f"a text that is no lens: {e}")
    try:
        calibio.read_lens(multi, 5)
        check(False, "a camera the profile has no line for was given one")
    except calibio.CalibFormatError as e:
        check("no line for camera 6" in str(e), "an Argus file with no line for the camera is refused by name")


# ================================================================ 16. R17
@section("R17 one shared frame reader: bodyview and render use it, I41 behaviour kept")
def _r17():
    from kinetrace import bodyview
    from kinetrace.video_source import FrameReader          # (not there before R17)
    r = FrameReader(PATHS[0], 5)
    f5, f6 = r.read(5), r.read(6)
    f9 = r.read(9)                               # a sampled run skips 7, 8
    check(f5 is not None and f6 is not None and f9 is not None and r.failed is None, "frames read in order")
    check(abs(int(f9[0, 0, 0]) - (40 + 9)) <= 6, f"frame 9 is frame 9 ({int(f9[0, 0, 0])})")
    gone = r.read(NF + 3)
    check(gone is None and r.failed == NF, f"the first frame that does not decode is remembered: {r.failed}")
    check(r.read(NF + 5) is None and r.failed == NF, "and stays the first")
    r.reset(0)
    check(r.failed is None and r.read(0) is not None, "reset starts over")
    r.release()
    br = bodyview._FrameReader(PATHS[0], 0)
    check(isinstance(br, FrameReader), "the Body run / export read through it")
    br.release()
    try:
        bodyview._FrameReader(os.path.join(OUT, "missing.mp4"), 0)
        check(False, "a missing video was opened")
    except bodyview.BodyRunProblem as e:
        check("could not open" in str(e), f"a video that does not open is the run's sentence: {e}")
    # the overlay export keeps its I41 behaviour
    s = TrackingSession(PATHS[0], NF, FPS, W, H)
    s.add_landmark("a")
    out = os.path.join(OUT, "ov.mp4")
    ren = OverlayRenderer(s, PATHS[0], out, OverlayOptions(30, 49, 0.5), [])
    done = {}
    ren.finished_ok.connect(lambda path, codec: done.update(ok=path))
    ren.error.connect(lambda m: done.update(err=m))
    ren.run()
    check("ok" in done and 9 <= ren.frames_written <= 10 and ren.note, f"a range past the end: {ren.frames_written} "
          f"frames, note {ren.note[:70]!r}")
    check(f"Frame {30 + ren.frames_written}" in ren.note and os.path.isfile(out), "the note names the first failed frame")
    ren2 = OverlayRenderer(s, PATHS[0], os.path.join(OUT, "ov2.mp4"), OverlayOptions(60, 70, 0.5), [])
    done.clear()
    ren2.finished_ok.connect(lambda path, codec: done.update(ok=path))
    ren2.error.connect(lambda m: done.update(err=m))
    ren2.run()
    check("err" in done and "could not be decoded" in done["err"] and not os.path.exists(os.path.join(OUT, "ov2.mp4")),
          "a range wholly past the end: a sentence and no empty file")
    ren3 = OverlayRenderer(s, os.path.join(OUT, "missing.mp4"), os.path.join(OUT, "ov3.mp4"), OverlayOptions(0, 3), [])
    done.clear()
    ren3.error.connect(lambda m: done.update(err=m))
    ren3.run()
    check("could not open" in done.get("err", ""), f"a video that does not open: {done.get('err')!r}")


# ================================================================ 17. R15
@section("R15 projectfile walks session.POINT_ARRAYS; the saved bytes are unchanged")
def _r15():
    names = [a.name for a in POINT_ARRAYS]
    check(sorted(names) == sorted(["tracks", "confidence", "visibility", "manual", "tracked", "occluded", "radius"]),
          f"the seven arrays {names}")
    old = dict(tracks=np.full((6, 3, 2), np.nan, np.float32), confidence=np.zeros((6, 3), np.float32),
               visibility=np.zeros((6, 3), bool), manual=np.zeros((6, 3), bool),
               tracked=np.zeros((6, 3), bool), occluded=np.zeros((6, 3), bool),
               radius=np.full((6, 3), np.nan, np.float32))
    new = projectfile._empty_tracks(6, 3)
    check(sorted(new) == sorted(old) and all(new[k].dtype == old[k].dtype and new[k].shape == old[k].shape
                                             and np.array_equal(new[k], old[k], equal_nan=True) for k in old),
          "_empty_tracks = the old dict, dtype by dtype")
    s = TrackingSession(PATHS[0], NF, FPS, W, H)
    s.add_landmark("a")
    s.set_position(3, 0, 10.5, 20.25)
    files = {}
    projectfile._freeze_camera(files, "cameras/c", s, True)
    check(all(f"cameras/c/{n}.npy" in files for n in names), "the recovery copy holds every array")
    check(all(np.array_equal(files[f"cameras/c/{n}.npy"], getattr(s, n), equal_nan=True) for n in names),
          "with the session's own values")
    # a project round-trips through the folder (the byte-for-byte proof against the old code is in the report)
    q = projectfile.load(PROJ)
    check(all(np.array_equal(getattr(q.sessions[0], n), getattr(make_project(PATHS).sessions[0], n), equal_nan=True)
              for n in names), "a saved project reads back identical")


# ================================================================ 18. R16
@section("R16 the project README names every file it writes")
def _r16():
    txt = projectfile._readme(["camA"], ["camA"])
    for nm in ("view.json", "segment.json", "spots.json", "ball_prompts.json", "skeleton.json",
               "reconstruction/meta.json"):
        check(nm in txt, f"README names {nm}")


# ================================================================ 19. Body result while closing
@section("a Body result landing during the close does not open the side-by-side window")
def _body_close():
    class BodyThread(QThread):
        done = Signal(object)

        def __init__(self, track):
            super().__init__()
            self.track = track
            self.target = None
            self._stop = False

        def result_fits(self, s):
            return True

        def request_cancel(self):
            self._stop = True

        def run(self):
            self.done.emit(self.track)
            while not self._stop:
                time.sleep(0.01)
    for closing in (False, True):
        w = new_window()
        try:
            assert open_project(w, PROJ)
            tr = BodyTrack(NF, rig_of("coco17"), 1)
            tr.set_person(3, 0, joints2d=np.random.default_rng(1).uniform(0, 200, (17, 2)), conf=np.ones(17),
                          score=0.9)
            tr.examined = np.array([3], np.int64)
            th = BodyThread(tr)
            th.done.connect(w._on_body_done)
            w._body_worker = th
            th.start()
            time.sleep(0.2)
            if closing:
                w._stop_runs_for_close()          # cancels the thread, delivers its queued result
            else:
                pump(0.3)
                th._stop = True
                th.wait(3000)
            pump(0.2)
            check(w.session.has_body(), f"the result is stored (closing={closing})")
            check((w.body_win is None) == closing,
                  f"the window {'stays shut while closing' if closing else 'opens as usual'} "
                  f"(body_win={w.body_win is not None})")
            check(getattr(w, "_stopping_for_close", False) is False, "the closing flag is down again")
        finally:
            w._body_worker = None
            w.close()


# ================================================================ 20. R14 hand snap = ray_target
@section("R14 the hand snap is retrack.ray_target: same position as before, fractional frames included")
def _r14():
    pid = win.session.pid_by_name("M")
    s = win.session

    def old_way(f):
        guides = win._epipolar_guides(pid)
        if not guides:
            return None
        cur = s.tracks[f, pid]
        near = cur if np.isfinite(cur).all() else np.array([s.width / 2.0, s.height / 2.0])
        target = intersect_polylines([g[0] for g in guides], near, {})
        return (float(np.clip(target[0], 0, s.width - 1)), float(np.clip(target[1], 0, s.height - 1)),
                len(guides))

    def one(f, label, on_truth=True):
        win._goto(f, force=True)
        win._on_select(pid)
        truth = s.tracks[f, pid].copy()
        s.set_position(f, pid, float(truth[0]) + 30.0, float(truth[1]) - 20.0)
        want = old_way(f)
        win.statusBar().clearMessage()
        win._snap_to_epipolar(pid)
        got = s.tracks[f, pid]
        exp = np.array(want[:2], np.float32)
        check(want is not None and np.array_equal(got, exp), f"{label}: snapped {got} == old {exp}")
        if on_truth:
            check(float(np.hypot(*(got - truth))) < 1.0,
                  f"{label}: and back on the truth ({np.hypot(*(got - truth)):.3f} px)")
        n = want[2]
        check(f"snapped onto {n} camera{'s' if n != 1 else ''}' rays" in win.statusBar().currentMessage(),
              f"{label}: the message names the {n} camera(s)")
    one(10, "integer offsets")
    p.set_offset(1, 0.4)
    p.set_offset(2, -0.3)
    one(12, "fractional offsets (the observations are interpolated, I253)")
    p.set_offset(1, 0.0)
    p.set_offset(2, 0.0)
    # two calibrated cameras are enough for the hand snap, not for the default re-track rule
    full = p.calibration
    p.calibration = Calibration(full.cameras[:2], "m", "two")
    try:
        check(retrack.ray_target(p, 0, "M", 14) is None, "the default rule still needs EVERY camera calibrated")
        one(14, "two calibrated cameras of three (one line: onto it, not along it)", on_truth=False)
    finally:
        p.calibration = full
    # a landmark the rays put outside the picture is refused with its sentence, nothing stored
    win._goto(16, force=True)
    win._on_select(pid)
    before = s.tracks[16, pid].copy()
    real = retrack.outside_by
    retrack.outside_by = lambda *a, **k: 500.0
    try:
        win._snap_to_epipolar(pid)
    finally:
        retrack.outside_by = real
    check(np.array_equal(s.tracks[16, pid], before) and "outside this camera's picture" in win.statusBar().currentMessage(),
          f"outside the picture: refused, {win.statusBar().currentMessage()[:70]!r}")
    # no other camera has it: the old sentence
    for c in (1, 2):
        p.sessions[c].tracked[16, pid] = False
    win._snap_to_epipolar(pid)
    check("no other camera has it" in win.statusBar().currentMessage(), "no other camera: said")


win.close()
print()
if FAILS:
    print(f"{len(FAILS)} FAILED:")
    for m in FAILS:
        print("  -", m)
    sys.exit(1)
print("VERIFY_REVIEW_LEFTOVERS PASSED")
