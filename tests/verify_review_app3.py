"""app.py (region 3: saving / opening / the 3D entries / Body handlers / exports / close) fixes from the
2026-10-02 code review (docs/AUDIT.md: I168 I169 I170 I172 I183 I197 I198 I201 I207 I208 I209 I242 I243
I244 G75 G76 G78 G96 G106 G107 G108 G109 G110 G113 G119 G140 I249 + the owner decision "DLTdv = points
only"). Each check FAILS on the code before the fixes. Offscreen, no GPU: a 3-camera synthetic project
(known DLT cameras, static landmarks + a moving one) is opened through the real MainWindow.

Run: .venv\\Scripts\\python.exe tests\\verify_review_app3.py
"""
import os
import sys
import time
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(errors="replace")

OUT = os.path.join(ROOT, "tests", "out", "review_app3")
import shutil  # noqa: E402

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "recovery"), exist_ok=True)
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_LOG_DIR"] = os.path.join(OUT, "logs")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
from PySide6.QtCore import QThread, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QMessageBox  # noqa: E402

from kinetrace import app as app_mod  # noqa: E402
from kinetrace import calib, calibio, projectfile, recovery  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, dlt_from_camera  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

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


def make_project(tag, paths, cal=CAL, n_frames=NF):
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
    proj = Project(sessions, ["camA", "camB", "camC"], [0, 0, 0])
    proj.calibration = cal
    return proj


PATHS = make_videos("rig")
PROJ = os.path.join(OUT, "rig.kinetrace")
make_project("rig", PATHS).save(PROJ)
PROJ_ID = projectfile.read_meta(PROJ)["project_id"]

# ---------------------------------------------------------------- dialogs stubbed
ASK = {"answer": QMessageBox.Yes, "log": [], "hook": None}
INFO, WARN, CRIT, TOASTS = [], [], [], []


def _question(*a, **k):
    text = str(a[2]) if len(a) > 2 else ""
    ASK["log"].append(text)
    if ASK["hook"] is not None:
        ASK["hook"](text)
    return ASK["answer"]


QMessageBox.question = staticmethod(_question)
QMessageBox.information = staticmethod(lambda *a, **k: INFO.append(str(a[2]) if len(a) > 2 else "") or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: WARN.append(str(a[2]) if len(a) > 2 else "") or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: CRIT.append(str(a[2]) if len(a) > 2 else "") or QMessageBox.Ok)
NEXT = {"save": "", "filter": "", "open": ""}
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (NEXT["save"], NEXT["filter"]))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (NEXT["open"], ""))

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


def drop_recon():
    """Clear the 3D result between sections (the old code has no _drop_reconstruction)."""
    if hasattr(win, "_drop_reconstruction"):
        win._drop_reconstruction()
    else:
        win.project.reconstruction = None
        win._hull_cache.clear()


def clear_logs():
    for L in (INFO, WARN, CRIT, TOASTS, ASK["log"]):
        L.clear()


win = new_window()
assert open_project(win, PROJ), "the synthetic project did not open"
p = win.project
print("project opened: 3 cameras, calibration", p.calibration is not None)


# ================================================================ before any 3D result
@section("G108 nothing to export: the reason only, never 'Exported'; no empty silhouette file")
def _g108_empty():
    clear_logs()
    NEXT["save"], NEXT["filter"] = os.path.join(OUT, "no3d_xyz.csv"), next(
        f[0] for f in win.EXPORT_FORMATS if f[2] == "xyz")
    win._export_dialog()
    pump()
    said = " ".join(TOASTS)
    check(not os.path.exists(NEXT["save"]), "no 3D result: no xyz file written")
    check("Exported" not in said and "3D" in said, f"no 'Exported' message, the reason instead: {said[:90]!r}")
    clear_logs()
    NEXT["save"], NEXT["filter"] = os.path.join(OUT, "nosil.json"), next(
        f[0] for f in win.EXPORT_FORMATS if f[2] == "sil_json")
    win._export_dialog()
    pump()
    check(not os.path.exists(NEXT["save"]) and "Exported" not in " ".join(TOASTS),
          "no silhouette: the JSON file is not even created, and nothing says 'Exported'")


@section("I170 Set World Axes without a 3D result says 'Reconstruct first'")
def _i170_none():
    clear_logs()
    check(hasattr(win, "act_set_axes"), "3D menu has a 'Set World Axes' action")
    win._set_world_axes()
    check(any("Reconstruct" in t for t in INFO), f"asks to reconstruct first: {INFO[:1]}")


# ================================================================ reconstruct, then the axes
win._reconstruct_3d(quiet=True)
pump(0.2)
REC0 = p.reconstruction
check(REC0 is not None and REC0.names == NAMES, "reconstruction of the synthetic rig")
IDX = {n: i for i, n in enumerate(NAMES)}
TRUE = {n: world(n, 0) for n in NAMES}


@section("I170 Set World Axes: the three landmarks land on the origin / +X / the XY plane, pixels unchanged")
def _i170_axes():
    cal_before = p.calibration
    orig_exec = app_mod._WorldAxesDialog.exec
    seen = {}

    def fake_exec(self):
        seen["ok_enabled_initially"] = self.buttons.button(QDialogButtonBox.Ok).isEnabled()
        self.cb_x.setCurrentIndex(self.cb_origin.currentIndex())          # same landmark twice
        seen["same_refused"] = not self.buttons.button(QDialogButtonBox.Ok).isEnabled()
        self.rb_now.setChecked(True)
        self.cb_origin.setCurrentText("O")
        self.cb_x.setCurrentText("PX")
        self.cb_y.setCurrentText("PY")
        self._accept()
        return QDialog.Accepted
    app_mod._WorldAxesDialog.exec = fake_exec
    try:
        clear_logs()
        win._set_world_axes()
        pump()
    finally:
        app_mod._WorldAxesDialog.exec = orig_exec
    cal2, rec2 = p.calibration, p.reconstruction
    check(cal2 is not cal_before and "world:" in (cal2.source or ""), f"the calibration is re-expressed ({cal2.source})")
    check(seen.get("ok_enabled_initially") and seen.get("same_refused"),
          "the dialog starts valid and refuses one landmark chosen twice")
    x = rec2.xyz[0]
    o_new, px_new, py_new = x[IDX["O"]], x[IDX["PX"]], x[IDX["PY"]]
    dx = np.linalg.norm(TRUE["PX"] - TRUE["O"])
    check(np.allclose(o_new, 0, atol=1e-5), f"the origin landmark is (0, 0, 0): {o_new}")
    check(np.allclose(px_new, [dx, 0, 0], atol=1e-5), f"the +X landmark is on +X at {dx:.4f}: {px_new}")
    check(abs(py_new[2]) < 1e-5 and py_new[1] > 0, f"the +Y landmark is in the XY plane with y > 0: {py_new}")
    ex = (TRUE["PX"] - TRUE["O"]) / dx
    rest = (TRUE["PY"] - TRUE["O"]) - ((TRUE["PY"] - TRUE["O"]) @ ex) * ex
    ey = rest / np.linalg.norm(rest)
    ez = np.cross(ex, ey)
    m = x[IDX["M"]]
    expect = np.array([(TRUE["M"] - TRUE["O"]) @ ex, (TRUE["M"] - TRUE["O"]) @ ey, (TRUE["M"] - TRUE["O"]) @ ez])
    check(np.allclose(m, expect, atol=1e-5), f"a fourth landmark follows (+Z by the right-hand rule): {m} vs {expect}")
    pts = np.random.RandomState(5).uniform(-0.4, 0.4, (30, 3))
    # new coords of an old point: X_new = B (X_old - o); recover B, o from the three landmarks
    B = np.stack([ex, ey, ez])
    worst = 0.0
    for c in range(3):
        a = cal_before.cameras[c].project(pts)
        b = cal2.cameras[c].project((pts - TRUE["O"]) @ B.T)
        worst = max(worst, float(np.abs(a - b).max()))
    check(worst < 1e-4, f"every camera projects every point to the same pixel (max change {worst:.2e} px)")
    check(p.dirty and not win._hull_cache and any("World axes set" in t for t in TOASTS),
          "marked changed, volumes cleared, the notice says what was done")
    # the mirrored (left-handed) case: the world is made right-handed too
    mir = cal2.reframed(np.diag([1.0, 1.0, -1.0]), np.zeros(3), "world: test mirror")
    check(calib.world_is_left_handed(mir) and not calib.world_is_left_handed(cal2),
          "a Z-mirrored calibration is left-handed, the chosen world is not")


# ================================================================ 3D exports go through the models
@section("I170 / I215 Ctrl+E 3D points are in the exported cameras' world; notes say what was left out")
def _i170_export():
    mirrored = CAL.reframed(np.diag([1.0, 1.0, -1.0]), np.zeros(3), "mirrored world")
    p.calibration = mirrored
    drop_recon()
    win._reconstruct_3d(quiet=True)
    pump(0.2)
    r = p.reconstruction
    z_raw = r.xyz[0, IDX["PX"], 2]
    check(abs(z_raw + TRUE["PX"][2]) < 1e-4, f"the left-handed world reconstructs z negated ({z_raw:.4f})")
    out = os.path.join(OUT, "mir_xyz.csv")
    files, notes = win._export_one("xyz", out)
    rows = open(out, encoding="utf-8").read().splitlines()
    head = rows[0].split(",")
    col = head.index("PX_Z")
    z_out = float(rows[1].split(",")[col])
    check(abs(z_out - TRUE["PX"][2]) < 1e-4, f"xyz CSV: exported z is the cameras' world ({z_out:.4f}, truth "
          f"{TRUE['PX'][2]:.4f})")
    out2 = os.path.join(OUT, "mir_xyzpts.csv")
    win._export_one("xyz_dltdv", out2)
    row0 = open(out2, encoding="utf-8").read().splitlines()[1].split(",")
    check(abs(float(row0[IDX["PX"] * 3 + 2]) - TRUE["PX"][2]) < 1e-4, "xyzpts: the same world")
    out3 = os.path.join(OUT, "mir_anipose.csv")
    win._export_one("xyz_anipose", out3)
    rows3 = open(out3, encoding="utf-8").read().splitlines()
    h3 = rows3[0].split(",")
    check(abs(float(rows3[1].split(",")[h3.index("PX_z")]) - TRUE["PX"][2]) < 1e-4, "Anipose points: the same world")
    check(any("mirrored" in n or "left-handed" in n for n in notes), f"a note says the world is mirrored: {notes[:1]}")
    # a result that starts before reference frame 0 (I215): the instants left out are said
    r2 = calib.Reconstruction(-3, list(r.names), np.concatenate([r.xyz[:3], r.xyz]), np.concatenate([r.residual[:3], r.residual]),
                              np.concatenate([r.n_cams[:3], r.n_cams]), r.unit, None)
    keep = p.reconstruction
    p.reconstruction = r2
    files, notes = win._export_one("xyz_dltdv", os.path.join(OUT, "early_xyzpts.csv"))
    p.reconstruction = keep
    check(any("before reference frame 0" in n for n in notes), f"left-out instants are said: {notes}")
    p.calibration = CAL
    drop_recon()
    win._reconstruct_3d(quiet=True)


@section("owner 2026-10-03: DLTdv exports are points only (no _segment.csv / _events.csv beside them)")
def _dltdv_only():
    s = p.sessions[0]
    s.ensure_animal()
    m = np.zeros((H, W), bool)
    m[100:140, 100:140] = True
    s.masks.set(3, m, 9.0)
    s.add_event("flap", 3, 8)
    for k in ("dltdv", "dltdv_bl"):
        d = os.path.join(OUT, "only_" + k)
        os.makedirs(d, exist_ok=True)
        files, _ = win._export_one(k, os.path.join(d, "t.csv"))
        names = sorted(os.listdir(d))
        check(names == ["t.csv", "t_pointnames.csv"], f"{k}: only the points + the pointnames sidecar: {names}")
    d = os.path.join(OUT, "only_multi")
    os.makedirs(d, exist_ok=True)
    win._export_one("multi", os.path.join(d, "m.csv"))
    names = sorted(os.listdir(d))
    check(not any(n.endswith("_segment.csv") or n.endswith("_events.csv") for n in names),
          f"all-cameras xypts: no silhouette / events file: {names}")
    d = os.path.join(OUT, "only_xyzpts")
    os.makedirs(d, exist_ok=True)
    win._export_one("xyz_dltdv", os.path.join(d, "x.csv"))
    names = sorted(os.listdir(d))
    check(names == ["x.csv", "x_pointnames.csv"], f"xyzpts: points + pointnames only: {names}")
    d = os.path.join(OUT, "only_wide")
    os.makedirs(d, exist_ok=True)
    win._export_one("wide", os.path.join(d, "w.csv"))
    names = sorted(os.listdir(d))
    check("w_segment.csv" in names and "w_events.csv" in names, f"the plain CSV export still carries them: {names}")
    # Everything: the events + segment files ONCE, beside the wide CSV
    d = os.path.join(OUT, "only_all")
    os.makedirs(d, exist_ok=True)
    NEXT["save"], NEXT["filter"] = os.path.join(d, "ev.csv"), next(f[0] for f in win.EXPORT_FORMATS if f[2] == "all")
    clear_logs()
    win._export_dialog()
    pump()
    names = sorted(os.listdir(d))
    segs = [n for n in names if n.endswith("_segment.csv")]
    evs = [n for n in names if n.endswith("_events.csv")]
    check(segs == ["ev_segment.csv"] and evs == ["ev_events.csv"], f"Everything: one segment, one events file: {segs} {evs}")
    check(not any(("dltdv" in n or "allcams" in n or "xyzpts" in n) and ("segment" in n or "events" in n) for n in names),
          "none of them beside a DLTdv file")


@section("G108 'Everything' goes on after a format fails and lists what failed and what was written")
def _g108_all():
    s = p.sessions[0]
    orig = s.export_mat

    def boom(path):
        raise PermissionError(13, "Permission denied", str(path))
    s.export_mat = boom
    d = os.path.join(OUT, "all_fail")
    os.makedirs(d, exist_ok=True)
    NEXT["save"], NEXT["filter"] = os.path.join(d, "x.csv"), next(f[0] for f in win.EXPORT_FORMATS if f[2] == "all")
    clear_logs()
    try:
        win._export_dialog()
        pump()
    finally:
        s.export_mat = orig
    names = os.listdir(d)
    check("x.csv" in names and any(n.startswith("x_xyz") for n in names), f"the other formats were still written: {len(names)} files")
    check(WARN and "MATLAB" in WARN[0] and "Written" in WARN[0], f"the failure is listed with what was written: {WARN[:1]}")
    check(not CRIT, "no crash dialog")


@section("G108 a cancelled PNG export says cancelled")
def _g108_png():
    s = p.sessions[0]
    d = os.path.join(OUT, "png_cancel")
    os.makedirs(d, exist_ok=True)
    files, notes = win._export_one("sil_png", os.path.join(d, "m.png"), cancelled=lambda: True)
    check(any(n.startswith("Cancelled") for n in notes), f"a note says cancelled: {notes}")


# ================================================================ imports
@section("I169 Import camera offsets: a file with no row for the reference camera asks first")
def _i169():
    f = os.path.join(OUT, "offs_noref.csv")
    open(f, "w", encoding="utf-8").write("camera,name,offset,rate\n2,camB,7,1\n3,camC,-3,1\n")
    NEXT["open"] = f
    ASK["answer"] = QMessageBox.No
    clear_logs()
    before = list(p.offsets)
    win._import_offsets()
    check(ASK["log"] and "camA" in ASK["log"][0], f"asked about the missing camA row: {ASK['log'][:1]}")
    check(list(p.offsets) == before, "No: nothing applied")
    ASK["answer"] = QMessageBox.Yes
    clear_logs()
    win._import_offsets()
    check(list(p.offsets) == [0.0, 7.0, -3.0], f"Yes: the file's numbers as written: {p.offsets}")
    p.offsets[:] = [0.0, 0.0, 0.0]
    f2 = os.path.join(OUT, "offs_rebase.csv")
    open(f2, "w", encoding="utf-8").write("camera,name,offset,rate\n1,camA,120,1\n2,camB,97,1\n3,camC,130,1\n")
    NEXT["open"] = f2
    clear_logs()
    win._import_offsets()
    check(list(p.offsets) == [0.0, -23.0, 10.0], f"a clap table is re-based on camA: {p.offsets}")
    check(any("re-based" in t for t in TOASTS), "the notice says the offsets were re-based")
    p.offsets[:] = [0.0, 0.0, 0.0]
    drop_recon()
    win._reconstruct_3d(quiet=True)


@section("I242 / G75 / G76 Sync keeps the 3D result when no offset moved; the estimate has progress + Cancel")
def _i242():
    from kinetrace import syncdialog
    orig = syncdialog.SyncDialog.exec
    syncdialog.SyncDialog.exec = lambda self: QDialog.Accepted        # ticked rows that change nothing
    try:
        rec = p.reconstruction
        win._sync_dialog()
        pump()
        check(p.reconstruction is rec and rec is not None, "an accepted sync that moved no offset keeps the 3D result")
    finally:
        syncdialog.SyncDialog.exec = orig
    got = {}
    real = calib.estimate_offsets

    def fake(sessions, cal, rates, offsets, t_range, names=None, search=1.0, step=0.05, refine=0.005, passes=2,
             report=None, progress=None, cancelled=None):
        got["progress"], got["cancelled"] = progress, cancelled
        if progress is not None:
            progress(1, 10)
        report.update(verdict="cancelled", why="Cancelled.", per_view={}, n_cells=0)
        return list(offsets), float("nan"), float("nan")
    calib.estimate_offsets = fake
    try:
        clear_logs()
        before = list(p.offsets)
        win._estimate_offsets_dialog()
    finally:
        calib.estimate_offsets = real
    check(got.get("progress") is not None and got.get("cancelled") is not None,
          "the search is given a progress callback and a cancel test")
    check(list(p.offsets) == before and not ASK["log"] and any("Cancelled" in t for t in TOASTS),
          "a cancelled search changes nothing, asks nothing and says so")

    # the rows say flat / none in words
    def fake2(sessions, cal, rates, offsets, t_range, names=None, search=1.0, step=0.05, refine=0.005, passes=2,
              report=None, progress=None, cancelled=None):
        report.update(verdict="flat", why="WHY-TEXT", per_view={1: "flat", 2: "none"}, n_cells=99, sharpness={})
        return [0.0, 0.2, 0.0], 1.0, 0.9
    calib.estimate_offsets = fake2
    ASK["answer"] = QMessageBox.No
    try:
        clear_logs()
        win._estimate_offsets_dialog()
    finally:
        calib.estimate_offsets = real
        ASK["answer"] = QMessageBox.Yes
    txt = ASK["log"][0] if ASK["log"] else ""
    check("WHY-TEXT" in txt and "do not move enough" in txt and "shares no landmark" in txt,
          f"the question uses the report's sentence and the flat / none labels: {txt[:120]!r}")


@section("I172 a hull with a camera cutting the animal is warned about")
def _i172():
    from kinetrace import hull as hullmod
    for c, s in enumerate(p.sessions):
        s.ensure_animal()
        uv = np.asarray(CAMS[c].project(world("M", 0))).reshape(-1, 2)[0]
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(uv[0]), int(uv[1])), 18, 1, -1)
        s.masks.set(0, m.astype(bool), 9.0)
    win._goto(0, force=True)
    orig = hullmod.Hull.thin_fraction
    hullmod.Hull.thin_fraction = lambda self, min_views=3: 0.4
    try:
        clear_logs()
        win._carve_hull_here(quiet=True)
        pump(0.2)
    finally:
        hullmod.Hull.thin_fraction = orig
    check(any("fewer than 3 cameras" in t for t in WARN), f"warned: {WARN[:1]}")


@section("G107 the 3D window and the hull cache use one rounding of the reference instant")
def _g107():
    win.act_view3d.setChecked(True)
    win._toggle_view3d(True)
    pump(0.1)
    p.set_offset(1, 0.5)
    win._set_active_view(1)
    win._goto(3, force=True)
    win._refresh_view3d(force=True)
    txt = win.view3d.info.text()
    # 2.5 is a tie: the way back to the reference rounds DOWN, like map_frame (I19 / I258), so the 3D
    # window, the hull key and the reference camera's tile all name frame 2
    ref_tile = p.map_frame(1, 0, 3)
    check("reference frame 2" in txt and win._reference_instant() == 2 == ref_tile,
          f"half-way instants follow map_frame: {txt[:50]!r}, tile {ref_tile}")
    win._set_active_view(0)
    p.set_offset(1, 0.0)


@section("I242 / G96 / I249 a new calibration drops the 3D result with its follow-ups; dialogs are deleted after use")
def _i242_more():
    from kinetrace import syncdialog
    counts = {"band": 0, "sync_deleted": 0, "lens_deleted": 0}
    real_band = win._update_disagreement

    def band():
        counts["band"] += 1
        return real_band()
    win._update_disagreement = band

    class FakeCalDlg:
        result_calibration = CAL

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.Accepted
    saved = app_mod.CalibrationDialog
    app_mod.CalibrationDialog = FakeCalDlg
    try:
        win._hull_cache[99] = ("v", "f", "h")
        win._import_calibration()
    finally:
        app_mod.CalibrationDialog = saved
        win._update_disagreement = real_band
    check(p.reconstruction is None and 99 not in win._hull_cache and counts["band"] >= 1,
          f"Import Calibration: result dropped, volumes cleared, the disagreement band refreshed ({counts['band']}x)")
    win._reconstruct_3d(quiet=True)
    orig_del, orig_exec = syncdialog.SyncDialog.deleteLater, syncdialog.SyncDialog.exec
    syncdialog.SyncDialog.exec = lambda self: QDialog.Rejected
    syncdialog.SyncDialog.deleteLater = lambda self: counts.__setitem__("sync_deleted", counts["sync_deleted"] + 1)
    try:
        win._sync_dialog()
    finally:
        syncdialog.SyncDialog.exec, syncdialog.SyncDialog.deleteLater = orig_exec, orig_del
    check(counts["sync_deleted"] == 1, "the sync dialog is deleted after use (G96)")
    from kinetrace import lenswizard

    class FakeWiz:
        result_profile, result_view = None, None

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.Rejected

        def result_views(self):
            return []

        def deleteLater(self):
            counts["lens_deleted"] += 1
    saved_w = lenswizard.LensWizard
    lenswizard.LensWizard = FakeWiz
    try:
        win._lens_wizard()
    finally:
        lenswizard.LensWizard = saved_w
    check(counts["lens_deleted"] == 1, "the lens wizard is deleted after its result is read (I249)")


@section("G113 Export Calibration names the files written before an error")
def _g113():
    from kinetrace import calibwizard

    class FakeExp:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.Accepted

        def chosen(self):
            return ["kcal"]
    import kinetrace.view3d as v3
    saved_dlg, saved_fn = v3.ExportCalibrationDialog, calibwizard.save_calibration_files
    v3.ExportCalibrationDialog = FakeExp

    def partial(res, grav, path, cal=None, written=None):
        written.append(path)
        raise PermissionError(13, "Permission denied", path + ".csv")
    calibwizard.save_calibration_files = partial
    NEXT["save"] = os.path.join(OUT, "cal_out.kcal.json")
    clear_logs()
    try:
        win._export_calibration()
    finally:
        v3.ExportCalibrationDialog, calibwizard.save_calibration_files = saved_dlg, saved_fn
    check(CRIT and "cal_out.kcal.json" in CRIT[0], f"the error names the file that was written: {CRIT[:1]}")


@section("G110 a save whose tidy-up failed says so instead of 'Nothing was changed'")
def _g110():
    from kinetrace import projectfile as pf
    orig = pf.write_folder_over

    def tidy(frozen, path, **kw):
        st = orig(frozen, path, **kw)
        st["tidy_up"] = ["the index in .cache: PermissionError"]
        return st
    pf.write_folder_over = tidy
    clear_logs()
    try:
        win.project.dirty = True
        ok = win._save_project()
    finally:
        pf.write_folder_over = orig
    check(ok and not CRIT and any("Not everything could be tidied up" in t for t in TOASTS),
          f"saved, with the tidy-up said: {TOASTS[:1]}")


@section("G109 opening a project ends a pending first-picture wait")
def _g109():
    w = new_window()
    tok = w._busy_push("Opening something", "")
    w._first_frame_token = tok
    w._first_frame_timer.start()
    assert open_project(w, PROJ)
    pump(1.0)
    check(not w._busy_stack, f"no stale loading card level is left ({len(w._busy_stack)})")
    w.close()


# ================================================================ save / open / recovery
@section("I201 / I244 / I243 saving: no orphaned writer, a failed first save does not block the name, the export is a copy")
def _save_section():
    from kinetrace import projectfile as pf
    # I244
    d = Path(OUT) / "firstfail.kinetrace"
    (d / ".cache").mkdir(parents=True)
    clear_logs()
    got = win._project_target(str(d))
    check(got is not None and not WARN, f"a folder holding only a failed save's .cache is accepted: {got}")
    # I243
    out = Path(OUT) / "copy.kinetrace"
    NEXT["save"] = str(out)
    win._export_single_file()
    pump()
    meta = pf.read(out)[2]
    check(out.is_file() and meta.get("project_id") != win._project_id,
          "Export Project as One File gives the copy its own project id")


@section("I201 the writer the autosave starts during the single-file question is not orphaned")
def _i201():
    from kinetrace import projectfile as pf
    single = os.path.join(OUT, "single.kinetrace")
    pf.save(make_project("single", PATHS), single, single_file=True)
    w = new_window()
    assert open_project(w, single)
    orig = pf.write

    def slow(frozen, path, **kw):
        time.sleep(1.2)
        return orig(frozen, path, **kw)
    pf.write = slow
    cap = []

    def hook(text):
        if text.startswith(("Save as a project folder", )) or "single-file project" in text:
            w.project.dirty = True
            w._recovery_sig = None
            w._autosave()                       # the 30 s timer firing while the question is open
            cap.append(w._save_worker)
    ASK["hook"] = hook
    try:
        w.project.dirty = True
        w.project.sessions[0].set_position(5, 0, 3.0, 4.0)
        ok = w._save_project()
    finally:
        pf.write = orig
        ASK["hook"] = None
    old = cap[0] if cap else None
    check(ok and old is not None, "the save went through the question")
    check(old is not None and (not old.isRunning() or old in app_mod._ORPHANS),
          "the earlier writer finished or is kept referenced (never dropped while running)")
    if old is not None:
        old.wait(5000)
    w.project.dirty = False
    w.close()


@section("I208 an empty stored video path is a missing video")
def _i208():
    clear_logs()
    NEXT["open"] = ""
    got = win._locate_video(Path(""), "camA")
    check(got is None and INFO, f"asked to locate it, nothing assumed: got {got}")


@section("I168 a project opened with a camera left out is detached (own id, no base save)")
def _i168():
    w = new_window()
    vids = make_videos("lost")                 # files no other window holds open (a held file cannot be renamed)
    f = os.path.join(OUT, "lost.kinetrace")
    make_project("lost", vids).save(f)
    saved_id = projectfile.read_meta(f)["project_id"]
    held = vids[2] + ".away"
    os.rename(vids[2], held)
    try:
        NEXT["open"] = ""
        clear_logs()
        ok = open_project(w, f, n=2)
    finally:
        os.rename(held, vids[2])
    check(ok, "opened with two cameras")
    check(w._project_id != saved_id and w._saved_at is None and w.project_path is None,
          f"detached: new id {w._project_id != saved_id}, no base save {w._saved_at is None}, "
          f"no path {w.project_path is None}")
    w.project.dirty = False
    w.close()


@section("I209 the recovery is set aside only after its copy opened")
def _i209():
    w = new_window()
    vids = make_videos("recov")
    f = os.path.join(OUT, "recov.kinetrace")
    make_project("recov", vids).save(f)
    assert open_project(w, f)
    w.project.sessions[0].set_position(7, 0, 9.0, 9.0)
    w.project.dirty = True
    w._autosave(wait=True)
    pid = w._project_id
    recovery.write_info(pid, base_saved_at="1999-01-01", last_path=f,
                        videos=[x.video_path for x in w.project.sessions], temporary=False)
    w.project.dirty = False
    w.close()                                  # releases the videos; the doctored recovery stays
    w = new_window()
    held = vids[0] + ".away"
    os.rename(vids[0], held)
    ASK["answer"] = QMessageBox.Yes            # "open them as a separate copy"
    NEXT["open"] = ""                          # ... and the video cannot be located
    try:
        w._open_project_from_path(f)
        pump(0.3)
    finally:
        os.rename(held, vids[0])
    still = recovery.find(pid)
    declined = Path(recovery.folder()[0]) / "declined"
    check(still is not None and not (declined.exists() and any(declined.iterdir())),
          f"an open that failed leaves the unsaved work where File > Recover finds it (found={still is not None})")
    w.close()


@section("I207 the working camera is limited to the project's length")
def _i207():
    long_paths = make_videos("long", 60)
    proj = make_project("short", long_paths, n_frames=40)
    f = os.path.join(OUT, "short.kinetrace")
    proj.save(f)
    w = new_window()
    assert open_project(w, f)
    check(w.n_frames == 40 and w.spin.maximum() == 39, f"frames {w.n_frames}, spin max {w.spin.maximum()}")
    w.project.dirty = False
    w.close()


# ================================================================ Body
@section("I183 / G78 Body: the miss message names the frames that kept their pose; Stop is a notice")
def _body():
    from kinetrace import bodyview
    from kinetrace.body import BodyTrack, rig_of
    s = win.session
    rig = rig_of("coco17")
    old = BodyTrack(s.n_frames, rig, 1)
    old.score[2, 0] = 0.9
    old.score[5, 0] = 0.9                  # a pose on a frame the next run examines and finds nobody on
    s.body = old
    calls = {}

    class FakeDlg:
        result_options = None

        def __init__(self, *a, **k):
            pass

        def exec(self):
            FakeDlg.result_options = type("O", (), {"backend": "vitpose", "start": 0, "end": 9, "step": 1})()
            return QDialog.Accepted

    class FakeWorker(QThread):
        progress = Signal(int, int, str)
        finished_ok = Signal(object)
        error = Signal(str)
        stopped = Signal(str)

        def __init__(self, *args, **kw):
            super().__init__()
            calls["args"], calls["kw"] = args, kw
            self.target = kw.get("target")

        def result_fits(self, session):
            return True

        def request_cancel(self):
            pass

        def run(self):
            if calls.get("mode") == "stop":
                self.stopped.emit("Stopped before the first frame was done.")
            else:
                new = BodyTrack(s.n_frames, rig, 1)
                new.examined = np.array([5], np.int64)
                self.finished_ok.emit(new)
    saved = (bodyview.BodyPoseWorker, bodyview.BodyRunDialog)
    bodyview.BodyPoseWorker, bodyview.BodyRunDialog = FakeWorker, FakeDlg
    try:
        calls["mode"] = "stop"
        clear_logs()
        win._body_run()
        for _ in range(100):
            pump(0.05)
            if win._body_worker is None:
                break
        check(len(calls["args"]) == 4, f"the no-op fps argument is no longer passed ({len(calls['args'])} positional)")
        check(any("Stopped before" in t for t in TOASTS), f"Stop is a notice: {TOASTS[:1]}")
        calls["mode"] = "miss"
        clear_logs()
        win._body_run()
        for _ in range(100):
            pump(0.05)
            if win._body_worker is None:
                break
    finally:
        bodyview.BodyPoseWorker, bodyview.BodyRunDialog = saved
    said = " ".join(TOASTS)
    check("it examined" in said and "earlier pose" in said, f"the miss says which frames kept their pose: {said[:140]!r}")
    check(win.session.body.n_posed() == 2, "both earlier poses are still there")
    s.body = None


# ================================================================ close
@section("I197 / I198 closing: no pass starts, the question comes first, a Body result lands, previews are cancelled")
def _close():
    from kinetrace.body import BodyTrack, rig_of
    w = new_window()
    assert open_project(w, PROJ)
    s = w.session
    rig = rig_of("coco17")

    class FakeBody(QThread):
        finished_ok = Signal(object)
        stopped_flag = False

        def __init__(self):
            super().__init__()
            self.target = s
            self.cancel = False

        def result_fits(self, session):
            return True

        def request_cancel(self):
            self.cancel = True

        def run(self):
            while not self.cancel:
                time.sleep(0.01)
            tr = BodyTrack(s.n_frames, rig, 1)
            tr.score[4, 0] = 0.8
            tr.examined = np.array([4], np.int64)
            self.finished_ok.emit(tr)

    class FakePreview(QThread):
        def __init__(self):
            super().__init__()
            self.cancelled = False

        def run(self):
            t_end = time.time() + 20
            while not self.cancelled and time.time() < t_end:
                time.sleep(0.01)

    fb, fp = FakeBody(), FakePreview()
    fb.finished_ok.connect(w._on_body_done)
    fb.finished.connect(w._end_body_run)
    w._body_worker = fb
    w._preview = fp
    fb.start()
    fp.start()
    w._passes = {"queue": [["x"]], "frame": 0, "snap": None, "msnaps": None}
    st = {"jobs": [1, 2], "done": [], "snaps": {}, "stretches": [], "before": {}, "thr": []}
    w._retrack = st
    w.project.dirty = True
    state = {}

    def hook(text):
        state["body_running_at_question"] = fb.isRunning()
        state["retrack_live_at_question"] = w._retrack is not None
    ASK["hook"], ASK["answer"] = hook, QMessageBox.Discard
    t0 = time.time()
    w.close()
    dt = time.time() - t0
    ASK["hook"], ASK["answer"] = None, QMessageBox.Yes
    check(state.get("body_running_at_question") is True and state.get("retrack_live_at_question") is True,
          "the question comes BEFORE a Body run / re-track is stopped")
    check(w._passes is None and st["jobs"] == [], "no second pass and no next re-track stretch can start")
    check(s.body is not None and s.body.n_posed() == 1, "the stopped Body run's result was merged, not thrown away")
    check(fp.cancelled and dt < 10, f"the mask preview was cancelled, not waited for 15 s ({dt:.1f} s)")
    fp.cancelled = True
    fp.wait(3000)


@section("G140 the close question no longer promises a recovery that Cancel does not give")
def _g140():
    w = new_window()
    assert open_project(w, PROJ)
    w.project.dirty = True
    clear_logs()
    ASK["answer"] = QMessageBox.Cancel
    w.close()
    ASK["answer"] = QMessageBox.Yes
    text = ASK["log"][0] if ASK["log"] else ""
    check(text and "Closing this dialog" not in text, f"question text: {text[:80]!r}")
    w.project.dirty = False
    w.close()


# ================================================================ G106 wand guidance
@section("G106 Calibrate Cameras with a Wand with no video explains what to do")
def _g106():
    w = new_window()
    clear_logs()
    w._wand_wizard()
    check(any("wand video" in t.lower() for t in INFO), f"guidance shown with no video open: {INFO[:1]}")
    w.close()


@section("G119 export names carry the camera in a multi-camera project")
def _g119():
    names = [win._default_output("_overlay.mp4"), win._default_output(".kcal.json", camera=False)]
    cam = win.project.name(win.project.active)
    check(cam in names[0] and cam not in names[1], f"{os.path.basename(names[0])} / {os.path.basename(names[1])}")


print()
if FAILS:
    print(f"VERIFY_REVIEW_APP3 FAILED ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    win.project.dirty = False
    sys.exit(1)
win.project.dirty = False
win.close()
print("VERIFY_REVIEW_APP3 PASSED")
