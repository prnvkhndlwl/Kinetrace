"""Offscreen GUI: multi-camera projects — adding videos, frame offsets, the
camera grid, per-view data separation, project round-trip and the all-cameras
export. No tracking runs, no GPU.

The two videos are the SAME world footage cut at different start frames, so a
correct offset makes both views show pixel-identical frames — that is what the
sync assertions check, rather than trusting the arithmetic alone.
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)

W, H, WORLD = 320, 240, 140
SHIFT = 17            # camA frame f and camB frame f-SHIFT are the same instant
LEN = 100


def build_pair():
    """One synthetic 'world', cut into two clips that start SHIFT frames apart."""
    frames = []
    for t in range(WORLD):
        img = np.full((H, W, 3), 18, np.uint8)
        img[:, :, 2] = 40 + (t % 17) * 3                  # a slow global ramp
        x = 30 + int(2.0 * t)
        y = 120 + int(50 * np.sin(t * 0.09))
        cv2.circle(img, (x % (W - 20) + 10, y), 9, (250, 250, 250), -1)
        cv2.putText(img, str(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 60), 2)
        frames.append(img)
    paths = []
    for name, start in (("camA.mp4", 0), ("camB.mp4", SHIFT)):
        p = os.path.join(SCRATCH, name)
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        for t in range(start, start + LEN):
            vw.write(frames[t])
        vw.release()
        paths.append(p)
    return paths


PATH_A, PATH_B = build_pair()
PROJ = os.path.join(SCRATCH, "multicam.cotrk")
for leftover in (PATH_A + ".cotracker.npz", PATH_B + ".cotracker.npz", PROJ):
    if os.path.exists(leftover):
        os.remove(leftover)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
_pick = {"files": [], "file": ""}
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (list(_pick["files"]), ""))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (_pick["file"], ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (_pick["file"], ""))

app = QApplication([])
from cotracker_app.app import MainWindow, READY
from cotracker_app.project import Project

win = MainWindow()
win.resize(1280, 860)
win.show()


def pump(cond, timeout, what):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def settle(seconds=0.35):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.005)


win._open_video(PATH_A)
pump(lambda: win.state == READY, 30, "open camA")
settle()
assert win.project is not None and win.project.n_views == 1, "one video = a one-view project"
assert len(win.grid.canvases) == 1, "a lone view must not grow grid chrome"
print("single-video project unchanged OK")

# ---- add a second camera ----------------------------------------------------
_pick["files"] = [PATH_B]
win._add_video_dialog()
settle()
p = win.project
assert p.n_views == 2 and len(win._views) == 2, f"expected 2 views, got {p.n_views}"
assert len(win.grid.canvases) == 2, "the grid must hold one canvas per camera"
assert p.active == 0, "adding a camera must not move the user off their working one"
assert win.grid.canvas(1) is not None and not win.grid.canvas(1)._interactive, \
    "a companion view must be view-only"
assert win.canvas is win.grid.canvas(0), "win.canvas must follow the ACTIVE view"
assert os.path.basename(PATH_B).startswith(p.name(1)[:4]), f"view name {p.name(1)!r}"
print("second camera added OK")

# ---- G3: every button of a camera row fits the panel at its narrowest -------
from PySide6.QtWidgets import QAbstractButton  # noqa: E402
_w0, _h0 = win.width(), win.height()
win.resize(win.minimumSizeHint().width(), max(_h0, 800))
settle()
for _ in range(10):
    QApplication.processEvents()
    time.sleep(0.02)
_lst = win.cameras.list
_vw = _lst.viewport().width()
assert not _lst.horizontalScrollBar().isVisible(), "a camera row must not need a horizontal scroll bar"
for _i, _row in enumerate(win.cameras._rows):
    assert _row.geometry().right() <= _vw, f"camera row {_i} is wider than the panel ({_row.width()} > {_vw})"
    for _b in _row.findChildren(QAbstractButton):
        _x = _b.mapTo(_lst.viewport(), _b.rect().topLeft()).x()
        assert _x + _b.width() <= _vw + 1, f"row {_i}: {_b.toolTip()[:30]!r} is cut off"
win.resize(_w0, _h0)
settle()
print(f"camera rows fit the narrowest panel ({_vw} px) OK")

# ---- the offset is what syncs them ------------------------------------------
win._on_view_offset(1, -SHIFT)     # camB frame f-SHIFT == camA frame f
assert p.offsets == [0, -SHIFT]
assert p.map_frame(0, 1, 40) == 40 - SHIFT
assert p.map_frame(0, 1, 3) is None, "before camB started rolling there is no frame"
assert p.map_frame(1, 0, 0) == SHIFT, "mapping must invert exactly"
win._goto(40, force=True)
settle(0.8)
assert win._views[1].want_frame == 40 - SHIFT, \
    f"companion asked for {win._views[1].want_frame}, expected {40 - SHIFT}"
# the real proof: with the offset right, both cameras show the same world frame
a = win._views[0].cache.get(40)
b = win._views[1].cache.get(40 - SHIFT)
assert a is not None and b is not None, "both views must have decoded"
assert np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() < 2.0, \
    "a correct offset must put both cameras on the SAME instant"
# and a wrong offset must not
c = win._views[1].cache.get(40 - SHIFT + 9)
if c is not None:
    assert np.abs(a.astype(np.int16) - c.astype(np.int16)).mean() > 5.0, \
        "the check would pass for any offset — it proves nothing"
lo, hi = p.coverage()
assert lo == SHIFT and hi == LEN - 1, f"overlap should be {SHIFT}..{LEN - 1}, got {lo}..{hi}"
print(f"frame offsets sync the views OK (overlap {lo}-{hi})")

# ---- the FIRST camera is the reference: its offset is 0, always ---------------
# Only differences between offsets affect anything, so the set could float; the
# numbers on screen would then be meaningless. Pinning view 0 at 0 makes every
# offset readable as "this many frames later than camera 1", whichever camera is
# being worked in.
from cotracker_app.project import REFERENCE_VIEW

assert REFERENCE_VIEW == 0
assert p.offsets[0] == 0, "the reference camera's offset must be 0"
_before = [p.map_frame(0, j, 40) for j in range(p.n_views)]
p.set_offset(0, 5)                       # editing the reference is meaningless
assert p.offsets[0] == 0, "the reference's offset must not be settable"
assert [p.map_frame(0, j, 40) for j in range(p.n_views)] == _before
p.set_active(1)                          # switching cameras must not renumber
assert p.offsets[0] == 0 and p.offsets == [0, -SHIFT], \
    f"switching the working camera must not move any offset: {p.offsets}"
assert [p.map_frame(0, j, 40) for j in range(p.n_views)] == _before
# aligning the reference against another camera shifts the OTHERS, keeping it 0
p.align_to(0, 20, 40)                    # cam1 frame 20 == cam2 frame 40
assert p.offsets[0] == 0, "the reference stays 0 even when it is what you aligned"
assert p.map_frame(0, 1, 20) == 40, "...and the alignment it expressed still holds"
p.set_active(0)
p.set_offset(1, -SHIFT)                  # back to the known-good alignment
assert p.offsets == [0, -SHIFT]
print("first camera is the reference (offset 0, not editable, stable) OK")

# ---- Align here: recover the offset from what the views are showing ----------
p.set_offset(1, 0)
win._goto(40, force=True)
settle(0.5)
win._views[1].want_frame = 40 - SHIFT      # the frame the user nudged it to
win._align_view_here(1)
assert p.offsets[1] == -SHIFT, f"Align here must recover the offset, got {p.offsets[1]}"
print("align-here OK")

# ---- each camera keeps its OWN points ---------------------------------------
win._goto(40, force=True)
win._on_add(100.0, 100.0)
win._on_add(140.0, 60.0)
app.processEvents()
assert p.sessions[0].n_points == 2 and p.sessions[1].n_points == 0, \
    "points must land in the working camera only"
tl_before = win.timeline.session
win._set_active_view(1)
settle()
assert p.active == 1 and win.session is p.sessions[1], "the working session must switch"
assert win.timeline.session is p.sessions[1] and win.timeline.session is not tl_before, \
    "the timeline must follow the working camera"
assert win.canvas is win.grid.canvas(1), "win.canvas must follow the active view"
assert win.current == 40 - SHIFT, \
    f"the playhead must land on the same instant, got frame {win.current}"
assert win.point_list.count() == 0, "the panel must show the new camera's points"
win._on_add(50.0, 50.0)
app.processEvents()
assert p.sessions[0].n_points == 2 and p.sessions[1].n_points == 1
assert not win.grid.canvas(0)._interactive and win.grid.canvas(1)._interactive, \
    "interactivity must move with the active view"
win._set_active_view(0)
settle()
assert win.current == 40, "switching back must return to the same instant"
print("per-camera data separation + view switching OK")

# ---- project round-trip: both views, names, offsets, active ------------------
win.project_path = None
_pick["file"] = PROJ
win._save_project_as()
assert os.path.exists(PROJ)
reopened = Project.load_npz(PROJ)
assert reopened.n_views == 2, "both cameras must be in the file"
assert reopened.offsets == [0, -SHIFT] and reopened.active == 0
assert [s.n_points for s in reopened.sessions] == [2, 1], "each view's points must survive"
assert reopened.names == p.names
assert os.path.basename(reopened.sessions[1].video_path) == "camB.mp4"
print("project round-trip (2 cameras) OK")

# reopen it through the app, which must re-open BOTH videos
_pick["file"] = PATH_A
win._open_project_from_path(PROJ)
pump(lambda: win.state == READY and win.project is not None and win.project.n_views == 2,
     30, "reopen multi-camera project")
settle()
p = win.project
assert p.n_views == 2 and len(win._views) == 2, "both cameras must reopen"
assert p.offsets == [0, -SHIFT]
assert [s.n_points for s in p.sessions] == [2, 1]
assert len(win.grid.canvases) == 2
print("reopen through the app OK")

# ---- all-cameras export -----------------------------------------------------
# give both views a landmark of the SAME name: that is what triangulation joins on
for i, s in enumerate(p.sessions):
    s.rename_point(0, "snout")
    f = p.map_frame(0, i, 40)
    s.set_position(f, 0, 10.0 + 5 * i, 20.0 + 5 * i)
out = os.path.join(SCRATCH, "allcams.csv")
written = p.export_multi_dltdv(out)
assert len(written) == 2 and os.path.exists(out)
head, *rows = open(out, encoding="utf-8").read().strip().split("\n")
cols = head.split(",")
assert cols[:4] == ["pt1_cam1_X", "pt1_cam1_Y", "pt1_cam2_X", "pt1_cam2_Y"], cols[:4]
# row k = frame k of the reference camera, from 0 (DLTdv8's layout; I18) -- it used
# to start at the overlap's first frame of the working camera, with nothing saying so
n_ref = p.sessions[0].n_frames
assert len(rows) == n_ref, f"{len(rows)} rows for {n_ref} reference frames"
lo = 0
cells = rows[40 - lo].split(",")
assert cells[0] and cells[2], "both cameras must contribute at a shared instant"
# DLTdv8 convention: top-left origin, first pixel = 1 (x + 1, y + 1) -- the frame the
# clicks and DLT coefficients of a DLTdv8 project are in
assert abs(float(cells[0]) - 11.0) < 1e-3 and abs(float(cells[2]) - 16.0) < 1e-3, cells[:4]
assert abs(float(cells[1]) - 21.0) < 1e-3, cells[1]
side = open(written[1], encoding="utf-8").read()
assert "convention,top-left origin; first pixel = 1" in side, side
assert "row k = frame k" in side, side
early = rows[0].split(",")
if p.map_frame(0, 1, 0) is None:          # camera 2 has no picture at reference frame 0
    assert early[2] == "NaN" and early[3] == "NaN", early[:4]   # NaN like DLTdv, never blank (I20)
# the older bottom-left variant is still available, explicitly
out_bl = os.path.join(SCRATCH, "allcams_bl.csv")
p.export_multi_dltdv(out_bl, flip_y=True)
cells_bl = open(out_bl, encoding="utf-8").read().strip().split("\n")[1 + 40 - lo].split(",")
assert abs(float(cells_bl[1]) - (H - 20.0)) < 1e-3, cells_bl[1]
print(f"all-cameras DLTdv export OK ({len(rows)} synced rows)")

# ---- solo hides (and stops decoding) the companions --------------------------
win.act_solo.setChecked(True)
settle()
assert win.grid.visible_indices() == [p.active], "solo shows only the working camera"
win._goto(50, force=True)
settle(0.3)
assert win._views[1].want_frame is None, "a hidden camera must not decode"
win.act_solo.setChecked(False)
settle(0.5)
assert win.grid.visible_indices() == [0, 1]
print("solo / show-all OK")

# ---- 15 cameras: the supported maximum ---------------------------------------
from cotracker_app.app import COMPANION_CACHE_BYTES
from cotracker_app.project import MAX_VIEWS
from cotracker_app.video_source import DEFAULT_CACHE_BYTES

assert MAX_VIEWS == 15
extra = []
for k in range(MAX_VIEWS - 2):                 # two are already open
    q = os.path.join(SCRATCH, f"cam{k + 3}.mp4")
    if not os.path.exists(q):
        import shutil
        shutil.copyfile(PATH_B, q)
    extra.append(q)
_pick["files"] = extra
t0 = time.time()
win._add_video_dialog()
settle(0.5)
p = win.project
assert p.n_views == MAX_VIEWS, f"expected {MAX_VIEWS} views, got {p.n_views}"
assert len(win.grid.canvases) == MAX_VIEWS and len(win._views) == MAX_VIEWS
print(f"15 cameras added in {time.time() - t0:.1f}s")

# one more must be refused, not silently dropped or crash
_pick["files"] = [PATH_A]
win._add_video_dialog()
assert win.project.n_views == MAX_VIEWS, "the 16th camera must be refused"

# memory: the WORKING view keeps the budget, companions get the small allowance
win._goto(60, force=True)
settle(1.5)
budgets = [rt.cache.max_bytes for rt in win._views]
assert budgets[p.active] == DEFAULT_CACHE_BYTES, "the working camera keeps the full cache"
assert all(b == COMPANION_CACHE_BYTES for i, b in enumerate(budgets) if i != p.active), \
    f"companions must share the small allowance, got {budgets}"
for i, rt in enumerate(win._views):
    if i != p.active:
        assert rt.cache.stats()[1] <= COMPANION_CACHE_BYTES, \
            f"companion {i} holds {rt.cache.stats()[1] / 1024**2:.0f} MB over its allowance"
held = sum(rt.cache.stats()[1] for rt in win._views)
print(f"15-camera caches hold {held / 1024**2:.0f} MB, every companion within its allowance")

# solo must hand back every companion decoder thread
win.act_solo.setChecked(True)
settle(0.8)
assert win.grid.visible_indices() == [p.active]
alive = [i for i, rt in enumerate(win._views) if rt.seek is not None]
assert alive == [p.active], f"solo must stop every companion decoder, still alive: {alive}"
win.act_solo.setChecked(False)
settle(0.8)
# ...and they come back ON DEMAND: a cached frame needs no decoder at all, so
# drop the companions' caches to force a real decode
for i, rt in enumerate(win._views):
    if i != p.active:
        rt.cache.clear()
        rt.want_frame = None
win._goto(61, force=True)
settle(2.0)
assert sum(rt.seek is not None for rt in win._views) > 1, \
    "companions must restart their decoders when a frame is not cached"
print("solo releases all 14 companion decoders, show-all restores them on demand")

# switching to a far camera still lands on the same instant
win._set_active_view(MAX_VIEWS - 1)
settle(0.6)
assert win.project.active == MAX_VIEWS - 1
assert win._views[MAX_VIEWS - 1].cache.max_bytes == DEFAULT_CACHE_BYTES, \
    "the cache budget must follow the working camera"
assert win._views[0].cache.max_bytes == COMPANION_CACHE_BYTES
win._set_active_view(0)
settle(0.4)
while win.project.n_views > 2:                 # back to two for the removal check
    win._remove_view(win.project.n_views - 1)
settle()
assert win.project.n_views == 2
print("15-camera scaling OK")

# ---- toolbar toggles must reach EVERY view, not just the one that existed
# when the button was wired. Binding `self.canvas.set_follow` captured canvas 0's
# bound method, so a camera added later kept _follow_enabled=True and auto-framed
# its silhouette while the Follow toggle said off.
p = win.project
win.btn_follow.setChecked(False)
app.processEvents()
assert [cv._follow_enabled for cv in win.grid.canvases] == [False] * p.n_views, \
    "Follow off must reach every camera"
win.btn_follow.setChecked(True)
app.processEvents()
assert all(cv._follow_enabled for cv in win.grid.canvases), "Follow on must reach every camera"
win.btn_follow.setChecked(False)
win.btn_pan.setChecked(True)
app.processEvents()
assert all(cv._pan_mode for cv in win.grid.canvases), "the pan tool must reach every camera"
win.btn_pan.setChecked(False)
win.marker_spin.setValue(11)
app.processEvents()
assert all(cv._marker_radius == 11 for cv in win.grid.canvases)

# a camera added WHILE a toggle is off must start from that state
_pick["files"] = [os.path.join(SCRATCH, "cam3.mp4")]
win._add_video_dialog()
settle()
assert not win.grid.canvases[-1]._follow_enabled, \
    "a newly added camera must inherit the current Follow state, not the class default"
assert win.grid.canvases[-1]._marker_radius == 11, "...and the current marker size"

# the behaviour that bug produced: with Follow off, a companion's view must not
# move when its silhouette does
comp = win.grid.canvas(1)
cs = win.project.sessions[1]
cs.ensure_animal()
for _f in range(0, 60):
    _mm = np.zeros((H, W), bool)
    _mm[80:140, 10 + _f * 4:70 + _f * 4] = True
    cs.masks.set(_f, _mm, 9.0)
win._goto(20, force=True)
settle(0.4)
_z = round(comp.transform().m11(), 4)
for _f in (25, 35, 45):
    win._goto(_f, force=True)
    settle(0.25)
assert round(comp.transform().m11(), 4) == _z, \
    "with Follow off a companion view must not zoom to its silhouette"
print("toolbar toggles reach every camera; Follow off means off everywhere OK")
while win.project.n_views > 2:
    win._remove_view(win.project.n_views - 1)
settle()

# ---- the panel says which camera is the reference and locks its offset --------
win._refresh_cameras()
app.processEvents()
_rows = win.cameras._rows
assert len(_rows) >= 2
assert not _rows[0].spin.isEnabled(), "the reference camera's offset must not be editable"
assert "reference" in _rows[0].name.text(), \
    f"the reference row must say so: {_rows[0].name.text()!r}"
assert _rows[1].spin.isEnabled(), "every other camera's offset stays editable"
assert not _rows[0].btn_align.isEnabled(), "there is nothing to align the reference to"
win._set_active_view(1)
settle()
win._refresh_cameras()
app.processEvents()
assert not _rows[0].spin.isEnabled(), \
    "the reference stays locked even when another camera is being worked in"
assert _rows[1].spin.isEnabled(), "the working camera's own offset stays editable"
win._set_active_view(0)
settle()
print("camera panel marks and locks the reference row OK")

# ---- removing the reference keeps everyone's RELATIVE timing -------------------
_pair = Project([p.sessions[0], p.sessions[1]], list(p.names), [0, -SHIFT])
_gap = _pair.offsets[1] - _pair.offsets[0]
_pair.remove_view(0)
assert _pair.offsets[0] == 0, "the promoted camera becomes the new zero"
_pair2 = Project([p.sessions[0], p.sessions[1]], list(p.names), [0, -SHIFT])
_pair2.remove_view(1)
assert _pair2.offsets == [0]
print("removing the reference re-zeroes without changing relative timing OK")

# ---- removing a camera -------------------------------------------------------
win._remove_view(1)
settle()
assert win.project.n_views == 1 and len(win._views) == 1 and len(win.grid.canvases) == 1
assert win.project.sessions[0].n_points == 2, "the surviving camera keeps its data"
assert win.session is win.project.sessions[0]
print("camera removal OK")

win._dev_probe.wait(180_000)
win.close()
app.processEvents()
for leftover in (PATH_A + ".cotracker.npz", PATH_B + ".cotracker.npz"):
    if os.path.exists(leftover):
        os.remove(leftover)
print("MULTICAM PASSED")
