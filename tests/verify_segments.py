"""(G149) Any number of segments, tracked in ONE pass (owner 2026-10-03: "the user should be able to
add and track as many unique segments as they want"; decisions: each landmark belongs to a segment,
a lost / swapped segment ends alone while the rest go on, segments are shared by name across cameras).

  [1] the session: add / rename (unique, landmarks follow) / remove (landmarks fall back), the active
      segment, per-segment masks, summaries dispatched by "seg", undo per segment, a skeleton on the
      second segment (its own landmark names, head), head_pid per segment.
  [2] the project file: three segments and the landmarks' segments round-trip; old one-segment layout.
  [3] two cameras: a segment made in one is in the other (by name), rename and remove act on both.
  [4] the worker with a SAM stand-in (no GPU): two segments in one session, each its own object; each
      derived landmark from its own segment; one segment taken by another object ends ALONE (its
      silhouette blank from the gap) and the other goes on to the end, no auto-pause; a single segment
      alone still stops the run.
  [5] the window (offscreen, real clicks): ＋ New segment, a row per segment, the row clicked is the one S
      adds to, both drawn, a timeline lane each, a marquee over one lane + Delete clears that segment only
      (Ctrl+Z back), the run's specs carry both, removing one removes it in both cameras, save + reopen.

.venv\\Scripts\\python.exe tests\\verify_segments.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "segments")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
app = QApplication.instance() or QApplication([])

from kinetrace import projectfile, segmenter  # noqa: E402
from kinetrace import tracker as trk  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.segmenter import summarize_mask  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402

FAILS = []
W, H, N = 320, 240, 60


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def blob(cx, cy, r=10):
    m = np.zeros((H, W), bool)
    cv2.circle(m.view(np.uint8), (int(round(cx)), int(round(cy))), r, 1, -1)
    return m


# ------------------------------------------------------------------ [1] the session
print("[1] the session")
s = TrackingSession("x.mp4", N, 30.0, W, H)
k0 = s.add_segment("lizard")
k1 = s.add_segment("lizard")
check((k0, k1) == (0, 1) and s.segment_names() == ["lizard", "lizard 2"] and s.active_seg == 1
      and s.animal.name == "lizard 2" and s.segments[0].color != s.segments[1].color,
      "two segments: unique names, own colours, the new one active", s.segment_names())
p0 = s.add_point(0, 10, 10, name="snout")
p1 = s.add_point(0, 50, 50, name="nose")
s.points[p1].segment = "lizard 2"
check(s.segment_of(p0) is None and s.segment_of(p1) == 1, "a landmark's animal ('' = Scene, G153)")
check(s.rename_animal("gecko", 1) == "gecko" and s.points[p1].segment == "gecko" and s.points[p1].name == "gecko nose",
      "a rename takes its landmarks along, named '<animal> <part>' (G153)", s.points[p1].name)
check(s.rename_animal("lizard", 1) == "lizard 2", "a name another segment has gets a number")
s.rename_animal("gecko", 1)
s.write_mask(3, blob(100, 100)[::1], 1.0, 5.0, i=0)
s.write_mask_summaries([dict(summarize_mask(blob(200, 120), 1.0, 5.0), frame=4, seg=1)])
check(s.seg_masks[0].has(3) and not s.seg_masks[1].has(3) and s.seg_masks[1].has(4) and not s.seg_masks[0].has(4),
      "masks and worker summaries go to their own segment (\"seg\")")
snap = s.snapshot()
s.clear_masks(0, N - 1, 1)
s.clear_masks(0, N - 1, 0)
s.restore(snap)
check(s.seg_masks[0].has(3) and s.seg_masks[1].has(4), "undo gives every segment its silhouettes back")
tpl = {"name": "t", "landmarks": ["head", "tail tip"], "bones": [["head", "tail tip"]],
       "derived": {"tail tip": "tip"}, "head": "head"}
s.apply_skeleton(tpl, segment=0)
new = s.apply_skeleton(tpl, segment=1)
names = [s.points[q].name for q in new]
sk1 = s.segments[1].skeleton
check(names == ["gecko head", "gecko tail tip"] and all(s.segment_of(q) == 1 for q in new)
      and sk1["head"] == "head" and ["head", "tail tip"] in sk1["bones"]
      and (s.pid_by_name("gecko head"), s.pid_by_name("gecko tail tip")) in s.bones(),
      "a skeleton on the second animal: its own landmark names, bones and head (G153: part names)", names)
check(s.head_pid(0) == s.pid_by_name("lizard head") and s.head_pid(1) == s.pid_by_name("gecko head"),
      "each animal's head landmark")
s.remove_segment(1)
check(s.segment_names() == ["lizard"] and s.points[p1].segment == "" and s.points[p1].name == "nose"
      and s.active_seg == 0, "remove: its landmarks go to Scene, named by their part (G153)", s.points[p1].name)

# ------------------------------------------------------------------ [2] the project file
print("[2] the project file")
s2 = TrackingSession(os.path.join(OUT, "a.mp4"), N, 30.0, W, H)
for nm in ("lizard", "gecko", "fly"):
    s2.add_segment(nm)
s2.segments[1].add_click(5, 10.0, 20.0)
s2.write_mask(6, blob(60, 60), 1.0, 4.0, i=2)
q = s2.add_point(0, 1, 1, name="wing")
s2.points[q].segment = "fly"
projectfile.save(Project([s2]), os.path.join(OUT, "three.kinetrace"))
p2, _st, _m = projectfile.read(os.path.join(OUT, "three.kinetrace"))
r = p2.sessions[0]
check(r.segment_names() == ["lizard", "gecko", "fly"] and r.segments[1].prompts == {5: [(10.0, 20.0, 1)]}
      and r.seg_masks[2].has(6) and not r.seg_masks[0].has(6) and r.points[0].segment == "fly",
      "three segments, their clicks / silhouettes and the landmark's segment round-trip", r.segment_names())

# ------------------------------------------------------------------ [3] two cameras
print("[3] shared by name across cameras")
a, b = TrackingSession("a.mp4", N, 30.0, W, H), TrackingSession("b.mp4", N, 30.0, W, H)
pr = Project([a, b])
a.add_segment("lizard")
a.add_segment("gecko")
pr.sync_landmarks(0)
check(b.segment_names() == ["lizard", "gecko"] and not b.segments[1].prompts and not b.seg_masks[1].n_masked(),
      "a segment made in one camera is in the other, empty")
check(pr.rename_segment("gecko", "skink") == "skink" and a.segment_names() == b.segment_names() == ["lizard", "skink"],
      "a rename acts in every camera")
check(pr.remove_segment("lizard") == [0, 1] and a.segment_names() == b.segment_names() == ["skink"],
      "a removal acts in every camera")

# ------------------------------------------------------------------ [4] the worker
print("[4] the worker, two segments in one SAM session (a stand-in)")
VIDEO = os.path.join(OUT, "plain.mp4")
vw = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in range(N):
    vw.write(np.full((H, W, 3), 60, np.uint8))
vw.release()


class Stand:
    """SAM stand-in: object 1 = a blob going right; object 2 = a blob going down, hidden at 20-29
    and then a look-alike far away (another object)."""

    def __init__(self, paths):
        self.paths = paths
        self.calls = []

    def new_session(self, start, size):
        stand = self

        class S:
            def step(self, native, idx, prompts):
                stand.calls.append((idx, sorted(p.obj_id for p in (prompts or []))))
                ids, ms, sc = [], [], []
                for obj, path in stand.paths.items():
                    c = path.get(idx)
                    ids.append(obj)
                    ms.append(np.zeros((H, W), bool) if c is None else blob(*c))
                    sc.append(8.0 if c is not None else -2.0)
                return segmenter.FrameMasks(idx, ids, np.stack(ms), np.array(sc, np.float32), (W, H), (W, H))
        return S()


right = {f: (30 + 3 * f, 60) for f in range(N)}
down = {**{f: (200, 40 + 2 * f) for f in range(20)}, **{f: None for f in range(20, 30)},
        **{f: (30, 220 - f) for f in range(30, N)}}


def run(paths, specs, derived=(), autopause=True):
    st = Stand(paths)
    saved = trk.get_segmenter
    trk.get_segmenter = lambda be, **k: st
    try:
        w = trk.TrackingWorker(VIDEO, 0, np.zeros((0, 2), np.float32), [], FrameCache(1 << 26), N, specs=[],
                               autopause=autopause, animals=specs, derived=list(derived))
        got = {"masks": {}, "auto": None, "end": None, "chunks": []}
        w.masks_ready.connect(lambda ss: [got["masks"].__setitem__((d["seg"], d["frame"]), d) for d in ss])
        w.chunk_ready.connect(lambda w0, tr, vi, cf, m, pend: got["chunks"].append((w0, tr.copy())))
        w.autopaused.connect(lambda f, p_: got.__setitem__("auto", (f, p_)))
        w.finished_ok.connect(lambda f, p_: got.__setitem__("end", (f, p_)))
        w.error.connect(lambda e: got.__setitem__("err", e))
        w.run()
        app.processEvents()
        got["w"], got["stand"] = w, st
        return got
    finally:
        trk.get_segmenter = saved


A0 = trk.AnimalSpec({0: [(30, 60, 1)]}, index=0, name="right")
A1 = trk.AnimalSpec({0: [(200, 40, 1)]}, index=1, name="down")
D = [trk.DerivedSpec(pid=7, spec="centroid", seg=0), trk.DerivedSpec(pid=8, spec="centroid", seg=1)]
g = run({1: right, 2: down}, [A0, A1], D)
check(not g.get("err") and g["stand"].calls[0] == (0, [1, 2]) and all(len(c[1]) == 0 for c in g["stand"].calls[1:]),
      "one SAM call per frame for both segments, each its own object (prompts on the first frame)",
      g.get("err", "")[-300:] or g["stand"].calls[:2])
m0 = [f for (k, f), d in g["masks"].items() if k == 0 and d["area"] > 0]
m1 = [f for (k, f), d in g["masks"].items() if k == 1 and d["area"] > 0]
check(len(m0) == N and sorted(m1) == list(range(20)), "segment 0 on every frame; segment 1 until it was hidden",
      (len(m0), sorted(m1)[-3:] if m1 else m1))
check(g["auto"] is None and g["end"][0] == N - 1 and g["w"]._seg_ended == {1: (20, "switched")},
      "the swapped segment ENDS alone at the gap (switched); the run goes on to the end, no auto-pause",
      (g["auto"], g["end"], g["w"]._seg_ended))
tr = np.concatenate([c[1] for c in sorted(g["chunks"], key=lambda c: c[0])])
check(np.allclose(tr[:20, 0], [right[f] for f in range(20)], atol=0.6)
      and np.allclose(tr[:20, 1], [down[f] for f in range(20)], atol=0.6)
      and np.isnan(tr[25:, 1]).all() and np.isfinite(tr[25:, 0]).all(),
      "each derived landmark comes from its OWN segment, the ended one's is blank")
g1 = run({1: down}, [trk.AnimalSpec({0: [(200, 40, 1)]}, index=0, name="down")])
check(g1["auto"] == (20, -1) and g1["w"]._autopause_reason == "switched",
      "a single segment alone still stops the run (nothing left to track)", (g1["auto"], g1["w"]._autopause_reason))
g2 = run({1: right, 2: right}, [trk.AnimalSpec({0: [(30, 60, 1)]}, index=0, name="a"),
                                trk.AnimalSpec({0: [(30, 60, 1)]}, index=3, name="b")])
check(all((3, f) in g2["masks"] for f in range(N)), "a summary carries the session's segment index (\"seg\")")

# ------------------------------------------------------------------ [5] the window
print("[5] through the window")
from kinetrace.app import READY, MainWindow  # noqa: E402


def clip(name, n=40):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for f in range(n):
        im = np.full((H, W, 3), 50, np.uint8)
        cv2.circle(im, (60 + f, 80), 12, (230, 230, 230), -1)
        cv2.circle(im, (240, 60 + f), 12, (40, 200, 40), -1)
        vw.write(im)
    vw.release()
    return path


VA, VB = clip("camA.mp4"), clip("camB.mp4")
win = MainWindow()
win.resize(1400, 900)
win.show()
app.setActiveWindow(win)
win._preview_mask = lambda: None           # no SAM here: the silhouettes are written by the test
win._open_video(VA)
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and not win._loading:
        break
pump(0.3)
check(win._add_view(VB), "two cameras")
win._set_active_view(0)
pump(0.3)
ss = win.session
win._on_animal_click(60, 80, True)          # S + click: the first animal
pump(0.1)
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump(0.2)
check(ss.n_segments == 2 and win._selected_segments() == [1] and win._animal_item(1) is not None
      and win.project.sessions[1].n_segments == 2,
      "+ Animal: a second LAYERS row, selected, in the other camera's list too (G154)")
win._on_animal_click(240, 60, True)
check(ss.segments[1].prompts == {0: [(240.0, 60.0, 1)]} and ss.segments[0].prompts == {0: [(60.0, 80.0, 1)]},
      "S adds the click to the selected animal")
lay = win.layers
rect = lay.visualItemRect(win._animal_item(0))
QTest.mouseClick(lay.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
pump(0.1)
check(win._selected_segments() == [0] and ss.active_seg == 0 and win._s_target() == 0,
      "a click on a row selects only it, and S's target follows (G154)", (win._selected_segments(), ss.active_seg))
for f in range(40):
    ss.write_mask(f, blob(60 + f, 80, 12), 1.0, 6.0, i=0)
    ss.write_mask(f, blob(240, 60 + f, 12), 1.0, 6.0, i=1)
win._refresh_overlay()
win._refresh_animal_panel()
pump(0.1)
cv = win.canvas
extra = getattr(cv, "_extra_masks", [])
check(cv._mask_item.isVisible() and extra and extra[0][0].isVisible(), "both silhouettes drawn")
from kinetrace.timeline import ANIMAL_H, EVENTS_H  # noqa: E402
tl = win.timeline
tl.set_session(ss)
check(tl._animal_h() == 2 * ANIMAL_H and tl._rows() == [("seg", 0), ("seg", 1)], "a timeline lane per animal")
y1 = EVENTS_H + 2 + ANIMAL_H + ANIMAL_H // 2
x0, x1 = int(tl._x_of(5)), int(tl._x_of(15))
QTest.mousePress(tl, Qt.LeftButton, Qt.ShiftModifier, QPoint(x0, y1))
QTest.mouseMove(tl, QPoint(x1, y1))
QTest.mouseRelease(tl, Qt.LeftButton, Qt.ShiftModifier, QPoint(x1, y1))
pump(0.1)
check(tl.sel_seg and tl.sel_segs == [1] and tl.sel_rows == [], "a Shift-drag along the second lane names that animal",
      (tl.sel_seg, tl.sel_segs, tl.sel_rows))
tl.setFocus()
QTest.keyClick(tl, Qt.Key_Delete)
pump(0.1)
check(not any(ss.seg_masks[1].has(f) for f in range(5, 16)) and ss.seg_masks[1].has(4) and ss.seg_masks[1].has(16)
      and all(ss.seg_masks[0].has(f) for f in range(5, 16)), "Delete cleared that animal's frames only")
QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
pump(0.1)
check(all(ss.seg_masks[1].has(f) for f in range(5, 16)), "Ctrl+Z brought them back")
win._select_segments([0, 1])
win._update_track_button()
specs = win._segment_specs([], set(), True, None, [])
check([a.index for a in specs.animals] == [0, 1] and [a.name for a in specs.animals] == ss.segment_names(),
      "a run with both rows selected carries both silhouettes (one session)", [a.index for a in specs.animals])
check(win.btn_track.text().startswith("Track · 2 silhouettes"), "the Track button says it", ascii(win.btn_track.text()))
target = os.path.join(OUT, "proj.kinetrace")
win.project_path = None
from PySide6.QtWidgets import QFileDialog  # noqa: E402
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
win._save_project()
pump(0.5)
rp, _st2, _m2 = projectfile.read(target)
check(rp.sessions[0].segment_names() == ss.segment_names() and rp.sessions[0].seg_masks[1].n_masked() == 40,
      "saved: both animals and their silhouettes", rp.sessions[0].segment_names())
# (G150, G154) Delete removes the SELECTED rows: both are selected; a Ctrl+click on the second row
# un-selects it
win._refresh_animal_panel()
win._select_segments([0, 1])
pump(0.1)
rect = lay.visualItemRect(win._animal_item(1))
QTest.mouseClick(lay.viewport(), Qt.LeftButton, Qt.ControlModifier, rect.center())
pump(0.1)
check(win._selected_segments() == [0] and win._s_target() == 0,
      "a Ctrl+click un-selects the second row (the first stays selected and is S's target)",
      (win._selected_segments(), ss.active_seg))
ASKED = []
_yes = QMessageBox.question
QMessageBox.question = staticmethod(lambda *a, **k: (ASKED.append(a[2]), QMessageBox.Yes)[1])
QTest.mouseClick(win.btn_clear_animal, Qt.LeftButton)
pump(0.1)
QMessageBox.question = _yes
check(ASKED and ASKED[0].startswith("Remove the animal animal:") and ss.segment_names() == ["animal 2"]
      and win.project.sessions[1].segment_names() == ["animal 2"]
      and win._animal_item(1) is None and tl._animal_h() == ANIMAL_H,
      "Delete removes the selected animal in both cameras, its row and its lane", (ASKED[:1], ss.segment_names()))
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump(0.2)
lay.select_only()
win._on_point_selection_changed()
QTest.mouseClick(win.btn_clear_animal, Qt.LeftButton)
pump(0.1)
check(ss.n_segments == 2 and "Select what to delete" in " ".join(n[0] for n in win.toast._notices),
      "nothing selected: Delete removes nothing and says what to select", ss.segment_names())
menu, acts = win._build_animal_menu(0)      # what a right-click on the first row builds (G151b)
win._animal_menu_action(acts["remove"], acts)
menu.deleteLater()
check(ss.n_segments == 1 and ss.segment_names() != ["animal 2"], "the row menu's Remove takes the row right-clicked",
      ss.segment_names())
win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_segments: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
