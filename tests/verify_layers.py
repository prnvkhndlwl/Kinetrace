"""(G152-G160) Animal layers (owner 2026-10-03: "if the user tracks a segment, lets say a squirrel, then that
can be a layer and the points tracked can be sublayers inside like adobe illustrator ... allow the user to
drag and drop points from one animal parent to another"), and the toolbar's two zooms.

  [0] G152  the timeline's zoom buttons sit on the timeline and zoom IT; the video's own buttons zoom the video
  [1] G153  the model: animals and Scene, "<animal> <part>" names, moving, renaming, the animal's skeleton
            (bones, head, template), undo, removal; two cameras by name; the project file (format 3)
  [2] G154/G155  LAYERS by real clicks: + Animal, N + click into the selected animal / the silhouette under
            the click / the selected point's animal / Scene (with a notice), + Point
  [3] G154  dragging points onto another animal or Scene (a real drop event): renamed in both cameras, one
            Ctrl+Z; a derived landmark is not dropped into Scene
  [4] G154/G156/G160  Track = the selection: an animal row = its silhouette and its points; a point alone
            runs alone unless its animal holds its points (opt-in); Scene points are never held
  [5] G157  skeleton: a template on the selected animal, a bone between two selected points (the menu),
            the head, a template saved from an animal and given to another
  [6] G154  the animal's row: rename (its points follow, both cameras), hold, remove keeping / deleting points
  [7] G158  timeline: each animal's silhouette lane above its points, Scene last; a lane click selects
  [8] G159  DeepLabCut multi-animal and SLEAP exports
  [9]       saved and opened again through the window

.venv\\Scripts\\python.exe tests\\verify_layers.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "layers")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

ANSWER = {"q": QMessageBox.Yes, "text": ""}
ASKED = []
QMessageBox.question = staticmethod(lambda *a, **k: (ASKED.append(a[2]), ANSWER["q"])[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: (ANSWER["text"], True))
QInputDialog.getItem = staticmethod(lambda p, t, l, items, cur=0, *a, **k: (items[cur], True))
app = QApplication.instance() or QApplication([])

from kinetrace import projectfile, skeletons  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession, qualified  # noqa: E402
from kinetrace.timeline import EVENTS_H, LANE_H  # noqa: E402

FAILS = []
W, H, N = 320, 240, 40


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    # the console is cp1252: tests print ASCII only (a "▾" in a message crashed the suite)
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.15):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def blob(cx, cy, r=14):
    m = np.zeros((H, W), bool)
    cv2.circle(m.view(np.uint8), (int(cx), int(cy)), r, 1, -1)
    return m


# ------------------------------------------------------------------ [1] the model
print("[1] the model")
s = TrackingSession("x.mp4", N, 30.0, W, H)
sq = s.add_segment("squirrel")
p_ref = s.add_point(0, 5, 5, name="rock")
check(s.segment_of(p_ref) is None and s.points_of(None) == [p_ref], "a point of no animal is in Scene")
moved = s.move_points([p_ref], sq)
check(moved == [("rock", "squirrel rock")] and s.part_name(p_ref) == "rock" and s.segment_of(p_ref) == sq,
      "moved into the animal: named '<animal> <part>', its part what the tree shows", moved)
t = {"name": "tiny", "head": "snout", "landmarks": ["snout", "neck", "tail"],
     "bones": [["snout", "neck"], ["neck", "tail"]], "derived": {"tail": "tip"}}
new = s.apply_skeleton(t, sq)
names = [s.points[q].name for q in new]
check(names == ["squirrel snout", "squirrel neck", "squirrel tail"] and s.points[new[2]].derived
      and s.segments[sq].skeleton["head"] == "snout" and len(s.bones()) == 2 and s.head_pid(sq) == new[0],
      "a template on the animal: its points, bones (part names) and head", names)
snap = s.snapshot()
check(s.rename_animal("chipmunk", sq) == "chipmunk" and s.points[new[0]].name == "chipmunk snout"
      and len(s.bones()) == 2, "renaming the animal renames its points; the bones still hold")
s.restore(snap)
check(s.segments[sq].name == "squirrel" and s.points[new[0]].name == "squirrel snout" and len(s.bones()) == 2,
      "Ctrl+Z gives the animal and its points their names back")
check(s.connect_bone(new[0], new[2]) and s.has_bone(new[0], new[2]) and len(s.bones()) == 3
      and s.bone_problem(new[0], p_ref) is None and s.bone_problem(new[0], new[0]),
      "a bone between two points of one animal")
sc = s.add_point(0, 9, 9, name="marker")
check(s.bone_problem(new[0], sc) is not None and not s.connect_bone(new[0], sc), "no bone to a Scene point")
check(s.set_head(new[1]) and s.head_pid(sq) == new[1] and not s.set_head(sc), "the head can be set; not a Scene one")
tpl = s.skeleton_template(sq, "mine")
check(tpl["landmarks"] == ["rock", "snout", "neck", "tail"] and ["snout", "tail"] in tpl["bones"]
      and tpl["head"] == "neck" and tpl["derived"] == {"tail": "tip"}, "the animal as a template, in parts", tpl)
s.remove_point(new[1])
check("neck" not in s.segments[sq].skeleton["landmarks"] and all("neck" not in b for b in s.segments[sq].skeleton["bones"])
      and "head" not in s.segments[sq].skeleton, "a deleted point leaves its animal's skeleton (I234)",
      s.segments[sq].skeleton)
gk = s.add_segment("gecko")
s.apply_skeleton(tpl, gk)
check(s.pid_by_name("gecko snout") is not None and s.pid_by_name("gecko neck") is not None
      and s.segments[gk].skeleton["head"] == "neck", "the same template on another animal: its own names")
ch = s.remove_segment(gk, keep_points=True)
check(s.segment_names() == ["squirrel"] and ("gecko snout", "snout") in ch and s.segment_of(s.pid_by_name("snout")) is None,
      "an animal removed keeping its points: they go to Scene by their part", ch[:2])
n0, n_sq = s.n_points, len(s.points_of(sq))
s.remove_segment(sq, keep_points=False)
check(n_sq == 3 and s.n_segments == 0 and s.n_points == n0 - n_sq, "an animal removed with its points",
      (n0, n_sq, s.n_points))
long_ = "a" * 64
for tail in ("left", "right"):                           # alike in their first 60 characters
    k = s.add_segment(long_ + tail)
    s.write_mask(0, blob(50, 50), 1.0, 6.0, i=k)
s.export_mat(os.path.join(OUT, "long.mat"))
from scipy.io import loadmat  # noqa: E402
segs_ = loadmat(os.path.join(OUT, "long.mat"), simplify_cells=True)["segments"]
check(sorted(v["name"] for v in segs_.values()) == [long_ + "left", long_ + "right"],
      "MATLAB export: two long animal names alike in 60 characters stay two structs", list(segs_))
for nm in (long_ + "left", long_ + "right"):
    s.remove_segment(s.segment_index(nm))

from kinetrace.tracker import AnimalSpec, TrackingWorker  # noqa: E402
spec_ = AnimalSpec(prompts={0: [(5.0, 5.0, 1)]})
TrackingWorker("x.mp4", 0, np.zeros((1, 2), np.float32), [3], None, N, animal=spec_, head_pid=3,
               on_body_pids=[3], constrain_pids=[3])
check(spec_.head_pid is None and not spec_.on_body and not spec_.constrain,
      "a worker leaves the caller's AnimalSpec as it was (its own copy takes the head / on-body points)")

print("[1b] two cameras, by name")
a, b = TrackingSession("a.mp4", N, 30.0, W, H), TrackingSession("b.mp4", N, 30.0, W, H)
pr = Project([a, b])
ka = a.add_segment("bat")
a.segments[ka].hold = True
a.apply_skeleton({"name": "w", "landmarks": ["wrist", "elbow"], "bones": [["wrist", "elbow"]], "head": "wrist"}, ka)
pr.sync_landmarks(0)
kb = b.segment_index("bat")
check(kb is not None and b.segments[kb].hold and b.segments[kb].skeleton == a.segments[ka].skeleton
      and not b.segments[kb].prompts and b.pid_by_name("bat wrist") is not None
      and b.points[b.pid_by_name("bat wrist")].segment == "bat",
      "the other camera gets the animal (skeleton, hold, no clicks) and its points, by name")
b.add_point(0, 1, 1, name="bat elbow (2)")          # a name only the other camera has
moved = pr.move_landmarks(["bat wrist"], "")
check(moved == [("bat wrist", "wrist")] and a.pid_by_name("wrist") is not None and b.pid_by_name("wrist") is not None
      and a.segment_of(a.pid_by_name("wrist")) is None, "moved to Scene in both cameras, one name", moved)
check(pr.rename_segment("bat", "vampire") == "vampire" and a.pid_by_name("vampire elbow") is not None
      and b.pid_by_name("vampire elbow") is not None, "an animal renamed in both cameras, its points too")
a.segments[a.segment_index("vampire")].hold = False
pr.share_animal("vampire", 0)
check(not b.segments[b.segment_index("vampire")].hold, "the animal's settings shared by name")

print("[1c] the project file (format 3)")
f = os.path.join(OUT, "model.kinetrace")
a.segments[0].shown = False
projectfile.save(pr, f)
q, _st, meta = projectfile.read(f)
qa = q.sessions[0]
check(projectfile.FORMAT_VERSION == 3 and json.load(open(os.path.join(f, "kinetrace.json")))["format_version"] == 3,
      "format 3")
check(qa.segment_names() == ["vampire"] and qa.segments[0].skeleton == a.segments[0].skeleton
      and not qa.segments[0].hold and not qa.segments[0].shown
      and [(p.name, p.segment) for p in qa.points] == [(p.name, p.segment) for p in a.points],
      "the animal (skeleton, hold, shown) and each point's animal round-trip")
rows = list(csv.reader(open(os.path.join(f, "cameras", "cam1", "points.csv"), encoding="utf-8-sig")))
check("animal" in rows[0] and not os.path.exists(os.path.join(f, "cameras", "cam1", "skeleton.json")),
      "points.csv has the `animal` column; no camera-level skeleton.json", rows[0])


# ------------------------------------------------------------------ the window
def clip(name, n=N):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for f in range(n):
        im = np.full((H, W, 3), 50, np.uint8)
        cv2.circle(im, (80 + f, 90), 14, (230, 230, 230), -1)
        vw.write(im)
    vw.release()
    return path


VA, VB = clip("camA.mp4"), clip("camB.mp4")
win = MainWindow()
win.resize(1500, 950)
win.show()
win._preview_mask = lambda: None
win._open_video(VA)
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and not win._loading:
        break
lay = win.layers
s = win.session

# ------------------------------------------------------------------ [0] the zooms
print("[0] the two zooms (G152)")
tl = win.timeline
check(win.btn_tz_in.parent() is tl and win.btn_tz_in.geometry().right() < 96,
      "the timeline's zoom buttons are the timeline's own, in its corner left of the ruler")
span0 = tl._view[1] - tl._view[0]
m0 = win.canvas.transform().m11()
QTest.mouseClick(win.btn_tz_in, Qt.LeftButton)
pump()
check(tl._view[1] - tl._view[0] < span0 and abs(win.canvas.transform().m11() - m0) < 1e-9,
      "the timeline's zoom-in zooms the timeline, not the video", (span0, tl._view))
QTest.mouseClick(win.btn_tz_fit, Qt.LeftButton)
pump()
cv = win.canvas
check(cv._zoom_bar.isVisible() and cv._zoom_bar.parent() is cv,
      "the video has its own zoom buttons (on the view, not its scrolling viewport: G168)")
QTest.mouseClick(cv.btn_zoom_in, Qt.LeftButton)
pump()
check(cv.transform().m11() > m0 * 1.1 and tl._view[1] - tl._view[0] == span0,
      "the video's zoom-in zooms the video, not the timeline", (m0, cv.transform().m11()))
QTest.mouseClick(cv.btn_zoom_fit, Qt.LeftButton)
pump()
check(abs(cv.transform().m11() - m0) < 1e-6, "fit puts the whole picture back")
check(win.btn_add.text() == "Point", "the point tool is called Point", win.btn_add.text())


def canvas_click(x, y, mod=Qt.NoModifier, canvas=None):
    c = canvas or win.canvas
    QTest.mouseClick(c.viewport(), Qt.LeftButton, mod, c.mapFromScene(QPointF(float(x), float(y))))
    pump(0.1)


def row_click(item, mod=Qt.NoModifier, button=Qt.LeftButton):
    lay.scrollToItem(item)
    pump(0.05)
    QTest.mouseClick(lay.viewport(), button, mod, lay.visualItemRect(item).center())
    pump(0.1)


def kids(k):
    """The point ids under animal k's row (None = Scene), in tree order."""
    top = win._animal_item(k) if k is not None else lay.scene_item()
    return [top.child(i).data(0, Qt.UserRole)[1] for i in range(top.childCount())] if top is not None else []


# ------------------------------------------------------------------ [2] making points
print("[2] LAYERS by real clicks (G154, G155)")
QTest.keyClick(win, Qt.Key_N)
canvas_click(20, 20)
check(s.n_points == 1 and s.points[0].segment == "" and lay.point_item(0).parent() is None,
      "with no animal at all, a point is a plain row", [(p.name, p.segment) for p in s.points])
check(win.btn_clear_animal.isEnabled(), "LAYERS Delete is usable with points and no animal (it deletes points too)")
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump()
check(s.segment_names() == ["animal"] and win._selected_segments() == [0] and lay.scene_item() is not None
      and kids(None) == [0], "+ Animal: a row, selected; the point that was there is in Scene")
QTest.keyClick(win, Qt.Key_N)
canvas_click(100, 100)
p1 = s.n_points - 1
check(s.points[p1].name == "animal P2" and kids(0) == [p1] and lay.point_item(p1).text(0) == "P2",
      "N + click with the animal selected: the point is its (named 'animal P2', shown as 'P2')",
      (s.points[p1].name, kids(0)))
QTest.keyClick(win, Qt.Key_N)
canvas_click(120, 110)
p2 = s.n_points - 1
check(s.points[p2].segment == "animal", "the next click, with its point selected: the same animal")
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump()
check(win._selected_segments() == [1] and win._selected_pids() == [] and win._s_target() == 1,
      "+ Animal with another animal's point selected: the new animal ALONE is selected, so S outlines it",
      (win._selected_segments(), win._selected_pids(), win._s_target()))
for fr in range(N):
    s.write_mask(fr, blob(250, 180), 1.0, 6.0, i=1)          # animal 2's silhouette
win._refresh_point_list()
row_click(lay.point_item(0))                                   # the Scene point
win.toast._notices.clear()
QTest.keyClick(win, Qt.Key_N)
canvas_click(250, 180)
p3 = s.n_points - 1
notes = " ".join(n[0] for n in win.toast._notices)
check(s.points[p3].segment == "animal 2" and "animal 2" in notes,
      "a click on another animal's silhouette with a Scene point selected: that animal, and a notice says so",
      (s.points[p3].segment, notes[:80]))
lay.select_only()
win._on_point_selection_changed()
win.toast._notices.clear()
QTest.keyClick(win, Qt.Key_N)
canvas_click(30, 200)
p4 = s.n_points - 1
check(s.points[p4].segment == "" and not win.toast._notices, "nothing selected, nothing under the click: Scene, quietly")
row_click(win._animal_item(0))
QTest.mouseClick(win.btn_new_point, Qt.LeftButton)
pump()
p5 = s.n_points - 1
check(s.points[p5].segment == "animal" and not s.tracked[:, p5].any(), "+ Point with the animal selected: its, no data")

# ------------------------------------------------------------------ [3] drag and drop
print("[3] dragging points (G154)")
check(win._add_view(VB), "a second camera")
win._set_active_view(0)
pump(0.3)
sB = win.project.sessions[1]
s = win.session
row_click(lay.point_item(p4))                                   # the Scene point "P5"
name4 = s.points[p4].name


def drop_on(item):
    lay.scrollToItem(item)
    pump(0.05)
    md = lay.mimeData([lay.point_item(q) for q in lay.selected_pids()])
    pos = QPointF(lay.visualItemRect(item).center())
    ev = QDropEvent(pos, Qt.MoveAction, md, Qt.LeftButton, Qt.NoModifier)
    lay.dropEvent(ev)
    pump(0.1)


drop_on(win._animal_item(1))
newname = s.points[p4].name
check(newname == f"animal 2 {name4}" and s.points[p4].segment == "animal 2" and sB.pid_by_name(newname) is not None
      and sB.points[sB.pid_by_name(newname)].segment == "animal 2" and p4 in kids(1),
      "dropped on animal 2: its, renamed, in both cameras", newname)
QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
pump()
check(s.points[p4].name == name4 and s.points[p4].segment == "" and sB.pid_by_name(name4) is not None,
      "one Ctrl+Z puts it back in both cameras")
row_click(lay.point_item(p1))
drop_on(lay.point_item(p3))                                      # onto a point of animal 2 = into animal 2
check(s.points[p1].segment == "animal 2" and s.points[p1].name == "animal 2 P2", "dropped on a point: into its animal")
drop_on(lay.scene_item())
check(s.points[p1].segment == "" and s.points[p1].name == "P2", "dropped on Scene: out of the animal")
d = s.add_landmark("animal tip", "silhouette", "tip")
s.points[d].segment = "animal"
win._refresh_point_list()
row_click(lay.point_item(d))
win.toast._notices.clear()
drop_on(lay.scene_item())
check(s.points[d].segment == "animal" and "derived" in " ".join(n[0] for n in win.toast._notices),
      "a landmark derived from the silhouette is not dropped into Scene, and says why")

# ------------------------------------------------------------------ [4] Track = the selection
print("[4] Track = the selection (G154, G156, G160)")
s.segments[1].add_click(0, 250, 180)
row_click(lay.point_item(p3))                                    # a point of animal 2, alone
win._update_track_button()
check(win._run_scope()[0] == {p3} and not win._run_segments({p3}) and "silhouette" not in win.btn_track.text(),
      "a point alone runs alone (its animal does not hold its points: opt-in)", win.btn_track.text())
s.segments[1].hold = True
win._update_track_button()
check(win._run_segments({p3}) == [1] and "silhouette" in win.btn_track.text(),
      "its animal holds its points: its silhouette rides along", win.btn_track.text())
s.points[p3].free = True
check(not win._run_segments({p3}), "a point marked 'may leave its silhouette': alone again")
s.points[p3].free = False
s.segments[1].hold = False
row_click(win._animal_item(1))
scope = win._run_scope()[0]
check(scope == set(s.points_of(1)) and win._run_segments(scope) == [1],
      "an animal row = its silhouette and all its points", (scope, s.points_of(1)))
row_click(lay.point_item(0))
row_click(win._animal_item(1), mod=Qt.ControlModifier)
check(win._selected_pids() == [0] and win._selected_segments() == [1], "Ctrl+click adds an animal to a point")
row_click(lay.point_item(0))
check(not win._run_segments({0}), "a Scene point never brings a silhouette")

# ------------------------------------------------------------------ [5] skeleton
print("[5] skeletons (G157)")
row_click(win._animal_item(0))
tmpl = next(t for t in skeletons.all_templates() if t["name"].startswith("Undulating"))
act = next(x for x in win.m_skeleton.actions() if x.text().startswith(f"Use template: {tmpl['name']}"))
act.trigger()
pump()
head0 = tmpl["head"]
check(s.pid_by_name(f"animal {head0}") is not None and s.segments[0].skeleton["head"] == head0
      and s.head_pid(0) == s.pid_by_name(f"animal {head0}"), "Skeleton menu on the selected animal: its points and head")
qa_, qb_ = s.pid_by_name("animal P3"), s.pid_by_name("animal P6")
lay.select_only([qa_, qb_])
win._on_point_selection_changed()
menu, acts = win._build_multi_menu([qa_, qb_])
check(acts["bone"].isEnabled() and acts["bone"].text().startswith("Connect"), "two points of one animal: Connect them")
nb = len(s.bones())
win._multi_menu_action(acts["bone"], acts, [qa_, qb_])
menu.deleteLater()
check(len(s.bones()) == nb + 1 and s.has_bone(qa_, qb_) and win.act_show_bones.isChecked(),
      "a bone drawn (and bones shown)")
QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
pump()
check(len(s.bones()) == nb, "Ctrl+Z takes the bone back")
menu, acts = win.canvas._build_context_menu(qb_)
win._point_menu_extra_action(acts["set_head"], acts, qb_)
menu.deleteLater()
check(s.head_pid(0) == qb_ and sB.segments[sB.segment_index("animal")].skeleton["head"] == "P6",
      "the point menu's 'Use it as ... head', in both cameras")
tmpdir = tempfile.mkdtemp(dir=OUT)
old_dir = skeletons.SKELETON_DIR
skeletons.SKELETON_DIR = Path(tmpdir)
try:
    ANSWER["text"] = "my animal"
    menu, acts = win._build_animal_menu(0)
    win._animal_menu_action(acts["save_tpl"], acts)
    menu.deleteLater()
    saved = json.load(open(os.path.join(tmpdir, "my animal.json"), encoding="utf-8"))
    check(head0 in saved["landmarks"] and saved["head"] == "P6" and not any(n.startswith("animal ") for n in saved["landmarks"]),
          "the animal saved as a template, in part names", saved.get("landmarks", [])[:4])
    row_click(win._animal_item(1))
    win._apply_skeleton_template(saved)
    check(s.pid_by_name(f"animal 2 {head0}") is not None and s.head_pid(1) == s.pid_by_name("animal 2 P6"),
          "the saved template on the other animal: its own points and head")
finally:
    skeletons.SKELETON_DIR = old_dir

# ------------------------------------------------------------------ [6] the animal's row
print("[6] the animal's row (G154)")
win._animal_item(0).setText(0, "squirrel")
pump()
check(s.segment_names()[0] == "squirrel" and s.pid_by_name(f"squirrel {head0}") is not None
      and sB.pid_by_name(f"squirrel {head0}") is not None and win._animal_item(0).text(0) == "squirrel",
      "a rename in the row renames the animal and its points in both cameras")
menu, acts = win._build_animal_menu(0)
acts["hold"].setChecked(True)
win._animal_menu_action(acts["hold"], acts)
menu.deleteLater()
check(s.segments[0].hold and sB.segments[sB.segment_index("squirrel")].hold, "Keep its points on its silhouette: both cameras")
n_sq = len(s.points_of(0))
ANSWER["q"] = QMessageBox.Yes                     # Yes = keep the points
ASKED.clear()
row_click(win._animal_item(0))
QTest.mouseClick(win.btn_clear_animal, Qt.LeftButton)
pump()
check(s.segment_names() == ["animal 2"] and len(s.points_of(None)) >= n_sq and s.pid_by_name(head0) is not None
      and ASKED and "Keep its" in ASKED[-1], "Delete on an animal (keep its points): they go to Scene by their part",
      (s.segment_names(), ASKED[-1:]))
n_all = s.n_points
n_a2 = len(s.points_of(0))
ANSWER["q"] = QMessageBox.No                      # No = delete them too
row_click(win._animal_item(0))
QTest.mouseClick(win.btn_clear_animal, Qt.LeftButton)
pump()
ANSWER["q"] = QMessageBox.Yes
check(s.n_segments == 0 and s.n_points == n_all - n_a2 and sB.n_points == s.n_points,
      "Delete on an animal (with its points): gone in both cameras", (s.n_points, n_all, n_a2))

# ------------------------------------------------------------------ [6b] undo of animal edits, both cameras
print("[6b] Ctrl+Z of an animal's bones / head / hold / name in EVERY camera (G153)")
ko = s.add_segment("owl")
qo = [s.add_point(0, 60.0 + 20 * j, 60.0) for j in range(2)]
s.move_points(qo, ko)
win._share_landmarks(undoable=False)
win._refresh_point_list()
pump()


def owl_b():
    return sB.segments[sB.segment_index("owl")]


def ctrl_z():
    QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
    pump()


lay.select_only(qo)
win._on_point_selection_changed()
menu, acts = win._build_multi_menu(qo)
win._multi_menu_action(acts["bone"], acts, qo)
menu.deleteLater()
check(s.has_bone(*qo) and (owl_b().skeleton or {}).get("bones"), "a bone, shared with the other camera")
ctrl_z()
check(not s.has_bone(*qo) and not (owl_b().skeleton or {}).get("bones"),
      "Ctrl+Z takes the bone back in the OTHER camera too", owl_b().skeleton)
menu, acts = win.canvas._build_context_menu(qo[1])
win._point_menu_extra_action(acts["set_head"], acts, qo[1])
menu.deleteLater()
check((owl_b().skeleton or {}).get("head") == s.part_name(qo[1]), "the head, shared with the other camera")
ctrl_z()
check(s.head_pid(ko) != qo[1] and not (owl_b().skeleton or {}).get("head"),
      "Ctrl+Z takes the head back in the OTHER camera too", owl_b().skeleton)
menu, acts = win._build_animal_menu(ko)
acts["hold"].setChecked(True)
win._animal_menu_action(acts["hold"], acts)
menu.deleteLater()
check(s.segments[ko].hold and owl_b().hold, "hold, shared with the other camera")
ctrl_z()
check(not s.segments[ko].hold and not owl_b().hold, "Ctrl+Z takes hold back in the OTHER camera too")
s.connect_bone(*qo)
win.project.share_animal("owl")
menu, acts = win._build_animal_menu(ko)
win._animal_menu_action(acts["forget_sk"], acts)
menu.deleteLater()
check(not s.segments[ko].skeleton and not owl_b().skeleton, "Forget its bones: both cameras")
ctrl_z()
check(s.has_bone(*qo) and (owl_b().skeleton or {}).get("bones"),
      "Ctrl+Z gives the bones back in the OTHER camera too", owl_b().skeleton)
kb_ = s.add_segment("bat")                                       # an animal with no points
win._share_landmarks(undoable=False)
win._refresh_point_list()
pump()
win._animal_item(kb_).setText(0, "fruit bat")
pump()
check(s.segment_names()[kb_] == "fruit bat" and sB.segment_index("fruit bat") is not None,
      "an animal with no points renamed in both cameras")
ctrl_z()
check(s.segment_names()[kb_] == "bat" and sB.segment_index("bat") is not None and sB.segment_index("fruit bat") is None,
      "Ctrl+Z gives its name back in the OTHER camera too (it has no points there)", sB.segment_names())
win.project.dirty = False
win._animal_item(ko).setCheckState(0, Qt.Unchecked)             # the animal's LAYERS checkbox
pump()
win._sync_ui_state()
check(not s.segments[ko].shown and not win.project.dirty and "owl" in s.ui_state.get("hidden_animals", []),
      "hiding an animal's silhouette is display state: not dirty, kept in the view (view.json / sidecar)",
      (s.segments[ko].shown, win.project.dirty, s.ui_state.get("hidden_animals")))
win._animal_item(ko).setCheckState(0, Qt.Checked)
pump()
for nm in ("owl", "bat"):                                        # leave the cameras as [7] expects them
    win.project.remove_segment(nm, keep_points=False)
win._undo_snap = None
win._refresh_point_list()
tl.set_session(s)
pump()
check(s.n_segments == 0 and sB.n_segments == 0 and sB.n_points == s.n_points, "the test animals removed again")

# ------------------------------------------------------------------ [7] timeline
print("[7] the timeline grouped by animal (G158)")
k = s.add_segment("lizard")
s.move_points([1, 2], k)
s.write_mask(3, blob(100, 100), 1.0, 6.0, i=k)
win._share_landmarks(undoable=False)
win._refresh_point_list()
tl.set_session(s)
pump(0.3)
rows = tl._rows()
check(rows[:3] == [("seg", 0), ("pt", 1), ("pt", 2)] and rows[3:] == [("pt", q) for q in s.points_of(None)],
      "the animal's silhouette lane, its points, then Scene", rows[:4])
y = EVENTS_H + 2 + LANE_H * (rows.index(("pt", 2)) - tl._scroll) + LANE_H // 2
lay.select_only()
QTest.mouseClick(tl, Qt.LeftButton, Qt.NoModifier, QPoint(20, int(y)))
pump()
check(win.selected == 2, "a click on a point's lane name selects it", win.selected)

# ------------------------------------------------------------------ [8] exports
print("[8] DeepLabCut multi-animal and SLEAP (G159)")
s.tracks[5, 1], s.tracked[5, 1], s.confidence[5, 1] = (10.0, 20.0), True, 0.9
s.tracks[5, 0], s.tracked[5, 0], s.confidence[5, 0] = (1.0, 2.0), True, 0.5
files, notes = win._export_one("dlc_ma", os.path.join(OUT, "ma.csv"))
rows = list(csv.reader(open(files[0], encoding="utf-8")))
check(rows[1][0] == "individuals" and "lizard" in rows[1] and "single" in rows[1]
      and rows[2][0] == "bodyparts" and rows[2][rows[1].index("lizard")] == s.part_name(1)
      and rows[3][1:4] == ["x", "y", "likelihood"] and rows[4 + 5][rows[1].index("lizard")] == "10.000",
      "DeepLabCut multi-animal: individuals / bodyparts / coords, the animal's part under its name",
      rows[1][:7])
files, notes = win._export_one("sleap", os.path.join(OUT, "sleap.csv"))
srows = list(csv.reader(open(files[0], encoding="utf-8")))
x_col = srows[0].index(f"{s.part_name(1)}.x")
lz = [r for r in srows[1:] if r[0] == "lizard" and r[1] == "5"]
check(srows[0][:3] == ["track", "frame_idx", "instance.score"] and lz and lz[0][x_col] == "10.000"
      and any(r[0] == "scene" for r in srows[1:]), "SLEAP analysis CSV: one track per animal, plus scene", srows[0][:6])

# ------------------------------------------------------------------ [9] save and open
print("[9] saved and opened again")
target = os.path.join(OUT, "layers.kinetrace")
k2 = s.add_segment("frog")
win._share_landmarks(undoable=False)
win._refresh_point_list()
row_click(win._animal_item(k2))                 # the second animal alone is selected (G161)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
win.project_path = None
win._save_project()
pump(0.5)
names_before = [(p.name, p.segment) for p in s.points]
win._open_project_from_path(target)
for _ in range(200):
    pump(0.05)
    if win.state == READY and win.session is not s and not win._loading:
        break
r = win.session
check([(p.name, p.segment) for p in r.points] == names_before and win._animal_item(0) is not None
      and win._animal_item(0).text(0) == "lizard" and [lay.kind_of(win._animal_item(0).child(i))[1]
                                                        for i in range(win._animal_item(0).childCount())] == [1, 2],
      "the project opens with its animals, their points under them, and Scene")
check(win._selected_segments() == [r.segment_index("frog")], "... and with the animal that was selected (G161)",
      win._selected_segments())

win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_layers: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
