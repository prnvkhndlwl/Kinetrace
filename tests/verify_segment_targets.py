"""(G151) Which segment an action works on, with several segments (owner 2026-10-03, after G150: "audit all
72 and fix the wrong ones"). Every check fails on the code before G151, where these actions took the
ACTIVE segment (the row clicked last, invisible) or the FIRST one.

Two cameras, three segments made with S + real canvas clicks and + New segment; rows (un)highlighted by real
clicks on the SEGMENT list.
  a  (G154) S outlines the animal selected in LAYERS; two selected = nowhere, said
  b  a row's right-click menu acts on THAT row
  c  Track's two passes put the pass holding ANY segment's head first
  d  the mask preview after an S click is anchored at THAT segment's head
  e  Data source -> silhouette: the highlighted segment, else asked
  f  Skeleton ▾ -> template: on the highlighted segment (prefixed, its own head), in both cameras
  g  Shift+> with no point selected: the highlighted segments' silhouettes
  h  a timeline silhouette clear that named no lane: the highlighted rows, else asked for every segment
  i  the stop notice names the segment that ended
  j  Import -> Silhouettes: into the highlighted segment
  k  Carve Volume: the highlighted segment's silhouettes
  l  Body: a silhouette of another segment than the active one drives the run
  m  silhouette exports: every segment, one file / folder each; Export enabled by any segment
  n  a re-track's segment = its point's segment
  o  convert info lists every segment; convert masks needs --animal when there are several

.venv\\Scripts\\python.exe tests\\verify_segment_targets.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "segment_targets")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMenu, QMessageBox  # noqa: E402

ASKED = []      # every question asked: (title, text)
PICKS = []      # every "Which segment?" asked: the label
PICK_ANSWER = [None]
QMessageBox.question = staticmethod(lambda *a, **k: (ASKED.append((a[1], a[2])), QMessageBox.Yes)[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)


def _get_item(parent, title, label, items, current=0, editable=False, *a, **k):
    if title in ("Which segment?", "Which animal?"):
        PICKS.append(label)
        return (PICK_ANSWER[0] or items[current]), True
    return items[current], True


QInputDialog.getItem = staticmethod(_get_item)
app = QApplication.instance() or QApplication([])

from kinetrace import app as appmod  # noqa: E402
from kinetrace import timeline as tlmod  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.skeletons import all_templates  # noqa: E402

FAILS = []
W, H, N = 320, 240, 40


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.15):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def blob(cx, cy, r=10):
    m = np.zeros((H, W), bool)
    cv2.circle(m.view(np.uint8), (int(round(cx)), int(round(cy))), r, 1, -1)
    return m


def clip(name):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for f in range(N):
        im = np.full((H, W, 3), 50, np.uint8)
        cv2.circle(im, (60 + f, 80), 12, (230, 230, 230), -1)
        vw.write(im)
    vw.release()
    return path


class _Sig:
    def connect(self, *a):
        pass


class FakePreview:
    """The mask preview's worker, without SAM: records what it was given."""
    last = None

    def __init__(self, path, frame, rgb, size, clicks, box, backend, head):
        FakePreview.last = dict(frame=frame, clicks=clicks, head=None if head is None else tuple(map(float, head)))
        self.loading = self.progress = self.done = self.error = _Sig()
        self.plain = self.cancelled = False

    def isRunning(self):
        return False

    def start(self):
        pass

    def wait(self, ms=0):
        return True


appmod._MaskPreviewWorker = FakePreview


class AutoMenu(QMenu):
    """A menu that 'clicks' the entry whose text starts with `pick` instead of waiting."""
    pick = ""
    shown = 0

    def exec(self, *a, **k):
        AutoMenu.shown += 1
        return next((x for x in self.actions() if x.text().startswith(AutoMenu.pick)), None)


VA, VB = clip("camA.mp4"), clip("camB.mp4")
win = MainWindow()
win.resize(1400, 900)
win.show()
win._open_video(VA)
for _ in range(300):
    pump(0.05)
    if win.state == READY and win.project is not None and not win._loading:
        break
check(win._add_view(VB), "two cameras")
win._set_active_view(0)
pump(0.3)
s = win.session
lay = win.layers


def canvas_click(x, y, mod=Qt.NoModifier):
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, mod, vp)
    pump(0.1)


def row_rect(k):
    lay.scrollToItem(win._animal_item(k))        # the user scrolls to a row before clicking it
    pump(0.05)
    return lay.visualItemRect(win._animal_item(k))


def row_click(k, button=Qt.LeftButton, mod=Qt.NoModifier):
    QTest.mouseClick(lay.viewport(), button, mod, row_rect(k).center())
    pump(0.1)


def highlight_only(ks):
    """Real clicks on the LAYERS rows, as in a file list (G154): the first animal plainly (it alone is
    selected), the others with Ctrl. None: the selection is emptied."""
    if not ks:
        lay.select_only()
        win._on_point_selection_changed()
        pump(0.1)
        return
    row_click(ks[0])
    for k in ks[1:]:
        row_click(k, mod=Qt.ControlModifier)


# ---------------------------------------------------------------- a: S's target
print("[a] the S tool's animal = the selected one")
QTest.keyClick(win, Qt.Key_S)
pump(0.1)
canvas_click(60, 80)                            # no animal yet: the click makes one
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump(0.2)
canvas_click(240, 60)                           # + Animal selected it: S outlines it
QTest.mouseClick(win.btn_new_segment, Qt.LeftButton)
pump(0.2)
canvas_click(160, 200)
check(s.segment_names() == ["animal", "animal 2", "animal 3"] and win._selected_segments() == [2]
      and s.active_seg == 2 and [len(a.prompts.get(0, [])) for a in s.segments] == [1, 1, 1],
      "three animals by S + clicks and + Animal, each click to the animal just made and selected",
      (s.segment_names(), win._selected_segments(), s.active_seg))
row_click(0)
check(win._selected_segments() == [0] and win._s_target() == 0, "a plain click selects only that row",
      win._selected_segments())
canvas_click(70, 90)
check([len(a.prompts.get(0, [])) for a in s.segments] == [2, 1, 1], "an S click goes to the selected animal",
      [len(a.prompts.get(0, [])) for a in s.segments])
row_click(1, mod=Qt.ControlModifier)
canvas_click(75, 95)
check(win._selected_segments() == [0, 1] and [len(a.prompts.get(0, [])) for a in s.segments] == [2, 1, 1]
      and "Select the animal to outline" in " ".join(n[0] for n in win.toast._notices),
      "two animals selected: the S click goes nowhere and says to select one",
      [len(a.prompts.get(0, [])) for a in s.segments])
QTest.keyClick(win, Qt.Key_S)                   # put S down
pump(0.1)
highlight_only([0])
for f in range(N):
    s.write_mask(f, blob(60 + f, 80, 12), 1.0, 6.0, i=0)
for f in range(5, 31):
    s.write_mask(f, blob(240, 60 + f // 2, 12), 1.0, 6.0, i=1)
for f in range(10, 21):
    s.write_mask(f, blob(160, 200, 12), 1.0, 6.0, i=2)
win._refresh_animal_panel()
win._refresh_overlay()
pump(0.1)

# ---------------------------------------------------------------- b: the row menu
print("[b] the row's right-click menu")
appmod.QMenu = AutoMenu
AutoMenu.pick, AutoMenu.shown = "Clear ALL", 0
row_click(1, Qt.RightButton)
if not AutoMenu.shown:
    # offscreen, a right click makes no context-menu event: send the one the platform would
    pos = row_rect(1).center()
    QApplication.sendEvent(lay.viewport(), QContextMenuEvent(QContextMenuEvent.Mouse, pos, lay.viewport().mapToGlobal(pos)))
    pump(0.1)
appmod.QMenu = QMenu
check(AutoMenu.shown == 1, "the row's context-menu event opened the segment menu once", AutoMenu.shown)
check(s.seg_masks[1].n_masked() == 0 and s.seg_masks[0].n_masked() == N and s.seg_masks[2].n_masked() == 11,
      "right-click on row 2 -> Clear ALL: that row's silhouettes only",
      ([m.n_masked() for m in s.seg_masks], s.active_seg))
QTest.keyClick(win, Qt.Key_Z, Qt.ControlModifier)
pump(0.1)
check(s.seg_masks[1].n_masked() == 26, "Ctrl+Z brings them back")

# ---------------------------------------------------------------- f: skeleton on a segment
print("[f] Skeleton > template")
t = all_templates()[0]
head = t["head"]


def use_template():
    act = next(a for a in win.m_skeleton.actions() if a.text().startswith(f"Use template: {t['name']}"))
    act.trigger()
    pump(0.1)


highlight_only([0])
use_template()
highlight_only([1])
use_template()
sB = win.project.sessions[1]
pre = f"animal 2 {head}"
hp1 = s.pid_by_name(pre)
check(s.pid_by_name(f"animal {head}") is not None and hp1 is not None and s.points[hp1].segment == "animal 2"
      and (s.segments[1].skeleton or {}).get("head") == head and s.head_pid(1) == hp1,
      "the second template went onto the selected animal 2 (its names, its own head)", (s.segments[1].skeleton, hp1))
kB = sB.segment_index("animal 2")
check(sB.pid_by_name(pre) is not None and kB is not None and (sB.segments[kB].skeleton or {}).get("head") == head,
      "... and in the other camera, by the animal's name", sB.segments[kB].skeleton if kB is not None else None)
if hp1 is None:          # [f] failed: build its scene so the checks below stay independent of it
    s.apply_skeleton(t, 1)
    hp1 = s.pid_by_name(pre)
    win._refresh_point_list()

# ---------------------------------------------------------------- c: passes
print("[c] two passes, the head's first")
hp0 = s.head_pid(0)
dot = s.add_landmark("dot", "track", "")
win._refresh_point_list()
s.points[dot].tracker = "alltracker"
s.points[hp1].tracker = "cotracker3"
passes = win._tracker_passes([dot, hp1])
check(hp0 is not None and hp0 != hp1 and len(passes) == 2 and hp1 in passes[0],
      "segment 2's head (CoTracker3) runs in the FIRST pass, though segment 1's head is the session's head",
      (passes, hp0, hp1))
s.points[hp1].tracker = ""

# ---------------------------------------------------------------- d: preview head
print("[d] the mask preview's head anchor")
highlight_only([1])
s.tracks[0, hp1] = (250.0, 70.0)
s.tracked[0, hp1] = True
s.visibility[0, hp1] = True
QTest.keyClick(win, Qt.Key_S)
pump(0.1)
FakePreview.last = None
canvas_click(242, 62)
QTest.keyClick(win, Qt.Key_S)
pump(0.1)
check(FakePreview.last is not None and FakePreview.last["head"] == (250.0, 70.0),
      "a click for segment 2 previews with segment 2's head as the midline anchor", FakePreview.last)

# ---------------------------------------------------------------- e: data source
print("[e] Data source -> silhouette")
mid = s.add_landmark("mid", "track", "")
win._refresh_point_list()
highlight_only([2])
PICKS.clear()
win._on_source_change(mid, "midline:0.5")
check(s.points[mid].segment == "animal 3" and s.points[mid].name == "animal 3 mid" and not PICKS,
      "derived from the one selected animal, and it becomes that animal's (no question)", (s.points[mid].name, PICKS))
mid2 = s.add_landmark("mid2", "track", "")
win._refresh_point_list()
highlight_only([])
PICK_ANSWER[0] = "animal 2"
win._on_source_change(mid2, "tip")
PICK_ANSWER[0] = None
check(PICKS and s.points[mid2].segment == "animal 2", "none selected: the user is asked which", PICKS)

# ---------------------------------------------------------------- g: Shift+>
print("[g] Shift+> with no point selected")
win.selected = None
highlight_only([1, 2])                           # (a plain click drops the point selection too)
win._goto(0)
pump(0.1)
QTest.keyClick(win, Qt.Key_Greater, Qt.ShiftModifier)
pump(0.2)
check(win.current == 30, "the last silhouette of the HIGHLIGHTED segments (segment 2's frame 30)",
      (win.current, s.active_seg))

# ---------------------------------------------------------------- h: timeline clear without a lane
print("[h] a timeline silhouette clear that named no segment lane")
tl = win.timeline
highlight_only([1, 2])
tl.sel_range, tl.sel_rows, tl.sel_seg, tl.sel_segs = (12, 14), None, False, []   # a drag along the ruler
tlmod_pop = tlmod._pop
tlmod._pop = lambda menu, pos: next((x for x in menu.actions() if x.text().startswith("Clear the selected animals'")),
                                    None)
x = int(tl._x_of(13))
pos = QPoint(x, tl.height() // 2)
tl.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse, pos, tl.mapToGlobal(pos)))
pump(0.1)
check(not any(s.seg_masks[k].has(f) for k in (1, 2) for f in (12, 13, 14)) and s.seg_masks[0].has(13),
      "-> the highlighted segments' silhouettes only", [s.seg_masks[k].has(13) for k in range(3)])
highlight_only([])
ASKED.clear()
tl.sel_range, tl.sel_rows, tl.sel_seg, tl.sel_segs = (16, 16), None, False, []
pos = QPoint(int(tl._x_of(16)), tl.height() // 2)
tl.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse, pos, tl.mapToGlobal(pos)))
pump(0.1)
tlmod._pop = tlmod_pop
check(any("ALL 3 animals" in q[1] for q in ASKED) and not any(s.seg_masks[k].has(16) for k in range(3)),
      "none highlighted: asked, then every segment's", ASKED[-1:])

# ---------------------------------------------------------------- i: stop notice
print("[i] the stop notice")
said = []
win._say_stop = lambda kind, name, frame, why="missing": said.append((kind, name, frame))
s.active_seg = 0
win.worker = SimpleNamespace(wait=lambda ms=0: True, isRunning=lambda: False, _autopause_reason="",
                             _ball_ended={}, _spot_ended={}, _seg_ended={2: (7, "lost")},
                             animals=[SimpleNamespace(index=2, constrain=set(), stored=None)])
win._autopause_info = (7, -1)
win._on_track_finished_impl(9, True)
pump(0.1)
check(said and said[0][1] == "animal 3", "names the animal that ended (animal 3), not the active one", said)

# ---------------------------------------------------------------- j: import silhouettes
print("[j] Import -> Silhouettes")
mdir = os.path.join(OUT, "masks")
os.makedirs(mdir, exist_ok=True)
for f in (35, 36, 37):
    cv2.imwrite(os.path.join(mdir, f"m_{f:06d}.png"), blob(100, 150, 15).astype(np.uint8) * 255)
highlight_only([1, 2])
highlight_only([2])                             # the last click UN-highlights segment 2's row
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: mdir)
before0 = s.seg_masks[0].n_masked()
act = next(a for a in win.findChildren(appmod.QAction) if a.text().replace("&", "").startswith("Silhouettes"))
act.trigger()
pump(0.3)
check(all(s.seg_masks[2].has(f) for f in (35, 36, 37)) and s.seg_masks[0].n_masked() == before0,
      "into the highlighted segment 3, segment 1 untouched")

# ---------------------------------------------------------------- k: carve
print("[k] Carve Volume")
names_seen = []
orig_masks = win._masks_at_instant
win._need_calibration = lambda what: True
win._reference_instant = lambda: 0
win._masks_at_instant = lambda t, name=None: (names_seen.append(name), [])[1]
highlight_only([1])
win._carve_hull_here(quiet=True)
check(names_seen == ["animal 2"], "the selected animal's silhouettes, by name", names_seen)
win._masks_at_instant = orig_masks

# ---------------------------------------------------------------- l: body
print("[l] Body -> Find People")
import kinetrace.bodyview as bv  # noqa: E402
got = []


class _BodyDlg:
    def __init__(self, parent, n, cur, sel, has_masks, *a, **k):
        got.append(has_masks)
        self.result_options = None

    def exec(self):
        return QDialog.Rejected


real_dlg = bv.BodyRunDialog
bv.BodyRunDialog = _BodyDlg
snap0 = s.seg_masks[0].copy()
masks2 = s.seg_masks[2].copy()
s.seg_masks[0].clear(0, N - 1)
s.seg_masks[2].clear(0, N - 1)
s.active_seg = 0
win._body_run()
bv.BodyRunDialog = real_dlg
check(got == [True], "segment 2 alone has a silhouette: the run is offered it (the active segment has none)", got)
s.seg_masks[0], s.seg_masks[2] = snap0, masks2

# ---------------------------------------------------------------- m: exports
print("[m] silhouette exports")
files, _notes = win._export_one("sil_json", os.path.join(OUT, "sil.json"))
check(sorted(os.path.basename(f) for f in files) == ["sil_animal.json", "sil_animal_2.json", "sil_animal_3.json"],
      "one JSON per segment", files)
files, _notes = win._export_one("sil_png", os.path.join(OUT, "png.json"))
n_png = sum(len(os.listdir(f)) for f in files)
check(len(files) == 3 and n_png == sum(m.n_masked() for m in s.seg_masks), "one PNG folder per segment",
      (files, n_png))
win._set_active_view(1)
pump(0.3)
sB = win.session
for q in range(sB.n_points - 1, -1, -1):
    sB.remove_point(q)
sB.seg_masks[1].clear(0, N - 1)
sB.write_mask(3, blob(100, 100, 10), 1.0, 6.0, i=1)
sB.active_seg = 0
win._apply_state()
check(win.act_export.isEnabled(), "Export is offered when only a segment other than the active one has a silhouette")
win._set_active_view(0)
pump(0.3)

# ---------------------------------------------------------------- n: re-track segment
print("[n] a re-track's segment")
highlight_only([])
s.active_seg = 2
check(win._seg_list([hp1], True) == [1], "the point's own animal (animal 2), not the active one",
      win._seg_list([hp1], True))

# ---------------------------------------------------------------- o: convert info
print("[o] convert info")
target = os.path.join(OUT, "proj.kinetrace")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (target, ""))
win.project_path = None
win._save_project()
pump(0.5)
r = subprocess.run([sys.executable, "-m", "kinetrace.convert", "info", target], cwd=ROOT,
                   capture_output=True, text=True)
out = r.stdout
check("segments: animal (silhouette on" in out and "animal 2 (silhouette on" in out and "animal 3" in out,
      "every segment by name", out[-400:])
print("[o2] convert masks names the animal (no hidden first one)")
mout = os.path.join(OUT, "masks_cli.json")
r = subprocess.run([sys.executable, "-m", "kinetrace.convert", "masks", target, mout], cwd=ROOT,
                   capture_output=True, text=True)
check(r.returncode != 0 and "--animal" in (r.stdout + r.stderr) and not os.path.exists(mout),
      "several animals and no --animal: refused, saying how to choose", (r.returncode, (r.stdout + r.stderr)[-200:]))
r = subprocess.run([sys.executable, "-m", "kinetrace.convert", "masks", target, mout, "--animal", "animal 2"],
                   cwd=ROOT, capture_output=True, text=True)
got = json.load(open(mout, encoding="utf-8")) if os.path.exists(mout) else {}
sv = win.session
check(r.returncode == 0 and got.get("name") == "animal 2"
      and len(got.get("frames", {})) == sv.seg_masks[sv.segment_index("animal 2")].n_masked(),
      "--animal 'animal 2': that animal's silhouettes", (r.returncode, got.get("name"), r.stderr[-200:]))

print("[p] a silhouette preview lands on the animal it was clicked for, renamed or not")
from kinetrace.segmenter import summarize_mask  # noqa: E402
sv = win.session
ka, kb = 0, 1
pf = next(f for f in range(sv.n_frames) if not sv.seg_masks[ka].has(f) and not sv.seg_masks[kb].has(f))
pm = np.zeros((sv.height, sv.width), bool)
pm[20:50, 30:70] = True
ps = summarize_mask(pm, 1.0, 9.0)
ps["frame"] = pf
win._preview = SimpleNamespace(target_session=sv, target_animal=sv.segments[ka], wait=lambda ms: None,
                               isRunning=lambda: False)
win._rename_segment(ka, "renamed meanwhile")          # renamed while SAM worked ...
sv.active_seg = kb                                    # ... and another animal is the active one now
win._on_preview_done(dict(ps))
check(sv.seg_masks[ka].has(pf) and not sv.seg_masks[kb].has(pf),
      "the preview is stored on the animal clicked for, under its new name (not the active one)")
gone = sv.segments[kb]
sv.remove_segment(kb)
win._preview = SimpleNamespace(target_session=sv, target_animal=gone, wait=lambda ms: None, isRunning=lambda: False)
before = [m.n_masked() for m in sv.seg_masks]
ps["frame"] = pf + 1 if pf + 1 < sv.n_frames else pf - 1
win._on_preview_done(dict(ps))
check([m.n_masked() for m in sv.seg_masks] == before, "the animal was removed meanwhile: the preview is dropped")

win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_segment_targets: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
