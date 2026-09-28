"""Save, backup and unsaved-work recovery through the app (offscreen, no GPU).

The project file changes only on Save; the 30 s autosave writes a recovery
copy in the recovery folder, matched to its project by id. Checked here:
autosave never touches the project and writes off the GUI thread; the window
does not stall during a background save; edits made while a save runs stay
unsaved; Save keeps the previous save as .bak; a failed write leaves the file
alone; a moved / renamed project still finds its unsaved work; Save As gives
a new id; unsaved work from another copy opens as a separate copy; leftovers
are announced and can be recovered; Discard / Cancel on close; and the whole
working state (toggles, panels, frame, zoom, timeline zoom, active camera)
comes back exactly on reopen, including a view-only change made after the
last save.

Run: .venv\\Scripts\\python.exe tests\\verify_recovery.py
"""
import os
import shutil
import sys
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _clean import forget_recovery  # noqa: E402
from kinetrace import projectfile, recovery  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "recovery_suite")
if os.path.isdir(OUT):
    shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

ASK = {"answer": QMessageBox.Yes, "close": None, "asked": [], "warned": [], "critical": []}


def _question(*a, **k):
    title = a[1] if len(a) > 1 else ""
    ASK["asked"].append(title)
    if title == "Save changes?":
        return ASK["close"]
    return ASK["answer"]


QMessageBox.question = staticmethod(_question)
QMessageBox.warning = staticmethod(lambda *a, **k: (ASK["warned"].append(a[1] if len(a) > 1 else ""), QMessageBox.Ok)[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: (ASK["critical"].append(str(a[2]) if len(a) > 2 else ""),
                                                     QMessageBox.Ok)[1])
SAVE_TO = {"path": ""}
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (SAVE_TO["path"], ""))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
app = QApplication.instance() or QApplication([])
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402

fails = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        fails.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def clip(path, n=90, w=320, h=240, shift=0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.RandomState(3)
    base = rng.randint(0, 255, (h + 40, w + 600, 3)).astype(np.uint8)
    for f in range(n):
        x = 4 * (f + shift)
        vw.write(np.ascontiguousarray(base[20:20 + h, x:x + w]))
    vw.release()
    return path


def window(video=None, project=None):
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    if video:
        w._open_video(video)
    else:
        w._open_project_from_path(project)
    for _ in range(300):
        pump(0.05)
        if w.state == READY and w.project is not None:
            break
    pump(0.3)
    assert w.state == READY and w.project is not None
    return w


def close(w, answer=None):
    ASK["close"] = answer
    w.close()
    pump(0.3)


def data(path):
    with open(path, "rb") as fh:
        return fh.read()


VA = clip(os.path.join(OUT, "camA.mp4"))
VB = clip(os.path.join(OUT, "camB.mp4"), shift=7)
forget_recovery(VA, VB)
real_write = projectfile.write

# ------------------------------------------------------------------ 1
print("\n[1] autosave: recovery folder only, off the GUI thread")
w = window(VA)
pid0 = w._project_id
w.session.add_point(0, 50.0, 60.0)
w.project.dirty = True
threads = []


def spy_write(*a, **k):
    threads.append(threading.current_thread() is threading.main_thread())
    return real_write(*a, **k)


projectfile.write = spy_write
before = sorted(os.listdir(OUT))
w._autosave(wait=True)
projectfile.write = real_write
check(threads == [False], "the recovery copy is written on a worker thread", str(threads))
check(recovery.find(pid0) is not None and recovery.find(pid0).get("temporary") is True,
      "a never-saved session's work is in the recovery folder, marked temporary")
check(sorted(os.listdir(OUT)) == before, "nothing is written beside the video")
P1 = os.path.join(OUT, "p1.kinetrace")
SAVE_TO["path"] = P1
check(w._save_project_as(), "Save As writes the project")
check(recovery.find(pid0) is None, "saving retires the never-saved recovery")
pid1, b1 = w._project_id, data(P1)
check(pid1 != pid0, "Save As gives the project its own id")
w.session.add_point(0, 70.0, 80.0)
w._autosave(wait=True)
check(data(P1) == b1, "autosave leaves the saved project byte-identical")
check(recovery.find(pid1) is not None and recovery.find(pid1)["base_saved_at"] == w._saved_at,
      "the recovery names the save it started from")

# ------------------------------------------------------------------ 2
print("\n[2] Save: .bak, edits during a save stay unsaved, a failed write changes nothing")
check(w._save_project(), "Save")
b2 = data(P1)
check(os.path.exists(P1 + ".bak") and data(P1 + ".bak") == b1, "the previous save is kept as .bak")
check(recovery.find(pid1) is None and not w.project.dirty, "Save retires the recovery and clears dirty")


def slow_write(*a, **k):
    time.sleep(0.4)
    return real_write(*a, **k)


projectfile.write = slow_write
QTimer.singleShot(100, lambda: w.session.add_point(0, 90.0, 90.0))
ok = w._save_project()
projectfile.write = real_write
check(ok and w.project.dirty, "an edit made while the save runs keeps the project unsaved")
check(projectfile.load(P1).sessions[0].n_points == 2, "the save holds the state at the moment Save was pressed")
b3 = data(P1)


def bad_write(*a, **k):
    raise OSError("disk full")


projectfile.write = bad_write
ASK["critical"] = []
ok = w._save_project()
projectfile.write = real_write
check(not ok and ASK["critical"] and "disk full" in ASK["critical"][-1], "a failed write is said", str(ASK["critical"]))
check(data(P1) == b3 and w.project.dirty, "a failed write leaves the file and the unsaved state alone")
check(not [f for f in os.listdir(OUT) if f.endswith(".tmp")], "no temp file is left behind")

# ------------------------------------------------------------------ 3
print("\n[3] the window stays live during a background save")
big = TrackingSession(VA, 40000, 30.0, 3840, 2160)
for i in range(10):
    big.add_point(0, 10.0 * i, 5.0 * i)
rng = np.random.default_rng(0)
big.write_segment(0, (rng.random((40000, 10, 2)) * 3000).astype(np.float32),
                  np.ones((40000, 10), bool), list(range(10)), rng.random((40000, 10)).astype(np.float32))
t0 = time.perf_counter()
fz = projectfile.freeze(Project([big]), {}, projectfile.new_id(), target=os.path.join(OUT, "big.kinetrace"))
t_freeze = time.perf_counter() - t0
gaps, last = [], [time.perf_counter()]


def tick():
    now = time.perf_counter()
    gaps.append(now - last[0])
    last[0] = now


timer = QTimer()
timer.timeout.connect(tick)
timer.start(10)
last[0] = time.perf_counter()
t0 = time.perf_counter()
ok, err = w._start_writer(fz, os.path.join(OUT, "big.kinetrace"), wait=True)
t_write = time.perf_counter() - t0
timer.stop()
worst = max(gaps[1:] or [0.0]) * 1000
print(f"  40000 x 10 text save: freeze {t_freeze * 1000:.0f} ms (GUI thread), write {t_write:.2f} s "
      f"(worker), longest GUI gap {worst:.0f} ms over {len(gaps)} ticks")
check(ok, "the big save succeeds", err)
# KINETRACE_PERF_SCALE relaxes the budgets on slow shared machines (GitHub's macOS
# runner showed a 165 ms gap; the workflow sets 4); 1 on a workstation
SLOW = float(os.environ.get("KINETRACE_PERF_SCALE", "1"))
check(t_freeze < 0.08 * SLOW, "the GUI-thread copy stays under 80 ms", f"{t_freeze * 1000:.0f} ms")
check(worst < 100 * SLOW, "the GUI thread never stalls 100 ms during the save", f"{worst:.0f} ms")
os.remove(os.path.join(OUT, "big.kinetrace"))

# ------------------------------------------------------------------ 4
print("\n[4] a moved and renamed project still finds its unsaved work")
w.session.add_point(0, 100.0, 110.0)
n_unsaved = w.session.n_points
w._autosave(wait=True)
close(w, None)                                  # the close question not answered: kept
check(recovery.find(pid1) is not None, "closing without an answer keeps the unsaved work")
moved_dir = os.path.join(OUT, "moved")
os.makedirs(moved_dir, exist_ok=True)
P1m = os.path.join(moved_dir, "renamed.kinetrace")
shutil.move(P1, P1m)
ASK["answer"], ASK["asked"] = QMessageBox.Yes, []
w = window(project=P1m)
check("Unsaved changes found" in ASK["asked"], "reopening offers the unsaved changes", str(ASK["asked"]))
check(w.session.n_points == n_unsaved and w.project.dirty, "Yes restores them (still unsaved)",
      f"{w.session.n_points} points")
check(str(w.project_path) == P1m and w._project_id == pid1, "Save goes to the moved file, same project")
check(w._save_project() and recovery.find(pid1) is None, "saving there retires the recovery")
check(projectfile.load(P1m).sessions[0].n_points == n_unsaved, "the moved file now holds the work")

# ------------------------------------------------------------------ 5
print("\n[5] unsaved work from another copy of the project opens as a separate copy")
copy_path = os.path.join(OUT, "copy.kinetrace")
shutil.copy(P1m, copy_path)                     # same id, same save
copy_saved_at = projectfile.read(copy_path)[2].get("saved_at")
w.session.add_point(0, 120.0, 130.0)
check(w._save_project(), "the original is saved again (the copy is now older)")
w.session.add_point(0, 140.0, 150.0)
w._autosave(wait=True)
# what the decision below hinges on (printed for the cross-OS CI: the first run on
# GitHub's Windows machine took the older copy for the SAME save)
found_before = recovery.find(pid1) or {}
print(f"  copy saved_at {copy_saved_at} | original now {w._saved_at} | "
      f"recovery base_saved_at {found_before.get('base_saved_at')} written {found_before.get('written_at')}")
check(copy_saved_at != w._saved_at, "the second save carries a new stamp", str(w._saved_at))
close(w, None)
ASK["answer"], ASK["asked"] = QMessageBox.Yes, []
w = window(project=copy_path)
check("Unsaved changes found" in ASK["asked"], "the older copy is told about the other copy's unsaved work")
print(f"  asked: {ASK['asked']} | opened as id {w._project_id[:8]}… (original {pid1[:8]}…), "
      f"path {w.project_path}, dirty {w.project.dirty}")
check(w.project_path is None and w._project_id != pid1 and w.project.dirty,
      "Yes opens them as a separate, unsaved copy with its own id")
check(recovery.find(pid1) is None, "the original's recovery is kept aside, not offered again")
declined = recovery.folder()[0] / "declined"
check(declined.is_dir() and any(pid1 in f.name for f in declined.iterdir()), "... in the declined folder")
close(w, QMessageBox.Discard)
check(recovery.find(w._project_id) is None, "Discard on close drops the unsaved work")

# ------------------------------------------------------------------ 6
print("\n[6] leftovers are announced at start and can be recovered")
w = window(VB)
w.session.add_point(0, 30.0, 30.0)
w.session.add_point(0, 40.0, 40.0)
w.project.dirty = True
w._autosave(wait=True)
left_id = w._project_id
ASK["close"] = QMessageBox.Cancel
w.close()
pump(0.2)
check(w.isVisible(), "Cancel on close keeps the window open")
close(w, None)
w = MainWindow()
w.show()
w._announce_recovery()
pump(0.1)
said = w.toast.text()
check(w.toast.isVisible() and "Recover Unsaved Work" in said and "click here" in said,
      "start-up says unsaved work is waiting, and that a click restores it", said)
labels_seen = []


def pick(parent, title, label, items, *a, **k):
    labels_seen.extend(items)
    return items[0], True


QInputDialog.getItem = staticmethod(pick)
# G39: CLICKING the notice opens the list (it used to only close the notice), with a real click
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

w.toast.show_message("An unrelated notice arrives meanwhile", "info", 4000)   # the click still restores
check("click here" in w.toast.text() and w.toast.toolTip() == "click to open it", "the action survives", w.toast.text())
QTest.mouseClick(w.toast, Qt.LeftButton)
for _ in range(300):
    pump(0.05)
    if w.state == READY and w.project is not None:
        break
pump(0.3)
check(labels_seen and "never saved" in labels_seen[0], "the list names what it is", str(labels_seen))
check(w.project is not None and w.session.n_points == 2 and w.project.dirty and w._project_id == left_id,
      "File -> Recover Unsaved Work reopens it, still unsaved")
labels_seen.clear()
w.state = TRACKING                     # as during a run (the gate reads only the state)
w._apply_state()
w._recover_dialog()
check(not w.act_recover.isEnabled() and not labels_seen, "recovering is not offered in the middle of a run")
w.state = READY
w._apply_state()
check(w.act_recover.isEnabled() and w.act_import_calib.text().replace("&", "") == "Calibration…",
      "... and is back after it; File -> Import names its calibration entry plainly")
close(w, QMessageBox.Discard)
check(recovery.find(left_id) is None, "and Discard drops it")

# ------------------------------------------------------------------ 7
print("\n[7] the whole working state comes back exactly")
w = window(VA)
w.session.add_point(0, 60.0, 60.0)
assert w._add_view(VB)
pump(0.5)
w.session.add_point(0, 80.0, 70.0)
w.session.add_point(0, 90.0, 75.0)
w._on_select(1)
w.btn_follow.setChecked(True)
w.btn_autopause.setChecked(False)
w.act_onion.setChecked(True)
w.act_loupe.setChecked(True)
w.marker_spin.setValue(7)
w.step_spin.setValue(5)
w.dock.setVisible(False)
w.act_onboarding.setChecked(False)
w.act_solo.setChecked(True)
sizes = w._split.sizes()
w._split.setSizes([sizes[0] + 60, max(sizes[1] - 60, 40)] + sizes[2:])
pump(0.2)
split_saved = w._split.sizes()
w._goto(33)
w.timeline._set_view(10, 40)
w.canvas.zoom_step(1)
pump(0.3)
view_saved = w.canvas.view_state()
P7 = os.path.join(OUT, "state.kinetrace")
SAVE_TO["path"] = P7
check(w._save_project_as(), "saved with two cameras")
active_saved = w.project.active
close(w, None)
ASK["asked"] = []
w = window(project=P7)
pump(0.5)
check("Unsaved changes found" not in ASK["asked"], "a clean save opens without questions", str(ASK["asked"]))
check(w.project.n_views == 2 and w.project.active == active_saved, "both cameras, the same one active")
check(w.current == 33, "the frame", str(w.current))
check(w.selected == 1, "the selected point", str(w.selected))
check(w.btn_follow.isChecked() and not w.btn_autopause.isChecked(), "Follow on, auto-pause off")
check(w.act_onion.isChecked() and w.act_loupe.isChecked(), "onion skin and loupe")
check(w.marker_spin.value() == 7 and w.step_spin.value() == 5, "marker size and step")
check(not w.dock.isVisible(), "the side panel stays hidden")
check(not w.act_onboarding.isChecked(), "the getting-started strip stays closed")
check(w.act_solo.isChecked(), "solo mode")
check(w._split.sizes() == split_saved, "the timeline height", f"{w._split.sizes()} vs {split_saved}")
check(tuple(w.timeline._view) == (10, 50), "the timeline zoom", str(w.timeline._view))
vs = w.canvas.view_state()
check(abs(vs["zoom"] - view_saved["zoom"]) < 1e-6 and abs(vs["center_x"] - view_saved["center_x"]) < 0.5,
      "the video zoom and position", f"{vs} vs {view_saved}")
check(not w.project.dirty, "reopening is not an unsaved change")
# a view-only change after the save: no question on close, back next time, file untouched
b7 = data(P7)
w._goto(47)
w.act_onion.setChecked(False)
w._set_display_filter("contrast")
pump(0.2)
check(not w.project.dirty, "the playhead, a toggle and a display filter are not unsaved data")
ASK["asked"] = []
close(w, None)
check("Save changes?" not in ASK["asked"], "so closing asks nothing", str(ASK["asked"]))
w = window(project=P7)
check(w.current == 47, "where the user left off comes back", str(w.current))
check(not w.act_onion.isChecked() and w._display_filter_key() == "contrast", "with the toggles as left")
check(data(P7) == b7, "without rewriting the project file")
close(w, None)

forget_recovery(VA, VB)
print("\n" + "=" * 62)
if fails:
    print(f"verify_recovery FAILED: {len(fails)}")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("verify_recovery PASSED")
