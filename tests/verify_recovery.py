"""Save, backup and unsaved-work recovery through the app (offscreen, no GPU).

The project file changes only on Save; the 30 s autosave writes a recovery
copy in the recovery folder, matched to its project by id. Checked here:
autosave never touches the project and writes off the GUI thread; the window
does not stall during a background save; edits made while a save runs stay
unsaved; Save keeps the previous save in the project folder's .history (I145);
a failed write leaves the folder alone; a moved / renamed project folder still
finds its unsaved work; Save As gives
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
    ASK.setdefault("texts", []).append(str(a[2]) if len(a) > 2 else "")
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
    """A project folder's files (not .cache / .history), or one file's bytes."""
    if os.path.isdir(path):
        out = {}
        for d, dirs, fs in os.walk(path):
            dirs[:] = [x for x in dirs if x not in (".cache", ".history", "exports")]
            for f in fs:
                out[os.path.relpath(os.path.join(d, f), path)] = open(os.path.join(d, f), "rb").read()
        return out
    with open(path, "rb") as fh:
        return fh.read()


VA = clip(os.path.join(OUT, "camA.mp4"))
VB = clip(os.path.join(OUT, "camB.mp4"), shift=7)
forget_recovery(VA, VB)
real_write = projectfile.write
real_folder = projectfile.write_folder_over

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
check(w._save_project_as() and os.path.isfile(os.path.join(P1, "kinetrace.json")),
      "Save As writes the project, as a folder (I145)")
check(recovery.find(pid0) is None, "saving retires the never-saved recovery")
pid1, b1 = w._project_id, data(P1)
check(pid1 != pid0, "Save As gives the project its own id")
w.session.add_point(0, 70.0, 80.0)
w._autosave(wait=True)
check(data(P1) == b1, "autosave leaves the saved project byte-identical")
check(recovery.find(pid1) is not None and recovery.find(pid1)["base_saved_at"] == w._saved_at,
      "the recovery names the save it started from")

# ------------------------------------------------------------------ 2
print("\n[2] Save: .history, edits during a save stay unsaved, a failed write changes nothing")
w.session.add_point(0, 75.0, 85.0)
check(w._save_project(), "Save")
b2 = data(P1)
hist = data(os.path.join(P1, ".history"))
kept = {k: v for k, v in hist.items() if k not in ("previous.json", "pending.json")}
check(kept.get("kinetrace.json") == b1["kinetrace.json"] and len(kept) >= 2
      and all(b1.get(k) == v for k, v in kept.items()),
      "the files the save replaced are kept as they were, in .history", str(sorted(kept)))
check(recovery.find(pid1) is None and not w.project.dirty, "Save retires the recovery and clears dirty")


def slow_write(*a, **k):
    time.sleep(0.4)
    return real_folder(*a, **k)


projectfile.write_folder_over = slow_write
QTimer.singleShot(100, lambda: w.session.add_point(0, 90.0, 90.0))
ok = w._save_project()
projectfile.write_folder_over = real_folder
check(ok and w.project.dirty, "an edit made while the save runs keeps the project unsaved")
check(projectfile.load(P1).sessions[0].n_points == 3, "the save holds the state at the moment Save was pressed")
b3 = data(P1)


def bad_write(*a, **k):
    raise OSError("disk full")


projectfile.write_folder_over = bad_write
ASK["critical"] = []
ok = w._save_project()
projectfile.write_folder_over = real_folder
check(not ok and ASK["critical"] and "disk full" in ASK["critical"][-1], "a failed write is said", str(ASK["critical"]))
check(data(P1) == b3 and w.project.dirty, "a failed write leaves the folder and the unsaved state alone")
ASK["critical"] = []
real_rep, n_rep = projectfile._replace, [0]


def locked(src, dst):                   # a CSV open in a spreadsheet: Windows refuses to replace it
    if ".cache" not in str(dst) and not str(dst).endswith("pending.json"):
        n_rep[0] += 1
        if n_rep[0] == 2:
            raise PermissionError(13, "The process cannot access the file", str(dst))
    return real_rep(src, dst)


projectfile._replace = locked
ok = w._save_project()
projectfile._replace = real_rep
check(not ok and ASK["critical"] and data(P1) == b3, "a file locked half-way through a save: the save is undone "
      "and said", str(ASK["critical"])[:120])
check(not [f for f in os.listdir(OUT) if f.endswith(".tmp")] and not os.path.exists(os.path.join(P1, ".saving")),
      "no temp file is left behind")

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
ok, err = w._start_writer(fz, os.path.join(OUT, "big.kinetrace"), wait=True, folder=True)
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
shutil.rmtree(os.path.join(OUT, "big.kinetrace"))

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
shutil.copytree(P1m, copy_path)                 # same id, same save
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

# ------------------------------------------------------------------ 8
print("\n[8] project folders: an older single file, Save As guards, one file, exports on save (I145, G42)")
import glob  # noqa: E402
import subprocess  # noqa: E402
import zipfile  # noqa: E402
from PySide6.QtCore import Qt as _Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QDialogButtonBox  # noqa: E402
w = window(VA)
w.session.add_point(0, 50.0, 60.0)
OLD = os.path.join(OUT, "older.kinetrace")
projectfile.save(w.project, OLD, w._ui_global_state(), w._project_id, single_file=True)   # the form before I145
one = data(OLD)
close(w, QMessageBox.Discard)
w = window(project=OLD)
w.session.add_point(0, 20.0, 30.0)
ASK["asked"], ASK["answer"] = [], QMessageBox.Yes
check(w._save_project() and "Save as a project folder?" in ASK["asked"] and os.path.isdir(OLD)
      and data(OLD + ".bak") == one, "Save offers to turn an older single file into a folder; the file is kept as .bak",
      str(ASK["asked"]))
check(projectfile.load(OLD).sessions[0].n_points == 2,
      "... which holds the work")
ASK["asked"] = []
w.session.add_point(0, 25.0, 35.0)
check(w._save_project() and "Save as a project folder?" not in ASK["asked"], "asked once: the next save just saves")
close(w, None)
OLD2 = os.path.join(OUT, "kept-single.kinetrace")
projectfile.save(projectfile.load(OLD), OLD2, single_file=True)
w = window(project=OLD2)
w.session.add_point(0, 15.0, 15.0)
ASK["answer"], ASK["asked"] = QMessageBox.No, []
check(w._save_project() and os.path.isfile(OLD2) and zipfile.is_zipfile(OLD2)
      and projectfile.load(OLD2).sessions[0].n_points == 4, "No keeps saving it as one file")
ASK["asked"] = []
w._save_project()
check("Save as a project folder?" not in ASK["asked"], "... without asking again")
ASK["answer"] = QMessageBox.Yes
close(w, None)
# Save As guards
w = window(project=OLD)
ASK["warned"], ASK["asked"] = [], []
# G55: a project is a folder, and Windows' save dialog OPENS a folder whose name is chosen, so
# choosing an existing project comes back as a path inside it. Inside the open project = save it
SAVE_TO["path"] = os.path.join(OLD, os.path.basename(OLD))
pid_before = w._project_id
w.session.add_point(0, 12.0, 12.0)
check(w._save_project_as() and str(w.project_path) == OLD and w._project_id == pid_before
      and not os.path.exists(SAVE_TO["path"]) and not ASK["asked"] and not w.project.dirty,
      "Save As into the open project's own folder saves that project (same id, nothing nested, no question)")
OTHER = os.path.join(OUT, "other-project.kinetrace")
projectfile.save(projectfile.load(OLD), OTHER)
other_before = projectfile.load(OTHER).sessions[0].n_points
w.session.add_point(0, 13.0, 13.0)
SAVE_TO["path"] = os.path.join(OTHER, "other-project.kinetrace")      # what the dialog returns after opening it
ASK["answer"], ASK["asked"] = QMessageBox.No, []
check(not w._save_project_as() and "Save as this project?" in ASK["asked"] and str(w.project_path) == OLD
      and projectfile.load(OTHER).sessions[0].n_points == other_before and not os.path.exists(SAVE_TO["path"]),
      "another project's folder: asked first; No changes nothing and nests nothing", str(ASK["asked"]))
ASK["answer"] = QMessageBox.Yes
junk = os.path.join(OUT, "thesis.kinetrace")
os.makedirs(junk, exist_ok=True)
open(os.path.join(junk, "chapter1.docx"), "w").write("x")
SAVE_TO["path"] = junk
ASK["warned"] = []
check(not w._save_project_as() and ASK["warned"] and os.listdir(junk) == ["chapter1.docx"],
      "... and a folder of that name that is not a project, leaving it alone")
check(str(w.project_path) == OLD, "after a refused Save As the project still saves where it did")
# Export Project as One File
SAVE_TO["path"] = os.path.join(OUT, "to-send.kinetrace")
w._export_single_file()
pump(0.2)
sent = SAVE_TO["path"]
check(os.path.isfile(sent) and zipfile.is_zipfile(sent) and "one file" in w.toast.text(),
      "File -> Export Project as One File writes one .kinetrace (a zip) and says so", w.toast.text())
check(projectfile.load(sent).sessions[0].n_points == w.session.n_points and str(w.project_path) == OLD,
      "... that opens with the same data, and the project folder stays the project")
SAVE_TO["path"] = os.path.join(OTHER, "other-project.kinetrace")
ASK["answer"], ASK["asked"] = QMessageBox.Yes, []
n_now = w.session.n_points
check(w._save_project_as() and str(w.project_path) == OTHER and projectfile.load(OTHER).sessions[0].n_points == n_now
      and not os.path.exists(SAVE_TO["path"]),
      "... Yes saves the work AS that project (no folder made inside it) (G55)")
close(w, None)
w = window(project=OLD)                                  # the exports below work on the first project
# exports on save, chosen by real clicks in File -> Keep Exports Up to Date...
from kinetrace import autoexport  # noqa: E402


def tick_and_ok():
    dlg = QApplication.activeModalWidget()
    for b in dlg.findChildren(QCheckBox):
        if b.text().startswith(("DeepLabCut", "DLTdv8 xypts — one file")):
            QTest.mouseClick(b, _Qt.LeftButton)
    QTest.mouseClick(dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), _Qt.LeftButton)


QTimer.singleShot(200, tick_and_ok)
w._exports_dialog()
check(w.project.exports == ["dltdv", "dlc"] and w.project.dirty, "the dialog's ticks are the project's exports "
      "(a change to save)", str(w.project.exports))
check(w._save_project(), "Save")
w._exports_worker.wait(30000)
pump(0.3)
ex = os.path.join(OLD, "exports")
made = sorted(os.path.basename(x) for x in glob.glob(os.path.join(ex, "*")))
check(made == ["cam1_DLC.csv", "cam1_xypts.csv", "cam1_xypts_pointnames.csv", "exports.json"],
      "after the save exports/ holds the DeepLabCut and DLTdv8 files", str(made))
dlc = os.path.join(ex, "cam1_DLC.csv")
t0_ = os.stat(dlc).st_mtime_ns
w._goto(5)
w._save_project()
w._exports_worker.wait(30000)
pump(0.2)
check(os.stat(dlc).st_mtime_ns == t0_, "a save that changed no data rewrites no export")
w.session.set_position(3, 0, 111.0, 99.0)
w._save_project()
w._exports_worker.wait(30000)
pump(0.2)
check(os.stat(dlc).st_mtime_ns != t0_ and "111.000" in open(dlc).read(), "a moved point: the export follows")
check(autoexport.refresh(OLD, ["dlc"])[0] == [] and not os.path.exists(os.path.join(ex, "cam1_xypts.csv")),
      "a format no longer chosen: its files go, the others stay")
close(w, None)
# a save cut short: the next open undoes it and says so
CRASH = r"""
import os, sys
sys.path.insert(0, {root!r})
from kinetrace import projectfile as pf
p = pf.load({path!r})
p.sessions[0].tracks[:, 0] += 5.0
real, n = pf._replace, [0]
def crash(src, dst):
    if ".cache" not in str(dst) and not str(dst).endswith("pending.json"):
        n[0] += 1
        if n[0] == 2:
            os._exit(3)
    real(src, dst)
pf._replace = crash
pf.save(p, {path!r})
"""
before_crash = data(OLD)
rc = subprocess.run([sys.executable, "-c", CRASH.format(root=ROOT, path=OLD)]).returncode
w = window(project=OLD)
pump(0.3)
check(rc == 3 and data(OLD) == before_crash and "cut short" in w.toast.text(),
      "a save cut short (the process died mid-save): opening puts the project back and says so", w.toast.text())
# G44: the close question names what changed since the last save
w.project.dirty = True
check(w._save_project(), "saved (the undone save left no fingerprints to compare with)")
w._on_select(0)
pname = w.session.points[0].name
w._goto(3)
pump(0.2)
w._on_annotate(222.0, 77.0)                      # a plain click with the point selected places it
ASK["asked"], ASK["texts"], ASK["close"] = [], [], QMessageBox.Cancel
w.close()
pump(0.3)
said = ASK["texts"][-1] if ASK["texts"] else ""
check(w.isVisible() and "Changed since your last save" in said and f"cam1: {pname} (positions)" in said,
      "the close question names what changed: the point a click placed", said[:200])
w._undo_run()
pump(0.2)
ASK["asked"], ASK["texts"] = [], []
w.close()
pump(0.3)
said = ASK["texts"][-1] if ASK["texts"] else ""
check(w.isVisible() and "Nothing differs from your last save" in said,
      "... and says so when the change was undone (Ctrl+Z)", said[:200])
# File -> Quit (G43): the same close as the window's x, by a real Ctrl+Q
from PySide6.QtWidgets import QMenu  # noqa: E402
file_menu = next(m for m in w.menuBar().findChildren(QMenu) if m.title().replace("&", "") == "File")
entries = [a for a in file_menu.actions() if not a.isSeparator()]
check(entries[-1] is w.act_quit and w.act_quit.text().replace("&", "") == "Quit"
      and w.act_quit.shortcut().toString() == "Ctrl+Q", "File ends with Quit (Ctrl+Q)")
w.session.add_point(0, 33.0, 44.0)
w.project.dirty = True
ASK["asked"], ASK["close"] = [], QMessageBox.Cancel
app.setActiveWindow(w)
w.activateWindow()
pump(0.2)
QTest.keyClick(w, _Qt.Key_Q, _Qt.ControlModifier)
pump(0.3)
check("Save changes?" in ASK["asked"] and w.isVisible(), "Ctrl+Q with unsaved changes asks first; Cancel stays",
      str(ASK["asked"]))
ASK["asked"], ASK["close"] = [], QMessageBox.Discard
w.act_quit.trigger()
pump(0.3)
check("Save changes?" in ASK["asked"] and not w.isVisible(), "File -> Quit, Discard: the window closes")

forget_recovery(VA, VB)
print("\n" + "=" * 62)
if fails:
    print(f"verify_recovery FAILED: {len(fails)}")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("verify_recovery PASSED")
