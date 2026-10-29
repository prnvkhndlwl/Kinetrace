"""(X40) The Kinetrace shortcut on Windows (owner 2026-10-10: start Kinetrace without clicking run.bat each
time): kinetrace/winapp.py writes Kinetrace.lnk, kinetrace/winlaunch.py is what it runs.

  [1] the shortcut: written through the shell's COM objects and read back -- target, arguments, working
      folder, icon, description AND the taskbar id (= the one app.main registers, so a pinned shortcut and
      the running window are one taskbar button); the .ico holds every size as a PNG
  [2] stale: fresh = nothing to do; a new shortcut layout or a moved folder rewrites it (and a Start menu
      entry the user added for the old place follows the folder)
  [3] the launcher: what is due (install / models / repair) and what it starts -- run.bat in a console
      when a step is due, else the app with no console; a shortcut moved out of its folder says so
  [4] a REAL double-click: the shell opens a .lnk made by write_link, the launcher runs and, with an
      install due, starts run.bat (it writes a marker); Help -> Check for Updates' restart goes the same way
  [5] run.bat makes the shortcut after an install and on every start (--if-stale), and still ends on ONE line;
      the closing text and the first-start pop-up name the shortcut when it is there
  [6] Help -> Add Kinetrace to the Start Menu (offscreen window, a scratch Start menu): asks, adds the
      entry for THIS folder, the menu then offers Remove, Remove takes it out; Help -> Kinetrace's Folders
      lists the entry while it is there

Windows only (prints SKIPPED elsewhere). .venv\\Scripts\\python.exe tests\\verify_winapp.py
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "out" / "winapp"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["KINETRACE_START_MENU_DIR"] = str(OUT / "start_menu")      # never the real Start menu
os.environ["KINETRACE_RECOVERY_DIR"] = str(OUT / "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

if sys.platform != "win32":
    print("verify_winapp SKIPPED (Windows only)")
    sys.exit(0)

FAILS = []


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if not ok:
        FAILS.append(what)


def wait_for(path: Path, seconds: float = 20.0) -> bool:
    t_end = time.time() + seconds
    while not path.exists() and time.time() < t_end:
        time.sleep(0.2)
    return path.exists()


from PySide6.QtWidgets import QApplication, QMenu, QMessageBox  # noqa: E402

# a full QApplication FIRST: winapp.write_icon makes a bare QGuiApplication when none exists (run.bat's case),
# and a window cannot be built on that one
app = QApplication.instance() or QApplication([])
from kinetrace import update, winapp, winlaunch  # noqa: E402

MARKER_BAT = b'@echo started %*> "%~dp0marker.txt"\r\nexit\r\n'


def fake_root(name: str) -> Path:
    """A Kinetrace folder with what winapp / winlaunch look at (no real environment)."""
    r = OUT / name
    (r / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (r / "kinetrace").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "kinetrace" / "winlaunch.py", r / "kinetrace" / "winlaunch.py")
    (r / "run.bat").write_bytes(MARKER_BAT)
    return r


# ---------------------------------------------------------------- [1] the shortcut
print("[1] the shortcut, written and read back")
A = fake_root("folder A & co")             # an ampersand in the folder name, as in I156
lnk = winapp.build(A)
got = winapp.read_link(lnk)
spec = winapp.link_spec(A)
check(lnk == A / "Kinetrace.lnk" and lnk.exists(), "Kinetrace.lnk is made in the Kinetrace folder", lnk)
check(all(got.get(k) == v for k, v in spec.items()), "target, arguments, folder, icon, description, taskbar id",
      {k: (got.get(k), v) for k, v in spec.items() if got.get(k) != v})
check(got.get("app_id") == "Kinetrace.Kinetrace" == winapp.APP_ID,
      "the taskbar id is the app's own (app.main registers Kinetrace.Kinetrace)", got.get("app_id"))
app_src = (ROOT / "kinetrace" / "app.py").read_text(encoding="utf-8")
check("from kinetrace.winapp import APP_ID" in app_src and "SetCurrentProcessExplicitAppUserModelID(APP_ID)" in app_src
      and app_src.count(winapp.APP_ID) == 0,
      "app.main registers the same id, read from winapp.APP_ID (one taskbar button for the pinned shortcut "
      "and the window; R56: no second copy of the text)")
check(spec["arguments"].startswith("-I ") and "winlaunch.py" in spec["arguments"],
      "the launcher runs isolated: standard library only", spec["arguments"])
ico = winapp.icon_path(A).read_bytes()
n = struct.unpack("<HHH", ico[:6])[2]
entries = [struct.unpack("<BBBBHHII", ico[6 + 16 * i:22 + 16 * i]) for i in range(n)]
check(n == len(winapp.ICON_SIZES) and all(ico[e[7]:e[7] + 8] == b"\x89PNG\r\n\x1a\n" for e in entries)
      and entries[-1][0] == 0, "the icon: every size, PNG entries, 256 written as 0", (n, entries[-1][:2]))
# (X51) another program (Explorer drawing the icon, the shell reading the shortcut) holds the file for a
# moment: the rewrite waits and retries like every other save (paths.replace_retry), it does not fail
import threading  # noqa: E402


def held(path: Path, seconds: float = 0.4) -> threading.Timer:
    f = open(path, "rb")                       # no FILE_SHARE_DELETE: os.replace onto it is refused while open
    t = threading.Timer(seconds, f.close)
    t.start()
    return t


for what, target, write in (("Kinetrace.lnk", lnk, lambda: winapp.write_link(lnk, spec)),
                            ("kinetrace.ico", winapp.icon_path(A), lambda: winapp.write_icon(winapp.icon_path(A)))):
    timer = held(target)
    try:
        write()
        ok, err = True, ""
    except OSError as e:
        ok, err = False, repr(e)
    timer.join()
    check(ok and target.exists(), f"{what} held open for 0.4 s by another program: still rewritten", err)

# ---------------------------------------------------------------- [2] stale
print("[2] stale: a new layout, a moved folder")
check(winapp.stale_reason(A) == "", "fresh: nothing to do")
real_version = winapp.SHORTCUT_VERSION
winapp.SHORTCUT_VERSION = "999"
check(winapp.stale_reason(A) == "version", "a new shortcut layout: rewrite")
winapp.SHORTCUT_VERSION = real_version
winapp.add_start_menu(A)                                     # the user added a Start menu entry
check(winapp.start_menu_points_to(A), "(setup) a Start menu entry for folder A")
B = OUT / "folder B"
shutil.rmtree(B, ignore_errors=True)
shutil.move(str(A), str(B))                                  # the user moves the whole folder
check(winapp.stale_reason(B) == "moved", "a moved folder: rewrite", winapp.stale_reason(B))
winapp.build(B)
check(winapp.read_link(B / "Kinetrace.lnk")["workdir"] == str(B) and winapp.stale_reason(B) == "",
      "rewritten for the new place")
check(winapp.start_menu_points_to(B) and not winapp.start_menu_points_to(A),
      "the Start menu entry follows the folder")
winapp.remove_start_menu()
check(not winapp.start_menu_link().exists(), "the Start menu entry can be removed")

# ---------------------------------------------------------------- [3] the launcher
print("[3] the launcher: what is due, what it starts")
C = fake_root("folder C")
check(winlaunch.due(C) == "install", "no install marker: install (first install, or an update asked for it)")
(C / ".venv" / "kinetrace-install.json").write_text("{}", encoding="utf-8")
check(winlaunch.due(C) == "models", "AllTracker missing: run.bat fetches it")
(C / "models" / "alltracker" / "nets").mkdir(parents=True)
(C / "models" / "alltracker" / "nets" / "alltracker.py").write_text("", encoding="utf-8")
check(winlaunch.due(C) == "repair", "no working environment: repair")
check(winlaunch.due(ROOT) == "", "this Kinetrace folder: nothing due")
calls = []
real_popen, real_run_bat, real_due = subprocess.Popen, winlaunch.run_bat, winlaunch.due
winlaunch.run_bat = lambda root, argv: calls.append(("run.bat", root, list(argv)))
winlaunch.subprocess.Popen = lambda args, **k: calls.append(("popen", list(args), k.get("creationflags")))
try:
    winlaunch.main(["x.kinetrace"], C)
    check(calls and calls[-1] == ("run.bat", C, ["x.kinetrace"]), "a step is due: run.bat, with the arguments",
          calls[-1:] if calls else calls)
    winlaunch.due = lambda root=None: ""
    winlaunch.main(["x.kinetrace"], C)
    check(calls[-1][0] == "popen" and calls[-1][1][0].endswith("pythonw.exe")
          and calls[-1][1][1:] == ["-m", "kinetrace", "x.kinetrace"] and calls[-1][2] & 0x00000008,
          "nothing due: the app through pythonw, detached (no console), with the arguments", calls[-1])
finally:
    winlaunch.subprocess.Popen, winlaunch.run_bat, winlaunch.due = real_popen, real_run_bat, real_due
told = []
real_tell = winlaunch.tell
winlaunch.tell = told.append
try:
    rc = winlaunch.main([], OUT / "no such folder")
finally:
    winlaunch.tell = real_tell
check(rc == 1 and told and "run.bat" in told[0], "a shortcut moved out of its folder says what to do", told)

# ---------------------------------------------------------------- [4] a real double-click
print("[4] a real double-click on the shortcut, and the update's restart")
D = fake_root("folder D & co")
spec = winapp.link_spec(D)
spec["target"] = str(winapp.launcher_python(ROOT))          # this folder's pythonw (D has none of its own)
winapp.write_link(D / "Kinetrace.lnk", spec)
os.startfile(str(D / "Kinetrace.lnk"))                       # what Explorer does on a double-click
check(wait_for(D / "marker.txt"), "the shell opens the shortcut, the launcher sees the install is due and runs run.bat")
(D / "marker.txt").unlink(missing_ok=True)
real_lp = winapp.launcher_python
winapp.launcher_python = lambda root=None: real_lp(ROOT)
try:
    update.relaunch(D)
finally:
    winapp.launcher_python = real_lp
check(wait_for(D / "marker.txt"), "Check for Updates' restart goes through the shortcut's launcher when it is there")
E = OUT / "folder E"                                          # no shortcut: run.bat as before (I156)
E.mkdir(exist_ok=True)
(E / "run.bat").write_bytes(MARKER_BAT)
update.relaunch(E)
check(wait_for(E / "marker.txt"), "... and run.bat when there is none")

# ---------------------------------------------------------------- [5] run.bat
print("[5] run.bat")
bat = (ROOT / "run.bat").read_text(encoding="utf-8")
inst = bat.index("kinetrace.welcome console")
check(bat.rfind("kinetrace.winapp --if-stale", 0, inst) > bat.index('if "%~1"=="--check" goto :done'),
      "after an install (not --check) the shortcut is made before the closing text that names it")
last = [ln for ln in bat.splitlines() if ln.strip() and not ln.lstrip().lower().startswith("rem")]
i_launch = bat.index(":launch")
check(bat.find("kinetrace.winapp --if-stale", i_launch) > i_launch
      and last[-3].startswith('".venv\\Scripts\\python.exe" -m kinetrace %*'),
      "every start refreshes it (--if-stale), and the app still starts on ONE line (I142)", last[-3:])
check(bat.count("\r\n") == bat.count("\n") or (ROOT / "run.bat").read_bytes().count(b"\r\n") > 100,
      "run.bat keeps its CRLF line ends")

from kinetrace import welcome  # noqa: E402
txt = welcome.finished_text("0.5.0", str(B), "console", "Windows")
check("the Kinetrace shortcut in this folder" in txt and "Pin to taskbar" in txt and all(ord(c) < 128 for c in txt),
      "the closing text after an install names the shortcut and how to pin it (ASCII)")
check("double-click run.bat" in welcome.finished_text("0.5.0", str(OUT / "folder E"), "console", "Windows"),
      "... and run.bat when no shortcut could be made")
check("Kinetrace shortcut" in welcome.popup_html("0.5.0", "Windows", str(B)), "the first-start pop-up names it too")

# ---------------------------------------------------------------- [6] Help -> Start menu
print("[6] Help -> Add Kinetrace to the Start Menu (offscreen)")
asked = []
QMessageBox.question = staticmethod(lambda *a, **k: (asked.append(a[2]), QMessageBox.Yes)[1])
QMessageBox.warning = staticmethod(lambda *a, **k: (asked.append("WARNING " + str(a[2])), QMessageBox.Ok)[1])
from kinetrace import paths  # noqa: E402
from kinetrace.app import MainWindow  # noqa: E402

win = MainWindow()
win.show()
for _ in range(20):
    app.processEvents()
    time.sleep(0.02)
menu = next(m for m in win.menuBar().findChildren(QMenu) if m.title().replace("&", "") == "Help")
menu.aboutToShow.emit()
check(win.act_start_menu.isVisible() and win.act_start_menu.text().startswith("Add Kinetrace"),
      "Help offers Add Kinetrace to the Start Menu", win.act_start_menu.text())
win.act_start_menu.trigger()
check(asked and "OUTSIDE the Kinetrace folder" in asked[-1], "it asks first and says the entry is outside the folder")
check(winapp.start_menu_points_to(ROOT) and winapp.read_link(winapp.start_menu_link())["app_id"] == winapp.APP_ID,
      "the entry starts THIS folder's Kinetrace, with the app's taskbar id")
check(any(p.what.startswith("Start menu entry") for p in paths.places()),
      "Help -> Kinetrace's Folders lists it (the one thing outside the folder)")
menu.aboutToShow.emit()
check(win.act_start_menu.text().startswith("Remove Kinetrace"), "the menu then offers Remove",
      win.act_start_menu.text())
win.act_start_menu.trigger()
check(not winapp.start_menu_link().exists(), "Remove takes it out")
menu.aboutToShow.emit()
check(win.act_start_menu.text().startswith("Add Kinetrace") and not any(
      p.what.startswith("Start menu entry") for p in paths.places()), "... and Add is offered again")
win.close()
for _ in range(20):
    app.processEvents()
    time.sleep(0.02)
win._dev_probe.wait(20000)

print()
if FAILS:
    print(f"verify_winapp FAILED ({len(FAILS)}): " + "; ".join(FAILS))
    sys.exit(1)
print("verify_winapp PASSED")
