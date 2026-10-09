"""(G178, X37, G179) After an install: the launcher's last words, the app's one-time pop-up, and the
offline reminder to check for updates.

  [1] the closing text: the version, which window may be closed (a Mac hands over to Kinetrace.app:
      "you can close it"; Windows / Linux: keep it open, closing it closes Kinetrace), how to start it
      next time, the folder and what must not be deleted (.venv, models, the launchers), what may be
      (the ZIP, the install logs); ASCII only (the Windows console is cp1252).
  [2] the flag: set by `python -m kinetrace.welcome`, taken ONCE; a flag that cannot be removed (a
      read-only folder) never shows the pop-up (it would show at every start).
  [3] the launchers: run.bat prints it only after an install in that run (not for --check); run.sh
      hands a Kinetrace.command start over to Kinetrace.app on a Mac (that Terminal can be closed) and
      warns about Documents / Desktop / Downloads; both still parse.
  [4] the pop-up (offscreen): `app.show_install_welcome` says it is installed, how to begin and to start
      next time, which window may be closed, what to keep.
  [5] the update reminder: offline; after 30 days without a check a clickable note (it opens Check for
      Updates), not again within a week; a check that reached GitHub starts the 30 days again.

.venv\\Scripts\\python.exe tests\\verify_welcome.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kinetrace import APP_VERSION, welcome  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if not ok:
        FAILS.append(what)


print("[1] the closing text")
mac = welcome.finished_text("0.5.0", "/Users/someone/Kinetrace", "mac-app", "Darwin")
win = welcome.finished_text("0.5.0", r"C:\Users\someone\Kinetrace", "console", "Windows")
lin = welcome.finished_text("0.5.0", "/home/someone/Kinetrace", "console", "Linux")
for name, t in (("Mac", mac), ("Windows", win), ("Linux", lin)):
    check("Kinetrace 0.5.0 is installed and ready" in t and "DO NOT delete" in t and ".venv" in t
          and "models" in t and "WHOLE folder is fine" in t and "Kinetrace-main.zip" in t
          and "logs/install-*.log" in t and "To uninstall, delete this folder" in t,
          f"{name}: installed, what to keep, what may be deleted, uninstall")
    check(all(ord(c) < 128 for c in t), f"{name}: ASCII only (the Windows console is cp1252)")
check("you can close it" in mac and "Kinetrace.app in this folder" in mac and "/Users/someone/Kinetrace" in mac
      and "Kinetrace.command" in mac, "Mac: the Terminal may be closed; next time Kinetrace.app; its files")
check("KEEP THIS WINDOW OPEN" in win and "closing it closes Kinetrace" in win and "run.bat" in win
      and "Kinetrace.app" not in win, "Windows: keep the window open; next time run.bat")
check("KEEP THIS WINDOW OPEN" in lin and "./run.sh" in lin, "Linux: keep the window open; next time ./run.sh")

print("\n[2] the flag")
tmp = Path(tempfile.mkdtemp(prefix="kt_welcome_"))
check(not welcome.take_pending(str(tmp)), "no install: no pop-up")
welcome.set_pending(str(tmp))
check(welcome.take_pending(str(tmp)) and not welcome.take_pending(str(tmp)), "after an install: once, then never")
welcome.set_pending(str(tmp))
real_unlink = Path.unlink
Path.unlink = lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("read-only"))
try:
    shown = welcome.take_pending(str(tmp))
finally:
    Path.unlink = real_unlink
check(not shown and welcome.flag_path(str(tmp)).exists(), "a flag that cannot be removed: no pop-up (not at every start)")
env = dict(os.environ, PYTHONIOENCODING="cp1252")
venv_like = tmp / "venv"
venv_like.mkdir()
r = subprocess.run([sys.executable, "-c",
                    "import sys; sys.prefix = sys.argv[1]; sys.argv = ['x', 'console']; "
                    "from kinetrace import welcome; welcome.main(sys.argv[1:])", str(venv_like)],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="cp1252")
check(r.returncode == 0 and f"Kinetrace {APP_VERSION} is installed" in r.stdout
      and welcome.flag_path(str(venv_like)).exists(),
      "python -m kinetrace.welcome: prints on a cp1252 console and leaves the flag", r.stderr[-300:])
shutil.rmtree(tmp, ignore_errors=True)

print("\n[3] the launchers")
bat = (ROOT / "run.bat").read_text(encoding="utf-8", errors="replace")
sh = (ROOT / "run.sh").read_text(encoding="utf-8")
i_check, i_wel, i_launch = (bat.index('if "%~1"=="--check" goto :done'), bat.index("-m kinetrace.welcome console"),
                            bat.index("\n:launch"))          # the label, not "goto :launch"
check(i_check < i_wel < i_launch, "run.bat: after an install in this run, not for --check, before the launch")
check('JUST_INSTALLED=1' in sh and sh.index('[ "$1" = "--check" ] && exit 0') < sh.index("JUST_INSTALLED=1"),
      "run.sh: an install in this run is noted after the --check exit")
check('[ "${KINETRACE_VIA:-}" = "command" ] && [ -d Kinetrace.app ]' in sh and "open Kinetrace.app --args" in sh
      and "kinetrace.welcome mac-app" in sh and "You can close this Terminal window" in sh,
      "run.sh: a Kinetrace.command start hands over to Kinetrace.app on a Mac")
check('"$HOME/Documents/"*|"$HOME/Desktop/"*|"$HOME/Downloads/"*)' in sh, "run.sh: warns about Documents / Desktop / Downloads")
cmd = (ROOT / "Kinetrace.command").read_text(encoding="utf-8")
check("KINETRACE_VIA=command exec ./run.sh" in cmd, "Kinetrace.command marks a Terminal start")
bash = shutil.which("bash")
if bash:
    check(subprocess.run([bash, "-n", str(ROOT / "run.sh")]).returncode == 0
          and subprocess.run([bash, "-n", str(ROOT / "Kinetrace.command")]).returncode == 0, "run.sh and Kinetrace.command parse")

print("\n[4] the pop-up")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
app = QApplication.instance() or QApplication([])
from kinetrace import app as kt_app  # noqa: E402
seen = {}


def fake_exec(self):
    seen["title"], seen["text"] = self.windowTitle(), self.text()
    return QMessageBox.Ok


QMessageBox.exec = fake_exec
kt_app.show_install_welcome(None)
text = seen.get("text", "")
check(seen.get("title") == "Kinetrace is installed" and f"Kinetrace {APP_VERSION} is installed and ready" in text
      and "Open Video" in text and "Next time" in text and "Keep the Kinetrace folder as it is" in text,
      "the pop-up: installed, how to begin, next time, what to keep", text[:200])
for system, words in (("Darwin", "you can close it"), ("Windows", "closing it closes Kinetrace")):
    check(words in welcome.popup_html(APP_VERSION, system), f"the pop-up on {system}: {words}")

print("\n[5] the update reminder (G179, offline)")
from kinetrace import update  # noqa: E402
D = 86400.0
T = 1_800_000_000.0
check(not update.reminder_due(T, 0, 0, 0), "never counted: no reminder")
check(not update.reminder_due(T + 29 * D, 0, T, 0) and update.reminder_due(T + 30 * D, 0, T, 0),
      "never checked: a reminder after 30 days of counting, not before")
check(not update.reminder_due(T + 60 * D, T + 50 * D, T, 0), "a check 10 days ago: no reminder")
check(not update.reminder_due(T + 33 * D, 0, T, T + 30 * D) and update.reminder_due(T + 37 * D, 0, T, T + 30 * D),
      "reminded 3 days ago: not again; a week later: again")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from kinetrace.app import MainWindow  # noqa: E402
w = MainWindow()
w.resize(1200, 800)
w.show()
s = w._settings()
for k in ("updates/last_check", "updates/since", "updates/last_reminder"):
    s.remove(k)
s.sync()
asked = []
w._check_updates = lambda: asked.append("check")          # what the note's click starts
check(not w._remind_updates(T) and float(w._settings().value("updates/since", 0)) == T,
      "the first start only starts counting")
check(not w._remind_updates(T + 20 * D), "20 days: nothing")
shown = w._remind_updates(T + 31 * D)
note = " ".join(str(n[0]) for n in w.toast._notices)
check(shown and w.toast.isVisible() and "31 days" in note and "Check for Updates" in note,
      "31 days: a note says so", note[:160])
QTest.mouseClick(w.toast, Qt.LeftButton)
app.processEvents()
check(asked == ["check"], "clicking the note opens Check for Updates", asked)
check(not w._remind_updates(T + 32 * D) and w._remind_updates(T + 38 * D), "not again the next day; a week later again")
del w._check_updates                                         # the real one, with a stand-in dialog


class _Dialog:
    def __init__(self, *a, **k):
        self.release = object()                              # GitHub answered

    def exec(self):
        return 0

    def deleteLater(self):
        pass


from kinetrace import updatedialog  # noqa: E402
real_dialog = updatedialog.UpdateDialog
updatedialog.UpdateDialog = _Dialog
try:
    w._check_updates()
finally:
    updatedialog.UpdateDialog = real_dialog
last = float(w._settings().value("updates/last_check", 0))
import time  # noqa: E402
check(abs(last - time.time()) < 60 and not w._remind_updates(time.time() + 5 * D),
      "a check that reached GitHub starts the 30 days again", last)
w.close()
app.processEvents()
w._dev_probe.wait(10000)

print("\nverify_welcome: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
