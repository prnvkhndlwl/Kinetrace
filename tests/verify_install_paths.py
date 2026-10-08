"""Where Kinetrace keeps things (Mac install audit P1 / P2-7): `--paths` and
Help -> Kinetrace's Folders.

  [1] `python -m kinetrace --paths` (no window): the environment, models, logs,
      recovery copies, skeletons and settings.ini listed INSIDE the folder, with
      sizes; projects said to be kept; exits 0 and opens no window
  [2] the Help menu entry, clicked through the menu: the same places in a list;
      Show (on the selected .venv row) opens that folder; a heading row opens
      nothing
CPU, ~20 s. Run: .venv\\Scripts\\python.exe tests\\verify_install_paths.py
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = sys.executable

# ---- [1] the CLI, as the launchers run it (not offscreen: the real folders) ----
env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
r = subprocess.run([PY, "-m", "kinetrace", "--paths"], cwd=ROOT, env=env, capture_output=True,
                   text=True, timeout=120)
out = r.stdout
assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
inside, _, outside = out.partition("Outside the Kinetrace folder:")
for what, path in (("Environment", ROOT / ".venv"), ("Models", ROOT / "models"), ("Error log", ROOT / "logs"),
                   ("recovery copies", ROOT / "recovery"), ("Settings", ROOT / "settings.ini"),
                   ("skeletons", ROOT / "skeletons")):
    line = next((ln for ln in inside.splitlines() if what in ln), "")
    assert str(path) in line, f"{what} not listed inside the folder: {line!r}\n{out}"
assert "[" in next(ln for ln in inside.splitlines() if "Environment" in ln), "no size for .venv"
assert "never deleted" in out and "projects" in out
assert out.isascii(), "the report must be ASCII (Windows consoles)"
print("[1] --paths lists every place inside the folder with sizes, no window: OK")

# ---- [2] Help -> Kinetrace's Folders, through the menu ----
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QAction, QDesktopServices  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QListWidget, QMenu, QPushButton  # noqa: E402

app = QApplication([])
from kinetrace.app import MainWindow  # noqa: E402

opened = []
QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toLocalFile()) or True)
seen = {}


def fake_exec(dlg):
    lst = dlg.findChild(QListWidget)
    rows = [lst.item(i).text() for i in range(lst.count())]
    seen["rows"] = rows
    show = next(b for b in dlg.findChildren(QPushButton) if b.text().startswith("Show in"))
    lst.setCurrentRow(0)                       # the heading: nothing to open
    QTest.mouseClick(show, Qt.LeftButton)
    seen["after_heading"] = list(opened)
    venv_row = next(i for i, t in enumerate(rows) if "Environment" in t)
    lst.setCurrentRow(venv_row)
    QTest.mouseClick(show, Qt.LeftButton)
    return 0


QDialog.exec = fake_exec
win = MainWindow()
win.show()
help_menu = next(m for m in win.findChildren(QMenu) if m.title() == "&Help")
help_actions = help_menu.actions()             # kept referenced (PySide6 harness pitfall)
act = next(a for a in help_actions if a.text().startswith("Kinetrace's &Folders"))
act.trigger()
app.processEvents()
rows = seen.get("rows") or []
assert rows and rows[0].startswith("Inside the Kinetrace folder"), rows
assert any("Environment" in t and str(ROOT / ".venv") in t and "[" in t for t in rows), rows
assert any(t.startswith("Outside the Kinetrace folder") for t in rows), rows
assert seen["after_heading"] == [], "a heading row must open nothing"
assert len(opened) == 1 and Path(opened[0]) == ROOT / ".venv", opened
print("[2] Help -> Kinetrace's Folders: the list, Show opens the selected folder: OK")
win._dev_probe.wait(30000)
win.close()

# ---- [3] uninstall.sh on a stand-in install (macOS / Linux; Git Bash on Windows) ----
import shutil  # noqa: E402
import tempfile  # noqa: E402
bash = shutil.which("bash")
if bash is None:
    print("[3] skipped: no bash on this machine")
else:
    tmp = Path(tempfile.mkdtemp(prefix="kt_uninstall_"))
    home = tmp / "home"
    fake = tmp / "Kinetrace novice"
    (fake / "kinetrace").mkdir(parents=True)
    (fake / "kinetrace" / "__init__.py").write_text("")
    (fake / "run.sh").write_text("")
    (fake / ".venv").mkdir()
    shutil.copy(ROOT / "uninstall.sh", fake / "uninstall.sh")
    data = home / ".local" / "share" / "kinetrace"          # the Linux / Git Bash fallback folder
    plist = home / "Library" / "Preferences" / "com.kinetrace.Kinetrace.plist"
    for p in (data / "recovery", plist.parent):
        p.mkdir(parents=True)
    plist.write_text("")
    keep = home / "Documents" / "trial.kinetrace"
    keep.mkdir(parents=True)

    def uninstall(answer):
        e = dict(env, HOME=str(home), KINETRACE_UNINSTALL_ANSWER=answer)
        e.pop("XDG_DATA_HOME", None)
        e.pop("XDG_CONFIG_HOME", None)
        return subprocess.run([bash, "uninstall.sh"], cwd=fake, env=e, capture_output=True, text=True,
                              timeout=60)

    (fake / "my.kinetrace").mkdir()                          # a project saved INSIDE: refused
    r = uninstall("DELETE")
    assert r.returncode == 1 and "my.kinetrace" in r.stdout and fake.exists(), r.stdout
    (fake / "my.kinetrace").rmdir()
    r = uninstall("no")
    assert r.returncode == 0 and "nothing was deleted" in r.stdout and fake.exists(), r.stdout
    r = uninstall("DELETE")
    assert r.returncode == 0, (r.stdout, r.stderr)
    mac = sys.platform == "darwin"
    assert not fake.exists(), "the folder must be gone"
    assert (not plist.exists()) if mac else (not data.exists()), "the outside folder must be gone"
    assert keep.exists(), "a project elsewhere must never be touched"
    shutil.rmtree(tmp, ignore_errors=True)
    print("[3] uninstall.sh: refuses with a project inside, asks, removes folder + outside traces, "
          "keeps projects: OK")
print("verify_install_paths PASSED")
