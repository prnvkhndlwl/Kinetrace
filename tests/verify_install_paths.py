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
# without the overrides run_suites.py sets (the recovery / log folders of the tests), so the
# install's own folders are what is listed
env = {k: v for k, v in os.environ.items()
       if k not in ("QT_QPA_PLATFORM", "KINETRACE_RECOVERY_DIR", "KINETRACE_LOG_DIR")}
r = subprocess.run([PY, "-m", "kinetrace", "--paths"], cwd=ROOT, env=env, capture_output=True,
                   text=True, timeout=120)
out = r.stdout
assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
inside, _, outside = out.partition("Outside the Kinetrace folder:")
for what, path in (("Environment", ROOT / ".venv"), ("Models", ROOT / "models"), ("Logs", ROOT / "logs"),
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

    def make_fake():
        (fake / "kinetrace").mkdir(parents=True)
        (fake / "kinetrace" / "__init__.py").write_text("")
        (fake / "run.sh").write_text("")
        (fake / ".venv").mkdir()
        shutil.copy(ROOT / "uninstall.sh", fake / "uninstall.sh")

    make_fake()
    data = home / ".local" / "share" / "kinetrace"          # the Linux / Git Bash fallback folder
    plist = home / "Library" / "Preferences" / "com.kinetrace.Kinetrace.plist"
    for p in (data / "recovery", plist.parent):
        p.mkdir(parents=True)
    plist.write_text("")
    keep = home / "Documents" / "trial.kinetrace"
    keep.mkdir(parents=True)

    def uninstall(answer, shared=""):
        e = dict(env, HOME=str(home), KINETRACE_UNINSTALL_ANSWER=answer, KINETRACE_UNINSTALL_SHARED=shared)
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
    mac = sys.platform == "darwin"
    outside = plist if mac else data
    # the folders outside are SHARED by every copy (Mac report): kept unless asked for
    r = uninstall("DELETE", shared="no")
    assert r.returncode == 0 and "shared by every Kinetrace copy" in r.stdout, (r.stdout, r.stderr)
    assert not fake.exists() and outside.exists(), "the folder goes; the shared outside folder stays"
    make_fake()
    r = uninstall("DELETE", shared="yes")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert not fake.exists() and not outside.exists(), "both gone when asked"
    assert keep.exists(), "a project elsewhere must never be touched"
    shutil.rmtree(tmp, ignore_errors=True)
    print("[3] uninstall.sh: refuses with a project inside, asks, removes the folder; the shared outside "
          "traces only when asked; keeps projects: OK")
# ---- [4] the gated models' folders, as install.py leaves them (Mac install audit P1) ----
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("kt_install", ROOT / "install.py")
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)
from kinetrace import bodypose, downloads, segmenter  # noqa: E402
tmp = Path(tempfile.mkdtemp(prefix="kt_gated_"))
real = downloads.MODELS_DIR, segmenter.MODELS_DIR, bodypose.MODELS_DIR
try:
    downloads.MODELS_DIR = segmenter.MODELS_DIR = bodypose.MODELS_DIR = tmp
    inst.gated_model_folders()
    inst.gated_model_folders()                               # twice: an install that resumes
    sam3 = (tmp / "sam3" / "PUT_FILES_HERE.txt").read_text(encoding="utf-8")
    assert "model.safetensors" in sam3 and downloads.HF_REVISIONS["facebook/sam3"] in sam3, sam3
    assert '--exclude "sam3.pt"' in sam3 and "Settings" in sam3, sam3
    # the CLI stays inside the folder and offline; no token on a command line (Mac report)
    assert 'HF_HOME="$PWD/models/hf"' in sam3 and "HF_HUB_DISABLE_UPDATE_CHECK=1" in sam3, sam3
    assert "--token" not in sam3 and "auth login" in sam3, sam3
    s3db = tmp / "sam-3d-body-dinov3" / "PUT_FILES_HERE.txt"
    if sys.platform == "darwin":
        assert not s3db.exists(), "SAM 3D Body cannot run on a Mac: no folder inviting its 2.8 GB"
    else:
        t = s3db.read_text(encoding="utf-8")
        assert "model.ckpt" in t and "assets/mhr_model.pt" in t and "NVIDIA" in t, t
    assert not (tmp / "sam-3d-body").exists(), "git clone needs models/sam-3d-body absent or empty"
    # a note alone is not a model: nothing in the app changes until the weights are there
    assert segmenter.local_dir("sam3") is None and segmenter.preferred_backend() == segmenter.DEFAULT_BACKEND
    assert bodypose.local_dir("sam-3d-body-dinov3") is None
finally:
    downloads.MODELS_DIR, segmenter.MODELS_DIR, bodypose.MODELS_DIR = real
    shutil.rmtree(tmp, ignore_errors=True)
# the install logs: only the newest few are kept (each holds pip's full log, ~13 MB; Mac report)
logs_tmp = Path(tempfile.mkdtemp(prefix="kt_logs_"))
(logs_tmp / "logs").mkdir()
for i in range(5):
    (logs_tmp / "logs" / f"install-2026-01-0{i + 1}-000000.log").write_text("x")
(logs_tmp / "logs" / "kinetrace.log").write_text("errors")
real_here = inst.HERE
try:
    inst.HERE = str(logs_tmp)
    newest = inst._new_log()
finally:
    inst.HERE = real_here
left = sorted(f.name for f in (logs_tmp / "logs").iterdir())
assert newest and len([f for f in left if f.startswith("install-")]) == inst.KEEP_INSTALL_LOGS, left
assert "kinetrace.log" in left and "install-2026-01-05-000000.log" in left, left
shutil.rmtree(logs_tmp, ignore_errors=True)
print("[4] install.py: models/sam3 (+ SAM 3D Body's off a Mac) with PUT_FILES_HERE.txt; the newest install logs; "
      "a note alone is not a model: OK")
# ---- [5] Kinetrace.app (macOS; built and its launcher run on a stand-in folder anywhere) ----
import plistlib  # noqa: E402
from kinetrace import APP_VERSION, macapp  # noqa: E402
if bash is None:
    print("[5] skipped: no bash on this machine")
else:
    tmp = Path(tempfile.mkdtemp(prefix="kt_app_"))
    fake = tmp / "Kinetrace moved"
    (fake / ".venv" / "bin").mkdir(parents=True)
    (fake / "run.sh").write_text('echo "RAN $*"\n', encoding="utf-8", newline="\n")
    py = fake / ".venv" / "bin" / "python"
    py.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8", newline="\n")
    py.chmod(0o755)
    stub = tmp / "stubs"
    stub.mkdir()
    (stub / "open").write_text(f'#!/bin/bash\necho "$*" > "{(tmp / "opened.txt").as_posix()}"\n',
                               encoding="utf-8", newline="\n")
    (stub / "open").chmod(0o755)
    app_dir = macapp.build(fake)
    info = plistlib.loads((app_dir / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleName"] == "Kinetrace" and info["CFBundleExecutable"] == "Kinetrace", info
    assert info["CFBundleShortVersionString"] == APP_VERSION and info["LSMinimumSystemVersion"] == "14.0"
    exe = app_dir / "Contents" / "MacOS" / "Kinetrace"
    assert subprocess.run([bash, "-n", str(exe)]).returncode == 0
    stamp = app_dir / "Contents" / "kinetrace-bundle-version"
    assert macapp.stale_reason(fake) == "", "a fresh app is not stale"
    # the folder moved (Mac report: the fallback path still named the old folder): the launcher only
    stamp.write_text(stamp.read_text(encoding="utf-8").replace(str(fake), str(tmp / "old name")),
                     encoding="utf-8")
    assert macapp.stale_reason(fake) == "moved"
    macapp._write_launcher(app_dir, fake)
    assert macapp.stale_reason(fake) == "" and str(fake) in exe.read_text(encoding="utf-8")
    stamp.write_text("0\n" + str(fake), encoding="utf-8")
    assert macapp.stale_reason(fake) == "version", "a changed launcher must rebuild the app"
    run_env = dict(env, PATH=str(stub) + os.pathsep + env.get("PATH", ""))

    def launch():
        return subprocess.run([bash, str(exe), "clip.mp4"], env=run_env, capture_output=True, text=True,
                              timeout=60)

    # an install still to do: it talks, so Terminal opens Kinetrace.command and nothing runs hidden
    r = launch()
    assert r.returncode == 0 and "Kinetrace.command" in (tmp / "opened.txt").read_text(), r
    assert not (fake / "logs" / "launcher.log").exists()
    (fake / ".venv" / "kinetrace-install.json").write_text("{}")
    r = launch()
    log = (fake / "logs" / "launcher.log").read_text()
    assert r.returncode == 0 and "RAN clip.mp4" in log, (r, log)
    # Help -> Check for Updates -> Restart now on a Mac: a NEW instance of the app, not Terminal
    import types  # noqa: E402
    from kinetrace import update  # noqa: E402
    started = []
    real_sys, real_popen = update.sys, update.subprocess.Popen
    update.sys = types.SimpleNamespace(platform="darwin")
    update.subprocess.Popen = lambda cmd, **kw: started.append(cmd)
    try:
        update.relaunch(fake)
        shutil.rmtree(app_dir)
        update.relaunch(fake)
    finally:
        update.sys, update.subprocess.Popen = real_sys, real_popen
    assert started[0] == ["open", "-n", str(app_dir)], started
    assert started[1] == ["open", str(fake / "Kinetrace.command")], started
    shutil.rmtree(tmp, ignore_errors=True)
    print("[5] Kinetrace.app: plist, rebuild when stale, launcher runs run.sh from its folder (output in "
          "logs/), Terminal for an install still to do: OK")
# ---- [6] key names on a Mac: the reference and the manual say what to press (P2-7) ----
from kinetrace.app import HOTKEYS_HTML  # noqa: E402
from kinetrace.widgets import native_keys  # noqa: E402
manual = (ROOT / "docs" / "MANUAL.md").read_text(encoding="utf-8")
for text in (HOTKEYS_HTML, manual):
    mac = native_keys(text, "darwin")
    assert "Ctrl" not in mac and "⌘+," in mac and "⌘+click" in mac, "Ctrl left in the Mac text"
    assert native_keys(text, "win32") == text and native_keys(text, "linux") == text
assert native_keys("Alternatively Alt+click", "darwin") == "Alternatively ⌥+click"
print("[6] on a Mac the key reference and the manual say Cmd / Option (Windows / Linux unchanged): OK")

# ---- [7] Settings in a menu-bar menu as Preferences (Qt puts it in Kinetrace's app menu on a Mac),
# and Ctrl+F / Cmd+F in the manual goes to its find box (Mac report 2026-10-08) ----
from PySide6.QtGui import QAction as _QAction  # noqa: E402
win2 = MainWindow()
win2.show()
edit_menu = next(m for m in win2.findChildren(QMenu) if m.title() == "&Edit")
edit_actions = edit_menu.actions()             # kept referenced (PySide6 harness pitfall)
assert win2.act_settings in edit_actions, "Settings must be in a menu-bar menu"
assert win2.act_settings.menuRole() == _QAction.PreferencesRole
win2._show_manual()
app.processEvents()
dlg = win2._manual_dlg
dlg.activateWindow()
dlg.view.setFocus()
app.processEvents()
QTest.keyClick(dlg.view, Qt.Key_F, Qt.ControlModifier)
app.processEvents()
assert dlg.find.hasFocus(), "Ctrl+F must put the cursor in the manual's find box"
dlg.close()
win2._dev_probe.wait(30000)
win2.close()
print("[7] Settings in Edit with the Preferences role; Ctrl+F reaches the manual's find box: OK")
print("verify_install_paths PASSED")
