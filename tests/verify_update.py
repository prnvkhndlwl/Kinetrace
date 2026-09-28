"""Help -> About / Check for Updates, the updater and the update scripts (G36,
G37, I142, M3). No internet: a local HTTP server stands in for GitHub's API,
the "installs" are temporary folders, the git case a temporary repository.

  [1] versions compare as numbers (1.10 > 1.9)
  [2] the release check: newer / not newer, 404 and an unreachable server said
      in words
  [3] a ZIP install: only changed files written, each user's folders never
      touched (even when the archive carries them), files the old release had
      and the new one dropped removed, the user's own files kept, the install
      marker dropped only when requirements.txt / install.py changed, unsafe or
      foreign archives refused with nothing changed; download + apply end to end
  [4] a git checkout: fast-forwarded to the release tag; a local change to a
      tracked file refuses, naming why
  [5] the launchers / scripts / installer / release workflow / licence
  [6] the app: Help entries, the About box, the update dialog driven by real
      clicks (available -> Update now -> installed -> Restart now), up to date,
      unreachable, busy, and nothing looked up before the dialog is shown

Run: .venv\\Scripts\\python.exe tests\\verify_update.py
"""
import http.server
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kinetrace import APP_VERSION, update  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="kt_update_"))


def check(cond, what):
    if not cond:
        raise AssertionError(what)


# ---------------------------------------------------------------- [1] versions
print("[1] versions")
check(update.parse_version("v1.10.2") == (1, 10, 2), "parse v1.10.2")
check(update.parse_version("2") == (2, 0, 0), "parse 2")
check(update.is_newer("1.10.0", "1.9.9") and not update.is_newer("1.2.0", "1.2.0"), "numeric compare")
check(update.is_newer("v1.2.1", "1.2.0") and not update.is_newer("1.1.9", "1.2.0"), "tag compare")
print("  1.10 > 1.9, equal is not newer OK")


# ---------------------------------------------------------- a fake GitHub
def make_release_zip(files: dict, top="prnvkhndlwl-Kinetrace-abc1234") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(top + "/", "")
        for rel, data in files.items():
            zf.writestr(f"{top}/{rel}", data)
    return buf.getvalue()


V2 = {"kinetrace/__init__.py": 'APP_VERSION = "1.1.0"\n', "install.py": "# installer v1\n",
      "requirements.txt": "numpy\nnewpackage\n", "run.bat": "@echo off\r\nrem same\r\n",
      "kinetrace/new_module.py": "x = 2\n", "docs/MANUAL.md": "# manual v2\n",
      "models/evil.bin": "must never be written", ".venv/kinetrace-install.json": "{}",
      "recovery/x.kinetrace": "no", "skeletons/lizard.json": "no"}
SERVED = {"latest": None, "zip": make_release_zip(V2)}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/api/releases/latest" and SERVED["latest"] is not None:
            body = json.dumps(SERVED["latest"]).encode()
            ctype = "application/json"
        elif self.path == "/zip":
            body, ctype = SERVED["zip"], "application/zip"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"
os.environ["KINETRACE_UPDATE_API"] = BASE + "/api"
SERVED["latest"] = {"tag_name": "v1.1.0", "name": "Kinetrace 1.1.0", "body": "* Faster tracking\n* A fix",
                    "zipball_url": BASE + "/zip", "html_url": BASE + "/page", "published_at": "2026-10-01T10:00:00Z"}

# ---------------------------------------------------------------- [2] check
print("[2] the release check")
rel, newer = update.check("1.0.0")
check(newer and rel.version == "1.1.0" and rel.tag == "v1.1.0" and rel.published == "2026-10-01", rel)
check("Faster tracking" in rel.notes and rel.zip_url.endswith("/zip"), rel)
check(update.check("1.1.0")[1] is False and update.check("1.2.0")[1] is False, "not newer")
SERVED["latest"] = None
try:
    update.check("1.0.0")
    raise AssertionError("a 404 must raise")
except update.UpdateError as e:
    check("no published version" in str(e) and "not public" in str(e), e)
SERVED["latest"] = {"tag_name": "v1.1.0", "name": "Kinetrace 1.1.0", "body": "* Faster tracking\n* A fix",
                    "zipball_url": BASE + "/zip", "html_url": BASE + "/page", "published_at": "2026-10-01T10:00:00Z"}
s = socket.socket()
s.bind(("127.0.0.1", 0))
dead = s.getsockname()[1]
s.close()
os.environ["KINETRACE_UPDATE_API"] = f"http://127.0.0.1:{dead}/api"
try:
    update.check("1.0.0", timeout=3)
    raise AssertionError("an unreachable server must raise")
except update.UpdateError as e:
    check("Could not reach GitHub" in str(e) and "internet connection" in str(e), e)
os.environ["KINETRACE_UPDATE_API"] = BASE + "/api"
print("  newer / not newer, 404 and unreachable said in words OK")


# ---------------------------------------------------------------- [3] ZIP install
def make_install(name: str, requirements="numpy\n") -> Path:
    r = TMP / name
    files = {"kinetrace/__init__.py": 'APP_VERSION = "1.0.0"\n', "install.py": "# installer v1\n",
             "requirements.txt": requirements, "run.bat": "@echo off\r\nrem same\r\n",
             "kinetrace/old_module.py": "x = 1\n", "docs/MANUAL.md": "# manual v1\n",
             "my_notes.txt": "the user's own file", ".venv/kinetrace-install.json": '{"torch": "x"}',
             "models/evil.bin": "the user's model", "recovery/x.kinetrace": "unsaved work",
             "skeletons/lizard.json": "the user's skeleton"}
    for rel, data in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode())
    (r / update.MANIFEST).write_text(json.dumps({"version": "1.0.0", "files": [
        "kinetrace/__init__.py", "install.py", "requirements.txt", "run.bat", "kinetrace/old_module.py",
        "docs/MANUAL.md"]}), encoding="utf-8")
    return r


def read(r: Path, rel: str) -> str:
    return (r / rel).read_text(encoding="utf-8")


print("[3] a ZIP install")
inst = make_install("zip_a")
run_bat_mtime = (inst / "run.bat").stat().st_mtime_ns
zp = TMP / "v2.zip"
zp.write_bytes(SERVED["zip"])
res = update.apply_zip(inst, zp, "1.1.0", "v1.1.0")
check(read(inst, "kinetrace/__init__.py") == 'APP_VERSION = "1.1.0"\n', "the version file is replaced")
check(read(inst, "kinetrace/new_module.py") == "x = 2\n" and read(inst, "docs/MANUAL.md") == "# manual v2\n", "new files")
check(not (inst / "kinetrace/old_module.py").exists() and res.removed == ["kinetrace/old_module.py"],
      f"a file the new release dropped is removed: {res.removed}")
check(read(inst, "my_notes.txt") == "the user's own file", "the user's own file is kept")
check(read(inst, "models/evil.bin") == "the user's model" and read(inst, "recovery/x.kinetrace") == "unsaved work"
      and read(inst, "skeletons/lizard.json") == "the user's skeleton", "each user's folders are never written")
check((inst / "run.bat").stat().st_mtime_ns == run_bat_mtime and "run.bat" not in res.changed,
      "an unchanged file is not rewritten (run.bat)")
check(res.reinstall and not (inst / ".venv/kinetrace-install.json").exists(),
      "requirements.txt changed -> the install marker is dropped so the launcher reinstalls")
man = json.loads(read(inst, update.MANIFEST))
check(man["version"] == "1.1.0" and "kinetrace/new_module.py" in man["files"], man)
check(not (inst / update.STAGING).exists(), "the staging folder is cleaned up")
check("1.1.0 is installed" in res.sentence() and "installs what the new version needs" in res.sentence(), res.sentence())
res2 = update.apply_zip(inst, zp, "1.1.0", "v1.1.0")
check(res2.changed == [] and res2.removed == [] and not res2.reinstall, "applying the same release again changes nothing")
print(f"  {len(res.changed)} changed, 1 removed, user folders untouched, marker dropped OK")

inst_b = make_install("zip_b", requirements="numpy\nnewpackage\n")      # same requirements as v2
res_b = update.apply_zip(inst_b, zp, "1.1.0")
check(not res_b.reinstall and (inst_b / ".venv/kinetrace-install.json").exists(),
      "requirements and installer unchanged -> the marker stays (no reinstall)")
print("  unchanged requirements keep the install marker OK")

for label, files, needle in (
        ("unsafe path", {**V2, "../escape.txt": "x"}, "unsafe path"),
        ("not Kinetrace", {"README.md": "something else"}, "not a Kinetrace release")):
    inst_c = make_install("zip_" + label.replace(" ", "_"))
    before = {p: p.read_bytes() for p in inst_c.rglob("*") if p.is_file()}
    bad = TMP / (label.replace(" ", "_") + ".zip")
    bad.write_bytes(make_release_zip(files))
    try:
        update.apply_zip(inst_c, bad, "1.1.0")
        raise AssertionError(f"{label}: must be refused")
    except update.UpdateError as e:
        check(needle in str(e) and "nothing was changed" in str(e), e)
    after = {p: p.read_bytes() for p in inst_c.rglob("*") if p.is_file()}
    check(before == after, f"{label}: the install changed")
damaged = TMP / "damaged.zip"
damaged.write_bytes(b"PK\x03\x04 not really a zip")
try:
    update.apply_zip(make_install("zip_damaged"), damaged, "1.1.0")
    raise AssertionError("a damaged download must be refused")
except update.UpdateError as e:
    check("damaged" in str(e), e)
print("  unsafe / foreign / damaged archives refused, nothing changed OK")

inst_d = make_install("zip_download")
seen = []
res_d = update.apply(update.check("1.0.0")[0], inst_d, lambda d, t: seen.append((d, t)))
check(res_d.how == "zip" and read(inst_d, "kinetrace/__init__.py").endswith('"1.1.0"\n'), "download + apply")
check(seen and seen[-1][0] == seen[-1][1] == len(SERVED["zip"]), f"download progress: {seen[-1:]}")
check(not (inst_d / update.STAGING).exists(), "the download is removed")
print("  download + apply end to end, with progress OK")


# ---------------------------------------------------------------- [4] git checkout
print("[4] a git checkout")
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main",
       "-c", "core.autocrlf=false"]


def git(cwd, *args):
    r = subprocess.run(GIT + list(args), cwd=cwd, capture_output=True, text=True)
    check(r.returncode == 0, f"git {args}: {r.stderr}")
    return r.stdout.strip()


if shutil.which("git") is None:
    print("  git not installed here - skipped")
else:
    origin = TMP / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    (origin / "kinetrace").mkdir()
    (origin / "kinetrace/__init__.py").write_text('APP_VERSION = "1.0.0"\n')
    (origin / "install.py").write_text("# v1\n")
    (origin / "requirements.txt").write_text("numpy\n")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "v1")
    clone = TMP / "clone"
    git(TMP, "clone", "-q", str(origin), str(clone))
    (clone / ".venv").mkdir()
    (clone / ".venv/kinetrace-install.json").write_text("{}")
    (origin / "kinetrace/__init__.py").write_text('APP_VERSION = "1.1.0"\n')
    (origin / "requirements.txt").write_text("numpy\nnewpackage\n")
    git(origin, "commit", "-qam", "v2")
    git(origin, "tag", "v1.1.0")
    head2 = git(origin, "rev-parse", "HEAD")
    (origin / "later.txt").write_text("unreleased work")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "work in progress after the release")
    check(update.install_kind(clone) == "git" and update.install_kind(TMP / "zip_a") == "zip", "install kind")
    (clone / "kinetrace/__init__.py").write_text('APP_VERSION = "1.0.0"  # edited\n')
    why = update.git_blocker(clone)
    check(why and "were changed in this folder" in why, why)
    try:
        update.apply_git(clone, "v1.1.0", "1.1.0")
        raise AssertionError("a local change must refuse")
    except update.UpdateError as e:
        check("git status" in str(e), e)
    git(clone, "checkout", "--", "kinetrace/__init__.py")
    (clone / "untracked_notes.txt").write_text("untracked files do not block")
    check(update.git_blocker(clone) is None, "untracked files do not block")
    r = update.apply_git(clone, "v1.1.0", "1.1.0")
    check(git(clone, "rev-parse", "HEAD") == head2, "moved exactly to the release tag, not past it")
    check(not (clone / "later.txt").exists(), "unreleased work on main is not pulled in")
    check(r.reinstall and not (clone / ".venv/kinetrace-install.json").exists(), "requirements changed -> marker dropped")
    print("  fast-forwarded to the tag (not past it), local change refused, marker dropped OK")


# ---------------------------------------------------------------- [5] scripts
print("[5] launchers, scripts, installer, release workflow, licence")
bat = (ROOT / "run.bat").read_text(encoding="utf-8")
launch = [ln for ln in bat.splitlines() if "-m kinetrace %*" in ln]
check(len(launch) == 1 and launch[0].rstrip().endswith("|| pause & exit /b"),
      f"run.bat launches on ONE line that ends the script (I142): {launch}")
ubat = (ROOT / "update.bat").read_text(encoding="utf-8")
uline = [ln for ln in ubat.splitlines() if "-m kinetrace.update" in ln]
check(len(uline) == 1 and uline[0].rstrip().endswith("& exit /b"), f"update.bat: one line: {uline}")
for f in ("update.sh", "Update.command"):
    lines = [ln for ln in (ROOT / f).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    check(lines[-1].startswith("exec "), f"{f} ends in exec")
attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
check("*.bat text eol=crlf" in attrs and "*.sh text eol=lf" in attrs and "*.command text eol=lf" in attrs,
      "line endings of the scripts are pinned")
src = (ROOT / "install.py").read_text(encoding="utf-8")
fast = src[src.index("if ok and torch_matches():"):]
fast = fast[:fast.index("return 0")]
check('pip("-r"' in fast and "fetch_alltracker()" in fast,
      "install.py's fast path installs what requirements.txt adds (I142)")
wf = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
check('paths: ["kinetrace/__init__.py"]' in wf and "gh release create" in wf and "--generate-notes" in wf,
      "the release workflow")
snippet = re.search(r'python3 -c "(.*)"\)', wf).group(1).replace('\\"', '"')
out = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, capture_output=True, text=True)
check(out.stdout.strip() == APP_VERSION, f"the workflow reads APP_VERSION: {out.stdout!r} {out.stderr[-300:]}")
lic = (ROOT / "LICENSE.md").read_text(encoding="utf-8")
check(lic.startswith("Required Notice: Copyright 2026 biomechLab@CMC") and "# PolyForm Noncommercial License 1.0.0" in lic
      and "educational institution, public research organization" in lic, "LICENSE.md")
gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
check(update.MANIFEST in gi and "/update/" in gi, ".gitignore keeps the manifest and the staging folder out")
print("  one-line launch, exec scripts, installer fast path, workflow reads the version, licence OK")


# ---------------------------------------------------------------- [6] the app
print("[6] the app")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
from kinetrace import updatedialog  # noqa: E402
from kinetrace.app import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.show()


def pump(cond=lambda: False, timeout=10.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return cond()


check(win.act_about.text().replace("&", "").startswith("About Kinetrace") and win.act_about.toolTip(), "About entry")
check("Updates" in win.act_updates.text() and "Nothing is checked unless you ask" in win.act_updates.toolTip(),
      "Check for Updates entry")
html = updatedialog.about_html()
check("biomechLab@CMC" in html and APP_VERSION in html and "PolyForm Noncommercial" in html, "About text")
seen_about = {}


def _read_about():
    w = QApplication.activeModalWidget()
    seen_about["text"] = w.about_text.text() if w is not None and hasattr(w, "about_text") else None
    if w is not None:
        w.reject()


QTimer.singleShot(200, _read_about)
win.act_about.trigger()                                  # blocks until closed
check(seen_about.get("text") and "Developed at <b>biomechLab@CMC</b>" in seen_about["text"], seen_about)
print("  Help -> About Kinetrace says 'Developed at biomechLab@CMC', version, licence OK")

# G41: the app icon, drawn in code: a viewfinder around three tracked points (air / water / land)
from kinetrace import appicon  # noqa: E402
import numpy as np  # noqa: E402


def _px(img):
    img = img.convertToFormat(img.Format.Format_ARGB32)
    return np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4].copy()


a, b = _px(appicon.render(256)), _px(appicon.render(256))
check(np.array_equal(a, b), "the icon draws the same pixels every time")
px = a.reshape(256, 256, 4)
rgb = px[..., :3][..., ::-1].astype(int)                                # BGRA in memory -> RGB
for nm, col in (("air", appicon.AIR), ("water", appicon.WATER), ("land", appicon.LAND)):
    n = int((np.abs(rgb - list(col)).sum(axis=2) < 50).sum())
    check(n > 120, f"the {nm} trail is drawn in its landmark colour ({n} px)")
white = int(((rgb > 235).all(axis=2)).sum())
check(white > 60, f"the three heads have bright cores ({white} px)")
check(px[0, 0, 3] == 0 and px[128, 128, 3] == 255, "the rounded corners transparent, the tile opaque")
check(not np.array_equal(_px(appicon.render(32)), _px(appicon.render(32, simple=False))),
      "small sizes get the simpler drawing")
ic = appicon.icon()
check(all(ic.pixmap(s, s).width() == s for s in (16, 32, 64, 256)), "every size is in the icon")
check(all((appicon.CACHE / f"appicon_v{appicon.ICON_VERSION}_{s}.png").exists() for s in appicon.SIZES),
      "rendered once into the theme cache")
seen_about.clear()


def _read_mark():
    w = QApplication.activeModalWidget()
    seen_about["mark"] = (w.about_mark.pixmap().width() if w is not None and hasattr(w, "about_mark") else 0)
    if w is not None:
        w.reject()


QTimer.singleShot(200, _read_mark)
win.act_about.trigger()
check(seen_about.get("mark") == 96, f"the About box shows the icon: {seen_about}")
print("  the app icon: three landmark-coloured trails, the same every draw, simpler when small, cached, in the About box (G41) OK")

# nothing goes to the network before the dialog is on screen
calls = []
real_check = update.check
update.check = lambda *a, **k: (calls.append(1), real_check(*a, **k))[1]
_exec = updatedialog.UpdateDialog.exec
updatedialog.UpdateDialog.exec = lambda self: 0
win.act_updates.trigger()
pump(timeout=0.3)
updatedialog.UpdateDialog.exec = _exec
check(calls == [], "the check must wait until the dialog is shown")

inst_g = make_install("gui")
restarted = []
dlg = updatedialog.UpdateDialog(win, root=inst_g, local="1.0.0", busy=win._update_busy,
                                restart=lambda: restarted.append(1))
dlg.show()
check(pump(lambda: dlg.state == "available"), f"available: {dlg.state} {dlg.body.toPlainText()[:200]}")
check(calls, "the check ran once shown")
check("1.1.0 is available" in dlg.title.text() and "Faster tracking" in dlg.body.toPlainText()
      and "projects, models" in dlg.body.toPlainText(), dlg.body.toPlainText())
check(dlg.buttons["Update now"].isEnabled(), "Update now enabled for a ZIP install")
QTest.mouseClick(dlg.buttons["Update now"], Qt.LeftButton)
check(pump(lambda: dlg.state == "done"), f"installed: {dlg.state} {dlg.body.toPlainText()[:300]}")
check(read(inst_g, "kinetrace/__init__.py").endswith('"1.1.0"\n') and "Restart" in dlg.body.toPlainText(),
      dlg.body.toPlainText())
QTest.mouseClick(dlg.buttons["Restart now"], Qt.LeftButton)
check(pump(lambda: restarted == [1]) and not dlg.isVisible(), "Restart now closes and restarts")
dlg.deleteLater()
print("  update dialog: available -> Update now -> installed -> Restart now (real clicks) OK")

dlg = updatedialog.UpdateDialog(win, root=make_install("gui_busy"), local="1.0.0",
                                busy=lambda: "Stop tracking first (X), then press Update now.")
dlg.show()
pump(lambda: dlg.state == "available")
QTest.mouseClick(dlg.buttons["Update now"], Qt.LeftButton)
pump(timeout=0.3)
check(dlg.state == "available" and "Stop tracking first" in dlg.body.toPlainText(), "busy: nothing installed, said why")
dlg.reject()

dlg = updatedialog.UpdateDialog(win, root=inst_g, local="1.1.0")
dlg.show()
check(pump(lambda: dlg.state == "uptodate") and "newest version" in dlg.title.text(), dlg.title.text())
dlg.reject()

os.environ["KINETRACE_UPDATE_API"] = f"http://127.0.0.1:{dead}/api"
dlg = updatedialog.UpdateDialog(win, root=inst_g, local="1.0.0")
dlg.show()
check(pump(lambda: dlg.state == "error", 20) and "Could not reach GitHub" in dlg.body.toPlainText(), dlg.state)
check("Try again" in dlg.buttons, "an error offers Try again")
dlg.reject()
os.environ["KINETRACE_UPDATE_API"] = BASE + "/api"
print("  busy / up to date / unreachable said in the dialog OK")

relaunched = []
update.relaunch = lambda *a, **k: relaunched.append(1)
check(win._update_busy() is None, "nothing blocks an update when idle")
win.close = lambda: False                                # the user pressed Cancel on Save / Discard
win._restart_after_update()
check(relaunched == [], "Cancel on close keeps this Kinetrace and starts no other")
del win.close
win._dev_probe.wait(20000)
win.close()
srv.shutdown()
shutil.rmtree(TMP, ignore_errors=True)
print("verify_update PASSED")
