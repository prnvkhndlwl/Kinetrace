"""Kinetrace.app for macOS (Mac install audit P2-3 / P2-5 / P2-6): a double-clickable app
in the Kinetrace folder, so nobody has to open a Terminal after the first install.

run.sh writes it after a finished install (and again when this file's BUNDLE_VERSION
changes); it is made ON the user's Mac, so it carries no download quarantine and opens
without the "unidentified developer" prompt the downloaded Kinetrace.command gets. No
Apple Developer ID is involved.

    Kinetrace.app/Contents/Info.plist            name, icon, minimum macOS
    Kinetrace.app/Contents/MacOS/Kinetrace       a bash launcher: runs run.sh from the folder
                                                 the app sits in, output to logs/launcher.log;
                                                 when an install or repair step is due (the
                                                 install marker is gone, .venv does not start)
                                                 it opens Kinetrace.command in Terminal instead,
                                                 because that step talks and takes minutes
    Kinetrace.app/Contents/Resources/Kinetrace.icns   appicon.render, through iconutil

The window's name in the menu bar, the Dock and Cmd-Tab comes from run.sh starting
Python through .venv/bin/Kinetrace (a link to python): an unbundled process is named by
its file name, and was "Python" before.

    python -m kinetrace.macapp [--if-stale]      (run.sh does this on macOS)
"""
from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "Kinetrace.app"
BUNDLE_VERSION = "1"          # bump when the launcher / plist below change: run.sh rebuilds the app
ICON_SIZES = (16, 32, 128, 256, 512)

LAUNCHER = """#!/bin/bash
# Kinetrace.app's launcher, written by kinetrace/macapp.py: starts Kinetrace from the folder
# this app sits in, without a Terminal window (output: logs/launcher.log).
DIR="$(cd "$(dirname "$0")/../../.." && pwd -P)"
[ -f "$DIR/run.sh" ] || DIR={baked}        # the app was moved out of its folder
cd "$DIR" || exit 1
if [ ! -f .venv/kinetrace-install.json ] || ! .venv/bin/python -c pass >/dev/null 2>&1; then
    # an install or repair step is due: it talks and takes minutes, so it runs in Terminal
    exec open -a Terminal "$DIR/Kinetrace.command"
fi
mkdir -p logs
exec /bin/bash ./run.sh "$@" >> logs/launcher.log 2>&1
"""


def info_plist(version: str) -> dict:
    return {
        "CFBundleName": "Kinetrace",
        "CFBundleDisplayName": "Kinetrace",
        "CFBundleIdentifier": "io.github.prnvkhndlwl.kinetrace",
        "CFBundleExecutable": "Kinetrace",
        "CFBundleIconFile": "Kinetrace",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
    }


def _stamp_text(root: Path) -> str:
    return f"{BUNDLE_VERSION}\n{root}"


def stale_reason(root: Path = ROOT) -> str:
    """'' (up to date), 'moved' (only the folder it bakes in as a fallback changed: the launcher is
    rewritten in place, Mac report 2026-10-08) or 'version' (rebuild)."""
    try:
        version, _, baked = (root / "Kinetrace.app" / "Contents" / "kinetrace-bundle-version") \
            .read_text(encoding="utf-8").partition("\n")
    except OSError:
        return "version"
    if version.strip() != BUNDLE_VERSION:
        return "version"
    return "" if baked.strip() == str(root) else "moved"


def is_stale(root: Path = ROOT) -> bool:
    return stale_reason(root) != ""


def _write_launcher(app: Path, root: Path) -> None:
    exe = app / "Contents" / "MacOS" / "Kinetrace"
    tmp = exe.with_name("Kinetrace.new")
    tmp.write_text(LAUNCHER.format(baked=shlex.quote(str(root))), encoding="utf-8", newline="\n")
    tmp.chmod(0o755)
    os.replace(tmp, exe)                      # in place: the app may be the one starting right now
    (app / "Contents" / "kinetrace-bundle-version").write_text(_stamp_text(root), encoding="utf-8")


def _icns(dest: Path) -> bool:
    """appicon at the sizes macOS wants, packed by iconutil (part of every macOS)."""
    if shutil.which("iconutil") is None:
        return False
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication([])     # noqa: F841 - QImage needs one
    from kinetrace import appicon
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "Kinetrace.iconset"
        iconset.mkdir()
        for s in ICON_SIZES:
            appicon.render(s).save(str(iconset / f"icon_{s}x{s}.png"))
            appicon.render(2 * s).save(str(iconset / f"icon_{s}x{s}@2x.png"))
        r = subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)],
                           capture_output=True, text=True)
        return r.returncode == 0 and dest.exists()


def build(root: Path = ROOT) -> Path:
    """(Re)write root/Kinetrace.app. Returns its path."""
    from kinetrace import APP_VERSION
    app = root / "Kinetrace.app"
    tmp = root / "Kinetrace.app.new"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "Contents" / "MacOS").mkdir(parents=True)
    (tmp / "Contents" / "Resources").mkdir()
    with open(tmp / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump(info_plist(APP_VERSION), fh)
    _write_launcher(tmp, root)
    _icns(tmp / "Contents" / "Resources" / "Kinetrace.icns")      # no icon is not worth failing over
    shutil.rmtree(app, ignore_errors=True)
    tmp.rename(app)
    if shutil.which("codesign"):
        # ad-hoc signature: no Developer ID; keeps Apple Silicon's "damaged app" check quiet
        subprocess.run(["codesign", "--force", "--sign", "-", str(app)], capture_output=True)
    return app


def main(argv: list[str]) -> int:
    if sys.platform != "darwin" and "--any-os" not in argv:
        print("Kinetrace.app is for macOS.")
        return 0
    reason = stale_reason(ROOT)
    if "--if-stale" in argv and reason == "":
        return 0
    if "--if-stale" in argv and reason == "moved":
        _write_launcher(APP, ROOT)            # the folder moved: only the baked fallback path changes
        return 0
    app = build()
    print(f"Made {app.name} in this folder: double-click it to start Kinetrace (no Terminal needed); "
          "drag it to the Dock to keep it there.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
