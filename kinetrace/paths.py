"""Every place Kinetrace reads or writes, in words (Mac install audit P1 / P2-7).

`python -m kinetrace --paths` (also run.sh / run.bat --paths) prints them with
their sizes; Help -> Kinetrace's Folders shows the same list with buttons that
open each one. Inside the Kinetrace folder: the environment, models, logs,
recovery copies, skeletons, the tour's demo video and settings.ini -- deleting the folder removes
them. Outside it: only the per-user fallback folder (used when the Kinetrace
folder cannot be written), settings Kinetrace 0.4.1 and earlier kept in the
registry / a macOS plist / ~/.config, PyTorch's compile cache in the temp
folder, and the projects and exports the user saves wherever they choose
(never deleted by an uninstall). No Qt here. ASCII output (Windows consoles).
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"           # (R48) every model's folder; each module keeps its own name (tests patch it)


def path_key(path) -> str:
    """(R44, I291) THE "same file" key of a path: absolute, case-folded where the file system ignores case
    (normcase), so `sub/../a.mp4`, a backslashed and a forward-slashed spelling of one video are one."""
    return os.path.normcase(os.path.abspath(str(path)))


def replace_retry(src, dst, tries: int = 20, delay: float = 0.1) -> None:
    """(R43) `os.replace`, retried while Windows refuses because another program (an antivirus scan,
    OneDrive, a viewer) holds the file for a moment; the last refusal is raised. THE replace of every
    save, autosave, update and download."""
    import time
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(delay)


def write_atomic(path, text: str, encoding: str = "utf-8", newline: str | None = None) -> None:
    """(I290, R43) Write `text` to `path` whole or not at all: a `<name>.tmp` beside it, then
    `replace_retry`. `newline=None` writes the platform's line endings, as `Path.write_text` does."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding, newline=newline) as fh:
        fh.write(text)
    replace_retry(tmp, path)


def write_image(path, img) -> bool:
    """(X44, I222) `cv2.imwrite` for any folder name: OpenCV on Windows cannot open a path with a
    non-ASCII character ("José", CJK, Greek) and answers False with no reason. The picture is encoded
    in memory (the same encoder, the same bytes) and written with Python's own file API. The format
    follows the suffix; False = the picture could not be encoded."""
    import cv2
    ok, enc = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        return False
    Path(path).write_bytes(enc.tobytes())
    return True


def read_image(path, flags: int | None = None):
    """(X44) `cv2.imread` for any folder name (see `write_image`): the picture, or None when the file
    is missing, empty or not a picture OpenCV can read. `flags` default: cv2.IMREAD_COLOR."""
    import cv2
    import numpy as np
    try:
        buf = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    try:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR if flags is None else flags)
    except cv2.error:
        return None


@dataclass
class Place:
    what: str            # one short phrase
    path: Path
    inside: bool         # inside the Kinetrace folder (removed with it)
    note: str = ""


def _legacy_settings() -> Place | None:
    """Where QSettings("Kinetrace", "Kinetrace") kept the annotator's name up to 0.4.1."""
    if sys.platform == "darwin":
        p = Path.home() / "Library" / "Preferences" / "com.kinetrace.Kinetrace.plist"
        return Place("Old settings (Kinetrace 0.4.1 and earlier)", p, False, "safe to delete") if p.exists() else None
    if sys.platform == "win32":
        try:
            import winreg
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Kinetrace"))
        except OSError:
            return None
        return Place("Old settings (Kinetrace 0.4.1 and earlier)", Path(r"HKEY_CURRENT_USER\Software\Kinetrace"), False,
                     "a registry key; safe to delete")
    p = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "Kinetrace"
    return Place("Old settings (Kinetrace 0.4.1 and earlier)", p, False, "safe to delete") if p.exists() else None


def _qt_settings() -> Place | None:
    """Qt's own settings, which its file dialogs write whatever the program (the last folder you
    browsed, the dialog's view): one place for EVERY Qt program, so never deleted by an uninstall
    (Mac report 2026-10-08)."""
    note = "Qt's file-dialog history (the last folder you browsed); shared by every Qt program"
    if sys.platform == "darwin":
        p = Path.home() / "Library" / "Preferences" / "com.qtproject.plist"
        return Place("Qt's settings", p, False, note) if p.exists() else None
    if sys.platform == "win32":
        try:
            import winreg
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\QtProject"))
        except OSError:
            return None
        return Place("Qt's settings", Path(r"HKEY_CURRENT_USER\Software\QtProject"), False,
                     note + "; a registry key")
    p = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "QtProject.conf"
    return Place("Qt's settings", p, False, note) if p.exists() else None


def _start_menu() -> Place | None:
    """(Windows) the Start menu entry the user added for this folder (Help -> Add Kinetrace to the Start
    Menu, winapp.py): the one thing of Kinetrace outside its folder."""
    if sys.platform != "win32":
        return None
    try:
        from kinetrace import winapp
        if not winapp.start_menu_points_to(ROOT):
            return None
        return Place("Start menu entry (you added it)", winapp.start_menu_link(), False,
                     "remove it with Help -> Remove Kinetrace from the Start Menu before deleting the folder")
    except Exception:       # noqa: BLE001 - the shell could not be asked: nothing to list
        return None


def places() -> list[Place]:
    from kinetrace import crashlog, demo, recovery
    rec, rec_fallback = recovery.folder()
    out = [
        Place("Kinetrace folder (the app)", ROOT, True, "deleting it uninstalls Kinetrace"),
        Place("Environment (Python + packages)", ROOT / ".venv", True),
        Place("Models", ROOT / "models", True, "downloaded once; kept by updates"),
        Place("Logs (errors; the last installs)", crashlog.folder(), crashlog.folder().is_relative_to(ROOT)),
        Place("Unsaved-work recovery copies", rec, not rec_fallback,
              "the Kinetrace folder could not be written" if rec_fallback else ""),
        Place("Settings (your name for notes)", recovery.settings_path(),
              recovery.settings_path().is_relative_to(ROOT)),
        Place("Saved skeletons", ROOT / "skeletons", True),
        Place("The tour's demo video", demo.demo_folder(), demo.demo_folder().is_relative_to(ROOT),
              "made by Help -> Take the Interface Tour; safe to delete: it is made again when needed"),
    ]
    fallback = recovery._user_dir().parent
    if fallback.exists() and not any(p.path.is_relative_to(fallback) for p in out):
        out.append(Place("Per-user fallback folder", fallback, False,
                         "used only when the Kinetrace folder could not be written"))
    menu = _start_menu()
    if menu is not None:
        out.append(menu)
    legacy = _legacy_settings()
    if legacy is not None:
        out.append(legacy)
    qt = _qt_settings()
    if qt is not None:
        out.append(qt)
    try:
        import getpass
        inductor = Path(tempfile.gettempdir()) / f"torchinductor_{getpass.getuser()}"
        if inductor.exists():
            out.append(Place("PyTorch's compile cache", inductor, False, "made by PyTorch; the system clears it"))
    except Exception:       # noqa: BLE001 - no user name: nothing to list
        pass
    return out


def size_bytes(path: Path) -> int | None:
    """Bytes under `path` (links not followed); None when it does not exist."""
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def human(n: int | None) -> str:
    if n is None:
        return "not created yet"
    for unit, k in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= k:
            return f"{n / k:.1f} {unit}"
    return f"{n} bytes"


def describe(with_sizes: bool = True) -> str:
    lines = ["Where Kinetrace keeps things", ""]
    ps = places()
    for inside in (True, False):
        lines.append("Inside the Kinetrace folder (deleting the folder removes these):" if inside else
                     "Outside the Kinetrace folder:")
        for p in (q for q in ps if q.inside == inside):
            size = "" if not with_sizes or p.what.startswith("Kinetrace folder") or "registry" in p.note \
                else f"  [{human(size_bytes(p.path))}]"
            note = f"  ({p.note})" if p.note else ""
            lines.append(f"  {p.what}: {p.path}{size}{note}")
        if not inside and not any(not q.inside for q in ps):
            lines.append("  nothing")
        lines.append("")
    lines.append("Your projects (name.kinetrace folders), exports and calibration files are saved where you")
    lines.append("choose them. They are never deleted by an update or by uninstalling.")
    return "\n".join(lines)


def cli() -> int:
    """`python -m kinetrace --paths`."""
    text = describe()
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    return 0
