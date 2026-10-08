"""Every place Kinetrace reads or writes, in words (Mac install audit P1 / P2-7).

`python -m kinetrace --paths` (also run.sh / run.bat --paths) prints them with
their sizes; Help -> Kinetrace's Folders shows the same list with buttons that
open each one. Inside the Kinetrace folder: the environment, models, logs,
recovery copies, skeletons and settings.ini -- deleting the folder removes
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


def places() -> list[Place]:
    from kinetrace import crashlog, recovery
    rec, rec_fallback = recovery.folder()
    out = [
        Place("Kinetrace folder (the app)", ROOT, True, "deleting it uninstalls Kinetrace"),
        Place("Environment (Python + packages)", ROOT / ".venv", True),
        Place("Models", ROOT / "models", True, "downloaded once; kept by updates"),
        Place("Error log", crashlog.folder(), crashlog.folder().is_relative_to(ROOT)),
        Place("Unsaved-work recovery copies", rec, not rec_fallback,
              "the Kinetrace folder could not be written" if rec_fallback else ""),
        Place("Settings (your name for notes)", recovery.settings_path(),
              recovery.settings_path().is_relative_to(ROOT)),
        Place("Saved skeletons", ROOT / "skeletons", True),
    ]
    fallback = recovery._user_dir().parent
    if fallback.exists() and not any(p.path.is_relative_to(fallback) for p in out):
        out.append(Place("Per-user fallback folder", fallback, False,
                         "used only when the Kinetrace folder could not be written"))
    legacy = _legacy_settings()
    if legacy is not None:
        out.append(legacy)
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
