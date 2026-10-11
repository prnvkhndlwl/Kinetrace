"""What Kinetrace.lnk runs (kinetrace/winapp.py writes the shortcut): start Kinetrace from this folder with
no console window -- or, when an install or repair step is due, run.bat in a console of its own, because
that step talks and takes minutes (the Mac app opens Terminal for the same reason).

Run by the folder's own pythonw with -I, so the standard library only: no package of the environment may be
needed here (this is what starts when the environment is broken). It never stops in silence: what it cannot
do is said in a message box.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_WINDOW = 0x08000000                  # CREATE_NO_WINDOW
DETACHED = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
NEW_CONSOLE = 0x00000010                # CREATE_NEW_CONSOLE


def due(root: Path = ROOT) -> str:
    """'' = the app can start; else why run.bat has to run first (it does the step and starts the app)."""
    root = Path(root)
    if not (root / ".venv" / "kinetrace-install.json").exists():
        return "install"                # first install, or an update asked for an install step
    if not (root / "models" / "alltracker" / "nets" / "alltracker.py").exists():
        return "models"
    py = root / ".venv" / "Scripts" / "python.exe"
    if not (root / ".venv" / "Scripts" / "pythonw.exe").exists() or not py.exists():
        return "repair"
    try:
        ok = subprocess.run([str(py), "-c", "pass"], cwd=str(root), creationflags=NO_WINDOW, timeout=120,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    return "" if ok else "repair"       # a moved folder (pyvenv.cfg names the old place): run.bat relinks it


def tell(text: str) -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "Kinetrace", 0x30 | 0x10000)   # warning, foreground
    except Exception:       # noqa: BLE001 - nothing else can be shown
        pass


def run_bat(root: Path, argv: list[str]) -> None:
    # `cmd /c ""path" args"` takes the quoted path literally (& ^ in a folder name, I156)
    args = "".join(f' "{a}"' for a in argv)
    subprocess.Popen(f'cmd /c ""{root / "run.bat"}"{args}"', cwd=str(root), creationflags=NEW_CONSOLE)


def main(argv: list[str], root: Path = ROOT) -> int:
    root = Path(root)
    if not (root / "run.bat").exists():
        tell(f"Kinetrace cannot find its folder:\n{root}\n\nThe shortcut was moved away from it, or the folder "
             "was moved or deleted. Open the Kinetrace folder and double-click run.bat: it makes a new shortcut.")
        return 1
    try:
        if due(root):
            run_bat(root, argv)
            return 0
        subprocess.Popen([str(root / ".venv" / "Scripts" / "pythonw.exe"), "-m", "kinetrace", *argv],
                         cwd=str(root), creationflags=DETACHED, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        tell(f"Kinetrace could not start ({e}).\n\nDouble-click run.bat in the Kinetrace folder:\n{root}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
