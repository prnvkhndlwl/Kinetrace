"""After an install: what the user reads last in the launcher's window, and a one-time pop-up the first
time the app opens (owner 2026-10-09: "users often don't read the terminal carefully").

The launcher (run.bat / run.sh) runs `python -m kinetrace.welcome <how>` right after an install that happened
in THAT run (never for `--check`): it prints `finished_text` and leaves the flag file `FLAG` in .venv.
The app (`app.main`, not `MainWindow`, so tests never see it) shows `popup_html` once when the flag is
there and it could be removed (a read-only folder must not show it at every start). No Qt here: install.py
imports it before the packages are known to work.

<how> = "mac-app" (macOS: Kinetrace.app is opened on its own; the Terminal can be closed), "console"
(Windows / Linux: the app runs in that window; closing it closes the app).
"""
from __future__ import annotations

import sys
from pathlib import Path

FLAG_NAME = "kinetrace-welcome"


def flag_path(prefix: str | None = None) -> Path:
    """The flag in the folder's .venv (`sys.prefix` of its interpreter)."""
    return Path(prefix or sys.prefix) / FLAG_NAME


def set_pending(prefix: str | None = None) -> None:
    try:
        flag_path(prefix).write_text("shown once by the app after an install\n", encoding="utf-8")
    except OSError:
        pass                                    # no pop-up is better than a failed install


def take_pending(prefix: str | None = None) -> bool:
    """True once after an install: the flag existed and was removed now."""
    p = flag_path(prefix)
    if not p.exists():
        return False
    try:
        p.unlink()
    except OSError:
        return False                            # cannot be removed: never show it at every start
    return True


def _start_next_time(system: str) -> str:
    if system == "Darwin":
        return "Kinetrace.app in this folder (drag it to the Dock to keep it there)"
    if system == "Windows":
        return "double-click run.bat in this folder"
    return "./run.sh in this folder"


def finished_text(version: str, folder: str, how: str, system: str) -> str:
    """The launcher's last words after an install (ASCII: the Windows console is cp1252)."""
    line = "=" * 72
    if how == "mac-app":
        window = ["Kinetrace is opening in its own window now.",
                  "This Terminal window is no longer needed: you can close it",
                  "(Terminal -> Quit Terminal, or the red button)."]
    else:
        window = ["Kinetrace is opening now. KEEP THIS WINDOW OPEN while you use Kinetrace:",
                  "closing it closes Kinetrace too. You can minimise it."]
    app_files = ("kinetrace, run.sh, Kinetrace.command, Kinetrace.app, update.sh, Update.command"
                 if system == "Darwin" else
                 "kinetrace, run.bat, update.bat" if system == "Windows" else "kinetrace, run.sh, update.sh")
    out = [line, f" Kinetrace {version} is installed and ready.", line, ""]
    out += [" " + s for s in window]
    out += ["",
            " Next time, start it with: " + _start_next_time(system) + ".",
            "",
            " DO NOT delete or move anything out of this folder:",
            f"   {folder}",
            "   .venv     Kinetrace's own Python and packages",
            "   models    the models it downloads",
            f"   {app_files}",
            "             the program and its launchers",
            " Moving or renaming the WHOLE folder is fine.",
            "",
            " Safe to delete:",
            "   the downloaded ZIP file (e.g. Kinetrace-main.zip in Downloads), if you used one",
            "   logs/install-*.log  (this installation's record; keep it if you want to report a problem)",
            "",
            " Your projects and exports are saved where you choose. To uninstall, delete this folder.",
            line]
    return "\n".join(out)


def popup_html(version: str, system: str) -> str:
    """The one-time pop-up the first time the app opens after an install."""
    if system == "Darwin":
        window = ("The Terminal window from the installation is no longer needed: you can close it.")
    else:
        window = ("Keep the window Kinetrace was started from open while you use it (you can minimise it): "
                  "closing it closes Kinetrace.")
    return (f"<p><b>Kinetrace {version} is installed and ready.</b></p>"
            "<p><b>To begin:</b> File → Open Video… (Ctrl+O), or drag a video onto this window. "
            "Press <b>F1</b> for the manual; Help → System Check… says what this computer runs.</p>"
            f"<p><b>Next time</b>, start it with {_start_next_time(system)}.</p>"
            f"<p>{window}</p>"
            "<p><b>Keep the Kinetrace folder as it is</b>: its <i>.venv</i> and <i>models</i> folders are the "
            "program and its models, and the launchers start it. Moving or renaming the whole folder is fine. "
            "Your projects are saved wherever you choose.</p>")


def main(argv: list[str]) -> int:
    """`python -m kinetrace.welcome <how>` (the launchers): print the closing text and leave the flag for the app."""
    import platform
    from kinetrace import APP_VERSION
    how = argv[0] if argv else "console"
    root = str(Path(__file__).resolve().parent.parent)
    print(finished_text(APP_VERSION, root, how, platform.system()), flush=True)
    set_pending()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
