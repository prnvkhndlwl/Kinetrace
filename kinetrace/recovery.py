"""Unsaved-work recovery. The project file changes only on Save; the 30 s
autosave writes here instead:

    <recovery folder>/<project_id>.kinetrace    the unsaved state (binary tables: fast)
    <recovery folder>/<project_id>.json         {project_id, base_saved_at, last_path,
                                                 videos, written_at, temporary}
    <recovery folder>/<project_id>.view.json    where the user was (frame, zoom, toggles)
                                                when they closed without data changes

A project is matched by the id inside it, not by its path, so moving or
renaming the project file still finds its unsaved work. The recovery folder
is inside the Kinetrace folder (self-contained); when that cannot be written
(installed read-only), the OS's per-user data folder. No Qt here.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ENV = "KINETRACE_RECOVERY_DIR"
_INSTALL = Path(__file__).resolve().parent.parent / "recovery"
_cached: tuple[Path, bool] | None = None


def _user_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Kinetrace"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Kinetrace"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "kinetrace"
    return base / "recovery"


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".probe-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def folder() -> tuple[Path, bool]:
    """(recovery folder, True if it is the per-user fallback)."""
    global _cached
    env = os.environ.get(ENV)
    if env:
        d = Path(env)
        d.mkdir(parents=True, exist_ok=True)
        return d, False
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        # the test suites run offscreen: their unsaved work must never appear
        # in the user's own recovery folder
        import tempfile
        d = Path(tempfile.gettempdir()) / "kinetrace-test-recovery"
        d.mkdir(parents=True, exist_ok=True)
        return d, False
    if _cached is None:
        _cached = (_INSTALL, False) if _writable(_INSTALL) else (_user_dir(), True)
        if _cached[1]:
            _cached[0].mkdir(parents=True, exist_ok=True)
    return _cached


def paths(project_id: str) -> tuple[Path, Path, Path]:
    d = folder()[0]
    return d / f"{project_id}.kinetrace", d / f"{project_id}.json", d / f"{project_id}.view.json"


def write_info(project_id: str, *, base_saved_at: str | None, last_path: str | None,
               videos: list[str], temporary: bool) -> None:
    info = {"project_id": project_id, "base_saved_at": base_saved_at, "last_path": last_path,
            "videos": list(videos), "temporary": bool(temporary),
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = paths(project_id)[1]
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def find(project_id: str) -> dict | None:
    """The recovery of this project, if its data file exists."""
    zp, jp, _ = paths(project_id)
    if not (zp.exists() and jp.exists()):
        return None
    try:
        return json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"project_id": project_id, "damaged": True}


def scan() -> list[dict]:
    """Every recovery in the folder, newest first."""
    out = []
    for jp in folder()[0].glob("*.json"):
        if jp.name.endswith(".view.json"):
            continue
        info = find(jp.stem)
        if info is not None:
            out.append(info)
    return sorted(out, key=lambda d: d.get("written_at", ""), reverse=True)


def for_video(video_path: str) -> dict | None:
    """A never-saved session's recovery that used this video (opening the video
    alone offers it)."""
    want = os.path.normcase(os.path.abspath(video_path))
    for info in scan():
        if info.get("temporary") and any(os.path.normcase(os.path.abspath(v)) == want
                                         for v in info.get("videos", [])):
            return info
    return None


def discard(project_id: str) -> None:
    for p in paths(project_id):
        try:
            p.unlink()
        except OSError:
            pass


def decline(project_id: str) -> Path | None:
    """The user chose the last saved version: keep the unsaved copy (recovery/
    declined/), but stop offering it."""
    return _move(project_id, "declined")


def quarantine(project_id: str) -> Path | None:
    """Move an unreadable recovery aside (recovery/damaged/) instead of deleting it."""
    return _move(project_id, "damaged")


def _move(project_id: str, sub: str) -> Path | None:
    d = folder()[0] / sub
    d.mkdir(exist_ok=True)
    moved = None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for p in paths(project_id):
        if p.exists():
            dst = d / f"{stamp}-{p.name}"
            shutil.move(str(p), str(dst))
            moved = moved or dst
    return moved


def write_view(project_id: str, view: dict, base_saved_at: str | None) -> None:
    p = paths(project_id)[2]
    p.write_text(json.dumps({"base_saved_at": base_saved_at, "view": view}, indent=1), encoding="utf-8")


def read_view(project_id: str, base_saved_at: str | None) -> dict | None:
    """The view left at close, only if the project file is still that save."""
    p = paths(project_id)[2]
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d.get("view") if d.get("base_saved_at") == base_saved_at else None


def cleanup_stale(max_age_s: float = 3600) -> None:
    """Temp files left by a crash in the middle of a write."""
    now = time.time()
    for p in folder()[0].glob("*.tmp"):
        try:
            if now - p.stat().st_mtime > max_age_s:
                p.unlink()
        except OSError:
            pass
