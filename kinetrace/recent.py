"""The recent projects of File -> Open Recent and the start screen (G205; owner 2026-10-16: "everytime I open
the app, i have to navigate through a bunch of folders to launch a saved project").

The list lives in the app's own settings.ini (`app._settings`, inside the Kinetrace folder: deleting the folder
removes it) as ONE JSON string under `KEY`: `[{"path", "id", "name"}, ...]`, newest first, at most `MAX`. Paths
are stored with forward slashes (`Path.as_posix`) and read back with `Path`, so the same code reads the list on
Windows, macOS and Linux and an INI's backslash escaping never matters.

A project that was MOVED keeps its entry: every project carries a `project_id` (`projectfile`), and opening or
saving a project whose id is listed at a path that no longer exists updates that entry instead of adding a second
one. A copied folder (same id, both paths still there) is two projects to the user: both are kept. An entry whose
path is gone stays listed; the app says so and offers Locate... (`nearest_folder` is where the picker opens).

No Qt here: `settings` is anything with `value(key, default)` / `setValue(key, value)`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

KEY = "recent/projects"
MAX = 10


def _norm(path) -> str:
    """The comparison form of a path: absolute, and case-folded where the OS ignores case (Windows)."""
    return os.path.normcase(os.path.abspath(str(path)))


def entry(path, pid: str | None) -> dict:
    """One list entry for the project at `path` (its folder, or a single .kinetrace file)."""
    p = Path(os.path.abspath(str(path)))
    return {"path": p.as_posix(), "id": str(pid or ""), "name": p.name}


def load(settings) -> list[dict]:
    """The stored list, newest first; anything unreadable gives an empty list (never an error)."""
    try:
        raw = json.loads(str(settings.value(KEY, "") or "[]"))
    except Exception:      # noqa: BLE001 - a damaged settings value is a list that starts over
        return []
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for it in raw:
        if not isinstance(it, dict) or not isinstance(it.get("path"), str) or not it["path"]:
            continue
        e = entry(Path(it["path"]), it.get("id"))
        if _norm(e["path"]) in seen:
            continue
        seen.add(_norm(e["path"]))
        out.append(e)
    return out[:MAX]


def store(settings, items: list[dict]) -> bool:
    """Write the list; False when the settings file cannot be written (not worth an error)."""
    try:
        settings.setValue(KEY, json.dumps(list(items)[:MAX]))
        return True
    except Exception:      # noqa: BLE001
        return False


def touch(items: list[dict], path, pid: str | None, exists=os.path.exists) -> list[dict]:
    """`items` with the project at `path` (id `pid`) on top: the same path, or the same id listed at a path that
    no longer `exists` (the project was moved), is replaced, never listed twice; at most `MAX` kept."""
    new = entry(path, pid)
    key = _norm(new["path"])
    out = [new]
    for it in items:
        if _norm(it["path"]) == key:
            continue
        if new["id"] and it.get("id") == new["id"] and not exists(it["path"]):
            continue                      # moved: its new place is the entry on top
        out.append(it)
    return out[:MAX]


def remove(items: list[dict], path) -> list[dict]:
    key = _norm(path)
    return [it for it in items if _norm(it["path"]) != key]


def clear(settings) -> bool:
    return store(settings, [])


def labels(items: list[dict]) -> list[str]:
    """What the menu and the start screen show for each entry: the project's folder name, followed by the
    folder it is in when two entries share a name (one `trial.kinetrace` per day folder)."""
    names = [it["name"] for it in items]
    return [f"{n}  —  {Path(it['path']).parent.name or Path(it['path']).parent}" if names.count(n) > 1 else n
            for n, it in zip(names, items)]


def nearest_folder(path, timeout: float = 0.5, isdir=os.path.isdir) -> Path | None:
    """The closest folder above `path` that still exists: where Locate... opens; None if not even its drive
    is there (or it does not answer within `timeout`, an unplugged share)."""
    parents = [str(p) for p in Path(str(path)).parents]
    found = exists_quick(parents, timeout, check=isdir)
    return next((Path(p) for p in parents if found.get(p)), None)


def exists_quick(paths: list[str], timeout: float = 0.5, check=os.path.exists) -> dict[str, bool | None]:
    """{path: True / False / None}: whether each path exists, asked on daemon threads and waited for at most
    `timeout` seconds in all. A disconnected network share can block a stat for many seconds on any OS: an
    answer that does not come in time is None (unknown: listed plainly, looked at again when it is clicked)."""
    got: dict[str, bool | None] = {p: None for p in paths}

    def ask(p):
        try:
            got[p] = bool(check(p))
        except OSError:
            got[p] = False

    threads = [threading.Thread(target=ask, args=(p,), daemon=True) for p in paths]
    for t in threads:
        t.start()
    end = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    return dict(got)
