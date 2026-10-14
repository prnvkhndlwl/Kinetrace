"""QThreads that outlived their wait (I112, I133, I198, I199; R24).

Dropping the last reference to a running QThread aborts the process (0xC0000409) when it finishes, so a
thread that did not stop within its wait is kept here until its `finished` fires, and the app's closeEvent
waits for every one of them (`wait_all`). One registry for the whole app: the window, the sync dialog, the
folder import and the point-model test each kept their own (R24). No Qt import: works on any QThread.
"""
from __future__ import annotations

ORPHANS: list = []


def retire(th) -> None:
    """Keep a still-running `th` referenced until it finishes; a finished thread, one already kept or one
    whose C++ object is gone needs nothing."""
    if th is None:
        return
    try:
        running = th.isRunning()
    except RuntimeError:            # deleted already
        return
    if running and th not in ORPHANS:
        ORPHANS.append(th)
        th.finished.connect(lambda t=th: ORPHANS.remove(t) if t in ORPHANS else None)


def wait_all(ms: int = 30000) -> None:
    """Wait for every kept thread (the app's closeEvent)."""
    for th in list(ORPHANS):
        try:
            if th.isRunning():
                th.wait(ms)
        except RuntimeError:
            pass
