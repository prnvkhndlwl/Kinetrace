"""CSV tables formatted in helper processes, for a big save (I145).

A full save of a large project spends nearly all its time turning float32
values into their shortest exact text, and numpy holds the GIL while it does,
so threads cannot share the work (measured: 1.3x at 16 threads). Helper
PROCESSES can: each runs `python -m kinetrace.csvpool` (numpy and
projectfile only, no Qt), reads jobs from its stdin, writes each table's CSV
to the path it is given with exactly `Table.csv`'s bytes, and answers on
stdout. `format_tables` hands a list of (Table, path) to them and returns the
jobs that were done; anything that goes wrong (a helper that will not start,
dies, or answers nonsense) leaves those jobs to the caller, which formats them
itself: a save never depends on the helpers. KINETRACE_SAVE_WORKERS = the
number of helpers (0 = none; default: the cores less two, at most 12).

Jobs cross the pipe as pickles of plain arrays between two processes of this
same program (never data from outside).
"""
from __future__ import annotations

import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

MIN_ROWS = 1_500_000          # below this a save formats in its own thread (about 1.5 s of work)
_ROOT = Path(__file__).resolve().parent.parent


def n_workers() -> int:
    env = os.environ.get("KINETRACE_SAVE_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            return 0
    return max(0, min(12, (os.cpu_count() or 1) - 2))


def _send(fh, obj) -> None:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    fh.write(struct.pack("<Q", len(b)))
    fh.write(b)
    fh.flush()


def _recv(fh):
    head = fh.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    b = fh.read(n)
    if len(b) < n:
        return None
    return pickle.loads(b)


def _start() -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    return subprocess.Popen([sys.executable, "-m", "kinetrace.csvpool"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(_ROOT), env=env,
                            creationflags=flags)


def format_tables(jobs: list, fsync: bool = True, workers: int | None = None,
                  timeout: float | None = None) -> set[int]:
    """Write each (Table, path) of `jobs` as its CSV in helper processes.
    -> the indices of the jobs that were written; the rest is the caller's.
    A helper that HANGS (not dies) is killed at the deadline — `timeout`,
    default 30 s + 5 us per row, five times the one-thread time — and its
    jobs are left to the caller: a save can never wait on it for ever."""
    if timeout is None:
        timeout = 30.0 + 5e-6 * sum(t.n for t, _p in jobs)
    k = n_workers() if workers is None else workers
    k = min(k, len(jobs))
    if k <= 0:
        return set()
    todo: queue.Queue = queue.Queue()
    for i in sorted(range(len(jobs)), key=lambda i: -jobs[i][0].n):     # the biggest first
        todo.put(i)
    done: set[int] = set()
    lock = threading.Lock()

    def feed(proc):
        try:
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                t, path = jobs[i]
                _send(proc.stdin, (i, str(path), t.cols, t.kinds, t.written, t.data, fsync))
                ans = _recv(proc.stdout)
                if not (isinstance(ans, tuple) and ans[0] == i and ans[1] is True):
                    todo.put(i) if ans is None else None     # a helper that died: the job goes back
                    return
                with lock:
                    done.add(i)
        except (OSError, ValueError, pickle.PickleError, EOFError):
            return

    procs = []
    try:
        for _ in range(k):
            try:
                procs.append(_start())
            except OSError:
                break
        threads = [threading.Thread(target=feed, args=(p,), daemon=True) for p in procs]
        for th in threads:
            th.start()
        deadline = time.monotonic() + timeout
        for th in threads:
            th.join(max(0.0, deadline - time.monotonic()))
        if any(th.is_alive() for th in threads):
            for p in procs:                  # hung: its pipe closes, the feeder returns
                p.kill()
            for th in threads:
                th.join(5)
    finally:
        for p in procs:
            try:
                p.stdin.close()
            except OSError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    return done


def _serve() -> int:
    """The helper: jobs in on stdin, answers out on stdout, until stdin closes."""
    from kinetrace.projectfile import Table
    src, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        job = _recv(src)
        if job is None:
            return 0
        i, path, cols, kinds, written, data, fsync = job
        try:
            text = Table(cols, kinds, data, written).csv().encode("utf-8")
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(text)
                fh.flush()
                if fsync:
                    os.fsync(fh.fileno())
            _send(out, (i, True, ""))
        except Exception as e:      # noqa: BLE001 - reported; the saving process writes it itself
            _send(out, (i, False, f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    sys.exit(_serve())
