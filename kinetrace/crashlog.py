"""The error log (2026-09-27, I140; owner: "log the crash behaviour").

Started without a console (a double-clicked launcher, pythonw) the program has
nowhere to print an error: `main()` points stderr at devnull, so an exception in
a button, a menu entry, a key handler or a background thread left no trace, and
a native crash left only the window gone. `install()` (called from `main()`,
never by the test suites) keeps a record instead:

    <log folder>/kinetrace.log        one timestamped entry per Python error
                                      (GUI thread, QThreads, Python threads),
                                      Qt's warnings / critical / fatal messages
                                      (a fatal one is written before Qt aborts),
                                      a line when a session starts and when it
                                      ends normally -- a start without an end is
                                      a session that died; rotated at 1 MB with
                                      2 older copies
    <log folder>/kinetrace-crash.log  Python's faulthandler: the stack of every
                                      thread when the process dies of a native
                                      fault (an access violation, a segfault)

The log folder is `logs/` inside the Kinetrace folder (self-contained), else the
per-user data folder when the install cannot be written (as the recovery
folder); KINETRACE_LOG_DIR overrides, offscreen runs use a temp folder. Nothing
is ever sent anywhere: Help -> Error Report... shows `report_text()` with a Copy
button for a bug report. Every hook calls the one it replaced, so a console run
still prints its tracebacks exactly as before, and nothing here may raise: when
the folder cannot be written, `install()` returns False and the program runs as
it did without it.

Not caught: an abort that bypasses signal handlers (on Windows, 0xC0000409 --
Qt's own fatal message before it IS logged) and an exception the code catches
and handles itself.
"""
from __future__ import annotations

import atexit
import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

ENV = "KINETRACE_LOG_DIR"
_INSTALL = Path(__file__).resolve().parent.parent / "logs"
LOG_NAME = "kinetrace.log"
CRASH_NAME = "kinetrace-crash.log"
MAX_BYTES = 1_000_000          # per log file; LOG_BACKUPS older copies are kept
LOG_BACKUPS = 2
NOTICE_EVERY_S = 10.0          # at most one on-screen notice this often; the log gets everything
REPORT_BYTES = 60_000          # how much of the log the Error Report shows

_logger: logging.Logger | None = None
_folder: Path | None = None
_crash_file = None             # kept open for faulthandler for the life of the process
_prev_excepthook = None
_prev_thread_hook = None
_prev_qt_handler = None
_context = None                # callable -> str: where the app was (plain attributes only)
_bridge = None                 # QObject whose signal carries a notice to the GUI thread
_last_notice = 0.0
_suppressed = 0
_head = ""                     # this session's header line (both files carry it)
_lock = threading.Lock()


def folder() -> Path:
    """Where the logs go (created on demand; see the module docstring)."""
    env = os.environ.get(ENV)
    if env:
        return Path(env)
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        import tempfile
        return Path(tempfile.gettempdir()) / "kinetrace-test-logs"
    from kinetrace import recovery
    if recovery._writable(_INSTALL):
        return _INSTALL
    return recovery._user_dir().parent / "logs"


def log_path() -> Path:
    return (_folder or folder()) / LOG_NAME


def crash_path() -> Path:
    return (_folder or folder()) / CRASH_NAME


def _stamp() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def _versions() -> str:
    parts = [f"Python {platform.python_version()}", f"{platform.system()} {platform.release()}"]
    for mod, label in (("PySide6", "PySide6"), ("cv2", "OpenCV"), ("numpy", "numpy")):
        m = sys.modules.get(mod)
        v = getattr(m, "__version__", None) if m is not None else None
        if v:
            parts.append(f"{label} {v}")
    return " | ".join(parts)


def install(app_version: str = "") -> bool:
    """Start the log for this process. Idempotent; False (and no hooks) when
    the log folder cannot be written."""
    global _logger, _folder, _crash_file, _prev_excepthook, _prev_thread_hook, _prev_qt_handler, _head
    if _logger is not None:
        return True
    try:
        d = folder()
        d.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(d / LOG_NAME, maxBytes=MAX_BYTES,
                                                       backupCount=LOG_BACKUPS, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        crash = d / CRASH_NAME
        if crash.exists() and crash.stat().st_size > MAX_BYTES:
            os.replace(crash, crash.with_name(CRASH_NAME + ".1"))
        cf = open(crash, "a", encoding="utf-8")
    except OSError:
        return False
    log = logging.getLogger("kinetrace.errors")
    log.setLevel(logging.INFO)
    log.propagate = False
    log.addHandler(handler)
    _logger, _folder, _crash_file = log, d, cf
    head = (f"==== Kinetrace {app_version} started {_stamp()}, pid {os.getpid()} | {_versions()}").strip()
    _head = head
    log.info("\n" + head)
    try:
        cf.write(head + "\n")
        cf.flush()
        faulthandler.enable(file=cf, all_threads=True)
    except (OSError, RuntimeError, ValueError):
        pass
    _prev_excepthook = sys.excepthook
    sys.excepthook = _excepthook
    _prev_thread_hook = threading.excepthook
    threading.excepthook = _thread_hook
    try:
        from PySide6.QtCore import qInstallMessageHandler
        _prev_qt_handler = qInstallMessageHandler(_qt_message)
    except Exception:           # noqa: BLE001 - no Qt: Python errors are still logged
        pass
    atexit.register(_ended)
    return True


def attach(context=None, notify=None) -> None:
    """Called by the window once it exists: `context()` -> one line of where
    the app was (it may run on a worker thread: plain attributes only);
    `notify(text)` shows the on-screen notice, always on the GUI thread."""
    global _context, _bridge
    _context = context
    if notify is None or _logger is None:
        return
    from PySide6.QtCore import QObject, Signal

    class _Bridge(QObject):
        notice = Signal(str)

    _bridge = _Bridge()
    _bridge.notice.connect(notify)      # queued when emitted from another thread


def _ended() -> None:
    if _logger is not None:
        try:
            _logger.info(f"==== ended normally {_stamp()}, pid {os.getpid()}")
        except Exception:       # noqa: BLE001
            pass


def _where(tb) -> str:
    """The innermost frame of Kinetrace's own code, else the innermost frame."""
    frames = traceback.extract_tb(tb) if tb is not None else []
    own = [f for f in frames if "kinetrace" in f.filename.replace("\\", "/").lower()]
    f = (own or frames or [None])[-1]
    return f"{f.name} ({Path(f.filename).name}:{f.lineno})" if f is not None else "unknown place"


def log_exception(exc_type, exc, tb, thread_name: str | None = None) -> None:
    """One entry for an exception, then (rate-limited) an on-screen notice."""
    global _last_notice, _suppressed
    if _logger is None or exc_type is None or issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        return
    try:
        ctx = ""
        if _context is not None:
            try:
                ctx = str(_context())
            except Exception:       # noqa: BLE001 - the context line is a courtesy
                ctx = "(context unavailable)"
        name = thread_name or threading.current_thread().name
        where = _where(tb)
        text = "".join(traceback.format_exception(exc_type, exc, tb)).rstrip()
        _logger.error(f"{_stamp()}  ERROR  {exc_type.__name__} in {where}, thread {name}"
                      + (f"  [{ctx}]" if ctx else "") + "\n" + text)
        with _lock:
            now = time.monotonic()
            if now - _last_notice < NOTICE_EVERY_S:
                _suppressed += 1
                return
            more = f" ({_suppressed} more since the last notice)" if _suppressed else ""
            _last_notice, _suppressed = now, 0
        if _bridge is not None:
            _bridge.notice.emit(f"Something went wrong: {exc_type.__name__} in {where}{more}. "
                                "The details were saved — Help → Error Report… shows them, ready to "
                                "copy into a bug report.")
    except Exception:           # noqa: BLE001 - logging must never break the program
        pass


def _excepthook(exc_type, exc, tb) -> None:
    log_exception(exc_type, exc, tb)
    hook = _prev_excepthook or sys.__excepthook__
    try:
        hook(exc_type, exc, tb)
    except Exception:           # noqa: BLE001
        pass


def _thread_hook(args) -> None:
    log_exception(args.exc_type, args.exc_value, args.exc_traceback,
                  args.thread.name if args.thread is not None else None)
    hook = _prev_thread_hook or threading.__excepthook__
    try:
        hook(args)
    except Exception:           # noqa: BLE001
        pass


def _qt_message(mode, context, message) -> None:
    """Qt's warnings, critical and fatal messages go in the log (a fatal one
    just before Qt aborts); debug / info messages do not. The message is still
    handed on (or printed) as Qt would have."""
    try:
        from PySide6.QtCore import QtMsgType
        kind = {QtMsgType.QtWarningMsg: "QT-WARNING", QtMsgType.QtCriticalMsg: "QT-CRITICAL",
                QtMsgType.QtFatalMsg: "QT-FATAL"}.get(mode)
        if kind is not None and _logger is not None:
            src = f"  ({Path(context.file).name}:{context.line})" if getattr(context, "file", None) else ""
            _logger.error(f"{_stamp()}  {kind}  {message}{src}")
    except Exception:           # noqa: BLE001
        pass
    try:
        if _prev_qt_handler is not None:
            _prev_qt_handler(mode, context, message)
        elif sys.stderr is not None:
            sys.stderr.write(f"{message}\n")
    except Exception:           # noqa: BLE001
        pass


def _tail(p: Path, n: int) -> str:
    try:
        with open(p, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n))
            # the log is written in text mode: CRLF on Windows, LF elsewhere
            data = fh.read().decode("utf-8", "replace").replace("\r\n", "\n")
        return data if size <= n else "…" + data[data.find("\n") + 1:]
    except OSError:
        return ""


def _has_dump(text: str) -> bool:
    """Whether a crash-log excerpt holds a faulthandler dump, not only session headers."""
    return any(ln.strip() and not ln.startswith("====") for ln in text.splitlines())


def _ended_sessions() -> set[str]:
    """The header lines of the sessions the log saw end normally (every log
    file, oldest first: a session's start and end can sit in two of them)."""
    lp = log_path()
    files = [lp.with_name(f"{LOG_NAME}.{k}") for k in range(LOG_BACKUPS, 0, -1)] + [lp]
    ended, head = set(), None
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for ln in text.splitlines():
            if ln.startswith("==== Kinetrace "):
                head = ln.strip()
            elif ln.startswith("==== ended normally") and head is not None:
                ended.add(head)
                head = None
    return ended


def _crash_sections(crash: str) -> list[str]:
    """Each session's faulthandler dump with what it means. faulthandler
    also sees a fault that the library raising it then handled (ctypes turns
    an access violation into a Python error, for one), so a dump from a session
    that went on and ended normally is labelled as handled."""
    ended = _ended_sessions()
    blocks, cur = [], None
    for ln in crash.splitlines():
        if ln.startswith("==== Kinetrace "):
            cur = [ln.strip(), []]
            blocks.append(cur)
        elif cur is not None:
            cur[1].append(ln)
        elif ln.strip():
            cur = ["(a session whose start is no longer in the crash log)", [ln]]
            blocks.append(cur)
    out = []
    for head, body in blocks:
        if not any(b.strip() for b in body):
            continue
        if head == _head:
            what = "THIS session, still running: the program went on, so the fault was handled"
        elif head in ended:
            what = ("that session went on and ended normally, so the library that raised the fault "
                    "handled it (usually harmless)")
        else:
            what = "that session did NOT end normally: this is most likely what closed the program"
        out.append(f"{head}\n  -> {what}\n" + "\n".join(body).strip("\n"))
    return out


def report_text() -> str:
    """What Help -> Error Report shows: where the logs are, the recent log
    (errors, Qt messages, session starts / ends) and any native crash dump."""
    lp, cp = log_path(), crash_path()
    log = _tail(lp, REPORT_BYTES)
    if len(log) < REPORT_BYTES // 2:
        older = lp.with_name(LOG_NAME + ".1")
        if older.exists():
            log = _tail(older, REPORT_BYTES - len(log)) + log
    crash = _tail(cp, REPORT_BYTES // 3)
    errors = sum(1 for ln in log.splitlines() if "  ERROR  " in ln or "  QT-CRITICAL  " in ln
                 or "  QT-FATAL  " in ln)
    out = [f"Kinetrace error report — {time.strftime('%Y-%m-%d %H:%M')}",
           f"Log: {lp}",
           f"Crash log: {cp}",
           "Nothing in these files is sent anywhere; paste this text (or attach the files) to a bug report.",
           ""]
    if not log.strip():
        out.append("No errors have been recorded" + ("." if _logger is not None else
                                                     " (the log is not running in this session)."))
    else:
        out.append(f"Recent log ({errors} error / Qt critical or fatal entries shown; "
                   "a 'started' line with no 'ended normally' after it is a session that did not close normally):")
        out.append(log.strip("\n"))
    if _has_dump(crash):
        out += ["", "Native fault dumps (the Python stack of every thread at the moment of the fault):"]
        out += _crash_sections(crash)
    return "\n".join(out)
