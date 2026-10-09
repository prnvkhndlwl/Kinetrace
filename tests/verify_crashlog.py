"""The error log (kinetrace/crashlog.py, 2026-09-27): every kind of error the
running program can meet is written to kinetrace.log with its traceback and a
line of context, the on-screen notice appears at most once per NOTICE_EVERY_S,
Help -> Error Report... shows and copies it, the log rotates, a native crash
leaves a stack dump, a Qt fatal message is written before the abort, an
unwritable folder changes nothing, and the real main() starts and ends a
logged session. CPU only.

Run: .venv\\Scripts\\python.exe tests\\verify_crashlog.py
"""
import os
import shutil
import subprocess
import sys
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

OUT = os.path.join(ROOT, "tests", "out", "crashlog")
shutil.rmtree(OUT, ignore_errors=True)
LOGS = os.path.join(OUT, "main")
os.makedirs(LOGS)
os.environ["KINETRACE_LOG_DIR"] = LOGS

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _clean import forget_recovery  # noqa: E402

VID = os.path.join(OUT, "clip.mp4")
vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
for f in range(30):
    vw.write(np.random.RandomState(f).randint(0, 255, (240, 320, 3)).astype(np.uint8))
vw.release()
forget_recovery(VID)

from kinetrace import crashlog  # noqa: E402

# the hooks must hand every error on to the ones they replaced (a console run
# still prints its tracebacks)
PREV = []
sys.excepthook = lambda t, v, tb: (PREV.append(("except", t.__name__)), sys.__excepthook__(t, v, tb))
threading.excepthook = lambda a: (PREV.append(("thread", a.exc_type.__name__)), threading.__excepthook__(a))
assert crashlog.install("test") and crashlog.install("test"), "install is idempotent"

from PySide6.QtCore import QEvent, QThread, QTimer, Qt, qWarning  # noqa: E402
from PySide6.QtGui import QAction, QKeyEvent  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QMessageBox,  # noqa: E402
                               QPlainTextEdit, QWidget)

for _n in ("question", "warning", "information", "critical"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: QMessageBox.No))
app = QApplication([])
from kinetrace.app import MainWindow, READY  # noqa: E402

win = MainWindow()
win.show()
NOTICES = []
crashlog.attach(win._error_context, lambda text: (NOTICES.append(text), win._on_error_logged(text)))


def pump(cond, sec):
    t = time.time()
    while time.time() - t < sec and not cond():
        app.processEvents()
        time.sleep(0.005)
    return cond()


win._open_video(VID)
assert pump(lambda: win.state == READY, 20), "video did not open"


def log_text():
    return open(crashlog.log_path(), encoding="utf-8").read()


# ---- 1. every kind of error, delivered by the event loop as in the real app
class KeyBoom(QWidget):
    def keyPressEvent(self, ev):
        raise KeyError("key handler boom")


class ThreadBoom(QThread):
    def run(self):
        raise RuntimeError("QThread boom")


kb = KeyBoom()
kb.show()
act = QAction("boom", win)
act.triggered.connect(lambda: 1 / 0)                     # a menu entry / button that raises
qt_thread = ThreadBoom()
py_thread = threading.Thread(target=lambda: [].pop(), name="py-boom")
QTimer.singleShot(50, act.trigger)
QTimer.singleShot(150, lambda: QApplication.postEvent(kb, QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier)))
QTimer.singleShot(250, qt_thread.start)
QTimer.singleShot(350, py_thread.start)
QTimer.singleShot(450, lambda: qWarning("kinetrace test warning"))
QTimer.singleShot(900, lambda: app.exit(0))      # exit, not quit: quit() closes the windows
app.exec()
qt_thread.wait()
py_thread.join()
pump(lambda: False, 0.2)                                  # queued notices from the threads
txt = log_text()
for needle in ("ZeroDivisionError in <lambda>", "KeyError in keyPressEvent", "RuntimeError in run",
               "IndexError in <lambda>", "thread py-boom", "QT-WARNING  kinetrace test warning",
               "Traceback (most recent call last)", "==== Kinetrace test started"):
    assert needle in txt, f"missing from the log: {needle!r}\n{txt[-2000:]}"
assert "[ready, frame 0, 1 camera(s), working in camera 1, point model" in txt, "context line"
assert sorted(PREV) == sorted([("except", "ZeroDivisionError"), ("except", "KeyError"),
                               ("except", "RuntimeError"), ("thread", "IndexError")]), PREV
print("errors in a slot, a key handler, a QThread and a Python thread are logged with context, "
      "Qt warnings too, and the old hooks still run OK")

# ---- 2. the notice: once, then rate-limited, then counting what it skipped
assert len(NOTICES) == 1 and NOTICES[0].startswith("Something went wrong: ZeroDivisionError"), NOTICES
assert "Help → Error Report…" in NOTICES[0]
assert "Something went wrong" in win.toast.text() and win.toast.isVisible(), \
    ("the notice is on screen", win.toast.isVisible(), win.toast.text()[:300])
crashlog._last_notice = 0.0                              # as if NOTICE_EVERY_S had passed
QTimer.singleShot(20, act.trigger)
QTimer.singleShot(200, lambda: app.exit(0))
app.exec()
assert len(NOTICES) == 2 and "(3 more since the last notice)" in NOTICES[1], NOTICES
print("one notice at a time; the next one counts the errors in between OK")

# ---- 3. Help -> Error Report...: shows the log and the System Check, Copy copies it
SEEN = {}


def _capture(dlg):
    view = dlg.findChild(QPlainTextEdit)
    SEEN["text"] = view.toPlainText()
    bb = dlg.findChild(QDialogButtonBox)
    SEEN["buttons"] = [b.text() for b in bb.buttons()]
    next(b for b in bb.buttons() if b.text() == "Copy").click()    # never "Open the log folder" here
    return QDialog.Rejected


QDialog.exec = _capture
assert win.act_error_report.isEnabled() and "bug report" in win.act_error_report.toolTip()
win.act_error_report.trigger()
rep = SEEN["text"]
for needle in (str(crashlog.log_path()), "ZeroDivisionError", "KeyError", "---- System check ----",
               "Nothing in these files is sent anywhere"):
    assert needle in rep, f"missing from the report: {needle!r}"
assert "Open the log folder" in SEEN["buttons"]
assert QApplication.clipboard().text() == rep, "Copy puts the whole report on the clipboard"
print("Help -> Error Report shows the log and the System Check; Copy copies it OK")

# ---- 4. rotation: the log never grows past its budget; the report stays bounded
for i in range(1400):
    crashlog._logger.error("x" * 1000)
files = sorted(os.listdir(LOGS))
logs = [f for f in files if f.startswith(crashlog.LOG_NAME)]
assert crashlog.LOG_NAME + ".1" in logs and len(logs) <= 1 + crashlog.LOG_BACKUPS, logs
assert all(os.path.getsize(os.path.join(LOGS, f)) <= crashlog.MAX_BYTES + 2000 for f in logs)
assert len(win._error_report_text()) < crashlog.REPORT_BYTES * 2 + 20000
print(f"the log rotates ({', '.join(logs)}), each file within {crashlog.MAX_BYTES} bytes OK")

win.close()
pump(lambda: False, 0.3)
win._dev_probe.wait(30000)
forget_recovery(VID)

# ---- 5. child processes: a native crash, a Qt fatal abort, an unwritable folder,
# a normal exit, and the real main()
CHILD = r'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
case = sys.argv[2]
from kinetrace import crashlog
ok = crashlog.install("child")
print("installed", ok, flush=True)
if case == "unwritable":
    assert not ok and sys.excepthook is sys.__excepthook__, "nothing hooked when the log cannot be written"
    assert "No errors have been recorded" in crashlog.report_text()
    print("fine", flush=True)
elif case == "segfault":
    import faulthandler
    faulthandler._sigsegv()         # a real native crash (CPython's own test helper)
elif case == "handled":
    import ctypes
    try:
        ctypes.string_at(0)         # a fault the library itself turns into a Python error
    except OSError:
        print("handled by ctypes", flush=True)
elif case == "report":
    print("REPORT<<" + crashlog.report_text() + ">>REPORT", flush=True)
elif case == "qtfatal":
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    import shiboken6
    app = QApplication([])
    class T(QThread):
        def run(self):
            time.sleep(5)
    t = T()
    t.start()
    time.sleep(0.2)
    shiboken6.delete(t)             # destroyed while running: Qt's fatal abort (the I133 class)
    time.sleep(1)
elif case == "main":
    from PySide6.QtCore import QTimer
    import kinetrace.crashlog as cl
    real_attach = cl.attach
    def attach(context=None, notify=None):
        real_attach(context, notify)
        win = notify.__self__
        QTimer.singleShot(1500, win.close)
    cl.attach = attach
    cl.install = lambda v="": True  # already installed above, as main() would
    from kinetrace.app import main
    sys.argv = ["kinetrace"]
    main()
print("exiting normally", flush=True)
'''


def child(case, logdir):
    os.makedirs(os.path.dirname(logdir), exist_ok=True)
    env = dict(os.environ, KINETRACE_LOG_DIR=logdir, KINETRACE_RECOVERY_DIR=os.path.join(OUT, "rec_" + case))
    r = subprocess.run([sys.executable, "-c", CHILD, ROOT, case], capture_output=True, text=True,
                       errors="replace", env=env, timeout=240)
    return r


def read(p):
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""


d = os.path.join(OUT, "segfault")
r = child("segfault", d)
crash, log = read(os.path.join(d, crashlog.CRASH_NAME)), read(os.path.join(d, crashlog.LOG_NAME))
assert r.returncode != 0 and "most recent call first" in crash, (r.returncode, crash)
assert "started" in log and "ended normally" not in log, "a session that died has no end line"
if sys.platform == "win32":                       # a null read is an OSError only on Windows (SEH);
    r2 = child("handled", d)                      # on macOS / Linux it is a real SIGSEGV (Mac report)
    assert r2.returncode == 0 and "handled by ctypes" in r2.stdout, (r2.stdout, r2.stderr[-500:])
rep = child("report", d).stdout                   # the next start opens Help -> Error Report
rep = rep[rep.index("REPORT<<"):rep.index(">>REPORT")]
secs = rep.split("\n==== Kinetrace child started")
crashed = [x for x in secs if "did NOT end normally" in x]
handled = [x for x in secs if "ended normally, so the library that raised the fault handled it" in x]
assert len(crashed) == 1 and "Segmentation fault" in crashed[0] and "most recent call first" in crashed[0], \
    rep[-3000:]
if sys.platform == "win32":                       # faulthandler sees first-chance faults on Windows only
    assert len(handled) == 1 and "string_at" in handled[0], rep[-3000:]
print(f"a native crash leaves every thread's stack in {crashlog.CRASH_NAME}; the report tells the crash "
      "from a fault a library handled OK")

d = os.path.join(OUT, "qtfatal")
r = child("qtfatal", d)
log = read(os.path.join(d, crashlog.LOG_NAME))
assert r.returncode != 0 and "QT-FATAL  QThread: Destroyed while thread" in log, (r.returncode, log[-800:])
assert "ended normally" not in log
print("Qt's fatal message is in the log before the abort OK")

d = os.path.join(OUT, "unwritable_is_a_file")
os.makedirs(OUT, exist_ok=True)
open(d, "w").write("a file where the folder should be")
r = child("unwritable", d)
assert r.returncode == 0 and "installed False" in r.stdout and "fine" in r.stdout, (r.stdout, r.stderr[-800:])
print("an unwritable log folder changes nothing (install returns False, no hooks) OK")

d = os.path.join(OUT, "normal")
r = child("normal", d)
log = read(os.path.join(d, crashlog.LOG_NAME))
assert r.returncode == 0 and log.index("started") < log.index("ended normally"), log
print("a normal exit ends its session with 'ended normally' OK")

d = os.path.join(OUT, "main")
shutil.rmtree(d, ignore_errors=True)
r = child("main", d)
log = read(os.path.join(d, crashlog.LOG_NAME))
assert r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-1500:])
assert "==== Kinetrace child started" in log and "ended normally" in log and "  ERROR  " not in log, log[-1500:]
print("the real main() starts, attaches the window, closes and ends its session OK")
print("verify_crashlog PASSED")
