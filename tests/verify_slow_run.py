"""Offscreen GUI: honest about slow and small computers (G240, G241).

- an out-of-memory error answers by CAUSE (the graphics card or the computer's memory) and by what was running
  (tracking or outlining), on the real endings PyTorch / NumPy print (G241);
- the Track label's time left is in hours for a long run, and a run on the processor with a long time left says
  so ONCE, with what the user can do (G240).

Run: .venv\\Scripts\\python.exe tests\\verify_slow_run.py   (needs test600.mp4)
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication, QMessageBox

from _clean import forget_recovery  # noqa: E402

VID = os.path.join(ROOT, "test600.mp4")
if not os.path.exists(VID):
    sys.exit("test600.mp4 missing - python make_test_video.py test600.mp4 --seed 0")
forget_recovery(VID)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

from kinetrace import app as kapp  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

FAILED = []


def check(ok, what, detail=""):
    print(("OK   " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(what)


# ---- G241: the memory sentence by cause -------------------------------------------------------
TB = "Traceback (most recent call last):\n  File \"x.py\", line 1, in f\n    y()\n"
CUDA = TB + "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 4.00 GiB"
MPS = TB + "RuntimeError: MPS backend out of memory (MPS allocated: 7.9 GB, other allocations: 1.2 GB, max allowed: 9.07 GB)."
CPU = TB + ("RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: "
            "you tried to allocate 18400000000 bytes.")
NPY = TB + "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.19 GiB for an array with shape (40000, 2000)"
PYM = TB + "MemoryError"


def hint(tb, doing):
    try:
        return kapp._model_error_hint(tb, doing=doing)
    except TypeError:
        return kapp._model_error_hint(tb)


for name, tb in (("CUDA", CUDA), ("Apple GPU", MPS)):
    h = hint(tb, "track")
    check("graphics card" in h.lower() and "segmentation model" not in h.lower(),
          f"{name} out of memory while tracking: the graphics card, and no word about the segmentation model", h)
    h = hint(tb, "outline")
    check("graphics card" in h.lower() and "segmentation model" in h.lower(),
          f"{name} out of memory while outlining: names the smaller segmentation model", h)
for name, tb in (("PyTorch on the processor", CPU), ("NumPy", NPY), ("Python", PYM)):
    h = hint(tb, "track")
    check(bool(h) and "graphics card" not in h.lower() and "gpu" not in h.lower() and "memory" in h.lower(),
          f"{name} out of memory: the computer's memory, never the graphics card", h)
check("cotracker3" in hint(CPU, "track").lower(),
      "out of memory while tracking: names the point model that needs less memory", hint(CPU, "track"))
check(hint(TB + "ValueError: something else", "track") == "", "another error gets no memory sentence")

# ---- G240: time left in hours; one notice for a long run on the processor ----------------------
eta = getattr(kapp, "_eta_text", None)
check(eta is not None and eta(39900) == "11 h 05 min", "39,900 s reads 11 h 05 min",
      "" if eta is None else eta(39900))
check(eta is not None and eta(125) == "2 min 05 s" and eta(40) == "40 s", "short runs read as before")

app = QApplication.instance() or QApplication([])
win = MainWindow()
win.resize(1500, 950)
win.show()


def pump(seconds=0.15):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


win._open_video(VID)
for _ in range(300):
    pump(0.05)
    if win.state == READY:
        break
assert win.state == READY
say = getattr(win, "_say_long_run", None)
check(say is not None, "the window can say a run will be long")
if say is not None:
    win._device_kind = "GPU"
    win._long_run_said = False
    win.toast.hide()
    say(39900.0)
    check(not win.toast.isVisible(), "a long run on the graphics card says nothing extra")
    win._device_kind = "CPU"
    say(600.0)
    check(not win.toast.isVisible(), "a 10-minute run on the processor says nothing extra")
    say(39900.0)
    t = win.toast.text()
    check(win.toast.isVisible() and "11 h 05 min" in t and "System Check" in t and "<b>X</b>" in t,
          "a long run on the processor says how long, why, and that X pauses and keeps everything", t[:200])
    win.toast.hide()
    say(39000.0)
    check(not win.toast.isVisible(), "... once per run")

win._dev_probe.wait(15000)
win.close()
forget_recovery(VID)
if FAILED:
    print(f"SLOW RUN FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("SLOW RUN PASSED")
