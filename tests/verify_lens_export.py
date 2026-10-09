"""(G148) A camera's lens profile can be exported at any time and loaded onto another camera.

Through the window (offscreen, two cameras, no GPU): 3D -> Export Lens Profile… with no profile says so
and writes nothing; with GoPro's lens on camA it writes a .klens.json that reads back identical (report
kept), an OpenCV .yml that reads back identical, and refuses a fisheye as an Argus .txt with a message;
3D -> Load a Lens Profile for Cameras… (its camera list, G176: the working camera ticked) puts the saved
file on camB; a profile that would replace camB's is not pre-ticked (nothing changes until it is ticked);
a profile of another picture size cannot be ticked; with profiles on both cameras the export asks which
camera's.

.venv\\Scripts\\python.exe tests\\verify_lens_export.py
"""
import json
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "lens_export")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

from _synth_gopro import make_gopro_video  # noqa: E402
import _lens_attach  # noqa: E402

ASK = {"save": ("", ""), "open": "", "item": None, "question": QMessageBox.No, "info": [], "warn": [], "items": []}
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ASK["save"])
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QMessageBox.information = staticmethod(lambda *a, **k: (ASK["info"].append(str(a[2]) if len(a) > 2 else ""),
                                                        QMessageBox.Ok)[1])
QMessageBox.warning = staticmethod(lambda *a, **k: (ASK["warn"].append(str(a[2]) if len(a) > 2 else ""),
                                                    QMessageBox.Ok)[1])
QMessageBox.question = staticmethod(lambda *a, **k: ASK["question"])


def _item(parent, title, label, items, current=0, editable=False):
    ASK["items"].append(list(items))
    return (ASK["item"](items) if ASK["item"] else items[current]), True


QInputDialog.getItem = staticmethod(_item)
_lens_attach.install()
app = QApplication.instance() or QApplication([])
from kinetrace import calibio, lens  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def same(a, b):
    return (a.width, a.height, a.fisheye) == (b.width, b.height, b.fisheye) and np.allclose(a.K, b.K) \
        and np.allclose(np.ravel(a.dist), np.ravel(b.dist))


VA = make_gopro_video(os.path.join(OUT, "camA.mp4"), seconds=2)
VB = make_gopro_video(os.path.join(OUT, "camB.mp4"), seconds=2, with_header=False)   # GoPro, but no lens model
w = MainWindow()
w.resize(1400, 900)
w.show()
w._open_video(VA)
for _ in range(300):
    pump(0.05)
    if w.state == READY and w.project is not None and not w._loading:
        break
pump(0.3)
check(w._add_view(VB), "two cameras")
pump(0.3)
p = w.project
w._apply_state()
check(w.act_export_lens.isEnabled() and w.act_load_lens.isEnabled(), "both 3D entries are enabled")

ASK["save"] = (os.path.join(OUT, "never.klens.json"), "Kinetrace lens (*.klens.json)")
w.act_export_lens.trigger()
pump(0.1)
check(ASK["info"] and "No camera of this project has a lens profile yet" in ASK["info"][-1]
      and not os.path.exists(ASK["save"][0]), "no profile: it says so and writes nothing")

w._use_gopro_lenses([0])
prof = p.lenses[0]
check(prof is not None and prof.report.get("gopro_nominal"), "camA has GoPro's lens model")

ASK["save"] = (os.path.join(OUT, "camA_lens"), "Kinetrace lens (*.klens.json)")
w.act_export_lens.trigger()
pump(0.1)
kl = os.path.join(OUT, "camA_lens.klens.json")
back = lens.LensProfile.load(kl) if os.path.exists(kl) else None
check(back is not None and same(back, prof) and back.report.get("gopro_nominal")
      and json.load(open(kl))["source"] == prof.source,
      "Kinetrace format: the file reads back identical, report and source kept")
ASK["save"] = (os.path.join(OUT, "camA_lens.yml"), "OpenCV lens (*.yml)")
w.act_export_lens.trigger()
pump(0.1)
yml = calibio.read_lens(ASK["save"][0]) if os.path.exists(ASK["save"][0]) else None
check(yml is not None and same(yml, prof), "OpenCV format: reads back identical")
n_warn = len(ASK["warn"])
ASK["save"] = (os.path.join(OUT, "camA_lens.txt"), "Argus / DLTdv camera profile (*.txt)")
w.act_export_lens.trigger()
pump(0.1)
check(len(ASK["warn"]) == n_warn + 1 and "fisheye" in ASK["warn"][-1] and not os.path.exists(ASK["save"][0]),
      "a fisheye lens as an Argus line is refused in words, nothing written", ASK["warn"][-1:])

# camB: load the saved file
w._set_active_view(1)
pump(0.3)
check(p.lenses[1] is None, "camB starts with no lens profile")
ASK["open"] = kl
w.act_load_lens.trigger()
pump(0.1)
check(p.lenses[1] is not None and same(p.lenses[1], prof) and p.dirty, "Load: camB has the saved profile")
other = lens.LensProfile(640, 480, prof.K * 2, prof.dist, True, float("nan"), 0, "elsewhere")
other_path = other.save(os.path.join(OUT, "other_size.klens.json"))
ASK["open"] = other_path
w.act_load_lens.trigger()
pump(0.1)
seen = _lens_attach.STATE["seen"][-1]
check(same(p.lenses[1], prof) and "640" in seen["rows"][1][0] and not seen["rows"][1][1] and not seen["chosen"],
      "a profile of another picture size cannot be ticked (the row says why), the camera keeps its own", seen)
mod = lens.LensProfile(prof.width, prof.height, prof.K * 1.01, prof.dist, True, 0.5, 30, "checkerboard (Kinetrace)")
mod_path = mod.save(os.path.join(OUT, "mod.klens.json"))
ASK["open"] = mod_path
w.act_load_lens.trigger()
pump(0.1)
seen = _lens_attach.STATE["seen"][-1]
check(same(p.lenses[1], prof) and "replaces" in seen["rows"][1][0] and not seen["chosen"],
      "replacing is not pre-ticked: nothing changes until camB is ticked", seen)
_lens_attach.STATE["do"] = lambda dlg: _lens_attach.tick(dlg, 1)
w.act_load_lens.trigger()
pump(0.1)
_lens_attach.STATE["do"] = None
check(same(p.lenses[1], mod) and "camB" in _lens_attach.STATE["seen"][-1]["summary"],
      "ticking camB replaces it (the dialog said so)", _lens_attach.STATE["seen"][-1])

# both cameras have a profile: the export asks whose
ASK["items"].clear()
ASK["item"] = lambda items: items[0]                  # the first camera (GoPro's lens)
ASK["save"] = (os.path.join(OUT, "chosen.klens.json"), "Kinetrace lens (*.klens.json)")
w.act_export_lens.trigger()
pump(0.1)
chosen = lens.LensProfile.load(ASK["save"][0]) if os.path.exists(ASK["save"][0]) else None
check(ASK["items"] and len(ASK["items"][-1]) == 2 and chosen is not None and same(chosen, prof),
      "with profiles on both cameras it asks which, and writes the one chosen", ASK["items"])

w.project.dirty = False
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_lens_export: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
