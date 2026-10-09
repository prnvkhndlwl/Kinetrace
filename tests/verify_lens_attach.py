"""(G176) One lens file attached to several cameras at once: a rig of identical cameras in groups.

Through the window (offscreen, five cameras, no GPU): 3D -> Load a Lens Profile for Cameras... shows every
camera with what the file does to it -- fits, fits turned (a camera filmed on its side, I269), another
picture size (cannot be ticked), replaces a profile (never pre-ticked) -- with the working camera ticked;
"Tick every camera it fits" + unticking one attaches to exactly the ticked cameras (the turned one gets
the turned profile, the other group and the unticked camera keep what they had); Cancel changes
nothing; an Argus file with one line per camera gives each ticked camera its own line.

.venv\\Scripts\\python.exe tests\\verify_lens_attach.py
"""
import os
import shutil
import subprocess
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "lens_attach")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

ASK = {"open": "", "warn": []}
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QMessageBox.warning = staticmethod(lambda *a, **k: (ASK["warn"].append(str(a[2]) if len(a) > 2 else ""),
                                                    QMessageBox.Ok)[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
app = QApplication.instance() or QApplication([])
import _lens_attach  # noqa: E402

_lens_attach.install()
from kinetrace import lens  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)      # the console is cp1252
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def same(a, b):
    return (a is not None and b is not None and (a.width, a.height, a.fisheye) == (b.width, b.height, b.fisheye)
            and np.allclose(a.K, b.K, rtol=0, atol=1e-9) and np.allclose(np.ravel(a.dist), np.ravel(b.dist)))


def video(path, w, h, seed):
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for _ in range(10):
        vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    vw.release()
    return path


A1 = video(os.path.join(OUT, "A1.mp4"), 320, 240, 1)
A2 = video(os.path.join(OUT, "A2.mp4"), 320, 240, 2)
SIDE = os.path.join(OUT, "A3_side.mp4")
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-display_rotation", "90",
                "-i", video(os.path.join(OUT, "A3_level.mp4"), 320, 240, 3), "-c", "copy", SIDE], check=True)
B1 = video(os.path.join(OUT, "B1.mp4"), 640, 480, 4)
A4 = video(os.path.join(OUT, "A4.mp4"), 320, 240, 5)

w = MainWindow()
w.resize(1400, 900)
w.show()
w._open_video(A1)
for _ in range(300):
    pump(0.05)
    if w.state == READY and w.project is not None and not w._loading:
        break
pump(0.3)
check(all(w._add_view(v) for v in (A2, SIDE, B1, A4)), "five cameras: A1 A2 A3 (on its side) B1 A4")
pump(0.3)
p = w.project
w._apply_state()
check(w.act_load_lens.text().startswith("Load a Lens Profile for Cameras"), "3D menu: Load a Lens Profile for Cameras…")

K = np.array([[300.0, 0, 163.5], [0, 302.0, 117.0], [0, 0, 1]])
lensA = lens.LensProfile(320, 240, K, np.array([-0.2, 0.04, 0.001, -0.002, 0.0]), False, 0.3, 20,
                         "checkerboard (Kinetrace)", {"verdict": "good", "verdict_reasons": ["clean"]}, rotation=0)
own4 = lens.LensProfile(320, 240, K * 1.02, np.zeros(5), False, 0.4, 20, "A4's own", rotation=0)
w._set_lenses({4: own4})
pathA = lensA.save(os.path.join(OUT, "groupA.klens.json"))

# 1. Cancel: nothing changes
w._set_active_view(0)
pump(0.3)
ASK["open"] = pathA
_lens_attach.STATE["cancel"] = True
w.act_load_lens.trigger()
pump(0.1)
_lens_attach.STATE["cancel"] = False
seen = _lens_attach.STATE["seen"][-1]
check(p.lenses[0] is None and p.lenses[4] is own4, "Cancel: nothing is attached")
rows = seen["rows"]
check(seen["chosen"] == [0], "the working camera starts ticked, alone", seen["chosen"])
check(rows[0][0] == "fits" and rows[1][0] == "fits" and "turned" in rows[2][0] and rows[2][1],
      "rows say: fits, fits, fits turned (filmed on its side)", rows)
check(not rows[3][1] and "640" not in rows[3][0] and "320 x 240" in rows[3][0],
      "the other group's camera (640 x 480) cannot be ticked, and says what the file was measured on", rows[3])
check(rows[4][1] and "replaces" in rows[4][0], "a camera with its own profile: 'replaces' (can be ticked)", rows[4])


# 2. Tick every camera it fits, untick A4, attach
def tick_all_but_a4(dlg):
    _lens_attach.click(dlg.btn_all)
    _lens_attach.tick(dlg, 4)


_lens_attach.STATE["do"] = tick_all_but_a4
w.act_load_lens.trigger()
pump(0.1)
_lens_attach.STATE["do"] = None
seen = _lens_attach.STATE["seen"][-1]
check(seen["chosen"] == [0, 1, 2] and seen["ok"] == "Attach to 3 cameras", "ticked: A1 A2 A3 ('Attach to 3 cameras')",
      seen)
check(same(p.lenses[0], lensA) and same(p.lenses[1], lensA), "A1 and A2 get the profile")
check(same(p.lenses[2], lens.turn_profile(lensA, 270)) and (p.lenses[2].width, p.lenses[2].height) == (240, 320),
      "A3 (filmed on its side) gets it turned to its 240 x 320 pictures")
check(p.lenses[3] is None and p.lenses[4] is own4, "B1 (other group) and A4 (unticked) keep what they had")
check(p.dirty, "the project is changed (save to keep it)")

# 3. the button also ticks a camera it would replace; the summary says so
_lens_attach.STATE["do"] = lambda dlg: _lens_attach.click(dlg.btn_all)
_lens_attach.STATE["cancel"] = True
w.act_load_lens.trigger()
pump(0.1)
_lens_attach.STATE["do"], _lens_attach.STATE["cancel"] = None, False
seen = _lens_attach.STATE["seen"][-1]
check(seen["rows"][0][0] == "already has this profile" and not seen["rows"][0][1]
      and seen["rows"][2][0] == "already has this profile",
      "cameras that have it already (turned too) say so and are not ticked again", seen["rows"])
check(seen["chosen"] == [4] and "A4" in seen["summary"] and "Replaces" in seen["summary"],
      "'Tick every camera it fits' then ticks A4 alone, and the summary says it replaces A4's profile", seen)

# 4. an Argus file with one line per camera: each camera its own line
argus = os.path.join(OUT, "rig.txt")
with open(argus, "w") as fh:
    fh.write("1 310 320 240 160 120 1 -0.1 0.01 0 0 0\n2 320 320 240 161 119 1 -0.12 0.02 0 0 0\n")
ASK["open"] = argus
_lens_attach.STATE["do"] = lambda dlg: (_lens_attach.tick(dlg, 1))
w.act_load_lens.trigger()
pump(0.1)
_lens_attach.STATE["do"] = None
seen = _lens_attach.STATE["seen"][-1]
check("no line" in seen["rows"][3][0] and not seen["rows"][3][1], "a camera the file has no line for cannot be ticked",
      seen["rows"])
check(p.lenses[1] is not None and np.isclose(p.lenses[1].f_square, 320.0)
      and p.lenses[0] is not None and np.isclose(p.lenses[0].f_square, 301.0),
      "A2 got the file's camera-2 line; A1 (not ticked: it would replace) kept its profile",
      [None if l is None else l.f_square for l in p.lenses])

w.project.dirty = False
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_lens_attach: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
