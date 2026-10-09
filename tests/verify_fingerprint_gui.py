"""(I266) Through the real window (offscreen, no GPU), with the user's gestures:

  [1] Ctrl+O a video: its frame fingerprint is made in the background, without marking the project
      changed; Ctrl+Shift+S saves it in the camera's folder
  [2] the video is replaced by one whose stream starts one frame later (an EDIT LIST, as phone files
      carry: same frame count, so the count check sees nothing); File -> Open Project: the check runs
      by itself and the evidence dialog opens -- shifted by -1, the sentence, the saved picture beside
      this computer's N-1 / N / N+1 with the best match marked; Next moment and Go to this frame
      work; nothing in the project is changed or re-indexed
  [3] a video whose timestamps skip a frame (a dropped frame): the frame typed into the frame box is
      the picture reading forward gives (the old seek showed the frame before it), and a warning says
      why
  [4] a still video: File -> Check Video Frames says it cannot tell and how little it matters (px);
      the frame on screen is added to the fingerprint with the dialog's button

.venv\\Scripts\\python.exe tests\\verify_fingerprint_gui.py
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
OUT = os.path.join(HERE, "out", "fingerprint_gui")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from kinetrace import framecheck, video_source  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
ASK = {"open": "", "save": ""}
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard if len(a) > 1 and a[1] == "Save changes?"
                                    else QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (ASK["save"], ""))
app = QApplication.instance() or QApplication([])
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


# ------------------------------------------------------------------ footage
W, H = 480, 270
_rng = np.random.RandomState(4)
_BG = cv2.GaussianBlur(_rng.randint(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), 2.0)
_TEX = cv2.GaussianBlur(_rng.randint(0, 255, (90, 90, 3)).astype(np.uint8), (0, 0), 1.5)


def picture(i, still=False):
    img = _BG.copy()
    if not still:
        x, y = 20 + (5 * i) % (W - 120), 60 + int(30 * np.sin(i * 0.07))
        img[y:y + 90, x:x + 90] = _TEX
    return img


def encode(path, frames, extra=()):
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "30",
                          "-bf", "2", "-pix_fmt", "yuv420p", *extra, path], stdin=subprocess.PIPE)
    for img in frames:
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


N = 150
VID = encode(os.path.join(OUT, "cam.mp4"), (picture(i) for i in range(N)))
LONG = encode(os.path.join(OUT, "cam_plus1.mp4"), (picture(i) for i in range(N + 1)))
SHIFTED = os.path.join(OUT, "cam_editlist.mp4")
subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{1 / 30:.6f}", "-i", LONG, "-c", "copy", SHIFTED], check=True)
GAP = encode(os.path.join(OUT, "gap.mp4"), (picture(i) for i in range(200)),
             extra=("-vf", "setpts=(N+gte(N\\,100))/(30*TB)", "-fps_mode", "passthrough", "-video_track_timescale",
                    "30000"))
STILL = encode(os.path.join(OUT, "still.mp4"), (picture(i, still=True) for i in range(60)))


class Shown:
    dialogs = []


class _Dlg(framecheck.FrameCheckDialog):
    """The real dialog (not modal: the main window stays usable for the hand check), recorded when it is
    shown; the test drives it with clicks."""

    def exec(self):                 # it must never block: the evidence is shown beside the work
        raise AssertionError("the frame-check window must not be modal")

    def show(self):
        Shown.dialogs.append(self)
        super().show()


framecheck.FrameCheckDialog = _Dlg


def settle(w, want=None):
    for _ in range(400):
        pump(0.05)
        if (w.state == READY and w.project is not None and not w._loading
                and (want is None or os.path.normcase(w.session.video_path) == os.path.normcase(want))):
            break
    pump(0.2)


def checks_done(w, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        pump(0.05)
        if w._fc_job is None and not w._fc_queue:
            pump(0.2)
            return True
    return False


def key(w, k, mod=Qt.NoModifier):
    w.activateWindow()
    app.setActiveWindow(w)          # offscreen: the user's click on the main window
    w.setFocus()
    QTest.keyClick(w, k, mod)
    pump(0.2)


w = MainWindow()
w.resize(1400, 900)
w.show()
TOASTS = []
_show = w.toast.show_message
w.toast.show_message = lambda text, *a, **k: (TOASTS.append(text), _show(text, *a, **k))

print("[1] a new video gets its fingerprint; Save keeps it")
ASK["open"] = VID
w.act_open.trigger()                 # File -> Open Video (Ctrl+O once a window has focus)
settle(w, VID)
check(checks_done(w), "the background check ends")
s = w.session
check(s.fingerprint is not None and len(s.fingerprint.marks) == 8, "the fingerprint is made by itself (8 moments)",
      str(s.fingerprint))
check(not w.project.dirty, "making it does not mark the project changed")
check(w._views[0].seek_plan is not None and w._views[0].seek_plan.exact, "the file's seeks were checked: exact")
w.session.add_point(10, 100.0, 100.0, name="snout")
w.session.tracks[20:40, 0] = (150.0, 120.0)
w.session.tracked[20:40, 0] = True
w.session.dirty = True
before = w.session.tracks.copy()
proj = os.path.join(OUT, "study.kinetrace")
ASK["save"] = proj
w.act_save_as.trigger()              # File -> Save Project As
pump(1.0)
camdir = os.path.join(proj, "cameras", w.project.name(0))
fpdir = os.path.join(camdir, "fingerprint")
check(os.path.isfile(os.path.join(fpdir, "meta.json")) and os.path.isfile(os.path.join(fpdir, "crops.npy")),
      "Save As writes cameras/<camera>/fingerprint/", str(os.listdir(camdir) if os.path.isdir(camdir) else proj))

print("[2] the video is replaced by one that starts a frame later; Open Project shows it")
ASK["open"] = STILL
w.act_open.trigger()
settle(w, STILL)
checks_done(w)
shutil.copyfile(SHIFTED, VID)                  # same name, same frame count, pictures one frame later
cnt = 0
cap = cv2.VideoCapture(VID)
while cap.grab():
    cnt += 1
cap.release()
check(cnt == N, "the replaced video has the same frame count", str(cnt))
Shown.dialogs.clear()
ASK["open"] = os.path.join(proj, "kinetrace.json")
w.act_open_proj.trigger()
settle(w, VID)
check(checks_done(w), "the check ends")
pump(0.5)
check(len(Shown.dialogs) == 1, "the evidence dialog opens by itself", str(len(Shown.dialogs)))
if Shown.dialogs:
    d = Shown.dialogs[-1]
    r = d.result
    check(r.verdict == "shifted" and r.shift == -1, "shifted by -1", f"{r.verdict} {r.shift} {r.detail}")
    check("1 frame earlier" in r.sentence and "cam" in r.sentence and "Nothing was changed" in d.sentence.text(),
          "one plain sentence names the camera and the shift", d.sentence.text())
    check(d.chip.text() == "POOR", "with a verdict", d.chip.text())
    check("best match" in d.caps[1].text() and "best match" not in d.caps[2].text(),
          "the saved picture beside here N-1 / N / N+1, the best match (N-1) marked", d.caps[1].text())
    check("Difference" in d.caps[4].text() and d.pics[4].pixmap() is not None, "with a difference view")
    k0 = d.k
    QTest.mouseClick(d.btn_next, Qt.LeftButton)
    pump(0.1)
    check(d.k == k0 + 1 and f"Moment {k0 + 2} of" in d.where.text(), "Next moment shows the next one", d.where.text())
    QTest.mouseClick(d.btn_go, Qt.LeftButton)
    pump(0.5)
    check(w.current == r.marks[d.k].frame, "Go to this frame goes there in the main window",
          f"{w.current} vs {r.marks[d.k].frame}")
    f0 = w.current
    key(w, Qt.Key_Right)
    check(d.isVisible() and w.current == f0 + 1, "the window stays open while the main window steps a frame",
          f"visible={d.isVisible()} {f0} -> {w.current}")
    d.close()
check(np.array_equal(w.session.tracks, before, equal_nan=True), "the tracks are untouched (never re-indexed)")
check(not w.project.dirty, "the project is not changed")

print("[3] a file whose timestamps skip a frame")
ref = []
cap = cv2.VideoCapture(GAP)
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    ref.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
cap.release()
plain = video_source.VideoSource(GAP)
old_wrong = [t for t in (150, 190) if int(np.argmin([np.abs(x - cv2.cvtColor(plain.get_frame(t), cv2.COLOR_RGB2GRAY)
                                                        ).mean() for x in ref])) != t]
plain.close()
TOASTS.clear()
ASK["open"] = GAP
w.act_open.trigger()
settle(w, GAP)
checks_done(w)
plan = w._views[0].seek_plan
target = old_wrong[0] if old_wrong else 190
w.spin.setFocus()
QTest.mouseDClick(w.spin.lineEdit(), Qt.LeftButton)
w.spin.lineEdit().selectAll()
QTest.keyClicks(w.spin.lineEdit(), str(target))
QTest.keyClick(w.spin.lineEdit(), Qt.Key_Return)
for _ in range(100):
    pump(0.05)
    if w.current == target and w.canvas._raw_rgb is not None and w._views[0].want_frame in (None, target):
        break
pump(0.5)
shown = w.canvas._raw_rgb
got = int(np.argmin([np.abs(x - cv2.cvtColor(shown, cv2.COLOR_RGB2GRAY)).mean() for x in ref])) if shown is not None else None
check(w.current == target and got == target, f"frame {target} typed in shows the picture of frame {target}",
      f"current {w.current}, picture of {got}; the plain seek showed {old_wrong}")
if old_wrong:
    check(plan is not None and not plan.exact and any("skip 1 frame" in t for t in TOASTS),
          "a warning says why (the timestamps skip a frame)", str(TOASTS[-3:]))

print("[4] a still video: cannot tell, and the fallback")
ASK["open"] = STILL
w.act_open.trigger()
settle(w, STILL)
checks_done(w)
check(w.session.fingerprint is not None and w.session.fingerprint.still, "its fingerprint knows nothing moves")
for _ in range(7):
    key(w, Qt.Key_Right)
check(w.current >= 3, "stepped forward with the Right key", str(w.current))
Shown.dialogs.clear()
w.act_check_frames.trigger()
checks_done(w)
pump(0.5)
check(len(Shown.dialogs) == 1, "File -> Check Video Frames opens the dialog", str(len(Shown.dialogs)))
if Shown.dialogs:
    d = Shown.dialogs[-1]
    check(d.result.verdict == "cannot_tell" and "px" in d.result.sentence and d.chip.text() == "GOOD",
          "cannot tell, how little it matters in px, GOOD", f"{d.result.verdict} {d.chip.text()} {d.result.sentence}")
    check(d.btn_add is not None, "the computer that made it may add the frame on screen")
    n0 = len(w.session.fingerprint.marks)
    if d.btn_add is not None:
        QTest.mouseClick(d.btn_add, Qt.LeftButton)
        pump(1.0)
        fp = w.session.fingerprint
        check(len(fp.marks) == n0 + 1 or any(m.picked == "user" for m in fp.marks),
              "the frame on screen is in the fingerprint", str([(m.frame, m.picked) for m in fp.marks]))

w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_fingerprint_gui: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
