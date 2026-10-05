"""(G143) Opening another video / project while the open one has unsaved work asks first, like
Quit (owner report 2026-10-03: Open Video replaced the work without a word; it only went to the
recovery folder).

Through the real window (offscreen, no GPU), Ctrl+O as a real key press:
  * nothing unsaved: no question, the new video opens;
  * Cancel (or Esc): nothing changes -- same video, still unsaved, its recovery copy kept;
  * Discard: the new video opens and the old work's recovery copy is gone (and not written back);
  * Save on a never-saved project: Save As, then the new video opens; a Save As that is cancelled
    opens nothing;
  * any other answer (a stubbed Yes): the old behaviour -- the work goes to the recovery folder;
  * the question names what is being opened; File -> Open Project and Open Folder of Videos ask too.

.venv\\Scripts\\python.exe tests\\verify_open_replace.py
"""
import os
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "open_replace")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox  # noqa: E402

from kinetrace import folderimport, recovery  # noqa: E402

ASK = {"answer": QMessageBox.Cancel, "texts": [], "open": "", "save": ""}


def _question(*a, **k):
    if (a[1] if len(a) > 1 else "") == "Save changes?":
        ASK["texts"].append(str(a[2]))
        return ASK["answer"]
    return QMessageBox.Yes


QMessageBox.question = staticmethod(_question)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (ASK["save"], ""))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: OUT)
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


def clip(name, n=40, shift=0):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    rng = np.random.RandomState(3 + shift)
    base = rng.randint(0, 255, (280, 320 + 4 * n, 3)).astype(np.uint8)
    for f in range(n):
        vw.write(np.ascontiguousarray(base[20:260, 4 * f:4 * f + 320]))
    vw.release()
    return path


A, B, C = clip("first.mp4"), clip("second.mp4", shift=1), clip("third.mp4", shift=2)


def settle(w, want=None):
    for _ in range(300):
        pump(0.05)
        if (w.state == READY and w.project is not None and not w._loading
                and (want is None or os.path.normcase(w.session.video_path) == os.path.normcase(want))):
            break
    pump(0.3)


def video(w):
    return os.path.basename(w.session.video_path)


def make_dirty(w):
    w.session.add_empty_point()
    w.session.dirty = True
    w._autosave(wait=True)
    pump(0.3)


def ctrl_o(w):
    w.activateWindow()
    w.setFocus()
    QTest.keyClick(w, Qt.Key_O, Qt.ControlModifier)
    pump(0.2)


w = MainWindow()
w.resize(1400, 900)
w.show()
w._open_video(A)
settle(w, A)
check(video(w) == "first.mp4", "first.mp4 open")

print("[1] nothing unsaved: no question")
ASK["texts"].clear()
ASK["open"] = B
ctrl_o(w)
settle(w, B)
check(video(w) == "second.mp4" and not ASK["texts"], "Ctrl+O opened second.mp4 without asking",
      f"{video(w)} {ASK['texts']}")

print("[2] Cancel keeps everything")
make_dirty(w)
pid = w._project_id
check(os.path.exists(recovery.paths(pid)[0]), "the unsaved work has its recovery copy")
ASK.update(answer=QMessageBox.Cancel, open=C)
ASK["texts"].clear()
ctrl_o(w)
settle(w)
check(len(ASK["texts"]) == 1 and "before opening third.mp4" in ASK["texts"][0],
      "the question was asked once and names third.mp4", f"{ASK['texts']}")
check(video(w) == "second.mp4" and w.project.dirty and w._project_id == pid,
      "Cancel: the same video, still unsaved, the same project", f"{video(w)} dirty={w.project.dirty}")
check(os.path.exists(recovery.paths(pid)[0]), "Cancel: the recovery copy is kept")

print("[3] Discard drops the unsaved work and opens the new video")
ASK.update(answer=QMessageBox.Discard, open=C)
ctrl_o(w)
settle(w, C)
pump(0.5)
check(video(w) == "third.mp4", "Discard: third.mp4 is open", video(w))
check(not os.path.exists(recovery.paths(pid)[0]), "Discard: the old work's recovery copy is gone (not written back)")
check(not any(r.get("project_id") == pid for r in recovery.scan()), "Discard: nothing of it in the recovery folder")

print("[4] Save: Save As first, then the new video")
make_dirty(w)
pid = w._project_id
ASK.update(answer=QMessageBox.Save, open=A, save="")
ctrl_o(w)
settle(w)
check(video(w) == "third.mp4" and w.project.dirty, "Save with Save As cancelled: nothing opens, still unsaved",
      f"{video(w)} dirty={w.project.dirty}")
target = os.path.join(OUT, "saved.kinetrace")
ASK.update(answer=QMessageBox.Save, open=A, save=target)
ctrl_o(w)
settle(w, A)
check(os.path.isfile(os.path.join(target, "kinetrace.json")), "Save: the project was saved", target)
check(video(w) == "first.mp4", "Save: then first.mp4 opened", video(w))

print("[5] any other answer keeps the old behaviour (the work to the recovery folder)")
make_dirty(w)
pid = w._project_id
ASK.update(answer=QMessageBox.Yes, open=B)
ctrl_o(w)
settle(w, B)
check(video(w) == "second.mp4" and os.path.exists(recovery.paths(pid)[0]),
      "a stubbed Yes: second.mp4 opened, the work kept in recovery", video(w))

print("[6] Open Project and Open Folder of Videos ask too")
make_dirty(w)
ASK.update(answer=QMessageBox.Cancel, open=os.path.join(target, "kinetrace.json"))
ASK["texts"].clear()
w.act_open_proj.trigger()
settle(w)
check(len(ASK["texts"]) == 1 and "before opening saved.kinetrace" in ASK["texts"][0]
      and video(w) == "second.mp4", "Open Project: asked, Cancel opened nothing", f"{ASK['texts']} {video(w)}")


class _Dlg:
    result_paths = [A]
    result_project = None

    def __init__(self, *a, **k):
        pass

    def exec(self):
        return QDialog.Accepted

    def stop_probe(self):
        pass

    def deleteLater(self):
        pass


saved_dlg = folderimport.VideoFolderDialog
folderimport.VideoFolderDialog = _Dlg
try:
    ASK["texts"].clear()
    w.act_open_folder.trigger()
    settle(w)
finally:
    folderimport.VideoFolderDialog = saved_dlg
check(len(ASK["texts"]) == 1 and "before opening the videos of open_replace" in ASK["texts"][0]
      and video(w) == "second.mp4", "Open Folder of Videos: asked, Cancel opened nothing", f"{ASK['texts']} {video(w)}")

ASK["answer"] = QMessageBox.Discard
w.close()
pump(0.3)
w._dev_probe.wait(10000)
print("\nverify_open_replace: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
