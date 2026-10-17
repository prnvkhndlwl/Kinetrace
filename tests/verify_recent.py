"""(G205) File -> Open Recent and the start screen's list of recent projects (owner 2026-10-16: "everytime I
open the app, i have to navigate through a bunch of folders to launch a saved project"; "make sure that this
option is OS agnostic and would still work if the user changes the project location").

[1] `kinetrace.recent` alone (no window): newest first, at most 10, one entry per project (the same path in
    another case on Windows; the same project id at a path that is gone = a MOVED project); a copied folder
    (same id, both still there) keeps both; Windows- and POSIX-style paths round-trip through the stored JSON;
    a damaged value is an empty list; where Locate... opens; a stalled existence check gives "unknown" in time.

.venv\\Scripts\\python.exe tests\\verify_recent.py
"""
import os
import shutil
import sys
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "recent")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from kinetrace import recent  # noqa: E402

# every dialog is stubbed: an unstubbed modal blocks the test
ASK = {"open": "", "save": "", "button": None, "boxes": []}


def _question(*a, **k):
    return QMessageBox.Discard if (a[1] if len(a) > 1 else "") == "Save changes?" else QMessageBox.Yes


def _exec(self):
    ASK["boxes"].append((self.windowTitle(), self.text() + "\n" + self.informativeText()))
    for b in self.buttons():
        if ASK["button"] is not None and b.text().replace("&", "") == ASK["button"]:
            b.click()
            break
    return 0


QMessageBox.exec = _exec
QMessageBox.question = staticmethod(_question)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: ASK["boxes"].append(("critical", str(a[1:3]))) or QMessageBox.Ok)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"], ""))
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (ASK["save"], ""))
app = QApplication.instance() or QApplication([])
from kinetrace.app import IDLE, READY, MainWindow  # noqa: E402

FAILS = []


def check(cond, what, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + what + (f"  ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(what)


class FakeSettings:
    def __init__(self):
        self.d = {}

    def value(self, key, default=None):
        return self.d.get(key, default)

    def setValue(self, key, value):      # noqa: N802 - the QSettings name
        self.d[key] = value


def unit():
    print("[1] the list on its own")
    a, b = os.path.join(OUT, "a.kinetrace"), os.path.join(OUT, "b.kinetrace")
    for p in (a, b):
        os.makedirs(p, exist_ok=True)
    items = recent.touch([], a, "ida")
    items = recent.touch(items, b, "idb")
    check([it["name"] for it in items] == ["b.kinetrace", "a.kinetrace"], "newest first")
    items = recent.touch(items, a, "ida")
    check([it["name"] for it in items] == ["a.kinetrace", "b.kinetrace"], "opening again moves it to the top, once")
    many = []
    for k in range(15):
        many = recent.touch(many, os.path.join(OUT, f"p{k}.kinetrace"), f"id{k}")
    check(len(many) == recent.MAX == 10 and many[0]["name"] == "p14.kinetrace", "at most 10 kept, newest first")
    if os.name == "nt":
        items2 = recent.touch(items, a.upper(), "ida")
        check(len(items2) == 2, "Windows: the same folder in another case is the same entry")

    # a MOVED project: same id, the listed path is gone -> one entry, at the new place
    moved = os.path.join(OUT, "moved", "a.kinetrace")
    os.makedirs(os.path.dirname(moved), exist_ok=True)
    gone = {recent.entry(a, "")["path"]}
    items3 = recent.touch(items, moved, "ida", exists=lambda p: p not in gone)
    paths = [it["path"] for it in items3]
    check(len(items3) == 2 and paths[0] == recent.entry(moved, "")["path"],
          "a moved project (same id, old path gone) replaces its entry")
    # a COPIED project: same id, both paths still there -> two entries
    items4 = recent.touch(items, moved, "ida", exists=lambda p: True)
    check(len(items4) == 3, "a copied folder (same id, both still there) keeps both entries")

    # round trip through the stored JSON: this OS's own path style, spaces and a non-ASCII folder name
    st = FakeSettings()
    native = r"D:\data\fish\trial 1.kinetrace" if os.name == "nt" else "/home/lab/data/trial 1.kinetrace"
    keep = [recent.entry(native, "n1"), recent.entry(os.path.join(OUT, "Lézard ñ", "t2.kinetrace"), "u1")]
    check(recent.store(st, keep), "stored")
    check(all("\\" not in it["path"] for it in recent.json.loads(st.d[recent.KEY])),
          "paths stored with forward slashes only")
    back = recent.load(st)
    check([(it["path"], it["id"], it["name"]) for it in back] == [(it["path"], it["id"], it["name"]) for it in keep],
          "paths with spaces and non-ASCII names round-trip unchanged")
    check(back[0]["name"] == "trial 1.kinetrace" and recent.Path(back[0]["path"]) == recent.Path(native),
          f"a {'Windows' if os.name == 'nt' else 'POSIX'} path reads back as the same folder")
    for bad in ("{not json", "42", '{"path": 1}', '[{"nopath": 1}, 7, {"path": ""}]'):
        st.d[recent.KEY] = bad
        check(recent.load(st) == [], f"a damaged value is an empty list: {bad!r}")
    check(recent.load(FakeSettings()) == [], "no value yet: an empty list")
    st.d[recent.KEY] = recent.json.dumps([recent.entry(a, "1"), recent.entry(a, "1")])
    check(len(recent.load(st)) == 1, "a value listing one project twice reads as one entry")
    recent.clear(st)
    check(recent.load(st) == [], "Clear List empties it")
    check(len(recent.remove(items, a)) == 1, "Remove from List drops exactly that entry")

    # where Locate... opens
    deep = os.path.join(OUT, "moved", "gone", "deeper", "x.kinetrace")
    check(recent.nearest_folder(deep) == recent.Path(os.path.join(OUT, "moved")),
          "Locate opens in the nearest folder that still exists")
    check(recent.nearest_folder("Q:/nothing/x.kinetrace", isdir=lambda p: False) is None,
          "no folder of it left (an unplugged drive): None")

    # a stalled check (an unplugged network share) does not hold the menu up
    release = threading.Event()

    def slow(p):
        if "slow" in p:
            release.wait(5)
        return True

    t0 = time.monotonic()
    got = recent.exists_quick(["fast", "slow"], timeout=0.3, check=slow)
    dt = time.monotonic() - t0
    release.set()
    check(got == {"fast": True, "slow": None} and dt < 1.0,
          f"a stalled existence check is 'unknown' within the time limit ({dt:.2f} s)")


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def clip(name, n=30):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    base = np.random.RandomState(5).randint(0, 255, (280, 320 + 4 * n, 3)).astype(np.uint8)
    for f in range(n):
        vw.write(np.ascontiguousarray(base[20:260, 4 * f:4 * f + 320]))
    vw.release()
    return path


def settle(w, want_project=None):
    for _ in range(300):
        pump(0.05)
        if (w.state == READY and w.project is not None and not w._loading
                and (want_project is None or (w.project_path is not None
                                              and os.path.normcase(str(w.project_path.resolve()))
                                              == os.path.normcase(os.path.abspath(want_project))))):
            break
    pump(0.3)


def window():
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    pump(0.3)
    return w


def close(w):
    w.close()
    pump(0.3)
    w._dev_probe.wait(10000)


def menu_entries(w):
    """File → Open Recent as it opens (Qt emits aboutToShow before showing it): {text: action}."""
    w.m_recent.aboutToShow.emit()
    return {a.text(): a for a in w.m_recent.actions() if not a.isSeparator()}


def listed(w):
    return [it["path"] for it in recent.load(w._settings())]


def posix(p):
    return recent.entry(p, "")["path"]


VID = clip("clip.mp4")
P1 = os.path.join(OUT, "first.kinetrace")


def gui_menu():
    print("[2] File -> Open Recent")
    w = window()
    recent.clear(w._settings())             # the offscreen test settings file is shared by the suites
    check(list(menu_entries(w)) == ["No recent projects", "&Clear List"]
          and not menu_entries(w)["&Clear List"].isEnabled(), "an empty list says so; Clear List greyed")
    w._open_video(VID)
    settle(w)
    w.session.add_empty_point()
    w.session.dirty = True
    ASK["save"] = P1
    w.activateWindow()
    QTest.keyClick(w, Qt.Key_S, Qt.ControlModifier)        # the user's Ctrl+S: Save As the first time
    pump(0.5)
    check(os.path.isfile(os.path.join(P1, "kinetrace.json")), "Ctrl+S saved the project")
    check(listed(w) == [posix(P1)], "a saved project is on the list", f"{listed(w)}")
    close(w)

    w = window()                            # the next start of the program
    ents = menu_entries(w)
    check("first.kinetrace" in ents, "after a restart it is in File -> Open Recent", f"{list(ents)}")
    check(ents["first.kinetrace"].toolTip().startswith(str(recent.Path(P1))), "its tooltip is the full path")
    check(w.m_recent.isEnabled(), "Open Recent is available when nothing is tracking")
    ents["first.kinetrace"].trigger()
    settle(w, P1)
    check(w.project_path is not None and os.path.normcase(str(w.project_path.resolve()))
          == os.path.normcase(os.path.abspath(P1)), "one click opened it", f"{w.project_path}")
    check(listed(w) == [posix(P1)], "opening it again keeps one entry")
    menu_entries(w)["&Clear List"].trigger()
    check(listed(w) == [] and "No recent projects" in menu_entries(w), "Clear List empties it")
    w._remember_project(w.project_path, w._project_id)      # back on the list for the next parts
    close(w)


unit()
gui_menu()

if FAILS:
    print(f"verify_recent FAILED ({len(FAILS)}): " + "; ".join(FAILS))
    sys.exit(1)
print("verify_recent PASSED")
