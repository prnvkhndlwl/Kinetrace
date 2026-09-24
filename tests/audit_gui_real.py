"""Real-event GUI audit (release sweep 2026-09-22).

Drives Kinetrace the way a user does -- QTest mouse clicks on the menu bar,
the toolbar, the panels, the canvas and the timeline, QTest key presses on
whichever widget holds the focus -- on the REAL Windows platform (fonts,
native popups; offscreen has no fonts), and screenshots every window,
dialog, wizard page and menu it meets.

Checked automatically (the rest goes to the screenshot / message review):
  * nothing raises (sys.excepthook + per-step try/except);
  * every enabled menu entry and toolbar button is reached by a real click;
  * menus show their tooltips (QMenu.toolTipsVisible);
  * a hotkey acts the same whichever widget holds the focus;
  * the view never moves on its own (canvas transform + scroll bars compared
    around every step that is not a view command);
  * a pause during tracking lands in < 1 s;
  * every message box / dialog / wizard page text is logged;
  * controls are inside the window and not clipped at 1600x1000 / 1366x768.

Outputs in tests/out/gui_real/: one PNG per distinct window / dialog / menu,
records.json (every step), report.txt (the automatic checks). Media it
builds lives in tests/out/gui_real_media/ (kept between runs).

Run: .venv\\Scripts\\python.exe tests\\audit_gui_real.py [--no-gpu] [--only=idle,video,...]
States: idle video points balls segment tracking twocams calibrated body
"""
import faulthandler
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import traceback

faulthandler.enable()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(errors="replace")
if "--offscreen" in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
NO_GPU = "--no-gpu" in sys.argv
ONLY = None
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = set(_a.split("=", 1)[1].split(","))

OUT = os.path.join(ROOT, "tests", "out", "gui_real")
MEDIA = os.path.join(ROOT, "tests", "out", "gui_real_media")
if os.path.isdir(OUT):
    shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(MEDIA, exist_ok=True)

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QAction, QContextMenuEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QCheckBox,  # noqa: E402
                               QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGroupBox,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QMenu, QMenuBar,
                               QMessageBox, QPlainTextEdit, QRadioButton, QTextEdit, QToolButton,
                               QWidget, QWizard)

import _synth  # noqa: E402

# ------------------------------------------------------------------ media
VID = os.path.join(MEDIA, "gui_animals.mp4")
if not os.path.exists(VID):
    _synth.build_video(VID)
VID_B = os.path.join(MEDIA, "gui_animals_cam2.mp4")
if not os.path.exists(VID_B):
    shutil.copy(VID, VID_B)
GT0 = _synth.gt_frame(0)[1][0]          # animal 0 on frame 0: head / eye / tail / feet / centre
# --video=<file>: run on a COPY of real footage (never the original: autosaves land next
# to the video). Click positions then come from fractions of the picture, not the
# synthetic animal's ground truth.
for _a in sys.argv:
    if _a.startswith("--video="):
        _src = _a.split("=", 1)[1]
        _dst = os.path.join(MEDIA, "real_" + os.path.basename(_src))
        if not os.path.exists(_dst) or os.path.getsize(_dst) != os.path.getsize(_src):
            shutil.copy(_src, _dst)
        VID = _dst
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(VID)
        _w, _h = _cap.get(_cv2.CAP_PROP_FRAME_WIDTH), _cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)
        _cap.release()
        GT0 = {"head": np.array([0.45 * _w, 0.45 * _h]), "tip": np.array([0.55 * _w, 0.55 * _h]),
               "centre": np.array([0.5 * _w, 0.5 * _h]), "feet": [np.array([0.6 * _w, 0.4 * _h])]}


def _clean_autosaves():
    for p in (VID, VID_B):
        if os.path.exists(p + ".cotracker.npz"):
            os.remove(p + ".cotracker.npz")
    for f in os.listdir(MEDIA):
        if f.endswith(".cotracker.npz"):
            try:
                os.remove(os.path.join(MEDIA, f))
            except OSError:
                pass


_clean_autosaves()

# --------------------------------------------------------------- records
REC: list[dict] = []
EXC: list[dict] = []
CTX = {"state": "start", "step": "-"}
SHOT_N = [0]
SEEN_SHOTS: dict[str, str] = {}


def rec(kind, **kw):
    d = {"state": CTX["state"], "step": CTX["step"], "kind": kind}
    d.update(kw)
    REC.append(d)
    return d


def _hook(tp, val, tb):
    tail = "".join(traceback.format_exception(tp, val, tb))[-2500:]
    EXC.append({"state": CTX["state"], "step": CTX["step"], "tb": tail})
    print(f"EXCEPTION [{CTX['state']}] {CTX['step']}\n{tail}", flush=True)


sys.excepthook = _hook
faulthandler.dump_traceback_later(int(os.environ.get("AUDIT_DUMP_S", "180")), repeat=True)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)[:48].strip("_") or "x"


def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>|</li>|</tr>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def shot(w, name, key=None):
    """Grab `w` to a PNG. `key` de-duplicates identical windows (same text)."""
    if key is not None:
        h = hashlib.md5(key.encode("utf-8", "replace")).hexdigest()[:12]
        if h in SEEN_SHOTS:
            return SEEN_SHOTS[h]
    SHOT_N[0] += 1
    fn = f"{SHOT_N[0]:04d}_{slug(CTX['state'])}_{slug(name)}.png"
    try:
        w.grab().save(os.path.join(OUT, fn))
    except Exception:                    # noqa: BLE001
        fn = None
    if key is not None:
        SEEN_SHOTS[h] = fn
    return fn


def widget_texts(w) -> list[str]:
    out = []
    for c in [w] + w.findChildren(QWidget):
        try:
            if c is not w and not c.isVisibleTo(w):
                continue
            if isinstance(c, QLabel) and c.text().strip():
                out.append(strip_html(c.text()))
            elif isinstance(c, (QTextEdit, QPlainTextEdit)):
                t = c.toPlainText().strip()
                if t:
                    out.append(t[:6000])
            elif isinstance(c, (QCheckBox, QRadioButton)) and c.text().strip():
                out.append(("[x] " if c.isChecked() else "[ ] ") + c.text())
            elif isinstance(c, QAbstractButton) and c.text().strip():
                out.append(f"<{c.text()}{'' if c.isEnabled() else ' (disabled)'}>")
            elif isinstance(c, QComboBox):
                out.append("{" + c.currentText() + "}")
            elif isinstance(c, QGroupBox) and c.title():
                out.append("## " + c.title())
        except RuntimeError:
            continue
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ------------------------------------------------------------- file dialogs
# Native OS file dialogs are not Qt widgets: they cannot be grabbed or clicked
# by QTest, and they are not the app's code. The statics answer from STUB.
STUB = {"open": "", "save": None, "dir": ""}
SAVE_DIR = os.path.join(OUT, "saved")
os.makedirs(SAVE_DIR, exist_ok=True)


def _save_name(*a, **k):
    cap = str(a[1]) if len(a) > 1 else str(k.get("caption", ""))
    filt = str(a[3]) if len(a) > 3 else str(k.get("filter", ""))
    rec("filedialog", which="save", caption=cap, filter=filt)
    if STUB["save"] is None:
        return "", ""
    m = re.search(r"\*\.([A-Za-z0-9.]+)", filt)
    ext = m.group(1) if m else "dat"
    return os.path.join(SAVE_DIR, f"{STUB['save']}.{ext}"), filt.split(";;")[0]


def _open_name(*a, **k):
    cap = str(a[1]) if len(a) > 1 else str(k.get("caption", ""))
    filt = str(a[3]) if len(a) > 3 else str(k.get("filter", ""))
    rec("filedialog", which="open", caption=cap, filter=filt)
    return (STUB["open"], filt) if STUB["open"] else ("", "")


QFileDialog.getSaveFileName = staticmethod(_save_name)
QFileDialog.getOpenFileName = staticmethod(_open_name)
QFileDialog.getOpenFileNames = staticmethod(
    lambda *a, **k: (rec("filedialog", which="open-many") and None) or (([STUB["open"]] if STUB["open"] else []), ""))
QFileDialog.getExistingDirectory = staticmethod(
    lambda *a, **k: (rec("filedialog", which="dir") and None) or STUB["dir"])

app = QApplication.instance() or QApplication([])
from cotracker_app.app import IDLE, READY, TRACKING, MainWindow  # noqa: E402

win = MainWindow()
win.resize(1600, 1000)
win.move(30, 30)
win.show()
_KEEP: list = []                         # wrappers from menu.actions(): never let Python free them


def pump(sec=0.12):
    t0 = time.time()
    while True:
        app.processEvents()
        if time.time() - t0 >= sec:
            break
        time.sleep(0.005)


def wait(cond, timeout, what):
    t0 = time.time()
    while not cond():
        pump(0.03)
        if time.time() - t0 > timeout:
            rec("timeout", what=what)
            print("TIMEOUT:", what, flush=True)
            return False
    return True


def activate():
    win.raise_()
    win.activateWindow()
    try:
        app.setActiveWindow(win)
    except Exception:                    # noqa: BLE001
        pass
    pump(0.05)


# ---------------------------------------------------------------- watchdog
# Modal dialogs and exec()'d popups block inside the click that opened them;
# this timer runs INSIDE their event loop, records + screenshots them and then
# answers through QTimer.singleShot (never blocking inside the watchdog, so a
# dialog opened BY the answer is handled too).
POLICY = {"mode": "cancel", "handler": None, "menu_pick": None, "text": "audit text",
          "answers": [], "wizard_pages": 12}
WATCH = {"busy": False, "ignore_popups": False, "wizard": {}}


_PENDING: list = []


def later(fn):
    """singleShot(0, fn) that keeps `fn` alive until it has run: PySide6
    dropped bare lambdas handed to singleShot, so a message box the watchdog
    'answered' stayed open forever (the harness hung on the first one)."""
    def run(f=fn):
        try:
            f()
        finally:
            if run in _PENDING:
                _PENDING.remove(run)
    _PENDING.append(run)
    QTimer.singleShot(0, run)


def _mb_answer(mb: QMessageBox):
    btns = mb.buttons()
    want = POLICY["answers"].pop(0) if POLICY["answers"] else None
    if want:
        for b in btns:
            if want.lower() in b.text().replace("&", "").lower():
                return b
    accept = POLICY["mode"] == "accept"
    roles_yes = (QMessageBox.YesRole, QMessageBox.AcceptRole, QMessageBox.ApplyRole)
    roles_no = (QMessageBox.NoRole, QMessageBox.RejectRole)
    for b in btns:
        if (mb.buttonRole(b) in roles_yes) == accept and mb.buttonRole(b) in (roles_yes + roles_no):
            return b
    return mb.escapeButton() or mb.defaultButton() or (btns[0] if btns else None)


def _handle_modal(m):
    title = m.windowTitle()
    if isinstance(m, QMessageBox):
        texts = [strip_html(m.text()), strip_html(m.informativeText()), m.detailedText() or ""]
        texts = [t for t in texts if t]
        btns = [b.text().replace("&", "") for b in m.buttons()]
        png = shot(m, "msg_" + title, key="msg|" + title + "|" + "|".join(texts))
        b = _mb_answer(m)
        rec("messagebox", title=title, text="\n".join(texts), buttons=btns, png=png,
            answered=b.text().replace("&", "") if b is not None else None)
        m.setProperty("audit_handled", True)
        if os.environ.get('AUDIT_DEBUG'):
            print('MB answer', repr(b.text() if b is not None else None), 'buttons', btns, flush=True)
        def _click(bb=b, mm=m):
            if os.environ.get('AUDIT_DEBUG'):
                print('MB click now', repr(bb.text()) if bb is not None else None, 'visible', mm.isVisible(), flush=True)
            (bb.click() if bb is not None else mm.reject())
        later(_click)
        return
    if isinstance(m, QWizard):
        _handle_wizard(m)
        return
    if isinstance(m, QInputDialog):
        png = shot(m, "input_" + title, key="input|" + title + "|" + m.labelText())
        rec("inputdialog", title=title, text=m.labelText(), png=png, mode=POLICY["mode"])
        m.setProperty("audit_handled", True)
        if POLICY["mode"] == "accept":
            def _acc(d=m):
                if d.inputMode() == QInputDialog.TextInput:
                    if d.comboBoxItems():
                        d.setTextValue(d.comboBoxItems()[0])
                    else:
                        d.setTextValue(POLICY["text"])
                d.accept()
            later(_acc)
        else:
            later(m.reject)
        return
    if isinstance(m, QDialog) and not m.property("audit_hooked"):
        m.setProperty("audit_hooked", True)
        m.finished.connect(lambda _r, d=m: d.setProperty("audit_handled", False))
    texts = widget_texts(m)
    png = shot(m, "dlg_" + (title or type(m).__name__), key="dlg|" + type(m).__name__ + "|" + title + "|" + "|".join(texts)[:3000])
    rec("dialog", cls=type(m).__name__, title=title, text="\n".join(texts), png=png, mode=POLICY["mode"])
    m.setProperty("audit_handled", True)
    h = POLICY["handler"]
    if h is not None and h(m):
        return
    if isinstance(m, QDialog):
        later(m.accept if POLICY["mode"] == "accept" else m.reject)
    else:
        later(m.close)


def _handle_wizard(wz: QWizard):
    key = wz.property("audit_key")
    if not key:
        WATCH["wizard_n"] = WATCH.get("wizard_n", 0) + 1
        key = f"wz{WATCH['wizard_n']}"
        wz.setProperty("audit_key", key)
    st = WATCH["wizard"].setdefault(key, {"seen": [], "wait": 0, "done": False})
    if st["done"]:
        return
    pid = wz.currentId()
    page = wz.currentPage()
    if pid not in st["seen"]:
        st["seen"].append(pid)
        st["wait"] = 0
        texts = widget_texts(page) if page is not None else []
        ttl = (page.title() if page is not None else "") or wz.windowTitle()
        png = shot(wz, f"wizard_{wz.windowTitle()}_p{pid}")
        nxt = wz.button(QWizard.NextButton)
        fin = wz.button(QWizard.FinishButton)
        rec("wizardpage", title=wz.windowTitle(), page=pid, page_title=ttl,
            subtitle=page.subTitle() if page is not None else "",
            text="\n".join(texts), png=png,
            next_enabled=bool(nxt.isVisible() and nxt.isEnabled()),
            finish_visible=bool(fin.isVisible()))
        h = POLICY["handler"]
        if h is not None and h(wz):
            return
    nxt = wz.button(QWizard.NextButton)
    tries = st.setdefault("tries", {})
    if len(st["seen"]) < POLICY["wizard_pages"] and nxt.isVisible() and nxt.isEnabled()             and tries.get(pid, 0) < 2:
        # a page whose Next is enabled but refuses (validatePage pops a message)
        # stays on the same id: two tries, then it counts as blocked
        tries[pid] = tries.get(pid, 0) + 1
        later(wz.next)
        return
    st["wait"] += 1
    if st["wait"] < 12:                  # ~1 s for a page that computes before enabling Next
        return
    if nxt.isVisible() and not nxt.isEnabled():
        rec("wizard_blocked", title=wz.windowTitle(), page=pid,
            text="\n".join(widget_texts(wz.currentPage()))[:3000] if wz.currentPage() else "")
    st["done"] = True
    wz.setProperty("audit_handled", True)
    later(wz.reject)


def _menu_entries(menu: QMenu) -> list[dict]:
    out = []
    for a in menu.actions():
        _KEEP.append(a)
        if a.isSeparator():
            continue
        out.append({"text": a.text().replace("&", ""), "enabled": a.isEnabled(), "tip": a.toolTip(),
                    "sub": a.menu() is not None, "checked": a.isChecked() if a.isCheckable() else None,
                    "shortcut": a.shortcut().toString()})
    return out


def _handle_popup(menu: QMenu):
    entries = _menu_entries(menu)
    png = shot(menu, "menu", key="menu|" + "|".join(e["text"] + str(e["enabled"]) for e in entries))
    rec("popup", entries=entries, png=png, tooltips_visible=menu.toolTipsVisible())
    menu.setProperty("audit_handled", True)
    if not menu.property("audit_hooked"):
        menu.setProperty("audit_hooked", True)
        menu.aboutToHide.connect(lambda m=menu: m.setProperty("audit_handled", False))
    pick = POLICY["menu_pick"]
    POLICY["menu_pick"] = None
    target = None
    if pick is not None:
        for a in menu.actions():
            if a.isSeparator() or not a.isEnabled():
                continue
            if (isinstance(pick, str) and pick.lower() in a.text().replace("&", "").lower()) or \
                    (callable(pick) and pick(a)):
                target = a
                break
        if target is None:
            rec("menu_pick_missing", pick=str(pick), entries=[e["text"] for e in entries])
    if target is not None:
        pos = menu.actionGeometry(target).center()
        later(lambda: QTest.mouseClick(menu, Qt.LeftButton, Qt.NoModifier, pos))
    else:
        later(lambda: QTest.keyClick(menu, Qt.Key_Escape))


def _watch():
    if WATCH["busy"]:
        return
    WATCH["busy"] = True
    try:
        if time.time() - WATCH.get("step_t0", time.time()) > 120:
            WATCH["step_t0"] = time.time()
            stuck = QApplication.activePopupWidget() or QApplication.activeModalWidget()
            if stuck is not None and stuck is not win:
                rec("stall", what=f"{type(stuck).__name__} '{stuck.windowTitle()}' closed after 120 s")
                print("STALL: closing", type(stuck).__name__, flush=True)
                if isinstance(stuck, QDialog):
                    later(stuck.reject)
                else:
                    later(stuck.close)
                return
        pop = QApplication.activePopupWidget()
        if isinstance(pop, QMenu) and not WATCH["ignore_popups"] and not pop.property("audit_handled"):
            _handle_popup(pop)
            return
        m = QApplication.activeModalWidget()
        if os.environ.get("AUDIT_DEBUG") and m is not None:
            print(f"WATCH modal={type(m).__name__} handled={m.property('audit_handled')} "
                  f"popup={type(pop).__name__ if pop else None} ignore={WATCH['ignore_popups']}", flush=True)
        if m is not None and m is not win:
            if isinstance(m, QWizard):
                _handle_wizard(m)
            elif not m.property("audit_handled"):
                _handle_modal(m)
    except Exception:                    # noqa: BLE001
        EXC.append({"state": CTX["state"], "step": CTX["step"] + " (watchdog)", "tb": traceback.format_exc()[-2000:]})
    finally:
        WATCH["busy"] = False


_wd = QTimer()
_wd.timeout.connect(_watch)
_wd.start(80)


# ------------------------------------------------------------ view guard
def view_sig():
    cv = win.canvas
    t = cv.transform()
    return (round(t.m11(), 5), round(t.m22(), 5), cv.horizontalScrollBar().value(),
            cv.verticalScrollBar().value())


VIEW_WORDS = ("zoom", "fit", "pan", "key R", "key Plus", "key Minus", "key Equal", "reset view", "wheel",
              "Only the", "solo", "Segment && Points panel", "Getting started", "open", "Open", "project",
              "switch", "view", "Add video", "camera", "remove", "Undo")


def closing_new_windows(before: set):
    """Screenshot + record + close every top-level window that appeared."""
    for w in QApplication.topLevelWidgets():
        try:
            if w is win or not w.isVisible() or id(w) in before:
                continue
            if w.windowType() in (Qt.ToolTip, Qt.Popup, Qt.SplashScreen) or isinstance(w, QMenu):
                continue
            texts = widget_texts(w)
            png = shot(w, "win_" + (w.windowTitle() or type(w).__name__),
                       key="win|" + type(w).__name__ + "|" + w.windowTitle())
            rec("window", cls=type(w).__name__, title=w.windowTitle(), text="\n".join(texts)[:6000], png=png)
            w.close()
        except RuntimeError:
            continue
    pump(0.05)


def step(name, fn, view_ok=None, settle=0.35):
    """Run one user action with the guards around it."""
    CTX["step"] = name
    WATCH["step_t0"] = time.time()
    if os.environ.get("AUDIT_TRACE"):
        print(f"  step [{CTX['state']}] {name}", flush=True)
    before_windows = {id(w) for w in QApplication.topLevelWidgets() if w.isVisible()}
    v0 = view_sig() if win.session is not None else None
    n_exc = len(EXC)
    st0 = win.state
    try:
        fn()
    except Exception:                    # noqa: BLE001
        EXC.append({"state": CTX["state"], "step": name, "tb": traceback.format_exc()[-2500:]})
    pump(settle)
    closing_new_windows(before_windows)
    if win.state == TRACKING and st0 != TRACKING and not name.startswith("track"):
        rec("state_change", frm=st0, to=win.state)
    if v0 is not None and win.session is not None:
        ok = view_ok if view_ok is not None else any(w.lower() in name.lower() for w in VIEW_WORDS)
        v1 = view_sig()
        if v1 != v0 and not ok:
            rec("view_moved", before=list(v0), after=list(v1))
    rec("step", ok=len(EXC) == n_exc, status=win.statusBar().currentMessage()[:300])


# ------------------------------------------------------------ interactions
def menubar_menus() -> list[QMenu]:
    """The menu bar's own menus (its hidden overflow menu has no title)."""
    mb = win.menuBar()
    return [m for m in mb.findChildren(QMenu, options=Qt.FindDirectChildrenOnly)
            if m.menuAction().isVisible() and m.title().replace("&", "").strip()]


def open_menubar(title_part: str) -> QMenu | None:
    mb = win.menuBar()
    for m in menubar_menus():
        if title_part.lower() in m.title().replace("&", "").lower():
            WATCH["ignore_popups"] = True
            r = mb.actionGeometry(m.menuAction())
            for _ in range(3):
                close_popups(keep_ignore=True)
                QTest.mouseClick(mb, Qt.LeftButton, Qt.NoModifier, r.center())
                pump(0.25)
                if m.isVisible():
                    return m
                # Escape out of the menu bar's keyboard mode, then click again
                QTest.keyClick(mb, Qt.Key_Escape)
                pump(0.1)
            return None
    return None


def close_popups(keep_ignore=False):
    for _ in range(4):
        p = QApplication.activePopupWidget()
        if p is None:
            break
        QTest.keyClick(p, Qt.Key_Escape)
        pump(0.08)
    if not keep_ignore:
        WATCH["ignore_popups"] = False


def click_menu_path(path: list[str], record_menu=True) -> bool:
    """Open a menu-bar menu and click through `path` (substrings) with real
    clicks; the final entry is clicked with the watchdog live."""
    m = open_menubar(path[0])
    if m is None:
        rec("menu_unreachable", path=path)
        close_popups()
        return False
    if record_menu:
        entries = _menu_entries(m)
        png = shot(m, "menubar_" + path[0], key="mb|" + path[0] + "|" + "|".join(e["text"] + str(e["enabled"]) for e in entries))
        rec("menubar", menu=path[0], entries=entries, png=png, tooltips_visible=m.toolTipsVisible())
    cur = m
    for i, part in enumerate(path[1:]):
        target = None
        for a in cur.actions():
            _KEEP.append(a)
            if not a.isSeparator() and part.lower() in a.text().replace("&", "").lower():
                target = a
                break
        if target is None or not target.isEnabled():
            rec("menu_entry_missing" if target is None else "menu_entry_disabled", path=path, at=part)
            close_popups()
            return False
        last = i == len(path) - 2
        if last:
            WATCH["ignore_popups"] = False
        pos = cur.actionGeometry(target).center()
        QTest.mouseClick(cur, Qt.LeftButton, Qt.NoModifier, pos)
        pump(0.25)
        if not last:
            sub = target.menu()
            if sub is None or not sub.isVisible():
                QTest.mouseMove(cur, pos)
                pump(0.4)
            if sub is None or not sub.isVisible():
                rec("submenu_unreachable", path=path, at=part)
                close_popups()
                return False
            if record_menu:
                entries = _menu_entries(sub)
                png = shot(sub, "submenu_" + part, key="sm|" + part + "|" + "|".join(e["text"] + str(e["enabled"]) for e in entries))
                rec("submenu", menu=" > ".join(path[:i + 2]), entries=entries, png=png,
                    tooltips_visible=sub.toolTipsVisible())
            cur = sub
    WATCH["ignore_popups"] = False
    return True


def walk_menubar(skip=("Open Video", "Open Project", "Exit", "Quit")):
    """Every enabled leaf entry of every menu-bar menu, clicked for real
    (dialogs answered with POLICY['mode'])."""
    plan = []
    for m in menubar_menus():
        top = m.title().replace("&", "")
        for a in m.actions():
            _KEEP.append(a)
            if a.isSeparator() or not a.text():
                continue
            if a.menu() is not None:
                sub = a.menu()
                _KEEP.append(sub)
                for b in sub.actions():
                    _KEEP.append(b)
                    if not b.isSeparator() and b.text():
                        plan.append([top, a.text().replace("&", ""), b.text().replace("&", "")])
            else:
                plan.append([top, a.text().replace("&", "")])
    for path in plan:
        label = " > ".join(path)
        if any(s in label for s in skip):
            continue
        if path[0] == "Skeleton" and CTX["state"] not in ("points",):
            continue                     # templates add landmarks: exercised in the points state only
        step("menu " + label, lambda p=path: click_menu_path(p))
        if win.state == TRACKING and CTX["state"] != "tracking":
            win._pause_tracking()
            wait(lambda: win.state != TRACKING, 30, "pause after menu")


def click_widget(w, button=Qt.LeftButton, pos=None, mods=Qt.NoModifier):
    if pos is None:
        pos = w.rect().center()
    QTest.mouseClick(w, button, mods, pos)


def click_button_arrow(btn: QToolButton, pick=None):
    """The ▾ part of a MenuButtonPopup tool button, with the watchdog picking `pick`."""
    POLICY["menu_pick"] = pick
    WATCH["ignore_popups"] = False
    pos = QPoint(btn.width() - 6, btn.height() // 2)
    QTest.mousePress(btn, Qt.LeftButton, Qt.NoModifier, pos)
    pump(0.6)
    QTest.mouseRelease(btn, Qt.LeftButton, Qt.NoModifier, pos)
    pump(0.3)


def canvas_pos(x, y, cv=None):
    cv = cv or win.canvas
    return cv.mapFromScene(QPointF(x, y))


def canvas_click(x, y, button=Qt.LeftButton, mods=Qt.NoModifier, cv=None):
    cv = cv or win.canvas
    QTest.mouseClick(cv.viewport(), button, mods, canvas_pos(x, y, cv))


def canvas_drag(p0, p1, cv=None, steps=8):
    cv = cv or win.canvas
    a, b = canvas_pos(*p0, cv), canvas_pos(*p1, cv)
    QTest.mousePress(cv.viewport(), Qt.LeftButton, Qt.NoModifier, a)
    for k in range(1, steps + 1):
        QTest.mouseMove(cv.viewport(), QPoint(a.x() + (b.x() - a.x()) * k // steps,
                                              a.y() + (b.y() - a.y()) * k // steps))
        pump(0.02)
    QTest.mouseRelease(cv.viewport(), Qt.LeftButton, Qt.NoModifier, b)


def list_item_pos(lst: QListWidget, row: int) -> QPoint:
    return lst.visualItemRect(lst.item(row)).center()


def context_menu_at(w: QWidget, pos: QPoint, pick=None):
    """A right-click: the press (the canvas opens its menu on press) and the
    context-menu event the platform sends after it (lists, timeline)."""
    POLICY["menu_pick"] = pick
    WATCH["ignore_popups"] = False
    QTest.mousePress(w, Qt.RightButton, Qt.NoModifier, pos)
    pump(0.2)
    QTest.mouseRelease(w, Qt.RightButton, Qt.NoModifier, pos)
    pump(0.2)
    if not any(r["kind"] == "popup" and r["step"] == CTX["step"] for r in REC[-6:]):
        target = w
        ev = QContextMenuEvent(QContextMenuEvent.Mouse, pos, w.mapToGlobal(pos))
        app.sendEvent(target, ev)
        pump(0.3)


def key(k, mods=Qt.NoModifier, target=None):
    t = target or QApplication.focusWidget() or win
    QTest.keyClick(t, k, mods)
    pump(0.08)


def focus_on(w):
    """Give `w` the keyboard focus the way a user does: click it."""
    activate()
    if isinstance(w, QAbstractSpinBox):
        le = w.findChild(QLineEdit)
        click_widget(le or w)
    elif isinstance(w, QListWidget):
        if w.count():
            click_widget(w.viewport(), pos=list_item_pos(w, 0))
        else:
            click_widget(w.viewport())
    else:
        click_widget(w)
    pump(0.1)
    return QApplication.focusWidget()


# -------------------------------------------------------- automatic checks
def check_controls(label):
    """Every visible button / spin box / list inside the window, unclipped."""
    bad = []
    for w in win.findChildren(QWidget):
        try:
            if not isinstance(w, (QAbstractButton, QAbstractSpinBox, QListWidget, QComboBox)):
                continue
            if not w.isVisible():
                continue
            g = w.rect()
            tl = w.mapTo(win, g.topLeft())
            br = w.mapTo(win, g.bottomRight())
            if tl.x() < 0 or tl.y() < 0 or br.x() > win.width() or br.y() > win.height():
                bad.append(f"{type(w).__name__} '{getattr(w, 'text', lambda: '')()}' outside the window")
                continue
            if w.visibleRegion().isEmpty():
                bad.append(f"{type(w).__name__} '{getattr(w, 'text', lambda: '')()}' fully covered / clipped")
                continue
            if isinstance(w, QToolButton) and w.text() and w.toolButtonStyle() != Qt.ToolButtonIconOnly:
                if w.sizeHint().width() - w.width() > 6:
                    bad.append(f"QToolButton '{w.text()}' squeezed: {w.width()} px for a {w.sizeHint().width()} px label")
        except RuntimeError:
            continue
    rec("controls", label=label, size=[win.width(), win.height()], problems=bad,
        png=shot(win, "window_" + label))
    return bad


FOCUS_KEYS = [("F", Qt.Key_F, Qt.NoModifier), ("B", Qt.Key_B, Qt.NoModifier),
              ("Shift+F", Qt.Key_F, Qt.ShiftModifier), ("Shift+B", Qt.Key_B, Qt.ShiftModifier),
              ("N", Qt.Key_N, Qt.NoModifier), ("H", Qt.Key_H, Qt.NoModifier),
              ("O", Qt.Key_O, Qt.NoModifier), ("L", Qt.Key_L, Qt.NoModifier),
              ("E", Qt.Key_E, Qt.NoModifier), ("Space", Qt.Key_Space, Qt.NoModifier),
              ("Right", Qt.Key_Right, Qt.NoModifier), ("Left", Qt.Key_Left, Qt.NoModifier),
              ("End", Qt.Key_End, Qt.NoModifier), ("Home", Qt.Key_Home, Qt.NoModifier),
              ("Shift+Right", Qt.Key_Right, Qt.ShiftModifier), ("J", Qt.Key_J, Qt.NoModifier),
              ("S", Qt.Key_S, Qt.NoModifier), ("Plus", Qt.Key_Plus, Qt.NoModifier),
              ("Minus", Qt.Key_Minus, Qt.NoModifier), ("Ctrl+1", Qt.Key_1, Qt.ControlModifier)]


def effect_sig():
    return {"frame": win.current, "add": win.btn_add.isChecked(), "pan": win.btn_pan.isChecked(),
            "onion": win.act_onion.isChecked(), "loupe": win.act_loupe.isChecked(),
            "play": win.btn_play.isChecked(), "event": win._pending_event is not None,
            "seg": win.btn_animal.isChecked(), "zoom": round(win.canvas.transform().m11(), 4),
            "dock": win.dock.isVisible()}


def undo_effect():
    """Put every toggle back after a probe so the next one starts equal."""
    if win.btn_play.isChecked():
        win.btn_play.setChecked(False)
    for b in (win.btn_add, win.btn_pan, win.btn_animal):
        if b.isChecked():
            b.setChecked(False)
    for a in (win.act_onion, win.act_loupe):
        if a.isChecked():
            a.setChecked(False)
    if win._pending_event is not None:
        win._pending_event = None
        win.timeline.set_pending_event(None)
    if not win.dock.isVisible():
        win.dock.setVisible(True)
    pump(0.1)


def hotkey_matrix(targets: dict):
    """Each key from each focus target; the canvas is the reference."""
    results = {}
    for tname, getter in targets.items():
        results[tname] = {}
        for kname, k, mods in FOCUS_KEYS:
            CTX["step"] = f"hotkey {kname} focus={tname}"
            try:
                win._goto(40)
                win.canvas.fit()
                pump(0.15)
                w = getter()
                fw = focus_on(w) if w is not None else None
                before = effect_sig()
                QTest.keyClick(QApplication.focusWidget() or win, k, mods)
                pump(0.25)
                after = effect_sig()
                diff = {kk: [before[kk], after[kk]] for kk in before if before[kk] != after[kk]}
                results[tname][kname] = {"diff": diff, "focus": type(fw).__name__ if fw else None}
            except Exception:            # noqa: BLE001
                EXC.append({"state": CTX["state"], "step": CTX["step"], "tb": traceback.format_exc()[-1500:]})
            undo_effect()
    ref = results.get("canvas", {})
    mism = []
    for tname, res in results.items():
        if tname == "canvas":
            continue
        for kname, r in res.items():
            rr = ref.get(kname)
            if rr is None:
                continue
            if set(r["diff"]) != set(rr["diff"]):
                mism.append({"focus": tname, "key": kname, "canvas": rr["diff"], "here": r["diff"],
                             "focus_widget": r["focus"]})
    rec("hotkey_matrix", results=results, mismatches=mism)
    return mism


def walk_toolbar():
    """Every visible tool button in the control bar and the panel, clicked."""
    btns = [b for b in win.findChildren(QToolButton) if b.isVisible() and b.window() is win]
    for b in btns:
        name = (b.text() or b.toolTip().split("\n")[0])[:40]
        if b is win.btn_track:
            continue
        CTX["step"] = f"toolbar {name}"
        rec("button", text=b.text(), tip=b.toolTip(), enabled=b.isEnabled(), checkable=b.isCheckable())
        if not b.isEnabled():
            continue
        if b.popupMode() == QToolButton.InstantPopup:
            step(f"toolbar {name} (menu)", lambda bb=b: (setattr_policy(None), click_widget(bb)))
            continue
        if b.text().startswith("×") or "Clear segment" in b.text():
            continue                     # destructive: covered by the scripted scenarios
        step(f"toolbar {name}", lambda bb=b: click_widget(bb, pos=QPoint(8, bb.height() // 2)))
        if b.isCheckable():
            step(f"toolbar {name} (again)", lambda bb=b: click_widget(bb, pos=QPoint(8, bb.height() // 2)))
        if b.popupMode() == QToolButton.MenuButtonPopup:
            step(f"toolbar {name} (arrow)", lambda bb=b: click_button_arrow(bb, None))
    reset_tools()


def setattr_policy(pick):
    POLICY["menu_pick"] = pick
    WATCH["ignore_popups"] = False


def reset_tools():
    """Disarm every tool a previous step may have left on (N, S, H, play), close
    popups, and log what was on: a state must start from a known footing."""
    close_popups()
    left_on = [b.text() for b in (win.btn_add, win.btn_animal, win.btn_pan, win.btn_play) if b.isChecked()]
    for b in (win.btn_add, win.btn_animal, win.btn_pan, win.btn_play):
        if b.isChecked():
            b.setChecked(False)
    if win._pending_event is not None:
        win._pending_event = None
        win.timeline.set_pending_event(None)
    win.canvas.cancel_gesture()
    pump(0.1)
    if left_on:
        rec("tools_left_on", tools=left_on)


def new_state(name):
    CTX["state"] = name
    CTX["step"] = "enter"
    POLICY.update(mode="cancel", handler=None, menu_pick=None, answers=[])
    print(f"==== state {name}", flush=True)
    reset_tools()


def want(name):
    return ONLY is None or name in ONLY


def open_video_by_menu(path):
    STUB["open"] = path
    POLICY["answers"] = []
    click_menu_path(["File", "Open Video"])
    STUB["open"] = ""
    return wait(lambda: win.state == READY, 60, "video opens")


# ================================================================ states
try:
    activate()
    win._dev_probe.wait(60000)

    if want("idle") or ONLY is None:
        new_state("idle")
        check_controls("idle_1600x1000")
        walk_menubar()
        walk_toolbar()
        for kname, k, mods in FOCUS_KEYS + [("T", Qt.Key_T, Qt.NoModifier), ("X", Qt.Key_X, Qt.NoModifier),
                                            ("Escape", Qt.Key_Escape, Qt.NoModifier), ("Delete", Qt.Key_Delete, Qt.NoModifier),
                                            ("F1", Qt.Key_F1, Qt.NoModifier)]:
            step(f"key {kname} (no video)", lambda k=k, m=mods: key(k, m, win))
        step("onboarding chip Open", lambda: click_widget(win.onboarding.findChildren(QAbstractButton)[0])
             if win.onboarding.findChildren(QAbstractButton) else None)

    # ---------------------------------------------------------------- one video
    new_state("video")
    step("open video via File menu", lambda: open_video_by_menu(VID))
    assert win.state == READY, "the video did not open"
    activate()
    if want("video"):
        check_controls("video_1600x1000")
        win.resize(1366, 768)
        pump(0.4)
        check_controls("video_1366x768")
        win.resize(1600, 1000)
        pump(0.4)
        # zoom in so a view that moved by itself would show
        step("zoom in with Plus x3", lambda: [key(Qt.Key_Plus, target=win.canvas.viewport()) for _ in range(3)])
        walk_menubar()
        walk_toolbar()
        step("reset view R", lambda: key(Qt.Key_R, target=win.canvas.viewport()))
        step("timeline right-click", lambda: context_menu_at(win.timeline, QPoint(win.timeline.width() // 2, 8)))
        step("play Space", lambda: key(Qt.Key_Space, target=win.canvas.viewport()), settle=1.0)
        step("pause Space", lambda: key(Qt.Key_Space, target=win.canvas.viewport()))
        step("frame box type 25 Enter", lambda: (focus_on(win.spin), win.spin.selectAll(),
                                                 QTest.keyClicks(QApplication.focusWidget(), "25"),
                                                 key(Qt.Key_Return)))
        rec("frame_after_typing", frame=win.current)

    # ---------------------------------------------------------------- points
    if want("points") or want("balls") or want("tracking"):
        new_state("points")
        win._goto(0)
        pump(0.3)
        hx, hy = map(float, GT0["head"])
        ex, ey = map(float, GT0["tip"])
        step("N then click the head", lambda: (key(Qt.Key_N, target=win.canvas.viewport()), canvas_click(hx, hy)))
        step("N then click the tail", lambda: (key(Qt.Key_N, target=win.canvas.viewport()), canvas_click(ex, ey)))
        rec("points_after_clicks", n=win.session.n_points)
        cx, cy = map(float, GT0["centre"])
        step("Add arrow -> rectangle region", lambda: click_button_arrow(win.btn_add, "rectangle"))
        step("N then drag a rectangle", lambda: (key(Qt.Key_N, target=win.canvas.viewport()),
                                                  canvas_drag((cx - 30, cy - 15), (cx + 30, cy + 15))))
        step("Add arrow -> circle region", lambda: click_button_arrow(win.btn_add, "circle"))
        rec("points_after_regions", n=win.session.n_points, kinds=[p.kind for p in win.session.points])
        if want("points"):
            step("select point 0 in the list", lambda: click_widget(win.point_list.viewport(),
                                                                     pos=list_item_pos(win.point_list, 0)))
            step("right-click marker 0", lambda: context_menu_at(win.canvas.viewport(), canvas_pos(hx, hy)))
            step("right-click list row 0", lambda: context_menu_at(win.point_list.viewport(),
                                                                    list_item_pos(win.point_list, 0)))
            POLICY["mode"] = "accept"
            step("rename by double-click + typing", lambda: (
                QTest.mouseDClick(win.point_list.viewport(), Qt.LeftButton, Qt.NoModifier,
                                  list_item_pos(win.point_list, 0)), pump(0.3),
                QTest.keyClicks(QApplication.focusWidget(), "snout"), key(Qt.Key_Return)))
            rec("rename_result", name=win.session.points[0].name if win.session.n_points else None,
                frame=win.current)
            step("unarmed click moves the selected point", lambda: canvas_click(hx + 6, hy + 4))
            if win.session.n_points:
                rec("annotate_result", pos=[float(v) for v in win.session.tracks[0, 0]],
                    manual=bool(win.session.manual[0, 0]))
            step("Ctrl+Z undoes the click", lambda: key(Qt.Key_Z, Qt.ControlModifier, win.canvas.viewport()))
            step("Shift+X hidden here", lambda: key(Qt.Key_X, Qt.ShiftModifier, win.canvas.viewport()))
            step("Shift+X again", lambda: key(Qt.Key_X, Qt.ShiftModifier, win.canvas.viewport()))
            step("Shift+N note", lambda: key(Qt.Key_N, Qt.ShiftModifier, win.canvas.viewport()))
            step("E start event", lambda: key(Qt.Key_E, target=win.canvas.viewport()))
            win._goto(10)
            step("E end event", lambda: key(Qt.Key_E, target=win.canvas.viewport()))
            POLICY["mode"] = "cancel"
            rec("events_after", n=len(win.session.events), notes=len(win.session.notes))
            step("Shift+drag timeline marquee", lambda: (
                QTest.mousePress(win.timeline, Qt.LeftButton, Qt.ShiftModifier, QPoint(140, 25)),
                QTest.mouseMove(win.timeline, QPoint(300, 60)),
                QTest.mouseRelease(win.timeline, Qt.LeftButton, Qt.ShiftModifier, QPoint(300, 60))))
            rec("timeline_selection", sel=list(win.timeline.sel_range) if win.timeline.sel_range else None,
                rows=win.timeline.sel_rows)
            step("timeline right-click on the selection", lambda: context_menu_at(win.timeline, QPoint(200, 40)))
            step("Esc clears the selection", lambda: key(Qt.Key_Escape, target=win.canvas.viewport()))
            mism = hotkey_matrix({
                "canvas": lambda: win.canvas.viewport(),
                "frame box": lambda: win.spin,
                "step box": lambda: win.step_spin,
                "marker box": lambda: win.marker_spin,
                "point list": lambda: win.point_list,
                "timeline": lambda: win.timeline,
                "no focus": lambda: None,
            })
            print(f"hotkey mismatches: {len(mism)}", flush=True)
            walk_menubar()
            step("Skeleton button menu", lambda: (setattr_policy(None), click_widget(win.btn_skeleton)))

    # ---------------------------------------------------------------- balls
    if want("balls"):
        new_state("balls")
        win._goto(0)
        fx, fy = map(float, GT0["feet"][0]) if len(GT0.get("feet", [])) else (hx + 40, hy + 40)
        step("Add arrow -> Ball marker", lambda: click_button_arrow(win.btn_add, "Ball marker"))
        step("click a ball", lambda: canvas_click(fx, fy))
        balls = [i for i, p in enumerate(win.session.points) if getattr(p, "is_ball", False)]
        rec("balls_after", n=len(balls))
        if balls:
            bx, by = map(float, win.session.tracks[0, balls[0]])
            step("right-click the ball", lambda: context_menu_at(win.canvas.viewport(), canvas_pos(bx, by)))
        check_controls("balls")

    # ---------------------------------------------------------------- segment (GPU)
    if want("segment") and not NO_GPU:
        new_state("segment")
        win._goto(0)
        cx, cy = map(float, GT0["centre"])
        step("S then click the animal", lambda: (key(Qt.Key_S, target=win.canvas.viewport()), canvas_click(cx, cy)))
        wait(lambda: win.session.masks is not None and win.session.masks.has(0), 180, "mask preview")
        pump(0.5)
        rec("segment_after", has_mask=bool(win.session.masks is not None and win.session.masks.has(0)))
        shot(win, "segment_preview")
        step("right-click the prompt click", lambda: context_menu_at(win.canvas.viewport(), canvas_pos(cx, cy)))
        step("Esc leaves the segment tool", lambda: key(Qt.Key_Escape, target=win.canvas.viewport()))
        if win.animal_list.isVisible() and win.animal_list.count():
            step("right-click the segment row", lambda: context_menu_at(win.animal_list.viewport(),
                                                                         list_item_pos(win.animal_list, 0)))
        check_controls("segment")

    # ---------------------------------------------------------------- tracking (GPU)
    if want("tracking") and not NO_GPU:
        new_state("tracking")
        win._goto(0)
        win.point_list.clearSelection()
        pump(0.2)
        POLICY["mode"] = "accept"
        step("track T", lambda: key(Qt.Key_T, target=win.canvas.viewport()), settle=0.2)
        ok = wait(lambda: win.state == TRACKING, 240, "run starts")
        if ok:
            wait(lambda: win.current > 5 or win.state != TRACKING, 240, "first frames")
            shot(win, "tracking_running")
            check_controls("tracking")
            walk_menubar()
            for kname, k, mods in [("F", Qt.Key_F, Qt.NoModifier), ("B", Qt.Key_B, Qt.NoModifier),
                                   ("N", Qt.Key_N, Qt.NoModifier), ("S", Qt.Key_S, Qt.NoModifier),
                                   ("Delete", Qt.Key_Delete, Qt.NoModifier), ("E", Qt.Key_E, Qt.NoModifier)]:
                step(f"key {kname} while tracking", lambda k=k, m=mods: key(k, m, win.canvas.viewport()))
            if win.state == TRACKING:
                t0 = time.time()
                CTX["step"] = "Space pauses"
                QTest.keyClick(win.canvas.viewport(), Qt.Key_Space)
                wait(lambda: win.state != TRACKING, 10, "pause")
                rec("pause_latency", key="Space", seconds=round(time.time() - t0, 3))
        new_state("paused")
        pump(0.5)
        shot(win, "paused")
        walk_menubar()
        step("track T again", lambda: key(Qt.Key_T, target=win.canvas.viewport()), settle=0.2)
        if wait(lambda: win.state == TRACKING, 120, "resume"):
            pump(1.0)
            t0 = time.time()
            CTX["step"] = "X pauses"
            QTest.keyClick(win.canvas.viewport(), Qt.Key_X)
            wait(lambda: win.state != TRACKING, 10, "pause X")
            rec("pause_latency", key="X", seconds=round(time.time() - t0, 3))
        step("track to the end", lambda: key(Qt.Key_T, target=win.canvas.viewport()), settle=0.2)
        wait(lambda: win.state == TRACKING, 120, "resume 2")
        wait(lambda: win.state != TRACKING, 600, "run ends")
        new_state("tracked")
        pump(0.5)
        shot(win, "tracked")
        walk_menubar()
        step("J next doubtful stretch", lambda: key(Qt.Key_J, target=win.canvas.viewport()))
        step("Ctrl+Z undo the run", lambda: key(Qt.Key_Z, Qt.ControlModifier, win.canvas.viewport()))
        POLICY["mode"] = "cancel"

    # ---------------------------------------------------------------- two cameras
    if want("twocams"):
        new_state("twocams")
        STUB["open"] = VID_B
        step("camera panel + Add video", lambda: click_widget(win.cameras.btn_add))
        STUB["open"] = ""
        wait(lambda: win.project is not None and win.project.n_views == 2, 60, "second camera")
        pump(0.8)
        rec("views", n=win.project.n_views if win.project else 0)
        check_controls("twocams")
        if win.project is not None and win.project.n_views == 2:
            cv2_ = win.grid.canvas(1)
            step("click the companion view (switch)", lambda: click_widget(cv2_.viewport()))
            rec("active_after_click", active=win.project.active)
            step("click the first view back (switch)", lambda: click_widget(win.grid.canvas(0).viewport()))
            walk_menubar()
            mism = hotkey_matrix({"canvas": lambda: win.canvas.viewport(),
                                  "camera offset box": lambda: next(
                                      (sb for sb in win.cameras.findChildren(QAbstractSpinBox)
                                       if sb.isVisible() and sb.isEnabled()), None)})
            print(f"hotkey mismatches (2 cams): {len(mism)}", flush=True)

    # ---------------------------------------------------------------- calibrated
    if want("calibrated"):
        new_state("calibrated")
        PROJ = os.path.join(ROOT, "tests", "out", "test3d_gui.cotrk")
        CSV = os.path.join(ROOT, "tests", "out", "test3d_gui_dltCoefs.csv")
        if not (os.path.exists(PROJ) and os.path.exists(CSV)):
            rec("skip", why="run tests/verify_3d_gui.py once to build the 3-camera project")
        else:
            STUB["open"] = PROJ
            POLICY["answers"] = ["No"]          # a resume / save-changes question
            step("File > Open Project", lambda: click_menu_path(["File", "Open Project"]))
            STUB["open"] = ""
            wait(lambda: win.project is not None and win.project.n_views == 3 and win.state == READY, 60, "3-cam project")
            pump(0.8)
            check_controls("calibrated_3cams")

            def _calib_handler(m):
                from PySide6.QtWidgets import QPushButton
                from cotracker_app import view3d as v3
                if isinstance(m, v3.CalibrationDialog):
                    def go(d=m):
                        try:
                            btn = next(b for b in d.findChildren(QPushButton) if "Choose file" in b.text())
                            QTest.mouseClick(btn, Qt.LeftButton)          # the (stubbed) file chooser answers CSV
                            pump(0.4)
                            d.conv.setCurrentIndex(2)                      # OpenCV, 0-based: how the rig was built
                            pump(0.3)
                            shot(d, "calibration_dialog_filled")
                            rec("dialog", cls="CalibrationDialog(filled)", title=d.windowTitle(),
                                text="\n".join(widget_texts(d)))
                            QTest.mouseClick(d.buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)
                        except Exception:    # noqa: BLE001
                            EXC.append({"state": CTX["state"], "step": "calibration dialog",
                                        "tb": traceback.format_exc()[-1500:]})
                            d.reject()
                    later(go)
                    return True
                return False

            STUB["open"] = CSV
            POLICY.update(mode="accept", handler=_calib_handler)
            step("3D > Import Calibration", lambda: click_menu_path(["3D", "Import Calibration"]))
            POLICY.update(mode="cancel", handler=None)
            STUB["open"] = ""
            rec("calibration_after", has=win.project.calibration is not None)
            POLICY["mode"] = "accept"
            POLICY["answers"] = ["No"]
            step("Ctrl+3 reconstruct", lambda: key(Qt.Key_3, Qt.ControlModifier, win.canvas.viewport()), settle=1.0)
            step("3D > Estimate Sub-frame Offsets", lambda: click_menu_path(["3D", "Estimate Sub-frame"]), settle=1.0)
            step("Ctrl+4 carve", lambda: key(Qt.Key_4, Qt.ControlModifier, win.canvas.viewport()), settle=1.0)
            POLICY["mode"] = "cancel"
            walk_menubar()
            if win.session.n_points:
                p0 = win.session.tracks[win.current, 0]
                if np.isfinite(p0).all():
                    step("right-click a landmark (calibrated)", lambda: context_menu_at(
                        win.canvas.viewport(), canvas_pos(float(p0[0]), float(p0[1]))))
            shot(win, "calibrated_guides")

    # ---------------------------------------------------------------- body
    if want("body"):
        new_state("body")
        step("Body > Find People", lambda: click_menu_path(["Body", "Find People"]))

except Exception:                        # noqa: BLE001 - a broken state must not lose the report
    EXC.append({"state": CTX["state"], "step": CTX["step"] + " (state aborted)",
                "tb": traceback.format_exc()[-2500:]})
    print("STATE ABORTED:", traceback.format_exc()[-1200:], flush=True)

# ================================================================ report
CTX["state"] = "report"
lines = []
lines.append(f"steps {sum(1 for r in REC if r['kind'] == 'step')}, screenshots {SHOT_N[0]}, exceptions {len(EXC)}")
lines.append("\n==== EXCEPTIONS")
for e in EXC:
    lines.append(f"---- [{e['state']}] {e['step']}\n{e['tb']}")
lines.append("\n==== VIEW MOVED WITHOUT A VIEW COMMAND")
for r in REC:
    if r["kind"] == "view_moved":
        lines.append(f"[{r['state']}] {r['step']}: {r['before']} -> {r['after']}")
lines.append("\n==== HOTKEY FOCUS MISMATCHES (vs the canvas)")
for r in REC:
    if r["kind"] == "hotkey_matrix":
        for m in r["mismatches"]:
            lines.append(f"[{r['state']}] {m['key']:12s} focus={m['focus']:18s} ({m['focus_widget']}) canvas={m['canvas']} here={m['here']}")
lines.append("\n==== PAUSE LATENCY")
for r in REC:
    if r["kind"] == "pause_latency":
        lines.append(f"[{r['state']}] {r['key']}: {r['seconds']} s" + ("   <-- over 1 s" if r["seconds"] > 1.0 else ""))
lines.append("\n==== MENUS WITHOUT VISIBLE TOOLTIPS (entries that carry one)")
seen_m = set()
for r in REC:
    if r["kind"] in ("menubar", "submenu", "popup") and not r.get("tooltips_visible"):
        tipped = [e["text"] for e in r["entries"] if e["tip"] and e["tip"].replace("&", "") != e["text"]]
        k = (r.get("menu") or "popup") + "|" + "|".join(tipped)
        if tipped and k not in seen_m:
            seen_m.add(k)
            lines.append(f"[{r['state']}] {r.get('menu', 'popup')}: {tipped}")
lines.append("\n==== DISABLED MENU ENTRIES PER STATE")
dis = {}
for r in REC:
    if r["kind"] in ("menubar", "submenu"):
        dis.setdefault(r["state"], {})[r["menu"]] = [e["text"] for e in r["entries"] if not e["enabled"]]
for st, d in dis.items():
    for mname, ents in d.items():
        if ents:
            lines.append(f"[{st}] {mname}: {ents}")
lines.append("\n==== CONTROL PROBLEMS")
for r in REC:
    if r["kind"] == "controls" and r["problems"]:
        lines.append(f"[{r['state']}] {r['label']} {r['size']}: " + "; ".join(r["problems"]))
lines.append("\n==== UNREACHABLE / MISSING / TIMEOUTS")
for r in REC:
    if r["kind"] in ("menu_unreachable", "submenu_unreachable", "menu_entry_missing", "timeout",
                     "menu_pick_missing", "wizard_blocked", "skip", "state_change"):
        lines.append(json.dumps(r, ensure_ascii=True)[:600])
lines.append("\n==== MESSAGES / DIALOGS / WIZARD PAGES (for the verdict review)")
seen_t = set()
for r in REC:
    if r["kind"] in ("messagebox", "dialog", "inputdialog", "wizardpage", "window"):
        k = r["kind"] + (r.get("title") or "") + (r.get("text") or "")[:400]
        if k in seen_t:
            continue
        seen_t.add(k)
        lines.append(f"\n--- [{r['state']}] {r['step']} :: {r['kind']} '{r.get('title', '')}' png={r.get('png')}")
        lines.append((r.get("text") or "")[:2500])
lines.append("\n==== STEP RESULTS WITH STATUS")
for r in REC:
    if r["kind"] == "step":
        lines.append(f"[{r['state']}] {r['step']}: {'ok' if r['ok'] else 'EXC'} | {r['status']}")
lines.append("\n==== INFO RECORDS")
for r in REC:
    if r["kind"] in ("rename_result", "annotate_result", "events_after", "timeline_selection", "balls_after",
                     "segment_after", "views", "active_after_click", "calibration_after", "points_after_clicks",
                     "points_after_regions", "frame_after_typing"):
        lines.append(json.dumps(r, ensure_ascii=True)[:500])
with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
with open(os.path.join(OUT, "records.json"), "w", encoding="utf-8") as fh:
    json.dump(REC, fh, ensure_ascii=True, indent=1, default=str)
print(f"\n{sum(1 for r in REC if r['kind'] == 'step')} steps, {SHOT_N[0]} screenshots, {len(EXC)} exceptions")
print("report:", os.path.join(OUT, "report.txt"))
try:
    win.close()
    pump(0.5)
except Exception:                        # noqa: BLE001
    pass
_clean_autosaves()
sys.stdout.flush()
# os._exit: a menu wrapper the app rebuilt must not be freed twice at teardown
os._exit(1 if EXC else 0)
