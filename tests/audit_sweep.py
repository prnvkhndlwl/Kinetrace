"""Audit sweep: every enabled menu action, hotkey and context-menu entry, in
every application state, must not raise -- with every dialog stubbed so
nothing blocks, once answering "cancel / no" and once "accept / yes".

Not a pass/fail suite in the usual sense: it prints a table of everything it
drove and a list of exceptions (with the state and the action that produced
them) and writes `tests/out/audit_sweep.txt`. Exit code 1 when any exception
was recorded. Offscreen, needs test600.mp4; the tracked state needs a GPU
(skipped with --no-track).
"""
import faulthandler
import os
import shutil
import sys
import time
import traceback

faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("KINETRACE_UPDATE_API", "http://127.0.0.1:9/api")   # never asks GitHub (G37)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QMenu,
                               QMessageBox, QWizard)

NO_TRACK = "--no-track" in sys.argv
VID = os.path.join(ROOT, "test600.mp4")
GT = np.load(VID + ".gt.npz")["gt"]
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(SCRATCH, exist_ok=True)
VID_B = os.path.join(SCRATCH, "audit_second_cam.mp4")
if not os.path.exists(VID_B):
    shutil.copy(VID, VID_B)
from _clean import forget_recovery  # noqa: E402
forget_recovery(VID, VID_B)

# ----------------------------------------------------------------- capture
EXC: list[tuple[str, str, str]] = []            # (state, what, traceback tail)
DIALOGS: list[tuple[str, str, str]] = []        # (state, what, message)
_CTX = {"state": "?", "what": "?", "mode": "cancel"}


LIVE = open(os.path.join(SCRATCH, "audit_sweep_live.txt"), "w", encoding="utf-8")


def _hook(tp, val, tb):
    tail = "".join(traceback.format_exception(tp, val, tb))[-1500:]
    EXC.append((_CTX["state"], _CTX["what"], tail))
    LIVE.write(f"\n---- [{_CTX['state']}] {_CTX['what']} ({_CTX['mode']})\n{tail}\n")
    LIVE.flush()


sys.excepthook = _hook
# a blocked run (an unstubbed modal dialog) prints its Python stack every 90 s
faulthandler.dump_traceback_later(90, repeat=True)

# ------------------------------------------------------------ dialog stubs
SAVE_TARGETS = {"csv": os.path.join(SCRATCH, "audit_out.csv"), "tsv": os.path.join(SCRATCH, "audit_out.tsv"),
                "mat": os.path.join(SCRATCH, "audit_out.mat"), "kinetrace": os.path.join(SCRATCH, "audit_out.kinetrace"),
                "mp4": os.path.join(SCRATCH, "audit_out.mp4"), "json": os.path.join(SCRATCH, "audit_out.json"),
                "obj": os.path.join(SCRATCH, "audit_out.obj"), "png": os.path.join(SCRATCH, "audit_out.png"),
                "txt": os.path.join(SCRATCH, "audit_out.txt")}


def _save_name(*a, **k):
    if _CTX["mode"] == "cancel":
        return "", ""
    filt = str(a[3]) if len(a) > 3 else str(k.get("filter", ""))
    for ext, path in SAVE_TARGETS.items():
        if f"*.{ext}" in filt:
            return path, filt.split(";;")[0]
    return os.path.join(SCRATCH, "audit_out.dat"), filt


def _open_name(*a, **k):
    if _CTX["mode"] == "cancel":
        return "", ""
    filt = str(a[3]) if len(a) > 3 else str(k.get("filter", ""))
    if "*.mp4" in filt or "Video" in filt:
        return VID_B, filt
    if "*.kinetrace" in filt:
        return SAVE_TARGETS["kinetrace"], filt
    return SAVE_TARGETS["txt"], filt


def _msg(kind):
    def f(*a, **k):
        DIALOGS.append((_CTX["state"], _CTX["what"], f"{kind}: {a[1] if len(a) > 1 else ''} | {str(a[2])[:200] if len(a) > 2 else ''}"))
        if kind == "question":
            return QMessageBox.Yes if _CTX["mode"] == "accept" else QMessageBox.No
        return QMessageBox.Ok
    return staticmethod(f)


QMessageBox.question = _msg("question")
QMessageBox.warning = _msg("warning")
QMessageBox.information = _msg("information")
QMessageBox.critical = _msg("critical")
QFileDialog.getSaveFileName = staticmethod(_save_name)
QFileDialog.getOpenFileName = staticmethod(_open_name)
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([] if _CTX["mode"] == "cancel" else [VID_B], ""))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "" if _CTX["mode"] == "cancel" else SCRATCH)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False) if _CTX["mode"] == "cancel" else ("audit name", True))
QInputDialog.getMultiLineText = staticmethod(lambda *a, **k: ("", False) if _CTX["mode"] == "cancel" else ("audit note", True))
QInputDialog.getItem = staticmethod(lambda *a, **k: ("", False) if _CTX["mode"] == "cancel" else ("audit item", True))
QInputDialog.getInt = staticmethod(lambda *a, **k: (0, False) if _CTX["mode"] == "cancel" else (1, True))
QInputDialog.getDouble = staticmethod(lambda *a, **k: (0.0, False) if _CTX["mode"] == "cancel" else (1.0, True))
_orig_dialog_exec = QDialog.exec
QDialog.exec = lambda self: QDialog.Rejected if _CTX["mode"] == "cancel" else QDialog.Accepted
QWizard.exec = lambda self: QDialog.Rejected
MENU_PICK = {"i": None}


def _menu_exec(self, *a, **k):
    from PySide6.QtGui import QAction
    titles = {m.title() for m in self.findChildren(QMenu)}
    acts = [x for x in self.findChildren(QAction)
            if x.isEnabled() and not x.isSeparator() and x.text() and x.text() not in titles]
    i = MENU_PICK["i"]
    if i is None or i >= len(acts):
        MENU_PICK["n"] = len(acts)
        return None
    MENU_PICK["n"] = len(acts)
    MENU_PICK["label"] = acts[i].text()
    return acts[i]


class _PickMenu(QMenu):
    """PySide6 ignores a class-level patch of QMenu.exec (the C++ dispatcher
    wins), so the modules that pop context menus get a Python subclass whose
    exec() the harness controls."""

    def exec(self, *a, **k):            # noqa: A003
        return _menu_exec(self, *a, **k)


app = QApplication([])
import kinetrace.canvas as _canvas_mod        # noqa: E402
import kinetrace.timeline as _timeline_mod    # noqa: E402
import kinetrace.app as _app_mod              # noqa: E402
for _mod in (_canvas_mod, _timeline_mod, _app_mod):
    _mod.QMenu = _PickMenu
from kinetrace.app import MainWindow, READY, TRACKING, IDLE       # noqa: E402
# `QAction.menu()` hands back a wrapper Python believes it owns: let it be
# garbage-collected and shiboken deletes the C++ menu under the app's feet.
# Keep every submenu wrapper the sweep ever touches alive.
_KEEP: list = []

win = MainWindow()
win.show()

# A modal dialog the stubs do not cover would hang the run inside
# processEvents(): a timer running INSIDE the event loop closes any modal it
# finds and logs it -- an unstubbed modal is itself a finding worth a line.
from PySide6.QtCore import QTimer                               # noqa: E402
MODALS: list[str] = []


def _modal_watchdog():
    m = QApplication.activeModalWidget()
    if m is not None and m is not win:
        MODALS.append(f"[{_CTX['state']}] {_CTX['what']}: {type(m).__name__} '{m.windowTitle()}'")
        print("MODAL closed:", MODALS[-1], flush=True)
        try:
            if isinstance(m, QDialog):
                m.reject()
            else:
                m.close()
        except Exception:               # noqa: BLE001
            pass


_wd = QTimer()
_wd.timeout.connect(_modal_watchdog)
_wd.start(1500)


def pump(sec=0.05):
    t0 = time.time()
    while True:
        app.processEvents()
        if time.time() - t0 >= sec:
            break
        time.sleep(0.005)


def wait(cond, timeout, what):
    t0 = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def close_strays():
    opened = []
    for w in QApplication.topLevelWidgets():
        if w is win or not w.isVisible():
            continue
        if w.windowType() in (Qt.ToolTip, Qt.Popup, Qt.SplashScreen):
            continue
        if w.parent() is None or True:
            opened.append(type(w).__name__)
            try:
                w.close()
            except Exception:           # noqa: BLE001
                pass
    return opened


def _menu_title(menu) -> str:
    parts = []
    w = menu
    while isinstance(w, QMenu):
        parts.append(w.title().replace("&", ""))
        w = w.parent()
    return " > ".join(reversed(parts))


def menu_actions(menu, prefix=""):
    """(label, action) for every non-separator entry under `menu`.

    Enumerated with findChildren, NOT menu.actions() / action.menu(): a wrapper
    from those that Python later garbage-collects deletes the C++ object (the
    Events and Skeleton menus died at the end of one walk), and keeping such
    wrappers alive is no better -- the app rebuilds those menus, the kept
    wrappers go stale, and freeing a stale wrapper double-frees (an access
    violation mid-sweep). findChildren wrappers proved safe both ways."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMenuBar, QToolButton
    titles = {m.title() for m in menu.findChildren(QMenu)}
    out = []
    for a in menu.findChildren(QAction):
        if a.isSeparator() or not a.text() or a.text() in titles:
            continue                     # a submenu's own title action opens the submenu: skip
        par = a.parent()
        # only the menu bar's and the tool buttons' menus: context menus the
        # app has already shown linger as children of the canvas / list /
        # timeline and their entries dispatch through exec(), not trigger()
        root = par
        while isinstance(root, QMenu):
            root = root.parent()
        if isinstance(par, QMenu) and not isinstance(root, (QMenuBar, QToolButton)):
            continue
        label = ((_menu_title(par) + " > ") if isinstance(par, QMenu) else prefix) + a.text().replace("&", "")
        out.append((label, a))
    return out


SKIP_WORDS = ("Exit", "Quit", "Open Video", "Open Project", "Add Camera", "Add camera")
LOG: list[str] = []


def log(line):
    LOG.append(line)
    print(line, flush=True)


def _find_action(label):
    """Menus are rebuilt (the Skeleton menu on every open / skeleton change),
    so look the action up by label right before triggering it."""
    for lab, act in menu_actions(win):
        if lab == label:
            return act
    return None


def drive_actions(state):
    labels = [lab for lab, _ in menu_actions(win)]
    for label in labels:
        if any(w in label for w in SKIP_WORDS):
            continue
        act = _find_action(label)
        try:
            if act is None or not act.isEnabled():
                continue
        except RuntimeError:
            continue
        if "Track" in label and "Undo" not in label and win.state == READY and win._track_blocked is None:
            continue                                # runs are driven explicitly
        for mode in ("cancel", "accept"):
            _CTX.update(state=state, what=f"menu: {label}", mode=mode)
            n0 = len(EXC)
            before = win.state
            act = _find_action(label)
            if act is None:
                break
            try:
                if not act.isEnabled():
                    break
                act.trigger()
                pump(0.15)
            except RuntimeError:
                break
            except Exception:           # noqa: BLE001
                EXC.append((state, f"menu: {label}", traceback.format_exc()[-1500:]))
            opened = close_strays()
            pump(0.05)
            if win.state == TRACKING:
                win._pause_tracking()
                wait(lambda: win.state != TRACKING, 30, "pause after menu action")
            status = "EXC" if len(EXC) > n0 else "ok"
            log(f"[{state}] {mode:6s} {label:60s} {status}" + (f" opened {opened}" if opened else "")
                + ("" if before == win.state else f" state {before}->{win.state}"))
            try:
                if act.isCheckable():
                    break
            except RuntimeError:
                break


KEYS = [(Qt.Key_F, Qt.NoModifier), (Qt.Key_B, Qt.NoModifier), (Qt.Key_F, Qt.ShiftModifier),
        (Qt.Key_B, Qt.ShiftModifier), (Qt.Key_Space, Qt.NoModifier), (Qt.Key_Space, Qt.NoModifier),
        (Qt.Key_X, Qt.NoModifier), (Qt.Key_N, Qt.NoModifier), (Qt.Key_Escape, Qt.NoModifier),
        (Qt.Key_S, Qt.NoModifier), (Qt.Key_Escape, Qt.NoModifier), (Qt.Key_Delete, Qt.NoModifier),
        (Qt.Key_E, Qt.NoModifier), (Qt.Key_F, Qt.NoModifier), (Qt.Key_E, Qt.NoModifier),
        (Qt.Key_Escape, Qt.NoModifier), (Qt.Key_R, Qt.NoModifier), (Qt.Key_H, Qt.NoModifier),
        (Qt.Key_H, Qt.NoModifier), (Qt.Key_Plus, Qt.NoModifier), (Qt.Key_Minus, Qt.NoModifier),
        (Qt.Key_Plus, Qt.ShiftModifier), (Qt.Key_Minus, Qt.ShiftModifier), (Qt.Key_C, Qt.ShiftModifier),
        (Qt.Key_Period, Qt.ShiftModifier), (Qt.Key_Comma, Qt.ShiftModifier), (Qt.Key_Period, Qt.NoModifier),
        (Qt.Key_Comma, Qt.NoModifier), (Qt.Key_J, Qt.NoModifier), (Qt.Key_J, Qt.ShiftModifier),
        (Qt.Key_O, Qt.NoModifier), (Qt.Key_L, Qt.NoModifier), (Qt.Key_X, Qt.ShiftModifier),
        (Qt.Key_N, Qt.ShiftModifier), (Qt.Key_Z, Qt.ControlModifier), (Qt.Key_1, Qt.ControlModifier),
        (Qt.Key_1, Qt.ControlModifier), (Qt.Key_Comma, Qt.ControlModifier), (Qt.Key_F1, Qt.NoModifier),
        (Qt.Key_Escape, Qt.NoModifier), (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Home, Qt.NoModifier),
        (Qt.Key_End, Qt.NoModifier), (Qt.Key_Left, Qt.NoModifier), (Qt.Key_Right, Qt.NoModifier),
        (Qt.Key_Up, Qt.NoModifier), (Qt.Key_Down, Qt.NoModifier), (Qt.Key_PageUp, Qt.NoModifier),
        (Qt.Key_PageDown, Qt.NoModifier), (Qt.Key_Tab, Qt.NoModifier), (Qt.Key_Backspace, Qt.NoModifier)]


def drive_keys(state, allow_track=False):
    for mode in ("cancel", "accept"):
        for key, mod in KEYS:
            if key == Qt.Key_T and not allow_track:
                continue
            name = f"key {Qt.Key(key).name} {'+' + str(mod) if mod != Qt.NoModifier else ''}".strip()
            _CTX.update(state=state, what=name, mode=mode)
            n0 = len(EXC)
            try:
                QTest.keyClick(win, key, mod)
                pump(0.05)
            except Exception:           # noqa: BLE001
                EXC.append((state, name, traceback.format_exc()[-1500:]))
            opened = close_strays()
            if win.state == TRACKING:
                win._pause_tracking()
                wait(lambda: win.state != TRACKING, 30, "pause after key")
            if len(EXC) > n0:
                log(f"[{state}] {mode:6s} {name:60s} EXC")
        log(f"[{state}] {mode:6s} {'all hotkeys':60s} done")


def drive_context_menus(state):
    """Every entry of the point menu (canvas AND list), the segment menu and
    the timeline menu, chosen one at a time through the patched QMenu.exec."""
    s = win.session
    if s is None or win.state != READY:
        return
    MAX_PIDS = 6                    # a skeleton template adds dozens of points; six cover every kind
    for mode in ("cancel", "accept"):
        for pid in range(min(s.n_points, MAX_PIDS)):
            i = 0
            while True:
                MENU_PICK["i"] = i
                MENU_PICK["label"] = "?"
                _CTX.update(state=state, what=f"canvas menu pid {pid} entry {i}", mode=mode)
                n0 = len(EXC)
                if not (0 <= pid < win.session.n_points):
                    break
                try:
                    win.canvas._context_menu(pid, QPoint(5, 5))
                    pump(0.05)
                except Exception:       # noqa: BLE001
                    EXC.append((state, _CTX["what"], traceback.format_exc()[-1500:]))
                close_strays()
                if i >= MENU_PICK.get("n", 0):
                    break
                lab = MENU_PICK.get("label", "?")
                log(f"[{state}] {mode:6s} {'canvas menu ' + str(pid) + ': ' + lab:60s} {'EXC' if len(EXC) > n0 else 'ok'}")
                if "Delete" in lab:
                    break
                i += 1
        for pid in range(min(s.n_points, MAX_PIDS)):
            i = 0
            while True:
                if pid >= win.point_list.count():
                    break
                item = win.point_list.item(pid)
                pos = win.point_list.visualItemRect(item).center()
                MENU_PICK["i"] = i
                MENU_PICK["label"] = "?"
                _CTX.update(state=state, what=f"list menu pid {pid} entry {i}", mode=mode)
                n0 = len(EXC)
                try:
                    win._point_list_menu(pos)
                    pump(0.05)
                except Exception:       # noqa: BLE001
                    EXC.append((state, _CTX["what"], traceback.format_exc()[-1500:]))
                close_strays()
                if i >= MENU_PICK.get("n", 0):
                    break
                lab = MENU_PICK.get("label", "?")
                log(f"[{state}] {mode:6s} {'list menu ' + str(pid) + ': ' + lab:60s} {'EXC' if len(EXC) > n0 else 'ok'}")
                if "Delete" in lab:
                    break
                i += 1
        if getattr(win.session, "animal", None) is not None:
            menu, acts = win._build_animal_menu()
            flat = []
            for k, v in acts.items():
                if isinstance(v, dict):
                    flat += list(v.keys())
                elif v is not None and hasattr(v, "text"):
                    flat.append(v)
            for a in flat:
                if not a.isEnabled():
                    continue
                _CTX.update(state=state, what=f"segment menu: {a.text()}", mode=mode)
                n0 = len(EXC)
                try:
                    win._animal_menu_action(a, acts)
                    pump(0.05)
                except Exception:       # noqa: BLE001
                    EXC.append((state, _CTX["what"], traceback.format_exc()[-1500:]))
                log(f"[{state}] {mode:6s} {'segment menu: ' + a.text():60s} {'EXC' if len(EXC) > n0 else 'ok'}")
                if "Remove" in a.text() or "remove" in a.text():
                    break
        MENU_PICK["i"] = None


def timeline_menus(state):
    tl = win.timeline
    if win.session is None:
        return
    for mode in ("cancel", "accept"):
        for f0, f1, rows in ((10, 50, None), (10, 50, [0]), (0, 0, None)):
            try:
                tl.sel_range = (f0, f1) if f1 > f0 else None
                tl.sel_rows = rows
            except Exception:           # noqa: BLE001
                pass
            i = 0
            while i < 12:
                MENU_PICK["i"] = i
                MENU_PICK["label"] = "?"
                _CTX.update(state=state, what=f"timeline menu sel={tl.sel_range} rows={rows} entry {i}", mode=mode)
                n0 = len(EXC)
                try:
                    from PySide6.QtGui import QContextMenuEvent
                    local = QPoint(tl.width() // 2, tl.height() // 2)
                    ev = QContextMenuEvent(QContextMenuEvent.Mouse, local, tl.mapToGlobal(local))
                    app.sendEvent(tl, ev)
                    pump(0.05)
                except Exception:       # noqa: BLE001
                    EXC.append((state, _CTX["what"], traceback.format_exc()[-1500:]))
                close_strays()
                if i >= MENU_PICK.get("n", 0):
                    break
                log(f"[{state}] {mode:6s} {'timeline menu: ' + MENU_PICK.get('label', '?'):60s} {'EXC' if len(EXC) > n0 else 'ok'}")
                i += 1
        MENU_PICK["i"] = None


def sweep(state, allow_track=False):
    log(f"==== state {state}: {win.state}, points {win.session.n_points if win.session else 0}")
    drive_actions(state)
    drive_keys(state, allow_track)
    drive_context_menus(state)
    timeline_menus(state)


# ---------------------------------------------------------------- states
win._dev_probe.wait(60000)
sweep("IDLE")

win._open_video(VID)
wait(lambda: win.state == READY, 30, "open")
sweep("READY-empty")

for d in range(2):
    win._on_add(float(GT[0, d, 0]), float(GT[0, d, 1]))
win._goto(0)
sweep("READY-2points")

if not NO_TRACK:
    # the sweeps above may have deleted points or switched the run mode
    win._goto(0)
    if win.session.n_points == 0 or not win.session.seedable_at(0):
        for d in range(2):
            win._on_add(float(GT[0, d, 0]), float(GT[0, d, 1]))
    if getattr(win, "_track_mode", "auto") != "auto":
        for a in win.findChildren(type(win.act_open)):
            if "utomatic" in a.text() and "emi" not in a.text() and a.isCheckable():
                a.setChecked(True)
                a.trigger()
        win._track_mode = "auto"
    win.point_list.clearSelection()
    _CTX.update(state="TRACKING", what="run", mode="accept")
    win._toggle_tracking()
    wait(lambda: win.state == TRACKING, 180, "run start")
    drive_keys("TRACKING")
    drive_actions("TRACKING")
    if win.state == TRACKING:
        wait(lambda: win.state == READY, 400, "run end")
    sweep("READY-tracked")

if win._add_view(VID_B):
    pump(0.3)
    sweep("READY-2cams")
    _CTX.update(state="READY-2cams", what="switch view", mode="accept")
    try:
        win._set_active_view(1)
    except Exception:                    # noqa: BLE001
        try:
            win.project.active = 1
            win.grid.set_active(1)
        except Exception:                # noqa: BLE001
            pass
    pump(0.2)
    sweep("READY-2cams-view2")

# ---------------------------------------------------------------- report
out = os.path.join(SCRATCH, "audit_sweep.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n\n==== DIALOGS\n")
    for st, what, msg in DIALOGS:
        fh.write(f"[{st}] {what}: {msg}\n")
    fh.write("\n==== UNSTUBBED MODALS (closed by the watchdog)\n")
    for m in MODALS:
        fh.write(m + "\n")
    fh.write("\n==== EXCEPTIONS\n")
    for st, what, tb in EXC:
        fh.write(f"\n---- [{st}] {what}\n{tb}\n")
print(f"\n{len(LOG)} actions driven, {len(DIALOGS)} dialogs shown, {len(EXC)} exceptions -> {out}")
for st, what, tb in EXC:
    print(f"\n---- [{st}] {what}\n{tb[-700:]}")
try:
    win.close()
    pump(0.3)
except Exception:                        # noqa: BLE001
    pass
forget_recovery(VID, VID_B)
print("AUDIT SWEEP " + ("CLEAN" if not EXC else f"FOUND {len(EXC)} EXCEPTIONS"))
sys.exit(1 if EXC else 0)
