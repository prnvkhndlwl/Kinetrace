"""The menu bar keeps every action, and every menu path the app and the docs name exists (G188).

Offscreen, no GPU. (a) Every action of the menu bar as it was before G188 (the golden list below: the window's
`act_*` attribute, or the label of a generated entry) is still reachable from the menu bar, with the same
keyboard shortcut. (b) Every "Menu → Entry (→ Entry)" path written in kinetrace/*.py, docs/MANUAL.md, README.md
and the wiki clone (when present) resolves in the live menu tree: a message that sends the user to a menu entry
that moved is a failure here, not a support question.
"""
import os
import re
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtGui import QAction  # noqa: E402
import shiboken6  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication([])
from kinetrace.app import MainWindow  # noqa: E402

# ---- (a) the golden list: every menu-bar action before G188, with its shortcut ----------------------------
# (the 3D menu's twin of File → Import → Calibration, `act_import_calib`, became that one action: `act_calib`)
GOLDEN = [
    ("act_open", "Ctrl+O"), ("act_open_folder", ""), ("act_open_proj", "Ctrl+Shift+O"), ("act_recover", ""),
    ("act_save", "Ctrl+S"), ("act_save_as", "Ctrl+Shift+S"),
    ("act_import_tracks", ""), ("act_import_xyz", ""), ("act_import_offsets", ""),
    ("act_import_masks", ""),
    ("act_export", "Ctrl+E"), ("act_exports", ""), ("act_overlay", ""), ("act_export_one", ""),
    ("act_check_frames", ""), ("act_quit", "Ctrl+Q"),
    ("act_undo", "Ctrl+Z"), ("act_select_all", "Ctrl+A"), ("act_note", ""), ("act_hidden", ""),
    ("act_annotator", ""), ("act_point_tools", ""), ("act_settings", "Ctrl+,"),
    ("text:Layers panel", "Ctrl+1"), ("act_onboarding", ""),
    ("act_sync_all", ""), ("act_sync_active", "Ctrl+Shift+2"), ("act_solo", "Ctrl+2"),
    ("act_show_all_views", ""), ("act_views_in_order", ""),
    ("act_show_midline", ""), ("act_show_bones", ""),
    ("text:Off", ""), ("text:Last 10 frames", ""), ("act_trail_custom", ""), ("act_trail_future", ""),
    ("act_onion", ""), ("act_loupe", ""), ("act_epipolar", ""),
    ("text:None", ""), ("text:Enhance contrast (CLAHE)", ""), ("text:Brighten dark footage (gamma)", ""),
    ("text:Frame difference (what moved since the previous frame)", ""),
    ("text:Use template: Lizard / iguana  (14 landmarks)", ""),
    ("text:Use template: Quadruped (generic)  (10 landmarks)", ""),
    ("text:Use template: Gliding mammal (flying squirrel)  (12 landmarks)", ""),
    ("text:Use template: Flying lizard (Draco)  (12 landmarks)", ""),
    ("text:Use template: Undulating body (fish / eel / swimming)  (5 landmarks)", ""),
    ("text:Use template: Bird / bat (wings)  (8 landmarks)", ""),
    ("text:Custom skeleton…", ""),
    ("act_mark_event", ""),
    ("act_sync", ""), ("act_cameras", ""), ("act_lens", ""), ("act_load_lens", ""), ("act_export_lens", ""),
    ("act_wand", ""), ("act_calib", ""), ("act_export_cal", ""), ("act_offsets3d", ""),
    ("act_recon", "Ctrl+3"), ("act_set_axes", ""), ("act_retrack", ""), ("act_hull", "Ctrl+4"),
    ("act_view3d", "Ctrl+5"), ("act_export_mesh", ""),
    ("act_body_run", ""), ("act_body_view", "Ctrl+6"), ("act_body_video", ""), ("act_body_joints", ""),
    ("act_body_angles", ""), ("act_body_clear", ""),
    ("act_manual", "F1"), ("act_tour", ""), ("act_help_models", ""), ("act_syscheck", ""),
    ("act_error_report", ""), ("act_folders", ""), ("act_start_menu", ""), ("text:Keyboard & Mouse Reference…", ""),
    ("act_updates", ""), ("act_about", ""),
]
assert len(GOLDEN) == 82, len(GOLDEN)


def label(a: QAction) -> str:
    """The entry's text as the user reads it: no mnemonic '&' ('&&' is a real '&'), no shortcut column."""
    return a.text().split("\t")[0].replace("&&", "\x00").replace("&", "").replace("\x00", "&")


def submenus(win) -> dict:
    """{C++ address of a menu's own action: the menu}. Found with findChildren: a wrapper from `QAction.menu()`
    that Python frees deletes the C++ menu (the PySide6 harness rule in .claude/rules/tests.md) - it took the
    Events and Skeleton menus with it at the first garbage collection."""
    return {shiboken6.getCppPointer(m.menuAction())[0]: m for m in win.menuBar().findChildren(QMenu)}


def tree(win, show=False):
    """[(path list, action, its submenu or None)] for every entry of the menu bar. With `show`, every menu is
    first sent aboutToShow (the ones that fill themselves then do) and nothing is collected: a refresh rebuilds a
    menu and deletes the actions a walk had already collected, so the caller shows first and walks second."""
    out, subs = [], submenus(win)

    def walk(menu, path):
        if show:
            menu.aboutToShow.emit()
            app.processEvents()
        for a in menu.actions():
            if a.isSeparator():
                continue
            sub = subs.get(shiboken6.getCppPointer(a)[0])
            out.append((path + [label(a)], a, sub))
            if sub is not None:
                walk(sub, path + [label(a)])
    for a in win.menuBar().actions():
        walk(subs[shiboken6.getCppPointer(a)[0]], [label(a)])
    return out


win = MainWindow()
win.resize(1280, 860)
win.show()
app.processEvents()

tree(win, show=True)
leaves = tree(win)
by_id = {id(a): p for p, a, sub in leaves if sub is None}
by_text = {p[-1]: a for p, a, sub in leaves if sub is None}
missing, wrong_key = [], []
for key, shortcut in GOLDEN:
    if key.startswith("text:"):
        act = by_text.get(key[5:])
        if act is None:
            missing.append(key)
            continue
    else:
        act = getattr(win, key, None)
        if act is None or id(act) not in by_id:
            missing.append(key)
            continue
    if act.shortcut().toString() != shortcut:
        wrong_key.append(f"{key}: {act.shortcut().toString()!r} != {shortcut!r}")
assert not missing, f"actions no longer reachable from the menu bar: {missing}"
assert not wrong_key, f"shortcuts changed: {wrong_key}"
print(f"(a) all {len(GOLDEN)} actions of the old menu bar are reachable, shortcuts unchanged OK")

# ---- (b) every menu path written in the code and the docs resolves ---------------------------------------
TOPS = ("File", "Edit", "View", "Skeleton", "Events", "3D", "Body", "Help")   # today's and every former top menu
# "3 Skeleton → 4 Track" (the getting-started steps) is not a menu path: no digit + space before the menu
START = re.compile(r"(?<![\w/-])(?<!\d )(" + "|".join(re.escape(t) for t in TOPS) + r")\s*(?:→|->)\s*")
ARROW = re.compile(r"→|->")
WORD = re.compile(r"[\w&+]+(?:['’\-][\w&+]+)*")
SKIP = re.compile(r"[\s*`_\"']*")
# entries whose label changes with the state (the text the docs may quote -> the label in a fresh window)
ALIASES = {"Add Kinetrace to the Start Menu…": ["Remove Kinetrace from the Start Menu…"]}


def words(s: str) -> list:
    return [w.lower().replace("'", "").replace("’", "") for w in WORD.findall(s)]


def core_words(name: str) -> list:
    """The ways prose quotes a label: whole, without its parenthesised hint, without an em-dash explanation."""
    out = []
    for text in [name] + ALIASES.get(name, []):
        for core in (text, re.sub(r"\s*\([^)]*\)[…\s]*$", "", text), text.split(" — ")[0],
                     re.split(r"\s{2,}\(", text)[0]):
            w = words(core)
            if w and w not in out:
                out.append(w)
    return out


children: dict = {}
for p, a, sub in leaves:
    children.setdefault(tuple(p[:-1]), []).append((p[-1], a, sub))
top_children = {label(a) for a in win.menuBar().actions()}


def match_entry(text: str, pos: int, path: list):
    """(entry name, its submenu or None, end position) of the entry the text at `pos` names, or None. Prose may shorten a
    label to its first words when no other entry of that menu starts with the same words ("3D → Reconstruct")."""
    end = ARROW.search(text, pos)
    end = min(end.start() if end else len(text), pos + 300)
    toks = [(m.group().lower().replace("'", "").replace("’", ""), m.end()) for m in WORD.finditer(text, pos, end)]
    sibs = children.get(tuple(path), [])
    best = None
    for name, act, sub in sibs:
        for cw in core_words(name):
            k = 0
            while k < len(cw) and k < len(toks) and toks[k][0] == cw[k]:
                k += 1
            if k == 0:
                continue
            whole = k == len(cw)
            unique = all(o is act or words(on)[:k] != cw[:k] for on, o, _ in sibs)
            if (whole or unique) and (best is None or k > best[0]):
                best = (k, name, sub, toks[k - 1][1])
    return None if best is None else best[1:]


def resolve(text: str, pos: int, top: str):
    """None when the path resolves, else the part that does not."""
    if top not in top_children:
        return f"no top-level menu {top!r}"
    path = [top]
    while True:
        pos = SKIP.match(text, pos).end()
        if text.startswith("…", pos) or text.startswith("...", pos):
            return None                       # "File → …": a placeholder, not an entry
        if text.startswith("](", pos):
            return None                       # "[Several cameras & 3D →](page)": a wiki link's arrow
        hit = match_entry(text, pos, path)
        if hit is None:
            return " > ".join(path) + " has no entry for " + repr(" ".join(text[pos:pos + 50].split()))
        name, sub, pos = hit
        path.append(name)
        nxt = ARROW.search(text, pos)
        between = text[pos:nxt.start()] if nxt else ""
        if sub is None or nxt is None or WORD.search(between):
            return None
        pos = nxt.end()


sources = [os.path.join(ROOT, "kinetrace", f) for f in sorted(os.listdir(os.path.join(ROOT, "kinetrace")))
           if f.endswith(".py")]
sources += [os.path.join(ROOT, "docs", "MANUAL.md"), os.path.join(ROOT, "README.md")]
wiki = os.path.join(ROOT, "wiki")
if not os.path.isdir(wiki):          # a worktree has no wiki clone: the main checkout's
    wiki = os.path.join(os.path.dirname(ROOT), "Kinetrace", "wiki")
if os.path.isdir(wiki):
    sources += [os.path.join(wiki, f) for f in sorted(os.listdir(wiki)) if f.endswith(".md")]
n_paths, stale = 0, []
for path in sources:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for m in START.finditer(text):
        n_paths += 1
        bad = resolve(text, m.end(), m.group(1))
        if bad:
            line = text.count("\n", 0, m.start()) + 1
            stale.append(f"{os.path.relpath(path, os.path.dirname(ROOT))}:{line}: {bad}")
for s in stale:
    print("  STALE", s.encode("ascii", "replace").decode())
if not stale:
    print(f"(b) all {n_paths} menu paths in the code and the docs resolve OK")

# ---- (c) the G188 layout --------------------------------------------------------------------------------
paths = {id(a): p for p, a, _ in leaves}
m_export = next(sub for p, a, sub in leaves if p == ["File", "Export"])
export_acts = [a for a in m_export.actions() if not a.isSeparator()]
want = [win.act_export, win.act_overlay, win.act_export_one, win.act_export_lens, win.act_export_cal,
        win.act_export_mesh, win.act_body_video, win.act_body_joints, win.act_body_angles, win.act_exports]
assert export_acts == want, [label(a) for a in export_acts]
for p, a, sub in leaves:
    if p[0] != "File" and sub is None:
        assert "export" not in p[-1].lower() or p[-1].startswith("Keep"), f"an export outside File > Export: {p}"
assert sum(a is win.act_calib for _, a, _s in leaves) == 1 and paths[id(win.act_calib)] == ["File", "Import", "Calibration…"]
print("(c) every export in File > Export, one calibration import OK")

# ---- (d) a shortcut whose action now sits in a submenu still fires: pressed with real keys ----------------
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from kinetrace.app import READY  # noqa: E402

VID = os.path.join(ROOT, "test600.mp4")
win._open_video(VID)
t0 = time.time()
while win.state != READY:
    app.processEvents()
    time.sleep(0.01)
    assert time.time() - t0 < 30, "the test video did not open"
win.activateWindow()
app.processEvents()
pressed = []
for key, shortcut in GOLDEN:
    act = getattr(win, key, None) if not key.startswith("text:") else None
    if act is None or not shortcut or len(paths[id(act)]) < 3:
        continue                                  # top-level entries' shortcuts are not in question
    chain = [sub for p, a, sub in leaves if sub is not None and paths[id(act)][:len(p)] == p]
    if not all(m.isEnabled() for m in chain):
        continue                                  # a menu off for a reason (Other cameras: one camera open)
    was = act.isEnabled()
    act.setEnabled(True)                          # the key's route is in question here, not the enable rule
    act.triggered.disconnect()                    # its dialog would block: count the press instead
    hits = []
    act.triggered.connect(lambda *_a, h=hits: h.append(1))
    QTest.keySequence(win, QKeySequence(shortcut))
    app.processEvents()
    act.setEnabled(was)
    assert hits, f"{shortcut} did not reach {key} ({' > '.join(paths[id(act)])})"
    pressed.append(shortcut)
assert "Ctrl+E" in pressed, pressed
print(f"(d) {', '.join(pressed)} fire(s) its action from inside the submenus OK")

win._dev_probe.wait(30000)
win.close()
app.processEvents()
assert not stale, f"(b) {len(stale)} of {n_paths} menu paths do not resolve (listed above)"
print("verify_menu_paths PASSED")
