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
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication([])
from kinetrace.app import MainWindow  # noqa: E402

# ---- (a) the golden list: every menu-bar action before G188, with its shortcut ----------------------------
GOLDEN = [
    ("act_open", "Ctrl+O"), ("act_open_folder", ""), ("act_open_proj", "Ctrl+Shift+O"), ("act_recover", ""),
    ("act_save", "Ctrl+S"), ("act_save_as", "Ctrl+Shift+S"),
    ("act_import_tracks", ""), ("act_import_calib", ""), ("act_import_xyz", ""), ("act_import_offsets", ""),
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
assert len(GOLDEN) == 83, len(GOLDEN)


def label(a: QAction) -> str:
    """The entry's text as the user reads it: no mnemonic '&' ('&&' is a real '&'), no shortcut column."""
    return a.text().split("\t")[0].replace("&&", "\x00").replace("&", "").replace("\x00", "&")


def tree(win, show=False):
    """[(path list, action)] for every entry of the menu bar. With `show`, every menu is first sent aboutToShow
    (the ones that fill themselves then do) and nothing is collected: a refresh rebuilds a menu and deletes the
    actions a walk had already collected, so the caller shows first and walks second."""
    out = []

    def walk(menu, path):
        if show:
            menu.aboutToShow.emit()
            app.processEvents()
        for a in menu.actions():
            if a.isSeparator():
                continue
            if a.menu() is not None:
                out.append((path + [label(a)], a))
                walk(a.menu(), path + [label(a)])
            else:
                out.append((path + [label(a)], a))
    for a in win.menuBar().actions():
        walk(a.menu(), [label(a)])
    return out


win = MainWindow()
win.resize(1280, 860)
win.show()
app.processEvents()

tree(win, show=True)
leaves = tree(win)
by_id = {id(a): p for p, a in leaves if a.menu() is None}
by_text = {p[-1]: a for p, a in leaves if a.menu() is None}
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
for p, a in leaves:
    children.setdefault(tuple(p[:-1]), []).append((p[-1], a))
top_children = {label(a): a for a in win.menuBar().actions()}


def match_entry(text: str, pos: int, path: list):
    """(entry name, action, end position) of the entry the text at `pos` names, or None. Prose may shorten a
    label to its first words when no other entry of that menu starts with the same words ("3D → Reconstruct")."""
    end = ARROW.search(text, pos)
    end = min(end.start() if end else len(text), pos + 300)
    toks = [(m.group().lower().replace("'", "").replace("’", ""), m.end()) for m in WORD.finditer(text, pos, end)]
    sibs = children.get(tuple(path), [])
    best = None
    for name, act in sibs:
        for cw in core_words(name):
            k = 0
            while k < len(cw) and k < len(toks) and toks[k][0] == cw[k]:
                k += 1
            if k == 0:
                continue
            whole = k == len(cw)
            unique = all(o is act or words(on)[:k] != cw[:k] for on, o in sibs)
            if (whole or unique) and (best is None or k > best[0]):
                best = (k, name, act, toks[k - 1][1])
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
        name, act, pos = hit
        path.append(name)
        nxt = ARROW.search(text, pos)
        between = text[pos:nxt.start()] if nxt else ""
        if act.menu() is None or nxt is None or WORD.search(between):
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
assert not stale, f"{len(stale)} of {n_paths} menu paths do not resolve"
print(f"(b) all {n_paths} menu paths in the code and the docs resolve OK")

win._dev_probe.wait(30000)
win.close()
app.processEvents()
print("verify_menu_paths PASSED")
