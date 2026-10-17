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

from kinetrace import recent  # noqa: E402

FAILS = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
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


unit()

if FAILS:
    print(f"verify_recent FAILED ({len(FAILS)}): " + "; ".join(FAILS))
    sys.exit(1)
print("verify_recent PASSED")
