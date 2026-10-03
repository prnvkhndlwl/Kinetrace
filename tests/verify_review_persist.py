"""Persistence fixes from the 2026-10-02 code review (docs/AUDIT.md: I163-I165,
I170, I210-I214, I244-I246, G110, G111, G141, G142, R16, R17): projectfile.py,
recovery.py, autoexport.py, crashlog.py, errors.py. Every check here FAILS on the
code before the fixes. No GPU; the exports check imports Qt (offscreen).

Run: .venv\\Scripts\\python.exe tests\\verify_review_persist.py
"""
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from kinetrace import projectfile as pf  # noqa: E402
from kinetrace import recovery  # noqa: E402
from kinetrace.body import BodyTrack, rig_of  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "review_persist")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
FAILS = []
RNG = np.random.default_rng(3)


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def section(title):
    """Run the function below as a section: an exception is a failed check, not a crash."""
    def deco(fn):
        print(f"\n[{title}]")
        try:
            fn()
        except Exception as e:      # noqa: BLE001 - on the old code most sections end here
            import traceback
            check(False, f"{title}: raised {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc(limit=3)
        return fn
    return deco


def mini(names=("cam1",), points=("snout", "tail"), T=60):
    sessions = []
    for cn in names:
        s = TrackingSession(os.path.join(OUT, "videos", cn + ".mp4"), T, 30.0, 640, 480)
        for j, pn in enumerate(points):
            s.add_point(0, 10.0 + j, 20.0 + j, name=pn)
        n = s.n_points
        s.tracks[:] = RNG.uniform(0, 600, (T, n, 2)).astype(np.float32)
        s.tracked[:] = True
        s.visibility[:] = True
        s.confidence[:] = 0.9
        sessions.append(s)
    return Project(sessions, list(names))


def tree(d, skip=(".cache", ".history", ".saving", "exports")):
    out = {}
    for dirpath, dirs, fnames in os.walk(d):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in fnames:
            fp = os.path.join(dirpath, f)
            out[os.path.relpath(fp, d).replace("\\", "/")] = open(fp, "rb").read()
    return out


def put(path, data="x", mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as fh:
        fh.write(data)


def refused(fn, needles, what):
    """fn must raise a ProjectFileError whose message holds every needle."""
    try:
        fn()
    except pf.ProjectFileError as e:
        check(all(n in str(e) for n in needles), f"{what}: refused naming {needles} -> '{str(e)[:110]}'")
    except Exception as e:      # noqa: BLE001
        check(False, f"{what}: a raw {type(e).__name__} instead of a ProjectFileError ({str(e)[:80]})")
    else:
        check(False, f"{what}: should have been refused")


# ------------------------------------------------------------------ I163
@section("I163 a rename by letter case keeps the data")
def _i163():
    path = os.path.join(OUT, "case.kinetrace")
    p = mini()
    pf.save(p, path)
    p.rename_landmark("snout", "Snout")
    st = pf.save(p, path)
    q = pf.load(path)
    check(q.sessions[0].points[0].name == "Snout" and int(q.sessions[0].tracked[:, 0].sum()) == 60,
          "landmark snout -> Snout: saved and reopened with all 60 frames")
    names = sorted(os.listdir(os.path.join(path, "cameras", "cam1", "tracks")))
    check(names == ["Snout.csv", "tail.csv"] and "tidy_up" not in st, f"the old file is gone, the new one stays {names}")
    p.names[0] = "Cam1"
    pf.save(p, path)
    q = pf.load(path)
    check(q.names == ["Cam1"] and int(q.sessions[0].tracked.sum()) == 120 and q.sessions[0].points[0].name == "Snout",
          "camera cam1 -> Cam1: the camera keeps all its data")
    p.rename_landmark("Snout", "snout")
    pf.save(p, path)
    check(int(pf.load(path).sessions[0].tracked.sum()) == 120, "... and back again")


# ------------------------------------------------------------------ I164 / I165
@section("I164 / I165 a save never takes the user's files; rollback works with hidden files around")
def _i164():
    path = os.path.join(OUT, "user.kinetrace")
    p = mini(("cam1", "cam2"), ("snout", "tail", "wing"))
    pf.save(p, path)
    extras = {"cameras/cam1/tracks/snout.xlsx": "an Excel copy", "cameras/cam1/tracks/notes.txt": "my note",
              "cameras/cam1/tracks/.DS_Store": "mac", "cameras/cam1/tracks/.~lock.snout.csv#": "lo",
              "cameras/cam1/.DS_Store": "mac", "cameras/cam1 - Copy/points.csv": "name\nold\n",
              "cameras/cam1 - Copy/tracks/snout.csv": "frame,x,y\n", ".DS_Store": "mac", "my-notes.txt": "mine",
              "cameras/cam2/silhouette/readme.txt": "mine too"}
    for rel, data in extras.items():
        put(os.path.join(path, *rel.split("/")), data)

    def intact(what):
        bad = [r for r, d in extras.items() if not os.path.isfile(os.path.join(path, *r.split("/")))
               or open(os.path.join(path, *r.split("/"))).read() != d]
        check(not bad, f"{what}: the user's files are all still there {bad[:2]}")
    p.sessions[0].tracks[5, 0] = (1.5, 2.5)
    pf.save(p, path)
    pf.save(p, path)
    intact("after two saves")
    p.rename_landmark("snout", "nose")
    pf.save(p, path)
    p.remove_view(1)
    pf.save(p, path)
    p.sessions[0].remove_point(1)
    pf.save(p, path)
    intact("after a rename, a removed camera and a removed landmark")
    check(not os.path.exists(os.path.join(path, "cameras", "cam2", "points.csv")), "the removed camera's own files went")
    # a save that fails half-way, with hidden files and odd folders around: undone, the real error says
    p.sessions[0].tracks[:, 0] += 1.0
    p.rename_landmark("nose", "snout")
    for rel, data in extras.items():                 # (put back what an older save took, so they are all there now)
        put(os.path.join(path, *rel.split("/")), data)
    before = tree(path)
    real, n = pf._replace, {"n": 0}

    def failing(src, dst):
        n["n"] += 1
        if n["n"] == 3:
            raise PermissionError("the file is open in another program")
        return real(src, dst)
    pf._replace = failing
    try:
        pf.save(p, path)
        check(False, "the failing save should raise")
    except PermissionError:
        check(True, "the failing save raises its own error (not a refusal from the rollback)")
    except Exception as e:      # noqa: BLE001
        check(False, f"the failing save raised {type(e).__name__}: {str(e)[:100]} instead of its own error")
    finally:
        pf._replace = real
    check(tree(path) == before, "every file as it was, hidden and user files included")
    q = pf.load(path)
    check(q.sessions[0].points[0].name == "nose", "the project opens as the last save")
    intact("after the failed save")
    # without an index (cache deleted): files of the project's own kinds are the project's, the user's are not
    check(not os.path.exists(os.path.join(path, ".cache", "index.json")), "(a failed save leaves no index)")
    p.rename_landmark("snout", "beak")
    pf.save(p, path)
    t = tree(path)
    check("cameras/cam1/tracks/nose.csv" not in t and "cameras/cam1/tracks/beak.csv" in t,
          "no index: a renamed landmark's old file still goes")
    intact("... and the user's files stay")


# ------------------------------------------------------------------ I210
@section("I210 hidden files (AppleDouble, .DS_Store) are not part of a project")
def _i210():
    path = os.path.join(OUT, "apple.kinetrace")
    p = mini()
    pf.save(p, path)
    for rel in ("cameras/cam1/tracks/._snout.csv", "cameras/cam1/._points.csv", "cameras/cam1/body/._joints3d.npy",
                "cameras/cam1/silhouette/._x.npy", "cameras/cam1/tracks/.DS_Store", "._kinetrace.json"):
        put(os.path.join(path, *rel.split("/")), b"AppleDouble\x00\x05\x16\x07", "wb")
    q = pf.load(path)
    check([x.name for x in q.sessions[0].points] == ["snout", "tail"] and q.sessions[0].body is None,
          "a folder with ._x.csv / ._x.npy files opens with the same landmarks")
    # the single file: macOS's zip adds __MACOSX/ and ._ members
    one = os.path.join(OUT, "apple-one.kinetrace")
    pf.save(p, one, single_file=True)
    with zipfile.ZipFile(one, "a") as z:
        z.writestr("cameras/cam1/tracks/._snout.csv", b"junk")
        z.writestr("__MACOSX/cameras/cam1/tracks/._snout.csv", b"junk")
        z.writestr("cameras/cam1/body/._joints3d.npy", b"junk")
    check([x.name for x in pf.load(one).sessions[0].points] == ["snout", "tail"],
          "the same in a single file")
    # and a project folder that lives under a folder whose own name starts with a dot opens
    hidden = os.path.join(OUT, ".hiddenparent", "dot.kinetrace")
    os.makedirs(os.path.dirname(hidden))
    pf.save(p, hidden)
    check(len(pf.load(hidden).sessions[0].points) == 2, "a project inside a folder named .something opens (only names inside count)")


# ------------------------------------------------------------------ I211 / I212
@section("I211 a name that differs by Unicode form or letter case is the same file")
def _i211():
    path = os.path.join(OUT, "names.kinetrace")
    p = mini(points=("snout", "\u00e9tude", "tail"))
    pf.save(p, path)
    tr = os.path.join(path, "cameras", "cam1", "tracks")
    nfc, nfd = "\u00e9tude.csv", "e\u0301tude.csv"
    if os.path.isfile(os.path.join(tr, nfc)):
        os.replace(os.path.join(tr, nfc), os.path.join(tr, nfd))      # as HFS+ / a zip made on a Mac lists it
    q = pf.load(path)
    names = [x.name for x in q.sessions[0].points]
    check(names == ["snout", "\u00e9tude", "tail"] and int(q.sessions[0].tracked[:, 1].sum()) == 60,
          f"a decomposed file name finds its landmark: one landmark with its data {names}")
    pcsv = os.path.join(path, "cameras", "cam1", "points.csv")
    text = open(pcsv, encoding="utf-8", newline="").read().replace("snout.csv", "SNOUT.CSV")
    open(pcsv, "w", encoding="utf-8", newline="").write(text)
    q = pf.load(path)
    check([x.name for x in q.sessions[0].points] == ["snout", "\u00e9tude", "tail"]
          and int(q.sessions[0].tracked[:, 0].sum()) == 60,
          "a points.csv `file` that differs from the disk only in case: not a second landmark")


@section("I212 one name rule for format-1 tracks.csv, ball_prompts.json and spots.json")
def _i212():
    hand = os.path.join(OUT, "format1")
    put(os.path.join(hand, "kinetrace.json"), json.dumps({"format": "kinetrace-project", "format_version": 1}))
    put(os.path.join(hand, "project.json"), json.dumps({"cameras": [{"name": "top", "folder": "top",
                                                                    "video": {"path": "top.mp4"}, "n_frames": 20,
                                                                    "fps": 30, "width": 640, "height": 480}]}))
    put(os.path.join(hand, "cameras", "top", "points.csv"), "name,color\n-Y axis,#ff0000\n")
    put(os.path.join(hand, "cameras", "top", "tracks.csv"), "frame,point,x,y\n0,-Y axis,10,20\n1,-Y axis,11,21\n")
    q = pf.load(hand)
    s = q.sessions[0]
    check([x.name for x in s.points] == ["Y axis"] and int(s.tracked.sum()) == 2,
          f"a landmark named '-Y axis' is ONE landmark with its settings and its data {[x.name for x in s.points]}")
    path = os.path.join(OUT, "keys.kinetrace")
    p = mini(points=("=ball", "-p", "other"))
    s = p.sessions[0]
    s.points[0].source = "ball"
    s.points[0].ball_prompts = {3: [[10.0, 11.0, 1]]}
    s.points[1].spot = {"cue": "bright", "radius": 6.0, "speed_gain": 0.5, "sigma": 1.5}
    pf.save(p, path)
    q = pf.load(path)
    s2 = q.sessions[0]
    check([x.name for x in s2.points] == ["ball", "p", "other"] and s2.points[0].ball_prompts == {3: [[10.0, 11.0, 1]]}
          and s2.points[1].spot is not None, "ball_prompts.json / spots.json keys match a name cleaned on reading")


# ------------------------------------------------------------------ I213
@section("I213 a file open in .history does not block a save")
def _i213():
    path = os.path.join(OUT, "locked.kinetrace")
    p = mini()
    pf.save(p, path)
    p.sessions[0].tracks[3, 0] = (5.5, 6.5)
    pf.save(p, path)
    held = None
    for dp, _d, fs in os.walk(os.path.join(path, ".history")):
        for f in fs:
            if f.endswith(".csv"):
                held = open(os.path.join(dp, f), "rb")      # a diff tool comparing the previous save (Windows: no delete)
                break
        if held:
            break
    check(held is not None, "a CSV of the previous save is held open")
    p.sessions[0].tracks[4, 0] = (7.5, 8.5)
    try:
        pf.save(p, path)
        ok = True
    except Exception as e:      # noqa: BLE001
        ok = False
        check(False, f"the save raised {type(e).__name__}: {str(e)[:100]}")
    finally:
        held.close()
    if ok:
        q = pf.load(path)
        check(np.allclose(q.sessions[0].tracks[4, 0], (7.5, 8.5)), "the save went through")
        check(os.path.isfile(os.path.join(path, ".history", "previous.json")), "and left its previous-save record")


# ------------------------------------------------------------------ I214 / G111
def _edit_project_json(path, fn):
    p = os.path.join(path, "project.json")
    d = json.load(open(p, encoding="utf-8"))
    fn(d)
    open(p, "w", encoding="utf-8").write(json.dumps(d))


@section("I214 rates and offsets in project.json are checked")
def _i214():
    path = os.path.join(OUT, "rates.kinetrace")
    pf.save(mini(("cam1", "cam2")), path)
    good = open(os.path.join(path, "project.json"), "rb").read()
    for what, setter in (("rate 0", lambda d: d["cameras"][1].__setitem__("rate", 0)),
                         ("a negative rate", lambda d: d["cameras"][1].__setitem__("rate", -2.0)),
                         ("rate null", lambda d: d["cameras"][1].__setitem__("rate", None)),
                         ("rate text", lambda d: d["cameras"][1].__setitem__("rate", "fast")),
                         ("offset null", lambda d: d["cameras"][1].__setitem__("offset", None)),
                         ("offset text", lambda d: d["cameras"][1].__setitem__("offset", "x"))):
        open(os.path.join(path, "project.json"), "wb").write(good)
        _edit_project_json(path, setter)
        refused(lambda: pf.read(path), ("project.json", "camera 2"), what)
    open(os.path.join(path, "project.json"), "wb").write(good.replace(b'"offset": 0.0', b'"offset": NaN', 1))
    refused(lambda: pf.read(path), ("project.json", "camera 1", "offset"), "a NaN offset")
    open(os.path.join(path, "project.json"), "wb").write(good)
    check(pf.read(path)[0].rates == [1.0, 1.0], "a good file still opens")


@section("G111 hand-edited files are refused with the file (and row), never a raw error")
def _g111():
    path = os.path.join(OUT, "g111.kinetrace")
    p = mini(points=("a", "ball"))
    p.sessions[0].points[1].source = "ball"
    p.sessions[0].points[1].ball_prompts = {2: [[1.0, 2.0, 1]]}
    p.sessions[0].set_note(4, "a note")
    from kinetrace.session import AnimalMeta
    p.sessions[0].animal = AnimalMeta("lizard", (10, 200, 30))
    bt = BodyTrack(60, rig_of("coco17"), 1)
    bt.set_person(0, 0, joints2d=RNG.uniform(0, 600, (17, 2)), conf=RNG.random(17), score=0.9)
    p.sessions[0].body = bt
    pf.save(p, path)
    base = tree(path)
    cam = os.path.join(path, "cameras", "cam1")

    def restore():
        for r, d in base.items():
            put(os.path.join(path, *r.split("/")), d, "wb")

    def edit(rel, fn, mode="text"):
        restore()
        fp = os.path.join(path, *rel.split("/"))
        if mode == "text":
            new = fn(open(fp, encoding="utf-8", newline="").read())
            open(fp, "w", encoding="utf-8", newline="").write(new)
        else:
            open(fp, "wb").write(fn)
    pcsv = open(os.path.join(cam, "points.csv"), encoding="utf-8").read().splitlines()
    hdr = pcsv[0].split(",")
    row = pcsv[1].split(",")
    row[hdr.index("radius")] = "abc"
    edit("cameras/cam1/points.csv", lambda t: "\n".join([pcsv[0], ",".join(row)] + pcsv[2:]) + "\n")
    refused(lambda: pf.read(path), ("points.csv", "row 2"), "a radius that is not a number")
    row = pcsv[1].split(",")
    row[hdr.index("outline")] = "1 x 3 4"
    edit("cameras/cam1/points.csv", lambda t: "\n".join([pcsv[0], ",".join(row)] + pcsv[2:]) + "\n")
    refused(lambda: pf.read(path), ("points.csv", "row 2"), "an outline that is not numbers")
    edit("cameras/cam1/notes.csv", lambda t: t.replace("4,a note", "4.0,a note"))
    refused(lambda: pf.read(path), ("notes.csv", "row 2"), "a note frame typed 4.0")
    edit("cameras/cam1/ball_prompts.json", lambda t: '{"ball": {"2": "oops"}}', "text")
    refused(lambda: pf.read(path), ("ball_prompts.json",), "a ball's clicks that are not clicks")
    edit("cameras/cam1/ball_prompts.json", lambda t: '[1, 2]', "text")
    refused(lambda: pf.read(path), ("ball_prompts.json",), "ball_prompts.json that is a list")
    edit("cameras/cam1/segment.json", lambda t: '{"name": ', "text")
    refused(lambda: pf.read(path), ("segment.json",), "a broken segment.json")
    edit("cameras/cam1/body/meta.json", lambda t: "not json", "text")
    refused(lambda: pf.read(path), ("body",), "a broken body meta.json")
    restore()
    lens = os.path.join(path, "lenses.json")
    open(lens, "w").write('[{"width": 1}]')
    refused(lambda: pf.read(path), ("lenses.json",), "a lens profile missing its numbers")
    os.remove(lens)
    cal = os.path.join(path, "calibration.json")
    open(cal, "w").write('{"coefs": [[1,2,3,4,5,6,7,8,9,10,11]], "cams": [1]}')
    refused(lambda: pf.read(path), ("calibration.json",), "a calibration with a camera that is not an object")
    os.remove(cal)
    restore()
    check(pf.read(path)[0].sessions[0].notes[4]["text"] == "a note", "the untouched project still opens")


# ------------------------------------------------------------------ I244
@section("I244 a folder holding only Kinetrace's own dot entries is reusable")
def _i244():
    d = os.path.join(OUT, "first.kinetrace")
    os.makedirs(os.path.join(d, ".cache"))
    os.makedirs(os.path.join(d, ".saving"))
    pf.save(mini(), d)
    check(pf.is_folder_project(d) and len(pf.load(d).sessions[0].points) == 2,
          "a first save over a folder a failed first save left (.cache, .saving)")
    d2 = os.path.join(OUT, "first2.kinetrace")
    os.makedirs(os.path.join(d2, ".cache"))
    open(os.path.join(d2, "thesis.docx"), "w").write("x")
    refused(lambda: pf.save(mini(), d2), ("not a Kinetrace project",), "a folder with a file of the user's")
    check(os.listdir(d2).count("thesis.docx") == 1 and not os.path.exists(os.path.join(d2, "kinetrace.json")),
          "... is left alone")


# ------------------------------------------------------------------ I246
@section("I246 a live lock is respected; a reused pid is not a lock")
def _i246():
    path = os.path.join(OUT, "lock.kinetrace")
    p = mini()
    pf.save(p, path)
    start = pf._proc_start(os.getpid()) if hasattr(pf, "_proc_start") else None
    me = {"pid": os.getpid(), "host": socket.gethostname(), "time": time.time(), "start": start}
    pend = os.path.join(path, ".history", "pending.json")
    put(pend, json.dumps({"saved_at": "another save", "saved_at_before": "x", "added": [], "replaced": [], "removed": []}))
    put(os.path.join(path, ".lock"), json.dumps(me))
    before = tree(path)
    refused(lambda: pf.read(path), ("being saved",), "opening while a live process is mid-save")
    check(os.path.isfile(pend) and tree(path) == before and os.path.isfile(os.path.join(path, ".lock")),
          "its pending record and its lock are left alone (the save is not rolled back under the writer)")
    os.remove(os.path.join(path, ".lock"))
    q, _s, meta = pf.read(path)
    check(meta["_interrupted"] == "undone", "without a live lock the cut-short save is undone as before")
    if start is None:
        print("  skip  no process start time on this system: the reused-pid check")
        return
    put(os.path.join(path, ".lock"), json.dumps(dict(me, start=str(start) + "0")))     # same pid, another process
    try:
        pf.save(p, path)
        check(True, "a lock whose pid is now another process does not refuse a save")
    except pf.ProjectFileError as e:
        check(False, f"a reused pid refused the save: {e}")
    check(not os.path.exists(os.path.join(path, ".lock")), "... and is released")
    put(os.path.join(path, ".lock"), json.dumps(me))                       # the real owner: refused
    refused(lambda: pf.save(p, path), ("being saved by another",), "a lock of a live process")
    os.remove(os.path.join(path, ".lock"))


# ------------------------------------------------------------------ G110
@section("G110 an error after the save counted is not 'nothing was changed'")
def _g110():
    path = os.path.join(OUT, "tidy.kinetrace")
    p = mini()
    pf.save(p, path)
    p.sessions[0].tracks[3, 0] = (50.5, 60.5)
    real_json = pf._write_json_atomic
    real_replace = pf._replace

    def no_index(target, obj, fsync=True):
        if Path(target).name == "index.json":
            raise OSError("the index is held by another program")
        return real_json(target, obj, fsync)

    def no_previous(src, dst):
        if Path(dst).name == "previous.json":
            raise OSError("previous.json is held by another program")
        return real_replace(src, dst)
    pf._write_json_atomic, pf._replace = no_index, no_previous
    try:
        st = pf.save(p, path)
        check(st is not None and st.get("tidy_up") and st["written"] >= 2,
              f"the save stands and says what did not work ({st})")
    except Exception as e:      # noqa: BLE001
        check(False, f"a post-commit problem failed the whole save: {type(e).__name__}: {str(e)[:100]}")
    finally:
        pf._write_json_atomic, pf._replace = real_json, real_replace
    q, _s, meta = pf.read(path)
    check(np.allclose(q.sessions[0].tracks[3, 0], (50.5, 60.5)) and meta["_interrupted"] in (None, "finished"),
          f"the folder holds the new save ({meta['_interrupted']})")
    p.sessions[0].tracks[4, 0] = (1.0, 2.0)
    st = pf.save(p, path)
    check(st["written"] >= 2 and "tidy_up" not in st, "the next save is normal")


# ------------------------------------------------------------------ R16 planner
@section("R16 one planner: the close question says what a save would do")
def _r16():
    path = os.path.join(OUT, "plan.kinetrace")
    p = mini()
    pf.save(p, path)
    fz = lambda: pf.freeze(p, {}, "p" * 32, target=path)      # noqa: E731
    check(pf.changes_since_save(fz(), path) == ([], []), "nothing differs after a save")
    gone = os.path.join(path, "cameras", "cam1", "tracks", "snout.csv")
    os.remove(gone)
    got = pf.changes_since_save(fz(), path)
    check(got is not None and "cameras/cam1/tracks/snout.csv" in got[0],
          f"a CSV deleted from the folder is a change a save would repair ({got})")
    st = pf.save(p, path)
    check(os.path.isfile(gone) and st["written"] == 2, "... and the save does repair it")
    put(os.path.join(path, "cameras", "cam1", "tracks", "snout.xlsx"), "mine")
    p.rename_landmark("snout", "nose")
    got = pf.changes_since_save(fz(), path)
    check(got == (["cameras/cam1/points.csv", "cameras/cam1/tracks/nose.csv"], ["cameras/cam1/tracks/snout.csv"]),
          f"a rename: the new file written, the old one removed, the user's xlsx not named ({got})")
    st = pf.save(p, path)
    check(st["removed"] == 1 and st["written"] == len(got[0]) + 1, f"the save does exactly that ({st})")
    check(recovery.safe_id is pf.safe_id and pf.is_camera_folder("cam1") and not pf.is_camera_folder(".x"),
          "one project-id rule and one camera-folder rule")


# ------------------------------------------------------------------ R17
@section("R17 the body mesh is not joined, copied and hashed again at every save")
def _r17():
    p = mini()
    s = p.sessions[0]
    bt = BodyTrack(60, rig_of("coco17"), 1)
    faces = np.array([[0, 1, 2], [2, 3, 0]], np.int32)
    for f in range(0, 60, 3):
        bt.set_person(f, 0, joints2d=RNG.uniform(0, 600, (17, 2)), conf=RNG.random(17), score=0.9,
                      vertices=RNG.uniform(-1, 1, (4, 3)).astype(np.float32), faces=faces)
    s.body = bt
    calls = {"n": 0}
    real = BodyTrack.mesh_arrays

    def counting(self, prefix):
        calls["n"] += 1
        return real(self, prefix)
    BodyTrack.mesh_arrays = counting
    try:
        f1 = pf.freeze(p, {}, "r" * 32, saved_at="t1")
        f2 = pf.freeze(p, {}, "r" * 32, saved_at="t2")
        check(calls["n"] == 1, f"two saves of an unchanged mesh join it once ({calls['n']} joins)")
        f3 = pf.freeze(p, {}, "r" * 32, binary_tracks=True, saved_at="t3")
        check(calls["n"] == 1, "... and the 30 s recovery copy shares it")
        bt.set_person(1, 0, vertices=RNG.uniform(-1, 1, (4, 3)).astype(np.float32))
        f4 = pf.freeze(p, {}, "r" * 32, saved_at="t4")
        check(calls["n"] == 2, "a changed mesh is joined again")
    finally:
        BodyTrack.mesh_arrays = real
    expect = {k: v for k, v in bt.to_arrays("body_", mesh=True).items() if k.startswith("body_mesh_")}
    for k, v in expect.items():
        buf = io.BytesIO()
        np.lib.format.write_array(buf, np.asarray(v), allow_pickle=False)
        got = f4.files[f"cameras/cam1/body/{k[5:]}.npy"]
        arr = got[1] if isinstance(got, tuple) else got
        buf2 = io.BytesIO()
        np.lib.format.write_array(buf2, np.asarray(arr), allow_pickle=False)
        check(buf.getvalue() == buf2.getvalue(), f"{k}: the same bytes as the array the old code wrote")
    # through a real save: bit-identical to the files of a project that never cached, and the second save skips it
    path = os.path.join(OUT, "mesh.kinetrace")
    pf.write_folder(f1, path)
    st = pf.write_folder(pf.freeze(p, {}, "r" * 32, saved_at="t5"), path)
    q = pf.load(path)
    check(sorted(q.sessions[0].body.mesh) == sorted(bt.mesh) and all(
        np.array_equal(q.sessions[0].body.mesh[k], bt.mesh[k]) for k in bt.mesh), "the mesh round-trips")
    check(st["written"] <= 4, f"a later save writes only what changed ({st})")


# ------------------------------------------------------------------ I170
@section("I170 exports/ writes 3D points in the world of the exported cameras")
def _i170():
    from kinetrace import autoexport, calibio
    from kinetrace.calib import Calibration, CameraCalibration, NoUndistort, Reconstruction, dlt_from_camera
    p = mini(("camA", "camB"))
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    cams = []
    for a in (0.2, 1.3):
        pos = np.array([3 * np.cos(a), 3 * np.sin(a), 1.0])
        z = -pos / np.linalg.norm(pos)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z]) @ np.diag([1.0, 1.0, -1.0])        # a left-handed world, as easyWand's
        cams.append(CameraCalibration(dlt_from_camera(K, R, -R @ pos), 640, 480, NoUndistort(), pixel_origin=0.0))
    p.calibration = Calibration(cams, "m", "unit test")
    p.calibration.origin_shift = np.array([0.1, -0.2, 0.3])
    xyz = RNG.normal(size=(20, 2, 3))
    p.reconstruction = Reconstruction(2, ["snout", "tail"], xyz, RNG.random((20, 2)),
                                      np.full((20, 2), 2, np.int32), "m", None)
    path = os.path.join(OUT, "world.kinetrace")
    pf.save(p, path)
    written, problems = autoexport.refresh(path, ["xyz_dltdv"])
    check("xyzpts.csv" in written and not problems, f"xyzpts.csv written ({written}, {problems})")
    rows = np.genfromtxt(os.path.join(path, "exports", "xyzpts.csv"), delimiter=",", skip_header=1)
    got = rows.reshape(rows.shape[0], 2, 3)[2:]
    rec = pf.load(path).reconstruction
    probe = np.nanmedian(rec.xyz.reshape(-1, 3), axis=0)
    models = calibio.to_models(pf.load(path).calibration, ["camA", "camB"], probe)
    want = models.world_to_export(rec.xyz.reshape(-1, 3)).reshape(rec.xyz.shape)
    check(models.mirrored, "the calibration is left-handed: its exported cameras are mirrored in Z")
    check(np.allclose(got, want, atol=1e-5) and not np.allclose(got, rec.xyz, atol=1e-3),
          "the points are the ones the exported cameras reproject (mirrored / shifted like them)")


# ------------------------------------------------------------------ crashlog
CHILD = r'''
import os, sys, time
sys.path.insert(0, {root!r})
from kinetrace import crashlog
crashlog.MAX_BYTES = {mx}
crashlog.LOG_BACKUPS = 200
ok = crashlog.install("child")
print("INSTALL", ok, os.getpid(), flush=True)
while not os.path.exists({go!r}):
    time.sleep(0.005)
pid = os.getpid()
for i in range({n}):
    crashlog._logger.error("ENTRY %d %d %s" % (pid, i, "x" * 90))
    if i % 7 == 0:
        time.sleep(0.001)
print("DONE", flush=True)
'''


@section("I245 two Kinetrace windows share one log without losing entries")
def _i245():
    d = os.path.join(OUT, "logs")
    os.makedirs(d)
    go = os.path.join(d, "go")
    env = dict(os.environ, KINETRACE_LOG_DIR=d, QT_QPA_PLATFORM="offscreen")
    n = 250
    kids = [subprocess.Popen([sys.executable, "-c", CHILD.format(root=ROOT, mx=4000, go=go, n=n)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    time.sleep(2.0)
    open(go, "w").write("go")
    outs = [k.communicate(timeout=120) for k in kids]
    pids = {int(o[0].split("INSTALL True")[1].split()[0]) for o in outs if "INSTALL True" in o[0]}
    text = ""
    for f in sorted(os.listdir(d)):
        if f.startswith("kinetrace.log"):
            text += open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
    seen = {(int(ln.split()[1]), int(ln.split()[2])) for ln in text.splitlines() if ln.startswith("ENTRY ")}
    missing = [(pid, i) for pid in pids for i in range(n) if (pid, i) not in seen]
    check(all("DONE" in o[0] for o in outs) and len(pids) == 2, "both windows finished")
    check(not missing, f"every entry of both windows is in the log ({len(missing)} of {2 * n} missing)")
    ended = text.count("==== ended normally")
    check(ended == 2, f"both 'ended normally' lines are there ({ended})")
    # a crash log another window holds open and that is over its size must not stop the log
    d2 = os.path.join(OUT, "logs2")
    os.makedirs(d2)
    held = open(os.path.join(d2, "kinetrace-crash.log"), "a")
    held.write("x" * 5000)
    held.flush()
    env2 = dict(env, KINETRACE_LOG_DIR=d2)
    open(os.path.join(d2, "go"), "w").write("go")
    out = subprocess.run([sys.executable, "-c", CHILD.format(root=ROOT, mx=1000, go=os.path.join(d2, "go"), n=1)],
                         env=env2, capture_output=True, text=True, timeout=60)
    held.close()
    check("INSTALL True" in out.stdout, f"a crash log held by another window does not turn logging off ({out.stdout.split()[:2]})")


@section("I245 an 'ended normally' line belongs to the session of its pid")
def _i245b():
    from kinetrace import crashlog
    d = os.path.join(OUT, "logs3")
    os.makedirs(d)
    a, b = "==== Kinetrace 1 started t, pid 1111 | A", "==== Kinetrace 1 started t, pid 2222 | B"
    open(os.path.join(d, "kinetrace.log"), "w").write("\n".join([a, b, "==== ended normally t, pid 1111", ""]))
    old = crashlog._folder
    crashlog._folder = Path(d)
    try:
        ended = crashlog._ended_sessions()
        crash = "\n".join([a, "Fatal Python error: A handled", b, "Fatal Python error: B real"])
        text = "\n".join(crashlog._crash_sections(crash))
    finally:
        crashlog._folder = old
    check(ended == {a}, f"only the window that ended is paired with its end line ({sorted(ended)})")
    check("B real" in text and text.index("pid 2222") < text.index("did NOT end normally") < text.index("B real"),
          "the report calls the window that did not end 'did NOT end normally'")


@section("G141 the red notice names Kinetrace's own frame, not a library under the install folder")
def _g141():
    from kinetrace import crashlog
    pkg = Path(crashlog.__file__).resolve().parent
    ns = {}
    exec(compile("def inner():\n    raise ValueError('x')\n", "C:/Users/someone/kinetrace-install/.venv/Lib/zipfile.py", "exec"), ns)
    ns2 = {"inner": ns["inner"]}
    exec(compile("def outer():\n    inner()\n", str(pkg / "fake_module.py"), "exec"), ns2)
    try:
        ns2["outer"]()
    except ValueError as e:
        where = crashlog._where(e.__traceback__)
    check("outer" in where and "fake_module.py" in where, f"the innermost frame of Kinetrace's own code: {where}")


@section("G142 plain_error")
def _g142():
    import logging
    from kinetrace.errors import plain_error

    class Grab(logging.Handler):
        def __init__(self):
            super().__init__()
            self.n = 0

        def emit(self, record):
            self.n += 1
    g = Grab()
    lg = logging.getLogger("kinetrace.errors")
    lg.addHandler(g)
    try:
        msg = plain_error(pf.ProjectFileError("a.kinetrace is being saved by another Kinetrace right now."),
                          "The save did not finish")
    finally:
        lg.removeHandler(g)
    check(msg == "a.kinetrace is being saved by another Kinetrace right now." and g.n == 0,
          "a ProjectFileError passes through unchanged, no type name, no traceback logged")
    rd = plain_error(PermissionError(13, "denied", "f.csv"), "f.csv could not be read")
    wr = plain_error(PermissionError(13, "denied", "f.csv"), "The save did not finish")
    check("read" in rd and "write" not in rd and "write" in wr, "a permission error on a read is a read problem")
    check("read" in plain_error(PermissionError(13, "x"), "x", reading=True)
          and "write" in plain_error(PermissionError(13, "x"), "x could not be read", reading=False), "... or say so with `reading`")


print()
if FAILS:
    print(f"VERIFY_REVIEW_PERSIST FAILED ({len(FAILS)})")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("VERIFY_REVIEW_PERSIST PASSED")
