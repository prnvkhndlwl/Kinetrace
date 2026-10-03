"""The project file (`kinetrace/projectfile.py`): every field round-trips bit
for bit, nothing the model holds is silently left out, hand-edited and
foreign files open or are refused with a sentence naming the file and row,
paths saved on another OS resolve, and the speed budgets hold. No GPU, no Qt.

Run: .venv\\Scripts\\python.exe tests\\verify_projectfile.py   (.venv/bin/python on Linux / macOS)
"""
import json
import os
import shutil
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from kinetrace import projectfile as pf  # noqa: E402
from kinetrace.body import BodyTrack, rig_of  # noqa: E402
from kinetrace.calib import (Calibration, CameraCalibration, LWMUndistort, NoUndistort,  # noqa: E402
                             OpenCVUndistort, Reconstruction, dlt_from_camera)
from kinetrace.lens import LensProfile  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.segmenter import MaskTrack  # noqa: E402
from kinetrace.session import DEFAULT_UI_STATE, AnimalMeta, TrackingSession  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "projectfile")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


# ----------------------------------------------------------------- a project that uses everything
def camera(name, T, fps, w, h, seed):
    r = np.random.default_rng(seed)
    s = TrackingSession(os.path.join(OUT, "videos", name + ".mp4"), T, fps, w, h)
    s.add_point(0, 10.0, 20.0, name="snout")
    s.add_point(0, 30.0, 40.0, name='wing, left "tip"\nsecond line')
    s.add_point(0, 50.0, 60.0, kind="group", radius=12.5, name="region-circle")
    s.add_point(0, 70.0, 80.0, kind="group", radius=9.0, shape="rect",
                outline=[[60.25, 70.5], [80.0, 70.5], [80.0, 90.125], [60.25, 90.125]])
    s.add_point(0, 90.0, 99.0, kind="group", radius=8.0, shape="polygon",
                outline=[[85.0, 95.0], [95.5, 97.0], [90.0, 104.75]])
    b = s.add_ball(0, 120.0, 130.0, name="ball é 中")
    s.add_ball_prompt(b, 7, 121.5, 131.25, 1)
    s.add_ball_prompt(b, 7, 140.0, 140.0, 0)
    tip = s.add_point(0, 5.0, 5.0, name="tail_tip")
    s.points[tip].source, s.points[tip].spec = "silhouette", "tip"
    s.points[0].anchor = True
    s.points[1].free = True
    s.points[1].spot = {"cue": "bright", "radius": 6.4, "speed_gain": 0.5, "sigma": 1.5}   # spots.json (I161)
    s.points[2].display = False
    N = s.n_points
    s.tracks[:] = r.uniform(0, w, (T, N, 2)).astype(np.float32)
    gaps = r.random((T, N)) < 0.3
    s.tracks[gaps] = np.nan
    s.tracked[:] = ~gaps
    s.visibility[:] = ~gaps & (r.random((T, N)) < 0.8)
    s.confidence[:] = np.where(gaps, 0.0, r.random((T, N))).astype(np.float32)
    s.manual[:] = ~gaps & (r.random((T, N)) < 0.05)
    s.occluded[:] = r.random((T, N)) < 0.03          # includes hidden-but-untracked cells
    s.radius[:] = np.nan
    s.radius[~gaps[:, b], b] = r.uniform(4, 30, (~gaps[:, b]).sum()).astype(np.float32)
    s.tracks[3, 0] = (np.float32(3839.9998), np.float32(1e-7))     # awkward float32 values
    s.tracked[3, 0] = True                                          # (tracked <=> finite, as in the app)
    s.add_event("strike, fast", 5, 9, note='first "strike"\nwith a comma, too')
    s.add_event("strike, fast", 12, 15)
    s.set_note(4, "fin flare, left", author="A. N. Other")
    s.annotator = "tester"
    s.animal = AnimalMeta("lizard", (10, 200, 30))
    s.animal.add_click(0, 100.5, 200.25, True)
    s.animal.add_click(0, 5.0, 6.0, False)
    s.animal.set_box(3, (10, 20, 110, 220))
    s.masks = MaskTrack(T, w, h)
    for f in range(0, T, 2):
        poly = r.integers(0, min(w, h), (40, 2)).astype(np.int32)
        s.masks.set_summary(f, {"area": 1234 + f, "bbox": np.array([1, 2, 300, 400]), "score": 8.5,
                                "centroid": np.array([150.5, 200.25], np.float32),
                                "polys": [poly, poly[:10]], "midline": r.uniform(0, w, (32, 2))})
    s.masks.set_summary(1, {"area": 0, "score": -1.5})               # blank frame with a score
    s.skeleton = {"name": "t", "landmarks": ["snout", "tail_tip"], "bones": [["snout", "tail_tip"]],
                  "head": "snout", "derived": {"tail_tip": "tip"}}
    bt = BodyTrack(T, rig_of("coco17"), 2)
    for f in range(0, T, 3):
        bt.set_person(f, 1, joints2d=r.uniform(0, w, (17, 2)), conf=r.random(17), score=0.9)
    bt.faces = np.array([[0, 1, 2], [2, 3, 0]], np.int32)
    v = r.uniform(-1, 1, (4, 3)).astype(np.float16)
    v.flags.writeable = False
    bt.mesh[(0, 1)] = v
    bt.runs, bt.backend, bt.notes = [(0, T - 1, 3)], "vitpose", "a note"
    s.body = bt
    ui = dict(DEFAULT_UI_STATE, selected=1, zoom=2.5, center_x=333.25, center_y=111.5, user_zoomed=True,
              timeline=[10, 90])
    s.ui_state = ui
    s.current_frame = T // 2
    return s


def make_project():
    T = 120
    ss = [camera("cam1", T, 60.0, 640, 480, 1), camera("cam2", T * 2, 120.0, 800, 600, 2),
          camera("cam 3", T, 60.0, 640, 480, 3)]
    ss[1].file_fps = 30.0          # a slow-motion file: says 30, recorded at 120 (G38)
    p = Project(ss, ["cam1", "cam2", "cam 3"], [0.0, -36.5, 12.25], 1, [1.0, 2.0, 1.0])
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    cams = []
    for i, a in enumerate((0.2, 1.3, 2.4)):
        pos = np.array([3 * np.cos(a), 3 * np.sin(a), 1.0])
        z = -pos / np.linalg.norm(pos)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        und = [NoUndistort(), OpenCVUndistort(K, np.array([-0.1, 0.01, 0.001, -0.002, 0.0])),
               LWMUndistort(rng.uniform(0, 640, (30, 2)), rng.uniform(0, 640, (30, 2)), 12)][i]
        cams.append(CameraCalibration(dlt_from_camera(K, R, -R @ pos), 640, 480, und, pixel_origin=float(i % 2),
                                      rmse=0.25 * i if i else float("nan")))
    p.calibration = Calibration(cams, "m", "unit test")
    p.calibration.origin_shift = np.array([0.1, -0.2, 0.3])
    p.calibration.notes = ["could not read the lens store"]
    p.lenses = [LensProfile(640, 480, K, np.array([-0.2, 0.05, 0.0, 0.0, 0.0]), rms=0.31, n_views=20,
                            source="board.mp4", report={"verdict": "good", "reach": 0.9}), None, None]
    xyz = rng.normal(size=(60, 3, 3))
    xyz[5] = np.nan
    p.reconstruction = Reconstruction(4, ["snout", "tail_tip", "ball é 中"], xyz,
                                      rng.random((60, 3)), rng.integers(0, 4, (60, 3)).astype(np.int32), "m",
                                      rng.random((60, 3, 3)).astype(np.float32))
    return p


STATE = {"tools": {"follow": True, "marker_size": 7, "display_filter": "contrast", "point_backend": "cotracker3",
                   "trail_len": 60}, "layout": {"dock_visible": False, "solo": True, "window": [10, 20, 1500, 900]}}


# ----------------------------------------------------------------- comparison
def same_array(a, b, name):
    if a is None or b is None:
        return check(a is None and b is None, f"{name}: both absent")
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return check(False, f"{name}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
    if a.dtype.kind == "f":
        na, nb = np.isnan(a), np.isnan(b)
        ok = np.array_equal(na, nb) and np.array_equal(a[~na].view(np.uint8 if a.itemsize == 1 else
                                                                   {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize]),
                                                         b[~nb].view({1: np.uint8, 2: np.uint16, 4: np.uint32,
                                                                      8: np.uint64}[b.itemsize]))
    else:
        ok = np.array_equal(a, b)
    return ok or check(False, f"{name}: values differ")


def compare(p, q):
    bad0 = len(FAILS)
    check(p.names == q.names and p.offsets == q.offsets and p.rates == q.rates and p.active == q.active,
          "cameras: names, offsets, rates, active camera")
    for i, (a, b) in enumerate(zip(p.sessions, q.sessions)):
        n = p.names[i]
        for k in ("n_frames", "fps", "file_fps", "width", "height", "current_frame", "annotator", "_name_counter",
                  "_event_counter"):
            if getattr(a, k) != getattr(b, k):
                check(False, f"{n}: {k} {getattr(a, k)!r} vs {getattr(b, k)!r}")
        for k in ("tracks", "confidence", "visibility", "manual", "tracked", "occluded", "radius"):
            same_array(getattr(a, k), getattr(b, k), f"{n}.{k}")
        if [vars(x) for x in a.points] != [vars(x) for x in b.points]:
            check(False, f"{n}: point metadata differs")
        if [vars(x) for x in a.events] != [vars(x) for x in b.events] or a.notes != b.notes:
            check(False, f"{n}: events / notes differ")
        if a.skeleton != b.skeleton:
            check(False, f"{n}: skeleton differs")
        if (a.animal is None) != (b.animal is None) or (a.animal and vars(a.animal) != vars(b.animal)):
            check(False, f"{n}: segment differs")
        if a.masks is not None:
            for k in ("bbox", "area", "centroid", "score"):
                same_array(getattr(a.masks, k), getattr(b.masks, k), f"{n}.masks.{k}")
            if sorted(a.masks.contours) != sorted(b.masks.contours) or any(
                    not all(np.array_equal(x, y) for x, y in zip(a.masks.contours[f], b.masks.contours[f]))
                    for f in a.masks.contours) or sorted(a.masks.midline) != sorted(b.masks.midline):
                check(False, f"{n}: silhouette outlines / midline differ")
        if a.body is not None:
            for k in ("joints3d", "joints2d", "conf", "score", "bbox", "focal", "cam_t", "faces"):
                same_array(getattr(a.body, k), getattr(b.body, k), f"{n}.body.{k}")
            meta_a = (a.body.runs, a.body.names, a.body.backend, a.body.rig.name, a.body.n_people, a.body.has_3d,
                      a.body.notes, a.body.n_requested, a.body.step)
            meta_b = (b.body.runs, b.body.names, b.body.backend, b.body.rig.name, b.body.n_people, b.body.has_3d,
                      b.body.notes, b.body.n_requested, b.body.step)
            if sorted(a.body.mesh) != sorted(b.body.mesh) or meta_a != meta_b or any(
                    not np.array_equal(a.body.mesh[k], b.body.mesh[k]) for k in a.body.mesh):
                check(False, f"{n}: body mesh / metadata differ")
        ua = {k: v for k, v in a.ui_state.items()}
        ub = {k: v for k, v in b.ui_state.items()}
        if ua != ub:
            diff = {k: (ua.get(k), ub.get(k)) for k in set(ua) | set(ub) if ua.get(k) != ub.get(k)}
            check(False, f"{n}: ui state differs {diff}")
    ca, cb = p.calibration, q.calibration
    if (ca is None) != (cb is None):
        check(False, "calibration present on one side only")
    elif ca is not None:
        for c1, c2 in zip(ca.cameras, cb.cameras):
            same_array(c1.coefs, c2.coefs, "calibration coefs")
            if (c1.width, c1.height, c1.pixel_origin, c1.y_flip) != (c2.width, c2.height, c2.pixel_origin, c2.y_flip) \
                    or c1.undistort.to_json() != c2.undistort.to_json() or \
                    not (c1.rmse == c2.rmse or (np.isnan(c1.rmse) and np.isnan(c2.rmse))):
                check(False, "calibration camera fields differ")
        check(ca.unit == cb.unit and ca.source == cb.source and np.array_equal(ca.origin_shift, cb.origin_shift)
              and list(ca.notes) == list(cb.notes), "calibration unit / source / origin shift / notes")
    if [None if l is None else l.to_json() for l in p.lenses] != [None if l is None else l.to_json() for l in q.lenses]:
        check(False, "lens profiles differ")
    ra, rb = p.reconstruction, q.reconstruction
    if ra is not None:
        for k in ("xyz", "residual", "n_cams", "per_cam"):
            same_array(getattr(ra, k), getattr(rb, k), f"reconstruction.{k}")
        check(ra.t0 == rb.t0 and ra.names == rb.names and ra.unit == rb.unit, "reconstruction t0 / names / unit")
    return len(FAILS) == bad0


# ----------------------------------------------------------------- 1. round trip, every encoding
print("\n[1] round trip")
p = make_project()
p.exports = ["dltdv_all", "dlc"]
path = os.path.join(OUT, "kitchen.kinetrace")
p.path = path
pid = pf.new_id()
pf.write_folder(pf.freeze(p, STATE, pid, target=path), path)
check(os.path.isdir(path) and os.path.isfile(os.path.join(path, "kinetrace.json"))
      and os.path.isfile(os.path.join(path, "cameras", "cam1", "tracks", "snout.csv"))
      and os.path.isfile(os.path.join(path, "README.txt")), "a project is a folder with one CSV per landmark")
q, state, meta = pf.read(path)
check(meta["project_id"] == pid and meta["format_version"] == pf.FORMAT_VERSION, "kinetrace.json id / version")
check(state == STATE, "state.json comes back as written")
tools = {k: v for k, v in STATE["tools"].items()}
for s in p.sessions:                                 # the window-wide toggles land in every camera
    s.ui_state.update(tools)
check(compare(p, q), "every field of a 3-camera project round-trips bit for bit (folder, tables from the cache)")
check(q.exports == ["dltdv_all", "dlc"], "the exports kept up to date on save are remembered (G42)")
shutil.rmtree(os.path.join(path, ".cache"))
q1, _, _ = pf.read(path)
check(compare(p, q1), "... and bit for bit from the CSV text alone (.cache deleted)")
zpath = os.path.join(OUT, "kitchen-one.kinetrace")
pf.write(pf.freeze(p, STATE, pid, target=zpath, layout="zip"), zpath)
q3, _, meta3 = pf.read(zpath)
check(compare(p, q3) and meta3["videos_relative_to"] == "container", "the single-file form (zip) round-trips too")
rpath = os.path.join(OUT, "kitchen.recovery.kinetrace")
pf.write(pf.freeze(p, STATE, pid, binary_tracks=True), rpath, compresslevel=0, fsync=False, backup=False)
q2, _, _ = pf.read(rpath)
check(compare(p, q2), "the binary encoding (recovery) round-trips too")

# ----------------------------------------------------------------- 2. nothing silently left out
print("\n[2] field coverage")
TRANSIENT = {
    "TrackingSession": {"_dirty", "data_version", "last_interp_hidden_skipped", "video_path"},
    "Project": {"path", "dirty", "edits"},
    "MaskTrack": {"native_w", "native_h"},
    "BodyTrack": {"mesh_version", "_version", "_cache", "examined", "stopped_early"},
}
PERSISTED = {
    "TrackingSession": {"n_frames", "fps", "file_fps", "width", "height", "tracks", "visibility", "manual", "tracked",
                        "confidence", "occluded", "radius", "points", "events", "notes", "annotator", "animal",
                        "masks", "body", "skeleton", "current_frame", "ui_state", "_name_counter", "_event_counter"},
    "Project": {"sessions", "names", "offsets", "rates", "active", "calibration", "reconstruction", "lenses",
                "exports"},
    "MaskTrack": {"n_frames", "bbox", "area", "centroid", "score", "contours", "midline"},
    "BodyTrack": {"joints3d", "joints2d", "conf", "score", "bbox", "focal", "cam_t", "names", "backend", "runs",
                  "faces", "mesh", "rig", "n_frames", "n_people", "has_3d", "notes", "n_requested", "step", "box_src"},
}
for cls, obj in (("TrackingSession", p.sessions[0]), ("Project", p), ("MaskTrack", p.sessions[0].masks),
                 ("BodyTrack", p.sessions[0].body)):
    unknown = set(vars(obj)) - PERSISTED[cls] - TRANSIENT.get(cls, set())
    check(not unknown, f"{cls}: every attribute is saved or listed as transient {sorted(unknown) or ''}")

# ----------------------------------------------------------------- 3. sparse rows, determinism
print("\n[3] sparse table, deterministic bytes")
big = TrackingSession(os.path.join(OUT, "big.mp4"), 40000, 240.0, 3840, 2160)
for j in range(10):
    big.add_point(0, 1.0, 1.0)
big.tracked[:] = False
big.manual[:] = False                                   # placing a point marks it hand-placed
big.tracks[:] = np.nan
big.confidence[:] = 0
big.visibility[:] = False
for f in (0, 10, 99, 20000, 39999):
    big.set_position(f, 3, 12.5, 7.25)
bp = Project([big])
sp = os.path.join(OUT, "sparse.kinetrace")
pf.write_folder(pf.freeze(bp, {}, "x" * 32, target=sp, saved_at="2026-01-01T00:00:00"), sp)
lm = pf.landmark_files([x.name for x in big.points])[3]
rows = open(os.path.join(sp, "cameras", "cam1", "tracks", lm), encoding="utf-8").read().strip().splitlines()
check(len(rows) == 1 + 5, f"40,000 frames with 5 tracked cells -> exactly 5 data rows ({len(rows) - 1})")
empty = open(os.path.join(sp, "cameras", "cam1", "tracks", pf.landmark_files([x.name for x in big.points])[0]),
             encoding="utf-8").read()
check(empty == ",".join(pf.LANDMARK_COLS[:-1]) + "\n", "a landmark with no data: the header alone, no radius column")
sp2 = os.path.join(OUT, "sparse2.kinetrace")
pf.write_folder(pf.freeze(bp, {}, "x" * 32, target=sp2, saved_at="2026-01-01T00:00:00"), sp2)


def tree(d, skip=(".cache", ".history", "exports")):
    out = {}
    for dirpath, dirs, fnames in os.walk(d):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in fnames:
            fp = os.path.join(dirpath, f)
            out[os.path.relpath(fp, d).replace("\\", "/")] = open(fp, "rb").read()
    return out


check(tree(sp) == tree(sp2), "the same project always gives the same bytes (every file of the folder)")
zp1, zp2 = os.path.join(OUT, "sparse-1.kinetrace"), os.path.join(OUT, "sparse-2.kinetrace")
for zp in (zp1, zp2):
    pf.write(pf.freeze(bp, {}, "x" * 32, target=zp, saved_at="2026-01-01T00:00:00", layout="zip"), zp)
check(open(zp1, "rb").read() == open(zp2, "rb").read(), "... and the single file too")
with zipfile.ZipFile(zp1) as z:
    check(sorted(z.namelist()) == sorted(tree(sp)), "the single file holds exactly the folder's files")

# ----------------------------------------------------------------- 3b. a save writes only what changed
print("\n[3b] incremental saves")
ip = os.path.join(OUT, "incremental.kinetrace")
inc = make_project()
st = pf.save(inc, ip, STATE, "i" * 32)
check(st["written"] == len(tree(ip)) and st["unchanged"] == 0 and st["removed"] == 0,
      f"the first save writes every file of the folder ({st})")
before = tree(ip)
st = pf.save(inc, ip, STATE, "i" * 32)
after = tree(ip)
check(st["written"] == 1 and [k for k in after if after[k] != before.get(k)] == ["kinetrace.json"],
      f"a save with nothing changed rewrites kinetrace.json alone ({st})")
inc.sessions[0].tracks[7, 0] = (1.25, 2.5)
inc.sessions[0].tracked[7, 0] = True
got = pf.changes_since_save(pf.freeze(inc, STATE, "i" * 32, target=ip), ip)
words = pf.describe_changes(*got, [("cam1", "cam1"), ("cam2", "cam2"), ("cam_3", "cam 3")],
                            {"cam1": {"snout.csv": "snout"}})
check(got == (["cameras/cam1/tracks/snout.csv"], []) and words == ["cam1: snout (positions)"],
      f"before saving, what changed is named: {words}")
check(pf.describe_changes(["project.json", "cameras/cam2/events.csv", "reconstruction/points/a.csv"],
                          ["cameras/cam2/tracks/old.csv"], [("cam2", "side")], {})
      == ["side: the events, old (removed)", "the cameras (videos, offsets, frame rates)", "the 3D result"],
      "... in words for every kind of file")
st = pf.save(inc, ip, STATE, "i" * 32)
after2 = tree(ip)
changed = sorted(k for k in after2 if after2[k] != after.get(k))
check(changed == ["cameras/cam1/tracks/snout.csv", "kinetrace.json"], f"one point moved: its file alone ({changed})")
check(sorted(os.listdir(os.path.join(ip, ".history", "cameras", "cam1", "tracks"))) == ["snout.csv"],
      "the file it replaced is kept in .history")
own = [os.path.join(ip, "notes-by-me.txt"), os.path.join(ip, "videos", "keep.txt"),
       os.path.join(ip, "cameras", "cam1", "my own file.txt")]
for f in own:
    os.makedirs(os.path.dirname(f), exist_ok=True)
    open(f, "w").write("mine")
inc.rename_landmark("snout", "nose")
inc.remove_view(2)
st = pf.save(inc, ip, STATE, "i" * 32)
t3 = tree(ip)
check("cameras/cam1/tracks/nose.csv" in t3 and "cameras/cam1/tracks/snout.csv" not in t3
      and not any(k.startswith("cameras/cam_3/") and k.endswith(".csv") for k in t3),
      f"a renamed landmark moves to its new file, a removed camera's files go ({st})")
check(all(os.path.isfile(f) and open(f).read() == "mine" for f in own),
      "files that are not the project's own (videos/, notes, a user file in a camera folder) are never touched")
q4, _, _ = pf.read(ip)
check(q4.names == inc.names and q4.sessions[0].points[0].name == "nose", "... and the folder reads back as saved")

# ----------------------------------------------------------------- 3c. the cache never wins over the CSV
print("\n[3c] cache and hand edits")
cp = os.path.join(OUT, "cache.kinetrace")
cproj = Project([camera("solo", 50, 30.0, 640, 480, 11)])
pf.save(cproj, cp, STATE, "c" * 32)
rel = "cameras/cam1/tracks/snout.csv"
cache_f = os.path.join(cp, ".cache", *rel.split("/")) + ".npy"
arr = np.load(cache_f)
r0 = arr["frame"][0]
x0 = arr["x"][0]
parsed = []
real_parse = pf._parse_landmark
pf._parse_landmark = lambda text, where: (parsed.append(where), real_parse(text, where))[1]
qa, _, _ = pf.read(cp)
check(not parsed and qa.sessions[0].tracks[r0, 0, 0] == x0, "an unchanged CSV is read from .cache (no CSV parsed)")
arr["x"][0] = 777.0                                   # a cache copy damaged (a power cut: caches are not fsynced)
np.save(cache_f, arr)
qa, _, _ = pf.read(cp)
check(parsed == [rel] and qa.sessions[0].tracks[r0, 0, 0] == x0,
      f"a damaged cache copy is caught by its fingerprint: the CSV is read instead {parsed}")
pf._parse_landmark = real_parse
csvf = os.path.join(cp, *rel.split("/"))
lines = open(csvf, encoding="utf-8").read().splitlines()
cells = lines[1].split(",")
cells[1] = "123.5"
lines[1] = ",".join(cells)
time.sleep(0.05)
open(csvf, "w", encoding="utf-8", newline="").write("\r\n".join(lines) + "\r\n")    # edited in a spreadsheet
qb, _, _ = pf.read(cp)
check(qb.sessions[0].tracks[r0, 0, 0] == np.float32(123.5), "a CSV edited by hand is read from its text")
check(pf.read(cp)[0].sessions[0].tracks[r0, 0, 0] == np.float32(123.5), "... every time it is opened")

# ----------------------------------------------------------------- 3d. a save cut short
print("\n[3d] crash safety")
kp = os.path.join(OUT, "crash.kinetrace")
kproj = Project([camera("solo", 50, 30.0, 640, 480, 12)])
pf.save(kproj, kp, STATE, "k" * 32)
saved = tree(kp)
kproj.sessions[0].tracks[:, 0] += 1.0
kproj.sessions[0].tracks[:, 1] += 2.0
real_replace = pf._replace
calls = {"n": 0}


def failing(src, dst):
    calls["n"] += 1
    if calls["n"] == 3:
        raise PermissionError("the file is open in another program")
    return real_replace(src, dst)


pf._replace = failing
try:
    pf.save(kproj, kp, STATE, "k" * 32)
    check(False, "the failing save should raise")
except PermissionError:
    pass
finally:
    pf._replace = real_replace
check(tree(kp) == saved and not os.path.exists(os.path.join(kp, ".saving")),
      "a save that fails half-way is undone: every file as it was, nothing left aside")
import subprocess  # noqa: E402
CRASH = r'''
import os, sys
sys.path.insert(0, {root!r})
from kinetrace import projectfile as pf
from kinetrace.project import Project
p = pf.load({path!r})
p.sessions[0].tracks[:, 0] += 5.0
real, n = pf._replace, [0]
def crash(src, dst):
    d = str(dst)
    if os.path.basename(d) == "kinetrace.json" and {after!r}:
        real(src, dst); os._exit(3)          # the power goes right AFTER the save counted
    if ".cache" not in d and os.path.basename(d) != "pending.json":
        n[0] += 1                            # a move of the project's own files
        if n[0] == 2 and not {after!r}:
            os._exit(3)                      # ... or in the middle, before it counted
    real(src, dst)
pf._replace = crash
pf.save(p, {path!r}, project_id="k" * 32)
'''
for after in (False, True):
    before_crash = tree(kp)
    base = pf.load(kp).sessions[0].tracks[:, 0].copy()
    rc = subprocess.run([sys.executable, "-c", CRASH.format(root=ROOT, path=kp, after=after)]).returncode
    qc, _, mc = pf.read(kp)
    if not after:
        check(rc == 3 and mc["_interrupted"] == "undone" and tree(kp) == before_crash,
              "a crash in the middle of a save: opening puts every file back as it was at the previous save")
    else:
        same = np.array_equal(qc.sessions[0].tracks[:, 0], base + np.float32(5.0), equal_nan=True)
        check(rc == 3 and mc["_interrupted"] == "finished" and same and not os.path.exists(
            os.path.join(kp, ".history", "pending.json")), "a crash right after the save counted: the new save stands")

# ----------------------------------------------------------------- 3f. a big save's CSVs in helper processes
print("\n[3f] helper processes format the CSVs of a big save")
from kinetrace import csvpool  # noqa: E402
hp_proj = make_project()
fixed = "2026-01-01T00:00:00.000001+00:00"


def save_to(name):
    d = os.path.join(OUT, name)
    shutil.rmtree(d, ignore_errors=True)
    pf.write_folder(pf.freeze(hp_proj, STATE, "h" * 32, target=d, saved_at=fixed), d)
    return tree(d)


env_before, min_before, start_before = os.environ.get("KINETRACE_SAVE_WORKERS"), csvpool.MIN_ROWS, csvpool._start
spy = {}
real_fmt = csvpool.format_tables


def spying(jobs, fsync=True, workers=None):
    r = real_fmt(jobs, fsync, workers)
    spy.update(jobs=len(jobs), done=len(r))
    return r


try:
    os.environ["KINETRACE_SAVE_WORKERS"] = "0"
    serial = save_to("helpers-none.kinetrace")
    os.environ["KINETRACE_SAVE_WORKERS"] = "3"
    csvpool.MIN_ROWS = 0                                   # use them even for this small project
    csvpool.format_tables = spying
    helped = save_to("helpers-three.kinetrace")
    check(spy.get("jobs", 0) > 20 and spy.get("done") == spy.get("jobs") and helped == serial,
          f"3 helpers wrote every table ({spy}) and the folder is byte for byte the serial one")
    csvpool.format_tables = real_fmt

    def no_start():
        raise OSError("antivirus says no")
    csvpool._start = no_start
    check(save_to("helpers-refused.kinetrace") == serial, "helpers that cannot start: the save formats itself")

    def dying():
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(8); sys.exit(1)"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    csvpool._start = dying
    check(save_to("helpers-dying.kinetrace") == serial, "helpers that die mid-job: the save formats itself")
finally:
    csvpool.format_tables, csvpool.MIN_ROWS, csvpool._start = real_fmt, min_before, start_before
    if env_before is None:
        os.environ.pop("KINETRACE_SAVE_WORKERS", None)
    else:
        os.environ["KINETRACE_SAVE_WORKERS"] = env_before

# ----------------------------------------------------------------- 3g. one save at a time, room, durability
print("\n[3g] the save lock, free space, a hung helper")
import socket  # noqa: E402
from pathlib import Path  # noqa: E402
lp = os.path.join(OUT, "lock.kinetrace")
lproj = Project([camera("solo", 50, 30.0, 640, 480, 14)])
pf.save(lproj, lp, STATE, "l" * 32)
lproj.sessions[0].tracks[:, 0] += 1.0
open(os.path.join(lp, ".lock"), "w").write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                                       "time": time.time()}))
lock_before = tree(lp)
try:
    pf.save(lproj, lp, STATE, "l" * 32)
    check(False, "a second save while one runs should be refused")
except pf.ProjectFileError as e:
    check("being saved by another Kinetrace" in str(e) and tree(lp) == lock_before,
          f"another save of the same project in progress: refused, nothing changed ('{str(e)[:60]}...')")
open(os.path.join(lp, ".lock"), "w").write(json.dumps({"pid": 999999, "host": socket.gethostname(), "time": time.time()}))
st_l = pf.save(lproj, lp, STATE, "l" * 32)
check(st_l["written"] >= 2 and not os.path.exists(os.path.join(lp, ".lock")),
      "a lock left by a process that is gone is taken over, and released after the save")
real_free = pf._free_bytes
pf._free_bytes = lambda root: 10 << 20
lproj.sessions[0].tracks[:, 0] += 1.0
lock_before = tree(lp)
try:
    pf.save(lproj, lp, STATE, "l" * 32)
    check(False, "a save without room should be refused")
except pf.ProjectFileError as e:
    check("not enough free space" in str(e) and tree(lp) == lock_before and not os.path.exists(os.path.join(lp, ".saving")),
          f"not enough room on the drive: said before anything is replaced ('{str(e)[:70]}...')")
finally:
    pf._free_bytes = real_free
csvpool._start = lambda: subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
try:
    t_h = time.perf_counter()
    hung = csvpool.format_tables([(pf._landmark_table(dict(tracks=np.zeros((10, 1, 2), np.float32),
                                                           confidence=np.ones((10, 1), np.float32),
                                                           visibility=np.ones((10, 1), bool),
                                                           manual=np.zeros((10, 1), bool), tracked=np.ones((10, 1), bool),
                                                           occluded=np.zeros((10, 1), bool),
                                                           radius=np.full((10, 1), np.nan, np.float32)), 0, False),
                                   Path(OUT) / "hung.csv")], workers=2, timeout=2.0)
    t_h = time.perf_counter() - t_h
finally:
    csvpool._start = start_before
check(hung == set() and t_h < 15, f"a helper that hangs is killed at its deadline ({t_h:.1f} s); the job is the saver's")
pf._fsync_dir(Path(OUT))                               # POSIX: renames made durable (a no-op on Windows)
check(True, "directory fsync runs (POSIX) or is skipped (Windows)")

# ----------------------------------------------------------------- 3e. back one save
print("\n[3e] the previous save")
bp2 = os.path.join(OUT, "previous.kinetrace")
bproj = Project([camera("solo", 50, 30.0, 640, 480, 13)])
pf.save(bproj, bp2, STATE, "b" * 32)
first_save = tree(bp2)
bproj.sessions[0].tracks[:, 0] = 1.0
bproj.sessions[0].add_point(0, 5.0, 5.0, name="late")
pf.save(bproj, bp2, STATE, "b" * 32)
pf.restore_previous(bp2)
check(tree(bp2) == first_save, "restore_previous puts the folder back at the save before the last one")
try:
    pf.restore_previous(bp2)
    check(False, "a second restore should be refused")
except pf.ProjectFileError as e:
    check("no earlier save" in str(e), f"... once: a second one is refused ('{e}')")

# ----------------------------------------------------------------- 3h. a project someone else made
print("\n[3h] a project from someone else cannot touch files outside it (I147-I149, I157)")
from kinetrace import autoexport, recovery  # noqa: E402

frozen_k = pf.freeze(p, STATE, pid, target=path)
bad_names = [r for r in frozen_k.files if not pf._layout_rel(r)]
check(not bad_names, f"every file a save writes is one a rollback accepts ({bad_names[:3]})")
refused = [r for r in ("../x.txt", "/etc/x", "C:/Users/x/thesis.docx", "cameras\\cam1\\points.csv", "",
                       "cameras/../../x", ".history/x", "cameras/cam1/../../x", "cameras/cam1/other.txt",
                       "exports/a.csv", "videos/cam1.mp4", "notes.txt", "cameras/cam1/tracks") if pf._layout_rel(r)]
check(not refused, f"paths outside the layout are refused ({refused})")
hostile = os.path.join(OUT, "hostile.kinetrace")
shutil.rmtree(hostile, ignore_errors=True)
shutil.copytree(path, hostile)
victim = os.path.join(OUT, "victim.txt")
open(victim, "w").write("the user's own file")
os.makedirs(os.path.join(hostile, ".history"), exist_ok=True)
open(os.path.join(hostile, ".history", "victim.txt"), "w").write("shipped by the attacker")
for record in ({"saved_at": "not this save", "added": [os.path.abspath(victim)]},
               {"saved_at": "not this save", "added": ["../victim.txt"]},
               {"saved_at": "not this save", "replaced": ["../victim.txt"]},
               {"saved_at": "not this save", "removed": "../victim.txt"}):
    json.dump(record, open(os.path.join(hostile, ".history", "pending.json"), "w"))
    before = tree(hostile)
    try:
        pf.read(hostile)
        check(False, f"a pending.json naming {record} should refuse the open")
    except pf.ProjectFileError as e:
        check(open(victim).read() == "the user's own file" and tree(hostile) == before and "Nothing was changed" in str(e),
              f"open refused, nothing outside or inside touched: {list(record)[1]} {str(list(record.values())[1])[-24:]!r}")
os.remove(os.path.join(hostile, ".history", "pending.json"))
open(os.path.join(hostile, ".history", "kinetrace.json"), "w").write("{}")
json.dump({"added": [os.path.abspath(victim)], "replaced": ["../victim.txt"]},
          open(os.path.join(hostile, ".history", "previous.json"), "w"))
try:
    pf.restore_previous(hostile)
    check(False, "a previous.json naming files outside should be refused")
except pf.ProjectFileError:
    check(open(victim).read() == "the user's own file", "restore_previous (convert previous) refuses it too")
shutil.rmtree(os.path.join(hostile, ".history"))
meta_p = os.path.join(hostile, "kinetrace.json")
km = json.load(open(meta_p, encoding="utf-8"))
km["project_id"] = "../../victim"
json.dump(km, open(meta_p, "w", encoding="utf-8"))
_, _, mh = pf.read(hostile)
check(pf.safe_id(mh["project_id"]) and mh["project_id"] != "../../victim",
      f"a project id that is a path is replaced by a new one ({mh['project_id'][:8]}...) (I148)")
try:
    recovery.paths("../../victim")
    check(False, "recovery.paths must refuse a path")
except ValueError:
    check(recovery.find("../victim") is None, "recovery.paths refuses an id with a path in it; find() ignores it")
open(os.path.join(hostile, pf.LOCK), "w").write('{"pid": "x", "time": [1], "host": 5}')
q_lock, _, _ = pf.read(hostile)
pf.write_folder(pf.freeze(q_lock, STATE, pid, target=hostile), hostile)
check(not os.path.exists(os.path.join(hostile, pf.LOCK)), "a lock Kinetrace did not write neither stops the open nor the save (I157)")
pj_p = os.path.join(hostile, "project.json")
pj_saved = open(pj_p, "rb").read()
pjh = json.loads(pj_saved)
pjh["cameras"][0]["n_frames"] = 10 ** 12
json.dump(pjh, open(pj_p, "w"))
try:
    pf.read(hostile)
    check(False, "a trillion-frame camera should be refused")
except pf.ProjectFileError as e:
    check("can hold" in str(e), f"a camera of 10^12 frames is refused, not allocated ('{str(e)[:60]}...')")
open(pj_p, "wb").write(pj_saved)
# exports/: exports.json names what a refresh may delete (I149)
ex = os.path.join(hostile, "exports")
os.makedirs(ex, exist_ok=True)
ours = os.path.join(ex, "cam1_DLC.csv")
open(ours, "w").write("made by an earlier refresh")
json.dump({"files": {os.path.abspath(victim): "x", "../../victim.txt": "x", "../victim.txt": "x",
                     "cam1_DLC.csv": "x", "notes.txt": "x"}}, open(os.path.join(ex, "exports.json"), "w"))
open(os.path.join(ex, "notes.txt"), "w").write("the user's")
written, problems = autoexport.refresh(hostile, ["wide"])
check(open(victim).read() == "the user's own file" and os.path.isfile(os.path.join(ex, "notes.txt"))
      and not os.path.exists(ours) and written and not problems,
      "a refresh removes only files it made (cam1_DLC.csv unticked), never a path exports.json names")
km = json.load(open(meta_p, encoding="utf-8"))
km["cameras"][0]["folder"] = "../../escape"
json.dump(km, open(meta_p, "w", encoding="utf-8"))
written, problems = autoexport.refresh(hostile, ["wide", "dlc"])
check(not written and problems and not os.path.exists(os.path.join(OUT, "escape_tracks.csv")),
      "a camera folder that is a path makes the refresh write nothing")

# ----------------------------------------------------------------- 4. a project written by another program
print("\n[4] hand-written and edited files")
hand = os.path.join(OUT, "hand")
os.makedirs(os.path.join(hand, "cameras", "top"), exist_ok=True)
json.dump({"format": "kinetrace-project", "format_version": 1}, open(os.path.join(hand, "kinetrace.json"), "w"))
json.dump({"cameras": [{"name": "top", "folder": "top", "video": {"path": "top.mp4"}, "n_frames": 100,
                        "fps": 30, "width": 640, "height": 480}]}, open(os.path.join(hand, "project.json"), "w"))
with open(os.path.join(hand, "cameras", "top", "tracks.csv"), "w", encoding="utf-8", newline="") as fh:
    fh.write("\ufeffframe,point,x,y,visible\r\n0,snout,10.5,20.25,TRUE\r\n5,snout,11,21,true\r\n5,tail,1,2,0\r\n")
hp, _, _ = pf.read(hand)
s = hp.sessions[0]
check([x.name for x in s.points] == ["snout", "tail"] and s.tracked.sum() == 3
      and s.tracks[5, 0].tolist() == [11.0, 21.0] and s.visibility[0, 0] and not s.visibility[5, 1],
      "3 files (BOM, CRLF, TRUE / true) open: points come from the names in tracks.csv")


def refused(write_fn, expect, what):
    d = os.path.join(OUT, "bad")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(hand, d)
    write_fn(d)
    try:
        pf.read(d)
        check(False, f"{what}: should have been refused")
    except pf.ProjectFileError as e:
        check(expect in str(e), f"{what}: refused with '{e}'")


tp = os.path.join("cameras", "top", "tracks.csv")
refused(lambda d: open(os.path.join(d, tp), "w").write("frame;point;x;y\n0;a;1,5;2\n"), "';'",
        "a semicolon / decimal-comma file")
refused(lambda d: open(os.path.join(d, tp), "w").write("frame,point,x,y\n0,a,1,2\n0,a,3,4\n"), "twice",
        "the same frame and point twice")
refused(lambda d: open(os.path.join(d, tp), "w").write("frame,point,x,y\n0,a,1,2\n100,a,3,4\n"), "row 3",
        "a frame beyond the video")
refused(lambda d: open(os.path.join(d, tp), "w").write("frame,point,x,y\n0,a,one,2\n"), "not a number",
        "text where a number belongs")
refused(lambda d: os.remove(os.path.join(d, "kinetrace.json")), "no kinetrace.json", "no kinetrace.json")
refused(lambda d: json.dump({"format": "kinetrace-project", "format_version": pf.FORMAT_VERSION + 1},
                            open(os.path.join(d, "kinetrace.json"), "w")), "newer", "a newer format version")
# a format-2 folder written by hand: the minimum is kinetrace.json + project.json + one landmark file
hand2 = os.path.join(OUT, "hand2.kinetrace")
os.makedirs(os.path.join(hand2, "cameras", "top", "tracks"), exist_ok=True)
json.dump({"format": "kinetrace-project", "format_version": 2}, open(os.path.join(hand2, "kinetrace.json"), "w"))
json.dump({"cameras": [{"name": "top", "folder": "top", "video": {"relative_path": "../top.mp4"}, "n_frames": 100,
                        "fps": 30, "width": 640, "height": 480}]}, open(os.path.join(hand2, "project.json"), "w"))
open(os.path.join(hand2, "cameras", "top", "tracks", "snout.csv"), "w", encoding="utf-8", newline="").write(
    "﻿frame,x,y\r\n0,10.5,20.25\r\n5,11,21\r\n")
open(os.path.join(hand2, "cameras", "top", "tracks", "tail tip.csv"), "w", newline="").write(
    "frame,x,y,hidden\n7,1,2,1\n")
h2 = pf.load(os.path.join(hand2, "kinetrace.json"))           # opened through its kinetrace.json
s2 = h2.sessions[0]
check([x.name for x in s2.points] == ["snout", "tail tip"] and s2.tracked.sum() == 3 and s2.occluded[7, 1]
      and s2.confidence[0, 0] == 1.0 and s2.visibility[5, 0], "a hand-written format-2 folder: a landmark per "
      "file (named by the file), missing columns take their defaults")
bad2 = os.path.join(hand2, "cameras", "top", "tracks", "snout.csv")
open(bad2, "w").write("frame,x,y\n0,1,2\n100,3,4\n")
try:
    pf.read(hand2)
    check(False, "a frame beyond the video in a landmark file should be refused")
except pf.ProjectFileError as e:
    check("snout.csv" in str(e) and "row 3" in str(e), f"a landmark file's bad row is named: '{e}'")
open(os.path.join(OUT, "garbage.kinetrace"), "wb").write(b"PK\x03\x04 not really a zip")
try:
    pf.read(os.path.join(OUT, "garbage.kinetrace"))
    check(False, "a damaged file should be refused")
except pf.ProjectFileError as e:
    check("damaged" in str(e) or "not a Kinetrace" in str(e), f"a damaged file: '{e}'")

# ----------------------------------------------------------------- 5. paths saved on another OS
print("\n[5] cross-platform paths")
vdir = os.path.join(OUT, "moved")
os.makedirs(os.path.join(vdir, "videos"), exist_ok=True)
open(os.path.join(vdir, "videos", "cam1.mp4"), "wb").close()
open(os.path.join(vdir, "e\u0301tude.mp4"), "wb").close()                 # decomposed (macOS style)
from pathlib import Path  # noqa: E402
got = pf.locate_video({"video": {"path": r"C:\Users\someone\data\videos\cam1.mp4",
                                 "relative_path": "videos/cam1.mp4"}}, Path(vdir))
check(got is not None and got.name == "cam1.mp4", "a Windows path + relative path resolves anywhere")
got = pf.locate_video({"video": {"path": r"D:\lab\cam1.mp4"}}, Path(vdir), [Path(vdir) / "videos"])
check(got is not None and got.name == "cam1.mp4", "only a foreign absolute path: found by its file name")
got = pf.locate_video({"video": {"path": "/Volumes/lab/\u00e9tude.mp4"}}, Path(vdir))
check(got is not None, "a composed name finds the decomposed file (macOS)")
here = Path(OUT).resolve()
check(pf.video_base(path, pf.read_meta(path)) == Path(path).resolve()
      and pf.video_base(zpath, pf.read_meta(zpath)) == here
      and pf.video_base(hand, {"format_version": 1}) == here,
      "relative video paths start at the project folder (format 2), beside a single file, and beside an "
      "unzipped format-1 folder (they were written relative to the zip's folder)")
check(json.load(open(os.path.join(path, "project.json")))["cameras"][0]["video"]["relative_path"]
      == "../videos/cam1.mp4", "a folder project stores its videos relative to itself")

# ----------------------------------------------------------------- 6. speed budgets
print("\n[6] speed (40,000 frames x 10 points, silhouette on every frame)")
T, N = 40000, 10
s = TrackingSession(os.path.join(OUT, "perf.mp4"), T, 240.0, 3840, 2160)
for j in range(N):
    s.add_point(0, 1.0, 1.0)
s.tracks[:] = rng.uniform(0, 3840, (T, N, 2)).astype(np.float32)
s.tracked[:] = True
s.visibility[:] = True
s.confidence[:] = rng.random((T, N)).astype(np.float32)
s.animal = AnimalMeta()
s.masks = MaskTrack(T, 3840, 2160)
poly = rng.integers(0, 3840, (100, 2)).astype(np.int32)
mid = rng.uniform(0, 3840, (32, 2)).astype(np.float32)
for f in range(T):
    s.masks.contours[f] = [poly]
    s.masks.midline[f] = mid
s.masks.area[:] = 5000
perf = Project([s])
t = time.perf_counter(); fz = pf.freeze(perf, {}, "p" * 32, binary_tracks=True); t_fz = time.perf_counter() - t
rp = os.path.join(OUT, "perf.recovery.kinetrace")
t = time.perf_counter(); pf.write(fz, rp, compresslevel=0, fsync=False, backup=False); t_rw = time.perf_counter() - t
t = time.perf_counter(); pf.read(rp); t_rr = time.perf_counter() - t
pp = os.path.join(OUT, "perf.kinetrace")
t = time.perf_counter(); pf.save(perf, pp, {}, "p" * 32); t_sw = time.perf_counter() - t
t = time.perf_counter(); pf.read(pp); t_sr = time.perf_counter() - t
t = time.perf_counter(); pf.save(perf, pp, {}, "p" * 32); t_s0 = time.perf_counter() - t
s.tracks[5, 3] = (7.5, 8.5)
t = time.perf_counter(); pf.changes_since_save(pf.freeze(perf, {}, "p" * 32, target=pp), pp); t_cq = time.perf_counter() - t
check(t_cq <= 0.5 * float(os.environ.get("KINETRACE_PERF_SCALE", "1")),
      f"what changed since the save, for the close question: {t_cq:.2f} s")
t = time.perf_counter(); pf.save(perf, pp, {}, "p" * 32); t_s1 = time.perf_counter() - t
shutil.rmtree(os.path.join(pp, ".cache"))
t = time.perf_counter(); pf.read(pp); t_sc = time.perf_counter() - t
mb = sum(os.path.getsize(os.path.join(d_, f_)) for d_, _, fs in os.walk(pp) for f_ in fs if ".history" not in d_) / 1e6
print(f"  freeze {t_fz * 1000:.0f} ms | recovery write {t_rw * 1000:.0f} ms, read {t_rr * 1000:.0f} ms | "
      f"project first save {t_sw:.2f} s, open {t_sr:.2f} s (from CSV alone {t_sc:.2f} s), save unchanged "
      f"{t_s0:.2f} s, after one edit {t_s1:.2f} s, {mb:.1f} MB")
SLOW = float(os.environ.get("KINETRACE_PERF_SCALE", "1"))
check(t_fz <= 0.08 * SLOW, "freeze (GUI thread) within budget")
check(t_rw <= 0.2 * SLOW and t_rr <= 0.1 * SLOW, "recovery write / read within budget")
check(t_sw <= 1.5 * SLOW and t_sr <= 0.5 * SLOW and t_sc <= 1.0 * SLOW,
      "project folder: first save / open / open from the CSVs alone within budget")
check(t_s0 <= 0.5 * SLOW and t_s1 <= 0.5 * SLOW, "a save with nothing / one point changed within budget")

# ----------------------------------------------------------------- 7. atomic write, backup
print("\n[7] atomic save, .bak")
ap = os.path.join(OUT, "atomic.kinetrace")
pf.write(pf.freeze(bp, {}, "a" * 32, saved_at="t1"), ap)
first = open(ap, "rb").read()
pf.write(pf.freeze(bp, {}, "a" * 32, saved_at="t2"), ap)
check(open(ap + ".bak", "rb").read() == first, "the previous save is kept as .bak")


class Boom(Exception):
    pass


fz = pf.freeze(bp, {}, "a" * 32, saved_at="t3")
before = open(ap, "rb").read()
try:
    pf.write(fz, ap, yield_gil=lambda: (_ for _ in ()).throw(Boom()))
except Boom:
    pass
check(open(ap, "rb").read() == before and not any(f.endswith(".tmp") for f in os.listdir(OUT)),
      "a write that fails half-way leaves the file untouched and no temp file")
fp_ = os.path.join(OUT, "atomic-folder.kinetrace")
pf.save(bp, fp_, {}, "a" * 32)
bp.sessions[0].tracks[1, 3] = (1.0, 1.0)
before_f = tree(fp_)
try:
    pf.write_folder(pf.freeze(bp, {}, "a" * 32, target=fp_), fp_, yield_gil=lambda: (_ for _ in ()).throw(Boom()))
except Boom:
    pass
check(tree(fp_) == before_f, "a folder save that fails half-way leaves every file untouched")

# ----------------------------------------------------------------- 8. older single files, wrong folders
print("\n[8] a single file becomes a folder")
sf = os.path.join(OUT, "older.kinetrace")
pf.save(bp, sf, {}, "o" * 32, single_file=True)
one = open(sf, "rb").read()
pf.save(pf.load(sf), sf, {}, "o" * 32)
check(os.path.isdir(sf) and open(sf + ".bak", "rb").read() == one and pf.load(sf).sessions[0].n_points == 10,
      "saving a single-file project as a folder keeps the file as .bak")
junk = os.path.join(OUT, "not-a-project")
os.makedirs(junk, exist_ok=True)
open(os.path.join(junk, "thesis.docx"), "w").write("x")
try:
    pf.save(bp, junk, {}, "j" * 32)
    check(False, "saving into a folder that is not a project should be refused")
except pf.ProjectFileError as e:
    check("not a Kinetrace project" in str(e) and os.listdir(junk) == ["thesis.docx"],
          f"a folder that is not a project is left alone ('{e}')")

# ----------------------------------------------------------------- 9. text helpers
print("\n[9] exact text")
ints = np.concatenate([np.array([0, -1, 9, 10, 99, 100, 12345678901]), rng.integers(-99, 10 ** 9, 20000)])
check(pf._join_rows([pf._int_codes(ints), pf._bool_codes(ints % 2 == 0)])
      == "".join(f"{int(v)},{int(v % 2 == 0)}\n" for v in ints), "whole numbers and 0 / 1 written as str() writes them")
f64 = np.concatenate([rng.normal(size=5000) * 10.0 ** rng.integers(-8, 8, 5000), [0.1, 1 / 3, -0.0, 1e300]])
back = np.array([float(x) for x in pf.f64_text(f64)])
check(np.array_equal(back.view(np.uint64), f64.view(np.uint64)), "float64 text reads back bit for bit")
names = ["snout", "Snout", "a/b", "CON", "nul.x", "...", "", "頭 (head)", "tail: tip?"]
files = pf.landmark_files(names)
check(len({f.casefold() for f in files}) == len(files) and all(not any(c in f for c in '<>:"/\\|?*') for f in files)
      and files[0] == "snout.csv" and files[3] == "CON_.csv" and files[7] == "頭 (head).csv",
      f"landmark file names are safe on every OS and unique ignoring case {files}")

print()
if FAILS:
    print(f"verify_projectfile FAILED ({len(FAILS)})")
    sys.exit(1)
print("verify_projectfile PASSED")
