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
        for k in ("n_frames", "fps", "width", "height", "current_frame", "annotator", "_name_counter",
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


# ----------------------------------------------------------------- 1. round trip, both encodings
print("\n[1] round trip")
p = make_project()
path = os.path.join(OUT, "kitchen.kinetrace")
p.path = path
pid = pf.new_id()
pf.write(pf.freeze(p, STATE, pid), path)
q, state, meta = pf.read(path)
check(meta["project_id"] == pid and meta["format_version"] == pf.FORMAT_VERSION, "kinetrace.json id / version")
check(state == STATE, "state.json comes back as written")
tools = {k: v for k, v in STATE["tools"].items()}
for s in p.sessions:                                 # the window-wide toggles land in every camera
    s.ui_state.update(tools)
check(compare(p, q), "every field of a 3-camera project round-trips bit for bit (CSV encoding)")
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
    "TrackingSession": {"n_frames", "fps", "width", "height", "tracks", "visibility", "manual", "tracked",
                        "confidence", "occluded", "radius", "points", "events", "notes", "annotator", "animal",
                        "masks", "body", "skeleton", "current_frame", "ui_state", "_name_counter", "_event_counter"},
    "Project": {"sessions", "names", "offsets", "rates", "active", "calibration", "reconstruction", "lenses"},
    "MaskTrack": {"n_frames", "bbox", "area", "centroid", "score", "contours", "midline"},
    "BodyTrack": {"joints3d", "joints2d", "conf", "score", "bbox", "focal", "cam_t", "names", "backend", "runs",
                  "faces", "mesh", "rig", "n_frames", "n_people", "has_3d", "notes", "n_requested", "step"},
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
pf.write(pf.freeze(bp, {}, "x" * 32, saved_at="2026-01-01T00:00:00"), sp)
with zipfile.ZipFile(sp) as z:
    rows = z.read("cameras/cam1/tracks.csv").decode().strip().splitlines()
check(len(rows) == 1 + 5, f"40,000 frames with 5 tracked cells -> exactly 5 data rows ({len(rows) - 1})")
sp2 = os.path.join(OUT, "sparse2.kinetrace")
pf.write(pf.freeze(bp, {}, "x" * 32, saved_at="2026-01-01T00:00:00"), sp2)
check(open(sp, "rb").read() == open(sp2, "rb").read(), "the same project always gives the same bytes")

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
refused(lambda d: json.dump({"format": "kinetrace-project", "format_version": 2},
                            open(os.path.join(d, "kinetrace.json"), "w")), "newer", "a newer format version")
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
t = time.perf_counter(); pf.write(pf.freeze(perf, {}, "p" * 32), pp); t_sw = time.perf_counter() - t
t = time.perf_counter(); pf.read(pp); t_sr = time.perf_counter() - t
mb = os.path.getsize(pp) / 1e6
print(f"  freeze {t_fz * 1000:.0f} ms | recovery write {t_rw * 1000:.0f} ms, read {t_rr * 1000:.0f} ms | "
      f"project write {t_sw:.2f} s, read {t_sr:.2f} s, {mb:.1f} MB")
SLOW = float(os.environ.get("KINETRACE_PERF_SCALE", "1"))
check(t_fz <= 0.08 * SLOW, "freeze (GUI thread) within budget")
check(t_rw <= 0.2 * SLOW and t_rr <= 0.1 * SLOW, "recovery write / read within budget")
check(t_sw <= 1.5 * SLOW and t_sr <= 1.0 * SLOW, "project write / read within budget")

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

print()
if FAILS:
    print(f"verify_projectfile FAILED ({len(FAILS)})")
    sys.exit(1)
print("verify_projectfile PASSED")
