"""The data-model fixes of the 2026-10-02 code review (session.py, trackio.py,
skeletons.py, render.py; no GPU, no window): I190, I175, I224, I225, I226, I254,
G72, G73, G121, G122, G123 and the R15 table of per-point arrays.

Every check is independent and prints ok / FAIL; the script exits non-zero when
any failed. `KINETRACE_SRC=<folder holding a kinetrace/ package>` runs the same
checks against another copy of the code (that is how each check was shown to
FAIL on the version before the fixes)."""
import csv
import io
import os
import sys
import tempfile

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.environ.get("KINETRACE_SRC") or ROOT
sys.path.insert(0, SRC)

import cv2
import numpy as np

VID = os.path.join(ROOT, "test600.mp4")
OUT = os.path.join(HERE, "out", "review_data")
os.makedirs(OUT, exist_ok=True)

import kinetrace
assert os.path.normcase(os.path.abspath(kinetrace.__file__)).startswith(os.path.normcase(os.path.abspath(SRC))), \
    f"kinetrace imported from {kinetrace.__file__}, not {SRC}"
from kinetrace import render, skeletons, trackio
from kinetrace.session import PALETTE, Event, TrackingSession, in_frame

W, H = 640, 480
results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            fn()
        except BaseException as e:      # noqa: BLE001 - a failing check must not stop the others
            results.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
        else:
            results.append((name, True, ""))
            print(f"ok    {name}")
        return fn
    return deco


def sess(T=100, n_points=0) -> TrackingSession:
    s = TrackingSession("x.mp4", T, 30.0, W, H)
    for k in range(n_points):
        s.add_point(0, 10.0 + k, 20.0 + k)
    return s


def imported(names, xy, conf=None, **kw):
    """xy: (R, N, 2) for one camera."""
    xy = np.asarray(xy, np.float32)[None]
    conf = np.ones(xy.shape[:3], np.float32) if conf is None else conf
    return trackio.Imported(kw.pop("kind", "dlc"), list(names), xy, conf, **kw)


def write(name, text):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p


# ---------------------------------------------------------------- I190 balls
@check("I190 add_ball_prompt keeps the frame's earlier clicks (negative ones too)")
def _():
    s = sess()
    pid = s.add_ball(10, 50.0, 50.0)
    s.add_ball_prompt(pid, 10, 52.0, 52.0, 0)
    s.add_ball_prompt(pid, 10, 51.0, 51.0, 1)
    got = s.points[pid].ball_prompts[10]
    assert got == [[50.0, 50.0, 1], [52.0, 52.0, 0], [51.0, 51.0, 1]], got
    assert tuple(s.tracks[10, pid]) == (51.0, 51.0) and s.manual[10, pid]      # still its hand-placed position
    s.set_position(10, pid, 60.0, 60.0)                       # a hand placement replaces them (as designed)
    assert s.points[pid].ball_prompts[10] == [[60.0, 60.0, 1]]


@check("I190 clear_window drops the ball prompts in the window, keeps the others, undo brings them back")
def _():
    s = sess()
    pid = s.add_ball(10, 50.0, 50.0)
    s.add_ball_prompt(pid, 20, 60.0, 60.0)
    s.add_ball_prompt(pid, 40, 70.0, 70.0)
    other = s.add_ball(20, 1.0, 1.0)
    snap = s.snapshot()
    s.clear_window([pid], 15, 25)
    assert sorted(s.points[pid].ball_prompts) == [10, 40], s.points[pid].ball_prompts
    assert sorted(s.points[other].ball_prompts) == [20]        # another point is untouched
    s.restore(snap)
    assert sorted(s.points[pid].ball_prompts) == [10, 20, 40]
    s.clear_window([pid], 0, 99)
    assert not s.points[pid].ball_prompts
    s.restore(snap)
    s.set_source(pid, "silhouette", "tip")                     # clears the whole track: the clicks go too
    assert not s.points[pid].ball_prompts


# ---------------------------------------------------------------- I175 / I225
def _xypts(names, n_cam, rows=3):
    header = [f"pt{i + 1}_cam{c + 1}_{ax}" for i in range(len(names)) for c in range(n_cam) for ax in "XY"]
    lines = [",".join(header)]
    for r in range(rows):
        lines.append(",".join(f"{100 + 10 * i + r + c + (0.5 if ax == 'Y' else 0):.3f}"
                              for i in range(len(names)) for c in range(n_cam) for ax in "XY"))
    return "\n".join(lines) + "\n"


@check("I175 all-cameras sidecar parsed by its header: names pt1 / cam2 / rows survive, quoted comma too")
def _():
    names = ["pt1", "cam2", "rows", "tail, tip"]
    p = write("allcam.csv", _xypts(names, 2))
    side = io.StringIO()
    w = csv.writer(side, lineterminator="\n")
    w.writerow(["name", "cameras"])
    for nm in names:
        w.writerow([nm, 2])
    w.writerow(["convention", "top-left origin; first pixel = 1 (DLTdv8 / MATLAB)"])
    w.writerow(["rows", "row k = frame k (from 0) of A, the reference camera"])
    w.writerow(["cam1", "A = a.mp4"])
    w.writerow(["cam2", "B = b.mp4"])
    write("allcam_pointnames.csv", side.getvalue())
    imp = trackio.read(p)
    assert imp.names == names, imp.names
    assert not imp.flip_y and imp.n_cameras == 2 and imp.rows == "reference"
    s = sess(n_points=0)
    trackio.apply(s, imp, 1)
    assert [q.name for q in s.points] == names, [q.name for q in s.points]


@check("I175 the all-cameras sidecar in the current unquoted layout still reads (bottom-left too)")
def _():
    p = write("allcam2.csv", _xypts(["a", "b"], 2))
    write("allcam2_pointnames.csv", "name,cameras\na,2\nb,2\nconvention,bottom-left origin; y counted from the "
          "bottom edge; first pixel = 1\nrows,row k = frame k\ncam1,A = a.mp4\ncam2,B = b.mp4\n")
    imp = trackio.read(p)
    assert imp.names == ["a", "b"] and imp.flip_y, (imp.names, imp.flip_y)


@check("I175 single-camera sidecar: session writer quotes a name with a comma and the reader returns it whole")
def _():
    s = sess(T=5)
    for nm in ("pt2", "tail, tip", "x"):
        s.add_point(0, 10.0, 20.0, name=nm)
    for f in range(5):
        for q in range(s.n_points):
            s.set_position(f, q, 11.0 + f + q, 21.0 + f)
    p = os.path.join(OUT, "single.csv")
    s.export_dltdv_csv(p)
    side = os.path.join(OUT, "single_pointnames.csv")
    rows = list(csv.reader(open(side, encoding="utf-8", newline="")))
    assert rows[0] == ["index", "name"] and rows[1] == ["pt1", "pt2"], rows
    assert ["pt2", "tail, tip"] in rows and ["pt3", "x"] in rows, rows
    imp = trackio.read(p)
    assert imp.names == [q.name for q in s.points], imp.names
    t = sess(T=5)
    trackio.apply(t, imp)
    assert [q.name for q in t.points] == [q.name for q in s.points]
    assert np.allclose(t.tracks[:, :, 0], s.tracks[:, :, 0])


@check("I175 apply: a name the file uses twice becomes its own point, never a merge")
def _():
    s = sess()
    xy = np.full((3, 2, 2), np.nan)
    xy[:, 0] = [[10, 10], [11, 11], [12, 12]]
    xy[:, 1] = [[200, 50], [201, 51], [202, 52]]
    res = trackio.apply(s, imported(["2", "2"], xy))
    assert [q.name for q in s.points] == ["2", "2 (2)"], [q.name for q in s.points]
    assert tuple(s.tracks[0, 0]) == (10.0, 10.0) and tuple(s.tracks[0, 1]) == (200.0, 50.0)
    assert res["new"] == 2 and res["duplicates"] == 1 and "more than once" in res["sentence"], res


@check("I225 apply matches a landmark through the exporters' sanitiser (no 'tail_ tip (2)')")
def _():
    s = sess()
    pid = s.add_point(0, 5.0, 5.0, name="tail, tip")
    xy = np.full((2, 1, 2), np.nan)
    xy[:, 0] = [[100, 100], [101, 101]]
    res = trackio.apply(s, imported(["tail_ tip"], xy))
    assert s.n_points == 1 and res["new"] == 0 and res["updated"] == 1, (s.n_points, res)
    assert tuple(s.tracks[1, pid]) == (101.0, 101.0)
    # the real round trip: Kinetrace's own wide DLC export of that session, imported back
    s.set_position(2, pid, 120.0, 130.0)
    p = os.path.join(OUT, "own_dlc.csv")
    s.export_dlc_csv(p)
    s2 = sess()
    pid2 = s2.add_point(0, 1.0, 1.0, name="tail, tip")
    trackio.apply(s2, trackio.read(p))
    assert s2.n_points == 1 and tuple(s2.tracks[2, pid2]) == (120.0, 130.0), [q.name for q in s2.points]


# ---------------------------------------------------------------- I224
def _png(folder, name, w=W, h=H, fill=True):
    img = np.zeros((h, w), np.uint8)
    if fill:
        img[100:200, 150:300] = 255
    assert cv2.imwrite(os.path.join(folder, name), img)


@check("I224 import_masks_png checks every file first: nothing is stored when one is wrong")
def _():
    d = tempfile.mkdtemp(dir=OUT)
    _png(d, "mask_000001.png")
    _png(d, "mask_000002.png")
    _png(d, "mask_000003.png", w=W + 10)                       # another size, LAST in the order
    s = sess(T=10)
    try:
        trackio.import_masks_png(s, d)
    except trackio.TrackImportError as e:
        assert "mask_000003.png" in str(e), e
    else:
        raise AssertionError("no error")
    assert s.animal is None and s.masks is None, "a segment was created by a failed import"
    s.ensure_animal()
    s.masks.set(0, np.pad(np.ones((10, 10), bool), ((50, H - 60), (50, W - 60))))
    before = (s.masks.area.copy(), s.mask_frames().tolist())
    try:
        trackio.import_masks_png(s, d)
    except trackio.TrackImportError:
        pass
    assert s.mask_frames().tolist() == before[1] == [0] and (s.masks.area == before[0]).all(), s.mask_frames()
    with open(os.path.join(d, "mask_000003.png"), "wb") as fh:
        fh.write(b"not an image")                              # unreadable instead of wrong
    try:
        trackio.import_masks_png(s, d)
    except trackio.TrackImportError as e:
        assert "not a readable image" in str(e)
    assert s.mask_frames().tolist() == [0]


@check("I224 a good folder still imports (same outlines as masks.set), frames beyond the video left out")
def _():
    d = tempfile.mkdtemp(dir=OUT)
    _png(d, "mask_000001.png")
    _png(d, "mask_000002.png")
    _png(d, "mask_000099.png")                                 # past the 10 frames
    s = sess(T=10)
    seen = []
    res = trackio.import_masks_png(s, d, lambda a, b: seen.append((a, b)))
    assert res["frames"] == 2 and "1 file(s)" in res["sentence"], res
    assert s.mask_frames().tolist() == [1, 2] and s.animal is not None
    ref = sess(T=10)
    ref.ensure_animal()
    m = np.zeros((H, W), bool)
    m[100:200, 150:300] = True
    ref.masks.set(1, m)
    assert (ref.masks.bbox[1] == s.masks.bbox[1]).all() and ref.masks.area[1] == s.masks.area[1]
    assert seen and seen[-1][1] == 3
    e = sess(T=10)
    trackio.import_masks_png(e, d)
    e2 = sess(T=5)                                             # all files beyond: no segment made for nothing
    d2 = tempfile.mkdtemp(dir=OUT)
    _png(d2, "mask_000050.png")
    trackio.import_masks_png(e2, d2)
    assert e2.animal is None


# ---------------------------------------------------------------- I254 / G72
@check("I254 import bounds = the session's rule: [-0.5, 0) is pixel 0, [W-0.5, W) is kept")
def _():
    s = sess()
    xy = np.array([[[-0.3, 100.0]], [[W - 0.3, 100.0]], [[-0.6, 100.0]], [[W + 0.1, 100.0]],
                   [[100.0, -0.4]], [[100.0, H - 0.2]], [[100.0, H + 0.2]]], np.float64)
    res = trackio.apply(s, imported(["p"], xy))
    t = s.tracked[:7, 0]
    assert t.tolist() == [True, True, False, False, True, True, False], t.tolist()
    assert res["outside"] == 3 and res["cells"] == 4, res
    assert tuple(s.tracks[0, 0]) == (0.0, 100.0) and tuple(s.tracks[4, 0]) == (100.0, 0.0)
    assert in_frame(s.tracks[:7][t, 0], W, H).all(), "stored but outside the session's rule"
    # and a tracking write keeps them: write_segment blanks anything in_frame refuses
    s.write_segment(0, s.tracks[:7].copy(), np.ones((7, 1), bool), [0])
    assert s.tracked[:7, 0].tolist() == t.tolist()


@check("G72 the sentence counts what was written and names the landmarks it skipped")
def _():
    s = sess()
    s.add_landmark("tip", "silhouette", "tip")
    s.add_ball(0, 5.0, 5.0, name="ball")
    s.add_point(0, 1.0, 1.0, name="plain")
    xy = np.full((4, 3, 2), 100.0)
    xy[:, 0] = [[10, 10]] * 4
    xy[:, 1] = [[20, 20]] * 4
    xy[:, 2] = [[30 + k, 30] for k in range(4)]
    res = trackio.apply(s, imported(["tip", "ball", "plain"], xy))
    assert res["cells"] == 4 and res["updated"] == 1, res
    assert res["skipped"] == ["tip", "ball"], res["skipped"]
    snt = res["sentence"]
    assert snt.startswith("4 position(s) of 1 point(s)") and "tip" in snt and "ball" in snt \
        and "silhouette" in snt, snt
    pid_tip = s.pid_by_name("tip")
    assert not s.tracked[:, pid_tip].any(), "a derived landmark was written"


# ---------------------------------------------------------------- G73
def _raises(path):
    try:
        trackio.read(path)
    except trackio.TrackImportError as e:
        assert os.path.basename(path) in str(e), f"the sentence does not name the file: {e}"
        return str(e)
    raise AssertionError("read() accepted a malformed file")


@check("G73 malformed but recognised files give a TrackImportError naming the file")
def _():
    _raises(write("commas.csv", ",,,\n1,2,3,4\n"))
    try:
        trackio.detect(os.path.join(OUT, "commas.csv"))
    except trackio.TrackImportError:
        pass
    else:
        raise AssertionError("a first line of commas was detected as a format")
    _raises(write("dlc_blank.csv", "scorer,s,s\n\ncoords,x,y\n0,1,2\n"))
    _raises(write("dlc_blank_row.csv", "scorer,s,s\nbodyparts,a,a\ncoords,x,y\n\n0,1,2\n"))
    _raises(write("dlc_empty.csv", "scorer,s,s\nbodyparts,a,a\ncoords,x,y\n"))
    _raises(write("sleap_score.csv", "track,frame_idx,instance.score,head.x,head.y,head.score\n"
                  "t0,0,abc,10,20,0.9\n"))
    _raises(write("sleap_nox.csv", "frame_idx,node,y\n0,head,1\n"))
    _raises(write("xy_short.csv", "pt1_cam1_X,pt1_cam1_Y\n1\n"))
    ok = write("dlc_ok.csv", "scorer,s,s\nbodyparts,a,a\ncoords,x,y\n0,1,2\n1,3,4\n")
    assert trackio.read(ok).names == ["a"]


# ---------------------------------------------------------------- G121 / G122
@check("G121 renaming an event to an existing type takes that type's colour")
def _():
    s = sess()
    a = s.add_event("run", 1, 5)
    b = s.add_event("jump", 10, 20)
    assert s.events[a].color != s.events[b].color
    s.update_event(b, name="run")
    assert s.events[b].color == s.events[a].color, (s.events[a].color, s.events[b].color)
    c = s.add_event("rest", 30, 40)
    keep = s.events[c].color
    s.update_event(c, name="sleep")                            # a brand-new name keeps its colour
    assert s.events[c].color == keep and s.events[c].name == "sleep"


@check("G122 validate_template drops duplicate landmark names with a sentence")
def _():
    t, problems = skeletons.validate_template({"name": "t", "landmarks": ["head", "tail", "head", " tail "],
                                               "bones": [["head", "tail"]], "head": "head"})
    assert t["landmarks"] == ["head", "tail"], t["landmarks"]
    assert len(problems) == 2 and "head" in problems[0] and "more than once" in problems[0], problems


@check("G122 apply_skeleton writes the names the points really got, so bones follow a unique_name rename")
def _():
    s = sess()
    s.add_landmark("a_b")                                      # the sanitiser makes "a,b" collide with it
    t = {"name": "t", "head": "a,b", "landmarks": ["a,b", "c"], "bones": [["a,b", "c"]], "derived": {"c": "tip"},
         "note": ""}
    new = s.apply_skeleton(t)
    names = [q.name for q in s.points]
    assert names == ["a_b", "a,b (2)", "c"], names
    assert s.skeleton["landmarks"] == ["a,b (2)", "c"] and s.skeleton["head"] == "a,b (2)", s.skeleton
    assert s.bones() == [(1, 2)], s.bones()
    assert new == [1, 2]


@check("G122 save_user_template never overwrites silently (overwrite=True does)")
def _():
    from pathlib import Path
    old = skeletons.SKELETON_DIR
    skeletons.SKELETON_DIR = Path(tempfile.mkdtemp(dir=OUT))
    try:
        t = {"name": "my skeleton", "landmarks": ["a"], "bones": [], "head": "a"}
        p = skeletons.save_user_template(t)
        assert p.exists()
        t2 = dict(t, landmarks=["a", "b"])
        try:
            skeletons.save_user_template(t2)
        except FileExistsError as e:
            assert "my skeleton.json" in str(e)
        else:
            raise AssertionError("an existing template was overwritten silently")
        assert '"b"' not in p.read_text(encoding="utf-8")
        skeletons.save_user_template(t2, overwrite=True)
        assert '"b"' in p.read_text(encoding="utf-8")
    finally:
        skeletons.SKELETON_DIR = old


# ---------------------------------------------------------------- I226 / G123
@check("I226 the overlay is written at the video file's rate the app measured, not the raw header's")
def _():
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.instance() or QCoreApplication([])
    s = TrackingSession(VID, 600, 30.0, W, H)
    s.file_fps = 50.0
    out = os.path.join(OUT, "fps50.mp4")
    if os.path.exists(out):
        os.remove(out)
    r = render.OverlayRenderer(s, VID, out, render.OverlayOptions(0, 9, 0.5, trails=0), [])
    errs, oks = [], []
    r.error.connect(errs.append)
    r.finished_ok.connect(lambda p, c: oks.append(p))
    r.run()                                                    # synchronously, in this thread
    assert oks and not errs, errs
    cap = cv2.VideoCapture(out)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert abs(fps - 50.0) < 1.0, f"written at {fps} fps"


@check("I226 a non-finite or non-positive rate is refused with a sentence, nothing left behind")
def _():
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.instance() or QCoreApplication([])
    s = TrackingSession(VID, 600, 30.0, W, H)
    out = os.path.join(OUT, "fps_bad.mp4")
    for bad in (float("nan"), 0.0, -5.0, float("inf")):
        if os.path.exists(out):
            os.remove(out)
        r = render.OverlayRenderer(s, VID, out, render.OverlayOptions(0, 9, 0.25, fps=bad), [])
        errs, oks = [], []
        r.error.connect(errs.append)
        r.finished_ok.connect(lambda p, c: oks.append(p))
        r.run()
        assert errs and not oks and "frame rate" in errs[0], (bad, errs, oks)
        assert not os.path.exists(out), "a file was left behind"
    # a bad session rate falls to the next good one; nothing good at all is refused
    s.file_fps = float("nan")
    s.fps = float("nan")
    r = render.OverlayRenderer(s, VID, out, render.OverlayOptions(0, 9, 0.25), [])
    assert r._output_fps(24.0) == 24.0
    try:
        r._output_fps(0.0)
    except RuntimeError as e:
        assert "frame rate" in str(e)
    else:
        raise AssertionError("no rate at all was accepted")
    s.file_fps = 25.0
    r = render.OverlayRenderer(s, VID, out, render.OverlayOptions(0, 9, 0.25, fps=12.5), [])
    assert r._output_fps(24.0) == 12.5                         # the caller's choice wins


@check("G123 overlay text is scaled once: a half-size 4K overlay no longer gets the minimum font")
def _():
    seen = []
    old = render._text
    render._text = lambda img, text, org, scale, color, thick=1: seen.append(scale)
    try:
        s = sess()
        for sc in (1.0, 0.5):
            seen.clear()
            render.draw_overlay(np.zeros((2160, 3840, 3), np.uint8), 5, s, render.OverlayOptions(0, 99, sc), [])
            assert seen, "no text drawn"
            f = seen[0]
            want = 0.55 * (3840 * sc / 1280.0) ** 0.5
            assert abs(f - max(0.4, want)) < 1e-9, (sc, f, want)
            if sc == 0.5:
                assert f > 0.6, f
    finally:
        render._text = old


# ---------------------------------------------------------------- R15
def _filled(T=40, N=4, seed=0):
    from kinetrace.session import POINT_ARRAYS
    rng = np.random.default_rng(seed)
    s = sess(T=T, n_points=N)
    for a in POINT_ARRAYS:
        shape = (T, N) + a.tail
        if a.dtype is bool:
            arr = rng.random(shape) < 0.5
        else:
            arr = rng.random(shape).astype(np.float32)
            arr[rng.random(shape) < 0.3] = np.nan
        setattr(s, a.name, arr)
    return s


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


@check("R15 the table lists every per-point array: session attributes, Snapshot fields, dtypes, shapes")
def _():
    import dataclasses
    from kinetrace.session import POINT_ARRAYS, Snapshot
    s = sess(T=12, n_points=3)
    names = [a.name for a in POINT_ARRAYS]
    assert len(names) == 7 and len(set(names)) == 7
    fields = {f.name for f in dataclasses.fields(Snapshot)}
    assert set(names) <= fields, set(names) - fields
    for a in POINT_ARRAYS:
        arr = getattr(s, a.name)
        assert arr.shape == (12, 3) + a.tail and arr.dtype == np.dtype(a.dtype), (a.name, arr.shape, arr.dtype)
    # no (T, N, ...) array of a session is missing from the table (the field-coverage guard)
    for k, v in vars(s).items():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == (12, 3):
            assert k in names, f"{k} is a per-point array that POINT_ARRAYS does not list"
    # an empty session has N = 0 columns of every array
    e = sess(T=7)
    assert all(getattr(e, a.name).shape == (7, 0) + a.tail for a in POINT_ARRAYS)


@check("R15 snapshot / restore round trip over every field, independent copies, older snapshots default")
def _():
    from kinetrace.session import POINT_ARRAYS
    s = _filled()
    ref = {a.name: getattr(s, a.name).copy() for a in POINT_ARRAYS}
    names = [q.name for q in s.points]
    snap = s.snapshot()
    s.tracks[:] = 1.0                                         # snapshot must not share memory
    s.add_point(3, 1.0, 1.0)
    s.remove_point(0)
    s.clear_window([0], 0, 10)
    s.restore(snap)
    for a in POINT_ARRAYS:
        assert _same(getattr(s, a.name), ref[a.name]), a.name
        assert _same(getattr(snap, a.name), ref[a.name]), "snapshot changed: " + a.name
    assert [q.name for q in s.points] == names
    snap.tracks[:] = 7.0                                      # and restore must not share memory with it
    assert not (s.tracks == 7.0).any()
    old_snap = s.snapshot()
    old_snap.occluded = None
    old_snap.radius = None
    s.restore(old_snap)
    assert s.occluded.shape == s.tracked.shape and not s.occluded.any() and s.occluded.dtype == bool
    assert s.radius.shape == s.tracked.shape and np.isnan(s.radius).all() and s.radius.dtype == np.float32


@check("R15 add / remove / reorder / clear_window keep every array in step and clear to the table's fill")
def _():
    from kinetrace.session import POINT_ARRAYS
    s = _filled(T=30, N=4, seed=1)
    ref = {a.name: getattr(s, a.name).copy() for a in POINT_ARRAYS}
    s.reorder_points([2, 0, 3, 1])
    for a in POINT_ARRAYS:
        assert _same(getattr(s, a.name), ref[a.name][:, [2, 0, 3, 1]]), a.name
    s.remove_point(1)
    for a in POINT_ARRAYS:
        assert _same(getattr(s, a.name), ref[a.name][:, [2, 3, 1]]), a.name
    pid = s.add_empty_point()
    for a in POINT_ARRAYS:
        arr = getattr(s, a.name)
        assert arr.shape == (30, 4) + a.tail and arr.dtype == np.dtype(a.dtype)
        col = arr[:, pid]
        assert (np.isnan(col) if isinstance(a.fill, float) and np.isnan(a.fill) else col == a.fill).all(), a.name
    n = s.clear_window([0, 1], 5, 9)
    for a in POINT_ARRAYS:
        blk = getattr(s, a.name)[5:10, :2]
        assert (np.isnan(blk) if isinstance(a.fill, float) and np.isnan(a.fill) else blk == a.fill).all(), a.name
    assert n == int(ref["tracked"][5:10][:, [2, 3]].sum())


@check("R15 restore_cells puts back only the asked points and frames (every array)")
def _():
    from kinetrace.session import POINT_ARRAYS
    s = _filled(T=30, N=4, seed=2)
    snap = s.snapshot()
    s2 = _filled(T=30, N=4, seed=3)
    s.tracks[:], s.tracked[:], s.radius[:], s.occluded[:] = s2.tracks, s2.tracked, s2.radius, s2.occluded
    s.manual[:], s.visibility[:], s.confidence[:] = s2.manual, s2.visibility, s2.confidence
    cur = {a.name: getattr(s, a.name).copy() for a in POINT_ARRAYS}
    s.restore_cells(snap, [1, 99], 5, 100)                     # an unknown point and a window past the end
    for a in POINT_ARRAYS:
        got, was, want = getattr(s, a.name), cur[a.name], getattr(snap, a.name)
        exp = was.copy()
        exp[5:, 1] = want[5:, 1]
        assert _same(got, exp), a.name
    v = s.data_version
    s.restore_cells(snap, [], 0, 5)
    s.restore_cells(snap, [1], 9, 5)
    assert s.data_version == v, "an empty restore marked the session changed"


@check("R15 one new-point helper: names, colours and counters as before for all five add paths")
def _():
    from kinetrace.session import PointMeta
    s = sess()
    a = s.add_point(0, 1.0, 1.0)                              # P1, palette 0
    b = s.add_ball(0, 2.0, 2.0)                               # ball 2, palette 1
    c = s.add_landmark("snout")                               # palette 2
    d = s.add_empty_point()                                   # P4, palette 3
    g = s.add_point(0, 3.0, 3.0, kind="group", radius=9.0)    # G5, palette 4
    src = PointMeta("far", (1, 2, 3), kind="group", radius=4.0, source="track", spec="", free=True,
                    shape="rect", outline=[[0, 0], [1, 0], [1, 1]], tracker="cotracker3")
    e = s.add_placeholder(src)                                # keeps its own colour, takes no number
    f = s.add_point(0, 4.0, 4.0)                              # P6, palette 5
    got = [(q.name, q.color) for q in s.points]
    assert got == [("P1", PALETTE[0]), ("ball 2", PALETTE[1]), ("snout", PALETTE[2]), ("P4", PALETTE[3]),
                   ("G5", PALETTE[4]), ("far", (1, 2, 3)), ("P6", PALETTE[5])], got
    assert s.points[b].is_ball and s.points[b].ball_prompts == {0: [[2.0, 2.0, 1]]}
    assert s.points[g].kind == "group" and s.points[g].radius == 9.0
    assert (s.points[e].kind, s.points[e].shape, s.points[e].free, s.points[e].outline) == ("group", "circle", True, None)
    assert s.points[e].tracker == "cotracker3", "the placeholder keeps its point's own tracker (G62)"
    assert s.points[c].source == "track" and s.add_landmark("tip", "silhouette", "tip") == 7
    assert s.points[7].derived and s.points[7].spec == "tip"
    assert s.add_landmark("snout") == 8 and s.points[8].name == "snout (2)"
    assert s.add_landmark("") == 9 and s.points[9].name == "point"
    h = s.add_point(0, 1.0, 1.0, name="{n}")                  # a typed name is never run through format()
    assert s.points[h].name == "{n}"


@check("R15 dead bits: keyframe interpolation never writes outside the picture, window honoured")
def _():
    s = sess(T=60)
    pid = s.add_point(10, 100.0, 100.0)
    s.set_position(30, pid, 700.0, 100.0)                      # key outside the picture on the right
    s.set_position(50, pid, 300.0, 300.0)
    n, span = s.interpolate_keyframes(pid, replace=False, window=(10, 40))   # the keys 10 and 30: a line
    assert n > 0 and span is not None
    inter = s.tracked[:, pid] & ~s.manual[:, pid]
    assert in_frame(s.tracks[inter, pid], W, H).all()
    assert not inter[29] and inter[11], "frame 29 would be at x = 670, outside the picture"
    assert inter[:10].sum() == 0 and inter[31:].sum() == 0, "wrote outside the window"
    low = sess(T=60)                                           # next_low_conf still walks backwards
    q = low.add_point(0, 5.0, 5.0)
    for f in (10, 11, 30):
        low.set_position(f, q, 5.0, 5.0)
        low.confidence[f, q] = 0.2
    assert low.next_low_conf(30, forward=False) == (10, 11) and low.next_low_conf(11, forward=False) is None


@check("R15 DLTdv writer: no pixel_origin parameter, the +1 convention text unchanged")
def _():
    import inspect
    from kinetrace.session import dltdv_convention_text
    assert "pixel_origin" not in inspect.signature(TrackingSession.export_dltdv_csv).parameters
    assert dltdv_convention_text(False, 1.0) == "top-left origin; first pixel = 1 (DLTdv8 / MATLAB)"
    assert dltdv_convention_text(False) == dltdv_convention_text(False, 1.0)
    assert "bottom-left" in dltdv_convention_text(True, 1.0)


@check("R15 OverlayOptions keeps only the fields a caller sets")
def _():
    import dataclasses
    names = {f.name for f in dataclasses.fields(render.OverlayOptions)}
    assert "midline" not in names and "skip_hidden" not in names and "fps" in names, names


bad = [r for r in results if not r[1]]
print(f"\n{len(results) - len(bad)} of {len(results)} checks passed")
if bad:
    print("FAILED: " + "; ".join(n for n, _, _ in bad))
    sys.exit(1)
print("VERIFY_REVIEW_DATA PASSED")
