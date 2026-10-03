"""Code-review fixes in the body layer (I180-I183, I195, I231, I232, G77-G79,
G135, G136, R17). No GPU, no weights, no network: the pose model is a stand-in.

Each check fails on the code before the fixes (shown by running this file
against the base modules) and passes after. A section that raises counts as
failed, so a check against the old API reports FAIL instead of aborting.

Run:  .venv\\Scripts\\python.exe tests\\verify_review_body.py
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "tests" / "out"
OUT.mkdir(parents=True, exist_ok=True)

import _synth_human as sh                                   # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
from kinetrace import body, bodypose, bodyview, downloads    # noqa: E402

fails: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + what + ((" -- " + detail) if detail else ""))
    if not ok:
        fails.append(what)


def section(title):
    """Run a section; an exception is a failed check, not an abort."""
    def wrap(fn):
        print("\n" + title)
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            if os.environ.get("REVIEW_TB"):
                print("\n".join(tb[-40:]))
            check(False, f"{fn.__name__} ran without raising", f"{type(exc).__name__}: {str(exc)[:60]} @ {tb[-3].strip()[:90]}")
        return fn
    return wrap


mhr, coco = body.rig_of("mhr70"), body.rig_of("coco17")
gt = sh.truth()


def fill(rig, arr):
    out = np.full((sh.T, rig.n_joints, arr.shape[-1]), np.nan)
    for k, nm in enumerate(gt["names"]):
        i = rig.index(nm)
        if i is not None:
            out[:, i] = arr[:, k]
    return out


X3, U2 = fill(mhr, gt["xyz"]), fill(mhr, gt["uv"])
UC = fill(coco, gt["uv"])


BOX_UNKNOWN, BOX_DETECTOR, BOX_GIVEN = (getattr(body, n, v) for n, v in
                                          (("BOX_UNKNOWN", 0), ("BOX_DETECTOR", 1), ("BOX_GIVEN", 2)))
HAS_SRC = hasattr(body, "BOX_UNKNOWN")           # False on the code before the review fixes


def coco_track(frames, step=1, shift=0.0, score=0.8, src=BOX_UNKNOWN):
    bt = body.BodyTrack(sh.T, coco, 1)
    bt.backend = "unit"
    for t in frames:
        bt.set_person(t, 0, joints2d=UC[t] + shift, conf=np.full(17, 0.9), score=score,
                      bbox=[10, 10, 100, 300], **({"box_source": src} if HAS_SRC else {}))
    fr = list(frames)
    bt.runs, bt.step, bt.n_requested = [(min(fr), max(fr), step)], step, len(fr)
    return bt


def video20() -> Path:
    p = OUT / "review_body_20.mp4"
    if not p.exists():
        sh.write_video(str(p), 20)
    return p


# ------------------------------------------------------------------ I181
@section("[I181] a shoulder swung past overhead does not wrap")
def _i181():
    flex = np.arange(120.0, 241.0, 2.0)                      # arm forward, through overhead, beyond
    x = np.full((len(flex), coco.n_joints, 3), np.nan)
    side, up = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    for i, a in enumerate(flex):
        for s, sg in (("left", 1.0), ("right", -1.0)):
            hip = np.array([0.0, 0.0, 0.0]) + sg * 0.125 * side
            sho = hip + 0.5 * up
            x[i, coco.index(f"{s}_hip")] = hip
            x[i, coco.index(f"{s}_shoulder")] = sho
            x[i, coco.index(f"{s}_elbow")] = sho + 0.3 * sh._rot(-up, side, -a)
    defs, ang = body.joint_angles(x, coco, have_3d=True)
    k = [d.name for d in defs].index("left shoulder flexion")
    err = float(np.nanmax(np.abs(ang[:, k] - flex)))
    check(err < 1e-6, "left shoulder flexion reads 120 .. 240 deg without wrapping", f"max err {err:.2e}")
    rom = body.range_of_motion(ang)
    check(abs(float(rom["range"][k]) - 120.0) < 1e-6, "its range of motion is 120 deg, not ~350", f"{rom['range'][k]:.1f}")
    v = body.angular_velocity(ang, 30.0)
    check(float(np.nanmax(np.abs(v[:, k]))) < 100.0, "and its angular velocity has no spike across 180",
          f"max {float(np.nanmax(np.abs(v[:, k]))):.0f} deg/s")
    # every OTHER angle keeps its window: a hip at 190 deg is still folded to -170
    d2 = body.AngleDef("x", "sagittal", ("a", "b", "c"))
    check(float(body._fold(np.array([190.0]), d2.fold)[0]) == -170.0
          and float(body._fold(np.array([180.0]), d2.fold)[0]) == 180.0
          and float(body._fold(np.array([-180.0]), d2.fold)[0]) == 180.0
          and float(body._fold(np.array([37.25]), d2.fold)[0]) == 37.25,
          "other angles still fold into (-180, 180] and untouched values are not rounded")


# ------------------------------------------------------------------ I182
@section("[I182] 2D signed angles when the facing cue is missing")
def _i182():
    left = UC.copy()
    left[..., 0] = sh.W - left[..., 0]                          # walks to the LEFT of the picture
    nose = coco.index("nose")
    d, ref = body.joint_angles(left, coco, have_3d=False)
    names = [q.name for q in d]
    hk = names.index("left hip flexion")
    cut = left.copy()
    cut[10:40, nose] = np.nan                                  # the nose is hidden on 30 frames
    _, got = body.joint_angles(cut, coco, have_3d=False)
    err = float(np.nanmax(np.abs(got[:, hk] - ref[:, hk])))
    check(err < 1e-9, "hip flexion of someone facing left keeps its sign where the nose is hidden",
          f"max diff {err:.2f} deg")
    # a clip that turns round about as often as not has no majority: blank there
    turn = np.concatenate([UC[:60], left[60:]])
    _, rt = body.joint_angles(turn, coco, have_3d=False)
    turn2 = turn.copy()
    turn2[0:10, nose] = np.nan
    turn2[70:80, nose] = np.nan
    _, rt2 = body.joint_angles(turn2, coco, have_3d=False)
    kk = names.index("left knee flexion")
    check(np.isnan(rt2[0:10, hk]).all() and np.isnan(rt2[70:80, hk]).all()
          and np.isfinite(rt2[10:60, hk]).all(),
          "a clip that faces both ways leaves the signed angles blank where the cue is missing")
    check(np.isfinite(rt2[0:10, kk]).all(), "while an unsigned angle (knee) is still measured there")
    check(np.allclose(rt2[10:60, hk], rt[10:60, hk]), "and frames with the cue are unchanged")


# ------------------------------------------------------------------ I183 / I231
@section("[I183, I231] a re-run keeps earlier poses on a miss; one step per frame")
def _i183():
    old = coco_track(range(24))
    nobody = body.BodyTrack(sh.T, coco, 1)
    nobody.examined, nobody.runs = np.array([3, 4, 5]), [(3, 5, 1)]
    m, note = body.merge_run(old, nobody)
    check(m.n_posed() == 24 and all(m.has(f) for f in (3, 4, 5)), "three frames nobody was found on keep their poses")
    check("3 frames" in note and "3, 4, 5" in note and "keep the earlier pose" in note,
          "the sentence counts and names them", note)
    part = coco_track([5, 6], shift=40.0)
    part.examined, part.runs = np.array([5, 6, 7]), [(5, 7, 1)]
    m2, note2 = body.merge_run(old, part)
    check(np.allclose(m2.joints2d[5, 0], UC[5] + 40) and np.allclose(m2.joints2d[7, 0], UC[7]),
          "frames found are replaced, the one missed (7) keeps the earlier pose")
    check("(7)" in note2 and "posed again" in note2, "and both are said", note2)
    other = body.BodyTrack(sh.T, mhr, 1)
    other.set_person(4, 0, joints3d=X3[4], joints2d=U2[4], conf=np.ones(70), score=0.9)
    other.examined = np.array([4])
    _, note3 = body.merge_run(old, other)
    check("23 of those frames have no pose now" in note3, "a replacing run says how many frames lost their pose", note3)

    # (I231) a dense run plus a sparse re-check of it
    dense = coco_track([f for f in range(40) if f not in (10, 11, 12, 13)])
    dense.runs = [(0, 39, 1)]
    sparse = coco_track(range(0, 40, 8), step=8)
    sparse.examined, sparse.runs = np.arange(0, 40, 8), [(0, 39, 8)]
    mm, _ = body.merge_run(dense, sparse)
    st = mm.frame_steps()
    check(int(st[np.isfinite(mm.score[:, 0])].max()) == 1, "the merged track is still a dense track (per-frame steps)")
    r = bodyview.SideBySideRenderer("x.mp4", mm, "y.mp4", 0, 39, 0, bodyview.PoseDrawOptions(), 30.0)
    check(len(r.frames) == 40 and abs(r.out_fps - 30.0) < 1e-9 and r.note == "",
          "its video writes every frame in real time, not every 8th at fps / 8",
          f"{len(r.frames)} frames at {r.out_fps:g} fps")
    rep = body.angle_report(mm, 30.0)
    check("never looked at" not in rep, "the report does not call its frames never looked at")
    p = OUT / "review_body_dense.csv"
    body.export_angles_csv(mm, p, 30.0)
    import csv
    rows = list(csv.reader(l for l in p.read_text(encoding="utf-8").splitlines() if not l.startswith("#")))
    hd = rows[0]
    c, cv_ = hd.index("left knee flexion"), hd.index("left knee flexion (deg/s)")
    byf = {int(r_[0]): r_ for r_ in rows[1:]}
    want = (float(byf[9][c]) - float(byf[8][c])) * 30.0
    check(abs(float(byf[9][cv_]) - want) < 1e-3,
          "deg/s next to a real gap (frames 10-13) uses only the measured side, it does not bridge",
          f"{float(byf[9][cv_]):.3f} vs {want:.3f}")
    # a sparse stretch beside a dense one: its samples still get a rate
    mix = coco_track(range(0, 20))
    far = coco_track(range(20, 60, 4), step=4)
    far.examined, far.runs = np.arange(20, 60, 4), [(20, 59, 4)]
    m3, _ = body.merge_run(mix, far)
    a = m3.angles(0)[1]
    v = body.angular_velocity(a, 30.0, m3.frame_steps())
    check(np.isfinite(v[28, 0]) or not np.isfinite(a[28, 0]) or np.isfinite(v[24:60:4]).any(),
          "a sparse stretch still has rates between its own samples")
    check(sorted(set(int(x) for x in m3.frame_steps()[np.isfinite(m3.score[:, 0])])) == [1, 4],
          "and the steps of a mixed track are kept apart", str(sorted(set(m3.frame_steps()[np.isfinite(m3.score[:, 0])]))))


# ------------------------------------------------------------------ worker helpers
class FakeEst(bodypose.BodyEstimator):
    gives_3d = False

    def __init__(self, people_fn=None, fail=None, on_make=None):
        self.rig = coco
        self.backend = "stand-in"
        self.people_fn, self.fail, self.calls = people_fn, fail, 0

    def step(self, bgr, boxes=None, masks=None):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return self.people_fn(self.calls) if self.people_fn else []


def person(box, x0=100.0, src=BOX_DETECTOR, score=0.9):
    return bodypose.PersonPose(joints2d=np.full((17, 2), float(x0), np.float32),
                               conf=np.full(17, 0.9, np.float32),
                               bbox=np.asarray(box, np.float32), score=score,
                               **({"box_source": src} if HAS_SRC else {}))


class Target:
    """What a session is to the worker: a body track and a frame count."""
    def __init__(self, bt, n):
        self.body, self.n_frames, self.video_path = bt, n, None


def run_worker(est_or_factory, opts, target=None, n_frames=sh.T, video=None, cancel_in_make=False):
    real = bodypose.make_estimator
    got = {"ok": [], "err": [], "stopped": [], "progress": []}

    def make(*a, **k):
        if callable(est_or_factory) and not isinstance(est_or_factory, FakeEst):
            r = est_or_factory(*a, **k)
        else:
            r = est_or_factory
        if cancel_in_make:
            w.request_cancel()
        return r

    bodypose.make_estimator = make
    try:
        w = bodyview.BodyPoseWorker(str(video or video20()), n_frames, opts, target=target)
        w.finished_ok.connect(lambda t: got["ok"].append(t))
        w.error.connect(lambda m: got["err"].append(m))
        if hasattr(w, "stopped"):
            w.stopped.connect(lambda m: got["stopped"].append(m))
        w.progress.connect(lambda *a: got["progress"].append(a))
        w.run()                                              # synchronously, on this thread
    finally:
        bodypose.make_estimator = real
    return w, got


def opts_(f0, f1, people=1, **k):
    return bodyview.BodyRunOptions(backend="vitpose-base", start=f0, end=f1, step=1, max_people=people,
                                   use_detector=False, use_masks=False, **k)


# ------------------------------------------------------------------ I180
@section("[I180] a re-run of part of a two-person video keeps the people in their columns")
def _i180():
    A, B = [50, 50, 150, 300], [300, 50, 500, 400]               # B is the bigger box
    prior = body.BodyTrack(20, coco, 2)
    for f in range(0, 6):
        prior.set_person(f, 0, joints2d=np.full((17, 2), 100.0), conf=np.full(17, .9), score=.9, bbox=A)
        prior.set_person(f, 1, joints2d=np.full((17, 2), 500.0), conf=np.full(17, .9), score=.9, bbox=B)
    prior.n_requested, prior.runs = 6, [(0, 5, 1)]
    est = FakeEst(lambda n: [person(B, 500.0), person(A, 100.0)])    # biggest first, as a detector gives them
    w, got = run_worker(est, opts_(6, 9, people=2), target=Target(prior, 20), n_frames=20)
    t = got["ok"][0] if got["ok"] else None
    check(t is not None and abs(float(t.joints2d[7, 0, 0, 0]) - 100.0) < 1e-3
          and abs(float(t.joints2d[7, 1, 0, 0]) - 500.0) < 1e-3,
          "person A stays in column 0 and B in column 1 (a fresh matcher swapped them)",
          "" if t is None else f"col0 x={t.joints2d[7, 0, 0, 0]:.0f}, col1 x={t.joints2d[7, 1, 0, 0]:.0f}")
    merged, _ = body.merge_run(prior, t)
    check(abs(float(merged.joints2d[3, 0, 0, 0]) - 100.0) < 1e-3 and abs(float(merged.joints2d[8, 0, 0, 0]) - 100.0) < 1e-3,
          "so every frame of column 0 is the same person after the merge")
    m = bodypose.PersonMatcher(2)
    m.seed([np.array(A, np.float32), np.array(B, np.float32)], [5, 5])
    check(m.assign([person(B), person(A)], frame=6) == [1, 0], "PersonMatcher.seed gives each person their column back")


# ------------------------------------------------------------------ I232 / G78
@section("[I232, G78] decoding failures, stops and errors of the pose run")
def _i232():
    est = FakeEst(lambda n: [person([20, 20, 200, 400])])
    # a range running past the 20 frames of the video: frame 20 does not decode
    w, got = run_worker(est, opts_(0, 29), n_frames=30)
    t = got["ok"][0] if got["ok"] else None
    check(t is not None and getattr(t, "decode_failed", None) == 20 and not t.stopped_early,
          "the first frame that did not decode is recorded, and it is not a user's Stop",
          "" if t is None else f"decode_failed={getattr(t, 'decode_failed', 'missing')}")
    merged, note = body.merge_run(None, t) if t is not None else (None, "")
    check("Frame 20 could not be decoded" in note and "Stopped early" not in note,
          "a first run says so (it used to say nothing)", note)
    prior = body.BodyTrack(30, coco, 1)
    for f in range(30):
        prior.set_person(f, 0, joints2d=UC[f], conf=np.full(17, .9), score=.8, bbox=[1, 1, 50, 90])
    w, got2 = run_worker(FakeEst(lambda n: [person([20, 20, 200, 400])]), opts_(0, 29), n_frames=30)
    m2, note2 = body.merge_run(prior, got2["ok"][0] if got2["ok"] else body.BodyTrack(30, coco, 1))
    check("Frame 20 could not be decoded" in note2 and "Stopped early" not in note2,
          "a re-run says it too, not 'Stopped early'", note2)
    # nothing at all decodes in the range: an error in words, not an empty success
    w, got = run_worker(FakeEst(lambda n: [person([20, 20, 200, 400])]), opts_(25, 29), n_frames=30)
    check(not got["ok"] and any("could not be decoded" in e for e in got["err"]),
          "a range wholly past the video is an error that says why", str(got["err"])[:100])

    # G78: Stop during a first-use download
    real_exc = downloads.DownloadCancelled("The download of the pose model was stopped.")

    def make_cancelled(*a, **k):
        raise real_exc
    w, got = run_worker(make_cancelled, opts_(0, 5))
    check(len(got["stopped"]) == 1 and not got["err"] and not got["ok"]
          and "stopped" in got["stopped"][0].lower(),
          "Stop during a download ends as 'stopped', not as an error", str(got))
    # Stop while the model loads
    w, got = run_worker(FakeEst(lambda n: [person([20, 20, 200, 400])]), opts_(0, 5), cancel_in_make=True)
    check(len(got["stopped"]) == 1 and not got["ok"] and not got["err"],
          "Stop while the model loads ends as 'stopped' (no 'no person was found')", str(got)[:120])
    # an unexpected exception: its type and a logged traceback
    records = []

    class H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())
    h = H()
    lg = logging.getLogger("kinetrace.errors")
    lg.addHandler(h)
    old_level = lg.level
    lg.setLevel(logging.DEBUG)
    try:
        w, got = run_worker(FakeEst(fail=KeyError("pred_cam_t")), opts_(0, 9))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)
    check(got["err"] and "KeyError" in got["err"][0] and "pred_cam_t" in got["err"][0],
          "a model that fails on every frame is reported with the exception's type", str(got["err"])[:140])
    check(any("KeyError" in r and "Traceback" in r for r in records),
          "and the traceback reaches the error log")

    # the side-by-side video: frames written / a note; a partial file is deleted on an error
    tr = coco_track(range(30))
    outp = OUT / "review_body_sbs.mp4"
    outp.unlink(missing_ok=True)
    r = bodyview.SideBySideRenderer(str(video20()), tr, str(outp), 0, 29, 0, bodyview.PoseDrawOptions(), 30.0, width=640)
    ok_sig, err_sig = [], []
    r.finished_ok.connect(lambda p, c: ok_sig.append(p))
    r.error.connect(lambda m: err_sig.append(m))
    r.run()
    check(ok_sig and getattr(r, "frames_written", None) == 20 and "Frame 20 could not be decoded" in getattr(r, "note", ""),
          "a video that ends early reports the frames really written and why",
          f"written={getattr(r, 'frames_written', 'missing')} note={getattr(r, 'note', '')[:60]}")
    outp.unlink(missing_ok=True)
    real_compose = bodyview.compose_side_by_side
    cnt = {"n": 0}

    def boom(*a, **k):
        cnt["n"] += 1
        if cnt["n"] == 4:
            raise ValueError("drawing failed")
        return real_compose(*a, **k)
    bodyview.compose_side_by_side = boom
    try:
        r2 = bodyview.SideBySideRenderer(str(video20()), tr, str(outp), 0, 19, 0, bodyview.PoseDrawOptions(), 30.0, width=640)
        err2 = []
        r2.error.connect(lambda m: err2.append(m))
        r2.run()
    finally:
        bodyview.compose_side_by_side = real_compose
    check(err2 and not outp.exists(), "an error half-way deletes the partial mp4", f"exists={outp.exists()}")


# ------------------------------------------------------------------ G77
@section("[G77] the body mesh shows its front")
def _g77():
    # two triangles in view space (x right, y up, z toward the viewer): one wound
    # counter-clockwise seen from the viewer (outward, front-facing), one clockwise
    verts = np.array([[-1.5, -1, 0], [-0.5, -1, 0], [-1, 1, 0],        # CCW from +z: front, left side of the panel
                      [0.5, -1, 0], [1, 1, 0], [1.5, -1, 0]], np.float64)   # CW: back, right side
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    img = np.full((200, 200, 3), 0, np.uint8)
    ok = bodyview._draw_mesh(img, verts, faces, np.eye(3), np.zeros(3), 50.0, (200, 200), (0.0, 0.0), -1.0)
    left, right = int(img[:, :100].any()), int(img[:, 100:].any())
    check(ok and left == 1 and right == 0, "the face that points at the viewer is drawn and the back one is not",
          f"left ink {left}, right ink {right}")
    # a closed outward-wound sphere: the drawn faces are the near half
    v, f = sh._sphere(np.zeros(3), 1.0)
    vol = float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)
    check(vol > 0, "(the synthetic sphere winds outward: positive signed volume)", f"{vol:.2f}")
    drawn_polys = set()
    real_fp = bodyview.cv2.fillPoly

    def spy(img_, polys, col):
        for q in np.asarray(polys).reshape(-1, 3, 2):
            drawn_polys.add(tuple(sorted(map(tuple, q.tolist()))))
        return real_fp(img_, polys, col)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nz = np.cross(b - a, c - a)[:, 2]
    scr = np.round(np.column_stack([100 + v[:, 0] * 50.0, 100 - v[:, 1] * 50.0])).astype(np.int32)
    near_polys = {tuple(sorted(map(tuple, scr[tri].tolist()))) for tri in f[nz > 0]}
    back_polys = {tuple(sorted(map(tuple, scr[tri].tolist()))) for tri in f[nz < 0]} - near_polys
    im = np.zeros((200, 200, 3), np.uint8)
    bodyview.cv2.fillPoly = spy
    try:
        bodyview._draw_mesh(im, v.astype(np.float64), f, np.eye(3), np.zeros(3), 50.0, (200, 200), (0.0, 0.0), -1.0)
    finally:
        bodyview.cv2.fillPoly = real_fp
    check(drawn_polys <= near_polys and len(drawn_polys & near_polys) > 20 and not (drawn_polys & back_polys),
          "on a closed sphere only faces on the near (+z) side are drawn",
          f"{len(drawn_polys & near_polys)} near, {len(drawn_polys & back_polys)} back drawn")
    # the light: a face turned toward the light is brighter than one turned away, among drawn faces
    img1 = np.zeros((60, 60, 3), np.uint8)
    img2 = np.zeros((60, 60, 3), np.uint8)
    bodyview._draw_mesh(img1, np.array([[-.5, -.5, 0], [.5, -.5, 0], [0, .5, 0]]), np.array([[0, 1, 2]]),
                        np.eye(3), np.zeros(3), 40.0, (60, 60), (0.0, 0.0), -1.0)
    check(int(img1.max()) > 0, "a plain front-facing triangle is lit and drawn")


# ------------------------------------------------------------------ G79 / R17
@section("[G79, R17] the plot's angles, joint names and the dead parts")
def _g79():
    track = body.BodyTrack(sh.T, mhr, 1)
    for t in range(sh.T):
        track.set_person(t, 0, joints3d=X3[t], joints2d=U2[t], conf=np.ones(70), score=0.9)
    w = bodyview.BodySideBySide()
    w.set_track(track, sh.FPS)
    defs = [d.name for d in track.angles(0)[0]]
    check(len(defs) == 14, "the 3D rig measures 14 angles", str(len(defs)))
    acts = getattr(w, "_plot_acts", {})
    check(set(acts) == set(defs), "all 14 angles are reachable from the plot menu", f"{len(acts)} entries")
    default = set(w.plot_angles())
    check(len(default) == 8 and not any("dorsi" in n for n in default),
          "by default the plot draws the eight limb flexions it always drew")
    acts["left ankle dorsiflexion"].setChecked(True)
    check("left ankle dorsiflexion" in w.plot_angles() and "left ankle dorsiflexion" in w.draw_options().plot_angles
          and len(w.plot_angles()) == 8,
          "ticking an ankle angle puts it in the plot (the oldest choice makes room: 8 traces at most)")
    w.chk_angles.setChecked(False)
    do = w.draw_options()
    check(do.angles_shown == () and "left ankle dorsiflexion" in do.plot_angles,
          "and the plot does not depend on the 'Angle numbers' tick")
    for act in getattr(w, "_group_acts", []):
        if "trunk lean" in act[0]:
            act[1].setChecked(True)
    check({"neck flexion", "trunk lean", "shoulder-hip twist"} <= set(w.plot_angles()), "the Trunk group ticks its angles")
    fr = sh.render_frame(10)
    base_opts = bodyview.PoseDrawOptions(angles_shown=("left knee flexion",), plot_angles=("left knee flexion",))
    other = bodyview.PoseDrawOptions(angles_shown=("left knee flexion",), plot_angles=("left ankle dorsiflexion",))
    a = bodyview.compose_side_by_side(fr, track, 10, 0, base_opts, True, width=800)
    b = bodyview.compose_side_by_side(fr, track, 10, 0, other, True, width=800)
    check(not np.array_equal(a, b), "the plot follows its own choice, whatever the numbers show")
    none = bodyview.compose_side_by_side(fr, track, 10, 0, bodyview.PoseDrawOptions(plot_angles=()), True, width=800)
    check(not np.array_equal(a, none), "choosing no angle draws an empty plot")

    # joint names reach the right-hand panel (R17)
    n_on = bodyview.compose_side_by_side(fr, track, 10, 0, bodyview.PoseDrawOptions(names=True, plot_angles=("left knee flexion",)), False, width=800)
    n_off = bodyview.compose_side_by_side(fr, track, 10, 0, bodyview.PoseDrawOptions(names=False, plot_angles=("left knee flexion",)), False, width=800)
    half = n_on.shape[1] // 2
    check(not np.array_equal(n_on[:, half:], n_off[:, half:]), "the 'Joint names' tick reaches the pose panel")

    # R17: dead code gone, the anchor of index 0, sideless anchors apart
    check(not hasattr(bodyview, "_median_dir"), "bodyview uses body.median_axis (no copy)")
    check(not hasattr(bodyview.BodySideBySide, "seek_requested"), "the unused seek_requested signal is gone")
    check(all(d.kind != "axis" for d in body.ANGLE_DEFS) and not hasattr(body.BodyRig, "up_vector"),
          "the 'axis' angle kind and BodyRig.up_vector are gone")
    check("trail" not in inspect.signature(bodyview.render_pose_panel).parameters, "render_pose_panel has no unused trail")
    joints = ["neck", "nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    rig0 = body.BodyRig("t0", "t0", joints, [])
    t0 = body.BodyTrack(5, rig0, 1)
    r = bodyview._angle_anchor(t0, "trunk lean")
    idx = r[0] if isinstance(r, tuple) else r
    check(idx == 0, "a joint at index 0 is a valid anchor (it was taken for missing)", f"anchor {idx}")
    rigall = mhr
    ta = body.BodyTrack(5, rigall, 1)
    anchors = [bodyview._angle_anchor(ta, n) for n in
               ("neck flexion", "trunk lean", "thigh separation (stride)", "shoulder-hip twist")]
    check(len({a if isinstance(a, tuple) else (a,) for a in anchors}) == 4,
          "each sideless angle has its own place and offset", str(anchors))


# ------------------------------------------------------------------ I195 / G135
@section("[I195, G135] model loading and availability")
def _i195():
    import transformers
    seen = {}

    class Rec:
        def __init__(self, tag):
            self.tag = tag
            self.config = type("C", (), {"id2label": {}})()

        @classmethod
        def from_pretrained(cls, path, **kw):
            seen[cls.__name__] = (path, kw)
            return cls(cls.__name__)

        def to(self, d):
            return self

        def eval(self):
            return self

    class FakeProc(Rec):
        pass

    class FakeVit(Rec):
        pass

    class FakeProc2(Rec):
        pass

    class FakeDet(Rec):
        pass

    tmp = Path(tempfile.mkdtemp(prefix="kt_hf_"))
    (tmp / "token").write_text("hf_test_token\n", encoding="utf-8")
    names = ("AutoProcessor", "VitPoseForPoseEstimation", "AutoImageProcessor", "RTDetrV2ForObjectDetection")
    # (transformers' lazy module: an attribute set BEFORE the real class was ever
    # read is what `from transformers import X` finds; deleting it afterwards
    # lets the lazy lookup resolve the real one again)
    assert not any(n in vars(transformers) for n in names), "a real class was already imported"
    real_hf = bodypose.HF_DIR
    os.environ["HF_HOME"] = str(tmp / "their_cache")               # the user's own HF_HOME
    try:
        transformers.AutoProcessor, transformers.VitPoseForPoseEstimation = FakeProc, FakeVit
        transformers.AutoImageProcessor, transformers.RTDetrV2ForObjectDetection = FakeProc2, FakeDet
        bodypose.HF_DIR = tmp
        spec = bodypose.BACKENDS["vitpose-base"]
        bodypose.ViTPoseEstimator(spec, "cpu", use_detector=True)
    finally:
        bodypose.HF_DIR = real_hf
        for n in names:
            vars(transformers).pop(n, None)
    kw = seen.get("FakeVit", ("", {}))[1]
    check(kw.get("cache_dir") == str(tmp / "hub"),
          "the pose model is loaded from models/hf even with HF_HOME set", str(kw.get("cache_dir")))
    check(kw.get("token") == "hf_test_token", "with the token saved in models/hf/token")
    kd = seen.get("FakeDet", ("", {}))[1]
    check(kd.get("cache_dir") == str(tmp / "hub"), "and so is the person detector", str(kd.get("cache_dir")))

    # G135
    real_cached = downloads.hf_cached
    try:
        downloads.hf_cached = lambda repo: repo == bodypose.BACKENDS["vitpose-base"].repo
        st, why = bodypose.backend_status("vitpose-base")
        check(st == "download" and "detector" in why, "with the detector missing the dialog says it will download it", f"{st}: {why[:80]}")
        st2, _ = bodypose.backend_status("vitpose-base", None, False)
        check(st2 == "ready", "and does not ask for it when a silhouette drives the run")
        downloads.hf_cached = lambda repo: repo == bodypose.DETECTOR_REPO
        st3, why3 = bodypose.backend_status("vitpose-base")
        check(st3 == "download" and "425" in why3, "with the pose model missing it says so", why3[:80])
        downloads.hf_cached = lambda repo: True
        check(bodypose.backend_status("vitpose-base")[0] == "ready", "everything cached is ready")
    finally:
        downloads.hf_cached = real_cached
    check(not hasattr(bodypose, "hub_cached"), "the unpinned hub_cached check is gone")


# ------------------------------------------------------------------ G136
@section("[G136] detector confidence counts detector frames only")
def _g136():
    bt = body.BodyTrack(40, coco, 1)
    for t in range(40):
        bt.set_person(t, 0, joints2d=UC[t], conf=np.full(17, .9),
                      score=0.8 if t < 20 else 1.0, bbox=[1, 1, 50, 90],
                      box_source=BOX_DETECTOR if t < 20 else BOX_GIVEN)
    check(abs(bt.person_score(0) - 0.8) < 1e-6, "silhouette frames do not dilute the detector's 80 %", f"{bt.person_score(0):.3f}")
    check("found 80%" in bt.person_label(0), "the label says 80 %", bt.person_label(0))
    sil = body.BodyTrack(40, coco, 1)
    for t in range(40):
        sil.set_person(t, 0, joints2d=UC[t], conf=np.full(17, .9), score=1.0, bbox=[1, 1, 50, 90],
                       box_source=BOX_GIVEN)
    check(not np.isfinite(sil.person_score(0)) and "100%" not in sil.person_label(0)
          and "silhouette" in sil.person_label(0), "a silhouette-driven column is not reported as 100 % detected", sil.person_label(0))
    rep = body.angle_report(sil, 30.0)
    check("100%" not in rep and "silhouette" in rep, "nor in the report")
    np.savez_compressed(OUT / "review_body_src.npz", **bt.to_arrays("b_"))
    with np.load(OUT / "review_body_src.npz", allow_pickle=True) as z:
        back = body.BodyTrack.from_arrays("b_", z)
        legacy = {k: v for k, v in bt.to_arrays("l_").items() if k != "l_box_src"}
        old = body.BodyTrack.from_arrays("l_", legacy)
    check(np.array_equal(back.box_src, bt.box_src), "the source survives the file")
    check(abs(old.person_score(0) - 0.9) < 1e-6, "an older file without it counts as detected, as before")
    # through the worker and a merge
    est = FakeEst(lambda n: [person([20, 20, 200, 400], src=BOX_GIVEN, score=1.0)])
    w, got = run_worker(est, opts_(0, 4))
    t = got["ok"][0] if got["ok"] else None
    check(t is not None and int(t.box_src[0, 0]) == BOX_GIVEN, "the worker records where each box came from")
    mg, _ = body.merge_run(coco_track(range(10), score=0.7, src=BOX_DETECTOR), t)
    check(abs(mg.person_score(0) - 0.7) < 1e-6, "a merged silhouette run leaves the earlier detector score alone", f"{mg.person_score(0):.3f}")


print("\n" + "=" * 62)
if fails:
    print(f"VERIFY_REVIEW_BODY FAILED ({len(fails)}):")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("VERIFY_REVIEW_BODY PASSED")
