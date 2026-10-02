"""The Moving spot point model (I160, I161, G56-G58, M8): a model-free tracker
for small, fast, featureless targets, the test that recommends a point model
from the user's clicks, and the guidance around it.

[1] core (spots.py) on synthetic clips with exact ground truth (_synth_spots):
    a flapping white spot over moving water with stronger glints, a dark bird
    against a bright sky, a dark "bat" over flickering water among rocks as
    dark as it; each followed to a fraction of a pixel by the right cue; the
    stops (vanishes -> "missing", two crossing -> "ambiguous", leaves the
    picture -> "left"); one click without a speed stops instead of grabbing a
    glint; the size / tiny-spot measurement; settings round trip; the test's
    click rule (20 in a row, gaps <= 2); the scorer picks the right cue on
    each scene with 0 corrections; the tie rule (AllTracker wins a tie).
[2] the worker (encoded clips, no model, no GPU): spot columns, the stop is an
    auto-pause with reason "spot" and the data ends the frame before, a spot
    leaving the picture ends the run without a pause, auto-pause off names it.
[3] the app: Track > Point model: Moving spot through the menu (the project is
    changed, the choice survives a save + reopen and does not leak into a new
    video); the tiny-spot hint on a new point; Track with two clicks follows
    the spot and stops where it vanishes (track cut, notice, point selected);
    Ctrl+Z; the test dialog refuses 10 clicks and says how many more, runs on
    25 and recommends Moving spot (bright), Use sets the project's model and
    the point's settings (saved, Ctrl+Z); the corrections hint; Help > Which
    Point Model opens the manual at that section; a region is left out of a
    Moving spot run with a notice.
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _synth_spots import make_scene, write_video  # noqa: E402
from kinetrace import spots  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "spots")
os.makedirs(OUT, exist_ok=True)


def follow(frames, gt, settings, start=0, vel=True):
    """One spot through in-memory frames: (tracker, {frame: (x, y)})."""
    v = gt[start + 1] - gt[start] if vel else None
    run = spots.SpotRun([(0, gt[start], v, settings)], (frames[0].shape[1], frames[0].shape[0]), start)
    run.resolve(frames[start])
    run.prefill(lambda k: frames[k] if 0 <= k < len(frames) else None, len(frames))
    out = {}
    for f in range(start, len(frames)):
        fx = run.step(f, frames[f])
        if 0 not in fx:
            break
        out[f] = (fx[0].x, fx[0].y)
    return run.trackers[0], out


def errors(out, gt):
    return np.array([np.hypot(out[f][0] - gt[f][0], out[f][1] - gt[f][1]) for f in out if np.isfinite(gt[f]).all()])


# ============================================================ [1] core
print("[1] core")
SC = {k: make_scene(k) for k in ("sea", "sky", "river")}
sea, sea_gt = SC["sea"]
sky, sky_gt = SC["sky"]
riv, riv_gt = SC["river"]

look = spots.measure_spot(sea[0], sea_gt[0])
assert look is not None and look.cue == "bright" and look.diameter < 8, look
look = spots.measure_spot(sky[0], sky_gt[0])
assert look is not None and look.cue == "dark" and look.diameter < 10, look
assert spots.looks_like_small_spot(sea[0], sea_gt[0]) is not None, "the squid-like spot is a tiny spot"
# a big textured animal is not a tiny spot
big = np.full((300, 300, 3), 90, np.uint8)
cv2.ellipse(big, (150, 150), (90, 50), 20, 0, 360, (170, 150, 120), -1)
rng = np.random.default_rng(1)
tex = (rng.normal(0, 25, (300, 300, 1)) * (big[..., :1] > 100)).astype(np.int16)
big = np.clip(big.astype(np.int16) + tex, 0, 255).astype(np.uint8)
assert spots.looks_like_small_spot(big, (150, 150)) is None, "a textured animal is not a tiny spot"
assert spots.measure_spot(sea[0], (-5, 10)) is None and spots.measure_spot(sea[0], (np.nan, 1)) is None
print("  size / tiny-spot measurement OK")

st = spots.SpotSettings("bright", 6.0, 0.5, 1.5)
assert spots.SpotSettings.from_dict(st.to_dict()) == st
for bad in (None, 7, {"cue": "laser"}, {"radius": "x"}, {"radius": 1e9}, {"sigma": float("nan")},
            {"speed_gain": -3}):
    got = spots.SpotSettings.from_dict(bad)
    assert got.cue in ("auto",) + spots.CUES and 0 <= got.radius <= spots.MAX_RADIUS, (bad, got)
assert spots.SpotSettings.from_dict({"cue": "laser"}).automatic
assert spots.SpotSettings().automatic and not st.automatic
print("  settings round trip, odd values fall back to automatic OK")

for name, (frames, gt), cue in (("sea", SC["sea"], "auto"), ("sky", SC["sky"], "auto")):
    t, out = follow(frames, gt, spots.SpotSettings())
    e = errors(out, gt)
    print(f"  {name}: automatic -> {t.settings.describe()}: {len(out)} frames, median {np.median(e):.2f} px, "
          f"max {e.max():.2f} px")
    assert len(out) == len(frames) and t.stopped is None and np.median(e) < 0.5 and e.max() < 1.5
assert follow(sea, sea_gt, spots.SpotSettings())[0].settings.cue == "bright"
assert follow(sky, sky_gt, spots.SpotSettings())[0].settings.cue == "dark"
t, out = follow(riv, riv_gt, spots.SpotSettings("change", 25.0, 2.5, 0))
e = errors(out, riv_gt)
print(f"  river: unusual change -> {len(out)} frames, median {np.median(e):.2f} px, max {e.max():.2f} px")
assert len(out) == len(riv) and np.median(e) < 1.5 and e.max() < 4.0
t, out = follow(riv, riv_gt, spots.SpotSettings("dark"))
assert len(out) < 10, "among rocks as dark as the bat, the dark-spot cue must stop, not follow a rock"
# the wide search on the sea grabs glints: it may not follow, and it must not go on for long
t, out = follow(sea, sea_gt, spots.SpotSettings("bright", 15.0, 0.0, 0))
assert t.stopped is not None and t.stopped[0] < 10, t.stopped
print("  the right cue follows each scene; the wrong ones stop OK")

# stops
fr, gt = make_scene("sea", vanish_at=40)
t, out = follow(fr, gt, spots.SpotSettings())
assert t.stopped == (40, "missing") and max(out) == 39, (t.stopped, max(out))
# a spot that FADES into the water (the owner's squid): it must stop, not follow the ripples
fr, gt = make_scene("sea", fade=(35, 50))
t, out = follow(fr, gt, spots.SpotSettings())
e = errors(out, gt)
print(f"  fading spot: followed to frame {max(out)} (fades 35-50), stopped {t.stopped}, worst {e.max():.2f} px")
assert t.stopped is not None and t.stopped[1] == "missing" and 38 <= t.stopped[0] <= 50 and e.max() < 1.5
fr, gt = make_scene("sea", twin=True)
t, out = follow(fr, gt, spots.SpotSettings())
assert t.stopped is not None and t.stopped[1] == "ambiguous" and 13 <= t.stopped[0] <= 18, t.stopped
e = errors({f: p for f, p in out.items() if f <= 12}, gt)
assert e.max() < 1.0, "before the crossing it is exact"
fr, gt = make_scene("sea", exit_right=True)
t, out = follow(fr, gt, spots.SpotSettings())
assert t.stopped is not None and t.stopped[1] == "left", t.stopped
# one click (no speed): the sky bird is unique -> followed; the sea spot has
# glints alike within the wide first search -> it stops, it does not take one
t, out = follow(sky, sky_gt, spots.SpotSettings(), vel=False)
assert len(out) == len(sky) and np.median(errors(out, sky_gt)) < 0.5
t, out = follow(sea, sea_gt, spots.SpotSettings(), vel=False)
assert t.stopped is not None and t.stopped[0] <= 2 and errors(out, sea_gt).max() < 1.0, (t.stopped, out)
print("  stops: missing at the vanish frame, ambiguous at the crossing, left at the edge; one click "
      "stops rather than taking a glint OK")

# the test's click rule
ok, st_, txt = spots.click_requirement(range(100, 119))
assert not ok and len(st_) == 19 and "20" in txt and "1 more frame" in txt, txt
ok, st_, txt = spots.click_requirement(list(range(0, 10)) + list(range(12, 22)))      # a gap of 2: one stretch
assert ok and len(st_) == 20, txt
ok, st_, txt = spots.click_requirement(list(range(0, 10)) + list(range(13, 23)))      # a gap of 3 breaks it
assert not ok and len(st_) == 10 and "10 more" in txt, txt
ok, st_, txt = spots.click_requirement([])
assert not ok and "no hand-placed frames" in txt
print("  click rule: 20 in a row, a skipped frame or two allowed, says how many more OK")

# the scorer on each scene with the clicks = ground truth + 0.7 px noise (frames 0-29)
for name, cue in (("sea", "bright"), ("sky", "dark"), ("river", "change")):
    frames, gt = SC[name]
    clicks = {f: gt[f] + np.random.default_rng(f).normal(0, 0.7, 2) for f in range(30)}
    look = spots.measure_spot(frames[0], clicks[0])
    sig = look.sigma if look else spots.DEFAULT_SIGMA
    limit = spots.drift_limit(frames[0].shape[1], 2.83 * sig)
    ahead = [(k, cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY)) for k in range(2, 31, 2)]
    res = spots.score_spot_settings(((f, frames[f]) for f in range(30)), clicks, spots.candidate_settings(sig),
                                    (frames[0].shape[1], frames[0].shape[0]), limit, ahead_frames=ahead)
    best = spots.best_spot(res)
    win_ = spots.recommend(list(best.values()))
    print(f"  test on {name}: {win_.label}, {win_.corrections} corrections, median {win_.median_error:.2f} px; "
          + ", ".join(f"{c}: {r.corrections}" for c, r in best.items()))
    assert win_.settings.cue == cue and win_.corrections == 0 and win_.first_ok == 29 and win_.median_error < 1.5
    assert all(r.corrections >= 5 for c, r in best.items() if c != cue and not (name == "sky" and c == "change"))
    txt = spots.verdict_text("P1", list(best.values()))
    assert "Moving spot" in txt and "no correction" in txt, txt
# the tie rule and the protocol for a whole-run model
a = spots.TestResult("AllTracker", "alltracker", frames=10, first_ok=10, errors=[1.0])
b = spots.TestResult("Moving spot: bright spot, search 6 px", "spot", spots.SpotSettings("bright", 6, 0, 1.5),
                     frames=10, first_ok=10, errors=[1.0])
assert spots.recommend([b, a]) is a, "a tie goes to AllTracker (no extra clicking)"
assert "Keep AllTracker" in spots.verdict_text("P1", [b, a])
clicks = {f: np.array([10.0 * f, 0.0]) for f in range(20)}


def drifter(f, xy):          # follows for 3 frames, then 20 px off
    return {g: (np.array([10.0 * g, 0.0]) if g - f <= 3 else np.array([10.0 * g, 20.0])) for g in range(f + 1, 20)}


r = spots.corrected_protocol(drifter, clicks, 6.0, "AllTracker", "alltracker")
assert r.corrections == 4 and r.drifts == 4 and r.first_ok == 3, (r.corrections, r.drifts, r.first_ok)


def stopper(f, xy):          # 2 frames, then nothing (an honest stop)
    return {g: np.array([10.0 * g, 0.0]) for g in range(f + 1, min(20, f + 3))}


r = spots.corrected_protocol(stopper, clicks, 6.0, "x", "spot")
assert r.stops == r.corrections > 0 and r.drifts == 0
print("  the scorer recommends the right cue on every scene (0 corrections); tie -> AllTracker; "
      "drifts and stops counted apart OK")

# I162: the test's search radii come from the clicks' motion; sizes up to ~60 px
steady = {f: sea_gt[f] + np.random.default_rng(f).normal(0, 0.7, 2) for f in range(30)}
zz_fr, zz_gt = make_scene("sky", zigzag=True)
dodge = {f: zz_gt[f] + np.random.default_rng(f).normal(0, 0.7, 2) for f in range(30)}
need_s, need_z = spots.motion_radius(steady), spots.motion_radius(dodge)
assert need_s < 9 and need_z > 15, (need_s, need_z)
grid_old = {(c.cue, c.radius) for c in spots.candidate_settings(1.0)}
grid_new = spots.candidate_settings(1.0, dodge)
for cue in spots.CUES:
    rs = [c.radius for c in grid_new if c.cue == cue]
    assert max(rs) >= need_z and min(rs) == min(r for c_, r in grid_old if c_ == cue), (cue, rs)
assert max(c.radius for c in grid_new if c.cue == "change") > 25, "the change cue is no longer capped at 25 px"
big_fr, big_gt = make_scene("sky", bird_sigma=13.0)
look = spots.measure_spot(big_fr[0], big_gt[0])
assert look.cue == "dark" and 15 < look.diameter < 50 and not look.at_limit, look
t, out = follow(big_fr, big_gt, spots.SpotSettings())
assert len(out) == len(big_fr) and np.median(errors(out, big_gt)) < 1.0
huge = make_scene("sky", n=2, bird_sigma=40.0)
look = spots.measure_spot(huge[0][0], huge[1][0])
assert look.at_limit and spots.looks_like_small_spot(huge[0][0], huge[1][0]) is None, look
print(f"  search radii from the clicks (steady {need_s:.1f} px, dodging {need_z:.1f} px: every cue reaches it, the "
      f"change cue past 25 px); a 13-sigma bird measured {spots.measure_spot(big_fr[0], big_gt[0]).diameter:.0f} px "
      "and followed; a ~110 px one at the size limit OK")

# ============================================================ [2] worker
print("[2] worker")
from kinetrace.tracker import SpotSpec, TrackingWorker  # noqa: E402
from kinetrace.video_source import FrameCache  # noqa: E402


def worker_run(path, n, specs, autopause=True):
    got, paused, fin = {}, [], []
    w = TrackingWorker(path, 0, None, None, FrameCache(1 << 30), n, specs=[], spots=specs, autopause=autopause)
    w.chunk_ready.connect(lambda w0, tr, vi, cf, mm, fr: got.update({w0 + i: tr[i].copy() for i in range(len(tr))}))
    w.autopaused.connect(lambda f, p: paused.append((f, p)))
    w.finished_ok.connect(lambda last, wp: fin.append((last, wp)))
    w.run()
    return w, got, paused, fin


fr, gt = make_scene("sea", vanish_at=40)
vid_v = os.path.join(OUT, "sea_vanish.mp4")
write_video(vid_v, fr)
w, got, paused, fin = worker_run(vid_v, len(fr), [SpotSpec(7, gt[0], gt[1] - gt[0], {})])
data = [f for f in sorted(got) if np.isfinite(got[f][0]).all()]
e = np.array([np.hypot(*(got[f][0] - gt[f])) for f in data])
print(f"  vanish clip: data on {data[0]}-{data[-1]}, median {np.median(e):.2f} px; paused {paused}, "
      f"reason {w._autopause_reason!r}, ended {w._spot_ended}")
assert w.point_ids == [7] and data == list(range(40)) and np.median(e) < 0.5
assert paused == [(40, 7)] and w._autopause_reason == "spot" and w._spot_ended == {7: (40, "missing")}
assert fin and fin[0][1] is True
w, got, paused, fin = worker_run(vid_v, len(fr), [SpotSpec(3, gt[0], gt[1] - gt[0], {}),
                                                  SpotSpec(9, gt[5], gt[6] - gt[5], {"cue": "bright"})],
                                 autopause=False)
assert w.point_ids == [3, 9] and paused == [] and w._spot_ended[3] == (40, "missing")
fr, gt = make_scene("sea", exit_right=True)
vid_x = os.path.join(OUT, "sea_exit.mp4")
write_video(vid_x, fr)
w, got, paused, fin = worker_run(vid_x, len(fr), [SpotSpec(0, gt[0], gt[1] - gt[0], {})])
last_in = int(np.nonzero(np.isfinite(gt[:, 0]))[0].max())
assert paused == [] and w._spot_ended == {} and fin[0][1] is False and fin[0][0] < len(fr) - 1, (paused, fin)
assert max(f for f in got if np.isfinite(got[f][0]).all()) >= last_in - 2
print(f"  spot columns, a stop = auto-pause 'spot' with the data ending the frame before, auto-pause off "
      f"records it, leaving the picture (frame {last_in}) ends the run at {fin[0][0]} without a pause OK")
try:
    TrackingWorker(vid_x, 0, None, None, FrameCache(1 << 20), 10, specs=[], spots=[])
    raise AssertionError("an empty run must be refused")
except ValueError:
    pass

# ============================================================ [3] app
print("[3] app")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from _clean import forget_recovery  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])
from kinetrace import pointtest  # noqa: E402
from kinetrace.app import READY, TRACKING, MainWindow  # noqa: E402


def pump(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout, what):
    t0 = time.time()
    while not cond():
        pump(0.03)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


fr, GT = make_scene("sea", n=60, vanish_at=45)
VID = os.path.join(OUT, "sea_app.mp4")
write_video(VID, fr)
VID2 = os.path.join(OUT, "sky_app.mp4")
write_video(VID2, make_scene("sky", n=30)[0])
forget_recovery(VID, VID2)
win = MainWindow()
win.resize(1400, 900)
win.show()
win._open_video(VID)
wait(lambda: win.state == READY, 60, "open")
pump(0.3)
default = win._point_backend
assert default in ("alltracker", "cotracker3") and win._pm_acts[default].isChecked()


def canvas_click(x, y):
    vp = win.canvas.mapFromScene(QPointF(float(x), float(y)))
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    pump(0.05)


# the tiny-spot hint: N + a real click on the spot, with AllTracker / CoTracker3 chosen
win._goto(0, force=True)
wait(lambda: win.cache.get(0) is not None, 10, "frame 0 cached")
win.canvas.fit()
pump(0.1)
win.btn_add.setChecked(True)
canvas_click(*GT[0])
s = win.session
assert s.n_points == 1
assert "one point" in win.toast.text() and "Moving spot" in win.toast.text(), win.toast.text()
assert "ONE point" in spots.WHICH_MODEL and "AllTracker" in spots.WHICH_MODEL
assert "one point" in win.act_pm_spot.text() and "visible shape" in win._pm_acts["alltracker"].text()
print("  a new point on a small spot: the hint names Moving spot OK")

# switch the point model through the menu; the project is changed by it
s.dirty = False
win.act_pm_spot.trigger()
pump(0.05)
assert win._point_backend == "spot" and win.act_pm_spot.isChecked() and s.ui_state["point_backend"] == "spot"
assert s.dirty, "the point model is a project setting: choosing it changes the project"
# give it its speed: click the spot on frame 1 too, go back to frame 0, track
win._goto(1, force=True)
pump(0.1)
win._on_annotate(*GT[1])
win._goto(0, force=True)
pump(0.1)
win.point_list.clearSelection()
win._start_tracking()
wait(lambda: win.state == TRACKING, 60, "tracking start")
wait(lambda: win.state == READY, 120, "tracking end")
pump(0.3)
have = np.nonzero(s.tracked[:, 0])[0]
e = np.hypot(*(s.tracks[have, 0] - GT[have]).T)
print(f"  app run: data on {have.min()}-{have.max()}, median {np.median(e):.2f} px, max {e.max():.2f} px; "
      f"playhead {win.current}; notice: {ascii(win.toast.text()[:80])}...")
assert have.min() == 0 and have.max() == 44 and np.median(e) < 0.6 and e.max() < 2.0
assert win.current == 45 and win.selected == 0 and "Stopped at frame 45" in win.toast.text()
assert "not found" in win.toast.text()
win._undo_run()
pump(0.1)
assert int(s.tracked[:, 0].sum()) == 2, "Ctrl+Z takes the run back (the two clicks stay)"
print("  Track with Moving spot follows the spot, stops where it vanishes (cut, notice, selected), Ctrl+Z OK")

# the test dialog: 10 clicks -> refused with how many more; 25 -> recommends Moving spot (bright)
seen = []


def fake_exec(self):
    seen.append(self)
    if self.enough:
        self.chk_models.setChecked(False)        # CPU suite: the Moving spot settings only
        self.run_test()
        wait(lambda: self.results is not None or self.error, 120, "the test")
        if self.winner is not None:
            self.btn_use.click()
    return 0


pointtest.PointModelTest.exec = fake_exec
for f in range(2, 10):
    s.set_position(f, 0, *(GT[f] + np.random.default_rng(f).normal(0, 0.6, 2)))
win.selected = 0
win._test_point_models()
d = seen[-1]
assert not d.enough and not d.btn_run.isEnabled() and "at least 20" in d.req.text() and "10 more" in d.req.text(), \
    d.req.text()
print(f"  10 clicks: refused -- {ascii(d.req.text()[:90])}...")
win._set_point_backend(default)                  # the test's Use must set it back to spot
for f in range(10, 25):
    s.set_position(f, 0, *(GT[f] + np.random.default_rng(f).normal(0, 0.6, 2)))
s.points[0].spot = None
win._test_point_models(0)
d = seen[-1]
assert d.enough and d.results is not None, d.error
print(f"  25 clicks: {ascii(d.verdict.text()[:160])}...")
assert d.winner.model == "spot" and d.winner.settings.cue == "bright" and d.winner.corrections == 0
assert win._point_backend == "spot" and win.act_pm_spot.isChecked()
assert s.points[0].spot and s.points[0].spot["cue"] == "bright"
win._undo_run()
pump(0.05)
assert s.points[0].spot is None, "Ctrl+Z takes the settings back"
win._apply_test_choice(0, d.winner)
print("  the test: refused below 20 clicks with how many more; recommends Moving spot (bright) with 0 "
      "corrections; Use sets the project's point model and the point's settings; Ctrl+Z OK")

# saved with the project, restored on open; a new video does not inherit it
proj = os.path.join(OUT, "spots_test.kinetrace")
win.project.save(proj)
from kinetrace.project import Project  # noqa: E402

p2 = Project.load(proj)
assert p2.sessions[0].ui_state.get("point_backend") == "spot" and p2.sessions[0].points[0].spot["cue"] == "bright"
win._open_video(VID2)
wait(lambda: win.state == READY and win.info is not None and win.info.path == VID2, 60, "open 2")
assert win._point_backend == default and win._pm_acts[default].isChecked(), win._point_backend
win._open_project_from_path(proj)
wait(lambda: win.state == READY and win.session is not None and win.session.points, 60, "reopen")
pump(0.2)
assert win._point_backend == "spot" and win.act_pm_spot.isChecked(), "the project's point model comes back"
assert win.session.points[0].spot["cue"] == "bright"
print("  the point model is saved with the project and restored on open; another video keeps the default OK")

# switch at any time, both ways
for k in ("alltracker", "cotracker3", "spot"):
    if win._pm_acts[k].isEnabled():
        win._pm_acts[k].trigger()
        assert win._point_backend == k and win._pm_acts[k].isChecked() and win.session.ui_state["point_backend"] == k

# the corrections hint: AllTracker / CoTracker3 data corrected by hand on 5 of 20 frames
win._pm_acts[default].trigger()
s = win.session
L = 40
tw = np.repeat(GT[None, :L, :], 1, 0).transpose(1, 0, 2).astype(np.float32) + 15.0
s.write_segment(0, tw, np.ones((L, 1), bool), [0], np.full((L, 1), 0.9, np.float32))
win.selected = 0
win.toast.hide()
for f in (20, 22, 24, 26):
    win._goto(f, force=True)
    win._on_annotate(*GT[f])
assert "corrected" not in win.toast.text() or not win.toast.isVisible()
win._goto(28, force=True)
win._on_annotate(*GT[28])
assert "corrected" in win.toast.text() and "Moving spot" in win.toast.text(), win.toast.text()
win.toast.hide()
win._goto(30, force=True)
win._on_annotate(*GT[30])
assert not win.toast.isVisible() or "corrected" not in win.toast.text(), "once per point"
print("  the corrections hint appears after 5 corrections in 20 frames, once OK")

# Help > Which Point Model Should I Use? opens the manual at that section
win.act_help_models.trigger()
pump(0.3)
man = win._manual_dlg
assert man.isVisible() and man.view.verticalScrollBar().value() > 0
assert man.go_to_heading("Which point model should I use?")
txt = man.view.toPlainText()
assert "Which point model should I use?" in txt and "20 frames in a" in txt
man.close()
print("  Help > Which Point Model Should I Use? opens the manual there OK")

# a region is left out of a Moving spot run, with a notice
win.act_pm_spot.trigger()
win._goto(0, force=True)
win._on_add_group(500.0, 200.0, 30.0)
pump(0.05)
win.point_list.clearSelection()
win._start_tracking(stop_after=5)
wait(lambda: win.state == READY and win.worker is None, 60, "region run")
assert "left out" in win.toast.text(), win.toast.text()
print("  a region is left out of a Moving spot run, said OK")

# [4] semi-automatic F steps and an every-camera run with Moving spot
print("[4] semi-automatic step, every camera")
frB, GTB = make_scene("sea", n=40, seed=3)
VIDB = os.path.join(OUT, "sea_camB.mp4")
write_video(VIDB, frB)
VIDA = os.path.join(OUT, "sea_camA.mp4")
frA, GTA = make_scene("sea", n=40, seed=0)
write_video(VIDA, frA)
forget_recovery(VIDA, VIDB)
win._open_video(VIDA)
wait(lambda: win.state == READY and win.info is not None and win.info.path == VIDA, 60, "open camA")
assert win._add_view(VIDB), "add camB"
wait(lambda: win.project is not None and win.project.n_views == 2 and win.state == READY, 60, "camB")
pump(0.3)
win.act_pm_spot.trigger()
p = win.project
win._goto(0, force=True)
win.btn_add.setChecked(True)
win.canvas.add_requested.emit(float(GTA[0, 0]), float(GTA[0, 1]))
pump(0.1)
name = p.sessions[0].points[0].name
win._goto(1, force=True)
win._on_annotate(*GTA[1])
for v, gt in ((1, GTB),):
    win._set_active_view(v)
    pump(0.2)
    j = p.sessions[v].pid_by_name(name)
    win.point_list.setCurrentRow(j)
    for f in (0, 1):
        win._goto(f, force=True)
        win._on_annotate(*gt[f])
win._set_active_view(0)
pump(0.2)
# semi-automatic: F at frame 1 tracks exactly frame 2
win.act_mode_semi.trigger()
win._goto(1, force=True)
win.point_list.clearSelection()
pump(0.1)
win._track_step()
wait(lambda: win.state == READY and win.worker is None, 60, "semi step")
sa = p.sessions[0]
assert sa.tracked[2, 0] and not sa.tracked[3, 0], "one F = one frame"
assert np.hypot(*(sa.tracks[2, 0] - GTA[2])) < 1.0, sa.tracks[2, 0]
win.act_mode_auto.trigger()
# every camera at once, from frame 1 (both cameras know the speed from frames 0-1)
win._goto(1, force=True)
win.act_track_all.setChecked(True)
pump(0.05)
win.point_list.clearSelection()
win._toggle_tracking(all_cameras=True)
wait(lambda: win._multi is not None or win.state == TRACKING, 30, "every-camera start")
wait(lambda: win._multi is None and win.state == READY, 120, "every-camera end")
pump(0.3)
for v, gt in ((0, GTA), (1, GTB)):
    s_ = p.sessions[v]
    j = s_.pid_by_name(name)
    have = np.nonzero(s_.tracked[:, j])[0]
    e = np.hypot(*(s_.tracks[have, j] - gt[have]).T)
    print(f"  camera {v}: data on {have.min()}-{have.max()}, median {np.median(e):.2f} px")
    assert have.max() >= 35 and np.median(e) < 0.6 and e.max() < 2.0
win.act_track_all.setChecked(False)
print("  semi-automatic F steps one frame; Track > Every camera follows the spot in both cameras OK")

win.close()
pump(0.3)
if getattr(win, "_dev_probe", None) is not None:
    win._dev_probe.wait(30000)
forget_recovery(VID, VID2, VIDA, VIDB)
print("verify_spots PASSED")
