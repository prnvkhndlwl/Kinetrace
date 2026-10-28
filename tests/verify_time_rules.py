"""(I364-I366, G225, R50) One rounding rule for frame numbers and frame-rate ratios (the 2026-10-25 sweep's PR 5).

  [map]  `Project.map_frame` is unchanged by `map_index` (the rule written once): every (src, dst, frame) of
         three cameras at rates 1, 2, 5/6 and offsets 0, 0.5, -37.5 gives what the old formula gave
  [I364] `rate_of` keeps the exact ratio (100 / 120 fps = 5/6, not 0.833333: 0.013 frames of drift at reference
         frame 40,000), snapped to a simple fraction so 2x / 4x rigs stay exactly 2.0 / 4.0; a saved rate reopens
         exactly, and a project saved with 0.833333 keeps it
  [G225] the timeline's thin "tracked in another camera" line and its tooltip follow the frame the other
         camera's TILE shows (a 60 fps camera in a 30 fps rig: a .5 tie goes DOWN towards the earlier camera)
  [I365] Re-track probes a stretch's quarter frames with half_up (10..15 -> 11, 13, 14, 15; round() gave 12)
  [I366] GoPro movement frames with half_up (t = 1.25 s at 2 fps -> frame 3; round() gave 2)
  [R50]  wanddata uses the shared time helpers; no Python round() / np.rint / np.round makes a frame number in
         project / retrack / gpmf / wanddata

.venv\\Scripts\\python.exe tests\\verify_time_rules.py
"""
import ast
import json
import os
import shutil
import sys
import time
import types
from fractions import Fraction

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out", "time_rules")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")
os.environ["KINETRACE_UPDATE_API"] = "http://127.0.0.1:9"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
app = QApplication.instance() or QApplication([])

from kinetrace import cameraorder, gpmf, retrack  # noqa: E402
from kinetrace import projectfile as pf  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.project import Project, rate_of  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402
from _stubs import stub_track_choice  # noqa: E402
TRACK_CHOICE = stub_track_choice()     # (G217) the Track pop-up answers "keep": no press waits on it

FAILS = []


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    if not ok:
        FAILS.append(what)


def pump(sec=0.15):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def wait(cond, timeout):
    t0 = time.time()
    while not cond():
        pump(0.02)
        if time.time() - t0 > timeout:
            return False
    return True


def sessions(*fps, n=200):
    return [TrackingSession(f"cam{k}.mp4", n, float(f), 640, 480) for k, f in enumerate(fps)]


# ------------------------------------------------------------------ [map]
print("[map] map_frame gives what it always gave (the rule now written once)")


def old_index(p, src, dst, frame):
    """map_frame's rounding as it was on main eaee38b (project.py:212-228), kept here as the reference."""
    t = (float(int(frame)) - p.offsets[src]) / p.rates[src]
    x = p.rates[dst] * float(t) + p.offsets[dst]
    return int(np.floor(x + 0.5)) if src <= dst else int(np.ceil(x - 0.5))


def old_map_frame(p, src, dst, frame):
    if not (0 <= src < p.n_views and 0 <= dst < p.n_views):
        return None
    local = old_index(p, src, dst, frame)
    if local < 0 or local >= p.sessions[dst].n_frames:
        return None
    return local


pm = Project(sessions(30, 60, 25, n=900), None, [0.0, 0.5, -37.5], rates=[1.0, 2.0, 0.8333333333333334])
diff = [(a, b, f) for a in range(3) for b in range(3) for f in range(0, 401)
        if pm.map_frame(a, b, f) != old_map_frame(pm, a, b, f)]
check(not diff, "every (src, dst, frame) of 3 cameras x 401 frames maps exactly as before", diff[:5])
if hasattr(pm, "map_index"):
    fr = np.arange(-50, 401)
    vec_ok = all(np.array_equal(pm.map_index(a, b, fr), [old_index(pm, a, b, f) for f in fr])
                 for a in range(3) for b in range(3))
    check(vec_ok, "Project.map_index (vectorised) = map_frame's rule on every frame, no range check")
else:
    check(False, "Project.map_index exists (map_frame's rule for many frames at once)")

# ------------------------------------------------------------------ [I364]
print("[I364] a camera's rate is the exact ratio of the frame rates")
p = Project(sessions(120, 100, n=40001))
exact = float(Fraction(5, 6))
check(p.rates[1] == exact, "a 100 fps camera in a 120 fps rig: rate 5/6 to the last bit", repr(p.rates[1]))
drift = abs(100.0 / 120.0 * 40000 - p.local_frame(1, 40000))
check(drift < 1e-9, "its frame at reference frame 40,000 is where 100/120 puts it (was 0.013 frames early)",
      f"{drift:.3e}")
cases = [((239.76, 119.88), 2.0), ((240.0, 120.0), 2.0), ((119.88011988011988, 29.97002997002997), 4.0),
         ((29.97, 30.0), float(Fraction(999, 1000))), ((60.0, 30.0), 2.0), ((15.0, 60.0), 0.25)]
for (a, b), want in cases:
    got = rate_of(a, b)
    check(got == want, f"rate_of({a}, {b}) = {want!r}", repr(got))
for a, b in ((23.976, 25.0), (240000 / 1001, 120.0), (59.94, 50.0), (120.0001, 120.0)):
    check(rate_of(a, b) == a / b, f"{a} / {b} (no simple fraction within a billionth) stays the raw ratio",
          repr(rate_of(a, b)))
check(rate_of(0.0, 30.0) == 1.0 and rate_of(30.0, 0.0) == 1.0, "an unknown rate still gives 1.0")
p2 = Project(sessions(120, n=100))
p2.add_view(sessions(100, n=100)[0], "cam2")
check(p2.rates[1] == exact, "a camera ADDED to a 120 fps project gets 5/6 too", repr(p2.rates[1]))

path = os.path.join(OUT, "rates.kinetrace")
p.dirty = True
pf.save(p, path)
q = pf.load(path)
check(q.rates[1] == p.rates[1], "Save and reopen: the rate comes back to the last bit", repr(q.rates[1]))
jsons = [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs
         if f.endswith(".json") and '"rate"' in open(os.path.join(dp, f), encoding="utf-8").read()]
if jsons:
    with open(jsons[0], encoding="utf-8") as fh:
        meta = json.load(fh)

    def set_rate(o):
        if isinstance(o, dict):
            if "rate" in o and isinstance(o["rate"], float) and o["rate"] != 1.0:
                o["rate"] = 0.833333
            for v in o.values():
                set_rate(v)
        elif isinstance(o, list):
            for v in o:
                set_rate(v)
    set_rate(meta)
    with open(jsons[0], "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    old = pf.load(path)
    check(old.rates[1] == 0.833333, "a project saved with rate 0.833333 reopens with 0.833333 (nothing re-written)",
          repr(old.rates[1]))
else:
    check(False, "(setup) the saved project has a json with the cameras' rates")

# ------------------------------------------------------------------ [I366]
print("[I366] GoPro movement frames: a .5 goes up")
info = gpmf.GoProInfo(path="x.mp4", fps=2.0)
got = [info.frame_of(t) for t in (1.25, 0.25, 2.25, 0.75)]
check(got == [3, 1, 5, 2], "frame_of at 2 fps for 1.25, 0.25, 2.25, 0.75 s = 3, 1, 5, 2", got)
ts = np.arange(0.0, 5.0, 0.01)
usual = np.array([0.0, 0.0, 1.0])
tilted = np.array([0.0, np.sin(np.radians(20)), np.cos(np.radians(20))])
g = np.where(((ts >= 1.25) & (ts < 2.25))[:, None], tilted, usual)
mv = gpmf._moves(ts, g, usual, info)
check(len(mv) == 1 and (mv[0]["frame"], mv[0]["end_frame"]) == (3, 5),
      "a tilt from 1.25 s to 2.25 s at 2 fps: 'Moved: frames 3-5' (was 2-4)",
      [(m["start_s"], m["end_s"], m["frame"], m["end_frame"]) for m in mv])

# ------------------------------------------------------------------ [I365]
print("[I365] Re-track probes a stretch's frames with half_up")
s_rt = TrackingSession("r.mp4", 40, 30.0, 640, 480)
s_rt.add_landmark("p")
asked = []
OUTSIDE = set()


def fake_ray_target(project, view, name, local_frame, probe=None, why=None, **kw):
    asked.append(int(local_frame))
    if local_frame in OUTSIDE and why is not None:
        why.append("the other cameras put it outside this camera's picture there (right edge)")
        return None
    return np.array([100.0, 100.0]), 2


def run_plan():
    asked.clear()
    st = retrack.Stretch(view=0, name="p", t0=10, t1=15, local0=10, local1=15, median_px=9.0)
    saved = (retrack.disagreeing_stretches, retrack._blame, retrack.ray_target)
    retrack.disagreeing_stretches = lambda project, thr, min_run=retrack.MIN_RUN: [st]
    retrack._blame = lambda project, out, thr: out
    retrack.ray_target = fake_ray_target
    try:
        proj_rt = types.SimpleNamespace(sessions=[s_rt], n_views=1, calibration=types.SimpleNamespace(n_calibrated=2))
        return retrack.plan(proj_rt, 5.0)
    finally:
        retrack.disagreeing_stretches, retrack._blame, retrack.ray_target = saved


res = run_plan()
check(asked == [10, 11, 13, 14, 15], "a stretch 10..15: the start, then 11, 13, 14, 15 (12.5 -> 13)", asked)
check(res and res[0].target is not None and not res[0].reason, "... and with every probe inside, it is re-tracked",
      (res[0].reason if res else None))
OUTSIDE.add(13)
res = run_plan()
check(res and res[0].target is None and "at frame 13" in res[0].reason,
      "the point leaving the picture at frame 13 is caught (the stretch is not re-tracked into blank frames)",
      (res[0].reason if res else None))

# ------------------------------------------------------------------ [R50]
print("[R50] the shared time helpers; no round() for a frame number")
src = {m: open(os.path.join(ROOT, "kinetrace", m + ".py"), encoding="utf-8").read()
       for m in ("project", "retrack", "gpmf", "wanddata")}
tree = ast.parse(src["wanddata"])
imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
check({"half_up", "local_frame"} <= imported, "wanddata imports half_up and calib.local_frame", sorted(imported))
check("floor(" not in "".join(ln for ln in src["wanddata"].splitlines() if "+ 0.5" in ln),
      "wanddata has no hand-written floor(x + 0.5)")


from kinetrace.calib import local_frame  # noqa: E402
from kinetrace.project import half_up  # noqa: E402
rng = np.random.default_rng(5)
same = True
for trial in range(200):
    rate = float(rng.choice([1.0, 0.5, 2.0, 1.001, 0.25, 0.8333333333333334, rng.uniform(0.2, 5.0)]))
    off = float(rng.choice([0.0, 0.5, -37.5, 237.5, rng.uniform(-300, 300)]))
    f = np.concatenate([np.sort(rng.integers(0, 5000, int(rng.integers(1, 300)))), [0, 1, 2]]).astype(np.int64)
    x = np.concatenate([(f - off) / rate, np.arange(-20, 20) + 0.5])           # exact .5 ties too
    same &= np.array_equal(half_up(x), np.floor(x + 0.5).astype(np.int64))
    same &= np.array_equal(local_frame(f, rate, off), rate * f.astype(np.float64) + off)
    same &= half_up(float(x[0])) == int(np.floor(x[0] + 0.5))
check(bool(same), "the shared helpers give exactly what wanddata's inlined maths gave (200 random trials, .5 ties)")


def round_calls(text):
    out = []
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name in ("round", "rint"):
                out.append(n.lineno)
    return out


ALLOWED = {("project", "has_fractional_offsets")}      # `o - round(o)`: is an offset whole? (not a frame number)
bad = []
for m, text in src.items():
    funcs = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for ln in round_calls(text):
        owner = min((f for f in funcs if f.lineno <= ln <= f.end_lineno), key=lambda f: f.end_lineno - f.lineno,
                    default=None)
        if (m, owner.name if owner else "") not in ALLOWED:
            bad.append(f"{m}.py:{ln} ({owner.name if owner else 'module'})")
check(not bad, "no round() / np.rint / np.round in project, retrack, gpmf, wanddata (one allowed: whole-offset test)",
      bad)

# ------------------------------------------------------------------ [G225]
print("[G225] the timeline's other-camera line follows the other camera's tile")
W, H = 320, 240


def clip(name, fps, n):
    path = os.path.join(OUT, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in range(n):
        im = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(im, str(f), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        vw.write(im)
    vw.release()
    return path


SLOW, FAST = clip("slow30.mp4", 30.0, 30), clip("fast60.mp4", 60.0, 60)
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([FAST], ""))
cameraorder.CameraOrderDialog.exec = lambda self: (self.buttons.button(QDialogButtonBox.Ok).click(), self.result())[1]
win = MainWindow()
win.resize(1500, 950)
win.show()
win._open_video(SLOW)
check(wait(lambda: win.state == READY and win.project is not None and not win._loading, 30), "(setup) camera 1 opens")
win._add_video_dialog()
check(wait(lambda: win.project is not None and win.project.n_views == 2 and not win._loading, 30),
      "(setup) the 60 fps camera is added")
pj = win.project
check(pj.rates == [1.0, 2.0], "(setup) the second camera runs at twice the rate", pj.rates)
QTest.mouseClick(win.btn_new_point, Qt.LeftButton)          # + Point: in every camera, no data
pump(0.3)
s_slow = pj.sessions[0]
j_slow = len(s_slow.points) - 1
name = s_slow.points[j_slow].name
s_slow.set_position(5, j_slow, 100.0, 100.0)               # tracked in camera 1 ONLY on its frame 5
s_slow.tracked[5, j_slow] = True
s_slow._touch()
cv_fast = win.grid.canvas(1)
QTest.mouseClick(cv_fast.viewport(), Qt.LeftButton, Qt.NoModifier, cv_fast.viewport().rect().center())
pump(0.4)
if pj.active != 1:
    win._set_active_view(1)                                 # (the click did not reach the tile offscreen)
    pump(0.4)
check(pj.active == 1, "(setup) working in the 60 fps camera")
s_fast = pj.sessions[1]
j_fast = s_fast.pid_by_name(name)
check(j_fast is not None and not s_fast.tracked[:, j_fast].any(), "(setup) the 60 fps camera has no data for it")
win._goto(7, force=True)
pump(0.3)
tl = win.timeline


def key_right():
    win.activateWindow()
    app.setActiveWindow(win)
    win.canvas.setFocus()
    QTest.keyClick(win.canvas, Qt.Key_Right)
    pump(0.25)


steps = []
for _ in range(7):                                           # frames 8..14: the frame camera 1's tile shows
    key_right()
    tile_frame = win._views[0].want_frame
    steps.append((win.current, tile_frame, tile_frame is not None and bool(s_slow.tracked[tile_frame, j_slow])))
key_right()                                                  # the playhead's own line away from 8..14
tl.update()
pump(0.3)
y0, y1 = tl._row_rect_y(j_fast)
ym = (y0 + y1) // 2                                          # the thin line runs along the lane's middle
img = tl.grab().toImage()
xb = int(round((tl._x_edge(40) + tl._x_edge(41)) / 2))       # frame 40: no camera has data there
bg = img.pixelColor(xb, ym)
rows = []
for f, tile_frame, tile_has in steps:
    x = int(round((tl._x_edge(f) + tl._x_edge(f + 1)) / 2))
    c = img.pixelColor(x, ym)
    lit = abs(c.red() - bg.red()) + abs(c.green() - bg.green()) + abs(c.blue() - bg.blue()) > 60
    QTest.mouseMove(tl, QPoint(x, ym))
    pump(0.1)
    rows.append((f, tile_frame, tile_has, lit, ("tracked in " + pj.name(0)) in tl.toolTip()))
check([r[0] for r in rows] == list(range(8, 15)), "(setup) Right steps the playhead one frame at a time",
      [r[0] for r in rows])
check([r[1] for r in rows] == [4, 4, 5, 5, 6, 6, 7], "(setup) camera 1's tile shows frames 4, 4, 5, 5, 6, 6, 7",
      [r[1] for r in rows])
check(all(r[3] == r[2] for r in rows), "the thin line is lit exactly where camera 1's tile shows the point "
      "(frames 10 and 11)", [(r[0], r[2], r[3]) for r in rows])
check(all(r[4] == r[2] for r in rows), "hovering it names camera 1 exactly there",
      [(r[0], r[2], r[4]) for r in rows])

win.project.dirty = False
win.close()
pump(0.3)
win._dev_probe.wait(10000)
print("\nverify_time_rules: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
