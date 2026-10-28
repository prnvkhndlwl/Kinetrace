"""(I269) A lens profile measured with the camera level fits the same camera filmed on its side.

A camera turned 90 degrees still records its normal landscape pictures; the video only carries a
rotation tag, and the decoder turns the pictures by it (a 4K GoPro clip filmed on its side plays
2160 x 3840). The lens is the same, so its profile is TURNED to fit -- exactly, by the difference
between the board video's turn and the camera's -- instead of refused as "another picture size".
An upside-down camera (same size, turned 180) gets its profile turned too, where the size check
alone attached it with the centre on the wrong side. A turn that is not known is refused in words.

Pure: the turn is exact for the standard, rational and fisheye models (projection identity), the
fitting rule, the file round trip; the decoder's turn on ffmpeg-tagged clips (and its direction:
the picture really is the stored one turned that way); a board scan records its video's turn.
Through the window (offscreen): 3D -> Load a Lens Profile for Cameras... on a camera filmed on its
side and on an upside-down one, an unknown turn refused, the wand wizard's "Use for all" targets.
(I275, G184) A profile made before the turn was recorded gets it from its checkerboard video: the pure
step (size check, refusals, GoPro OREN kept), then the offer through the wand wizard's "Load file...",
Load a Lens Profile for Cameras... and the one-camera entry.

.venv\\Scripts\\python.exe tests\\verify_lens_turn.py
"""
import os
import shutil
import subprocess
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(1, HERE)
OUT = os.path.join(HERE, "out", "lens_turn")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["KINETRACE_RECOVERY_DIR"] = os.path.join(OUT, "recovery")

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

ASK = {"open": "", "question": QMessageBox.Yes, "warn": [], "asked": []}
# a list answers one dialog after the other (the lens file, then its checkerboard video: I275)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ASK["open"].pop(0) if isinstance(ASK["open"], list)
                                                            else ASK["open"], ""))
QMessageBox.warning = staticmethod(lambda *a, **k: (ASK["warn"].append(str(a[2]) if len(a) > 2 else ""),
                                                    QMessageBox.Ok)[1])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: (ASK["asked"].append(str(a[2]) if len(a) > 2 else ""),
                                                     ASK["question"])[1])
app = QApplication.instance() or QApplication([])
import _lens_attach  # noqa: E402  (the camera list: the working camera stays ticked)

_lens_attach.install()

from kinetrace import gpmf, lens  # noqa: E402
from kinetrace.app import READY, MainWindow  # noqa: E402
from kinetrace.calibwizard import share_targets  # noqa: E402
from kinetrace.video_source import applied_rotation, display_rotation, open_capture, probe_video  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    line = ("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else "")
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)      # the console is cp1252
    if not ok:
        FAILS.append(what)


def pump(sec=0.2):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.005)


def same(a, b, tol=1e-9):
    return ((a.width, a.height, a.fisheye) == (b.width, b.height, b.fisheye)
            and np.allclose(a.K, b.K, rtol=0, atol=tol) and np.allclose(np.ravel(a.dist), np.ravel(b.dist), rtol=0, atol=tol))


FF = imageio_ffmpeg.get_ffmpeg_exe()


def tagged(src, deg, dst):
    """`src` with ffmpeg's display rotation `deg` (counter-clockwise, ffmpeg's sign), stream copied."""
    subprocess.run([FF, "-y", "-loglevel", "error", "-display_rotation", str(deg), "-i", src, "-c", "copy", dst],
                   check=True)
    return dst


def write_video(path, frames, fps=10.0):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


# ------------------------------------------------------------------ 1. the turn is exact
print("[1] turning a profile is exact")
W, H = 3840, 2160
rng = np.random.default_rng(3)
obj = np.column_stack([rng.uniform(-1, 1, 400), rng.uniform(-0.6, 0.6, 400), rng.uniform(1.5, 4, 400)])
K = np.array([[1800.0, 0, 1931.3], [0, 1795.0, 1069.8], [0, 0, 1]])
MODELS = {
    "standard": lens.LensProfile(W, H, K, np.array([-0.21, 0.05, 0.0013, -0.0021, -0.004]), False, rotation=0),
    "rational + thin prism": lens.LensProfile(W, H, K, np.array([-0.21, 0.05, 0.0013, -0.0021, -0.004, 0.01, 0.002,
                                                                 -0.001, 0.0011, -0.0004, 0.0007, 0.0002]),
                                              False, rotation=0),
    "fisheye": lens.LensProfile(W, H, np.array([[1500.0, 0, 1925.0], [0, 1500.0, 1075.0], [0, 0, 1]]),
                                np.array([0.05, -0.01, 0.003, -0.0005]), True, rotation=0),
}


def project(p, X):
    if p.fisheye:
        return cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), p.K,
                                         p.dist.reshape(4, 1))[0].reshape(-1, 2)
    return cv2.projectPoints(X, np.zeros(3), np.zeros(3), p.K, p.dist)[0].reshape(-1, 2)


def turn_pixels(uv, w, h, cw):          # what the decoder does to a pixel centre, cw degrees clockwise
    for _ in range(cw // 90):
        uv = np.column_stack([(h - 1) - uv[:, 1], uv[:, 0]])
        w, h = h, w
    return uv


def turn_points(X, cw):                 # the camera turned about its optical axis the same way
    for _ in range(cw // 90):
        X = np.column_stack([-X[:, 1], X[:, 0], X[:, 2]])
    return X


for name, p0 in MODELS.items():
    worst = 0.0
    for cw in (90, 180, 270):
        t = lens.turn_profile(p0, cw)
        worst = max(worst, float(np.max(np.abs(turn_pixels(project(p0, obj), W, H, cw)
                                                - project(t, turn_points(obj, cw))))))
        sz = (H, W) if cw in (90, 270) else (W, H)
        check((t.width, t.height) == sz and t.rotation == cw, f"{name} {cw}: size {sz}, records the turn")
    check(worst < 1e-8, f"{name}: every point projects to the turned pixel ({worst:.1e} px)")
    back = lens.turn_profile(lens.turn_profile(p0, 270), 90)
    check(same(back, p0, 1e-12) and back.rotation == 0, f"{name}: turned 270 then 90 is the profile again")
check(lens.turn_profile(MODELS["standard"], 0) is MODELS["standard"], "a turn of 0 is the profile itself")
skew = lens.LensProfile(W, H, np.array([[1800.0, 3.0, 1920], [0, 1800.0, 1080], [0, 0, 1]]), np.zeros(5), rotation=0)
try:
    lens.turn_profile(skew, 90)
    check(False, "a skewed camera matrix is refused, not turned wrongly")
except ValueError:
    check(True, "a skewed camera matrix is refused, not turned wrongly")

# ------------------------------------------------------------------ 2. the fitting rule
print("\n[2] which profile fits which camera")
p0 = MODELS["standard"]
f, said = lens.fit_profile(p0, (H, W), 270, "CAM1")
check(f is not None and same(f, lens.turn_profile(p0, 270)) and f.rotation == 270
      and "turned 90° counter-clockwise" in said,
      "level profile on a camera filmed on its side (the GoPro case): turned, and said", said)
check(f is not None and np.isclose(f.f_square, p0.f_square), "the focal length is the same lens's")
f, said = lens.fit_profile(p0, (W, H), 180, "CAM1")
check(f is not None and np.allclose(f.principal, (W - 1 - p0.principal[0], H - 1 - p0.principal[1]))
      and "upside down" in said, "an upside-down camera: same size, the centre moved to the other side", said)
f, said = lens.fit_profile(p0, (W, H), 0, "CAM1")
check(f is p0 and said == "", "same turn, same size: the profile as it is")
unknown = lens.LensProfile(W, H, K, p0.dist)
f, said = lens.fit_profile(unknown, (H, W), 270, "CAM1")
check(f is None and "not known which way" in said and "Calibrate a Lens" in said,
      "a profile that does not record its turn is refused on a turned camera, in words", said)
f, said = lens.fit_profile(unknown, (W, H), 0, "CAM1")
check(f is unknown and said == "", "...and taken as it is at the same size (as always)")
f, said = lens.fit_profile(p0, (H, W), None, "CAM1")
check(f is None and "does not say which way" in said, "a video whose turn is not known: refused, in words", said)
f, said = lens.fit_profile(p0, (1920, 1080), 0, "CAM1")
check(f is None and "1920 x 1080" in said, "another picture size: refused as before (I31)", said)
f, said = lens.fit_profile(p0, (1080, 1920), 270, "CAM1")
check(f is None and "1080 x 1920" in said, "another size even when turned: refused", said)
rt = lens.LensProfile.from_json(lens.turn_profile(p0, 270).to_json())
check(rt.rotation == 270 and same(rt, lens.turn_profile(p0, 270)), "the turn is saved with the profile")
d = p0.to_json()
d.pop("rotation")
check(lens.LensProfile.from_json(d).rotation is None, "a file without it: turn not known")
check(lens.same_profile(p0, lens.turn_profile(p0, 270)) and lens.same_profile(lens.turn_profile(p0, 90), p0)
      and not lens.same_profile(p0, lens.LensProfile(H, W, K, p0.dist)),
      "a turned copy is the same lens (a shared profile, G40); another lens is not")

# ------------------------------------------------------------------ 3. the decoder's turn
print("\n[3] what the decoder does with a rotation tag")
frames = []
for k in range(10):
    img = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(img, (20 + k, 30), (120 + k, 90), (40, 200, 250), -1)          # marks the top-left
    cv2.circle(img, (260, 190), 25, (250, 80, 40), -1)
    frames.append(img)
LEVEL = write_video(os.path.join(OUT, "level.mp4"), frames)
SIDE = tagged(LEVEL, 90, os.path.join(OUT, "side.mp4"))          # ffmpeg 90 = counter-clockwise, as the GoPro clip
UPSIDE = tagged(LEVEL, 180, os.path.join(OUT, "upside.mp4"))
for path, want, size in ((LEVEL, 0, (320, 240)), (SIDE, 270, (240, 320)), (UPSIDE, 180, (320, 240))):
    info = probe_video(path)
    check(info.rotation == want and (info.width, info.height) == size and display_rotation(path) == want,
          f"{os.path.basename(path)}: turned {want} clockwise, {size[0]} x {size[1]}",
          (info.rotation, info.width, info.height))
caps = [open_capture(p) for p in (LEVEL, SIDE)]
(_, a), (_, b) = caps[0].read(), caps[1].read()
for c in caps:
    c.release()
check(b is not None and np.array_equal(cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE), b),
      "the turned clip's picture IS the stored one turned 270 clockwise (the direction the profile is turned)")
check(display_rotation(os.path.join(OUT, "never_opened.mp4")) is None, "a video never opened: turn not known")

# the GoPro lens model is in the stored picture
gi = gpmf.GoProInfo("x.mp4")
gi.width, gi.height, gi.poly, gi.zmpl = 3840, 2160, [0.0, 1.2, 0.0, -0.05], 1.0
gp = gpmf.lens_profile(gi)
check(gp is not None and gp.rotation == 0 and lens.fit_profile(gp, (2160, 3840), 270)[0] is not None,
      "GoPro's lens model (stored 3840 x 2160) fits the same GoPro filmed on its side")

# a board scan records its video's turn
board = lens.checkerboard_image(9, 6, 40, 40)
bframes = []
for k in range(6):
    canvas = np.full((480, 640), 255, np.uint8)
    M = cv2.getRotationMatrix2D((board.shape[1] / 2, board.shape[0] / 2), 4.0 * k, 0.9)
    warped = cv2.warpAffine(board, M, (board.shape[1], board.shape[0]), borderValue=255)
    y, x = 20 + 4 * k, 40 + 6 * k
    canvas[y:y + warped.shape[0], x:x + warped.shape[1]] = warped[:480 - y, :640 - x]
    bframes.append(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
BOARD = tagged(write_video(os.path.join(OUT, "board_level.mp4"), bframes), 90, os.path.join(OUT, "board_side.mp4"))
scan = lens.scan_video(BOARD, (9, 6), max_candidates=6)
check(scan.rotation == 270 and scan.size == (480, 640) and len(scan.corners) >= 3,
      "a board video filmed on its side: the scan records its turn", (scan.rotation, scan.size, len(scan.corners)))
BOARD_LEVEL = os.path.join(OUT, "board_level.mp4")


def ident(a, b):
    """The same numbers, bit for bit (Tier A: a recorded turn must not move the lens)."""
    return (a is not None and b is not None and (a.width, a.height, a.fisheye) == (b.width, b.height, b.fisheye)
            and np.array_equal(a.K, b.K) and np.array_equal(np.ravel(a.dist), np.ravel(b.dist)))


# ------------------------------------------------------------------ 5. the turn read from the board video
print("\n[5] a profile with no turn recorded gets it from its checkerboard video (I275)")
OLD = lens.LensProfile(320, 240, np.array([[300.0, 0, 163.5], [0, 302.0, 117.0], [0, 0, 1]]),
                       np.array([-0.2, 0.04, 0.001, -0.002, 0.0]), False, 0.3, 20, "checkerboard (made before I269)",
                       {"verdict": "good", "verdict_reasons": ["clean"], "principal_px": [163.5, 117.0]})
check(OLD.rotation is None, "the old profile records no turn")
tu = getattr(lens, "turn_unknown", None)
check(tu is not None and tu(OLD, (240, 320), 270) and not tu(OLD, (320, 240), 0) and not tu(OLD, (240, 320), None)
      and not tu(lens.turn_profile(MODELS["standard"], 0), (2160, 3840), 270) and not tu(OLD, (640, 480), 0),
      "turn_unknown: only an unknown PROFILE turn on a quarter-turned camera whose own turn is known")
vt, rt_ = getattr(lens, "video_turn", None), getattr(lens, "record_turn", None)
if vt is None or rt_ is None:
    check(False, "lens.video_turn / lens.record_turn exist")
else:
    size, rot, oren = vt(LEVEL)
    check(size == (320, 240) and rot == 0 and oren == "", "video_turn: the level video, not GoPro footage",
          (size, rot, oren))
    filled, said = rt_(OLD, LEVEL, size, rot, oren)
    check(filled is not None and filled.rotation == 0 and ident(filled, OLD) and filled is not OLD and OLD.rotation is None
          and filled.board.get("named_later") and filled.board.get("video") == "level.mp4"
          and "level.mp4" in said, "the level board video: turn 0 recorded in a copy, the lens numbers untouched", said)
    check(filled is not None and "named afterwards" in lens.board_sentence(filled.board)
          and "0 boards" not in lens.board_sentence(filled.board), "the report says the video was named afterwards",
          lens.board_sentence(filled.board) if filled is not None else None)
    if filled is not None:
        got, said = lens.fit_profile(filled, (240, 320), 270, "CAM1")
        check(ident(got, lens.turn_profile(OLD, 270)) and "turned" in said,
              "...then fits the camera filmed on its side, bit for bit the exact turn", said)
        rt2 = lens.LensProfile.from_json(filled.to_json())
        check(rt2.rotation == 0 and rt2.board.get("named_later") and ident(rt2, OLD), "the recorded turn is saved")
        again, said = rt_(filled, LEVEL, size, rot, oren)
        check(again is None and "already" in said, "a profile that records its turn is not changed", said)
    for vid, what in ((SIDE, "240 x 320"), (BOARD_LEVEL, "640 x 480")):
        s_, r_, o_ = vt(vid)
        bad, said = rt_(OLD, vid, s_, r_, o_)
        check(bad is None and what in said and "320 x 240" in said,
              f"{os.path.basename(vid)}: another picture size is not this profile's video, in words", said)
    bad, said = rt_(OLD, LEVEL, (320, 240), None, "")
    check(bad is None and "cannot say" in said, "a decoder that cannot say the turn: refused, in words", said)
    side_old = lens.LensProfile(240, 320, np.array([[302.0, 0, 117.0], [0, 300.0, 163.5], [0, 0, 1]]),
                                OLD.dist.copy(), False, 0.3, 20, "measured on a board filmed on its side")
    s_, r_, o_ = vt(SIDE)
    side_f, said = rt_(side_old, SIDE, s_, r_, o_)
    check(side_f is not None and side_f.rotation == 270, "measured on a board filmed on its side: turn 270", said)
    if side_f is not None:
        lev, _ = lens.fit_profile(side_f, (320, 240), 0, "CAM2")
        sid, _ = lens.fit_profile(side_f, (240, 320), 270, "CAM1")
        check(ident(lev, lens.turn_profile(side_old, 90)) and sid is side_f,
              "...fits a level camera turned 90 and the camera on its side as it is")
    import _synth_gopro  # noqa: E402
    for o in ("U", "L"):
        gp_vid = _synth_gopro.make_gopro_video(os.path.join(OUT, f"gopro_{o}.mp4"), 320, 240, seconds=1, fps=10,
                                               header={"oren": o}, with_sensors=False)
        s_, r_, o_ = vt(gp_vid)
        g_f, _ = rt_(OLD, gp_vid, s_, r_, o_)
        check(o_ == o and g_f is not None and g_f.board.get("gopro_orientation") == o,
              f"GoPro board video (OREN {o}): its mounting tag is kept in the board record", (s_, r_, o_))
    gs = lens.scan_video(os.path.join(OUT, "gopro_U.mp4"), (9, 6), max_candidates=3)
    check(getattr(gs, "gopro_orientation", None) == "U"
          and lens.board_record(gs, [], (9, 6), 0.025).get("gopro_orientation") == "U"
          and "gopro_orientation" not in lens.board_record(scan, [], (9, 6), 0.025),
          "a board scan of GoPro footage records its OREN in the board record (other footage: no entry)")

# ------------------------------------------------------------------ 4. through the window
print("\n[4] Load a Lens Profile through the window")
w = MainWindow()
w.resize(1400, 900)
w.show()
w._open_video(SIDE)
for _ in range(300):
    pump(0.05)
    if w.state == READY and w.project is not None and not w._loading:
        break
pump(0.3)
check(w._add_view(UPSIDE) and w._add_view(LEVEL), "three cameras: on its side, upside down, level")
pump(0.3)
p = w.project
w._apply_state()
small = lens.LensProfile(320, 240, np.array([[300.0, 0, 163.5], [0, 302.0, 117.0], [0, 0, 1]]),
                         np.array([-0.2, 0.04, 0.001, -0.002, 0.0]), False, 0.3, 20, "checkerboard (Kinetrace)",
                         {"verdict": "good", "verdict_reasons": ["clean"], "principal_px": [163.5, 117.0]},
                         rotation=0)
kl = small.save(os.path.join(OUT, "level_board.klens.json"))

w._set_active_view(0)
pump(0.3)
ASK["open"] = kl
n_warn = len(ASK["warn"])
w.act_load_lens.trigger()
pump(0.1)
got = p.lenses[0] if p.lenses else None
check(len(ASK["warn"]) == n_warn and got is not None and same(got, lens.turn_profile(small, 270))
      and (got.width, got.height) == (240, 320) and p.dirty,
      "camera filmed on its side: the level profile is attached TURNED (it used to be refused)",
      ASK["warn"][n_warn:])
# 270 clockwise = 90 counter-clockwise: pixel (x, y) of the stored 320 x 240 picture goes to (y, 319 - x)
check(got is not None and got.principal == (117.0, 320 - 1 - 163.5) and got.report.get("principal_px") == [117.0, 155.5],
      "its centre and report are in the turned picture's pixels", got.principal if got else None)

w._set_active_view(1)
pump(0.3)
w.act_load_lens.trigger()
pump(0.1)
got = p.lenses[1] if len(p.lenses) > 1 else None
check(got is not None and np.allclose(got.principal, (320 - 1 - 163.5, 240 - 1 - 117.0)),
      "upside-down camera: the centre is moved to the other side (it used to be attached unturned)",
      got.principal if got else None)

w._set_active_view(2)
pump(0.3)
w.act_load_lens.trigger()
pump(0.1)
got = p.lenses[2] if len(p.lenses) > 2 else None
check(got is not None and same(got, small), "level camera: the profile as it is")

argus = os.path.join(OUT, "argus_profile.txt")
with open(argus, "w") as fh:
    fh.write("1 300 320 240 163.5 117 1 -0.2 0.04 0.001 -0.002 0\n")
w._set_active_view(0)
pump(0.3)
before = p.lenses[0]
ASK["open"] = argus
n_asked = len(ASK["asked"])
w.act_load_lens.trigger()
pump(0.1)
seen = _lens_attach.STATE["seen"][-1]
check(p.lenses[0] is before and "not known" in seen["rows"][0][0] and not seen["rows"][0][1],
      "a profile that does not record its turn cannot be ticked for the turned camera, in words", seen)
check(len(ASK["asked"]) == n_asked, "an Argus file (one line per camera) is not offered a checkerboard video (G184)",
      ASK["asked"][n_asked:])

# the wand wizard's "Use for all": the level camera's profile reaches the turned cameras, turned
p.lenses = [None, None, small]
fill, replace = share_targets(p, 2)
check(sorted(fill) == [0, 1] and replace == [], "Use for all: the same camera filmed turned is a target", (fill, replace))

# ------------------------------------------------------------------ 6. the offer, through the user's clicks
print("\n[6] the offer to read the turn from the checkerboard video (G184)")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from kinetrace import calibwizard as cw  # noqa: E402

OLD_FILE = OLD.save(os.path.join(OUT, "old_board.klens.json"))
WANT_SIDE = lens.turn_profile(OLD, 270)

# (a) the wand wizard's Cameras page: the camera filmed on its side, "Load file..."
p.lenses = [None, None, None]
wiz = cw.WandWizard(w, p, OUT)
wiz.show()
page = wiz.page_cams          # (G223) its lens rows change the wizard's pending copy, wiz.lenses
page._refresh_lens_rows()
pump(0.1)
ASK.update(open=[OLD_FILE, LEVEL], question=QMessageBox.Yes)
n_asked, n_warn = len(ASK["asked"]), len(ASK["warn"])
QTest.mouseClick(page.lens_buttons[0][1], Qt.LeftButton)
pump(0.3)
got = wiz.lenses[0] if wiz.lenses else None
check(len(ASK["asked"]) == n_asked + 1 and "checkerboard video" in "".join(ASK["asked"][n_asked:]),
      "wand wizard: an old profile on the camera filmed on its side -> asked to find its checkerboard video",
      ASK["asked"][n_asked:])
check(ident(got, WANT_SIDE) and got is not None and got.rotation == 270 and len(ASK["warn"]) == n_warn,
      "...Yes + the level board video: attached, turned exactly (bit for bit turn_profile)", ASK["warn"][n_warn:])
check(page.lens_labels and "turned" in page.lens_labels[0].text() and "level.mp4" in page.lens_labels[0].text(),
      "...and its row says it was turned and names the checkerboard video the turn came from",
      page.lens_labels[0].text() if page.lens_labels else "")
wiz.lenses[:] = [None, None, None]
page._refresh_lens_rows()
ASK.update(open=[OLD_FILE], question=QMessageBox.No)
QTest.mouseClick(page.lens_buttons[0][1], Qt.LeftButton)
pump(0.2)
check(wiz.lenses[0] is None and "not known" in page.lens_labels[0].text(),
      "...No: not attached, the old refusal on the row", page.lens_labels[0].text())
ASK.update(open=[OLD_FILE, BOARD_LEVEL], question=QMessageBox.Yes)
n_warn = len(ASK["warn"])
QTest.mouseClick(page.lens_buttons[0][1], Qt.LeftButton)
pump(0.3)
check(wiz.lenses[0] is None and any("640 x 480" in t for t in ASK["warn"][n_warn:]),
      "...a board video of another size: refused in words, nothing attached", ASK["warn"][n_warn:])
# a lens file from another program (OpenCV .yml, same numbers): it may have been fitted on the STORED
# pictures, so the turn Kinetrace's decoder gives a board video is not known to be its: never offered
from kinetrace import calibio  # noqa: E402

OCV_FILE = os.path.join(OUT, "old_board_opencv.yml")
calibio.write_lens(OLD, OCV_FILE)
wiz.lenses[:] = [None, None, None]
page._refresh_lens_rows()
ASK.update(open=[OCV_FILE, LEVEL], question=QMessageBox.Yes)
n_asked = len(ASK["asked"])
QTest.mouseClick(page.lens_buttons[0][1], Qt.LeftButton)
pump(0.2)
ASK["open"] = ""
check(len(ASK["asked"]) == n_asked and wiz.lenses[0] is None and "not known" in page.lens_labels[0].text(),
      "an OpenCV lens file from another program is not offered a checkerboard video: refused as before",
      (ASK["asked"][n_asked:], page.lens_labels[0].text()))
wiz.reject()
wiz.deleteLater()
pump(0.2)

# (b) 3D -> Load a Lens Profile for Cameras... (three cameras)
p.lenses = [None, None, None]
w._set_active_view(0)
pump(0.3)
ASK.update(open=[OLD_FILE, LEVEL], question=QMessageBox.Yes)
n_asked = len(ASK["asked"])
_lens_attach.STATE["do"] = lambda dlg: _lens_attach.tick(dlg, 2)
w.act_load_lens.trigger()
pump(0.3)
_lens_attach.STATE["do"] = None
seen = _lens_attach.STATE["seen"][-1]
check(len(ASK["asked"]) == n_asked + 1 and seen["rows"][0][1] and "turned" in seen["rows"][0][0],
      "Load a Lens Profile for Cameras: asked, then the camera on its side can be ticked, 'fits, turned'", seen)
check(ident(p.lenses[0], WANT_SIDE) and ident(p.lenses[2], OLD) and p.lenses[2].rotation == 0 and p.lenses[1] is None,
      "...it gets the exact turn, the level camera the profile's own numbers, the unticked camera nothing",
      [None if x is None else (x.width, x.height, x.rotation) for x in p.lenses])
# with its turn known the profile follows I269 like one made today: the upside-down camera (tag 180) is offered
# it turned 180 (before, the unknown turn attached it unturned, centre on the wrong side)
check(seen["rows"][1][1] and "turned" in seen["rows"][1][0],
      "...and the upside-down camera's row reads 'fits, turned' (I269's rule, now that the turn is known)", seen)
OCV_N = len(ASK["asked"])
ASK.update(open=[OCV_FILE], question=QMessageBox.Yes)
w.act_load_lens.trigger()
pump(0.3)
check(len(ASK["asked"]) == OCV_N and "not known" in _lens_attach.STATE["seen"][-1]["rows"][0][0],
      "Load a Lens Profile for Cameras: an OpenCV file is not offered either", _lens_attach.STATE["seen"][-1])

w.project.dirty = False
w.close()
pump(0.3)
w._dev_probe.wait(10000)

# (d) one camera: 3D -> Load a Lens Profile for This Camera...
w2 = MainWindow()
w2.resize(1200, 800)
w2.show()
w2._open_video(SIDE)
for _ in range(300):
    pump(0.05)
    if w2.state == READY and w2.project is not None and not w2._loading:
        break
pump(0.3)
ASK.update(open=[OLD_FILE, LEVEL], question=QMessageBox.Yes)
n_asked, n_warn = len(ASK["asked"]), len(ASK["warn"])
w2.act_load_lens.trigger()
pump(0.3)
got = w2.project.lenses[0] if w2.project.lenses else None
check(len(ASK["asked"]) == n_asked + 1 and ident(got, WANT_SIDE) and len(ASK["warn"]) == n_warn,
      "one camera: asked, then attached with the exact turn", (ASK["asked"][n_asked:], ASK["warn"][n_warn:]))
w2.project.dirty = False
w2.close()
pump(0.3)
w2._dev_probe.wait(10000)
print("\nverify_lens_turn: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
