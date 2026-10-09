"""(I266) A video that decodes differently on another computer is found, measured and shown --
never silently accepted, never silently fixed. No Qt, no GPU.

  [1] the fingerprint: the busiest moments, spread over the video, full-resolution crops, the
      decoder and the file; it round-trips through its arrays; a damaged one is refused in words
  [2] a video whose start is shifted by an EDIT LIST (the stream copied from one frame later, as
      phone files do) against the fingerprint of the unshifted one -> "shifted", k = -1; one with
      a frame put in front -> k = +1; the same file -> "same"; another video -> "different"
  [3] a high-speed clip whose subject moves ~1 px per frame: a 1-frame shift is still found
      through the full-resolution crops
  [4] a still clip -> "cannot tell", with how far a shift could move a track (px) and a verdict
  [5] a capture whose seeks land one frame late (a wrapper) still gives exact frames once
      `check_seeks` has looked at it; a file whose seeks are exact keeps the plain seek (no file
      gets slower: the same number of seeks and decodes as before)
  [6] a real file whose timestamps skip a frame (a camera that dropped one): on this computer its
      seeks land one frame early after the gap -- the old VideoSource showed the wrong pictures;
      with the plan every frame is the one reading forward gives

.venv\\Scripts\\python.exe tests\\verify_fingerprint.py
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "fingerprint")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402

from kinetrace import fingerprint as fpm  # noqa: E402
from kinetrace import video_source as vs  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
FAILS = []


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


# ------------------------------------------------------------------ synthetic footage
W, H = 640, 360
_rng = np.random.RandomState(11)
_BG = cv2.GaussianBlur(_rng.randint(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), 2.0)
_TEX = cv2.GaussianBlur(_rng.randint(0, 255, (120, 120, 3)).astype(np.uint8), (0, 0), 1.5)
_YY, _XX = np.mgrid[0:120, 0:120]
_DISC = ((_XX - 59.5) ** 2 + (_YY - 59.5) ** 2) <= 55 ** 2


def picture(i: float, speed: float = 6.0, still: bool = False, wobble: float = 40.0) -> np.ndarray:
    """A textured disc crossing a textured background: `speed` px per frame across, up to
    `wobble` px up and down (sub-pixel exact)."""
    img = _BG.copy()
    if not still:
        x = 40.0 + (speed * i) % (W - 200)
        y = 120.0 + wobble * np.sin(i * 0.05)
        M = np.float32([[1, 0, x], [0, 1, y]])
        disc = cv2.warpAffine(_TEX, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(_DISC.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        img = (img * (1 - mask) + disc * mask).astype(np.uint8)
    return img


def encode(path: str, frames, fps: float = 30.0, extra=()):
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                          "-g", "30", "-bf", "2", "-pix_fmt", "yuv420p", *extra, path], stdin=subprocess.PIPE)
    for img in frames:
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed for " + path)
    return path


def count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    k = 0
    while cap.grab():
        k += 1
    cap.release()
    return k


N = 240
A = encode(os.path.join(OUT, "a.mp4"), (picture(i) for i in range(N)))
# an edit list: the same kind of encode one frame longer, its stream copied from the second frame on
# (what phone and action-camera files carry): the picture of frame N+1 is shown as frame N
_long = encode(os.path.join(OUT, "a_plus1.mp4"), (picture(i) for i in range(N + 1)))
B_EDIT = os.path.join(OUT, "b_editlist.mp4")
subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{1 / 30:.6f}", "-i", _long, "-c", "copy", B_EDIT], check=True)
B_LATE = encode(os.path.join(OUT, "b_late.mp4"), (picture(i - 1) for i in range(N)))
OTHER = encode(os.path.join(OUT, "other.mp4"), (picture(i + 97, speed=-4.0) for i in range(N)))

print("[1] the fingerprint")
t0 = time.perf_counter()
fp = fpm.make(A, N)
dt = time.perf_counter() - t0
check(fp is not None and len(fp.marks) == fpm.N_MARKS, f"{fpm.N_MARKS} moments kept", f"{fp and len(fp.marks)}")
frames = [m.frame for m in fp.marks]
check(frames == sorted(frames) and all(fpm.NEAR <= f <= N - 1 - fpm.NEAR for f in frames),
      "in frame order, each with two frames on both sides", str(frames))
parts = [(f * fpm.N_MARKS) // N for f in frames]
check(parts == list(range(fpm.N_MARKS)), "one moment per eighth of the video (spread over it)", str(parts))
check(fp.crops.shape == (fpm.N_MARKS, fpm.CROP, fpm.CROP, 3) and fp.crops.dtype == np.uint8,
      "full-resolution colour crops of 160 px", str(fp.crops.shape))
# the crop sits where the disc moves (not on the still background)
cap = cv2.VideoCapture(A)
inside = 0
for m in fp.marks:
    x = 40.0 + (6.0 * m.frame) % (W - 200) + 60
    y = 120.0 + 40.0 * np.sin(m.frame * 0.05) + 60
    bx, by, s = m.box
    inside += (bx - 30 <= x <= bx + s + 30) and (by - 30 <= y <= by + s + 30)
cap.release()
check(inside == len(fp.marks), "every crop is on the moving disc", f"{inside} of {len(fp.marks)}")
check(all(m.sep > 0.01 for m in fp.marks), "each crop differs clearly from its neighbours",
      str([round(m.sep, 4) for m in fp.marks]))
dec = fp.decoder
check(dec.get("opencv") == cv2.__version__ and dec.get("os") and dec.get("avcodec") is not None,
      "the decoder is recorded (OS, OpenCV, FFmpeg)", str(dec))
check(fp.file.get("size") == os.path.getsize(A) and fp.file.get("name") == "a.mp4", "the file is recorded", str(fp.file))
back = fpm.VideoFingerprint.from_arrays(fp.to_arrays())
check(back.same(fp), "it round-trips through its arrays")
check(not fpm.needs_check(fp, A), "same computer, same file: no check needed")
bad = fp.to_arrays()
bad["crops"] = bad["crops"][:3]
try:
    fpm.VideoFingerprint.from_arrays(bad)
    check(False, "a damaged fingerprint is refused")
except ValueError as e:
    check("crops" in str(e), "a damaged fingerprint is refused in words", str(e))
print(f"    made in {dt:.2f} s for {N} frames of {W}x{H}")
check(count(B_EDIT) == N, "the edit-list file decodes the same number of frames (a count check sees nothing)",
      str(count(B_EDIT)))

print("[2] shifted starts, the same file, another video")
r = fpm.compare(fp, A, N, "cam1")
check(r.verdict == "same" and r.shift == 0 and r.quality == "good", "the same file: same, good",
      f"{r.verdict} {r.shift} {r.detail}")
r = fpm.compare(fp, B_EDIT, count(B_EDIT), "cam1")
check(r.verdict == "shifted" and r.shift == -1, "the edit-list file: shifted by -1 (one frame earlier)",
      f"{r.verdict} {r.shift} {r.detail}")
check("1 frame earlier" in r.sentence and "Nothing was changed" in r.sentence, "the sentence says it in words",
      r.sentence)
check(r.needs_eyes, "a shift asks for the evidence dialog")
r = fpm.compare(fp, B_LATE, N, "cam1")
check(r.verdict == "shifted" and r.shift == 1 and "1 frame later" in r.sentence, "a frame in front: shifted by +1",
      f"{r.verdict} {r.shift} {r.sentence}")
r = fpm.compare(fp, OTHER, N, "cam1")
check(r.verdict == "different", "another video: different", f"{r.verdict} {r.detail}")
fp_late = fpm.VideoFingerprint.from_arrays(fp.to_arrays())
fp_late.decoder = dict(fp_late.decoder, os="Darwin", avcodec="60.31.102")
check(fpm.needs_check(fp_late, A), "another decoder: the check is due")
fp_late.decoder = dict(fp.decoder)
fp_late.file = dict(fp.file, mtime_ns=1)
check(fpm.needs_check(fp_late, A), "another copy of the file: the check is due")
check("Windows" in fpm.describe_decoder(dec) or "Linux" in fpm.describe_decoder(dec)
      or "macOS" in fpm.describe_decoder(dec), "the decoder in words", fpm.describe_decoder(dec))

print("[3] high frame rate, ~1 px per frame")
NH = 300
HI = encode(os.path.join(OUT, "hi.mp4"), (picture(i, speed=1.0, wobble=0.0) for i in range(NH + 1)), fps=240.0)
HI_EDIT = os.path.join(OUT, "hi_editlist.mp4")
subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{1 / 240:.6f}", "-i", HI, "-c", "copy", HI_EDIT], check=True)
HI_LATE = encode(os.path.join(OUT, "hi_late.mp4"), (picture(i - 1, speed=1.0, wobble=0.0) for i in range(NH + 1)),
                 fps=240.0)
fph = fpm.make(HI, NH + 1)
print("    moves (px/frame):", [round(m.move_px, 2) for m in fph.marks], " sep:", [round(m.sep, 4) for m in fph.marks])
check(0.6 <= fph.move_px <= 1.5, "the measured motion is about 1 px per frame", f"{fph.move_px:.2f}")
r = fpm.compare(fph, HI_EDIT, count(HI_EDIT), "cam2")
check(r.verdict == "shifted" and r.shift == -1 and r.marks and all(k.verdict == "clear" for k in r.marks),
      "the same file read one frame later (edit list): every moment clearly -1", f"{r.verdict} {r.detail}")
r = fpm.compare(fph, HI_LATE, NH + 1, "cam2")
check(r.verdict == "shifted" and r.shift == 1, "even a re-encoded copy one frame late is found (+1): the moments "
      "agree although compression noise is as large as a 1-px move", f"{r.verdict} {r.detail}")
print("    " + r.sentence)
r = fpm.compare(fph, HI, NH + 1, "cam2")
check(r.verdict == "same" and r.quality == "good", "and the same file is the same", f"{r.verdict} {r.detail}")
# another OpenCV build rounds its colour conversion differently: +-1 grey level on ~half the pixels (Mac
# round 2), up to 3 levels in a few; the saved crops as such a build would have made them
_rr = np.random.RandomState(21)
other_build = fpm.VideoFingerprint.from_arrays(fph.to_arrays())
noise = _rr.choice([-1, 0, 0, 1], size=other_build.crops.shape) + _rr.choice([0] * 49 + [3], size=other_build.crops.shape)
other_build.crops = np.clip(other_build.crops.astype(np.int16) + noise, 0, 255).astype(np.uint8)
r = fpm.compare(other_build, HI, NH + 1, "cam2")
check(r.verdict == "same" and all(k.verdict == "clear" for k in r.marks),
      "another build's 1-level rounding: still SAME, every moment clear at 1 px/frame", f"{r.verdict} {r.detail}")
r = fpm.compare(other_build, HI_EDIT, count(HI_EDIT), "cam2")
check(r.verdict == "shifted" and r.shift == -1, "... and a 1-frame shift is still told apart from it",
      f"{r.verdict} {r.detail}")

print("[3b] a small mover far away (X36: under 1 % of the picture)")
WS, HS, NS = 1920, 1080, 90
_bgs = cv2.GaussianBlur(np.random.RandomState(8).randint(0, 255, (HS, WS, 3)).astype(np.uint8), (0, 0), 3.0)
_bug = cv2.GaussianBlur(np.random.RandomState(9).randint(0, 255, (24, 24, 3)).astype(np.uint8), (0, 0), 1.0)


def small(i):
    img = _bgs.copy()
    x, y = 300 + 3 * i, 700 - 2 * i
    img[y:y + 24, x:x + 24] = _bug
    return img


def encode_size(path, frames, w, h):
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "30",
                          "-bf", "2", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for img in frames:
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


SM = encode_size(os.path.join(OUT, "small.mp4"), (small(i) for i in range(NS + 1)), WS, HS)
SM_EDIT = os.path.join(OUT, "small_editlist.mp4")
subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{1 / 60:.6f}", "-i", SM, "-c", "copy", SM_EDIT], check=True)
from kinetrace import sync  # noqa: E402
sig = sync.motion_signal(SM, 0, 30)
print(f"    the sync motion measure (99th percentile, 160 px) sees at most {np.nanmax(sig):.2f} levels")
fps2 = fpm.make(SM, NS + 1)
on_bug = sum(abs(m.box[0] + 80 - (300 + 3 * m.frame + 12)) < 80 and abs(m.box[1] + 80 - (700 - 2 * m.frame + 12)) < 80
             for m in fps2.marks)
check(not fps2.still and on_bug == len(fps2.marks), "the fingerprint finds it: not 'still', every crop on the mover",
      f"still={fps2.still}, {on_bug} of {len(fps2.marks)}, sep {[round(m.sep, 4) for m in fps2.marks]}")
r = fpm.compare(fps2, SM_EDIT, NS, "cam4")
check(r.verdict == "shifted" and r.shift == -1, "and a one-frame shift of it is found", f"{r.verdict} {r.detail}")

print("[4] a still clip")
ST = encode(os.path.join(OUT, "still.mp4"), (picture(i, still=True) for i in range(90)))
fps_ = fpm.make(ST, 90)
check(fps_.still, "the fingerprint knows nothing moves")
r = fpm.compare(fps_, ST, 90, "cam3")
check(r.verdict == "cannot_tell" and r.quality == "good" and r.move_px < 0.5 and "px" in r.sentence,
      "cannot tell, with the bound in px and a verdict (good: harmless)", f"{r.verdict} {r.quality} {r.sentence}")
check(not r.needs_eyes, "a harmless 'cannot tell' does not interrupt with a dialog")


print("[5] a capture whose seeks land one frame late")
IDS = os.path.join(OUT, "ids.mp4")
encode(IDS, (picture(i) for i in range(150)), extra=("-g", "24"))


def ident(frames_ref, rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return int(np.argmin([np.abs(r - g).mean() for r in frames_ref]))


ref = []
cap = cv2.VideoCapture(IDS)
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    ref.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
cap.release()
keys, _ = vs.packet_times(IDS)[1], None
COUNT = {"set": 0, "grab": 0, "read": 0}


class LateSeeks:
    """A capture whose seeks to a frame that is not a keyframe land one frame late (an inexact
    CAP_PROP_POS_FRAMES); its timestamps stay honest, as FFmpeg's are."""

    def __init__(self, path, late=True):
        self.c = cv2.VideoCapture(path)
        self.late = late

    def set(self, prop, val):
        COUNT["set"] += 1
        if prop == cv2.CAP_PROP_POS_FRAMES and self.late and int(val) not in keys and int(val) > 0:
            val = int(val) + 1
        return self.c.set(prop, val)

    def grab(self):
        COUNT["grab"] += 1
        return self.c.grab()

    def read(self):
        COUNT["read"] += 1
        return self.c.read()

    def __getattr__(self, k):
        return getattr(self.c, k)


real_open = vs.open_capture
vs.open_capture = lambda p: LateSeeks(p, late=True)
try:
    src = vs.VideoSource(IDS)
    wrong_old = sum(ident(ref, src.get_frame(t)) != t for t in (40, 77, 101, 131) if not src.cache.clear())
    src.close()
    check(wrong_old > 0, "without the check, such a capture shows wrong frames (the old behaviour)", str(wrong_old))
    plan = vs.check_seeks(IDS, len(ref))
    check(not plan.exact and plan.wrong and plan.pts_map is not None, "the check finds the late seeks",
          f"exact={plan.exact} wrong={plan.wrong}")
    check("re-encode" in plan.note and "every frame is exact" in plan.note, "and says so in words", plan.note)
    vs.set_seek_plan(IDS, plan)
    src = vs.VideoSource(IDS)
    rng = np.random.RandomState(5)
    targets = [int(t) for t in rng.randint(0, len(ref), 40)] + [0, 1, 23, 24, 25, len(ref) - 1]
    bad = []
    for t in targets:
        src.cache.clear()
        g = ident(ref, src.get_frame(t))
        if g != t:
            bad.append((t, g))
    src.close()
    check(not bad, f"with the plan, all {len(targets)} random jumps give the exact frame", str(bad))
    rd = vs.FrameReader(IDS, 61)
    got = [ident(ref, cv2.cvtColor(rd.read(f), cv2.COLOR_BGR2RGB)) for f in (61, 70, 99)]
    rd.release()
    check(got == [61, 70, 99], "FrameReader (exports, pose runs) too", str(got))
    fpi = fpm.make(IDS, len(ref), seeks="read forward")
    caps = cv2.VideoCapture(IDS)
    ok_marks = 0
    for i, m in enumerate(fpi.marks):
        caps.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(m.frame):
            caps.grab()
        _, bgr = caps.read()
        ok_marks += fpm.dist(fpi.crops[i], cv2.cvtColor(fpm._crop(bgr, m.box), cv2.COLOR_BGR2RGB)) < 1e-6
    caps.release()
    check(ok_marks == len(fpi.marks), "the fingerprint's crops are the exact frames (read from the start)",
          f"{ok_marks} of {len(fpi.marks)}")
    vs.set_seek_plan(IDS, None)
    vs.open_capture = lambda p: LateSeeks(p, late=False)
    plan = vs.check_seeks(IDS, len(ref))
    check(plan.exact, "an exact capture: the plain seek stays", str(plan.wrong))
    vs.set_seek_plan(IDS, plan)
    for k in COUNT:
        COUNT[k] = 0
    src = vs.VideoSource(IDS)
    for t in (100, 20, 140, 60):
        src.cache.clear()
        src.get_frame(t)
    src.close()
    with_plan = dict(COUNT)
    vs.set_seek_plan(IDS, None)
    for k in COUNT:
        COUNT[k] = 0
    src = vs.VideoSource(IDS)
    for t in (100, 20, 140, 60):
        src.cache.clear()
        src.get_frame(t)
    src.close()
    check(with_plan == COUNT, "a file with exact seeks is not slower (the same seeks and decodes)",
          f"{with_plan} vs {COUNT}")
finally:
    vs.open_capture = real_open
    vs.set_seek_plan(IDS, None)

print("[6] a file whose timestamps skip a frame (a dropped frame)")
GAP = os.path.join(OUT, "gap.mp4")
encode(GAP, (picture(i) for i in range(200)),
       extra=("-vf", "setpts=(N+gte(N\\,100))/(30*TB)", "-fps_mode", "passthrough", "-video_track_timescale", "30000"))
ref = []
cap = cv2.VideoCapture(GAP)
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    ref.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
cap.release()
src = vs.VideoSource(GAP)
old = []
for t in (60, 130, 150, 190):
    src.cache.clear()
    g = ident(ref, src.get_frame(t))
    if g != t:
        old.append((t, g))
src.close()
print("    the plain seek on this computer:", old or "exact")
plan = vs.check_seeks(GAP, len(ref))
check(plan.gaps == 1, "the gap in the timestamps is seen", str(plan.gaps))
if old:
    check(not plan.exact and "skip 1 frame" in plan.note, "the check finds the wrong jumps and names the cause",
          plan.note)
else:
    check(plan.exact, "this computer's seeks happen to be exact on it: the plain seek stays")
vs.set_seek_plan(GAP, plan)
src = vs.VideoSource(GAP)
bad = []
for t in list(range(90, 115)) + [130, 150, 190, len(ref) - 1, 5]:
    src.cache.clear()
    g = ident(ref, src.get_frame(t))
    if g != t:
        bad.append((t, g))
src.close()
vs.set_seek_plan(GAP, None)
check(not bad, "every frame is the one reading forward gives", str(bad))

print("\nverify_fingerprint: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
