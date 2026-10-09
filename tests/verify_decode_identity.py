"""(I266) Does THIS computer's OpenCV / FFmpeg decode the same picture at each frame number as the
Windows build the expectations were made with? Run by the install check on Ubuntu, macOS and
Windows (.github/workflows/install-check.yml); no GPU, no Qt.

Each OS's opencv-python wheel ships its own FFmpeg build. A handful of small clips that stress
what differs between builds -- B-frames behind an EDIT LIST (phone and action-camera files), a
stream that starts late (an empty edit), a timestamp that skips a frame (a dropped frame) -- are
generated here with imageio-ffmpeg from deterministic pictures and fixed settings, then decoded with
cv2 exactly as the app does:

  * the number of frames that decode must be the Windows number;
  * every frame reached by the app's own reader (`VideoSource` with the seek plan `check_seeks`
    found here) must be the frame reading forward from the start gives;
  * the fingerprint made on Windows (tests/data/decode_identity/<clip>.npz) compared with this
    computer's frames (`fingerprint.compare`) must say "same" -- a shift here is exactly what a
    project moved to this OS would show its user.

The encoder of another OS may write other bytes (another x264 build): the clip's SHA-256 against the
Windows one is printed for information; the comparison is of pictures, tolerant to compression and
colour-conversion differences, not of bytes.

  .venv\\Scripts\\python.exe tests\\verify_decode_identity.py            check
  .venv\\Scripts\\python.exe tests\\verify_decode_identity.py --write    (Windows) write the expectations
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "out", "decode_identity")
DATA = os.path.join(HERE, "data", "decode_identity")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402

from kinetrace import fingerprint as fpm  # noqa: E402
from kinetrace import video_source as vs  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
WRITE = "--write" in sys.argv
FAILS = []
W, H, N, FPS = 320, 240, 96, 30
K, SIDE = 6, 64                 # a small fingerprint: the expectations live in the repository


def check(ok, what, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + what + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(what)


def picture(i: int) -> np.ndarray:
    """Deterministic on every OS: integer arithmetic only (no RNG, no float rounding that could
    differ): a smooth gradient, a checkered square moving 3 px per frame, a bar moving down."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.empty((H, W, 3), np.uint8)
    img[..., 0] = (x * 255 // (W - 1)).astype(np.uint8)
    img[..., 1] = (y * 255 // (H - 1)).astype(np.uint8)
    img[..., 2] = ((x + y) * 255 // (W + H - 2)).astype(np.uint8)
    sx, sy = 10 + (3 * i) % (W - 70), 40 + (i % 40)
    sq = (((np.arange(48)[:, None] // 6) + (np.arange(48)[None, :] // 6)) % 2 * 200 + 30).astype(np.uint8)
    img[sy:sy + 48, sx:sx + 48] = sq[..., None]
    by = (5 * i) % (H - 12)
    img[by:by + 12, W - 60:W - 20] = (250, 250, 40)
    return img


def run(args):
    subprocess.run([FF, "-y", "-loglevel", "error", *args], check=True)


def encode(path, n, codec_args, extra=()):
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-threads", "1", *codec_args, "-pix_fmt", "yuv420p",
                          "-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1", *extra, path],
                         stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(picture(i).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg could not write {os.path.basename(path)}")
    return path


X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-g", "24", "-bf", "3", "-x264-params", "threads=1"]


def clips() -> dict:
    """{name: path} of the clips, made here."""
    out = {}
    base = encode(os.path.join(OUT, "base.mp4"), N + 2, X264)
    out["bframes"] = base
    edit = os.path.join(OUT, "editlist.mp4")                 # the stream from frame 2 on: an edit list hides 2
    run(["-ss", f"{2 / FPS:.6f}", "-i", base, "-c", "copy", "-fflags", "+bitexact", "-map_metadata", "-1", edit])
    out["editlist"] = edit
    late = os.path.join(OUT, "late_start.mp4")               # starts 0.5 s late: an empty edit in front
    run(["-itsoffset", "0.5", "-i", base, "-c", "copy", "-fflags", "+bitexact", "-map_metadata", "-1", late])
    out["late_start"] = late
    out["dropped_frame"] = encode(os.path.join(OUT, "dropped_frame.mp4"), N, X264,
                                  ["-vf", "setpts=(N+gte(N\\,50))/(30*TB)", "-fps_mode", "passthrough",
                                   "-video_track_timescale", "30000"])
    return out


def sequential(path) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def sha(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


ident = fpm.decoder_identity()
print("this computer decodes with:", fpm.describe_decoder(ident))
made = clips()
os.makedirs(DATA, exist_ok=True)
expect_path = os.path.join(DATA, "expected.json")
expected = {}
if os.path.isfile(expect_path):
    with open(expect_path, encoding="utf-8") as fh:
        expected = json.load(fh)
if WRITE:
    expected = {"decoder": ident, "clips": {}}
for name, path in made.items():
    print(f"[{name}]")
    seq = sequential(path)
    n = len(seq)
    plan = vs.check_seeks(path, n)
    print(f"    {n} frames; seeks {'exact' if plan.exact else 'NOT exact: ' + str(plan.wrong)}; timestamp gaps "
          f"{plan.gaps}; sha256 {sha(path)}")
    vs.set_seek_plan(path, plan)
    src = vs.VideoSource(path)
    bad = []
    for t in sorted({1, 2, 3, n // 3, n // 2, n // 2 + 1, n - 2, n - 1, 25, 49, 50, 51, 70}):
        if not 0 <= t < n:
            continue
        src.cache.clear()
        rgb = src.get_frame(t)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb is not None else None
        if g is None or float(np.abs(g.astype(np.float32) - seq[t]).mean()) > 0.5:
            bad.append(t)
    src.close()
    vs.set_seek_plan(path, None)
    check(not bad, f"{name}: every frame the app reaches is the one reading forward gives", f"wrong at {bad}")
    store = os.path.join(DATA, f"{name}.npz")
    if WRITE:
        fp = fpm.make(path, n, k=K, crop=SIDE)
        arrays = fp.to_arrays()
        np.savez_compressed(store, meta=np.frombuffer(arrays["meta"].encode("utf-8"), np.uint8),
                            crops=arrays["crops"], thumbs=np.zeros((len(fp.marks), 1, 1), np.uint8))
        expected["clips"][name] = {"n_frames": n, "sha256_16": sha(path), "seeks_exact": bool(plan.exact),
                                   "gaps": int(plan.gaps)}
        print(f"    expectation written: {len(fp.marks)} moments at {[m.frame for m in fp.marks]}")
        continue
    want = (expected.get("clips") or {}).get(name)
    if want is None or not os.path.isfile(store):
        check(False, f"{name}: an expectation from Windows exists", store)
        continue
    print(f"    Windows: {want['n_frames']} frames, sha256 {want['sha256_16']}"
          + (" (the same bytes)" if want["sha256_16"] == sha(path) else " (another encoder build: other bytes)"))
    check(n == want["n_frames"], f"{name}: the same number of frames decode as on Windows",
          f"{n} here, {want['n_frames']} on Windows")
    z = np.load(store)
    fp = fpm.VideoFingerprint.from_arrays({"meta": bytes(z["meta"]).decode("utf-8"), "crops": z["crops"]})
    r = fpm.compare(fp, path, n, name)
    print("    " + r.sentence)
    check(r.verdict == "same", f"{name}: each frame number shows the picture it shows on Windows",
          f"{r.verdict} {r.shift}: {r.detail}")
if WRITE:
    with open(expect_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(expected, fh, indent=1)
        fh.write("\n")
    print(f"\nexpectations written to {os.path.relpath(DATA, ROOT)}")
print("\nverify_decode_identity: " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
