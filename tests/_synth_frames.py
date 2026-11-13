"""Clips whose every picture carries its own number, for the frame-exactness suites.

`picture(i)` draws frame number i as 16 blocks (bright = 1) with the inverted code below, over a
moving bar and square; `barcode(bgr)` reads it back from a decoded picture. `make_clips` writes the
clip kinds with the environment's own ffmpeg (imageio-ffmpeg): even timestamps, a camera that
dropped frames (timestamps that skip), a variable rate, an edit list, Matroska.
"""
import os
import subprocess

import cv2
import numpy as np

BITS = 16
_BG = {}


def picture(i: int, W: int = 320, H: int = 240) -> np.ndarray:
    bg = _BG.get((W, H))
    if bg is None:
        y, x = np.mgrid[0:H, 0:W]
        bg = np.empty((H, W, 3), np.uint8)
        bg[..., 0] = (x * 255 // (W - 1)).astype(np.uint8)
        bg[..., 1] = (y * 255 // (H - 1)).astype(np.uint8)
        bg[..., 2] = ((x + y) * 255 // (W + H - 2)).astype(np.uint8)
        _BG[(W, H)] = bg
    img = bg.copy()
    bh, bw = max(8, H // 8), W // BITS
    for b in range(BITS):
        bit = (i >> (BITS - 1 - b)) & 1
        x0 = b * bw
        img[0:bh, x0:x0 + bw] = 230 if bit else 25
        img[bh:2 * bh, x0:x0 + bw] = 25 if bit else 230
    barw = max(4, W // 20)
    x = (i * 7 * W // 320) % (W - barw)
    img[2 * bh:H, x:x + barw] = (40, 200, 250)
    s = max(16, H // 5)
    s -= s % 8
    sx = (i * 3 * W // 320) % (W - s)
    sy = 2 * bh + (i * 2) % max(1, H - 2 * bh - s)
    c = s // 8
    sq = (((np.arange(s)[:, None] // c) + (np.arange(s)[None, :] // c)) % 2 * 200 + 30).astype(np.uint8)
    img[sy:sy + s, sx:sx + s] = sq[..., None]
    return img


def barcode(bgr) -> int | None:
    """The frame number drawn by `picture`, or None (not one of its pictures)."""
    if bgr is None or getattr(bgr, "ndim", 0) != 3:
        return None
    H, W = bgr.shape[:2]
    bh, bw = max(8, H // 8), W // BITS
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    v = 0
    for b in range(BITS):
        xs = slice(b * bw + bw // 4, b * bw + (3 * bw) // 4)
        top = float(g[bh // 4:(3 * bh) // 4, xs].mean())
        low = float(g[bh + bh // 4:bh + (3 * bh) // 4, xs].mean())
        if abs(top - low) < 60:
            return None
        v = (v << 1) | int(top > low)
    return v


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"]


def _x264(g: int, bf: int) -> list:
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", str(g), "-bf", str(bf),
            "-threads", "1", "-x264-params", "threads=1:scenecut=0"]


def _encode(path: str, n: int, codec: list, extra=()) -> None:
    p = subprocess.Popen([_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", "320x240", "-r", "30", "-i", "-", *codec, "-pix_fmt", "yuv420p",
                          *BITEXACT, *extra, path], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(picture(i).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg could not write {os.path.basename(path)}")


def _remux(src: str, dst: str, pre=()) -> None:
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", *pre, "-i", src, "-c", "copy", *BITEXACT, dst], check=True)


def _gap(expr: str) -> list:
    return ["-vf", f"setpts={expr}/(30*TB)", "-fps_mode", "passthrough", "-video_track_timescale", "30000"]


# name -> (file, uneven timestamps?, what it is)
KINDS = {
    "plain": ("plain.mp4", False, "H.264, keyframe every 12, no B-frames"),
    "bframes": ("bframes.mp4", False, "H.264, 3 B-frames, keyframe every 24"),
    "editlist": ("editlist.mp4", False, "B-frames behind an edit list that hides 2 pictures"),
    "dropped_early": ("dropped_early.mp4", True, "timestamps skip 1 frame at frame 50"),
    "dropped_late": ("dropped_late.mp4", True, "timestamps skip 3 frames at frame 400 of 600"),
    "dropped_late_nobf": ("dropped_late_nobf.mp4", True, "timestamps skip 1 frame at frame 400, no B-frames"),
    "mkv_dropped": ("mkv_dropped.mkv", True, "the dropped_late stream in Matroska"),
    "vfr": ("vfr.mp4", True, "variable rate: 30 fps for 150 frames, then 60 fps"),
}


def make_clips(folder: str, names=None) -> dict:
    """{name: path}; a clip already there is kept (the makers are deterministic)."""
    os.makedirs(folder, exist_ok=True)
    P = {k: os.path.join(folder, v[0]) for k, v in KINDS.items()}
    makers = {
        "plain": lambda: _encode(P["plain"], 300, _x264(12, 0)),
        "bframes": lambda: _encode(P["bframes"], 302, _x264(24, 3)),
        "editlist": lambda: _remux(P["bframes"], P["editlist"], pre=["-ss", f"{2 / 30:.6f}"]),
        "dropped_early": lambda: _encode(P["dropped_early"], 300, _x264(24, 3), _gap("(N+gte(N\\,50))")),
        "dropped_late": lambda: _encode(P["dropped_late"], 600, _x264(24, 3), _gap("(N+3*gte(N\\,400))")),
        "dropped_late_nobf": lambda: _encode(P["dropped_late_nobf"], 600, _x264(30, 0), _gap("(N+gte(N\\,400))")),
        "mkv_dropped": lambda: _remux(P["dropped_late"], P["mkv_dropped"]),
        # a time base of 1/600 so the 60 fps half-steps are whole ticks
        "vfr": lambda: _encode(P["vfr"], 300, _x264(24, 3),
                               ["-vf", "settb=1/600,setpts='if(lt(N,150),N,150+(N-150)/2)/(30*TB)'",
                                "-fps_mode", "passthrough", "-video_track_timescale", "600"]),
    }
    order = [k for k in KINDS if names is None or k in names]
    for dep, need in (("editlist", "bframes"), ("mkv_dropped", "dropped_late")):
        if dep in order and need not in order:
            order.insert(0, need)
    for k in order:
        if not os.path.exists(P[k]) or os.path.getsize(P[k]) == 0:
            makers[k]()
    return {k: P[k] for k in order}


def forward(path: str) -> list:
    """The numbers drawn in the pictures of `path`, read from the start one after another: THE
    reference (frame N = the N-th picture reading forward)."""
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(barcode(bgr))
    cap.release()
    return out
