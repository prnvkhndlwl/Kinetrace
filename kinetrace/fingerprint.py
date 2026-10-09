"""Does this computer show a video's frames under the same numbers as the computer that made the
project? (I266)

A project stores every track, silhouette and event by FRAME NUMBER. Frame N is the N-th picture
the decoder gives, and the decoder is not the same everywhere: each operating system's
opencv-python ships its own FFmpeg build, and phone / action-camera files carry edit lists and
start offsets that FFmpeg versions have handled differently. A video that decodes one frame
"later" on the Mac than on the PC where it was tracked has the same frame count -- nothing
noticed it -- and every track sits one frame off there.

A FINGERPRINT, made where the project is made and saved with it (cameras/<cam>/fingerprint/):
the K moments of the video where the picture changes most from frame to frame, spread over the
video; for each, a full-resolution crop where that change is largest (a 1-pixel-per-frame motion
of a high-speed camera is invisible on a thumbnail, not in a full-resolution crop) and a small
whole-frame grey copy for the eye. With them the decoder (OS, OpenCV, FFmpeg's libraries) and the
file (size, time). The change is measured like `sync.motion_signal` (grey, downscaled, |frame -
neighbour|) but as the smaller of the change to the frame before and to the frame after (the
moment must differ from BOTH neighbours) and as the MEAN over the crop's window, not the 99th
percentile over the whole picture: that percentile does not see a mover smaller than ~1 % of the
picture (X36), and an animal far away is such a mover.

The CHECK, on a computer whose decoder or file differs: this computer's frames N-2 .. N+2 of each
saved moment are compared with the saved crop by normalised correlation (blind to the few-levels
YUV->RGB differences between builds), the best match must stand clearly above the others, and
the moments must agree: "same", "shifted by k frames", "mixed" (the shift changes along the video),
"different" (no saved picture is anywhere near: another take or cut), or "cannot tell" (too little
motion -- then a shift moves the tracks by at most that tiny per-frame motion, given in pixels).
Nothing is ever re-indexed: the app shows the evidence (`FrameCheckDialog`) and the user decides.

Every frame is read EXACTLY: the file's seeks are checked first (`video_source.check_seeks`) and
every read goes through `video_source.seek_capture`. No Qt here; the app runs it on a worker.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from kinetrace.sync import THUMB_W
from kinetrace.video_source import check_seeks, file_identity, open_capture, seek_capture, seek_plan, set_seek_plan

FORMAT_VERSION = 1
N_MARKS = 8              # moments kept per video
CROP = 160               # side of the full-resolution crop (px)
NEAR = 2                 # frames either side compared on another computer
WIDE = 12                # ... and when nothing matches there, this far
SCAN_BUDGET = 1600       # frames decoded at most to find the busiest moments (a longer video is sampled)
STRETCH = 48             # frames per sampled stretch of a long video
ANALYSIS_W = 320         # width of the grey copies the change is measured on (to rank moments, place crops)
MIN_SEP = 0.004          # 1 - correlation between a saved crop and its neighbours below which it cannot tell them apart
NO_MATCH = 0.20          # 1 - correlation above which a picture is not the saved one
CLEAR = 3.0              # the best match must be this many times closer than the next ...
LEAN = 1.2               # ... or at least this, in several moments that all agree (`_verdict`)


# ------------------------------------------------------------------ who decodes, what file
_IDENT: dict | None = None


def decoder_identity() -> dict:
    """What decides how this computer decodes a video: the OS, OpenCV and the FFmpeg libraries in
    its build, and the decode backend chosen (KINETRACE_DECODE)."""
    global _IDENT
    if _IDENT is None:
        info = cv2.getBuildInformation()

        def lib(name: str) -> str:
            m = re.search(rf"^\s*{name}:\s*YES\s*\(([^)]*)\)", info, re.M)
            return m.group(1).strip() if m else ""
        _IDENT = {"os": platform.system() or "unknown", "machine": platform.machine() or "",
                  "opencv": cv2.__version__, "avcodec": lib("avcodec"), "avformat": lib("avformat"),
                  "swscale": lib("swscale")}
    return dict(_IDENT, backend=os.environ.get("KINETRACE_DECODE", "").strip().lower() or "auto")


_DECODER_KEYS = ("os", "opencv", "avcodec", "avformat", "swscale", "backend")


def same_decoder(a: dict | None, b: dict | None) -> bool:
    return bool(a) and bool(b) and all(str(a.get(k, "")) == str(b.get(k, "")) for k in _DECODER_KEYS)


def same_file(a: dict | None, b: dict | None) -> bool:
    return bool(a) and bool(b) and a.get("size") == b.get("size") and a.get("mtime_ns") == b.get("mtime_ns")


def describe_decoder(d: dict | None) -> str:
    """'Windows, OpenCV 5.0.0, FFmpeg avcodec 61.19.100' (and the backend when it is not the default)."""
    if not d:
        return "an unknown computer"
    os_name = {"Darwin": "macOS"}.get(str(d.get("os")), str(d.get("os") or "unknown OS"))
    s = f"{os_name}, OpenCV {d.get('opencv', '?')}, FFmpeg avcodec {d.get('avcodec') or '?'}"
    if d.get("backend") not in (None, "", "auto"):
        s += f", decoder {d['backend']}"
    return s


# ------------------------------------------------------------------ the fingerprint
@dataclass
class Mark:
    """One saved moment: frame `frame`, its crop's place (x, y, side) in the full picture, how much
    the picture changes there (`motion`: mean grey levels over the crop's window, the smaller of the
    changes to the two neighbours), how far the crop is from its own neighbours (`sep`,
    1 - correlation; tiny = it cannot tell them apart) and how far its content moves from the frame
    before (`move_px`, pixels)."""
    frame: int
    box: tuple
    motion: float
    sep: float
    move_px: float
    picked: str = "auto"        # "auto" | "user" (a frame the user pointed at)


@dataclass
class VideoFingerprint:
    decoder: dict
    file: dict
    n_frames: int
    width: int
    height: int
    made: str
    marks: list = field(default_factory=list)          # [Mark], in frame order
    crops: np.ndarray | None = None                     # (K, S, S, 3) uint8 RGB, frame N of each mark
    thumbs: np.ndarray | None = None                    # (K, h, w) uint8 grey, the whole frame N
    seeks: str = "exact"                                # how frames were reached where it was made
    data_before: bool = False                           # the camera already had data when it was made

    @property
    def still(self) -> bool:
        """Nothing moves enough in any saved moment to tell neighbouring frames apart (judged on the
        full-resolution crops: a small mover barely moves the downscaled measure, X36)."""
        return not any(m.sep >= MIN_SEP for m in self.marks)

    @property
    def move_px(self) -> float:
        return max((m.move_px for m in self.marks), default=0.0)

    def to_arrays(self, prefix: str = "") -> dict:
        meta = {"format_version": FORMAT_VERSION, "decoder": self.decoder, "file": self.file,
                "n_frames": int(self.n_frames), "width": int(self.width), "height": int(self.height),
                "made": self.made, "seeks": self.seeks, "data_before": bool(self.data_before),
                "what": "the moments of this camera's video where the picture changes most, to check that "
                        "another computer decodes the same picture under the same frame number "
                        "(crops.npy: full-resolution crops of those frames, thumbs.npy: whole frames, grey)",
                "marks": [{"frame": int(m.frame), "box": [int(v) for v in m.box], "motion": round(float(m.motion), 3),
                           "sep": round(float(m.sep), 5), "move_px": round(float(m.move_px), 3), "picked": m.picked}
                          for m in self.marks]}
        out = {f"{prefix}meta": json.dumps(meta, indent=1) + "\n"}
        if self.crops is not None:
            out[f"{prefix}crops"] = np.ascontiguousarray(self.crops, np.uint8)
        if self.thumbs is not None:
            out[f"{prefix}thumbs"] = np.ascontiguousarray(self.thumbs, np.uint8)
        return out

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str = "") -> "VideoFingerprint":
        """The inverse of `to_arrays`; ValueError (a sentence) for anything that does not fit."""
        raw = arrays.get(f"{prefix}meta")
        if raw is None:
            raise ValueError("meta.json is missing")
        m = json.loads(raw) if isinstance(raw, str) else raw
        if int(m.get("format_version", 0)) > FORMAT_VERSION:
            raise ValueError("made by a newer Kinetrace")
        marks = [Mark(int(k["frame"]), tuple(int(v) for v in k["box"]), float(k.get("motion", 0.0)),
                      float(k.get("sep", 0.0)), float(k.get("move_px", 0.0)), str(k.get("picked", "auto")))
                 for k in m.get("marks") or []]
        crops, thumbs = arrays.get(f"{prefix}crops"), arrays.get(f"{prefix}thumbs")
        K = len(marks)
        if crops is not None and (crops.ndim != 4 or crops.shape[0] != K or crops.shape[3] != 3
                                  or crops.dtype != np.uint8):
            raise ValueError(f"crops.npy holds {crops.shape} {crops.dtype}, not {K} colour crops")
        if thumbs is not None and (thumbs.ndim != 3 or thumbs.shape[0] != K or thumbs.dtype != np.uint8):
            raise ValueError(f"thumbs.npy holds {thumbs.shape} {thumbs.dtype}, not {K} grey pictures")
        if crops is None and K:
            raise ValueError("crops.npy is missing")
        for k in marks:
            x, y, side = k.box
            if crops is not None and (side != crops.shape[1] or x < 0 or y < 0):
                raise ValueError(f"frame {k.frame}: crop box {k.box} does not fit the crops")
        return cls(dict(m.get("decoder") or {}), dict(m.get("file") or {}), int(m["n_frames"]), int(m["width"]),
                   int(m["height"]), str(m.get("made", "")), marks, crops, thumbs, str(m.get("seeks", "exact")),
                   bool(m.get("data_before", False)))

    def same(self, other: "VideoFingerprint | None") -> bool:
        """Equal content (the project file's round-trip test)."""
        if other is None:
            return False
        a, b = self.to_arrays(), other.to_arrays()
        return (a["meta"] == b["meta"] and all(np.array_equal(a.get(k), b.get(k)) for k in ("crops", "thumbs"))
                and (("crops" in a) == ("crops" in b)) and (("thumbs" in a) == ("thumbs" in b)))


# ------------------------------------------------------------------ pictures
def _grey(bgr) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _analysis(bgr) -> np.ndarray:
    """The picture the change is measured on: grey, `ANALYSIS_W` wide (float)."""
    w = min(ANALYSIS_W, bgr.shape[1])
    h = max(2, int(round(bgr.shape[0] * w / max(1, bgr.shape[1]))))
    return cv2.resize(_grey(bgr), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _thumb(a: np.ndarray) -> np.ndarray:
    """The whole frame kept for the eye: grey, `THUMB_W` wide, from the analysis copy."""
    h = max(2, int(round(a.shape[0] * THUMB_W / max(1, a.shape[1]))))
    return np.clip(cv2.resize(a, (THUMB_W, h), interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    """1 - normalised correlation of two pictures (grey or colour, any dtype): 0 = the same picture
    up to brightness and contrast (what differs between two decoders' colour conversions), larger =
    a different picture."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)
    if min(a.shape) >= 16:      # a light blur: compression noise down, a 1-px movement still there
        a = cv2.GaussianBlur(a, (0, 0), 1.0)
        b = cv2.GaussianBlur(b, (0, 0), 1.0)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(float((a * a).sum()) * float((b * b).sum())))
    if den < 1e-6:
        return 0.0 if float(np.abs(a).max(initial=0)) < 1e-3 and float(np.abs(b).max(initial=0)) < 1e-3 else 1.0
    return float(max(0.0, 1.0 - float((a * b).sum()) / den))


def _crop(img: np.ndarray, box) -> np.ndarray:
    x, y, s = (int(v) for v in box)
    return img[y:y + s, x:x + s]


def _move_px(prev_grey: np.ndarray, grey: np.ndarray) -> float:
    """How far the crop's content moves from the frame before, in pixels (the 95th percentile of
    the optical-flow length): the most a one-frame shift can move a tracked point there."""
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_grey, grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    except cv2.error:
        return 0.0
    return float(np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 95))


def _place(t_prev, t, t_next, width: int, height: int, side: int) -> tuple[tuple, float]:
    """Where the crop goes and how much changes there: the box (in the full picture) in which the
    frame differs most from BOTH neighbours (the smaller of the two changes, averaged over the box,
    on the analysis copy), and that average (grey levels)."""
    d = np.minimum(np.abs(t - t_prev), np.abs(t - t_next))
    s = width / float(t.shape[1])                       # full pixels per analysis pixel
    k = max(1, int(round(side / s)))
    acc = cv2.boxFilter(d, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
    yy, xx = np.unravel_index(int(np.argmax(acc)), acc.shape)
    cx, cy = (xx + 0.5) * s - 0.5, (yy + 0.5) * s - 0.5
    x0 = int(np.clip(round(cx - side / 2), 0, max(0, width - side)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(0, height - side)))
    return (x0, y0, side), float(acc[yy, xx]) / float(k * k)


def _stretches(n: int, k: int = N_MARKS, budget: int = SCAN_BUDGET, length: int = STRETCH) -> list:
    """[(first, count)] of frames to read: the whole video when it fits the budget, else a few
    evenly spread stretches in each of the `k` parts (every part gets the same share)."""
    if n <= budget:
        return [(0, n)]
    per = max(1, budget // (k * length))
    out = []
    for w in range(k):
        a, b = (n * w) // k, (n * (w + 1)) // k
        for j in range(per):
            first = a + ((b - a - length) * (2 * j + 1)) // (2 * per) if b - a > length else a
            out.append((max(0, first), min(length, n - first)))
    return out


@dataclass
class _Cand:
    frame: int
    score: float
    box: tuple
    crop: np.ndarray            # RGB, frame N
    prev: np.ndarray            # grey crops of N - 1 and N + 1
    nxt: np.ndarray
    thumb: np.ndarray
    picked: str = "auto"


def _candidate(f: int, bgr3: list, an3: list, box: tuple, score: float, picked: str = "auto") -> _Cand:
    rgb = cv2.cvtColor(np.ascontiguousarray(_crop(bgr3[1], box)), cv2.COLOR_BGR2RGB)
    return _Cand(f, score, box, rgb, _grey(np.ascontiguousarray(_crop(bgr3[0], box))),
                 _grey(np.ascontiguousarray(_crop(bgr3[2], box))), _thumb(an3[1]), picked)


def _finish(cands: list, path: str, n: int, width: int, height: int, seeks: str, data_before: bool
            ) -> VideoFingerprint:
    cands = sorted(cands, key=lambda c: c.frame)
    marks = []
    for c in cands:
        g = _grey(cv2.cvtColor(c.crop, cv2.COLOR_RGB2BGR))
        marks.append(Mark(c.frame, c.box, c.score, min(dist(g, c.prev), dist(g, c.nxt)), _move_px(c.prev, g),
                          c.picked))
    crops = np.stack([c.crop for c in cands]) if cands else None
    thumbs = np.stack([c.thumb for c in cands]) if cands else None
    return VideoFingerprint(decoder_identity(), file_identity(path), int(n), int(width), int(height),
                            time.strftime("%Y-%m-%d %H:%M:%S"), marks, crops, thumbs, seeks, bool(data_before))


def exact_reads(path: str, n_frames: int, should_cancel=lambda: False) -> None:
    """Make sure this file's seeks were checked on this computer (`check_seeks`), so every read
    below is exact -- the app's worker has usually done it already; a caller of this module on its
    own (a test, the CI check) gets it here."""
    plan = seek_plan(path)
    if plan is None or plan.file != file_identity(path):
        plan = check_seeks(path, n_frames, should_cancel)
        if not should_cancel():
            set_seek_plan(path, plan)


def _reader(path: str):
    cap = open_capture(str(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"could not open {path}")
    return cap


def make(path: str, n_frames: int, *, k: int = N_MARKS, seeks: str = "exact", data_before: bool = False,
         crop: int = CROP, progress=None, should_cancel=lambda: False) -> VideoFingerprint | None:
    """The fingerprint of `path` as THIS computer decodes it: one moment per part of the video (k
    parts), the frame there that differs most from both neighbours. A video of up to SCAN_BUDGET
    frames is read whole from the start (no seek at all); a longer one in evenly spread stretches
    of STRETCH frames. One full-resolution frame and its two neighbours are held at a time, never
    the video. None when cancelled."""
    n = int(n_frames)
    if n > SCAN_BUDGET:                 # a short video is read from the start: no seek at all
        exact_reads(path, n, should_cancel)
    cap = _reader(path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        side = int(min(crop, width, height))
        best: dict[int, _Cand] = {}
        plan = _stretches(n, k)
        total = sum(c for _, c in plan)
        done = 0
        for first, count in plan:
            if first > 0 and not seek_capture(cap, path, first):
                continue
            bgr3, an3 = [], []
            for f in range(first, first + count):
                if should_cancel():
                    return None
                ok, bgr = cap.read()
                if not ok:
                    break
                done += 1
                if progress is not None and done % 50 == 0:
                    progress(done / max(1, total))
                bgr3.append(bgr)
                an3.append(_analysis(bgr))
                if len(bgr3) > 3:
                    bgr3.pop(0)
                    an3.pop(0)
                if len(bgr3) < 3:
                    continue
                c = f - 1                                    # the middle of the three
                if not NEAR <= c <= n - 1 - NEAR:
                    continue
                box, score = _place(an3[0], an3[1], an3[2], width, height, side)
                part = min(k - 1, (c * k) // n)
                if part not in best or score > best[part].score:
                    best[part] = _candidate(c, bgr3, an3, box, score)
        return _finish(list(best.values()), path, n, width, height, seeks, data_before)
    finally:
        cap.release()


def add_mark(fp: VideoFingerprint, path: str, frame: int) -> VideoFingerprint:
    """The fingerprint with one more moment at `frame` -- one the user pointed at because the
    video barely moves elsewhere (it replaces a saved moment at the same frame). Only on the
    computer that made the fingerprint: the caller checks `same_decoder`."""
    f = int(frame)
    if not 1 <= f <= fp.n_frames - 2:
        raise ValueError(f"frame {f} has no frame on both sides")
    exact_reads(path, fp.n_frames)
    cap = _reader(path)
    try:
        if not seek_capture(cap, path, f - 1):
            raise OSError(f"frame {f - 1} could not be read")
        bgr3 = []
        for _ in range(3):
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"frame {f - 1 + len(bgr3)} could not be read")
            bgr3.append(bgr)
    finally:
        cap.release()
    an3 = [_analysis(b) for b in bgr3]
    side = int(fp.crops.shape[1]) if fp.crops is not None and len(fp.crops) else int(min(CROP, fp.width, fp.height))
    box, score = _place(an3[0], an3[1], an3[2], fp.width, fp.height, side)
    one = _finish([_candidate(f, bgr3, an3, box, score, "user")], path, fp.n_frames,
                  fp.width, fp.height, fp.seeks, fp.data_before)
    rows = [(m, fp.crops[i], fp.thumbs[i] if fp.thumbs is not None else None)
            for i, m in enumerate(fp.marks) if m.frame != f]
    rows.append((one.marks[0], one.crops[0], one.thumbs[0]))
    rows.sort(key=lambda r: r[0].frame)
    thumbs = None
    if all(r[2] is not None for r in rows) and len({r[2].shape for r in rows}) == 1:
        thumbs = np.stack([r[2] for r in rows])
    return VideoFingerprint(dict(fp.decoder), dict(fp.file), fp.n_frames, fp.width, fp.height, fp.made,
                            [r[0] for r in rows], np.stack([r[1] for r in rows]), thumbs, fp.seeks, fp.data_before)


# ------------------------------------------------------------------ the check
@dataclass
class MarkCheck:
    """One saved moment on this computer: its distances to this computer's frames N + d
    (`dists` {d: 1 - correlation}), the best d, and a verdict: "clear" (one d stands out),
    "unclear" (several fit about as well), "none" (none of them is the saved picture)."""
    frame: int
    dists: dict
    best: int | None
    verdict: str
    here: dict = field(default_factory=dict)        # {d: RGB crop of this computer's frame N + d}
    here_thumb: dict = field(default_factory=dict)  # {d: grey whole frame (thumb size)}

    @property
    def ratio(self) -> float:
        """How much closer the best match is than the next: >= CLEAR = clear."""
        if self.best is None or len(self.dists) < 2:
            return float("nan")
        others = [v for d, v in self.dists.items() if d != self.best]
        return min(others) / max(self.dists[self.best], 1e-4)


@dataclass
class CheckResult:
    verdict: str                    # "same" | "shifted" | "mixed" | "different" | "cannot_tell"
    shift: int | None               # frames: this computer shows the saved picture of frame N at N + shift
    quality: str                    # "good" | "ok" | "poor"
    sentence: str                   # the one plain sentence
    detail: str                     # the numbers behind it, and the hand check
    marks: list                     # [MarkCheck]
    made_on: str
    here: str
    move_px: float = 0.0

    @property
    def needs_eyes(self) -> bool:
        """The evidence dialog is shown: a shift, a mismatch, or 'cannot tell' where it matters."""
        return self.verdict in ("shifted", "mixed", "different") or (
            self.verdict == "cannot_tell" and self.quality == "poor")


def needs_check(fp: VideoFingerprint | None, path: str) -> bool:
    """True when this computer may decode `path` differently from where `fp` was made: another
    decoder (OS, OpenCV, FFmpeg, backend) or another file (size, time)."""
    if fp is None or not fp.marks:
        return False
    return not (same_decoder(fp.decoder, decoder_identity()) and same_file(fp.file, file_identity(path)))


def _judge(dists: dict) -> tuple[int | None, str]:
    """A moment's verdict from its distances {d: 1 - correlation}: "clear" (the best is CLEAR times
    closer than every other), "lean" (LEAN times: on its own not proof, several that agree are),
    "unclear", or "none" (not even the best is the saved picture)."""
    best = min(dists, key=lambda d: (dists[d], abs(d)))
    b = dists[best]
    if b > NO_MATCH:
        return best, "none"
    others = [v for d, v in dists.items() if d != best]
    if not others:
        return best, "unclear"
    nxt = min(others)
    if nxt - b < MIN_SEP:
        return best, "unclear"
    if nxt >= CLEAR * b:
        return best, "clear"
    if nxt >= LEAN * b:
        return best, "lean"
    return best, "unclear"


def _read_near(cap, path: str, f: int, reach: int, n_here: int, box, thumbs: bool) -> tuple[dict, dict]:
    a, b = max(0, f - reach), min(n_here - 1, f + reach)
    crops, ths = {}, {}
    if not seek_capture(cap, path, a):
        return crops, ths
    for g in range(a, b + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        c = _crop(bgr, box)
        if c.shape[0] != box[2] or c.shape[1] != box[2]:
            break                                       # another picture size: nothing to compare
        crops[g - f] = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_BGR2RGB)
        if thumbs:
            ths[g - f] = _thumb(_analysis(bgr))
    return crops, ths


def compare(fp: VideoFingerprint, path: str, n_frames: int, camera: str = "this camera", *,
            progress=None, should_cancel=lambda: False) -> CheckResult | None:
    """This computer's frames against the saved moments (see the module text). None when cancelled."""
    n_here = int(n_frames)
    here_id = describe_decoder(decoder_identity())
    made_on = describe_decoder(fp.decoder)
    marks: list[MarkCheck] = []
    exact_reads(path, n_here, should_cancel)
    cap = _reader(path)
    try:
        for i, m in enumerate(fp.marks):
            if should_cancel():
                return None
            crops, ths = _read_near(cap, path, m.frame, NEAR, n_here, m.box, True)
            saved = fp.crops[i]
            dists = {d: dist(saved, c) for d, c in crops.items()}
            best, verdict = _judge(dists) if dists else (None, "none")
            marks.append(MarkCheck(m.frame, dists, best, verdict, crops, ths))
            if progress is not None:
                progress((i + 1) / max(1, 2 * len(fp.marks)))
        if marks and not any(k.verdict == "clear" for k in marks) and \
                sum(k.verdict == "none" for k in marks) * 2 >= len(marks):
            # nothing is near its own frame: look further (a bigger shift, or another video)
            for i, (m, k) in enumerate(zip(fp.marks, marks)):
                if should_cancel():
                    return None
                crops, ths = _read_near(cap, path, m.frame, WIDE, n_here, m.box, True)
                dists = {d: dist(fp.crops[i], c) for d, c in crops.items()}
                if dists:
                    best, verdict = _judge(dists)
                    marks[i] = MarkCheck(m.frame, dists, best, verdict, crops, ths)
                if progress is not None:
                    progress(0.5 + (i + 1) / max(1, 2 * len(fp.marks)))
    finally:
        cap.release()
    return _verdict(fp, marks, camera, made_on, here_id)


def _frames(k: int) -> str:
    return f"{abs(k)} frame{'s' if abs(k) != 1 else ''}"


def _verdict(fp: VideoFingerprint, marks: list, camera: str, made_on: str, here: str) -> CheckResult:
    """The moments' verdicts -> one answer. Clear moments decide; without one, moments that only
    lean decide when at least 3 and at least half of all lean and every one of them agrees (8 moments
    that each pick the same neighbour by chance is not what noise does); otherwise it cannot tell."""
    clear = [k for k in marks if k.verdict == "clear"]
    lean = [k for k in marks if k.verdict == "lean"]
    none = [k for k in marks if k.verdict == "none"]
    move = fp.move_px
    usable = [k for k, m in zip(marks, fp.marks) if m.sep >= MIN_SEP]
    tail = f" (made on {made_on}; this computer: {here}.)"
    hand = (f"Hand check: open {camera}, go to frame {fp.marks[0].frame if fp.marks else 0} and compare the "
            "picture with the saved one shown here (the crop's place is marked on the whole frame).")
    nums = "; ".join(f"frame {k.frame}: " + (f"best {k.best:+d}, {k.ratio:.1f}x closer than the next ({k.verdict})"
                                             if k.best is not None and np.isfinite(k.ratio) else k.verdict)
                     for k in marks) + "." + tail + " " + hand
    evidence, firm = clear, True
    if not clear and len(lean) >= 3 and 2 * len(lean) >= len(marks) and len({k.best for k in lean}) == 1:
        evidence, firm = lean, False
    shifts = sorted({k.best for k in evidence})
    if firm:
        how = f"{len(clear)} of {len(marks)} moments clearly"
    else:
        rs = [k.ratio for k in lean]
        how = (f"all {len(lean)} moments that stand out point the same way, each only {min(rs):.1f}-{max(rs):.1f}x "
               "closer than the next: look at them below")
    if len(shifts) == 1:
        k = shifts[0]
        if k == 0:
            return CheckResult("same", 0, "good" if firm and len(clear) >= 2 else "ok",
                               f"{camera}: this computer shows every saved picture at its own frame number ({how}); "
                               "the project's frames are the same pictures here.", nums, marks, made_on, here, move)
        when = "later" if k > 0 else "earlier"
        return CheckResult("shifted", k, "poor",
                           f"On this computer {camera} shows the picture {_frames(k)} {when} than where the "
                           f"project was made ({how}): its tracks, silhouettes and events sit {_frames(k)} off "
                           "here. Nothing was changed.", nums, marks, made_on, here, move)
    if len(shifts) > 1:
        return CheckResult("mixed", None, "poor",
                           f"On this computer {camera} shows some saved pictures at other frame numbers than "
                           "others (" + ", ".join(f"{k.best:+d} at frame {k.frame}" for k in evidence) + "): the "
                           "shift changes along the video, so a frame is missing or doubled somewhere here. "
                           "Nothing was changed.", nums, marks, made_on, here, move)
    if marks and len(none) == len(marks):
        return CheckResult("different", None, "poor",
                           f"None of the {len(marks)} saved pictures of {camera} is found within {WIDE} frames of "
                           "its frame on this computer: this is probably another video (another take or cut) "
                           "or it decodes very differently here. Nothing was changed.", nums, marks, made_on, here,
                           move)
    # cannot tell: say how much it could matter
    if move < 0.5:
        q, size = "good", "harmless"
    elif move < 2.0:
        q, size = "ok", "small"
    else:
        q, size = "poor", "not small"
    why = ("the video barely moves, so neighbouring frames look alike" if fp.still or not usable
           else "the saved moments do not stand out clearly from their neighbours here")
    return CheckResult("cannot_tell", None, q,
                       f"Kinetrace cannot tell whether {camera} shows the same picture at each frame number on "
                       f"this computer: {why}. Where it moves most, the picture moves {move:.2f} px from one "
                       f"frame to the next, so a shift of one frame would move its tracks by about that much "
                       f"({size}).", nums
                       + (" Point at a frame where something moves fast and add it to the fingerprint on the "
                          "computer that made the project." if q == "poor" else ""),
                       marks, made_on, here, move)
