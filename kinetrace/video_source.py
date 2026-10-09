"""Streaming video frame access for long (40k+ frame, 4K) videos.

Never loads the whole video into memory. Three cooperating pieces:

- FrameCache: byte-budgeted, thread-safe LRU of decoded RGB frames, shared
  between the GUI-side seek thread and tracking workers.
- VideoSource: cv2.VideoCapture wrapper with a forward-read fast path and
  frame-accurate seeks. One instance per thread (VideoCapture isn't
  thread-safe).
- SeekService: QThread that owns the scrubbing VideoSource. The GUI thread
  never touches cv2. Latest-wins request queue: rapid scrubbing collapses
  to the newest target so there is never a decode backlog.
"""

from __future__ import annotations

import bisect
import os
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


def _total_ram_bytes() -> int:
    """Physical RAM, or 0 when it cannot be determined (`device.total_ram_bytes`:
    ctypes on Windows, sysconf on macOS and Linux)."""
    from kinetrace.device import total_ram_bytes
    return total_ram_bytes()


def _default_cache_bytes() -> int:
    """Frame-cache budget: a quarter of physical RAM, clamped to [2, 24] GB.

    The cache is what makes re-scrubbing a stretch you are correcting
    instant (pure cache hits, zero decode), so on a workstation it should be
    much larger than the old fixed 2 GB (~85 frames of 4K RGB). It fills
    lazily — this is a ceiling, not an allocation. Override with
    KINETRACE_CACHE_GB.
    """
    override = os.environ.get("KINETRACE_CACHE_GB")
    if override:
        try:
            return max(1, int(float(override) * 1024**3))
        except ValueError:
            pass
    total = _total_ram_bytes()
    if total <= 0:
        return 2 * 1024**3  # unknown RAM: keep the old conservative default
    return int(min(max(total // 4, 2 * 1024**3), 24 * 1024**3))


DEFAULT_CACHE_BYTES = _default_cache_bytes()  # ~640 4K frames on a 64 GB box
SEEK_FORWARD_MAX = 30  # read forward instead of seeking when target is this close ahead
SEEK_BACK_PREFETCH = 12  # on a backward seek, decode this run-up too (B-key steps)
PREFETCH_AHEAD = 8       # idle prefetch after serving a seek (F-key steps, playback)
MAX_HEADER_FPS = 100_000.0   # a header rate above this is a timebase, not a camera (I37)


class FrameCache:
    """Thread-safe byte-budgeted LRU of {frame_index: RGB uint8 array}."""

    def __init__(self, max_bytes: int = DEFAULT_CACHE_BYTES):
        self._lock = threading.Lock()
        self._data: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes = 0
        self.max_bytes = max_bytes

    def get(self, idx: int) -> np.ndarray | None:
        with self._lock:
            frame = self._data.get(idx)
            if frame is not None:
                self._data.move_to_end(idx)
            return frame

    def put(self, idx: int, frame: np.ndarray) -> None:
        with self._lock:
            old = self._data.pop(idx, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._data[idx] = frame
            self._bytes += frame.nbytes
            while self._bytes > self.max_bytes and len(self._data) > 1:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= evicted.nbytes

    def nearest(self, idx: int) -> tuple[int, np.ndarray] | None:
        """Cached frame with index closest to idx (for instant scrub preview)."""
        with self._lock:
            if not self._data:
                return None
            best = min(self._data, key=lambda k: abs(k - idx))
            return best, self._data[best]

    def stats(self) -> tuple[int, int]:
        """(cached frame count, bytes held) — for the cache-reset feedback."""
        with self._lock:
            return len(self._data), self._bytes

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def trim(self) -> int:
        """Evict down to the CURRENT budget and return the bytes released.
        `put` only evicts when something is added, so a cache whose budget was
        just lowered (a camera that stopped being the working one) would sit on
        the old memory until its next decode — with 15 cameras that is the
        difference between a bounded app and an unbounded one."""
        with self._lock:
            freed = 0
            while self._bytes > self.max_bytes and len(self._data) > 1:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= evicted.nbytes
                freed += evicted.nbytes
            return freed


@dataclass
class VideoInfo:
    path: str
    n_frames: int                  # frames that can actually be DECODED (verified)
    fps: float
    width: int
    height: int
    vfr_suspected: bool = False
    header_frames: int = 0         # what the container claimed; differs when the file ends early
    # where `fps` came from (I37): "header" (the file states it), "timestamps"
    # (measured from the frames' own clock because the header gave no usable
    # rate) or "assumed" (neither did: 30 fps is a guess). Every time, speed,
    # camera rate and sync offset hangs on this number, so anything but
    # "header" must be said to the user (`fps_note`).
    fps_source: str = "header"
    # (G145) GoPro footage: the `gpmf.GoProInfo` read from the file (the flag that turns on the GoPro
    # workflow -- its lens model, settings checks, sensors); None for every other camera
    gopro: object = None

    @property
    def header_overcount(self) -> int:
        """Phantom frames the header promised but the decoder cannot deliver."""
        return max(0, int(self.header_frames) - int(self.n_frames)) if self.header_frames else 0

    @property
    def fps_note(self) -> str:
        """A plain-language warning when the frame rate did not come from the
        file's header, else "" (I37)."""
        if self.fps_source == "timestamps":
            return (f"This file does not state a usable frame rate, so Kinetrace measured "
                    f"{self.fps:.6g} fps from its frame timestamps. Times, speeds and camera sync "
                    "use this number: check it against the rate the camera recorded at. "
                    "If it is wrong, set the real rate with the camera's fps button in the CAMERAS panel.")
        if self.fps_source == "assumed":
            return (f"This file states no frame rate and its timestamps give none, so Kinetrace is "
                    f"ASSUMING {self.fps:.6g} fps. Times, speeds and camera sync will be wrong unless "
                    "the camera really recorded at that rate: set the real rate with the camera's fps "
                    "button in the CAMERAS panel.")
        return ""


def verified_frame_count(cap, n_header: int, max_probes: int = 24) -> int:
    """The number of frames that really decode, from the header's claim.

    Containers over-report: real camera clips have claimed 607 / 614 frames
    and decoded 601, and every phantom frame became a blank column in the timeline,
    the coverage window and the exports. Seeking to the last claimed frame and
    reading it is the cheap check; when that fails, a binary search over the
    seekable range finds the last frame that decodes (measured: 10 probes,
    1.7 s, exactly the sequential count on both clips). A file whose last
    frame reads fine costs one seek."""
    n = int(n_header)
    if n <= 0:
        return n

    def ok_at(k: int) -> bool:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        good, _ = cap.read()
        return bool(good)

    if ok_at(n - 1):
        return n
    lo, hi = 0, n - 1                 # lo decodes (assumed for frame 0), hi does not
    if not ok_at(0):
        return n                      # seeking is broken on this file: trust the header
    probes = 0
    while hi - lo > 1 and probes < max_probes:
        mid = (lo + hi) // 2
        probes += 1
        if ok_at(mid):
            lo = mid
        else:
            hi = mid
    return lo + 1


def probe_video(path: str, vfr_samples: int = 60, progress=None) -> VideoInfo:
    """Read metadata and sniff for variable frame rate. Raises ValueError on failure.
    `progress(stage, facts)`, if given, is called as the work goes -- "open",
    "rate" (facts: width, height, fps, frames), "frames" -- so a caller can say
    what is taking the time (a 4K file spends ~2 s decoding its first frames and
    its last one); it changes nothing in the result."""
    def say(stage: str, **facts) -> None:
        if progress is not None:
            try:
                progress(stage, facts)
            except Exception:       # noqa: BLE001 -- a progress display must never fail the probe
                pass

    say("open")
    cap = open_capture(path)  # same backend as decoding: frame counts must agree
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video:\n{path}\n\n"
            "The file may use an unsupported codec. Re-encoding usually fixes it:\n"
            f'ffmpeg -i "{Path(path).name}" -c:v libx264 -crf 18 fixed.mp4'
        )
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n_frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video reports invalid metadata (frames={n_frames}, {width}x{height}).")
        say("rate", width=width, height=height, fps=raw_fps, frames=n_frames)

        # VFR sniff: timestamps of the first frames should tick at ~1/fps.
        times = []
        for _ in range(min(vfr_samples, n_frames)):
            ok = cap.grab()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        vfr = False
        ts_fps = 0.0                   # the rate the timestamps themselves imply
        span = 0.0
        n_steps = 0
        if len(times) > 10:
            ks = [k for k, t in enumerate(times) if t > 0]
            tpos = np.asarray([times[k] for k in ks], np.float64)
            deltas = np.diff(tpos)
            deltas = deltas[deltas > 0]
            if len(deltas) > 5:
                # (I42) MKV / WebM / FLV clocks tick in whole milliseconds: a
                # constant 400 fps stream reads 2, 3, 2, 3 ms (a std of up to half
                # the 1 ms quantum), which is the container rounding, not a
                # variable rate. MP4 timestamps are not whole ms, so unaffected.
                quantum_half = 0.5 if np.allclose(tpos, np.round(tpos), atol=1e-3) else 0.0
                vfr = bool(np.std(deltas) > max(0.15 * np.mean(deltas), quantum_half + 1e-6))
                span, n_steps = float(tpos[-1] - tpos[0]), ks[-1] - ks[0]
                if span > 0:
                    ts_fps = 1000.0 * n_steps / span   # mean rate: exact through ms rounding

        # (I37) any finite header rate up to MAX_HEADER_FPS is believed —
        # high-speed camera files can state a real 1000-25000 fps, and the old
        # "> 1000 means 30" rule silently made every time, speed and sync offset
        # wrong by that factor. A header that contradicts regular
        # timestamps by more than the ms quantum + 10 % is a container TIMEBASE
        # (1000, 90000...), not a rate, and the timestamps win; with no usable
        # header either, 30 fps is an announced guess (`fps_source`).
        fps, fps_source = 30.0, "assumed"
        if np.isfinite(raw_fps) and 0.0 < raw_fps <= MAX_HEADER_FPS:
            fps, fps_source = raw_fps, "header"
            if (ts_fps > 0 and not vfr
                    and abs(n_steps * 1000.0 / raw_fps - span) > max(2.0, 0.1 * span)):
                fps, fps_source = ts_fps, "timestamps"
        elif 0.0 < ts_fps <= MAX_HEADER_FPS:
            fps, fps_source = ts_fps, "timestamps"
        header = n_frames
        say("frames", width=width, height=height, fps=fps, frames=header)
        n_frames = verified_frame_count(cap, header)
        if not times:
            # (I235) the sniff above decoded nothing: if frame 0 does not read either, "trust the
            # header" (verified_frame_count's answer for a file that cannot seek) would accept a
            # video no frame of which decodes, and every later step would toast instead
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not cap.read()[0]:
                n_frames = 0
        if n_frames <= 0:
            raise ValueError(
                f"No frame of this video could be decoded ({header} claimed).\n\n"
                "The file may be damaged, or use a codec this computer cannot read. Re-encoding usually fixes it:\n"
                f'ffmpeg -i "{Path(path).name}" -c:v libx264 -crf 18 fixed.mp4')
        return VideoInfo(path, n_frames, fps, width, height, vfr, header, fps_source)
    finally:
        cap.release()


def open_capture(path: str) -> cv2.VideoCapture:
    """Open a VideoCapture using the process-wide decode backend.

    Default is OpenCV's own choice (FFmpeg software) — measured most
    reliable for the random seeks that dominate scrubbing. Set
    KINETRACE_DECODE to trade that for hardware decode, which measured ~2x
    faster SEQUENTIAL decode at 4K (helps long tracking runs) but slower
    seeks:
        (unset) / auto  software FFmpeg  — the default
        msmf            Media Foundation, hardware
        hw              FFmpeg + D3D11VA hardware
        ffmpeg          FFmpeg, explicitly software

    Backends must never be MIXED inside one session: they agree on frame
    indices, but their YUV->RGB conversion differs by a few levels (measured
    max 28-56), and the tracker's subpixel refinement should not see
    different pixels than the user corrected on. The choice is the environment
    variable, process-wide (read at every open, never per call site), which is
    what guarantees that.
    """
    mode = os.environ.get("KINETRACE_DECODE", "").strip().lower()
    if mode == "msmf":
        return cv2.VideoCapture(path, cv2.CAP_MSMF)
    if mode == "hw":
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_HW_ACCELERATION,
                                 cv2.VIDEO_ACCELERATION_ANY])
    if mode == "ffmpeg":
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(path)


class VideoOpenError(RuntimeError):
    """A video that could not be opened; the message is a sentence."""


# ------------------------------------------------------------------ exact frames (I266)
# A project stores everything by frame number, so frame N must be the same picture wherever it is
# shown. "Frame N" is the N-th picture a decoder gives when it reads the file from the start. A seek
# (CAP_PROP_POS_FRAMES) is OpenCV's arithmetic from timestamps and the average frame rate: on a file
# whose timestamps skip (a camera that dropped frames), vary, or start late it lands frames away from
# where reading forward is (measured here: a 5-frame gap moved seeks by up to 2 frames). `check_seeks`
# compares seeks with reading forward for each file on THIS computer; a file whose seeks are exact
# keeps the plain seek (no file gets slower), any other is reached by `seek_capture` landing BEFORE
# the frame, naming the frame it landed on by its own timestamp (`PtsMap`, from the file's packets)
# and reading forward to the one asked for.

@dataclass
class PtsMap:
    """Frame number of a decoded picture from its timestamp (CAP_PROP_PTS, stream units): the
    file's packet timestamps in picture order (`pts`), the packets an edit list hides before the
    first picture (`lead`), and the difference between the two readings' clocks (`delta`)."""
    pts: np.ndarray
    lead: int = 0
    delta: float = 0.0
    tol: float = 0.25

    def index(self, p: float) -> int | None:
        """The frame number of the picture whose timestamp is `p`, or None (not one of the file's)."""
        if p is None or not np.isfinite(p):
            return None
        q = float(p) - self.delta
        j = int(np.searchsorted(self.pts, q - self.tol))
        if j < len(self.pts) and abs(float(self.pts[j]) - q) <= self.tol:
            return j - self.lead
        return None


@dataclass
class SeekPlan:
    """How frames of one file are reached on this computer (`check_seeks`)."""
    exact: bool                      # a seek gives the picture reading forward gives
    pts_map: PtsMap | None = None    # not exact: names the frame a seek landed on (None = read from the start)
    checked: int = 0                 # seeks compared with frames read forward
    wrong: list = field(default_factory=list)   # [(asked, got)] where a seek landed elsewhere (got None = unknown)
    file: dict | None = None         # the file's size / time when it was checked
    gaps: int = 0                    # frames the file's timestamps skip (a camera that dropped frames)
    note: str = ""                   # the sentence for the user when not exact


_PLANS: dict = {}
_PLANS_LOCK = threading.Lock()


def _plan_key(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def file_identity(path) -> dict:
    """The file as it is on disk: name, size, modification time (ns). {} when it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    return {"name": Path(path).name, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def seek_plan(path) -> SeekPlan | None:
    """The plan `check_seeks` found for this file on this computer, or None (not checked yet: plain
    seeks, as before the check existed)."""
    with _PLANS_LOCK:
        return _PLANS.get(_plan_key(path))


def set_seek_plan(path, plan: SeekPlan | None) -> None:
    with _PLANS_LOCK:
        if plan is None:
            _PLANS.pop(_plan_key(path), None)
        else:
            _PLANS[_plan_key(path)] = plan


def seek_capture(cap, path, idx: int) -> bool:
    """Position `cap` (a capture of `path`) so its next read returns frame `idx`: THE seek for every
    capture. A plain seek unless this file's seeks were found inexact here; then it lands before
    `idx`, names the frame it landed on from that picture's timestamp and reads forward (decoding
    without converting, `grab`) -- landing earlier again while it overshoots, and from the start of
    the file when the timestamps cannot name the frame. False when the file ends before `idx`."""
    idx = max(0, int(idx))
    plan = seek_plan(path)
    if plan is None or plan.exact or idx == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        return True
    m = plan.pts_map
    if m is not None:
        want = idx - 1                     # land on a frame <= idx - 1: the next read is then a fresh one
        back = 1
        guess = want
        for _ in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, guess))
            if not cap.grab():
                break
            c = m.index(cap.get(cv2.CAP_PROP_PTS))
            if c is None:
                break
            if c <= want:
                for _ in range(want - c):
                    if not cap.grab():
                        return False
                return True
            if guess <= 0:
                break
            guess = max(0, guess - (c - want) - back)
            back *= 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    # the start is where a decoder begins anyway: exact, slow
    for _ in range(idx):
        if not cap.grab():
            return False
    return True


def packet_times(path, max_packets: int | None = None, should_cancel=None) -> tuple[np.ndarray, list, int]:
    """(timestamps in picture order, keyframes as picture ranks, packets read) from the file's
    packets alone -- nothing is decoded, so it is fast (5000 packets of a 1080p file in 0.03 s):
    OpenCV's raw-stream mode gives each packet's timestamp and keyframe flag; packets come in
    decode order (B-frames out of picture order), so a packet's picture rank is the rank of its
    timestamp. `max_packets` stops early (the last few ranks of a partial read may still move);
    `should_cancel()` is asked every 500 packets. (empty, [], 0) when the file cannot be read this
    way or the read was cancelled."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)     # packets only, no picture: no backend mixing
    pts, key = [], []
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_FORMAT, -1):
            return np.empty(0), [], 0
        while max_packets is None or len(pts) < max_packets:
            if should_cancel is not None and len(pts) % 500 == 0 and should_cancel():
                return np.empty(0), [], 0
            if not cap.grab():
                break
            pts.append(cap.get(cv2.CAP_PROP_PTS))
            key.append(bool(cap.get(cv2.CAP_PROP_LRF_HAS_KEY_FRAME)))
    except cv2.error:
        return np.empty(0), [], 0
    finally:
        cap.release()
    if not pts:
        return np.empty(0), [], 0
    p = np.asarray(pts, np.float64)
    order = np.argsort(p, kind="stable")
    rank = np.empty(len(p), np.int64)
    rank[order] = np.arange(len(p))
    return p[order], sorted(int(r) for r, k in zip(rank, key) if k), len(p)


SEEK_TOL = 0.75          # mean |grey| difference (levels) below which two decodes of one frame are the same
_CHECK_W = 96            # width of the grey copies the seek check compares


def _small(bgr) -> np.ndarray:
    h = max(2, int(round(bgr.shape[0] * _CHECK_W / max(1, bgr.shape[1]))))
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (_CHECK_W, h),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _same(a, b) -> bool:
    return a is not None and b is not None and a.shape == b.shape and float(np.abs(a - b).mean()) <= SEEK_TOL


def _calibrate(s: np.ndarray, first: list, hidden: int) -> PtsMap | None:
    """The `PtsMap` under which the first pictures read from the start are frames 0, 1, 2 ...:
    `first` = their timestamps as the decoding capture reports them, `s` = the packets' (sorted).
    An edit list hides packets before the first picture and may shift the clock: the hidden count
    (packets - frames) is tried first, then none, then up to 64."""
    f = np.asarray(first, np.float64)
    if len(f) < 2 or not np.all(np.isfinite(f)) or not np.all(np.diff(f) > 0) or len(s) < len(f):
        return None
    step = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    tol = 0.25 * step if step > 0 else 0.25
    for lead in dict.fromkeys([hidden, 0, *range(1, 65)]):
        if not 0 <= lead <= len(s) - len(f):
            continue
        delta = float(f[0] - s[lead])
        if np.all(np.abs(f - delta - s[lead:lead + len(f)]) <= tol):
            return PtsMap(s, int(lead), delta, tol)
    return None


def check_seeks(path, n_frames: int, should_cancel=lambda: False) -> SeekPlan:
    """Do seeks in this file land where reading forward does, on this computer? (I266)

    1. Every packet's timestamp is read (no decoding): their picture order is the frame numbering,
       and a timestamp that skips is a frame the camera dropped.
    2. The first pictures are read from the start, one after another -- exact by definition -- up to
       just past the second keyframe (48-300 frames); their timestamps tie the packets' clock to the
       decoder's (`_calibrate`).
    3. Seeks into that stretch, next to keyframes and between them, must give the same pictures.
    4. Seeks spread over the whole file must land on the picture whose timestamp is that frame's.
       (Without usable timestamps: a far seek must agree with reading forward from 2 GOPs before.)
    Exact: plain seeks stay. Not exact: the plan carries the timestamp map (or none: read from the
    start) for `seek_capture`, and `note` says what was found in words. Runs on a worker."""
    n = int(n_frames)
    plan = SeekPlan(True, None, 0, [], file_identity(path))
    if n < 3:
        return plan
    s, keys, npk = packet_times(path, should_cancel=should_cancel)
    if should_cancel():
        return plan
    inner = [k for k in keys if 0 < k < n - 8]
    m = min(n, max(48, min(300, (inner[1] if len(inner) > 1 else inner[0] if inner else 48) + 16)))
    cap = open_capture(str(path))
    if not cap.isOpened():
        cap.release()
        return plan
    try:
        prefix, first = [], []
        for _ in range(m):
            if should_cancel():
                return plan
            ok, bgr = cap.read()
            if not ok:
                break
            prefix.append(_small(bgr))
            first.append(cap.get(cv2.CAP_PROP_PTS))
        m = len(prefix)
        pmap = _calibrate(s, first, max(0, npk - n)) if m >= 2 else None
        if pmap is not None and len(s) > 1:
            step = float(np.median(np.diff(s)))
            if step > 0:
                q = (s - s[0]) / step
                if np.all(np.abs(q - np.round(q)) < 0.25):
                    plan.gaps = int(round(q[-1])) - (len(s) - 1)
        targets = {1, m // 2, m - 1} | {k + d for k in inner[:2] for d in (-1, 1, 3)}
        for t in sorted(t for t in targets if 0 < t < m):
            if should_cancel():
                return plan
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, bgr = cap.read()
            got = _small(bgr) if ok else None
            plan.checked += 1
            if not _same(got, prefix[t]):
                hit = [j for j in range(m) if _same(got, prefix[j])]
                plan.wrong.append((t, hit[0] if len(hit) == 1 else None))
        gop = (inner[1] - inner[0]) if len(inner) > 1 else (inner[0] if inner else 30)
        if pmap is not None:
            far = sorted({t for t in (n // 4 + 3, n // 2 + 5, (3 * n) // 4 + 7, n - 2) if m <= t < n})
            for t in far:
                if should_cancel():
                    return plan
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                got = pmap.index(cap.get(cv2.CAP_PROP_PTS)) if cap.grab() else None
                plan.checked += 1
                if got != t:
                    plan.wrong.append((t, got))
        else:
            for t in [t for t in (n // 2 + 5, n - 2) if t > m + 2 * gop]:
                if should_cancel():
                    return plan
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, bgr = cap.read()
                direct = _small(bgr) if ok else None
                back = max(0, t - max(32, 2 * gop))
                cap.set(cv2.CAP_PROP_POS_FRAMES, back)
                for _ in range(t - back):
                    if not cap.grab():
                        break
                ok, bgr = cap.read()
                plan.checked += 1
                if not _same(direct, _small(bgr) if ok else None):
                    plan.wrong.append((t, None))
    finally:
        cap.release()
    if not plan.wrong:
        return plan
    plan.exact = False
    plan.pts_map = pmap
    t, got = plan.wrong[0]
    how = (f"asked for frame {t}, it showed frame {got}" if got is not None
           else f"asked for frame {t}, it showed another picture")
    why = (f" The file's timestamps skip {plan.gaps} frame(s) (the camera probably dropped them), and "
           "jumping goes by the timestamps while reading counts pictures." if plan.gaps > 0 else "")
    slow = ("jumping is a little slower" if pmap is not None
            else "jumping far back is SLOW because this file's timestamps cannot name its frames")
    plan.note = (f"Jumping to a frame of {Path(path).name} lands on the wrong picture on this computer ({how}; "
                 f"{len(plan.wrong)} of {plan.checked} jumps checked were wrong).{why} Kinetrace now reaches "
                 f"every frame of this video by reading forward to it: every frame is exact, {slow}. To make it "
                 f'fast again, re-encode the video once: ffmpeg -i "{Path(path).name}" -vsync cfr -c:v libx264 '
                 "-crf 18 fixed.mp4 (then track on the new file).")
    return plan


class FrameReader:
    """One decoder for frames wanted in increasing order (I232, R17): it seeks to
    the first, skips cheaply between sampled frames, and remembers the FIRST frame
    it could not read (`failed`). The pose run, the side-by-side export and the
    overlay export each carried a copy of this loop; none could tell a damaged
    file from a user's Stop, so a run said 'stopped early' and an export said
    'written'. One per thread (a VideoCapture is not thread-safe); `cap` is
    exposed for the properties (fps, size)."""

    def __init__(self, path: str, first: int = 0):
        self.path = str(path)
        self.cap = open_capture(self.path)
        if not self.cap.isOpened():
            self.cap.release()
            raise VideoOpenError("could not open " + str(path))
        self.reset(first)

    def reset(self, first: int) -> None:
        self.failed: int | None = None
        if not seek_capture(self.cap, self.path, int(first)):       # (I266) exact on every file
            self.failed = int(first)
        self.nxt = int(first)

    def read(self, f: int):
        """Frame `f` (BGR), or None -- then `failed` is the first frame that
        did not decode. `f` must not be lower than the one before."""
        if self.failed is not None:
            return None
        while self.nxt < f:                              # sampled run: skip cheaply
            if not self.cap.grab():
                self.failed = self.nxt
                return None
            self.nxt += 1
        ok, bgr = self.cap.read()
        if not ok:
            self.failed = int(f)
            return None
        self.nxt = int(f) + 1
        return bgr

    def release(self) -> None:
        self.cap.release()


class VideoSource:
    """Sequential + random frame access. One instance per thread."""

    def __init__(self, path: str, cache: FrameCache | None = None):
        self.path = path
        self.cache = cache if cache is not None else FrameCache()
        self._cap = open_capture(path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self._pos = 0  # index of the frame the next read() returns

    def read_next(self) -> tuple[int, np.ndarray] | None:
        """Read the next frame sequentially. Returns (index, RGB) or None at EOF."""
        idx = self._pos        # the frame this read was ASKED for, taken before it can block (I39)
        ok, bgr = self._cap.read()
        if not ok:
            return None
        self._pos += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.cache.put(idx, rgb)
        return idx, rgb

    def seek(self, idx: int) -> None:
        """Position so the next read_next() returns frame idx (`seek_capture`: a plain seek, or for
        a file whose seeks are inexact on this computer a landing before it and a read forward, I266)."""
        if idx == self._pos:
            return
        seek_capture(self._cap, self.path, idx)
        self._pos = idx

    def get_frame(self, idx: int) -> np.ndarray | None:
        """Random access with cache + forward-read fast path. None past EOF."""
        cached = self.cache.get(idx)
        if cached is not None:
            return cached
        if not (self._pos <= idx <= self._pos + SEEK_FORWARD_MAX):
            # Backward scrubbing steps through the neighbors next, and cv2
            # decodes forward from the previous keyframe either way — so land a
            # little early and cache the run-up: the following back-steps become
            # cache hits instead of full GOP re-decodes each.
            self.seek(max(0, idx - SEEK_BACK_PREFETCH) if idx < self._pos else idx)
        frame = None
        while self._pos <= idx:
            nxt = self.read_next()
            if nxt is None:
                return None
            _, frame = nxt
        return frame

    def close(self) -> None:
        self._cap.release()


class ReadAhead:
    """Decode a couple of frames ahead of the consumer, on its own thread.

    The tracking worker's loop is decode -> segment -> model -> emit, strictly
    serial, so every frame pays full decode latency before the GPU starts. On
    the 4K clip decode alone is ~16 ms of a ~60 ms frame. OpenCV releases the
    GIL inside `read()` and `cvtColor`, so moving those to a second thread
    genuinely overlaps them with the model step.

    This changes only WHEN a frame is decoded: same frames, same order, same
    pixels, so tracking output is bit-identical. Frames read ahead but never
    consumed still land in the shared cache, so nothing is wasted.

    While a ReadAhead is running it is the ONLY owner of the VideoSource —
    `stop()` before seeking, reusing or closing it. After stopping, the source
    is positioned up to `depth` frames further on than the last frame consumed;
    every caller seeks before the next segment, which resets that.

    A decode EXCEPTION is re-raised to the consumer after the frames before it
    (I40): reporting it as the end of the video made a failed run read
    "Tracking complete".
    """

    _EOF = object()

    def __init__(self, src: "VideoSource", depth: int = 2):
        self._src = src
        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="kinetrace-readahead")
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                nxt = self._src.read_next()
                if nxt is None:
                    self._put(self._EOF)
                    return
                if not self._put(nxt):
                    return
        except Exception as e:   # noqa: BLE001 — re-raised in the consumer's read_next
            self._exc = e
            self._put(self._EOF)

    def _put(self, item) -> bool:
        """Block until there is room, but stay responsive to stop()."""
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def read_next(self) -> tuple[int, np.ndarray] | None:
        """Same contract as `VideoSource.read_next`: (index, RGB) or None."""
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is self._EOF:
                if self._exc is not None:
                    raise self._exc
                return None
            return item
        return None

    def stop(self) -> None:
        self._stop.set()
        while True:           # unblock a producer parked in put()
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        # (I39) wait until the producer has LEFT src.read_next(), however long
        # one read stalls (a network share can hold a 4K read for seconds).
        # Every caller seeks, reuses or closes src right after this; a bounded
        # join handed back a capture the producer was still inside, which files
        # the stale frame under the new seek target or crashes in release().
        while self._thread.is_alive():
            self._thread.join(timeout=0.25)


class SeekService(QThread):
    """Owns the scrubbing VideoSource; the GUI thread never decodes.

    Latest-wins: request(idx) overwrites any pending target, so scrub spam
    only ever decodes the newest position.

    A failed read is retried once on a FRESH capture before anything is
    reported (I40): a transient read error (a network share hiccup) used to
    shorten the video for the rest of the session. `n_frames` is the VERIFIED
    count (`probe_video`); a frame that still fails is a `decode_failed` (the
    app always passes the verified count, so the end of the file is never
    guessed here). An exception never ends the thread: it is reported and the
    next request is served.
    """

    frame_ready = Signal(int, object)  # (frame index, RGB ndarray)
    seek_slow = Signal(int)            # decode in progress for idx (show "Seeking...")
    decode_failed = Signal(int, str)   # (frame, reason) a frame that exists could not be
                                       # decoded twice; frame -1 = the video would not open

    def __init__(self, path: str, cache: FrameCache, n_frames: int | None = None):
        super().__init__()
        self._path = path
        self._cache = cache
        self._n_frames = int(n_frames) if n_frames else None
        self._cond = threading.Condition()
        self._target: int | None = None
        self._stop = False

    def request(self, idx: int) -> None:
        with self._cond:
            self._target = idx
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self.wait(5000)

    def _fetch(self, src: "VideoSource | None", idx: int):
        """`get_frame` with one retry on a freshly opened capture (I40).
        Returns (frame or None, error text or None, the source to keep using —
        None when even reopening failed; the next request tries again)."""
        err = None
        if src is not None:
            try:
                frame = src.get_frame(idx)
                if frame is not None:
                    return frame, None, src
            except Exception as e:      # noqa: BLE001 — retried below, then reported
                err = f"{type(e).__name__}: {e}"
            src.close()
            src = None
        try:
            src = VideoSource(self._path, self._cache)
            frame = src.get_frame(idx)
            if frame is not None:
                return frame, None, src
        except Exception as e:          # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        return None, err, src

    def run(self) -> None:
        src = None
        try:
            src = VideoSource(self._path, self._cache)
        except Exception as e:          # noqa: BLE001 — said, and retried on the next request
            self.decode_failed.emit(-1, f"the video could not be opened for display: {e}")
        try:
            while True:
                with self._cond:
                    while self._target is None and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        return
                    idx = self._target
                    self._target = None

                if self._cache.get(idx) is None:
                    self.seek_slow.emit(idx)
                frame, err, src = self._fetch(src, idx)

                # a newer request may have arrived while decoding; serve it next loop
                with self._cond:
                    superseded = self._target is not None
                if frame is None:
                    self.decode_failed.emit(
                        idx, err or "the decoder returned no picture (tried twice)")
                elif not superseded:
                    self.frame_ready.emit(idx, frame)
                    # idle prefetch: warm the frames a forward step / playback
                    # will want next; aborts the moment a new request lands
                    for j in range(idx + 1, idx + 1 + PREFETCH_AHEAD):
                        with self._cond:
                            if self._target is not None or self._stop:
                                break
                        try:
                            if self._cache.get(j) is None and src.get_frame(j) is None:
                                break  # EOF during prefetch is not an error
                        except Exception:   # noqa: BLE001 — opportunistic; a request retries
                            break
        finally:
            if src is not None:
                src.close()
