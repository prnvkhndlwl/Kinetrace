"""Overlay video render: the tracked result burned into an MP4.

For talks, for a collaborator without the app, and for QC at a glance: every
frame of a chosen range is decoded, the markers / names / skeleton bones /
silhouette / trails / frame counter / active events / notes are drawn on it
with OpenCV, and the result is written with `cv2.VideoWriter`.

`draw_overlay` is a pure function (numpy + cv2, no Qt) so it is testable and
reusable; `OverlayRenderer` is the QThread that runs it over a frame range
with its OWN VideoCapture (one capture per thread — the rule everywhere in
this app). Coordinates are native video pixels; the output can be scaled
down (`OverlayOptions.scale`) and everything drawn scales with it.

Codec: H.264 (`avc1`) is tried first because it plays everywhere; OpenCV's
bundled FFmpeg may not carry an encoder for it, in which case the writer
falls back to MPEG-4 part 2 (`mp4v`), which every player also opens.
"""

from __future__ import annotations

import sys
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class OverlayOptions:
    start: int = 0
    end: int = 0                 # inclusive
    scale: float = 1.0           # output size factor (1.0 = native)
    markers: bool = True
    names: bool = True
    bones: bool = True
    mask: bool = True
    trails: int = 30             # frames of trail, 0 = none
    frame_number: bool = True
    events: bool = True
    notes: bool = True
    marker_px: int = 7           # marker radius in OUTPUT pixels
    fps: float | None = None     # None = the video file's own rate (the session's `file_fps`)
    slowdown: float | None = None    # k = "k x slower than real time" (G194); None = as the file plays
    speed_label: bool = True     # draw "4x slower" under the frame number (nothing at real time)


# (G194) a player shows smooth motion from about 24 pictures a second (film's 23.976 counts as 24, so
# it is not doubled for nothing); above 60 most screens drop them
MIN_PLAY_FPS = 23.5
MAX_PLAY_FPS = 60.0


def playback_plan(real_fps: float, file_fps: float, slowdown: float) -> tuple[int, int, float]:
    """How to write a video `slowdown` times slower than real time (G194) from a camera that recorded
    `real_fps` frames a second, into a file that would play at `file_fps`: (step, repeat, out_fps).
    Source frames must play at r = real_fps / slowdown a second. Above max(file_fps, 60) (a high-speed
    clip sped up) every `step`-th frame is kept; below 24 each frame is written `repeat` times so
    players stay smooth. Either way 1 s of action lasts exactly `slowdown` s, and no frame is blended
    or invented."""
    r = float(real_fps) / float(slowdown)
    top = max(float(file_fps), MAX_PLAY_FPS)
    if r > top:
        step = int(np.ceil(r / top - 1e-9))
        return step, 1, r / step
    if r < MIN_PLAY_FPS:
        repeat = int(np.ceil(MIN_PLAY_FPS / r - 1e-9))
        return 1, repeat, r * repeat
    return 1, 1, r


def _times(k: float) -> str:
    return f"{k:.0f}" if k >= 100 else f"{k:.3g}"


def speed_words(slowdown: float | None) -> str:
    """'4x slower' / '16.7x slower' / '1.25x faster' against real time; '' at real time or unknown."""
    if slowdown is None or not is_rate(slowdown) or abs(float(slowdown) - 1.0) < 0.005:
        return ""
    k = float(slowdown)
    return f"{_times(k)}x slower" if k > 1.0 else f"{_times(1.0 / k)}x faster"


def _nth(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def slowdown_sentence(real_fps: float | None, file_fps: float, slowdown: float | None, n: int) -> str:
    """The overlay dialog's line under Speed (G194): how fast the video plays against real time and how
    long `n` source frames last. `slowdown` None = as the file plays; `real_fps` None = not known."""
    if real_fps is None or not is_rate(real_fps):
        return (f"Plays as the file does: these {n} frames make a {n / file_fps:.1f} s video. The camera's "
                "real frame rate is not known: set it with the fps button in the CAMERAS panel to slow "
                "the video against real time.")
    k = real_fps / file_fps if slowdown is None else float(slowdown)
    words = speed_words(k)
    secs = n / real_fps * k
    if not words:
        text = f"Plays at real time: these {n} frames make a {secs:.1f} s video."
    elif slowdown is None:
        text = (f"The file already plays {words} than real time: these {n} frames make a {secs:.1f} s "
                "video.")
    else:
        text = (f"Plays {words} than real time: 1 s of action lasts {_times(k)} s; these {n} frames make a "
                f"{secs:.1f} s video.")
    if slowdown is not None:
        step = playback_plan(real_fps, file_fps, k)[0]
        if step > 1:
            text += (f" It keeps every {_nth(step)} frame: a player shows at most "
                     f"{max(file_fps, MAX_PLAY_FPS):g} pictures a second.")
    return text


def _bgr(color) -> tuple[int, int, int]:
    r, g, b = (int(c) for c in color)
    return (b, g, r)


def _font_scale(w: int) -> float:
    """The overlay's text size for a picture `w` pixels wide as written (G123: `w` is already the
    scaled width -- scaling by the output scale again gave a half-size 4K overlay the minimum font)."""
    return max(0.4, 0.55 * (w / 1280.0) ** 0.5) if w >= 640 else 0.4


def _text(img, text: str, org: tuple[int, int], scale: float, color, thick: int = 1) -> None:
    """Outlined text (dark halo) so it reads on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(bgr: np.ndarray, frame: int, session, opts: OverlayOptions,
                 bones: list[tuple[int, int]] | None = None,
                 mask_color=(77, 227, 176), mask_opacity: float = 0.35,
                 slowdown: float | None = None, frame_word: bool = True) -> np.ndarray:
    """Draw the session's state at `frame` on a BGR image (native size or a
    scaled copy — `opts.scale` says which). Returns the image (drawn in place
    when it is already the output size, else the scaled copy). `slowdown` = how
    many times slower than real time the video plays (G194): labelled under the
    frame number when `opts.speed_label`, nothing at real time. `frame_word`
    False leaves "frame N" out of the counter line and keeps the time: a tile
    of several cameras has its frame in its title bar (G196)."""
    sc = float(opts.scale)
    if sc != 1.0:
        bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    out = bgr
    h, w = out.shape[:2]
    font_scale = _font_scale(w)
    s = session
    T = s.n_frames
    if not (0 <= frame < T):
        return out
    r = max(2, int(round(opts.marker_px)))

    # silhouettes (under everything else): every animal in its colour (G149); a frozen copy carries
    # `segs`, a live session gives its animals and their masks, so both draw every animal alike
    segs = getattr(s, "segs", None)
    if segs is None:
        segs = list(zip(getattr(s, "segments", []), getattr(s, "seg_masks", [])))
    for animal, masks in (segs if opts.mask else []):
        polys = masks.contours.get(frame) if masks is not None else None
        if polys:
            col = _bgr(getattr(animal, "color", mask_color))
            pts = [np.round(np.asarray(p, np.float64) * sc).astype(np.int32).reshape(-1, 1, 2)
                   for p in polys if len(p) >= 3]
            if pts:
                # blend only the polygons' bounding box (clipped to the picture): outside it the
                # filled layer equals the frame, so a whole-frame blend changed nothing there
                bx, by, bw, bh = cv2.boundingRect(np.concatenate(pts))
                x0, y0, x1, y1 = max(bx, 0), max(by, 0), min(bx + bw, w), min(by + bh, h)
                if x0 < x1 and y0 < y1:
                    roi = out[y0:y1, x0:x1]
                    layer = roi.copy()
                    cv2.fillPoly(layer, pts, col, offset=(-x0, -y0))
                    out[y0:y1, x0:x1] = cv2.addWeighted(layer, float(mask_opacity), roi,
                                                        1.0 - float(mask_opacity), 0)
                cv2.polylines(out, pts, True, col, 1, cv2.LINE_AA)

    occ = s.occluded[frame] if getattr(s, "occluded", None) is not None else np.zeros(s.n_points, bool)
    pos = s.tracks[frame] * sc
    valid = s.tracked[frame] & np.isfinite(s.tracks[frame]).all(axis=1)
    valid &= ~occ                       # cells marked hidden by hand are not drawn

    # trails
    if opts.trails > 0:
        lo = max(0, frame - int(opts.trails))
        for j, meta in enumerate(s.points):
            if not meta.display:
                continue
            seg = s.tracks[lo:frame + 1, j]
            ok = s.tracked[lo:frame + 1, j] & np.isfinite(seg).all(axis=1)
            if s.occluded is not None:
                ok &= ~s.occluded[lo:frame + 1, j]
            if ok.sum() < 2:
                continue
            col = _bgr(meta.color)
            pts = np.round(seg * sc).astype(np.int32)
            # break the polyline at gaps
            run: list[np.ndarray] = []
            for k in range(len(pts)):
                if ok[k]:
                    run.append(pts[k])
                elif len(run) >= 2:
                    cv2.polylines(out, [np.asarray(run).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
                    run = []
                else:
                    run = []
            if len(run) >= 2:
                cv2.polylines(out, [np.asarray(run).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)

    # bones
    if opts.bones and bones:
        for a, b in bones:
            if a < s.n_points and b < s.n_points and valid[a] and valid[b]:
                pa = tuple(int(v) for v in np.round(pos[a]))
                pb = tuple(int(v) for v in np.round(pos[b]))
                cv2.line(out, pa, pb, (235, 235, 235), 1, cv2.LINE_AA)

    # markers + names
    if opts.markers:
        for j, meta in enumerate(s.points):
            if not meta.display or not valid[j]:
                continue
            c = (int(round(pos[j][0])), int(round(pos[j][1])))
            col = _bgr(meta.color)
            if s.visibility[frame, j]:
                cv2.circle(out, c, r, col, -1, cv2.LINE_AA)
                cv2.circle(out, c, r, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, c, r, col, 2, cv2.LINE_AA)
            if meta.kind == "group":
                ol = meta.outline_at(pos[j] / sc) if hasattr(meta, "outline_at") else None
                if ol is not None:
                    cv2.polylines(out, [np.round(ol * sc).astype(np.int32).reshape(-1, 1, 2)], True,
                                  col, 1, cv2.LINE_AA)
                else:
                    cv2.circle(out, c, int(round(meta.radius * sc)), col, 1, cv2.LINE_AA)
            if opts.names:
                _text(out, meta.name, (c[0] + r + 3, c[1] - r - 2), font_scale, col)

    # frame counter + events + notes (top-left block, bottom-left notes)
    y = int(22 * max(font_scale / 0.55, 0.8)) + 4
    line_h = int(24 * max(font_scale / 0.55, 0.8))
    if opts.frame_number:
        if frame_word:
            t = f"frame {frame}"
            if s.fps:
                t += f"   {frame / s.fps:8.3f} s"
        else:
            t = f"{frame / s.fps:.3f} s" if s.fps else ""
        if t:
            _text(out, t, (8, y), font_scale, (255, 255, 255))
            y += line_h
    speed = speed_words(slowdown) if opts.speed_label else ""
    if speed:
        _text(out, speed, (8, y), font_scale, (255, 255, 255))
        y += line_h
    if opts.events:
        for e in s.events:
            if e.start <= frame <= e.end:
                _text(out, f"{e.name}  [{e.start}-{e.end}]", (8, y), font_scale, _bgr(e.color))
                y += line_h
    if opts.notes:
        n = s.notes.get(frame) if getattr(s, "notes", None) else None
        if n:
            _text(out, "note: " + str(n.get("text", ""))[:120], (8, h - 10), font_scale, (120, 255, 200))
    return out


def freeze_session(s, start: int = 0, end: int | None = None) -> SimpleNamespace:
    """A private copy of everything `draw_overlay` reads, taken on the GUI
    thread (I38). The renderer draws from its own thread while the app stays
    editable, and the session's mutators swap arrays one attribute at a time
    (a point is appended to `points` before the arrays grow), so drawing from
    the live session crashed on "index 10 is out of bounds" or mixed pre- and
    post-edit data. Silhouettes are copied only for `start..end`, the frames
    the render draws; the arrays whole (trails look back before `start`)."""
    end = s.n_frames - 1 if end is None else int(end)

    def frozen(m):
        return SimpleNamespace(
            contours={f: [np.array(p, copy=True) for p in ps]
                      for f, ps in m.contours.items() if start <= f <= end},
            midline={f: np.array(mm, copy=True)
                     for f, mm in m.midline.items() if start <= f <= end})
    segs = [(a.copy(), frozen(m)) for a, m in zip(getattr(s, "segments", []), getattr(s, "seg_masks", []))]
    occ = getattr(s, "occluded", None)
    return SimpleNamespace(
        n_frames=s.n_frames, n_points=s.n_points, fps=s.fps, width=s.width, height=s.height,
        file_fps=getattr(s, "file_fps", s.fps),
        tracks=s.tracks.copy(), tracked=s.tracked.copy(), visibility=s.visibility.copy(),
        occluded=None if occ is None else occ.copy(),
        points=[p.copy() for p in s.points],
        events=[copy.copy(e) for e in s.events],
        notes=copy.deepcopy(getattr(s, "notes", None) or {}),
        segs=segs)


def is_rate(fps: float) -> bool:
    """A frame rate a video can be written at: a finite number above 0 (I226 overlay, I276 side-by-side)."""
    return bool(np.isfinite(fps) and fps > 0)


def open_writer(path: str | Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 when the bundled FFmpeg can encode it, else MPEG-4. Returns the
    writer and the codec tag that worked."""
    path = str(path)
    # Windows Media Foundation encodes H.264 natively; asking for it FIRST
    # avoids OpenCV's FFmpeg backend logging a failed libopenh264 attempt
    # before it falls through to MSMF on its own
    attempts = [("avc1", cv2.CAP_ANY), ("mp4v", cv2.CAP_ANY)]
    if sys.platform.startswith("win"):           # Media Foundation exists only on Windows
        attempts.insert(0, ("avc1", cv2.CAP_MSMF))
    for tag, api in attempts:
        vw = cv2.VideoWriter(path, api, cv2.VideoWriter_fourcc(*tag), float(fps),
                             (int(size[0]), int(size[1])))
        if vw.isOpened():
            return vw, tag
        vw.release()
    raise RuntimeError("OpenCV could not open a video writer for " + path)


class VideoWriteJob(QThread):
    """Writes frames of a video, each drawn by the subclass, into `out_path` (R28: the overlay and the
    side-by-side video had one loop in two copies). Emits progress (done, total), then
    finished_ok(path, codec) or error(message); `request_cancel` stops after the current frame and
    deletes the partial file. After `finished_ok`, `frames_written` is how many SOURCE frames are really in the
    file (each written `_repeat` times in a slow-motion overlay, G194) and `note` says why it is short of what was asked ("" when it is not) -- a video can end before
    the range the session believed it had (I41, I232). A truncated file is never a result.

    A subclass gives `_reader()` (it may refuse in words first), `_plan(reader)` -> (frames, first,
    fps, size), `_compose(bgr, frame)`, and the sentences `_none_written(first)` /
    `_short_sentence(reader, frames, done)` / `_error_text(exc)`."""
    progress = Signal(int, int)
    finished_ok = Signal(str, str)
    error = Signal(str)

    def __init__(self, out_path: str):
        super().__init__()
        self.out_path = str(out_path)
        self.frames_written = 0
        self.note = ""
        self._cancel = False
        self._repeat = 1            # times each composed frame is written (a slow-motion overlay, G194)

    def request_cancel(self) -> None:
        self._cancel = True

    def _discard(self) -> None:
        try:
            Path(self.out_path).unlink(missing_ok=True)
        except OSError:
            pass

    def run(self) -> None:
        reader = vw = None
        created = False             # the writer made / truncated out_path: ours to delete
        try:
            reader = self._reader()
            frames, first, fps, size = self._plan(reader)
            total = len(frames)
            vw, codec = open_writer(self.out_path, fps, size)
            created = True
            reader.reset(first)
            done = 0
            for f in frames:
                if self._cancel:
                    break
                bgr = reader.read(f)
                if bgr is None:
                    break
                img = self._compose(bgr, f)
                if img.shape[1] != size[0] or img.shape[0] != size[1]:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                for _ in range(self._repeat):
                    if self._cancel:        # a 1000x slower frame is hundreds of writes
                        break
                    vw.write(img)
                done += 1
                if done % 5 == 0 or done == total:
                    self.progress.emit(done, total)
            vw.release()
            vw = None
            self.frames_written = done
            if self._cancel:
                self._discard()
                self.error.emit("cancelled")
                return
            if done == 0:
                # (I41) a range wholly past the frames that decode: an empty MP4 reported as
                # "written" is worse than saying so
                self._discard()
                self.error.emit(self._none_written(first))
                return
            if done < total:
                # (I41) the loop used to stop at the first failed read and report the REQUESTED count
                self.note = (self.note + "; " if self.note else "") + self._short_sentence(reader, frames, done)
                self.progress.emit(done, total)
            self.finished_ok.emit(self.out_path, codec)
        except Exception as e:      # noqa: BLE001
            if vw is not None:
                vw.release()
                vw = None
            if created:
                self._discard()     # a truncated file is not a result (I232)
            self.error.emit(self._error_text(e))
        finally:
            if vw is not None:
                vw.release()
            if reader is not None:
                reader.release()


class OverlayRenderer(VideoWriteJob):
    """Renders `opts.start..opts.end` of the session's video with the overlay
    into `out_path`. Emits progress (done, total), then finished_ok(path,
    codec) or error(message). `request_cancel` stops after the current
    frame and deletes the partial file.

    Built on the GUI thread: it draws from a FROZEN copy of the session
    (`freeze_session`, I38), so edits made while it renders are not in the
    video and cannot crash it. After `finished_ok`, `frames_written` is the
    number of source frames really in the file and `note` a sentence saying why it is
    short of the range ("" when it is not) — a video can end before the range
    the session believed it had (I41)."""

    def __init__(self, session, video_path: str, out_path: str, opts: OverlayOptions,
                 bones: list[tuple[int, int]] | None, mask_opacity: float = 0.35):
        super().__init__(out_path)
        self.session = freeze_session(session, opts.start, opts.end)
        self.video_path = str(video_path)
        self.opts = opts
        # (I44) the dialog's "Skeleton bones" tick decides, not the canvas's
        # View toggle: with the tick on, an empty list means the skeleton's bones
        if opts.bones and not bones and hasattr(session, "bones"):
            bones = session.bones()
        self.bones = list(bones or [])
        self.mask_opacity = float(mask_opacity)
        self.slowdown: float | None = None      # times slower than real time, known once planned (G194)

    def _output_fps(self, header_fps: float) -> float:
        """The rate the overlay plays at (I226): the caller's `opts.fps`, else the
        rate the app measured for the video FILE (`file_fps`: from its timestamps
        when the header is wrong), else the container header's. A rate that is not a
        positive finite number is refused with a sentence -- NaN used to give a
        one-frame MP4 reported as complete."""
        if self.opts.fps is not None:
            candidates = [("the requested frame rate", self.opts.fps)]
        else:
            candidates = [("the frame rate measured for this video", getattr(self.session, "file_fps", None)),
                          ("the frame rate in the session", getattr(self.session, "fps", None)),
                          ("the frame rate in the video's header", header_fps)]
        for what, v in candidates:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if is_rate(v):
                return v
            if self.opts.fps is not None:
                raise RuntimeError(f"{what} is {v:g}: it must be a number above 0 frames per second")
        raise RuntimeError("the video's frame rate is not known (none is stored in its file and none was "
                           "measured), so the overlay cannot be timed. Set the camera's frame rate with the "
                           "fps button in the CAMERAS panel first.")

    def _reader(self):
        from kinetrace.video_source import FrameReader      # the one shared frame reader (R17)
        return FrameReader(self.video_path, int(self.opts.start))

    def _real_fps(self) -> float | None:
        """The rate the camera really recorded at (the fps button, G38), or None when it is not a rate."""
        try:
            v = float(getattr(self.session, "fps", None))
        except (TypeError, ValueError):
            return None
        return v if is_rate(v) else None

    def _plan(self, reader):
        f0, f1 = int(self.opts.start), int(self.opts.end)
        cap = reader.cap
        fps = self._output_fps(float(cap.get(cv2.CAP_PROP_FPS)))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sc = float(self.opts.scale)
        size = (max(2, int(round(w * sc))), max(2, int(round(h * sc))))
        real = self._real_fps()
        k = self.opts.slowdown
        if k is None:                       # as the file plays: today's rate, every frame once
            self.slowdown = real / fps if real is not None else None
            return range(f0, f1 + 1), f0, fps, size
        if real is None:
            raise RuntimeError("the camera's real frame rate is not known, so the video cannot be slowed "
                               "against real time. Set it with the fps button in the CAMERAS panel, or "
                               "choose 'As the file plays'.")
        if not is_rate(k):
            raise RuntimeError(f"the speed {k:g} is not a number above 0")
        step, self._repeat, out_fps = playback_plan(real, fps, float(k))
        self.slowdown = float(k)
        return range(f0, f1 + 1, step), f0, out_fps, size

    def _compose(self, bgr, f):
        return draw_overlay(bgr, f, self.session, self.opts, self.bones, mask_opacity=self.mask_opacity,
                            slowdown=self.slowdown)

    def _none_written(self, first: int) -> str:
        return (f"frame {first} could not be decoded (the video ends before it, or the file is damaged "
                "there), so nothing was written. Choose a range inside the video.")

    def _short_sentence(self, reader, frames, done: int) -> str:
        # `frames` may skip (a high-speed clip sped up, G194): name the frames, not f0 + done
        bad = reader.failed if reader.failed is not None else frames[done]
        return (f"Frame {bad} could not be decoded (the video ends there, or the file is damaged), so "
                f"this overlay stops at frame {frames[done - 1]}: {done} of {len(frames)} frames were written.")

    def _error_text(self, e: BaseException) -> str:
        return str(e)


# ------------------------------------------------------------------ several cameras in one video (G195)

# the grid of cameras fits a 4K frame: many players refuse anything wider than 4096 px
MAX_GRID_W, MAX_GRID_H = 3840, 2160
BAR_BGR = (0x23, 0x1E, 0x1E)        # a title bar: the window's chrome (theme.BG_WINDOW #1E1E23, as BGR)


def bar_height(cell_w: int) -> int:
    """A title bar's height over a cell `cell_w` px wide (G196): its words' height + 8 px, even."""
    (_, th), base = cv2.getTextSize("Ag", FONT, _font_scale(cell_w), 1)
    return (th + base + 8 + 1) // 2 * 2


def tile_layout(sizes: list[tuple[int, int]], cols: int, rows: int, cell_aspect: float,
                out_scale: float = 1.0):
    """Where each camera goes in a multi-view overlay (G195, G196): every cell has the window's cell
    shape (`cell_aspect` = w / h) with a title bar above it. A cell is the smallest of that shape that
    holds every camera at its own size (nothing is enlarged past native), the grid with its bars is
    scaled to fit MAX_GRID_W x MAX_GRID_H and then by `out_scale`, both sides padded up to even
    (H.264). Each camera is fit in its cell (aspect kept, centred), as the window shows it. Returns
    the canvas (W, H), per camera (x, y, w, h, scale) and per camera its bar (x, y, w, h), in `sizes`
    order, filling the grid row by row."""
    a = float(cell_aspect)
    wc = max(max(float(w), float(h) * a) for w, h in sizes)
    hc = wc / a
    # fitted with the native cell's bar: the bar only shrinks with the cell, so the grid stays inside
    bar0 = bar_height(int(wc))
    s = min(1.0, MAX_GRID_W / (cols * wc), max(1.0, MAX_GRID_H - rows * bar0) / (rows * hc)) * float(out_scale)
    cell_w, cell_h = max(2, int(wc * s)), max(2, int(hc * s))
    bar = bar_height(cell_w)
    W, H = cols * cell_w, rows * (bar + cell_h)
    rects, bars = [], []
    for k, (w, h) in enumerate(sizes):
        c, r = k % cols, k // cols
        sc = min(cell_w / float(w), cell_h / float(h))
        tw, th = min(cell_w, int(round(w * sc))), min(cell_h, int(round(h * sc)))
        y0 = r * (bar + cell_h)
        rects.append((c * cell_w + (cell_w - tw) // 2, y0 + bar + (cell_h - th) // 2, tw, th, sc))
        bars.append((c * cell_w, y0, cell_w, bar))
    return (W + W % 2, H + H % 2), rects, bars


class _ViewsReader:
    """The frames of several cameras for each frame of the working camera (G195): the reader protocol
    `VideoWriteJob.run` uses (`cap`, `reset`, `read`, `failed`, `release`). One `FrameReader` per camera,
    opened here on the worker at that camera's first matched frame. `read(f)` returns None when the
    WORKING camera's frame does not decode (the run stops there, I41), else per camera ("ok", image) /
    ("off", None: not recording at that instant) / ("bad", frame that did not decode). `tables[k]` maps
    the working camera's frame f to camera k's frame (`table[f - f0]`, -1 = not recording; None for the
    working camera): `Project.map_frame`, frozen on the GUI thread. A camera's frame asked twice in a
    row (a slower camera) is the last image again; a frame that fails is reported once in `notes`."""

    def __init__(self, views, active: int, f0: int):
        from kinetrace.video_source import FrameReader, VideoOpenError
        self.views, self.active, self.f0 = views, int(active), int(f0)
        self.readers: list = []
        self.notes: list[str] = []
        self._last = [(-1, None)] * len(views)          # (frame, image) last read per camera
        self._bad: list[int | None] = [None] * len(views)
        try:
            for k, v in enumerate(views):
                if k == self.active:
                    first = self.f0
                else:
                    valid = v.table[v.table >= 0]
                    first = int(valid[0]) if len(valid) else None
                try:
                    self.readers.append(None if first is None else FrameReader(v.path, first))
                except VideoOpenError:
                    raise RuntimeError(f"Camera {v.camera + 1} ({v.name}): its video could not be opened "
                                       f"({v.path}), so nothing was written.") from None
        except Exception:
            self.release()
            raise

    @property
    def cap(self):
        return self.readers[self.active].cap

    @property
    def failed(self):
        return self.readers[self.active].failed

    def reset(self, first: int) -> None:
        self.readers[self.active].reset(first)

    def read(self, f: int):
        main = self.readers[self.active].read(f)
        if main is None:
            return None
        out = []
        for k, v in enumerate(self.views):
            if k == self.active:
                out.append(("ok", main))
                continue
            lf = int(v.table[f - self.f0])
            if lf < 0:
                out.append(("off", None))
            elif self._bad[k] is not None:
                out.append(("bad", self._bad[k]))
            elif lf == self._last[k][0]:
                out.append(("ok", self._last[k][1]))
            else:
                img = self.readers[k].read(lf)
                if img is None:
                    bad = self.readers[k].failed if self.readers[k].failed is not None else lf
                    self._bad[k] = bad
                    self.notes.append(f"Camera {v.camera + 1} ({v.name}): frame {bad} could not be decoded "
                                      "(its video ends there, or the file is damaged), so its tile is dark "
                                      "from there")
                    out.append(("bad", bad))
                else:
                    self._last[k] = (lf, img)
                    out.append(("ok", img))
        return out

    def release(self) -> None:
        for r in self.readers:
            if r is not None:
                r.release()
        self._last = [(-1, None)] * len(self.views)     # (review) one full frame per camera, not kept


class ViewsOverlayRenderer(OverlayRenderer):
    """The overlay of several cameras in one video (G195): `views` = [(session, video_path, name, bones,
    table, camera index, offset)] in their order on screen, the working camera's `table` None (it is the
    clock: the range, the Speed and the frame rate are its). `cell` = (w, h) of a view on screen
    (`ViewGrid.cell_size`): the video's cells take its shape, as the window shows them (G196; None =
    the working camera's shape). Every camera is drawn with `draw_overlay` in its cell at the frame
    `table` gives, under a title bar with the window's caption words (name, frame, offset); the tiles
    leave out "frame N" (the bar has it) and keep the time; "4x slower" only in the working camera's
    tile. A camera not recording at an instant is a dark tile saying so."""

    def __init__(self, views, out_path: str, opts: OverlayOptions, mask_opacity: float = 0.35,
                 cell: tuple[int, int] | None = None):
        active = next(k for k, v in enumerate(views) if v[4] is None)
        s, path, _name, bones, _table, _cam, _off = views[active]
        super().__init__(s, path, out_path, opts, bones, mask_opacity)
        self.active = active
        self.cell = None if cell is None else (int(cell[0]), int(cell[1]))
        self.views = []
        for k, (sv, pv, name, bv, table, cam, off) in enumerate(views):
            if k == active:
                frozen, bv = self.session, self.bones
            else:
                table = np.asarray(table, np.int64)
                valid = table[table >= 0]
                frozen = freeze_session(sv, int(valid.min()) if len(valid) else 0,
                                        int(valid.max()) if len(valid) else -1)
                if opts.bones and not bv and hasattr(sv, "bones"):      # I44, as for the working camera
                    bv = sv.bones()
            self.views.append(SimpleNamespace(session=frozen, path=str(pv), name=str(name),
                                              bones=list(bv or []), table=table, camera=int(cam),
                                              offset=float(off or 0.0)))
        self.tiles: list[tuple] = []        # (camera, x, y, w, h, scale) per tile, known once planned
        self.bars: list[tuple] = []         # (x, y, w, h) of each tile's title bar (G196)
        self._views_reader: _ViewsReader | None = None

    def _reader(self):
        self._views_reader = _ViewsReader(self.views, self.active, int(self.opts.start))
        return self._views_reader

    def _plan(self, reader):
        from kinetrace.viewgrid import grid_shape        # the window's grid rule (lazy: Qt widgets)
        frames, first, fps, _size = super()._plan(reader)
        sizes = []
        for v, r in zip(self.views, reader.readers):
            if r is not None:
                sizes.append((int(r.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(r.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            else:
                sizes.append((int(v.session.width), int(v.session.height)))
        cols, rows = grid_shape(len(self.views))
        cw, ch = self.cell if self.cell is not None else sizes[self.active]
        canvas, rects, self.bars = tile_layout(sizes, cols, rows, cw / float(ch), float(self.opts.scale))
        self.tiles = [(v.camera,) + tuple(r) for v, r in zip(self.views, rects)]
        self._canvas = canvas
        return frames, first, fps, canvas

    def _compose(self, items, f):
        from dataclasses import replace
        from kinetrace.viewgrid import caption_for       # the window's caption words (lazy: Qt widgets)
        W, H = self._canvas
        out = np.zeros((H, W, 3), np.uint8)
        for k, (v, (_cam, x, y, w, h, sc), (bx, by, bw, bh), (state, img)) in enumerate(
                zip(self.views, self.tiles, self.bars, items)):
            fs = _font_scale(w)
            lf = f if k == self.active else int(v.table[f - self.f0])
            if state == "ok":
                # (ui rule) drawing copies its frame first: a slower camera's image is shown twice
                tile = draw_overlay(img if k == self.active else img.copy(), lf, v.session,
                                    replace(self.opts, scale=sc), v.bones, mask_opacity=self.mask_opacity,
                                    slowdown=self.slowdown if k == self.active else None, frame_word=False)
                th, tw = min(h, tile.shape[0], H - y), min(w, tile.shape[1], W - x)
                out[y:y + th, x:x + tw] = tile[:th, :tw]
            else:
                # (review) the tile's words are written on a copy of its area and pasted back: clipped to
                # the tile, never spilling into a neighbour's
                area = out[y:y + h, x:x + w].copy()
                say = ("not recording at this instant" if state == "off"
                       else f"frame {img} could not be decoded")
                (tw, _), _ = cv2.getTextSize(say, FONT, fs, 1)
                _text(area, say, (max(4, (w - tw) // 2), h // 2), fs, (220, 220, 220))
                out[y:y + h, x:x + w] = area
            # (G196) the title bar above the picture, as the window's: its words clipped to the bar
            bar = np.empty((bh, bw, 3), np.uint8)
            bar[:] = BAR_BGR
            words = caption_for(v.name, lf if lf >= 0 else None, v.offset, v.session.n_frames, sep="|")
            bfs = _font_scale(bw)
            (_, th), _ = cv2.getTextSize(words, FONT, bfs, 1)
            # plain text on the dark bar: `_text`'s halo is drawn thicker and drifts off the letters
            cv2.putText(bar, words, (6, (bh + th) // 2), FONT, bfs, (236, 232, 232), 1, cv2.LINE_AA)
            out[by:by + bh, bx:bx + bw] = bar[:H - by, :W - bx]
        if self._views_reader is not None and self._views_reader.notes:
            self.note = "; ".join(filter(None, [self.note] + self._views_reader.notes))
            self._views_reader.notes = []
        return out

    @property
    def f0(self) -> int:
        return int(self.opts.start)


# ------------------------------------------------------------------ dialog


class OverlayDialog:
    """Options for File → Export → Overlay Video. Built lazily as a QDialog so
    the pure-function part of this module imports without QtWidgets."""

    def __new__(cls, parent, session, default_path: str, current: int,
                sel_range: tuple[int, int] | None = None, n_frames: int | None = None,
                views: list[str] | None = None, together_path: str | None = None,
                together_has: tuple[bool, bool] = (False, False)):
        """`views`: the names of the cameras shown on screen, in their order (G195); two or more offer
        "All shown cameras together", whose suggested file is `together_path`. `together_has`: (any
        shown camera has silhouettes, any has skeleton bones) -- the Draw ticks follow the cameras the
        video will hold, not only the working one."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                                       QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                                       QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)
        from kinetrace import theme

        def _rate(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if is_rate(v) else None

        # (G194) speeds are against REAL time: the camera's rate (the fps button) vs the file's
        real = _rate(session.fps)
        file_rate = _rate(getattr(session, "file_fps", None)) or real
        k0 = real / file_rate if real is not None and file_rate is not None else None
        OTHER = "other"

        # (I41) the frames that DECODE (the view's verified count) bound every
        # range: a project saved with a header's over-count keeps the longer
        # session, and a range in that phantom tail wrote a short or empty MP4
        T = session.n_frames if not n_frames else max(1, min(int(session.n_frames), int(n_frames)))
        if sel_range is not None:
            sel_range = None if sel_range[0] > T - 1 else (sel_range[0], min(sel_range[1], T - 1))
        events = [(e, min(e.end, T - 1)) for e in session.events if e.start <= T - 1]
        # what the Silhouette / Skeleton bones ticks can draw: this camera's, or (together) any shown one's
        one_has = (session.masks is not None and session.animal is not None, bool(session.bones()))

        class _Dlg(QDialog):
            def __init__(self):
                super().__init__(parent)
                self.setWindowTitle("Export overlay video")
                self.setMinimumWidth(560)
                self.result_options: OverlayOptions | None = None
                self.result_path = ""
                self.result_together = False
                lay = QVBoxLayout(self)
                intro = QLabel("Writes an MP4 with the tracked result drawn on every frame: markers, "
                               "names, skeleton, silhouette, trails, the frame counter, the speed, "
                               "events and notes. The original video is never modified.")
                intro.setWordWrap(True)
                lay.addWidget(intro)
                form = QFormLayout()
                # (G195) this camera, or the cameras shown on screen together (their own widget: the
                # range's radio buttons are another group)
                shown = list(views or [])
                vbox = self._views_box = QWidget(self)      # owned: hidden for one camera, not freed
                vl = QVBoxLayout(vbox)
                vl.setContentsMargins(0, 0, 0, 0)
                self.r_one = QRadioButton("This camera")
                # (review) at most 4 names on the line: a long label sets the dialog's width
                listed = ", ".join(shown[:4]) + (", …" if len(shown) > 4 else "")
                self.r_together = QRadioButton(f"All shown cameras together ({len(shown)})"
                                               + (": " + listed if len(shown) > 1 else ""))
                self.r_together.setEnabled(len(shown) > 1)
                self.r_together.setToolTip(
                    "The cameras shown in the window (" + ", ".join(shown) + "), laid out as the window shows "
                    "them (grid, order, cell shape), each under its title bar with its own overlay; this "
                    "camera's frames are the clock and every other camera shows the frame that matches it "
                    "(its sync offset and rate)." if len(shown) > 1 else
                    "Only one camera is shown: show another with its eye in the CAMERAS panel (and Solo off) "
                    "to put them together.")
                self.r_one.setChecked(True)
                vl.addWidget(self.r_one)
                vl.addWidget(self.r_together)
                if views is not None:               # a one-camera project has nothing to choose
                    form.addRow("Views", vbox)
                else:
                    vbox.hide()
                # range
                rng = QWidget()
                rl = QVBoxLayout(rng)
                rl.setContentsMargins(0, 0, 0, 0)
                self.r_all = QRadioButton(f"Whole video (frames 0–{T - 1})")
                self.r_sel = QRadioButton("Selected frame window on the timeline"
                                          + (f" ({sel_range[0]}–{sel_range[1]})" if sel_range else ""))
                self.r_sel.setEnabled(sel_range is not None)
                self.r_ev = QRadioButton("An event:")
                self.ev_box = QComboBox()
                for e, end in events:
                    self.ev_box.addItem(f"{e.name}  [{e.start}–{end}]", (e.start, end))
                self.r_ev.setEnabled(bool(events))
                self.ev_box.setEnabled(bool(events))
                self.r_custom = QRadioButton("Frames:")
                self.f0 = QSpinBox()
                self.f0.setRange(0, T - 1)
                self.f1 = QSpinBox()
                self.f1.setRange(0, T - 1)
                self.f0.setValue(max(0, current - 150))
                self.f1.setValue(min(T - 1, current + 150))
                row_ev = QHBoxLayout()
                row_ev.addWidget(self.r_ev)
                row_ev.addWidget(self.ev_box, 1)
                row_c = QHBoxLayout()
                row_c.addWidget(self.r_custom)
                row_c.addWidget(self.f0)
                row_c.addWidget(QLabel("to"))
                row_c.addWidget(self.f1)
                row_c.addStretch(1)
                rl.addWidget(self.r_all)
                rl.addWidget(self.r_sel)
                rl.addLayout(row_ev)
                rl.addLayout(row_c)
                (self.r_sel if sel_range else self.r_all).setChecked(True)
                form.addRow("Range", rng)
                self.scale = QComboBox()
                for label, v in (("Full size", 1.0), ("Half size", 0.5), ("Quarter size", 0.25)):
                    self.scale.addItem(label, v)
                self._scale_auto = 1 if session.width > 2000 else 0
                self.scale.setCurrentIndex(self._scale_auto)
                self.scale.setToolTip("Half size is plenty for a talk and renders 4x faster")
                form.addRow("Output size", self.scale)
                # (G194) speed against real time: presets + any number; the line under it says how
                # long the video will be
                srow = QWidget()
                sl = QHBoxLayout(srow)
                sl.setContentsMargins(0, 0, 0, 0)
                self.speed = QComboBox()
                if k0 is None or not speed_words(k0):
                    self.speed.addItem("Real time (as the file plays)" if k0 is not None else
                                       "As the file plays", None)
                else:
                    self.speed.addItem(f"As the file plays ({speed_words(k0)})", None)
                    self.speed.addItem("Real time", 1.0)
                if real is not None:
                    for k in (2.0, 4.0, 5.0, 10.0, 20.0):
                        if k0 is None or abs(k - k0) > 0.005 * k:      # not the file's own speed twice
                            self.speed.addItem(f"{speed_words(k)}", k)
                    self.speed.addItem("Other:", OTHER)
                self.speed_x = QDoubleSpinBox()
                self.speed_x.setRange(1.0, 1000.0)
                self.speed_x.setDecimals(2)
                self.speed_x.setValue(3.0)
                self.speed_x.setSuffix("x slower")
                self.speed_x.setEnabled(False)
                sl.addWidget(self.speed, 1)
                sl.addWidget(self.speed_x)
                self.speed.setToolTip("How much slower than real time (the camera's own rate) the video "
                                      "plays. A high-speed camera's file may already play slower.")
                form.addRow("Speed", srow)
                self.speed_info = QLabel()
                self.speed_info.setWordWrap(True)
                self.speed_info.setStyleSheet(f"color: {theme.TEXT_DIM};")
                form.addRow("", self.speed_info)
                opts = QWidget()
                ol = QVBoxLayout(opts)
                ol.setContentsMargins(0, 0, 0, 0)
                self.c_markers = QCheckBox("Markers")
                self.c_names = QCheckBox("Point names")
                self.c_bones = QCheckBox("Skeleton bones")
                self.c_mask = QCheckBox("Silhouette")
                self.c_trails = QCheckBox("Trails (last 30 frames)")
                self.c_frame = QCheckBox("Frame number and time")
                self.c_events = QCheckBox("Event names while active")
                self.c_notes = QCheckBox("Frame notes")
                self.c_speed = QCheckBox("Speed (e.g. '4x slower'; nothing at real time)")
                for c in (self.c_markers, self.c_names, self.c_bones, self.c_mask, self.c_trails,
                          self.c_frame, self.c_speed, self.c_events, self.c_notes):
                    c.setChecked(True)
                    ol.addWidget(c)
                self.c_mask.setEnabled(one_has[0])
                self.c_bones.setEnabled(one_has[1])
                form.addRow("Draw", opts)
                self.marker = QSpinBox()
                self.marker.setRange(2, 30)
                self.marker.setValue(7)
                self.marker.setSuffix(" px")
                form.addRow("Marker size", self.marker)
                prow = QWidget()
                pl = QHBoxLayout(prow)
                pl.setContentsMargins(0, 0, 0, 0)
                self.path = QLineEdit(default_path)
                btn = QPushButton("Choose…")
                btn.clicked.connect(self._pick)
                pl.addWidget(self.path, 1)
                pl.addWidget(btn)
                form.addRow("Save as", prow)
                lay.addLayout(form)
                hint = QLabel("H.264 when this OpenCV build can encode it, otherwise MPEG-4; both play "
                              "in any player. Rendering runs in the background — you can keep working; "
                              "edits made while it renders are not in this video.")
                hint.setWordWrap(True)
                hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
                lay.addWidget(hint)
                btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                btns.button(QDialogButtonBox.Ok).setText("Render")
                btns.accepted.connect(self._accept)
                btns.rejected.connect(self.reject)
                lay.addWidget(btns)
                self.speed.currentIndexChanged.connect(self._speed_changed)
                self.speed_x.valueChanged.connect(self._update_speed_info)
                for rb in (self.r_all, self.r_sel, self.r_ev, self.r_custom):
                    rb.toggled.connect(self._update_speed_info)
                self.ev_box.currentIndexChanged.connect(self._update_speed_info)
                self.f0.valueChanged.connect(self._update_speed_info)
                self.f1.valueChanged.connect(self._update_speed_info)
                self._update_speed_info()
                self.r_together.toggled.connect(self._views_changed)

            def _views_changed(self, together: bool):
                # (review) a silhouette or skeleton in ANOTHER shown camera is drawable together
                has = (bool(together_has[0]) or one_has[0], bool(together_has[1]) or one_has[1]) \
                    if together else one_has
                self.c_mask.setEnabled(bool(has[0]))
                self.c_bones.setEnabled(bool(has[1]))
                # (review) Half size is the starting choice for ONE wide camera; the cameras together
                # are already fitted into 4K, so they start at Full -- unless the user picked a size
                auto = 0 if together else self._scale_auto
                if self.scale.currentIndex() == (self._scale_auto if together else 0):
                    self.scale.setCurrentIndex(auto)
                # the suggested file follows the choice, unless the user typed their own
                if together_path:
                    old, new = (default_path, together_path) if together else (together_path, default_path)
                    if self.path.text().strip() == old:
                        self.path.setText(new)

            def _slowdown(self) -> float | None:
                d = self.speed.currentData()
                return float(self.speed_x.value()) if d == OTHER else d

            def _speed_changed(self):
                self.speed_x.setEnabled(self.speed.currentData() == OTHER)
                self._update_speed_info()

            def _update_speed_info(self):
                if file_rate is None:
                    self.speed_info.setText("")
                    return
                a, b = self._range()
                self.speed_info.setText(slowdown_sentence(real, file_rate, self._slowdown(), b - a + 1))

            def _pick(self):
                p, _ = QFileDialog.getSaveFileName(self, "Overlay video", self.path.text(),
                                                   "MP4 video (*.mp4)")
                if p:
                    self.path.setText(p if p.lower().endswith(".mp4") else p + ".mp4")

            def _range(self) -> tuple[int, int]:
                if self.r_sel.isChecked() and sel_range:
                    return int(sel_range[0]), int(sel_range[1])
                if self.r_ev.isChecked() and self.ev_box.count():
                    a, b = self.ev_box.currentData()
                    return int(a), int(b)
                if self.r_custom.isChecked():
                    a, b = self.f0.value(), self.f1.value()
                    return (a, b) if a <= b else (b, a)
                return 0, T - 1

            def _accept(self):
                a, b = self._range()
                self.result_options = OverlayOptions(
                    start=a, end=b, scale=float(self.scale.currentData()),
                    markers=self.c_markers.isChecked(), names=self.c_names.isChecked(),
                    bones=self.c_bones.isChecked() and self.c_bones.isEnabled(),
                    mask=self.c_mask.isChecked() and self.c_mask.isEnabled(),
                    trails=30 if self.c_trails.isChecked() else 0,
                    frame_number=self.c_frame.isChecked(), events=self.c_events.isChecked(),
                    notes=self.c_notes.isChecked(), marker_px=self.marker.value(),
                    slowdown=self._slowdown(), speed_label=self.c_speed.isChecked())
                p = self.path.text().strip()
                self.result_path = p if p.lower().endswith(".mp4") else p + ".mp4"
                self.result_together = self.r_together.isChecked() and self.r_together.isEnabled()
                self.accept()

        return _Dlg()
