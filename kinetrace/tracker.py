"""CoTracker3 online tracking engine.

Runs Meta's CoTracker3 online model over a forward segment of the video in a
QThread, emitting per-window results so the GUI can render progression live.

Windowing contract (mirrors the model's online bookkeeping exactly):
- model.step == 8, window length 16.
- After 8 frames: one is_first_step=True call (initializes queries; no output).
- At every multiple of 8 frames: a processing call with the last 16 frames.
  The model rewrites its predictions for that whole window (the first half is
  a refined overlap), so we emit the full 16-row window and the session
  overwrites the overlap rows.
- Non-first processing chunks must be 9..16 frames long; EOF tails and short
  segments are handled explicitly below.

Coordinates: frames are downscaled to a working resolution (max dim ~1280) on
CPU before upload — the model resizes to ~512x384 internally, so accuracy is
unchanged while transfers shrink ~16x for 4K video. Model output is rescaled
to native pixels here; optionally refined to true subpixel by a gated
Lucas-Kanade pass on the native-resolution frames (chained from the exact
user seed, re-anchored to the CoTracker prediction every frame by the gate).

Robustness layers on top of the base pipeline (all per-run configurable):
- ROI zoom: when the tracked constellation is small relative to a high-res
  frame, the model is fed a crop around it instead of the whole frame, so
  small objects keep real texture at the model's internal ~512x384. The crop
  is fixed for a segment (online queries can't move mid-segment); when a
  point nears the crop edge the run transparently re-seeds a new segment
  from its own predictions at the last emitted frame.
- Region groups (kind == "group"): a circle of member points is tracked
  around the group seed; each frame a RANSAC similarity fit (member seeds ->
  member positions) yields the emitted center, so single-member failures are
  outliers, not track loss. Members are worker-internal and re-sampled fresh
  at every (re)seed.
- Confidence: the online wrapper binarizes visibility*confidence at 0.6 and
  discards the rest; processing calls here run the inner model directly to
  keep the raw confidence (visibility stays binarized with the exact same
  numerics). Confidence is the track-correctness score — an occluded point
  keeps high confidence, a *lost* point does not — which is what the
  auto-pause detector needs to stop at failures without pausing on ordinary
  occlusion.
- Appearance re-anchor (PointMeta.anchor, opt-in): an NCC template captured
  at the user's seed snaps the track back when a strong match disagrees —
  catches slow drift on distinct rigid features. Off by default because it
  changes the "track the placed location" semantics.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from kinetrace import alltracker_backend as at_backend
from kinetrace.segmenter import (DEFAULT_BACKEND, MIDLINE_SAMPLES, Prompt, get_segmenter,
                                     score_to_confidence, summarize_mask)
from kinetrace.silhouette import extremity_roles, midline as silhouette_midline, oriented, resample
from kinetrace.session import in_frame
from kinetrace.video_source import FrameCache, ReadAhead, VideoSource

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WORKING_MAX_DIM = 1280
# LK refinement is the largest non-overlapped cost at 4K (measured 174 ms per
# 16-frame window, 33% of wall — MORE than the model itself), because
# calcOpticalFlowPyrLK pyramids the whole 8-megapixel frame to polish a few
# points. Two ways to cut it were measured and NEITHER is currently lossless:
#   - cropping to the points' neighbourhood: 1.4x on LK, +15% overall, but the
#     cropped pyramid differs from the frame's own by 1.8e-3 px even with the
#     origin snapped to 2**maxLevel.
#   - reusing one pyramid per frame (exact by construction): OpenCV 5.0 dropped
#     the Python overload of calcOpticalFlowPyrLK that accepts a prebuilt
#     pyramid, so it cannot be expressed here.
# Left exact-but-slow deliberately; see docs/HANDOFF.md.
POINT_BACKENDS = ("cotracker3", "alltracker")
ALLTRACKER_MAX_DIM = 1024        # AllTracker correlates every pixel pair of a window: memory ~ (H*W/64)^2

# auto-pause: confidence below the threshold for this many consecutive
# in-frame frames = the model has lost the point. Occlusion also drops
# confidence, but only in SHORT dips (measured max 7-8 consecutive frames
# under a 46-frame opaque occlusion vs 130+ after a real loss) — the run
# length is the separator, so keep >= 2x the occlusion worst case.
CONF_PAUSE_THRESHOLD = 0.25
CONF_PAUSE_RUN = 16

# region groups
GROUP_MIN_FIT_RADIUS = 6.0   # below this the members are too close for a fit
GROUP_RANSAC_MIN_INLIERS = 3

# ROI zoom
ROI_MIN_W, ROI_MIN_H = 768, 576  # never crop tighter than this (over-zoom is useless)
ROI_MARGIN_FRAC = 0.35           # padding around the constellation bbox, each side
ROI_EDGE_FRAC = 0.12             # re-seed when a point enters this edge band
ROI_MIN_ZOOM = 2.0               # engage only when the crop at least doubles the
                                 # pixels on target — every re-seed costs a little
                                 # accuracy, so marginal zoom is a net loss
ROI_GROW_SEGMENT = 48            # a segment shorter than this means the crop is
                                 # too tight for the motion -> grow the next one
ROI_GROW_FACTOR = 1.5

# appearance re-anchor
ANCHOR_TEMPLATE = 25             # native-px template side (odd)
ANCHOR_MIN_CORR = 0.75           # NCC peak needed before a snap is trusted

# decode read-ahead of the point-tracking loop, in frames (I134): one model step's
# worth, so the decoder keeps going through each model / LK burst instead of
# stopping after 2 and making the worker wait for it afterwards. Only WHEN a frame
# is decoded changes (same frames, same order: coordinates bit-identical, measured);
# at most this many decoded frames wait in the queue (~200 MB at 4K). Real 4K clip:
# CoTracker3 27.5 -> 32.2 fps, AllTracker 13.0 -> 14.2 fps; 16 gains nothing more.
READ_AHEAD_FRAMES = 8

_model_lock = threading.Lock()
_model = None
_device: str | None = None


@dataclass
class PointSpec:
    """One user-facing tracked output: a plain point or a region-group center."""
    pid: int
    seed: np.ndarray                 # (2,) float32 native px
    kind: str = "point"              # "point" | "group"
    radius: float = 0.0              # group region radius, native px
    anchor: bool = False             # appearance re-anchor opt-in (points only)
    outline: np.ndarray | None = None  # rect/polygon region: (M, 2) vertex OFFSETS from the seed


@dataclass
class AnimalSpec:
    """The segment to segment during a run: the user's prompts per frame and,
    when resuming, the mask at the start frame (working resolution) that
    seeds the segmentation session."""
    prompts: dict = field(default_factory=dict)     # frame -> [(x, y, label)] native px
    boxes: dict = field(default_factory=dict)       # frame -> (x0, y0, x1, y1)
    seed_mask: np.ndarray | None = None             # bool, working res, at start_frame
    backend: str = DEFAULT_BACKEND


class BallSpec:
    """A ball marker (session source "ball"): SAM segments it, a circle is
    fitted per frame, the centre is the output (see balls.py). `prompts` =
    {frame: [[x, y, label], ...]} native px; `seed` = its position on the run's
    first frame when no prompt sits there (re-seeds from the previous run);
    `radius` = a size hint for the first prompt's box (None = unknown)."""

    def __init__(self, pid: int, prompts: dict, seed=None, radius: float | None = None,
                 backend: str = ""):
        self.pid = int(pid)
        self.prompts = {int(f): [list(c) for c in cs] for f, cs in (prompts or {}).items()}
        self.seed = None if seed is None else np.asarray(seed, np.float32)
        self.radius = None if radius is None or not np.isfinite(radius) or radius <= 0 else float(radius)
        self.backend = backend


@dataclass
class DerivedSpec:
    """An output column computed from the segment's silhouette, not tracked."""
    pid: int
    spec: str        # "tip" | "midline:<f>" | "centroid" | "ext:L|R|FL|FR|HL|HR"


SUPPORT_POINTS = 24        # extra CoTracker queries sampled inside the mask (never emitted)
OFF_BODY_CONF = 0.2        # confidence cap for a skeleton point that sits outside the mask
ANIMAL_LOST_RUN = 16       # frames of "not present" before auto-pause reports the animal lost
MASK_HIST = 96             # per-frame mask cache depth in the worker
SNAP_RESTART_FRAC = 0.02   # a snap longer than this fraction of the mask diagonal re-seeds CoTracker
SNAP_RESTART_MIN_PX = 6.0  # ... but never for sub-pixel jitter
# Landmark identity + derived-skeleton continuity (lessons from a real stereo test):
ANCHOR_SHORT_FRAC = 0.6    # an anchored midline shorter than this x the previous one = anchor mid-body (snout slid)
ANCHOR_END_FRAC = 0.2      # the head anchor must sit within this x body length of an END of the free midline
ANCHOR_CHECK_EVERY = 8     # frames between the (costlier) free-midline anchor checks; a short midline forces one
ANCHOR_LOST_CONF = 0.45    # confidence cap for derived landmarks on a frame whose head anchor could not be trusted
EXT_CONF_CAP = 0.7         # silhouette feet / wing tips are never as certain as a tracked point
DERIVED_JUMP_FRAC = 0.15   # a derived landmark jumping more than this x body length in one frame ...
DERIVED_JUMP_CONF = 0.3    # ... gets this confidence (a flip, a lost anchor, a leg swapped for a tail)
BORDER_PX = 3              # a derived point this close to the picture edge, on a mask touching it, is the edge
COLLAPSE_FRAC = 0.02       # two constrained landmarks closer than this x mask diagonal have merged ...
COLLAPSE_MIN_PX = 3.0      # ... (never below this, in working pixels)
# A landmark that LEAVES the segment must stop the run at that
# frame -- snapping it back does not guarantee the same spot. Outline jitter of
# a few pixels is not leaving: within this band the point is nudged onto the
# silhouette, beyond it the run stops.
EXIT_BAND_FRAC = 0.03      # band outside the silhouette still counted as "on it", x mask diagonal ...
EXIT_BAND_MIN_PX = 8.0     # ... never thinner than this, in native pixels


class VideoDecodeError(RuntimeError):
    """A frame in the MIDDLE of the video could not be decoded (I40). The run
    reports it through `error` with a plain sentence (not a traceback) after
    emitting everything tracked before that frame."""


def _frame_decodes(path: str, idx: int) -> bool:
    """True when frame `idx` can be sought to and read. Uses its own capture, so
    the run's source and the shared frame cache are left alone."""
    cap = None
    try:
        from kinetrace.video_source import open_capture   # same backend as the run
        cap = open_capture(path)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, _ = cap.read()
        return bool(ok)
    except Exception:   # noqa: BLE001 -- a probe that cannot run proves nothing
        return False
    finally:
        if cap is not None:
            cap.release()


def pick_device() -> tuple[str, str]:
    """Returns (torch device string, human-readable label). ONE rule for every
    model (`device.pick_device`: CUDA, else Apple's GPU, else the CPU;
    `KINETRACE_DEVICE` forces it) - kept here by name for its callers."""
    from kinetrace.device import pick_device as _pick
    return _pick()


def get_model():
    """Process-wide CoTracker3 online predictor singleton. Safe to call from any thread."""
    global _model, _device
    with _model_lock:
        if _model is None:
            device, _ = pick_device()
            torch.hub.set_dir(str(MODELS_DIR))  # keep downloads inside the tool folder
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online",
                                   trust_repo=True)
            _model = model.to(device).eval()
            _device = device
        return _model, _device


def model_is_cached() -> bool:
    """True if the repo+checkpoint are already on disk (first run needs internet)."""
    return (MODELS_DIR / "checkpoints" / "scaled_online.pth").exists()


def sample_members(center: np.ndarray, radius: float, w: int, h: int,
                   outline: np.ndarray | None = None) -> np.ndarray:
    """Member constellation for a group: center + 4 @ 0.4r + 8 @ 0.75r, clamped
    just inside the frame. (M, 2) float32. Rings stay well INSIDE the drawn
    circle — the user outlines the object, so members must sit on it, not on
    the background at its boundary.

    A rectangle / polygon region (`outline` = vertex offsets from the center)
    is sampled on a grid over its bounding box shrunk to 80% towards the
    center, keeping the samples inside the polygon (same "inside, not on the
    boundary" rule), up to 16 members plus the center."""
    if outline is not None and len(outline) >= 3:
        poly = np.asarray(outline, np.float32).reshape(-1, 2)
        lo, hi = poly.min(axis=0) * 0.8, poly.max(axis=0) * 0.8
        gx = np.linspace(lo[0], hi[0], 5)
        gy = np.linspace(lo[1], hi[1], 5)
        cand = np.array([(x, y) for y in gy for x in gx], np.float32)
        inside = np.array([cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
                           for x, y in cand])
        cand = cand[inside]
        # drop the sample nearest the center (the center itself is member 0)
        cand = cand[np.argsort(np.hypot(cand[:, 0], cand[:, 1]))]
        if len(cand) and np.hypot(*cand[0]) < 1e-3:
            cand = cand[1:]
        if len(cand) > 16:      # thin evenly: keep the spatial spread
            cand = cand[np.linspace(0, len(cand) - 1, 16).round().astype(int)]
        offs = np.concatenate([np.zeros((1, 2), np.float32), cand.reshape(-1, 2)], axis=0)
        pts = np.asarray(center, np.float32)[None] + offs
        pts[:, 0] = np.clip(pts[:, 0], 1.0, w - 2.0)
        pts[:, 1] = np.clip(pts[:, 1], 1.0, h - 2.0)
        return pts
    offs = [(0.0, 0.0)]
    for r_frac, n, phase in ((0.4, 4, np.pi / 4), (0.75, 8, 0.0)):
        ang = phase + np.arange(n) * (2 * np.pi / n)
        offs += [(radius * r_frac * np.cos(a), radius * r_frac * np.sin(a)) for a in ang]
    pts = np.asarray(center, np.float32)[None] + np.asarray(offs, np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 1.0, w - 2.0)
    pts[:, 1] = np.clip(pts[:, 1], 1.0, h - 2.0)
    return pts


def fit_group(seed_pts: np.ndarray, seed_center: np.ndarray, cur_pts: np.ndarray,
              conf_m: np.ndarray, vis_m: np.ndarray, radius: float,
              frame_wh: tuple[int, int]) -> tuple[np.ndarray, bool, float]:
    """Robust center of a member constellation for one frame.

    Similarity fit (translation+rotation+scale, RANSAC) from the seed
    constellation when enough members survive; falls back to the median
    member displacement (handles deformation, at reduced confidence).
    Returns (center (2,) float32, visible, confidence in [0,1]).
    """
    M = len(seed_pts)
    fw, fh = frame_wh
    valid = in_frame(cur_pts, fw, fh)
    if not valid.any():
        return np.array([np.nan, np.nan], np.float32), False, 0.0

    if valid.sum() >= GROUP_RANSAC_MIN_INLIERS and radius >= GROUP_MIN_FIT_RADIUS:
        A, inliers = cv2.estimateAffinePartial2D(
            seed_pts[valid].astype(np.float32), cur_pts[valid].astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=max(3.0, 0.1 * radius),
            maxIters=200, confidence=0.995)
        if A is not None and inliers is not None:
            inl = inliers.ravel().astype(bool)
            scale = float(np.hypot(A[0, 0], A[0, 1]))
            if inl.sum() >= GROUP_RANSAC_MIN_INLIERS and 0.33 < scale < 3.0:
                center = (A[:, :2] @ np.asarray(seed_center, np.float64)
                          + A[:, 2]).astype(np.float32)
                conf = float(inl.sum() / M) * float(np.median(conf_m[valid][inl]))
                visible = bool(np.mean(vis_m[valid][inl]) > 0.5)
                return center, visible, min(conf, 1.0)

    # fallback: rigid fit unavailable/degenerate -> median displacement.
    # A healthy-but-unfittable constellation (deforming object, tiny radius)
    # must stay above CONF_PAUSE_THRESHOLD; one losing its members must not.
    disp = np.median(cur_pts[valid] - seed_pts[valid], axis=0)
    center = (np.asarray(seed_center, np.float32) + disp.astype(np.float32))
    conf = 0.6 * float(valid.mean()) * float(np.median(conf_m[valid]))
    visible = bool(np.mean(vis_m[valid]) > 0.5)
    return center, visible, min(conf, 1.0)


def _drain(gen):
    """Run a step generator to its end and return its return value (I141)."""
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


class MultiTrackingWorker(QThread):
    """Several cameras' TrackingWorkers tracked AT THE SAME TIME (I141, owner
    2026-09-27: "only the current active camera tracks"). They are advanced in
    turn on this ONE thread, a frame each (`TrackingWorker.steps`): every camera
    moves on together and draws live, each worker decodes on its own read-ahead
    thread, one model step runs on the GPU at a time (so the memory of ONE run,
    whatever the camera count), and each worker emits its own signals exactly
    as if it ran alone -- its output is bit-identical to a run of its own.
    `request_pause` stops every camera at once. The workers are never started as
    threads themselves; this thread ends when the last of them has."""

    def __init__(self, workers: list):
        super().__init__()
        self.workers = list(workers)
        for w in self.workers[1:]:
            w.private_model = True

    def request_pause(self) -> None:
        for w in self.workers:
            w.request_pause()

    def run(self) -> None:
        live = [(w, w.steps()) for w in self.workers]
        try:
            while live:
                for item in list(live):
                    try:
                        next(item[1])
                    except StopIteration:
                        live.remove(item)
        finally:
            for _w, g in live:          # only on an unexpected exit: each closes its reader / video
                g.close()


class TrackingWorker(QThread):
    """Tracks all seeded points forward from start_frame until EOF or pause."""

    model_loading = Signal()
    started_ok = Signal()
    # (window start abs frame, tracks (L,K,2) float32 native px, vis (L,K) bool,
    #  conf (L,K) float32, members {out_index: (L,M,2) native} for live overlay,
    #  new_frames: list[(abs_idx, display-res RGB)])
    chunk_ready = Signal(int, object, object, object, object, object)
    # per-chunk list of mask summaries (see segmenter.summarize_mask + "frame"),
    # emitted BEFORE chunk_ready so the GUI holds the mask of every frame it displays
    masks_ready = Signal(object)
    # ball markers: [(frame, pid, radius), ...] of the circles fitted in a chunk,
    # emitted BEFORE chunk_ready (the centre travels in the chunk itself)
    balls_ready = Signal(object)
    autopaused = Signal(int, int)    # (first low-confidence frame, point id; -1 = the animal)
    finished_ok = Signal(int, bool)  # (last emitted abs frame, was_paused)
    error = Signal(str)              # a traceback -- or a plain sentence for a VideoDecodeError

    def __init__(self, video_path: str, start_frame: int, seed_xy: np.ndarray,
                 point_ids: list[int], cache: FrameCache, n_frames: int,
                 refine: bool = True, specs: list[PointSpec] | None = None,
                 roi: bool = True, autopause: bool = True,
                 animal: AnimalSpec | None = None,
                 derived: list[DerivedSpec] | None = None,
                 head_pid: int | None = None, on_body_pids=None, constrain_pids=None,
                 point_backend: str = "cotracker3", balls: list[BallSpec] | None = None):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        if specs is None:  # legacy signature: plain points from (K,2) seeds
            seed_xy = np.asarray(seed_xy, np.float32)
            specs = [PointSpec(pid, seed_xy[i].copy())
                     for i, pid in enumerate(point_ids)]
        self.specs = list(specs)
        self.derived = list(derived or [])
        self.balls = list(balls or [])
        # emitted columns: tracked specs first, then silhouette-derived outputs,
        # then the ball markers (SAM circle centres)
        self.point_ids = ([sp.pid for sp in self.specs] + [d.pid for d in self.derived]
                          + [b.pid for b in self.balls])
        self.n_cols = len(self.point_ids)
        self.seed_xy = (np.stack([sp.seed for sp in self.specs]).astype(np.float32)
                        if self.specs else np.zeros((0, 2), np.float32))
        self.cache = cache
        self.n_frames = n_frames
        self.refine = refine
        self.roi = roi
        self.autopause = autopause
        self.animal = animal
        self.head_pid = head_pid
        self.on_body = set(on_body_pids or [])
        self.point_backend = point_backend if point_backend in POINT_BACKENDS else "cotracker3"
        # physical constraint: these tracked points can never leave the animal's
        # silhouette — predictions outside it are snapped to the nearest mask pixel
        self.constrain = set(constrain_pids or [])
        if self.animal is None and self.derived:
            raise ValueError("silhouette-derived points need a segment to derive from")
        if not self.specs and self.animal is None and not self.balls:
            raise ValueError("nothing to track: no seeded points, no balls and no segment")
        self._pause = False
        self._autopause_hit: tuple[int, int] | None = None
        self._autopause_reason = ""          # "exit" | "lowconf" | "lost": what the app tells the user
        self._exit_hit: tuple[int, int] | None = None   # (frame, pid) of the first landmark to leave the segment
        # animal layer state (per run)
        self._seg = None
        self._sxy = None               # (sx, sy) native / working, set by the first segmented frame (I61)
        self._seg_last = start_frame - 1
        self._mask_hist: dict[int, tuple[np.ndarray, float]] = {}   # frame -> (mask_work, scale)
        self._summ: dict[int, dict] = {}                             # frame -> summarize_mask(...)
        self._mid_cache: dict[int, tuple] = {}
        self._dil_cache: dict[int, np.ndarray] = {}
        self._snap_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # frame -> (labels, lut)
        self._snap_big = False                                              # last row snapped far
        self._prev_head: np.ndarray | None = None
        # derived-skeleton continuity + landmark identity (see the constants)
        self._prev_len: float | None = None            # previous midline length, working px
        self._prev_path: np.ndarray | None = None      # previous midline path, working px
        self._anchor_lost: set[int] = set()            # frames whose head anchor could not be trusted
        self._anchor_verdict: tuple | None = None      # (frame, anchor at an end?, free midline) last checked
        self._derived_hist: dict[int, dict[int, np.ndarray]] = {}   # column -> {frame: native xy}
        self._snapped: dict[int, set[int]] = {}        # frame -> query indices the constraint moved
        self._collapsed: dict[int, set[int]] = {}      # frame -> query indices merged with another
        self._col2q: dict[int, int] = {}               # spec index -> query index (point specs)
        self._q2col: dict[int, int] = {}               # query index -> OUTPUT column (point specs)
        self._back_run: dict[int, int] = {}            # pid -> frames back since it was declared lost
        # first frame the decoder could not deliver although the video goes on (I40)
        self.decode_failed_at: int | None = None
        self._animal_counted_until = start_frame - 1
        self._lost_run = 0
        self._lost_start = start_frame
        # ball markers (balls.py): one SAM session on a crop, every ball an object
        self._ball_trk = None
        self._ball_last = start_frame - 1
        self._ball_res: dict[int, dict] = {}          # frame -> {pid: CircleFit}
        self._ball_obj = {b.pid: 1000 + k for k, b in enumerate(self.balls)}   # pid -> SAM object id
        self._ball_gone: set[int] = set()             # balls dropped inside the picture (data ends)
        self._ball_ended: dict[int, tuple[int, str]] = {}  # pid -> (first frame without it, "lost"|"apart")

    def request_pause(self) -> None:
        self._pause = True

    # set by MultiTrackingWorker for every camera after the first: its own copy of
    # the CoTracker3 model (the online wrapper keeps a run's state on the model)
    private_model = False

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        for _ in self.steps():          # the whole run on this worker's own thread
            pass

    def steps(self):
        """The run as a generator that yields after every frame it reads (I141).
        `run()` drains it on this worker's thread; `MultiTrackingWorker` advances
        several workers' generators in turn on ONE thread, so every camera tracks
        at the same time. Only the pauses between frames differ -- the same frames,
        model calls and signals -- so the output is bit-identical either way."""
        src = None
        try:
            src = self._open_source()
            yield from self._run_steps(src)
        except VideoDecodeError as e:
            self.error.emit(str(e))     # a sentence for the user, not a traceback
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if src is not None:
                src.close()

    def _open_source(self) -> VideoSource:
        src = VideoSource(self.video_path, self.cache)
        src.seek(self.start_frame)
        return src

    def _run(self, src: VideoSource) -> None:
        _drain(self._run_steps(src))

    def _run_steps(self, src: VideoSource):
        self.model_loading.emit()
        model = device = None
        if self.specs:
            if self.point_backend == "alltracker":
                model, device = at_backend.get_alltracker()
            else:
                model, device = get_model()
                if self.private_model:
                    # CoTracker3's online wrapper keeps the run's queries and window
                    # state on the model object: runs interleaved on one thread each
                    # need their own copy (same weights, so the same numbers; I141)
                    import copy
                    model = copy.deepcopy(model)

        probe = src.get_frame(self.start_frame)
        if probe is None:
            raise RuntimeError(f"Could not decode frame {self.start_frame}")
        nh, nw = probe.shape[:2]
        self._frame_wh = (nw, nh)
        if self.animal is not None:
            engine = get_segmenter(self.animal.backend)
            self._seg = engine.new_session(self.start_frame, (nw, nh))
        if self.balls:
            from kinetrace.balls import BallTracker
            from kinetrace.segmenter import preferred_backend
            backend = (self.balls[0].backend or (self.animal.backend if self.animal is not None else "")
                       or preferred_backend())
            self._ball_trk = BallTracker(get_segmenter(backend), (nw, nh))
        if not self.specs:
            # animal / balls only: no point tracker — segment every frame, derive landmarks
            self.started_ok.emit()
            last = yield from self._animal_only_steps(src, self.start_frame)
            if self._autopause_hit is not None:
                self.autopaused.emit(*self._autopause_hit)
            elif self.decode_failed_at is not None:
                raise VideoDecodeError(self._decode_failure_text())
            self.finished_ok.emit(last, self._pause or self._autopause_hit is not None)
            return
        # LK acceptance gate ~ 2x the model's effective quantization at native res
        self._gate = 2.0 * max(1.0, max(nw, nh) / 512.0)
        # LK window scales mildly with resolution (more context at 4K)
        w = int(round(21 * max(1.0, max(nw, nh) / 1920.0)))
        self._lk_win = min(w | 1, 51)
        # refinement only pays off when native res is well above the model's
        # internal ~512px resolution (at 4K it cuts error ~5x; at <=1024 it
        # just adds noise on top of already-native-scale predictions)
        self._do_refine = self.refine and max(nw, nh) / 512.0 > 1.5
        self._templates: dict[int, np.ndarray] = {}   # pid -> seed NCC template
        self._template_frac: dict[int, np.ndarray] = {}   # pid -> seed minus the template's centre pixel
        self._low_run = {sp.pid: 0 for sp in self.specs}  # autopause counters
        self._low_start = {sp.pid: 0 for sp in self.specs}  # first frame of each run
        self._gone_run = {sp.pid: 0 for sp in self.specs}   # low conf AND not visible
        self._gone_start = {sp.pid: 0 for sp in self.specs}
        self._back_run = {sp.pid: 0 for sp in self.specs}   # recovery run of a lost point
        self._lost: set[int] = set()      # points declared lost: their data is blanked
        self._counted_until = self.start_frame - 1        # overlap rows counted once
        self._autopause_hit: tuple[int, int] | None = None
        self._crop_growth = 1.0                           # ROI churn adaptation

        self.started_ok.emit()

        # emitted windows ALWAYS carry all original columns; a point dropped at
        # an ROI restart keeps its column and emits NaN (= "no data") from then
        # on, so the session/pid mapping never shifts mid-run
        specs = list(self.specs)
        col_idx = list(range(len(specs)))  # original column of each active spec
        seg_start = self.start_frame
        first_seg = True
        last_emitted = self.start_frame - 1
        while True:
            reason, last_emitted, next_seeds = yield from self._segment_steps(
                src, model, device, seg_start, specs, col_idx, first_seg, last_emitted)
            if reason != "restart":
                break
            # transparent ROI re-seed: continue from our own predictions at the
            # last emitted frame; points with no valid position there stay
            # blank until the user re-places them (the OOB invariant)
            new_specs, new_cols = [], []
            for sp, k, pos in zip(specs, col_idx, next_seeds):
                if pos is not None:
                    new_specs.append(PointSpec(sp.pid, np.asarray(pos, np.float32),
                                               sp.kind, sp.radius, sp.anchor, sp.outline))
                    new_cols.append(k)
            if not new_specs:
                if (self.animal is not None or self.balls) and last_emitted + 1 < self.n_frames:
                    # every tracked point was dropped, but the animal / the
                    # ball markers go on: continue with masks + derived
                    # landmarks + balls only (balls without a segment used to
                    # stop here, short of the video's end -- I120)
                    last_emitted = yield from self._animal_only_steps(src, last_emitted + 1)
                break
            if last_emitted - seg_start < ROI_GROW_SEGMENT:
                self._crop_growth *= ROI_GROW_FACTOR  # crop too tight for the motion
            specs, col_idx = new_specs, new_cols
            seg_start = last_emitted
            first_seg = False

        if self._autopause_hit is not None:
            self.autopaused.emit(*self._autopause_hit)
        elif self.decode_failed_at is not None:
            # everything before the damaged frame is emitted; say why the run
            # stopped instead of finishing as if the video ended there (I40)
            raise VideoDecodeError(self._decode_failure_text())
        self.finished_ok.emit(last_emitted, self._pause or self._autopause_hit is not None)

    def _check_read_end(self, k: int) -> None:
        """The decoder delivered no frame `k` (I40). At the run's last frame that
        is the end of the video. Before it, it is either a frame the decoder
        cannot read in the middle of the file, or a header that promised more
        frames than the file holds (the count is verified at open, but a file
        that cannot seek keeps the header's number). A LATER frame that decodes
        tells the two apart; only then is the run reported as stopped by a
        damaged frame. Clean runs never get here with k < n_frames."""
        if k >= self.n_frames or self.decode_failed_at is not None:
            return
        for idx in sorted({k + 1, self.n_frames - 1}):
            if k < idx < self.n_frames and _frame_decodes(self.video_path, idx):
                self.decode_failed_at = int(k)
                return

    def _decode_failure_text(self) -> str:
        k = self.decode_failed_at
        name = Path(self.video_path).name
        return (f"Frame {k} of the video could not be decoded, although the file goes on after it, "
                f"so tracking stopped at frame {k - 1}. Everything tracked up to there is kept. The file "
                f"is probably damaged at that frame. To go on, place the points again on a frame after {k} "
                f"and track from there, or re-encode the video (for example: "
                f'ffmpeg -i "{name}" -c:v libx264 -crf 18 fixed.mp4) and open the copy.')

    # -------------------------------------------------------------- segments

    def _compute_crop(self, specs: list[PointSpec],
                      seg_start: int | None = None) -> tuple[int, int, int, int] | None:
        """Fixed crop for one segment, or None for full-frame processing. With
        an segment, the crop covers its mask at the segment start too — the
        crop follows the segment, not just the points."""
        nw, nh = self._frame_wh
        if not self.roi or max(nw, nh) / 512.0 <= 1.5:
            return None
        pts = []
        for sp in specs:
            r = sp.radius if sp.kind == "group" else 0.0
            pts += [(sp.seed[0] - r, sp.seed[1] - r), (sp.seed[0] + r, sp.seed[1] + r)]
        summ = self._summ.get(seg_start) if seg_start is not None else None
        if summ is not None and summ["area"] > 0:
            x0, y0, x1, y1 = summ["bbox"]
            pts += [(x0, y0), (x1, y1)]
        if not pts:
            return None
        pts = np.asarray(pts, np.float64)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        grow = self._crop_growth  # widens after churn-y (short) segments
        cw = int(min(nw, max((x1 - x0) * (1 + 2 * ROI_MARGIN_FRAC), ROI_MIN_W) * grow))
        ch = int(min(nh, max((y1 - y0) * (1 + 2 * ROI_MARGIN_FRAC), ROI_MIN_H) * grow))
        if max(nw, nh) / max(cw, ch) < ROI_MIN_ZOOM:
            return None
        cx0 = int(np.clip(round((x0 + x1) / 2 - cw / 2), 0, nw - cw))
        cy0 = int(np.clip(round((y0 + y1) / 2 - ch / 2), 0, nh - ch))
        return (cx0, cy0, cw, ch)

    def _model_step(self, model, frames: list[np.ndarray], device: str, first: bool,
                    queries: torch.Tensor | None):
        """One model call. First call goes through the wrapper (query setup);
        processing calls run the inner model to keep the raw confidence while
        reproducing the wrapper's outputs exactly (vis = (vis*conf) > 0.6)."""
        chunk = torch.from_numpy(np.stack(frames)).to(device).float().permute(0, 3, 1, 2)[None]
        with torch.no_grad():
            if first:
                model(video_chunk=chunk, is_first_step=True, queries=queries,
                      add_support_grid=True)
                return None
            B, T, C, H, W = chunk.shape
            video = chunk.reshape(B * T, C, H, W)
            video = F.interpolate(video, tuple(model.interp_shape), mode="bilinear",
                                  align_corners=True)
            video = video.reshape(B, T, 3, model.interp_shape[0], model.interp_shape[1])
            if model.v2:
                tracks, vis_f, _ = model.model(video=video, queries=model.queries,
                                               iters=6, is_online=True)
                conf_f = torch.ones_like(vis_f)
            else:
                tracks, vis_f, conf_f, _ = model.model(video=video, queries=model.queries,
                                                       iters=6, is_online=True)
            # slice the current window on-GPU BEFORE the CPU copy: the model
            # accumulates predictions for every frame fed so far, and copying
            # that whole tensor each call would be O(T^2) transfer over a run
            keep = 2 * model.step
            n = model.N  # strip the support grid
            tracks, vis_f, conf_f = (tracks[:, -keep:, :n], vis_f[:, -keep:, :n],
                                     conf_f[:, -keep:, :n])
            vis = (vis_f * conf_f) > 0.6
            tracks = tracks * tracks.new_tensor(
                [(W - 1) / (model.interp_shape[1] - 1),
                 (H - 1) / (model.interp_shape[0] - 1)])
            return (tracks[0].cpu().numpy(), vis[0].cpu().numpy(),
                    conf_f[0].cpu().numpy().astype(np.float32))

    def _run_segment(self, src: VideoSource, model, device: str, seg_start: int,
                     specs: list[PointSpec], col_idx: list[int], first_seg: bool,
                     last_emitted: int):
        return _drain(self._segment_steps(src, model, device, seg_start, specs, col_idx,
                                          first_seg, last_emitted))

    def _segment_steps(self, src: VideoSource, model, device: str, seg_start: int,
                       specs: list[PointSpec], col_idx: list[int], first_seg: bool,
                       last_emitted: int):
        """(A generator: yields after every frame, I141.) One online-model segment (fixed queries, fixed crop). Returns
        (reason, last_emitted, restart_seeds) with reason in
        {"eof", "pause", "autopause", "restart"}. Emitted arrays keep the
        run's original column count; specs[a] fills column col_idx[a]."""
        use_at = self.point_backend == "alltracker"
        step = at_backend.STRIDE if use_at else model.step  # 8 either way
        nw, nh = self._frame_wh
        K = self.n_cols      # original column count (tracked + derived), NOT len(specs)
        n_track = len(self.specs)

        # query layout: singles map 1:1 to an output column; groups contribute
        # a member block and their output column is the per-frame fitted center
        q_seeds: list[np.ndarray] = []
        out_map: list[tuple] = []  # ("point", q_idx) | ("group", slice, seed_members, center)
        for sp in specs:
            if sp.kind == "group":
                mem = sample_members(sp.seed, sp.radius, nw, nh, sp.outline)
                out_map.append(("group", slice(len(q_seeds), len(q_seeds) + len(mem)),
                                mem, sp.seed.copy()))
                q_seeds += list(mem)
            else:
                out_map.append(("point", len(q_seeds)))
                q_seeds.append(sp.seed.astype(np.float32))
        # the segment's first frame must be segmented before the crop and the
        # support points can use its mask (a restart re-reads an already
        # segmented frame; the very first frame is segmented here)
        if self._seg is not None and seg_start > self._seg_last:
            first = src.get_frame(seg_start)
            if first is not None:
                self._seg_step(seg_start, first)
        # support queries: a sparse grid inside the animal's mask so the joint
        # solve is anchored to the body; never emitted, re-sampled every segment
        q_seeds += list(self._support_points(seg_start))
        q_seeds_arr = np.asarray(q_seeds, np.float32)   # (Q, 2) native px

        crop = self._compute_crop(specs, seg_start)
        if crop is not None:
            cx0, cy0, cw, ch = crop
        else:
            cx0 = cy0 = 0
            cw, ch = nw, nh
        origin = np.array([cx0, cy0], np.float32)
        if use_at:
            # 1024 on CUDA; smaller on the CPU / an Apple GPU, where the window
            # lives in RAM (18.4 GB at 1024 on a full 4K frame, measured)
            from kinetrace.device import alltracker_max_dim
            at_dim = min(ALLTRACKER_MAX_DIM, alltracker_max_dim(str(device))) if device is not None \
                else ALLTRACKER_MAX_DIM
        scale = min(1.0, (at_dim if use_at else WORKING_MAX_DIM) / max(cw, ch))
        ww, wh = max(2, round(cw * scale)), max(2, round(ch * scale))
        # align-corners convention, matching the model's internal rescaling
        to_native = np.array([(cw - 1) / max(ww - 1, 1), (ch - 1) / max(wh - 1, 1)], np.float32)
        to_working = 1.0 / to_native
        # display path: full frame at working res (crops would break the canvas)
        disp_scale = min(1.0, WORKING_MAX_DIM / max(nw, nh))
        dw, dh = max(2, round(nw * disp_scale)), max(2, round(nh * disp_scale))

        q_work = ((q_seeds_arr - origin) * to_working).astype(np.float32)   # (Q, 2) crop-working px
        queries = None
        if not use_at:
            queries = torch.tensor(
                np.concatenate([np.zeros((len(q_seeds_arr), 1), np.float32), q_work], axis=1),
                device=device)[None]  # (1, Q, 3): (t_rel=0, x, y) in crop-working px
        at_stream = None

        window: deque[np.ndarray] = deque(maxlen=2 * step)   # crop-working RGB
        gray_hist: deque[tuple[int, np.ndarray]] = deque(maxlen=2 * step + 2)  # native gray
        pending: list[tuple[int, np.ndarray]] = []           # display frames since last emit
        self._refined: dict[int, np.ndarray] = {}            # abs frame -> (Q,2) native
        anchor_q = {om[1]: sp for sp, om in zip(specs, out_map)
                    if sp.kind == "point" and sp.anchor}     # q_idx -> spec
        need_gray = self._do_refine or bool(anchor_q)
        n = 0
        is_first = True
        restart_seeds: list | None = None
        result = {"last": last_emitted, "restart": False}
        src.seek(seg_start)

        def model_input_of(native: np.ndarray) -> np.ndarray:
            region = native[cy0:cy0 + ch, cx0:cx0 + cw] if crop is not None else native
            if scale >= 1.0:
                # copy: a view would keep the whole native 4K frame alive for as
                # long as it sits in the 16-slot window (~400 MB pinned)
                return np.ascontiguousarray(region) if crop is not None else region
            return cv2.resize(region, (ww, wh), interpolation=cv2.INTER_AREA)

        def display_of(native: np.ndarray, model_frame: np.ndarray) -> np.ndarray:
            if crop is None:
                return model_frame  # identical path, no second resize
            if disp_scale >= 1.0:
                return native
            return cv2.resize(native, (dw, dh), interpolation=cv2.INTER_AREA)

        def emit_rows(step_out, n_frames_fed: int):
            """Map one rewritten window (last <= 16 rows) to native coords,
            refine, fit groups, run detectors, and emit."""
            nonlocal pending, restart_seeds
            tr_q, vis_q, conf_q = step_out
            L = min(tr_q.shape[0], 2 * step, n_frames_fed)
            tr = tr_q[-L:] * to_native[None, None, :] + origin[None, None, :]
            vi = vis_q[-L:].astype(bool)
            cf = conf_q[-L:]
            w0 = seg_start + n_frames_fed - L
            if w0 == seg_start:
                # root the window (and the LK chain below) at the exact segment
                # seeds: the user's click on the first segment, our own emitted
                # positions on a restart — never the model's re-prediction
                tr[0] = q_seeds_arr
                if first_seg:
                    vi[0] = True  # user input is ground truth
            if need_gray:
                tr = self._refine_window(w0, tr.astype(np.float32), vi, gray_hist,
                                         seg_start, anchor_q)
            tr = tr.astype(np.float32)
            self._constrain_to_mask(w0, tr, specs, out_map, first_seg, col_idx)

            out_tr = np.full((L, K, 2), np.nan, np.float32)
            out_vi = np.zeros((L, K), bool)
            out_cf = np.zeros((L, K), np.float32)
            members: dict[int, np.ndarray] = {}
            for a, om in enumerate(out_map):
                k = col_idx[a]
                if om[0] == "point":
                    q = om[1]
                    out_tr[:, k] = tr[:, q]
                    out_vi[:, k] = vi[:, q]
                    out_cf[:, k] = cf[:, q]
                else:
                    sl, seed_m, seed_c = om[1], om[2], om[3]
                    members[k] = tr[:, sl].copy()
                    for i in range(L):
                        c, v, g = fit_group(seed_m, seed_c, tr[i, sl], cf[i, sl],
                                            vi[i, sl], specs[a].radius, self._frame_wh)
                        out_tr[i, k] = c
                        out_vi[i, k] = v
                        out_cf[i, k] = g
            if first_seg and w0 == self.start_frame:
                # the seed frame is user ground truth — keep it exact
                out_tr[0, :n_track] = self.seed_xy
                out_vi[0, :n_track] = True
                out_cf[0, :n_track] = 1.0

            self._demote_off_body(w0, out_tr, out_cf)
            self._demote_merged(w0, out_cf)
            self._fill_derived(w0, out_tr, out_vi, out_cf)
            self._fill_balls(w0, out_tr, out_vi, out_cf)
            self._check_autopause(w0, out_tr, out_vi, out_cf, specs, col_idx)
            self._check_animal_lost(w0, L)
            self._emit_masks(w0, L)
            self.chunk_ready.emit(w0, out_tr, out_vi, out_cf, members, list(pending))
            pending = []
            result["last"] = w0 + L - 1

            # never restart into a stub segment: the last <=16 frames track fine
            # in the current crop, and sub-window segments have no clean shape
            room = (self.n_frames - (result["last"] + 1) > 2 * step
                    and self.decode_failed_at is None)   # nothing to restart into past a damaged frame
            if self._snap_big and room and self._autopause_hit is None and not self._pause:
                # a constrained point was pulled back onto the animal by a large
                # step: CoTracker still believes the old location, so re-seed it
                # from the corrected positions (same mechanism as an ROI restart)
                result["restart"] = True
                restart_seeds = []
                for a in range(len(specs)):
                    p = out_tr[-1, col_idx[a]]
                    good = bool(in_frame(p, nw, nh))
                    restart_seeds.append(p.copy() if good else None)
            if (crop is not None and room and self._autopause_hit is None and not self._pause
                    and not result["restart"]):
                pts = [out_tr[-1, :n_track]] + [members[k][-1] for k in members]
                summ_last = self._summ.get(w0 + L - 1)
                if summ_last is not None and summ_last["area"] > 0:
                    bx0, by0, bx1, by1 = summ_last["bbox"]   # the animal nearing the edge restarts too
                    pts.append(np.array([[bx0, by0], [bx1, by1]], np.float32))
                pts = np.concatenate(pts, axis=0)
                ok = np.isfinite(pts).all(axis=1)
                mx, my = ROI_EDGE_FRAC * cw, ROI_EDGE_FRAC * ch
                # only a crop side INSIDE the picture is an edge to escape: a
                # side on the picture border cannot move further out, so a
                # restart there got the same crop again, restarted after one
                # window, grew the crop and soon switched ROI off for the rest
                # of the run (I119; balls._near_edge_in has the same rule)
                near = (ok & (((pts[:, 0] < cx0 + mx) & (cx0 > 0))
                              | ((pts[:, 0] > cx0 + cw - mx) & (cx0 + cw < nw))
                              | ((pts[:, 1] < cy0 + my) & (cy0 > 0))
                              | ((pts[:, 1] > cy0 + ch - my) & (cy0 + ch < nh))))
                if near.any():
                    result["restart"] = True
                    restart_seeds = []
                    for a in range(len(specs)):
                        p = out_tr[-1, col_idx[a]]
                        good = bool(in_frame(p, nw, nh))
                        restart_seeds.append(p.copy() if good else None)

        # decode runs a couple of frames ahead on its own thread, overlapping
        # with the segment + model steps below. Same frames in the same order
        # (bit-identical output) — only the timing changes. It OWNS src until
        # stopped, which is why every exit path below goes through the finally.
        reader = ReadAhead(src, depth=READ_AHEAD_FRAMES)
        read_end = None     # the frame the decoder did not deliver, if the loop ended on one
        try:
            while True:
                if self._pause or self._autopause_hit is not None or result["restart"]:
                    break
                nxt = reader.read_next()
                if nxt is None:
                    read_end = seg_start + n
                    break
                if nxt[0] >= self.n_frames:
                    break
                abs_idx, native = nxt
                if self._seg is not None and abs_idx > self._seg_last:
                    self._seg_step(abs_idx, native)
                if self._ball_trk is not None and abs_idx > self._ball_last:
                    self._ball_step(abs_idx, native)
                model_frame = model_input_of(native)
                window.append(model_frame)
                if need_gray:
                    gray_hist.append((abs_idx, cv2.cvtColor(native, cv2.COLOR_RGB2GRAY)))
                    if abs_idx == self.start_frame and first_seg and anchor_q:
                        self._capture_templates(gray_hist[-1][1], specs, out_map)
                pending.append((abs_idx, display_of(native, model_frame)))
                n += 1
                if use_at:
                    # AllTracker: the segment's first frame is the query frame; every
                    # completed 16-frame window (stride 8) rewrites its rows
                    if n == 1:
                        at_stream = at_backend.AllTrackerStream()
                        at_stream.start(model_frame, q_work)
                    else:
                        out = at_stream.push(model_frame)
                        if out is not None:
                            emit_rows(self._at_rows(out), n)
                elif n % step == 0:
                    if is_first:
                        self._model_step(model, list(window), device, True, queries)
                        is_first = False
                    else:
                        out = self._model_step(model, list(window), device, False, None)
                        emit_rows(out, n)
                yield                           # a frame done: another camera's turn (I141)

        finally:
            reader.stop()   # hand src back before any seek / reuse / close
        if read_end is not None:
            self._check_read_end(read_end)   # the end of the video, or a damaged frame? (I40)

        # EOF / pause tail. On pause we discard unprocessed frames (< 8) — the
        # user acts on what they saw, which is the last emitted window.
        interrupted = (self._pause or self._autopause_hit is not None or result["restart"])
        if use_at and not interrupted and n > 1:
            out = at_stream.flush()      # tail shorter than a window: pad, run, keep real rows
            if out is not None:
                emit_rows(self._at_rows(out), n)
        elif not interrupted and n > 0:
            if is_first or use_at:
                if n == 1:
                    if first_seg:
                        # single-frame run: the seed row is already known
                        out_tr = np.full((1, K, 2), np.nan, np.float32)
                        out_vi = np.zeros((1, K), bool)
                        out_cf = np.zeros((1, K), np.float32)
                        for a, sp in enumerate(specs):
                            out_tr[0, col_idx[a]] = sp.seed
                            out_vi[0, col_idx[a]] = True
                            out_cf[0, col_idx[a]] = 1.0
                        out_tr[0, :n_track] = self.seed_xy
                        self._fill_derived(seg_start, out_tr, out_vi, out_cf)
                        self._fill_balls(seg_start, out_tr, out_vi, out_cf)
                        self._emit_masks(seg_start, 1)
                        self.chunk_ready.emit(seg_start, out_tr, out_vi, out_cf,
                                              {}, list(pending))
                    # a restarted segment's seg_start frame was already emitted
                    # with real model data — never overwrite it with fabricated
                    # vis/conf (restarts can't reach here anyway: no restart is
                    # taken within 16 frames of EOF)
                    result["last"] = seg_start
                else:
                    self._model_step(model, list(window), device, True, queries)
                    out = self._model_step(model, list(window), device, False, None)
                    emit_rows(out, n)
            elif n % step != 0:
                tail = list(window)[-(n % step + step):]  # 9..15 frames, starts at online_ind
                out = self._model_step(model, tail, device, False, None)
                emit_rows(out, n)
            elif n == step:
                # exactly 8 frames then EOF: the loop's init call consumed them
                # and no processing call ever ran. Re-init (fresh state, same
                # queries) and process the same window — identical to how a
                # short first segment (2..7 frames) is handled above.
                self._model_step(model, list(window), device, True, queries)
                out = self._model_step(model, list(window), device, False, None)
                emit_rows(out, n)

        if result["restart"]:
            reason = "restart"
        elif self._autopause_hit is not None:
            reason = "autopause"
        elif self._pause:
            reason = "pause"
        else:
            reason = "eof"
        return reason, result["last"], restart_seeds

    @staticmethod
    def _at_rows(out) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """AllTracker window -> (tracks (L,Q,2) working px, vis bool, conf [0,1]).
        Its confidence is well calibrated for the pause threshold (measured:
        0.95 on clean dots, 0.82 on good real tracks, 0.01 on a lost track)."""
        _, tr, vis_p, conf_p = out
        return tr.astype(np.float32), vis_p > 0.5, conf_p.astype(np.float32)

    # ----------------------------------------------------------- animal layer

    def _w2n(self, p, scale: float) -> np.ndarray:
        """working-res px -> native px (pixel-centre convention), per axis once
        the segmenter has said its (sx, sy); `scale` (one number) stays the
        factor for LENGTHS (I61)."""
        s = self._sxy if self._sxy is not None else scale
        return (np.asarray(p, np.float32) + 0.5) * s - 0.5

    def _n2w(self, p, scale: float) -> np.ndarray:
        s = self._sxy if self._sxy is not None else scale
        return (np.asarray(p, np.float32) + 0.5) / s - 0.5

    def _seg_step(self, abs_idx: int, native: np.ndarray) -> None:
        """Segment one frame (strictly sequential). Prompts on this frame — the
        user's clicks/boxes, or the seed mask on the start frame — are applied
        first, so a click placed on a later frame corrects the run when the
        run gets there."""
        a = self.animal
        prompts: list[Prompt] = []
        clicks = a.prompts.get(abs_idx, [])
        box = a.boxes.get(abs_idx)
        if clicks or box is not None:
            pts = np.array([[c[0], c[1]] for c in clicks], np.float32) if clicks else None
            labs = np.array([c[2] for c in clicks], np.int64) if clicks else None
            prompts.append(Prompt(1, pts, labs, box))
        elif abs_idx == self.start_frame and a.seed_mask is not None and a.seed_mask.any():
            prompts.append(Prompt(1, mask=a.seed_mask))
        if abs_idx == self.start_frame and not prompts:
            raise RuntimeError(
                f"The animal has no click, box or mask on frame {abs_idx}. Press S and click "
                "the animal on this frame, then start tracking again.")
        fm = self._seg.step(native, abs_idx, prompts or None)
        mask = fm.mask(1)
        score = float(fm.scores[fm.index(1)])
        self._seg_last = abs_idx
        self._mask_hist[abs_idx] = (mask, fm.scale)
        sxy = getattr(fm, "scale_xy", None)
        if sxy is not None:
            self._sxy = np.asarray(sxy, np.float32)
        summ = summarize_mask(mask, sxy if sxy is not None else fm.scale, score)
        summ["frame"] = abs_idx
        self._summ[abs_idx] = summ
        for k in [k for k in self._mask_hist if k < abs_idx - MASK_HIST]:
            del self._mask_hist[k]
            self._summ.pop(k, None)
            self._mid_cache.pop(k, None)
            self._dil_cache.pop(k, None)
            self._snap_cache.pop(k, None)
            self._snapped.pop(k, None)
            self._collapsed.pop(k, None)
            self._anchor_lost.discard(k)
            for hist in self._derived_hist.values():
                hist.pop(k, None)

    def _support_points(self, frame: int) -> np.ndarray:
        """A sparse grid of native-px points inside the segment's mask at `frame`."""
        hist = self._mask_hist.get(frame)
        if hist is None:
            return np.zeros((0, 2), np.float32)
        mask, scale = hist
        m8 = mask.astype(np.uint8)
        er = cv2.erode(m8, np.ones((5, 5), np.uint8))
        if not er.any():
            er = m8
        area = int(er.sum())
        if area < 4:
            return np.zeros((0, 2), np.float32)
        step = max(2, int(np.sqrt(area / SUPPORT_POINTS)))
        ys, xs = np.nonzero(er)
        sel = (ys % step == 0) & (xs % step == 0)
        if sel.sum() < 4:
            idx = np.linspace(0, len(xs) - 1, min(SUPPORT_POINTS, len(xs))).astype(int)
            pts = np.stack([xs[idx], ys[idx]], axis=1)
        else:
            pts = np.stack([xs[sel], ys[sel]], axis=1)
            if len(pts) > SUPPORT_POINTS:
                pts = pts[np.linspace(0, len(pts) - 1, SUPPORT_POINTS).astype(int)]
        return self._w2n(pts.astype(np.float32), scale)

    def _midline_for(self, f: int, head_native: np.ndarray | None):
        """Cached silhouette midline of frame f in working-res coordinates."""
        hist = self._mask_hist.get(f)
        if hist is None:
            return None
        mask, scale = hist
        key = None if head_native is None else (round(float(head_native[0]), 1),
                                                 round(float(head_native[1]), 1))
        cached = self._mid_cache.get(f)
        if cached is not None and cached[0] == key:
            return cached[1]
        anchor = None if head_native is None else tuple(self._n2w(head_native, scale))
        ml = silhouette_midline(mask, anchor=anchor)
        lost = anchor is None
        if ml is not None and anchor is not None:
            # The anchor may not be at an end of the body (the tracked snout
            # slid along the flank in a real stereo test; or it was clicked
            # mid-body): a geodesic from mid-body still reaches the tail tip,
            # so the derived tail looks fine while every midline fraction is
            # wrong. The direct test is geometric: the anchor must lie near an
            # END of the body's own (unanchored) midline. That midline costs
            # as much as the anchored one, so it is computed every
            # ANCHOR_CHECK_EVERY frames (and whenever the anchored midline is
            # suspiciously short), and the verdict is held in between.
            summ = self._summ.get(f)
            diag_w = float("nan")
            if summ is not None and summ["area"] > 0:
                x0, y0, x1, y1 = summ["bbox"]                      # native px
                diag_w = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1)) / scale   # -> working px, like ml.length
            short = ((self._prev_len is not None and ml.length < ANCHOR_SHORT_FRAC * self._prev_len)
                     or (np.isfinite(diag_w) and ml.length < 0.5 * diag_w))
            last = self._anchor_verdict
            due = short or last is None or f - last[0] >= ANCHOR_CHECK_EVERY or f < last[0]
            if due:
                free = silhouette_midline(mask, anchor=None)
                ok = True
                if free is not None and free.length > 0:
                    a = np.asarray(anchor, np.float64)
                    d_end = min(float(np.linalg.norm(a - free.head)), float(np.linalg.norm(a - free.tip)))
                    ok = d_end <= ANCHOR_END_FRAC * float(free.length)
                    if short and free.length * ANCHOR_SHORT_FRAC > ml.length:
                        ok = False
                self._anchor_verdict = (f, ok, free)
            else:
                ok, free = last[1], last[2]
            if not ok:
                if free is None or (f != self._anchor_verdict[0]):
                    free = silhouette_midline(mask, anchor=None)      # this frame's own diameter
                if free is not None:
                    ml = free
                    lost = True
        if ml is not None and lost:
            ml = self._orient_continuous(ml, scale)
        if ml is not None:
            self._prev_head = self._w2n(ml.head, scale)
            self._prev_len = float(ml.length)
            self._prev_path = np.asarray(ml.path, np.float32).copy()
        if lost and ml is not None and self.head_pid is not None:
            # only a flag when there WAS a head landmark to anchor on: a run
            # with no head point is unanchored by design, not by failure
            self._anchor_lost.add(f)
        self._mid_cache[f] = (key, ml)
        return ml

    def _orient_continuous(self, ml, scale: float):
        """Head/tail labelling without an anchor, by continuity with the
        previous frame's WHOLE midline (both ends), not its head alone: the
        thick-end heuristic silently flipped an iguana (thick tail base) and
        every derived landmark with it, with full confidence."""
        if self._prev_path is None or len(self._prev_path) < 2 or len(ml.path) < 2:
            return oriented(ml, None if self._prev_head is None else self._n2w(self._prev_head, scale))
        ph, pt = self._prev_path[0], self._prev_path[-1]
        same = float(np.linalg.norm(ml.head - ph) + np.linalg.norm(ml.tip - pt))
        flip = float(np.linalg.norm(ml.head - pt) + np.linalg.norm(ml.tip - ph))
        if flip < same:
            return oriented(ml, ph)          # oriented() flips when the head is farther from the old head
        return ml

    def _derived_xy(self, spec: str, ml, summ: dict, scale: float, roles: dict | None):
        if spec == "centroid":
            return np.asarray(summ["centroid"], np.float32)
        if ml is None:
            return np.array([np.nan, np.nan], np.float32)
        if spec == "tip":
            return self._w2n(ml.tip, scale)
        if spec.startswith("midline:"):
            try:
                frac = float(spec.split(":", 1)[1])
            except ValueError:
                # a rule that does not read as a number is no position, not mid-body (I62)
                return np.array([np.nan, np.nan], np.float32)
            return self._w2n(ml.sample([frac])[0], scale)
        if spec.startswith("ext:"):
            role = spec.split(":", 1)[1]
            if roles is not None and role in roles:
                return self._w2n(roles[role], scale)
        return np.array([np.nan, np.nan], np.float32)

    def _fill_derived(self, w0: int, out_tr: np.ndarray, out_vi: np.ndarray,
                      out_cf: np.ndarray) -> None:
        """Silhouette-derived columns for every row of a window, anchored on
        the tracked head point where it exists. Also attaches the frame's
        midline to its mask summary for display/export."""
        if self.animal is None:
            return
        L = out_tr.shape[0]
        n_track = len(self.specs)
        head_col = None
        if self.head_pid is not None and self.head_pid in self.point_ids[:n_track]:
            head_col = self.point_ids.index(self.head_pid)
        need_midline = bool(self.derived) or True   # the midline is always shown/exported
        for i in range(L):
            f = w0 + i
            summ = self._summ.get(f)
            if summ is None or summ["area"] <= 0:
                continue
            mask, scale = self._mask_hist[f]
            head = None
            if head_col is not None and np.isfinite(out_tr[i, head_col]).all():
                head = out_tr[i, head_col]
            ml = self._midline_for(f, head) if need_midline else None
            if ml is not None and len(ml.path) >= 2:
                summ["midline"] = self._w2n(resample(ml.path, MIDLINE_SAMPLES), scale)
            if not self.derived:
                continue
            present = summ["score"] > 0
            conf = score_to_confidence(summ["score"])
            # A silhouette cut by the picture edge: its tail tip / feet at the
            # edge are the CUT, not the body part (a lizard's tail leaving
            # the frame exported a "tail tip" walking along the border).
            touches = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
            hw, ww = mask.shape[:2]
            anchor_lost = f in self._anchor_lost
            # `scale` is NATIVE / WORKING (3.75 at 4K): working lengths are
            # multiplied to reach native pixels. Dividing here made the body
            # 14x too short at 4K and demoted a real lizard's tail tip on 90 %
            # of frames as a "jump".
            body_len_native = (float(ml.length) * scale) if ml is not None else float("nan")
            roles = None
            for j, d in enumerate(self.derived):
                if d.spec.startswith("ext:") and roles is None and ml is not None:
                    roles = extremity_roles(ml)
                p = self._derived_xy(d.spec, ml, summ, scale, roles)
                k = n_track + j
                c = conf
                if touches and d.spec != "centroid" and np.isfinite(p).all():
                    xw, yw = self._n2w(p, scale)
                    if min(xw, yw, ww - 1 - xw, hw - 1 - yw) <= BORDER_PX:
                        p = np.array([np.nan, np.nan], np.float32)      # out-of-frame = no data
                if anchor_lost:
                    c = min(c, ANCHOR_LOST_CONF)
                if d.spec.startswith("ext:"):
                    c = min(c, EXT_CONF_CAP)
                hist = self._derived_hist.setdefault(k, {})
                prev = hist.get(f - 1)
                if (prev is not None and np.isfinite(p).all() and np.isfinite(prev).all()
                        and np.isfinite(body_len_native) and body_len_native > 0
                        and float(np.linalg.norm(p - prev)) > DERIVED_JUMP_FRAC * body_len_native):
                    c = min(c, DERIVED_JUMP_CONF)          # a jump this size is a flip or a swap
                if np.isfinite(p).all():
                    hist[f] = np.asarray(p, np.float32).copy()
                else:
                    hist.pop(f, None)
                out_tr[i, k] = p
                good = bool(np.isfinite(p).all())
                out_vi[i, k] = good and present
                out_cf[i, k] = c if good else 0.0

    def _snap_lut(self, f: int):
        """(labels image, label -> (x, y)) so any pixel maps to its nearest mask pixel."""
        cached = self._snap_cache.get(f)
        if cached is not None:
            return cached
        hist = self._mask_hist.get(f)
        if hist is None:
            return None
        mask, _ = hist
        m8 = mask.astype(np.uint8)
        if not m8.any():
            return None
        inv = (1 - m8).astype(np.uint8)          # zeros = mask pixels = snap targets
        _, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(m8)
        lut = np.zeros((int(labels.max()) + 1, 2), np.float32)
        lut[labels[ys, xs]] = np.stack([xs, ys], axis=1)
        out = (labels, lut)
        self._snap_cache[f] = out
        return out

    def _constrain_to_mask(self, w0: int, tr: np.ndarray, specs: list[PointSpec],
                           out_map: list[tuple], first_seg: bool,
                           col_idx: list[int] | None = None) -> None:
        """Physical constraint: a point on the segment cannot be off the segment.
        Any constrained query predicted outside the frame's silhouette is moved
        to the nearest silhouette pixel (in place, so the LK chain continues
        from there). The user's own seed row is never touched. A large snap on
        the window's last row asks for a segment restart so CoTracker itself
        re-seeds from the corrected position. `col_idx[a]` is the output
        column of specs[a] (they differ once an ROI restart dropped a spec)."""
        self._snap_big = False
        if col_idx is None:
            col_idx = list(range(len(specs)))
        self._col2q = {a: om[1] for a, om in enumerate(out_map) if om[0] == "point"}
        # the merged-landmark demotion writes OUTPUT columns: after a restart
        # that dropped a spec, the spec index is not the column any more and
        # the cap landed on an innocent neighbour (I116)
        self._q2col = {om[1]: col_idx[a] for a, om in enumerate(out_map) if om[0] == "point"}
        for i in range(tr.shape[0]):
            # every window rewrites its rows' flags from scratch: a flag left
            # from the previous segment would name that segment's query indices
            self._snapped.pop(w0 + i, None)
            self._collapsed.pop(w0 + i, None)
        if not self.constrain or self.animal is None:
            return
        qs = []
        landmark_qs: list[tuple[int, np.ndarray]] = []    # (query, native seed) of skeleton POINTS
        landmark_pid: dict[int, int] = {}                 # query -> pid, for the exit report
        for a, (sp, om) in enumerate(zip(specs, out_map)):
            if sp.pid not in self.constrain:
                continue
            if om[0] == "point":
                qs.append(om[1])
                # "clearly apart when clicked" must mean the USER's seeds: after
                # a segment restart the specs are re-seeded from the emitted
                # positions, and two merged landmarks then restart on one pixel.
                # seed_xy is indexed by output column, not by spec index (I116)
                k = col_idx[a]
                seed0 = (self.seed_xy[k] if k < len(self.seed_xy) and np.isfinite(self.seed_xy[k]).all()
                         else sp.seed)
                landmark_qs.append((om[1], np.asarray(seed0, np.float64)))
                landmark_pid[om[1]] = sp.pid
            else:
                qs.extend(range(om[1].start, om[1].stop))   # group members stay on the body too
        if not qs:
            return
        L = tr.shape[0]
        for i in range(L):
            f = w0 + i
            if first_seg and f == self.start_frame:
                continue
            hist = self._mask_hist.get(f)
            lut = self._snap_lut(f)
            if hist is None or lut is None:
                continue
            mask, scale = hist
            labels, table = lut
            h, w = mask.shape
            summ = self._summ.get(f)
            if summ is not None and summ["area"] > 0:
                x0, y0, x1, y1 = summ["bbox"]
                diag = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1)) / scale
            else:
                diag = float(np.hypot(w, h))
            band_w = max(EXIT_BAND_MIN_PX / scale, EXIT_BAND_FRAC * diag)   # working px, like `diag`
            snapped = self._snapped.setdefault(f, set())
            snapped.clear()                     # overlap rows are rewritten by each window
            for q in qs:
                p = tr[i, q]
                if not np.isfinite(p).all():
                    continue
                xw, yw = self._n2w(p, scale)
                xi = min(max(int(round(xw)), 0), w - 1)
                yi = min(max(int(round(yw)), 0), h - 1)
                if mask[yi, xi]:
                    continue
                tx, ty = table[labels[yi, xi]]
                moved = float(np.hypot(tx - xw, ty - yw))
                if q in landmark_pid and moved > band_w:
                    # The landmark has LEFT the segment. Pulling it back to the
                    # nearest silhouette pixel would put it on some other spot
                    # of the animal and carry on as if nothing happened (it
                    # does not guarantee the same spot). Stop the
                    # run at this frame instead, and invent nothing: this frame
                    # and the rest of the window are blank for this point.
                    if self._exit_hit is None or f < self._exit_hit[0]:
                        self._exit_hit = (f, landmark_pid[q])
                    tr[i:, q] = np.nan
                    continue
                # within the band: outline jitter, nudge onto the silhouette
                # (group members are internal samples and are always nudged)
                tr[i, q] = self._w2n((tx, ty), scale)
                snapped.add(q)
            # Landmark identity: snapping to the nearest silhouette pixel lets
            # neighbouring landmarks land on the SAME pixel and stay merged
            # (snout / eye / shoulder in a real lizard test). Two landmarks that
            # were clearly apart when clicked and now sit within a couple of
            # per cent of the body diagonal of each other have collapsed: flag
            # both, so their confidence drops and auto-pause can stop the run.
            # Positions are left alone -- pushing them apart would invent data.
            collapsed = self._collapsed.setdefault(f, set())
            collapsed.clear()
            if len(landmark_qs) >= 2:
                # `diag` is in WORKING px (bbox is native, divided by scale above);
                # scale = native / working, so native = working * scale
                min_sep_w = max(COLLAPSE_MIN_PX, COLLAPSE_FRAC * diag)             # working px
                min_sep_n = min_sep_w * scale                                       # native px
                for u in range(len(landmark_qs)):
                    qa, sa = landmark_qs[u]
                    pa = tr[i, qa]
                    if not np.isfinite(pa).all():
                        continue
                    for v in range(u + 1, len(landmark_qs)):
                        qb, sb = landmark_qs[v]
                        pb = tr[i, qb]
                        if not np.isfinite(pb).all():
                            continue
                        if (float(np.linalg.norm(pa - pb)) < min_sep_n
                                and float(np.linalg.norm(sa - sb)) > 2.0 * min_sep_n):
                            collapsed.add(qa)
                            collapsed.add(qb)
            ref = self._refined.get(f)
            if ref is not None:
                for q in qs:
                    ref[q] = tr[i, q]
        if self._exit_hit is not None and self._autopause_hit is None:
            # stops the segment loop after this window is emitted; the app cuts
            # the point's track at the exit frame and takes the user there
            self._autopause_hit = self._exit_hit
            self._autopause_reason = "exit"

    def _demote_off_body(self, w0: int, out_tr: np.ndarray, out_cf: np.ndarray) -> None:
        """A skeleton point that sits outside the (dilated) segment mask has
        drifted onto the background: cap its confidence so the timeline turns
        red and auto-pause can stop the run. Coordinates are left alone."""
        if not self.on_body or self.animal is None:
            return
        L = out_tr.shape[0]
        cols = [a for a, sp in enumerate(self.specs) if sp.pid in self.on_body]
        if not cols:
            return
        for i in range(L):
            f = w0 + i
            hist = self._mask_hist.get(f)
            summ = self._summ.get(f)
            if hist is None or summ is None or summ["area"] <= 0:
                continue
            mask, scale = hist
            dil = self._dil_cache.get(f)
            if dil is None:
                x0, y0, x1, y1 = summ["bbox"]
                pad = max(8.0, 0.03 * float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))) / scale
                k = 2 * int(np.ceil(pad)) + 1
                dil = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8))
                self._dil_cache[f] = dil
            h, w = dil.shape
            for a in cols:
                p = out_tr[i, a]
                if not np.isfinite(p).all():
                    continue
                xw, yw = self._n2w(p, scale)
                xi, yi = int(round(xw)), int(round(yw))
                if 0 <= xi < w and 0 <= yi < h and dil[yi, xi] == 0:
                    out_cf[i, a] = min(out_cf[i, a], OFF_BODY_CONF)

    def _demote_merged(self, w0: int, out_cf: np.ndarray) -> None:
        """Merged landmarks (flagged in _constrain_to_mask): the same cap as an
        off-body point, so the timeline turns red and auto-pause treats them
        as lost. Positions are left alone. This runs for every CONSTRAINED
        point, skeleton or not: the app constrains points placed with N on a
        segment without a skeleton too, and behind the skeleton-only guard of
        _demote_off_body two of them could sit on one pixel at full
        confidence (I117). Columns come from `_q2col`, not the spec index
        (I116)."""
        if not self.constrain or self.animal is None:
            return
        for i in range(out_cf.shape[0]):
            merged = self._collapsed.get(w0 + i)
            if not merged:
                continue
            for q in merged:
                k = self._q2col.get(q)
                if k is not None and k < out_cf.shape[1]:
                    out_cf[i, k] = min(out_cf[i, k], OFF_BODY_CONF)

    def _check_animal_lost(self, w0: int, L: int) -> None:
        """Auto-pause when the segment has not been present for ANIMAL_LOST_RUN
        consecutive frames (left the frame, or the segmentation lost it)."""
        if not self.autopause or self.animal is None or self._autopause_hit is not None:
            return
        for f in range(w0, w0 + L):
            if f <= self._animal_counted_until:
                continue
            self._animal_counted_until = f
            summ = self._summ.get(f)
            if summ is None:
                continue
            if summ["area"] <= 0 or summ["score"] <= 0:
                if self._lost_run == 0:
                    self._lost_start = f
                self._lost_run += 1
                if self._lost_run >= ANIMAL_LOST_RUN:
                    self._autopause_hit = (self._lost_start, -1)
                    self._autopause_reason = "lost"
                    return
            else:
                self._lost_run = 0

    def _emit_masks(self, w0: int, L: int) -> None:
        if self.animal is None:
            return
        summaries = [self._summ[f] for f in range(w0, w0 + L) if f in self._summ]
        if summaries:
            self.masks_ready.emit(summaries)

    def _run_animal_only(self, src: VideoSource, start: int) -> int:
        return _drain(self._animal_only_steps(src, start))

    def _animal_only_steps(self, src: VideoSource, start: int):
        """Segment every frame from `start` (no point tracker): masks, midline
        and derived landmarks, emitted in 8-frame chunks. Returns the last
        emitted frame (start - 1 if none). A generator: yields after every
        frame (I141)."""
        step = 8
        nw, nh = self._frame_wh
        disp_scale = min(1.0, WORKING_MAX_DIM / max(nw, nh))
        dw, dh = max(2, round(nw * disp_scale)), max(2, round(nh * disp_scale))
        nd = len(self.derived)
        n_track = len(self.specs)   # 0 on a pure animal run; >0 after every point was dropped
        K = self.n_cols
        pending: list[tuple[int, np.ndarray]] = []
        buf: list[int] = []
        last = start - 1
        src.seek(start)

        def flush():
            nonlocal last, pending, buf
            if not buf:
                return
            w0, L = buf[0], len(buf)
            out_tr = np.full((L, K, 2), np.nan, np.float32)
            out_vi = np.zeros((L, K), bool)
            out_cf = np.zeros((L, K), np.float32)
            self._fill_derived(w0, out_tr, out_vi, out_cf)
            self._fill_balls(w0, out_tr, out_vi, out_cf)
            self._check_animal_lost(w0, L)
            self._emit_masks(w0, L)
            self.chunk_ready.emit(w0, out_tr, out_vi, out_cf, {}, list(pending))
            last = w0 + L - 1
            pending, buf = [], []

        reader = ReadAhead(src)   # same overlap for the segment-only path
        want, read_end = start, None
        try:
            while not self._pause and self._autopause_hit is None:
                nxt = reader.read_next()
                if nxt is None:
                    read_end = want
                    break
                if nxt[0] >= self.n_frames:
                    break
                abs_idx, native = nxt
                want = abs_idx + 1
                if self._seg is not None and abs_idx > self._seg_last:
                    self._seg_step(abs_idx, native)
                if self._ball_trk is not None and abs_idx > self._ball_last:
                    self._ball_step(abs_idx, native)
                pending.append((abs_idx, native if disp_scale >= 1.0
                                else cv2.resize(native, (dw, dh), interpolation=cv2.INTER_AREA)))
                buf.append(abs_idx)
                if len(buf) >= step:
                    flush()
                yield                           # (I141)
        finally:
            reader.stop()
        if read_end is not None:
            self._check_read_end(read_end)   # (I40)
        flush()
        return last

    # ----------------------------------------------------------- ball markers

    def _ball_step(self, abs_idx: int, native: np.ndarray) -> None:
        """SAM-segment every ball on one frame (strictly sequential) and fit
        the circles. Prompts placed on this frame are applied first; on the
        run's first frame a ball without a prompt is seeded from its position.
        A ball SAM cannot find any more is dropped: if its last circle sat at
        the picture edge it simply left (no data, no pause - the out-of-frame
        invariant); otherwise it was lost inside the picture and the run
        pauses there so the user can click it again."""
        from kinetrace.balls import BallPrompt
        trk = self._ball_trk
        prompts: list[BallPrompt] = []
        for b in self.balls:
            if b.pid in self._ball_gone:
                continue
            obj = self._ball_obj[b.pid]
            cs = b.prompts.get(abs_idx)
            if cs:
                pts = np.array([[c[0], c[1]] for c in cs], np.float32)
                labs = np.array([int(c[2]) for c in cs], np.int64)
                r = b.radius
                st = trk.active.get(obj)
                if st is not None and st.r:
                    r = float(st.r)
                prompts.append(BallPrompt(obj, points=pts, labels=labs, radius=r))
            elif abs_idx == self.start_frame and b.seed is not None and np.isfinite(b.seed).all() \
                    and not trk.has(obj):
                prompts.append(BallPrompt(obj, points=b.seed.reshape(1, 2), labels=np.array([1]),
                                          radius=b.radius))
        before = {obj: (st.xy, st.r, st.miss) for obj, st in trk.active.items()}
        fits = trk.step(native, abs_idx, prompts)
        self._ball_last = abs_idx
        inv = {obj: pid for pid, obj in self._ball_obj.items()}
        self._ball_res[abs_idx] = {inv[obj]: fit for obj, fit in fits.items() if obj in inv}
        nw, nh = self._frame_wh
        for obj, (xy, r, miss) in before.items():
            if trk.has(obj) or obj not in inv:
                continue
            pid = inv[obj]
            rr = float(r or 8.0)
            edge = min(xy[0], xy[1], nw - 1 - xy[0], nh - 1 - xy[1]) <= 2.0 * rr + 6.0
            self._ball_gone.add(pid)
            if edge:
                continue
            # the first frame without an accepted circle
            first_bad = max(self.start_frame, abs_idx - miss)
            # "apart" (the balls no longer fit one crop, I58) is kept as a fallback:
            # since I143 far-apart balls get a crop per group and are not dropped for it
            why = "apart" if getattr(trk, "drop_reason", {}).get(obj) == "apart" else "lost"
            # kept with auto-pause OFF too, so the app can say which ball ended where (I127)
            self._ball_ended[int(pid)] = (int(first_bad), why)
            if self.autopause and self._autopause_hit is None:
                self._autopause_hit = (int(first_bad), int(pid))
                self._autopause_reason = why
        for k in [k for k in self._ball_res if k < abs_idx - 4 * MASK_HIST]:
            del self._ball_res[k]

    def _fill_balls(self, w0: int, out_tr: np.ndarray, out_vi: np.ndarray, out_cf: np.ndarray) -> None:
        """Ball columns of a window from the per-frame circle fits; emits the
        radii (the chunk carries the centres)."""
        if not self.balls:
            return
        L = out_tr.shape[0]
        k0 = len(self.specs) + len(self.derived)
        radii = []
        for i in range(L):
            f = w0 + i
            res = self._ball_res.get(f, {})
            for j, b in enumerate(self.balls):
                fit = res.get(b.pid)
                k = k0 + j
                if fit is None:
                    continue
                out_tr[i, k] = (fit.x, fit.y)
                out_vi[i, k] = True
                out_cf[i, k] = fit.confidence
                radii.append((int(f), int(b.pid), float(fit.r)))
        if radii:
            self.balls_ready.emit(radii)

    # -------------------------------------------------------------- detectors

    def _check_autopause(self, w0: int, out_tr: np.ndarray, out_vi: np.ndarray,
                         out_cf: np.ndarray, specs: list[PointSpec],
                         col_idx: list[int]) -> None:
        """Sustained low confidence on an IN-FRAME prediction means the model has
        LOST the point, not that it is occluded — an occluded point keeps high
        confidence (measured: max 7-8 consecutive low frames under a 46-frame
        opaque occlusion, against 130+ after a real loss), which is why the run
        length is the separator.

        Two consequences, deliberately independent:

        - **auto-pause** (opt-in, default on) stops the run and takes the user to
          the failing frame.
        - **the point's data is blanked** from the moment it is declared lost,
          for as long as confidence stays down. This is the visibility-based
          fallback the out-of-frame rule cannot provide on its own: when a point
          walks out of the frame the model parks its guess INSIDE the frame, so
          the bounds check never fires and the export keeps emitting coordinates
          for something that is not there (measured on the exit clip: 100% of the
          out-of-frame window kept bogus coordinates). Blanking runs even with
          auto-pause off, because it is about what the data MEANS, not about
          interrupting the user.

        The first CONF_PAUSE_RUN frames of a dip are kept: while it could still
        be an occlusion, the coordinates are the best estimate available.
        Overlap rows are rewritten by each window, so each frame is counted once.

        A lost point counts as recovered only after CONF_PAUSE_RUN consecutive
        frames back (I118). Each window re-emits its 8 overlap rows, and the
        session keeps the LAST write: dropping the lost state on the first good
        frame let the next window write the model's parked guess back into the
        gone frames just before a recovery, or around a one-frame flicker.
        Rows that are not gone are never blanked, so the hysteresis costs a
        recovered point nothing.
        """
        nw, nh = self._frame_wh
        L = out_tr.shape[0]

        def blank(rows, k):
            out_tr[rows, k] = np.nan
            out_vi[rows, k] = False
            out_cf[rows, k] = 0.0

        for i in range(L):
            f = w0 + i
            new_row = f > self._counted_until
            if new_row:
                self._counted_until = f
            for a, sp in enumerate(specs):
                k = col_idx[a]
                p = out_tr[i, k]
                inside = bool(in_frame(p, nw, nh))
                low = inside and out_cf[i, k] < CONF_PAUSE_THRESHOLD
                # Blanking is STRICTLY narrower than pausing: the model must
                # also report the point as not visible. Low confidence alone is
                # not enough, because the on-body constraint deliberately
                # demotes a point that sits off the silhouette to OFF_BODY_CONF
                # (below the pause threshold) while it is still being tracked
                # perfectly well — that is a demotion, not a loss.
                # A point under the on-body constraint is judged differently: the
                # constraint DELIBERATELY overrides the model (it snaps the point
                # back onto the silhouette and re-seeds from there), so a plain
                # low-confidence is not a loss. But when the model has given up
                # (not visible, low confidence) AND the constraint had to move
                # the point onto the body on this frame, the position is the
                # constraint's invention, not a track: that IS lost. Without
                # this, one camera's neck landmark in a real stereo test exported 601 frames
                # of confident-but-wrong coordinates.
                constrained = sp.pid in self.constrain
                q = self._col2q.get(a)
                snapped_here = (constrained and q is not None
                                and q in self._snapped.get(f, ()))
                gone = (low and not out_vi[i, k] and (not constrained or snapped_here))
                if new_row and inside:     # OOB neither counts nor resets the run
                    if low:
                        if self._low_run[sp.pid] == 0:
                            self._low_start[sp.pid] = f  # OOB gaps make f-RUN+1 wrong
                        self._low_run[sp.pid] += 1
                        if (self._low_run[sp.pid] >= CONF_PAUSE_RUN
                                and self.autopause and self._autopause_hit is None):
                            self._autopause_hit = (self._low_start[sp.pid], sp.pid)
                            self._autopause_reason = "lowconf"
                    else:
                        self._low_run[sp.pid] = 0
                    if gone:
                        if self._gone_run[sp.pid] == 0:
                            self._gone_start[sp.pid] = f
                        self._gone_run[sp.pid] += 1
                        self._back_run[sp.pid] = 0
                        if (self._gone_run[sp.pid] >= CONF_PAUSE_RUN
                                and sp.pid not in self._lost):
                            self._lost.add(sp.pid)
                            # blank back to the start of the run, as far as this
                            # window still reaches (the session overwrites
                            # overlap rows, so for those it is not too late)
                            start = max(self._gone_start[sp.pid], w0)
                            blank(slice(start - w0, i + 1), k)
                    else:
                        self._gone_run[sp.pid] = 0
                        if sp.pid in self._lost:
                            # recovered only once it has been back for a full
                            # run: see the docstring (I118)
                            self._back_run[sp.pid] = self._back_run.get(sp.pid, 0) + 1
                            if self._back_run[sp.pid] >= CONF_PAUSE_RUN:
                                self._lost.discard(sp.pid)
                                self._back_run[sp.pid] = 0
                if sp.pid in self._lost and gone:
                    blank(i, k)

    # ----------------------------------------------------------- refinement

    def _capture_templates(self, gray: np.ndarray, specs: list[PointSpec],
                           out_map: list[tuple]) -> None:
        """Grab the seed-appearance NCC template for each anchor point."""
        r = ANCHOR_TEMPLATE // 2
        h, w = gray.shape
        for sp, om in zip(specs, out_map):
            if sp.kind != "point" or not sp.anchor:
                continue
            x, y = int(round(sp.seed[0])), int(round(sp.seed[1]))
            # a clipped patch would put the feature off the template center and
            # bias every snap — near the frame edge the anchor simply stays off
            if x - r < 0 or y - r < 0 or x + r + 1 > w or y + r + 1 > h:
                continue
            self._templates[sp.pid] = gray[y - r:y + r + 1, x - r:x + r + 1].copy()
            # the patch is centred on the ROUNDED seed; the click sits this far
            # from that pixel, and every snap must add it back or the track
            # shifts by up to half a pixel per axis (I121)
            self._template_frac[sp.pid] = (np.asarray(sp.seed, np.float32)
                                           - np.array([x, y], np.float32))

    def _reanchor(self, gray: np.ndarray, template: np.ndarray,
                  xy: np.ndarray) -> np.ndarray | None:
        """NCC search around xy; returns the subpixel peak if trustworthy."""
        th, tw = template.shape
        sr = int(np.ceil(4 * self._gate))          # search radius beyond template
        h, w = gray.shape
        x, y = int(round(xy[0])), int(round(xy[1]))
        x0 = max(0, x - tw // 2 - sr)
        y0 = max(0, y - th // 2 - sr)
        x1 = min(w, x + tw // 2 + sr + 1)
        y1 = min(h, y + th // 2 + sr + 1)
        search = gray[y0:y1, x0:x1]
        if search.shape[0] < th + 2 or search.shape[1] < tw + 2:
            return None
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        if peak < ANCHOR_MIN_CORR:
            return None
        px, py = loc
        # 3-point parabola per axis for subpixel localization
        def parab(v0, v1, v2):
            d = v0 - 2 * v1 + v2
            return 0.0 if abs(d) < 1e-9 else 0.5 * (v0 - v2) / d
        dx = parab(res[py, px - 1], res[py, px], res[py, px + 1]) \
            if 0 < px < res.shape[1] - 1 else 0.0
        dy = parab(res[py - 1, px], res[py, px], res[py + 1, px]) \
            if 0 < py < res.shape[0] - 1 else 0.0
        return np.array([x0 + px + dx + tw // 2, y0 + py + dy + th // 2], np.float32)

    def _refine_window(self, w0: int, tracks: np.ndarray, vis: np.ndarray,
                       gray_hist: deque[tuple[int, np.ndarray]], seg_start: int,
                       anchor_q: dict[int, PointSpec]) -> np.ndarray:
        """Native-resolution subpixel polish, chained from the segment seed.

        For each frame f in the window, LK-track each visible point from its
        refined position at f-1 into f, seeded with CoTracker's prediction as
        the initial guess. Accept only within `gate` px of the CoTracker
        prediction — the gate both bounds LK drift and re-anchors the chain.
        Anchor points additionally snap to their seed-appearance NCC peak
        when the match is strong.
        """
        grays = dict(gray_hist)
        gate = self._gate
        out = tracks.copy()
        L = tracks.shape[0]
        for i in range(L):
            f = w0 + i
            g1 = grays.get(f)
            if f == seg_start:
                self._refined[f] = out[i].copy()
                continue
            prev_pts = self._refined.get(f - 1)
            g0 = grays.get(f - 1)
            if self._do_refine and prev_pts is not None and g0 is not None and g1 is not None:
                valid = vis[i] & np.isfinite(prev_pts).all(axis=1)
                if valid.any():
                    p0 = prev_pts[valid].reshape(-1, 1, 2).astype(np.float32)
                    init = out[i][valid].reshape(-1, 1, 2).astype(np.float32).copy()
                    p1, _st, _ = cv2.calcOpticalFlowPyrLK(
                        g0, g1, p0, init,
                        winSize=(self._lk_win, self._lk_win), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                    p1 = p1.reshape(-1, 2)
                    # The gate vs CoTracker's prediction is the safety mechanism (LK's
                    # own status bit is unreliable on locally-flat patches): accept a
                    # refinement only if finite, inside the frame, and within the gate.
                    fw, fh = self._frame_wh
                    ok = (in_frame(p1, fw, fh)
                          & (np.linalg.norm(p1 - out[i][valid], axis=1) <= gate))
                    sel = np.nonzero(valid)[0][ok]
                    out[i][sel] = p1[ok]
            if anchor_q and g1 is not None:
                for q, sp in anchor_q.items():
                    tpl = self._templates.get(sp.pid)
                    if tpl is None or not np.isfinite(out[i][q]).all():
                        continue
                    snapped = self._reanchor(g1, tpl, out[i][q])
                    if snapped is not None:
                        out[i][q] = snapped + self._template_frac.get(sp.pid, 0.0)
            self._refined[f] = out[i].copy()
        # keep the chain buffer bounded
        for k in list(self._refined):
            if k < w0 - 2:
                del self._refined[k]
        return out
