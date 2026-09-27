"""Tracking session data model: point tracks, the segment silhouette, the
skeleton, persistence, exports.

Arrays are (T, N, ...) where T = video frame count and N = number of points.
NaN in `tracks` (mirrored by `tracked == False`) means "no data for this
point at this frame". Even a 40k-frame, 50-point session is ~20 MB, so the
whole model lives in RAM; it is saved as part of a `.kinetrace` project file
(projectfile.py).

What a session holds, beyond the tracks:
- `confidence` (T, N): CoTracker3's per-frame track-correctness score in
  [0, 1] (1.0 for manual placements). Drives the timeline coloring and the
  auto-pause detector. Distinct from `visibility` — an occluded point keeps
  high confidence; a *lost* point does not.
- Region groups: a point with kind == "group" is the fitted center of a
  user-drawn circular region (radius in native px). Its member points are
  tracker-internal and never persisted — the group behaves like a normal
  point everywhere else (exports, undo, corrections).
- Events: named frame windows for navigation, persisted and exported.
- `ui_state`: everything needed to reopen a project exactly as it was left
  (view transform, selection, toolbar toggles).

The segment layer:
- ONE segment per project (`AnimalMeta`): the user's click/box prompts per
  frame, and its per-frame silhouette in a compact `MaskTrack` (outlines,
  midline, bbox, area, centroid, presence score).
- Points carry a `source`: "track" (appearance-tracked by CoTracker3, the
  user seeds it) or "silhouette" (derived from the segment's mask by `spec`,
  e.g. "tip", "midline:0.5", "ext:FL"). Derived points are written into the
  same (T, N) arrays by the tracking run, so exports, timeline, undo and
  corrections work unchanged; they simply cannot be seeded by hand.
- A named `skeleton` (landmark names, bones, head landmark, derived defaults)
  from a template; landmarks are ordinary points matched by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kinetrace.body import BodyTrack, export_angles_csv, export_joints_csv
from kinetrace.segmenter import MIDLINE_SAMPLES, MaskTrack

# 12 visually distinct point colors (RGB), cycled by point index.
PALETTE: list[tuple[int, int, int]] = [
    (230, 60, 60), (60, 200, 60), (70, 130, 255), (255, 200, 40),
    (200, 80, 220), (40, 220, 220), (255, 130, 40), (150, 220, 60),
    (255, 100, 160), (120, 120, 255), (180, 140, 90), (240, 240, 240),
]
ANIMAL_COLOR: tuple[int, int, int] = (77, 227, 176)   # mint — outside the point palette


# Reopening a project must restore the exact working state, not just tracks.
DEFAULT_UI_STATE: dict = {
    "selected": -1,          # selected point id, -1 = none
    "follow": False,         # ⌖ Follow toggle — OFF by default:
                             # an automatic re-frame surprises more than it helps
    "autopause": True,       # pause tracking on confidence collapse
    "roi": True,             # ROI-zoom tracking
    "track_mode": "auto",    # "auto" = run to end | "semi" = F steps one frame
    "marker_size": 3,        # marker radius, screen px (was 7 until 2026-09-22)
    "zoom": 0.0,             # canvas scale; 0 = never zoomed (fit on open)
    "center_x": 0.0,         # scene point at the viewport center
    "center_y": 0.0,
    "user_zoomed": False,
    "show_mask": True,       # animal silhouette overlay
    "epipolar": True,        # dashed guides from the other cameras for the selected landmark (needs a calibration)
    "mask_opacity": 0.35,
    "show_midline": True,
    "show_bones": True,
    "seg_backend": "",       # "" = the app's preferred backend (SAM 3 if its weights are present)
    "on_body": True,         # tracked points are kept inside the animal's silhouette
    "point_backend": "",     # "" = preferred (AllTracker when vendored, else CoTracker3)
    "trail_len": 30,         # trajectory trail length in frames (0 = off)
    "trail_future": False,   # also draw the upcoming path (dashed)
    "onion": False,          # onion skin: ghosts of the previous / next frame
    "loupe": False,          # magnifier under the cursor
    "display_filter": "none",  # none | contrast | bright | diff (display only)
    "region_shape": "circle",  # what an armed drag draws: circle | rect | polygon
}

SOURCES = ("track", "silhouette", "ball")


def _sanitize(name: str) -> str:
    """Make a point/event name safe for CSV/TSV headers and cells."""
    return name.replace(",", "_").replace("\t", "_").replace("\n", " ").strip() or "point"


def dltdv_convention_text(flip_y: bool, pixel_origin: float) -> str:
    """One plain sentence naming the pixel convention an xypts file was
    written in, for its sidecar and the export dialog."""
    if flip_y:
        return "bottom-left origin; y counted from the bottom edge; first pixel = 1 (older DLTdv, Argus Clicker)"
    return (f"top-left origin; first pixel = {pixel_origin:g} "
            f"({'DLTdv8 / MATLAB' if abs(pixel_origin - 1.0) < 1e-9 else 'OpenCV / Kinetrace'})")


@dataclass
class PointMeta:
    name: str
    color: tuple[int, int, int]
    display: bool = True
    kind: str = "point"     # "point" | "group" (future shapes: rect, polygon)
    radius: float = 0.0     # group region radius in native px (0 for points)
    anchor: bool = False    # opt-in appearance re-anchor (NCC snap-back)
    source: str = "track"   # "track" (CoTracker3) | "silhouette" (derived from the mask)
    spec: str = ""          # derivation spec for silhouette points (see skeletons.py)
    free: bool = False      # may leave the animal (exempt from the on-body constraint)
    shape: str = "circle"   # region outline: "circle" | "rect" | "polygon" (groups only)
    outline: list | None = None   # rect/polygon vertices [[x, y], ...] native px at the seed
    # ball markers (source "ball"): the user's SAM clicks per frame,
    # {frame: [[x, y, label], ...]} native px, label 1 = the ball / 0 = not it
    ball_prompts: dict | None = None

    @property
    def derived(self) -> bool:
        return self.source == "silhouette"

    @property
    def is_ball(self) -> bool:
        """Tracked by segmenting the ball with SAM and fitting a circle (balls.py)."""
        return self.source == "ball"

    def copy(self) -> "PointMeta":
        return PointMeta(self.name, self.color, self.display,
                         self.kind, self.radius, self.anchor, self.source, self.spec,
                         self.free, self.shape,
                         None if self.outline is None else [list(v) for v in self.outline],
                         None if self.ball_prompts is None
                         else {int(f): [list(c) for c in cs] for f, cs in self.ball_prompts.items()})

    def outline_at(self, center) -> np.ndarray | None:
        """The region outline translated to `center` (its fitted position on a
        frame): (M, 2) float32, or None for a circle / plain point."""
        if self.outline is None or len(self.outline) < 3:
            return None
        pts = np.asarray(self.outline, np.float32).reshape(-1, 2)
        seed_c = pts.mean(axis=0)
        return pts - seed_c + np.asarray(center, np.float32)


@dataclass
class Event:
    """A named frame window for navigation (start <= end, inclusive)."""
    name: str
    start: int
    end: int
    color: tuple[int, int, int]
    note: str = ""          # free text (what happened, why it matters)
    author: str = ""        # who marked it (the annotator name from Settings)


@dataclass
class AnimalMeta:
    """The one tracked segment: name, color, and the user's segmentation
    prompts per frame (clicks with 1 = segment / 0 = not segment, and boxes)."""
    name: str = "segment"
    color: tuple[int, int, int] = ANIMAL_COLOR
    prompts: dict[int, list[tuple[float, float, int]]] = field(default_factory=dict)
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)

    def add_click(self, frame: int, x: float, y: float, positive: bool = True) -> None:
        self.prompts.setdefault(int(frame), []).append((float(x), float(y), 1 if positive else 0))

    def set_box(self, frame: int, box) -> None:
        x0, y0, x1, y1 = (float(v) for v in box)
        self.boxes[int(frame)] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


    def has_prompt(self, frame: int) -> bool:
        return bool(self.prompts.get(int(frame))) or int(frame) in self.boxes

    def prompt_frames(self) -> list[int]:
        return sorted(set(self.prompts) | set(self.boxes))

    def n_prompts(self) -> int:
        return sum(len(v) for v in self.prompts.values()) + len(self.boxes)

    def copy(self) -> "AnimalMeta":
        return AnimalMeta(self.name, self.color,
                          {f: list(v) for f, v in self.prompts.items()}, dict(self.boxes))

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "color": list(self.color),
                           "prompts": {str(f): v for f, v in self.prompts.items()},
                           "boxes": {str(f): list(b) for f, b in self.boxes.items()}})

    @staticmethod
    def from_json(s: str) -> "AnimalMeta":
        d = json.loads(s)
        a = AnimalMeta(str(d.get("name", "segment")), tuple(int(c) for c in d.get("color", ANIMAL_COLOR)))
        for f, v in d.get("prompts", {}).items():
            a.prompts[int(f)] = [(float(p[0]), float(p[1]), int(p[2])) for p in v]
        for f, b in d.get("boxes", {}).items():
            a.boxes[int(f)] = tuple(float(x) for x in b)
        return a


@dataclass
class Snapshot:
    tracks: np.ndarray
    visibility: np.ndarray
    manual: np.ndarray
    tracked: np.ndarray
    confidence: np.ndarray
    points: list[PointMeta] = field(default_factory=list)
    masks: MaskTrack | None = None
    has_animal: bool = False
    occluded: np.ndarray | None = None
    body: BodyTrack | None = None
    radius: np.ndarray | None = None       # ball markers: fitted radius per frame
    skeleton: dict | None = None           # names / bones / head: renamed with the points (I22)


HEAD_HINTS = ("snout", "nose", "head", "beak", "rostrum", "mouth")


class TrackingSession:
    def __init__(self, video_path: str, n_frames: int, fps: float, width: int, height: int):
        self.video_path = video_path
        self.n_frames = n_frames
        self.fps = fps
        self.width = width
        self.height = height
        self.tracks = np.full((n_frames, 0, 2), np.nan, np.float32)
        self.visibility = np.zeros((n_frames, 0), bool)
        self.manual = np.zeros((n_frames, 0), bool)
        self.tracked = np.zeros((n_frames, 0), bool)
        self.confidence = np.zeros((n_frames, 0), np.float32)
        # hand-marked "hidden here": the data stays (so the mark can be undone)
        # but the cell is NOT exported and NOT used for 3D
        self.occluded = np.zeros((n_frames, 0), bool)
        # ball markers: the fitted circle's radius per frame (NaN elsewhere) so
        # the canvas can draw the circle SAM found, not just its centre
        self.radius = np.full((n_frames, 0), np.nan, np.float32)
        self.points: list[PointMeta] = []
        self.events: list[Event] = []
        self.notes: dict[int, dict] = {}    # frame -> {"text", "author", "time"}
        self.annotator = ""                 # who is digitizing (Settings)
        self.animal: AnimalMeta | None = None
        self.masks: MaskTrack | None = None
        # Human body poses (joints + angles) from the body layer. None until a
        # pose run has produced something; it is a whole-video array, not a
        # per-point one, so it lives beside `masks` rather than in `points`.
        self.body: BodyTrack | None = None
        self.skeleton: dict | None = None
        self.current_frame = 0  # persisted so projects reopen where the user left off
        self.ui_state: dict = dict(DEFAULT_UI_STATE)
        self.data_version = 0   # bumped on every mutation (timeline repaint/cache key)
        self.dirty = False
        self._name_counter = 0
        self._event_counter = 0

    @property
    def dirty(self) -> bool:
        return self._dirty

    @dirty.setter
    def dirty(self, value: bool) -> None:
        # Marking a change also moves data_version on: a save compares it
        # before and after, so an edit made while a save ran stays unsaved
        # whichever code path marked it.
        if value:
            self.data_version += 1
        self._dirty = bool(value)

    def _touch(self) -> None:
        self.dirty = True

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        """This one camera as a Kinetrace project file (see projectfile.py)."""
        from kinetrace import projectfile
        from kinetrace.project import Project
        projectfile.save(Project([self]), path)

    @staticmethod
    def load(path) -> "TrackingSession":
        """The active camera of a project file."""
        from kinetrace import projectfile
        p = projectfile.load(path)
        return p.sessions[p.active]

    # ----------------------------------------------------------------- points

    @property
    def n_points(self) -> int:
        return len(self.points)

    def unique_name(self, desired: str, exclude_pid: int | None = None) -> str:
        """`desired` if free, else 'desired (2)', 'desired (3)', … so a rename
        can never shadow another point's data. Compared through the export
        sanitizer too — "a,b" and "a_b" would otherwise merge into one CSV
        column even though the raw names differ."""
        desired = desired.strip() or "point"
        taken = {_sanitize(p.name) for i, p in enumerate(self.points) if i != exclude_pid}
        if _sanitize(desired) not in taken:
            return desired
        k = 2
        while _sanitize(f"{desired} ({k})") in taken:
            k += 1
        return f"{desired} ({k})"

    def _append_point(self, meta: PointMeta) -> int:
        self.points.append(meta)
        T = self.n_frames
        self.tracks = np.concatenate([self.tracks, np.full((T, 1, 2), np.nan, np.float32)], axis=1)
        self.visibility = np.concatenate([self.visibility, np.zeros((T, 1), bool)], axis=1)
        self.manual = np.concatenate([self.manual, np.zeros((T, 1), bool)], axis=1)
        self.tracked = np.concatenate([self.tracked, np.zeros((T, 1), bool)], axis=1)
        self.confidence = np.concatenate([self.confidence, np.zeros((T, 1), np.float32)], axis=1)
        self.occluded = np.concatenate([self.occluded, np.zeros((T, 1), bool)], axis=1)
        self.radius = np.concatenate([self.radius, np.full((T, 1), np.nan, np.float32)], axis=1)
        return self.n_points - 1

    def add_point(self, frame: int, x: float, y: float,
                  kind: str = "point", radius: float = 0.0, name: str | None = None,
                  shape: str = "circle", outline=None) -> int:
        self._name_counter += 1
        prefix = "G" if kind == "group" else "P"
        meta = PointMeta(self.unique_name(name or f"{prefix}{self._name_counter}"),
                         PALETTE[(self._name_counter - 1) % len(PALETTE)],
                         kind=kind, radius=float(radius),
                         shape=shape if shape in ("circle", "rect", "polygon") else "circle",
                         outline=None if outline is None else [[float(a), float(b)] for a, b in outline])
        pid = self._append_point(meta)
        self.set_position(frame, pid, x, y)
        return pid

    def add_ball(self, frame: int, x: float, y: float, name: str | None = None) -> int:
        """A ball marker: SAM segments the ball from this click, a circle is
        fitted per frame and its centre is the point (balls.py). The click is
        both the seed position and the first SAM prompt."""
        self._name_counter += 1
        meta = PointMeta(self.unique_name(name or f"ball {self._name_counter}"),
                         PALETTE[(self._name_counter - 1) % len(PALETTE)], source="ball",
                         ball_prompts={int(frame): [[float(x), float(y), 1]]})
        pid = self._append_point(meta)
        self.set_position(frame, pid, x, y)
        return pid

    def add_ball_prompt(self, pid: int, frame: int, x: float, y: float, label: int = 1) -> None:
        """Another SAM click on a ball on `frame` (a correction, or where the
        ball reappeared). A positive click also becomes its hand-placed position."""
        q = self.points[pid]
        if not q.is_ball:
            return
        if q.ball_prompts is None:
            q.ball_prompts = {}
        q.ball_prompts.setdefault(int(frame), []).append([float(x), float(y), int(label)])
        if label:
            self.set_position(frame, pid, x, y)
        self._touch()

    def ball_pids(self) -> list[int]:
        return [i for i, q in enumerate(self.points) if q.is_ball]

    def write_ball_radii(self, rows) -> None:
        """(frame, pid, radius) triples from the worker's circle fits."""
        for f, pid, r in rows:
            if 0 <= f < self.n_frames and 0 <= pid < self.n_points:
                self.radius[f, pid] = float(r)

    def add_landmark(self, name: str, source: str = "track", spec: str = "") -> int:
        """A named point with no data yet (a skeleton landmark waiting to be
        placed, or a silhouette-derived one filled in by the next run)."""
        self._name_counter += 1
        meta = PointMeta(self.unique_name(name), PALETTE[(self._name_counter - 1) % len(PALETTE)],
                         source=source if source in SOURCES else "track", spec=spec)
        pid = self._append_point(meta)
        self._touch()
        return pid

    def add_placeholder(self, meta: PointMeta) -> int:
        """Another camera's landmark `meta`, with no data here: same name, colour,
        kind and data source, so selecting it and clicking the video places it in
        THIS camera (project.sync_landmarks). A rectangle / polygon outline is that
        camera's own geometry, so the region arrives here as a circle of the same
        radius; ball clicks and the appearance lock are per camera too."""
        m = PointMeta(meta.name, meta.color, True, meta.kind, meta.radius, False, meta.source,
                      meta.spec, meta.free, meta.shape if meta.outline is None else "circle")
        pid = self._append_point(m)
        self._touch()
        return pid

    def remove_point(self, pid: int) -> None:
        keep = [i for i in range(self.n_points) if i != pid]
        self.tracks = self.tracks[:, keep]
        self.visibility = self.visibility[:, keep]
        self.manual = self.manual[:, keep]
        self.tracked = self.tracked[:, keep]
        self.confidence = self.confidence[:, keep]
        self.occluded = self.occluded[:, keep]
        self.radius = self.radius[:, keep]
        del self.points[pid]
        self._touch()

    def rename_point(self, pid: int, desired: str) -> str:
        """Rename with collision protection; returns the name actually applied.
        Skeleton references (landmarks, bones, head) follow the rename."""
        old = self.points[pid].name
        name = self.unique_name(desired, exclude_pid=pid)
        self.points[pid].name = name
        sk = self.skeleton
        if sk is not None and old in sk.get("landmarks", []):
            sk["landmarks"] = [name if n == old else n for n in sk["landmarks"]]
            sk["bones"] = [[name if n == old else n for n in b] for b in sk.get("bones", [])]
            if sk.get("head") == old:
                sk["head"] = name
            if old in sk.get("derived", {}):
                sk["derived"][name] = sk["derived"].pop(old)
        self._touch()
        return name

    def set_source(self, pid: int, source: str, spec: str = "") -> None:
        """Switch a point between appearance tracking and silhouette derivation.
        Data derived under the old rule is cleared (it would be stale)."""
        meta = self.points[pid]
        source = source if source in SOURCES else "track"
        if source == "silhouette":
            meta.spec = spec
            if meta.source != "silhouette" or spec != meta.spec:
                pass
            self.clear_window([pid], 0, self.n_frames - 1)
        elif meta.source == "silhouette":
            self.clear_window([pid], 0, self.n_frames - 1)
            meta.spec = ""
        meta.source = source
        self._touch()

    def set_position(self, frame: int, pid: int, x: float, y: float) -> None:
        """Manual placement/correction of one point at one frame."""
        self.tracks[frame, pid] = (x, y)
        self.visibility[frame, pid] = True
        self.manual[frame, pid] = True
        self.tracked[frame, pid] = True
        self.confidence[frame, pid] = 1.0  # user input is ground truth
        q = self.points[pid]
        if q.is_ball:
            # the hand-placed centre is where SAM is prompted on the next run
            if q.ball_prompts is None:
                q.ball_prompts = {}
            q.ball_prompts[int(frame)] = [[float(x), float(y), 1]]
        self._touch()

    def manual_frames(self, pid: int) -> np.ndarray:
        """Frames where `pid` was placed by hand (ascending int64)."""
        if not (0 <= pid < self.n_points):
            return np.zeros(0, np.int64)
        return np.nonzero(self.manual[:, pid] & self.tracked[:, pid])[0].astype(np.int64)

    def data_frames(self, pid: int) -> np.ndarray:
        """Frames where `pid` has a position at all (ascending int64) — what
        "its first / last frame" means for navigation. Hand-marked hidden
        cells are included: you go there precisely to look at them."""
        if not (0 <= pid < self.n_points):
            return np.zeros(0, np.int64)
        return np.nonzero(self.tracked[:, pid])[0].astype(np.int64)

    INTERP_CONF = 0.6      # confidence written by interpolate_keyframes: shows on the timeline as "not tracked"

    def interpolate_keyframes(self, pid: int, replace: bool = False,
                              window: tuple[int, int] | None = None) -> tuple[int, tuple[int, int] | None]:
        """Keyframe digitizing: a smooth curve through `pid`'s hand-placed frames
        (a natural cubic spline with three or more, a straight line with two)
        fills the frames between the first and the last of them. `replace=False`
        writes only frames that have NO data (the gaps); `replace=True` writes
        every non-hand-placed frame in between (the interpolation becomes the
        track). `window` (f0, f1) limits both to that frame range. Filled cells
        get confidence INTERP_CONF and are not flagged hand-placed. Returns
        (frames written, (first, last) frame written) - (0, None) with fewer
        than two hand-placed frames."""
        keys = self.manual_frames(pid)
        if window is not None:
            a, b = min(window), max(window)
            keys = keys[(keys >= a) & (keys <= b)]
        if len(keys) < 2:
            return 0, None
        kx = self.tracks[keys, pid, 0].astype(np.float64)
        ky = self.tracks[keys, pid, 1].astype(np.float64)
        f = np.arange(int(keys[0]), int(keys[-1]) + 1)
        if len(keys) >= 3:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(keys.astype(np.float64), np.column_stack([kx, ky]), bc_type="natural")
            xy = cs(f.astype(np.float64))
        else:
            xy = np.column_stack([np.interp(f, keys, kx), np.interp(f, keys, ky)])
        is_key = np.isin(f, keys)
        if replace:
            write = ~is_key
        else:
            write = ~is_key & ~self.tracked[f, pid]
        if window is not None:
            write &= (f >= min(window)) & (f <= max(window))
        # never write outside the picture
        write &= (xy[:, 0] >= 0) & (xy[:, 0] < self.width) & (xy[:, 1] >= 0) & (xy[:, 1] < self.height)
        # nor into a frame the user marked hidden: that mark is a statement about
        # the footage, and filling it exported an invented position there (I17)
        self.last_interp_hidden_skipped = int((write & self.occluded[f, pid]).sum())
        write &= ~self.occluded[f, pid]
        rows = f[write]
        if len(rows) == 0:
            return 0, None
        self.tracks[rows, pid] = xy[write].astype(np.float32)
        self.visibility[rows, pid] = True
        self.tracked[rows, pid] = True
        self.manual[rows, pid] = False
        self.confidence[rows, pid] = self.INTERP_CONF
        self._touch()
        return int(len(rows)), (int(rows[0]), int(rows[-1]))

    def next_manual_frame(self, pid: int, frame: int, forward: bool = True) -> int | None:
        """The nearest hand-placed frame of `pid` strictly after (or before) `frame`."""
        fr = self.manual_frames(pid)
        if len(fr) == 0:
            return None
        cand = fr[fr > frame] if forward else fr[fr < frame]
        if len(cand) == 0:
            return None
        return int(cand[0] if forward else cand[-1])

    def low_conf_runs(self, pids: list[int] | None = None, threshold: float = 0.5) -> list[tuple[int, int]]:
        """[(start, end)] inclusive frame runs where any of `pids` (default: all
        points) is tracked with confidence below `threshold` and not marked
        occluded by hand — the red stretches on the timeline."""
        cols = list(range(self.n_points)) if pids is None else [p for p in pids if 0 <= p < self.n_points]
        if not cols:
            return []
        low = (self.tracked[:, cols] & (self.confidence[:, cols] < threshold)
               & ~self.occluded[:, cols]).any(axis=1)
        if not low.any():
            return []
        d = np.diff(low.astype(np.int8), prepend=0, append=0)
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    def next_low_conf(self, frame: int, forward: bool = True, pids: list[int] | None = None,
                      threshold: float = 0.5) -> tuple[int, int] | None:
        """The next (previous) low-confidence run whose start is after (before)
        `frame`; a run containing `frame` is skipped so repeated presses walk on."""
        runs = self.low_conf_runs(pids, threshold)
        if forward:
            for a, b in runs:
                if a > frame:
                    return (a, b)
        else:
            for a, b in reversed(runs):
                if a < frame and not (a <= frame <= b):
                    return (a, b)
                if a <= frame <= b:
                    continue
        return None

    # ---- hand-marked occlusion --------------------------------------------

    def set_occluded(self, frame: int, pid: int, on: bool) -> None:
        """Mark a cell "hidden here": kept in the data (so the mark is reversible)
        but excluded from every export and from 3D. Marking never invents a
        position: an untracked cell only carries the flag."""
        if not (0 <= pid < self.n_points and 0 <= frame < self.n_frames):
            return
        self.occluded[frame, pid] = bool(on)
        self._touch()

    def set_occluded_window(self, pids: list[int], start: int, end: int, on: bool) -> int:
        if end < start:
            start, end = end, start
        start = max(0, min(self.n_frames - 1, int(start)))
        end = max(0, min(self.n_frames - 1, int(end)))
        cols = np.asarray([p for p in pids if 0 <= p < self.n_points], np.int64)
        if len(cols) == 0:
            return 0
        self.occluded[start:end + 1, cols] = bool(on)
        self._touch()
        return int((end - start + 1) * len(cols))

    @property
    def exportable(self) -> np.ndarray:
        """(T, N) bool: cells that leave the app — tracked AND not hand-marked hidden."""
        return self.tracked & ~self.occluded

    # ---- notes --------------------------------------------------------------

    def set_note(self, frame: int, text: str, author: str = "") -> None:
        """Attach (or clear, with empty text) a free-text note to a frame."""
        import time as _time
        frame = int(frame)
        text = (text or "").strip()
        if not text:
            self.notes.pop(frame, None)
        else:
            self.notes[frame] = {"text": text, "author": author or self.annotator or "",
                                 "time": _time.strftime("%Y-%m-%d %H:%M")}
        self._touch()

    def note_frames(self) -> list[int]:
        return sorted(int(f) for f in self.notes)

    def positions_at(self, frame: int) -> np.ndarray:
        """(N, 2) float32 with NaN rows where the point has no data at `frame`."""
        return self.tracks[frame].copy()

    def seedable_at(self, frame: int) -> list[int]:
        """Appearance-tracked point ids that have a defined position at `frame`
        (silhouette-derived points come from the mask, never from a seed)."""
        return [i for i in range(self.n_points)
                if self.tracked[frame, i] and not self.points[i].derived]

    def derived_pids(self) -> list[int]:
        return [i for i, p in enumerate(self.points) if p.derived]

    def last_tracked_frame(self) -> int | None:
        rows = np.nonzero(self.tracked.any(axis=1))[0]
        return int(rows[-1]) if len(rows) else None


    # ---------------------------------------------------------------- animal

    def ensure_animal(self) -> AnimalMeta:
        if self.animal is None:
            self.animal = AnimalMeta()
        if self.masks is None:
            self.masks = MaskTrack(self.n_frames)
            self.masks.native_w, self.masks.native_h = self.width, self.height
        self._touch()
        return self.animal

    # ------------------------------------------------------------------ body


    def clear_body(self) -> None:
        self.body = None
        self._touch()

    def body_frames(self) -> np.ndarray:
        return self.body.frames() if self.body is not None else np.zeros(0, np.int64)

    def clear_body_window(self, start: int, end: int) -> int:
        if self.body is None:
            return 0
        n = self.body.clear(start, end)
        if n:
            self._touch()
        return n

    def has_body(self) -> bool:
        return self.body is not None and self.body.n_posed() > 0

    def export_body_joints_csv(self, path: str | Path) -> None:
        if self.body is None:
            raise ValueError("this view has no body pose to export")
        export_joints_csv(self.body, path)

    def export_body_angles_csv(self, path: str | Path) -> None:
        if self.body is None:
            raise ValueError("this view has no body pose to export")
        export_angles_csv(self.body, path, float(self.fps or 0.0))

    def clear_animal(self) -> None:
        self.animal = None
        self.masks = None
        self._touch()

    def rename_animal(self, name: str) -> str:
        """Rename the segment (the panel row and the timeline lane follow).
        Returns the name actually applied; empty falls back to "segment"."""
        if self.animal is None:
            return ""
        self.animal.name = (name or "").strip() or "segment"
        self._touch()
        return self.animal.name

    def mask_frames(self) -> np.ndarray:
        """Frames that carry a silhouette (ascending int64)."""
        if self.masks is None:
            return np.zeros(0, np.int64)
        return np.asarray(self.masks.frames(), np.int64).reshape(-1)

    def animal_seedable_at(self, frame: int) -> bool:
        """The segment can be (re)started at `frame`: the user prompted there,
        or a mask from an earlier run exists there (it seeds the model)."""
        if self.animal is None:
            return False
        if self.animal.has_prompt(frame):
            return True
        return self.masks is not None and self.masks.has(frame)

    def write_mask(self, frame: int, mask_work: np.ndarray, scale: float, score: float,
                   midline: np.ndarray | None = None) -> None:
        self.ensure_animal()
        self.masks.set_from_work(frame, mask_work, scale, score, midline)
        self._touch()

    def write_mask_summaries(self, summaries: list[dict]) -> None:
        """Store per-frame mask summaries emitted by the tracking worker."""
        if not summaries:
            return
        self.ensure_animal()
        for d in summaries:
            f = int(d["frame"])
            if 0 <= f < self.n_frames:
                self.masks.set_summary(f, d)
        self._touch()

    def clear_masks(self, start: int, end: int) -> int:
        if self.masks is None:
            return 0
        n = int((self.masks.area[max(0, start):end + 1] > 0).sum())
        self.masks.clear(start, end)
        self._touch()
        return n

    # -------------------------------------------------------------- skeleton

    def apply_skeleton(self, template: dict) -> list[int]:
        """Adopt a template: create the landmarks that do not exist yet (as
        points without data; derived ones carry their spec). Existing points
        with the same name are kept — their data is never touched. Returns the
        new point ids."""
        self.skeleton = {k: json.loads(json.dumps(template[k]))
                         for k in ("name", "head", "landmarks", "bones", "derived", "note")
                         if k in template}
        existing = {p.name: i for i, p in enumerate(self.points)}
        derived = template.get("derived", {})
        new: list[int] = []
        for name in template["landmarks"]:
            spec = derived.get(name, "")
            if name in existing:
                pid = existing[name]
                if spec and not self.tracked[:, pid].any() and not self.points[pid].derived:
                    self.points[pid].source, self.points[pid].spec = "silhouette", spec
                continue
            new.append(self.add_landmark(name, "silhouette" if spec else "track", spec))
        self._touch()
        return new

    def clear_skeleton(self) -> None:
        self.skeleton = None
        self._touch()

    def pid_by_name(self, name: str) -> int | None:
        return next((i for i, p in enumerate(self.points) if p.name == name), None)

    def head_pid(self) -> int | None:
        """The appearance-tracked point that anchors the silhouette midline."""
        if self.skeleton and self.skeleton.get("head"):
            pid = self.pid_by_name(self.skeleton["head"])
            if pid is not None and not self.points[pid].derived:
                return pid
        for i, p in enumerate(self.points):
            if not p.derived and any(h in p.name.lower() for h in HEAD_HINTS):
                return i
        return None

    def bones(self) -> list[tuple[int, int]]:
        if not self.skeleton:
            return []
        out = []
        for a, b in self.skeleton.get("bones", []):
            pa, pb = self.pid_by_name(a), self.pid_by_name(b)
            if pa is not None and pb is not None:
                out.append((pa, pb))
        return out

    # ----------------------------------------------------------------- events

    def add_event(self, name: str, start: int, end: int, note: str = "",
                  author: str | None = None) -> int:
        """Add a named frame window; swaps/clamps a reversed or out-of-range
        range instead of rejecting it. Returns the event index.

        Events are TYPES with occurrences: the same name marked again is a
        repeat occurrence and keeps the type's color — colors are assigned
        per distinct name, not per window."""
        if end < start:
            start, end = end, start
        start = max(0, min(self.n_frames - 1, int(start)))
        end = max(0, min(self.n_frames - 1, int(end)))
        self._event_counter += 1
        name = name.strip() or f"event {self._event_counter}"
        color = next((e.color for e in self.events if e.name == name),
                     PALETTE[len({e.name for e in self.events}) % len(PALETTE)])
        self.events.append(Event(name, start, end, color, (note or "").strip(),
                                 self.annotator if author is None else author))
        self._touch()
        return len(self.events) - 1

    def remove_event(self, index: int) -> None:
        del self.events[index]
        self._touch()

    def update_event(self, index: int, name: str | None = None,
                     start: int | None = None, end: int | None = None,
                     note: str | None = None) -> None:
        ev = self.events[index]
        if name is not None and name.strip():
            ev.name = name.strip()
        if note is not None:
            ev.note = note.strip()
            if ev.note and not ev.author:
                ev.author = self.annotator
        s = ev.start if start is None else int(start)
        e = ev.end if end is None else int(end)
        if e < s:
            s, e = e, s
        ev.start = max(0, min(self.n_frames - 1, s))
        ev.end = max(0, min(self.n_frames - 1, e))
        self._touch()

    # --------------------------------------------------------------- tracking

    def write_segment(self, t0: int, tracks_win: np.ndarray, vis_win: np.ndarray,
                      point_ids: list[int], conf_win: np.ndarray | None = None) -> None:
        """Scatter a window of tracker output into the session.

        tracks_win: (L, K, 2) float32 in native pixels; vis_win: (L, K) bool;
        conf_win: (L, K) float32 in [0, 1] (None -> 1.0 where tracked); rows
        map to frames t0..t0+L (clipped to the video length), columns map to
        session point ids via point_ids.
        """
        L = tracks_win.shape[0]
        end = min(t0 + L, self.n_frames)
        # an animal-only run with no landmark points emits chunks with zero
        # columns (masks travel separately): nothing to scatter, and an empty
        # index array would be float-typed and blow up the fancy indexing
        if end <= t0 or len(point_ids) == 0 or tracks_win.shape[1] == 0:
            return
        rows = slice(t0, end)
        cols = np.asarray(point_ids, np.int64)
        tw = tracks_win[: end - t0].copy()
        vw = vis_win[: end - t0].copy()
        cw = (np.ones(vw.shape, np.float32) if conf_win is None
              else conf_win[: end - t0].astype(np.float32).copy())
        # A point that left the frame has no real coordinates — the model keeps
        # predicting (it must, for joint tracking), but we store blanks so the
        # marker disappears and the export has empty cells for those frames.
        oob = ~((tw[..., 0] >= 0) & (tw[..., 0] < self.width)
                & (tw[..., 1] >= 0) & (tw[..., 1] < self.height))
        tw[oob] = np.nan
        vw[oob] = False
        cw[oob] = 0.0
        self.tracks[rows, cols, :] = tw
        self.visibility[rows, cols] = vw
        self.tracked[rows, cols] = ~oob
        self.confidence[rows, cols] = np.clip(np.nan_to_num(cw), 0.0, 1.0)
        self.manual[rows, cols] = False  # model output supersedes stale manual flags
        rad = self.radius[rows, cols]
        rad[oob] = np.nan
        self.radius[rows, cols] = rad
        self._touch()

    def clear_window(self, pids: list[int], start: int, end: int) -> int:
        """Blank the tracked data of `pids` in frames [start, end] inclusive —
        the points themselves survive (bulk cleanup of a bad stretch).
        Returns how many tracked cells were cleared."""
        if end < start:
            start, end = end, start
        start = max(0, min(self.n_frames - 1, int(start)))
        end = max(0, min(self.n_frames - 1, int(end)))
        rows = slice(start, end + 1)
        cols = np.asarray([p for p in pids if 0 <= p < self.n_points], np.int64)
        if len(cols) == 0:
            return 0
        n = int(self.tracked[rows, cols].sum())
        self.tracks[rows, cols] = np.nan
        self.visibility[rows, cols] = False
        self.manual[rows, cols] = False
        self.tracked[rows, cols] = False
        self.confidence[rows, cols] = 0.0
        self.occluded[rows, cols] = False
        self.radius[rows, cols] = np.nan
        self._touch()
        return n

    def snapshot(self) -> Snapshot:
        return Snapshot(self.tracks.copy(), self.visibility.copy(), self.manual.copy(),
                        self.tracked.copy(), self.confidence.copy(),
                        [p.copy() for p in self.points],
                        self.masks.copy() if self.masks is not None else None,
                        self.animal is not None, self.occluded.copy(),
                        self.body.copy() if self.body is not None else None,
                        self.radius.copy(),
                        json.loads(json.dumps(self.skeleton)) if self.skeleton else None)

    def restore(self, snap: Snapshot) -> None:
        self.tracks = snap.tracks.copy()
        self.visibility = snap.visibility.copy()
        self.manual = snap.manual.copy()
        self.tracked = snap.tracked.copy()
        self.confidence = snap.confidence.copy()
        self.occluded = (snap.occluded.copy() if snap.occluded is not None
                         else np.zeros(self.tracked.shape, bool))
        self.points = [p.copy() for p in snap.points]
        # the skeleton names points: restored with them, or a rename / template
        # applied after the snapshot left bones and the head on names that no
        # longer exist (I22)
        self.skeleton = json.loads(json.dumps(snap.skeleton)) if snap.skeleton else None
        self.radius = (snap.radius.copy() if snap.radius is not None
                       else np.full(self.tracked.shape, np.nan, np.float32))
        if snap.has_animal and self.animal is not None:
            self.masks = snap.masks.copy() if snap.masks is not None else None
            if self.masks is None:
                self.ensure_animal()
        # A body track is restored whether or not there was one: undoing the
        # very first pose run has to be able to take it back to nothing.
        self.body = snap.body.copy() if snap.body is not None else None
        self._touch()


    # ---------------------------------------------------------------- exports

    def export_csv(self, path: str | Path) -> None:
        """Wide format: frame, {name}_x, {name}_y, {name}_visible, ... Blank = untracked."""
        names = [_sanitize(p.name) for p in self.points]
        header = "frame," + ",".join(f"{n}_x,{n}_y,{n}_visible" for n in names)
        lines = [header]
        tracks, vis, tracked = self.tracks, self.visibility, self.exportable
        for t in range(self.n_frames):
            cells = [str(t)]
            for j in range(self.n_points):
                if tracked[t, j]:
                    x, y = tracks[t, j]
                    cells.append(f"{x:.3f},{y:.3f},{int(vis[t, j])}")
                else:
                    cells.append(",,")
            lines.append(",".join(cells))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    def export_tsv_sparse(self, path: str | Path) -> None:
        """Long format, tracked cells only: frame  point  x  y  visible."""
        lines = ["frame\tpoint\tx\ty\tvisible"]
        names = [_sanitize(p.name) for p in self.points]
        ts, js = np.nonzero(self.exportable)
        for t, j in zip(ts.tolist(), js.tolist()):
            x, y = self.tracks[t, j]
            lines.append(f"{t}\t{names[j]}\t{x:.3f}\t{y:.3f}\t{int(self.visibility[t, j])}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    def export_dlc_csv(self, path: str | Path, scorer: str = "Kinetrace") -> None:
        """DeepLabCut-style CSV (3 header rows: scorer / bodyparts / coords),
        one row per frame, likelihood = tracking confidence. Readable by
        DeepLabCut, Anipose (as 2D pose) and most pose-analysis notebooks."""
        names = [_sanitize(p.name) for p in self.points]
        lines = ["scorer," + ",".join([scorer] * (3 * len(names))),
                 "bodyparts," + ",".join(f"{n},{n},{n}" for n in names),
                 "coords," + ",".join("x,y,likelihood" for _ in names)]
        tracks, conf, tracked = self.tracks, self.confidence, self.exportable
        for t in range(self.n_frames):
            cells = [str(t)]
            for j in range(self.n_points):
                if tracked[t, j]:
                    x, y = tracks[t, j]
                    cells.append(f"{x:.3f},{y:.3f},{conf[t, j]:.4f}")
                else:
                    cells.append(",,0")
            lines.append(",".join(cells))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    def export_dltdv_csv(self, path: str | Path, flip_y: bool = False,
                         pixel_origin: float = 1.0) -> None:
        """DLTdv-style xypts CSV: pt{k}_cam1_X, pt{k}_cam1_Y per point, one
        row per frame, NaN where untracked.

        Convention (this is where 3D pipelines silently break): **DLTdv8
        numbers pixels from 1 with the origin at the TOP-left** -- verified on
        a real 6-camera DLTdv8 project, whose clicks reproduce DLTdv's own 3D to
        0.07 mm only in that frame (`calib.CameraCalibration`). That is the
        default: x + 1, y + 1. `flip_y=True` writes the older bottom-left
        variant (y' = H - y, which is 1-based from the bottom edge) for
        DLTdv5-era files and Argus Clicker. A *_pointnames.csv sidecar maps
        pt indices to names and states the convention written."""
        header = ",".join(f"pt{k + 1}_cam1_X,pt{k + 1}_cam1_Y" for k in range(self.n_points))
        lines = [header]
        exp = self.exportable
        po = float(pixel_origin)
        for t in range(self.n_frames):
            cells = []
            for j in range(self.n_points):
                if exp[t, j]:
                    x, y = self.tracks[t, j]
                    yy = (self.height - float(y)) if flip_y else float(y) + po
                    cells.append(f"{float(x) + po:.3f},{yy:.3f}")
                else:
                    cells.append("NaN,NaN")
            lines.append(",".join(cells))
        p = Path(path)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        side = p.with_name(p.stem + "_pointnames.csv")
        side.write_text("index,name\n" + "\n".join(f"pt{k + 1},{_sanitize(pt.name)}"
                                                   for k, pt in enumerate(self.points))
                        + f"\nconvention,{dltdv_convention_text(flip_y, po)}\n",
                        encoding="utf-8", newline="")

    def export_animal_csv(self, path: str | Path) -> None:
        """Segment silhouette per frame: presence, score, bbox, centroid, area,
        and the 32-point midline (x/y columns; blank where absent)."""
        m = self.masks
        cols = ["frame", "present", "score", "x0", "y0", "x1", "y1", "cx", "cy", "area"]
        cols += [f"mid{k}_{ax}" for k in range(MIDLINE_SAMPLES) for ax in ("x", "y")]
        lines = [",".join(cols)]
        for t in range(self.n_frames):
            if m is not None and m.has(t):
                b = m.bbox[t]
                c = m.centroid[t]
                sc = m.score[t]
                cells = [str(t), "1", f"{sc:.3f}" if np.isfinite(sc) else "",
                         str(b[0]), str(b[1]), str(b[2]), str(b[3]),
                         f"{c[0]:.2f}", f"{c[1]:.2f}", str(int(m.area[t]))]
                mid = m.midline.get(t)
                if mid is not None and len(mid) == MIDLINE_SAMPLES:
                    cells += [f"{v:.2f}" for xy in mid for v in xy]
                else:
                    cells += [""] * (2 * MIDLINE_SAMPLES)
            else:
                cells = [str(t), "0"] + [""] * (len(cols) - 2)
            lines.append(",".join(cells))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    def export_mat(self, path: str | Path) -> None:
        """MATLAB .mat: tracks (T,N,2) with NaN where untracked, plus metadata,
        the segment silhouette summary and the skeleton."""
        from scipy.io import savemat
        T = self.n_frames
        tracks = self.tracks.astype(np.float64).copy()
        tracks[~self.exportable] = np.nan      # hand-marked hidden cells leave as NaN
        d = {
            "tracks": tracks,
            "visibility": self.visibility,
            "manual": self.manual,
            "tracked": self.exportable,
            "occluded": self.occluded,
            "confidence": self.confidence.astype(np.float64),
            "annotator": str(self.annotator),
            "note_frames": np.array(self.note_frames(), np.float64),
            "note_text": np.array([self.notes[f]["text"] for f in self.note_frames()], dtype=object),
            "point_names": np.array([_sanitize(p.name) for p in self.points], dtype=object),
            "point_kind": np.array([p.kind for p in self.points], dtype=object),
            "point_radius": np.array([p.radius for p in self.points], np.float64),
            "point_source": np.array([p.source for p in self.points], dtype=object),
            "point_spec": np.array([p.spec for p in self.points], dtype=object),
            "event_names": np.array([_sanitize(e.name) for e in self.events], dtype=object),
            "event_start": np.array([e.start for e in self.events], np.float64),
            "event_end": np.array([e.end for e in self.events], np.float64),
            "event_notes": np.array([e.note for e in self.events], dtype=object),
            "fps": self.fps,
            "video_path": str(self.video_path),
            "video_size": np.array([self.width, self.height]),
            # MATLAB users expect 1-based pixels; say what these are rather
            # than leave them to find a one-pixel shift in their 3D
            "pixel_convention": "top-left origin, first pixel = 0 (OpenCV); add 1 for MATLAB / DLTdv8 coordinates",
            "frame_convention": "frame index 0 = first frame of the video",
        }
        if self.skeleton:
            d["skeleton_name"] = str(self.skeleton.get("name", ""))
            d["skeleton_head"] = str(self.skeleton.get("head", ""))
            d["skeleton_bones"] = np.array([[_sanitize(a), _sanitize(b)]
                                            for a, b in self.skeleton.get("bones", [])],
                                           dtype=object).reshape(-1, 2)
        if self.masks is not None:
            m = self.masks
            mid = np.full((T, MIDLINE_SAMPLES, 2), np.nan, np.float64)
            for f, pts in m.midline.items():
                if len(pts) == MIDLINE_SAMPLES:
                    mid[f] = pts
            d.update({
                "segment_name": self.animal.name if self.animal else "segment",
                "segment_present": (m.area > 0),
                "segment_score": m.score.astype(np.float64),
                "segment_bbox": m.bbox.astype(np.float64),
                "segment_centroid": m.centroid.astype(np.float64),
                "segment_area": m.area.astype(np.float64),
                "segment_midline": mid,
                "segment_prompt_frames": np.array(self.animal.prompt_frames() if self.animal else [],
                                                 np.float64),
            })
        savemat(str(path), d, do_compression=True)

    def export_events_csv(self, path: str | Path) -> None:
        """Events sidecar (written next to CSV/TSV exports): name, start, end,
        note, author — then one row per frame note (name = "note"). Standard CSV
        quoting: a comma, quote or line break in a name or note survives (they
        used to be replaced by '_')."""
        import csv
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["name", "start_frame", "end_frame", "note", "author"])
            w.writerows([e.name, e.start, e.end, e.note, e.author] for e in self.events)
            w.writerows(["note", f, f, self.notes[f]["text"], self.notes[f].get("author", "")]
                        for f in self.note_frames())
