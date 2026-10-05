"""Tracking session data model: point tracks, the segment silhouette, the
skeleton, persistence, exports.

Arrays are (T, N, ...) where T = video frame count and N = number of points
(the seven of `POINT_ARRAYS`). NaN in `tracks` (mirrored by `tracked ==
False`) means "no data for this point at this frame". Even a 40k-frame,
50-point session is ~20 MB, so the whole model lives in RAM; it is saved as
part of a project folder (projectfile.py).

What a session holds, beyond the tracks:
- `confidence` (T, N): the point model's per-frame track-correctness score in
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
- Points carry a `source`: "track" (tracked by the point model chosen in
  Track ▾ -- AllTracker, CoTracker3 or Moving spot; the user seeds it),
  "silhouette" (derived from the segment's mask by `spec`, e.g. "tip",
  "midline:0.5", "ext:FL") or "ball" (a ball marker: SAM + a fitted circle).
  Derived points are written into the same (T, N) arrays by the tracking run,
  so exports, timeline, undo and corrections work unchanged; they simply
  cannot be seeded by hand. `spot` = the point's Moving spot settings found by
  the test on the user's clicks (None = automatic; spots.py).
- A named `skeleton` (landmark names, bones, head landmark, derived defaults)
  from a template; landmarks are ordinary points matched by name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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
# (G149) the segments after the first: soft colours, none of them the point palette's
SEGMENT_COLORS: tuple[tuple[int, int, int], ...] = (ANIMAL_COLOR, (255, 159, 207), (130, 180, 255), (255, 210, 110),
                                                    (190, 150, 255), (120, 230, 240), (255, 140, 110), (200, 230, 120))


# Reopening a project must restore the exact working state, not just tracks.
DEFAULT_UI_STATE: dict = {
    "selected": -1,          # selected point id, -1 = none
    "follow": False,         # ⌖ Follow toggle — OFF by default:
                             # an automatic re-frame surprises more than it helps
    "autopause": True,       # pause tracking on confidence collapse
    "roi": True,             # ROI-zoom tracking
    "track_mode": "auto",    # "auto" = run to end | "semi" = F steps one frame
    "track_all": False,      # Track ▾ → Every camera: track the points in each camera that has them (G29)
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
    "point_backend": "",     # "" = preferred (AllTracker when vendored, else CoTracker3)
    "trail_len": 10,         # trajectory trail length in frames (0 = off; View -> Trails, G33)
    "trail_future": False,   # also draw the upcoming path (dashed)
    "onion": False,          # onion skin: ghosts of the previous / next frame
    "loupe": False,          # magnifier under the cursor
    "display_filter": "none",  # none | contrast | bright | diff (display only)
    "region_shape": "circle",  # what an armed drag draws: circle | rect | polygon
}

SOURCES = ("track", "silhouette", "ball")
TRACKERS = ("alltracker", "cotracker3", "spot")   # a point's own tracker (G62); "" = the project default


class PointArray(NamedTuple):
    """One of the per-point (T, N, ...) arrays of a session (R15)."""
    name: str          # the TrackingSession attribute, and the Snapshot field
    dtype: type
    fill: object       # an empty / cleared cell
    tail: tuple = ()   # trailing shape of one (frame, point) cell

    def empty(self, n_frames: int, n_points: int) -> np.ndarray:
        return np.full((n_frames, n_points) + self.tail, self.fill, self.dtype)


# THE list of the per-point arrays: allocation, a new / removed / reordered
# point, clear_window, snapshot / restore and `restore_cells` all walk it, so a
# new array is added here once (projectfile and the app still name them
# themselves; verify_core's field-coverage check pins this table to Snapshot)
POINT_ARRAYS: tuple[PointArray, ...] = (
    PointArray("tracks", np.float32, np.nan, (2,)),
    PointArray("visibility", bool, False),
    PointArray("manual", bool, False),
    PointArray("tracked", bool, False),
    PointArray("confidence", np.float32, 0.0),
    # hand-marked "hidden here": the data stays (so the mark can be undone) but
    # the cell is NOT exported and NOT used for 3D
    PointArray("occluded", bool, False),
    # ball markers: the fitted circle's radius per frame (NaN elsewhere) so the
    # canvas can draw the circle SAM found, not just its centre
    PointArray("radius", np.float32, np.nan),
)


def in_frame(pts, w: float, h: float):
    """Which positions lie inside a w x h picture: finite, 0 <= x < w and
    0 <= y < h, over the last axis (x, y) -- a numpy bool for one (2,) point, a
    bool array for (..., 2). THE picture-bounds rule of the tracker and the
    session, which lived in six copies (I138). A NaN or infinite coordinate
    fails the bounds anyway; the finite test only says so outright."""
    pts = np.asarray(pts)
    x, y = pts[..., 0], pts[..., 1]
    return np.isfinite(pts).all(axis=-1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)


# a spreadsheet runs a cell that starts with one of these as a FORMULA (CSV
# injection): no landmark, event or segment name may start with one (M7, owner
# 2026-09-30: typed names are refused, names from files lose them)
FORMULA_START = ("=", "+", "-", "@", chr(9), chr(13))       # tab and carriage return too


def starts_formula(name: str) -> bool:
    return (name or "").strip().startswith(FORMULA_START)


def formula_safe(name: str, fallback: str = "point") -> str:
    """`name` without the leading characters a spreadsheet takes for a formula."""
    s = (name or "").strip()
    while s.startswith(FORMULA_START):
        s = s[1:].strip()
    return s or fallback


def _formula_safe_template(t: dict) -> dict:
    """A skeleton template with every landmark name made formula-safe."""
    f = formula_safe
    out = dict(t)
    out["landmarks"] = [f(n) for n in t.get("landmarks", [])]
    if "bones" in t:
        out["bones"] = [[f(n) for n in b] for b in t["bones"]]
    if t.get("head"):
        out["head"] = f(t["head"])
    if "derived" in t:
        out["derived"] = {f(k): v for k, v in t["derived"].items()}
    return out


def _copy_json(d):
    return json.loads(json.dumps(d)) if d else None


def _clean_skeleton(sk: dict | None) -> dict | None:
    """An animal's skeleton (G153: part names) with every field the right type; None when empty.
    Reads what a hand-edited project file holds without trusting it."""
    if not isinstance(sk, dict):
        return None
    f = formula_safe
    lm = [f(str(n)) for n in sk.get("landmarks", []) if str(n).strip()]
    bones = [[f(str(a)), f(str(b))] for a, b in (x for x in sk.get("bones", []) if isinstance(x, (list, tuple))
                                                  and len(x) == 2) if str(a).strip() and str(b).strip()]
    derived = {f(str(k)): str(v) for k, v in (sk.get("derived") or {}).items()} \
        if isinstance(sk.get("derived"), dict) else {}
    head = f(str(sk["head"])) if sk.get("head") else None
    out = {"name": str(sk.get("name", "") or ""), "landmarks": lm, "bones": bones, "derived": derived}
    if head:
        out["head"] = head
    if sk.get("note"):
        out["note"] = str(sk["note"])
    return out if (lm or bones or head or out["name"]) else None


def qualified(animal: str, part: str) -> str:
    """(G153) A landmark's name: "<animal> <part>" for a point of an animal, the part alone for a
    Scene point. It is the key every camera, the 3D layer and the exports join on."""
    return f"{animal} {part}" if animal else part


def _sanitize(name: str) -> str:
    """Make a point/event name safe for CSV/TSV headers and cells."""
    return name.replace(",", "_").replace("\t", "_").replace("\n", " ").strip() or "point"


def dltdv_convention_text(flip_y: bool, pixel_origin: float = 1.0) -> str:
    """One plain sentence naming the pixel convention an xypts file was
    written in, for its sidecar and the export dialog. Both writers write
    DLTdv8's first-pixel-1 numbering, so `pixel_origin` is always 1."""
    if flip_y:
        return "bottom-left origin; y counted from the bottom edge; first pixel = 1 (older DLTdv, Argus Clicker)"
    return "top-left origin; first pixel = 1 (DLTdv8 / MATLAB)"


@dataclass
class PointMeta:
    name: str
    color: tuple[int, int, int]
    display: bool = True
    kind: str = "point"     # "point" | "group" (a region; its outline is `shape`)
    radius: float = 0.0     # group region radius in native px (0 for points)
    anchor: bool = False    # opt-in appearance re-anchor (NCC snap-back)
    source: str = "track"   # "track" (the point model) | "silhouette" (derived from the mask) | "ball"
    spec: str = ""          # derivation spec for silhouette points (see skeletons.py)
    free: bool = False      # may leave the animal (exempt from the on-body constraint)
    shape: str = "circle"   # region outline: "circle" | "rect" | "polygon" (groups only)
    outline: list | None = None   # rect/polygon vertices [[x, y], ...] native px at the seed
    # ball markers (source "ball"): the user's SAM clicks per frame,
    # {frame: [[x, y, label], ...]} native px, label 1 = the ball / 0 = not it
    ball_prompts: dict | None = None
    # the Moving spot point model's settings for this point (spots.SpotSettings
    # as a dict, from Track ▾ -> Test the point models on my clicks); None =
    # automatic. Per camera: each camera sees the spot differently (I161)
    spot: dict | None = None
    # the point's own tracker (G62): "alltracker" | "cotracker3" | "spot", or "" =
    # the project's default point model (Track ▾). Shared across cameras (by name)
    tracker: str = ""
    # (G153) the ANIMAL this landmark belongs to (its name; "" = Scene, no animal): its name is
    # "<animal> <part>", a derived landmark is computed from the animal's silhouette, and a point the
    # animal holds is kept on it (G156)
    segment: str = ""

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
                         else {int(f): [list(c) for c in cs] for f, cs in self.ball_prompts.items()},
                         None if self.spot is None else dict(self.spot), self.tracker, self.segment)

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
    """One ANIMAL layer (G153; a session has any number, G149): name, colour, the user's
    segmentation prompts per frame for its OPTIONAL silhouette (clicks with 1 = it / 0 = not it,
    and boxes), its skeleton and whether its points are held on its silhouette. Its points are
    the PointMeta whose `segment` is this name, named "<animal> <part>".

    `skeleton` is in PART names (the animal's prefix left off): {"name": template name,
    "landmarks": [part, ...], "bones": [[part, part], ...], "derived": {part: spec},
    "head": part} -- None until a template is applied or a bone drawn. `hold` (G156): its
    appearance-tracked points not marked 'may leave' are kept on its silhouette."""
    name: str = "animal"
    color: tuple[int, int, int] = ANIMAL_COLOR
    prompts: dict[int, list[tuple[float, float, int]]] = field(default_factory=dict)
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    skeleton: dict | None = None
    hold: bool = False        # (G160, owner) opt-in: the points of an animal track alone by default
    shown: bool = True        # its LAYERS checkbox: its silhouette drawn (its points have their own)

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
                          {f: list(v) for f, v in self.prompts.items()}, dict(self.boxes),
                          _copy_json(self.skeleton), self.hold, self.shown)

    def shared_copy(self) -> "AnimalMeta":
        """What another camera's copy of this animal shares (G153): name, colour, skeleton, hold --
        not the clicks, which belong to this camera's pictures."""
        return AnimalMeta(self.name, self.color, skeleton=_copy_json(self.skeleton), hold=self.hold,
                          shown=self.shown)

    def to_json(self) -> str:
        d = {"name": self.name, "color": list(self.color),
             "prompts": {str(f): v for f, v in self.prompts.items()},
             "boxes": {str(f): list(b) for f, b in self.boxes.items()},
             "hold": bool(self.hold)}           # `shown` is display state: view.json (G154)
        if self.skeleton:
            d["skeleton"] = self.skeleton
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "AnimalMeta":
        d = json.loads(s)
        a = AnimalMeta(str(d.get("name", "animal")), tuple(int(c) for c in d.get("color", ANIMAL_COLOR)))
        for f, v in d.get("prompts", {}).items():
            a.prompts[int(f)] = [(float(p[0]), float(p[1]), int(p[2])) for p in v]
        for f, b in d.get("boxes", {}).items():
            a.boxes[int(f)] = tuple(float(x) for x in b)
        sk = d.get("skeleton")
        a.skeleton = _clean_skeleton(sk) if isinstance(sk, dict) else None
        a.hold = bool(d.get("hold", False))
        return a


@dataclass
class Snapshot:
    tracks: np.ndarray
    visibility: np.ndarray
    manual: np.ndarray
    tracked: np.ndarray
    confidence: np.ndarray
    points: list[PointMeta] = field(default_factory=list)
    occluded: np.ndarray | None = None
    seg_masks: dict | None = None          # (G149) segment name -> its MaskTrack copy
    body: BodyTrack | None = None
    radius: np.ndarray | None = None       # ball markers: fitted radius per frame
    # (G153) each animal's name, skeleton and hold, in order: renamed / moved with its points (I22) --
    # its clicks are not undone (segment clicks cannot be, G70)
    animals: list | None = None


HEAD_HINTS = ("snout", "nose", "head", "beak", "rostrum", "mouth")


class TrackingSession:
    def __init__(self, video_path: str, n_frames: int, fps: float, width: int, height: int):
        self.video_path = video_path
        self.n_frames = n_frames
        self.fps = fps          # the rate the camera REALLY recorded at: times, velocities, camera rates (G38)
        self.file_fps = fps     # the rate the video file says (differs when a header lies, e.g. slow-motion files)
        self.width = width
        self.height = height
        for a in POINT_ARRAYS:      # tracks, visibility, manual, tracked, confidence, occluded, radius (R15)
            setattr(self, a.name, a.empty(n_frames, 0))
        self.points: list[PointMeta] = []
        self.events: list[Event] = []
        self.notes: dict[int, dict] = {}    # frame -> {"text", "author", "time"}
        self.annotator = ""                 # who is digitizing (Settings)
        # (G149) the segments: any number, each its own silhouette track; `animal` / `masks` are
        # the ACTIVE one (the S tool's clicks, the SEGMENT row it selects, its menus)
        self.segments: list[AnimalMeta] = []
        self.seg_masks: list[MaskTrack] = []
        self.active_seg = 0
        # Human body poses (joints + angles) from the body layer. None until a
        # pose run has produced something; it is a whole-video array, not a
        # per-point one, so it lives beside `masks` rather than in `points`.
        self.body: BodyTrack | None = None
        # (G153) there is no per-camera skeleton: each animal carries its own (AnimalMeta.skeleton)
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
        desired = formula_safe(desired)
        taken = {_sanitize(p.name) for i, p in enumerate(self.points) if i != exclude_pid}
        if _sanitize(desired) not in taken:
            return desired
        k = 2
        while _sanitize(f"{desired} ({k})") in taken:
            k += 1
        return f"{desired} ({k})"

    def _append_point(self, meta: PointMeta) -> int:
        self.points.append(meta)
        for a in POINT_ARRAYS:
            setattr(self, a.name, np.concatenate([getattr(self, a.name), a.empty(self.n_frames, 1)], axis=1))
        return self.n_points - 1

    def _new_point(self, name: str | None, default: str, color=None, counted: bool = True,
                   unique: bool = True, **fields) -> int:
        """The one way a point is born (R15): a column appended with `fields` as
        its PointMeta. `name`, or `default` with `{n}` = the point counter AFTER
        it is counted (P3, ball 3). `counted` = the point takes the next palette
        colour and number (every placement but a camera's placeholder, which keeps
        its origin's `color`); `unique` = the name goes through `unique_name`."""
        if counted:
            self._name_counter += 1
        name = name or default.format(n=self._name_counter)
        if unique:
            name = self.unique_name(name)
        if color is None:
            color = PALETTE[(self._name_counter - 1) % len(PALETTE)]
        return self._append_point(PointMeta(name, color, **fields))

    def add_point(self, frame: int, x: float, y: float,
                  kind: str = "point", radius: float = 0.0, name: str | None = None,
                  shape: str = "circle", outline=None) -> int:
        pid = self._new_point(name, "G{n}" if kind == "group" else "P{n}",
                              kind=kind, radius=float(radius),
                              shape=shape if shape in ("circle", "rect", "polygon") else "circle",
                              outline=None if outline is None else [[float(a), float(b)] for a, b in outline])
        self.set_position(frame, pid, x, y)
        return pid

    def add_ball(self, frame: int, x: float, y: float, name: str | None = None) -> int:
        """A ball marker: SAM segments the ball from this click, a circle is
        fitted per frame and its centre is the point (balls.py). The click is
        both the seed position and the first SAM prompt."""
        pid = self._new_point(name, "ball {n}", source="ball",
                              ball_prompts={int(frame): [[float(x), float(y), 1]]})
        self.set_position(frame, pid, x, y)
        return pid

    def add_ball_prompt(self, pid: int, frame: int, x: float, y: float, label: int = 1) -> None:
        """Another SAM click on a ball on `frame` (a correction, or where the
        ball reappeared). A positive click also becomes its hand-placed position.
        The frame's earlier clicks stay, negative ones too (I190): a hand
        placement (`set_position`) is what replaces them, a second click does not."""
        q = self.points[pid]
        if not q.is_ball:
            return
        if q.ball_prompts is None:
            q.ball_prompts = {}
        q.ball_prompts.setdefault(int(frame), []).append([float(x), float(y), int(label)])
        if label:
            self._place(frame, pid, x, y)
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
        pid = self._new_point(name, "point", source=source if source in SOURCES else "track", spec=spec)
        self._touch()
        return pid

    def add_placeholder(self, meta: PointMeta) -> int:
        """Another camera's landmark `meta`, with no data here: same name, colour,
        kind and data source, so selecting it and clicking the video places it in
        THIS camera (project.sync_landmarks). A rectangle / polygon outline is that
        camera's own geometry, so the region arrives here as a circle of the same
        radius; ball clicks and the appearance lock are per camera too."""
        pid = self._new_point(meta.name, "point", color=meta.color, counted=False, unique=False,
                              kind=meta.kind, radius=meta.radius, source=meta.source, spec=meta.spec,
                              free=meta.free, shape=meta.shape if meta.outline is None else "circle",
                              tracker=meta.tracker, segment=meta.segment)
        self._touch()
        return pid

    def add_empty_point(self) -> int:
        """POINTS → ＋ New point: a named point with no data yet (G26), numbered and
        coloured like one placed with N + click. With several cameras the app
        gives it to every camera, then a click in each camera places it."""
        pid = self._new_point(None, "P{n}")
        self._touch()
        return pid

    def _keep_columns(self, cols: list[int]) -> None:
        for a in POINT_ARRAYS:
            setattr(self, a.name, getattr(self, a.name)[:, cols])

    def remove_point(self, pid: int) -> None:
        """Delete point `pid` with its data. A point of an animal leaves its skeleton too (its part,
        its bones, the head when it was the head, G153): a camera added later or a re-applied
        template does not bring it back through the skeleton (I234)."""
        k = self.segment_of(pid)
        part = self.part_name(pid) if k is not None else None
        self._keep_columns([i for i in range(self.n_points) if i != pid])
        del self.points[pid]
        sk = self.segments[k].skeleton if k is not None else None
        if sk and not any(self.part_name(q) == part for q in self.points_of(k)):
            sk["landmarks"] = [n for n in sk.get("landmarks", []) if n != part]
            sk["bones"] = [b for b in sk.get("bones", []) if part not in b]
            sk.get("derived", {}).pop(part, None)
            if sk.get("head") == part:
                sk.pop("head", None)
        self._touch()

    def reorder_points(self, order: list[int]) -> None:
        """Put the points in `order` (a permutation of the point ids): one order
        in every camera's POINTS list (G26). Data travels with its point; bones
        and the head anchor name points, so they need nothing."""
        order = [int(i) for i in order]
        if sorted(order) != list(range(self.n_points)) or order == list(range(self.n_points)):
            return
        sel = int(self.ui_state.get("selected", -1))
        if 0 <= sel < self.n_points:
            self.ui_state["selected"] = order.index(sel)     # the saved selection stays on its point
        self._keep_columns(order)
        self.points = [self.points[i] for i in order]
        self._touch()

    def rename_point(self, pid: int, desired: str) -> str:
        """Rename with collision protection; returns the name actually applied. A point of an
        animal keeps the animal's prefix ("snout" typed for a squirrel point = "squirrel snout",
        G153), and the animal's skeleton (landmarks, bones, head) follows the new part."""
        k = self.segment_of(pid)
        old_part = self.part_name(pid)
        if k is not None:
            a = self.segments[k].name
            if not desired.startswith(a + " "):
                desired = qualified(a, formula_safe(desired))
        name = self.unique_name(desired, exclude_pid=pid)
        self.points[pid].name = name
        if k is not None:
            self._rename_part(k, old_part, self.part_name(pid))
        self._touch()
        return name

    def set_source(self, pid: int, source: str, spec: str = "") -> None:
        """Switch a point between appearance tracking and silhouette derivation.
        Data derived under the old rule is cleared (it would be stale)."""
        meta = self.points[pid]
        source = source if source in SOURCES else "track"
        if source == "silhouette":
            meta.spec = spec
            self.clear_window([pid], 0, self.n_frames - 1)
        elif meta.source == "silhouette":
            self.clear_window([pid], 0, self.n_frames - 1)
            meta.spec = ""
        meta.source = source
        self._touch()

    def _place(self, frame: int, pid: int, x: float, y: float) -> None:
        """The cells of a hand placement (no prompt bookkeeping, no touch)."""
        self.tracks[frame, pid] = (x, y)
        self.visibility[frame, pid] = True
        self.manual[frame, pid] = True
        self.tracked[frame, pid] = True
        self.confidence[frame, pid] = 1.0  # user input is ground truth

    def set_position(self, frame: int, pid: int, x: float, y: float) -> None:
        """Manual placement/correction of one point at one frame."""
        self._place(frame, pid, x, y)
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
        # never write outside the picture
        write &= in_frame(xy, self.width, self.height)
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

    def exportable_at(self, frames, pid: int):
        """`exportable[frames, pid]` -- one cell, or a column for an index array --
        without building the whole (T, N) array first: the epipolar guides ask
        for one cell per camera, several times per camera, on every playhead
        move (0.4 ms per array at 40k frames x 20 points; I137)."""
        return self.tracked[frames, pid] & ~self.occluded[frames, pid]

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


    # ---------------------------------------------------------------- animals
    # (G149, G153) any number of ANIMALS (layers): `segments[i]` with its OPTIONAL silhouette
    # `seg_masks[i]` and its points (PointMeta.segment = its name, named "<animal> <part>"); points of
    # no animal are in Scene. `animal` / `masks` = the ACTIVE one (`active_seg`): the S tool's target,
    # which the app sets from the LAYERS selection (G154). Anything that works on one particular
    # animal passes its index: a default that means "the active / first one" is how G150 / G151 hid.

    @property
    def animal(self) -> AnimalMeta | None:
        return self.segments[self.active_seg] if 0 <= self.active_seg < len(self.segments) else None

    @animal.setter
    def animal(self, meta: AnimalMeta | None) -> None:
        """(the single-animal form) None removes the active animal; a meta replaces the active
        one's, or makes the first animal."""
        if meta is None:
            if self.animal is not None:
                self.remove_segment(self.active_seg)
            return
        if self.animal is None:
            self.add_segment(meta.name, meta=meta)
        else:
            self.segments[self.active_seg] = meta

    @property
    def masks(self) -> MaskTrack | None:
        return self.seg_masks[self.active_seg] if 0 <= self.active_seg < len(self.seg_masks) else None

    @masks.setter
    def masks(self, track: MaskTrack | None) -> None:
        if self.animal is None:
            if track is None:
                return
            self.add_segment()
        self.seg_masks[self.active_seg] = track if track is not None else self._new_masks()

    def masks_of(self, i: int | None = None) -> MaskTrack | None:
        """Animal `i`'s silhouette track (None = the active animal's); None when there is no such
        animal (simplify 2026-10-04: the one form of that lookup)."""
        if i is None:
            return self.masks
        return self.seg_masks[i] if 0 <= i < len(self.seg_masks) else None

    def has_silhouette(self, k: int) -> bool:
        """Animal k has (or can make) a silhouette: clicks or stored silhouettes (G153: optional)."""
        return 0 <= k < self.n_segments and bool(self.segments[k].n_prompts() or self.seg_masks[k].n_masked())

    def _new_masks(self) -> MaskTrack:
        m = MaskTrack(self.n_frames)
        m.native_w, m.native_h = self.width, self.height
        return m

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def segment_names(self) -> list[str]:
        return [s.name for s in self.segments]

    def segment_index(self, name: str) -> int | None:
        return next((i for i, s in enumerate(self.segments) if s.name == name), None) if name else None

    def unique_segment_name(self, desired: str, exclude: int | None = None) -> str:
        desired = formula_safe(desired or "animal", "animal")
        taken = {_sanitize(s.name).lower() for i, s in enumerate(self.segments) if i != exclude}
        if _sanitize(desired).lower() not in taken:
            return desired
        k = 2
        while _sanitize(f"{desired} {k}").lower() in taken:
            k += 1
        return f"{desired} {k}"

    def add_segment(self, name: str | None = None, color=None, meta: AnimalMeta | None = None,
                    shared: bool = False) -> int:
        """A new animal (no clicks, no silhouette, no points yet), made the active one. Its name is
        unique among the animals ("animal", "animal 2", …), its colour the next animal colour.
        `meta`: a copy of it (`shared`: only what cameras share -- name, colour, skeleton, hold)."""
        k = len(self.segments)
        m = (meta.shared_copy() if shared else meta.copy()) if meta is not None else AnimalMeta()
        m.name = self.unique_segment_name(name or m.name)
        m.color = tuple(color) if color is not None else (m.color if meta is not None
                                                          else SEGMENT_COLORS[k % len(SEGMENT_COLORS)])
        self.segments.append(m)
        self.seg_masks.append(self._new_masks())
        self.active_seg = k
        self._touch()
        return k

    def remove_segment(self, i: int, keep_points: bool = True) -> list[tuple[str, str | None]]:
        """Remove animal i with its silhouette. Its points go to Scene (renamed to their part) or,
        `keep_points=False`, are deleted too. Returns [(old name, new name or None if deleted)]."""
        if not (0 <= i < len(self.segments)):
            return []
        mine = self.points_of(i)
        changes: list[tuple[str, str | None]] = []
        if keep_points:
            changes = self.move_points(mine, None)
        else:
            for pid in sorted(mine, reverse=True):
                changes.append((self.points[pid].name, None))
                self.remove_point(pid)
        del self.segments[i]
        del self.seg_masks[i]
        self.active_seg = min(self.active_seg if self.active_seg < i else max(0, self.active_seg - 1),
                              len(self.segments) - 1)
        self.active_seg = max(self.active_seg, 0)
        self._touch()
        return changes

    def segment_of(self, pid: int) -> int | None:
        """The animal landmark `pid` belongs to (its index); None = Scene (G153: "" is no animal, as
        is the name of an animal that is not there)."""
        if not (0 <= pid < self.n_points):
            return None
        return self.segment_index(self.points[pid].segment)

    def points_of(self, k: int | None) -> list[int]:
        """The points of animal k, in list order (None = the Scene points)."""
        if k is not None and not (0 <= k < len(self.segments)):
            return []
        of = self._segment_lookup()
        return [i for i, p in enumerate(self.points) if of(p) == k]

    def _segment_lookup(self):
        """p -> the index of point p's animal, None = Scene: `segment_of`'s rule (the FIRST animal of
        that name; "" is no animal) through one name -> index dict (simplify 2026-10-04)."""
        idx: dict[str, int] = {}
        for i, a in enumerate(self.segments):
            idx.setdefault(a.name, i)
        return lambda p: idx.get(p.segment) if p.segment else None

    def layer_groups(self) -> tuple[list[list[int]], list[int]]:
        """(per_animal, scene): per_animal[k] = animal k's point ids in list order, scene = the Scene
        points, in ONE pass over the points (= `points_of(k)` for every k, and `points_of(None)`)."""
        of = self._segment_lookup()
        per_animal: list[list[int]] = [[] for _ in self.segments]
        scene: list[int] = []
        for i, p in enumerate(self.points):
            k = of(p)
            (scene if k is None else per_animal[k]).append(i)
        return per_animal, scene

    def part_name(self, pid: int) -> str:
        """The point's name without its animal's prefix ("squirrel snout" -> "snout"): what the
        LAYERS tree shows under the animal and what its skeleton lists."""
        p = self.points[pid]
        a = p.segment if self.segment_of(pid) is not None else ""
        if a and p.name.startswith(a + " ") and len(p.name) > len(a) + 1:
            return p.name[len(a) + 1:]
        return p.name

    def move_points(self, pids, k: int | None) -> list[tuple[str, str]]:
        """Points `pids` into animal k (None = Scene), G153: given to it and renamed "<animal> <part>"
        (a free name); their data is untouched. The skeleton of the animal a point leaves lists its
        PART, so its bones simply stop resolving; the one it joins draws them if it lists that part.
        Returns [(old name, new name)] of the points that changed."""
        target = self.segments[k].name if (k is not None and 0 <= k < len(self.segments)) else ""
        if k is not None and not target:
            return []
        out = []
        for pid in pids:
            if not (0 <= pid < self.n_points) or self.segment_of(pid) == k:
                continue
            p = self.points[pid]
            part, old = self.part_name(pid), p.name
            p.segment = target
            p.name = self.unique_name(qualified(target, part), exclude_pid=pid)
            out.append((old, p.name))
        if out:
            self._touch()
        return out

    def ensure_animal(self) -> AnimalMeta:
        if self.animal is None:
            self.add_segment()          # with its (empty) silhouette track: segments and seg_masks go in pairs
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

    def rename_animal(self, name: str, i: int | None = None) -> str:
        """Rename animal `i` (the active one by default; the LAYERS row and the timeline lane
        follow), and its points with it: "squirrel snout" -> "<new> snout" (G153). Returns the name
        actually applied: empty falls back to "animal", a name another animal has gets a number."""
        i = self.active_seg if i is None else i
        if not (0 <= i < len(self.segments)):
            return ""
        old = self.segments[i].name
        new = self.unique_segment_name(formula_safe(name, "animal"), exclude=i)
        mine = [(pid, self.part_name(pid)) for pid in self.points_of(i)]
        self.segments[i].name = new
        for pid, part in mine:
            self.points[pid].segment = new
            self.points[pid].name = self.unique_name(qualified(new, part), exclude_pid=pid)
        if new != old:
            self._touch()
        return new

    def mask_frames(self, i: int | None = None) -> np.ndarray:
        """Frames that carry a silhouette of animal `i` (the active one; ascending int64)."""
        m = self.masks_of(i)
        if m is None:
            return np.zeros(0, np.int64)
        return np.asarray(m.frames(), np.int64).reshape(-1)

    def animal_seedable_at(self, frame: int, i: int | None = None) -> bool:
        """Animal `i`'s silhouette (the active one) can be (re)started at `frame`: the user prompted
        there, or a mask from an earlier run exists there (it seeds the model)."""
        i = self.active_seg if i is None else i
        if not (0 <= i < len(self.segments)):
            return False
        if self.segments[i].has_prompt(frame):
            return True
        return self.seg_masks[i].has(frame)

    def write_mask(self, frame: int, mask_work: np.ndarray, scale: float, score: float,
                   midline: np.ndarray | None = None, i: int | None = None) -> None:
        if i is None:
            self.ensure_animal()
            m = self.masks
        else:
            m = self.seg_masks[i]
        m.set_from_work(frame, mask_work, scale, score, midline)
        self._touch()

    def write_mask_summaries(self, summaries: list[dict]) -> None:
        """Store per-frame mask summaries emitted by the tracking worker: each carries the animal
        it belongs to ("seg", G149; none = the active one)."""
        if not summaries:
            return
        if not self.segments:
            self.ensure_animal()
        for d in summaries:
            f = int(d["frame"])
            i = int(d.get("seg", self.active_seg))
            if 0 <= f < self.n_frames and 0 <= i < len(self.seg_masks):
                self.seg_masks[i].set_summary(f, d)
        self._touch()

    def clear_masks(self, start: int, end: int, i: int | None = None) -> int:
        m = self.masks_of(i)
        if m is None:
            return 0
        n = int((m.area[max(0, start):end + 1] > 0).sum())
        m.clear(start, end)
        self._touch()
        return n

    # -------------------------------------------------------------- skeleton
    # (G153, G157) each animal has its own skeleton, in PART names; a point's full name is
    # "<animal> <part>". Bones and the head resolve through `qualified`, so a point that leaves the
    # animal leaves its bones and one that joins an animal listing its part takes them up.

    def _skeleton_of(self, k: int, create: bool = False) -> dict | None:
        a = self.segments[k]
        if a.skeleton is None and create:
            a.skeleton = {"name": "", "landmarks": [], "bones": [], "derived": {}}
        return a.skeleton

    def apply_skeleton(self, template: dict, segment: int | None = None) -> list[int]:
        """Put a template on animal `segment` (G153; None = the active animal, a new "animal" when
        there is none): its landmarks become the animal's points, "<animal> <part>", made where they
        do not exist yet (no data; derived ones carry their spec) -- an existing point keeps its
        data. The animal's skeleton takes the template's bones, head and name; one already there
        keeps its bones and landmarks beside the new ones. Returns the new point ids."""
        if not self.segments:
            self.add_segment()
        k = segment if (segment is not None and 0 <= segment < len(self.segments)) else self.active_seg
        a = self.segments[k]
        t = _formula_safe_template(template)
        part = _clean_skeleton({x: t[x] for x in ("name", "head", "landmarks", "bones", "derived", "note")
                                if x in t}) or {"name": "", "landmarks": [], "bones": [], "derived": {}}
        sk = a.skeleton
        if sk:
            sk["landmarks"] = list(sk.get("landmarks", [])) + [n for n in part["landmarks"]
                                                               if n not in sk.get("landmarks", [])]
            have = {tuple(b) for b in sk.get("bones", [])} | {tuple(b[::-1]) for b in sk.get("bones", [])}
            sk["bones"] = list(sk.get("bones", [])) + [b for b in part["bones"] if tuple(b) not in have]
            sk.setdefault("derived", {}).update(part.get("derived", {}))
            for x in ("head", "name", "note"):
                if part.get(x):
                    sk[x] = part[x]
        else:
            a.skeleton = part
        derived = part.get("derived", {})
        new: list[int] = []
        for p in part["landmarks"]:
            spec = derived.get(p, "")
            full = qualified(a.name, p)
            pid = self.pid_by_name(full)
            if pid is not None:
                if self.points[pid].segment != a.name:
                    self.points[pid].segment = a.name
                if spec and not self.tracked[:, pid].any() and not self.points[pid].derived:
                    self.points[pid].source, self.points[pid].spec = "silhouette", spec
                continue
            pid = self.add_landmark(full, "silhouette" if spec else "track", spec)
            self.points[pid].segment = a.name
            # (G122) the name the point really got ("a,b" beside "a_b" becomes "a,b (2)"):
            # the skeleton must use it
            self._rename_part(k, p, self.part_name(pid))
            new.append(pid)
        self._touch()
        return new

    def _rename_part(self, k: int, old: str, new: str) -> None:
        """Animal k's skeleton follows a renamed part (landmarks, bones, head, derived rules)."""
        sk = self.segments[k].skeleton if 0 <= k < len(self.segments) else None
        if not sk or old == new:
            return
        sk["landmarks"] = [new if n == old else n for n in sk.get("landmarks", [])]
        sk["bones"] = [[new if n == old else n for n in b] for b in sk.get("bones", [])]
        if sk.get("head") == old:
            sk["head"] = new
        if old in sk.get("derived", {}):
            sk["derived"][new] = sk["derived"].pop(old)

    def clear_skeleton(self, k: int | None = None) -> None:
        """Forget animal k's skeleton (None = every animal's); the points stay."""
        for i in (range(len(self.segments)) if k is None else [k]):
            if 0 <= i < len(self.segments):
                self.segments[i].skeleton = None
        self._touch()

    def pid_by_name(self, name: str) -> int | None:
        return next((i for i, p in enumerate(self.points) if p.name == name), None)

    def head_pid(self, seg: int | None = None) -> int | None:
        """The appearance-tracked point that anchors animal `seg`'s midline (None = the first
        animal): its skeleton's head, else one of ITS points named like a head (snout, nose, ...).
        A Scene point is never a head."""
        if seg is None:
            seg = 0
        if not (0 <= seg < len(self.segments)):
            return None
        a = self.segments[seg]

        def ok(pid):
            return (pid is not None and not self.points[pid].derived and not self.points[pid].is_ball
                    and self.segment_of(pid) == seg)
        if a.skeleton and a.skeleton.get("head"):
            pid = self.pid_by_name(qualified(a.name, a.skeleton["head"]))
            if ok(pid):
                return pid
        for i in self.points_of(seg):
            if ok(i) and any(h in self.part_name(i).lower() for h in HEAD_HINTS):
                return i
        return None

    def bone_names(self) -> list[tuple[str, str]]:
        """Every animal's bones as point names ("squirrel snout", "squirrel neck")."""
        out = []
        for a in self.segments:
            for x, y in (a.skeleton or {}).get("bones", []):
                out.append((qualified(a.name, x), qualified(a.name, y)))
        return out

    def bones(self) -> list[tuple[int, int]]:
        """Every animal's bones between points that exist, as point ids."""
        at = {p.name: i for i, p in enumerate(self.points)}
        return [(at[x], at[y]) for x, y in self.bone_names() if x in at and y in at]

    def bone_problem(self, a: int, b: int) -> str | None:
        """Why points a and b cannot be joined by a bone (None = they can, G157)."""
        if a == b:
            return "Pick two different points."
        ka, kb = self.segment_of(a), self.segment_of(b)
        if ka is None or kb is None:
            return "A bone joins two points of one animal: put both points in an animal first."
        if ka != kb:
            return "A bone joins two points of ONE animal."
        return None

    def has_bone(self, a: int, b: int) -> bool:
        k = self.segment_of(a)
        if k is None or k != self.segment_of(b):
            return False
        pa, pb = self.part_name(a), self.part_name(b)
        return any(sorted(x) == sorted((pa, pb)) for x in (self.segments[k].skeleton or {}).get("bones", []))

    def connect_bone(self, a: int, b: int) -> bool:
        """Join points a and b of one animal with a bone (G157). True when a bone was added."""
        if self.bone_problem(a, b) or self.has_bone(a, b):
            return False
        k = self.segment_of(a)
        sk = self._skeleton_of(k, create=True)
        pa, pb = self.part_name(a), self.part_name(b)
        for p in (pa, pb):
            if p not in sk["landmarks"]:
                sk["landmarks"].append(p)
        sk["bones"].append([pa, pb])
        self._touch()
        return True

    def remove_bone(self, a: int, b: int) -> bool:
        if not self.has_bone(a, b):
            return False
        k = self.segment_of(a)
        pa, pb = self.part_name(a), self.part_name(b)
        sk = self.segments[k].skeleton
        sk["bones"] = [x for x in sk["bones"] if sorted(x) != sorted((pa, pb))]
        self._touch()
        return True

    def set_head(self, pid: int) -> bool:
        """Point `pid` becomes its animal's head (the midline's anchor, G157). False for a Scene
        point or one derived from the silhouette."""
        k = self.segment_of(pid)
        if k is None or self.points[pid].derived or self.points[pid].is_ball:
            return False
        sk = self._skeleton_of(k, create=True)
        p = self.part_name(pid)
        if p not in sk["landmarks"]:
            sk["landmarks"].append(p)
        sk["head"] = p
        self._touch()
        return True

    def skeleton_template(self, k: int, name: str) -> dict:
        """Animal k as a template (G157, "Save its skeleton as a template…"): every point of it by
        part, its bones, its head, the rules of its derived points."""
        a = self.segments[k]
        mine = self.points_of(k)
        sk = a.skeleton or {}
        derived = {self.part_name(q): self.points[q].spec for q in mine if self.points[q].derived}
        head = sk.get("head") or (self.part_name(self.head_pid(k)) if self.head_pid(k) is not None else None)
        t = {"name": name, "landmarks": [self.part_name(q) for q in mine],
             "bones": [list(b) for b in sk.get("bones", [])], "derived": derived}
        if head:
            t["head"] = head
        return t
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
        name = formula_safe(name, f"event {self._event_counter}")
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
            ev.name = formula_safe(name, ev.name)
            # (G121) one type, one colour: renamed to an existing type = that type's colour
            same = next((e.color for i, e in enumerate(self.events) if i != index and e.name == ev.name), None)
            if same is not None:
                ev.color = same
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
        oob = ~in_frame(tw, self.width, self.height)
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
        for a in POINT_ARRAYS:
            getattr(self, a.name)[rows, cols] = a.fill
        # (I190) a ball's SAM clicks in the window go with its data: the next run
        # would otherwise prompt SAM at a click the user removed
        for c in cols.tolist():
            bp = self.points[c].ball_prompts
            if bp:
                for f in [f for f in bp if start <= f <= end]:
                    del bp[f]
        self._touch()
        return n

    def snapshot(self) -> Snapshot:
        return Snapshot(**{a.name: getattr(self, a.name).copy() for a in POINT_ARRAYS},
                        points=[p.copy() for p in self.points],
                        seg_masks={s.name: m.copy() for s, m in zip(self.segments, self.seg_masks)},
                        body=self.body.copy() if self.body is not None else None,
                        animals=[(a.name, _copy_json(a.skeleton), a.hold) for a in self.segments])

    def restore(self, snap: Snapshot) -> None:
        shape = snap.tracks.shape[:2]
        for a in POINT_ARRAYS:
            src = getattr(snap, a.name, None)      # an older snapshot has no occluded / radius
            setattr(self, a.name, a.empty(*shape) if src is None else src.copy())
        self.points = [p.copy() for p in snap.points]
        # each animal's name, skeleton and hold come back with its points (G153): the points name
        # their animal and its skeleton names their parts -- a rename, a move or a template applied
        # after the snapshot would otherwise leave them pointing at nothing (I22). Animals made
        # since stay (a new animal's clicks cannot be undone, G70): restored by position when the
        # same animals are there, else by name
        if snap.animals is not None:
            if len(snap.animals) == len(self.segments):
                for a, (nm, sk, hold) in zip(self.segments, snap.animals):
                    a.name, a.skeleton, a.hold = nm, _copy_json(sk), hold
            else:
                by = {nm: (sk, hold) for nm, sk, hold in snap.animals}
                for a in self.segments:
                    if a.name in by:
                        a.skeleton, a.hold = _copy_json(by[a.name][0]), by[a.name][1]
        if snap.seg_masks is not None:
            # (G149) every segment that is still here gets its silhouettes back (segments and their
            # clicks are not undone: segment clicks cannot be, G70)
            for i, s in enumerate(self.segments):
                if s.name in snap.seg_masks:
                    self.seg_masks[i] = snap.seg_masks[s.name].copy()
        # A body track is restored whether or not there was one: undoing the
        # very first pose run has to be able to take it back to nothing.
        self.body = snap.body.copy() if snap.body is not None else None
        self._touch()

    def restore_cells(self, snap: Snapshot, pids, f0: int, f1: int) -> None:
        """Put `snap`'s data back for the points `pids` on frames f0..f1 (a later
        pass of a run ended earlier): every per-point array, nothing else. A point
        id past either side's columns is left alone (R15; was the app's
        `_restore_frames`)."""
        f1 = min(int(f1), self.n_frames - 1)
        f0 = max(int(f0), 0)
        if f0 > f1 or not len(pids):
            return
        sl = slice(f0, f1 + 1)
        for a in POINT_ARRAYS:
            src, dst = getattr(snap, a.name, None), getattr(self, a.name)
            if src is None:
                continue
            for q in pids:
                if 0 <= q < src.shape[1] and q < dst.shape[1]:
                    dst[sl, q] = src[sl, q]
        self._touch()

    # ------------------------------------------------- point tools (G147)
    # Fixing identities: a tracker that swapped two body parts, a stretch tracked under the
    # wrong name, one landmark tracked in two pieces, a point that should be two. Every
    # per-point array travels (POINT_ARRAYS: positions, flags, confidence, hidden, radius),
    # and a ball's SAM clicks on those frames go with its data. The app makes each one undo step.

    def point_tool_problem(self, a: int, b: int | None = None) -> str | None:
        """Why a point tool cannot work on `a` (and `b`), in words; None = it can."""
        n = self.n_points
        if not (0 <= a < n) or (b is not None and not (0 <= b < n)):
            return "choose the points first"
        if b is not None and a == b:
            return "choose two different points"
        for q in (a,) if b is None else (a, b):
            if self.points[q].derived:
                return (f"{self.points[q].name} is computed from the silhouette (each run recomputes it); "
                        "right-click it -> Data source -> Track by appearance to edit its data")
        if b is not None and self.points[a].is_ball != self.points[b].is_ball:
            return "a ball marker can only exchange data with another ball marker (its data is a fitted circle)"
        return None

    def _range(self, f0: int, f1: int) -> slice:
        lo, hi = sorted((int(f0), int(f1)))
        return slice(max(0, lo), min(self.n_frames - 1, hi) + 1)

    def _move_prompts(self, src: int, dst: int, frames, swap: bool = False) -> None:
        """A ball's SAM clicks on `frames` follow its data (I190's rule: clicks belong to the data)."""
        ps, pd = self.points[src], self.points[dst]
        if not (ps.is_ball and pd.is_ball):
            return
        a, b = dict(ps.ball_prompts or {}), dict(pd.ball_prompts or {})
        for f in [int(f) for f in frames]:
            ca, cb = a.pop(f, None), b.pop(f, None)
            if ca is not None:
                b[f] = ca
            if swap and cb is not None:
                a[f] = cb
        ps.ball_prompts, pd.ball_prompts = a, b

    def swap_points(self, a: int, b: int, f0: int, f1: int) -> int:
        """Exchange the data of points `a` and `b` on frames f0..f1 (a tracker swapped them, e.g. the
        left and right foot). Returns the frames where either had data."""
        if self.point_tool_problem(a, b):
            return 0
        sl = self._range(f0, f1)
        n = int((self.tracked[sl, a] | self.tracked[sl, b]).sum())
        if not n:
            return 0
        for arr in POINT_ARRAYS:
            x = getattr(self, arr.name)
            x[sl, a], x[sl, b] = x[sl, b].copy(), x[sl, a].copy()
        self._move_prompts(a, b, range(sl.start, sl.stop), swap=True)
        self._touch()
        return n

    def move_point_data(self, src: int, dst: int, f0: int, f1: int) -> int:
        """`dst` takes `src`'s data on every frame of f0..f1 where `src` has some (overwriting `dst`
        there), and `src` is cleared on those frames: a stretch tracked under the wrong name.
        Returns the frames moved."""
        if self.point_tool_problem(src, dst):
            return 0
        sl = self._range(f0, f1)
        rows = np.nonzero(self.tracked[sl, src])[0] + sl.start
        if not len(rows):
            return 0
        for arr in POINT_ARRAYS:
            x = getattr(self, arr.name)
            x[rows, dst] = x[rows, src]
            x[rows, src] = arr.fill
        self._move_prompts(src, dst, rows)
        self._touch()
        return int(len(rows))

    def fill_point_gaps(self, src: int, dst: int, f0: int, f1: int) -> int:
        """`dst` takes `src`'s data on the frames of f0..f1 where `dst` has none and `src` has some;
        `src` is left as it is (one landmark tracked in two pieces: fill, then delete the other).
        Returns the frames filled."""
        if self.point_tool_problem(src, dst):
            return 0
        sl = self._range(f0, f1)
        rows = np.nonzero(self.tracked[sl, src] & ~self.tracked[sl, dst])[0] + sl.start
        if not len(rows):
            return 0
        for arr in POINT_ARRAYS:
            x = getattr(self, arr.name)
            x[rows, dst] = x[rows, src]
        ps, pd = self.points[src], self.points[dst]
        if ps.is_ball and pd.is_ball and ps.ball_prompts:
            pd.ball_prompts = dict(pd.ball_prompts or {})
            for f in rows.tolist():
                if f in ps.ball_prompts and f not in pd.ball_prompts:
                    pd.ball_prompts[f] = [list(c) for c in ps.ball_prompts[f]]
        self._touch()
        return int(len(rows))

    def split_point(self, pid: int, frame: int, name: str | None = None) -> tuple[int, int]:
        """`pid`'s data from `frame` on becomes a NEW point (same kind, tracker and data source; the
        next colour), and `pid` ends on the frame before: a point that turned into another part.
        Returns (the new point's id, the frames it took); (-1, 0) when there is nothing to split."""
        if self.point_tool_problem(pid) or not (0 <= int(frame) < self.n_frames):
            return -1, 0
        if not self.tracked[int(frame):, pid].any():
            return -1, 0
        q = self.points[pid]
        new = self._new_point(name or f"{q.name} (2)", "P{n}", kind=q.kind, radius=q.radius, source=q.source,
                              free=q.free, shape=q.shape if q.outline is None else "circle",
                              tracker=q.tracker, ball_prompts={} if q.is_ball else None,
                              segment=q.segment)          # in the same animal (G153)
        n = self.move_point_data(pid, new, int(frame), self.n_frames - 1)
        return new, n


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
        cols = [(None, _sanitize(p.name), j) for j, p in enumerate(self.points)]
        self._write_dlc_csv(path, scorer, cols, individuals=False)

    def _write_dlc_csv(self, path: str | Path, scorer: str, cols, individuals: bool) -> None:
        """The one DeepLabCut CSV writer (simplify 2026-10-04): `cols` = [(individual or None,
        body part, pid or None)], three cells x, y, likelihood each (blank / 0 without a position);
        `individuals` adds DeepLabCut's multi-animal 'individuals' header row."""
        lines = ["scorer," + ",".join([scorer] * (3 * len(cols)))]
        if individuals:
            lines.append("individuals," + ",".join(f"{c[0]},{c[0]},{c[0]}" for c in cols))
        lines += ["bodyparts," + ",".join(f"{c[1]},{c[1]},{c[1]}" for c in cols),
                  "coords," + ",".join("x,y,likelihood" for _ in cols)]
        tracks, conf, ok = self.tracks, self.confidence, self.exportable
        for t in range(self.n_frames):
            cells = [str(t)]
            for _i, _p, q in cols:
                if q is not None and ok[t, q]:
                    x, y = tracks[t, q]
                    cells.append(f"{x:.3f},{y:.3f},{conf[t, q]:.4f}")
                else:
                    cells.append(",,0")
            lines.append(",".join(cells))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    def _individual_columns(self) -> tuple[list[str], list[tuple[str, list[int | None]]]]:
        """(G159) The body parts every individual shares (the union of the animals' parts, in order)
        and [(individual, [pid or None per part])]: each animal, then 'single' for the Scene points
        (DeepLabCut's name for body parts of no individual)."""
        parts: list[str] = []
        for k in range(self.n_segments):
            for q in self.points_of(k):
                p = self.part_name(q)
                if p not in parts:
                    parts.append(p)
        inds = []
        for k, a in enumerate(self.segments):
            at = {self.part_name(q): q for q in self.points_of(k)}
            inds.append((a.name, [at.get(p) for p in parts]))
        return parts, inds

    def export_dlc_multi_csv(self, path: str | Path, scorer: str = "Kinetrace") -> None:
        """DeepLabCut MULTI-ANIMAL CSV (G159): header rows scorer / individuals / bodyparts / coords;
        each animal is an individual with the body parts all animals share (blank cells for a part it
        lacks), the Scene points are DeepLabCut's unique body parts under 'single'; x, y, likelihood
        (= tracking confidence), blank / 0 where there is no position."""
        parts, inds = self._individual_columns()
        cols = [(_sanitize(n), _sanitize(p), q) for n, qs in inds for p, q in zip(parts, qs)]
        cols += [("single", _sanitize(self.points[q].name), q) for q in self.points_of(None)]
        self._write_dlc_csv(path, scorer, cols, individuals=True)

    def export_sleap_csv(self, path: str | Path) -> None:
        """SLEAP analysis CSV (G159): track, frame_idx, instance.score, then <node>.x, <node>.y,
        <node>.score per body part -- one track per animal (the parts all animals share), and a track
        'scene' for the points of no animal when there are any. One row per track and frame that has
        a position; instance.score = the mean confidence of its placed nodes."""
        parts, inds = self._individual_columns()
        scene = self.points_of(None)
        nodes = parts + [self.points[q].name for q in scene if self.points[q].name not in parts]
        tracks_ = [(n, dict(zip(parts, qs))) for n, qs in inds]
        if scene:
            tracks_.append(("scene", {self.points[q].name: q for q in scene}))
        head = ["track", "frame_idx", "instance.score"]
        for n in nodes:
            n = _sanitize(n)
            head += [f"{n}.x", f"{n}.y", f"{n}.score"]
        lines = [",".join(head)]
        ok, tr, conf = self.exportable, self.tracks, self.confidence
        for name, at in tracks_:
            qs = [at.get(n) for n in nodes]
            cols = [q for q in qs if q is not None]
            if not cols:
                continue
            any_t = ok[:, cols].any(axis=1)
            for t in np.nonzero(any_t)[0].tolist():
                cells, scores = [], []
                for q in qs:
                    if q is not None and ok[t, q]:
                        x, y = tr[t, q]
                        cells += [f"{x:.3f}", f"{y:.3f}", f"{conf[t, q]:.4f}"]
                        scores.append(float(conf[t, q]))
                    else:
                        cells += ["", "", ""]
                lines.append(",".join([_sanitize(name), str(t), f"{np.mean(scores):.4f}"] + cells))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    def export_dltdv_csv(self, path: str | Path, flip_y: bool = False) -> None:
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
        po = 1.0                       # DLTdv8 numbers pixels from 1 (the only variant written)
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
        import csv                     # (I175) quoted: a name with a comma survives the round trip
        with open(side, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["index", "name"])
            w.writerows([f"pt{k + 1}", pt.name] for k, pt in enumerate(self.points))
            w.writerow(["convention", dltdv_convention_text(flip_y, po)])

    def export_animal_csv(self, path: str | Path, i: int | None = None) -> None:
        """Segment `i`'s (the active one's) silhouette per frame: presence, score, bbox, centroid,
        area, and the 32-point midline (x/y columns; blank where absent)."""
        m = self.masks_of(i)
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
        # (G153) each point's animal ('' = Scene), and every animal's bones by point name
        d["point_animal"] = np.array([p.segment if self.segment_of(i) is not None else ""
                                      for i, p in enumerate(self.points)], dtype=object)
        bones = self.bone_names()
        if bones:
            d["skeleton_bones"] = np.array([[_sanitize(a), _sanitize(b)] for a, b in bones],
                                           dtype=object).reshape(-1, 2)
            heads = [self.points[h].name for h in (self.head_pid(k) for k in range(self.n_segments))
                     if h is not None]
            d["skeleton_head"] = np.array([_sanitize(h) for h in heads], dtype=object)
            d["skeleton_name"] = "; ".join(sorted({str((a.skeleton or {}).get("name", "")) for a in self.segments
                                                   if (a.skeleton or {}).get("name")}))
        if self.seg_masks:
            # (G149) every segment in `segments` (a struct per segment); the first one's fields stay at
            # the top level as before
            segs = {}
            for k, (sm, sg) in enumerate(zip(self.seg_masks, self.segments)):
                mk = np.full((T, MIDLINE_SAMPLES, 2), np.nan, np.float64)
                for f, pts in sm.midline.items():
                    if len(pts) == MIDLINE_SAMPLES:
                        mk[f] = pts
                key = re.sub(r"[^A-Za-z0-9_]", "_", sg.name) or f"segment{k + 1}"
                if not key[0].isalpha():
                    key = "s_" + key
                # a MATLAB field name: at most 63 characters (60 here), unique -- two names alike after the
                # cleaning or in their first 60 characters get the animal's number
                key = key[:60]
                if key in segs:
                    key = f"{key[:55]}_{k + 1}"
                segs[key] = {
                    "name": sg.name, "present": (sm.area > 0), "score": sm.score.astype(np.float64),
                    "bbox": sm.bbox.astype(np.float64), "centroid": sm.centroid.astype(np.float64),
                    "area": sm.area.astype(np.float64), "midline": mk,
                    "prompt_frames": np.array(sg.prompt_frames(), np.float64)}
            if segs:
                d["segments"] = segs
            m = self.seg_masks[0]
            mid = np.full((T, MIDLINE_SAMPLES, 2), np.nan, np.float64)
            for f, pts in m.midline.items():
                if len(pts) == MIDLINE_SAMPLES:
                    mid[f] = pts
            first = self.segments[0] if self.segments else None
            d.update({
                "segment_name": first.name if first else "segment",
                "segment_present": (m.area > 0),
                "segment_score": m.score.astype(np.float64),
                "segment_bbox": m.bbox.astype(np.float64),
                "segment_centroid": m.centroid.astype(np.float64),
                "segment_area": m.area.astype(np.float64),
                "segment_midline": mid,
                "segment_prompt_frames": np.array(first.prompt_frames() if first else [], np.float64),
            })
        # (I263) long_field_names: MATLAB takes 63-character field names, scipy only 31 without it (an animal named
        # longer than that made the whole export fail)
        savemat(str(path), d, do_compression=True, long_field_names=True)

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
