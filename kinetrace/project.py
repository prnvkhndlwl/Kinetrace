"""Multi-camera projects: several videos of the same event, kept in sync.

A project is one or more **views**. A view is one video plus its own complete
`TrackingSession` — its own points, tracks, silhouette and events — because the
cameras see different things and each needs its own 2D digitizing. What ties
them together is a per-view **frame offset**.

Sync model
----------
**The first camera is the reference clock: its offset is always 0**, and every
other camera's offset says how many frames later that camera's recording is
against it. Only *differences* between offsets ever affect anything, so the
whole set could float freely — but then the numbers on screen would be
meaningless ("camera 2 is at -17" against what?). Pinning view 0 at 0 makes
every offset directly readable, and `_normalize` re-establishes it after any
change by shifting the whole set, which cannot alter a single mapping.

Formally, `offset` is *the frame number in this video at the shared instant
t = 0*, so::

    local_frame_of_view_i = t + offset_i

Given the frame you are looking at in one view, the matching frame in another
is therefore::

    local_j = local_i - offset_i + offset_j          (`map_frame`)

That is the ONLY thing offsets do. The playhead, the timeline, every array
index, every export and the undo stack all stay in the **active view's own**
frame numbering exactly as in a single-video project — so nothing about
tracking changes when a second camera is added. Switching the active view
remaps the playhead through `map_frame`, so the picture never jumps in time.

Frame rates and sub-frame sync (2026-09-13)
------------------------------------------
Cameras may run at different frame rates and resolutions. Each view carries a
**rate** = its fps divided by the reference camera's (2.0 for a 240 fps camera
in a 120 fps rig), and offsets are **fractional**::

    local_i = rate_i * t + offset_i          (t in reference frames)
    local_j = rate_j * (local_i - offset_i) / rate_i + offset_j

Display rounds to the nearest frame; the 3D layer (`calib.py`) interpolates
the tracks at the exact fractional frame. Whole-frame offsets are what you
set by eye for tracking; the fractional part comes from `calib.estimate_offsets`
afterwards, from the triangulation residual itself. The reference is still
pinned at 0 — `_normalize` shifts every other offset by `rate_i * offset_0`,
a pure re-timing that changes no mapping.

Persistence
-----------
One `.kinetrace` file (a zip of CSV / JSON / .npy, see projectfile.py and
docs/FORMAT.md): every camera with its tracks, the offsets and rates, which
camera was active, the calibration, lens profiles and the last 3D result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinetrace.calib import Calibration, Reconstruction
from kinetrace.session import TrackingSession

MAX_VIEWS = 15         # a 4x4 grid; past this the tiles stop being readable at all
REFERENCE_VIEW = 0     # the first camera loaded: the clock everything is measured against


def default_view_name(index: int) -> str:
    return f"cam{index + 1}"


def rate_of(fps: float, ref_fps: float) -> float:
    """Frames of a view per reference frame, from the two nominal rates.
    Rounded so 239.76 / 119.88 is exactly 2.0 rather than 2.0000000001."""
    if fps <= 0 or ref_fps <= 0:
        return 1.0
    return float(round(fps / ref_fps, 6))


class Project:
    """The views of one project and the offsets that align them."""

    def __init__(self, sessions: list[TrackingSession] | None = None,
                 names: list[str] | None = None,
                 offsets: list[float] | None = None,
                 active: int = 0,
                 rates: list[float] | None = None):
        self.sessions: list[TrackingSession] = list(sessions or [])
        n = len(self.sessions)
        self.names: list[str] = list(names) if names else [default_view_name(i) for i in range(n)]
        self.offsets: list[float] = [float(o) for o in offsets] if offsets else [0.0] * n
        # keep the lists the same length whatever a caller passed
        self.names = (self.names + [default_view_name(i) for i in range(n)])[:n]
        self.offsets = (self.offsets + [0.0] * n)[:n]
        ref_fps = self.sessions[REFERENCE_VIEW].fps if n else 0.0
        self.rates: list[float] = ([float(r) for r in rates] if rates
                                   else [rate_of(s.fps, ref_fps) for s in self.sessions])
        self.rates = (self.rates + [1.0] * n)[:n]
        self.active = int(np.clip(active, 0, max(n - 1, 0)))
        self.path: str | None = None       # where this project was last saved
        self.calibration: Calibration | None = None    # DLT per view, view order
        self.reconstruction: Reconstruction | None = None   # last 3D result
        self.lenses: list = [None] * n                  # lens.LensProfile per view (or None)
        # the formats written into the project's exports/ folder at every save (G42)
        self.exports: list[str] = []
        self._normalize()

    # -------------------------------------------------------------- accessors

    @property
    def n_views(self) -> int:
        return len(self.sessions)

    @property
    def session(self) -> TrackingSession | None:
        """The view the user is working in."""
        return self.sessions[self.active] if self.sessions else None

    def name(self, i: int) -> str:
        return self.names[i] if 0 <= i < len(self.names) else ""

    def others(self) -> list[int]:
        return [i for i in range(self.n_views) if i != self.active]

    @property
    def dirty(self) -> bool:
        return any(s.dirty for s in self.sessions)

    @dirty.setter
    def dirty(self, value: bool) -> None:
        for s in self.sessions:
            s.dirty = bool(value)

    # ------------------------------------------------------------------- sync

    def _normalize(self) -> None:
        """Pin the reference camera's offset at 0 by shifting the whole set.

        Only differences between offsets affect anything, so subtracting a
        constant from all of them changes no mapping, no overlap and no export —
        it just makes the numbers on screen mean something: every offset now
        reads as "this many frames later than camera 1"."""
        if self.offsets and self.offsets[REFERENCE_VIEW] != 0:
            shift = self.offsets[REFERENCE_VIEW]
            # re-time t by `shift` reference frames: view i moves by rate_i * shift
            self.offsets = [o - r * shift for o, r in zip(self.offsets, self.rates)]
            self.offsets[REFERENCE_VIEW] = 0.0

    def reference_time(self, view: int, frame: float) -> float:
        """Instant (in reference frames, fractional) shown by `frame` of `view`."""
        return (float(frame) - self.offsets[view]) / self.rates[view]

    def local_frame(self, view: int, t: float) -> float:
        """Fractional local frame of `view` at reference instant `t`."""
        return self.rates[view] * float(t) + self.offsets[view]

    def map_frame_exact(self, src: int, dst: int, frame: float) -> float:
        """Fractional frame in `dst` at the instant of `frame` in `src` (no
        range check) — what the 3D layer interpolates at."""
        return self.local_frame(dst, self.reference_time(src, frame))

    def map_frame(self, src: int, dst: int, frame: int) -> int | None:
        """The frame in view `dst` showing the same instant as `frame` in view
        `src` (nearest whole frame) — or None when that instant is outside
        `dst`'s recording (the camera had not started yet, or had already
        stopped)."""
        if not (0 <= src < self.n_views and 0 <= dst < self.n_views):
            return None
        # Python's round() sends .5 to the EVEN integer, so with a half-frame
        # offset or a 2x camera the mapped frames alternated down / up: some
        # repeated, their neighbours never shown or exported (I19). Ties go up
        # one way and down the other way, so a switch there and back lands on
        # the frame it started from.
        x = self.map_frame_exact(src, dst, int(frame))
        local = int(np.floor(x + 0.5)) if src <= dst else int(np.ceil(x - 0.5))
        if local < 0 or local >= self.sessions[dst].n_frames:
            return None
        return local


    def align_to(self, i: int, shown_frame: int, active_frame: int) -> float:
        """Set view `i`'s offset so that `shown_frame` in it lines up with
        `active_frame` in the active view (the flash/clap alignment step).
        Whole frames only — the fractional part is a later, measured step.

        On the REFERENCE's row (another camera active) the reference cannot
        move — it is the clock — so the ACTIVE camera's offset is set instead:
        the same result as working in the reference and aligning the active
        camera, and every other camera keeps its alignment to the reference
        (G13, owner 2026-09-28). Returns the offset that changed."""
        a = self.active
        if i == REFERENCE_VIEW and a != REFERENCE_VIEW:
            t = self.reference_time(REFERENCE_VIEW, int(shown_frame))
            moved, frame = a, int(active_frame)
        else:
            t = self.reference_time(a, int(active_frame))
            moved, frame = i, int(shown_frame)
        before = list(self.offsets)
        self.offsets[moved] = float(frame) - self.rates[moved] * t
        self._normalize()
        self.sessions[moved].dirty = True
        if any(abs(x - y) > 1e-12 for x, y in zip(before, self.offsets)):
            self.reconstruction = None          # triangulated under the old timing (I23)
        return self.offsets[moved]

    def set_offset(self, i: int, offset: float) -> None:
        """Retime one camera against the reference. The reference itself has no
        offset to set — it IS the zero — so shifting it is meaningless and is
        ignored rather than silently snapping back."""
        if i == REFERENCE_VIEW or not (0 <= i < self.n_views):
            return
        if abs(self.offsets[i] - float(offset)) > 1e-12:
            self.offsets[i] = float(offset)
            self.sessions[i].dirty = True
            self.reconstruction = None          # triangulated under the old timing (I23)

    def set_rate(self, i: int, rate: float) -> None:
        """Override a view's frame-rate ratio (the default comes from the two
        videos' nominal fps, which is right unless a header lies)."""
        if i == REFERENCE_VIEW or not (0 <= i < self.n_views) or rate <= 0:
            return
        if abs(self.rates[i] - float(rate)) > 1e-12:
            self.rates[i] = float(rate)
            self.sessions[i].dirty = True
            self.reconstruction = None          # triangulated under the old timing (I23)

    def set_fps(self, i: int, fps: float) -> bool:
        """The rate camera `i` REALLY recorded at (G38: a file header can lie -
        high-speed footage saved for slow-motion playback says 30). Rescales the
        rates instead of recomputing them, so offsets and any rate set by hand
        or imported are kept: this camera's rate scales by new / old, and for the
        reference (whose rate is 1 by definition) every other camera's rate
        scales the other way. True when something changed."""
        if not (0 <= i < self.n_views) or not fps or fps <= 0:
            return False
        s = self.sessions[i]
        old = float(s.fps or 0.0)
        if old > 0 and abs(old - float(fps)) <= 1e-9:
            return False
        if old > 0:
            if i == REFERENCE_VIEW:
                for j in range(1, self.n_views):
                    self.rates[j] *= old / float(fps)
            else:
                self.rates[i] *= float(fps) / old
        s.fps = float(fps)
        for v in self.sessions:
            v.dirty = True
        self.reconstruction = None          # triangulated under the old timing (I23)
        return True

    def fps_mismatch(self) -> bool:
        """True when the views were not all shot at the same rate. That is
        handled (each view has its own rate), but worth showing: a companion
        then skips or repeats frames as the playhead moves one at a time."""
        return any(abs(r - 1.0) > 1e-9 for r in self.rates)

    def has_fractional_offsets(self) -> bool:
        return any(abs(o - round(o)) > 1e-9 for o in self.offsets)

    def reference_span(self, min_views: int | None = None) -> tuple[float, float] | None:
        """(first, last) reference instant at which at least `min_views` views
        have a frame (None = every view). Triangulation needs only two: a short
        overview clip must not shrink every 3D output to its own span (I15).
        None when no instant qualifies."""
        k = self.n_views if min_views is None else max(1, min(int(min_views), self.n_views))
        spans = [(self.reference_time(i, 0), self.reference_time(i, self.sessions[i].n_frames - 1))
                 for i in range(self.n_views)]
        return self.span_of(spans, k)

    @staticmethod
    def span_of(spans: list[tuple[float, float]], k: int) -> tuple[float, float] | None:
        """(first, last) instant covered by at least `k` of the (start, end)
        intervals, or None."""
        if k < 1 or len(spans) < k:
            return None
        edges = sorted([(a, 0) for a, _ in spans] + [(b, 1) for _, b in spans])   # starts before ends at a tie
        count, first, last = 0, None, None
        for t, kind in edges:
            if kind == 0:
                count += 1
                if count >= k and first is None:
                    first = t
            else:
                if count >= k:
                    last = t
                count -= 1
        return None if first is None or last is None or last < first else (first, last)

    def coverage(self, min_views: int | None = None) -> tuple[int, int]:
        """(first, last) frame of the ACTIVE view inside `reference_span`
        (every view by default -- the camera panel's caption; 2 = what can be
        triangulated)."""
        s = self.session
        if s is None:
            return (0, 0)
        span = self.reference_span(min_views)
        if span is None:
            return (0, -1)
        t_lo, t_hi = span
        lo = int(np.ceil(self.local_frame(self.active, t_lo) - 1e-9))
        hi = int(np.floor(self.local_frame(self.active, t_hi) + 1e-9))
        lo = max(lo, 0)
        hi = min(hi, s.n_frames - 1)
        return (lo, hi) if lo <= hi else (0, -1)

    # ------------------------------------------------------------------ views

    def add_view(self, session: TrackingSession, name: str | None = None,
                 offset: float = 0.0, rate: float | None = None) -> int:
        """Append a camera. Its rate defaults to fps / reference fps, so a
        240 fps camera in a 120 fps rig steps two frames per reference frame
        from the moment it is added."""
        ref_fps = self.sessions[REFERENCE_VIEW].fps if self.sessions else session.fps
        self.sessions.append(session)
        self.names.append(name or default_view_name(self.n_views - 1))
        self.offsets.append(float(offset))
        self.rates.append(float(rate) if rate else rate_of(session.fps, ref_fps))
        self.lenses.append(None)
        self._normalize()
        session.dirty = True
        self.reconstruction = None      # a different camera set: 3D is stale
        return self.n_views - 1

    def remove_view(self, i: int) -> None:
        if not (0 <= i < self.n_views) or self.n_views <= 1:
            return
        del self.sessions[i], self.names[i], self.offsets[i], self.rates[i]
        if i < len(self.lenses):
            del self.lenses[i]
        if self.calibration is not None and i < len(self.calibration.cameras):
            del self.calibration.cameras[i]
        self.reconstruction = None
        # removing the reference promotes the next camera: the new reference's
        # rate becomes the unit, and re-zeroing shifts the set — nothing moves
        if i == REFERENCE_VIEW and self.rates:
            r0 = self.rates[REFERENCE_VIEW]
            if r0 > 0 and abs(r0 - 1.0) > 1e-12:
                self.rates = [r / r0 for r in self.rates]
        self._normalize()
        if self.active >= self.n_views:
            self.active = self.n_views - 1
        elif self.active > i:
            self.active -= 1
        self.dirty = True

    # ------------------------------------------- one landmark list (G19)
    # Every camera is digitized separately, but 3D, the wand calibration and the
    # all-cameras export join cameras BY LANDMARK NAME. A point made in one camera
    # therefore exists in every camera (with no data where it has not been placed),
    # so it can be selected there, clicked on the epipolar line and tracked.

    def missing_landmarks(self) -> dict[int, list]:
        """{view: [PointMeta another camera has and this one lacks, ...]} (the
        first camera that has a name gives its colour, kind and data source)."""
        first: dict[str, object] = {}
        for s in self.sessions:
            for meta in s.points:
                first.setdefault(meta.name, meta)
        out: dict[int, list] = {}
        for v, s in enumerate(self.sessions):
            have = {m.name for m in s.points}
            lack = [m for n, m in first.items() if n not in have]
            if lack:
                out[v] = lack
        return out

    def landmark_order(self, primary: int | None = None) -> list[str]:
        """The one order of the shared list (G26): camera `primary`'s points
        first (the working camera, so its point ids never move under the app),
        then any other names in camera order."""
        first = self.active if primary is None else int(primary)
        seen: dict[str, None] = {}
        for v in [first] + [k for k in range(self.n_views) if k != first]:
            if 0 <= v < self.n_views:
                for m in self.sessions[v].points:
                    seen.setdefault(m.name, None)
        return list(seen)

    def landmark_changes(self, primary: int | None = None) -> list[int]:
        """The cameras `sync_landmarks(primary)` would change (a name missing,
        or the list in another order) -- what an undo step must snapshot."""
        if self.n_views < 2:
            return []
        order = self.landmark_order(primary)
        return [v for v, s in enumerate(self.sessions) if [m.name for m in s.points] != order]

    def sync_landmarks(self, primary: int | None = None) -> int:
        """Give every camera every landmark any camera has, in ONE order
        (`landmark_order(primary)`, G26); returns how many points were added.
        Never touches data. The default-name counters are kept in step, so the
        next new point is not called 'P1' in one camera while another camera's
        'P1' is a different landmark."""
        if self.n_views < 2:
            return 0
        added = 0
        for v, metas in self.missing_landmarks().items():
            for meta in metas:
                self.sessions[v].add_placeholder(meta)
                added += 1
        order = self.landmark_order(primary)
        for s in self.sessions:
            names = [m.name for m in s.points]
            if names != order:
                at = {n: i for i, n in enumerate(names)}
                s.reorder_points([at[n] for n in order if n in at])
        top = max(s._name_counter for s in self.sessions)
        for s in self.sessions:
            s._name_counter = top
        return added

    def landmark_views(self, name: str) -> list[tuple[int, int]]:
        """[(view, pid), ...] of the cameras that have landmark `name`."""
        out = []
        for v, s in enumerate(self.sessions):
            pid = s.pid_by_name(name)
            if pid is not None:
                out.append((v, pid))
        return out

    def rename_landmark(self, old: str, desired: str) -> str:
        """Rename a landmark in EVERY camera to one name that is free in all of
        them (a per-camera suffix would split one landmark into two for 3D).
        Returns the name applied."""
        where = self.landmark_views(old)
        base = desired.strip() or "point"
        cand, k = base, 2
        while any(s.unique_name(cand, exclude_pid=s.pid_by_name(old)) != cand for s in self.sessions):
            cand, k = f"{base} ({k})", k + 1
        for v, pid in where:
            self.sessions[v].rename_point(pid, cand)
        return cand

    def set_active(self, i: int) -> int | None:
        """Switch the working view. Returns the frame the new view should show
        so the playhead stays on the same instant (None = it has no frame
        there, in which case the caller should keep whatever it had)."""
        if not (0 <= i < self.n_views) or i == self.active:
            return None
        s = self.session
        target = self.map_frame(self.active, i, s.current_frame if s else 0)
        self.active = i
        return target

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        """Write the Kinetrace project file (see projectfile.py)."""
        from kinetrace import projectfile
        projectfile.save(self, path)

    @staticmethod
    def load(path) -> "Project":
        from kinetrace import projectfile
        return projectfile.load(path)

    # ---------------------------------------------------------------- exports

    def export_multi_dltdv(self, path: str | Path, flip_y: bool = False,
                           pixel_origin: float = 1.0) -> list[str]:
        """One DLTdv/Argus xypts file covering EVERY view, which is what a 3D
        reconstruction consumes: one row per instant, columns grouped
        `pt{n}_cam{k}_X/Y`, cameras in view order. Rows are the overlapping
        reference camera's frames from 0 -- row k = frame k of the first
        camera, DLTdv8's own layout, and the xyz export's `frame` column -- and
        each view is sampled through its own offset (I18: rows used to start at
        the overlap's first frame of the WORKING camera, with nothing saying so).

        Pixel convention: DLTdv8's by default -- TOP-left origin, first pixel
        = 1 (x + 1, y + 1), the frame a DLTdv8 project's clicks and DLT
        coefficients are in (see `calib.CameraCalibration`). `flip_y=True`
        writes the older bottom-left variant (y' = H - y per view height).

        Landmarks are matched BY NAME across views (that is what triangulation
        needs); a name missing from a view, or no data there, exports as NaN
        like DLTdv's own files (I20: blank cells read as 0 in MATLAB's csvread).
        Writes a `*_pointnames.csv` sidecar (names, the convention, which video
        is camK, the frame basis) and returns the files written."""
        from kinetrace.session import dltdv_convention_text
        path = Path(path)
        po = float(pixel_origin)
        names: list[str] = []
        for s in self.sessions:                  # union, first-seen order
            for q in s.points:
                if q.name not in names:
                    names.append(q.name)
        cols = [f"pt{i + 1}_cam{c + 1}_{ax}"
                for i in range(len(names)) for c in range(self.n_views) for ax in ("X", "Y")]
        lines = [",".join(cols)]
        index = [{q.name: j for j, q in enumerate(s.points)} for s in self.sessions]
        exp = [s.exportable for s in self.sessions]   # once, not per cell: 150 s -> 4 s at 40k frames (I21)
        ref = REFERENCE_VIEW
        for t in range(self.sessions[ref].n_frames):
            fr = [self.map_frame(ref, c, t) for c in range(self.n_views)]
            cells: list[str] = []
            for nm in names:
                for c, s in enumerate(self.sessions):
                    f = fr[c]
                    j = index[c].get(nm)
                    if f is None or j is None or not exp[c][f, j]:
                        cells.append("NaN,NaN")
                        continue
                    x, y = s.tracks[f, j]
                    yy = (s.height - float(y)) if flip_y else float(y) + po
                    cells.append(f"{float(x) + po:.4f},{yy:.4f}")
            lines.append(",".join(cells))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        side = path.with_name(path.stem + "_pointnames.csv")
        cams = [f"cam{c + 1},{self.name(c)} = {Path(s.video_path).name}".replace("\n", " ")
                for c, s in enumerate(self.sessions)]
        side.write_text("\n".join(["name,cameras"]
                                  + [f"{nm},{self.n_views}" for nm in names]
                                  + [f"convention,{dltdv_convention_text(flip_y, po)}",
                                     f"rows,row k = frame k (from 0) of {self.name(ref)}, the reference camera; "
                                     "the other cameras are sampled at the same instant through their offsets"]
                                  + cams) + "\n",
                        encoding="utf-8", newline="")
        return [str(path), str(side)]
