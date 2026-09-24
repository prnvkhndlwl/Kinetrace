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
Schema v5 writes every view into one `.cotrk` under a `v{i}_` key prefix
(`TrackingSession.to_arrays(prefix)`), plus the view names, float offsets,
rates, which one was active, the calibration (`calib_*`) and the last 3D
reconstruction (`xyz_*`). v4 (integer offsets, no rates) and v3-or-older
(one unprefixed session) still load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from kinetrace import APP_VERSION
from kinetrace.calib import Calibration, Reconstruction
from kinetrace.session import TrackingSession

PROJECT_SCHEMA = 5
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

    def frame_for(self, i: int, frame: int | None = None) -> int | None:
        """Frame in view `i` matching the active view's `frame` (default: the
        active session's current frame)."""
        if frame is None:
            s = self.session
            frame = s.current_frame if s is not None else 0
        return self.map_frame(self.active, i, frame)

    def align_to(self, i: int, shown_frame: int, active_frame: int) -> float:
        """Set view `i`'s offset so that `shown_frame` in it lines up with
        `active_frame` in the active view (the flash/clap alignment step).
        Whole frames only — the fractional part is a later, measured step.
        Returns the new offset."""
        t = self.reference_time(self.active, int(active_frame))
        before = list(self.offsets)
        self.offsets[i] = float(int(shown_frame)) - self.rates[i] * t
        self._normalize()   # aligning the reference shifts everyone else instead
        self.sessions[i].dirty = True
        if any(abs(a - b) > 1e-12 for a, b in zip(before, self.offsets)):
            self.reconstruction = None          # triangulated under the old timing (I23)
        return self.offsets[i]

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

    def rename_view(self, i: int, name: str) -> str:
        """Rename with collision protection — the names become export column
        prefixes, so two views may not share one."""
        name = (name or "").strip() or default_view_name(i)
        taken = {n for j, n in enumerate(self.names) if j != i}
        if name in taken:
            k = 2
            while f"{name} ({k})" in taken:
                k += 1
            name = f"{name} ({k})"
        self.names[i] = name
        self.sessions[i].dirty = True
        return name

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

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        arrays: dict = {}
        for i, s in enumerate(self.sessions):
            # a body MESH is large and barely compressible (float16 vertices):
            # compressing it froze every 30 s autosave for seconds; it is written
            # stored, below (I93: 1.7 s -> 0.1 s for 500 meshed frames, same file)
            arrays.update(s.to_arrays(f"v{i}_", body_mesh=False))
        # saved whatever its camera count: after a camera is added the 3D menu
        # waits for a calibration of every camera, but the one made for the
        # first cameras must not vanish from the file (I16)
        if self.calibration is not None and len(self.calibration) > 0:
            arrays.update(self.calibration.to_arrays("calib_"))
        if any(l is not None for l in self.lenses):
            arrays["lens_meta"] = json.dumps([None if l is None else l.to_json() for l in self.lenses])
        if self.reconstruction is not None:
            r = self.reconstruction
            arrays.update({"xyz_t0": int(r.t0), "xyz_names": np.array(r.names, dtype=object),
                           "xyz_pts": r.xyz.astype(np.float64), "xyz_res": r.residual.astype(np.float64),
                           "xyz_ncams": r.n_cams.astype(np.int32), "xyz_unit": str(r.unit)})
            if r.per_cam is not None:
                arrays["xyz_percam"] = np.asarray(r.per_cam, np.float32)
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                schema=PROJECT_SCHEMA,
                app_version=APP_VERSION,
                n_views=len(self.sessions),
                view_names=np.array(self.names, dtype=object),
                view_offsets=np.array(self.offsets, np.float64),
                view_rates=np.array(self.rates, np.float64),
                active_view=int(self.active),
                **arrays,
            )
        meshes: dict = {}
        for i, s in enumerate(self.sessions):
            if s.body is not None:
                meshes.update(s.body.mesh_arrays(f"v{i}_body_"))
        if meshes:
            import zipfile
            with zipfile.ZipFile(tmp, "a", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for key, arr in meshes.items():
                    with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                        np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)
        os.replace(tmp, path)   # atomic: a crash never corrupts the previous save
        self.dirty = False
        self.path = str(path)

    @staticmethod
    def load_npz(path: str | Path) -> "Project":
        """Open a project. v5 files carry every view plus rates, calibration
        and 3D; v4 has integer offsets and no rates (derived from the fps);
        v3 and older are a single unprefixed session and load as a one-view
        project."""
        with np.load(path, allow_pickle=True) as z:
            have = set(z.files)
            if "n_views" not in have:            # v3 or older: one view, no prefix
                p = Project([TrackingSession.from_arrays(z)])
            else:
                n = int(z["n_views"])
                sessions = [TrackingSession.from_arrays(z, f"v{i}_") for i in range(n)]
                names = [str(v) for v in z["view_names"]] if "view_names" in have else None
                offsets = ([float(v) for v in z["view_offsets"]]
                           if "view_offsets" in have else None)
                rates = ([float(v) for v in z["view_rates"]] if "view_rates" in have else None)
                active = int(z["active_view"]) if "active_view" in have else 0
                p = Project(sessions, names, offsets, active, rates)
                cal = Calibration.from_arrays(z, "calib_")
                if cal is not None and 0 < len(cal) <= n:     # fewer = cameras added since (I16)
                    p.calibration = cal
                if "lens_meta" in have:
                    try:
                        from kinetrace.lens import LensProfile
                        raw = json.loads(str(z["lens_meta"]))
                        lenses = [None if d is None else LensProfile.from_json(d) for d in raw]
                        if len(lenses) == n:
                            p.lenses = lenses
                    except (ValueError, TypeError, KeyError):
                        pass          # a lens profile that does not parse is simply absent
                if "xyz_pts" in have:
                    p.reconstruction = Reconstruction(
                        int(z["xyz_t0"]), [str(v) for v in z["xyz_names"]],
                        np.asarray(z["xyz_pts"], np.float64), np.asarray(z["xyz_res"], np.float64),
                        np.asarray(z["xyz_ncams"], np.int32), str(z["xyz_unit"]),
                        np.asarray(z["xyz_percam"], np.float32) if "xyz_percam" in have else None)
        p.dirty = False
        p.path = str(path)
        return p

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
