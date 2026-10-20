"""A DLTdv8 project (`*_dvProject.mat`) read as a whole (I310): File > Import > DLTdv Project.

DLTdv8 keeps everything in one MATLAB struct, `udExport`, saved as MATLAB v7.3 (HDF5, read with h5py) or, when
re-saved by hand, v7 (read with scipy). Both forms give the same `DLTdvProject`:

* `data.xypts`: a MATLAB sparse matrix. Rows are frames of the project's clock (video 1); columns are, for each
  point, for each video, x then y in DLTdv's pixels (first pixel 1, y down from the top). An empty cell means
  "not clicked". Only the rows holding any click are kept, so a 40,000-row project with 200 clicked rows costs
  200 rows.
* `data.dltpts` / `dltres`: DLTdv's own 3D and residual per row, used only as a CHECK of Kinetrace's 3D.
* `data.dltcoef` (11 x V), `movsizes` (h, w per video), `movFrameRates`, `movFrameRateRatios`, `pointNames`
  (DLTdv 8.5+; older ones are numbered pt1..ptN), `camd` (one LWM lens profile per video, a MATLAB object).
* `video.files` (the clicking machine's paths), `video.offset` (frames, as text).
* `data.offset` (rows x videos): the offset in force when each row was clicked. On every real project seen, a
  row clicked in a camera carries that camera's `video.offset`; any that does not is counted in `notes`.

DLTdv's frame rule (1-based rows and frames): video c shows frame `ratio_c * row + offset_c` at clock row
`row`. In 0-based terms camera c's frame of row r is `ratio_c * (r + 1) + offset_c - 1` (checked on real
projects, including a ratio-2 camera). What frame number a picture has depends on the MATLAB that decoded the
video: MATLAB 2024b renumbered GoPro files, so a project's clicks can sit whole frames away from Kinetrace's
pictures (I310). `clickshift.py` measures that; `frame_map` applies the shift the user takes.

Every file is opened read-only (h5py mode "r", scipy loadmat); nothing is ever written next to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kinetrace.calib import Calibration, CameraCalibration, LWMUndistort, NoUndistort
from kinetrace.project import half_up

MAX_BYTES = 2 * 1024 ** 3          # the dense block of clicked rows (rows x points x videos x 2 x 8 bytes)


class DLTdvImportError(ValueError):
    """A file that cannot be imported as a DLTdv8 project; the message says why in plain words."""


@dataclass
class DLTdvProject:
    path: str
    version: float                     # DLTdv's version (8.5, ...); NaN when the file does not say
    point_names: list[str]
    video_files: list[str]             # base names (what is searched for)
    stored_files: list[str]            # as stored (the clicking machine's paths)
    frame_rates: np.ndarray            # (V,) fps
    ratios: np.ndarray                 # (V,) video rate / clock rate
    sizes: np.ndarray                  # (V, 2) int: width, height
    offsets: np.ndarray                # (V,) video.offset, frames
    dltcoef: np.ndarray                # (11, V) in DLTdv's pixels; a column of zeros = not calibrated
    n_rows: int                        # the project's row count (the clock's frames)
    rows: np.ndarray                   # (R,) 0-based rows that hold any click or 3D, increasing
    xy: np.ndarray                     # (R, P, V, 2) DLTdv pixels (1-based); NaN = not clicked
    xyz: np.ndarray                    # (R, P, 3) DLTdv's own 3D; NaN where none
    res: np.ndarray                    # (R, P) DLTdv's residual; NaN where none
    lenses: list                       # (V,) LWMUndistort in DLTdv pixels, or NoUndistort
    lens_names: list[str]              # (V,) the profile each video names ("" = none)
    notes: list[str] = field(default_factory=list)

    @property
    def n_videos(self) -> int:
        return len(self.video_files)

    @property
    def name(self) -> str:
        """The project's name for a new Kinetrace project: the file's stem without DLTdv's suffix."""
        stem = Path(self.path).stem
        return stem[: -len("_dvProject")] if stem.lower().endswith("_dvproject") else stem

    def clicked(self, v: int) -> np.ndarray:
        """(R, P) bool: camera v has a click."""
        return np.isfinite(self.xy[:, :, v, 0])

    def n_clicks(self, v: int) -> int:
        return int(self.clicked(v).sum())

    def calibrated(self, v: int) -> bool:
        c = self.dltcoef[:, v]
        return bool(np.isfinite(c).all() and np.any(c != 0))

    def calibration(self) -> Calibration:
        """The DLT cameras in DLTdv's own convention (first pixel 1, y down; stated by the file), each with its
        lens; a video DLTdv has no coefficients for is not calibrated (None). Callers convert with
        `to_internal()` where it enters the project."""
        cams = [CameraCalibration(self.dltcoef[:, c].copy(), int(self.sizes[c, 0]), int(self.sizes[c, 1]),
                                  self.lenses[c], 1.0, False) if self.calibrated(c) else None
                for c in range(self.n_videos)]
        cal = Calibration(cams, source=Path(self.path).name)
        cal.notes = [n for n in self.notes if "lens" in n.lower() or "undistort" in n.lower()]
        return cal

    # ---------------------------------------------------------------- frames
    def local_frames(self, v: int, shift: float = 0.0) -> np.ndarray:
        """(R,) camera v's 0-based frame of each kept row, by DLTdv's rule plus `shift` (unrounded)."""
        return self.ratios[v] * (self.rows + 1.0) + self.offsets[v] - 1.0 + shift

    def timing(self, views: list[int], shifts: list[float]) -> tuple[list[float], list[float]]:
        """Kinetrace rates and offsets for the cameras `views` (the first is the reference clock), with each
        camera's frame-check shift: local = rate * t + offset, t = the first camera's frame. Exact for every
        row: camera c's frame of row r is rate_c * F_0(r) + offset_c."""
        r0 = self.ratios[views[0]]
        rates = [float(self.ratios[v] / r0) for v in views]
        f0 = self.ratios[views[0]] + self.offsets[views[0]] - 1.0 + shifts[0]            # F_0 at row 0
        offs = [float(self.ratios[v] + self.offsets[v] - 1.0 + s - rate * f0)
                for v, s, rate in zip(views, shifts, rates)]
        return rates, offs

    def check_inputs(self, v: int) -> tuple[np.ndarray, np.ndarray]:
        """What the click-shift check needs for camera v: (n, 2) clicks in Kinetrace pixels (0-based) and (n,)
        their frames by DLTdv's rule (no shift)."""
        r, k = np.nonzero(self.clicked(v))
        frames = half_up(self.local_frames(v)[r])
        return self.xy[r, k, v] - 1.0, np.atleast_1d(frames).astype(np.int64)

    def imported(self, views: list[int], shifts: list[float]):
        """A `trackio.Imported` holding the cameras `views` (in that order) as hand clicks, and per camera the
        function `frame_of_row` that `trackio.apply` needs (row index into `rows` -> 0-based local frame)."""
        from kinetrace.trackio import Imported
        # DLTdv's first pixel is 1: made 0-based in float64 BEFORE the float32 a session holds, so each value is
        # float32(x - 1) exactly, never float32(x) - 1 rounded twice
        xy = (np.ascontiguousarray(np.moveaxis(self.xy[:, :, views], 2, 0)) - 1.0).astype(np.float32)  # (C,R,P,2)
        has = np.isfinite(xy[..., 0])
        imp = Imported("dvproject", list(self.point_names), xy, has.astype(np.float32),
                       visible=has, manual=has, hidden=np.zeros_like(has), pixel_origin=0.0, flip_y=False,
                       rows="frames")
        maps = []
        for v, s in zip(views, shifts):
            fr = np.atleast_1d(half_up(self.local_frames(v, s))).astype(np.int64)
            maps.append(lambda r, fr=fr: int(fr[r]))
        return imp, maps

    def shared_frames(self, v: int, shift: float = 0.0) -> int:
        """Clicks of camera v that land on a frame another of its rows also lands on (a ratio below 1 or a
        fractional offset): the later row is the one kept."""
        fr = np.atleast_1d(half_up(self.local_frames(v, shift)))
        has = self.clicked(v).any(axis=1)
        _, counts = np.unique(fr[has], return_counts=True)
        return int((counts - 1).sum())


# -------------------------------------------------------------------- reading

def read(path) -> DLTdvProject:
    """Read a DLTdv8 project, either MATLAB form. DLTdvImportError (a sentence) when it is not one or is
    damaged."""
    path = str(path)
    name = Path(path).name
    try:
        import h5py
        hdf = h5py.is_hdf5(path)
    except ImportError:
        hdf = False
        h5py = None
    if not hdf:
        with open(path, "rb") as fh:
            head = fh.read(20)
        if head.startswith(b"MATLAB 7.3") and h5py is None:
            raise DLTdvImportError(f"{name} is a MATLAB v7.3 file and the h5py package is missing: run the "
                                   "launcher again (it installs what is missing), or re-save the project as "
                                   "MATLAB v7 in MATLAB")
        if not head.startswith(b"MATLAB 5.0"):
            raise DLTdvImportError(f"{name} is not a MATLAB file, so not a DLTdv8 project (DLTdv8 saves "
                                   "…_dvProject.mat)")
        return _read_v7(path)
    return _read_v73(path)


def _finish(path, version, names, stored, rates, ratios, sizes_hw, offsets_txt, coef, shape, coo_xy, coo_xyz,
            coo_res, coo_off, lenses, lens_names, notes) -> DLTdvProject:
    """Check the parts against each other and build the project from the sparse matrices' (row, col, value)
    triplets (only the rows that hold anything become dense)."""
    name = Path(path).name
    nrows, ncols = shape
    P, V = len(names), len(stored)
    if not V:
        raise DLTdvImportError(f"{name}: the project names no videos")
    if ncols != 2 * P * V:
        raise DLTdvImportError(f"{name}: its clicks have {ncols} columns where {P} point(s) in {V} video(s) need "
                               f"{2 * P * V}; the file is damaged or not a DLTdv8 project")
    if np.isfinite(version) and version < 8:
        raise DLTdvImportError(f"{name} was saved by DLTdv {version:g}; only DLTdv8 projects can be imported "
                               "(open it in DLTdv8 and save it again)")
    ratios = np.asarray(ratios, np.float64).ravel()
    rates = np.asarray(rates, np.float64).ravel()
    if len(ratios) != V or not np.all(np.isfinite(ratios) & (ratios > 0)):
        raise DLTdvImportError(f"{name}: the videos' frame-rate ratios are missing or not positive")
    offs = []
    for c, s in enumerate(offsets_txt):
        try:
            offs.append(float(str(s).strip() or 0))
        except ValueError:
            raise DLTdvImportError(f"{name}: video {c + 1}'s offset is '{s}', not a number of frames") from None
    if len(offs) != V:
        raise DLTdvImportError(f"{name}: {len(offs)} video offset(s) for {V} videos")
    sizes_hw = np.asarray(sizes_hw, np.float64).reshape(V, 2)
    coef = np.asarray(coef, np.float64)
    if coef.size == 0:
        coef = np.zeros((11, V))
    if coef.shape != (11, V):
        raise DLTdvImportError(f"{name}: the DLT coefficients are {coef.shape[0]} x {coef.shape[1]}, not 11 x {V}")
    used = np.unique(np.concatenate([coo_xy[0], coo_xyz[0], coo_res[0]]).astype(np.int64))
    need = len(used) * P * V * 2 * 8
    if need > MAX_BYTES:
        raise DLTdvImportError(f"{name}: {len(used):,} rows hold clicks, {need / 1e9:.1f} GB in memory (the "
                               f"limit is {MAX_BYTES / 1e9:.0f} GB); split the project in DLTdv8")

    def dense(coo, cols):
        """The kept rows of a sparse matrix (a row outside them, e.g. an offset recorded on a row with no
        click, is not needed)."""
        out = np.full((len(used), cols), np.nan)
        r, c, x = coo
        i = np.searchsorted(used, np.asarray(r, np.int64))
        keep = (i < len(used)) & (used[np.minimum(i, len(used) - 1)] == np.asarray(r, np.int64)) \
            if len(used) else np.zeros(len(r), bool)
        out[i[keep], np.asarray(c, np.int64)[keep]] = np.asarray(x)[keep]
        return out
    xy = dense(coo_xy, ncols).reshape(len(used), P, V, 2)
    zero = xy == 0                      # never a DLTdv pixel (the first is 1): an empty cell stored explicitly
    if zero.any():
        notes.append(f"{int(zero.sum())} click coordinate(s) are exactly 0 and were read as not clicked")
        xy[zero] = np.nan
    xy[np.isnan(xy).any(axis=-1)] = np.nan
    xyz = dense(coo_xyz, 3 * P).reshape(len(used), P, 3) if len(coo_xyz[0]) else np.full((len(used), P, 3), np.nan)
    res = dense(coo_res, P) if len(coo_res[0]) else np.full((len(used), P), np.nan)
    if coo_off is not None:
        roff = dense(coo_off, V)
        for c in range(V):
            has = np.isfinite(xy[:, :, c, 0]).any(axis=1)
            rec = np.nan_to_num(roff[has, c], nan=0.0)           # an empty cell is an offset of 0
            odd = int((rec != offs[c]).sum())
            if odd:
                notes.append(f"{odd} row(s) clicked in video {c + 1} were clicked while its offset was "
                             f"{', '.join(f'{v:g}' for v in np.unique(rec[rec != offs[c]])[:3])}, not today's "
                             f"{offs[c]:g}: they are placed by today's offset; check them on the pictures")
    stored = [str(s) for s in stored]
    files = [s.replace("\\", "/").rstrip("/").split("/")[-1] for s in stored]
    return DLTdvProject(path=path, version=version, point_names=list(names), video_files=files,
                        stored_files=stored, frame_rates=rates if len(rates) == V else np.full(V, np.nan),
                        ratios=ratios, sizes=np.stack([sizes_hw[:, 1], sizes_hw[:, 0]], 1).astype(int),
                        offsets=np.asarray(offs), dltcoef=coef, n_rows=int(nrows), rows=used, xy=xy,
                        xyz=xyz, res=res, lenses=lenses, lens_names=lens_names, notes=notes)


def _names(names, P: int, notes: list) -> list[str]:
    """The points' names, unique the way Kinetrace names landmarks: a name already used (ignoring capitals, as
    the exporters write it) becomes "name (2)", "name (3)" ... so every camera pairs the same DLTdv point."""
    from kinetrace.trackio import name_key
    names = [str(n).strip() for n in names]
    if len(names) != P or not all(names):
        notes.append("the project stores no point names (DLTdv8 before 8.5): the points are numbered pt1, pt2, ...")
        return [f"pt{k + 1}" for k in range(P)]
    out, seen, renamed = [], set(), []
    for n in names:
        new, k = n, 1
        while name_key(new).casefold() in seen:
            k += 1
            new = f"{n} ({k})"
        seen.add(name_key(new).casefold())
        out.append(new)
        if new != n:
            renamed.append(new)
    if renamed:
        notes.append(f"{len(renamed)} point name(s) repeat another (capitals ignored) and were numbered: "
                     + ", ".join(renamed[:6]) + (" ..." if len(renamed) > 6 else ""))
    return out


def _lens_name(s: str) -> str:
    return s.split(":", 1)[1].strip() if ":" in s else ""


# ---------------------------------------------------------------- MATLAB v7 (scipy)

def _read_v7(path) -> DLTdvProject:
    from scipy.io import loadmat
    name = Path(path).name
    try:
        m = loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as e:  # noqa: BLE001 - a damaged file in words
        raise DLTdvImportError(f"{name} could not be read as a MATLAB file ({type(e).__name__}: {e})") from None
    ud = m.get("udExport")
    if ud is None or not hasattr(ud, "data") or not hasattr(ud, "video"):
        raise DLTdvImportError(f"{name} is not a DLTdv project (it holds no udExport): choose the "
                               "…_dvProject.mat DLTdv8 saved")
    d, v = ud.data, ud.video

    def cells(x) -> list:
        return [] if x is None else [("" if (isinstance(e, np.ndarray) and e.size == 0) else e)
                                     for e in np.atleast_1d(np.asarray(x, dtype=object)).ravel()]

    def coo(sp):
        if sp is None or not hasattr(sp, "tocoo"):
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        c = sp.tocoo()
        return c.row, c.col, np.asarray(c.data, np.float64)

    xyp = getattr(d, "xypts", None)
    if xyp is None or not hasattr(xyp, "shape"):
        raise DLTdvImportError(f"{name} is not a DLTdv8 project (it holds no clicks, data.xypts)")
    V = int(np.asarray(getattr(d, "nvid", len(cells(v.files)))).item())
    P = int(np.asarray(getattr(d, "numpts", xyp.shape[1] // max(1, 2 * V))).item())
    notes: list[str] = []
    names = _names(cells(getattr(d, "pointNames", None)), P, notes)
    # the lenses (and their sentences) exactly as File > Import > Calibration has always read them (I102)
    lenses: list = [NoUndistort() for _ in range(V)]
    if hasattr(d, "camd") and hasattr(d, "dltcoef"):
        try:     # the lenses (and their sentences) exactly as File > Import > Calibration has read them (I102)
            cal = Calibration.load_dltdv_project(path)
            if len(cal.cameras) == V:
                lenses = [cam.undistort for cam in cal.cameras]
            notes += cal.notes
        except Exception as e:  # noqa: BLE001 - the clicks still import; SAY the lenses did not
            notes.append(f"The project's lens undistortion could not be read ({type(e).__name__}: {e}), so every "
                         "camera would be used as filmed.")
    coef = np.asarray(getattr(d, "dltcoef", []), np.float64)
    coef = coef.reshape(11, -1) if coef.size else np.zeros((11, V))
    off = getattr(d, "offset", None)
    version = np.asarray(getattr(ud, "DLTdv_version", np.nan), np.float64).ravel()
    lens_names = [_lens_name(str(s)) for s in cells(getattr(v, "undistortString", None))] or [""] * V
    return _finish(path, float(version[0]) if version.size else np.nan, names, cells(v.files),
                   np.atleast_1d(getattr(d, "movFrameRates", [])),
                   np.atleast_1d(getattr(d, "movFrameRateRatios", np.ones(V))),
                   np.atleast_2d(np.asarray(d.movsizes, np.float64)).reshape(-1, 2), cells(v.offset), coef,
                   xyp.shape, coo(xyp), coo(getattr(d, "dltpts", None)), coo(getattr(d, "dltres", None)),
                   coo(off) if off is not None else None, lenses, lens_names, notes)


# ---------------------------------------------------------------- MATLAB v7.3 (HDF5)

def _cls(obj) -> str:
    c = obj.attrs.get("MATLAB_class", b"")
    return c.decode() if isinstance(c, bytes) else str(c)


def _char(obj) -> str:
    if obj.attrs.get("MATLAB_empty", 0):
        return ""
    return "".join(chr(int(c)) for c in np.asarray(obj[()]).flatten(order="F") if c)


def _cell(f, ds) -> list:
    """A MATLAB cell array: its elements as str (char), float (a numeric scalar) or "" (anything else)."""
    import h5py
    out = []
    for ref in np.asarray(ds[()]).flatten(order="F"):
        o = f[ref]
        if isinstance(o, h5py.Dataset) and _cls(o) == "char":
            out.append(_char(o))
        elif isinstance(o, h5py.Dataset) and _cls(o) == "double" and not o.attrs.get("MATLAB_empty", 0) \
                and o.size == 1:
            out.append(f"{float(np.asarray(o[()]).ravel()[0]):g}")
        else:
            out.append("")
    return out


def _coo(grp):
    """A MATLAB sparse matrix (CSC: data, ir = row indices, jc = column starts) as (rows, cols, values) and its
    shape."""
    nrows = int(grp.attrs["MATLAB_sparse"])
    jc = np.asarray(grp["jc"][()], np.int64).ravel()
    ncols = len(jc) - 1
    if "data" in grp and "ir" in grp:
        data = np.asarray(grp["data"][()], np.float64).ravel()
        ir = np.asarray(grp["ir"][()], np.int64).ravel()
    else:                                         # an all-empty sparse matrix keeps only jc
        data, ir = np.zeros(0), np.zeros(0, np.int64)
    cols = np.repeat(np.arange(ncols), np.diff(jc))
    return (ir, cols, data), (nrows, ncols)


def _num(ds, default=np.nan) -> np.ndarray:
    if ds is None or ds.attrs.get("MATLAB_empty", 0):
        return np.atleast_1d(np.asarray(default, np.float64))
    return np.asarray(ds[()], np.float64)


def _object_id(ds) -> int | None:
    """The MCOS object id a MATLAB object reference names ([0xDD000000, 2, 1, 1, id, class]), else None."""
    import h5py
    if not isinstance(ds, h5py.Dataset) or ds.attrs.get("MATLAB_empty", 0):
        return None
    meta = np.asarray(ds[()]).astype(np.uint64).ravel()
    if meta.size >= 6 and meta[0] == 0xDD000000 and meta[1] == 2 and meta[2] == 1 and meta[3] == 1:
        return int(meta[4])
    return None


def _lenses_v73(f, d, V: int, name: str) -> tuple[list, list[str]]:
    """One LWM profile per video from the MCOS store: object id k's saved properties are the struct at cell
    1 + k of `#subsystem#/MCOS` (the same rule as the v7 reader, `calib._assign_lwm_profiles`). A video with
    no object reference is used as filmed. A profile that is named but cannot be read leaves EVERY camera
    as filmed and says so (never a silent mix, I102)."""
    import h5py
    if "camd" not in d:
        return [NoUndistort() for _ in range(V)], []
    try:
        ids = [_object_id(f[ref]) for ref in np.asarray(d["camd"][()]).flatten(order="F")]
        if not any(k is not None for k in ids):
            return [NoUndistort() for _ in range(V)], []
        cells = np.asarray(f["#subsystem#"]["MCOS"][()]).ravel()
        out = []
        for c, k in enumerate(ids[:V]):
            if k is None:
                out.append(NoUndistort())
                continue
            if not 1 + k < len(cells):
                raise ValueError(f"camera {c + 1} names lens object {k}, which the file does not hold")
            g = f[cells[1 + k]]
            if not isinstance(g, h5py.Group) or "uvPoints" not in g or "xyPoints" not in g:
                raise ValueError(f"camera {c + 1}'s lens object {k} is not an LWM profile")
            raw = np.asarray(g["uvPoints"][()], np.float64).T          # MATLAB (M, 2) is HDF5 (2, M)
            und = np.asarray(g["xyPoints"][()], np.float64).T
            n = int(np.asarray(g["N"][()]).item()) if "N" in g else 12
            out.append(LWMUndistort(raw, und, n))
        out += [NoUndistort() for _ in range(V - len(out))]
        bare = [c + 1 for c, k in enumerate(ids[:V]) if k is None]
        return out, ([] if not bare else [
            "Lens undistortion was found for every camera except camera(s) " + ", ".join(map(str, bare))
            + ", which the DLTdv project leaves as filmed."])
    except Exception as e:  # noqa: BLE001 - the DLT and the clicks still load; SAY the lenses did not
        return [NoUndistort() for _ in range(V)], [
            f"The project's lens undistortion could not be read ({e}), so every camera would be used as filmed "
            f"-- pixels of error at the edges of a GoPro-style lens. Import the cameras' lens profiles another "
            f"way before trusting the 3D."]


def _read_v73(path) -> DLTdvProject:
    import h5py
    name = Path(path).name
    try:
        f = h5py.File(path, "r")                                     # read-only
    except Exception as e:  # noqa: BLE001
        raise DLTdvImportError(f"{name} could not be opened ({type(e).__name__}: {e})") from None
    with f:
        if "udExport" not in f:
            raise DLTdvImportError(f"{name} is not a DLTdv project (it holds no udExport): choose the "
                                   "…_dvProject.mat DLTdv8 saved")
        ud = f["udExport"]
        if "data" not in ud or "video" not in ud or "xypts" not in ud["data"]:
            raise DLTdvImportError(f"{name} is not a DLTdv8 project (it holds no clicks, data.xypts)")
        d, v = ud["data"], ud["video"]
        stored = _cell(f, v["files"]) if "files" in v else []
        V = int(_num(d.get("nvid"), len(stored)).ravel()[0])
        coo_xy, shape = _coo(d["xypts"])
        P = int(_num(d.get("numpts"), shape[1] // max(1, 2 * V)).ravel()[0])
        notes: list[str] = []
        names = _names(_cell(f, d["pointNames"]) if "pointNames" in d else [], P, notes)
        lenses, lnotes = _lenses_v73(f, d, V, name)
        notes += lnotes
        empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
        coef = _num(d.get("dltcoef"), np.zeros((V, 11))).T                     # MATLAB (11, V)
        sizes = _num(d.get("movsizes"), np.zeros((2, V))).T                    # MATLAB (V, 2): h, w
        version = float(_num(ud.get("DLTdv_version")).ravel()[0])
        return _finish(path, version, names, stored, _num(d.get("movFrameRates")).ravel(),
                       _num(d.get("movFrameRateRatios"), np.ones(V)).ravel(), sizes,
                       _cell(f, v["offset"]) if "offset" in v else ["0"] * V, coef.reshape(11, -1) if coef.size else coef,
                       shape, coo_xy,
                       _coo(d["dltpts"])[0] if "dltpts" in d else empty,
                       _coo(d["dltres"])[0] if "dltres" in d else empty,
                       _coo(d["offset"])[0] if isinstance(d.get("offset"), h5py.Group) else None,
                       lenses,
                       [_lens_name(s) for s in _cell(f, v["undistortString"])] if "undistortString" in v
                       else [""] * V, notes)


# -------------------------------------------------------------------- videos and the 3D check

def _key(p) -> str:
    import os
    return os.path.normcase(os.path.abspath(str(p)))


def find_videos(dv: DLTdvProject, folder: str | None = None) -> list[str | None]:
    """Each video, by its base name: where the project says, then beside the .mat, in the .mat's parent folder
    and in `folder` (the one the user was asked for) -- in each of those also inside a sub-folder named like
    the stored path's folder (a GoPro rig's cam1/GX010001.MP4, cam2/GX010001.MP4). Two cameras never get the
    same file: a candidate another camera already has is skipped (None = not found). Touches the file system
    (a stored network path can take seconds): call it off the GUI thread."""
    here = Path(dv.path).resolve().parent
    places = [here, here.parent] + ([Path(folder)] if folder else [])
    taken: set[str] = set()
    out: list[str | None] = []
    for stored, base in zip(dv.stored_files, dv.video_files):
        sub = stored.replace("\\", "/").rstrip("/").split("/")[-2:-1]          # the stored folder's name
        cands = [Path(stored)] if stored else []
        for p in places:
            cands += [p / sub[0] / base] if sub and sub[0] else []
            cands.append(p / base)
        hit = None
        for c in cands:
            try:
                if _key(c) not in taken and c.is_file():
                    hit = str(c)
                    break
            except OSError:
                continue
        if hit is not None:
            taken.add(_key(hit))
        out.append(hit)
    return out


def compare_3d(dv: DLTdvProject, rec, views: list[int], shifts: list[float], name_of=None) -> dict:
    """Kinetrace's 3D (a calib.Reconstruction, reference = camera views[0]'s frames) against DLTdv's stored 3D,
    on every row DLTdv has a 3D point for. Returns {"n", "median", "p95", "extent", "verdict", "hand", "missing"}
    or {"n": 0, "why": ...}. The verdict is relative to the size of the clicked volume (the units are the
    calibration's own): good <= 0.1 %, ok <= 1 %, poor beyond (the same math, so anything above good means the
    frames, pixels or lenses were read differently)."""
    has = np.isfinite(dv.xyz[..., 0])
    if not has.any():
        return {"n": 0, "why": "DLTdv stored no 3D points in this project, so there is nothing to compare"}
    if rec is None:
        return {"n": 0, "why": "Kinetrace made no 3D (fewer than two calibrated cameras)"}
    t_of_row = np.atleast_1d(half_up(dv.local_frames(views[0], shifts[0])))
    name_of = name_of or (lambda n: n)
    col = {n: j for j, n in enumerate(rec.names)}
    d, pairs, missing = [], [], 0
    for i, k in zip(*np.nonzero(has)):
        j = col.get(name_of(dv.point_names[k]))
        ti = int(t_of_row[i]) - rec.t0
        if j is None or not (0 <= ti < rec.xyz.shape[0]) or not np.isfinite(rec.xyz[ti, j, 0]):
            missing += 1
            continue
        d.append(float(np.linalg.norm(rec.xyz[ti, j] - dv.xyz[i, k])))
        pairs.append((int(t_of_row[i]), int(dv.rows[i]), dv.point_names[k], rec.xyz[ti, j], dv.xyz[i, k]))
    if not d:
        return {"n": 0, "missing": missing,
                "why": "no instant has both DLTdv's 3D and Kinetrace's (check the cameras' calibration)"}
    d = np.asarray(d)
    pts = dv.xyz[has]
    extent = float(np.max(np.ptp(pts, axis=0))) if len(pts) > 1 else 0.0
    med = float(np.median(d))
    rel = med / extent if extent > 0 else (0.0 if med == 0 else np.inf)
    verdict = "good" if rel <= 1e-3 else ("ok" if rel <= 1e-2 else "poor")
    t, r, nm, a, b = pairs[int(np.argsort(d)[len(d) // 2])]                # the median pair
    hand = (f"frame {t} of {dv.video_files[views[0]]} (DLTdv row {r + 1}), "
            f"point '{nm}': Kinetrace ({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}), "
            f"DLTdv ({b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f})")
    return {"n": len(d), "median": med, "p95": float(np.percentile(d, 95)), "extent": extent, "rel": rel,
            "verdict": verdict, "hand": hand, "missing": missing}
