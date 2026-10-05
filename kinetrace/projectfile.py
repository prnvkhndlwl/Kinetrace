"""The Kinetrace project: a FOLDER `name.kinetrace/` of plain files that other
programs can read and write (format 2, I145). See docs/FORMAT.md.

    kinetrace.json            format, version, app version, project id, saved at, cameras
    README.txt                what every file is (written by Kinetrace)
    project.json              per camera: name, video (path + path relative to the
                              project folder), frames, fps, size, offset, rate; the active camera
    state.json                window-wide toggles and layout (restored exactly)
    calibration.json          camera calibration (DLT coefficients + conventions), if any
    lenses.json               lens profiles per camera, if any
    reconstruction/meta.json + points/<landmark>.csv   the last 3D result, if any
    cameras/<folder>/view.json        this camera's frame, zoom, selection, timeline zoom
    cameras/<folder>/points.csv       one row per landmark (its tracks file in `file`, its animal in
                                      `animal`, "" = Scene; G153)
    cameras/<folder>/tracks/<landmark>.csv   frame,x,y,confidence,visible,hand_placed,hidden
                                      (+ radius for ball markers); only frames with data
    cameras/<folder>/events.csv, notes.csv, ball_prompts.json, spots.json, segment.json (the first
                                      animal: name, colour, clicks, its skeleton and hold)
    cameras/<folder>/silhouette/summary.csv + *.npy   per-frame area / centroid / box
    cameras/<folder>/segments/<name>/segment.json + silhouette/   every further animal (G149)
                                      (readable), outlines and midline (binary)
    cameras/<folder>/body/*.npy + meta.json  body poses and mesh (binary)
    exports/                  files for other programs, refreshed on save when asked (G42)
    .cache/ .history/         binary copies for fast opening / the previous save (not data)

A save writes only the files whose content changed (`write_folder`); opening
reads a table from `.cache/` when the CSV is still the one saved, and parses
the CSV when it was edited by hand, so the CSV is always what counts. The
same layout in a ZIP is the single-file form (`write`: File -> Export Project
as One File, and the recovery copies with binary tables); format-1 projects
(a zip with one long tracks.csv per camera) still open. Frames count from 0;
pixels are OpenCV pixel centres counted from 0.

No Qt here: `write` / `write_folder` run on a worker thread, `freeze` (cheap
copies) on the GUI thread first, so a save never sees data change under it.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
import uuid
import weakref
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from kinetrace import APP_VERSION
from kinetrace.recovery import safe_id  # noqa: F401 - the one project-id rule (R16); re-exported here

_log = logging.getLogger("kinetrace.errors")        # the error log keeps what is logged here (crashlog.py)

FORMAT = "kinetrace-project"
FORMAT_VERSION = 3              # 3 = animal layers (G153: points.csv `animal`, the skeleton per animal); 2 = a
#                                  folder, one CSV per landmark (I145); 1 = a zip, one tracks.csv per camera
SUFFIX = ".kinetrace"
META = "kinetrace.json"
CACHE_DIR, HISTORY_DIR, SAVING_DIR = ".cache", ".history", ".saving"
EXPORTS_DIR, VIDEOS_DIR = "exports", "videos"
_NOT_DATA = (CACHE_DIR, HISTORY_DIR, SAVING_DIR, EXPORTS_DIR, VIDEOS_DIR)   # never read as project data
# the ui_state entries kept per camera (view.json); every other one is window-wide (state.json)
# (G103) selected_names / segment_selected / segments_selected = the run scope: the points selected in LAYERS
# (BY NAME), whether an animal row is, and which animals (by name, G161); view.json keeps them as
# selected_points / segment_selected / selected_animals
VIEW_KEYS = ("selected", "selected_names", "segment_selected", "segments_selected", "zoom", "center_x", "center_y",
             "user_zoomed", "timeline", "hidden_animals")
# (G167, G154) "hidden_animals" = the animals whose LAYERS checkbox is off (silhouette not drawn): display state, so it
# lives in view.json (and the view sidecar), never makes the project dirty, and is not in segment.json
TRACK_COLS = ("frame", "point", "x", "y", "confidence", "visible", "hand_placed", "hidden", "radius")  # format 1
LANDMARK_COLS = ("frame", "x", "y", "confidence", "visible", "hand_placed", "hidden", "radius")
SILHOUETTE_COLS = ("frame", "area", "score", "centroid_x", "centroid_y", "x0", "y0", "x1", "y1")
POINT3D_COLS = ("frame", "x", "y", "z", "residual", "n_cams")         # + one <camera>_px column per camera
POINT_COLS = ("name", "color", "shown", "kind", "radius", "anchor", "source", "spec", "free", "shape", "outline",
              "file", "tracker", "animal")         # (G153) the animal it belongs to; "" = Scene
EVENT_COLS = ("name", "start", "end", "color", "note", "author")
NOTE_COLS = ("frame", "text", "author", "time")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)          # fixed: the same project always gives the same bytes
_CHUNK = 25_000                             # rows formatted per step (keeps the GUI thread responsive)


class ProjectFileError(Exception):
    """A project that cannot be opened; the message names the file (and row)."""


# ------------------------------------------------------------------ small codecs
def _float_text(a, dtype) -> np.ndarray:
    """Shortest text that reads back as exactly the same `dtype`; '' for NaN."""
    a = np.asarray(a, dtype)
    s = a.astype(str)
    s[np.isnan(a)] = ""
    return s


def f32_text(a: np.ndarray) -> np.ndarray:
    """Shortest text that reads back as exactly the same float32; '' for NaN."""
    return _float_text(a, np.float32)


def text_f32(col, where: str) -> np.ndarray:
    """Inverse of f32_text; exact for text written by it, correctly rounded
    for anything typed by hand. '' (or 'nan') = NaN."""
    col = col if isinstance(col, (list, tuple)) else list(col)
    try:
        d = np.fromiter(map(float, col), np.float64, len(col))    # fast path: no blank cells
        blank = np.zeros(len(col), bool)
    except ValueError:
        blank = np.fromiter((not c.strip() for c in col), bool, len(col))
        try:
            d = np.fromiter((float(c) if c.strip() else np.nan for c in col), np.float64, len(col))
        except ValueError as e:
            raise ProjectFileError(f"{where}: a value is not a number ({e})") from None
    v = d.astype(np.float32)
    # Parsing through float64 can double-round, but only when the float64
    # lands (almost) exactly halfway between two float32s. Those rare cells
    # are resolved against the text itself; everything else is already the
    # correctly rounded float32.
    with np.errstate(invalid="ignore"):
        half = np.spacing(np.abs(v)).astype(np.float64) / 2
        near_half = ~blank & (np.abs(np.abs(d - v.astype(np.float64)) - half) <= half * 1e-6)
    for i in np.flatnonzero(near_half):
        for cand in (np.nextafter(v[i], np.float32(np.inf)), np.nextafter(v[i], np.float32(-np.inf))):
            if str(cand) == col[i].strip():
                v[i] = cand
                break
    return v


def _bool_col(col, where: str) -> np.ndarray:
    joined = "".join(col)
    if len(joined) == len(col) and set(joined) <= {"0", "1"}:   # fast path: our own 0 / 1 columns
        return np.frombuffer(joined.encode("ascii"), np.uint8) == ord("1")
    u, inv = np.unique(np.asarray(col, dtype=str), return_inverse=True)   # a handful of distinct values
    val = []
    for x in u:
        t = x.strip().lower()
        if t not in ("0", "1", "true", "false", ""):
            raise ProjectFileError(f"{where}: expected 0 / 1, found {x!r}")
        val.append(t in ("1", "true"))
    return np.asarray(val, bool)[inv]


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def _rgb(s: str, where: str) -> tuple[int, int, int]:
    s = (s or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise ProjectFileError(f"{where}: colour {s!r} is not #rrggbb")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _csv_text(header, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n"


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON: {type(o).__name__}")


def _clean_json(obj):
    """NaN / inf are not JSON: write them as null."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean_json(obj.item())
    if isinstance(obj, np.ndarray):
        return _clean_json(obj.tolist())
    return obj


_FOLDER_CHARS = "A-Za-z0-9._-"
CAMERA_FOLDER = re.compile(f"[{_FOLDER_CHARS}]+")          # (R16) the one camera-folder rule: read, layout check, autoexport


def is_camera_folder(name) -> bool:
    """A camera folder name a project may hold: [A-Za-z0-9._-], never hidden."""
    return isinstance(name, str) and CAMERA_FOLDER.fullmatch(name) is not None and not name.startswith(".")


def _key(name: str) -> str:
    """A file name as the matching rule sees it: composed (NFC) and case-folded,
    so a decomposed name (HFS+) or a case-only difference (NTFS / macOS) is the
    same file (I163, I211)."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", str(name)).casefold())


def camera_folders(names: list[str]) -> list[str]:
    """Zip-safe, case-insensitively unique folder names ([A-Za-z0-9._-])."""
    out, seen = [], set()
    for i, n in enumerate(names):
        base = re.sub(f"[^{_FOLDER_CHARS}]+", "_", unicodedata.normalize("NFKD", n)).strip("._") or f"cam{i + 1}"
        f, k = base, 2
        while f.lower() in seen:
            f, k = f"{base}-{k}", k + 1
        seen.add(f.lower())
        out.append(f)
    return out


_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def landmark_files(names: list[str]) -> list[str]:
    """One file name per landmark (`snout.csv`): the name itself where every
    OS allows it (letters of any language kept), characters Windows forbids
    as `_`, no leading dot, no reserved device name, unique ignoring case."""
    out, seen = [], set()
    for i, n in enumerate(names):
        base = _FORBIDDEN.sub("_", unicodedata.normalize("NFC", str(n))).strip().strip(".").strip()[:80]
        if not base:
            base = f"point{i + 1}"
        if base.split(".")[0].upper() in _RESERVED:
            base += "_"
        f, k = base, 2
        while f.casefold() in seen:
            f, k = f"{base}-{k}", k + 1
        seen.add(f.casefold())
        out.append(f + ".csv")
    return out


def f64_text(a: np.ndarray) -> np.ndarray:
    """Shortest text that reads back as exactly the same float64; '' for NaN."""
    return _float_text(a, np.float64)


def _codes(c: np.ndarray) -> np.ndarray:
    """A column as (n, width) code points, 0 = padding (dropped by _join_rows)."""
    if c.ndim == 2 and c.dtype == np.uint32:
        return c                                  # already code points (_int_codes, _bool_codes)
    c = np.ascontiguousarray(c if c.dtype.kind == "U" else c.astype(str))
    w = c.dtype.itemsize // 4
    return c.view(np.uint32).reshape(len(c), w) if w else np.zeros((len(c), 0), np.uint32)


def _int_codes(a: np.ndarray) -> np.ndarray:
    """Whole numbers as code points without going through Python strings:
    the same text as str(int), digits right-aligned behind 0-padding."""
    a = np.asarray(a, np.int64)
    v = np.abs(a)
    w = len(str(int(v.max()))) if v.size else 1
    p10 = 10 ** np.arange(w - 1, -1, -1, dtype=np.int64)
    lead = v[:, None] < p10
    lead[:, -1] = False                           # "0" itself has one digit
    codes = np.where(lead, 0, (v[:, None] // p10) % 10 + 48).astype(np.uint32)
    if (a < 0).any():
        codes = np.concatenate([np.where(a < 0, 45, 0).astype(np.uint32)[:, None], codes], axis=1)
    return codes


def _bool_codes(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, bool).astype(np.uint32) + 48)[:, None]


def _join_rows(cols: list[np.ndarray]) -> str:
    """Rows of comma-separated cells from equally long columns (strings or
    code points), built as ONE array of code points: a Python join per row
    cost 0.77 us a row; this is the same text in a fraction of that."""
    n = len(cols[0]) if cols else 0
    if n == 0:
        return ""
    parts, last = [], len(cols) - 1
    for k, c in enumerate(cols):
        parts.append(_codes(c))
        parts.append(np.full((n, 1), 10 if k == last else 44, np.uint32))    # "\n" / ","
    m = np.concatenate(parts, axis=1).ravel()
    m = m[m != 0]                            # the padding of the shorter cells
    if not m.size or int(m.max()) < 128:
        return m.astype(np.uint8).tobytes().decode("ascii")
    return m.astype("<u4").tobytes().decode("utf-32-le")


def _digest(*parts) -> str:
    """Fingerprint of a file's content, taken from the data it is written
    from (a save skips a file whose fingerprint and file on disk are unchanged)."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, str):
            h.update(b"s" + p.encode("utf-8"))
        else:
            a = np.ascontiguousarray(p)
            h.update(f"a{a.dtype.str}{a.shape}".encode())
            if a.size:                           # (an empty array has no bytes to add, and cannot be cast)
                h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


class Table:
    """A long table: named 1-D columns of one length. `kinds` per column:
    "i" whole number, "f4" / "f8" float32 / float64 (blank = NaN, written as
    the shortest exact text), "b" 0 / 1. `written` = the columns the CSV
    carries (a column left out reads back as its default)."""

    __slots__ = ("cols", "kinds", "data", "written")

    def __init__(self, cols, kinds, data: dict, written=None):
        self.cols, self.kinds, self.data = tuple(cols), tuple(kinds), data
        self.written = tuple(written or cols)

    @property
    def n(self) -> int:
        return len(self.data[self.cols[0]])

    def digest(self) -> str:
        return _digest(",".join(self.written), *(self.data[c] for c in self.written))

    def _text_col(self, c: str, a: np.ndarray) -> np.ndarray:
        k = self.kinds[self.cols.index(c)]
        if k == "f4":
            return f32_text(a)
        if k == "f8":
            return f64_text(a)
        if k == "b":
            return _bool_codes(a)
        return _int_codes(a)

    def csv(self, yield_gil=lambda: None) -> str:
        out = [",".join(self.written) + "\n"]
        for a in range(0, self.n, _CHUNK):
            out.append(_join_rows([self._text_col(c, self.data[c][a:a + _CHUNK]) for c in self.written]))
            yield_gil()
        return "".join(out)

    def struct(self) -> np.ndarray:
        """The binary copy kept in .cache/ (every column, written or not)."""
        dt = [(c, {"i": "<i8", "f4": "<f4", "f8": "<f8", "b": "?"}[k]) for c, k in zip(self.cols, self.kinds)]
        s = np.empty(self.n, dt)
        for c in self.cols:
            s[c] = self.data[c]
        return s

    @classmethod
    def from_struct(cls, s: np.ndarray, cols, kinds) -> "Table | None":
        if s.dtype.names is None or tuple(s.dtype.names) != tuple(cols):
            return None
        return cls(cols, kinds, {c: np.ascontiguousarray(s[c]) for c in cols})


# ------------------------------------------------------------------ freeze (GUI thread)
@dataclass
class Frozen:
    """Everything a save writes, detached from the live objects."""
    files: dict = field(default_factory=dict)      # zip member -> str (text) or np.ndarray (npy)


def _point_row(p, file: str) -> list:
    """One landmark's points.csv row, in POINT_COLS order (`_read_point` is its
    inverse: the columns are declared once, here and there)."""
    return [p.name, _hex(p.color), int(p.display), p.kind, repr(float(p.radius)), int(p.anchor), p.source,
            p.spec, int(p.free), p.shape,
            "" if not p.outline else " ".join(repr(float(v)) for xy in p.outline for v in xy), file, p.tracker,
            p.segment]


def _put_arrays(files: dict, folder: str, arrays: dict, prefix: str) -> None:
    """A `to_arrays` dict -> folder/<key>.npy for arrays, folder/<key>.json for
    the JSON strings. The same keys come back for `from_arrays`."""
    for key, val in arrays.items():
        name = key[len(prefix):] if key.startswith(prefix) else key
        if isinstance(val, str):
            files[f"{folder}/{name}.json"] = val if val.endswith("\n") else val + "\n"
        else:
            files[f"{folder}/{name}.npy"] = np.array(val, copy=True)


def freeze(project, state: dict, project_id: str, *, target: str | Path | None = None,
           binary_tracks: bool = False, saved_at: str | None = None, layout: str = "folder") -> Frozen:
    """Copies of everything to save. Cheap: array copies and small lists;
    formatting happens later, in `write` / `write_folder`, off the GUI thread.
    `target` = the project being written; `layout` "folder" (a project folder:
    video paths relative to the folder itself) or "zip" (one file: relative to
    the folder containing it). `binary_tracks` = the recovery copy's binary
    tables (fast, not meant to be read by people)."""
    fz = Frozen()
    files = fz.files
    folders = camera_folders(project.names)
    saved_at = saved_at or timestamp()
    files[META] = _json({
        "format": FORMAT, "format_version": FORMAT_VERSION, "app_version": APP_VERSION,
        "project_id": project_id, "saved_at": saved_at,
        "cameras": [{"folder": f, "name": n} for f, n in zip(folders, project.names)],
        "videos_relative_to": "project" if layout == "folder" else "container",
        "conventions": {"frames": "counted from 0",
                        "pixels": "OpenCV pixel centres, counted from 0 (top-left pixel centre = 0, 0)",
                        "blank": "no data"}})
    cams = _freeze_cameras(project, folders, target or getattr(project, "path", None), layout)
    files["project.json"] = _json({"cameras": cams, "active_camera": project.names[project.active],
                                   "exports_on_save": list(getattr(project, "exports", []) or [])})   # (G42)
    files["state.json"] = _json(_clean_json(state))
    if project.calibration is not None and len(project.calibration) > 0:
        arr = project.calibration.to_arrays("calib_")
        cal = json.loads(arr["calib_meta"])
        cal["coefs"] = np.asarray(arr["calib_coefs"], np.float64).tolist()
        cal["notes"] = list(getattr(project.calibration, "notes", []) or [])
        files["calibration.json"] = _json(_clean_json(cal))
    if any(l is not None for l in project.lenses):
        files["lenses.json"] = _json(_clean_json([None if l is None else l.to_json() for l in project.lenses]))
    _freeze_reconstruction(files, project.reconstruction, folders, binary_tracks)
    for i, s in enumerate(project.sessions):
        _freeze_camera(files, f"cameras/{folders[i]}", s, binary_tracks)
    if not binary_tracks:
        files["README.txt"] = _readme(project.names, folders)
    return fz


def _freeze_cameras(project, folders: list[str], where, layout: str) -> list[dict]:
    """project.json's camera entries: video path (absolute and relative to the
    project), size, rate, offset."""
    base = (Path(where).resolve() if layout == "folder" else Path(where).resolve().parent) if where else None
    cams = []
    for i, s in enumerate(project.sessions):
        rel = None
        if base is not None:
            try:
                rel = PurePosixPath(Path(os.path.relpath(s.video_path, base)).as_posix()).as_posix()
            except ValueError:                       # another drive on Windows
                rel = None
        cams.append({"name": project.names[i], "folder": folders[i],
                     "video": {"path": str(s.video_path), "relative_path": rel},
                     "n_frames": int(s.n_frames), "fps": float(s.fps),
                     "file_fps": float(getattr(s, "file_fps", s.fps)),   # the file's own rate (G38)
                     "width": int(s.width), "height": int(s.height),
                     "offset": float(project.offsets[i]), "rate": float(project.rates[i])})
    return cams


def _freeze_reconstruction(files: dict, r, folders: list[str], binary_tracks: bool) -> None:
    """The 3D result: binary arrays in a recovery copy, else one readable table per landmark (I145)."""
    if r is not None and binary_tracks:
        rec = {"xyz": r.xyz.astype(np.float64), "residual": r.residual.astype(np.float64),
               "n_cams": r.n_cams.astype(np.int32),
               "meta": json.dumps({"t0": int(r.t0), "names": list(r.names), "unit": str(r.unit)})}
        if r.per_cam is not None:
            rec["per_cam"] = np.asarray(r.per_cam, np.float32)
        _put_arrays(files, "reconstruction", rec, "")
    elif r is not None:
        # one readable table per landmark, rows in REFERENCE frames (I145)
        arrays = {"xyz": r.xyz.astype(np.float64), "residual": r.residual.astype(np.float64),
                  "n_cams": r.n_cams.astype(np.int64),
                  "per_cam": None if r.per_cam is None else np.asarray(r.per_cam, np.float32)}
        C = 0 if r.per_cam is None else int(np.asarray(r.per_cam).shape[2])
        cams_px = [f"{f}_px" for f in (folders if C == len(folders) else [f"cam{k + 1}" for k in range(C)])]
        rfiles = landmark_files(list(r.names))
        files["reconstruction/meta.json"] = _json({
            "t0": int(r.t0), "n_frames": int(r.xyz.shape[0]), "unit": str(r.unit),
            "points": [{"name": n, "file": f"points/{f}"} for n, f in zip(r.names, rfiles)],
            "per_camera_columns": cams_px,
            "frames": "reference camera's frames (the first camera's)"})
        for j, f in enumerate(rfiles):
            files[f"reconstruction/points/{f}"] = ("p3", arrays, j, int(r.t0), cams_px)


def _freeze_camera(files: dict, d: str, s, binary_tracks: bool) -> None:
    st = s.ui_state
    sel = int(st.get("selected", -1))
    scope = {}                                  # the run scope (G103), only when the app has recorded one
    if "selected_names" in st:
        scope["selected_points"] = [str(x) for x in (st.get("selected_names") or [])]
    if "segment_selected" in st:
        scope["segment_selected"] = bool(st.get("segment_selected"))
    if isinstance(st.get("segments_selected"), list):           # (G161) WHICH animals, by name
        scope["selected_animals"] = [str(x) for x in st.get("segments_selected") or []]
    files[f"{d}/view.json"] = _json(_clean_json({
        "current_frame": int(s.current_frame),
        "selected_point": s.points[sel].name if 0 <= sel < s.n_points else None,
        **scope,
        "zoom": st.get("zoom", 0.0), "center_x": st.get("center_x", 0.0), "center_y": st.get("center_y", 0.0),
        "user_zoomed": bool(st.get("user_zoomed", False)),
        "timeline": st.get("timeline"),
        "hidden_animals": [a.name for a in s.segments if not a.shown],      # (G154) display state
        "annotator": s.annotator, "counters": {"points": s._name_counter, "events": s._event_counter}}))
    lfiles = landmark_files([p.name for p in s.points])
    files[f"{d}/points.csv"] = ("points", [_point_row(p, "" if binary_tracks else lfiles[j])
                                           for j, p in enumerate(s.points)])
    from kinetrace.session import POINT_ARRAYS
    arrays = {a.name: getattr(s, a.name).copy() for a in POINT_ARRAYS}       # (R15) the one table
    if binary_tracks:
        for k, v in arrays.items():
            files[f"{d}/{k}.npy"] = v
    else:
        for j, p in enumerate(s.points):        # one readable table per landmark (I145)
            files[f"{d}/tracks/{lfiles[j]}"] = ("lm", arrays, j, bool(p.is_ball))
    files[f"{d}/events.csv"] = ("events", [[e.name, e.start, e.end, _hex(e.color), e.note, e.author]
                                           for e in s.events])
    files[f"{d}/notes.csv"] = ("notes", [[f, n.get("text", ""), n.get("author", ""), n.get("time", "")]
                                         for f, n in sorted(s.notes.items())])
    balls = {p.name: {str(f): cs for f, cs in (p.ball_prompts or {}).items()} for p in s.points if p.is_ball}
    if balls:
        files[f"{d}/ball_prompts.json"] = _json(_clean_json(balls))
    spot = {p.name: dict(p.spot) for p in s.points if p.spot}
    if spot:            # the Moving spot settings the test found (I161)
        files[f"{d}/spots.json"] = _json(_clean_json(spot))
    # (G153) no camera-level skeleton.json: each animal's skeleton and hold are in its segment.json
    # (G149) the first segment where a one-segment project always had it; every further one in
    # segments/<its name>/ with the same two parts (its order in segment.json)
    extra = landmark_files([a.name for a in s.segments[1:]])
    for k, (a, mt) in enumerate(zip(s.segments, s.seg_masks)):
        base = d if k == 0 else f"{d}/segments/{extra[k - 1][:-4]}"
        meta = json.loads(a.to_json())
        if k:
            meta["order"] = k
        files[f"{base}/segment.json"] = json.dumps(meta) + "\n"
        m = mt.to_arrays("mask_")
        if not binary_tracks:               # the per-frame numbers as a readable table (I145)
            summ = {kk: np.array(m.pop(f"mask_{kk}"), copy=True) for kk in ("bbox", "area", "centroid", "score")}
            files[f"{base}/silhouette/summary.csv"] = ("sil", summ)
        _put_arrays(files, f"{base}/silhouette", m, "mask_")
    if s.body is not None:
        _put_arrays(files, f"{d}/body", s.body.to_arrays("body_", mesh=False), "body_")
        _put_mesh(files, f"{d}/body", s.body)


_MESH_BLOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()     # BodyTrack -> (key, {name: entry})


def _put_mesh(files: dict, folder: str, body) -> None:
    """The body's mesh block (`mesh_*.npy`): joined and hashed once, then reused
    for as long as `mesh_version` says the mesh did not change -- it used to be
    joined, copied and hashed at every save and every 30 s autosave (R17: 110 ms
    and 180 MB per 1000 meshed frames on the GUI thread). The arrays are never
    written in place, so a frozen project and the cache share them."""
    key = (body.mesh_version, len(body.mesh), None if body.faces is None else id(body.faces))
    hit = _MESH_BLOCKS.get(body)
    if hit is None or hit[0] != key:
        block = {}
        for name, arr in body.mesh_arrays("").items():
            arr.flags.writeable = False
            block[name] = ("npy", arr, [None])           # [digest], filled once by _materialize
        hit = (key, block)
        _MESH_BLOCKS[body] = hit
    for name, entry in hit[1].items():
        files[f"{folder}/{name}.npy"] = entry


# ------------------------------------------------------------------ tables (worker thread)
_LM_KINDS = ("i", "f4", "f4", "f4", "b", "b", "b", "f4")
_SIL_KINDS = ("i", "i", "f4", "f4", "f4", "i", "i", "i", "i")


def _landmark_table(arrays: dict, j: int, is_ball: bool) -> Table:
    """One landmark's rows: every frame whose cell is not all-default (what
    a read would give back for a frame without a row)."""
    tr, cf, vi = arrays["tracks"][:, j], arrays["confidence"][:, j], arrays["visibility"][:, j]
    ma, tk, oc, ra = arrays["manual"][:, j], arrays["tracked"][:, j], arrays["occluded"][:, j], arrays["radius"][:, j]
    f = np.flatnonzero(tk | ma | oc | vi | np.isfinite(ra) | (cf != 0))
    data = {"frame": f.astype(np.int64), "x": tr[f, 0], "y": tr[f, 1], "confidence": cf[f],
            "visible": vi[f], "hand_placed": ma[f], "hidden": oc[f], "radius": ra[f]}
    with_r = is_ball or bool(np.isfinite(data["radius"]).any())       # radius: ball markers only
    return Table(LANDMARK_COLS, _LM_KINDS, data, LANDMARK_COLS if with_r else LANDMARK_COLS[:-1])


def _silhouette_table(summ: dict) -> Table:
    bb, ar, ce, sc = summ["bbox"], summ["area"], summ["centroid"], summ["score"]
    f = np.flatnonzero((ar != 0) | (bb != -1).any(1) | np.isfinite(ce).any(1) | np.isfinite(sc))
    data = {"frame": f.astype(np.int64), "area": ar[f].astype(np.int64), "score": sc[f].astype(np.float32),
            "centroid_x": ce[f, 0].astype(np.float32), "centroid_y": ce[f, 1].astype(np.float32),
            **{c: bb[f, k].astype(np.int64) for k, c in enumerate(("x0", "y0", "x1", "y1"))}}
    return Table(SILHOUETTE_COLS, _SIL_KINDS, data)


def _point3d_table(arrays: dict, j: int, t0: int, cams_px: list[str]) -> Table:
    xyz, res, nc, pc = arrays["xyz"][:, j], arrays["residual"][:, j], arrays["n_cams"][:, j], arrays["per_cam"]
    keep = (nc != 0) | np.isfinite(res) | np.isfinite(xyz).any(1)
    if pc is not None:
        pc = pc[:, j, :]
        keep |= np.isfinite(pc).any(1)
    f = np.flatnonzero(keep)
    data = {"frame": (f + int(t0)).astype(np.int64), "x": xyz[f, 0], "y": xyz[f, 1], "z": xyz[f, 2],
            "residual": res[f], "n_cams": nc[f].astype(np.int64)}
    for k, c in enumerate(cams_px):
        data[c] = pc[f, k] if pc is not None else np.full(len(f), np.nan, np.float32)
    return Table(POINT3D_COLS + tuple(cams_px), ("i", "f8", "f8", "f8", "f8", "i") + ("f4",) * len(cams_px), data)


def _int_col(col, where: str, what: str) -> np.ndarray:
    try:
        return np.fromiter(map(int, col), np.int64, len(col))
    except ValueError:
        raise ProjectFileError(f"{where}: a {what} is not a whole number") from None


def _parse_landmark(text: str, where: str) -> Table:
    cols, n = parse_table(text, where, ("frame", "x", "y"), LANDMARK_COLS[3:])
    x, y = text_f32(cols["x"], where), text_f32(cols["y"], where)
    has = np.isfinite(x) & np.isfinite(y)
    opt = lambda c, conv, default: conv(cols[c], where) if cols.get(c) is not None else default  # noqa: E731
    data = {"frame": _int_col(cols["frame"], where, "frame number"), "x": x, "y": y,
            "confidence": opt("confidence", text_f32, np.where(has, 1.0, 0.0).astype(np.float32)),
            "visible": opt("visible", _bool_col, has.copy()),
            "hand_placed": opt("hand_placed", _bool_col, np.zeros(n, bool)),
            "hidden": opt("hidden", _bool_col, np.zeros(n, bool)),
            "radius": opt("radius", text_f32, np.full(n, np.nan, np.float32))}
    return Table(LANDMARK_COLS, _LM_KINDS, data,
                 [c for c in LANDMARK_COLS if c in ("frame", "x", "y") or cols.get(c) is not None])


def _parse_silhouette(text: str, where: str) -> Table:
    cols, n = parse_table(text, where, ("frame",), SILHOUETTE_COLS[1:])
    ints = lambda c, d: (_int_col([v or str(d) for v in cols[c]], where, c)       # noqa: E731
                         if cols.get(c) is not None else np.full(n, d, np.int64))
    f32 = lambda c: text_f32(cols[c], where) if cols.get(c) is not None else np.full(n, np.nan, np.float32)  # noqa: E731
    data = {"frame": _int_col(cols["frame"], where, "frame number"), "area": ints("area", 0), "score": f32("score"),
            "centroid_x": f32("centroid_x"), "centroid_y": f32("centroid_y"),
            **{c: ints(c, -1) for c in ("x0", "y0", "x1", "y1")}}
    return Table(SILHOUETTE_COLS, _SIL_KINDS, data)


def _point3d_parser(cams_px: list[str]):
    def parse(text: str, where: str) -> Table:
        cols, n = parse_table(text, where, ("frame",), POINT3D_COLS[1:] + tuple(c.lower() for c in cams_px))
        f64 = lambda c: (np.array([float(v) if v.strip() else np.nan for v in cols[c]], np.float64)   # noqa: E731
                         if cols.get(c) is not None else np.full(n, np.nan))
        try:
            data = {"frame": _int_col(cols["frame"], where, "frame number"), "x": f64("x"), "y": f64("y"),
                    "z": f64("z"), "residual": f64("residual"),
                    "n_cams": (_int_col([v or "0" for v in cols["n_cams"]], where, "camera count")
                               if cols.get("n_cams") is not None else np.zeros(n, np.int64))}
        except ValueError as e:
            raise ProjectFileError(f"{where}: a value is not a number ({e})") from None
        for c in cams_px:
            v = cols.get(c.lower())
            data[c] = text_f32(v, where) if v is not None else np.full(n, np.nan, np.float32)
        return Table(POINT3D_COLS + tuple(cams_px), ("i", "f8", "f8", "f8", "f8", "i") + ("f4",) * len(cams_px), data)
    return parse


def _check_frames(fr: np.ndarray, n: int, where: str, first: int = 0) -> None:
    bad = (fr < first) | (fr >= first + n)
    if bad.any():
        raise ProjectFileError(f"{where}: row {int(np.flatnonzero(bad)[0]) + 2}: frame {fr[bad][0]} "
                               f"is outside the video ({first} - {first + n - 1})")
    if fr.size > 1 and not (np.diff(fr) > 0).all() and np.unique(fr).size != fr.size:   # ours are ascending
        raise ProjectFileError(f"{where}: the same frame appears twice")


def _fill_landmark(arr: dict, j: int, t: Table, T: int, where: str) -> None:
    d = t.data
    fr = d["frame"]
    _check_frames(fr, T, where)
    arr["tracks"][fr, j, 0], arr["tracks"][fr, j, 1] = d["x"], d["y"]
    arr["tracked"][fr, j] = np.isfinite(d["x"]) & np.isfinite(d["y"])
    arr["confidence"][fr, j] = d["confidence"]
    arr["visibility"][fr, j] = d["visible"]
    arr["manual"][fr, j] = d["hand_placed"]
    arr["occluded"][fr, j] = d["hidden"]
    arr["radius"][fr, j] = d["radius"]


def _empty_tracks(T: int, N: int) -> dict:
    from kinetrace.session import POINT_ARRAYS
    return {a.name: a.empty(T, N) for a in POINT_ARRAYS}                     # (R15)


def _materialize(item, need_digest: bool = True):
    """A frozen entry -> (content, digest): content is text, an array (.npy)
    or a Table (CSV, formatted only if it is written)."""
    if isinstance(item, tuple):
        kind = item[0]
        if kind == "npy":                       # an array whose fingerprint is kept with it (R17)
            memo = item[2]
            if need_digest and memo[0] is None:
                memo[0] = _digest(item[1])
            return item[1], (memo[0] or "") if need_digest else ""
        if kind == "lm":
            t = _landmark_table(*item[1:])
        elif kind == "sil":
            t = _silhouette_table(item[1])
        elif kind == "p3":
            t = _point3d_table(*item[1:])
        else:
            header = {"points": POINT_COLS, "events": EVENT_COLS, "notes": NOTE_COLS}[kind]
            text = _csv_text(header, item[1])
            return text, (_digest(text) if need_digest else "")
        return t, (t.digest() if need_digest else "")
    return item, (_digest(item) if need_digest else "")


# ------------------------------------------------------------------ write (worker thread)
def write(frozen: Frozen, path: str | Path, *, compresslevel: int = 1, fsync: bool = True,
          backup: bool = True, yield_gil=lambda: time.sleep(0)) -> None:
    """The single-file form (a ZIP of the same layout: File -> Export Project
    as One File, and the recovery copies). Written atomically: a temp file in
    the same folder, optionally a copy of the previous file as `<name>.bak`,
    then one os.replace. The file at `path` is always either the old save or
    the new one, never half of either."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            with zipfile.ZipFile(fh, "w", allowZip64=True) as z:
                for name in sorted(frozen.files):
                    data, _ = _materialize(frozen.files[name], need_digest=False)
                    if isinstance(data, Table):
                        data = data.csv(yield_gil)
                    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
                    info.external_attr = 0o644 << 16
                    if isinstance(data, np.ndarray):
                        info.compress_type = zipfile.ZIP_STORED          # float data barely compresses
                        with z.open(info, "w", force_zip64=True) as m:
                            np.lib.format.write_array(m, data, allow_pickle=False)
                    else:
                        info.compress_type = zipfile.ZIP_DEFLATED if compresslevel > 0 else zipfile.ZIP_STORED
                        z.writestr(info, data.encode("utf-8"), compresslevel=compresslevel or None)
                    yield_gil()
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        if backup and path.exists():
            bak_tmp = path.with_name(path.name + ".bak.tmp")
            shutil.copy2(path, bak_tmp)
            _replace(bak_tmp, path.with_name(path.name + ".bak"))
        _replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _replace(src: Path, dst: Path) -> None:
    for attempt in range(20):                # Windows: antivirus / OneDrive hold files briefly
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


# ------------------------------------------------------------------ the project folder (I145)
_TOP_FILES = {META, "project.json", "state.json", "calibration.json", "lenses.json", "README.txt"}
_CAM_FILES = {"view.json", "points.csv", "tracks.csv", "events.csv", "notes.csv", "ball_prompts.json", "spots.json",
              "skeleton.json", "segment.json", "tracks.npy", "confidence.npy", "visibility.npy", "manual.npy",
              "tracked.npy", "occluded.npy", "radius.npy"}
_CAM_DIRS = ("tracks", "silhouette", "body", "segments")
# files an older layout left in a camera folder (format 1's tracks.csv, the recovery form's dense arrays): a save
# removes them if they are there, besides what the previous save wrote (I164)
_LEGACY_CAM_FILES = ("skeleton.json", "tracks.csv", "tracks.npy", "confidence.npy", "visibility.npy", "manual.npy", "tracked.npy",
                     "occluded.npy", "radius.npy")
_OWN_DOT = (CACHE_DIR, HISTORY_DIR, SAVING_DIR, ".lock")      # the dot entries Kinetrace itself makes


def project_root(path: str | Path) -> Path:
    """The project a path names: a folder (also given as its kinetrace.json) or a single file."""
    p = Path(path)
    return p.parent if p.name.lower() == META and p.is_file() else p


def is_project(path: str | Path) -> bool:
    root = project_root(path)
    if root.is_dir():
        return (root / META).is_file()
    return root.is_file() and zipfile.is_zipfile(root)


def is_folder_project(path: str | Path) -> bool:
    root = project_root(path)
    return root.is_dir() and (root / META).is_file()


def read_meta(path: str | Path) -> dict:
    """kinetrace.json alone (which save a project is, without reading it)."""
    root = project_root(path)
    try:
        if root.is_dir():
            return json.loads((root / META).read_text(encoding="utf-8-sig"))
        with zipfile.ZipFile(root) as z:
            return json.loads(z.read(META).decode("utf-8-sig"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
        raise ProjectFileError(f"{root.name}: kinetrace.json cannot be read ({e})") from None


def video_base(path: str | Path, meta: dict) -> Path:
    """The folder a project's `relative_path`s start from: the project folder
    itself (format 2 folders) or the folder containing the project (a single
    file, and every format-1 project -- an unzipped one too)."""
    root = project_root(path).resolve()
    rel_to = meta.get("videos_relative_to") or ("project" if int(meta.get("format_version") or 1) >= 2
                                                 and root.is_dir() else "container")
    return root if rel_to == "project" and root.is_dir() else root.parent


def _csvpool():
    from kinetrace import csvpool
    return csvpool


def _hide(p: Path) -> None:
    """.cache / .history / .saving are the program's own: hidden in Explorer."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x02)
        except Exception:        # noqa: BLE001 - cosmetic
            pass


def _write_bytes(p: Path, data: bytes, fsync: bool) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data)
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def _fsync_dir(d: Path) -> None:
    """Make the renames in `d` durable (POSIX: a rename is on disk only once
    its directory is; Windows has no directory fsync and needs none)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


LOCK = ".lock"
# the largest video a project may describe: every T x N array is sized from it (I157)
MAX_FRAMES = 20_000_000                 # 18 h at 300 fps
MAX_SIDE = 65_536
LOCK_STALE_S = 6 * 3600                  # a lock this old is a crash's leftover, whoever wrote it


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259                 # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _proc_start(pid: int) -> str | None:
    """A token that differs when `pid` is another process than it was: the
    process's start time as the OS keeps it (a pid reused after a reboot is not
    the process that took the lock, I246). None when it cannot be told."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
                if not k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
                    return None
                return str((c.dwHighDateTime << 32) | c.dwLowDateTime)
            finally:
                k.CloseHandle(h)
        if sys.platform.startswith("linux"):
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
            return stat.rsplit(")", 1)[1].split()[19]             # field 22: start time, ticks since boot
        import subprocess
        out = subprocess.run(["ps", "-p", str(int(pid)), "-o", "lstart="], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        return out or None
    except Exception:        # noqa: BLE001 - a courtesy check: unknown = fall back on "is it alive"
        return None


def _read_lock(lk: Path) -> dict | None:
    """The .lock's content, None when it is not one Kinetrace wrote."""
    try:
        other = json.loads(lk.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return other if isinstance(other, dict) else None


def _lock_stale(other: dict | None) -> bool:
    """True for a lock whose writer is gone: not a lock Kinetrace wrote (I157),
    hours old, its process dead (on this computer), or its pid now another
    process (the start time it recorded no longer matches, I246)."""
    import socket
    if other is None:
        return True
    try:
        if time.time() - float(other.get("time") or 0) > LOCK_STALE_S:
            return True
        if other.get("host") != socket.gethostname():
            return False
        pid = int(other.get("pid") or 0)
        if not _pid_alive(pid):
            return True
        start = other.get("start")
        if isinstance(start, str) and start:
            now = _proc_start(pid)
            return now is not None and now != start
        return False
    except (ValueError, TypeError, AttributeError, OverflowError):
        return True


def clear_stale_lock(root: str | Path) -> dict | None:
    """A save that crashed left its .lock: removed when its process is gone.
    -> the lock of a save that is still running (a live process, a fresh
    lock), else None."""
    lk = Path(root) / LOCK
    if not lk.is_file():
        return None
    other = _read_lock(lk)
    if _lock_stale(other):
        lk.unlink(missing_ok=True)
        return None
    return other


def _take_lock(root: Path) -> Path:
    """One save at a time per project folder: `.lock` created exclusively
    (process id, its start time, computer, time). Two Kinetrace windows saving
    the same project would otherwise clear each other's .saving / .history. A
    lock whose process is gone (on this computer) or that is hours old is taken over."""
    import socket
    lk = root / LOCK
    me = {"pid": os.getpid(), "host": socket.gethostname(), "time": time.time(), "start": _proc_start(os.getpid())}
    for _attempt in range(3):
        try:
            fd = os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            other = _read_lock(lk)
            if _lock_stale(other):
                lk.unlink(missing_ok=True)          # a crash's leftover
                continue
            other = other or {}
            raise ProjectFileError(
                f"{root.name} is being saved by another Kinetrace right now (process {other.get('pid')} on "
                f"{other.get('host')}). Close the other window (or wait for its save), then save again.") from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(me, fh)
        _hide(lk)
        return lk
    raise ProjectFileError(f"{root.name}: could not take the save lock ({lk})")


def _free_bytes(root: Path) -> int:
    try:
        return int(shutil.disk_usage(root).free)
    except OSError:
        return 1 << 62                      # unknown: do not refuse


def _write_json_atomic(p: Path, obj, fsync: bool = True) -> None:
    tmp = p.with_name(p.name + ".tmp")
    _write_bytes(tmp, json.dumps(obj, indent=1).encode("utf-8"), fsync)
    _replace(tmp, p)


def _saved_at(root: Path) -> str | None:
    try:
        return json.loads((root / META).read_text(encoding="utf-8-sig")).get("saved_at")
    except (OSError, ValueError, AttributeError):
        return None


def _read_index(root: Path) -> dict:
    """.cache/index.json: {file: [size, mtime_ns, fingerprint]} of the last
    save -- only when it belongs to the save kinetrace.json describes."""
    try:
        d = json.loads((root / CACHE_DIR / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict) or d.get("saved_at") != _saved_at(root) or not isinstance(d.get("files"), dict):
        return {}
    return d["files"]


def _unchanged_on_disk(p: Path, entry) -> bool:
    try:
        st = p.stat()
    except OSError:
        return False
    return [st.st_size, st.st_mtime_ns] == list(entry[:2])


def _layout_rel(rel) -> bool:
    """True for a relative path of the project layout (what a save writes):
    a top file, reconstruction/..., cameras/<folder>/<camera file or folder>/...
    -- never absolute, a drive, a backslash, '.' / '..' or a hidden part."""
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel or "\0" in rel:
        return False
    parts = PurePosixPath(rel).parts
    if not parts or rel.startswith("/") or any(p in (".", "..") or p.startswith(".") for p in parts):
        return False
    if len(parts) == 1:
        return parts[0] in _TOP_FILES
    if parts[0] == "reconstruction":
        return True
    return (parts[0] == "cameras" and len(parts) >= 3 and is_camera_folder(parts[1])
            and ((len(parts) == 3 and parts[2] in _CAM_FILES) or (len(parts) >= 4 and parts[2] in _CAM_DIRS)))


# the kinds of file a project's sub-folders hold: what a save may treat as its own when it has no record
_OWN_SUFFIX = {"tracks": (".csv",), "silhouette": (".csv", ".npy", ".json"), "body": (".npy", ".json"),
               "segments": (".csv", ".npy", ".json"),
               "reconstruction": (".csv", ".npy", ".json")}


def _scan_layout(root: Path) -> set[str]:
    """The project's own files on disk by their KIND (a table of a landmark, a
    silhouette / body array): used only when there is no record of what the
    previous save wrote. A spreadsheet's `snout.xlsx`, a note, a hidden file
    and any folder the layout does not name are not the project's."""
    out = {n for n in _TOP_FILES if (root / n).is_file()}

    def take(d: Path, kind: str):
        try:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in _OWN_SUFFIX[kind]:
                    out.add(p.relative_to(root).as_posix())
        except OSError:
            pass
    take(root / "reconstruction", "reconstruction")
    cams = root / "cameras"
    try:
        dirs = [d for d in cams.iterdir() if d.is_dir() and is_camera_folder(d.name)] if cams.is_dir() else []
    except OSError:
        dirs = []
    for d in dirs:
        out |= {f"cameras/{d.name}/{n}" for n in _CAM_FILES if (d / n).is_file()}
        for sub in _CAM_DIRS:
            if (d / sub).is_dir():
                take(d / sub, sub)
    return out


def _own_files(root: Path, index: dict) -> set[str]:
    """THE rule for "the project's files" (R16, I164, I165): what a save may
    replace or remove -- the files the previous save wrote (its index), the
    leftovers an older layout is known to have made, and, only when there is no
    index (deleted, or an undone save), the files that are of the project's own
    kinds. Never a user's file (`snout.xlsx`, a note, `cam1 - Copy/`), never a
    hidden one (`.DS_Store`, a lock file), nothing `_layout_rel` rejects --
    what a rollback would refuse to name -- and only what is on disk."""
    out = {r for r in index if isinstance(r, str)}
    cams = root / "cameras"
    try:
        dirs = [d for d in cams.iterdir() if d.is_dir() and is_camera_folder(d.name)] if cams.is_dir() else []
    except OSError:
        dirs = []
    for d in dirs:
        out |= {f"cameras/{d.name}/{n}" for n in _LEGACY_CAM_FILES if (d / n).is_file()}
    if not index:
        out |= _scan_layout(root)
    return {r for r in out if _layout_rel(r) and (root / r).is_file()}


def holds_only_own_entries(folder: str | Path) -> bool:
    """True for a folder a first save may use: empty, or holding only what
    Kinetrace itself makes in a project folder (.cache, .history, .saving,
    .lock -- what a first save that failed leaves behind, I244)."""
    try:
        return all(e.name in _OWN_DOT for e in Path(folder).iterdir())
    except OSError:
        return False


def _history_paths(root: Path, record: dict, what: str) -> dict:
    """The file lists of a .history record (pending.json / previous.json),
    each path checked to be one of the project's own files (I147): the record
    is read from a folder someone else may have made, and a rollback deletes
    and moves what it names. One unsafe entry refuses the whole record."""
    if not isinstance(record, dict):
        record = {}
    out = {}
    base = root.resolve()
    for key in ("added", "replaced", "removed"):
        items = record.get(key, [])
        if not isinstance(items, list):
            items = [items]
        for rel in items:
            ok = _layout_rel(rel)
            if ok:
                try:
                    ok = (root / rel).resolve().is_relative_to(base)
                except (OSError, ValueError):
                    ok = False
            if not ok:
                raise ProjectFileError(f"{root.name}: .history/{what} names {str(rel)[:80]!r}, which is not a file "
                                       f"of this project. Nothing was changed; delete the .history folder to open it.")
        out[key] = items
    return out


def _rollback(root: Path, pending: dict) -> None:
    """Undo a save that did not finish: every file back as it was."""
    hist = root / HISTORY_DIR
    pending = _history_paths(root, pending, "pending.json")
    for rel in pending.get("added", []):
        try:
            (root / rel).unlink()
        except FileNotFoundError:
            pass
    for rel in pending.get("replaced", []) + pending.get("removed", []):
        h = hist / rel
        if h.exists():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            _replace(h, root / rel)
    try:
        (hist / "pending.json").unlink()
    except FileNotFoundError:
        pass
    (root / CACHE_DIR / "index.json").unlink(missing_ok=True)       # caches untrusted: CSVs are read
    shutil.rmtree(root / SAVING_DIR, ignore_errors=True)


def finish_interrupted(root: str | Path) -> str | None:
    """A save cut short (a crash, a power cut): when its last step (the new
    kinetrace.json) happened, only the tidying is finished; otherwise every
    file goes back to the previous save. -> "finished" / "undone" / None."""
    root = Path(root)
    pend = root / HISTORY_DIR / "pending.json"
    said = None
    if pend.is_file():
        try:
            p = json.loads(pend.read_text(encoding="utf-8"))
        except ValueError:
            p = {}
        if not isinstance(p, dict):
            p = {}
        if p.get("saved_at") and _saved_at(root) == p.get("saved_at"):
            _replace(pend, pend.with_name("previous.json"))
            said = "finished"
        else:
            _rollback(root, p)
            said = "undone"
    if (root / SAVING_DIR).exists():
        shutil.rmtree(root / SAVING_DIR, ignore_errors=True)
    return said


def _fresh_dir(d: Path) -> None:
    """An empty folder at `d`, best effort: a file in it that another program
    holds open (the previous save's copy, being compared in a diff tool) must
    not stop a save (I213) -- what cannot be removed stays, and the files this
    save puts there replace it."""
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(exist_ok=True)


def _move_aside(src: Path, dst: Path) -> None:
    """src -> dst, for the previous save's copy in .history. Where `dst` is a file
    another program holds open (a diff tool comparing the previous save) and so
    cannot be replaced, its content is overwritten in place, which Windows
    allows -- the save goes on (I213)."""
    try:
        _replace(src, dst)
    except PermissionError:
        if not dst.is_file():
            raise
        shutil.copyfile(src, dst)
        src.unlink()


def write_folder(frozen: Frozen, path: str | Path, *, fsync: bool = True,
                 yield_gil=lambda: time.sleep(0)) -> dict:
    """Save into the project folder, writing ONLY the files whose content
    changed. Crash-safe: new files are written aside (.saving/), the files
    they replace go to .history/ (the previous save), and the new
    kinetrace.json -- one atomic replace -- is the moment the save counts; a
    save cut short before it is undone the next time the project is opened.
    -> {"written", "unchanged", "removed"} counts (+ "tidy_up": the steps after
    the commit that did not work, in words: the save itself stood, G110)."""
    root = Path(path)
    if root.exists() and not root.is_dir():
        raise ProjectFileError(f"{root.name} is a file, not a project folder")
    if root.is_dir() and not (root / META).is_file() and not holds_only_own_entries(root):
        raise ProjectFileError(f"{root} is a folder that is not a Kinetrace project: choose another name")
    if not root.parent.is_dir():                        # a folder that is gone is said, not re-made (I104)
        raise FileNotFoundError(f"the folder {root.parent} does not exist (a disconnected drive?)")
    root.mkdir(exist_ok=True)
    lock = _take_lock(root)
    try:
        return _write_folder_locked(frozen, root, fsync, yield_gil)
    finally:
        lock.unlink(missing_ok=True)


def _write_folder_locked(frozen: Frozen, root: Path, fsync: bool, yield_gil) -> dict:
    finish_interrupted(root)
    new_saved = json.loads(frozen.files[META])["saved_at"]
    old_saved = _saved_at(root)
    index = _read_index(root)
    cache, tmp = root / CACHE_DIR, root / SAVING_DIR
    cache.mkdir(exist_ok=True)
    _hide(cache)
    (cache / "index.json").unlink(missing_ok=True)      # caches untrusted until this save is complete
    _fresh_dir(tmp)
    _hide(tmp)
    try:
        return _write_folder_steps(frozen, root, index, old_saved, new_saved, fsync, yield_gil)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)         # whatever happened, nothing is left aside


@dataclass
class _Item:
    """One file of a save: what it will hold, its fingerprint, the index entry
    of the previous save, and whether the folder already holds exactly that."""
    content: object
    digest: str
    entry: list | None
    same: bool


def _plan(frozen: Frozen, root: Path, index: dict) -> tuple[dict, list[str], set[str]]:
    """THE planner of a save, shared by the save itself and the close question
    (R16): for every file to write, whether the folder already holds it (same
    fingerprint AND the same file on disk), and which of the project's files go
    (`_own_files`, not what is in the folder). -> ({rel: _Item}, removed,
    case_news): `case_news` = new files whose name differs only by letter case
    or Unicode form from a file that goes (a landmark renamed `snout` ->
    `Snout`, I163): file systems that do not tell them apart must see the old
    one set aside BEFORE the new one goes in, or the new one replaces it and
    then the "removal" takes the new one away."""
    items = {}
    for rel in sorted(frozen.files):
        content, digest = _materialize(frozen.files[rel])
        entry = index.get(rel)
        same = rel != META and entry is not None and entry[2] == digest and _unchanged_on_disk(root / rel, entry)
        items[rel] = _Item(content, digest, entry, same)
    removed = sorted(r for r in _own_files(root, index) if r not in frozen.files)
    gone = {_key(r) for r in removed}
    case_news = {r for r in frozen.files if _key(r) in gone}
    return items, removed, case_news


def _stage(items: dict, root: Path, fsync: bool, yield_gil) -> tuple[list, list, dict, dict]:
    """Stage 1: every file that changed is written into .saving/ (the tables
    are only collected: they are formatted next), and the binary copies of the
    tables into .cache/. -> (changed [(rel, digest)], tables, keep, cols)."""
    cache, tmp = root / CACHE_DIR, root / SAVING_DIR
    keep, changed, tables, cols = {}, [], [], {}
    for rel, it in items.items():
        content, digest, entry, same = it.content, it.digest, it.entry, it.same
        if same and not isinstance(content, Table):
            keep[rel] = entry
            continue
        if isinstance(content, Table):
            cols[rel] = ",".join(content.written)
            cpath = cache / (rel + ".npy")
            if not (same and cpath.is_file()):
                buf = io.BytesIO()
                np.save(buf, content.struct(), allow_pickle=False)
                _write_bytes(cpath.with_name(cpath.name + ".tmp"), buf.getvalue(), False)
                _replace(cpath.with_name(cpath.name + ".tmp"), cpath)
            if same:
                keep[rel] = [*entry[:3], cols[rel]]
                continue
            tables.append((rel, content, digest))       # formatted next, in helpers when there is a lot
            continue
        elif isinstance(content, np.ndarray):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, content, allow_pickle=False)
            data = buf.getvalue()
        else:
            data = content.encode("utf-8")
        _write_bytes(tmp / rel, data, fsync)
        changed.append((rel, digest))
        yield_gil()
    return changed, tables, keep, cols


def _format_tables(tables: list, changed: list, root: Path, fsync: bool, yield_gil) -> None:
    """Stage 2: the tables' CSV text. numpy's exact float text holds the GIL, so
    a big save hands it to helper processes (csvpool.py); whatever they do not
    finish is formatted here, the same bytes either way."""
    tmp = root / SAVING_DIR
    need = sum(t.n * (12 * len(t.written) + 1) for _r, t, _d in tables) + (64 << 20)     # generous
    free = _free_bytes(root)
    if free < need:
        raise ProjectFileError(f"not enough free space on the drive of {root}: about {need >> 20} MB are needed "
                               f"for this save and {free >> 20} MB are free. Nothing was changed.")
    done = set()
    if sum(t.n for _r, t, _d in tables) >= _csvpool().MIN_ROWS:
        done = _csvpool().format_tables([(t, tmp / r) for r, t, _d in tables], fsync)
    for k, (rel, t, digest) in enumerate(tables):
        if k not in done:
            _write_bytes(tmp / rel, t.csv(yield_gil).encode("utf-8"), fsync)
            yield_gil()
        changed.append((rel, digest))
    changed.sort()


def _commit(root: Path, changed: list, removed: list[str], case_news: set[str], old_saved, new_saved,
            fsync: bool) -> None:
    """Stage 3: the files they replace (and the ones that go) move into
    .history/, the new ones move in, and the replace of kinetrace.json is the
    moment the save counts. Anything that goes wrong before it is undone."""
    tmp, hist = root / SAVING_DIR, root / HISTORY_DIR
    new = [r for r, _ in changed if r != META]
    # a case-only rename is a removal plus an addition: `exists()` on a file system that ignores case would
    # call the new name "replaced" (I163)
    pending = {"saved_at_before": old_saved, "saved_at": new_saved, "removed": removed,
               "replaced": [r for r in new if r not in case_news and (root / r).exists()],
               "added": [r for r in new if r in case_news or not (root / r).exists()]}
    _fresh_dir(hist)
    _hide(hist)
    for stale in ("pending.json", "previous.json", META):       # (a file kept open by another program stays)
        try:
            (hist / stale).unlink(missing_ok=True)
        except OSError:
            pass
    _write_json_atomic(hist / "pending.json", pending, fsync)
    committed = False
    try:
        for rel in removed:                                  # the old ones first (I163)
            (hist / rel).parent.mkdir(parents=True, exist_ok=True)
            _move_aside(root / rel, hist / rel)
        for rel in new:
            dst = root / rel
            if dst.exists():
                (hist / rel).parent.mkdir(parents=True, exist_ok=True)
                _move_aside(dst, hist / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            _replace(tmp / rel, dst)
        if (root / META).is_file():
            shutil.copy2(root / META, hist / META)
        if fsync:                                         # every move on disk before the commit
            for d in sorted({(root / r).parent for r in new} | {(hist / r).parent for r in removed}
                            | {(hist / r).parent for r in pending["replaced"]}):
                _fsync_dir(d)
        _replace(tmp / META, root / META)                 # the save counts from here
        committed = True
        if fsync:
            _fsync_dir(root)
    except BaseException:
        if not committed:
            _rollback(root, pending)
        raise


def _finish(root: Path, frozen: Frozen, keep: dict, changed: list, cols: dict, removed: list[str],
            new_saved: str) -> dict:
    """Stage 4, after the commit: nothing here may undo or fail the save --
    it counted. A step that does not work (a file or folder another program
    holds) is logged and said in the stats' "tidy_up", never raised (G110): the
    next open finishes a `pending.json` left behind, and a missing index only
    makes the next save write everything."""
    cache, hist = root / CACHE_DIR, root / HISTORY_DIR
    tidy = []

    def attempt(what, fn):
        try:
            return fn()
        except OSError as e:
            _log.warning("save of %s: %s: %s", root.name, what, e)
            tidy.append(f"{what}: {e}")
    attempt("the previous-save record", lambda: _replace(hist / "pending.json", hist / "previous.json"))
    files = dict(keep)
    for rel, digest in changed:
        st = attempt(f"reading back {rel}", lambda rel=rel: (root / rel).stat())
        if st is not None:
            files[rel] = [st.st_size, st.st_mtime_ns, digest] + ([cols[rel]] if rel in cols else [])
    attempt("the index in .cache", lambda: _write_json_atomic(cache / "index.json", {"saved_at": new_saved,
                                                                                      "files": files}, fsync=False))
    for rel in removed:                                  # a removed camera's empty folders
        def tidy_dirs(rel=rel):
            d = (root / rel).parent
            while d != root and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
        attempt(f"removing the emptied folder of {rel}", tidy_dirs)
    stats = {"written": len(changed), "unchanged": len(frozen.files) - len(changed), "removed": len(removed)}
    if tidy:
        stats["tidy_up"] = tidy
    return stats


def _write_folder_steps(frozen, root, index, old_saved, new_saved, fsync, yield_gil) -> dict:
    cache = root / CACHE_DIR
    items, removed, case_news = _plan(frozen, root, index)
    for rel in removed:        # their cache copies go first: on a file system that ignores case one may be the new file's
        try:
            (cache / (rel + ".npy")).unlink(missing_ok=True)
        except OSError:
            pass
    changed, tables, keep, cols = _stage(items, root, fsync, yield_gil)
    _format_tables(tables, changed, root, fsync, yield_gil)
    _commit(root, changed, removed, case_news, old_saved, new_saved, fsync)
    return _finish(root, frozen, keep, changed, cols, removed, new_saved)


_NOT_CHANGES = {META, "README.txt", "state.json"}      # where the user was / derived: never "a change"


def changes_since_save(frozen: Frozen, path: str | Path) -> tuple[list[str], list[str]] | None:
    """What a save would do now, from the same plan the save makes (`_plan`):
    (files whose content would be written, files that would be removed),
    without writing anything. None when that cannot be told (not a folder, no
    index)."""
    root = project_root(path)
    if not root.is_dir():
        return None
    index = _read_index(root)
    if not index:
        return None
    skip = lambda r: r in _NOT_CHANGES or r.endswith("/view.json")      # noqa: E731
    items, removed, _news = _plan(frozen, root, index)
    changed = [rel for rel, it in items.items() if not it.same and not skip(rel)]
    return changed, [r for r in removed if not skip(r)]


def describe_changes(changed: list[str], removed: list[str], cameras: list[tuple[str, str]],
                     landmarks: dict[str, dict[str, str]]) -> list[str]:
    """`changes_since_save`'s files in words, one line per camera / part:
    'cam1: P1 (positions), the events'. `cameras` = (folder, name);
    `landmarks` = {folder: {file: landmark name}} (the file a landmark had
    at the last save may be gone: its file stem is used then)."""
    top = {"project.json": "the cameras (videos, offsets, frame rates)", "calibration.json": "the calibration",
           "lenses.json": "the lens profiles"}
    part = {"points.csv": "the landmark list", "events.csv": "the events", "notes.csv": "the notes",
            "ball_prompts.json": "the ball markers' clicks", "spots.json": "the Moving spot settings",
            "skeleton.json": "the skeleton (an older project's)",
            "segment.json": "the animal (its clicks, skeleton, hold)"}
    names = dict(cameras)
    lines, per_cam, rest = [], {}, []

    def say(rel: str, gone: bool):
        bits = rel.split("/")
        if bits[0] == "cameras" and len(bits) >= 3:
            folder, what = bits[1], "/".join(bits[2:])
            if what.startswith("tracks/"):
                f = what[len("tracks/"):]
                w = f"{landmarks.get(folder, {}).get(f, PurePosixPath(f).stem)} ({'removed' if gone else 'positions'})"
            elif what.startswith("silhouette/"):
                w = "the silhouette"
            elif what.startswith("segments/"):
                w = f"the segment {what.split('/')[1]}"
            elif what.startswith("body/"):
                w = "the body poses"
            else:
                w = part.get(what, what)
            per_cam.setdefault(names.get(folder, folder) + (" (removed)" if gone and what == "points.csv" else ""),
                               []).append(w)
        elif rel.startswith("reconstruction/"):
            rest.append("the 3D result")
        else:
            rest.append(top.get(rel, rel))
    for r in changed:
        say(r, False)
    for r in removed:
        say(r, True)
    for cam, ws in per_cam.items():
        lines.append(f"{cam}: " + ", ".join(dict.fromkeys(ws)))
    lines += list(dict.fromkeys(rest))
    return lines


def restore_previous(path: str | Path) -> str:
    """Put the project folder back as it was at the save before the last one
    (the files the last save replaced or removed are in .history/). -> the
    restored save's time. The history is used up: it cannot be done twice."""
    root = project_root(path)
    hist = root / HISTORY_DIR
    try:
        p = json.loads((hist / "previous.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ProjectFileError(f"{root.name}: no earlier save is kept") from None
    if not (hist / META).is_file():
        raise ProjectFileError(f"{root.name}: the last save was the first one; there is nothing earlier")
    saved_before = p.get("saved_at_before") if isinstance(p, dict) else None
    p = _history_paths(root, p, "previous.json")
    for rel in p.get("added", []):
        (root / rel).unlink(missing_ok=True)
    for rel in p.get("replaced", []) + p.get("removed", []):
        if (hist / rel).exists():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            _replace(hist / rel, root / rel)
    _replace(hist / META, root / META)
    (hist / "previous.json").unlink()
    (root / CACHE_DIR / "index.json").unlink(missing_ok=True)
    return str(saved_before or "")


# ------------------------------------------------------------------ read
class _Source:
    """A project folder or a single file (zip) with the same layout. Names are
    matched by `_key` (composed, case-folded) when the exact name is not there:
    a file system that lists `Cam1` as `cam1`, or a decomposed name, is still
    the file (I163, I211); hidden files (`.DS_Store`, AppleDouble `._x.csv`) are
    not part of the project (I210)."""

    def __init__(self, path: Path):
        self.path = path
        self.index: dict = {}
        if path.is_dir():
            self.zip = None
            names = set()
            for dirpath, dirs, fnames in os.walk(path):
                rel = Path(dirpath).relative_to(path).as_posix()
                if rel == ".":
                    dirs[:] = [d for d in dirs if d not in _NOT_DATA and not d.startswith(".")]
                    rel = ""
                else:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                names |= {f"{rel}/{f}" if rel else f for f in fnames if not f.startswith(".")}
            self.index = _read_index(path)
        else:
            try:
                self.zip = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                raise ProjectFileError(f"{path.name} is not a Kinetrace project (not a zip file, or damaged)") from None
            names = set(self.zip.namelist())
        for n in names:
            if ".." in PurePosixPath(n).parts or n.startswith("/"):
                raise ProjectFileError(f"{path.name}: unsafe member name {n!r}")
        # a member with a hidden part (macOS's `._x.csv`, `.DS_Store`) is never the project's (I210)
        self.names = {n for n in names if not any(p.startswith(".") for p in PurePosixPath(n).parts)}
        self._keys = {n: _key(n) for n in self.names}
        self._actual: dict[str, str] = {}
        for n in sorted(self.names):
            self._actual.setdefault(self._keys[n], n)

    def resolve(self, name: str) -> str | None:
        """The name as the project holds it: exactly as asked, else the same name
        up to case / Unicode form; None when there is none."""
        return name if name in self.names else self._actual.get(_key(name))

    def has(self, name: str) -> bool:
        return self.resolve(name) is not None

    def bytes(self, name: str) -> bytes:
        name = self.resolve(name) or name
        try:
            return self.zip.read(name) if self.zip else (self.path / name).read_bytes()
        except zipfile.BadZipFile as e:                      # CRC mismatch: damaged
            raise ProjectFileError(f"{name}: damaged ({e})") from None

    def text(self, name: str) -> str:
        try:
            return self.bytes(name).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ProjectFileError(f"{name}: not UTF-8 text") from None

    def json(self, name: str):
        try:
            return json.loads(self.text(name))
        except json.JSONDecodeError as e:
            raise ProjectFileError(f"{name}: line {e.lineno}: {e.msg}") from None

    def npy(self, name: str) -> np.ndarray:
        try:
            return np.lib.format.read_array(io.BytesIO(self.bytes(name)), allow_pickle=False)
        except ValueError as e:
            raise ProjectFileError(f"{name}: {e}") from None

    def arrays(self, folder: str) -> dict:
        """folder/*.npy + *.json -> the mapping `from_arrays` expects."""
        out = {}
        for n in self.files_in(folder):
            key = PurePosixPath(n).name
            if key.endswith(".npy"):
                out[key[:-4]] = self.npy(n)
            elif key.endswith(".json"):
                out[key[:-5]] = self.text(n)
        return out

    def table(self, name: str, required: tuple, optional: tuple = ()) -> tuple[dict, int]:
        return parse_table(self.text(name), name, required, optional)

    def load_table(self, name: str, parse, cols, kinds) -> Table:
        """A CSV table -- from .cache/ when the file is still the one the
        last save wrote (same size and time), else parsed from its text (a
        hand edit): the CSV is always what counts."""
        name = self.resolve(name) or name
        entry = self.index.get(name)
        if entry is not None and len(entry) > 3 and _unchanged_on_disk(self.path / name, entry):
            try:
                s = np.load(self.path / CACHE_DIR / (name + ".npy"), allow_pickle=False)
                t = Table.from_struct(s, cols, kinds)
                if t is not None and set(entry[3].split(",")) <= set(cols):
                    t.written = tuple(entry[3].split(","))
                    # its fingerprint must be the one the save recorded: a copy damaged
                    # by a power cut (caches are not fsynced) is never read
                    if t.digest() == entry[2]:
                        return t
            except (OSError, ValueError, EOFError):
                pass
        return parse(self.text(name), name)

    def _under(self, folder: str) -> list[tuple[str, str]]:
        """(rest of the name below `folder`, the name as held) for everything under it."""
        pre = _key(folder) + "/"
        return [(k[len(pre):], n) for n, k in self._keys.items() if k.startswith(pre)]

    def has_dir(self, folder: str) -> bool:
        return bool(self._under(folder))

    def files_in(self, folder: str) -> list[str]:
        return sorted(n for rest, n in self._under(folder) if "/" not in rest)


def parse_table(text: str, name: str, required: tuple, optional: tuple = ()) -> tuple[dict, int]:
    """A CSV as {column: list of strings}, found by header name (lower case).
    Also reads tracks files from outside a project (trackio.py)."""
    head = text.split("\n", 1)[0]
    if ";" in head and "," not in head:
        raise ProjectFileError(f"{name}: separated by ';' - saved by a spreadsheet set to a "
                               "comma-decimal locale. Save it with ',' separators and '.' decimals.")
    if '"' in text:
        rows = list(csv.reader(io.StringIO(text)))
    else:
        rows = [ln.split(",") for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while rows and (not rows[-1] or rows[-1] == [""]):          # trailing blank lines
        rows.pop()
    if not rows:
        return {c: [] for c in required + optional}, 0
    header = [h.strip().lower() for h in rows[0]]
    missing = [c for c in required if c not in header]
    if missing:
        raise ProjectFileError(f"{name}: missing column(s) {', '.join(missing)}")
    n = len(rows) - 1
    width = len(header)
    if set(map(len, rows[1:])) - {width}:                   # only then find the row, for the message
        k, r = next((k, r) for k, r in enumerate(rows[1:], start=2) if len(r) != width)
        raise ProjectFileError(f"{name}: row {k} has {len(r)} values, the header has {width}")
    cols = list(zip(*rows[1:])) if n else [()] * width
    out = {h: list(c) for h, c in zip(header, cols)}
    for c in optional:
        out.setdefault(c, None)
    return out, n

def _foreign_name(p: str) -> str:
    """The file name of a path saved on any OS (C:\\a\\b.mp4 on a Mac -> b.mp4)."""
    if re.match(r"^[A-Za-z]:[\\/]|^\\\\", p or ""):
        return PureWindowsPath(p).name
    return PurePosixPath(p or "").name


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def locate_video(entry: dict, project_dir: Path | None, extra_dirs=()) -> Path | None:
    """The camera's video on THIS computer: relative path, absolute path, then
    the same file name in the project folder or folders already used."""
    v = entry.get("video") or {}
    cands = []
    rel = v.get("relative_path")
    if rel and project_dir is not None:
        cands.append(project_dir / Path(*PurePosixPath(rel).parts))
    if v.get("path"):
        cands.append(Path(v["path"]))
    name = _nfc(_foreign_name(v.get("path") or rel or ""))
    for d in [project_dir, *extra_dirs]:
        if d is not None and name:
            cands.append(Path(d) / name)
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    # macOS may hand back a decomposed name: compare normalised names in the folders
    for d in [project_dir, *extra_dirs]:
        if d is None or not name or not Path(d).is_dir():
            continue
        fold = sys.platform in ("win32", "darwin")     # case-insensitive file systems
        for f in Path(d).iterdir():
            if _nfc(f.name) == name or (fold and _nfc(f.name).lower() == name.lower()):
                return f
    return None


@contextmanager
def _parsing(where: str):
    """An odd value in a hand-edited file is a sentence naming the file (and the
    row when the caller knows it), never a raw KeyError / ValueError (G111)."""
    try:
        yield
    except ProjectFileError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as e:
        what = f"{e} is missing" if isinstance(e, KeyError) else str(e)
        raise ProjectFileError(f"{where}: a value cannot be read ({what})") from None


def read(path: str | Path):
    """-> (Project, state dict, kinetrace.json dict). Raises ProjectFileError
    with the file (and row) for anything that cannot be read."""
    from kinetrace.project import Project
    from kinetrace.session import DEFAULT_UI_STATE

    root = project_root(path)
    interrupted = None
    live = None
    if root.is_dir() and (root / LOCK).is_file():
        try:
            live = clear_stale_lock(root)
        except OSError:
            pass
    if root.is_dir() and (root / HISTORY_DIR / "pending.json").is_file():
        if live is not None:        # a save is running right now: its files are mid-move, not an interrupted save (I246)
            raise ProjectFileError(
                f"{root.name} is being saved by another Kinetrace right now (process {live.get('pid')} on "
                f"{live.get('host')}). Open it again when that save has finished.")
        try:
            interrupted = finish_interrupted(root)
        except OSError as e:
            raise ProjectFileError(f"{root.name}: an earlier save of this project was cut short and could not "
                                   f"be undone ({e}). Make the folder writable and open it again.") from None
    src = _Source(root)
    if not src.has("kinetrace.json"):
        raise ProjectFileError(f"{root.name}: no kinetrace.json - not a Kinetrace project")
    meta = src.json("kinetrace.json")
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise ProjectFileError("kinetrace.json: not a Kinetrace project file")
    with _parsing("kinetrace.json"):
        newer = int(meta.get("format_version", 0)) > FORMAT_VERSION
    if newer:
        raise ProjectFileError(f"kinetrace.json: made by a newer Kinetrace (format "
                               f"{meta.get('format_version')}); update Kinetrace to open it")
    if not safe_id(meta.get("project_id")):
        meta["project_id"] = new_id()             # a file name in the recovery folder: never a path (I148)
    pj = src.json("project.json") if src.has("project.json") else None
    if not isinstance(pj, dict) or not isinstance(pj.get("cameras"), list) or not pj["cameras"]:
        raise ProjectFileError("project.json: missing, or no cameras listed")
    state = src.json("state.json") if src.has("state.json") else {}
    if not isinstance(state, dict):
        state = {}                                # a hand-edited state.json: defaults, not a refusal
    if not isinstance(state.get("tools"), dict):
        state["tools"] = {}
    sessions, names, offsets, rates = _read_cameras(src, pj)
    active = names.index(pj["active_camera"]) if pj.get("active_camera") in names else 0
    p = Project(sessions, names, offsets, active, rates)
    for s in p.sessions:                         # window-wide toggles live once, in state.json
        s.ui_state.update({k: v for k, v in state["tools"].items() if k in DEFAULT_UI_STATE})
    p.calibration = _read_calibration(src)
    lenses = _read_lenses(src, len(sessions))
    if lenses is not None:
        p.lenses = lenses
    p.reconstruction = _read_reconstruction(src)
    p.exports = [str(x) for x in (pj.get("exports_on_save") or []) if isinstance(x, str)]
    p.dirty = False
    # the app locates each camera's video from these, and says when a cut-short save was undone
    meta = dict(meta, _cameras=pj["cameras"], _interrupted=interrupted)
    return p, state, meta


def _read_cameras(src: _Source, pj: dict) -> tuple[list, list, list, list]:
    """project.json's cameras -> (sessions with their data, names, offsets, rates)."""
    from kinetrace.session import DEFAULT_UI_STATE, TrackingSession
    sessions, names, offsets, rates = [], [], [], []
    for k, cam in enumerate(pj["cameras"]):
        where = f"project.json camera {k + 1}"
        if not isinstance(cam, dict):
            raise ProjectFileError(f"{where}: expected the camera's settings")
        folder = cam.get("folder") or camera_folders([cam.get("name", f"cam{k + 1}")])[0]
        if not is_camera_folder(folder):
            raise ProjectFileError(f"project.json: camera folder {folder!r} is not allowed")
        try:
            T, fps = int(cam["n_frames"]), float(cam["fps"])
            w, h = int(cam["width"]), int(cam["height"])
        except (KeyError, TypeError, ValueError):
            raise ProjectFileError(f"{where}: needs n_frames, fps, width and height") from None
        if not (1 <= T <= MAX_FRAMES and 1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):   # sizes every array (I157)
            raise ProjectFileError(f"{where}: {T} frames of {w} x {h} pixels is not a video Kinetrace can hold")
        # a rate of 0 would divide every playhead move; a NaN / null offset would raise at the first one (I214)
        try:
            offset, rate = float(cam.get("offset", 0.0)), float(cam.get("rate", 1.0))
        except (TypeError, ValueError):
            raise ProjectFileError(f"{where}: offset and rate must be numbers (found offset "
                                   f"{cam.get('offset')!r}, rate {cam.get('rate')!r})") from None
        if not np.isfinite(offset):
            raise ProjectFileError(f"{where}: offset {cam.get('offset')!r} is not a finite number of frames")
        if not (np.isfinite(rate) and rate > 0):
            raise ProjectFileError(f"{where}: rate {cam.get('rate')!r} must be a number above 0 "
                                   "(this camera's frame rate / the first camera's)")
        s = TrackingSession(str((cam.get("video") or {}).get("path", "")), T, fps, w, h)
        try:                                     # a project from before G38 has no file_fps: the same rate
            s.file_fps = float(cam.get("file_fps", fps)) or fps
        except (TypeError, ValueError):
            s.file_fps = fps
        s.ui_state = dict(DEFAULT_UI_STATE)
        _read_camera(src, f"cameras/{folder}", s)
        sessions.append(s)
        names.append(str(cam.get("name", folder)))
        offsets.append(offset)
        rates.append(rate)
    return sessions, names, offsets, rates


def _read_calibration(src: _Source):
    from kinetrace.calib import Calibration
    if not src.has("calibration.json"):
        return None
    c = src.json("calibration.json")
    if not isinstance(c, dict):
        raise ProjectFileError("calibration.json: expected the calibration's settings")
    try:
        coefs = np.asarray(c.pop("coefs"), np.float64)
    except (KeyError, TypeError, ValueError):
        raise ProjectFileError("calibration.json: 'coefs' missing or not 11 numbers per camera") from None
    with _parsing("calibration.json"):
        notes = c.pop("notes", [])
        for cam in c.get("cams") or []:
            if cam.get("rmse") is None:                  # NaN is written as null
                cam["rmse"] = float("nan")
        cal = Calibration.from_arrays({"calib_coefs": coefs, "calib_meta": json.dumps(c)}, "calib_")
        if cal is not None:
            cal.notes = list(notes or [])
    return cal


def _read_lenses(src: _Source, n_cameras: int):
    if not src.has("lenses.json"):
        return None
    from kinetrace.lens import LensProfile
    raw = src.json("lenses.json")
    if not isinstance(raw, list) or len(raw) != n_cameras:
        return None
    out = []
    for k, d in enumerate(raw):
        with _parsing(f"lenses.json entry {k + 1}"):
            out.append(None if d is None else LensProfile.from_json(d))
    return out


def _read_reconstruction(src: _Source):
    from kinetrace.calib import Reconstruction
    rec = src.arrays("reconstruction")
    if "xyz" in rec:                              # binary (recovery copies, format 1)
        with _parsing("reconstruction"):
            m = json.loads(rec["meta"])
            return Reconstruction(int(m["t0"]), list(m["names"]), rec["xyz"].astype(np.float64),
                                  rec["residual"].astype(np.float64), rec["n_cams"].astype(np.int32),
                                  str(m.get("unit", "")),
                                  rec["per_cam"].astype(np.float32) if "per_cam" in rec else None)
    if src.has("reconstruction/meta.json"):
        return _read_points3d(src)
    return None


def _read_points3d(src: _Source):
    """reconstruction/meta.json + one CSV per landmark (rows = reference frames)."""
    from kinetrace.calib import Reconstruction
    m = src.json("reconstruction/meta.json")
    try:
        t0, R = int(m["t0"]), int(m["n_frames"])
        pts = list(m.get("points") or [])
        names = [str(q["name"]) for q in pts]
        cams_px = [str(c) for c in (m.get("per_camera_columns") or [])]
    except (KeyError, TypeError, ValueError):
        raise ProjectFileError("reconstruction/meta.json: needs t0, n_frames and points") from None
    if not 0 <= R <= MAX_FRAMES:
        raise ProjectFileError(f"reconstruction/meta.json: {R} frames is not a length Kinetrace can hold")
    N, C = len(names), len(cams_px)
    xyz = np.full((R, N, 3), np.nan)
    res = np.full((R, N), np.nan)
    ncam = np.zeros((R, N), np.int32)
    per = np.full((R, N, C), np.nan, np.float32) if C else None
    parse = _point3d_parser(cams_px)
    kinds = ("i", "f8", "f8", "f8", "f8", "i") + ("f4",) * C
    for j, q in enumerate(pts):
        rel = f"reconstruction/{q.get('file', '')}"
        if not src.has(rel):
            continue
        d = src.load_table(rel, parse, POINT3D_COLS + tuple(cams_px), kinds).data
        _check_frames(d["frame"], R, rel, first=t0)
        r = d["frame"] - t0
        xyz[r, j, 0], xyz[r, j, 1], xyz[r, j, 2] = d["x"], d["y"], d["z"]
        res[r, j], ncam[r, j] = d["residual"], d["n_cams"]
        for k, c in enumerate(cams_px):
            per[r, j, k] = d[c]
    return Reconstruction(t0, names, xyz, res, ncam, str(m.get("unit", "")), per)


def _clean_name(n: str) -> str:
    """A name read from a file, without what a spreadsheet takes for a formula
    (M7): ONE rule for every place a landmark's name comes from -- points.csv,
    a format-1 tracks.csv, the keys of ball_prompts.json / spots.json -- or the
    same landmark would be two (I212)."""
    from kinetrace.session import formula_safe, starts_formula
    return formula_safe(n) if starts_formula(n) else n


def _read_point(cols: dict, n: int, i: int, where: str):
    """Row `i` of points.csv -> PointMeta (`_point_row` is its inverse)."""
    from kinetrace.session import SOURCES, TRACKERS, PointMeta

    def g(c, default=""):
        return cols[c][i] if cols.get(c) is not None else default
    with _parsing(where):
        outline = g("outline").split()
        source = g("source", "track")
        return PointMeta(
            _clean_name(g("name")), _rgb(g("color", "#ffffff"), where), g("shown", "1") != "0",
            g("kind", "point") or "point", float(g("radius", "0") or 0), g("anchor", "0") == "1",
            source if source in SOURCES else "track", g("spec"), g("free", "0") == "1",
            g("shape", "circle") or "circle",
            [[float(outline[j]), float(outline[j + 1])] for j in range(0, len(outline) - 1, 2)] or None,
            tracker=g("tracker", "") if g("tracker", "") in TRACKERS else "",
            segment=_clean_name(g("animal", "")) if g("animal", "") else "")


def _read_camera(src: _Source, d: str, s) -> None:
    points, pfiles = _read_points(src, d)
    arr, points = _read_tracks(src, d, points, pfiles, s.n_frames)
    index = {p.name: i for i, p in enumerate(points)}
    s.points = points
    from kinetrace.session import POINT_ARRAYS
    for a in POINT_ARRAYS:                          # (R15)
        setattr(s, a.name, arr[a.name])
    _read_events_notes(src, d, s)
    _read_markers(src, d, s, index)
    _read_segment_body(src, d, s)
    _read_view(src, d, s, index)


def _read_points(src: _Source, d: str) -> tuple[list, list]:
    """points.csv -> (the points with their settings, the `file` column)."""
    points, pfiles = [], []
    if src.has(f"{d}/points.csv"):
        cols, n = src.table(f"{d}/points.csv", ("name",), POINT_COLS[1:])
        pfiles = list(cols.get("file") or [""] * n)
        for i in range(n):
            points.append(_read_point(cols, n, i, f"{d}/points.csv row {i + 2}"))
    return points, pfiles


def _read_tracks(src: _Source, d: str, points: list, pfiles: list, T: int) -> tuple[dict, list]:
    """The positions: dense .npy in recovery files, one CSV per landmark in
    projects, one long tracks.csv in format 1. -> (the T x N arrays, the points
    -- a landmark file points.csv does not name adds one)."""
    if src.has(f"{d}/tracks.npy"):
        from kinetrace.session import POINT_ARRAYS
        arr = {a.name: src.npy(f"{d}/{a.name}.npy") for a in POINT_ARRAYS}      # (R15)
        for k, v in arr.items():
            if v.shape[0] != T or v.shape[1] != len(points):
                raise ProjectFileError(f"{d}/{k}.npy: shape {v.shape} does not match {T} frames x "
                                       f"{len(points)} points")
        return arr, points
    if src.has_dir(f"{d}/tracks") or not src.has(f"{d}/tracks.csv"):
        derived = landmark_files([p.name for p in points])
        named = [f"{d}/tracks/{(pfiles[j] if j < len(pfiles) and pfiles[j].strip() else derived[j]).strip()}"
                 for j in range(len(points))]
        known = {_key(r) for r in named}
        for rel in src.files_in(f"{d}/tracks"):     # a file points.csv does not name: a new landmark
            if rel.lower().endswith(".csv") and _key(rel) not in known:
                points.append(_new_point(PurePosixPath(rel).stem, len(points)))
                named.append(rel)
                known.add(_key(rel))
        arr = _empty_tracks(T, len(points))
        for j, rel in enumerate(named):
            if src.has(rel):
                _fill_landmark(arr, j, src.load_table(rel, _parse_landmark, LANDMARK_COLS, _LM_KINDS), T, rel)
        return arr, points
    cols, n = src.table(f"{d}/tracks.csv", ("frame", "point", "x", "y"), TRACK_COLS[4:])
    return tracks_from_table(cols, n, points, T, f"{d}/tracks.csv")


def _new_point(raw_name: str, k: int):
    """A landmark only a file knows (named by the file / the table)."""
    from kinetrace.session import PointMeta
    return PointMeta(_clean_name(unicodedata.normalize("NFC", raw_name)), _PALETTE[k % len(_PALETTE)])


def _read_events_notes(src: _Source, d: str, s) -> None:
    from kinetrace.session import Event
    if src.has(f"{d}/events.csv"):
        cols, n = src.table(f"{d}/events.csv", ("name", "start", "end"), EVENT_COLS[3:])
        for i in range(n):
            where = f"{d}/events.csv row {i + 2}"
            try:
                a, b = int(cols["start"][i]), int(cols["end"][i])
            except ValueError:
                raise ProjectFileError(f"{where}: start / end must be whole numbers") from None
            s.events.append(Event(_clean_name(cols["name"][i]), min(a, b), max(a, b),
                                  _rgb(cols["color"][i], where) if cols.get("color") else (255, 200, 0),
                                  (cols.get("note") or [""] * n)[i], (cols.get("author") or [""] * n)[i]))
    if src.has(f"{d}/notes.csv"):
        cols, n = src.table(f"{d}/notes.csv", ("frame", "text"), NOTE_COLS[2:])
        for i in range(n):
            if cols["text"][i].strip():
                try:
                    frame = int(cols["frame"][i])
                except ValueError:
                    raise ProjectFileError(f"{d}/notes.csv row {i + 2}: the frame {cols['frame'][i]!r} must be a "
                                           "whole number") from None
                s.notes[frame] = {"text": cols["text"][i], "author": (cols.get("author") or [""] * n)[i],
                                  "time": (cols.get("time") or [""] * n)[i]}


def _read_markers(src: _Source, d: str, s, index: dict) -> None:
    """ball_prompts.json / spots.json (keyed by landmark name)."""
    rel = f"{d}/ball_prompts.json"
    if src.has(rel):
        raw = src.json(rel)
        if not isinstance(raw, dict):
            raise ProjectFileError(f"{rel}: expected one entry per ball marker")
        for nm, per in raw.items():
            nm = _clean_name(nm)
            if nm in index and s.points[index[nm]].is_ball:
                with _parsing(f"{rel} ({nm})"):
                    s.points[index[nm]].ball_prompts = {
                        int(f): [[float(c[0]), float(c[1]), int(c[2])] for c in cs] for f, cs in per.items()}
    if src.has(f"{d}/spots.json"):
        from kinetrace.spots import SpotSettings
        per = src.json(f"{d}/spots.json")
        for nm, st in (per.items() if isinstance(per, dict) else ()):
            nm = _clean_name(nm)
            if nm in index and isinstance(st, dict):       # odd values fall back to automatic
                s.points[index[nm]].spot = SpotSettings.from_dict(st).to_dict()


def _read_segment_body(src: _Source, d: str, s) -> None:
    from kinetrace.body import BodyTrack
    from kinetrace.segmenter import MIDLINE_SAMPLES, MaskTrack
    from kinetrace.session import AnimalMeta
    T = s.n_frames
    # (G149) the first segment where a one-segment project had it, then segments/<name>/ in order
    folders = [d] if src.has(f"{d}/segment.json") else []
    extra = sorted({rest.split("/")[0] for rest, _n in src._under(f"{d}/segments")
                    if rest.count("/") == 1 and rest.endswith("/segment.json")})
    order = {}
    for name in extra:
        with _parsing(f"{d}/segments/{name}/segment.json"):
            order[name] = int(json.loads(src.text(f"{d}/segments/{name}/segment.json")).get("order", 1_000_000))
    folders += [f"{d}/segments/{name}" for name in sorted(extra, key=lambda n: (order[n], n))]
    for base in folders:
        _read_one_segment(src, base, s, T, AnimalMeta, MaskTrack, MIDLINE_SAMPLES)
    s.active_seg = 0
    b = src.arrays(f"{d}/body")
    if b:
        with _parsing(f"{d}/body"):
            s.body = BodyTrack.from_arrays("", b)
        if s.body.n_frames != T:
            raise ProjectFileError(f"{d}/body: {s.body.n_frames} frames, the video has {T}")


def _read_one_segment(src: "_Source", d: str, s, T: int, AnimalMeta, MaskTrack, MIDLINE_SAMPLES) -> None:
    """One segment (its segment.json + silhouette/) appended to the session (G149)."""
    with _parsing(f"{d}/segment.json"):
        meta = AnimalMeta.from_json(src.text(f"{d}/segment.json"))
    if meta.name in s.segment_names():
        raise ProjectFileError(f"{d}/segment.json: a second segment named {meta.name!r}")
    s.add_segment(meta.name, meta=meta)
    m = src.arrays(f"{d}/silhouette")
    rel = f"{d}/silhouette/summary.csv"
    if "bbox" not in m and src.has(rel):          # format 2: the per-frame numbers are a CSV
        t = src.load_table(rel, _parse_silhouette, SILHOUETTE_COLS, _SIL_KINDS).data
        _check_frames(t["frame"], T, rel)
        f = t["frame"]
        m["bbox"] = np.full((T, 4), -1, np.int32)
        m["area"] = np.zeros(T, np.int32)
        m["centroid"] = np.full((T, 2), np.nan, np.float32)
        m["score"] = np.full(T, np.nan, np.float32)
        m["area"][f], m["score"][f] = t["area"], t["score"]
        m["centroid"][f, 0], m["centroid"][f, 1] = t["centroid_x"], t["centroid_y"]
        for k, c in enumerate(("x0", "y0", "x1", "y1")):
            m["bbox"][f, k] = t[c]
        for k, v in (("cpts", np.zeros((0, 2), np.int32)), ("coff", np.zeros(1, np.int64)),
                     ("cframe", np.zeros(0, np.int32)), ("mframe", np.zeros(0, np.int32)),
                     ("mpts", np.zeros((0, MIDLINE_SAMPLES, 2), np.float32))):
            m.setdefault(k, v)                 # a hand-written summary without outlines
    with _parsing(f"{d}/silhouette"):
        s.masks = MaskTrack.from_arrays("", m) if "bbox" in m else MaskTrack(T)
    if s.masks.n_frames != T:
        raise ProjectFileError(f"{d}/silhouette: {s.masks.n_frames} frames, the video has {T}")
    s.masks.native_w, s.masks.native_h = s.width, s.height


def apply_hidden_animals(s) -> None:
    """(G167) `ui_state["hidden_animals"]` (the view sidecar, `_sync_ui_state`) onto the animals' `shown`;
    view.json itself is read straight onto them (`_read_view`)."""
    hidden = s.ui_state.get("hidden_animals")
    if isinstance(hidden, list):
        for a in s.segments:
            a.shown = a.name not in hidden


def _read_view(src: _Source, d: str, s, index: dict) -> None:
    """view.json: where the user was (never data: odd values fall back)."""
    T = s.n_frames
    s._name_counter, s._event_counter = len(s.points), len(s.events)
    v = src.json(f"{d}/view.json") if src.has(f"{d}/view.json") else {}
    if not isinstance(v, dict):
        v = {}

    def view_value(key, conv, default):
        # where the user was is not data: an odd hand-edited value falls back
        try:
            return conv(v[key]) if v.get(key) is not None else default
        except (TypeError, ValueError):
            return default
    s.current_frame = int(np.clip(view_value("current_frame", int, 0), 0, T - 1))
    sel = v.get("selected_point")
    s.ui_state["selected"] = index.get(sel, -1) if isinstance(sel, str) else -1
    names = v.get("selected_points")                    # the run scope (G103): names, never indices
    if isinstance(names, list):
        s.ui_state["selected_names"] = [x for x in names if isinstance(x, str)]
    if "segment_selected" in v:
        s.ui_state["segment_selected"] = bool(v.get("segment_selected"))
    animals = v.get("selected_animals")                 # (G161) which animals, by name
    if isinstance(animals, list):
        s.ui_state["segments_selected"] = [x for x in animals if isinstance(x, str)]
    for k in ("zoom", "center_x", "center_y"):
        s.ui_state[k] = view_value(k, float, s.ui_state.get(k, 0.0))
    s.ui_state["user_zoomed"] = bool(v.get("user_zoomed", False))
    if v.get("timeline") is not None:
        s.ui_state["timeline"] = v["timeline"]
    hidden = v.get("hidden_animals")                    # (G167) the animals whose silhouette is not drawn:
    if isinstance(hidden, list):                        # onto the animals themselves (their `shown` is the state)
        for a in s.segments:
            a.shown = a.name not in hidden
    s.annotator = str(v.get("annotator", "") or "")
    c = v.get("counters") if isinstance(v.get("counters"), dict) else {}
    try:
        s._name_counter = max(int(c.get("points") or 0), len(s.points))
        s._event_counter = max(int(c.get("events") or 0), len(s.events))
    except (TypeError, ValueError):
        pass                                     # the point / event counts set above stand
    s.dirty = False


def tracks_from_table(cols: dict, n: int, points: list, T: int, where: str):
    """tracks.csv columns -> the T x N arrays of a session (NaN / False where a
    cell has no row). A point name only the table knows becomes a new point,
    appended to `points`. -> (arrays, points). Also used by trackio.py."""
    index = {p.name: i for i, p in enumerate(points)}
    if n:
        try:
            fr = np.fromiter(map(int, cols["frame"]), np.int64, n)
        except ValueError:
            raise ProjectFileError(f"{where}: a frame number is not a whole number") from None
        bad = (fr < 0) | (fr >= T)
        if bad.any():
            raise ProjectFileError(f"{where}: row {int(np.flatnonzero(bad)[0]) + 2}: frame {fr[bad][0]} "
                                   f"is outside the video (0 - {T - 1})")
        pnames = [_clean_name(x) for x in cols["point"]]      # the same name rule as points.csv (I212)
        for nm in sorted(set(pnames) - set(index), key=pnames.index):
            index[nm] = len(points)                   # a name only tracks.csv knows: a new point
            points.append(_new_point(nm, len(points)))
        pi = np.fromiter(map(index.__getitem__, pnames), np.int64, n)
    N = len(points)
    arr = _empty_tracks(T, N)
    if n:
        dup = np.unique(fr * N + pi, return_counts=True)[1] > 1
        if dup.any():
            raise ProjectFileError(f"{where}: the same frame and point appear twice")
        x, y = text_f32(cols["x"], where), text_f32(cols["y"], where)
        arr["tracks"][fr, pi, 0], arr["tracks"][fr, pi, 1] = x, y
        has = np.isfinite(x) & np.isfinite(y)
        arr["tracked"][fr, pi] = has
        arr["confidence"][fr, pi] = (text_f32(cols["confidence"], where) if cols.get("confidence") is not None
                                     else np.where(has, 1.0, 0.0).astype(np.float32))
        for key, col in (("visibility", "visible"), ("manual", "hand_placed"), ("occluded", "hidden")):
            if cols.get(col) is not None:
                arr[key][fr, pi] = _bool_col(cols[col], where)
            elif key == "visibility":
                arr[key][fr, pi] = has
        if cols.get("radius") is not None:
            arr["radius"][fr, pi] = text_f32(cols["radius"], where)
    return arr, points


_PALETTE = [(255, 77, 77), (77, 210, 255), (255, 210, 77), (140, 255, 120), (220, 120, 255),
            (255, 150, 60), (80, 160, 255), (255, 110, 190)]


# ------------------------------------------------------------------ conveniences (tests, CLI)
def new_id() -> str:
    return uuid.uuid4().hex


def timestamp() -> str:
    """UTC, to the microsecond: two saves in one second must still differ, or
    a recovery could be matched to the wrong save."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _readme(names: list[str], folders: list[str]) -> str:
    cams = "\n".join(f"  {n}: cameras/{f}/" for n, f in zip(names, folders))
    return f"""This folder is a Kinetrace project.

Open it in Kinetrace with File > Open Project... (choose kinetrace.json in
this folder), or read its files with any program: every table is a CSV that
Excel, MATLAB, R or Python open directly. To move the project, move or copy
this WHOLE folder; keep the videos beside it (or in videos/ inside it) and
they are found again on any computer.

Cameras:
{cams}

What is where
  kinetrace.json          which project and which save this is
  project.json            the cameras: videos, frame counts, frame rates, sizes, offsets
  state.json              the window: toggles and panels
  calibration.json        the camera calibration (if any)
  lenses.json             lens profiles (if any)
  reconstruction/points/<landmark>.csv
                          the 3D result: frame,x,y,z,residual,n_cams,<camera>_px
                          (frames of the reference camera, the first one)
  cameras/<camera>/points.csv
                          the landmarks and their settings ("file" = its tracks file)
  cameras/<camera>/tracks/<landmark>.csv
                          frame,x,y,confidence,visible,hand_placed,hidden (+ radius for
                          ball markers); one row per frame that has data
  cameras/<camera>/events.csv, notes.csv
                          marked events and notes on frames
  cameras/<camera>/view.json
                          where you were: frame, zoom, the selected points, timeline zoom
  cameras/<camera>/segment.json, segments/<animal>/segment.json
                          each animal: name, colour, the clicks and boxes of its silhouette,
                          its skeleton (part names) and whether its points are held on it
  cameras/<camera>/spots.json, ball_prompts.json
                          Moving-spot settings per point; the clicks that start each ball marker
  reconstruction/meta.json
                          the 3D result's settings: first frame, unit, cameras
  cameras/<camera>/silhouette/summary.csv
                          the animal's silhouette per frame: frame,area,score,centroid_x,centroid_y,x0,y0,x1,y1
  cameras/<camera>/silhouette/*.npy, body/*.npy
                          outlines, midline and body poses (NumPy arrays)
  exports/                files for DeepLabCut, DLTdv or MATLAB, refreshed at every save
                          when chosen in File > Keep Exports Up to Date
  videos/                 optional: put the videos here to keep everything in one folder
  .cache/                 copies of the tables for fast opening (safe to delete)
  .history/               the files as they were before the last save

Conventions: frames count from 0. Pixels are OpenCV pixel centres counted
from 0 (the centre of the top-left pixel is 0,0; add 1 for MATLAB or DLTdv8).
A blank cell means no data. 1 / 0 = yes / no.

Editing by hand: close the project in Kinetrace first, and save a CSV with
',' separators and '.' decimals. Kinetrace reads an edited CSV the next time
the project opens.

Going back one save: python -m kinetrace.convert previous <this folder>
Full description: docs/FORMAT.md in the Kinetrace folder.
"""


def set_aside(path: str | Path) -> Path:
    """A single-file project where a folder is about to be saved: kept as
    `<name>.bak` (replacing an older .bak)."""
    p = Path(path)
    bak = p.with_name(p.name + ".bak")
    if bak.is_dir():
        shutil.rmtree(bak)
    _replace(p, bak)
    return bak


def write_folder_over(frozen: Frozen, path: str | Path, **kw) -> dict:
    """`write_folder`, where a single-file project may be in the way: that
    file is kept as `<name>.bak`, and comes back if the folder cannot be written."""
    path = Path(path)
    bak = set_aside(path) if path.is_file() else None
    try:
        return write_folder(frozen, path, **kw)
    except BaseException:
        if bak is not None and not (path / META).is_file():
            shutil.rmtree(path, ignore_errors=True)
            _replace(bak, path)
        raise


def save(project, path, state: dict | None = None, project_id: str | None = None, *,
         single_file: bool = False, **kw) -> dict | None:
    """Freeze + write in one call (tests, the converter). Without a state, the
    active camera's toggles are kept as the window-wide ones. A project
    folder by default (an older single file at `path` is kept as .bak);
    `single_file` = the zip form. A folder saved again keeps its project id."""
    path = Path(path)
    if state is None:
        ui = project.sessions[project.active].ui_state
        state = {"tools": {k: v for k, v in ui.items() if k not in VIEW_KEYS}}
    if project_id is None and is_folder_project(path):
        try:
            project_id = read_meta(path).get("project_id")
        except ProjectFileError:
            project_id = None
    pid = project_id or new_id()
    stats = None
    if single_file:
        write(freeze(project, state, pid, target=path, layout="zip"), path, **kw)
    else:
        stats = write_folder_over(freeze(project, state, pid, target=path), path, **kw)
    project.path = str(path)
    project.dirty = False
    return stats


def load(path):
    """-> Project (tests, the converter)."""
    p, _state, _meta = read(path)
    p.path = str(path)
    return p
