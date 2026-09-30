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
    cameras/<folder>/points.csv       one row per landmark (its tracks file in `file`)
    cameras/<folder>/tracks/<landmark>.csv   frame,x,y,confidence,visible,hand_placed,hidden
                                      (+ radius for ball markers); only frames with data
    cameras/<folder>/events.csv, notes.csv, ball_prompts.json, skeleton.json, segment.json
    cameras/<folder>/silhouette/summary.csv + *.npy   per-frame area / centroid / box
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
import os
import re
import shutil
import sys
import time
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from kinetrace import APP_VERSION

FORMAT = "kinetrace-project"
FORMAT_VERSION = 2              # 2 = a folder, one CSV per landmark (I145); 1 = a zip, one tracks.csv per camera
SUFFIX = ".kinetrace"
META = "kinetrace.json"
CACHE_DIR, HISTORY_DIR, SAVING_DIR = ".cache", ".history", ".saving"
EXPORTS_DIR, VIDEOS_DIR = "exports", "videos"
_NOT_DATA = (CACHE_DIR, HISTORY_DIR, SAVING_DIR, EXPORTS_DIR, VIDEOS_DIR)   # never read as project data
# the ui_state entries kept per camera (view.json); every other one is window-wide (state.json)
VIEW_KEYS = ("selected", "zoom", "center_x", "center_y", "user_zoomed", "timeline")
TRACK_COLS = ("frame", "point", "x", "y", "confidence", "visible", "hand_placed", "hidden", "radius")  # format 1
LANDMARK_COLS = ("frame", "x", "y", "confidence", "visible", "hand_placed", "hidden", "radius")
SILHOUETTE_COLS = ("frame", "area", "score", "centroid_x", "centroid_y", "x0", "y0", "x1", "y1")
POINT3D_COLS = ("frame", "x", "y", "z", "residual", "n_cams")         # + one <camera>_px column per camera
POINT_COLS = ("name", "color", "shown", "kind", "radius", "anchor", "source", "spec", "free", "shape", "outline",
              "file")
EVENT_COLS = ("name", "start", "end", "color", "note", "author")
NOTE_COLS = ("frame", "text", "author", "time")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)          # fixed: the same project always gives the same bytes
_CHUNK = 25_000                             # rows formatted per step (keeps the GUI thread responsive)


class ProjectFileError(Exception):
    """A project that cannot be opened; the message names the file (and row)."""


# ------------------------------------------------------------------ small codecs
def f32_text(a: np.ndarray) -> np.ndarray:
    """Shortest text that reads back as exactly the same float32; '' for NaN."""
    a = np.asarray(a, np.float32)
    s = a.astype(str)
    s[np.isnan(a)] = ""
    return s


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


def camera_folders(names: list[str]) -> list[str]:
    """Zip-safe, case-insensitively unique folder names ([A-Za-z0-9._-])."""
    out, seen = [], set()
    for i, n in enumerate(names):
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", unicodedata.normalize("NFKD", n)).strip("._") or f"cam{i + 1}"
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
    a = np.asarray(a, np.float64)
    s = a.astype(str)
    s[np.isnan(a)] = ""
    return s


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
    where = target or getattr(project, "path", None)
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
    r = project.reconstruction
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
    for i, s in enumerate(project.sessions):
        _freeze_camera(files, f"cameras/{folders[i]}", s, binary_tracks)
    if not binary_tracks:
        files["README.txt"] = _readme(project.names, folders)
    return fz


def _freeze_camera(files: dict, d: str, s, binary_tracks: bool) -> None:
    st = s.ui_state
    sel = int(st.get("selected", -1))
    files[f"{d}/view.json"] = _json(_clean_json({
        "current_frame": int(s.current_frame),
        "selected_point": s.points[sel].name if 0 <= sel < s.n_points else None,
        "zoom": st.get("zoom", 0.0), "center_x": st.get("center_x", 0.0), "center_y": st.get("center_y", 0.0),
        "user_zoomed": bool(st.get("user_zoomed", False)),
        "timeline": st.get("timeline"),
        "annotator": s.annotator, "counters": {"points": s._name_counter, "events": s._event_counter}}))
    lfiles = landmark_files([p.name for p in s.points])
    files[f"{d}/points.csv"] = ("points", [
        [p.name, _hex(p.color), int(p.display), p.kind, repr(float(p.radius)), int(p.anchor), p.source,
         p.spec, int(p.free), p.shape,
         "" if not p.outline else " ".join(repr(float(v)) for xy in p.outline for v in xy),
         "" if binary_tracks else lfiles[j]]
        for j, p in enumerate(s.points)])
    arrays = dict(tracks=s.tracks.copy(), confidence=s.confidence.copy(), visibility=s.visibility.copy(),
                  manual=s.manual.copy(), tracked=s.tracked.copy(), occluded=s.occluded.copy(),
                  radius=s.radius.copy())
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
    if s.skeleton:
        files[f"{d}/skeleton.json"] = _json(_clean_json(s.skeleton))
    if s.animal is not None:
        files[f"{d}/segment.json"] = s.animal.to_json() + "\n"
        if s.masks is not None:
            m = s.masks.to_arrays("mask_")
            if not binary_tracks:               # the per-frame numbers as a readable table (I145)
                summ = {k: np.array(m.pop(f"mask_{k}"), copy=True) for k in ("bbox", "area", "centroid", "score")}
                files[f"{d}/silhouette/summary.csv"] = ("sil", summ)
            _put_arrays(files, f"{d}/silhouette", m, "mask_")
    if s.body is not None:
        _put_arrays(files, f"{d}/body", s.body.to_arrays("body_", mesh=True), "body_")


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
    return dict(tracks=np.full((T, N, 2), np.nan, np.float32), confidence=np.zeros((T, N), np.float32),
                visibility=np.zeros((T, N), bool), manual=np.zeros((T, N), bool),
                tracked=np.zeros((T, N), bool), occluded=np.zeros((T, N), bool),
                radius=np.full((T, N), np.nan, np.float32))


def _materialize(item, yield_gil, need_digest: bool = True):
    """A frozen entry -> (content, digest): content is text, an array (.npy)
    or a Table (CSV, formatted only if it is written)."""
    if isinstance(item, tuple):
        kind = item[0]
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
                    data, _ = _materialize(frozen.files[name], yield_gil, need_digest=False)
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
_CAM_FILES = {"view.json", "points.csv", "tracks.csv", "events.csv", "notes.csv", "ball_prompts.json",
              "skeleton.json", "segment.json", "tracks.npy", "confidence.npy", "visibility.npy", "manual.npy",
              "tracked.npy", "occluded.npy", "radius.npy"}
_CAM_DIRS = ("tracks", "silhouette", "body")


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


def _lock_stale(lk: Path) -> bool:
    import socket
    try:
        other = json.loads(lk.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if time.time() - float(other.get("time") or 0) > LOCK_STALE_S:
        return True
    return other.get("host") == socket.gethostname() and not _pid_alive(int(other.get("pid") or 0))


def clear_stale_lock(root: str | Path) -> None:
    """A save that crashed left its .lock: removed when its process is gone."""
    lk = Path(root) / LOCK
    if lk.is_file() and _lock_stale(lk):
        lk.unlink(missing_ok=True)


def _take_lock(root: Path) -> Path:
    """One save at a time per project folder: `.lock` created exclusively
    (process id, computer, time). Two Kinetrace windows saving the same
    project would otherwise clear each other's .saving / .history. A lock
    whose process is gone (on this computer) or that is hours old is taken over."""
    import socket
    lk = root / LOCK
    me = {"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()}
    for _attempt in range(3):
        try:
            fd = os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                other = json.loads(lk.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                other = {}
            if _lock_stale(lk):
                lk.unlink(missing_ok=True)          # a crash's leftover
                continue
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


def _managed(root: Path) -> set[str]:
    """The files of the project layout that are on disk (what a save may
    replace or remove). Anything else in the folder -- videos/, exports/, a
    user's own notes -- is never touched."""
    out = {n for n in _TOP_FILES if (root / n).is_file()}
    rec = root / "reconstruction"
    if rec.is_dir():
        out |= {p.relative_to(root).as_posix() for p in rec.rglob("*") if p.is_file()}
    cams = root / "cameras"
    if cams.is_dir():
        for d in cams.iterdir():
            if not d.is_dir():
                continue
            out |= {p.relative_to(root).as_posix() for p in d.iterdir() if p.is_file() and p.name in _CAM_FILES}
            for sub in _CAM_DIRS:
                if (d / sub).is_dir():
                    out |= {p.relative_to(root).as_posix() for p in (d / sub).rglob("*") if p.is_file()}
    return out


def _rollback(root: Path, pending: dict) -> None:
    """Undo a save that did not finish: every file back as it was."""
    hist = root / HISTORY_DIR
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
        if p.get("saved_at") and _saved_at(root) == p.get("saved_at"):
            _replace(pend, pend.with_name("previous.json"))
            said = "finished"
        else:
            _rollback(root, p)
            said = "undone"
    if (root / SAVING_DIR).exists():
        shutil.rmtree(root / SAVING_DIR, ignore_errors=True)
    return said


def write_folder(frozen: Frozen, path: str | Path, *, fsync: bool = True,
                 yield_gil=lambda: time.sleep(0), **_ignored) -> dict:
    """Save into the project folder, writing ONLY the files whose content
    changed. Crash-safe: new files are written aside (.saving/), the files
    they replace go to .history/ (the previous save), and the new
    kinetrace.json -- one atomic replace -- is the moment the save counts; a
    save cut short before it is undone the next time the project is opened.
    -> {"written", "unchanged", "removed"} counts."""
    root = Path(path)
    if root.exists() and not root.is_dir():
        raise ProjectFileError(f"{root.name} is a file, not a project folder")
    if root.is_dir() and not (root / META).is_file() and any(root.iterdir()):
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
    cache, tmp, hist = root / CACHE_DIR, root / SAVING_DIR, root / HISTORY_DIR
    cache.mkdir(exist_ok=True)
    _hide(cache)
    (cache / "index.json").unlink(missing_ok=True)      # caches untrusted until this save is complete
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    _hide(tmp)
    try:
        return _write_folder_steps(frozen, root, index, old_saved, new_saved, fsync, yield_gil)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)         # whatever happened, nothing is left aside


def _write_folder_steps(frozen, root, index, old_saved, new_saved, fsync, yield_gil) -> dict:
    cache, tmp, hist = root / CACHE_DIR, root / SAVING_DIR, root / HISTORY_DIR
    keep, changed, tables, cols = {}, [], [], {}
    for rel in sorted(frozen.files):
        content, digest = _materialize(frozen.files[rel], yield_gil)
        entry = index.get(rel)
        same = rel != META and entry is not None and entry[2] == digest and _unchanged_on_disk(root / rel, entry)
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
            tables.append((rel, content, digest))       # formatted below, in helpers when there is a lot
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
    # the tables' CSV text: numpy's exact float text holds the GIL, so a big save
    # hands it to helper processes (csvpool.py); whatever they do not finish is
    # formatted here, the same bytes either way
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
    removed = sorted(_managed(root) - set(frozen.files))
    pending = {"saved_at_before": old_saved, "saved_at": new_saved, "removed": removed,
               "replaced": [r for r, _ in changed if r != META and (root / r).exists()],
               "added": [r for r, _ in changed if r != META and not (root / r).exists()]}
    shutil.rmtree(hist, ignore_errors=True)
    hist.mkdir()
    _hide(hist)
    _write_json_atomic(hist / "pending.json", pending, fsync)
    try:
        for rel, _d in changed:
            if rel == META:
                continue
            dst = root / rel
            if dst.exists():
                (hist / rel).parent.mkdir(parents=True, exist_ok=True)
                _replace(dst, hist / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            _replace(tmp / rel, dst)
        for rel in removed:
            (hist / rel).parent.mkdir(parents=True, exist_ok=True)
            _replace(root / rel, hist / rel)
        if (root / META).is_file():
            shutil.copy2(root / META, hist / META)
        if fsync:                                         # every move on disk before the commit
            for d in sorted({(root / r).parent for r, _ in changed} | {(hist / r).parent for r in removed}
                            | {(hist / r).parent for r in pending["replaced"]}):
                _fsync_dir(d)
        _replace(tmp / META, root / META)                 # the save counts from here
        if fsync:
            _fsync_dir(root)
    except BaseException:
        _rollback(root, pending)
        raise
    _replace(hist / "pending.json", hist / "previous.json")
    files = dict(keep)
    for rel, digest in changed:
        st = (root / rel).stat()
        files[rel] = [st.st_size, st.st_mtime_ns, digest] + ([cols[rel]] if rel in cols else [])
    _write_json_atomic(cache / "index.json", {"saved_at": new_saved, "files": files}, fsync=False)
    for rel in removed:
        (cache / (rel + ".npy")).unlink(missing_ok=True)
        d = (root / rel).parent
        while d != root and d.is_dir() and not any(d.iterdir()):     # a removed camera's empty folders
            d.rmdir()
            d = d.parent
    return {"written": len(changed), "unchanged": len(frozen.files) - len(changed), "removed": len(removed)}


_NOT_CHANGES = {META, "README.txt", "state.json"}      # where the user was / derived: never "a change"


def changes_since_save(frozen: Frozen, path: str | Path) -> tuple[list[str], list[str]] | None:
    """What differs from the last save of the project folder at `path`:
    (files whose content would change, files that would be removed), from
    the fingerprints that save left (the same ones a save compares; nothing
    is written). None when that cannot be told (not a folder, no index)."""
    root = project_root(path)
    if not root.is_dir():
        return None
    index = _read_index(root)
    if not index:
        return None
    skip = lambda r: r in _NOT_CHANGES or r.endswith("/view.json")      # noqa: E731
    changed = []
    for rel in sorted(frozen.files):
        if skip(rel):
            continue
        entry = index.get(rel)
        if entry is None or entry[2] != _materialize(frozen.files[rel], lambda: None)[1]:
            changed.append(rel)
    removed = sorted(r for r in index if r not in frozen.files and not skip(r))
    return changed, removed


def describe_changes(changed: list[str], removed: list[str], cameras: list[tuple[str, str]],
                     landmarks: dict[str, dict[str, str]]) -> list[str]:
    """`changes_since_save`'s files in words, one line per camera / part:
    'cam1: P1 (positions), the events'. `cameras` = (folder, name);
    `landmarks` = {folder: {file: landmark name}} (the file a landmark had
    at the last save may be gone: its file stem is used then)."""
    top = {"project.json": "the cameras (videos, offsets, frame rates)", "calibration.json": "the calibration",
           "lenses.json": "the lens profiles"}
    part = {"points.csv": "the landmark list", "events.csv": "the events", "notes.csv": "the notes",
            "ball_prompts.json": "the ball markers' clicks", "skeleton.json": "the skeleton",
            "segment.json": "the segment's clicks"}
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
    for rel in p.get("added", []):
        (root / rel).unlink(missing_ok=True)
    for rel in p.get("replaced", []) + p.get("removed", []):
        if (hist / rel).exists():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            _replace(hist / rel, root / rel)
    _replace(hist / META, root / META)
    (hist / "previous.json").unlink()
    (root / CACHE_DIR / "index.json").unlink(missing_ok=True)
    return str(p.get("saved_at_before") or "")


# ------------------------------------------------------------------ read
class _Source:
    """A project folder or a single file (zip) with the same layout."""

    def __init__(self, path: Path):
        self.path = path
        self.index: dict = {}
        if path.is_dir():
            self.zip = None
            self.names = set()
            for dirpath, dirs, fnames in os.walk(path):
                rel = Path(dirpath).relative_to(path).as_posix()
                if rel == ".":
                    dirs[:] = [d for d in dirs if d not in _NOT_DATA and not d.startswith(".")]
                    rel = ""
                self.names |= {f"{rel}/{f}" if rel else f for f in fnames}
            self.index = _read_index(path)
        else:
            try:
                self.zip = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                raise ProjectFileError(f"{path.name} is not a Kinetrace project (not a zip file, or damaged)") from None
            self.names = set(self.zip.namelist())
        for n in self.names:
            if ".." in PurePosixPath(n).parts or n.startswith("/"):
                raise ProjectFileError(f"{path.name}: unsafe member name {n!r}")

    def has(self, name: str) -> bool:
        return name in self.names

    def bytes(self, name: str) -> bytes:
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
        pre = folder + "/"
        for n in self.names:
            if n.startswith(pre) and "/" not in n[len(pre):]:
                key = n[len(pre):]
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

    def has_dir(self, folder: str) -> bool:
        pre = folder + "/"
        return any(n.startswith(pre) for n in self.names)

    def files_in(self, folder: str) -> list[str]:
        pre = folder + "/"
        return sorted(n for n in self.names if n.startswith(pre) and "/" not in n[len(pre):])


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


def read(path: str | Path):
    """-> (Project, state dict, kinetrace.json dict). Raises ProjectFileError
    with the file (and row) for anything that cannot be read."""
    from kinetrace.calib import Calibration, Reconstruction
    from kinetrace.project import Project
    from kinetrace.session import DEFAULT_UI_STATE, TrackingSession

    root = project_root(path)
    interrupted = None
    if root.is_dir() and (root / LOCK).is_file():
        try:
            clear_stale_lock(root)
        except OSError:
            pass
    if root.is_dir() and (root / HISTORY_DIR / "pending.json").is_file():
        try:
            interrupted = finish_interrupted(root)
        except OSError as e:
            raise ProjectFileError(f"{root.name}: an earlier save of this project was cut short and could not "
                                   f"be undone ({e}). Make the folder writable and open it again.") from None
    src = _Source(root)
    if not src.has("kinetrace.json"):
        raise ProjectFileError(f"{root.name}: no kinetrace.json - not a Kinetrace project")
    meta = src.json("kinetrace.json")
    if meta.get("format") != FORMAT:
        raise ProjectFileError("kinetrace.json: not a Kinetrace project file")
    if int(meta.get("format_version", 0)) > FORMAT_VERSION:
        raise ProjectFileError(f"kinetrace.json: made by a newer Kinetrace (format "
                               f"{meta.get('format_version')}); update Kinetrace to open it")
    pj = src.json("project.json") if src.has("project.json") else None
    if not pj or not pj.get("cameras"):
        raise ProjectFileError("project.json: missing, or no cameras listed")
    state = src.json("state.json") if src.has("state.json") else {}
    if not isinstance(state, dict):
        state = {}                                # a hand-edited state.json: defaults, not a refusal
    if not isinstance(state.get("tools"), dict):
        state["tools"] = {}
    sessions, names, offsets, rates = [], [], [], []
    for k, cam in enumerate(pj["cameras"]):
        folder = cam.get("folder") or camera_folders([cam.get("name", f"cam{k + 1}")])[0]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", folder) or folder.startswith("."):
            raise ProjectFileError(f"project.json: camera folder {folder!r} is not allowed")
        where = f"project.json camera {k + 1}"
        try:
            T, fps = int(cam["n_frames"]), float(cam["fps"])
            w, h = int(cam["width"]), int(cam["height"])
        except (KeyError, TypeError, ValueError):
            raise ProjectFileError(f"{where}: needs n_frames, fps, width and height") from None
        s = TrackingSession(str((cam.get("video") or {}).get("path", "")), T, fps, w, h)
        try:                                     # a project from before G38 has no file_fps: the same rate
            s.file_fps = float(cam.get("file_fps", fps)) or fps
        except (TypeError, ValueError):
            s.file_fps = fps
        s.ui_state = dict(DEFAULT_UI_STATE)
        _read_camera(src, f"cameras/{folder}", s)
        sessions.append(s)
        names.append(str(cam.get("name", folder)))
        offsets.append(float(cam.get("offset", 0.0)))
        rates.append(float(cam.get("rate", 1.0)))
    active = names.index(pj["active_camera"]) if pj.get("active_camera") in names else 0
    p = Project(sessions, names, offsets, active, rates)
    for s in p.sessions:                         # window-wide toggles live once, in state.json
        s.ui_state.update({k: v for k, v in state["tools"].items() if k in DEFAULT_UI_STATE})
    if src.has("calibration.json"):
        c = src.json("calibration.json")
        try:
            coefs = np.asarray(c.pop("coefs"), np.float64)
        except (KeyError, TypeError, ValueError):
            raise ProjectFileError("calibration.json: 'coefs' missing or not 11 numbers per camera") from None
        notes = c.pop("notes", [])
        for cam in c.get("cams") or []:
            if cam.get("rmse") is None:                  # NaN is written as null
                cam["rmse"] = float("nan")
        cal = Calibration.from_arrays({"calib_coefs": coefs, "calib_meta": json.dumps(c)}, "calib_")
        if cal is not None:
            cal.notes = list(notes or [])
            p.calibration = cal
    if src.has("lenses.json"):
        from kinetrace.lens import LensProfile
        raw = src.json("lenses.json")
        if isinstance(raw, list) and len(raw) == len(sessions):
            p.lenses = [None if d is None else LensProfile.from_json(d) for d in raw]
    rec = src.arrays("reconstruction")
    if "xyz" in rec:                              # binary (recovery copies, format 1)
        m = json.loads(rec["meta"])
        p.reconstruction = Reconstruction(int(m["t0"]), list(m["names"]), rec["xyz"].astype(np.float64),
                                          rec["residual"].astype(np.float64), rec["n_cams"].astype(np.int32),
                                          str(m.get("unit", "")),
                                          rec["per_cam"].astype(np.float32) if "per_cam" in rec else None)
    elif src.has("reconstruction/meta.json"):
        p.reconstruction = _read_points3d(src)
    p.exports = [str(x) for x in (pj.get("exports_on_save") or []) if isinstance(x, str)]
    p.dirty = False
    # the app locates each camera's video from these, and says when a cut-short save was undone
    meta = dict(meta, _cameras=pj["cameras"], _interrupted=interrupted)
    return p, state, meta


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


def _read_camera(src: _Source, d: str, s) -> None:
    from kinetrace.body import BodyTrack
    from kinetrace.segmenter import MIDLINE_SAMPLES, MaskTrack
    from kinetrace.session import SOURCES, AnimalMeta, Event, PointMeta

    T = s.n_frames
    # ---- points
    points, pfiles = [], []
    if src.has(f"{d}/points.csv"):
        cols, n = src.table(f"{d}/points.csv", ("name",), POINT_COLS[1:])
        pfiles = list(cols.get("file") or [""] * n)
        for i in range(n):
            g = lambda c, default="": (cols[c][i] if cols.get(c) is not None else default)  # noqa: E731
            where = f"{d}/points.csv row {i + 2}"
            outline = g("outline").split()
            points.append(PointMeta(
                g("name"), _rgb(g("color", "#ffffff"), where), g("shown", "1") != "0", g("kind", "point") or "point",
                float(g("radius", "0") or 0), g("anchor", "0") == "1",
                g("source", "track") if g("source", "track") in SOURCES else "track", g("spec"),
                g("free", "0") == "1", g("shape", "circle") or "circle",
                [[float(outline[j]), float(outline[j + 1])] for j in range(0, len(outline) - 1, 2)] or None))
    index = {p.name: i for i, p in enumerate(points)}
    # ---- tracks (dense .npy in recovery files, one CSV per landmark in projects,
    # one long tracks.csv in format 1)
    if src.has(f"{d}/tracks.npy"):
        arr = {k: src.npy(f"{d}/{k}.npy") for k in
               ("tracks", "confidence", "visibility", "manual", "tracked", "occluded", "radius")}
        for k, v in arr.items():
            if v.shape[0] != T or v.shape[1] != len(points):
                raise ProjectFileError(f"{d}/{k}.npy: shape {v.shape} does not match {T} frames x "
                                       f"{len(points)} points")
    elif src.has_dir(f"{d}/tracks") or not src.has(f"{d}/tracks.csv"):
        derived = landmark_files([p.name for p in points])
        named = [f"{d}/tracks/{(pfiles[j] if j < len(pfiles) and pfiles[j].strip() else derived[j]).strip()}"
                 for j in range(len(points))]
        for rel in src.files_in(f"{d}/tracks"):     # a file points.csv does not name: a new landmark
            if rel.lower().endswith(".csv") and rel not in named:
                points.append(PointMeta(unicodedata.normalize("NFC", PurePosixPath(rel).stem),
                                        _PALETTE[len(points) % len(_PALETTE)]))
                named.append(rel)
        arr = _empty_tracks(T, len(points))
        for j, rel in enumerate(named):
            if src.has(rel):
                _fill_landmark(arr, j, src.load_table(rel, _parse_landmark, LANDMARK_COLS, _LM_KINDS), T, rel)
        index = {p.name: i for i, p in enumerate(points)}
    else:
        cols, n = src.table(f"{d}/tracks.csv", ("frame", "point", "x", "y"), TRACK_COLS[4:])
        arr, points = tracks_from_table(cols, n, points, T, f"{d}/tracks.csv")
        index = {p.name: i for i, p in enumerate(points)}
    s.points = points
    s.tracks, s.confidence, s.visibility = arr["tracks"], arr["confidence"], arr["visibility"]
    s.manual, s.tracked, s.occluded, s.radius = arr["manual"], arr["tracked"], arr["occluded"], arr["radius"]
    # ---- events / notes / prompts / skeleton / segment
    if src.has(f"{d}/events.csv"):
        cols, n = src.table(f"{d}/events.csv", ("name", "start", "end"), EVENT_COLS[3:])
        for i in range(n):
            where = f"{d}/events.csv row {i + 2}"
            try:
                a, b = int(cols["start"][i]), int(cols["end"][i])
            except ValueError:
                raise ProjectFileError(f"{where}: start / end must be whole numbers") from None
            s.events.append(Event(cols["name"][i], min(a, b), max(a, b),
                                  _rgb(cols["color"][i], where) if cols.get("color") else (255, 200, 0),
                                  (cols.get("note") or [""] * n)[i], (cols.get("author") or [""] * n)[i]))
    if src.has(f"{d}/notes.csv"):
        cols, n = src.table(f"{d}/notes.csv", ("frame", "text"), NOTE_COLS[2:])
        for i in range(n):
            if cols["text"][i].strip():
                s.notes[int(cols["frame"][i])] = {"text": cols["text"][i],
                                                  "author": (cols.get("author") or [""] * n)[i],
                                                  "time": (cols.get("time") or [""] * n)[i]}
    if src.has(f"{d}/ball_prompts.json"):
        for nm, per in (src.json(f"{d}/ball_prompts.json") or {}).items():
            if nm in index and s.points[index[nm]].is_ball:
                s.points[index[nm]].ball_prompts = {int(f): [[float(c[0]), float(c[1]), int(c[2])] for c in cs]
                                                    for f, cs in per.items()}
    if src.has(f"{d}/skeleton.json"):
        sk = src.json(f"{d}/skeleton.json")
        if isinstance(sk, dict) and sk.get("landmarks"):
            s.skeleton = sk
    if src.has(f"{d}/segment.json"):
        s.animal = AnimalMeta.from_json(src.text(f"{d}/segment.json"))
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
        s.masks = MaskTrack.from_arrays("", m) if "bbox" in m else MaskTrack(T)
        if s.masks.n_frames != T:
            raise ProjectFileError(f"{d}/silhouette: {s.masks.n_frames} frames, the video has {T}")
        s.masks.native_w, s.masks.native_h = s.width, s.height
    b = src.arrays(f"{d}/body")
    if b:
        s.body = BodyTrack.from_arrays("", b)
        if s.body.n_frames != T:
            raise ProjectFileError(f"{d}/body: {s.body.n_frames} frames, the video has {T}")
    # ---- view
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
    for k in ("zoom", "center_x", "center_y"):
        s.ui_state[k] = view_value(k, float, s.ui_state.get(k, 0.0))
    s.ui_state["user_zoomed"] = bool(v.get("user_zoomed", False))
    if v.get("timeline") is not None:
        s.ui_state["timeline"] = v["timeline"]
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
    from kinetrace.session import PointMeta
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
        for nm in sorted(set(cols["point"]) - set(index), key=cols["point"].index):
            index[nm] = len(points)                   # a name only tracks.csv knows: a new point
            points.append(PointMeta(nm, _PALETTE[len(points) % len(_PALETTE)]))
        pi = np.fromiter(map(index.__getitem__, cols["point"]), np.int64, n)
    N = len(points)
    arr = dict(tracks=np.full((T, N, 2), np.nan, np.float32), confidence=np.zeros((T, N), np.float32),
               visibility=np.zeros((T, N), bool), manual=np.zeros((T, N), bool),
               tracked=np.zeros((T, N), bool), occluded=np.zeros((T, N), bool),
               radius=np.full((T, N), np.nan, np.float32))
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
  cameras/<camera>/silhouette/summary.csv
                          the segment per frame: frame,area,score,centroid_x,centroid_y,x0,y0,x1,y1
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
