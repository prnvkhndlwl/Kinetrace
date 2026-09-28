"""The Kinetrace project file (`name.kinetrace`): a standard ZIP of plain
files that other programs can read and write. See docs/FORMAT.md.

    kinetrace.json            format, version, app version, project id, saved at, cameras
    project.json              per camera: name, video (path + relative path), frames, fps,
                              size, offset, rate; the active camera
    state.json                window-wide toggles and layout (restored exactly)
    calibration.json          camera calibration (DLT coefficients + conventions), if any
    lenses.json               lens profiles per camera, if any
    reconstruction/*.npy|json the last 3D result, if any
    cameras/<folder>/view.json        this camera's frame, zoom, selection, timeline zoom
    cameras/<folder>/points.csv       one row per point
    cameras/<folder>/tracks.csv       frame,point,x,y,confidence,visible,hand_placed,hidden,radius
                                      (only cells with data; tracks.npy etc. in recovery files)
    cameras/<folder>/events.csv, notes.csv, ball_prompts.json, skeleton.json, segment.json
    cameras/<folder>/silhouette/*.npy  outlines, midline, per-frame summary (binary)
    cameras/<folder>/body/*.npy + meta.json  body poses and mesh (binary)

Tables a person edits or another program writes are CSV / JSON; bulk model
output (silhouettes, body) is .npy, which is what keeps a save fast. Frames
count from 0; pixels are OpenCV pixel centres counted from 0.

No Qt here: `write` runs on a worker thread, `freeze` (cheap copies) on the
GUI thread first, so a save never sees data change under it.
"""
from __future__ import annotations

import csv
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
FORMAT_VERSION = 1
SUFFIX = ".kinetrace"
# the ui_state entries kept per camera (view.json); every other one is window-wide (state.json)
VIEW_KEYS = ("selected", "zoom", "center_x", "center_y", "user_zoomed", "timeline")
TRACK_COLS = ("frame", "point", "x", "y", "confidence", "visible", "hand_placed", "hidden", "radius")
POINT_COLS = ("name", "color", "shown", "kind", "radius", "anchor", "source", "spec", "free", "shape", "outline")
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


def _cell(s: str) -> str:
    """One CSV cell, quoted only when it has to be (comma, quote, newline, or empty)."""
    if s == "" or any(c in s for c in ',"\n\r'):
        return '"' + s.replace('"', '""') + '"'
    return s


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
           binary_tracks: bool = False, saved_at: str | None = None) -> Frozen:
    """Copies of everything to save. Cheap: array copies and small lists;
    formatting happens later, in `write`, off the GUI thread. `target` = the
    file being written (video paths are stored relative to its folder)."""
    fz = Frozen()
    files = fz.files
    folders = camera_folders(project.names)
    saved_at = saved_at or timestamp()
    files["kinetrace.json"] = _json({
        "format": FORMAT, "format_version": FORMAT_VERSION, "app_version": APP_VERSION,
        "project_id": project_id, "saved_at": saved_at,
        "cameras": [{"folder": f, "name": n} for f, n in zip(folders, project.names)],
        "conventions": {"frames": "counted from 0",
                        "pixels": "OpenCV pixel centres, counted from 0 (top-left pixel centre = 0, 0)",
                        "blank": "no data"}})
    where = target or getattr(project, "path", None)
    base = Path(where).resolve().parent if where else None
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
    files["project.json"] = _json({"cameras": cams, "active_camera": project.names[project.active]})
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
    if r is not None:
        rec = {"xyz": r.xyz.astype(np.float64), "residual": r.residual.astype(np.float64),
               "n_cams": r.n_cams.astype(np.int32),
               "meta": json.dumps({"t0": int(r.t0), "names": list(r.names), "unit": str(r.unit)})}
        if r.per_cam is not None:
            rec["per_cam"] = np.asarray(r.per_cam, np.float32)
        _put_arrays(files, "reconstruction", rec, "")
    for i, s in enumerate(project.sessions):
        _freeze_camera(files, f"cameras/{folders[i]}", s, binary_tracks)
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
    files[f"{d}/points.csv"] = ("points", [
        [p.name, _hex(p.color), int(p.display), p.kind, repr(float(p.radius)), int(p.anchor), p.source,
         p.spec, int(p.free), p.shape,
         "" if not p.outline else " ".join(repr(float(v)) for xy in p.outline for v in xy)]
        for p in s.points])
    arrays = dict(tracks=s.tracks.copy(), confidence=s.confidence.copy(), visibility=s.visibility.copy(),
                  manual=s.manual.copy(), tracked=s.tracked.copy(), occluded=s.occluded.copy(),
                  radius=s.radius.copy())
    if binary_tracks:
        for k, v in arrays.items():
            files[f"{d}/{k}.npy"] = v
    else:
        files[f"{d}/tracks.csv"] = ("tracks", arrays, [p.name for p in s.points])
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
            _put_arrays(files, f"{d}/silhouette", s.masks.to_arrays("mask_"), "mask_")
    if s.body is not None:
        _put_arrays(files, f"{d}/body", s.body.to_arrays("body_", mesh=True), "body_")


# ------------------------------------------------------------------ write (worker thread)
def _tracks_csv(arrays: dict, names: list[str], yield_gil) -> str:
    """Long table: one row per (frame, point) whose cell is not all-default."""
    tr, cf, vi = arrays["tracks"], arrays["confidence"], arrays["visibility"]
    ma, tk, oc, ra = arrays["manual"], arrays["tracked"], arrays["occluded"], arrays["radius"]
    keep = tk | ma | oc | vi | np.isfinite(ra) | (cf != 0)
    f_idx, p_idx = np.nonzero(keep)
    qnames = np.array([_cell(n) for n in names] or [""], dtype=object)
    out = [",".join(TRACK_COLS) + "\n"]
    for a in range(0, len(f_idx), _CHUNK):
        f, p = f_idx[a:a + _CHUNK], p_idx[a:a + _CHUNK]
        xy = tr[f, p]
        cols = [f.astype(str), qnames[p], f32_text(xy[:, 0]), f32_text(xy[:, 1]), f32_text(cf[f, p]),
                vi[f, p].astype(np.uint8).astype(str), ma[f, p].astype(np.uint8).astype(str),
                oc[f, p].astype(np.uint8).astype(str), f32_text(ra[f, p])]
        out.append("\n".join(map(",".join, zip(*cols))) + "\n")
        yield_gil()
    return "".join(out)


def _render(item, yield_gil) -> str | np.ndarray:
    if isinstance(item, tuple):
        kind = item[0]
        if kind == "tracks":
            return _tracks_csv(item[1], item[2], yield_gil)
        header = {"points": POINT_COLS, "events": EVENT_COLS, "notes": NOTE_COLS}[kind]
        return _csv_text(header, item[1])
    return item


def write(frozen: Frozen, path: str | Path, *, compresslevel: int = 1, fsync: bool = True,
          backup: bool = True, yield_gil=lambda: time.sleep(0)) -> None:
    """Write atomically: a temp file in the same folder, optionally a copy of
    the previous file as `<name>.bak`, then one os.replace. The file at `path`
    is always either the old save or the new one, never half of either."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            with zipfile.ZipFile(fh, "w", allowZip64=True) as z:
                for name in sorted(frozen.files):
                    data = _render(frozen.files[name], yield_gil)
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


# ------------------------------------------------------------------ read
class _Source:
    """A zip file or an unzipped folder with the same layout."""

    def __init__(self, path: Path):
        self.path = path
        if path.is_dir():
            self.zip = None
            self.names = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
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

    src = _Source(Path(path))
    if not src.has("kinetrace.json"):
        raise ProjectFileError(f"{Path(path).name}: no kinetrace.json - not a Kinetrace project")
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
    if rec:
        m = json.loads(rec["meta"])
        p.reconstruction = Reconstruction(int(m["t0"]), list(m["names"]), rec["xyz"].astype(np.float64),
                                          rec["residual"].astype(np.float64), rec["n_cams"].astype(np.int32),
                                          str(m.get("unit", "")),
                                          rec["per_cam"].astype(np.float32) if "per_cam" in rec else None)
    p.dirty = False
    meta = dict(meta, _cameras=pj["cameras"])       # the app locates each camera's video from these
    return p, state, meta


def _read_camera(src: _Source, d: str, s) -> None:
    from kinetrace.body import BodyTrack
    from kinetrace.segmenter import MaskTrack
    from kinetrace.session import SOURCES, AnimalMeta, Event, PointMeta

    T = s.n_frames
    # ---- points
    points = []
    if src.has(f"{d}/points.csv"):
        cols, n = src.table(f"{d}/points.csv", ("name",), POINT_COLS[1:])
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
    # ---- tracks (dense .npy in recovery files, long CSV in projects)
    if src.has(f"{d}/tracks.npy"):
        arr = {k: src.npy(f"{d}/{k}.npy") for k in
               ("tracks", "confidence", "visibility", "manual", "tracked", "occluded", "radius")}
        for k, v in arr.items():
            if v.shape[0] != T or v.shape[1] != len(points):
                raise ProjectFileError(f"{d}/{k}.npy: shape {v.shape} does not match {T} frames x "
                                       f"{len(points)} points")
    else:
        cols, n = (src.table(f"{d}/tracks.csv", ("frame", "point", "x", "y"), TRACK_COLS[4:])
                   if src.has(f"{d}/tracks.csv") else ({}, 0))
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


def save(project, path, state: dict | None = None, project_id: str | None = None, **kw) -> None:
    """Freeze + write in one call (tests, the converter). Without a state, the
    active camera's toggles are kept as the window-wide ones."""
    if state is None:
        ui = project.sessions[project.active].ui_state
        state = {"tools": {k: v for k, v in ui.items() if k not in VIEW_KEYS}}
    write(freeze(project, state, project_id or new_id(), target=path), path, **kw)
    project.path = str(path)
    project.dirty = False


def load(path):
    """-> Project (tests, the converter)."""
    p, _state, _meta = read(path)
    p.path = str(path)
    return p
