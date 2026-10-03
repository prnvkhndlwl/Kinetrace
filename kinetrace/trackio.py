"""2D tracks from other programs into a Kinetrace camera (no Qt).

Formats, recognised from their contents (`detect`), never from the name:

    DeepLabCut CSV      header rows scorer / bodyparts / coords (x, y[, likelihood]),
                        first column = frame. Single animal: a multi-animal file
                        (an `individuals` row) and a training-data file (image
                        names instead of frames) are refused with a sentence.
    DLTdv / Argus xypts pt{i}_cam{j}_X / _Y columns, one row per frame, NaN =
                        missing; one camera or all of them. First pixel = 1,
                        top-left (DLTdv8), or bottom-left when the
                        `*_pointnames.csv` sidecar says so (older DLTdv, Argus).
    SLEAP analysis CSV  track, frame_idx, instance.score, {node}.x, {node}.y, {node}.score
                        (one track: Kinetrace follows one animal per video).
    Kinetrace tracks    a project folder's tracks/<landmark>.csv (frame, x, y[, confidence,
                        visible, hand_placed, hidden, radius]), or an older project's
                        tracks.csv (frame, point, x, y, ...).

`read(path)` -> `Imported` (positions still in the file's pixel convention);
`apply(session, imported, camera, frame_of_row)` converts to Kinetrace pixels
(OpenCV pixel centres from 0), matches points BY NAME (through the same
sanitiser the exporters write names with; a new name becomes a new point),
writes every cell that has data inside the picture and returns a summary with
a plain sentence. Cells the file has no data for are left as they are. A file
that is recognised but damaged raises `TrackImportError` naming the file.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kinetrace import projectfile

KINDS = {"dlc": "DeepLabCut CSV", "dltdv": "DLTdv / Argus xypts CSV",
         "sleap": "SLEAP analysis CSV", "kinetrace": "Kinetrace tracks (tracks.csv, or one landmark's file)"}
_XYPTS = re.compile(r"^pt(\d+)_cam(\d+)_([xy])$", re.IGNORECASE)


class TrackImportError(Exception):
    """A file that cannot be imported; the message says why in plain words."""


@dataclass
class Imported:
    kind: str                         # a key of KINDS
    names: list[str]
    xy: np.ndarray                    # (C, R, N, 2) float32 in the FILE's pixels, NaN = no data
    conf: np.ndarray                  # (C, R, N) float32
    visible: np.ndarray | None = None  # (C, R, N) bool; None = wherever there is data
    manual: np.ndarray | None = None
    hidden: np.ndarray | None = None
    pixel_origin: float = 0.0         # first pixel's coordinate in the file (DLTdv: 1)
    flip_y: bool = False              # y counted up from the bottom edge (older DLTdv / Argus)
    rows: str = "frames"              # "frames": row k = frame k; "reference": frame k of the reference camera
    notes: list[str] = field(default_factory=list)

    @property
    def n_cameras(self) -> int:
        return int(self.xy.shape[0])

    @property
    def label(self) -> str:
        return KINDS[self.kind]


def _text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise TrackImportError(f"{path.name}: cannot be read ({e.strerror or e})") from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _rows(text: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(text)))
    while rows and not any(c.strip() for c in rows[-1]):
        rows.pop()
    return rows


def detect(path) -> str:
    """The format of a tracks file, from its first lines."""
    path = Path(path)
    return _detect(_text(path), path.name)


def _detect(text: str, name: str) -> str:
    head = text.split("\n", 3)
    first = [c.strip().lower() for c in next(csv.reader([head[0]]))] if head and head[0] else []
    if ";" in (head[0] if head else "") and "," not in head[0]:
        raise TrackImportError(f"{name}: separated by ';' - saved by a spreadsheet set to a "
                               "comma-decimal locale. Save it with ',' separators and '.' decimals.")
    if first[:1] == ["scorer"]:
        return "dlc"
    # (G73) at least one real column: a first line of only commas is no xypts header
    if any(first) and all(_XYPTS.match(c) for c in first if c):
        return "dltdv"
    if "frame_idx" in first and ("node" in first or any(c.endswith(".x") for c in first)):
        return "sleap"
    if first[:4] == ["frame", "point", "x", "y"]:
        return "kinetrace"
    if first[:3] == ["frame", "x", "y"] and "point" not in first:
        return "kinetrace"          # one landmark's file of a project folder: tracks/<name>.csv (I145)
    raise TrackImportError(
        f"{name}: not a tracks file Kinetrace recognises. It reads DeepLabCut CSV (header rows "
        "scorer / bodyparts / coords), DLTdv / Argus xypts CSV (pt1_cam1_X ...), SLEAP analysis CSV "
        "(track, frame_idx, ...) and Kinetrace's own tracks (a project folder's tracks/<landmark>.csv: "
        "frame, x, y; or an older project's tracks.csv: frame, point, x, y).")


def read(path) -> Imported:
    path = Path(path)
    text = _text(path)                  # the file is read once; detect and the reader share it
    kind = _detect(text, path.name)
    reader = {"dlc": _read_dlc, "dltdv": _read_dltdv, "sleap": _read_sleap, "kinetrace": _read_kinetrace}[kind]
    try:
        return reader(path, text)
    except TrackImportError:
        raise
    except (ValueError, IndexError, KeyError, OverflowError) as e:
        # (G73) a recognised file that is not what its header promises: a sentence naming it
        raise TrackImportError(f"{path.name}: looks like a {KINDS[kind]} but cannot be read as one "
                               f"({type(e).__name__}: {e}). Check that the file is complete and was "
                               "not edited by hand.") from None


def _floats(cells, where) -> np.ndarray:
    try:
        return projectfile.text_f32(list(cells), where)
    except projectfile.ProjectFileError as e:
        raise TrackImportError(str(e)) from None


def _frames(cells, where) -> np.ndarray:
    try:
        fr = np.array([int(float(c)) for c in cells], np.int64)
    except (ValueError, OverflowError):
        raise TrackImportError(f"{where}: the first column must be frame numbers") from None
    if (fr < 0).any():
        raise TrackImportError(f"{where}: negative frame numbers")
    return fr


# ------------------------------------------------------------------ DeepLabCut
def _read_dlc(path: Path, text: str) -> Imported:
    rows = _rows(text)
    if len(rows) < 3:
        raise TrackImportError(f"{path.name}: a DeepLabCut CSV needs three header rows")
    first = [(r[0].strip().lower() if r else "") for r in rows[:3]]      # (G73) a blank row has no cells
    if first[1] == "individuals":
        raise TrackImportError(
            f"{path.name}: a multi-animal DeepLabCut file. Kinetrace follows one animal per video: "
            "export each animal separately from DeepLabCut (or filter its CSV to one individual).")
    if first[1] != "bodyparts" or first[2] != "coords":
        raise TrackImportError(f"{path.name}: expected header rows scorer / bodyparts / coords")
    parts, coords = rows[1][1:], [c.strip().lower() for c in rows[2][1:]]
    data = rows[3:]
    if not data:
        raise TrackImportError(f"{path.name}: has the three header rows but no frames below them")
    if data[0] and not re.fullmatch(r"\s*\d+(\.0*)?\s*", data[0][0] or "x"):
        raise TrackImportError(
            f"{path.name}: a DeepLabCut training-data file (image names, not frame numbers). "
            "Import the CSV DeepLabCut writes when it ANALYSES a video (one row per frame).")
    names: list[str] = []
    cols: dict[str, dict[str, int]] = {}
    for k, (bp, c) in enumerate(zip(parts, coords)):
        bp = bp.strip()
        if bp not in cols:
            cols[bp] = {}
            names.append(bp)
        cols[bp][c] = k + 1
    for bp in names:
        if "x" not in cols[bp] or "y" not in cols[bp]:
            raise TrackImportError(f"{path.name}: body part {bp!r} has no x / y columns")
    width = len(rows[0])
    short = next((i for i, r in enumerate(data) if len(r) != width), None)
    if short is not None:
        raise TrackImportError(f"{path.name}: row {short + 4} has {len(data[short])} values, the header {width}")
    fr = _frames([r[0] for r in data], path.name)
    R = int(fr.max()) + 1 if len(fr) else 0
    N = len(names)
    xy = np.full((1, R, N, 2), np.nan, np.float32)
    conf = np.zeros((1, R, N), np.float32)
    colz = list(zip(*data)) if data else []
    for j, bp in enumerate(names):
        xy[0, fr, j, 0] = _floats(colz[cols[bp]["x"]], f"{path.name} {bp} x")
        xy[0, fr, j, 1] = _floats(colz[cols[bp]["y"]], f"{path.name} {bp} y")
        if "likelihood" in cols[bp]:
            lk = _floats(colz[cols[bp]["likelihood"]], f"{path.name} {bp} likelihood")
            conf[0, fr, j] = np.clip(np.nan_to_num(lk, nan=0.0), 0.0, 1.0)
        else:
            conf[0, fr, j] = 1.0
    notes = [f"DeepLabCut scorer: {rows[0][1].strip()}"] if len(rows[0]) > 1 and rows[0][1].strip() else []
    return Imported("dlc", names, xy, conf, notes=notes)


# ------------------------------------------------------------------ DLTdv / Argus
def _sidecar(path: Path, n_pts: int) -> tuple[list[str], bool, list[str]]:
    """(point names by index, bottom-left?, notes) from `<stem>_pointnames.csv`,
    read by its HEADER (I175), CSV-quoted:
      `index,name`   one camera: a row `pt<k>,<name>` per point, then `convention,...`;
      `name,cameras` all cameras: the `n_pts` rows after the header ARE the names in
                     order (whatever they are called: pt1, cam2, rows ...), then
                     `convention,...`, `rows,...` and a `cam<k>,...` row per camera."""
    side = path.with_name(path.stem + "_pointnames.csv")
    if not side.is_file():
        return [], False, [f"No {side.name} beside it: points are named pt1, pt2, ... and the pixels are "
                           "read as DLTdv8's (first pixel = 1, origin top-left)."]
    rows = _rows(_text(side))
    head = [c.strip().lower() for c in rows[0]][:2] if rows else []
    names: list[str] = []
    if head == ["name", "cameras"]:
        body = rows[1:]
        names = [(r[0].strip() if r else "") for r in body[:n_pts]]
        tail = body[n_pts:]
    else:                                                       # single camera: index,name
        tail = []
        by_index: dict[int, str] = {}
        for r in rows[1:]:
            key = r[0].strip() if r else ""
            m = re.fullmatch(r"pt(\d+)", key)
            if m and len(r) > 1:
                by_index[int(m.group(1))] = r[1].strip()
            else:
                tail.append(r)
        names = [by_index.get(k, "") for k in range(1, (max(by_index) if by_index else 0) + 1)]
    flip = any(r and r[0].strip() == "convention" and "bottom-left" in ",".join(r[1:]).lower() for r in tail)
    notes = []
    if any(not n for n in names):                               # a blank name: the point keeps ptK
        notes.append(f"{side.name} leaves some points unnamed: they are called pt<k>, k = their column number.")
    return names, flip, notes


def _read_dltdv(path: Path, text: str) -> Imported:
    rows = _rows(text)
    header = [c.strip() for c in rows[0]]
    idx = {}
    for k, h in enumerate(header):
        m = _XYPTS.match(h)
        if m:
            idx[(int(m.group(1)), int(m.group(2)), m.group(3).lower())] = k
    if not idx:
        raise TrackImportError(f"{path.name}: no pt<k>_cam<j>_X / _Y columns in its first line")
    n_pts = max(p for p, _, _ in idx)
    n_cam = max(c for _, c, _ in idx)
    data = rows[1:]
    short = next((i for i, r in enumerate(data) if len(r) != len(header)), None)
    if short is not None:
        raise TrackImportError(f"{path.name}: row {short + 2} has {len(data[short])} values, the header "
                               f"{len(header)}")
    R = len(data)
    colz = list(zip(*data)) if data else [()] * len(header)
    xy = np.full((n_cam, R, n_pts, 2), np.nan, np.float32)
    for (p, c, ax), k in idx.items():
        xy[c - 1, :, p - 1, 0 if ax == "x" else 1] = _floats(colz[k], f"{path.name} {header[k]}")
    names, flip, notes = _sidecar(path, n_pts)
    names = [names[i] if i < len(names) and names[i] else f"pt{i + 1}" for i in range(n_pts)]
    conf = np.where(np.isfinite(xy).all(-1), 1.0, 0.0).astype(np.float32)
    return Imported("dltdv", names, xy, conf, pixel_origin=1.0, flip_y=flip,
                    rows="reference" if n_cam > 1 else "frames", notes=notes)


# ------------------------------------------------------------------ SLEAP
def _score(s: str, path: Path) -> float:
    """A SLEAP instance score from its cell (G73: a non-number is a sentence, not a ValueError)."""
    try:
        return float(s)
    except ValueError:
        raise TrackImportError(f"{path.name}: the instance score {s.strip()!r} is not a number "
                               "(leave it blank for a hand-labelled instance)") from None


def _read_sleap(path: Path, text: str) -> Imported:
    """SLEAP 1.x's analysis CSV and sleap-io's layouts ("sleap", "instances",
    "points", "frames"), told apart by their headers as sleap-io itself does.
    Columns are found by name (the layouts differ in order). A user-labelled
    instance (blank instance score) is hand-placed and wins over a
    prediction on the same frame; otherwise the higher instance score wins."""
    rows = _rows(text)
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    short = next((i for i, r in enumerate(data) if len(r) != len(header)), None)
    if short is not None:
        raise TrackImportError(f"{path.name}: row {short + 2} has {len(data[short])} values, the header "
                               f"{len(header)}")
    col = {h: [r[k] for r in data] for k, h in enumerate(header)}
    if "node" in col:
        for need in ("x", "y"):
            if need not in col:
                raise TrackImportError(f"{path.name}: has a node column but no '{need}' column")
    n = len(data)
    fr_all = _frames(col["frame_idx"], path.name) if n else np.zeros(0, np.int64)
    blank = [""] * n
    # one record per (instance, node): frame, track, instance score text, node, x, y, node score
    recs: list[tuple] = []
    if "node" in col:                                   # "points": one row per node
        x, y = _floats(col["x"], f"{path.name} x"), _floats(col["y"], f"{path.name} y")
        sc = _floats(col["score"], f"{path.name} score") if "score" in col else np.full(n, np.nan, np.float32)
        tr, inst = col.get("track", blank), col.get("instance_score", blank)
        recs = [(fr_all[i], tr[i], inst[i], col["node"][i], x[i], y[i], sc[i]) for i in range(n)]
        nodes = list(dict.fromkeys(col["node"]))
    else:
        groups: dict[str, dict[str, dict[str, int]]] = {}     # prefix -> node -> coord -> column
        frames_layout = any(re.match(r"^inst\d+\.", h) for h in header)
        for k, h in enumerate(header):
            parts = h.split(".")
            if parts[-1] not in ("x", "y", "score") or h in ("instance.score",) or len(parts) < 2:
                continue
            if frames_layout or (len(parts) >= 3 and f"{parts[0]}.track" in col):
                pre, node = parts[0], ".".join(parts[1:-1])
            else:
                pre, node = "", ".".join(parts[:-1])
            if node:                                    # inst0.score is the instance's, not a node's
                groups.setdefault(pre, {}).setdefault(node, {})[parts[-1]] = k
        nodes = list(dict.fromkeys(nd for g in groups.values() for nd in g))
        if not nodes:
            raise TrackImportError(f"{path.name}: no node columns (head.x, head.y, ...) found")
        cache: dict[int, np.ndarray] = {}

        def fcol(k):
            if k not in cache:
                cache[k] = _floats(col[header[k]], f"{path.name} {header[k]}")
            return cache[k]
        for pre, g in groups.items():
            # frames layout: inst0.track names the track; with track-named prefixes the prefix is it
            tr = col.get(f"{pre}.track" if pre else "track", [pre] * n if pre else blank)
            inst = col.get(f"{pre}.score" if pre else ("instance.score" if "instance.score" in col else "score"),
                           blank)
            for node, cc in g.items():
                if "x" not in cc or "y" not in cc:
                    continue
                xs, ys = fcol(cc["x"]), fcol(cc["y"])
                ss = fcol(cc["score"]) if "score" in cc else np.full(n, np.nan, np.float32)
                recs.extend((fr_all[i], tr[i] or (pre if frames_layout else ""), inst[i], node,
                             xs[i], ys[i], ss[i]) for i in range(n))
    have = [r for r in recs if np.isfinite(r[4]) and np.isfinite(r[5])]
    tracks = sorted({r[1] for r in have})
    if len(tracks) > 1:
        shown = ", ".join(t or "(no track)" for t in tracks[:6]) + (" ..." if len(tracks) > 6 else "")
        raise TrackImportError(
            f"{path.name}: {len(tracks)} tracks ({shown}). Kinetrace follows one animal per video: "
            "export one track per file from SLEAP.")
    # the instance kept per frame: a user label first, then the highest instance score
    best: dict[int, tuple] = {}
    for r in have:
        f, s = int(r[0]), r[2].strip()
        rank = (1, 0.0) if not s else (0, _score(s, path))
        if f not in best or rank > best[f]:
            best[f] = rank
    R = (max(best) + 1) if best else 0
    N = len(nodes)
    ni = {nd: j for j, nd in enumerate(nodes)}
    xy = np.full((1, R, N, 2), np.nan, np.float32)
    conf = np.zeros((1, R, N), np.float32)
    manual = np.zeros((1, R, N), bool)
    for f, tr, inst, node, x, y, s in have:
        f = int(f)
        rank = (1, 0.0) if not inst.strip() else (0, _score(inst, path))
        if rank != best[f]:
            continue
        j = ni[node]
        xy[0, f, j] = (x, y)
        user = rank[0] == 1
        conf[0, f, j] = 1.0 if user or not np.isfinite(s) else float(np.clip(s, 0.0, 1.0))
        manual[0, f, j] = user
    notes = [f"SLEAP track: {tracks[0]}"] if tracks and tracks[0] else []
    return Imported("sleap", nodes, xy, conf, manual=manual, notes=notes)


# ------------------------------------------------------------------ Kinetrace
def _read_kinetrace(path: Path, text: str) -> Imported:
    from kinetrace.session import PALETTE, PointMeta, formula_safe
    head = [c.strip().lower() for c in next(csv.reader([text.split("\n", 1)[0]]))] if text else []
    try:
        if "point" not in head:
            # one landmark's file from a project folder (tracks/<name>.csv): named by the file
            t = projectfile._parse_landmark(text, path.name)
            T = int(t.data["frame"].max()) + 1 if t.n else 0
            arr = projectfile._empty_tracks(T, 1)
            projectfile._fill_landmark(arr, 0, t, T, path.name)
            points = [PointMeta(formula_safe(unicodedata.normalize("NFC", path.stem)), PALETTE[0])]
        else:
            cols, n = projectfile.parse_table(text, path.name, ("frame", "point", "x", "y"),
                                              projectfile.TRACK_COLS[4:])
            points: list = []
            T = (max(map(int, cols["frame"])) + 1) if n else 0
            arr, points = projectfile.tracks_from_table(cols, n, points, T, path.name)
    except (projectfile.ProjectFileError, ValueError) as e:
        raise TrackImportError(str(e)) from None
    assert all(isinstance(p, PointMeta) for p in points)
    return Imported("kinetrace", [p.name for p in points], arr["tracks"][None], arr["confidence"][None],
                    visible=arr["visibility"][None], manual=arr["manual"][None], hidden=arr["occluded"][None])


# ------------------------------------------------------------------ into a session
def apply(session, imp: Imported, camera: int = 0, frame_of_row=None) -> dict:
    """Write camera `camera` of `imp` into `session`. `frame_of_row(r)` = the
    session frame of file row r (None = row r is frame r; a multi-camera
    DLTdv file is in the reference camera's frames, so the app passes the
    project's frame mapping). Returns {new, updated, cells, outside, beyond,
    skipped, duplicates, sentence}: `cells` counts what was WRITTEN, `skipped`
    names the landmarks the file had data for that are not placed from a file
    (silhouette-derived, ball markers)."""
    from kinetrace.session import PALETTE, _sanitize, formula_safe, in_frame
    T, W, H = session.n_frames, session.width, session.height
    xy = imp.xy[camera].astype(np.float64)
    R, N = xy.shape[:2]
    x = xy[..., 0] - imp.pixel_origin
    y = (H - xy[..., 1]) if imp.flip_y else xy[..., 1] - imp.pixel_origin
    rows = np.arange(R)
    fr = rows if frame_of_row is None else np.array(
        [-1 if (f := frame_of_row(int(r))) is None else int(f) for r in rows], np.int64)
    has = np.isfinite(x) & np.isfinite(y)
    # (I254) pixel 0's outer half-pixel counts as pixel 0 (the canvas does the same),
    # then the session's own picture rule: 0 <= x < W, 0 <= y < H
    x = np.where((x >= -0.5) & (x < 0.0), 0.0, x)
    y = np.where((y >= -0.5) & (y < 0.0), 0.0, y)
    inside = in_frame(np.stack([x, y], axis=-1), W, H)
    in_video = (fr >= 0) & (fr < T)
    beyond = int((has & ~in_video[:, None]).sum())
    outside = int((has & in_video[:, None] & ~inside).sum())
    keep = inside & in_video[:, None]

    def key(name: str) -> str:
        return _sanitize(formula_safe(name))        # the form the exporters write a name in (I225)
    index = {key(p.name): i for i, p in enumerate(session.points)}
    claimed: set[int] = set()
    new = updated = cells = duplicates = 0
    skipped: list[tuple[str, str]] = []
    for j, name in enumerate(imp.names):
        rr = np.flatnonzero(keep[:, j])
        if not len(rr):
            continue
        pid = index.get(key(name))
        if pid is not None and (session.points[pid].derived or session.points[pid].is_ball):
            # a silhouette landmark / ball is not placed from a file (G72: and says so)
            skipped.append((session.points[pid].name, "taken from the silhouette"
                            if session.points[pid].derived else "a ball marker, tracked with SAM"))
            continue
        if pid is not None and pid in claimed:
            pid = None                  # (I175) a name the file uses twice: its own point, never a merge
            duplicates += 1
        if pid is None:
            pid = session._new_point(name, "point", color=PALETTE[len(session.points) % len(PALETTE)],
                                     counted=False)
            index.setdefault(key(name), pid)
            new += 1
        else:
            updated += 1
        claimed.add(pid)
        cells += len(rr)
        f = fr[rr]
        session.tracks[f, pid, 0], session.tracks[f, pid, 1] = x[rr, j], y[rr, j]
        session.tracked[f, pid] = True
        session.confidence[f, pid] = imp.conf[camera, rr, j]
        session.visibility[f, pid] = True if imp.visible is None else imp.visible[camera, rr, j]
        session.manual[f, pid] = False if imp.manual is None else imp.manual[camera, rr, j]
        session.occluded[f, pid] = False if imp.hidden is None else imp.hidden[camera, rr, j]
    session._touch()
    parts = [f"{cells} position(s) of {new + updated} point(s) imported from the {imp.label}"
             + (f" ({new} new)" if new else "")]
    skipped = list(dict.fromkeys(skipped))
    if skipped:
        shown = ", ".join(f"{nm} ({why})" for nm, why in skipped[:6]) + (" ..." if len(skipped) > 6 else "")
        parts.append(f"{len(skipped)} landmark(s) in the file were left as they are because they are not "
                     f"placed by hand: {shown}")
    if duplicates:
        parts.append(f"{duplicates} name(s) occur more than once in the file; each repeat became its own "
                     "point (name (2), ...) instead of being merged")
    if outside:
        parts.append(f"{outside} fell outside the picture and were left out - if that is many, check that "
                     "this is the right video and the file's pixel convention")
    if beyond:
        parts.append(f"{beyond} are on frames this video does not have and were left out")
    return {"new": new, "updated": updated, "cells": cells, "outside": outside, "beyond": beyond,
            "skipped": [nm for nm, _ in skipped], "duplicates": duplicates,
            "sentence": "; ".join(parts) + "."}


# ------------------------------------------------------------------ silhouettes
def export_masks_json(session, path) -> int:
    """The segment's outline per frame as polygons (OpenCV pixel centres from
    0): {"frames": {"12": [[[x, y], ...], ...]}, "width", "height", ...}.
    Returns the number of frames written."""
    import json
    m = session.masks
    frames = {} if m is None else {str(f): [p.reshape(-1, 2).tolist() for p in m.contours[f]]
                                    for f in sorted(m.contours) if m.has(f)}
    Path(path).write_text(json.dumps({
        "format": "kinetrace-silhouette", "name": session.animal.name if session.animal else "segment",
        "width": session.width, "height": session.height,
        "pixels": "OpenCV pixel centres, counted from 0, top-left", "frames": frames}), encoding="utf-8")
    return len(frames)


def export_masks_png(session, folder, progress=None) -> int:
    """One black / white PNG per frame with a silhouette (white = the segment),
    named <video name>_<frame, 6 digits>.png. `progress(done, total)` may
    return False to stop. Returns the number of files written."""
    import cv2
    m = session.masks
    frames = [] if m is None else [int(f) for f in m.frames()]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(session.video_path).stem or "mask"
    for k, f in enumerate(frames):
        if progress is not None and progress(k, len(frames)) is False:
            return k
        mask = m.rasterize(f, session.height, session.width)
        if not cv2.imwrite(str(folder / f"{stem}_{f:06d}.png"), mask.astype(np.uint8) * 255):
            raise TrackImportError(f"{folder}: cannot write the PNG files there")
    return len(frames)


def import_masks_png(session, folder, progress=None) -> dict:
    """PNG masks (any image; nonzero = the segment) named with their frame
    number (the last number in the name: mask_000012.png, frame12.png) ->
    the camera's segment. They must be the video's size. -> {frames, sentence}.
    `progress(done, total)` after each file (G52). EVERY file is read and
    checked before the session is touched (I224): one unreadable or wrongly
    sized image refuses the whole folder with nothing stored."""
    import cv2
    from kinetrace.segmenter import summarize_mask
    folder = Path(folder)
    files = []
    for p in sorted(folder.iterdir()) if folder.is_dir() else []:
        if p.suffix.lower() in (".png", ".tif", ".tiff", ".bmp"):
            nums = re.findall(r"\d+", p.stem)
            if nums:
                files.append((int(nums[-1]), p))
    if not files:
        raise TrackImportError(f"{folder.name}: no mask images named with a frame number "
                               "(for example mask_000012.png)")
    T, W, H = session.n_frames, session.width, session.height
    staged: list[tuple[int, dict]] = []     # outlines are compact: the images are not kept
    n = beyond = 0
    for f, p in files:
        if not 0 <= f < T:
            beyond += 1
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise TrackImportError(f"{p.name}: not a readable image")
        if img.shape != (H, W):
            raise TrackImportError(f"{p.name}: {img.shape[1]} x {img.shape[0]} pixels, the video is {W} x {H}. "
                                   "Masks must be the size of the video.")
        staged.append((f, summarize_mask(img > 0, 1.0, 1.0)))
        n += 1
        if progress is not None:
            progress(n + beyond, len(files))
    if staged:
        session.ensure_animal()
        for f, d in staged:
            session.masks.set_summary(f, d)
        session._touch()
    sentence = f"{n} silhouette(s) imported from {folder.name}"
    if beyond:
        sentence += f"; {beyond} file(s) name frames this video does not have and were left out"
    return {"frames": n, "sentence": sentence + "."}
