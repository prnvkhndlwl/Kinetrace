"""Kinetrace's file converter, for scripts and other programs' pipelines.
The same functions as the menus (calibio.py, trackio.py, projectfile.py);
no window, no Qt widgets.

    python -m kinetrace.convert info     FILE
    python -m kinetrace.convert check    FILE                  (exit 0 clean, 1 warnings, 2 errors)
    python -m kinetrace.convert calibration IN OUT [--to F] [--size WxH ...] [--pixels P] [--names a,b,c]
    python -m kinetrace.convert tracks   IN OUT (--video V | --size WxH --frames N) [--to F] [--camera K]
    python -m kinetrace.convert points3d IN OUT [--to F]
    python -m kinetrace.convert masks    PROJECT OUT [--camera NAME] [--to json|png]
    python -m kinetrace.convert offsets  PROJECT OUT
    python -m kinetrace.convert import   PROJECT --tracks FILE [--camera NAME] [--out NEW.kinetrace]
    python -m kinetrace.convert pack     FOLDER OUT.kinetrace

On Windows run it with .venv\\Scripts\\python, on Linux / macOS with
.venv/bin/python. Formats are chosen from --to or else from OUT's name.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CAL_FORMATS = {"anipose": "Anipose calibration.toml", "opencv-yaml": "OpenCV .yml", "opencv-json": "OpenCV .json",
               "matlab": "MATLAB .mat", "blender": "Blender script (.py)", "dlt": "DLT coefficients (dltCoefs.csv, "
               "MATLAB pixels)", "kcal": "Kinetrace .kcal.json"}
TRACK_FORMATS = {"dlc": "DeepLabCut CSV", "dltdv": "DLTdv8 xypts (first pixel 1, top-left)",
                 "dltdv-bottomleft": "DLTdv / Argus xypts, y from the bottom", "wide": "wide CSV",
                 "sparse": "sparse TSV", "kinetrace": "a Kinetrace project (.kinetrace)"}
P3_FORMATS = {"kinetrace": "Kinetrace xyz CSV", "dltdv": "DLTdv xyzpts", "anipose": "Anipose points_3d CSV"}
PIXELS = {"matlab": (1.0, False), "bottom-left": (1.0, True), "opencv": (0.0, False)}


class Failure(Exception):
    """Stop with this sentence (exit 2)."""


def _say(*parts) -> None:
    print(" ".join(str(p) for p in parts), flush=True)


def _guess(out: str, table: dict, by_suffix: dict) -> str:
    low = out.lower()
    for suf, key in by_suffix.items():
        if low.endswith(suf):
            return key
    raise Failure(f"cannot tell the format from {Path(out).name}: give --to (one of {', '.join(table)})")


def _sizes(values, n: int):
    if not values:
        return None
    out = []
    for v in values:
        try:
            w, h = (int(x) for x in v.lower().split("x"))
        except ValueError:
            raise Failure(f"--size {v}: write it as WIDTHxHEIGHT, e.g. 1920x1080") from None
        out.append((w, h))
    if len(out) == 1:
        out = out * n
    if len(out) != n:
        raise Failure(f"{len(out)} --size values for {n} cameras: give one for all, or one per camera")
    return out


# ------------------------------------------------------------------ calibration
def cmd_calibration(a) -> int:
    from kinetrace import calibio
    cal = calibio.load_calibration(a.input)
    if a.pixels:
        po, flip = PIXELS[a.pixels]
        for c in cal.cameras:
            c.pixel_origin, c.y_flip = po, flip
    sizes = _sizes(a.size, len(cal.cameras))
    if sizes:
        for c, (w, h) in zip(cal.cameras, sizes):
            c.width, c.height = w, h
    to = a.to or _guess(a.output, CAL_FORMATS, {".kcal.json": "kcal", ".toml": "anipose", ".yml": "opencv-yaml",
                                                ".yaml": "opencv-yaml", ".json": "opencv-json", ".mat": "matlab",
                                                ".py": "blender", ".csv": "dlt"})
    if to == "kcal":
        Path(a.output).write_text(json.dumps(calibio.calibration_to_kcal(cal), indent=1), encoding="utf-8")
    elif to == "dlt":
        calibio.dlt_csv_matlab(cal, a.output)
        if any(type(c.undistort).__name__ != "NoUndistort" for c in cal.cameras):
            _say("note: these cameras have a lens correction a dltCoefs.csv cannot hold: its coefficients fit "
                 "lens-corrected pixels only")
    else:
        if any(c.width <= 0 or c.height <= 0 for c in cal.cameras):
            raise Failure(f"{Path(a.input).name} does not record the cameras' picture sizes (a dltCoefs.csv "
                          "never does): give --size WIDTHxHEIGHT, once for all cameras or once per camera")
        names = a.names.split(",") if a.names else None
        models = calibio.to_models(cal, names)
        {"anipose": calibio.write_anipose, "opencv-yaml": calibio.write_opencv, "opencv-json": calibio.write_opencv,
         "matlab": calibio.write_matlab, "blender": calibio.write_blender}[to](models, a.output)
        for n in models.notes:
            _say("note:", n)
    _say(f"wrote {a.output} ({CAL_FORMATS[to]}, {len(cal.cameras)} cameras)")
    return 0


# ------------------------------------------------------------------ 2D tracks
def _session_for(a):
    from kinetrace.session import TrackingSession
    if a.video:
        from kinetrace.video_source import probe_video
        try:
            info = probe_video(a.video)
        except (ValueError, OSError) as e:
            raise Failure(f"{a.video}: {e}") from None
        return TrackingSession(info.path, info.n_frames, info.fps, info.width, info.height)
    if not (a.size and a.frames):
        raise Failure("tracks need the video (--video FILE) or its size and length (--size WxH --frames N)")
    (w, h), = _sizes(a.size[:1], 1)
    return TrackingSession("", int(a.frames), float(a.fps), w, h)


def cmd_tracks(a) -> int:
    from kinetrace import projectfile, trackio
    from kinetrace.project import Project
    imp = trackio.read(a.input)
    if not 1 <= a.camera <= imp.n_cameras:
        raise Failure(f"--camera {a.camera}: the file has {imp.n_cameras} camera(s)")
    s = _session_for(a)
    summ = trackio.apply(s, imp, a.camera - 1)
    to = a.to or _guess(a.output, TRACK_FORMATS, {".kinetrace": "kinetrace", ".tsv": "sparse"})
    if to == "dlc":
        s.export_dlc_csv(a.output)
    elif to in ("dltdv", "dltdv-bottomleft"):
        s.export_dltdv_csv(a.output, flip_y=(to == "dltdv-bottomleft"))
    elif to == "wide":
        s.export_csv(a.output)
    elif to == "sparse":
        s.export_tsv_sparse(a.output)
    else:
        projectfile.save(Project([s]), a.output)
    _say(summ["sentence"])
    _say(f"wrote {a.output} ({TRACK_FORMATS[to]})")
    return 0


# ------------------------------------------------------------------ 3D points
def cmd_points3d(a) -> int:
    from kinetrace import calibio
    rec, notes = calibio.read_points3d(a.input)
    to = a.to or "kinetrace"
    calibio.write_points3d(rec, a.output, to)
    for n in notes:
        _say("note:", n)
    _say(f"wrote {a.output} ({P3_FORMATS[to]})")
    return 0


# ------------------------------------------------------------------ project-based
def _camera(project, name: str | None) -> int:
    if name is None:
        return project.active
    if name in project.names:
        return project.names.index(name)
    if name.isdigit() and 1 <= int(name) <= project.n_views:
        return int(name) - 1
    raise Failure(f"no camera {name!r}: the project has {', '.join(project.names)}")


def cmd_masks(a) -> int:
    from kinetrace import projectfile, trackio
    p = projectfile.load(a.project)
    s = p.sessions[_camera(p, a.camera)]
    to = a.to or ("json" if a.output.lower().endswith(".json") else "png")
    n = trackio.export_masks_json(s, a.output) if to == "json" else trackio.export_masks_png(s, a.output)
    _say(f"wrote {n} silhouette frame(s) to {a.output}")
    return 0


def cmd_offsets(a) -> int:
    from kinetrace import calibio, projectfile
    calibio.write_offsets(projectfile.load(a.project), a.output)
    _say(f"wrote {a.output}")
    return 0


def cmd_import(a) -> int:
    from kinetrace import projectfile, trackio
    proj, state, meta = projectfile.read(a.project)
    view = _camera(proj, a.camera)
    imp = trackio.read(a.tracks)
    col = 0
    if imp.n_cameras > 1:
        col = view if imp.n_cameras == proj.n_views else None
        if col is None:
            raise Failure(f"{Path(a.tracks).name} has {imp.n_cameras} cameras, the project {proj.n_views}")
    f_of_row = None if imp.rows == "frames" else (lambda r: proj.map_frame(0, view, r))
    summ = trackio.apply(proj.sessions[view], imp, col, f_of_row)
    proj.sync_landmarks()               # one landmark list for every camera (G19), as the app does
    out = a.out or a.project
    projectfile.write(projectfile.freeze(proj, state, meta.get("project_id") or projectfile.new_id(), target=out),
                      out)
    _say(f"{proj.name(view)}: {summ['sentence']}")
    _say(f"wrote {out}" + (" (the previous version is kept as .bak)" if out == a.project else ""))
    return 0


def cmd_pack(a) -> int:
    from kinetrace import projectfile
    proj, state, meta = projectfile.read(a.folder)
    projectfile.write(projectfile.freeze(proj, state, meta.get("project_id") or projectfile.new_id(),
                                         target=a.output), a.output)
    _say(f"wrote {a.output}: {proj.n_views} camera(s), "
         f"{sum(s.n_points for s in proj.sessions)} point(s)")
    return 0


# ------------------------------------------------------------------ info / check
def _project_lines(path) -> tuple[list[str], list[str]]:
    """(summary lines, warnings) of a project file or folder."""
    from kinetrace import projectfile
    proj, state, meta = projectfile.read(path)
    lines = [f"Kinetrace project, format {meta.get('format_version')}, saved {meta.get('saved_at', '?')} "
             f"by Kinetrace {meta.get('app_version', '?')}"]
    warn = []
    pdir = Path(path).resolve().parent if Path(path).is_file() else Path(path).resolve()
    entries = meta.get("_cameras") or []
    for i, s in enumerate(proj.sessions):
        cells = int(s.tracked.sum())
        lines.append(f"  camera {i + 1} {proj.name(i)!r}: {s.n_frames} frames, {s.fps:g} fps, {s.width}x{s.height}, "
                     f"offset {proj.offsets[i]:g}, rate {proj.rates[i]:g}; {s.n_points} point(s), {cells} "
                     f"position(s), {len(s.events)} event(s)"
                     + (f", silhouette on {s.masks.n_masked()} frame(s)" if s.masks is not None else "")
                     + (" [active]" if i == proj.active else ""))
        entry = entries[i] if i < len(entries) else {"video": {"path": s.video_path}}
        if projectfile.locate_video(entry, pdir) is None:
            warn.append(f"camera {i + 1} {proj.name(i)!r}: its video is not found "
                        f"({(entry.get('video') or {}).get('relative_path') or s.video_path})")
    if proj.calibration is not None:
        lines.append(f"  calibration: {len(proj.calibration)} camera(s), unit {proj.calibration.unit or '?'}, "
                     f"from {proj.calibration.source or '?'}")
        if len(proj.calibration) != proj.n_views:
            warn.append(f"the calibration has {len(proj.calibration)} cameras, the project {proj.n_views}")
    if proj.reconstruction is not None:
        r = proj.reconstruction
        lines.append(f"  3D: {len(r.names)} landmark(s), reference frames {r.t0} - {r.t0 + r.n_frames - 1}")
    return lines, warn


def _describe(path: str) -> tuple[list[str], list[str]]:
    from kinetrace import calibio, trackio
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".kinetrace") or (p.is_dir() and (p / "kinetrace.json").is_file()):
        return _project_lines(path)
    if low.endswith(".csv"):
        try:
            imp = trackio.read(path)
        except trackio.TrackImportError as e_tracks:
            try:
                rec, notes = calibio.read_points3d(path)
                return [f"3D points: {notes[0]}"] + notes[1:], []
            except calibio.CalibFormatError:
                pass
            try:
                cal = calibio.load_calibration(path)
                return [f"DLT coefficients: {len(cal.cameras)} camera(s) (a dltCoefs.csv records no picture "
                        "sizes)"], []
            except Exception:  # noqa: BLE001
                raise Failure(str(e_tracks)) from None
        import numpy as np
        n = int(np.isfinite(imp.xy).all(axis=-1).sum())
        return [f"{imp.label}: {imp.n_cameras} camera(s), {len(imp.names)} point(s) "
                f"({', '.join(imp.names[:8])}{' ...' if len(imp.names) > 8 else ''}), "
                f"{imp.xy.shape[1]} row(s), {n} position(s)"] + imp.notes, []
    cal = calibio.load_calibration(path)
    lines = [f"calibration ({cal.source or p.name}): {len(cal.cameras)} camera(s), unit {cal.unit or '?'}"]
    lines += [f"  camera {i + 1}: {c.width}x{c.height}, pixels from {c.pixel_origin:g}"
              f"{', y from the bottom' if c.y_flip else ''}, lens correction: {getattr(c.undistort, 'kind', 'none')}"
              for i, c in enumerate(cal.cameras)]
    return lines, cal.notes


def cmd_info(a) -> int:
    lines, warn = _describe(a.file)
    for ln in lines:
        _say(ln)
    for w in warn:
        _say("warning:", w)
    return 0


def cmd_check(a) -> int:
    try:
        lines, warn = _describe(a.file)
    except Exception as e:  # noqa: BLE001 - every failure is a finding here
        _say("error:", e)
        return 2
    for w in warn:
        _say("warning:", w)
    _say("OK" if not warn else f"{len(warn)} warning(s)")
    return 1 if warn else 0


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(prog="python -m kinetrace.convert", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="what a file holds")
    s.add_argument("file")
    s = sub.add_parser("check", help="validate a file made elsewhere (exit 0 clean, 1 warnings, 2 errors)")
    s.add_argument("file")
    s = sub.add_parser("calibration", help="a calibration into another program's format")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--to", choices=list(CAL_FORMATS))
    s.add_argument("--size", action="append", help="WIDTHxHEIGHT, once for all cameras or once per camera "
                                                   "(a dltCoefs.csv records no sizes)")
    s.add_argument("--pixels", choices=list(PIXELS), help="how a dltCoefs.csv counted pixels (default matlab)")
    s.add_argument("--names", help="camera names, comma separated")
    s = sub.add_parser("tracks", help="2D tracks from one program's format into another's")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--to", choices=list(TRACK_FORMATS))
    s.add_argument("--video", help="the video the tracks belong to (its size and length)")
    s.add_argument("--size", action="append", help="WIDTHxHEIGHT of the video, without --video")
    s.add_argument("--frames", type=int, help="number of frames, without --video")
    s.add_argument("--fps", type=float, default=30.0, help="frame rate, without --video (default 30)")
    s.add_argument("--camera", type=int, default=1, help="which camera of an all-cameras file (from 1)")
    s = sub.add_parser("points3d", help="3D points between Kinetrace, DLTdv xyzpts and Anipose")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--to", choices=list(P3_FORMATS))
    s = sub.add_parser("masks", help="a project's silhouettes as polygons (JSON) or PNG masks")
    s.add_argument("project")
    s.add_argument("output", help="a .json file, or a folder for PNGs")
    s.add_argument("--camera", help="camera name or number (default: the active one)")
    s.add_argument("--to", choices=["json", "png"])
    s = sub.add_parser("offsets", help="a project's camera offsets and rates as CSV")
    s.add_argument("project")
    s.add_argument("output")
    s = sub.add_parser("import", help="tracks from another program into a project file")
    s.add_argument("project")
    s.add_argument("--tracks", required=True)
    s.add_argument("--camera", help="camera name or number (default: the active one)")
    s.add_argument("--out", help="write a new project instead of updating this one")
    s = sub.add_parser("pack", help="an unzipped project folder into one .kinetrace file")
    s.add_argument("folder")
    s.add_argument("output")
    a = ap.parse_args(argv)
    from kinetrace import calibio, projectfile, trackio
    try:
        return {"info": cmd_info, "check": cmd_check, "calibration": cmd_calibration, "tracks": cmd_tracks,
                "points3d": cmd_points3d, "masks": cmd_masks, "offsets": cmd_offsets, "import": cmd_import,
                "pack": cmd_pack}[a.cmd](a)
    except (Failure, projectfile.ProjectFileError, trackio.TrackImportError, calibio.CalibFormatError,
            ValueError, OSError) as e:
        _say("error:", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
