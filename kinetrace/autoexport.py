"""The project folder's exports/: files for other programs, refreshed at every
save (G42).

File -> Keep Exports Up to Date... ticks the formats (saved in project.json as
`exports_on_save`). After a save of the project folder, `refresh` brings
exports/ up to date FROM THE SAVED FOLDER ITSELF (read back through the cache,
so the files match the save exactly and never race the live session), and
only rewrites a file whose inputs changed: every save leaves a fingerprint per
file in .cache/index.json, and exports/exports.json remembers the
fingerprints each export was made from. The export functions are the ones
Ctrl+E uses (session.py / project.py / calibio.py).

    exports/<camera>_xypts.csv (+ _pointnames.csv)   DLTdv8, one camera
    exports/all_cameras_xypts.csv (+ _pointnames.csv) DLTdv8, every camera, for 3D
    exports/<camera>_DLC.csv                         DeepLabCut
    exports/<camera>.mat                             MATLAB
    exports/<camera>_tracks.csv                      wide CSV
    exports/xyzpts.csv (+ _pointnames.csv)           3D landmarks, DLTdv xyzpts
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from kinetrace import projectfile
from kinetrace.errors import plain_error

# (key, label, file name pattern ({cam} = the camera's folder name), per camera)
FORMATS = (
    ("dltdv_all", "DLTdv8 xypts — every camera in one file, for 3D (offsets applied)", "all_cameras_xypts.csv", False),
    ("dltdv", "DLTdv8 xypts — one file per camera (top-left, first pixel = 1)", "{cam}_xypts.csv", True),
    ("dlc", "DeepLabCut CSV — one file per camera", "{cam}_DLC.csv", True),
    ("mat", "MATLAB .mat — one file per camera", "{cam}.mat", True),
    ("wide", "Wide CSV (x, y, visible per landmark) — one file per camera", "{cam}_tracks.csv", True),
    ("xyz_dltdv", "3D landmarks — DLTdv xyzpts (after 3D → Reconstruct)", "xyzpts.csv", False),
)
KEYS = tuple(f[0] for f in FORMATS)
STATE = "exports.json"


_FOLDER = re.compile(r"[A-Za-z0-9._-]+")
_MADE = [re.compile(_FOLDER.pattern.join(map(re.escape, p.split("{cam}")))) for _k, _l, p, _c in FORMATS]


def _ours(out: Path, name) -> bool:
    """True for a file name this module can have written into `out` (I149):
    exports.json is read from a folder someone else may have made, and every
    name in it that is no longer made is DELETED -- so never a path, never
    anything but one of FORMATS' file names."""
    if not isinstance(name, str) or not _FOLDER.fullmatch(name) or name.startswith("."):
        return False
    if not any(m.fullmatch(name) for m in _MADE):
        return False
    try:
        return (out / name).resolve().parent == out.resolve()
    except (OSError, ValueError):
        return False


def _sidecars(name: str) -> list[str]:
    stem = name.rsplit(".", 1)[0]
    return [f"{stem}_pointnames.csv"] if name.endswith(("xypts.csv", "xyzpts.csv")) else []


def _fingerprint(index: dict, prefixes: tuple, *extra: str) -> str:
    """What an export is made from, as the fingerprints the save left behind;
    '' (always rewrite) when the save left none."""
    if not index:
        return ""
    parts = [f"{rel}={e[2]}" for rel, e in sorted(index.items())
             if rel.startswith(prefixes) and not rel.endswith("/view.json")]
    return projectfile._digest(*extra, *parts)


def refresh(root: str | Path, formats: list[str], scorer: str = "Kinetrace", cancel=lambda: False
            ) -> tuple[list[str], list[str]]:
    """Bring `root`/exports up to date for `formats`. -> (files written,
    problems in words). Files this function made earlier and no longer makes
    (a format unticked, a camera removed) are removed; nothing else is."""
    from kinetrace import calibio
    root = projectfile.project_root(root)
    out = root / projectfile.EXPORTS_DIR
    out.mkdir(exist_ok=True)
    try:
        prev = json.loads((out / STATE).read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError, AttributeError):
        prev = {}
    if not isinstance(prev, dict):
        prev = {}
    prev = {k: v for k, v in prev.items() if _ours(out, k)}
    index = projectfile._read_index(root)
    meta = projectfile.read_meta(root)
    cams = [(c.get("folder"), c.get("name")) for c in meta.get("cameras") or []]
    if not all(isinstance(f, str) and _FOLDER.fullmatch(f) and not f.startswith(".") for f, _n in cams):
        return [], ["kinetrace.json names a camera folder that is not a plain name: exports/ was not refreshed"]
    has_3d = (root / "reconstruction" / "meta.json").is_file()
    jobs = []
    for key, _label, pattern, per_cam in FORMATS:
        if key not in formats:
            continue
        if key == "dltdv_all":
            if len(cams) >= 2:
                jobs.append((pattern, _fingerprint(index, ("cameras/", "project.json"), key),
                             lambda p, path: p.export_multi_dltdv(path)))
        elif key == "xyz_dltdv":
            if has_3d:
                jobs.append((pattern, _fingerprint(index, ("reconstruction/",), key),
                             lambda p, path: calibio.write_points3d(p.reconstruction, path, "dltdv")))
        else:
            for i, (folder, name) in enumerate(cams):
                fp = _fingerprint(index, (f"cameras/{folder}/", "project.json"), key, str(name))
                if key == "dltdv":
                    fn = lambda p, path, i=i: p.sessions[i].export_dltdv_csv(path)       # noqa: E731
                elif key == "dlc":
                    fn = lambda p, path, i=i: p.sessions[i].export_dlc_csv(path, scorer=scorer)   # noqa: E731
                elif key == "mat":
                    fn = lambda p, path, i=i: p.sessions[i].export_mat(path)             # noqa: E731
                else:
                    fn = lambda p, path, i=i: p.sessions[i].export_csv(path)             # noqa: E731
                jobs.append((pattern.format(cam=folder), fp, fn))
    project = None
    written, problems, made = [], [], {}
    for name, fp, fn in jobs:
        if cancel():
            return written, problems
        path = out / name
        if fp and prev.get(name) == fp and path.is_file():
            made[name] = fp
            continue
        try:
            if project is None:
                project = projectfile.load(root)
            fn(project, str(path))
            made[name] = fp
            written.append(name)
        except Exception as e:      # noqa: BLE001 - one format failing never stops the others
            problems.append(plain_error(e, f"{name} not written", short=True))      # in words (G54)
    for name in set(prev) - set(made):
        for f in [name, *_sidecars(name)]:
            (out / f).unlink(missing_ok=True)
    (out / STATE).write_text(json.dumps({"files": made}, indent=1), encoding="utf-8")
    return written, problems


# ------------------------------------------------------------------ Qt: the worker and the dialog
from PySide6.QtCore import QThread, Signal  # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout)  # noqa: E402


class ExportsWorker(QThread):
    """`refresh` off the GUI thread, after a save."""
    done = Signal(list, list)

    def __init__(self, root, formats, scorer):
        super().__init__()
        self._root, self._formats, self._scorer, self._stop = root, formats, scorer, False

    def cancel(self) -> None:
        self._stop = True

    def run(self):
        try:
            w, p = refresh(self._root, self._formats, self._scorer, cancel=lambda: self._stop)
        except Exception as e:      # noqa: BLE001 - said to the user
            w, p = [], [plain_error(e, "exports/ could not be refreshed", short=True)]
        self.done.emit(w, p)


class ExportsDialog(QDialog):
    """File -> Keep Exports Up to Date...: which files to keep in exports/."""

    def __init__(self, parent, current: list[str], n_views: int, has_3d: bool, folder: Path | None):
        super().__init__(parent)
        self.setWindowTitle("Keep exports up to date")
        lay = QVBoxLayout(self)
        where = (f"<b>{Path(folder).name}/exports/</b>" if folder is not None else
                 "the project folder's <b>exports/</b> (save the project as a folder first)")
        intro = QLabel(f"The files ticked here are written into {where} every time you save, so the latest "
                       "tracks are always ready for another program. Only a file whose data changed is "
                       "rewritten, in the background after the save. Everything else stays in Ctrl+E.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self._boxes = {}
        for key, label, _pattern, _per in FORMATS:
            b = QCheckBox(label)
            b.setChecked(key in current)
            if key == "dltdv_all" and n_views < 2:
                b.setEnabled(key in current)
                b.setToolTip("Needs a second camera")
            if key == "xyz_dltdv" and not has_3d:
                b.setToolTip("Written once the project has a 3D result (3D → Reconstruct 3D Landmarks)")
            self._boxes[key] = b
            lay.addWidget(b)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def chosen(self) -> list[str]:
        return [k for k in KEYS if self._boxes[k].isChecked()]
