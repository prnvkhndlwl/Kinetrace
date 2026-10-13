"""The 3D layer's user interface: the calibration import dialog and the 3D
view window (triangulated landmarks, camera positions, the carved hull).

Rendering is software (numpy + OpenCV into a QImage) so the app keeps its
no-OpenGL, self-contained-folder promise; a few hundred thousand triangles
still draw in well under a second, and the view only redraws on demand.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from kinetrace import theme
from kinetrace.calib import Calibration, CameraCalibration
from kinetrace.calibio import load_calibration
from kinetrace.hull import _view_rotation

CALIB_FILTER = ("Calibration files (*.json *.csv *.mat *.txt *.toml *.yml *.yaml *.xml);;"
                "Kinetrace calibration (*.kcal.json);;"
                "DLTdv / easyWand / Argus DLT coefficients (*.csv);;"
                "easyWand data (*easyWandData.mat);;DLTdv8 project, MATLAB v7 (*dvProject.mat);;"
                "Anipose calibration (*.toml);;OpenCV cameras (*.yml *.yaml *.xml *.json);;"
                "MATLAB cameras (*.mat);;OpenCV cameras K + R/t (*.json *.txt);;All files (*)")

CONVENTIONS = [
    # label, pixel_origin, y_flip
    ("DLTdv8 / easyWand — MATLAB pixels (1-based, y down from the top)", 1.0, False),
    ("DLTdv5-style — y measured from the BOTTOM edge", 1.0, True),
    ("OpenCV / Python — 0-based pixels, y down", 0.0, False),
]


# the importer dispatch lives in calibio (no Qt: the command-line converter uses it too)
load_calibration_file = load_calibration

EXPORT_CAL_FORMATS = [
    # key, label, ticked by default
    ("kcal", "Kinetrace (.kcal.json) + DLT coefficients for DLTdv / easyWand (dltCoefs.csv) + report", True),
    ("anipose", "Anipose / aniposelib (calibration.toml)", False),
    ("opencv_yml", "OpenCV / Python (.yml): K, distortion, R, t per camera", False),
    ("opencv_json", "OpenCV / Python (.json)", False),
    ("matlab", "MATLAB (.mat): K / IntrinsicMatrix, distortion, R / t per camera", False),
    ("blender", "Blender (a Python script that makes the cameras)", False),
    ("lenses", "Lens profiles, one OpenCV .yml per camera that has one", False),
    ("offsets", "Camera offsets and frame rates (offsets.csv)", False),
]


def match_columns(view_sizes, file_cameras, n_views: int) -> list[int]:
    """Which column of a calibration file each project camera starts on (G74).
    A column is matched to a camera by the picture size the file records for
    it: columns whose size only ONE camera has go to that camera first (a
    swapped pair, a rig's lone 848x480 camera), then cameras that share a size
    take the columns of that size in order, and every camera left over takes
    the next column nobody has taken -- so a file with no sizes is simply
    column i for camera i. The old rule only accepted a size hit that was not
    another camera's identity default, which in a swap is always the case.
    (I273) A camera left over when every column is taken gets -1: not calibrated
    (it used to share the last column, which the dialog then refused)."""
    n_cols = len(file_cameras)
    out: list[int | None] = [None] * n_views
    taken: set[int] = set()
    hits = []
    for i in range(n_views):
        size = view_sizes[i] if i < len(view_sizes) else None
        hits.append([] if size is None else
                    [k for k, c in enumerate(file_cameras)
                     if c.width and (int(c.width), int(c.height)) == (int(size[0]), int(size[1]))])
    for i in range(n_views):                        # sizes only this camera has first
        if len(hits[i]) == 1 and hits[i][0] not in taken:
            others = [j for j in range(n_views) if j != i and hits[i][0] in hits[j]]
            if not others:
                out[i] = hits[i][0]
                taken.add(hits[i][0])
    for i in range(n_views):                        # then shared sizes, in order
        if out[i] is None:
            free = [k for k in hits[i] if k not in taken]
            if free:
                out[i] = free[0]
                taken.add(free[0])
    spare = [k for k in range(n_cols) if k not in taken]
    for i in range(n_views):                        # the rest: the next free column
        if out[i] is None:
            out[i] = spare.pop(0) if spare else -1
    return [int(k) for k in out]


class ExportCalibrationDialog(QDialog):
    """File -> Export -> Calibration: which formats. Every file is written with the
    same base name; the conventions are converted (and checked) by the program."""

    def __init__(self, parent, has_lenses: bool):
        super().__init__(parent)
        self.setWindowTitle("Export calibration")
        lay = QVBoxLayout(self)
        head = QLabel("Save the cameras for other programs. Pixel and axis conventions are converted for "
                      "you, and each file is checked against this project's calibration.")
        head.setWordWrap(True)
        lay.addWidget(head)
        self.boxes: dict[str, QCheckBox] = {}
        for key, label, on in EXPORT_CAL_FORMATS:
            cb = QCheckBox(label)
            cb.setChecked(on)
            if key == "lenses" and not has_lenses:
                cb.setEnabled(False)
                cb.setToolTip("No camera of this project has a lens profile")
            self.boxes[key] = cb
            lay.addWidget(cb)
        row = QHBoxLayout()
        allb = QPushButton("Tick everything")
        allb.clicked.connect(lambda: [cb.setChecked(True) for cb in self.boxes.values() if cb.isEnabled()])
        row.addWidget(allb)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Choose where…")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def chosen(self) -> list[str]:
        return [k for k, cb in self.boxes.items() if cb.isChecked() and cb.isEnabled()]


class CalibrationDialog(QDialog):
    """Pick a calibration file, say which column is which camera, and how its
    pixels were counted. Returns a `Calibration` in VIEW order."""

    def __init__(self, parent, view_names: list[str], view_sizes: list[tuple[int, int]],
                 start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import camera calibration")
        self.view_names = list(view_names)
        self.view_sizes = list(view_sizes)
        self._loaded: Calibration | None = None
        self.result_calibration: Calibration | None = None
        lay = QVBoxLayout(self)
        intro = QLabel(
            "The calibration tells the 3D layer where every camera is. Kinetrace reads the "
            "11-parameter DLT that DLTdv, easyWand and Argus produce: a <i>dltCoefs.csv</i> "
            "(one column per camera), an <i>easyWandData.mat</i>, or a DLTdv8 project saved "
            "as MATLAB v7 (which also carries its lens undistortion). Cameras described the "
            "OpenCV way -- a K matrix per camera plus the R / t (or a 4x4 T) between them, as "
            "a JSON or a plain text file -- are converted; the file's origin (usually camera 1) "
            "is moved to a point the cameras look at, because a DLT cannot have a camera at its "
            "origin, and the report says by how much.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        self.path_label = QLabel("no file")
        self.path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        btn = QPushButton("Choose file…")
        btn.clicked.connect(lambda: self._pick(start_dir))
        row.addWidget(btn)
        row.addWidget(self.path_label, 1)
        lay.addLayout(row)
        form = QFormLayout()
        self.conv = QComboBox()
        for label, _, _ in CONVENTIONS:
            self.conv.addItem(label)
        self.conv.setToolTip("How the calibration software counted pixels. Wrong by one pixel or a\n"
                             "flipped y and every 3D point is off — this is the usual mistake.")
        form.addRow("Pixel convention", self.conv)
        lay.addLayout(form)
        self.grid = QGridLayout()
        lay.addLayout(self.grid)
        self._combos: list[QComboBox] = []
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.info)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        lay.addWidget(self.buttons)

    def _pick(self, start_dir: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Calibration file", start_dir, CALIB_FILTER)
        if path:
            self.load(path)

    def load(self, path: str) -> bool:
        try:
            # (I99) no video sizes: a dltCoefs.csv records none, and sizes taken
            # from the videos in view order were shown as "the file's own record"
            # and turned the mismatch safeguard against any non-identity mapping
            cal = load_calibration_file(path)
        except Exception as e:      # noqa: BLE001
            self.info.setText(f"Could not read {Path(path).name}: {e}")
            return False
        self._loaded = cal
        self.path_label.setText(Path(path).name)
        self.path_label.setToolTip(path)
        # a Kinetrace file states its own convention, and a K + R/t file (JSON or
        # the plain-text form) is OpenCV's by construction: no guessing to offer.
        # (I96) the .txt form used to keep the MATLAB default, and every track
        # was triangulated from (x + 1, y + 1) with a perfect-looking residual.
        low = path.lower()
        # K + R/t cameras (Anipose, OpenCV, MATLAB cameras) are always OpenCV pixels;
        # from_krt is the only importer that moves the origin, so it marks them
        self._self_described = low.endswith((".json", ".txt")) or cal.origin_shift is not None
        # (I273) the file's calibrated cameras (a .kcal.json may leave cameras out: null)
        cols = cal.views()
        fcams = [cal.camera(k) for k in cols]
        if fcams:
            # (I171) ALWAYS take the importer's own convention: a self-described file locks it, and
            # a file that is not (dltCoefs.csv, easyWand, a DLTdv project: MATLAB's 1-based pixels)
            # must RESET it -- a previous self-described file left OpenCV's selected, and the
            # next dltCoefs.csv was then imported 0-based: the I96 one-pixel error
            c0 = fcams[0]
            match = [k for k, (_, o, f) in enumerate(CONVENTIONS) if o == c0.pixel_origin and f == c0.y_flip]
            self.conv.setCurrentIndex(match[0] if match else (2 if c0.pixel_origin == 0.0 else 0))
        self.conv.setEnabled(not self._self_described)
        # per-view mapping to a calibration column
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._combos = []
        self.grid.addWidget(QLabel("camera in this project"), 0, 0)
        self.grid.addWidget(QLabel("column in the file"), 0, 1)
        # Default mapping: column i for camera i -- unless the file records
        # picture sizes and exactly one column matches a view's size (e.g. a
        # rig's lone 848x480 camera), in which case that column is taken.
        n_v = len(self.view_names)
        if len(cols) < len(cal.cameras) <= n_v:
            # a file that already says which cameras are not calibrated (a null): camera i = entry i, the
            # cameras past its end not calibrated (a saved calibration does not write trailing gaps), never
            # packed by size - that gave camera 3 camera 4's DLT (review of I273)
            default = [cols.index(i) if i in cols else -1 for i in range(n_v)]
        else:
            default = match_columns(self.view_sizes[:n_v], fcams, n_v)
        for i, nm in enumerate(self.view_names):
            self.grid.addWidget(QLabel(nm), i + 1, 0)
            cb = QComboBox()
            for k, c in zip(cols, fcams):
                size = f" ({c.width}×{c.height})" if c.width else ""
                cb.addItem(f"cam {k + 1}{size}", k)
            cb.addItem("not calibrated", -1)          # (I273) left out of 3D
            cb.setCurrentIndex(default[i] if default[i] >= 0 else cb.count() - 1)
            cb.setToolTip(("Which column of the file is this camera. The sizes in brackets are the "
                           "file's own record of each camera's picture -- match them to the videos."
                           if any(c.width for c in fcams) else
                           "Which column of the file is this camera. The file records no picture sizes: "
                           "use the order your calibration software listed the cameras in.")
                          + " \"not calibrated\" leaves the camera out of 3D.")
            self.grid.addWidget(cb, i + 1, 1)
            self._combos.append(cb)
        und = {c.undistort.kind for c in fcams}
        parts = [f"{len(fcams)} camera(s) in the file, {len(self.view_names)} in the project."]
        if len(fcams) < len(self.view_names):
            parts.append("A camera the file does not cover is marked \"not calibrated\": 3D uses the "
                         "calibrated cameras and leaves it out.")
        if not any(c.width for c in fcams):
            parts.append("This file does not record picture sizes: match the columns by the order your "
                         "calibration software listed the cameras.")
        notes = list(getattr(cal, "notes", []) or [])
        if "lwm" in und:
            parts.append("Lens undistortion (MATLAB local weighted mean) was found and will be applied.")
        elif "opencv" in und:
            parts.append("OpenCV lens coefficients were found and will be applied.")
        elif notes:
            pass                    # the importer says exactly what it could not read (I102)
        elif "dvproject" in low:
            parts.append("This DLTdv project carries no lens undistortion: the tracks are used as filmed. "
                         "Fine for rectilinear lenses; for a GoPro-style lens, add its profile in DLTdv8 "
                         "and save the project again.")
        elif self._self_described:
            parts.append("No lens distortion in this file: the tracks are used as filmed. Fine for "
                         "rectilinear lenses.")
        else:
            parts.append("No undistortion in this file: the tracks are used as filmed. Fine for "
                         "rectilinear lenses; a GoPro-style lens needs the DLTdv project file.")
        parts += notes
        if cal.unit:
            parts.append(f"World unit: {cal.unit}.")
        rm = [c.rmse for c in fcams if np.isfinite(c.rmse)]
        if rm:
            parts.append(f"Calibration RMSE per camera: {', '.join(f'{r:.2f}' for r in rm)} px.")
        self.info.setText(" ".join(parts))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(len(fcams) >= 2)
        return True

    def _accept(self) -> None:
        if self._loaded is None:
            return
        from PySide6.QtWidgets import QMessageBox
        chosen = [int(cb.currentData()) for cb in self._combos]     # the file's entry, -1 = not calibrated
        # Two project cameras on one column is never right: the same DLT would
        # place both cameras at the same spot and the 3D is garbage with a
        # perfectly normal-looking residual. Refuse, do not warn.
        used = [k for k in chosen if k >= 0]
        if len(set(used)) != len(used):
            dup = sorted({k for k in used if used.count(k) > 1})
            QMessageBox.warning(
                self, "Two cameras share a column",
                "Column(s) " + ", ".join(f"cam {k + 1}" for k in dup) + " of the file are assigned to "
                "more than one camera of this project. Every camera needs its own column -- "
                "match them by image size and by the order the calibration software listed them.")
            return
        # A size mismatch between a column and the video it is assigned to is
        # the usual sign of a wrong mapping (e.g. a rig of five 1920x1080
        # cameras and one 848x480). Say which ones, and let the user decide.
        if len(used) < 2:
            QMessageBox.warning(
                self, "Two cameras are needed",
                "Give at least two cameras of this project a column of the file: 3D needs two "
                "calibrated cameras.")
            return
        mism = []
        for i, k in enumerate(chosen):
            if k < 0:
                continue
            src = self._loaded.cameras[k]
            if i < len(self.view_sizes) and src.width and src.height:
                w, h = self.view_sizes[i]
                if (int(w), int(h)) != (int(src.width), int(src.height)):
                    mism.append(f"{self.view_names[i]} ({w}x{h}) <- cam {k + 1} ({src.width}x{src.height})")
        if mism:
            if QMessageBox.question(
                    self, "Image sizes do not match",
                    "The calibration file records a different picture size for these cameras than "
                    "the videos in this project:\n\n  " + "\n  ".join(mism) +
                    "\n\nThat usually means the columns are matched to the wrong cameras (or the "
                    "videos were resized after calibrating). Use this mapping anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        label, origin, flip = CONVENTIONS[self.conv.currentIndex()]
        cams = []
        for i, k in enumerate(chosen):
            if k < 0:                                   # (I273) not calibrated
                cams.append(None)
                continue
            src = self._loaded.cameras[k]
            w, h = self.view_sizes[i] if i < len(self.view_sizes) else (src.width, src.height)
            if getattr(self, "_self_described", False):
                origin, flip = src.pixel_origin, src.y_flip
            cams.append(CameraCalibration(src.coefs.copy(), int(w), int(h), src.undistort,
                                          origin, flip, src.rmse))
        self.result_calibration = Calibration(cams, self._loaded.unit, self._loaded.source)
        # (I96) a K + R/t file moved the world origin; 3D results need the shift
        # to return to the file's frame, so it must survive the re-mapping
        self.result_calibration.origin_shift = self._loaded.origin_shift
        self.result_calibration.notes = list(getattr(self._loaded, "notes", []) or [])
        self.accept()


# ------------------------------------------------------------------ 3D view


class Scene3D:
    """What the 3D window draws: landmarks over time (for a stable framing),
    the current instant's landmarks + bones, camera centres, and a mesh."""

    def __init__(self):
        self.points_all: np.ndarray | None = None      # (T, N, 3) for the framing box
        self.names: list[str] = []
        self.points: np.ndarray | None = None          # (N, 3) at the shown instant
        self.bones: list[tuple[int, int]] = []
        self.cameras: np.ndarray | None = None         # (C, 3)
        self.camera_names: list[str] = []
        self.mesh: tuple[np.ndarray, np.ndarray] | None = None
        self.unit = ""
        self.calib = None               # (G185) calibview.CalibPicture: the wand calibration layer
        self.cover_n = 2                # (G186) the covered space drawn: seen by >= this many cameras
        self.calib_views = None         # the cameras (CamDraw.view) whose pyramid / field of view are drawn

    def framing(self, include_cameras: bool, include_calib: bool = False) -> tuple[np.ndarray, float]:
        """Centre and span to frame: the CURRENT instant's landmarks and hull
        (the animal fills the window and the view follows it through the
        glide); the whole trajectory only when the instant has nothing, and
        the cameras only when asked (they are metres away from a centimetre
        animal, so including them shrinks it to a dot). With the wand calibration layer: the wand cloud and
        the cameras (G185)."""
        if include_calib and self.calib is not None:
            cal = [self.calib.finite_ends()] + [np.array([c.centre for c in self.calib.cams]).reshape(-1, 3)]
            allp = np.concatenate([p for p in cal if len(p)]) if sum(len(p) for p in cal) else np.zeros((1, 3))
            centre = (allp.min(axis=0) + allp.max(axis=0)) / 2
            return centre, max(float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0))), 0.05)
        pts = []
        if self.points is not None and len(self.points):
            p = np.asarray(self.points).reshape(-1, 3)
            pts.append(p[np.isfinite(p).all(axis=1)])
        if self.mesh is not None and len(self.mesh[0]):
            pts.append(self.mesh[0])
        if not sum(len(p) for p in pts) and self.points_all is not None:
            p = self.points_all.reshape(-1, 3)
            pts.append(p[np.isfinite(p).all(axis=1)])
        if include_cameras and self.cameras is not None:
            pts.append(self.cameras)
        if not pts or not sum(len(p) for p in pts):
            return np.zeros(3), 1.0
        allp = np.concatenate([p for p in pts if len(p)])
        centre = (allp.min(axis=0) + allp.max(axis=0)) / 2
        span = float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0)))
        # a lone landmark or a tiny constellation still needs a visible frame
        return centre, max(span, 0.05)


def screen_transform(scene: Scene3D, size: tuple[int, int], azimuth: float, elevation: float,
                     zoom: float = 1.0, pan: tuple[float, float] = (0.0, 0.0), up: int = 2,
                     show_cameras: bool = False, show_calib: bool = False):
    """(to_screen, R, scale): world points -> ((N, 2) pixels of the rendered image, (N,) depth towards the
    viewer), the view rotation, pixels per world unit. THE projection `render_scene` draws with (picking a
    point under the mouse uses the same, G185)."""
    W, H = size
    R = _view_rotation(azimuth, elevation, up)
    centre, span = scene.framing(show_cameras, show_calib)
    scale = 0.85 * min(W, H) / span * zoom

    def to_screen(p):
        q = (np.asarray(p, np.float64).reshape(-1, 3) - centre) @ R.T
        return (np.column_stack([W / 2 + q[:, 0] * scale + pan[0], H / 2 - q[:, 1] * scale + pan[1]]),
                q[:, 2])
    return to_screen, R, scale


# (G185) the wand ends by state (BGR): good, ok, poor, rejected / disagreeing
END_COLOURS = ((110, 210, 110), (10, 214, 255), (40, 140, 255), (70, 70, 240))   # ok = theme.OK (G192)
CAM_COLOURS = ((255, 170, 80), (120, 220, 255), (200, 120, 255), (120, 255, 170), (255, 120, 170),
               (170, 200, 255), (80, 200, 200), (230, 230, 120), (180, 160, 255), (140, 255, 255))


def _nice_length(x: float) -> float:
    """1, 2 or 5 x 10^k at or below x."""
    if not np.isfinite(x) or x <= 0:
        return 0.0
    e = 10.0 ** np.floor(np.log10(x))
    for m in (5.0, 2.0, 1.0):
        if m * e <= x:
            return m * e
    return e


FOV_SIDE_ALPHA = 0.10          # (G187, owner 2026-10-12) each field-of-view side: 10 % of its camera's colour


def _px(v: float, px: float) -> int:
    """A size of `v` screen pixels at the display's pixel density `px` (at least 1; halves round up, never to
    even - the same at 150 % as at 250 %)."""
    return max(1, int(np.floor(v * px + 0.5)))


SCREEN_CLAMP = 1e7              # px: a far-plane point of a ray near 90 degrees, zoomed far in, stays in int32


def _to_int_px(scr: np.ndarray) -> np.ndarray:
    return np.round(np.clip(scr, -SCREEN_CLAMP, SCREEN_CLAMP)).astype(np.int32)


def _draw_poly(img, scr: np.ndarray, colour, closed: bool, thickness: int = 1) -> None:
    ok = np.isfinite(scr).all(axis=1)
    if ok.sum() >= 2:
        cv2.polylines(img, [_to_int_px(scr[ok])], closed, colour, thickness, cv2.LINE_AA)


def _fov_sides(cd, to_screen) -> list:
    """(G187) The four sides of one camera's field of view as (mean depth, (K, 2) int32 screen polygon): the
    camera centre, the border points of that side (the lens bend kept) and the next side's first point, so
    the corner closes."""
    f = cd.fov
    if f is None or f.side is None or not len(f.border):
        return []
    firsts = {s: f.border[f.side == s][0] for s in range(4) if (f.side == s).any()}
    out = []
    for s in range(4):
        pts = f.border[f.side == s]
        if not len(pts):
            continue
        nxt = next((firsts[(s + j) % 4] for j in range(1, 4) if (s + j) % 4 in firsts), None)
        poly = np.vstack([f.centre[None], pts] + ([nxt[None]] if nxt is not None else []))
        scr, depth = to_screen(poly)
        if len(scr) >= 3 and np.isfinite(scr).all():
            out.append((float(depth.mean()), _to_int_px(scr)))
    return out


def _blend_poly(img, poly: np.ndarray, colour, alpha: float) -> None:
    """Fill `poly` at `alpha` over `img`, in place, inside the polygon's own box only."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(poly[:, 0].min())), max(0, int(poly[:, 1].min()))
    x1, y1 = min(W, int(poly[:, 0].max()) + 1), min(H, int(poly[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.fillPoly(layer, [poly - np.array([x0, y0], np.int32)], colour)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_calib(img, scene: Scene3D, to_screen, R, scale: float, show_ends: bool, show_pyramids: bool,
                show_fov: bool, show_cover: bool, hits: dict | None, px: float = 1.0) -> np.ndarray:
    """(G185, G186, G187) the fields of view's translucent sides, the covered space (translucent, over them),
    the fields of view's edges, the cameras' pose pyramids, the wand lines and ends, and the scale bar. Every
    fixed size is in screen pixels times `px` (the display's pixel density). Returns the image (the
    translucent layer makes a new one)."""
    pic = scene.calib
    H, W = img.shape[:2]
    if show_fov:
        faces = []
        for k, cd in enumerate(pic.cams):
            if scene.calib_views is not None and cd.view not in scene.calib_views:
                continue
            col = CAM_COLOURS[k % len(CAM_COLOURS)]
            faces += [(d, poly, col) for d, poly in _fov_sides(cd, to_screen)]
        for d, poly, col in sorted(faces, key=lambda f: f[0]):          # far ones first
            _blend_poly(img, poly, col, FOV_SIDE_ALPHA)
    if show_cover and pic.cover.get(scene.cover_n) is not None and len(pic.cover[scene.cover_n][1]):
        verts, faces = pic.cover[scene.cover_n]
        scr, depth = to_screen(verts)
        order = np.argsort(depth[faces].mean(axis=1))
        tri = np.round(scr[faces]).astype(np.int32)
        fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        light = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
        shade = np.clip(0.35 + 0.65 * np.abs((fn @ R.T) @ light), 0, 1)
        base = np.array([200.0, 150.0, 80.0])                         # BGR: a blue glass
        layer = img.copy()
        for k in order:
            cv2.fillConvexPoly(layer, tri[k], (base * shade[k]).astype(int).tolist(), lineType=cv2.LINE_AA)
        img = cv2.addWeighted(layer, 0.3, img, 0.7, 0)
    for k, cd in enumerate(pic.cams):
        if scene.calib_views is not None and cd.view not in scene.calib_views:
            continue
        col = CAM_COLOURS[k % len(CAM_COLOURS)]
        dim = tuple(int(0.55 * v) for v in col)
        if show_fov and cd.fov is not None and len(cd.fov.border):
            _draw_poly(img, to_screen(cd.fov.border)[0], dim, True, _px(1, px))
            for c0, c1 in cd.fov.rays:
                _draw_poly(img, to_screen(np.vstack([c0, c1]))[0], dim, False, _px(1, px))
        if show_pyramids and cd.pose is not None and len(cd.pose.border):
            _draw_poly(img, to_screen(cd.pose.border)[0], col, True, _px(1, px))            # (G187) 1 px
            for c0, c1 in cd.pose.rays:
                _draw_poly(img, to_screen(np.vstack([c0, c1]))[0], col, False, _px(1, px))
        cs = to_screen(cd.centre)[0][0]
        if (show_pyramids or show_fov) and np.isfinite(cs).all():
            cv2.putText(img, cd.name, (int(cs[0]) + _px(8, px), int(cs[1]) - _px(6, px)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * px, col, _px(1, px), cv2.LINE_AA)
    es, _ = to_screen(pic.ends)
    if hits is not None:
        hits["ends"] = es if show_ends else None
    if not show_ends:
        es = np.full_like(es, np.nan)
    for a, b in pic.pairs:
        if np.isfinite(es[a]).all() and np.isfinite(es[b]).all():
            cv2.line(img, (int(es[a, 0]), int(es[a, 1])), (int(es[b, 0]), int(es[b, 1])), (70, 70, 70), _px(1, px),
                     cv2.LINE_AA)
    order = np.argsort(pic.end_state, kind="stable")             # rejected ones last: on top
    for k in order:
        if np.isfinite(es[k]).all():
            st = int(pic.end_state[k])
            # (G187, owner: "the wand points are too big") radius 1, a rejected end 2, at the display's density
            cv2.circle(img, (int(es[k, 0]), int(es[k, 1])), _px(2 if st == 3 else 1, px), END_COLOURS[st], -1,
                       cv2.LINE_AA)
    L = _nice_length(110.0 * px / max(scale, 1e-12))
    if L > 0:
        n = int(round(L * scale))
        x1, y = W - _px(20, px), H - _px(24, px)
        cv2.line(img, (x1 - n, y), (x1, y), (220, 220, 220), _px(2, px), cv2.LINE_AA)
        for x in (x1 - n, x1):
            cv2.line(img, (x, y - _px(5, px)), (x, y + _px(5, px)), (220, 220, 220), _px(1, px), cv2.LINE_AA)
        lab = f"{L:g} {pic.unit or scene.unit}".strip()
        cv2.putText(img, lab, (x1 - n, y - _px(9, px)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * px, (220, 220, 220),
                    _px(1, px), cv2.LINE_AA)
    return img


def render_scene(scene: Scene3D, size: tuple[int, int], azimuth: float, elevation: float,
                 zoom: float = 1.0, show_mesh: bool = True, show_points: bool = True,
                 show_cameras: bool = False, up: int = 2, pan: tuple[float, float] = (0.0, 0.0),
                 show_calib: bool = False, show_fov: bool = False, show_cover: bool = False,
                 hits: dict | None = None, show_pyramids: bool | None = None, px: float = 1.0) -> np.ndarray:
    """The 3D window's picture (BGR). The wand calibration layers (G185, G186) are drawn only when asked and
    the scene has a picture: `show_calib` = the wand points (and the scale bar), `show_pyramids` the
    cameras' pose pyramids (None = with the wand points), `show_fov` their fields of view, `show_cover`
    the covered space; each on its own (owner 2026-10-12). `hits`, when given, receives "ends" = the wand
    ends' (P, 2) image pixels (None when they are not drawn). `px` (G187): the display's pixel density -
    `size` is in device pixels, `pan` in screen (logical) pixels, and every fixed size (lines, dots, text)
    is scaled by it, so the picture is sharp on a scaled display; px = 1 draws exactly as before."""
    W, H = size
    img = np.full((H, W, 3), (24, 26, 30), np.uint8)
    pyr = show_calib if show_pyramids is None else show_pyramids
    calib_on = (show_calib or pyr or show_fov or show_cover) and scene.calib is not None
    to_screen, R, scale = screen_transform(scene, size, azimuth, elevation, zoom,
                                           (pan[0] * px, pan[1] * px), up, show_cameras, calib_on)
    fs, t1 = 0.45 * px, _px(1, px)                                  # text scale, a 1 px line

    # axes triad (bottom-left), world axes after the same rotation
    o = np.array([48.0 * px, H - 48.0 * px])
    for k, (col, lab) in enumerate((((60, 60, 230), "X"), ((60, 200, 60), "Y"), ((230, 120, 60), "Z"))):
        a = np.zeros(3)
        a[k] = 1.0
        d = (R @ a)[:2] * 32 * px
        tip = (int(o[0] + d[0]), int(o[1] - d[1]))
        cv2.line(img, (int(o[0]), int(o[1])), tip, col, _px(2, px), cv2.LINE_AA)
        cv2.putText(img, lab, (tip[0] + _px(3, px), tip[1] + _px(4, px)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, t1,
                    cv2.LINE_AA)
    if show_mesh and scene.mesh is not None and len(scene.mesh[1]):
        verts, faces = scene.mesh
        fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        light = np.array([0.3, 0.5, 0.8])
        light /= np.linalg.norm(light)
        shade = np.clip(0.25 + 0.75 * np.clip((fn @ R.T) @ light, 0, 1), 0, 1)
        scr, depth = to_screen(verts)
        fd = depth[faces].mean(axis=1)
        order = np.argsort(fd)
        base = np.array([120, 190, 240], np.float64)[::-1]      # BGR of the hull colour
        tri = np.round(scr[faces]).astype(np.int32)
        for k in order:
            c = (base * shade[k]).astype(int).tolist()
            cv2.fillConvexPoly(img, tri[k], c, lineType=cv2.LINE_AA)
    if calib_on:
        img = _draw_calib(img, scene, to_screen, R, scale, show_calib, pyr, show_fov, show_cover, hits, px)
    if show_cameras and not calib_on and scene.cameras is not None and len(scene.cameras):
        cs, _ = to_screen(scene.cameras)
        m = 50 * px
        for k, (x, y) in enumerate(cs):
            if np.isfinite(x) and np.isfinite(y) and -m < x < W + m and -m < y < H + m:
                cv2.drawMarker(img, (int(x), int(y)), (80, 120, 255), cv2.MARKER_TRIANGLE_UP, _px(14, px),
                               _px(2, px), cv2.LINE_AA)
                nm = scene.camera_names[k] if k < len(scene.camera_names) else f"cam {k + 1}"
                cv2.putText(img, nm, (int(x) + _px(8, px), int(y) + _px(4, px)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (120, 150, 255), t1, cv2.LINE_AA)
    if show_points and scene.points is not None and len(scene.points):
        ps, _ = to_screen(scene.points)
        ok = np.isfinite(scene.points).all(axis=1)
        for i, j in scene.bones:
            if i < len(ok) and j < len(ok) and ok[i] and ok[j]:
                cv2.line(img, (int(ps[i, 0]), int(ps[i, 1])), (int(ps[j, 0]), int(ps[j, 1])),
                         (200, 200, 200), t1, cv2.LINE_AA)
        for k, (x, y) in enumerate(ps):
            if ok[k] and np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), _px(4, px), (60, 230, 120), -1, cv2.LINE_AA)
                if k < len(scene.names):
                    cv2.putText(img, scene.names[k], (int(x) + _px(6, px), int(y) - _px(4, px)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * px, (170, 240, 190), t1, cv2.LINE_AA)
    return img


class _CoverageThread(QThread):
    """(G186) `CalibPicture.coverage_for` off the window's thread."""
    done = Signal(object, int)

    def __init__(self, picture, views, gen: int):
        super().__init__()
        self.picture, self.views, self.gen = picture, views, gen

    def run(self):
        try:
            res = self.picture.coverage_for(self.views)
        except Exception as e:      # noqa: BLE001 - said in the info line, never a crash
            res = e
        self.done.emit(res, self.gen)


class View3D(QWidget):
    """Floating 3D window: orbit with the left mouse button, zoom with the
    wheel, pan with the middle button (or Shift+drag), double-click to reset.
    Redraws on demand, coalesced through a timer. With a wand calibration picture (G185, G186): the wand
    points, the camera pyramids, the fields of view and the covered space are separate ticks, the
    Cameras menu chooses which cameras are drawn and counted, a wand end under the mouse says its frame,
    and a click on it emits `end_clicked` (its reference frame)."""

    closed = Signal()
    end_clicked = Signal(int)
    calib_wanted = Signal()         # a calibration tick was switched on and there is no picture yet

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("3D view")
        self.resize(760, 620)
        self.scene = Scene3D()
        self.azimuth, self.elevation, self.zoom = 35.0, 25.0, 1.0
        self.pan = [0.0, 0.0]
        self._drag = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.chk_points = QCheckBox("Landmarks")
        self.chk_points.setChecked(True)
        self.chk_mesh = QCheckBox("Volume hull")
        self.chk_mesh.setChecked(True)
        self.chk_cams = QCheckBox("Cameras")
        self.chk_cams.setChecked(False)
        self.chk_cams.setToolTip("Include the camera positions (zooms the view out to fit them)")
        for c in (self.chk_points, self.chk_mesh, self.chk_cams):
            c.toggled.connect(lambda _=False: self.request_render())
            top.addWidget(c)
        top.addStretch(1)
        self.info = QLabel("")
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.addWidget(self.info, 1)
        lay.addLayout(top)
        # (G185, G186) the wand calibration's layers, each on its own, and the cameras drawn / counted
        self.calib_row = QWidget()
        cr = QHBoxLayout(self.calib_row)
        cr.setContentsMargins(0, 0, 0, 0)
        self.chk_calib = QCheckBox("Wand points")
        self.chk_calib.setToolTip("Every wand position the calibration used, both ends, coloured by error: "
                                  "green good, yellow ok, orange poor, red rejected (or, recomputed from the "
                                  "tracks, disagreeing). Hover one for its frame; click it to go there.")
        self.chk_pyr = QCheckBox("Camera pyramids")
        self.chk_pyr.setToolTip("Where each camera stands and which way it looks")
        self.chk_fov = QCheckBox("Fields of view")
        self.chk_fov.setToolTip("Each camera's whole field of view, its lens included, to a flat far plane past "
                                "the wand: four translucent sides in the camera's colour (denser where fields "
                                "overlap)")
        self.chk_cover = QCheckBox("Covered by ≥")
        self.chk_cover.setToolTip("The space at least this many of the chosen cameras see: where 3D is possible "
                                  "(2) and well determined (3 or more)")
        self.cmb_cover = QComboBox()
        from PySide6.QtWidgets import QMenu, QToolButton
        self.btn_cams = QToolButton()
        self.btn_cams.setText("Cameras ▾")
        self.btn_cams.setPopupMode(QToolButton.InstantPopup)
        self.btn_cams.setToolTip("Choose the cameras whose pyramid and field of view are drawn, and that the "
                                 "covered space is counted from")
        self.cams_menu = QMenu(self)
        self.btn_cams.setMenu(self.cams_menu)
        self.cam_actions: list = []
        for w in (self.chk_calib, self.chk_pyr, self.chk_fov, self.chk_cover, self.cmb_cover, self.btn_cams):
            cr.addWidget(w)
        cr.addStretch(1)
        self.calib_row.setVisible(False)
        lay.addWidget(self.calib_row)
        for c in (self.chk_calib, self.chk_pyr, self.chk_fov, self.chk_cover):
            c.toggled.connect(self._calib_ticked)
        self.cmb_cover.currentIndexChanged.connect(self._cover_n_changed)
        self.canvas = QLabel()
        self.canvas.setMinimumSize(320, 240)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background: #181a1e;")
        self.canvas.setMouseTracking(True)
        self.canvas.installEventFilter(self)
        lay.addWidget(self.canvas, 1)
        hint = QLabel("drag to orbit · wheel to zoom · middle-drag or Shift+drag to pan · double-click to reset")
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(hint)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._render)
        self.last_image: np.ndarray | None = None
        self._last_dpr = 1.0            # the pixel density last_image was drawn at (G187)
        self._hits: dict = {}
        self._cover_threads: list = []          # every worker until it has finished (see _start_coverage)
        self._cover_gen = 0
        self._cover_key = None
        self._base_info = ""

    # ---- data
    def set_scene(self, scene: Scene3D, info: str = "") -> None:
        old = self.scene.calib if self.scene is not None else None
        scene.calib_views = self.scene.calib_views if (self.scene is not None and scene.calib is old) else None
        scene.cover_n = self.scene.cover_n if self.scene is not None else scene.cover_n
        self.scene = scene
        self._base_info = info
        if scene.calib is not old:
            self._picture_changed()
            if self.chk_cover.isChecked() and scene.calib is not None:
                self._start_coverage()            # (review) Covered was ticked before this picture came
        self._show_info()
        self.request_render()

    # ---- the wand calibration layers (G185, G186)
    def set_calib_available(self, on: bool) -> None:
        """Show the calibration ticks (there is a calibration to draw)."""
        self.calib_row.setVisible(bool(on))

    def set_calib_picture(self, picture, ticks: tuple = ()) -> None:
        """Draw `picture` (a calibview.CalibPicture); `ticks` = layers to switch on ("points", "pyramids",
        "fov", "cover")."""
        self.calib_row.setVisible(True)
        self.scene.calib = picture
        self._picture_changed()
        boxes = {"points": self.chk_calib, "pyramids": self.chk_pyr, "fov": self.chk_fov, "cover": self.chk_cover}
        for k in ticks:
            boxes[k].blockSignals(True)
            boxes[k].setChecked(True)
            boxes[k].blockSignals(False)
        self._calib_ticked()                  # counts the covered space when that tick is on

    def _calib_on(self) -> bool:
        return any(c.isChecked() for c in (self.chk_calib, self.chk_pyr, self.chk_fov, self.chk_cover))

    def _picture_changed(self) -> None:
        """A new picture: its cameras in the Cameras menu (all chosen), the covered space to count again."""
        pic = self.scene.calib
        self.cams_menu.clear()
        self.cam_actions = []
        self._cover_key = None
        self._cover_gen += 1
        self._hits = {}                 # the last render's ends belong to the old picture
        if pic is None:
            self.scene.calib_views = None
            return
        for label, on in (("All cameras", True), ("No camera", False)):
            a = self.cams_menu.addAction(label)
            a.triggered.connect(lambda _=False, on=on: self._set_all_cams(on))
        self.cams_menu.addSeparator()
        for cd in pic.cams:
            a = self.cams_menu.addAction(cd.name)
            a.setCheckable(True)
            a.setChecked(True)
            a.setData(cd.view)
            a.toggled.connect(lambda _=False: self._cams_changed())
            self.cam_actions.append(a)
        self.scene.calib_views = None
        self._fill_cover_combo()

    def _set_all_cams(self, on: bool) -> None:
        for a in self.cam_actions:
            a.blockSignals(True)
            a.setChecked(on)
            a.blockSignals(False)
        self._cams_changed()

    def _chosen_views(self):
        if not self.cam_actions or all(a.isChecked() for a in self.cam_actions):
            return None
        return {int(a.data()) for a in self.cam_actions if a.isChecked()}

    def _cams_changed(self) -> None:
        self.scene.calib_views = self._chosen_views()
        self._fill_cover_combo()
        self._cover_key = None
        if self.chk_cover.isChecked():
            self._start_coverage()
        self._show_info()
        self.request_render()

    def _fill_cover_combo(self) -> None:
        n = len(self.cam_actions) if self.scene.calib_views is None else len(self.scene.calib_views)
        keep = self.scene.cover_n
        self.cmb_cover.blockSignals(True)
        self.cmb_cover.clear()
        for k in range(2, n + 1):
            self.cmb_cover.addItem(f"{k} cameras" if k < n else f"all {n}", k)
        i = self.cmb_cover.findData(min(max(keep, 2), max(n, 2)))
        self.cmb_cover.setCurrentIndex(max(i, 0))
        self.cmb_cover.blockSignals(False)
        if self.cmb_cover.count():
            self.scene.cover_n = int(self.cmb_cover.currentData())
        self.chk_cover.setEnabled(n >= 2)
        self.cmb_cover.setEnabled(n >= 2)

    def _cover_n_changed(self, _i=None) -> None:
        if self.cmb_cover.count():
            self.scene.cover_n = int(self.cmb_cover.currentData())
        self.request_render()

    def _calib_ticked(self, _on=None) -> None:
        if self._calib_on() and self.scene.calib is None:
            self.calib_wanted.emit()
        if self.chk_cover.isChecked() and self.scene.calib is not None:
            self._start_coverage()
        self._show_info()
        self.request_render()

    def _start_coverage(self) -> None:
        pic = self.scene.calib
        key = (id(pic), None if self.scene.calib_views is None else tuple(sorted(self.scene.calib_views)))
        if pic is None or key == self._cover_key:
            return
        self._cover_key = key
        self._cover_gen += 1
        th = _CoverageThread(pic, self.scene.calib_views, self._cover_gen)
        th.done.connect(self._coverage_done)
        # every worker stays referenced until it has finished: an earlier one still counting for another
        # choice of cameras is not dropped (a running QThread garbage-collected aborts the app); its result
        # is ignored by generation
        self._cover_threads.append(th)
        th.finished.connect(lambda t=th: self._cover_thread_ended(t))
        self.info.setText((self._base_info + "  ·  " if self._base_info else "") + "counting the covered space…")
        th.start()

    def _cover_thread_ended(self, th) -> None:
        if th in self._cover_threads:
            self._cover_threads.remove(th)
        th.deleteLater()

    def _coverage_done(self, res, gen: int) -> None:
        if gen != self._cover_gen or self.scene.calib is None:
            return
        if isinstance(res, Exception):
            self.info.setText(f"The covered space could not be counted: {res}")
            return
        self.scene.calib.use_coverage(res)
        self._show_info()
        self.request_render()

    def stop_threads(self, wait: bool = True) -> None:
        """The coverage workers: their results are dropped (generation); with `wait`, until all have ended
        (a count takes well under a few seconds; it cannot be interrupted half-way)."""
        self._cover_gen += 1
        self._cover_key = None          # a dropped count is counted again when asked (review)
        if wait:
            for th in list(self._cover_threads):
                th.wait()

    def _show_info(self) -> None:
        pic = self.scene.calib
        txt = self._base_info
        if pic is not None and self._calib_on():
            lines = list(pic.lines)
            if not self.chk_cover.isChecked():
                lines = [ln for ln in lines if not ln.startswith("The wand cloud lies")]
            if lines:
                txt = (txt + "  ·  " if txt else "") + "  ".join(lines)
            self.info.setToolTip("\n".join(pic.lines))
        self.info.setText(txt)

    def end_screen_points(self) -> np.ndarray | None:
        """The wand ends' (P, 2) positions on the canvas in the last picture, in screen (logical) pixels - the
        picture itself is drawn at the display's pixel density (G187) - NaN where not drawn."""
        es = self._hits.get("ends")
        if es is None or self.last_image is None:
            return None
        dpr = self._last_dpr
        h, w = self.last_image.shape[:2]
        ox, oy = (self.canvas.width() - w / dpr) / 2.0, (self.canvas.height() - h / dpr) / 2.0
        return es / dpr + np.array([ox, oy])

    def _pick(self, x: float, y: float, radius: float = 6.0) -> int | None:
        pts = self.end_screen_points()
        if pts is None or not len(pts):
            return None
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        d[~np.isfinite(d)] = np.inf
        k = int(np.argmin(d))
        return k if d[k] <= radius else None

    def eventFilter(self, obj, ev):
        """Mouse on the picture: hover says a wand end's frame and error, a click (no drag) on one emits
        `end_clicked`; dragging orbits / pans as before."""
        if obj is self.canvas:
            t = ev.type()
            if t == QEvent.MouseMove and self._drag is None:
                k = self._pick(ev.position().x(), ev.position().y())
                pic = self.scene.calib
                tip = pic.hover_text(k) if (k is not None and pic is not None and k < len(pic.end_frame)) else ""
                if tip != self.canvas.toolTip():
                    self.canvas.setToolTip(tip)
                    if tip:
                        from PySide6.QtWidgets import QToolTip
                        QToolTip.showText(self.canvas.mapToGlobal(QPoint(int(ev.position().x()) + 12,
                                                                          int(ev.position().y()) + 12)), tip, self.canvas)
                return False
            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._press_at = (ev.position().x(), ev.position().y())
            elif t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                p0 = getattr(self, "_press_at", None)
                self._press_at = None
                if p0 is not None and abs(ev.position().x() - p0[0]) + abs(ev.position().y() - p0[1]) < 3:
                    k = self._pick(ev.position().x(), ev.position().y())
                    pic = self.scene.calib
                    if k is not None and pic is not None and k < len(pic.end_frame):
                        self.end_clicked.emit(int(pic.end_frame[k]))
        return super().eventFilter(obj, ev)


    def request_render(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _render(self) -> None:
        # (G187) drawn at the display's pixel density: a display scaled above 100 % shows it sharp, not stretched
        dpr = max(1.0, float(self.canvas.devicePixelRatioF()))
        w = max(64, self.canvas.width())
        h = max(64, self.canvas.height())
        hits: dict = {}
        img = render_scene(self.scene, (int(round(w * dpr)), int(round(h * dpr))), self.azimuth, self.elevation,
                           self.zoom, self.chk_mesh.isChecked(), self.chk_points.isChecked(),
                           self.chk_cams.isChecked(), pan=tuple(self.pan),
                           show_calib=self.chk_calib.isChecked(), show_fov=self.chk_fov.isChecked(),
                           show_cover=self.chk_cover.isChecked(), hits=hits,
                           show_pyramids=self.chk_pyr.isChecked(), px=dpr)
        self._hits = hits
        self._last_dpr = dpr
        self.last_image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg.copy())
        pm.setDevicePixelRatio(dpr)
        self.canvas.setPixmap(pm)

    def save_png(self, path: str) -> None:
        if self.last_image is None:
            self._render()
        cv2.imwrite(path, self.last_image)

    # ---- interaction
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.request_render()

    def mousePressEvent(self, ev):
        self._drag = (ev.position().x(), ev.position().y(), ev.button(), ev.modifiers())

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        x0, y0, button, mods = self._drag
        dx, dy = ev.position().x() - x0, ev.position().y() - y0
        if button == Qt.MiddleButton or (mods & Qt.ShiftModifier):
            self.pan[0] += dx
            self.pan[1] += dy
        else:
            self.azimuth = (self.azimuth + dx * 0.5) % 360
            self.elevation = float(np.clip(self.elevation + dy * 0.5, -89, 89))
        self._drag = (ev.position().x(), ev.position().y(), button, mods)
        self.request_render()

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def wheelEvent(self, ev):
        self.zoom = float(np.clip(self.zoom * (1.15 if ev.angleDelta().y() > 0 else 1 / 1.15), 0.05, 50))
        self.request_render()

    def mouseDoubleClickEvent(self, ev):
        self.azimuth, self.elevation, self.zoom = 35.0, 25.0, 1.0
        self.pan = [0.0, 0.0]
        self.request_render()

    def closeEvent(self, ev):
        self._cover_gen += 1
        self.stop_threads()
        self.closed.emit()
        super().closeEvent(ev)
