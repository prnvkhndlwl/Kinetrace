"""The 3D layer's user interface: the calibration import dialog and the 3D
view window (triangulated landmarks, camera positions, the carved hull).

Rendering is software (numpy + OpenCV into a QImage) so the app keeps its
no-OpenGL, self-contained-folder promise; a few hundred thousand triangles
still draw in well under a second, and the view only redraws on demand.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from kinetrace import theme
from kinetrace.canvas import bgr_pixmap
from kinetrace.calib import PIXEL_CONVENTIONS, Calibration, CameraCalibration
from kinetrace.calibio import load_calibration
from kinetrace.hull import _view_rotation

CALIB_FILTER = ("Calibration files (*.json *.csv *.mat *.txt *.toml *.yml *.yaml *.xml);;"
                "Kinetrace calibration (*.kcal.json);;"
                "DLTdv / easyWand / Argus DLT coefficients (*.csv);;"
                "easyWand data (*easyWandData.mat);;DLTdv8 project (*dvProject.mat);;"
                "Anipose calibration (*.toml);;OpenCV cameras (*.yml *.yaml *.xml *.json);;"
                "MATLAB cameras (*.mat);;OpenCV cameras K + R/t (*.json *.txt);;All files (*)")

CONVENTIONS = [
    # label, pixel_origin, y_flip (R32: calib.PIXEL_CONVENTIONS, in the same order)
    ("DLTdv8 / easyWand — MATLAB pixels (1-based, y down from the top)", *PIXEL_CONVENTIONS["matlab"]),
    ("DLTdv5-style — y measured from the BOTTOM edge", *PIXEL_CONVENTIONS["bottom-left"]),
    ("OpenCV / Python — 0-based pixels, y down", *PIXEL_CONVENTIONS["opencv"]),
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
            "(one column per camera), an <i>easyWandData.mat</i>, or a DLTdv8 project "
            "(<i>…_dvProject.mat</i>, which also carries its lens undistortion; to bring in its "
            "videos and clicks too, use File → Import → DLTdv Project…). Cameras described the "
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
        self.conv.setPlaceholderText("Choose how this file numbers pixels")
        self.conv.setCurrentIndex(-1)
        self.conv.setToolTip("How the calibration software counted pixels. Wrong by one pixel or a\n"
                             "flipped y and every 3D point is off — this is the usual mistake.")
        self.conv.currentIndexChanged.connect(lambda _i: self._update_ok())
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
        # (I278) a Kinetrace file, a K + R/t / OpenCV / Anipose / MATLAB-cameras file and a DLTdv8
        # project state their convention (the importer says so: `convention_stated`); a bare
        # dltCoefs.csv (DLTdv, easyWand or Argus wrote it) and an easyWand file (y from the top
        # or the bottom) do not, and the user is ASKED -- never a default (I96: a guessed
        # convention moved every 3D point by a pixel with a perfect-looking residual)
        low = path.lower()
        self._self_described = bool(getattr(cal, "convention_stated", True))
        # (I273) the file's calibrated cameras (a .kcal.json may leave cameras out: null)
        cols = cal.views()
        fcams = [cal.camera(k) for k in cols]
        if fcams and self._self_described:
            c0 = fcams[0]
            match = [k for k, (_, o, f) in enumerate(CONVENTIONS) if o == c0.pixel_origin and f == c0.y_flip]
            self.conv.setCurrentIndex(match[0] if match else -1)
        else:
            # (I171) a file that does not say RESETS the choice: the previous file's must not carry over
            self.conv.setCurrentIndex(-1)
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
        if "dltdv" in und:
            parts.append("DLTdv's own lens undistortion was found and will be applied exactly as DLTdv applies it"
                         + (" (a camera's profile matches no DLTdv model: it is approximated, see below)."
                            if "lwm" in und else "."))
        elif "lwm" in und:
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
        if not self._self_described:
            parts.append("This file does not say how its pixels were numbered, so choose it above: DLTdv8 and "
                         "easyWand (MATLAB) count the first pixel as 1, y down from the top; files from older "
                         "DLTdv versions may measure y from the bottom edge; OpenCV and Python count from 0. "
                         "A wrong choice moves every 3D point by about a pixel's worth (or turns the picture "
                         "upside down) while the residual still looks normal, so Kinetrace asks instead of "
                         "guessing.")
        if cal.unit:
            parts.append(f"World unit: {cal.unit}.")
        rm = [c.rmse for c in fcams if np.isfinite(c.rmse)]
        if rm:
            parts.append(f"Calibration RMSE per camera: {', '.join(f'{r:.2f}' for r in rm)} px.")
        self.info.setText(" ".join(parts))
        self._n_file_cams = len(fcams)
        self._update_ok()
        return True

    def _update_ok(self) -> None:
        """OK needs a file of two or more cameras and its pixel convention (I278: asked when the
        file does not say)."""
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            self._loaded is not None and getattr(self, "_n_file_cams", 0) >= 2
            and (self.conv.currentIndex() >= 0 or getattr(self, "_self_described", False)))

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
        self_described = getattr(self, "_self_described", False)
        if self.conv.currentIndex() < 0 and not self_described:
            return                                      # (I278) OK is grey until the convention is chosen
        origin, flip = CONVENTIONS[self.conv.currentIndex()][1:] if self.conv.currentIndex() >= 0 else (0.0, False)
        cams = []
        for i, k in enumerate(chosen):
            if k < 0:                                   # (I273) not calibrated
                cams.append(None)
                continue
            src = self._loaded.cameras[k]
            w, h = self.view_sizes[i] if i < len(self.view_sizes) else (src.width, src.height)
            if self_described:
                origin, flip = src.pixel_origin, src.y_flip
            cams.append(CameraCalibration(src.coefs.copy(), int(w), int(h), src.undistort,
                                          origin, flip, src.rmse))
        # (I96) a K + R/t file moved the world origin; 3D results need the shift to return to the
        # file's frame, so it survives the re-mapping, with the notes and the quality report
        try:
            # (I278) into Kinetrace's own pixels, once, here: the videos' heights are known
            self.result_calibration = self._loaded.with_cameras(cams).to_internal()
        except ValueError as e:
            QMessageBox.warning(self, "Import camera calibration", f"This calibration cannot be used: {e}.")
            return
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
        # (G203) the landmarks' paths: the row of the instant in `points_all`, and how many rows either side a
        # fading trail spans (the app: 1 s at the reference camera's rate)
        self.t_index: int | None = None
        self.trail_frames = 0
        self.verdicts: list[str] = []   # per landmark at the instant: good / ok / poor / "" (cell_verdict)
        self.world: dict | None = None  # (G201) Calibration.world: the origin / +X / +Y landmarks' names
        # (I309) during a run `points_all` is a window around the instant: the grid's box is the one of the whole
        # result (`box_points`, the same array between runs, so its box is cached) widened by the run's live
        # points (`box_extra`, small)
        self.box_points: np.ndarray | None = None
        self.box_extra: np.ndarray | None = None
        self.missing: list[tuple[str, str]] = []    # (G206) (landmark, why it has no 3D at this instant)
        # (G207) per landmark at the instant: the cameras that made its 3D point, [{"view", "camera" (its name),
        # "err" (px), "verdict"}]; each calibrated camera's centre by view; the landmark selected in the window
        self.rays: list[list[dict]] = []
        self.cam_centres: dict[int, np.ndarray] = {}
        self.selected: str | None = None
        # (G208) the legend: the view of each entry of `cameras` / `camera_names`, and per landmark its state at
        # the instant as (short text, verdict) - "2 cams", "only C3 has it"; (G209) the cameras whose field of
        # view is drawn (None = every one)
        self.camera_views: list[int] = []
        self.status: list[tuple[str, str]] = []
        self.fov_views: set | None = None
        # (G224) the calibration's world is a mirror image (left-handed: easyWand / DLTdv): drawn reflected back
        # (its Y axis), so the scene looks as the room does; the axes keep their own labels
        self.mirrored = False

    def framing(self, include_cameras: bool, include_calib: bool = False,
                box: tuple | None = None) -> tuple[np.ndarray, float]:
        """Centre and span to frame: the CURRENT instant's landmarks and hull
        (the animal fills the window and the view follows it through the
        glide); the whole trajectory only when the instant has nothing, and
        the cameras only when asked (they are metres away from a centimetre
        animal, so including them shrinks it to a dot). With the wand calibration layer: the wand cloud and
        the cameras (G185). With the grid (G202, `box` = `grid_box`): the box, which does not move with the
        animal, so the grid stays put while the time moves."""
        if include_calib and self.calib is not None:
            cal = [self.calib.finite_ends()] + [np.array([c.centre for c in self.calib.cams]).reshape(-1, 3)]
            allp = np.concatenate([p for p in cal if len(p)]) if sum(len(p) for p in cal) else np.zeros((1, 3))
            centre = (allp.min(axis=0) + allp.max(axis=0)) / 2
            return centre, max(float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0))), 0.05)
        if box is not None:
            lo, hi = box[0], box[1]
            return (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 0.05)
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
                     show_cameras: bool = False, show_calib: bool = False, box: tuple | None = None):
    """(to_screen, R, scale): world points -> ((N, 2) pixels of the rendered image, (N,) depth towards the
    viewer), the view rotation, pixels per world unit. THE projection `render_scene` draws with (picking a
    point under the mouse uses the same, G185)."""
    W, H = size
    R = _view_rotation(azimuth, elevation, up)
    if getattr(scene, "mirrored", False):
        # (G224) a left-handed world drawn through a proper rotation is the room's mirror image (a left foot
        # on the right): reflect its Y back - Z stays up, X keeps its direction. Everything goes through this
        # R (points, cameras, grid, axes, picking); a mesh's winding flips with it (`render_scene`)
        R = R @ np.diag([1.0, -1.0, 1.0])
    centre, span = scene.framing(show_cameras, show_calib, box)
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
FOV_WALL_ALPHA = 0.18          # (G209) a box wall's cells a picked camera sees: this much of its colour

# (G202, owner 2026-10-16: "make the figures and vis look clean and publication ready with pastel shades") one
# palette per background, BGR. The dark background is the one the window always had (the wand layers' colours
# are blended over it).
PALETTES = {
    "dark": {"bg": (24, 26, 30), "pane": (33, 36, 41), "grid": (55, 59, 66), "edge": (92, 98, 108),
             "text": (214, 219, 224), "dim": (150, 156, 165), "bone": (150, 152, 156),
             "axes": ((130, 130, 240), (140, 210, 150), (236, 180, 128))},
    "light": {"bg": (250, 249, 247), "pane": (242, 240, 237), "grid": (220, 216, 212), "edge": (168, 163, 158),
              "text": (58, 54, 50), "dim": (118, 113, 108), "bone": (150, 147, 144),
              "axes": ((88, 88, 214), (92, 168, 104), (196, 138, 78))},
}
# one soft colour per landmark (dot and trail), in order: blue, salmon, mint, butter, lavender, pink, teal, tan
LANDMARK_COLOURS = ((222, 184, 143), (160, 167, 242), (181, 217, 158), (138, 211, 246),
                    (224, 166, 195), (208, 179, 245), (209, 209, 142), (155, 185, 217))
TRAIL_MIN_ALPHA = 0.12         # a trail's far end: this much of its colour over the background
LABEL_PX = 11                  # label text height in screen pixels (x the display's density)
TICK_PX = 10


def _bgr(hex_colour: str) -> tuple:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _mix(a, b, t: float) -> tuple:
    """`a` moved toward `b` by t (0 = a)."""
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _draw_labels(img: np.ndarray, labels: list, px: float) -> np.ndarray:
    """Text drawn last, anti-aliased in the application's sans font (QPainter on the picture's own pixels, so
    Save PNG keeps it). labels: (x, y, text, BGR, size in screen px, align "l" / "c" / "r" / "+"); y = the
    baseline; "+" (G208) starts where the previous label ended, `x` being the gap (pieces of one line in
    different colours). Without a Qt application (a script): OpenCV's font."""
    if not labels:
        return img
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
    from PySide6.QtCore import QPointF
    buf = np.ascontiguousarray(img)
    if QGuiApplication.instance() is None:
        end = 0.0
        for x, y, text, col, size, align in labels:
            fs = size / 24.0 * px
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, _px(1, px))[0][0]
            x0 = end + x if align == "+" else (x if align == "l" else (x - tw / 2.0 if align == "c" else x - tw))
            cv2.putText(buf, text, (int(x0), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, _px(1, px), cv2.LINE_AA)
            end = x0 + tw
        return buf
    h, w = buf.shape[:2]
    qi = QImage(buf.data, w, h, 3 * w, QImage.Format_BGR888)
    pa = QPainter(qi)
    pa.setRenderHint(QPainter.TextAntialiasing, True)
    pa.setRenderHint(QPainter.Antialiasing, True)
    font = QFont()
    end = 0.0
    for x, y, text, col, size, align in labels:
        font.setPixelSize(max(6, int(round(size * px))))
        pa.setFont(font)
        pa.setPen(QColor(int(col[2]), int(col[1]), int(col[0])))
        tw = pa.fontMetrics().horizontalAdvance(text)
        if align == "+":
            x0 = end + float(x)
        else:
            x0 = float(x) + (0.0 if align == "l" else (-tw / 2.0 if align == "c" else -float(tw)))
        pa.drawText(QPointF(x0, float(y)), text)
        end = x0 + tw
    pa.end()
    return buf


_GRID_CACHE: list = [None, None, None]        # (the points array, the extra points, its box)


def _grid_box_cached(points, extra=()) -> tuple | None:
    """`grid_box` once per 3D result: the window is drawn again on every frame change with the SAME
    `points_all` (a new result is a new array), and the box over 40k x 20 points took ~35 ms."""
    key = tuple(tuple(np.asarray(e, float).ravel()) for e in extra)
    if _GRID_CACHE[0] is points and _GRID_CACHE[1] == key:
        return _GRID_CACHE[2]
    box = grid_box(points, extra)
    _GRID_CACHE[:] = [points, key, box]
    return box


def _scene_box(scene: "Scene3D", world_on: bool) -> tuple | None:
    """The grid's box: `grid_box` of the whole 3D result (cached), widened by the run's live points (I309)."""
    extra = [np.zeros(3)] if world_on else []
    src = scene.box_points if scene.box_points is not None else scene.points_all
    box = _grid_box_cached(src, extra) if src is not None else None
    if scene.box_extra is None:
        return box
    live = np.asarray(scene.box_extra, np.float64).reshape(-1, 3)
    live = live[np.isfinite(live).all(axis=1)]
    if not len(live):
        return box
    if box is not None and (live >= box[0]).all() and (live <= box[1]).all():
        return box                                  # inside the box: it stays as it is
    return grid_box(None, ([box[0], box[1]] if box is not None else []) + [live] + extra)


def grid_box(points, extra=()) -> tuple | None:
    """(G202) (lo (3,), hi (3,), step): a box on round ticks around every finite point of `points` (T, N, 3)
    and of `extra` (e.g. the world's origin), ONE step for the three axes (1, 2 or 5 x 10^k, 4 to 10
    intervals across the widest side) so the box is not stretched; None without a point."""
    pts = []
    if points is not None:
        a = np.asarray(points, np.float64).reshape(-1, 3)
        pts.append(a[np.isfinite(a).all(axis=1)])
    for e in extra:
        e = np.asarray(e, np.float64).reshape(-1, 3)
        pts.append(e[np.isfinite(e).all(axis=1)])
    if not pts or not sum(len(q) for q in pts):
        return None
    allp = np.concatenate([q for q in pts if len(q)])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = float((hi - lo).max())
    if not span > 0:
        span = max(float(np.abs(lo).max()), 1e-3) * 0.2
    step = _nice_length(span / 4.0)
    lo = np.floor(lo / step - 1e-9) * step
    hi = np.ceil(hi / step + 1e-9) * step
    hi = np.where(hi - lo < step / 2, lo + step, hi)
    return lo, hi, step


def back_walls(R: np.ndarray) -> np.ndarray:
    """(G202) For each world axis, which side of the box its wall stands on: 1 = the high end, 0 = the low end
    - the side AWAY from the viewer, so the walls never hide the points and swap as the view orbits. (With
    Z up and the view from above, Z's wall is the floor.)"""
    toward = np.asarray(R, np.float64).T @ np.array([0.0, 0.0, 1.0])     # world direction toward the viewer
    return (toward < 0).astype(int)


def _tick_text(v: float, step: float) -> str:
    d = max(0, int(-np.floor(np.log10(step) + 1e-9)))
    t = f"{v:.{d}f}"
    return "0" if t.strip("-0.") == "" else t


def _draw_grid(img, box, to_screen, R, pal: dict, unit: str, labels: list, px: float) -> None:
    """(G202) The floor and the two back walls (filled, under everything), their grid lines at every tick,
    the box's edges, and the tick numbers and axis names along the edges nearest the viewer."""
    lo, hi, step = box
    side = back_walls(R)
    wall = np.where(side == 1, hi, lo)              # each axis' wall coordinate
    near = np.where(side == 1, lo, hi)              # the opposite (front) side
    ticks = [np.arange(lo[k], hi[k] + step / 2, step) for k in range(3)]
    for k in range(3):                               # the pane perpendicular to axis k
        a, b = [i for i in range(3) if i != k]
        quad = np.zeros((4, 3))
        quad[:, k] = wall[k]
        quad[:, a] = [lo[a], hi[a], hi[a], lo[a]]
        quad[:, b] = [lo[b], lo[b], hi[b], hi[b]]
        scr = to_screen(quad)[0]
        if np.isfinite(scr).all():
            cv2.fillPoly(img, [_to_int_px(scr)], pal["pane"], cv2.LINE_AA)
    for k in range(3):
        a, b = [i for i in range(3) if i != k]
        for u, v in ((a, b), (b, a)):                 # lines across the pane at u's ticks
            for tv in ticks[u][1:-1]:
                seg = np.zeros((2, 3))
                seg[:, k] = wall[k]
                seg[:, u] = tv
                seg[:, v] = [lo[v], hi[v]]
                _draw_poly(img, to_screen(seg)[0], pal["grid"], False, _px(1, px))
        quad = np.zeros((4, 3))
        quad[:, k] = wall[k]
        quad[:, a] = [lo[a], hi[a], hi[a], lo[a]]
        quad[:, b] = [lo[b], lo[b], hi[b], hi[b]]
        _draw_poly(img, to_screen(quad)[0], pal["edge"], True, _px(1, px))
    centre_scr = to_screen(((lo + hi) / 2)[None])[0][0]
    # each axis is numbered along one front edge: X and Y on the floor (Z at its wall) at the other one's near
    # side; Z up the vertical edge where the X and Y walls' near sides meet
    for k in range(3):
        if k == 2:
            fixed = {0: near[0], 1: wall[1]}
        else:
            o = 1 - k
            fixed = {o: near[o], 2: wall[2]}
        pts = np.zeros((len(ticks[k]), 3))
        pts[:, k] = ticks[k]
        for i, v in fixed.items():
            pts[:, i] = v
        scr = to_screen(pts)[0]
        if not np.isfinite(scr).all() or len(scr) < 2:
            continue
        out = scr.mean(axis=0) - centre_scr          # push the numbers away from the box
        out = out / max(float(np.linalg.norm(out)), 1e-9)
        gap = float(np.median(np.linalg.norm(np.diff(scr, axis=0), axis=1)))
        every = max(1, int(np.ceil(34.0 * px / max(gap, 1e-9))))      # never closer than ~34 px
        # a corner two numbered edges share is numbered once: Y's end at X's edge, Z's foot at Y's edge
        corner = near[0] if k == 0 else (wall[2] if k == 2 else None)
        for j in range(0, len(scr), every):
            if corner is not None and abs(ticks[k][j] - corner) < step / 2:
                continue
            q = scr[j] + out * 14 * px
            labels.append((q[0], q[1] + 4 * px, _tick_text(ticks[k][j], step), pal["dim"], TICK_PX, "c"))
        mid = (scr[0] + scr[-1]) / 2 + out * 32 * px
        name = "XYZ"[k] + (f" ({unit})" if unit else "")
        labels.append((mid[0], mid[1] + 4 * px, name, pal["text"], LABEL_PX, "c"))


TRAIL_LEVELS = 8               # a fading trail is drawn in this many shades (one cv2 call each)
TRAIL_MAX_POINTS = 4000        # the whole path: at most this many points per landmark


def _draw_trails(img, scene: Scene3D, to_screen, pal: dict, trail, px: float) -> tuple | None:
    """(G203) Each landmark's path in its own colour: `trail` = "all" (the whole result, one soft line) or a
    number of rows either side of the instant (fading with the distance in time, blended toward the
    background, in TRAIL_LEVELS shades). Segments are drawn in batches: one call per landmark and shade,
    not one per frame. -> the rows drawn (lo, hi), None when nothing."""
    P, k = scene.points_all, scene.t_index
    if P is None or not len(P) or trail in (None, 0):
        return None
    T, N = P.shape[0], P.shape[1]
    if trail == "all":
        lo, hi = 0, T - 1
    else:
        if k is None:
            return None
        lo, hi = max(0, k - int(trail)), min(T - 1, k + int(trail))
    seg = P[lo:hi + 1]
    if len(seg) < 2:
        return lo, hi
    have = np.isfinite(seg).all(axis=2)                                      # (L, N)
    if trail == "all" and len(seg) > TRAIL_MAX_POINTS:
        # the whole path of a long result: every step-th row (40k rows are far more than the picture's
        # pixels); a segment only where EVERY row it spans has data, so a gap is never bridged
        step = -(-len(seg) // TRAIL_MAX_POINTS)
        rows = np.r_[np.arange(0, len(seg), step), len(seg) - 1]
        rows = np.unique(rows)
        missing = np.concatenate([np.zeros((1, N), int), np.cumsum(~have, axis=0)])
        span_ok = (missing[rows[1:] + 1] - missing[rows[:-1]]) == 0          # (M - 1, N)
        seg, have = seg[rows], have[rows]
    else:
        span_ok = have[:-1] & have[1:]
    scr = to_screen(seg.reshape(-1, 3))[0].reshape(len(seg), N, 2)
    ok = have & np.isfinite(scr).all(axis=2)                                 # (L, N)
    pairs = ok[:-1] & ok[1:] & span_ok                                       # segment i -> i + 1
    ip = _to_int_px(np.where(np.isfinite(scr), scr, 0.0))
    if trail == "all":
        level = np.zeros(len(seg) - 1, int)
        shades = [0.55]
    else:
        dt = np.abs(lo + np.arange(len(seg) - 1) + 0.5 - k) / max(float(trail), 1.0)
        alpha = np.maximum(TRAIL_MIN_ALPHA, 1.0 - dt)
        level = np.minimum((alpha * TRAIL_LEVELS).astype(int), TRAIL_LEVELS - 1)
        shades = [(q + 0.5) / TRAIL_LEVELS for q in range(TRAIL_LEVELS)]
    thick = _px(1, px) if trail == "all" else _px(1.5, px)
    for j in range(N):
        col = LANDMARK_COLOURS[j % len(LANDMARK_COLOURS)]
        for q, a in enumerate(shades):                                       # faint ones first
            idx = np.nonzero(pairs[:, j] & (level == q))[0]
            if not len(idx):
                continue
            lines = np.stack([ip[idx, j], ip[idx + 1, j]], axis=1)          # (M, 2, 2)
            cv2.polylines(img, list(lines), False, _mix(pal["bg"], col, a), thick, cv2.LINE_AA)
    return lo, hi


def _draw_world(img, scene: Scene3D, to_screen, pal: dict, length: float, unit: str, labels: list,
                px: float) -> np.ndarray | None:
    """(G201) The chosen world: X / Y / Z arrows from (0, 0, 0), `length` long, labelled, and the length said
    once. -> the origin on screen (None when off screen)."""
    o = to_screen(np.zeros((1, 3)))[0][0]
    if not np.isfinite(o).all():
        return None
    for k in range(3):
        tip = np.zeros((1, 3))
        tip[0, k] = length
        t = to_screen(tip)[0][0]
        if not np.isfinite(t).all():
            continue
        col = pal["axes"][k]
        p0, p1 = tuple(int(v) for v in _to_int_px(o[None])[0]), tuple(int(v) for v in _to_int_px(t[None])[0])
        n = max(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])), 1.0)
        cv2.arrowedLine(img, p0, p1, col, _px(2, px), cv2.LINE_AA, tipLength=min(0.3, 12.0 * px / n))
        d = (t - o) / n
        labels.append((t[0] + d[0] * 12 * px, t[1] + d[1] * 12 * px + 4 * px, "+" + "XYZ"[k], col, LABEL_PX, "c"))
    cv2.circle(img, tuple(int(v) for v in _to_int_px(o[None])[0]), _px(3, px), pal["text"], -1, cv2.LINE_AA)
    labels.append((o[0] - 8 * px, o[1] + 16 * px, f"0 · arrows {length:g} {unit}".strip(), pal["dim"], TICK_PX, "r"))
    return o


RAY_MIX = 0.6                  # (G207, G208: "thinner like optic rays") its verdict's colour this much over the
                               # background, half a screen pixel wide (a 1 px line at 100 %, faint)
LEGEND_MIN_PX = 200            # (G208) a side strip narrower than this (screen px) gets the legend on a panel
LEGEND_ROW_PX = 17


def cam_tag(view: int) -> str:
    """(G208) A camera's short name on the figure ("C7" = the 7th camera); the legend gives its full name."""
    return f"C{int(view) + 1}"


def ray_names(scene: Scene3D, rays) -> set:
    """(G210) The landmarks whose rays are drawn: `rays` = a collection of names, or "off" / "selected" (the
    scene's selected landmark) / "all"."""
    if rays is None or rays == "off":
        return set()
    if rays == "all":
        return set(scene.names)
    if rays == "selected":
        return {scene.selected} if scene.selected else set()
    return set(rays)


def _draw_rays(img, scene: Scene3D, ps: np.ndarray, to_screen, pal: dict, wanted: set, labels: list,
               px: float, xr: tuple | None = None) -> list[dict]:
    """(G207, owner 2026-10-17: rays "from the corresponding cameras that contributed to the point as is shown
    in ... qualisys") Each 3D point (all, or the selected landmark's) with a ray from every camera that made
    it, from the camera's centre to the dot, coloured by that camera's own verdict (its error against its
    picture's bands). The camera end is clipped to the scene's part of the picture (`xr` = its x range: the
    legend's strips are kept clear; a camera is often metres off screen) and tagged there with the camera's
    short name, once per camera (G208: its full name and the error are in the legend). -> the rays drawn (for
    picking, the legend, tests): {"name", "view", "err", "verdict", "start", "end"} in image pixels."""
    H, W = img.shape[:2]
    x0, x1 = (0, W) if xr is None else (int(xr[0]), int(xr[1]))
    drawn, tagged = [], {}
    thick = _px(0.5, px)                    # (G208, owner: "make the camera rays thinner like optic rays")
    for j, lst in enumerate(scene.rays):
        if j >= len(scene.names) or j >= len(ps) or not lst:
            continue
        nm = scene.names[j]
        if nm not in wanted:
            continue
        p1 = ps[j]
        if not np.isfinite(p1).all():
            continue
        for ry in lst:
            c = scene.cam_centres.get(ry["view"])
            if c is None:
                continue
            p0 = to_screen(np.asarray(c, np.float64).reshape(1, 3))[0][0]
            if not np.isfinite(p0).all():
                continue
            a = tuple(int(v) for v in np.clip(np.round(p0), -SCREEN_CLAMP, SCREEN_CLAMP))
            b = tuple(int(v) for v in np.clip(np.round(p1), -SCREEN_CLAMP, SCREEN_CLAMP))
            ok, a, b = cv2.clipLine((x0, 0, x1 - x0, H), a, b)
            if not ok:
                continue
            col = _mix(pal["bg"], _bgr(theme.VERDICT.get(ry["verdict"], theme.VERDICT["ok"])), RAY_MIX)
            cv2.line(img, a, b, col, thick, cv2.LINE_AA)
            inside = x0 <= p1[0] < x1 and 0 <= p1[1] < H
            drawn.append({"name": nm, "view": ry["view"], "err": ry["err"], "verdict": ry["verdict"],
                          "start": (float(a[0]), float(a[1])),
                          "end": (float(p1[0]), float(p1[1])) if inside else (float(b[0]), float(b[1]))})
            tagged.setdefault(ry["view"], (a, col))
    m = 6 * px
    for view, (a, col) in tagged.items():
        x = float(np.clip(a[0], x0 + m, x1 - m))
        y = float(np.clip(a[1] - m, 2 * m + LABEL_PX * px, H - m))
        labels.append((x, y, cam_tag(view), _mix(col, pal["text"], 0.35), TICK_PX, "l" if x < (x0 + x1) / 2 else "r"))
    return drawn


def legend_layout(size: tuple[int, int], px: float) -> tuple[float, float, bool]:
    """(G208) (left column's width, right column's width, on a panel): the legend takes the strips beside the
    square the scene is framed in (`screen_transform` scales by the picture's SHORT side, so on a wide window
    these are empty); a strip narrower than LEGEND_MIN_PX gets a fixed-width column drawn on a translucent
    panel over the scene instead."""
    W, H = size
    strip = (W - H) / 2.0
    want = LEGEND_MIN_PX * px
    if strip >= want:
        return strip, strip, False
    w = min(want, W * 0.32)
    return w, w, True


def _legend_text(text: str, width: float, px: float) -> str:
    """`text` cut to about `width` pixels of TICK_PX text (an estimate: the font is measured only when drawn)."""
    n = max(4, int(width / (0.56 * TICK_PX * px)))
    return text if len(text) <= n else text[:n - 1] + "…"


LEGEND_PER_LANDMARK = 3        # (G208) with up to this many landmarks' rays, a camera lists each one's error


def _draw_legend(img, scene: Scene3D, pal: dict, labels: list, px: float, rays_drawn: list,
                 wanted: set = frozenset(), fov_seen: dict | None = None) -> dict:
    """(G208, owner 2026-10-18: "there is a lot of white space on either side, can that be used for text/legend
    instead") Left: the cameras - their colour (the one their field of view is drawn in), the tag the figure
    shows, the full name and, with rays, right after the name the error of each of that camera's rays in its
    verdict's colour: one number per landmark (in the order the header names them in their own colours) when
    the rays of up to LEGEND_PER_LANDMARK landmarks are drawn, else the worst one and its landmark; a footer
    says what the number is. Right: the landmarks - colour, name, how many cameras made the point and its
    verdict, or why there is none here; (G210) a ring around the dot of each landmark whose rays are on, and
    a click on a row toggles them. -> {"left": [line texts], "right": [line texts], "rows": [(landmark, x0, y0,
    x1, y1) in image pixels]}."""
    H, W = img.shape[:2]
    lw, rw, panel = legend_layout((W, H), px)
    row, pad, gap = LEGEND_ROW_PX * px, 10 * px, 6 * px
    vc = {k: _mix(_bgr(c), pal["text"], 0.25) for k, c in theme.VERDICT.items()}
    lm_col = {nm: LANDMARK_COLOURS[j % len(LANDMARK_COLOURS)] for j, nm in enumerate(scene.names)}
    ray_names = [nm for nm in scene.names if any(ry["name"] == nm for ry in rays_drawn)]
    each = 0 < len(ray_names) <= LEGEND_PER_LANDMARK
    errs = {}                                       # view -> {landmark: (err, verdict)}
    for ry in rays_drawn:
        errs.setdefault(ry["view"], {})[ry["name"]] = (ry["err"], ry["verdict"])

    def est(text):                                  # the width a text takes (an estimate: measured when drawn)
        return len(text) * 0.56 * TICK_PX * px

    def cam_parts(v):
        out = fov_parts(v)
        e = errs.get(v)
        if not e:
            return out
        return ray_parts(v, e) + ([("  ", pal["dim"])] + out if out else [])

    def fov_parts(v):                               # (G209) its field of view is drawn: how much of the box it sees
        f = (fov_seen or {}).get(v)
        return [] if f is None else [(f"{100 * f:.0f} %", pal["dim"])]

    def ray_parts(v, e):
        if each:
            out = []
            for i, nm in enumerate(ray_names):
                if i:
                    out.append((" · ", pal["dim"]))
                out.append((f"{e[nm][0]:.1f}", vc.get(e[nm][1], pal["dim"])) if nm in e else ("–", pal["dim"]))
            return out + [(" px", pal["dim"])]
        nm = max(e, key=lambda k: e[k][0])
        return [(f"{e[nm][0]:.1f} px", vc.get(e[nm][1], pal["dim"])), (f" ({nm})", pal["dim"])]

    cams = [(("box", CAM_COLOURS[i % len(CAM_COLOURS)]),
             f"{cam_tag(v)}  {scene.camera_names[i] if i < len(scene.camera_names) else ''}", cam_parts(v))
            for i, v in enumerate(scene.camera_views)]
    head_l = [[("Cameras", pal["dim"])]]
    if fov_seen:
        head_l.append([("field of view: % of the box it sees", pal["dim"])])
    foot_l = []
    if rays_drawn:
        if each:
            line = [("ray error in px, for ", pal["dim"])]
            for i, nm in enumerate(ray_names):
                if i:
                    line.append((" · ", pal["dim"]))
                line.append((nm, _mix(lm_col[nm], pal["text"], 0.2)))
            head_l.append(line)
        else:
            head_l.append([("ray error in px: the worst of each camera's rays", pal["dim"])])
        foot_l = ["px: how far the 3D point, seen from that camera,", "lands from the camera's own point (green good,",
                  "yellow ok, red poor; a ray's colour says the same)"]
    lms = [(("dot", LANDMARK_COLOURS[j % len(LANDMARK_COLOURS)], nm in wanted), nm,
            [(scene.status[j][0], vc.get(scene.status[j][1], pal["dim"]))] if j < len(scene.status) else [])
           for j, nm in enumerate(scene.names)]
    rows = []

    def column(x0, width, heads, entries, foots):
        if not entries:
            return []
        room_rows = max(1, int((H - 2 * pad) / row) - len(heads) - len(foots))
        shown = entries if len(entries) <= room_rows else entries[:room_rows - 1]
        n = len(heads) + len(shown) + (len(shown) < len(entries)) + len(foots)
        if panel:
            h = n * row + 2 * pad
            _blend_poly(img, np.array([[x0, 0], [x0 + width, 0], [x0 + width, h], [x0, h]], np.int32), pal["bg"], 0.88)
        y = pad + row * 0.75
        texts = []
        for k, parts in enumerate(heads):
            if k:
                y += row
            labels.append((x0 + pad, y, parts[0][0], parts[0][1], TICK_PX, "l"))
            labels.extend((0.0, y, t, c, TICK_PX, "+") for t, c in parts[1:])
            texts.append("".join(t for t, _ in parts))
        for sw, main, parts in shown:
            y += row
            cy = int(round(y - row * 0.3))
            if sw[0] == "box":
                s_ = _px(4, px)
                cv2.rectangle(img, (int(x0 + pad), cy - s_), (int(x0 + pad) + 2 * s_, cy + s_), sw[1], -1)
            else:
                cv2.circle(img, (int(x0 + pad + _px(4, px)), cy), _px(4, px), sw[1], -1, cv2.LINE_AA)
                if sw[2]:                                   # (G210) its rays are on
                    cv2.circle(img, (int(x0 + pad + _px(4, px)), cy), _px(6.5, px), pal["text"], _px(1, px),
                               cv2.LINE_AA)
                rows.append((main, x0, y - row * 0.8, x0 + width, y + row * 0.2))
            room = width - 2 * pad - 14 * px - (sum(est(t) for t, _ in parts) + gap if parts else 0)
            t = _legend_text(main, room, px)
            labels.append((x0 + pad + 14 * px, y, t, pal["text"], TICK_PX, "l"))
            for i, (pt, pc) in enumerate(parts):            # (G208) right after the name, not at the far edge
                labels.append((gap if i == 0 else 0.0, y, pt, pc, TICK_PX, "+"))
            texts.append(t + ("  " + "".join(pt for pt, _ in parts) if parts else ""))
        if len(shown) < len(entries):
            y += row
            labels.append((x0 + pad, y, f"+ {len(entries) - len(shown)} more", pal["dim"], TICK_PX, "l"))
        for ft in foots:
            y += row
            labels.append((x0 + pad, y, _legend_text(ft, width - 2 * pad, px), pal["dim"], TICK_PX, "l"))
            texts.append(ft)
        return texts

    right_foot = ["click one: rays on / off"] if scene.rays else []
    return {"left": column(0.0, lw, head_l, cams, foot_l),
            "right": column(W - rw, rw, [[("Landmarks", pal["dim"])]], lms, right_foot), "rows": rows}


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


def clip_to_box(poly: np.ndarray, lo, hi) -> np.ndarray:
    """(G209) The part of a 3D polygon (K, 3) inside the box lo..hi (Sutherland-Hodgman against its six
    faces) -> (M, 3), empty when nothing is inside."""
    out = np.asarray(poly, np.float64).reshape(-1, 3)
    for ax in range(3):
        for bound, sign in ((float(lo[ax]), 1.0), (float(hi[ax]), -1.0)):
            if not len(out):
                return out
            d = sign * (out[:, ax] - bound)                  # >= 0 inside
            nxt, dn = np.roll(out, -1, axis=0), np.roll(d, -1)
            res = []
            for p_, q_, dp, dq in zip(out, nxt, d, dn):
                if dp >= 0:
                    res.append(p_)
                if (dp >= 0) != (dq >= 0):
                    res.append(p_ + (dp / (dp - dq)) * (q_ - p_))
            out = np.array(res, np.float64).reshape(-1, 3)
    return out


def _fov_sides(cd, to_screen, box=None) -> list:
    """(G187) The four sides of one camera's field of view as (mean depth, (K, 2) int32 screen polygon): the
    camera centre, the border points of that side (the lens bend kept) and the next side's first point, so
    the corner closes. (G209) `box` (lo, hi): each side cut to it first - the part of the field of view inside
    the working volume, not wedges reaching far past it."""
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
        if box is not None:
            poly = clip_to_box(poly, box[0], box[1])
            if len(poly) < 3:
                continue
        scr, depth = to_screen(poly)
        if len(scr) >= 3 and np.isfinite(scr).all():
            out.append((float(depth.mean()), _to_int_px(scr)))
    return out


FOV_CELLS = 24                 # (G209) each box wall is checked in this many x this many cells per camera
_FOOT_CACHE: dict = {}         # (id(picture), view, box) -> {(axis, side): (cells' corners (M, 4, 3), seen (M,))}


def _wall_cells(lo, hi, axis: int, side: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The n x n cells of the box wall at `axis` = lo / hi (`side` 0 / 1): (corners (n*n, 4, 3), centres)."""
    u, w = [a for a in range(3) if a != axis]
    g = np.linspace(0.0, 1.0, n + 1)
    a0, a1 = np.meshgrid(g[:-1], g[:-1], indexing="ij")
    b0, b1 = a0 + 1.0 / n, a1 + 1.0 / n
    corners = np.zeros((n * n, 4, 3))
    for k, (pu, pw) in enumerate(((a0, a1), (b0, a1), (b0, b1), (a0, b1))):
        corners[:, k, u] = lo[u] + pu.ravel() * (hi[u] - lo[u])
        corners[:, k, w] = lo[w] + pw.ravel() * (hi[w] - lo[w])
        corners[:, k, axis] = hi[axis] if side else lo[axis]
    return corners, corners.mean(axis=1)


def _footprint(pic, cd, lo, hi) -> dict:
    """(G209) Which cells of the box's six walls camera `cd` sees (calibview.camera_sees), cached: the box and
    the picture change rarely, the view on every drag."""
    key = (id(pic), int(cd.view), tuple(np.round(lo, 9)), tuple(np.round(hi, 9)))
    got = _FOOT_CACHE.get(key)
    if got is not None:
        return got
    from kinetrace.calib import working_probe
    from kinetrace.calibview import camera_sees
    cal = pic.cal
    cam = cal.camera(cd.view) if cal is not None else None
    out = {}
    if cam is not None:
        probe = working_probe(cal)
        for axis in range(3):
            for side in (0, 1):
                corners, centres = _wall_cells(np.asarray(lo, float), np.asarray(hi, float), axis, side, FOV_CELLS)
                out[(axis, side)] = (corners, camera_sees(cam, centres, probe))
    if len(_FOOT_CACHE) > 64:
        _FOOT_CACHE.clear()
    _FOOT_CACHE[key] = out
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
                show_fov: bool, show_cover: bool, hits: dict | None, px: float = 1.0,
                labels: list | None = None, fov_box=None, fov_seen: dict | None = None) -> np.ndarray:
    """(G185, G186, G187) the fields of view's translucent sides, the covered space (translucent, over them),
    the fields of view's edges, the cameras' pose pyramids, the wand lines and ends, and the scale bar. Every
    fixed size is in screen pixels times `px` (the display's pixel density). Text goes into `labels` (drawn
    last, `_draw_labels`). Returns the image (the translucent layer makes a new one)."""
    pic = scene.calib
    labels = [] if labels is None else labels
    H, W = img.shape[:2]
    faces = []
    if show_fov:
        # (G209) the cameras picked in the Fields of view menu, each cut to the working volume's box: on the
        # box's three back walls (the grid's) the cells the camera sees, in its colour - a camera that sees the
        # whole box tints them all, which its cut side walls alone would not show (they pass outside the box)
        walls = back_walls(R)
        for k, cd in enumerate(pic.cams):
            if scene.fov_views is not None and cd.view not in scene.fov_views:
                continue
            col = CAM_COLOURS[k % len(CAM_COLOURS)]
            if fov_box is not None:
                fp = _footprint(pic, cd, fov_box[0], fov_box[1])
                if fov_seen is not None and fp:              # (G209) for the legend: how much of the box it sees
                    fov_seen[cd.view] = float(np.mean(np.concatenate([sn for _, sn in fp.values()])))
                quads = []
                for axis in range(3):
                    corners, seen = fp.get((axis, int(walls[axis])), (None, None))
                    if corners is None or not seen.any():
                        continue
                    scr = to_screen(corners[seen].reshape(-1, 3))[0].reshape(-1, 4, 2)
                    quads += list(_to_int_px(scr))
                if quads:
                    mask = np.zeros(img.shape[:2], np.uint8)
                    cv2.fillPoly(mask, quads, 1)
                    m = mask.astype(bool)
                    img[m] = (img[m] * (1.0 - FOV_WALL_ALPHA) + np.array(col, float) * FOV_WALL_ALPHA).astype(np.uint8)
                    if hits is not None:
                        hits.setdefault("fov_walls", {})[cd.view] = int(m.sum())
            faces += [(d, poly, col) for d, poly in _fov_sides(cd, to_screen, fov_box)]
        for d, poly, col in sorted(faces, key=lambda f: f[0]):          # far ones first
            _blend_poly(img, poly, col, FOV_SIDE_ALPHA)
        for d, poly, col in faces:                                      # the cut sides' outlines
            cv2.polylines(img, [poly], True, tuple(int(0.55 * v) for v in col), _px(1, px), cv2.LINE_AA)
        if hits is not None:
            hits["fov"] = [poly for _, poly, _ in faces]
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
        if show_pyramids and cd.pose is not None and len(cd.pose.border):
            _draw_poly(img, to_screen(cd.pose.border)[0], col, True, _px(1, px))            # (G187) 1 px
            for c0, c1 in cd.pose.rays:
                _draw_poly(img, to_screen(np.vstack([c0, c1]))[0], col, False, _px(1, px))
        cs = to_screen(cd.centre)[0][0]
        shown = show_pyramids or (show_fov and (scene.fov_views is None or cd.view in scene.fov_views))
        if shown and np.isfinite(cs).all():
            labels.append((cs[0] + 8 * px, cs[1] - 6 * px, cam_tag(cd.view), col, LABEL_PX, "l"))     # (G208)
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
        labels.append((x1 - n, y - 9 * px, lab, (220, 220, 220), LABEL_PX, "l"))
    return img


def render_scene(scene: Scene3D, size: tuple[int, int], azimuth: float, elevation: float,
                 zoom: float = 1.0, show_mesh: bool = True, show_points: bool = True,
                 show_cameras: bool = False, up: int = 2, pan: tuple[float, float] = (0.0, 0.0),
                 show_calib: bool = False, show_fov: bool = False, show_cover: bool = False,
                 hits: dict | None = None, show_pyramids: bool | None = None, px: float = 1.0,
                 show_grid: bool = False, show_world: bool = False, trail=None, palette: str = "dark",
                 rays: str = "off", legend: bool = False) -> np.ndarray:
    """The 3D window's picture (BGR). The wand calibration layers (G185, G186) are drawn only when asked and
    the scene has a picture: `show_calib` = the wand points (and the scale bar), `show_pyramids` the
    cameras' pose pyramids (None = with the wand points), `show_fov` their fields of view, `show_cover`
    the covered space; each on its own (owner 2026-10-12). `hits`, when given, receives "ends" = the wand
    ends' (P, 2) image pixels (None when they are not drawn). `px` (G187): the display's pixel density -
    `size` is in device pixels, `pan` in screen (logical) pixels, and every fixed size (lines, dots, text)
    is scaled by it, so the picture is sharp on a scaled display; px = 1 draws exactly as before.
    (G201-G203) `show_grid`: the box grid around the whole 3D result (and the view frames it); `show_world`:
    the chosen world's arrows (with `scene.world`); `trail`: None, a number of rows either side, or "all";
    `palette`: "dark" | "light". `hits` also receives "labels" (every text drawn), "grid" (the box),
    "origin" (the world's origin on screen) and "trail" (the rows drawn). (G207) `rays`: "off" | "selected" |
    "all" or (G210) a set of landmark names - the cameras' rays to each 3D point; `hits` "rays" (`_draw_rays`)
    and "dots" ((N, 2) each
    landmark's dot on the picture, NaN where none is drawn). (G208) `legend`: the cameras and the landmarks in
    the strips beside the scene (`_draw_legend`; `hits` "legend")."""
    W, H = size
    pal = PALETTES.get(palette, PALETTES["dark"])
    img = np.full((H, W, 3), pal["bg"], np.uint8)
    pyr = show_calib if show_pyramids is None else show_pyramids
    calib_on = (show_calib or pyr or show_fov or show_cover) and scene.calib is not None
    # (G209) the view frames the wand cloud and the cameras for the layers that reach them; fields of view alone
    # are cut to the 3D points' box, so with points the view keeps framing those
    frame_calib = calib_on and (show_calib or pyr or show_cover or scene.points_all is None)
    world_on = bool(show_world and scene.world)
    box = _scene_box(scene, world_on) if show_grid else None
    to_screen, R, scale = screen_transform(scene, size, azimuth, elevation, zoom,
                                           (pan[0] * px, pan[1] * px), up, show_cameras, frame_calib,
                                           None if frame_calib else box)
    t1 = _px(1, px)                                                  # a 1 px line
    fov_seen: dict = {}                                              # (G209) view -> share of the box it sees
    labels: list = []
    unit = scene.unit or (scene.calib.unit if scene.calib is not None else "")
    if box is not None:
        _draw_grid(img, box, to_screen, R, pal, unit, labels, px)
    if hits is not None:
        hits["grid"] = box

    if box is None and not world_on:
        # axes triad (bottom-left), world axes after the same rotation: the one orientation cue left when
        # neither the grid nor the world's arrows say it (no redundant UI)
        o = np.array([48.0 * px, H - 48.0 * px])
        for k, lab in enumerate("XYZ"):
            col = pal["axes"][k]
            a = np.zeros(3)
            a[k] = 1.0
            d = (R @ a)[:2] * 32 * px
            tip = (int(o[0] + d[0]), int(o[1] - d[1]))
            cv2.line(img, (int(o[0]), int(o[1])), tip, col, _px(2, px), cv2.LINE_AA)
            labels.append((tip[0] + 3 * px, tip[1] + 4 * px, lab, col, LABEL_PX, "l"))
    if show_mesh and scene.mesh is not None and len(scene.mesh[1]):
        verts, faces = scene.mesh
        fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        if np.linalg.det(R) < 0:
            fn = -fn                                 # (G224) a reflected view turns every face inside out
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
        # (G209) the fields of view end at the grid's box (the 3D result's), else at the wand cloud's
        if box is None and scene.points_all is not None:
            box = _scene_box(scene, world_on)          # (review) the points' box with the grid off too
            fov_box = box[:2] if box is not None else scene.calib.box(0.25)
            box = None
        else:
            fov_box = box[:2] if box is not None else scene.calib.box(0.25)
        img = _draw_calib(img, scene, to_screen, R, scale, show_calib, pyr, show_fov, show_cover, hits, px, labels,
                          fov_box, fov_seen)
    if show_cameras and not calib_on and scene.cameras is not None and len(scene.cameras):
        cs, _ = to_screen(scene.cameras)
        m = 50 * px
        for k, (x, y) in enumerate(cs):
            if np.isfinite(x) and np.isfinite(y) and -m < x < W + m and -m < y < H + m:
                cv2.drawMarker(img, (int(x), int(y)), (80, 120, 255), cv2.MARKER_TRIANGLE_UP, _px(14, px),
                               _px(2, px), cv2.LINE_AA)
                v = scene.camera_views[k] if k < len(scene.camera_views) else k
                labels.append((x + 8 * px, y + 4 * px, cam_tag(v), (120, 150, 255), LABEL_PX, "l"))    # (G208)
    if show_points:
        drawn = _draw_trails(img, scene, to_screen, pal, trail, px)
        if hits is not None:
            hits["trail"] = drawn
    drawn_rays: list = []
    if hits is not None:
        hits["rays"], hits["dots"] = [], None
    if show_points and scene.points is not None and len(scene.points):
        ps, _ = to_screen(scene.points)
        ok = np.isfinite(scene.points).all(axis=1)
        wanted = ray_names(scene, rays)
        if wanted:
            lw, rw, _ = legend_layout((W, H), px)
            drawn_rays = _draw_rays(img, scene, ps, to_screen, pal, wanted, labels, px,
                                    (lw, W - rw) if legend else None)          # (G208) clear of the legend
            if hits is not None:
                hits["rays"] = drawn_rays
        if hits is not None:
            hits["dots"] = np.where((ok & np.isfinite(ps).all(axis=1))[:, None], ps, np.nan)
        for i, j in scene.bones:
            if i < len(ok) and j < len(ok) and ok[i] and ok[j]:
                cv2.line(img, (int(ps[i, 0]), int(ps[i, 1])), (int(ps[j, 0]), int(ps[j, 1])),
                         pal["bone"], t1, cv2.LINE_AA)
        roles = {}
        if world_on:
            roles = {scene.world.get("origin"): "origin", scene.world.get("x"): "+X", scene.world.get("y"): "+Y"}
        ring = {"ok": _bgr(theme.VERDICT["ok"]), "poor": _bgr(theme.VERDICT["poor"])}
        for k, (x, y) in enumerate(ps):
            if ok[k] and np.isfinite(x) and np.isfinite(y):
                col = LANDMARK_COLOURS[k % len(LANDMARK_COLOURS)]
                c = (int(x), int(y))
                vd = scene.verdicts[k] if k < len(scene.verdicts) else ""
                if vd in ring:                                   # ok / poor: a ring in the verdict's colour
                    cv2.circle(img, c, _px(7, px), ring[vd], _px(1.5, px), cv2.LINE_AA)
                cv2.circle(img, c, _px(4.5, px), col, -1, cv2.LINE_AA)
                cv2.circle(img, c, _px(4.5, px), _mix(col, (0, 0, 0), 0.4), t1, cv2.LINE_AA)
                if k < len(scene.names):
                    nm = scene.names[k]
                    tag = roles.get(nm)
                    labels.append((x + 8 * px, y - 6 * px, nm + (f" · {tag}" if tag else ""), pal["text"],
                                   LABEL_PX, "l"))
    if world_on:
        span = float(np.linalg.norm(box[1] - box[0])) if box is not None else scene.framing(show_cameras)[1]
        o = _draw_world(img, scene, to_screen, pal, _nice_length(0.3 * span), unit, labels, px)
        if hits is not None:
            hits["origin"] = o
    if legend and (scene.camera_views or scene.names):
        leg = _draw_legend(img, scene, pal, labels, px, drawn_rays, ray_names(scene, rays), fov_seen)
        if hits is not None:
            hits["legend"] = leg
    if hits is not None:
        hits["labels"] = [(x, y, t) for x, y, t, *_ in labels]
    return _draw_labels(img, labels, px)


def _light_preference(value: bool | None = None) -> bool:
    """(G202) The 3D window's Light background, remembered for every project (the app's settings file, as the
    annotator's name): read it, or store `value`."""
    from PySide6.QtCore import QSettings
    from kinetrace import recovery
    try:
        st = QSettings(str(recovery.settings_path()), QSettings.IniFormat)
        if value is not None:
            st.setValue("view3d/light_background", bool(value))
            return bool(value)
        return str(st.value("view3d/light_background", "false")).lower() in ("true", "1")
    except Exception:           # noqa: BLE001 - a preference that cannot be read is the default
        return False if value is None else bool(value)


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
        # (G202, G201, G203) the figure: the grid, the chosen world, the paths, the background
        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.setToolTip("A box around the whole 3D result with a grid on the floor and the two walls behind "
                                 "the points, numbered in the calibration's unit: read a position off it. The box "
                                 "stays put while the landmarks move.")
        self.chk_world = QCheckBox("World axes")
        self.chk_world.setChecked(True)
        self.chk_world.setToolTip("The world chosen with 3D → Set World Axes: arrows from the origin along +X, +Y "
                                  "and +Z, and the three landmarks that set it")
        self.cmb_trail = QComboBox()
        self.cmb_trail.addItems(["No trail", "Trail ±1 s", "Whole path"])
        self.cmb_trail.setCurrentIndex(1)
        self.cmb_trail.setToolTip("Each landmark's path in its own colour: the second before and after the frame on "
                                  "screen (fading), or the whole 3D result")
        # (G207) the cameras' rays to each 3D point, as motion-capture software draws them; (G210, owner
        # 2026-10-18) for the landmarks ticked in this menu, and the main window's selected one if followed
        from PySide6.QtWidgets import QMenu as _QMenu, QToolButton as _QToolButton
        self.btn_rays = _QToolButton()
        self.btn_rays.setPopupMode(_QToolButton.InstantPopup)
        self.btn_rays.setToolTip("Draw a line from each camera that made a 3D point to that point, for the landmarks "
                                 "ticked here (or click a landmark in the legend, or its dot), and for the one "
                                 "selected in the main window if you tick Follow. Each line is coloured by how well "
                                 "that camera agrees with the point: green good, yellow ok, red poor. A red line is "
                                 "the camera to check (a wrong click or track, or its timing).")
        self.rays_menu = _QMenu(self)
        self.btn_rays.setMenu(self.rays_menu)
        self.ray_actions: list = []
        self._ray_pick: set = set()         # the landmarks ticked (by name, kept while the window is open)
        self._ray_follow = False            # also the main window's selected landmark
        self._ray_menu_names: list = []
        self._fill_rays_menu()
        self.chk_light = QCheckBox("Light background")
        self.chk_light.setToolTip("A white figure for a paper or a slide; remembered")
        self.chk_light.setChecked(_light_preference())
        for c in (self.chk_points, self.chk_mesh, self.chk_cams, self.chk_grid, self.chk_world):
            c.toggled.connect(lambda _=False: self.request_render())
            top.addWidget(c)
        self.cmb_trail.currentIndexChanged.connect(lambda _=0: self.request_render())
        top.addWidget(self.cmb_trail)
        top.addWidget(self.btn_rays)
        self.chk_light.toggled.connect(self._light_toggled)
        top.addWidget(self.chk_light)
        self.btn_save = QPushButton("Save picture…")
        self.btn_save.setToolTip("Write the picture as it is on screen to a PNG file, at twice the window's "
                                 "resolution (for a paper or a slide)")
        self.btn_save.clicked.connect(self._save_clicked)
        top.addWidget(self.btn_save)
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
        from PySide6.QtWidgets import QMenu, QToolButton
        # (G209, owner 2026-10-18) the fields of view of the cameras picked here (none at first): ten at once
        # are a tangle; each is cut to the working volume's box
        self.btn_fov = QToolButton()
        self.btn_fov.setPopupMode(QToolButton.InstantPopup)
        self.btn_fov.setToolTip("Pick the cameras whose field of view is drawn: what each one sees, its lens "
                                "included, inside the box around the 3D points (the wand's space without them): "
                                "translucent sides in the camera's colour, denser where fields overlap. Pick one "
                                "or two to check a camera's framing.")
        self.fov_menu = QMenu(self)
        self.btn_fov.setMenu(self.fov_menu)
        self.fov_actions: list = []
        self._fov_pick: set = set()             # the views picked, kept across new pictures of the calibration
        self._fov_menu_cams: list = []          # (view, name) the menu lists
        self.chk_cover = QCheckBox("Covered by ≥")
        self.chk_cover.setToolTip("The space at least this many of the chosen cameras see: where 3D is possible "
                                  "(2) and well determined (3 or more)")
        self.cmb_cover = QComboBox()
        self.cmb_cover.setSizeAdjustPolicy(QComboBox.AdjustToContents)    # (G209) "2 cameras", not "2 came"
        self.btn_cams = QToolButton()
        self.btn_cams.setText("Cameras ▾")
        self.btn_cams.setPopupMode(QToolButton.InstantPopup)
        self.btn_cams.setToolTip("Choose the cameras whose pyramid is drawn, and that the covered space is "
                                 "counted from")
        self.cams_menu = QMenu(self)
        self.btn_cams.setMenu(self.cams_menu)
        self.cam_actions: list = []
        self._fov_label()
        for w in (self.chk_calib, self.chk_pyr, self.btn_fov, self.chk_cover, self.cmb_cover, self.btn_cams):
            cr.addWidget(w)
        cr.addStretch(1)
        self.calib_row.setVisible(False)
        lay.addWidget(self.calib_row)
        for c in (self.chk_calib, self.chk_pyr, self.chk_cover):
            c.toggled.connect(self._calib_ticked)
        self.cmb_cover.currentIndexChanged.connect(self._cover_n_changed)
        self.canvas = QLabel()
        self.canvas.setMinimumSize(320, 240)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setAlignment(Qt.AlignCenter)
        self._apply_background()
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
        self.last_render_s = 0.0
        self._last_dpr = 1.0            # the pixel density last_image was drawn at (G187)
        self._hits: dict = {}
        self._cover_threads: list = []          # every worker until it has finished (see _start_coverage)
        self._cover_gen = 0
        self._cover_key = None
        self._base_info = ""

    # ---- the figure's background (G202)
    def _palette(self) -> str:
        return "light" if self.chk_light.isChecked() else "dark"

    def _apply_background(self) -> None:
        b, g, r = PALETTES[self._palette()]["bg"]
        self.canvas.setStyleSheet(f"background: #{r:02x}{g:02x}{b:02x};")

    def _light_toggled(self, on: bool) -> None:
        _light_preference(bool(on))
        self._apply_background()
        self.request_render()

    def _trail(self):
        i = self.cmb_trail.currentIndex()
        return None if i == 0 else ("all" if i == 2 else max(1, int(self.scene.trail_frames)))

    def _rays(self) -> set:
        """(G210) The landmarks whose rays are drawn: the ticked ones, and the selected one when followed."""
        out = {nm for nm in self._ray_pick if nm in self.scene.names}
        if self._ray_follow and self.scene.selected:
            out.add(self.scene.selected)
        return out

    def _fill_rays_menu(self) -> None:
        """(G210) The Rays menu for the scene's landmarks: All / None, Follow the main window's selection, a tick
        per landmark (kept by name)."""
        names = list(self.scene.names) if self.scene is not None else []
        self._ray_menu_names = names
        self.rays_menu.clear()
        self.ray_actions = []
        for label, on in (("All landmarks", True), ("None", False)):
            a = self.rays_menu.addAction(label)
            a.triggered.connect(lambda _=False, on=on: self._set_rays_all(on))
        self.act_ray_follow = self.rays_menu.addAction("Follow the main window's selection")
        self.act_ray_follow.setCheckable(True)
        self.act_ray_follow.setChecked(self._ray_follow)
        self.act_ray_follow.toggled.connect(self._ray_follow_toggled)
        self.rays_menu.addSeparator()
        for nm in names:
            a = self.rays_menu.addAction(nm)
            a.setCheckable(True)
            a.setChecked(nm in self._ray_pick)
            a.setData(nm)
            a.toggled.connect(lambda on, nm=nm: self._ray_ticked(nm, on))
            self.ray_actions.append(a)
        self._rays_label()

    def _rays_label(self) -> None:
        n = len({nm for nm in self._ray_pick if nm in self._ray_menu_names})
        what = " + ".join(([str(n)] if n else []) + (["selected"] if self._ray_follow else []))
        self.btn_rays.setText(f"Rays ({what}) ▾" if what else "Rays ▾")

    def _rays_changed(self) -> None:
        self._rays_label()
        self._show_info()
        self.request_render()

    def _ray_ticked(self, name: str, on: bool) -> None:
        (self._ray_pick.add if on else self._ray_pick.discard)(name)
        self._rays_changed()

    def _set_rays_all(self, on: bool) -> None:
        self._ray_pick = set(self._ray_menu_names) if on else set()
        for a in self.ray_actions:
            a.blockSignals(True)
            a.setChecked(on)
            a.blockSignals(False)
        self._rays_changed()

    def _ray_follow_toggled(self, on: bool) -> None:
        self._ray_follow = bool(on)
        self._rays_changed()

    def toggle_rays(self, name: str) -> None:
        """(G210) A click on a landmark's legend row or dot: its rays on / off (its tick in the menu)."""
        for a in self.ray_actions:
            if a.data() == name:
                a.setChecked(not a.isChecked())         # -> _ray_ticked
                return

    def set_selected(self, name: str | None) -> None:
        """(G207) The landmark selected in the main window: its rays when Follow is ticked."""
        if name == self.scene.selected:
            return
        self.scene.selected = name
        if self._ray_follow:
            self._show_info()
            self.request_render()

    # ---- data
    def set_scene(self, scene: Scene3D, info: str = "") -> None:
        old = self.scene.calib if self.scene is not None else None
        scene.calib_views = self.scene.calib_views if (self.scene is not None and scene.calib is old) else None
        scene.cover_n = self.scene.cover_n if self.scene is not None else scene.cover_n
        self.scene = scene
        self._base_info = info
        if list(scene.names) != self._ray_menu_names:        # (G210) the landmarks changed: the menu follows
            self._fill_rays_menu()
        if scene.calib is old and self._fov_cams() != self._fov_menu_cams:     # (G209) the cameras changed
            self._fill_fov_menu()
        self.chk_world.setEnabled(bool(scene.world))          # (G201) greyed until Set World Axes
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
        boxes = {"points": self.chk_calib, "pyramids": self.chk_pyr, "cover": self.chk_cover}
        for k in ticks:
            if k == "fov":                    # asked for by name (the wand wizard's Show in 3D): every camera
                self._set_fov_all(True, quiet=True)
                continue
            boxes[k].blockSignals(True)
            boxes[k].setChecked(True)
            boxes[k].blockSignals(False)
        self._calib_ticked()                  # counts the covered space when that tick is on

    def _calib_on(self) -> bool:
        return self._fov_on() or any(c.isChecked() for c in (self.chk_calib, self.chk_pyr, self.chk_cover))

    # ---- (G209) the Fields of view menu
    def _fov_views(self) -> set:
        """The picked cameras the menu lists (the pick itself is kept even while a camera is not listed)."""
        return {v for v in self._fov_pick if any(v == c for c, _ in self._fov_menu_cams)}

    def _fov_on(self) -> bool:
        return bool(self._fov_views())

    def _fov_label(self) -> None:
        n = len(self._fov_views())
        self.btn_fov.setText(f"Fields of view ({n}) ▾" if n else "Fields of view ▾")

    def _fov_cams(self) -> list:
        """(view, name) of the cameras a field of view can be drawn for: the picture's, else the calibrated
        cameras of the scene (review: without a picture the menu was empty, so nothing could ask for one)."""
        pic = self.scene.calib
        if pic is not None:
            return [(int(cd.view), cd.name) for cd in pic.cams]
        return [(int(v), self.scene.camera_names[i] if i < len(self.scene.camera_names) else "")
                for i, v in enumerate(self.scene.camera_views)]

    def _fill_fov_menu(self) -> None:
        cams = self._fov_cams()
        self._fov_menu_cams = cams
        self.fov_menu.clear()
        self.fov_actions = []
        if cams:
            for label, on in (("All cameras", True), ("No camera", False)):
                a = self.fov_menu.addAction(label)
                a.triggered.connect(lambda _=False, on=on: self._set_fov_all(on))
            self.fov_menu.addSeparator()
            for v, nm in cams:
                a = self.fov_menu.addAction(f"{cam_tag(v)}  {nm}")
                a.setCheckable(True)
                a.setChecked(v in self._fov_pick)
                a.setData(v)
                a.toggled.connect(lambda on, v=v: self._fov_ticked(v, on))
                self.fov_actions.append(a)
        self._fov_label()

    def _set_fov_all(self, on: bool, quiet: bool = False) -> None:
        for a in self.fov_actions:
            a.blockSignals(True)
            a.setChecked(on)
            a.blockSignals(False)
        self._fov_pick = {v for v, _ in self._fov_menu_cams} if on else set()
        self._fov_label()
        if not quiet:
            self._calib_ticked()

    def _fov_ticked(self, view: int, on: bool) -> None:
        (self._fov_pick.add if on else self._fov_pick.discard)(int(view))
        self._fov_label()
        self._calib_ticked()                  # asks for the picture when there is none yet

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
            self._fill_fov_menu()
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
        self._fill_fov_menu()

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
        tips = []
        miss = self.scene.missing
        if miss:                            # (G206) which landmark has no 3D here, and why
            # (G208) the line stays short: the legend gives each one's state, the tooltip the whole reason
            txt += (f" · no 3D here: {miss[0][0]}" if len(miss) == 1 else
                    f" · {len(miss)} landmarks without 3D here")
            tips.append("No 3D position at this instant:\n" + "\n".join(f"  {nm}: {why}" for nm, why, *_ in miss))
        if self._ray_follow:                # (G207, G210) say why the followed landmark has no ray
            sel = self.scene.selected
            if sel is None:
                txt += " · rays: no landmark selected in the main window"
            elif not any(nm == sel and j < len(self.scene.rays) and self.scene.rays[j]
                         for j, nm in enumerate(self.scene.names)):
                txt += f" · rays: {sel} has no 3D point here"
        if pic is not None and self._calib_on():
            lines = list(pic.lines)
            if not self.chk_cover.isChecked():
                lines = [ln for ln in lines if not ln.startswith("The wand cloud lies")]
            if lines:
                txt = (txt + "  ·  " if txt else "") + "  ".join(lines)
            tips.append("\n".join(pic.lines))
        self.info.setToolTip("\n\n".join(tips))
        self.info.setText(txt)

    def end_screen_points(self, what: str = "ends") -> np.ndarray | None:
        """The wand ends' (P, 2) positions on the canvas in the last picture (`what` "dots": the landmarks'
        dots, G207), in screen (logical) pixels - the picture itself is drawn at the display's pixel density
        (G187) - NaN where not drawn."""
        es = self._hits.get(what)
        if es is None or self.last_image is None:
            return None
        dpr = self._last_dpr
        h, w = self.last_image.shape[:2]
        ox, oy = (self.canvas.width() - w / dpr) / 2.0, (self.canvas.height() - h / dpr) / 2.0
        return np.asarray(es, np.float64) / dpr + np.array([ox, oy])

    def _legend_row_at(self, x: float, y: float) -> str | None:
        """(G210) The landmark whose legend row is under the canvas point (x, y), if any."""
        leg = self._hits.get("legend") or {}
        if not leg.get("rows") or self.last_image is None:
            return None
        dpr = self._last_dpr
        h, w = self.last_image.shape[:2]
        ix = (x - (self.canvas.width() - w / dpr) / 2.0) * dpr
        iy = (y - (self.canvas.height() - h / dpr) / 2.0) * dpr
        for nm, x0, y0, x1, y1 in leg["rows"]:
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                return nm
        return None

    def _pick(self, x: float, y: float, radius: float = 6.0, what: str = "ends") -> int | None:
        pts = self.end_screen_points(what)
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
                    else:                   # (G210) a landmark's legend row or dot: its rays on / off
                        nm = self._legend_row_at(ev.position().x(), ev.position().y())
                        if nm is None:
                            j = self._pick(ev.position().x(), ev.position().y(), radius=8.0, what="dots")
                            nm = self.scene.names[j] if j is not None and j < len(self.scene.names) else None
                        if nm is not None:
                            self.toggle_rays(nm)
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
        self.scene.fov_views = self._fov_views()
        t0 = time.perf_counter()
        img = render_scene(self.scene, (int(round(w * dpr)), int(round(h * dpr))), self.azimuth, self.elevation,
                           self.zoom, self.chk_mesh.isChecked(), self.chk_points.isChecked(),
                           self.chk_cams.isChecked(), pan=tuple(self.pan),
                           show_calib=self.chk_calib.isChecked(), show_fov=self._fov_on(),
                           show_cover=self.chk_cover.isChecked(), hits=hits,
                           show_pyramids=self.chk_pyr.isChecked(), px=dpr,
                           show_grid=self.chk_grid.isChecked(), show_world=self.chk_world.isChecked(),
                           trail=self._trail(), palette=self._palette(), rays=self._rays(), legend=True)
        self.last_render_s = time.perf_counter() - t0       # (I309) the live updates' time budget reads it
        self._hits = hits
        self._last_dpr = dpr
        self.last_image = img
        self.canvas.setPixmap(bgr_pixmap(img, dpr))

    def save_png(self, path: str, scale: float | None = None) -> None:
        """The picture on screen to a PNG; `scale` (G202): drawn again at that many times the window's
        resolution (lines and text scale with it), else the picture as shown."""
        if scale is None:
            if self.last_image is None:
                self._render()
            img = self.last_image
        else:
            w, h = max(64, self.canvas.width()), max(64, self.canvas.height())
            self.scene.fov_views = self._fov_views()
            img = render_scene(self.scene, (int(round(w * scale)), int(round(h * scale))), self.azimuth,
                               self.elevation, self.zoom, self.chk_mesh.isChecked(), self.chk_points.isChecked(),
                               self.chk_cams.isChecked(), pan=tuple(self.pan),
                               show_calib=self.chk_calib.isChecked(), show_fov=self._fov_on(),
                               show_cover=self.chk_cover.isChecked(), show_pyramids=self.chk_pyr.isChecked(),
                               px=float(scale), show_grid=self.chk_grid.isChecked(),
                               show_world=self.chk_world.isChecked(), trail=self._trail(), palette=self._palette(),
                               rays=self._rays(), legend=True)
        if not cv2.imwrite(path, img):
            raise OSError(f"the picture could not be written to {path}")

    def _save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the 3D picture", "3d_view.png", "PNG picture (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self.save_png(path, scale=2.0)
        except OSError as e:
            self.info.setText(str(e))
            return
        self.info.setText(f"Saved {Path(path).name} (twice the window's resolution)")

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
