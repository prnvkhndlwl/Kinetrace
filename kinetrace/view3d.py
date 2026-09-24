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
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from kinetrace import theme
from kinetrace.calib import Calibration, CameraCalibration
from kinetrace.hull import _view_rotation

CALIB_FILTER = ("Calibration files (*.json *.csv *.mat *.txt);;Kinetrace calibration (*.kcal.json);;"
                "DLTdv / easyWand / Argus DLT coefficients (*.csv);;"
                "easyWand data (*easyWandData.mat);;DLTdv8 project, MATLAB v7 (*dvProject.mat);;"
                "OpenCV cameras K + R/t (*.json *.txt);;All files (*)")

CONVENTIONS = [
    # label, pixel_origin, y_flip
    ("DLTdv8 / easyWand — MATLAB pixels (1-based, y down from the top)", 1.0, False),
    ("DLTdv5-style — y measured from the BOTTOM edge", 1.0, True),
    ("OpenCV / Python — 0-based pixels, y down", 0.0, False),
]


def load_calibration_file(path: str, sizes: list[tuple[int, int]] | None = None) -> Calibration:
    """Pick the importer from the file name / contents. `sizes` only labels the
    columns of a size-less dltCoefs.csv; the import dialog does NOT pass the
    videos' sizes, because a size the file never recorded must not be shown
    or compared as if it were the file's (I99)."""
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".json"):
        from kinetrace.calibwizard import load_kcal
        try:
            return load_kcal(p)
        except (ValueError, KeyError, TypeError):
            return Calibration.load_krt(p)          # an OpenCV-style K + R/t JSON
    if low.endswith(".txt"):
        return Calibration.load_krt(p)
    if low.endswith(".mat"):
        if "easywand" in low:
            return Calibration.load_easywand_mat(p)
        if "dvproject" in low:
            return Calibration.load_dltdv_project(p)
        try:
            return Calibration.load_easywand_mat(p)
        except Exception:      # noqa: BLE001
            return Calibration.load_dltdv_project(p)
    return Calibration.load_dlt_csv(p, sizes)


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
        self._self_described = low.endswith(".json") or low.endswith(".txt")
        if self._self_described and cal.cameras:
            c0 = cal.cameras[0]
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
        default = [min(i, len(cal.cameras) - 1) for i in range(len(self.view_names))]
        for i, size in enumerate(self.view_sizes[:len(self.view_names)]):
            hits = [k for k, c in enumerate(cal.cameras)
                    if c.width and (int(c.width), int(c.height)) == (int(size[0]), int(size[1]))]
            if len(hits) == 1 and hits[0] not in [default[j] for j in range(len(default)) if j != i]:
                default[i] = hits[0]
        for i, nm in enumerate(self.view_names):
            self.grid.addWidget(QLabel(nm), i + 1, 0)
            cb = QComboBox()
            for k, c in enumerate(cal.cameras):
                size = f" ({c.width}×{c.height})" if c.width else ""
                cb.addItem(f"cam {k + 1}{size}")
            cb.setCurrentIndex(default[i])
            cb.setToolTip("Which column of the file is this camera. The sizes in brackets are the "
                          "file's own record of each camera's picture -- match them to the videos."
                          if any(c.width for c in cal.cameras) else
                          "Which column of the file is this camera. The file records no picture sizes: "
                          "use the order your calibration software listed the cameras in.")
            self.grid.addWidget(cb, i + 1, 1)
            self._combos.append(cb)
        und = {c.undistort.kind for c in cal.cameras}
        parts = [f"{len(cal.cameras)} camera(s) in the file, {len(self.view_names)} in the project."]
        if not any(c.width for c in cal.cameras):
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
        rm = [c.rmse for c in cal.cameras if np.isfinite(c.rmse)]
        if rm:
            parts.append(f"Calibration RMSE per camera: {', '.join(f'{r:.2f}' for r in rm)} px.")
        self.info.setText(" ".join(parts))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(len(cal.cameras) >= 2)
        return True

    def _accept(self) -> None:
        if self._loaded is None:
            return
        from PySide6.QtWidgets import QMessageBox
        chosen = [cb.currentIndex() for cb in self._combos]
        # Two project cameras on one column is never right: the same DLT would
        # place both cameras at the same spot and the 3D is garbage with a
        # perfectly normal-looking residual. Refuse, do not warn.
        if len(set(chosen)) != len(chosen):
            dup = sorted({k for k in chosen if chosen.count(k) > 1})
            QMessageBox.warning(
                self, "Two cameras share a column",
                "Column(s) " + ", ".join(f"cam {k + 1}" for k in dup) + " of the file are assigned to "
                "more than one camera of this project. Every camera needs its own column -- "
                "match them by image size and by the order the calibration software listed them.")
            return
        # A size mismatch between a column and the video it is assigned to is
        # the usual sign of a wrong mapping (e.g. a rig of five 1920x1080
        # cameras and one 848x480). Say which ones, and let the user decide.
        mism = []
        for i, k in enumerate(chosen):
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

    def framing(self, include_cameras: bool) -> tuple[np.ndarray, float]:
        """Centre and span to frame: the CURRENT instant's landmarks and hull
        (the animal fills the window and the view follows it through the
        glide); the whole trajectory only when the instant has nothing, and
        the cameras only when asked (they are metres away from a centimetre
        animal, so including them shrinks it to a dot)."""
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


def render_scene(scene: Scene3D, size: tuple[int, int], azimuth: float, elevation: float,
                 zoom: float = 1.0, show_mesh: bool = True, show_points: bool = True,
                 show_cameras: bool = False, up: int = 2, pan: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    W, H = size
    img = np.full((H, W, 3), (24, 26, 30), np.uint8)
    R = _view_rotation(azimuth, elevation, up)
    centre, span = scene.framing(show_cameras)
    scale = 0.85 * min(W, H) / span * zoom

    def to_screen(p):
        q = (np.asarray(p, np.float64).reshape(-1, 3) - centre) @ R.T
        return (np.column_stack([W / 2 + q[:, 0] * scale + pan[0], H / 2 - q[:, 1] * scale + pan[1]]),
                q[:, 2])

    # axes triad (bottom-left), world axes after the same rotation
    o = np.array([48.0, H - 48.0])
    for k, (col, lab) in enumerate((((60, 60, 230), "X"), ((60, 200, 60), "Y"), ((230, 120, 60), "Z"))):
        a = np.zeros(3)
        a[k] = 1.0
        d = (R @ a)[:2] * 32
        tip = (int(o[0] + d[0]), int(o[1] - d[1]))
        cv2.line(img, (int(o[0]), int(o[1])), tip, col, 2, cv2.LINE_AA)
        cv2.putText(img, lab, (tip[0] + 3, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
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
    if show_cameras and scene.cameras is not None and len(scene.cameras):
        cs, _ = to_screen(scene.cameras)
        for k, (x, y) in enumerate(cs):
            if np.isfinite(x) and np.isfinite(y) and -50 < x < W + 50 and -50 < y < H + 50:
                cv2.drawMarker(img, (int(x), int(y)), (80, 120, 255), cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
                nm = scene.camera_names[k] if k < len(scene.camera_names) else f"cam {k + 1}"
                cv2.putText(img, nm, (int(x) + 8, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 150, 255), 1, cv2.LINE_AA)
    if show_points and scene.points is not None and len(scene.points):
        ps, _ = to_screen(scene.points)
        ok = np.isfinite(scene.points).all(axis=1)
        for i, j in scene.bones:
            if i < len(ok) and j < len(ok) and ok[i] and ok[j]:
                cv2.line(img, (int(ps[i, 0]), int(ps[i, 1])), (int(ps[j, 0]), int(ps[j, 1])),
                         (200, 200, 200), 1, cv2.LINE_AA)
        for k, (x, y) in enumerate(ps):
            if ok[k] and np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), 4, (60, 230, 120), -1, cv2.LINE_AA)
                if k < len(scene.names):
                    cv2.putText(img, scene.names[k], (int(x) + 6, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (170, 240, 190), 1, cv2.LINE_AA)
    return img


class View3D(QWidget):
    """Floating 3D window: orbit with the left mouse button, zoom with the
    wheel, pan with the middle button (or Shift+drag), double-click to reset.
    Redraws on demand, coalesced through a timer."""

    closed = Signal()

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
        self.canvas = QLabel()
        self.canvas.setMinimumSize(320, 240)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background: #181a1e;")
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

    # ---- data
    def set_scene(self, scene: Scene3D, info: str = "") -> None:
        self.scene = scene
        self.info.setText(info)
        self.request_render()

    def set_info(self, info: str) -> None:
        self.info.setText(info)

    def request_render(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _render(self) -> None:
        w = max(64, self.canvas.width())
        h = max(64, self.canvas.height())
        img = render_scene(self.scene, (w, h), self.azimuth, self.elevation, self.zoom,
                           self.chk_mesh.isChecked(), self.chk_points.isChecked(),
                           self.chk_cams.isChecked(), pan=tuple(self.pan))
        self.last_image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(qimg.copy()))

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
        self.closed.emit()
        super().closeEvent(ev)
