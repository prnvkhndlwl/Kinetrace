"""Video canvas: frame display, point markers, and all mouse interaction.

Scene coordinates are identical to native video pixel coordinates by
construction (the pixmap item fills the scene rect at native size), so
mapToScene() of a mouse event yields video pixels directly — there is no
manual coordinate mapping anywhere.

Gestures (active only when interaction is enabled; G59, owner 2026-10-02):
    click                -> place the SELECTED point here (on a marker too);
                            nothing selected: a click on a marker selects it
    hold left + move     -> pan (starting on a marker too: points are never dragged)
    right click a marker -> select it and clear it on THIS frame only
    hold right on marker -> (LONG_PRESS_MS) its menu: rename / delete / ...
    Ctrl+click anywhere  -> reposition the selected point there
    Alt+click            -> look here (where this spot can be in the other cameras)
    wheel                -> zoom (anchored under cursor); R fits, middle-drag pans
Add (N) armed: click = a new point (or continue the selected one), drag = a
circle region tracked as a group (Add ▾ picks circle / rectangle / polygon: a
rectangle is a drag, a polygon is click-click-click + Enter).
Segment tool (A, armed): click = "this is the segment", Shift+click = "not the
segment", drag = box around it; right-click a prompt marker to remove it.
Overlays: the segment's silhouette (translucent fill + outline), its midline,
prompt markers, and skeleton bones between landmarks. Motion overlays (one
`_MotionOverlay` item per canvas): fading trajectory trails (past solid,
future dashed), onion-skin ghosts of the previous / next frame. A loupe
(`_Loupe`, a child widget of the viewport) magnifies the pixels under the
cursor for sub-pixel placement. Display filters (`set_display_filter`) change
only what is SHOWN — contrast, brightness, frame difference — never the
frames the tracker sees.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen,
                           QPixmap, QPolygonF, QTransform)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QLabel, QMenu)

HIT_RADIUS_PX = 14      # screen px within which a click grabs a point
MARKER_RADIUS = 3.0     # screen px (3 by default, was 7)
TRAIL_FRAMES = 10          # View -> Trails' preset ("Last 10 frames") and the default (G33)
TRAIL_MAX = 1000          # the longest Custom... trail
DRAG_CIRCLE_PX = 6      # screen px of movement before a press becomes a drag (a pan, a circle)
LONG_PRESS_MS = 500     # a right press held this long opens the point's menu (G59)
MIN_GROUP_RADIUS = 8.0  # native px; a smaller circle is treated as a plain click
MIN_BOX_PX = 10.0       # native px; a smaller animal box is treated as a click
ZOOM_STEP = 1.25       # video zoom factor per keypress / wheel notch
REGION_SHAPES = ("circle", "rect", "polygon")
DISPLAY_FILTERS = ("none", "contrast", "bright", "diff")
LOUPE_PX = 176         # loupe widget side, screen px
LOUPE_FACTOR = 4.0     # magnification relative to the current view scale
GHOST_RADIUS = 4.5     # onion-skin ghost marker radius, screen px


def np_to_qimage(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    # straight to RGB32, the format QPixmap.fromImage converts a frame to anyway:
    # the conversion is also the deep copy that frees the image from the numpy
    # buffer, so one pass replaces a copy + a conversion (a 4K frame 17.0 ->
    # 8.7 ms on the GUI thread, the same pixels; I135)
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).convertToFormat(QImage.Format_RGB32)


# the display filters' lookup tables, built once: cv2.LUT on a 4K frame is ~25x
# faster than numpy's lut[rgb] / an int32 clip, with the same pixels (I136)
_BRIGHT_LUT = (255.0 * (np.arange(256) / 255.0) ** 0.5).astype(np.uint8)    # gamma 0.5
_DIFF_GAIN_LUT = np.clip(np.arange(256) * 4, 0, 255).astype(np.uint8)       # x4 gain, saturating


def apply_display_filter(rgb: np.ndarray, kind: str, prev: np.ndarray | None = None) -> np.ndarray:
    """Display-only image filters. `rgb` is HxWx3 uint8; returns a new array
    of the same shape (or `rgb` itself for "none"). Pure pixel ops — the
    tracker never sees the result."""
    if kind == "contrast":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if kind == "bright":
        return cv2.LUT(rgb, _BRIGHT_LUT)
    if kind == "diff":
        if prev is None or prev.shape != rgb.shape:
            return rgb
        d = cv2.absdiff(rgb, prev)
        g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
        g = cv2.LUT(g, _DIFF_GAIN_LUT)                                  # x4 gain
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    return rgb


class _GuideOverlay(QGraphicsItem):
    """Epipolar guides: where the SELECTED landmark, as the other cameras see
    it at this instant, can lie in this picture -- one dashed polyline per
    other camera (a polyline, because lens undistortion bends it). Drawn in
    screen-constant width; geometry in native video pixels. `faint`: the parts
    of a line where the lens model is guessing (I132); `preds`: where two or
    more other cameras put the landmark together (G23) -- the lines then step
    back (`dim`)."""

    PRED_R = 8.0                       # screen px: half-diagonal of the predicted-point diamond

    def __init__(self):
        super().__init__()
        self.setZValue(4)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.rect = QRectF()
        self.lines: list = []          # (pts (M, 2), (r, g, b), label)
        self.faint: list = []          # (pts (M, 2), (r, g, b))
        self.preds: list = []          # (x, y, (r, g, b), label, (px, py) placed here or None)
        self.notes: list = []          # (x, y, text, (r, g, b)): the 3D rmse beside a placed point (G28)
        self.dim = False

    def boundingRect(self) -> QRectF:
        return self.rect

    @staticmethod
    def _path(pts) -> QPainterPath:
        path = QPainterPath(QPointF(float(pts[0, 0]), float(pts[0, 1])))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        return path

    def _text(self, painter, x, y, dx, dy, text, color, scale):
        painter.save()
        painter.setPen(QPen(QColor(*color, 235), 1.0 / scale))
        painter.translate(x, y)
        painter.scale(1.0 / scale, 1.0 / scale)
        painter.drawText(QPointF(dx, dy), text)
        painter.restore()

    def paint(self, painter: QPainter, option, widget=None):
        if not (self.lines or self.faint or self.preds or self.notes):
            return
        scale = painter.worldTransform().m11() or 1.0
        painter.setRenderHint(QPainter.Antialiasing, True)
        for pts, color, label in self.lines:
            pts = np.asarray(pts, np.float64)
            if len(pts) < 2:
                continue
            pen = QPen(QColor(*color, 110 if self.dim else 210), (1.2 if self.dim else 1.6) / scale)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(self._path(pts))
            if label:
                self._text(painter, float(pts[-1, 0]), float(pts[-1, 1]), 4, -4, label, color, scale)
        for pts, color in self.faint:
            pts = np.asarray(pts, np.float64)
            if len(pts) < 2:
                continue
            pen = QPen(QColor(*color, 80), 1.0 / scale)
            pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawPath(self._path(pts))
        r = self.PRED_R / scale
        for x, y, color, label, placed in self.preds:
            if placed is not None:
                pen = QPen(QColor(*color, 170), 1.0 / scale)
                painter.setPen(pen)
                painter.drawLine(QPointF(float(placed[0]), float(placed[1])), QPointF(x, y))
            diamond = QPolygonF([QPointF(x, y - r), QPointF(x + r, y), QPointF(x, y + r), QPointF(x - r, y)])
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 160), 3.2 / scale))       # a dark halo: visible on any footage
            painter.drawPolygon(diamond)
            painter.setPen(QPen(QColor(*color, 240), 1.6 / scale))
            painter.drawPolygon(diamond)
            painter.setBrush(QBrush(QColor(*color, 240)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), 1.4 / scale, 1.4 / scale)
            if label:
                self._text(painter, x, y, self.PRED_R + 4, -self.PRED_R - 2, label, color, scale)
        for x, y, text, color in self.notes:
            # below-right of the marker, on a dark plate: readable on any footage
            painter.save()
            painter.translate(float(x), float(y))
            painter.scale(1.0 / scale, 1.0 / scale)
            fm = painter.fontMetrics()
            r = fm.boundingRect(text).adjusted(-4, -2, 4, 2).translated(10, 14 + fm.ascent())
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
            painter.drawRoundedRect(r, 3, 3)
            painter.setPen(QPen(QColor(*color, 245), 1.0))
            painter.drawText(QPointF(10, 14 + fm.ascent()), text)
            painter.restore()


class _MotionOverlay(QGraphicsItem):
    """Trails and onion-skin ghosts for every point, drawn in one paint call.
    Line widths and ghost radii are in SCREEN pixels (computed from the view
    scale at paint time), the geometry is native video pixels."""

    def __init__(self):
        super().__init__()
        self.setZValue(5)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.rect = QRectF()
        self.colors: list = []
        self.display: list = []
        self.past: list = []     # per point (K, 2) oldest..newest ending at the current frame
        self.future: list = []   # per point (K, 2) starting after the current frame
        self.ghost_prev: np.ndarray | None = None   # (N, 2) previous-frame positions
        self.ghost_next: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.selected: int | None = None

    def boundingRect(self) -> QRectF:
        return self.rect

    def paint(self, painter: QPainter, option, widget=None):
        scale = painter.worldTransform().m11() or 1.0
        painter.setRenderHint(QPainter.Antialiasing, True)
        n = len(self.colors)
        for i in range(n):
            if i >= len(self.display) or not self.display[i]:
                continue
            color = self.colors[i]
            bold = i == self.selected
            # past trail: fade from transparent (oldest) to opaque (now)
            pts = self.past[i] if i < len(self.past) else None
            if pts is not None and len(pts) >= 2:
                k = len(pts) - 1
                for j in range(k):
                    a, b = pts[j], pts[j + 1]
                    if not (np.isfinite(a).all() and np.isfinite(b).all()):
                        continue
                    alpha = int(40 + 200 * (j + 1) / k)
                    pen = QPen(QColor(*color, alpha), (2.2 if bold else 1.5) / scale)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))
            # future path: dashed, dimmer, fading away from now
            pts = self.future[i] if i < len(self.future) else None
            if pts is not None and len(pts) >= 2:
                k = len(pts) - 1
                for j in range(k):
                    a, b = pts[j], pts[j + 1]
                    if not (np.isfinite(a).all() and np.isfinite(b).all()):
                        continue
                    alpha = int(30 + 150 * (k - j) / k)
                    pen = QPen(QColor(*color, alpha), 1.2 / scale, Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))
            # onion skin: hollow ghost at the previous (thin) and next (dashed) frame
            r = GHOST_RADIUS / scale
            cur = self.current[i] if self.current is not None and i < len(self.current) else None
            for ghost, style in ((self.ghost_prev, Qt.SolidLine), (self.ghost_next, Qt.DashLine)):
                if ghost is None or i >= len(ghost) or not np.isfinite(ghost[i]).all():
                    continue
                g = ghost[i]
                pen = QPen(QColor(*color, 170), 1.2 / scale, style)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(float(g[0]), float(g[1])), r, r)
                if cur is not None and np.isfinite(cur).all():
                    pen = QPen(QColor(*color, 110), 1.0 / scale, Qt.DotLine)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(float(g[0]), float(g[1])), QPointF(float(cur[0]), float(cur[1])))


class _Loupe(QLabel):
    """Magnifier that follows the cursor over the video (child of the viewport).
    Shows the native pixels around the cursor at LOUPE_FACTOR x the current
    view scale with nearest-neighbour scaling, so single pixels are visible
    when placing or dragging a point at 4K."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(LOUPE_PX, LOUPE_PX)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("QLabel { border: 1px solid rgba(255,255,255,180); background: #000; }")
        self.hide()

    def show_at(self, pixmap: QPixmap, pix_transform: QTransform, scene_pt: QPointF,
                view_pos: QPoint, view_scale: float, marker: QPointF | None,
                color: tuple[int, int, int] | None) -> None:
        if pixmap.isNull():
            self.hide()
            return
        # the pixmap may be lower-res than native (working-res frames during a
        # run): map native px -> pixmap px through the item transform
        sx = pix_transform.m11() or 1.0
        sy = pix_transform.m22() or 1.0
        mag = max(view_scale * LOUPE_FACTOR, 1.0)
        win = max(8.0, LOUPE_PX / mag)                # native px shown across the loupe
        cx, cy = scene_pt.x() / sx, scene_pt.y() / sy
        half_x, half_y = win / sx / 2, win / sy / 2
        src = QRect(int(round(cx - half_x)), int(round(cy - half_y)),
                    max(2, int(round(2 * half_x))), max(2, int(round(2 * half_y))))
        crop = pixmap.copy(src)
        img = crop.scaled(LOUPE_PX, LOUPE_PX, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # crosshair at the cursor's exact sub-pixel position
        fx = (cx - src.x()) / max(src.width(), 1) * LOUPE_PX
        fy = (cy - src.y()) / max(src.height(), 1) * LOUPE_PX
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
        painter.drawLine(QPointF(fx - 14, fy), QPointF(fx - 4, fy))
        painter.drawLine(QPointF(fx + 4, fy), QPointF(fx + 14, fy))
        painter.drawLine(QPointF(fx, fy - 14), QPointF(fx, fy - 4))
        painter.drawLine(QPointF(fx, fy + 4), QPointF(fx, fy + 14))
        if marker is not None and color is not None:
            mx = (marker.x() / sx - src.x()) / max(src.width(), 1) * LOUPE_PX
            my = (marker.y() / sy - src.y()) / max(src.height(), 1) * LOUPE_PX
            painter.setPen(QPen(QColor(*color), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(mx, my), 6, 6)
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
        painter.drawText(4, LOUPE_PX - 5, f"{mag:.1f}x")
        painter.end()
        self.setPixmap(img)
        # sit to the lower-right of the cursor; flip when that leaves the viewport
        pw, ph = self.parent().width(), self.parent().height()
        x = view_pos.x() + 22
        y = view_pos.y() + 22
        if x + LOUPE_PX > pw:
            x = view_pos.x() - 22 - LOUPE_PX
        if y + LOUPE_PX > ph:
            y = view_pos.y() - 22 - LOUPE_PX
        self.move(max(0, x), max(0, y))
        self.show()
        self.raise_()


class _Marker(QGraphicsEllipseItem):
    """Fixed-screen-size point marker with a name tag."""

    def __init__(self, pid: int, r: float = MARKER_RADIUS):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.pid = pid
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(10)
        self.label = QGraphicsSimpleTextItem(self)
        self.label.setPos(r + 3, -r - 3)
        self._cross = QGraphicsPathItem(self)     # hand-marked hidden: an X inside the ring
        self._cross.setVisible(False)
        self.setAcceptedMouseButtons(Qt.NoButton)  # the view handles all mouse logic

    def set_radius(self, r: float) -> None:
        self.setRect(-r, -r, 2 * r, 2 * r)
        self.label.setPos(r + 3, -r - 3)

    def style(self, color: tuple[int, int, int], visible: bool, selected: bool, name: str,
              derived: bool = False, occluded: bool = False):
        qc = QColor(*color)
        pen = QPen(qc, 3 if selected else 2)
        pen.setCosmetic(True)
        # derived (silhouette) landmarks: dotted ring — computed, not tracked
        pen.setStyle(Qt.DotLine if derived else Qt.SolidLine)
        self.setPen(pen)
        # filled = visible to the tracker, hollow ring = predicted occluded;
        # hand-marked hidden = hollow with a cross ("not exported here")
        self.setBrush(QBrush(qc) if (visible and not occluded) else QBrush(Qt.NoBrush))
        self._cross.setVisible(bool(occluded))
        if occluded:
            r = self.rect().width() / 2
            path = QPainterPath()
            path.moveTo(-r * 0.7, -r * 0.7)
            path.lineTo(r * 0.7, r * 0.7)
            path.moveTo(-r * 0.7, r * 0.7)
            path.lineTo(r * 0.7, -r * 0.7)
            self._cross.setPath(path)
            cp = QPen(qc, 2)
            cp.setCosmetic(True)
            self._cross.setPen(cp)
        self.label.setText(name + ("  (hidden)" if occluded else ""))
        self.label.setBrush(QBrush(qc))
        f = self.label.font()
        f.setBold(selected)
        f.setItalic(derived)
        self.label.setFont(f)


class _PromptMarker(QGraphicsEllipseItem):
    """Fixed-screen-size segment prompt: + (this is the segment) / − (it is not)."""

    def __init__(self, index: int, positive: bool, color: tuple[int, int, int], r: float = 6.0):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.index = index
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(12)
        self.setAcceptedMouseButtons(Qt.NoButton)
        qc = QColor(*color) if positive else QColor(255, 90, 80)
        pen = QPen(QColor(20, 20, 24), 1.5)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(qc))
        self.label = QGraphicsSimpleTextItem("+" if positive else "−", self)
        self.label.setBrush(QBrush(QColor(20, 20, 24)))
        f = self.label.font()
        f.setBold(True)
        f.setPointSizeF(9)
        self.label.setFont(f)
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2, -br.height() / 2)


class VideoCanvas(QGraphicsView):
    add_requested = Signal(float, float)
    annotate_requested = Signal(float, float)      # plain click, unarmed: place the SELECTED point here
    group_requested = Signal(float, float, float)  # (cx, cy, radius) circle drag
    region_requested = Signal(str, object)         # ("rect" | "polygon", [[x, y], ...]) armed drag / clicks
    occluded_toggled = Signal(int, bool)           # (pid, hidden on this frame)
    point_selected = Signal(int)
    point_moved = Signal(int, float, float)        # live while dragging
    move_committed = Signal(int, float, float)     # on release
    reposition_requested = Signal(float, float)    # Ctrl+click
    clear_frame_requested = Signal(int)            # right click on a marker: clear it on this frame (G59)
    delete_requested = Signal(int)
    rename_requested = Signal(int)
    anchor_toggled = Signal(int, bool)             # appearance re-anchor opt-in
    source_change_requested = Signal(int, str)     # (pid, derived spec or "" = track)
    free_toggled = Signal(int, bool)               # (pid, may leave the animal)
    animal_click = Signal(float, float, bool)      # (x, y, positive) — animal tool
    animal_box = Signal(float, float, float, float)  # (x0, y0, x1, y1) — animal tool drag
    prompt_remove_requested = Signal(int)          # index into this frame's prompt list
    probe_requested = Signal(float, float)         # Alt+click: where is this spot in the other cameras?
    view_clicked = Signal()                        # any press — the camera grid uses
    #                                                it to make a clicked view active

    def __init__(self):
        super().__init__()
        from kinetrace.theme import BG_CANVAS
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(BG_CANVAS))  # darkest surface: video pops
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._pixitem = QGraphicsPixmapItem()
        self._pixitem.setTransformationMode(Qt.SmoothTransformation)
        # Scene coordinates are OpenCV pixel coordinates: the CENTRE of image
        # pixel i is at i, the convention of the tracker, SAM, the silhouettes,
        # the calibration and every export. With the pixmap at (0, 0) pixel i's
        # centre sat at i + 0.5, so a click on the centre of a pixel was stored
        # half a pixel right / down of it and tracked markers were drawn half a
        # pixel up-left of their pixel (I24). `pos` is applied after the
        # working-resolution scale set in set_frame, so a working pixel j also
        # lands at (j + 0.5) * s - 0.5, the worker's _w2n.
        self._pixitem.setPos(-0.5, -0.5)
        self._scene.addItem(self._pixitem)
        self._placeholder = self._scene.addSimpleText("Open a video to begin  (Ctrl+O)")
        self._placeholder.setBrush(QBrush(QColor(160, 160, 170)))

        self._native_size: tuple[int, int] | None = None  # (w, h)
        self._marker_radius = MARKER_RADIUS  # screen px, user-adjustable
        self._markers: list[_Marker] = []
        self._trails: list[QGraphicsPathItem] = []
        self._regions: list[QGraphicsPathItem] = []       # dashed region outlines (circle/rect/polygon)
        self._occluded = np.zeros(0, bool)
        self._region_shape = "circle"       # Add ▾: what an armed drag draws
        self._poly_pts: list[QPointF] = []  # polygon-in-progress vertices (native px)
        self._display_filter = "none"
        self._raw_rgb: np.ndarray | None = None
        self._prev_rgb: np.ndarray | None = None
        self._loupe_enabled = False
        # the app fills the session-aware half of the point menu (see
        # _build_context_menu); left None, the menu is the canvas-only one
        self.menu_extra = None          # callable(menu, acts, pid)
        self.menu_extra_action = None   # callable(chosen, acts, pid) -> handled?
        self.multi_menu = None          # callable(pid, global pos) -> handled? (several selected, G64)
        self._trail_len = TRAIL_FRAMES
        self._trail_future = False
        self._onion = False
        self._member_items: dict[int, QGraphicsPathItem] = {}  # pid -> live member dots
        self._interactive = True
        self._switchable = False   # a view-only companion a click switches to (pointing hand, not busy)
        self._user_zoomed = False
        self._follow_enabled = True   # mirrors the ⌖ Follow toggle
        self._selected: int | None = None
        self._dragging: int | None = None
        self._drag_moved = False
        self._panning = False
        self._pan_mode = False   # ✋: left-drag pans instead of editing points
        self._place_mode = False  # N: next click places a point (crosshair)
        self._click_only = False  # ball marker: armed clicks ignore the region shape
        self._plain_press = False  # unarmed left press on the video (annotate on release, pan on a move)
        self._press_hit: int | None = None   # the marker under an unarmed left press, if any
        # right press on a marker: a click clears it on this frame, a long press opens its menu (G59)
        self._rpress: tuple | None = None    # (pid, global pos) while the right button is down
        self._rpress_timer = QTimer(self)
        self._rpress_timer.setSingleShot(True)
        self._rpress_timer.setInterval(LONG_PRESS_MS)
        self._rpress_timer.timeout.connect(self._on_long_right_press)
        self._animal_mode = False  # A: clicks/boxes prompt the animal segmentation
        self._pan_last = QPointF()
        self._positions = np.zeros((0, 2), np.float32)
        self._metas: list = []
        self._bones: list[tuple[int, int]] = []
        self._mask_rect: QRectF | None = None   # segment bbox on this frame (follow target)
        # deferred click / circle-drag gesture state
        self._press_scene: QPointF | None = None
        self._press_view: QPointF | None = None
        self._circle_active = False
        self._circle_preview = QGraphicsEllipseItem()
        pen = QPen(QColor(255, 255, 255, 200), 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        self._circle_preview.setPen(pen)
        self._circle_preview.setBrush(QBrush(QColor(255, 255, 255, 30)))
        self._circle_preview.setZValue(20)
        self._circle_preview.setVisible(False)
        self._scene.addItem(self._circle_preview)
        # animal tool: box-drag preview
        self._box_active = False
        self._box_preview = QGraphicsRectItem()
        self._box_preview.setPen(pen)
        self._box_preview.setBrush(QBrush(QColor(255, 255, 255, 25)))
        self._box_preview.setZValue(20)
        self._box_preview.setVisible(False)
        self._scene.addItem(self._box_preview)
        # armed rectangle / polygon region previews
        self._rect_preview = QGraphicsRectItem()
        self._rect_preview.setPen(pen)
        self._rect_preview.setBrush(QBrush(QColor(255, 255, 255, 30)))
        self._rect_preview.setZValue(20)
        self._rect_preview.setVisible(False)
        self._scene.addItem(self._rect_preview)
        self._poly_preview = QGraphicsPathItem()
        self._poly_preview.setPen(pen)
        self._poly_preview.setBrush(QBrush(QColor(255, 255, 255, 30)))
        self._poly_preview.setZValue(20)
        self._poly_preview.setVisible(False)
        self._scene.addItem(self._poly_preview)
        self._rect_active = False
        # motion overlay (trails + onion skin) and the loupe
        self._motion = _MotionOverlay()
        self._scene.addItem(self._motion)
        self._guides = _GuideOverlay()
        self._scene.addItem(self._guides)
        # a companion that has not caught up with the playhead (Active view only, G24)
        self._stale = QGraphicsRectItem()
        self._stale.setPen(QPen(Qt.NoPen))
        self._stale.setBrush(QBrush(QColor(0, 0, 0, 120)))
        self._stale.setZValue(30)
        self._stale.setVisible(False)
        self._scene.addItem(self._stale)
        self._loupe = _Loupe(self.viewport())
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        # animal overlays: silhouette, midline, prompt markers, skeleton bones
        self._mask_item = QGraphicsPathItem()
        self._mask_item.setZValue(3)
        self._mask_item.setVisible(False)
        self._scene.addItem(self._mask_item)
        self._midline_item = QGraphicsPathItem()
        self._midline_item.setZValue(4)
        self._midline_item.setVisible(False)
        self._scene.addItem(self._midline_item)
        self._bones_item = QGraphicsPathItem()
        self._bones_item.setZValue(7)
        self._bones_item.setVisible(False)
        self._scene.addItem(self._bones_item)
        self._prompt_items: list[_PromptMarker] = []
        self._prompt_positions = np.zeros((0, 2), np.float32)
        self._prompt_box_item = QGraphicsRectItem()
        self._prompt_box_item.setZValue(11)
        self._prompt_box_item.setVisible(False)
        self._scene.addItem(self._prompt_box_item)

    # ---------------------------------------------------------------- frames

    def set_video_size(self, w: int, h: int) -> None:
        self._native_size = (w, h)
        self._scene.setSceneRect(-0.5, -0.5, w, h)       # the picture's outer edges (I24)
        self._motion.prepareGeometryChange()
        self._motion.rect = QRectF(-0.5, -0.5, w, h)
        self._guides.prepareGeometryChange()
        self._guides.rect = QRectF(-0.5, -0.5, w, h)
        self._stale.setRect(QRectF(-0.5, -0.5, w, h))
        self._raw_rgb = None
        self._prev_rgb = None
        self._placeholder.setVisible(False)
        self._user_zoomed = False
        self.fit()

    def clear_video(self) -> None:
        self._native_size = None
        self._pixitem.setPixmap(QPixmap())
        self._placeholder.setVisible(True)
        self.set_points(np.zeros((0, 2)), np.zeros(0, bool), [], None)
        self.clear_group_members()
        self.set_mask(None)
        self.set_midline(None)
        self.set_prompts([], None)
        self.set_guides([])
        self.set_stale(False)
        self.cancel_gesture()

    def set_stale(self, on: bool) -> None:
        """Veil a companion view whose picture is not at the playhead's instant
        (Active view only: it is not decoded until Sync all views is back, G27)."""
        self._stale.setVisible(bool(on) and self._native_size is not None)

    def is_stale(self) -> bool:
        return self._stale.isVisible()

    # ------------------------------------------------------- animal overlays

    def set_mask(self, polys, color: tuple[int, int, int] = (77, 227, 176),
                 opacity: float = 0.35) -> None:
        """Silhouette overlay from native-px outline polygons (None hides it)."""
        if not polys:
            self._mask_item.setVisible(False)
            self._mask_rect = None
            return
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        allpts = np.concatenate([np.asarray(q, np.float64).reshape(-1, 2) for q in polys if len(q) >= 3]) if polys else np.zeros((0, 2))
        if len(allpts):
            lo, hi = allpts.min(axis=0), allpts.max(axis=0)
            self._mask_rect = QRectF(float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        for poly in polys:
            pts = np.asarray(poly, np.float64)
            if len(pts) < 3:
                continue
            # outlines are OpenCV pixel-centre points, like the scene (I24)
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in pts]))
            path.closeSubpath()
        self._mask_item.setPath(path)
        fill = QColor(*color)
        fill.setAlpha(int(np.clip(opacity, 0.0, 1.0) * 255))
        self._mask_item.setBrush(QBrush(fill))
        pen = QPen(QColor(*color, 230), 1.5)
        pen.setCosmetic(True)
        self._mask_item.setPen(pen)
        self._mask_item.setVisible(True)

    def set_midline(self, pts, color: tuple[int, int, int] = (77, 227, 176)) -> None:
        if pts is None or len(pts) < 2:
            self._midline_item.setVisible(False)
            return
        path = QPainterPath()
        pts = np.asarray(pts, np.float64)
        path.moveTo(float(pts[0][0]), float(pts[0][1]))
        for p in pts[1:]:
            path.lineTo(float(p[0]), float(p[1]))
        pen = QPen(QColor(*color, 210), 2.0)
        pen.setCosmetic(True)
        self._midline_item.setPen(pen)
        self._midline_item.setPath(path)
        self._midline_item.setVisible(True)

    def set_prompts(self, clicks, box, color: tuple[int, int, int] = (77, 227, 176)) -> None:
        """Show this frame's segment prompts: clicks [(x, y, label)] and a box."""
        for it in self._prompt_items:
            self._scene.removeItem(it)
        self._prompt_items = []
        pos = []
        for i, (x, y, lab) in enumerate(clicks or []):
            it = _PromptMarker(i, bool(lab), color)
            it.setPos(float(x), float(y))
            self._scene.addItem(it)
            self._prompt_items.append(it)
            pos.append((x, y))
        self._prompt_positions = np.asarray(pos, np.float32).reshape(-1, 2)
        if box is not None:
            x0, y0, x1, y1 = box
            self._prompt_box_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            pen = QPen(QColor(*color, 220), 1.5, Qt.DashLine)
            pen.setCosmetic(True)
            self._prompt_box_item.setPen(pen)
            self._prompt_box_item.setBrush(QBrush(Qt.NoBrush))
            self._prompt_box_item.setVisible(True)
        else:
            self._prompt_box_item.setVisible(False)

    def set_bones(self, pairs: list[tuple[int, int]]) -> None:
        """Skeleton connections drawn between landmarks that have positions."""
        self._bones = list(pairs)
        self._draw_bones()

    def _draw_bones(self) -> None:
        if not self._bones or len(self._positions) == 0:
            self._bones_item.setVisible(False)
            return
        path = QPainterPath()
        n = len(self._positions)
        for a, b in self._bones:
            if a < n and b < n and np.isfinite(self._positions[a]).all() \
                    and np.isfinite(self._positions[b]).all():
                path.moveTo(float(self._positions[a][0]), float(self._positions[a][1]))
                path.lineTo(float(self._positions[b][0]), float(self._positions[b][1]))
        pen = QPen(QColor(255, 255, 255, 120), 1.6)
        pen.setCosmetic(True)
        self._bones_item.setPen(pen)
        self._bones_item.setPath(path)
        self._bones_item.setVisible(not path.isEmpty())

    def set_animal_mode(self, enabled: bool) -> None:
        """A: clicks and box-drags prompt the segment segmentation instead of
        editing points. Stays armed until toggled off (several clicks are
        normal when refining a mask)."""
        self._animal_mode = enabled
        if enabled:
            self._place_mode = False
        else:
            self.cancel_gesture()
        self._update_cursor()

    def _hit_prompt(self, scene_pos: QPointF) -> int | None:
        if len(self._prompt_positions) == 0:
            return None
        scale = self.transform().m11() or 1.0
        d = np.linalg.norm(self._prompt_positions - [scene_pos.x(), scene_pos.y()], axis=1)
        k = int(np.argmin(d))
        return k if d[k] <= 12 / scale else None

    def set_frame(self, rgb: np.ndarray) -> None:
        """Display a frame. May be lower-res than native (e.g. the tracking
        working resolution): it is scaled to fill the native scene rect, so
        point coordinates stay in native pixels regardless."""
        if self._native_size is None:
            return
        nw, nh = self._native_size
        h, w = rgb.shape[:2]
        if self._display_filter != "none":
            shown = apply_display_filter(rgb, self._display_filter, self._raw_rgb)
        else:
            shown = rgb
        self._prev_rgb = self._raw_rgb
        self._raw_rgb = rgb
        self._pixitem.setPixmap(QPixmap.fromImage(np_to_qimage(shown)))
        if (w, h) != (nw, nh):
            self._pixitem.setTransform(QTransform().scale(nw / w, nh / h))
        else:
            self._pixitem.setTransform(QTransform())

    def fit(self) -> None:
        if self._native_size:
            self.fitInView(QRectF(-0.5, -0.5, *self._native_size), Qt.KeepAspectRatio)
            self._user_zoomed = False

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self._user_zoomed:
            self.fit()

    # ---------------------------------------------------------------- points

    def set_interactive(self, enabled: bool) -> None:
        self._interactive = enabled
        if not enabled:
            # a gesture must not survive into TRACKING: its release would
            # commit an edit at whatever frame the render tick reached
            self.cancel_gesture()
        self._update_cursor()

    def set_switchable(self, on: bool) -> None:
        """A view-only companion that a click makes the working camera: it shows
        a pointing hand, not the busy cursor a tracking run shows (G22)."""
        self._switchable = bool(on)
        self._update_cursor()

    def set_pan_mode(self, enabled: bool) -> None:
        """✋ pan tool: left-drag pans the view; point editing is suspended.
        View-only, so it works during tracking too."""
        self._pan_mode = enabled
        if enabled:
            self.cancel_gesture()
        self._update_cursor()

    def set_place_mode(self, enabled: bool) -> None:
        """N: arm point placement. A stray click must never edit the data —
        adding a point (or circling a region) requires this explicit mode,
        shown by the crosshair cursor. One-shot: the app disarms after use."""
        self._place_mode = enabled
        if not enabled:
            self.cancel_gesture()
        self._update_cursor()

    def set_click_only(self, on: bool) -> None:
        """Armed placement makes single clicks only (a ball marker): the region
        shape (polygon corners, circle / rectangle drags) is ignored (I48)."""
        self._click_only = bool(on)
        self.cancel_gesture()

    def set_region_shape(self, shape: str) -> None:
        """Add ▾: what an armed drag draws — circle (drag), rectangle (drag)
        or polygon (click each corner, Enter closes)."""
        if shape in REGION_SHAPES:
            self._region_shape = shape
            self.cancel_gesture()

    def region_shape(self) -> str:
        return self._region_shape

    def set_display_filter(self, kind: str) -> None:
        """Display-only filter: none / contrast (CLAHE) / bright (gamma) /
        diff (|this frame - previous shown frame| x4). Re-renders the current
        frame; the tracker never sees filtered pixels."""
        if kind not in DISPLAY_FILTERS:
            kind = "none"
        if kind == self._display_filter:
            return
        self._display_filter = kind
        if self._raw_rgb is not None:
            rgb, prev = self._raw_rgb, self._prev_rgb
            shown = apply_display_filter(rgb, kind, prev) if kind != "none" else rgb
            self._pixitem.setPixmap(QPixmap.fromImage(np_to_qimage(shown)))

    def display_filter(self) -> str:
        return self._display_filter

    def set_loupe(self, enabled: bool) -> None:
        self._loupe_enabled = bool(enabled)
        if not enabled:
            self._loupe.hide()

    def set_trails(self, length: int, future: bool) -> None:
        """Trail length in frames (0 = off) and whether the future path is drawn."""
        self._trail_len = max(0, int(length))
        self._trail_future = bool(future)

    def set_onion(self, enabled: bool) -> None:
        self._onion = bool(enabled)
        if not enabled:
            self._motion.ghost_prev = None
            self._motion.ghost_next = None
            self._motion.update()


    def polygon_in_progress(self) -> bool:
        return bool(self._poly_pts)

    def finish_polygon(self) -> bool:
        """Enter while a polygon is being drawn: close it (>= 3 corners)."""
        if len(self._poly_pts) < 3:
            return False
        pts = [[p.x(), p.y()] for p in self._poly_pts]
        self.cancel_gesture()
        self.region_requested.emit("polygon", pts)
        return True

    def _update_loupe(self, view_pos, scene_pt: QPointF) -> None:
        if not (self._loupe_enabled and self._interactive and self._native_size
                and not self._pan_mode and self._in_video(scene_pt)):
            self._loupe.hide()
            return
        marker = None
        color = None
        pid = self._dragging if self._dragging is not None else self._selected
        if pid is not None and pid < len(self._positions) and np.isfinite(self._positions[pid]).all():
            marker = QPointF(float(self._positions[pid][0]), float(self._positions[pid][1]))
            if pid < len(self._metas):
                color = tuple(self._metas[pid].color)
        # the loupe works in pixmap coordinates, whose origin is the picture's
        # corner: scene (pixel-centre) coordinates are shifted by the item's pos (I24)
        o = self._pixitem.pos()
        shift = (lambda q: None if q is None else QPointF(q.x() - o.x(), q.y() - o.y()))
        self._loupe.show_at(self._pixitem.pixmap(), self._pixitem.transform(), shift(scene_pt),
                            QPoint(int(view_pos.x()), int(view_pos.y())),
                            self.transform().m11() or 1.0, shift(marker), color)

    def _update_cursor(self) -> None:
        if self._pan_mode:
            self.setCursor(Qt.OpenHandCursor)
        elif self._place_mode and self._interactive:
            self.setCursor(Qt.CrossCursor)
        elif self._animal_mode and self._interactive:
            self.setCursor(Qt.PointingHandCursor)
        elif not self._interactive and self._switchable:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor if self._interactive else Qt.BusyCursor)

    def set_points(self, positions: np.ndarray, visible: np.ndarray,
                   metas: list, selected: int | None,
                   trails: list[np.ndarray] | None = None,
                   future: list[np.ndarray] | None = None,
                   ghost_prev: np.ndarray | None = None,
                   ghost_next: np.ndarray | None = None,
                   occluded: np.ndarray | None = None,
                   radii: np.ndarray | None = None) -> None:
        """positions (N,2) native px with NaN where undefined at this frame.
        `trails` / `future`: per point (K, 2) past / upcoming positions (the
        canvas fades them); `ghost_prev` / `ghost_next`: (N, 2) positions on
        the neighbouring frames (onion skin); `occluded`: (N,) hand-marked
        hidden on this frame; `radii`: (N,) the circle SAM fitted to a ball
        marker on this frame (NaN for everything else) - drawn in scene px so
        the user sees the segment, not only its centre."""
        self._positions = np.asarray(positions, np.float32).reshape(-1, 2)
        rad = (np.asarray(radii, np.float32).reshape(-1) if radii is not None
               else np.full(len(metas), np.nan, np.float32))
        self._metas = list(metas)  # context menu reads kind/anchor
        self._selected = selected
        n = len(metas)
        occ = (np.asarray(occluded, bool).reshape(-1) if occluded is not None
               else np.zeros(n, bool))
        self._occluded = occ
        while len(self._markers) < n:
            m = _Marker(len(self._markers), self._marker_radius)
            self._scene.addItem(m)
            self._markers.append(m)
            reg = QGraphicsPathItem()   # region outline, scene (native px) scale
            reg.setZValue(6)
            self._scene.addItem(reg)
            self._regions.append(reg)
        for i, (marker, region) in enumerate(zip(self._markers, self._regions)):
            if i >= n:
                marker.setVisible(False)
                region.setVisible(False)
                continue
            meta = metas[i]
            has_pos = bool(i < len(self._positions) and np.isfinite(self._positions[i]).all())
            show = bool(meta.display and has_pos)
            marker.setVisible(show)
            is_group = getattr(meta, "kind", "point") == "group"
            is_ball = getattr(meta, "source", "track") == "ball"
            ball_r = float(rad[i]) if (is_ball and i < len(rad) and np.isfinite(rad[i]) and rad[i] > 0) else 0.0
            region.setVisible(show and (is_group or ball_r > 0))
            if has_pos:
                marker.setPos(*self._positions[i])
                marker.style(meta.color, bool(visible[i]) if i < len(visible) else True,
                             i == selected, meta.name,
                             getattr(meta, "source", "track") == "silhouette",
                             bool(occ[i]) if i < len(occ) else False)
                if ball_r > 0:
                    # the circle fitted to the SAM mask, in native px (scales with zoom)
                    x, y = self._positions[i]
                    path = QPainterPath()
                    path.addEllipse(QRectF(x - ball_r, y - ball_r, 2 * ball_r, 2 * ball_r))
                    region.setPath(path)
                    pen = QPen(QColor(*meta.color, 230 if i == selected else 160),
                               2 if i == selected else 1.2, Qt.SolidLine)
                    pen.setCosmetic(True)
                    region.setPen(pen)
                    region.setBrush(QBrush(Qt.NoBrush))
                elif is_group:
                    x, y = self._positions[i]
                    path = QPainterPath()
                    outline = meta.outline_at((x, y)) if hasattr(meta, "outline_at") else None
                    if outline is not None:
                        path.addPolygon(QPolygonF([QPointF(float(a), float(b)) for a, b in outline]))
                        path.closeSubpath()
                    else:
                        r = float(meta.radius)
                        path.addEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
                    region.setPath(path)
                    pen = QPen(QColor(*meta.color, 220 if i == selected else 130),
                               2 if i == selected else 1.2, Qt.DashLine)
                    pen.setCosmetic(True)
                    region.setPen(pen)
                    region.setBrush(QBrush(Qt.NoBrush))
        # motion overlay: trails (past / future) and onion-skin ghosts
        mo = self._motion
        mo.colors = [tuple(m.color) for m in metas]
        mo.display = [bool(m.display) for m in metas]
        mo.past = list(trails) if (trails is not None and self._trail_len > 0) else []
        mo.future = list(future) if (future is not None and self._trail_future and self._trail_len > 0) else []
        mo.ghost_prev = ghost_prev if self._onion else None
        mo.ghost_next = ghost_next if self._onion else None
        mo.current = self._positions
        mo.selected = selected
        mo.update()
        self._draw_bones()
        self._apply_follow()

    def update_marker(self, pid: int, x: float, y: float) -> None:
        if 0 <= pid < len(self._markers):
            self._markers[pid].setPos(x, y)

    def set_marker_size(self, px: float) -> None:
        """Marker radius in screen px. Small markers let the user see the exact
        pixel under a target whose apparent size changes with camera distance."""
        self._marker_radius = float(np.clip(px, 2.0, 30.0))
        for m in self._markers:
            m.set_radius(self._marker_radius)

    def set_guides(self, lines: list, faint: list | None = None, preds: list | None = None,
                   dim: bool | None = None, notes: list | None = None) -> None:
        """Epipolar guides for the selected landmark: [(pts (M, 2) native px,
        (r, g, b), label), ...]; `faint`: [(pts, (r, g, b)), ...] parts where
        the lens model is guessing (I132); `preds`: [(x, y, (r, g, b), label,
        (px, py) or None), ...] where two or more other cameras put it (G23);
        `dim`: draw the lines faint (default: when there is a ◇; the app also
        dims them once the point is placed in two cameras, G28); `notes`:
        [(x, y, text, (r, g, b)), ...] beside a marker. Empty lists clear them."""
        self._guides.lines = list(lines or [])
        self._guides.faint = list(faint or [])
        self._guides.preds = list(preds or [])
        self._guides.notes = list(notes or [])
        self._guides.dim = bool(self._guides.preds) if dim is None else bool(dim)
        self._guides.update()

    def prediction_count(self) -> int:
        return len(self._guides.preds)

    def scene_px_per_screen_px(self) -> float:
        """How many native video pixels one screen pixel covers at this zoom."""
        return 1.0 / (self.transform().m11() or 1.0)

    def guide_count(self) -> int:
        """How many epipolar guide lines are drawn now."""
        return len(self._guides.lines)

    def set_group_members(self, members: dict[int, np.ndarray]) -> None:
        """Live overlay of a group's internal member points during tracking.
        members: {pid: (M, 2) native px}. Stale pids are hidden."""
        for pid, item in self._member_items.items():
            if pid not in members:
                item.setVisible(False)
        for pid, pts in members.items():
            item = self._member_items.get(pid)
            if item is None:
                item = QGraphicsPathItem()
                item.setZValue(8)
                pen = QPen(QColor(255, 255, 255, 160), 1)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.setBrush(QBrush(QColor(255, 255, 255, 90)))
                self._scene.addItem(item)
                self._member_items[pid] = item
            path = QPainterPath()
            for p in np.asarray(pts, np.float32):
                if np.isfinite(p).all():
                    path.addEllipse(QPointF(float(p[0]), float(p[1])), 2.2, 2.2)
            item.setPath(path)
            item.setVisible(True)

    def clear_group_members(self) -> None:
        for item in self._member_items.values():
            item.setVisible(False)

    # -------------------------------------------------------------- view state

    def view_state(self) -> dict:
        """Zoom/pan needed to restore this exact view later."""
        c = self.mapToScene(self.viewport().rect().center())
        return {"zoom": float(self.transform().m11()), "center_x": float(c.x()),
                "center_y": float(c.y()), "user_zoomed": bool(self._user_zoomed)}

    def set_view_state(self, zoom: float, cx: float, cy: float, user_zoomed: bool) -> None:
        if user_zoomed and zoom > 0:
            self.setTransform(QTransform().scale(zoom, zoom))
            self.centerOn(cx, cy)
            self._user_zoomed = True
        else:
            self.fit()

    def set_follow(self, enabled: bool) -> None:
        self._follow_enabled = bool(enabled)
        if enabled:
            self._apply_follow()  # toggling it on brings the points back NOW

    def _apply_follow(self, margin: int = 70) -> None:
        """Keep the tracked content in view — with predictable rules:

        - a selected point WITH data here: minimal pan to keep it inside the
          margin box (never zooms — your zoom is respected for close work);
        - a selected point WITHOUT data here: leave the view alone (jumping to
          some other target would be disorienting);
        - no selection: AUTO-FRAME all valid points — fit the min/max bbox of
          their positions plus a buffer, clamped to the frame edges. Zooms in
          AND out as the constellation moves and spreads; a small hysteresis
          band keeps it from re-fitting on every pixel of motion.

        Runs after every set_points() and after every zoom change, so scrub,
        tracking ticks, wheel and keyboard zoom all behave identically.
        """
        if self._native_size is None or not self._follow_enabled:
            return
        pts = self._positions
        if len(pts) == 0 and self._mask_rect is None:
            return
        vp = self.viewport().rect()
        m = min(margin, vp.width() // 4, vp.height() // 4)
        sel = self._selected

        if sel is None or sel >= len(pts):
            # auto-frame: bbox of all valid points AND the segment silhouette (so
            # follow works with a segment alone) + buffer, clamped to the frame
            valid = pts[np.isfinite(pts).all(axis=1)] if len(pts) else np.zeros((0, 2), np.float32)
            if self._mask_rect is not None:
                r = self._mask_rect
                corners = np.array([[r.left(), r.top()], [r.right(), r.bottom()]], np.float32)
                valid = np.concatenate([valid.reshape(-1, 2), corners], axis=0)
            if len(valid) == 0:
                return
            lo, hi = valid.min(axis=0), valid.max(axis=0)
            nw, nh = self._native_size
            buf = max(50.0, 0.25 * float(max(hi[0] - lo[0], hi[1] - lo[1])))
            rect = QRectF(float(lo[0]) - buf, float(lo[1]) - buf,
                          float(hi[0] - lo[0]) + 2 * buf,
                          float(hi[1] - lo[1]) + 2 * buf)
            rect = rect.intersected(QRectF(-0.5, -0.5, nw, nh))  # frame-edge clamp
            view = self.mapToScene(vp).boundingRect()
            # refit only when the target escapes the view or the view is far
            # too loose — not on every frame of ordinary motion
            if (not view.contains(rect)
                    or view.width() > 1.8 * rect.width()
                    or view.height() > 1.8 * rect.height()):
                self.fitInView(rect, Qt.KeepAspectRatio)
            return

        if not np.isfinite(pts[sel]).all():
            return  # gap in the selected track: hold the view steady
        p = self.mapFromScene(QPointF(float(pts[sel][0]), float(pts[sel][1])))
        dx = dy = 0
        if p.x() < m:
            dx = p.x() - m
        elif p.x() > vp.width() - m:
            dx = p.x() - (vp.width() - m)
        if p.y() < m:
            dy = p.y() - m
        elif p.y() > vp.height() - m:
            dy = p.y() - (vp.height() - m)
        if dx:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(dx))
        if dy:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(dy))

    # ----------------------------------------------------------- interaction

    def _hit_test(self, scene_pos: QPointF) -> int | None:
        if len(self._positions) == 0:
            return None
        scale = self.transform().m11() or 1.0
        # grabbing must work on the whole marker even when it is drawn large
        radius = max(HIT_RADIUS_PX, self._marker_radius + 4) / scale
        d = np.linalg.norm(self._positions - [scene_pos.x(), scene_pos.y()], axis=1)
        # a point whose panel tick is off is not drawn: grabbing it moved an
        # invisible point instead of the visible one under the cursor (I46)
        shown = np.array([bool(getattr(m, "display", True)) for m in self._metas[:len(d)]], bool)
        if len(shown):
            d[:len(shown)][~shown] = np.inf
        with np.errstate(invalid="ignore"):
            cand = np.where(np.nan_to_num(d, nan=np.inf) <= radius)[0]
        return int(cand[np.argmin(d[cand])]) if len(cand) else None

    def _in_video(self, p: QPointF) -> bool:
        if self._native_size is None:
            return False
        w, h = self._native_size
        return -0.5 <= p.x() < w - 0.5 and -0.5 <= p.y() < h - 0.5

    @staticmethod
    def _onpic(p: QPointF) -> QPointF:
        """A position the session accepts: pixel 0 spans [-0.5, 0.5), but a
        coordinate below 0 counts as out of the picture there, so a click in
        that outer half-pixel became a point that tracking blanked (I126)."""
        return QPointF(max(0.0, p.x()), max(0.0, p.y()))

    def cancel_gesture(self) -> None:
        """Abort any in-progress gesture: circle drag, deferred click, box drag,
        or marker drag (Esc, interaction disabled, point set changed)."""
        self._press_scene = None
        self._press_view = None
        self._circle_active = False
        self._circle_preview.setVisible(False)
        self._box_active = False
        self._box_preview.setVisible(False)
        self._plain_press = False
        self._press_hit = None
        self._rpress = None
        self._rpress_timer.stop()
        self._rect_active = False
        self._rect_preview.setVisible(False)
        self._poly_pts = []
        self._poly_preview.setVisible(False)
        self._dragging = None
        self._drag_moved = False

    def mousePressEvent(self, ev):
        # announced before anything else: in a multi-camera grid a click on a
        # companion view must make it the working one, and it is view-only
        # until then, so no edit can be lost by the switch
        was_interactive = self._interactive
        self.view_clicked.emit()
        armed_here = self._interactive and self._place_mode and ev.button() == Qt.LeftButton
        if not was_interactive and ev.button() != Qt.MiddleButton and not self._pan_mode and not armed_here:
            # a click on a view-only (companion) tile only SWITCHES to it: the
            # switch makes it interactive synchronously, and the same press then
            # hand-placed its selected point (I110). With Add armed the click is
            # an explicit placement: the switch carries Add over, and the press
            # places the point in the camera clicked (G20) -- before, it was
            # swallowed and the NEXT click, unarmed, placed nothing
            return
        if ev.button() == Qt.MiddleButton or \
                (ev.button() == Qt.LeftButton and self._pan_mode and self._native_size):
            self._panning = True
            self._pan_button = ev.button()
            self._pan_last = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        sp = self.mapToScene(ev.position().toPoint())
        if self._animal_mode and self._interactive and self._native_size:
            if ev.button() == Qt.LeftButton and self._in_video(sp):
                # deferred: release decides between a click (prompt) and a box drag
                self._press_scene = self._onpic(sp)
                self._press_view = ev.position()
                self._box_active = False
                return
            if ev.button() == Qt.RightButton:
                k = self._hit_prompt(sp)
                if k is not None:
                    menu = QMenu(self)
                    menu.setToolTipsVisible(True)      # (G15)
                    act = menu.addAction("Remove this click")
                    chosen = menu.exec(ev.globalPosition().toPoint())
                    menu.deleteLater()
                    if chosen == act:
                        self.prompt_remove_requested.emit(k)
                return
            super().mousePressEvent(ev)
            return
        if ev.button() == Qt.LeftButton and self._interactive and self._native_size:
            if ev.modifiers() & Qt.AltModifier and not self._place_mode:
                # look-here: never an edit, so it cannot pass for a placement (G21)
                if self._in_video(sp):
                    sp = self._onpic(sp)
                    self.probe_requested.emit(sp.x(), sp.y())
                return
            hit = self._hit_test(sp)
            if ev.modifiers() & Qt.ControlModifier:
                if self._in_video(sp):
                    sp = self._onpic(sp)
                    self.reposition_requested.emit(sp.x(), sp.y())
                return
            if not self._place_mode:
                # unarmed (G59): the release decides -- a click places the
                # selected point (on a marker too), a press held and moved
                # pans the view, starting on a marker as anywhere else
                # (markers are no longer dragged: a click places the point)
                self._press_scene = self._onpic(sp) if self._in_video(sp) else None
                self._press_view = ev.position()
                self._press_hit = hit
                self._plain_press = True
                self._circle_active = False
                self._rect_active = False
                return
            if self._in_video(sp):
                if self._place_mode and self._region_shape == "polygon" and not self._click_only:
                    # polygon region: every click adds a corner; Enter closes
                    # it (>= 3 corners), Esc cancels. Double-click also closes.
                    self._poly_pts.append(self._onpic(sp))
                    self._draw_poly_preview(sp)
                    return
                # deferred: release decides between a click (add point) and a
                # circle / rectangle drag (add region group). Unarmed, a clean
                # click is a manual annotation of the SELECTED point at this
                # frame (the app decides; nothing selected = nothing edited) —
                # a drag without N does nothing at all.
                self._press_scene = self._onpic(sp)
                self._press_view = ev.position()
                self._circle_active = False
                self._rect_active = False
                self._plain_press = not self._place_mode
                return
        if ev.button() == Qt.RightButton and self._interactive and self._native_size:
            # never during a left press / pan: the menu's popup grab would swallow
            # the release and leave the gesture armed forever
            busy = self._dragging is not None or self._plain_press or self._panning
            hit = self._hit_test(sp) if not busy else None
            if hit is not None:
                # G59: a click clears this point on this frame, a long press opens its menu
                self._rpress = (hit, ev.globalPosition().toPoint())
                self._rpress_timer.start()
                return
        super().mousePressEvent(ev)

    def _on_long_right_press(self) -> None:
        """The right button has been held on a marker: its menu (G59)."""
        if self._rpress is None:
            return
        pid, pos = self._rpress
        self._rpress = None
        if self._interactive:
            if callable(self.multi_menu) and self.multi_menu(pid, pos):
                return                          # one of several selected points: the menu for all (G64)
            self._context_menu(pid, pos)

    def mouseMoveEvent(self, ev):
        if self._panning:
            delta = ev.position() - self._pan_last
            self._pan_last = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if self._loupe_enabled:
            self._update_loupe(ev.position(), self.mapToScene(ev.position().toPoint()))
        if self._poly_pts and self._place_mode:
            self._draw_poly_preview(self.mapToScene(ev.position().toPoint()))
            return
        if not self._interactive and (self._dragging is not None
                                      or self._press_scene is not None):
            self.cancel_gesture()  # interaction revoked mid-gesture
            return
        if self._dragging is not None:
            sp = self.mapToScene(ev.position().toPoint())
            if self._in_video(sp):
                self._drag_moved = True
                sp = self._onpic(sp)
                self.update_marker(self._dragging, sp.x(), sp.y())
                self.point_moved.emit(self._dragging, sp.x(), sp.y())
            return
        if self._press_scene is not None and self._animal_mode:
            if (not self._box_active
                    and (ev.position() - self._press_view).manhattanLength() > DRAG_CIRCLE_PX):
                self._box_active = True
                self._box_preview.setVisible(True)
            if self._box_active:
                sp = self.mapToScene(ev.position().toPoint())
                self._box_preview.setRect(QRectF(self._press_scene, sp).normalized())
            return
        if self._plain_press:
            if (ev.position() - self._press_view).manhattanLength() > DRAG_CIRCLE_PX:
                # press, hold and move = pan (G59); a drag is never a placement
                start = self._press_view
                self.cancel_gesture()
                self._panning = True
                self._pan_button = Qt.LeftButton
                self._pan_last = start
                self.setCursor(Qt.ClosedHandCursor)
                self.mouseMoveEvent(ev)
            return
        if self._press_scene is not None and self._click_only:
            return                   # a ball marker: a click, never a region gesture (I48)
        if self._press_scene is not None and self._region_shape == "rect":
            if (not self._rect_active
                    and (ev.position() - self._press_view).manhattanLength() > DRAG_CIRCLE_PX):
                self._rect_active = True
                self._rect_preview.setVisible(True)
            if self._rect_active:
                sp = self.mapToScene(ev.position().toPoint())
                self._rect_preview.setRect(QRectF(self._press_scene, sp).normalized())
            return
        if self._press_scene is not None:
            if (not self._circle_active
                    and (ev.position() - self._press_view).manhattanLength() > DRAG_CIRCLE_PX):
                self._circle_active = True
                self._circle_preview.setVisible(True)
            if self._circle_active:
                sp = self.mapToScene(ev.position().toPoint())
                r = float(np.hypot(sp.x() - self._press_scene.x(),
                                   sp.y() - self._press_scene.y()))
                c = self._press_scene
                self._circle_preview.setRect(c.x() - r, c.y() - r, 2 * r, 2 * r)
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton and self._rpress is not None:
            pid = self._rpress[0]
            self._rpress = None
            self._rpress_timer.stop()
            if self._interactive:              # a short right click: clear it on this frame (G59);
                self.clear_frame_requested.emit(pid)   # the app selects it, or clears every selected one
            return
        if self._panning and ev.button() == getattr(self, "_pan_button", Qt.MiddleButton):
            self._panning = False
            self._update_cursor()
            return
        if not self._interactive and (self._dragging is not None
                                      or self._press_scene is not None):
            self.cancel_gesture()  # never commit an edit after interaction was revoked
            return
        if ev.button() == Qt.LeftButton and self._dragging is not None:
            pid = self._dragging
            self._dragging = None
            if self._drag_moved:
                sp = self.mapToScene(ev.position().toPoint())
                if self._in_video(sp):
                    sp = self._onpic(sp)
                    self.move_committed.emit(pid, sp.x(), sp.y())
            return
        if ev.button() == Qt.LeftButton and self._press_scene is not None and self._animal_mode:
            press = self._press_scene
            was_box = self._box_active
            self.cancel_gesture()
            sp = self.mapToScene(ev.position().toPoint())
            if was_box:
                r = QRectF(press, sp).normalized()
                if min(r.width(), r.height()) >= MIN_BOX_PX:
                    self.animal_box.emit(r.left(), r.top(), r.right(), r.bottom())
                    return
            positive = not bool(ev.modifiers() & Qt.ShiftModifier)
            self.animal_click.emit(press.x(), press.y(), positive)
            return
        if ev.button() == Qt.LeftButton and self._plain_press:
            press, hit = self._press_scene, self._press_hit
            self.cancel_gesture()
            if self._selected is None and hit is not None:
                self.point_selected.emit(hit)       # nothing to place: the click picks the point
            elif press is not None:
                self.annotate_requested.emit(press.x(), press.y())
            return
        if ev.button() == Qt.LeftButton and self._press_scene is not None and self._rect_active:
            press = self._press_scene
            self.cancel_gesture()
            sp = self.mapToScene(ev.position().toPoint())
            r = QRectF(press, sp).normalized()
            if min(r.width(), r.height()) >= MIN_GROUP_RADIUS:
                self.region_requested.emit("rect", [[r.left(), r.top()], [r.right(), r.top()],
                                                    [r.right(), r.bottom()], [r.left(), r.bottom()]])
            else:
                self.add_requested.emit(press.x(), press.y())
            return
        if ev.button() == Qt.LeftButton and self._press_scene is not None:
            press = self._press_scene
            was_circle = self._circle_active
            self.cancel_gesture()
            sp = self.mapToScene(ev.position().toPoint())
            if was_circle:
                r = float(np.hypot(sp.x() - press.x(), sp.y() - press.y()))
                if r >= MIN_GROUP_RADIUS:
                    self.group_requested.emit(press.x(), press.y(), r)
                else:  # a wobbly click, not a region
                    self.add_requested.emit(press.x(), press.y())
            else:
                self.add_requested.emit(press.x(), press.y())
            return
        super().mouseReleaseEvent(ev)

    def _draw_poly_preview(self, cursor: QPointF | None) -> None:
        if not self._poly_pts:
            self._poly_preview.setVisible(False)
            return
        path = QPainterPath(self._poly_pts[0])
        for q in self._poly_pts[1:]:
            path.lineTo(q)
        if cursor is not None:
            path.lineTo(cursor)
        path.closeSubpath()
        self._poly_preview.setPath(path)
        self._poly_preview.setVisible(True)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._poly_pts and self._place_mode:
            # the double-click's second press already added a duplicate corner
            if len(self._poly_pts) >= 2:
                a, b = self._poly_pts[-1], self._poly_pts[-2]
                if abs(a.x() - b.x()) < 1e-6 and abs(a.y() - b.y()) < 1e-6:
                    self._poly_pts.pop()
            if not self.finish_polygon():
                self.cancel_gesture()
            return
        super().mouseDoubleClickEvent(ev)

    def leaveEvent(self, ev):
        self._loupe.hide()
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        if self._native_size is None:
            return
        factor = ZOOM_STEP if ev.angleDelta().y() > 0 else 1 / ZOOM_STEP
        current = self.transform().m11()
        if 0.02 < current * factor < 60:
            self.scale(factor, factor)
            self._user_zoomed = True
            self._apply_follow()

    def zoom_step(self, factor: float) -> None:
        """Keyboard zoom (+/-), anchored under the mouse pointer when it is
        over the video, else at the viewport center."""
        if self._native_size is None:
            return
        current = self.transform().m11()
        if not (0.02 < current * factor < 60):
            return
        vp = self.viewport()
        pos = vp.mapFromGlobal(QCursor.pos())
        if not vp.rect().contains(pos):
            pos = vp.rect().center()
        anchor = self.mapToScene(pos)
        prev = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.scale(factor, factor)
        moved = self.mapToScene(pos) - anchor  # pull the anchor back under the pointer
        self.translate(moved.x(), moved.y())
        self.setTransformationAnchor(prev)
        self._user_zoomed = True
        self._apply_follow()

    def _build_context_menu(self, pid: int) -> tuple[QMenu, dict]:
        """Point context menu, shared by the canvas right-click and the point
        list panel. Returned separately from exec() so tests can inspect it.

        The canvas knows nothing about frames, so the app hangs the
        session-aware half (jump to this point's first / last frame, clear its
        position here or over a window) on the end through `menu_extra`, and
        handles those choices through `menu_extra_action`."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        acts = {"rename": menu.addAction("Rename point")}
        meta = self._metas[pid] if pid < len(getattr(self, "_metas", [])) else None
        is_point = meta is not None and getattr(meta, "kind", "point") == "point"
        act_anchor = menu.addAction("Lock to seed appearance (re-anchor)")
        act_anchor.setCheckable(True)
        if is_point:
            act_anchor.setChecked(bool(getattr(meta, "anchor", False)))
            act_anchor.setToolTip(
                "Snap the track back whenever this point's original appearance is "
                "found nearby — catches slow drift on distinct rigid features. "
                "Changes the plain 'track the placed location' behavior; off by default.")
        else:
            # visible-but-disabled beats hidden: the user learns why it's not
            # available for regions instead of never finding the feature
            act_anchor.setEnabled(False)
            act_anchor.setToolTip(
                "Appearance lock applies to plain points only — a region group's "
                "center is already stabilized by its member-constellation fit.")
        acts["anchor"] = act_anchor
        # where the point's data comes from: appearance tracking (default) or
        # the animal's silhouette (tail tip, midline fractions, extremities)
        from kinetrace.skeletons import DERIVED_CHOICES
        sub = menu.addMenu("Data source")
        cur_spec = getattr(meta, "spec", "") if meta is not None else ""
        derived = meta is not None and getattr(meta, "source", "track") == "silhouette"
        acts["source"] = {}
        a = sub.addAction("Track by appearance (click to seed it)")
        a.setCheckable(True)
        a.setChecked(not derived)
        acts["source"][a] = ""
        sub.addSeparator()
        for spec, label in DERIVED_CHOICES:
            a = sub.addAction(f"From silhouette: {label}")
            a.setCheckable(True)
            a.setChecked(derived and cur_spec == spec)
            acts["source"][a] = spec
        sub.menuAction().setToolTip("Silhouette-derived landmarks need a segment (S) and fill in "
                       "when you track")
        act_occ = menu.addAction("Hidden on this frame (keep, do not export)")
        act_occ.setCheckable(True)
        act_occ.setChecked(bool(self._occluded[pid]) if pid < len(self._occluded) else False)
        act_occ.setToolTip("Mark this point as not really visible here: the position stays in the\n"
                           "project (so you can unmark it) but is left blank in every export and\n"
                           "ignored by 3D. Shift+X toggles it for the selected point; on the\n"
                           "timeline, Shift+drag a window and right-click to mark a stretch.")
        acts["occluded"] = act_occ
        act_free = menu.addAction("May leave the segment (free point)")
        act_free.setCheckable(True)
        act_free.setChecked(bool(getattr(meta, "free", False)) if meta is not None else False)
        act_free.setToolTip("With the Body toggle on, tracked points are kept inside the segment's "
                            "silhouette, and one that clearly leaves it stops the run. Check this for a "
                            "point that legitimately lives elsewhere (a marker on the ground, a reference "
                            "object).")
        acts["free"] = act_free
        acts["delete"] = menu.addAction("Delete point")
        if callable(self.menu_extra):
            self.menu_extra(menu, acts, pid)
        return menu, acts

    def _context_menu(self, pid: int, global_pos) -> None:
        menu, acts = self._build_context_menu(pid)
        chosen = menu.exec(global_pos)
        menu.deleteLater()          # else every right-click leaves a menu behind as a child
        if callable(self.menu_extra_action) and self.menu_extra_action(chosen, acts, pid):
            return
        if chosen == acts["rename"]:
            self.rename_requested.emit(pid)
        elif chosen == acts["anchor"]:
            self.anchor_toggled.emit(pid, acts["anchor"].isChecked())
        elif chosen == acts["delete"]:
            self.delete_requested.emit(pid)
        elif chosen in acts["source"]:
            self.source_change_requested.emit(pid, acts["source"][chosen])
        elif chosen == acts["free"]:
            self.free_toggled.emit(pid, acts["free"].isChecked())
        elif chosen == acts["occluded"]:
            self.occluded_toggled.emit(pid, acts["occluded"].isChecked())
