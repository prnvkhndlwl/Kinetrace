"""Toolbar icons drawn in code (no image assets, crisp at any DPI, theme colors).

Every icon is a 20 px glyph rendered at 2x device pixel ratio with the
design-token text color; the checked/hover states come from the button
stylesheet, not from the icon.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from kinetrace import theme

SIZE = 20


def _icon(draw, color: str | None = None, width: float = 1.8) -> QIcon:
    dpr = 2
    pm = QPixmap(SIZE * dpr, SIZE * dpr)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color or theme.TEXT), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    draw(p, QColor(color or theme.TEXT))
    p.end()
    return QIcon(pm)


def _tri(p, c, pts):
    p.setBrush(QBrush(c))
    p.drawPolygon(QPolygonF([QPointF(*q) for q in pts]))
    p.setBrush(Qt.NoBrush)


@lru_cache(maxsize=None)
def prev() -> QIcon:
    return _icon(lambda p, c: (_tri(p, c, [(14, 4.5), (14, 15.5), (6.5, 10)]),
                               p.drawLine(QPointF(5, 5), QPointF(5, 15))))


@lru_cache(maxsize=None)
def next_() -> QIcon:
    return _icon(lambda p, c: (_tri(p, c, [(6, 4.5), (6, 15.5), (13.5, 10)]),
                               p.drawLine(QPointF(15, 5), QPointF(15, 15))))


@lru_cache(maxsize=None)
def play() -> QIcon:
    return _icon(lambda p, c: _tri(p, c, [(6, 4), (6, 16), (16, 10)]))


@lru_cache(maxsize=None)
def pause() -> QIcon:
    def d(p, c):
        p.setBrush(QBrush(c))
        p.drawRoundedRect(QRectF(5, 4, 3.5, 12), 1, 1)
        p.drawRoundedRect(QRectF(11.5, 4, 3.5, 12), 1, 1)
    return _icon(d)


def _lens(p):
    p.drawEllipse(QRectF(3.5, 3.5, 9.5, 9.5))
    p.drawLine(QPointF(11.5, 11.5), QPointF(16.5, 16.5))


@lru_cache(maxsize=None)
def zoom_in() -> QIcon:
    def d(p, c):
        _lens(p)
        p.drawLine(QPointF(6, 8.25), QPointF(10.5, 8.25))
        p.drawLine(QPointF(8.25, 6), QPointF(8.25, 10.5))
    return _icon(d)


@lru_cache(maxsize=None)
def zoom_out() -> QIcon:
    def d(p, c):
        _lens(p)
        p.drawLine(QPointF(6, 8.25), QPointF(10.5, 8.25))
    return _icon(d)


@lru_cache(maxsize=None)
def zoom_fit() -> QIcon:
    def d(p, c):
        for x0, y0, dx, dy in ((4, 4, 1, 1), (16, 16, -1, -1)):
            p.drawLine(QPointF(x0, y0), QPointF(x0 + 4.5 * dx, y0))
            p.drawLine(QPointF(x0, y0), QPointF(x0, y0 + 4.5 * dy))
        p.drawLine(QPointF(7, 7), QPointF(13, 13))
    return _icon(d)


@lru_cache(maxsize=None)
def add() -> QIcon:
    def d(p, c):
        p.drawEllipse(QRectF(5, 5, 10, 10))
        p.drawLine(QPointF(10, 2), QPointF(10, 6.5))
        p.drawLine(QPointF(10, 13.5), QPointF(10, 18))
        p.drawLine(QPointF(2, 10), QPointF(6.5, 10))
        p.drawLine(QPointF(13.5, 10), QPointF(18, 10))
    return _icon(d)


@lru_cache(maxsize=None)
def segment() -> QIcon:
    def d(p, c):
        path = QPainterPath(QPointF(4, 11))
        path.cubicTo(QPointF(3, 4), QPointF(11, 2), QPointF(14, 6))
        path.cubicTo(QPointF(18, 9), QPointF(16, 16), QPointF(10, 16.5))
        path.cubicTo(QPointF(6, 17), QPointF(4.5, 14), QPointF(4, 11))
        p.drawPath(path)
        p.setBrush(QBrush(c))
        p.drawEllipse(QRectF(8.5, 8.5, 3.5, 3.5))
    return _icon(d)


@lru_cache(maxsize=None)
def mask() -> QIcon:
    """The silhouette overlay: a filled, translucent blob with its outline."""
    def d(p, c):
        path = QPainterPath(QPointF(4, 11))
        path.cubicTo(QPointF(3, 4), QPointF(11, 2), QPointF(14, 6))
        path.cubicTo(QPointF(18, 9), QPointF(16, 16), QPointF(10, 16.5))
        path.cubicTo(QPointF(6, 17), QPointF(4.5, 14), QPointF(4, 11))
        fill = QColor(c)
        fill.setAlpha(90)
        p.setBrush(QBrush(fill))
        p.drawPath(path)
        p.setBrush(Qt.NoBrush)
    return _icon(d)


@lru_cache(maxsize=None)
def body() -> QIcon:
    """On-body constraint: a point held inside an outline."""
    def d(p, c):
        p.drawEllipse(QRectF(3, 4, 14, 12))
        p.setBrush(QBrush(c))
        p.drawEllipse(QRectF(8.25, 8.25, 3.5, 3.5))
        p.setBrush(Qt.NoBrush)
    return _icon(d)


@lru_cache(maxsize=None)
def follow() -> QIcon:
    """Follow: a crosshair reticle."""
    def d(p, c):
        p.drawEllipse(QRectF(5, 5, 10, 10))
        for a, b in (((10, 2), (10, 5)), ((10, 15), (10, 18)), ((2, 10), (5, 10)), ((15, 10), (18, 10))):
            p.drawLine(QPointF(*a), QPointF(*b))
        p.setBrush(QBrush(c))
        p.drawEllipse(QRectF(9, 9, 2, 2))
        p.setBrush(Qt.NoBrush)
    return _icon(d)


@lru_cache(maxsize=None)
def autopause() -> QIcon:
    """Auto-pause: a pause glyph with a warning tick."""
    def d(p, c):
        p.setBrush(QBrush(c))
        p.drawRect(QRectF(5, 4, 3, 12))
        p.drawRect(QRectF(11, 4, 3, 12))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(15.5, 13.5), QPointF(17.5, 15.5))
        p.drawLine(QPointF(17.5, 15.5), QPointF(19, 11.5))
    return _icon(d)


@lru_cache(maxsize=None)
def roi() -> QIcon:
    """ROI zoom: a dashed crop frame with a small target inside."""
    def d(p, c):
        pen = p.pen()
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRect(QRectF(3, 3, 14, 14))
        pen.setStyle(Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(QBrush(c))
        p.drawEllipse(QRectF(8.25, 8.25, 3.5, 3.5))
        p.setBrush(Qt.NoBrush)
    return _icon(d)


@lru_cache(maxsize=None)
def pan() -> QIcon:
    """Pan: four-way arrows."""
    def d(p, c):
        p.drawLine(QPointF(10, 3), QPointF(10, 17))
        p.drawLine(QPointF(3, 10), QPointF(17, 10))
        for tip, a, b in (((10, 2.5), (7.5, 5), (12.5, 5)), ((10, 17.5), (7.5, 15), (12.5, 15)),
                          ((2.5, 10), (5, 7.5), (5, 12.5)), ((17.5, 10), (15, 7.5), (15, 12.5))):
            _tri(p, c, [tip, a, b])
    return _icon(d)


@lru_cache(maxsize=None)
def panel() -> QIcon:
    """The side panel: a window with a right-hand column."""
    def d(p, c):
        p.drawRoundedRect(QRectF(3, 4, 14, 12), 2, 2)
        p.drawLine(QPointF(12, 4), QPointF(12, 16))
    return _icon(d)


def _eye_outline(p):
    path = QPainterPath(QPointF(2.5, 10))
    path.quadTo(QPointF(10, 2.5), QPointF(17.5, 10))
    path.quadTo(QPointF(10, 17.5), QPointF(2.5, 10))
    p.drawPath(path)


@lru_cache(maxsize=None)
def eye() -> QIcon:
    """(G169) A camera's view is shown: an open eye."""
    def d(p, c):
        _eye_outline(p)
        p.setBrush(QBrush(c))
        p.drawEllipse(QRectF(7.75, 7.75, 4.5, 4.5))
        p.setBrush(Qt.NoBrush)
    return _icon(d)


@lru_cache(maxsize=None)
def eye_off() -> QIcon:
    """(G169) A camera's view is hidden: the eye struck through, dimmed."""
    def d(p, c):
        _eye_outline(p)
        p.drawLine(QPointF(4, 16), QPointF(16, 4))
    return _icon(d, theme.TEXT_DIM)


@lru_cache(maxsize=None)
def check() -> QIcon:
    """A done tick, in the accent colour."""
    def d(p, c):
        p.drawLine(QPointF(4, 10.5), QPointF(8.5, 15))
        p.drawLine(QPointF(8.5, 15), QPointF(16, 6))
    return _icon(d, theme.ACCENT, 2.2)
