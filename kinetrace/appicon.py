"""The app icon, drawn in code - no image files.

The mark: a camera viewfinder (four rounded corner brackets) around three
tracked points, each with its own colour and a dotted trail that fades
behind it, ending in a glowing head (bright core, coloured halo). The three
move three ways - the three media Kinetrace tracks in:

* air: a glide, descending from high on the right to its head at the lower left (amber),
* water: an undulating swimming wave (cyan),
* land: a bounding gait, one hump per stride (blue).

The colours are the app's own: the dark window surface and the theme's
hairline rim (`theme.py`), the brackets in the dim text grey, and the three
trails in the point colours the app gives landmarks (`session.PALETTE`), so
the icon's dots look like the markers on screen. Small sizes (below
`DETAIL_MIN` px) drop the soft motion streaks and the halos and thicken the
lines. Rendered sizes are cached as PNGs in `kinetrace/_theme_cache/` (bump
`ICON_VERSION` after changing the drawing).

    python -m kinetrace.appicon kinetrace-icon.png [--size 1024] [--simple]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient)

from kinetrace import theme
from kinetrace.session import PALETTE

ICON_VERSION = 5               # 4: viewfinder + three trails (air / water / land); 5: the glide descends
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256, 512)
DETAIL_MIN = 48                # below this: no streaks or halos, thicker lines
CACHE = Path(__file__).resolve().parent / "_theme_cache"
TILE_RADIUS = 0.22             # rounded-square corner, x size (the usual app-tile shape)


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


STOCK = _rgb(theme.BG_WINDOW)
RIM = _rgb(theme.HAIRLINE)
BRACKET = _rgb(theme.TEXT_DIM)
AIR, WATER, LAND = PALETTE[3], PALETTE[5], PALETTE[2]     # amber, cyan, blue: landmark colours


# ------------------------------------------------------------------ the paths
def _air(t: np.ndarray) -> np.ndarray:
    """A glide: from high on the right, descending to the lower left (gliding
    goes from high to low), so the head is at the low end."""
    t = 1.0 - t
    x = 0.21 + 0.55 * t
    y = 0.46 - 0.24 * np.sin(0.5 * np.pi * t) ** 1.3 + 0.03 * t
    return np.stack([x, y], axis=1)


def _water(t: np.ndarray) -> np.ndarray:
    """A swimming wave: one and a half undulations."""
    x = 0.20 + 0.49 * t
    y = 0.62 - 0.12 * t + 0.042 * np.sin(3.0 * np.pi * t)
    return np.stack([x, y], axis=1)


def _land(t: np.ndarray) -> np.ndarray:
    """A bounding gait: a hump per stride, the feet down between them."""
    x = 0.22 + 0.41 * t
    y = 0.80 - 0.09 * t - 0.075 * np.abs(np.sin(2.0 * np.pi * t)) ** 0.8
    return np.stack([x, y], axis=1)


TRAILS = ((_air, AIR), (_water, WATER), (_land, LAND))


def _even(fn, n: int) -> np.ndarray:
    """n points evenly spaced ALONG the path (not in its parameter), so the
    dots do not crowd on the curves."""
    dense = fn(np.linspace(0.0, 1.0, 600))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    at = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(at, s, dense[:, 0]), np.interp(at, s, dense[:, 1])], axis=1)


# ------------------------------------------------------------------ drawing
def _brackets(p: QPainter, S: float, width: float) -> None:
    """The viewfinder: four rounded L corners."""
    pen = QPen(QColor(*BRACKET, 235), width * S)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    m, arm, rr = 0.15, 0.15, 0.07                  # inset, arm length, corner radius
    for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x0 = m if sx == 0 else 1 - m
        y0 = m if sy == 0 else 1 - m
        dx = 1 if sx == 0 else -1
        dy = 1 if sy == 0 else -1
        path = QPainterPath(QPointF((x0 + dx * arm) * S, y0 * S))
        path.lineTo((x0 + dx * rr) * S, y0 * S)
        path.quadTo(x0 * S, y0 * S, x0 * S, (y0 + dy * rr) * S)
        path.lineTo(x0 * S, (y0 + dy * arm) * S)
        p.drawPath(path)


def _streak(p: QPainter, S: float, fn, colour) -> None:
    """A soft motion streak under the older half of a trail."""
    t = np.linspace(0.0, 0.72, 40)
    pts = fn(t)
    for w, a in ((0.050, 14), (0.030, 20)):
        pen = QPen(QColor(*colour, a), w * S)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath(QPointF(*(pts[0] * S)))
        for q in pts[1:]:
            path.lineTo(*(q * S))
        p.drawPath(path)


def _trail(p: QPainter, S: float, fn, colour, n: int, big: bool) -> QPointF:
    """Dots that grow and brighten towards the head; returns the head position."""
    p.setPen(Qt.NoPen)
    pts = _even(fn, n)
    for k in range(n - 1):
        f = k / (n - 2) if n > 2 else 1.0
        r = (0.006 + 0.012 * f) if big else (0.020 + 0.018 * f)
        p.setBrush(QColor(*colour, int(30 + 200 * f ** 1.4)))
        p.drawEllipse(QPointF(*(pts[k] * S)), r * S, r * S)
    return QPointF(*(pts[-1] * S))


def _head(p: QPainter, S: float, at: QPointF, colour, big: bool) -> None:
    """The tracked point: a coloured halo (large sizes), a coloured ring, a bright core."""
    if big:
        g = QRadialGradient(at, 0.075 * S)
        g.setColorAt(0.0, QColor(*colour, 150))
        g.setColorAt(0.45, QColor(*colour, 55))
        g.setColorAt(1.0, QColor(*colour, 0))
        p.setBrush(g)
        p.setPen(Qt.NoPen)
        p.drawEllipse(at, 0.075 * S, 0.075 * S)
    r = (0.030 if big else 0.060) * S
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(*colour))
    p.drawEllipse(at, r, r)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawEllipse(at, r * 0.55, r * 0.55)


def render(size: int, simple: bool | None = None) -> QImage:
    """The icon at `size` px (`simple` = the small-size drawing; default by size)."""
    simple = size < DETAIL_MIN if simple is None else bool(simple)
    S = float(size)
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(0, 0, S, S), TILE_RADIUS * S, TILE_RADIUS * S)
    p.fillPath(tile, QColor(*STOCK))
    p.setClipPath(tile)
    _brackets(p, S, 0.036 if not simple else 0.060)
    heads = []
    for fn, colour in TRAILS:
        if not simple:
            _streak(p, S, fn, colour)
        heads.append((_trail(p, S, fn, colour, 16 if not simple else 4, not simple), colour))
    for at, colour in heads:                       # the heads on top of every trail
        _head(p, S, at, colour, not simple)
    p.setClipping(False)
    w = max(1.0, S * 0.008)                        # the theme's hairline rim
    p.setPen(QPen(QColor(*RIM), w))
    p.setBrush(Qt.NoBrush)
    h = w / 2
    p.drawRoundedRect(QRectF(h, h, S - w, S - w), TILE_RADIUS * S - h, TILE_RADIUS * S - h)
    p.end()
    return img


def icon() -> QIcon:
    """The window / taskbar icon at every size, from the cache when it is there."""
    ic = QIcon()
    for s in SIZES:
        f = CACHE / f"appicon_v{ICON_VERSION}_{s}.png"
        pm = QPixmap(str(f)) if f.exists() else QPixmap()
        if pm.isNull():
            img = render(s)
            try:
                CACHE.mkdir(exist_ok=True)
                img.save(str(f))
            except OSError:
                pass                              # a read-only install: draw it every start
            pm = QPixmap.fromImage(img)
        ic.addPixmap(pm)
    return ic


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        print("usage: python -m kinetrace.appicon OUT.png [--size N] [--simple]")
        return 2
    size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 1024
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication.instance() or QGuiApplication([])     # noqa: F841 - QPainter wants one
    ok = render(size, simple=True if "--simple" in argv else None).save(argv[0])
    print(("wrote " if ok else "could not write ") + argv[0])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
