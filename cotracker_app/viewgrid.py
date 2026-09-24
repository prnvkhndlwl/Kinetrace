"""The camera grid: one `VideoCanvas` per view, laid out side by side.

With a single video this is exactly the old single-canvas layout — the grid
adds one widget with no margins around the one canvas, so nothing about a
one-camera project changes.

With several videos every view gets its own canvas showing the frame that
matches the playhead through its offset (`Project.map_frame`), with that
view's own tracked points drawn on it. Only the ACTIVE canvas takes edits;
the others are view-only and a click on one makes it active, which is the
fastest way to switch cameras. Each canvas carries a small caption with the
camera name, its own frame number and its offset, and the active one is
outlined in the accent color.

The grid owns no data: `MainWindow` feeds it frames and points. It does own
the per-view decode runtimes (cache + seek thread), because their lifetime is
exactly the lifetime of a view's canvas.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from cotracker_app import theme
from cotracker_app.canvas import VideoCanvas

CAPTION_H = 18


class _ViewCell(QWidget):
    """One canvas plus its caption, framed when active."""

    clicked = Signal(int)

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.canvas = VideoCanvas()
        self.caption = QLabel("")
        self.caption.setFixedHeight(CAPTION_H)
        # the caption must never dictate the window width (the QLabel pitfall
        # that once forced a 3376 px minimum) — ignore its width entirely
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.caption)
        lay.addWidget(self.canvas, 1)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        edge = theme.ACCENT if active else theme.HAIRLINE
        self.setStyleSheet(f"_ViewCell {{ border: 1px solid {edge}; }}")
        self.caption.setStyleSheet(
            f"color: {theme.TEXT if active else theme.TEXT_DIM}; "
            f"background: {theme.BG_WINDOW}; padding: 0 6px;")

    def set_caption(self, text: str) -> None:
        if text != self.caption.text():
            self.caption.setText(text)


class ViewGrid(QWidget):
    """A grid of camera views. `set_count(1)` is the plain single-canvas UI."""

    view_activated = Signal(int)     # a click landed on a non-active view

    def __init__(self):
        super().__init__()
        self._cells: list[_ViewCell] = []
        self._active = 0
        self._solo = False           # show only the active view even with several
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._lay = lay
        self.set_count(1)

    # ----------------------------------------------------------------- layout

    @property
    def canvases(self) -> list[VideoCanvas]:
        return [c.canvas for c in self._cells]

    @property
    def active_canvas(self) -> VideoCanvas:
        return self._cells[self._active].canvas

    def canvas(self, i: int) -> VideoCanvas | None:
        return self._cells[i].canvas if 0 <= i < len(self._cells) else None

    def set_count(self, n: int) -> list[VideoCanvas]:
        """Grow or shrink to `n` views; returns the canvases that are NEW, so
        the caller can wire their signals exactly once."""
        n = max(1, int(n))
        new: list[VideoCanvas] = []
        while len(self._cells) < n:
            cell = _ViewCell(len(self._cells))
            cell.canvas.view_clicked.connect(
                lambda _=None, k=cell.index: self._on_cell_clicked(k))
            self._cells.append(cell)
            new.append(cell.canvas)
        while len(self._cells) > n:
            cell = self._cells.pop()
            self._lay.removeWidget(cell)
            cell.canvas.clear_video()
            cell.setParent(None)
            cell.deleteLater()
        self._active = min(self._active, n - 1)
        self._relayout()
        return new

    def set_solo(self, solo: bool) -> None:
        """Solo hides the companion views (and, via `visible_indices`, stops the
        app decoding them) without forgetting them."""
        if solo != self._solo:
            self._solo = bool(solo)
            self._relayout()

    @property
    def solo(self) -> bool:
        return self._solo

    def visible_indices(self) -> list[int]:
        """Views actually on screen — the app only decodes these."""
        if self._solo:
            return [self._active]
        return list(range(len(self._cells)))

    def _relayout(self) -> None:
        for cell in self._cells:
            self._lay.removeWidget(cell)
            cell.setVisible(False)
        shown = self.visible_indices()
        # near-square grid, wider than tall (video is landscape)
        cols = max(1, math.ceil(math.sqrt(len(shown))))
        rows = max(1, math.ceil(len(shown) / cols))
        if rows > cols:
            cols, rows = rows, cols
        for k, i in enumerate(shown):
            cell = self._cells[i]
            self._lay.addWidget(cell, k // cols, k % cols)
            cell.setVisible(True)
        for c in range(self._lay.columnCount()):
            self._lay.setColumnStretch(c, 1 if c < cols else 0)
        for r in range(self._lay.rowCount()):
            self._lay.setRowStretch(r, 1 if r < rows else 0)
        single = len(shown) <= 1
        self._lay.setContentsMargins(0, 0, 0, 0)
        for cell in self._cells:
            cell.caption.setVisible(not single)
            cell.setStyleSheet("" if single else cell.styleSheet())
        if single:                       # a lone view keeps the bare old look
            self._cells[self._active].setStyleSheet("")
        else:
            self.set_active(self._active)

    # ----------------------------------------------------------------- active

    def set_active(self, i: int) -> None:
        if not (0 <= i < len(self._cells)):
            return
        self._active = i
        single = len(self.visible_indices()) <= 1
        for cell in self._cells:
            if single:
                cell.setStyleSheet("")
            else:
                cell.set_active(cell.index == i)
        if self._solo:
            self._relayout()

    def _on_cell_clicked(self, i: int) -> None:
        if i != self._active:
            self.view_activated.emit(i)

    # --------------------------------------------------------------- captions

    def set_caption(self, i: int, text: str) -> None:
        if 0 <= i < len(self._cells):
            self._cells[i].set_caption(text)


def caption_for(name: str, frame: int | None, offset: float, n_frames: int) -> str:
    """Caption text for one view: name, where it is, and how it is aligned.
    Whole-frame offsets print as integers; a measured sub-frame offset keeps
    its decimals so the sync is visible at a glance."""
    off = ""
    if offset:
        off = (f"  ·  offset {int(round(offset)):+d}" if abs(offset - round(offset)) < 1e-9
               else f"  ·  offset {offset:+.3f}")
    if frame is None:
        return f"{name}  ·  no frame at this instant{off}"
    return f"{name}  ·  frame {frame} / {max(n_frames - 1, 0)}{off}"
