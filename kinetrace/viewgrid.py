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

The grid owns no data: `MainWindow` feeds it frames and points, and keeps
the per-view decode runtimes (cache + seek thread) itself.

Which views are on screen and where (G169, G170) is the grid's own DISPLAY
state: `display_order()` (the tiles' order on screen) and `hidden()` (views the
user switched off with their eye in CAMERAS) hold camera indices; cell `i`
always shows camera `i`. A view is dragged by its title bar onto another view
to take that place, the others shifting (as in a video call). None of this
changes the cameras' numbers: their ORDER is data (the reference, calibrations,
exports) and changes only through Camera order... (`MainWindow`), which then
`remap`s the arrangement so every camera stays where it was on screen.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QCursor, QDrag
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QMenu, QSizePolicy, QVBoxLayout, QWidget

from kinetrace import theme
from kinetrace.canvas import VideoCanvas

CAPTION_H = 18
VIEW_MIME = "application/x-kinetrace-view"      # a view dragged by its title bar: its index (G170)
DRAG_THUMB_W = 180                             # px: the picture that follows the pointer while dragging


class _Caption(QLabel):
    """A view's title bar: its text, and the handle the view is dragged by (G170). A press that moves
    past the drag distance starts the drag (`_ViewCell.start_drag`); a right click asks for the
    view's menu."""

    def __init__(self, cell: "_ViewCell"):
        super().__init__("")
        self._cell = cell
        self._press: QPoint | None = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    def mousePressEvent(self, ev):          # noqa: N802 - Qt name
        if ev.button() == Qt.LeftButton:
            self._press = ev.position().toPoint()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):           # noqa: N802 - Qt name
        if (self._press is not None and ev.buttons() & Qt.LeftButton
                and (ev.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance()):
            self._press = None
            self._cell.start_drag()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):        # noqa: N802 - Qt name
        self._press = None
        super().mouseReleaseEvent(ev)


class _ViewCell(QWidget):
    """One canvas plus its caption, framed when active. It takes another view dropped on it (G170)."""

    clicked = Signal(int)
    dropped = Signal(int, int)       # (the view dragged, this view): the first takes this one's place
    drag_wanted = Signal(int)        # its title bar was dragged
    menu_wanted = Signal(int, QPoint)

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.canvas = VideoCanvas()
        # a dropped view must reach the cell: the canvas takes no drops of its own
        self.canvas.setAcceptDrops(False)
        self.canvas.viewport().setAcceptDrops(False)
        self.caption = _Caption(self)
        self.caption.setFixedHeight(CAPTION_H)
        self.caption.customContextMenuRequested.connect(
            lambda pos: self.menu_wanted.emit(self.index, self.caption.mapToGlobal(pos)))
        # the caption must never dictate the window width (the QLabel pitfall
        # that once forced a 3376 px minimum) — ignore its width entirely
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # a QSS border on a plain QWidget is only painted with WA_StyledBackground,
        # and the children must leave room for it: a 1 px margin while framed (G127)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._lay.addWidget(self.caption)
        self._lay.addWidget(self.canvas, 1)
        self._active = False
        self._framed = False
        self._target = False             # a view is being dragged over this one
        self.set_active(False)

    def set_frame(self, framed: bool) -> None:
        """Framed (several views): a 1 px outline in `set_active`'s colour; a lone
        view is bare, as it always was."""
        m = 1 if framed else 0
        self._lay.setContentsMargins(m, m, m, m)
        if not framed:
            self.setStyleSheet("")

    def set_active(self, active: bool, framed: bool = True) -> None:
        self._active, self._framed = bool(active), bool(framed)
        self.set_frame(framed)
        if framed:
            # a view dragged over this one: its place is taken -- the outline says where it lands
            edge = theme.ACCENT if (active or self._target) else theme.HAIRLINE
            width = 2 if self._target else 1
            self.setStyleSheet(f"_ViewCell {{ border: {width}px solid {edge}; }}")
        self.caption.setStyleSheet(
            f"color: {theme.TEXT if active else theme.TEXT_DIM}; "
            f"background: {theme.BG_WINDOW}; padding: 0 6px;")

    def set_caption(self, text: str) -> None:
        if text != self.caption.text():
            self.caption.setText(text)

    def set_draggable(self, on: bool, tip: str = "") -> None:
        """Several views on screen: the title bar is a handle (G170)."""
        self.caption.setCursor(Qt.OpenHandCursor if on else Qt.ArrowCursor)
        self.caption.setToolTip(tip if on else "")

    # ---------------------------------------------------------- drag and drop (G170)

    def start_drag(self) -> None:
        self.drag_wanted.emit(self.index)

    @staticmethod
    def _source(ev) -> int | None:
        md = ev.mimeData()
        if md is None or not md.hasFormat(VIEW_MIME):
            return None
        try:
            return int(bytes(md.data(VIEW_MIME)).decode("ascii"))
        except ValueError:
            return None

    def _set_target(self, on: bool) -> None:
        if on != self._target:
            self._target = on
            self.set_active(self._active, self._framed)

    def dragEnterEvent(self, ev):           # noqa: N802 - Qt name
        src = self._source(ev)
        if src is not None and src != self.index:
            ev.acceptProposedAction()
            self._set_target(True)
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):            # noqa: N802 - Qt name
        src = self._source(ev)
        if src is not None and src != self.index:
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev):           # noqa: N802 - Qt name
        self._set_target(False)

    def dropEvent(self, ev):                # noqa: N802 - Qt name
        self._set_target(False)
        src = self._source(ev)
        if src is None or src == self.index:
            ev.ignore()
            return
        ev.acceptProposedAction()
        self.dropped.emit(src, self.index)


class ViewGrid(QWidget):
    """A grid of camera views. `set_count(1)` is the plain single-canvas UI."""

    view_activated = Signal(int)     # a click landed on a non-active view
    # (G169, G170) the user changed which views are shown or where (a drag, the title bar's menu):
    # the app re-decodes / redraws and keeps the CAMERAS eyes in step
    arrangement_changed = Signal()

    def __init__(self):
        super().__init__()
        self._cells: list[_ViewCell] = []
        self._active = 0
        self._solo = False           # show only the active view even with several
        self._order: list[int] = []  # (G170) the views' order on screen: camera indices, each once
        self._hidden: set[int] = set()   # (G169) views switched off with their eye
        self.name_of = None          # callable(index) -> camera name, for the title bar's menu
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
            cell.dropped.connect(self._on_dropped)
            cell.drag_wanted.connect(self._start_drag)
            cell.menu_wanted.connect(self._view_menu)
            self._cells.append(cell)
            self._order.append(cell.index)            # a new view goes last on screen, shown
            new.append(cell.canvas)
        while len(self._cells) > n:
            cell = self._cells.pop()
            self._lay.removeWidget(cell)
            cell.canvas.clear_video()
            cell.setParent(None)
            cell.deleteLater()
        self._order = [i for i in self._order if i < n]
        self._hidden = {i for i in self._hidden if i < n}
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
        """Views actually on screen, in their order on screen — the app only decodes these. The
        working view is always among them (its eye cannot hide it)."""
        if self._solo:
            return [self._active]
        return [i for i in self._order if i not in self._hidden or i == self._active]

    # ------------------------------------------------- which views, where (G169, G170)

    def display_order(self) -> list[int]:
        return list(self._order)

    def hidden(self) -> set[int]:
        return set(self._hidden)

    def set_arrangement(self, order=None, hidden=None) -> None:
        """Put the views in `order` (camera indices; missing ones follow in camera order, unknown
        ones are left out) with `hidden` switched off. None keeps that part as it is. Not a user
        action: no `arrangement_changed`."""
        n = len(self._cells)
        if order is not None:
            seen: list[int] = []
            for i in order:
                if isinstance(i, int) and 0 <= i < n and i not in seen:
                    seen.append(i)
            self._order = seen + [i for i in range(n) if i not in seen]
        if hidden is not None:
            self._hidden = {i for i in hidden if isinstance(i, int) and 0 <= i < n}
        self._relayout()

    def set_hidden(self, i: int, hidden: bool, user: bool = True) -> None:
        """View `i`'s eye (G169). `user`: say so (`arrangement_changed`)."""
        if not (0 <= i < len(self._cells)) or (i in self._hidden) == bool(hidden):
            return
        (self._hidden.add if hidden else self._hidden.discard)(i)
        self._relayout()
        if user:
            self.arrangement_changed.emit()

    def move_view(self, src: int, dst: int) -> None:
        """View `src` takes view `dst`'s place on screen; the views between shift by one (G170)."""
        if src == dst or src not in self._order or dst not in self._order:
            return
        order = [i for i in self._order if i != src]
        at = order.index(dst)
        # dragged forward it lands AFTER the view it was dropped on, backward BEFORE it: either way it
        # ends up exactly where that view was
        if self._order.index(src) < self._order.index(dst):
            at += 1
        order.insert(at, src)
        self._order = order
        self._relayout()
        self.arrangement_changed.emit()

    def reset_arrangement(self, user: bool = False) -> None:
        """Every view shown, in camera order."""
        changed = self._order != list(range(len(self._cells))) or bool(self._hidden)
        self._order = list(range(len(self._cells)))
        self._hidden = set()
        self._relayout()
        if user and changed:
            self.arrangement_changed.emit()

    def remap(self, new_of: dict[int, int]) -> None:
        """The cameras were renumbered (`new_of[old] = new`; a camera missing from it is gone): the
        arrangement follows them, so every camera keeps its place on screen and its eye."""
        self._order = [new_of[i] for i in self._order if i in new_of]
        self._hidden = {new_of[i] for i in self._hidden if i in new_of}
        self._active = new_of.get(self._active, 0)

    def drop_view(self, i: int) -> None:
        """Camera `i` is being removed: the ones after it move down one (call before `set_count`)."""
        self.remap({k: (k if k < i else k - 1) for k in range(len(self._cells)) if k != i})

    def _on_dropped(self, src: int, dst: int) -> None:
        self.move_view(src, dst)

    def _start_drag(self, i: int) -> None:
        """A view's title bar was dragged: the view follows the pointer as a small picture (G170)."""
        if len(self.visible_indices()) < 2 or not (0 <= i < len(self._cells)):
            return
        cell = self._cells[i]
        md = QMimeData()
        md.setData(VIEW_MIME, str(i).encode("ascii"))
        drag = QDrag(cell)
        drag.setMimeData(md)
        pm = cell.grab()
        if not pm.isNull() and pm.width() > DRAG_THUMB_W:
            pm = pm.scaledToWidth(DRAG_THUMB_W, Qt.SmoothTransformation)
        if not pm.isNull():
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, min(8, pm.height() // 2)))
        drag.exec(Qt.MoveAction)

    def _name(self, i: int) -> str:
        try:
            return str(self.name_of(i)) if self.name_of is not None else f"view {i + 1}"
        except Exception:                # noqa: BLE001 -- a label only
            return f"view {i + 1}"

    def build_view_menu(self, i: int):
        """The title bar's menu (G169, G170) and its actions, built apart so tests can drive it."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        acts = {}
        acts["hide"] = menu.addAction(f"Hide {self._name(i)}")
        acts["hide"].setEnabled(i != self._active and not self._solo)
        acts["hide"].setToolTip("The working camera is always shown" if i == self._active else
                                "Its picture is not read while hidden; its eye in CAMERAS shows it again")
        acts["show_all"] = menu.addAction("Show every camera")
        acts["show_all"].setEnabled(bool(self._hidden))
        acts["camera_order"] = menu.addAction("Arrange the views in camera order")
        acts["camera_order"].setEnabled(self._order != list(range(len(self._cells))))
        return menu, acts

    def view_menu_action(self, chosen, acts, i: int) -> None:
        if chosen is None:
            return
        if chosen is acts["hide"]:
            self.set_hidden(i, True)
        elif chosen is acts["show_all"]:
            self._hidden = set()
            self._relayout()
            self.arrangement_changed.emit()
        elif chosen is acts["camera_order"]:
            self._order = list(range(len(self._cells)))
            self._relayout()
            self.arrangement_changed.emit()

    def _view_menu(self, i: int, gpos) -> None:
        if len(self._cells) < 2:
            return
        menu, acts = self.build_view_menu(i)
        chosen = menu.exec(gpos if gpos is not None else QCursor.pos())
        menu.deleteLater()               # a shown menu is released (see canvas._context_menu)
        self.view_menu_action(chosen, acts, i)

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
            cell.set_active(cell.index == self._active, framed=not single)   # a lone view: no frame
            cell.set_draggable(not single, "Drag this title onto another view to move the view there (only the "
                                           "screen changes: the cameras keep their numbers).\nRight-click: hide "
                                           "it, show every camera, or arrange the views in camera order.")

    # ----------------------------------------------------------------- active

    def set_active(self, i: int) -> None:
        if not (0 <= i < len(self._cells)):
            return
        before = self.visible_indices()
        self._active = i
        # the working view is always on screen (G169): a hidden one taken up is shown while it is the
        # working one, its eye left as it was -- the app's own switches (an every-camera run builds its
        # cameras' runs one camera at a time) must not undo the user's choice of views
        if self._solo or self.visible_indices() != before:
            self._relayout()
            return
        single = len(self.visible_indices()) <= 1
        for cell in self._cells:
            cell.set_active(cell.index == i, framed=not single)

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
