"""The LAYERS panel (G153-G157, owner 2026-10-03: "if the user tracks a segment, lets say a squirrel,
then that can be a layer and the points tracked can be sublayers inside like adobe illustrator ...
allow the user to drag and drop points from one animal parent to another").

One tree: every ANIMAL (an individual: its optional silhouette, its skeleton, whether its points
are held on it) with its points as children, then SCENE (the points of no animal: wand ends,
reference markers). One selection for both: a plain click selects only that row, Ctrl / Shift add.
An animal row stands for its silhouette AND all its points (Track runs them; Delete removes the
animal). Dragging point rows onto an animal, one of its points or Scene asks the app to move them
(`move_requested`): the app renames them "<animal> <part>" in every camera, as one undo step, and
rebuilds the tree.

The widget only shows and reports: it never changes the session (the app does, then calls
`rebuild`). Rows carry ("animal", k) / ("scene",) / ("point", pid) in Qt.UserRole and the point's
full name in UserRole + 2, so a rebuild keeps the selection and the expanded animals BY NAME.
"""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from kinetrace import theme

ROLE_KIND = Qt.UserRole              # ("animal", k) | ("scene",) | ("point", pid)
ROLE_HAS = Qt.UserRole + 1           # point: has data in this camera (the dimmed style)
ROLE_NAME = Qt.UserRole + 2          # point: its full name / animal: its name (selection by name)
SCENE_LABEL = "Scene"
ICON_W, ICON_H = 34, 14


def point_icon(meta, tag: str = "") -> QIcon:
    """A point's swatch: a filled square (tracked by appearance), a dotted outline (derived from
    the silhouette) or a ring (a ball marker), and its tracker's tag (AT / CT / MS, G62)."""
    pm = QPixmap(ICON_W, ICON_H)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    if meta.derived:
        p.setPen(QPen(QColor(*meta.color), 1.5, Qt.DotLine))
        p.drawRect(2, 2, 10, 10)
    elif meta.is_ball:
        p.setPen(QPen(QColor(*meta.color), 2.0))
        p.drawEllipse(2, 2, 10, 10)
    else:
        p.fillRect(1, 1, 12, 12, QColor(*meta.color))
    if tag:
        f = p.font()
        f.setPixelSize(9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(16, 0, 18, 14, Qt.AlignVCenter | Qt.AlignLeft, tag)
    p.end()
    return QIcon(pm)


def animal_icon(color, has_silhouette: bool, hold: bool) -> QIcon:
    """An animal's swatch: a filled blob when it has a silhouette, an outline when it has none
    yet; a small dot inside = its points are held on the silhouette (G156)."""
    pm = QPixmap(ICON_W, ICON_H)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(*color)
    p.setPen(QPen(c, 1.5))
    if has_silhouette:
        p.setBrush(QColor(c.red(), c.green(), c.blue(), 150))
    p.drawEllipse(1, 2, 14, 10)
    if hold and has_silhouette:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.TEXT))
        p.drawEllipse(6, 5, 4, 4)
    p.end()
    return QIcon(pm)


def scene_icon() -> QIcon:
    pm = QPixmap(ICON_W, ICON_H)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(theme.TEXT_DIM), 1.2, Qt.DashLine))
    p.drawRect(1, 2, 13, 10)
    p.end()
    return QIcon(pm)


class LayersPanel(QTreeWidget):
    """The animals with their points, then Scene (see the module docstring)."""

    move_requested = Signal(object, int)     # (point ids, target animal index; -1 = Scene)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setIconSize(QSize(ICON_W, ICON_H))
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(False)      # a double-click renames
        self.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self._points: dict[int, QTreeWidgetItem] = {}
        self._animals: dict[int, QTreeWidgetItem] = {}
        self._scene: QTreeWidgetItem | None = None
        self._collapsed: set[str] = set()        # animals the user folded, by name ("" = Scene)
        self.itemCollapsed.connect(lambda it: self._collapsed.add(self._fold_key(it)))
        self.itemExpanded.connect(lambda it: self._collapsed.discard(self._fold_key(it)))

    # ------------------------------------------------------------------ build

    def _fold_key(self, it) -> str:
        kind = it.data(0, ROLE_KIND)
        return "" if kind and kind[0] == "scene" else str(it.data(0, ROLE_NAME))

    def rebuild(self, s, tag_of=None, style_point=None, keep=None, current=None) -> None:
        """The tree for session `s` (None = empty). `tag_of(pid)` = its tracker tag; `style_point(item,
        pid)` sets the point row's tooltip / dimming; `keep` = (point names, animal names) to select
        (None = keep what is selected now, by name); `current` = the point id to make current."""
        if keep is None:
            keep = self.selected_names()
        pnames, anames = keep
        self.blockSignals(True)
        self.clear()
        self._points, self._animals, self._scene = {}, {}, None
        if s is not None:
            per_animal, scene = s.layer_groups()
            for k, pids in [*enumerate(per_animal), (None, scene)]:
                if k is None:
                    if not s.n_segments:
                        top = self.invisibleRootItem()         # no animal at all: a plain list
                    else:
                        top = QTreeWidgetItem([SCENE_LABEL])
                        top.setData(0, ROLE_KIND, ("scene",))
                        top.setData(0, ROLE_NAME, "")
                        top.setIcon(0, scene_icon())
                        top.setFlags((top.flags() | Qt.ItemIsDropEnabled | Qt.ItemIsUserCheckable)
                                     & ~Qt.ItemIsDragEnabled & ~Qt.ItemIsEditable)
                        shown = [s.points[q].display for q in pids]
                        top.setCheckState(0, Qt.Checked if (all(shown) if shown else True) else Qt.Unchecked)
                        top.setToolTip(0, "Points that belong to no animal (wand ends, reference markers): never "
                                          "held on a silhouette. Drag a point here to take it out of its animal.")
                        self.addTopLevelItem(top)
                        self._scene = top
                else:
                    a = s.segments[k]
                    top = QTreeWidgetItem([a.name])
                    top.setData(0, ROLE_KIND, ("animal", k))
                    top.setData(0, ROLE_NAME, a.name)
                    n_masked = s.seg_masks[k].n_masked()
                    has_sil = bool(a.n_prompts() or n_masked)      # = s.has_silhouette(k), the count reused
                    top.setIcon(0, animal_icon(a.color, has_sil, a.hold))
                    top.setFlags((top.flags() | Qt.ItemIsDropEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
                                 & ~Qt.ItemIsDragEnabled)
                    top.setCheckState(0, Qt.Checked if a.shown else Qt.Unchecked)
                    top.setToolTip(0, f"{a.name}: {len(pids)} point(s); silhouette on {n_masked:,} of "
                                      f"{s.n_frames:,} frames from {a.n_prompts()} click(s)/box(es)"
                                   + ("; its points are kept on the silhouette" if a.hold else "")
                                   + ".\nSelect it (with S on, a click on the video outlines it). Its checkbox "
                                     "shows / hides its silhouette; double-click renames it; right-click for "
                                     "its skeleton, silhouettes and more. Drag points onto it to make them its.")
                    if not has_sil and not pids:
                        top.setForeground(0, QColor(theme.TEXT_DIM))
                    self.addTopLevelItem(top)
                    self._animals[k] = top
                for q in pids:
                    meta = s.points[q]
                    it = QTreeWidgetItem([s.part_name(q)])
                    it.setData(0, ROLE_KIND, ("point", q))
                    it.setData(0, ROLE_NAME, meta.name)
                    it.setIcon(0, point_icon(meta, tag_of(q) if tag_of is not None else ""))
                    it.setFlags((it.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
                                & ~Qt.ItemIsDropEnabled)
                    it.setCheckState(0, Qt.Checked if meta.display else Qt.Unchecked)
                    if style_point is not None:
                        style_point(it, q)
                    if top is self.invisibleRootItem():
                        self.addTopLevelItem(it)
                    else:
                        top.addChild(it)
                    self._points[q] = it
            for i in range(self.topLevelItemCount()):
                it = self.topLevelItem(i)
                if it.childCount():
                    it.setExpanded(self._fold_key(it) not in self._collapsed)
            for q, it in self._points.items():
                if it.data(0, ROLE_NAME) in pnames:
                    it.setSelected(True)
            for k, it in self._animals.items():
                if it.data(0, ROLE_NAME) in anames:
                    it.setSelected(True)
            if current is not None and current in self._points:
                self.setCurrentItem(self._points[current], 0, QItemSelectionModel.NoUpdate)
        self.blockSignals(False)

    # ------------------------------------------------------------- look-ups

    def point_item(self, pid: int) -> QTreeWidgetItem | None:
        return self._points.get(pid)

    def animal_item(self, k: int) -> QTreeWidgetItem | None:
        return self._animals.get(k)

    def scene_item(self) -> QTreeWidgetItem | None:
        return self._scene

    def n_point_rows(self) -> int:
        return len(self._points)

    @staticmethod
    def kind_of(item) -> tuple:
        return tuple(item.data(0, ROLE_KIND) or ()) if item is not None else ()

    def selected_pids(self) -> list[int]:
        """The point rows selected (not the points of a selected animal row)."""
        return sorted(q for q, it in self._points.items() if it.isSelected())

    def selected_animals(self) -> list[int]:
        return sorted(k for k, it in self._animals.items() if it.isSelected())

    def selected_names(self) -> tuple[set, set]:
        """(the full names of the selected points, the names of the selected animals)."""
        return ({it.data(0, ROLE_NAME) for it in self._points.values() if it.isSelected()},
                {it.data(0, ROLE_NAME) for it in self._animals.values() if it.isSelected()})

    def current_pid(self) -> int | None:
        kind = self.kind_of(self.currentItem())
        return kind[1] if kind and kind[0] == "point" else None

    # ------------------------------------------------------------ selection

    def select_only(self, pids=(), animals=(), current: int | None = None) -> None:
        """THE selection becomes these rows (signals blocked: the caller refreshes what follows)."""
        self.blockSignals(True)
        self.clearSelection()
        for q in pids:
            if q in self._points:
                self._points[q].setSelected(True)
        for k in animals:
            if k in self._animals:
                self._animals[k].setSelected(True)
        if current is not None and current in self._points:
            self.setCurrentItem(self._points[current], 0, QItemSelectionModel.NoUpdate)
        self.blockSignals(False)

    def set_selected(self, pids=(), animals=(), on: bool = True) -> None:
        for q in pids:
            if q in self._points:
                self._points[q].setSelected(on)
        for k in animals:
            if k in self._animals:
                self._animals[k].setSelected(on)

    # ---------------------------------------------------------- drag & drop

    def _drop_target(self, item) -> int | None:
        """The animal a drop on `item` gives the points to (-1 = Scene, None = not a target)."""
        kind = self.kind_of(item)
        if not kind:
            return None
        if kind[0] == "animal":
            return kind[1]
        if kind[0] == "scene":
            return -1
        parent = item.parent()
        pk = self.kind_of(parent)
        if pk and pk[0] == "animal":
            return pk[1]
        if pk and pk[0] == "scene":
            return -1
        return None

    def _ours(self, ev) -> bool:
        """A drag of this tree's own rows (another program's drag carries other data)."""
        md = ev.mimeData()
        return md is not None and any(md.hasFormat(f) for f in self.mimeTypes())

    def dragEnterEvent(self, ev):
        if self._ours(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        target = self._drop_target(self.itemAt(ev.position().toPoint()))
        if target is None or not self.selected_pids() or not self._ours(ev):
            ev.ignore()
            return
        ev.setDropAction(Qt.MoveAction)
        ev.accept()

    def dropEvent(self, ev):
        """Never let Qt move the rows itself: the app moves the points (every camera, one undo step)
        and rebuilds the tree from the session."""
        target = self._drop_target(self.itemAt(ev.position().toPoint()))
        pids = self.selected_pids()
        ev.setDropAction(Qt.IgnoreAction)
        ev.accept()
        if target is not None and pids and self._ours(ev):
            self.move_requested.emit(pids, int(target))
