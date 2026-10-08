"""The cameras' ORDER (G172): camera 1 is the reference clock every offset is measured against, and the
order is the order of the cameras in calibrations (a DLTdv / easyWand / Argus file lists its cameras
in order), in 3D and in every export -- it must match the calibration's, so the user sets it, never
the file dialog's sort.

`CameraOrderDialog` lists the cameras with their numbers; a row moves with Move up / Move down (or
Alt+Up / Alt+Down) or by dragging it. Used when videos are added (the project's cameras and the new
ones, the new ones marked) and by CAMERAS -> Camera order... for an open project. File -> Open Folder
of Videos orders its videos in its own table (folderimport.py).

The dialog is data-free: it returns `order`, the entries' keys in the chosen order.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout)

from kinetrace import theme

KEY_ROLE = Qt.UserRole + 1
NOTE_ROLE = Qt.UserRole + 2


class CameraOrderDialog(QDialog):
    """`entries` = [(key, name, note)] in the current order (note: a few words after the name, e.g.
    "new"). `problem(keys in order)` -> a sentence why that order is refused (OK greys out), or None.
    After OK: `order` = the keys in the chosen order."""

    def __init__(self, parent, entries, title: str = "Camera order", intro: str = "", consequence: str = "",
                 problem=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.order: list = [e[0] for e in entries]
        self._problem = problem
        lay = QVBoxLayout(self)
        head = QLabel(intro or (
            "Put the cameras in their order. <b>Camera 1 is the reference</b>: the clock every other camera's "
            "offset is measured against. The order is also the order of the cameras in calibrations, in 3D and in "
            "every export: keep it the same as your calibration's (a DLTdv, easyWand or Argus file lists its "
            "cameras in order)."))
        head.setWordWrap(True)
        lay.addWidget(head)
        row = QHBoxLayout()
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)        # drag a row to its place
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setWordWrap(True)                  # a long file name wraps instead of a sideways scroll bar
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setToolTip("Drag a camera to its place, or select it and use Move up / Move down "
                             "(Alt+Up / Alt+Down)")
        for key, name, note in entries:
            it = QListWidgetItem()
            it.setData(KEY_ROLE, key)
            it.setData(Qt.UserRole, str(name))
            it.setData(NOTE_ROLE, str(note or ""))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
            self.list.addItem(it)
        self.list.model().rowsMoved.connect(lambda *_: self._renumber())
        self.list.currentRowChanged.connect(lambda _r: self._update_buttons())
        row.addWidget(self.list, 1)
        side = QVBoxLayout()
        self.btn_up = QPushButton("Move up")
        self.btn_down = QPushButton("Move down")
        for b, d in ((self.btn_up, -1), (self.btn_down, +1)):
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, dd=d: self.move_current(dd))
            side.addWidget(b)
        side.addStretch(1)
        row.addLayout(side)
        lay.addLayout(row, 1)
        QShortcut(QKeySequence("Alt+Up"), self, activated=lambda: self.move_current(-1))
        QShortcut(QKeySequence("Alt+Down"), self, activated=lambda: self.move_current(+1))
        self.note = QLabel(consequence)
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.note)
        self._consequence = consequence
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.resize(520, 420)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._renumber()

    # ------------------------------------------------------------------ order

    def keys(self) -> list:
        return [self.list.item(r).data(KEY_ROLE) for r in range(self.list.count())]

    def move_current(self, delta: int) -> None:
        """The selected camera one place up (-1) or down (+1)."""
        r = self.list.currentRow()
        to = r + int(delta)
        if r < 0 or not (0 <= to < self.list.count()):
            return
        it = self.list.takeItem(r)
        self.list.insertItem(to, it)
        self.list.setCurrentRow(to)
        self._renumber()

    def _renumber(self) -> None:
        """Every row says the number it will have; camera 1 says it is the reference."""
        for r in range(self.list.count()):
            it = self.list.item(r)
            note = it.data(NOTE_ROLE)
            text = f"{r + 1}   {it.data(Qt.UserRole)}" + ("   (reference: the clock)" if r == 0 else "")
            if note:
                text += f"   · {note}"
            it.setText(text)
        self._update_buttons()

    def _update_buttons(self) -> None:
        r = self.list.currentRow()
        self.btn_up.setEnabled(r > 0)
        self.btn_down.setEnabled(0 <= r < self.list.count() - 1)
        why = self._problem(self.keys()) if self._problem is not None else None
        self.note.setText(why[:1].upper() + why[1:] + "." if why else self._consequence)
        self.note.setStyleSheet(f"color: {theme.AMBER if why else theme.TEXT_DIM};")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not why)

    def _accept(self) -> None:
        if self._problem is not None and self._problem(self.keys()):
            return
        self.order = self.keys()
        self.accept()
