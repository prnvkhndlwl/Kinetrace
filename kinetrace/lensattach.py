"""3D → Load a Lens Profile for Cameras… (G176): one saved lens file attached to every camera ticked in
a list, so a rig of 5 + 5 or 4 + 4 + 2 identical cameras gets its profiles in one go instead of one
camera at a time. Each row says what the file would do to that camera -- fits, fits turned (a camera
filmed on its side, I269), does not fit and why, or replaces the profile it has -- before anything is
attached. `attach_rows` decides (no Qt); `LensAttachDialog` only shows the rows and reports the ticks;
the app attaches (`app._load_lens_profile`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from kinetrace import lens, theme

COLUMNS = ("Camera", "Recording", "Lens profile now", "With this file")


@dataclass
class AttachRow:
    view: int
    name: str
    recording: str              # what the camera records, to tell the groups of a rig apart
    current: str                # `lens.lens_label` of the profile it has now
    profile: object = None      # the profile it would get (turned for its video), None = cannot
    effect: str = ""            # what the file does to this camera, in a few words
    detail: str = ""            # the whole sentence (the turn, or why it does not fit)
    replaces: bool = False      # it has a DIFFERENT profile now
    already: bool = False       # it has this very profile now

    @property
    def can_attach(self) -> bool:
        return self.profile is not None and not self.already


def attach_rows(project, path: str, recordings: list[str]) -> list[AttachRow]:
    """One row per camera of `project`: what lens file `path` would do to it. A file with one line per
    camera (Argus) gives each camera its own line (`lens.read_lens_for`). Raises what the reader raises
    for a file that cannot be read at all."""
    from kinetrace.calibwizard import lens_for_camera
    rows = []
    for v in range(project.n_views):
        name = project.name(v)
        cur = project.lenses[v] if v < len(project.lenses) else None
        row = AttachRow(v, name, recordings[v] if v < len(recordings) else "", lens.lens_label(cur))
        prof, which = lens.read_lens_for(path, v, name)
        if prof is None:
            row.effect, row.detail = "no line for this camera in the file", which
            rows.append(row)
            continue
        fitted, said = lens_for_camera(prof, project.sessions[v], name)
        if fitted is None:
            s = project.sessions[v]
            swapped = (int(prof.width), int(prof.height)) == (int(s.height), int(s.width))
            row.effect = ("does not fit: the same pictures turned, but which way is not known" if swapped else
                          f"does not fit: measured on {int(prof.width)} x {int(prof.height)} pictures")
            row.detail = said
            rows.append(row)
            continue
        row.profile, row.detail = fitted, said
        row.already = cur is not None and lens.same_profile(cur, fitted)
        row.replaces = cur is not None and not row.already
        turned = fitted is not prof
        row.effect = ("already has this profile" if row.already else
                      ("fits, turned to the camera's picture" if turned else "fits")
                      + (": replaces its profile" if row.replaces else ""))
        rows.append(row)
    return rows


class LensAttachDialog(QDialog):
    """The cameras and what lens file `path` would do to each (`rows`, from `attach_rows`); the ticked
    ones are `chosen()` after Accept. The working camera `active` starts ticked when the file fits it
    and it has no other profile; "Tick every camera it fits" ticks the rest."""

    def __init__(self, parent, path: str, rows: list[AttachRow], active: int):
        super().__init__(parent)
        self.setWindowTitle("Load a lens profile for cameras")
        self.rows = rows
        lay = QVBoxLayout(self)
        intro = QLabel(f"Attach <b>{Path(path).name}</b> to the cameras ticked below. Tick only cameras of the "
                       "same model, lens, zoom and recording mode as the one it was measured on: the same picture "
                       "size does not prove the same lens. Hover a row for the whole story.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(len(rows), len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)     # what the file does: the column read
        self.table.setWordWrap(False)
        for r, row in enumerate(rows):
            for c, text in enumerate((row.name, row.recording, row.current, row.effect)):
                it = QTableWidgetItem(text)
                it.setToolTip(row.detail or row.effect)
                if c == 0:
                    if row.can_attach:
                        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                        # replacing a profile is never pre-ticked: it takes the user's own tick (G148)
                        it.setCheckState(Qt.Checked if row.view == active and not row.replaces else Qt.Unchecked)
                    else:
                        it.setFlags(Qt.ItemIsEnabled)
                if not row.can_attach:
                    it.setForeground(QColor(theme.TEXT_DIM))
                elif c == 3 and row.replaces:
                    it.setForeground(QColor(theme.AMBER))
                self.table.setItem(r, c, it)
        # the themed tick box is wider than the size Qt measures for it: room for it, or names are cut
        self.table.resizeColumnToContents(0)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, self.table.columnWidth(0) + 24)
        self.table.itemChanged.connect(lambda *_: self._update())
        lay.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.btn_all = QPushButton("Tick every camera it fits")
        self.btn_all.setToolTip("Every camera this file fits (its picture size, or the same pictures turned). "
                                "Untick the cameras of another model, lens or mode.")
        self.btn_all.clicked.connect(lambda: self._tick_all(True))
        self.btn_none = QPushButton("Tick none")
        self.btn_none.clicked.connect(lambda: self._tick_all(False))
        for b in (self.btn_all, self.btn_none):
            b.setFocusPolicy(Qt.NoFocus)
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.box.accepted.connect(self.accept)
        self.box.rejected.connect(self.reject)
        lay.addWidget(self.box)
        self.resize(1100, 230 + 30 * min(len(rows), 16))      # every row of a 16-camera rig without scrolling
        self._update()

    def _tick_all(self, on: bool) -> None:
        for r, row in enumerate(self.rows):
            if row.can_attach:
                self.table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def chosen(self) -> list[int]:
        """The cameras ticked (project order)."""
        return [row.view for r, row in enumerate(self.rows)
                if row.can_attach and self.table.item(r, 0).checkState() == Qt.Checked]

    def _update(self) -> None:
        views = set(self.chosen())
        n = len(views)
        repl = [row.name for row in self.rows if row.view in views and row.replaces]
        ok = self.box.button(QDialogButtonBox.Ok)
        ok.setText(f"Attach to {n} camera{'s' if n != 1 else ''}" if n else "Attach")
        ok.setEnabled(bool(n))
        fits = sum(1 for row in self.rows if row.can_attach)
        self.summary.setText(
            (f"Replaces the lens profile of {', '.join(repl)}. " if repl else "")
            + (f"The file fits {fits} of the {len(self.rows)} cameras." if fits else
               "The file fits none of these cameras: hover a row to see why."))
