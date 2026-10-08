"""File → Open Folder of Videos… (G30): pick a folder, see every
video in it, tick the ones to import, put them in CAMERA ORDER (G171: camera 1,
the reference clock, is the first ticked video; the order is the order of the
cameras in calibrations, 3D and exports), and they become one multi-camera
project -- saved at once if asked.

`list_videos` is pure (no Qt), so the scan is testable on its own. The dialog
reads each file's header (picture size, frame rate, frame count) on a
background thread: the GUI thread never opens a cv2.VideoCapture.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from kinetrace import theme
from kinetrace.errors import plain_error
from kinetrace.project import MAX_VIEWS

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg")


def _natural_key(p: Path):
    """cam2 before cam10: digits compare as numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def list_videos(folder: str | os.PathLike, recursive: bool = False) -> list[Path]:
    """The video files in `folder` (by extension, any case), in natural order;
    with `recursive` its subfolders too. Hidden files and macOS '._' shadows are
    left out."""
    root = Path(folder)
    if not root.is_dir():
        return []
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS
           and not p.name.startswith(".")]      # (also macOS "._" shadows)
    return sorted(out, key=_natural_key)


def default_project_path(folder: str | os.PathLike) -> Path:
    """<folder>/<folder name>.kinetrace"""
    root = Path(folder)
    return root / f"{root.name or 'project'}.kinetrace"


# header threads that were still reading when their dialog went away: kept alive
# until `finished` (like app._ORPHANS); the app's closeEvent can `wait_orphans` (I199)
_ORPHANS: list = []


def wait_orphans(ms: int = 30000) -> None:
    """Wait for header threads that outlived their dialog (called when the app closes)."""
    for th in list(_ORPHANS):
        try:
            if th.isRunning():
                th.wait(ms)
        except RuntimeError:
            pass


class _HeaderProbe(QThread):
    """Reads each file's header off the GUI thread: (row, w, h, fps, frames) or (row, error)."""

    got = Signal(int, object)

    def __init__(self, paths: list[Path]):
        super().__init__()
        self._paths = list(paths)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import cv2

        from kinetrace.video_source import open_capture
        for row, p in enumerate(self._paths):
            if self._stop:
                return
            try:
                cap = open_capture(str(p))
                if not cap.isOpened():
                    self.got.emit(row, "could not be opened")
                    continue
                info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        float(cap.get(cv2.CAP_PROP_FPS) or 0.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
                cap.release()
                self.got.emit(row, info)
            except Exception as e:      # noqa: BLE001 -- one odd file must not stop the list
                self.got.emit(row, plain_error(e, "could not be read", short=True))      # (G54)


class VideoFolderDialog(QDialog):
    """Every video of a folder, a tick per file (import it), in CAMERA ORDER (G171): Move up / Move down
    (Alt+Up / Alt+Down) put the selected video earlier / later, Sort by name restores the folder's
    order; the first ticked video is camera 1, the reference clock every other camera's offset is
    measured against. OK: `result_paths` (the ticked videos in that order) and `result_project` (the
    .kinetrace to save to at once, or None)."""

    COLS = ("Import", "Camera", "Video", "Picture", "Frame rate", "Frames", "Size")

    def __init__(self, parent, folder: str):
        super().__init__(parent)
        self.setWindowTitle("Open a folder of videos")
        self.folder = Path(folder)
        self.result_paths: list[str] = []
        self.result_project: str | None = None
        self._paths: list[Path] = []          # the scan, in the folder's (natural) order; never reordered
        self._rows: list[int] = []            # the table's rows = indices into _paths, in CAMERA order
        self._ticked: set[int] = set()        # indices into _paths
        self._ticks: list[QCheckBox] = []     # the tick of each table row
        self._probe: _HeaderProbe | None = None
        self._headers: dict[int, object] = {}     # index into _paths -> header (or a sentence)
        lay = QVBoxLayout(self)
        intro = QLabel(
            f"<b>{self.folder}</b><br>Tick the videos to put in one project (each is a camera of the same "
            "event) and put them in <b>camera order</b> with Move up / Move down: the first ticked video is "
            "<b>camera 1, the reference clock</b> — every other camera's offset says how many frames it is from "
            "it. The order is also the order of the cameras in calibrations, in 3D and in every export: keep it the "
            "same as your calibration's.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        self.chk_sub = QCheckBox("Include subfolders")
        self.chk_sub.toggled.connect(self._rescan)
        row.addWidget(self.chk_sub)
        row.addStretch(1)
        for text, on in (("Tick all", True), ("Tick none", False)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, v=on: self._tick_all(v))
            row.addWidget(b)
        lay.addLayout(row)
        mid = QHBoxLayout()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(list(self.COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(lambda *_: self._update_moves())
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        mid.addWidget(self.table, 1)
        side = QVBoxLayout()
        self.btn_up = QPushButton("Move up")
        self.btn_down = QPushButton("Move down")
        self.btn_sort = QPushButton("Sort by name")
        self.btn_up.setToolTip("The selected video one camera earlier (Alt+Up)")
        self.btn_down.setToolTip("The selected video one camera later (Alt+Down)")
        self.btn_sort.setToolTip("Back to the folder's order (cam2 before cam10)")
        for b, d in ((self.btn_up, -1), (self.btn_down, +1)):
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, dd=d: self.move_current(dd))
            side.addWidget(b)
        self.btn_sort.setAutoDefault(False)
        self.btn_sort.clicked.connect(self.sort_by_name)
        side.addWidget(self.btn_sort)
        side.addStretch(1)
        mid.addLayout(side)
        lay.addLayout(mid, 1)
        QShortcut(QKeySequence("Alt+Up"), self, activated=lambda: self.move_current(-1))
        QShortcut(QKeySequence("Alt+Down"), self, activated=lambda: self.move_current(+1))
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.note)
        prow = QHBoxLayout()
        self.chk_save = QCheckBox("Save the project now as")
        self.chk_save.setChecked(True)
        self.chk_save.setToolTip("The project file keeps the videos (by path), their order, and "
                                 "later everything you track. Untick to decide later (Ctrl+S).")
        self.edit_proj = QLineEdit(str(default_project_path(self.folder)))
        self.chk_save.toggled.connect(self.edit_proj.setEnabled)
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        prow.addWidget(self.chk_save)
        prow.addWidget(self.edit_proj, 1)
        prow.addWidget(browse)
        lay.addLayout(prow)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.resize(860, 540)
        self._rescan()

    # ------------------------------------------------------------------ table

    def _rescan(self) -> None:
        self.stop_probe()
        self._paths = list_videos(self.folder, self.chk_sub.isChecked())
        self._headers = {}
        self._rows = list(range(len(self._paths)))
        self._ticked = {k for k in self._rows if k < MAX_VIEWS}
        self._fill()
        if self._paths:
            self.table.selectRow(0)
            self._probe = _HeaderProbe(self._paths)
            self._probe.got.connect(self._on_header)
            self._probe.start()
        self._update()

    def _fill(self) -> None:
        """The table from the state: one row per video, in camera order."""
        self.table.setRowCount(len(self._rows))
        self._ticks = []
        for r, k in enumerate(self._rows):
            p = self._paths[k]
            tick = QCheckBox()
            tick.setChecked(k in self._ticked)
            tick.toggled.connect(lambda on, kk=k: self._on_tick(kk, on))
            self._ticks.append(tick)
            self.table.setCellWidget(r, 0, self._centred(tick))
            self.table.setItem(r, 1, QTableWidgetItem(""))
            rel = p.relative_to(self.folder) if self.chk_sub.isChecked() else Path(p.name)
            self.table.setItem(r, 2, QTableWidgetItem(str(rel)))
            self._show_header(r, k)
            try:
                size = f"{p.stat().st_size / 1024 ** 2:,.0f} MB"
            except OSError:
                size = "?"
            self.table.setItem(r, 6, QTableWidgetItem(size))
        self._number()

    def _number(self) -> None:
        """The Camera column: each ticked video's number in the order; camera 1 is the reference."""
        n = 0
        for r, k in enumerate(self._rows):
            text = ""
            if k in self._ticked:
                n += 1
                text = "1 · reference" if n == 1 else str(n)
            it = self.table.item(r, 1)
            if it is not None and it.text() != text:
                it.setText(text)

    def _show_header(self, r: int, k: int) -> None:
        info = self._headers.get(k)
        if info is None:
            for c in (3, 4, 5):
                self.table.setItem(r, c, QTableWidgetItem("…"))
        elif isinstance(info, str):
            self.table.setItem(r, 3, QTableWidgetItem(info))
            self.table.setItem(r, 4, QTableWidgetItem(""))
            self.table.setItem(r, 5, QTableWidgetItem(""))
        else:
            w, h, fps, n = info
            self.table.setItem(r, 3, QTableWidgetItem(f"{w} × {h}"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{fps:.3f}".rstrip("0").rstrip(".") + " fps"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{n:,}"))

    @staticmethod
    def _centred(w) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch(1)
        h.addWidget(w)
        h.addStretch(1)
        return box

    def _on_header(self, k: int, info) -> None:
        """A header read off the GUI thread: `k` = the file's index in the scan (its row may have moved)."""
        if not (0 <= k < len(self._paths)):
            return
        self._headers[k] = info
        if k in self._rows:
            self._show_header(self._rows.index(k), k)
        self._update()

    def _tick_all(self, on: bool) -> None:
        for r, t in enumerate(self._ticks):
            t.setChecked(on and r < MAX_VIEWS)

    def _on_tick(self, k: int, on: bool) -> None:
        (self._ticked.add if on else self._ticked.discard)(k)
        self._number()
        self._update()

    def move_current(self, delta: int) -> None:
        """The selected video one place earlier (-1) or later (+1) in the camera order (G171)."""
        r = self.table.currentRow()
        to = r + int(delta)
        if r < 0 or not (0 <= to < len(self._rows)):
            return
        self._rows[r], self._rows[to] = self._rows[to], self._rows[r]
        self._fill()
        self.table.selectRow(to)
        self._update()

    def sort_by_name(self) -> None:
        """Back to the folder's (natural) order."""
        r = self.table.currentRow()
        k = self._rows[r] if 0 <= r < len(self._rows) else None
        self._rows = list(range(len(self._paths)))
        self._fill()
        if k is not None:
            self.table.selectRow(self._rows.index(k))
        self._update()

    def ticked(self) -> list[int]:
        """The ticked videos (indices into the scan) in camera order: the first is camera 1."""
        return [k for k in self._rows if k in self._ticked]

    def order_names(self) -> list[str]:
        """The ticked videos' file names in camera order."""
        return [self._paths[k].name for k in self.ticked()]

    def _update_moves(self) -> None:
        r = self.table.currentRow()
        self.btn_up.setEnabled(r > 0)
        self.btn_down.setEnabled(0 <= r < len(self._rows) - 1)
        self.btn_sort.setEnabled(self._rows != list(range(len(self._paths))))

    def _update(self) -> None:
        rows = self.ticked()
        ok_btn = self.buttons.button(QDialogButtonBox.Ok)
        problems = []
        if not self._paths:
            problems.append("No video files in this folder" + ("" if self.chk_sub.isChecked() else
                                                              " (tick Include subfolders to look deeper)") + ".")
        elif not rows:
            problems.append("Tick at least one video.")
        elif len(rows) > MAX_VIEWS:
            problems.append(f"A project holds up to {MAX_VIEWS} cameras: untick {len(rows) - MAX_VIEWS}.")
        bad = [self._paths[k].name for k in rows if isinstance(self._headers.get(k), str)]
        if bad:
            problems.append(f"Cannot be opened: {', '.join(bad)} — untick {'it' if len(bad) == 1 else 'them'}.")
        fps = sorted({round(self._headers[k][2], 3) for k in rows if isinstance(self._headers.get(k), tuple)})
        info = ""
        if len(fps) > 1:
            info = (f"Frame rates differ ({', '.join(f'{f:g}' for f in fps)} fps): that is fine — each camera "
                    "keeps its own rate against camera 1 (×2 for a camera twice as fast).")
        if rows and not problems:
            info = (f"{len(rows)} camera(s); camera 1 (the reference): <b>{self._paths[rows[0]].name}</b>. After "
                    "importing, line them up in time with 3D → Sync Cameras (Sound / Motion). " + info)
        self.note.setText(" ".join(problems) if problems else info)
        self.note.setStyleSheet(f"color: {theme.AMBER if problems else theme.TEXT_DIM};")
        ok_btn.setEnabled(not problems)
        ok_btn.setText(f"Import {len(rows)} video{'s' if len(rows) != 1 else ''}" if rows else "Import")
        self._update_moves()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the project as", self.edit_proj.text(),
                                              "Kinetrace project (*.kinetrace)")
        if path:
            self.edit_proj.setText(path if path.endswith(".kinetrace") else path + ".kinetrace")

    def _accept(self) -> None:
        rows = self.ticked()
        if not rows or len(rows) > MAX_VIEWS:
            return
        self.result_paths = [str(self._paths[k]) for k in rows]          # camera order (G171)
        proj = self.edit_proj.text().strip()
        self.result_project = (proj if proj.endswith(".kinetrace") else proj + ".kinetrace") \
            if (self.chk_save.isChecked() and proj) else None
        self.accept()

    def stop_probe(self) -> None:
        """Stop the header thread (a QThread must never be destroyed running). It
        is told to stop and cut off from the dialog, and kept referenced in
        `_ORPHANS` until its `finished` fires when a slow header read is still in
        progress: dropping a running QThread aborts the process, and waiting for
        it froze the dialog for seconds (I199)."""
        th, self._probe = self._probe, None
        if th is None:
            return
        th.stop()
        try:
            th.got.disconnect(self._on_header)
        except (RuntimeError, TypeError):
            pass
        if th.isRunning() and th not in _ORPHANS:
            _ORPHANS.append(th)
            th.finished.connect(lambda t=th: _ORPHANS.remove(t) if t in _ORPHANS else None)

    def done(self, r: int) -> None:        # noqa: D102 - the header thread stops with the dialog
        self.stop_probe()
        super().done(r)
