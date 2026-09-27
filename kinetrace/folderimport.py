"""File → Open Folder of Videos… (G30, owner 2026-09-26): pick a folder, see every
video in it, tick the ones to import, pick the BASE (reference) camera, and they
become one multi-camera project -- saved at once if asked.

`list_videos` is pure (no Qt), so the scan is testable on its own. The dialog
reads each file's header (picture size, frame rate, frame count) on a
background thread: the GUI thread never opens a cv2.VideoCapture.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from kinetrace import theme
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
           and not p.name.startswith(".") and not p.name.startswith("._")]
    return sorted(out, key=_natural_key)


def default_project_path(folder: str | os.PathLike) -> Path:
    """<folder>/<folder name>.kinetrace"""
    root = Path(folder)
    return root / f"{root.name or 'project'}.kinetrace"


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
                self.got.emit(row, f"{type(e).__name__}: {e}")


class VideoFolderDialog(QDialog):
    """Every video of a folder, a tick per file (import it) and a round button
    per file (it is the BASE camera: camera 1, the reference clock every other
    camera's offset is measured against). OK: `result_paths` (the base first,
    then the others in the listed order) and `result_project` (the .kinetrace
    to save to at once, or None)."""

    COLS = ("Import", "Base", "Video", "Picture", "Frame rate", "Frames", "Size")

    def __init__(self, parent, folder: str):
        super().__init__(parent)
        self.setWindowTitle("Open a folder of videos")
        self.folder = Path(folder)
        self.result_paths: list[str] = []
        self.result_project: str | None = None
        self._paths: list[Path] = []
        self._ticks: list[QCheckBox] = []
        self._probe: _HeaderProbe | None = None
        self._headers: dict[int, object] = {}
        self._base = QButtonGroup(self)
        self._base.setExclusive(True)
        self._base.idToggled.connect(lambda _i, _on: self._update())
        lay = QVBoxLayout(self)
        intro = QLabel(
            f"<b>{self.folder}</b><br>Tick the videos to put in one project (each is a camera of the same "
            "event) and choose the <b>base</b> camera: it is camera 1, the reference clock — every other "
            "camera's offset says how many frames it is from it. The base must be one of the ticked videos.")
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
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(list(self.COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        lay.addWidget(self.table, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.note)
        prow = QHBoxLayout()
        self.chk_save = QCheckBox("Save the project now as")
        self.chk_save.setChecked(True)
        self.chk_save.setToolTip("The project file keeps the videos (by path), which one is the base, and "
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
        self.resize(820, 520)
        self._rescan()

    # ------------------------------------------------------------------ table

    def _rescan(self) -> None:
        self.stop_probe()
        self._paths = list_videos(self.folder, self.chk_sub.isChecked())
        self._headers = {}
        self._ticks = []
        for b in self._base.buttons():
            self._base.removeButton(b)
        self.table.setRowCount(len(self._paths))
        for r, p in enumerate(self._paths):
            tick = QCheckBox()
            tick.setChecked(r < MAX_VIEWS)
            tick.toggled.connect(lambda _on, row=r: self._on_tick(row))
            self._ticks.append(tick)
            self.table.setCellWidget(r, 0, self._centred(tick))
            radio = QRadioButton()
            radio.setToolTip("The base camera: camera 1, the reference clock")
            self._base.addButton(radio, r)
            self.table.setCellWidget(r, 1, self._centred(radio))
            rel = p.relative_to(self.folder) if self.chk_sub.isChecked() else Path(p.name)
            self.table.setItem(r, 2, QTableWidgetItem(str(rel)))
            for c in (3, 4, 5):
                self.table.setItem(r, c, QTableWidgetItem("…"))
            try:
                size = f"{p.stat().st_size / 1024 ** 2:,.0f} MB"
            except OSError:
                size = "?"
            self.table.setItem(r, 6, QTableWidgetItem(size))
        if self._paths:
            self._base.button(0).setChecked(True)
            self._probe = _HeaderProbe(self._paths)
            self._probe.got.connect(self._on_header)
            self._probe.start()
        self._update()

    @staticmethod
    def _centred(w) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch(1)
        h.addWidget(w)
        h.addStretch(1)
        return box

    def _on_header(self, row: int, info) -> None:
        if not (0 <= row < self.table.rowCount()):
            return
        self._headers[row] = info
        if isinstance(info, str):
            self.table.setItem(row, 3, QTableWidgetItem(info))
            self.table.setItem(row, 4, QTableWidgetItem(""))
            self.table.setItem(row, 5, QTableWidgetItem(""))
        else:
            w, h, fps, n = info
            self.table.setItem(row, 3, QTableWidgetItem(f"{w} × {h}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{fps:.3f}".rstrip("0").rstrip(".") + " fps"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{n:,}"))
        self._update()

    def _tick_all(self, on: bool) -> None:
        for r, t in enumerate(self._ticks):
            t.setChecked(on and r < MAX_VIEWS)

    def _on_tick(self, _row: int) -> None:
        # the base must be imported: an unticked base goes to the first ticked row
        b = self._base.checkedId()
        if b < 0 or not self._ticks[b].isChecked():
            first = next((r for r, t in enumerate(self._ticks) if t.isChecked()), None)
            if first is not None:
                self._base.button(first).setChecked(True)
        self._update()

    def ticked(self) -> list[int]:
        return [r for r, t in enumerate(self._ticks) if t.isChecked()]

    def base_row(self) -> int:
        return self._base.checkedId()

    def _update(self) -> None:
        rows = self.ticked()
        base = self.base_row()
        ok_btn = self.buttons.button(QDialogButtonBox.Ok)
        problems = []
        if not self._paths:
            problems.append("No video files in this folder" + ("" if self.chk_sub.isChecked() else
                                                              " (tick Include subfolders to look deeper)") + ".")
        elif not rows:
            problems.append("Tick at least one video.")
        elif len(rows) > MAX_VIEWS:
            problems.append(f"A project holds up to {MAX_VIEWS} cameras: untick {len(rows) - MAX_VIEWS}.")
        elif base not in rows:
            problems.append("Choose the base camera among the ticked videos.")
        bad = [self._paths[r].name for r in rows if isinstance(self._headers.get(r), str)]
        if bad:
            problems.append(f"Cannot be opened: {', '.join(bad)} — untick {'it' if len(bad) == 1 else 'them'}.")
        fps = sorted({round(self._headers[r][2], 3) for r in rows if isinstance(self._headers.get(r), tuple)})
        info = ""
        if len(fps) > 1:
            info = (f"Frame rates differ ({', '.join(f'{f:g}' for f in fps)} fps): that is fine — each camera "
                    "keeps its own rate against the base (×2 for a camera twice as fast).")
        if rows and base in rows and not problems:
            info = (f"{len(rows)} camera(s); base: <b>{self._paths[base].name}</b>. After importing, line them "
                    "up in time with 3D → Sync Cameras (Sound / Motion). " + info)
        self.note.setText(" ".join(problems) if problems else info)
        self.note.setStyleSheet(f"color: {theme.AMBER if problems else theme.TEXT_DIM};")
        ok_btn.setEnabled(not problems)
        ok_btn.setText(f"Import {len(rows)} video{'s' if len(rows) != 1 else ''}" if rows else "Import")

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the project as", self.edit_proj.text(),
                                              "Kinetrace project (*.kinetrace)")
        if path:
            self.edit_proj.setText(path if path.endswith(".kinetrace") else path + ".kinetrace")

    def _accept(self) -> None:
        rows, base = self.ticked(), self.base_row()
        if not rows or base not in rows or len(rows) > MAX_VIEWS:
            return
        order = [base] + [r for r in rows if r != base]
        self.result_paths = [str(self._paths[r]) for r in order]
        proj = self.edit_proj.text().strip()
        self.result_project = (proj if proj.endswith(".kinetrace") else proj + ".kinetrace") \
            if (self.chk_save.isChecked() and proj) else None
        self.accept()

    def stop_probe(self) -> None:
        """Stop the header thread (a QThread must never be destroyed running)."""
        if self._probe is not None:
            self._probe.stop()
            self._probe.wait(5000)
            self._probe = None

    def done(self, r: int) -> None:        # noqa: D102 - the header thread stops with the dialog
        self.stop_probe()
        super().done(r)
