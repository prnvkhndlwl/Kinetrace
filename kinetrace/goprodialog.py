"""3D → GoPro Cameras… (G145): what each GoPro camera of the project says about itself -- the
recording settings, its tilt from the gravity sensor, dropped frames, when it moved -- and a button
that gives GoPro's own lens model to the cameras that have no lens profile yet. A camera whose video
is not GoPro footage is listed as such and nothing is read from it (owner 2026-10-03: GoPro only).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from kinetrace import gpmf, lens, theme

COLUMNS = ("Camera", "Footage", "Stabilisation", "Shutter / ISO", "Points (tilt, roll)", "Dropped frames",
           "Moved", "Lens profile")


lens_text = lens.lens_label          # the one lens-profile label (app.py uses it too)


class GoProDialog(QDialog):
    """`names` = camera names, `infos` = one gpmf.GoProInfo or None per camera, `lenses` = a callable
    giving the project's lens profile per camera (read again after the button attached some). `on_use_lenses(views)` attaches GoPro's lens to those cameras and returns
    a sentence; `on_goto(view, frame)` shows that camera at that frame."""

    def __init__(self, parent, names, infos, lenses, on_use_lenses=None, on_goto=None):
        super().__init__(parent)
        self.setWindowTitle("GoPro cameras")
        self.names, self.infos, self._lenses = list(names), list(infos), lenses
        self.lenses = list(lenses())
        self.on_use_lenses, self.on_goto = on_use_lenses, on_goto
        lay = QVBoxLayout(self)
        intro = QLabel("What each GoPro video says about itself (read from the file; only GoPro footage is read). "
                       "Click a <b>Moved</b> cell to go to the camera and frame where it moved.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(len(self.names), len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.cellClicked.connect(self._clicked)
        lay.addWidget(self.table, 1)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.RichText)
        lay.addWidget(self.notes)
        self.btn_lens = QPushButton("")
        self.btn_lens.clicked.connect(self._use_lenses)
        lay.addWidget(self.btn_lens, 0, Qt.AlignLeft)
        self.said = QLabel("")
        self.said.setWordWrap(True)
        lay.addWidget(self.said)
        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)
        lay.addWidget(box)
        self._fill()
        self.resize(1100, 260 + 30 * len(self.names))

    def _views_without_lens(self) -> list[int]:
        return [v for v, i in enumerate(self.infos)
                if i is not None and i.has_lens and (v >= len(self.lenses) or self.lenses[v] is None)]

    def _fill(self) -> None:
        warn = theme.AMBER
        for v, (name, info) in enumerate(zip(self.names, self.infos)):
            prof = self.lenses[v] if v < len(self.lenses) else None
            if info is None:
                cells = [name, "not GoPro footage (nothing read)", "", "", "", "", "", lens_text(prof)]
            else:
                tl = info.tilt()
                tilt = (f"{abs(tl[0]):.0f}° {'down' if tl[0] >= 0 else 'up'}, roll {tl[1]:+.0f}°"
                        + (f" ({gpmf.OREN_NAMES.get(info.orientation, info.orientation)})"
                           if info.orientation not in ("", "U") else "") if tl else "")
                shut = (f"1/{1 / info.shutter_s:.0f} s" if np.isfinite(info.shutter_s) and info.shutter_s > 0 else "")
                iso = f", ISO {info.iso:.0f}" if np.isfinite(info.iso) else ""
                moves = "; ".join((f"from frame {m['frame']} on ({m['tilt_deg']:.1f}°)" if m["to_end"] and
                                   m["start_s"] > 0.5 else f"frames {m['frame']}–{m['end_frame']} "
                                   f"({m['tilt_deg']:.1f}°)") for m in info.moves)
                cells = [name, info.label + ("" if info.header_found else " (settings not in this copy)"),
                         info.stabilisation or "?", shut + iso, tilt,
                         str(info.dropped_frames) if info.sensors_found else "?",
                         moves or ("no" if info.sensors_found else "?"),
                         lens_text(prof) + (" (GoPro's available)" if prof is None and info.has_lens else "")]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                bad = info is not None and ((c == 2 and info.stabilised) or (c == 5 and info.dropped_frames)
                                            or (c == 6 and info.moves))
                if bad:
                    it.setForeground(Qt.GlobalColor.white)
                    it.setBackground(_qcolor(warn))
                self.table.setItem(v, c, it)
        named = {n: i for n, i in zip(self.names, self.infos) if i is not None}
        lines = [s for n, i in named.items() for s in gpmf.problems(i, n)] + gpmf.rig_problems(named)
        prior = gpmf.timecode_prior(self.infos, [i.fps for i in self.infos]) if all(self.infos) else None
        if prior is not None and len(prior) > 1:
            lines.append("The cameras' timecode tracks put them within " + ", ".join(
                f"{n} {p:+.0f}" for n, p in zip(self.names[1:], prior[1:])) + " frames of the first camera: a "
                "starting point for 3D → Sync Cameras (the camera clocks are set to about a second, so this is not "
                "the sync itself).")
        self.notes.setText("<br>".join(f"• {s}" for s in lines) if lines else
                           "Nothing to warn about: stabilisation off, no dropped frames, no camera moved.")
        todo = self._views_without_lens()
        self.btn_lens.setText(f"Use GoPro's lens model for the {len(todo)} camera(s) without a lens profile"
                              if todo else "Every GoPro camera has a lens profile")
        self.btn_lens.setEnabled(bool(todo) and self.on_use_lenses is not None)
        self.btn_lens.setToolTip("GoPro's own model of the lens for the mode each video was recorded in, read from "
                                 "the video: it covers the whole picture. It is the lens design, not each unit "
                                 "(about 1 % in focal length and ~10 px in centre between units): the wand "
                                 "calibration refines the focal length, a checkerboard (3D → Calibrate a Lens → "
                                 "GoPro lens + your boards) measures both.")

    def _use_lenses(self) -> None:
        views = self._views_without_lens()
        if not views or self.on_use_lenses is None:
            return
        self.said.setText(self.on_use_lenses(views) or "")
        self.lenses = list(self._lenses())
        self._fill()

    def _clicked(self, row: int, col: int) -> None:
        info = self.infos[row] if 0 <= row < len(self.infos) else None
        if col == 6 and info is not None and info.moves and self.on_goto is not None:
            self.on_goto(row, int(info.moves[0]["frame"]))


def _qcolor(hex_color: str):
    from PySide6.QtGui import QColor
    return QColor(hex_color)
