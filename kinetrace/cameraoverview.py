"""3D → Cameras Overview… (G177): every camera of the project in one table -- what it records (picture
size, frame rate, how its video is turned), its lens profile and whether it is used, whether the 3D
calibration covers it -- and, for GoPro footage, what the camera says about itself (G145: settings,
tilt from the gravity sensor, dropped frames, when it moved), with GoPro's lens model for the GoPro
cameras that have no profile. It replaces the GoPro-only dialog (owner 2026-10-08: no quick way to see
the cameras' properties and lenses).

`camera_facts` (no Qt) is also what the CAMERAS panel's lens badge and its tooltip show (`badge`,
`facts_tooltip`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from kinetrace import gpmf, lens, theme

BASE_COLUMNS = ("Camera", "Records", "Lens profile", "3D calibration")
GOPRO_COLUMNS = ("GoPro footage", "Stabilisation", "Shutter / ISO", "Points (tilt, roll)", "Dropped frames", "Moved")
MOVED_COL = len(BASE_COLUMNS) + GOPRO_COLUMNS.index("Moved")


@dataclass
class CameraFacts:
    view: int
    name: str
    video: str
    width: int
    height: int
    fps: float
    file_fps: float
    rotation: int | None         # the turn the decoder applies (`video_source.applied_rotation`)
    gopro: object = None         # gpmf.GoProInfo for GoPro footage
    lens_state: str = "none"     # "ok" (attached and used) / "unused" (attached, does not fit) / "none"
    lens_label: str = "none"     # `lens.lens_label`
    lens_note: str = ""          # the profile's summary, or why it is not used
    calibrated: bool | None = None   # None = the project has no calibration yet
    n_calibrated: int = 0

    @property
    def records(self) -> str:
        """What the camera records, in a few words."""
        t = f"{self.width} x {self.height}, {self.fps:g} fps"
        if abs(self.file_fps - self.fps) > 1e-6:
            t += f" (file says {self.file_fps:g})"
        turned = lens.turn_words(self.rotation)
        return t + (f", plays turned {turned}" if turned else "")

    @property
    def calibration_text(self) -> str:
        if self.calibrated is None:
            return "no calibration yet"
        return "calibrated" if self.calibrated else f"not covered ({self.n_calibrated} cameras calibrated)"


def camera_facts(project, infos) -> list[CameraFacts]:
    """One `CameraFacts` per camera of `project`; `infos` = the cameras' `VideoInfo` (or None) in
    project order. The lens is judged by the attach rule (`calibwizard.lens_for_camera`)."""
    from kinetrace.calibwizard import lens_for_camera
    from kinetrace.video_source import display_rotation
    out = []
    cal = getattr(project, "calibration", None)
    n_cal = len(cal) if cal is not None else 0
    for v in range(project.n_views):
        s = project.sessions[v]
        info = infos[v] if v < len(infos) else None
        rot = getattr(info, "rotation", None) if info is not None else display_rotation(s.video_path)
        f = CameraFacts(v, project.name(v), Path(s.video_path).name if s.video_path else "", int(s.width),
                        int(s.height), float(s.fps), float(getattr(s, "file_fps", s.fps)), rot,
                        getattr(info, "gopro", None))
        prof = project.lenses[v] if v < len(project.lenses) else None
        if prof is not None:
            f.lens_label = lens.lens_label(prof)
            fitted, why = lens_for_camera(prof, s, f.name)
            if fitted is None:
                f.lens_state, f.lens_note = "unused", (why[0].upper() + why[1:] if why else "")
            else:
                f.lens_state, f.lens_note = "ok", fitted.summary()
        f.calibrated = None if cal is None else v < n_cal
        f.n_calibrated = n_cal
        out.append(f)
    return out


def badge(f: CameraFacts) -> tuple[str, str]:
    """The CAMERAS row's lens badge: (text, colour)."""
    if f.lens_state == "ok":
        return "lens ✓", theme.GREEN
    if f.lens_state == "unused":
        return "lens ✗", theme.AMBER
    return "no lens", theme.TEXT_DIM


def facts_tooltip(f: CameraFacts) -> str:
    """Everything about one camera in a few lines: the badge's tooltip."""
    lines = [f"{f.name}" + (f" — {f.video}" if f.video else ""), f"Records {f.records}."]
    if f.rotation:
        lines.append("Its video carries a rotation tag (a camera filmed on its side or upside down): the "
                     "pictures are shown upright, and a lens profile measured with the camera level is "
                     "turned to fit.")
    if f.gopro is not None:
        g = f.gopro
        lines.append(f"GoPro: {g.label}, stabilisation {g.stabilisation or '?'}"
                     + (f", {g.dropped_frames} dropped frame(s)" if getattr(g, "dropped_frames", 0) else "") + ".")
    if f.lens_state == "ok":
        lines.append(f"Lens profile: {f.lens_label}. {f.lens_note}.")
    elif f.lens_state == "unused":
        lines.append(f"Lens profile ATTACHED BUT NOT USED: {f.lens_note}")
    else:
        lines.append("No lens profile: the wand calibration estimates the focal length and corrects no distortion "
                     "(3D → Calibrate a Lens, or 3D → Load a Lens Profile for Cameras…).")
    lines.append(f"3D calibration: {f.calibration_text}.")
    lines.append("Click for 3D → Cameras Overview…")
    return "\n".join(lines)


class CamerasOverview(QDialog):
    """`facts()` = the cameras' `CameraFacts` (read again after a button changed something).
    `on_use_gopro_lenses(views)` gives GoPro's lens model to those cameras and returns a sentence;
    `on_load_lens()` runs 3D → Load a Lens Profile for Cameras…; `on_goto(view, frame)` shows that
    camera at that frame (a GoPro camera's Moved cell)."""

    def __init__(self, parent, facts, on_use_gopro_lenses=None, on_load_lens=None, on_goto=None):
        super().__init__(parent)
        self.setWindowTitle("Cameras overview")
        self._facts_fn = facts
        self.facts = list(facts())
        self.on_use_gopro_lenses, self.on_load_lens, self.on_goto = on_use_gopro_lenses, on_load_lens, on_goto
        self.has_gopro = any(f.gopro is not None for f in self.facts)
        self.columns = BASE_COLUMNS + (GOPRO_COLUMNS if self.has_gopro else ())
        lay = QVBoxLayout(self)
        intro = QLabel("Every camera of the project: what it records, its lens profile and whether it is used, and "
                       "whether the 3D calibration covers it. Hover a cell for the whole sentence."
                       + (" GoPro footage also says what the camera reported about itself (read from the file); "
                          "click a <b>Moved</b> cell to go to that camera and frame." if self.has_gopro else ""))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(len(self.facts), len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
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
        btns = QHBoxLayout()
        self.btn_load = QPushButton("Load a lens profile for cameras…")
        self.btn_load.setToolTip("3D → Load a Lens Profile for Cameras…: one saved lens file for every camera you tick")
        self.btn_load.clicked.connect(self._load_lens)
        self.btn_load.setEnabled(on_load_lens is not None)
        self.btn_lens = QPushButton("")
        self.btn_lens.clicked.connect(self._use_gopro_lenses)
        self.btn_lens.setVisible(self.has_gopro)
        for b in (self.btn_load, self.btn_lens):
            b.setFocusPolicy(Qt.NoFocus)
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.said = QLabel("")
        self.said.setWordWrap(True)
        lay.addWidget(self.said)
        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)
        lay.addWidget(box)
        self._fill()
        # as wide as the columns (no sideways scrolling) within a sane window; every row of a 16-camera rig
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()              # measured now: the header is not laid out yet
        need =self.table.horizontalHeader().length() + 2 * self.table.frameWidth() + 40
        self.resize(min(max(need, 700), 1600), 260 + 30 * min(len(self.facts), 16))

    def _gopro_without_lens(self) -> list[int]:
        return [f.view for f in self.facts if f.gopro is not None and f.gopro.has_lens and f.lens_state == "none"]

    def _fill(self) -> None:
        for r, f in enumerate(self.facts):
            lens_cell = ("none" if f.lens_state == "none" else
                         f.lens_label + (" — NOT USED" if f.lens_state == "unused" else ""))
            if f.lens_state == "none" and f.gopro is not None and f.gopro.has_lens:
                lens_cell += " (GoPro's available)"
            cells = [(f.name, f.video), (f.records, ""), (lens_cell, f.lens_note), (f.calibration_text, "")]
            if self.has_gopro:
                cells += self._gopro_cells(f)
            for c, (text, tip) in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setToolTip(tip or text)
                warn = ((c == 2 and f.lens_state == "unused")
                        or (f.gopro is not None and c >= len(BASE_COLUMNS) and self._gopro_bad(f, c)))
                if warn:
                    it.setForeground(Qt.GlobalColor.white)
                    it.setBackground(QColor(theme.AMBER))
                elif c == 2 and f.lens_state == "none":
                    it.setForeground(QColor(theme.TEXT_DIM))
                self.table.setItem(r, c, it)
        lines = []
        if self.has_gopro:
            named = {f.name: f.gopro for f in self.facts if f.gopro is not None}
            lines += [s for n, i in named.items() for s in gpmf.problems(i, n)] + gpmf.rig_problems(named)
            infos = [f.gopro for f in self.facts]
            prior = gpmf.timecode_prior(infos, [i.fps for i in infos]) if all(infos) else None
            if prior is not None and len(prior) > 1:
                lines.append("The cameras' timecode tracks put them within " + ", ".join(
                    f"{f.name} {pr:+.0f}" for f, pr in zip(self.facts[1:], prior[1:])) + " frames of the first "
                    "camera: a starting point for 3D → Sync Cameras (the camera clocks are set to about a second, "
                    "so this is not the sync itself).")
        none = [f.name for f in self.facts if f.lens_state == "none"]
        unused = [f.name for f in self.facts if f.lens_state == "unused"]
        if unused:
            lines.append(f"Attached but NOT used (another picture size, or a turn not known): {', '.join(unused)}. "
                         "Hover its lens cell for why.")
        if none and len(none) < len(self.facts):
            lines.append(f"No lens profile: {', '.join(none)}.")
        self.notes.setText("<br>".join(f"• {s}" for s in lines) if lines else
                           ("Nothing to warn about." if self.has_gopro else ""))
        self.notes.setVisible(bool(self.notes.text()))
        todo = self._gopro_without_lens()
        self.btn_lens.setText(f"Use GoPro's lens model for the {len(todo)} GoPro camera(s) without a lens profile"
                              if todo else "Every GoPro camera has a lens profile")
        self.btn_lens.setEnabled(bool(todo) and self.on_use_gopro_lenses is not None)
        self.btn_lens.setToolTip("GoPro's own model of the lens for the mode each video was recorded in, read from "
                                 "the video: it covers the whole picture. It is the lens design, not each unit "
                                 "(about 1 % in focal length and ~10 px in centre between units): the wand "
                                 "calibration refines the focal length, a checkerboard (3D → Calibrate a Lens → "
                                 "GoPro lens + your boards) measures both.")

    @staticmethod
    def _gopro_cells(f: CameraFacts) -> list[tuple[str, str]]:
        info = f.gopro
        if info is None:
            return [("not GoPro footage", "Nothing is read from other cameras' files")] + [("", "")] * 5
        tl = info.tilt()
        tilt = (f"{abs(tl[0]):.0f}° {'down' if tl[0] >= 0 else 'up'}, roll {tl[1]:+.0f}°"
                + (f" ({gpmf.OREN_NAMES.get(info.orientation, info.orientation)})"
                   if info.orientation not in ("", "U") else "") if tl else "")
        shut = (f"1/{1 / info.shutter_s:.0f} s" if np.isfinite(info.shutter_s) and info.shutter_s > 0 else "")
        iso = f", ISO {info.iso:.0f}" if np.isfinite(info.iso) else ""
        moves = "; ".join((f"from frame {m['frame']} on ({m['tilt_deg']:.1f}°)" if m["to_end"] and
                           m["start_s"] > 0.5 else f"frames {m['frame']}–{m['end_frame']} "
                           f"({m['tilt_deg']:.1f}°)") for m in info.moves)
        return [(info.label + ("" if info.header_found else " (settings not in this copy)"), ""),
                (info.stabilisation or "?", ""), (shut + iso, ""), (tilt, ""),
                (str(info.dropped_frames) if info.sensors_found else "?", ""),
                (moves or ("no" if info.sensors_found else "?"), "")]

    @staticmethod
    def _gopro_bad(f: CameraFacts, col: int) -> bool:
        info, name = f.gopro, (BASE_COLUMNS + GOPRO_COLUMNS)[col]
        return bool((name == "Stabilisation" and info.stabilised) or (name == "Dropped frames" and info.dropped_frames)
                    or (name == "Moved" and info.moves))

    def _refresh(self) -> None:
        self.facts = list(self._facts_fn())
        self._fill()

    def _use_gopro_lenses(self) -> None:
        views = self._gopro_without_lens()
        if not views or self.on_use_gopro_lenses is None:
            return
        self.said.setText(self.on_use_gopro_lenses(views) or "")
        self._refresh()

    def _load_lens(self) -> None:
        if self.on_load_lens is not None:
            self.on_load_lens()
            self._refresh()

    def _clicked(self, row: int, col: int) -> None:
        f = self.facts[row] if 0 <= row < len(self.facts) else None
        if (f is not None and self.has_gopro and col == MOVED_COL and f.gopro is not None and f.gopro.moves
                and self.on_goto is not None):
            self.on_goto(row, int(f.gopro.moves[0]["frame"]))
