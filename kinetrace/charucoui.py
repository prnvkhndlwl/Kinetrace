"""The ChArUco board's boxes, shared by the lens wizard and the camera calibration wizard
(G220, G221): squares across and down, the square and marker size with their unit, the
ArUco dictionary, and a line that says at once what is wrong with a typed board.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QSpinBox, QWidget)

from kinetrace import charuco, theme
from kinetrace.widgets import wrap_caption

UNITS = [("millimetres (mm)", 0.001), ("centimetres (cm)", 0.01), ("metres (m)", 1.0), ("inches (in)", 0.0254)]


class CharucoFields(QWidget):
    """The board typed in. `spec()` = a `charuco.BoardSpec` (None while `problem()` says why
    it cannot exist); `changed` fires on every edit."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._legacy = False            # the older OpenCV layout, when a scan found the board printed so
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        self.sx = QSpinBox()
        self.sx.setRange(3, 40)
        self.sx.setValue(charuco.DEFAULT_SQUARES[0])
        self.sy = QSpinBox()
        self.sy.setRange(3, 40)
        self.sy.setValue(charuco.DEFAULT_SQUARES[1])
        rl.addWidget(self.sx)
        rl.addWidget(QLabel("×"))
        rl.addWidget(self.sy)
        hint = QLabel("squares across × down (count the squares, not the corners; the printed board has 7 × 5)")
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        hint.setWordWrap(True)
        rl.addWidget(hint, 1)
        form.addRow("ChArUco squares", row)
        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        self.square = QDoubleSpinBox()
        self.square.setDecimals(3)
        self.square.setRange(0.001, 10000.0)
        self.square.setValue(charuco.DEFAULT_SQUARE_MM)
        self.marker = QDoubleSpinBox()
        self.marker.setDecimals(3)
        self.marker.setRange(0.001, 10000.0)
        self.marker.setValue(charuco.DEFAULT_MARKER_MM)
        self.unit = QComboBox()
        for label, _ in UNITS:
            self.unit.addItem(label)
        sl.addWidget(QLabel("square"))
        sl.addWidget(self.square)
        sl.addWidget(QLabel("marker"))
        sl.addWidget(self.marker)
        sl.addWidget(self.unit)
        sl.addStretch(1)
        form.addRow("Measured sizes", srow)
        self.square.setToolTip("One black square's side, measured with a ruler on the printed board")
        self.marker.setToolTip("One marker's side (the black outline of the small pattern inside a white square)")
        self.dictionary = QComboBox()
        for d in charuco.DICTIONARIES:
            self.dictionary.addItem(d)
        self.dictionary.setCurrentText(charuco.DEFAULT_DICTIONARY)
        self.dictionary.setToolTip("The family of markers the board was printed with: the program or website that "
                                   "made the board names it (Kinetrace's own board is DICT_4X4_50)")
        form.addRow("Marker dictionary", self.dictionary)
        self.note = QLabel("")
        wrap_caption(self.note)
        self.note.setStyleSheet(f"color: {theme.VERDICT['poor']};")
        form.addRow("", self.note)
        for w in (self.sx, self.sy, self.square, self.marker):
            w.valueChanged.connect(self._edited)
        self.unit.currentIndexChanged.connect(self._edited)
        self.dictionary.currentIndexChanged.connect(self._edited)
        self._edited()

    def _edited(self, *_):
        self._legacy = False            # a board typed anew: the scan decides the layout again
        bad = self._spec().problem()
        self.note.setText(bad[0].upper() + bad[1:] + "." if bad else "")
        self.note.setVisible(bool(bad))
        self.changed.emit()

    def _spec(self) -> charuco.BoardSpec:
        u = UNITS[self.unit.currentIndex()][1]
        return charuco.BoardSpec(int(self.sx.value()), int(self.sy.value()), float(self.square.value()) * u,
                                 float(self.marker.value()) * u, self.dictionary.currentText(), self._legacy)

    def problem(self) -> str:
        return self._spec().problem()

    def spec(self) -> charuco.BoardSpec | None:
        s = self._spec()
        return None if s.problem() else s

    def set_spec(self, spec: charuco.BoardSpec) -> None:
        """Show `spec` (the board a scan used, its layout included) without firing `changed`."""
        u = UNITS[self.unit.currentIndex()][1]
        for w, v in ((self.sx, spec.squares_x), (self.sy, spec.squares_y), (self.square, spec.square_m / u),
                     (self.marker, spec.marker_m / u)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.dictionary.blockSignals(True)
        self.dictionary.setCurrentText(spec.dictionary)
        self.dictionary.blockSignals(False)
        self._legacy = bool(spec.legacy)

    def set_enabled_inputs(self, on: bool) -> None:
        for w in (self.sx, self.sy, self.square, self.marker, self.unit, self.dictionary):
            w.setEnabled(on)
