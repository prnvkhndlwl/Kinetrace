"""The CAMERAS panel: one row per view, with the frame offset that aligns it.

**The first camera loaded is the reference**: its offset is 0 by definition and
is not editable, and every other camera's offset is "the frame this camera
shows when the reference is at frame 0" (a camera switched on N frames after the
reference has offset -N). That holds whichever camera you are working in — the
numbers never shift under you when you switch.

A row shows the camera name, a spin box for its **frame offset**, and a nudge
pair. Raising the offset by one shows that camera one frame later against the
same playhead — you nudge until the flash / clap / first contact lines up with
the working view, and that is the whole alignment workflow. "Align here" does
the same in one click once both views are parked on the same event. On the
reference's row it sets the WORKING camera's offset instead (the reference is
the clock and never moves; every other camera keeps its offset, G13).

The active row is the camera being tracked; selecting another row switches to
it (the playhead follows through the offsets, so the picture stays on the same
instant).

The panel is data-free: `MainWindow` rebuilds it from the `Project` and acts on
its signals.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from kinetrace import theme
from kinetrace.project import REFERENCE_VIEW
from kinetrace.widgets import ElidedLabel

OFFSET_LIMIT = 10_000_000     # frames; far beyond any real clip
OFFSET_DECIMALS = 3           # sub-frame sync is measured to ~0.01 frame; show a little more


def _dim_label(text: str = "") -> QLabel:
    """A caption that must never dictate the dock's minimum width (the QLabel
    pitfall — a long single-line label once forced a 3376 px window). It ends
    in "…" when cut, with the whole text as its tooltip."""
    lab = ElidedLabel(text)
    lab.setStyleSheet(f"color: {theme.TEXT_DIM};")
    lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return lab


class _CameraRow(QWidget):
    """Name + offset spin + nudge buttons + align, for one view."""

    offset_changed = Signal(int, float)   # (view index, new offset in this view's frames)
    align_requested = Signal(int)
    remove_requested = Signal(int)
    fps_requested = Signal(int)           # the frame-rate button (G38)

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(4)
        self.name = _dim_label()
        top.addWidget(self.name, 1)
        # "Align here" sits on the name line: on the offset line it made the row
        # wider than the panel and was clipped with the x and the > (G3)
        self.btn_align = QToolButton()
        self.btn_align.setText("Align here")
        self.btn_align.setToolTip(
            "Take the frame this camera is showing right now as the match for the\n"
            "working camera's current frame, and set the offset from that.")
        self.btn_align.clicked.connect(lambda: self.align_requested.emit(self.index))
        top.addWidget(self.btn_align)
        self.btn_remove = QToolButton()
        self.btn_remove.setText("×")
        self.btn_remove.setToolTip("Remove this camera from the project (its tracks are dropped)")
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self.index))
        top.addWidget(self.btn_remove)
        lay.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(3)
        cap = QLabel("offset")                  # short: may size the row
        cap.setStyleSheet(f"color: {theme.TEXT_DIM};")
        row.addWidget(cap)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-OFFSET_LIMIT, OFFSET_LIMIT)
        self.spin.setDecimals(OFFSET_DECIMALS)
        self.spin.setSingleStep(1.0)          # by hand you align whole frames
        self.spin.setFixedWidth(88)            # the offset line must fit the narrowest panel
        self.spin.setToolTip("")   # set per row: the reference reads differently
        self.spin.valueChanged.connect(
            lambda v: self.offset_changed.emit(self.index, float(v)))
        # give the keyboard back as soon as the edit is done (hotkeys, and no
        # field left highlighted); rows are built after the window's own
        # focus-policy pass, so this is set here
        self.spin.setFocusPolicy(Qt.ClickFocus)
        self.spin.editingFinished.connect(self.spin.clearFocus)
        row.addWidget(self.spin)
        self.nudges: list[QToolButton] = []
        for text, delta, tip in (("◂", -1, "one frame earlier"), ("▸", +1, "one frame later")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(f"Show this camera {tip}")
            b.setProperty("compact", True)       # theme: tight padding, the row must fit the panel
            b.clicked.connect(lambda _=False, d=delta: self.spin.setValue(self.spin.value() + d))
            row.addWidget(b)
            self.nudges.append(b)
        self.rate = QLabel("")                # "×2" for a camera at twice the reference rate
        self.rate.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.rate.setToolTip("Frames of this camera per frame of the reference camera\n"
                             "(its fps divided by the reference fps). Companion views step\n"
                             "through the video at this rate.")
        row.addWidget(self.rate)
        row.addStretch(1)
        lay.addLayout(row)

        # the rate this camera recorded at, in front of the status line (the status
        # shortens with "…"; the name line and the offset line have no room to give,
        # G3): click to correct a file whose header gives the wrong rate (G38)
        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        self.btn_fps = QToolButton()
        self.btn_fps.setProperty("compact", True)
        self.btn_fps.clicked.connect(lambda: self.fps_requested.emit(self.index))
        bottom.addWidget(self.btn_fps)
        self.status = _dim_label()
        bottom.addWidget(self.status, 1)
        # no taller than the text line it shares, or every row grows and the list scrolls
        # (the theme's min-height + padding would make it 28 px)
        self.btn_fps.setStyleSheet("QToolButton { min-height: 0px; padding: 0px 6px; }")
        lay.addLayout(bottom)

    def update_row(self, name: str, offset: float, active: bool, status: str,
                   removable: bool, reference: bool = False, rate: float = 1.0,
                   fps: float | None = None, file_fps: float | None = None) -> None:
        weight = "600" if active else "400"
        color = theme.TEXT if active else theme.TEXT_DIM
        tag = "  (reference)" if reference else ""
        self.name.setText(name + tag)
        self.name.setStyleSheet(f"color: {color}; font-weight: {weight};")
        if abs(self.spin.value() - float(offset)) > 0.5 * 10 ** -OFFSET_DECIMALS:
            self.spin.blockSignals(True)
            self.spin.setValue(float(offset))
            self.spin.blockSignals(False)
        self.rate.setText("" if abs(rate - 1.0) < 1e-9 else f"×{rate:g}")
        self.rate.setVisible(abs(rate - 1.0) >= 1e-9)
        # The FIRST camera is the clock: its offset is 0 by definition, so there
        # is nothing to type there. Every other camera stays editable whichever
        # one you happen to be working in — its offset means the same thing
        # either way ("its frame when the reference is at frame 0").
        self.spin.setEnabled(not reference)
        for b in self.nudges:               # the arrows only move the box: off with it (G128)
            b.setEnabled(not reference)
        self.spin.setToolTip(
            "This camera is the reference: every other offset is measured\n"
            "against it, so it is 0 by definition. To retime the set, change\n"
            "the other cameras."
            if reference else
            "The frame this camera shows when the reference camera is at its frame 0.\n"
            "A camera switched on N frames AFTER the reference has offset -N;\n"
            "one switched on earlier has +N. Nudge with the arrows until the same\n"
            "moment shows in both views, or park it on that moment and press Align here.")
        # on the reference's row it sets the WORKING camera's offset (G13)
        self.btn_align.setEnabled(not active)
        # a disabled button still shows its tooltip: say WHY it is off (G13)
        self.btn_align.setToolTip(
            "This is the working camera: Align here is on the OTHER cameras' rows.\n"
            "Park this camera on a moment, park another camera on the same moment,\n"
            "and press Align here on that camera's row."
            if active else
            "Take the frame the reference is showing right now as the match for the\n"
            "working camera's current frame. The reference is the clock (offset 0)\n"
            "and does not move: the WORKING camera's offset is set, and every other\n"
            "camera keeps its own."
            if reference else
            "Take the frame this camera is showing right now as the match for the\n"
            "working camera's current frame, and set the offset from that.")
        self.btn_remove.setEnabled(removable)
        self.status.setText(status)
        self.btn_fps.setVisible(bool(fps))
        if fps:
            changed = file_fps is not None and abs(float(file_fps) - float(fps)) > 1e-6
            self.btn_fps.setText(f"{float(fps):g} fps" + (" *" if changed else ""))
            self.btn_fps.setToolTip(
                (f"This camera recorded at {float(fps):g} frames per second - set by hand; its video file "
                 f"says {float(file_fps):g}." if changed else
                 f"This camera recorded at {float(fps):g} frames per second (read from its video file).")
                + "\nClick if that is wrong: high-speed footage is often saved for slow-motion playback,\n"
                "so the file says 30 while the camera filmed at 240 or 1000. Times, speeds and the\n"
                "matching of cameras all use this number.")


class CameraPanel(QWidget):
    """The CAMERAS section of the right dock."""

    activate_requested = Signal(int)
    offset_changed = Signal(int, float)
    align_requested = Signal(int)
    remove_requested = Signal(int)
    add_requested = Signal()
    fps_requested = Signal(int)          # a row's frame-rate button (G38)
    sync_toggled = Signal(bool)          # Sync all views (True) / Active view only (False), G24

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(4)
        title = QLabel("CAMERAS")
        title.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-weight: 600; letter-spacing: 1px;")
        head.addWidget(title)
        head.addStretch(1)
        self.btn_sync = QToolButton()
        self.btn_sync.setText("Sync all")
        self.btn_sync.setCheckable(True)
        self.btn_sync.setChecked(True)
        self.btn_sync.setToolTip(
            "Sync all: every camera follows the playhead.\n"
            "Off (Active view only): only the working camera reads its video; the others stay on\n"
            "the picture they last showed until Sync all is back — much faster with many 4K cameras.\n"
            "Click another camera to work in it. Also under View → Other cameras.")
        self.btn_sync.toggled.connect(self.sync_toggled)
        self.btn_sync.setVisible(False)       # shown with a second camera
        head.addWidget(self.btn_sync)
        self.btn_add = QToolButton()
        self.btn_add.setText("＋ Add video")
        self.btn_add.setToolTip(
            "Add another camera's video of the same event.\n"
            "Each camera keeps its own points and silhouette; the frame offset\n"
            "keeps them on the same instant.")
        self.btn_add.clicked.connect(self.add_requested)
        head.addWidget(self.btn_add)
        lay.addLayout(head)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SingleSelection)
        self.list.setToolTip("The camera being tracked. Click another to switch to it.")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # re-lay the rows out when the viewport changes width (a vertical scroll
        # bar appearing, the panel resized): the default Fixed mode keeps the old
        # width and clips the row's right-hand buttons
        self.list.setResizeMode(QListWidget.Adjust)
        # ... but a top-to-bottom list re-lays out only when its HEIGHT changes:
        # a narrower panel left the row widgets at their old width (G3)
        self.list.viewport().installEventFilter(self)
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list, 1)
        self.note = QLabel()                  # wraps (several lines), so it is not elided
        self.note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self._rows: list[_CameraRow] = []
        self._suppress = False

    def eventFilter(self, obj, ev):             # noqa: N802 - Qt name
        if obj is self.list.viewport() and ev.type() == QEvent.Resize:
            QTimer.singleShot(0, self.list.doItemsLayout)
        return False

    def _on_row(self, row: int) -> None:
        if not self._suppress and row >= 0:
            self.activate_requested.emit(row)

    def rebuild(self, n: int) -> None:
        """Make the list hold exactly `n` rows, keeping existing widgets."""
        while len(self._rows) < n:
            row = _CameraRow(len(self._rows))
            row.offset_changed.connect(self.offset_changed)
            row.align_requested.connect(self.align_requested)
            row.remove_requested.connect(self.remove_requested)
            row.fps_requested.connect(self.fps_requested)
            item = QListWidgetItem()          # NOT QListWidgetItem(self.list):
            # that inserts it, and addItem would again. Height from the row; width
            # left to the list, which stretches rows to its viewport (a natural-width
            # hint put a horizontal scroll bar under a clipped row, G3)
            item.setSizeHint(QSize(1, row.sizeHint().height()))
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            self._rows.append(row)
        while len(self._rows) > n:
            self._rows.pop()
            self.list.takeItem(self.list.count() - 1)

    def update_rows(self, names: list[str], offsets: list[float], active: int,
                    statuses: list[str], note: str = "",
                    rates: list[float] | None = None, fps: list[float] | None = None,
                    file_fps: list[float] | None = None) -> None:
        self.rebuild(len(names))
        for i, row in enumerate(self._rows):
            row.update_row(names[i], offsets[i], i == active, statuses[i],
                           len(names) > 1, reference=(i == REFERENCE_VIEW),
                           rate=(rates[i] if rates and i < len(rates) else 1.0),
                           fps=(fps[i] if fps and i < len(fps) else None),
                           file_fps=(file_fps[i] if file_fps and i < len(file_fps) else None))
        self._suppress = True            # programmatic selection must not re-emit
        self.list.setCurrentRow(active)
        self._suppress = False
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.btn_sync.setVisible(len(names) > 1)

    def set_sync(self, on: bool) -> None:
        """Mirror the app's choice without emitting `sync_toggled` back."""
        self.btn_sync.blockSignals(True)
        self.btn_sync.setChecked(bool(on))
        self.btn_sync.blockSignals(False)
