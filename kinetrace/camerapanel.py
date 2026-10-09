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

(G169) With many cameras every row is ONE compact line -- its disclosure ▶, its
eye (show / hide the camera's view: a hidden view is not decoded, and the views
left get the room), its number, name and offset -- so all of them fit; ▶ opens
the row's controls (Align here, the offset box and its nudges, the frame rate,
remove). The numbers are the cameras' ORDER (camera 1 = the reference), the
order of calibrations, 3D and exports; Camera order... changes it.

The panel is data-free: `MainWindow` rebuilds it from the `Project` and acts on
its signals.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from kinetrace import icons, theme
from kinetrace.project import REFERENCE_VIEW
from kinetrace.widgets import ElidedLabel

OFFSET_LIMIT = 10_000_000     # frames; far beyond any real clip
OFFSET_DECIMALS = 3           # sub-frame sync is measured to ~0.01 frame; show a little more
LIST_MAX_H = 330              # px: the CAMERAS list grows with its rows up to this, then scrolls (G169)


def _dim_label(text: str = "") -> QLabel:
    """A caption that must never dictate the dock's minimum width (the QLabel
    pitfall — a long single-line label once forced a 3376 px window). It ends
    in "…" when cut, with the whole text as its tooltip."""
    lab = ElidedLabel(text)
    lab.setStyleSheet(f"color: {theme.TEXT_DIM};")
    lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return lab


class _CameraRow(QWidget):
    """One camera: a compact line -- disclosure, eye, number + name, offset (G169) -- and, opened,
    its controls: Align here and remove on that line, the offset box with its nudges, the frame rate
    and the status line."""

    offset_changed = Signal(int, float)   # (view index, new offset in this view's frames)
    align_requested = Signal(int)
    remove_requested = Signal(int)
    fps_requested = Signal(int)           # the frame-rate button (G38)
    shown_toggled = Signal(int, bool)     # (G169) its eye: (view index, shown)
    expand_toggled = Signal(int)          # (G169) its disclosure was clicked (the row's height changed)
    overview_requested = Signal(int)      # (G177) its lens badge was clicked

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self._offset_text = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 6, 2)
        lay.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(3)
        # (G169) the disclosure and the eye lead the line: a glyph button and a drawn icon, never a QStyle
        # icon (black on the dark theme)
        # (G169) no taller than the text line: the theme's button min-height + padding made every camera
        # row ~35 px, and ten cameras no longer fitted the list; the eye shows its state by its picture
        # (open / struck through), not by the theme's checked-button fill
        flat = ("QToolButton { min-height: 0px; min-width: 0px; padding: 0px; border: none; background: transparent; }"
                f"QToolButton:hover {{ background: {theme.HAIRLINE}; border-radius: 3px; }}")
        self.btn_expand = QToolButton()
        self.btn_expand.setAutoRaise(True)
        self.btn_expand.setFocusPolicy(Qt.NoFocus)
        self.btn_expand.setStyleSheet(flat)
        self.btn_expand.setFixedSize(18, 20)
        self.btn_expand.clicked.connect(lambda: self.set_expanded(not self.expanded()))
        top.addWidget(self.btn_expand)
        self.btn_eye = QToolButton()
        self.btn_eye.setAutoRaise(True)
        self.btn_eye.setCheckable(True)
        self.btn_eye.setChecked(True)
        self.btn_eye.setFocusPolicy(Qt.NoFocus)
        self.btn_eye.setStyleSheet(flat)
        self.btn_eye.setIconSize(QSize(16, 16))
        self.btn_eye.setFixedSize(22, 20)
        self.btn_eye.toggled.connect(self._on_eye)
        top.addWidget(self.btn_eye)
        self.name = _dim_label()
        top.addWidget(self.name, 1)
        # the offset at a glance while the row is shut (the box shows it when open)
        self.summary = ElidedLabel()
        self.summary.setStyleSheet(f"color: {theme.TEXT_DIM};")
        top.addWidget(self.summary)
        # (G177) the lens at a glance: "lens ✓" / "lens ✗" (attached, not used) / "no lens"; its tooltip says
        # what the camera records, its lens and its calibration; a click opens 3D -> Cameras Overview
        self.lens_badge = QToolButton()
        self.lens_badge.setAutoRaise(True)
        self.lens_badge.setFocusPolicy(Qt.NoFocus)
        self.lens_badge.clicked.connect(lambda: self.overview_requested.emit(self.index))
        self.lens_badge.setVisible(False)
        top.addWidget(self.lens_badge)
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

        # what ▶ opens: the offset line and the frame-rate / status line
        self.details = QWidget()
        dl = QVBoxLayout(self.details)
        dl.setContentsMargins(0, 0, 0, 2)
        dl.setSpacing(2)
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
        dl.addLayout(row)

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
        dl.addLayout(bottom)
        lay.addWidget(self.details)
        self._expanded = True             # so set_expanded(False) below takes effect
        self.set_expanded(False)

    # ------------------------------------------------------------- open / shut (G169)

    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, on: bool, emit: bool = True) -> None:
        on = bool(on)
        if on == self._expanded:
            return
        self._expanded = on
        self.details.setVisible(on)
        self.btn_align.setVisible(on)
        self.btn_remove.setVisible(on)
        self.summary.setVisible(not on and bool(self._offset_text))
        self.btn_expand.setText("▼" if on else "▶")      # not the nudges' ◂ ▸ (one frame earlier / later)
        self.btn_expand.setToolTip("Hide this camera's controls" if on else
                                   "Show this camera's controls: Align here, its offset and nudges, its frame "
                                   "rate, remove")
        if emit:
            self.expand_toggled.emit(self.index)

    def _on_eye(self, on: bool) -> None:
        self.btn_eye.setIcon(icons.eye() if on else icons.eye_off())
        self.shown_toggled.emit(self.index, bool(on))

    def set_shown(self, shown: bool, active: bool, solo: bool) -> None:
        """The eye, without emitting: on = the view is on screen. The working camera is always
        shown; with View -> Other cameras -> Only the working camera the eyes rest (G169)."""
        self.btn_eye.blockSignals(True)
        self.btn_eye.setChecked(bool(shown or active))
        self.btn_eye.blockSignals(False)
        self.btn_eye.setIcon(icons.eye() if (shown or active) else icons.eye_off())
        self.btn_eye.setEnabled(not active and not solo)
        self.btn_eye.setToolTip(
            "The working camera is always shown" if active else
            "Only the working camera is shown (View → Other cameras → Only the working camera, Ctrl+2): "
            "untick that to choose the cameras here" if solo else
            "Shown: click to hide this camera's view. A hidden view is not read from its video (faster) and "
            "the other views get its room; its tracks still count for 3D and the guides." if shown else
            "Hidden: click to show this camera's view again")

    def set_lens(self, lens: tuple[str, str, str] | None) -> None:
        """(G177) The lens badge: (text, colour, tooltip), or None to hide it."""
        self.lens_badge.setVisible(lens is not None)
        if lens is None:
            return
        text, color, tip = lens
        self.lens_badge.setText(text)
        # as tall as the text line (the theme's button min-height would grow every row, G169)
        self.lens_badge.setStyleSheet(
            f"QToolButton {{ min-height: 0px; min-width: 0px; padding: 0px 4px; border: none; background: transparent;"
            f" color: {color}; }} QToolButton:hover {{ background: {theme.HAIRLINE}; border-radius: 3px; }}")
        self.lens_badge.setToolTip(tip)

    def update_row(self, name: str, offset: float, active: bool, status: str,
                   removable: bool, reference: bool = False, rate: float = 1.0,
                   fps: float | None = None, file_fps: float | None = None, number: int | None = None,
                   lens: tuple[str, str, str] | None = None) -> None:
        weight = "600" if active else "400"
        self.set_lens(lens)
        color = theme.TEXT if active else theme.TEXT_DIM
        tag = "  (reference)" if reference else ""
        # (G169) its number = its place in the camera ORDER (calibration, 3D, exports)
        self.name.setText((f"{number}  " if number is not None else "") + name + tag)
        self.name.setStyleSheet(f"color: {color}; font-weight: {weight};")
        off = f"{float(offset):+.3f}".rstrip("0").rstrip(".")
        self._offset_text = "" if reference else (f"offset {off}" + ("" if abs(rate - 1.0) < 1e-9 else f" · ×{rate:g}"))
        self.summary.setText(self._offset_text)
        self.summary.setVisible(not self._expanded and bool(self._offset_text))
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
    overview_requested = Signal(int)     # (G177) a row's lens badge
    sync_toggled = Signal(bool)          # Sync all views (True) / Active view only (False), G24
    shown_toggled = Signal(int, bool)    # (G169) a row's eye: (view index, shown)
    show_all_requested = Signal()        # (G169) Show all: every hidden view back
    order_requested = Signal()           # (G172) Camera order...

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
        self.list.setToolTip("The cameras, in their ORDER (camera 1 = the reference clock; the order of\n"
                             "calibrations, 3D and exports). The highlighted one is being tracked: click\n"
                             "another to switch to it. The eye shows / hides a camera's view; ▶ opens its\n"
                             "controls (Align here, offset, frame rate).")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # (G175) no vertical item padding: an item is exactly its row's height (`_fit_row`), and the
        # theme's 4 px above and below squeezed the 23 px row to 15 px -- the name lost its descenders
        # and underscores on real fonts
        self.list.setStyleSheet("QListWidget::item { padding: 0px 2px; }")
        # re-lay the rows out when the viewport changes width (a vertical scroll
        # bar appearing, the panel resized): the default Fixed mode keeps the old
        # width and clips the row's right-hand buttons
        self.list.setResizeMode(QListWidget.Adjust)
        # ... but a top-to-bottom list re-lays out only when its HEIGHT changes:
        # a narrower panel left the row widgets at their old width (G3)
        self.list.viewport().installEventFilter(self)
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list, 1)
        # (G169, G172) under the list, only with several cameras: every hidden view back (only while
        # some are hidden: it says how many), and the cameras' order
        foot = QHBoxLayout()
        foot.setSpacing(4)
        self.btn_show_all = QToolButton()
        self.btn_show_all.setFocusPolicy(Qt.NoFocus)
        self.btn_show_all.clicked.connect(self.show_all_requested)
        foot.addWidget(self.btn_show_all)
        foot.addStretch(1)
        self.btn_order = QToolButton()
        self.btn_order.setText("Camera order…")
        self.btn_order.setFocusPolicy(Qt.NoFocus)
        self.btn_order.setToolTip(
            "Put the cameras in another order: camera 1 is the reference clock, and the order is the order\n"
            "of the cameras in calibrations, 3D and exports (keep it the same as your calibration's).\n"
            "Each camera keeps its points, offset, frame rate, lens and calibration. Dragging a view's title\n"
            "bar only moves it on screen.")
        self.btn_order.clicked.connect(self.order_requested)
        foot.addWidget(self.btn_order)
        lay.addLayout(foot)
        self.note = QLabel()                  # wraps (several lines), so it is not elided
        self.note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self._rows: list[_CameraRow] = []
        self._suppress = False
        self.btn_show_all.setVisible(False)
        self.btn_order.setVisible(False)

    def eventFilter(self, obj, ev):             # noqa: N802 - Qt name
        if obj is self.list.viewport() and ev.type() == QEvent.Resize:
            QTimer.singleShot(0, self.list.doItemsLayout)
        return False

    def _on_row(self, row: int) -> None:
        if not self._suppress and row >= 0:
            self.activate_requested.emit(row)

    def rebuild(self, n: int) -> None:
        """Make the list hold exactly `n` rows, keeping existing widgets."""
        changed = False
        while len(self._rows) < n:
            row = _CameraRow(len(self._rows))
            row.offset_changed.connect(self.offset_changed)
            row.align_requested.connect(self.align_requested)
            row.remove_requested.connect(self.remove_requested)
            row.fps_requested.connect(self.fps_requested)
            row.shown_toggled.connect(self.shown_toggled)
            row.expand_toggled.connect(self._fit_row)
            row.overview_requested.connect(self.overview_requested)
            item = QListWidgetItem()          # NOT QListWidgetItem(self.list):
            # that inserts it, and addItem would again. Height from the row; width
            # left to the list, which stretches rows to its viewport (a natural-width
            # hint put a horizontal scroll bar under a clipped row, G3)
            item.setSizeHint(QSize(1, row.sizeHint().height()))
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            self._rows.append(row)
            changed = True
        while len(self._rows) > n:
            self._rows.pop()
            self.list.takeItem(self.list.count() - 1)
            changed = True
        if changed:
            self._fit_height()

    def _fit_row(self, i: int) -> None:
        """Row `i` was opened / shut (G169): its item takes the row's new height, and the list its rows'."""
        if not (0 <= i < len(self._rows)):
            return
        row = self._rows[i]
        row.layout().activate()
        self.list.item(i).setSizeHint(QSize(1, row.sizeHint().height()))
        self.list.doItemsLayout()
        self._fit_height()

    def _fit_height(self) -> None:
        """(G169) The list is as tall as its rows (all of them visible at once, one compact line each),
        up to LIST_MAX_H; past that it scrolls."""
        rows_h = sum(self.list.item(k).sizeHint().height() for k in range(self.list.count()))
        self.list.setFixedHeight(max(28, min(LIST_MAX_H, rows_h + 2 * self.list.frameWidth() + 2)))

    def update_rows(self, names: list[str], offsets: list[float], active: int,
                    statuses: list[str], note: str = "",
                    rates: list[float] | None = None, fps: list[float] | None = None,
                    file_fps: list[float] | None = None, shown: list[bool] | None = None,
                    solo: bool = False, lenses: list | None = None) -> None:
        self.rebuild(len(names))
        several = len(names) > 1
        for i, row in enumerate(self._rows):
            row.update_row(names[i], offsets[i], i == active, statuses[i],
                           several, reference=(i == REFERENCE_VIEW),
                           rate=(rates[i] if rates and i < len(rates) else 1.0),
                           fps=(fps[i] if fps and i < len(fps) else None),
                           file_fps=(file_fps[i] if file_fps and i < len(file_fps) else None),
                           number=(i + 1) if several else None,
                           lens=(lenses[i] if lenses and i < len(lenses) else None))
            row.set_shown(bool(shown[i]) if shown and i < len(shown) else True, i == active, solo)
            row.btn_eye.setVisible(several)          # one camera: nothing to show or hide
        self._suppress = True            # programmatic selection must not re-emit
        self.list.setCurrentRow(active)
        self._suppress = False
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.btn_sync.setVisible(several)
        n_hidden = sum(1 for i, s in enumerate(shown or []) if not s and i != active) if not solo else 0
        self.btn_show_all.setText(f"Show all ({n_hidden} hidden)")
        self.btn_show_all.setToolTip("Show every camera's view again (each eye in the list shows / hides one)")
        self.btn_show_all.setVisible(several and n_hidden > 0)
        self.btn_order.setVisible(several)

    def set_sync(self, on: bool) -> None:
        """Mirror the app's choice without emitting `sync_toggled` back."""
        self.btn_sync.blockSignals(True)
        self.btn_sync.setChecked(bool(on))
        self.btn_sync.blockSignals(False)
