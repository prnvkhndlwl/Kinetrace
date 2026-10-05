"""Checkerboard review: look at every board the scan found, fix a corner that
went astray, and choose which views the calibration is fitted from.

Why this exists: a lens calibration is a number you cannot
check by looking at it. The only honest way to trust one is to see the boards
it was fitted from -- that the corners sit on the corners, that they were found
in the same order every time, and that the board's own axes point the way the
board actually lies. One view with its corners found in the wrong order, or a
blurred frame, quietly drags the whole fit; before this page there was no way
to notice, let alone to do anything about it.

Three things the page gives you:

* **every board, scrollable**, with its corners, the order they were found in
  and the board's X/Y/Z axes drawn on it, plus how far that view reprojects;
* **drag a corner** to where it should be, on the full-resolution frame;
* **a tick per image**, so a view you do not trust is simply left out -- with
  an automatic choice as the starting point rather than the only option.

Drawing is OpenCV into a QPixmap, like the rest of the app; no OpenGL.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from kinetrace import lens, theme

TILE_W = 300                 # thumbnail width in the gallery


def pixmap(bgr: np.ndarray, width: int | None = None, max_w: int | None = None) -> QPixmap:
    """A BGR picture as a QPixmap: scaled to exactly `width` px wide, or only
    DOWN to `max_w` (one helper for the gallery and the result page, R19)."""
    img = bgr
    if width and img.shape[1] != width:
        h = max(1, round(img.shape[0] * width / img.shape[1]))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    elif max_w and img.shape[1] > max_w:
        h = max(1, round(img.shape[0] * max_w / img.shape[1]))
        img = cv2.resize(img, (max_w, h), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


def quality(err: float, size: tuple[int, int] | None = None) -> tuple[str, str]:
    """(word, colour) for one view's reprojection error: green up to the lens
    report's "good" limit, amber up to its "ok" limit, red beyond - the SAME
    limits the report grades the fit with (`lens.rms_limits`, R19; the tiles
    used a fixed 0.6 / 1.5 px beside the report's resolution-scaled ones)."""
    if not np.isfinite(err):
        return "not scored", theme.TEXT_DIM
    good, ok = lens.rms_limits(*(size or (1920, 1080)))
    if err <= good:
        return f"{err:.2f} px", theme.GREEN
    if err <= ok:
        return f"{err:.2f} px", "#E0A030"
    return f"{err:.2f} px", theme.RED


class _FrameReader(QThread):
    """Reads ONE full-resolution frame for the corner editor, off the GUI
    thread and with its own VideoCapture (one capture per thread) (I81).
    Emits (board index, BGR frame or None, why it failed)."""

    frame_ready = Signal(int, object, str)

    def __init__(self, i: int, video: str, frame_no: int):
        super().__init__()
        self.i, self.video, self.frame_no = int(i), str(video), int(frame_no)

    def run(self):
        bgr, why = None, ""
        name = Path(self.video).name
        try:
            if not Path(self.video).exists():
                why = (f"the video is no longer at {self.video} (moved, renamed, or on a drive that is "
                       "not connected)")
            else:
                from kinetrace.video_source import open_capture
                cap = open_capture(self.video)
                try:
                    if not cap.isOpened():
                        why = f"{name} could not be opened as a video"
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no)
                        ok, img = cap.read()
                        if ok and img is not None:
                            bgr = img
                        else:
                            why = f"frame {self.frame_no} of {name} could not be decoded"
                finally:
                    cap.release()
        except Exception as exc:                                 # noqa: BLE001
            why = f"reading {name} failed ({exc})"
        self.frame_ready.emit(self.i, bgr, why)


class CornerEditor(QDialog):
    """One board at full resolution: drag any corner onto where it belongs.

    The frame is re-read from the video rather than kept in memory -- the scan
    holds 480 px thumbnails so that a hundred 4K boards cost 40 MB instead of
    2.5 GB, and this is the one place the real pixels are needed. The caller
    deletes the dialog after use (I249: each one kept ~47 MB of a 5.3K frame
    for the whole session).
    """

    def __init__(self, parent, bgr: np.ndarray, corners: np.ndarray,
                 pattern: tuple[int, int], square: float, prof, frame_no: int, orient: str = ""):
        super().__init__(parent)
        self.setWindowTitle(f"Frame {frame_no} - drag a corner to correct it")
        self.resize(1100, 820)
        self.bgr = bgr
        self.original = np.asarray(corners, np.float64).reshape(-1, 2).copy()
        self.corners = self.original.copy()
        self.pattern = pattern
        self.square = square
        self.prof = prof
        self._drag: int | None = None
        self._dirty = False
        self._gray: np.ndarray | None = None

        lay = QVBoxLayout(self)
        # (G138) the hint follows how THIS board's corner 0 was chosen: "the same physical corner on
        # every board" is false for a symmetric board and for one too faint to tell by colour
        hint = QLabel("Drag any green dot onto the true corner. The ringed dot is "
                      + lens.corner_zero_text(orient, pattern)
                      + " The arrows are the board's own axes: if they point somewhere silly, the corners "
                        "were found in the wrong order and the view is better left out.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(hint)
        self.view = _EditorCanvas(self)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.view, 1)
        self.info = QLabel("")
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.info)
        row = QHBoxLayout()
        btn_reset = QPushButton("Put them back")
        btn_reset.clicked.connect(self._reset)
        row.addWidget(btn_reset)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Keep the change")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)
        self._refresh()

    def gray(self) -> np.ndarray:
        """The frame in grey, made once (the corner snap on every mouse release)."""
        if self._gray is None:
            self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    def _reset(self):
        self.corners = self.original.copy()
        self._dirty = False
        self._refresh()

    def moved(self) -> bool:
        return self._dirty

    def _refresh(self):
        self.view.update()
        try:
            e = lens.per_view_errors([self.corners], self.pattern, self.square, self.prof)[0]
        except Exception:                                        # noqa: BLE001
            e = float("nan")
        moved = int((np.linalg.norm(self.corners - self.original, axis=1) > 0.5).sum())
        word = "unchanged" if not moved else f"{moved} corner{'s' if moved != 1 else ''} moved"
        self.info.setText(f"{len(self.corners)} corners - {word} - this view reprojects at "
                          + ("?" if not np.isfinite(e) else f"{e:.2f} px"))


class _EditorCanvas(QWidget):
    """The picture, fitted to the widget, with the corners drawn on top and
    draggable. Kept separate so the dialog stays readable. The frame is turned
    into a QPixmap ONCE; every repaint draws it scaled and paints the corners,
    the first-row line, the ring and the axes with QPainter (the whole frame was
    re-marked with OpenCV and re-converted on every mouse move: 28 ms at 4K, R19)."""

    def __init__(self, dlg: CornerEditor):
        super().__init__(dlg)
        self.dlg = dlg
        self.setMouseTracking(True)
        self._hover: int | None = None
        self._base = pixmap(dlg.bgr)

    # ---- geometry ----
    def _fit(self) -> tuple[float, float, float]:
        """(scale, offset x, offset y) mapping video pixels to widget pixels."""
        ih, iw = self.dlg.bgr.shape[:2]
        s = min(self.width() / max(1, iw), self.height() / max(1, ih))
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def _to_widget(self, p):
        s, ox, oy = self._fit()
        return np.asarray(p, np.float64) * s + [ox, oy]

    def _to_image(self, x, y):
        s, ox, oy = self._fit()
        return np.array([(x - ox) / max(s, 1e-9), (y - oy) / max(s, 1e-9)])

    # ---- painting ----
    def paintEvent(self, _ev):
        d = self.dlg
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(theme.BG_CANVAS))
        s, ox, oy = self._fit()
        ih, iw = d.bgr.shape[:2]
        p.drawPixmap(QRect(int(ox), int(oy), int(iw * s), int(ih * s)), self._base)
        pts = self._to_widget(d.corners)
        cols = int(d.pattern[0])
        if len(pts) >= cols:
            # the first row, so a flipped or rotated detection is obvious (what lens.draw_board_review draws)
            p.setPen(QPen(QColor(90, 210, 250), 2))
            p.drawPolyline([QPointF(float(x), float(y)) for x, y in pts[:cols]])
        if d.prof is not None:
            for lab, a, b in lens.board_axes(d.corners, d.pattern, d.square, d.prof):
                col = QColor(*lens.AXIS_COLORS_BGR[lab][::-1])
                wa, wb = self._to_widget(a), self._to_widget(b)
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(*wa), QPointF(*wb))
                v = wb - wa
                n = float(np.linalg.norm(v))
                if n > 1e-6:
                    u = v / n
                    nrm = np.array([-u[1], u[0]])
                    for sgn in (1.0, -1.0):
                        tip = wb - 10.0 * u + sgn * 5.0 * nrm
                        p.drawLine(QPointF(*wb), QPointF(*tip))
                p.drawText(QPointF(float(wb[0]) + 4, float(wb[1]) + 5), lab)
        # the draggable handles, in widget space so they stay a usable size
        for i, (x, y) in enumerate(pts):
            hot = (i == self._hover) or (i == d._drag)
            p.setPen(QPen(QColor(theme.ACCENT if hot else "#40E080"), 2))
            r = 9 if hot else 5
            p.drawEllipse(QPoint(int(x), int(y)), r, r)
        if len(pts):
            p.setPen(QPen(QColor(255, 120, 60), 2))                    # corner 0
            p.drawEllipse(QPoint(int(pts[0, 0]), int(pts[0, 1])), 14, 14)
        moved = np.linalg.norm(d.corners - d.original, axis=1) > 0.5
        p.setPen(QPen(QColor(theme.RED), 2))
        for x, y in pts[moved]:
            p.drawEllipse(QPoint(int(x), int(y)), 11, 11)
        p.end()

    # ---- interaction ----
    def _nearest(self, x, y) -> int | None:
        pts = self._to_widget(self.dlg.corners)
        d = np.linalg.norm(pts - [x, y], axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= 18 else None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.dlg._drag = self._nearest(ev.position().x(), ev.position().y())
            self.update()

    def mouseMoveEvent(self, ev):
        d = self.dlg
        if d._drag is None:
            h = self._nearest(ev.position().x(), ev.position().y())
            if h != self._hover:
                self._hover = h
                self.update()
            return
        d.corners[d._drag] = self._to_image(ev.position().x(), ev.position().y())
        d._dirty = True
        d._refresh()

    def mouseReleaseEvent(self, _ev):
        d = self.dlg
        if d._drag is not None:
            # snap to the true corner near where it was dropped, so a rough
            # drag still lands sub-pixel -- the same refinement the detector
            # itself uses
            pt = np.array([[d.corners[d._drag]]], np.float32)
            try:
                ref = cv2.cornerSubPix(d.gray(), pt, (9, 9), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                                        40, 0.001))
                cand = np.asarray(ref, np.float64).reshape(2)
                if np.linalg.norm(cand - d.corners[d._drag]) < 12:
                    d.corners[d._drag] = cand
            except cv2.error:
                pass
            d._drag = None
            d._refresh()


class _Tile(QFrame):
    """One board in the gallery: picture, frame number, error, include tick."""

    toggled = Signal(int, bool)
    opened = Signal(int)

    def __init__(self, i: int, frame_no: int, pix: QPixmap, err: float, used: bool,
                 size: tuple[int, int] | None = None):
        super().__init__()
        self.i = i
        self._size = size
        self.setFrameShape(QFrame.StyledPanel)
        # A fixed width, or the grid stretches each tile across its cell and
        # the picture floats in a sea of panel.
        self.setFixedWidth(TILE_W + 14)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignCenter)
        self.pic.setPixmap(pix)
        self.pic.setCursor(Qt.PointingHandCursor)
        self.pic.setToolTip("Click to open this board and drag a corner")
        self.pic.mouseReleaseEvent = lambda _ev: self.opened.emit(self.i)
        lay.addWidget(self.pic)
        row = QHBoxLayout()
        self.chk = QCheckBox(f"frame {frame_no}")
        self.chk.setChecked(used)
        self.chk.setToolTip("Include this view in the calibration")
        self.chk.toggled.connect(lambda v: self.toggled.emit(self.i, bool(v)))
        row.addWidget(self.chk)
        row.addStretch(1)
        word, col = quality(err, size)
        self.err = QLabel(word)
        self.err.setStyleSheet(f"color: {col};")
        self.err.setToolTip("How far this view's corners sit from where the fitted lens "
                            f"says they should be. Under {lens.rms_limits(*(size or (1920, 1080)))[0]:.1f} px "
                            "is good.")
        row.addWidget(self.err)
        lay.addLayout(row)
        self._restyle(used)

    def set_used(self, used: bool):
        self.chk.blockSignals(True)
        self.chk.setChecked(used)
        self.chk.blockSignals(False)
        self._restyle(used)

    def set_error(self, err: float):
        word, col = quality(err, self._size)
        self.err.setText(word)
        self.err.setStyleSheet(f"color: {col};")

    def set_pixmap(self, pix: QPixmap):
        self.pic.setPixmap(pix)

    def _restyle(self, used: bool):
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {theme.ACCENT if used else theme.HAIRLINE};"
            f" border-radius: 6px; background: {theme.BG_PANEL}; }}"
            if used else
            f"QFrame {{ border: 1px dashed {theme.HAIRLINE}; border-radius: 6px;"
            f" background: {theme.BG_WINDOW}; }}")


class BoardReview(QWidget):
    """The scrollable gallery plus its controls. Owns which views are used."""

    changed = Signal()           # the chosen set or the corners moved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scan = None
        self.pattern = (9, 6)
        self.square = 0.024
        self.prof = None
        self.errors = np.zeros(0)
        self.used: list[bool] = []
        self.edited: set[int] = set()
        # bumped on every accepted corner edit, so a page can tell whether its
        # fit still describes these corners (I74)
        self.corner_rev = 0
        self.model = "auto"          # the lens model "Best spread" fits with
        self.base = None             # (G146) GoPro's lens curve for model "gopro"
        self._reader: _FrameReader | None = None
        self._reader_scan = None
        self._reading = -1           # the board whose full frame is being read
        self._tiles: list[_Tile] = []
        # how a slow call ("Best spread" fits a lens) is run: by default right here; a wizard hands
        # in its off-thread runner so the page keeps repainting (G51, R19)
        self.runner = lambda fn: fn()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.summary, 1)
        for label, slot, tip in (
                ("Best spread", self._auto, "Let Kinetrace choose: a spread over the picture, "
                                            "near and far, tilted and square-on, minus any view "
                                            "that fits badly"),
                ("All", lambda: self._set_all(True), "Use every board found"),
                ("None", lambda: self._set_all(False), "Clear the selection"),
                ("Drop the worst", self._drop_worst,
                 "Untick every view that reprojects more than three times worse than the median")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        lay.addLayout(bar)
        # How corner 0 was recognised -- the sentence that tells a first-time
        # user whether the rings SHOULD all sit in the same place on the board.
        self.orient_label = QLabel("")
        self.orient_label.setWordWrap(True)
        self.orient_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.orient_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.orient_label)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setMinimumHeight(260)
        self.host = QWidget()
        self.grid = QGridLayout(self.host)
        self.grid.setSpacing(8)
        self.area.setWidget(self.host)
        lay.addWidget(self.area, 1)

    # ------------------------------------------------------------- content
    def set_scan(self, scan, pattern, square, prof, chosen=None, errors=None, model="auto"):
        self.scan = scan
        self.pattern = pattern
        self.square = float(square)
        self.prof = prof
        self.model = model
        n = len(scan.corners) if scan is not None else 0
        self.errors = (np.asarray(errors, float) if errors is not None
                       else np.full(n, np.nan))
        if chosen is None:
            chosen = list(range(n))
        self.used = [i in set(chosen) for i in range(n)]
        self.edited.clear()
        orient = list(getattr(scan, "orient", []) or []) if scan is not None else []
        if scan is not None and len(orient) != n:          # an older scan without the tags
            orient = ["colour" if lens.board_is_asymmetric(pattern) else "image"] * n
        self.orient_label.setText(lens.orientation_summary(orient, pattern))
        self._rebuild()

    def chosen(self) -> list[int]:
        return [i for i, u in enumerate(self.used) if u]

    def set_fit(self, prof, errors, note: str = "") -> None:
        """Show `prof` and the per-view `errors` OF THAT profile on every tile
        (the one place that writes them: the review page used to reach into
        `prof`, `errors` and the tiles itself, R19)."""
        self.prof = prof
        self.errors = np.asarray(errors, float)
        for i, tile in enumerate(self._tiles):
            tile.set_error(self.errors[i])
            tile.set_pixmap(self._tile_image(i))
        self._summarise(note)

    def _tile_image(self, i: int) -> QPixmap:
        s = self.scan
        return pixmap(lens.draw_board_review(s.thumbs[i], s.corners[i], self.pattern,
                                             self.square, self.prof,
                                             scale=s.thumb_scale(i)), TILE_W)

    def _rebuild(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._tiles = []
        if self.scan is None:
            return
        per_row = self._per_row()
        for i in range(len(self.scan.corners)):
            t = _Tile(i, self.scan.frames[i], self._tile_image(i), self.errors[i], self.used[i],
                      self.scan.size)
            t.toggled.connect(self._on_toggle)
            t.opened.connect(self._edit)
            self.grid.addWidget(t, i // per_row, i % per_row, Qt.AlignTop | Qt.AlignLeft)
            self._tiles.append(t)
        # soak up the spare width on the right instead of spreading the tiles
        self.grid.setColumnStretch(per_row, 1)
        self._per = per_row
        self._summarise()

    def _per_row(self) -> int:
        # The viewport has no meaningful width until the widget has been laid
        # out, so fall back to the scroll area and then to the widget, less an
        # allowance for the vertical scrollbar.
        w = max(self.area.viewport().width(), self.area.width() - 24,
                self.width() - 24, 960)
        return max(1, int(w) // (TILE_W + 22))

    def _reflow(self) -> None:
        if not self._tiles:
            return
        per_row = self._per_row()
        if per_row == getattr(self, "_per", 0):
            return
        self.grid.setColumnStretch(getattr(self, "_per", 0), 0)
        for k, t in enumerate(self._tiles):
            self.grid.addWidget(t, k // per_row, k % per_row, Qt.AlignTop | Qt.AlignLeft)
        self.grid.setColumnStretch(per_row, 1)
        self._per = per_row

    def showEvent(self, ev):
        super().showEvent(ev)
        self._reflow()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reflow()          # re-flow when the window changes width

    # ------------------------------------------------------------ actions
    def _on_toggle(self, i: int, on: bool):
        self.used[i] = on
        self._tiles[i].set_used(on)
        self._summarise()
        self.changed.emit()

    def _set_all(self, on: bool):
        self.used = [on] * len(self.used)
        for t in self._tiles:
            t.set_used(on)
        self._summarise()
        self.changed.emit()

    def _auto(self):
        if self.scan is None:
            return
        scan, prof = self.scan, self.prof
        corners = list(scan.corners)

        def work():
            # the lens model the user chose on the video page, not always "auto"
            idx, err, why = lens.auto_select(corners, self.pattern, self.square, scan.size, self.model,
                                             base=self.base)
            if prof is not None:
                # the errors of the profile the tiles DRAW (its axes), not of the provisional fit
                # auto_select made on its way (R19)
                err = lens.per_view_errors(corners, self.pattern, self.square, prof)
            return idx, err, why
        idx, err, why = self.runner(work)
        self.errors = err
        self.used = [i in set(idx) for i in range(len(self.used))]
        for i, t in enumerate(self._tiles):
            t.set_used(self.used[i])
            t.set_error(err[i])
        self._summarise(why)
        self.changed.emit()

    def _drop_worst(self):
        e = self.errors
        if not np.isfinite(e).any():
            self._summarise("nothing to go on yet - fit once first")
            return
        limit = lens.outlier_limit(e, lens.OUTLIER_FLOOR_PICK_PX)
        for i in range(len(self.used)):
            if np.isfinite(e[i]) and e[i] > limit:
                self.used[i] = False
                self._tiles[i].set_used(False)
        self._summarise(f"dropped every view worse than {limit:.2f} px")
        self.changed.emit()

    def _edit(self, i: int):
        """Open the full-resolution frame so a corner can be dragged.

        The frame is read by a `_FrameReader` thread (I81): the GUI thread
        never owns a VideoCapture, and a long-GOP 4K seek no longer freezes
        the window. The editor opens when the frame arrives."""
        s = self.scan
        if s is None:
            return
        if not s.video:
            self._read_failed(i, "the scan does not say which video it came from")
            return
        if self._reader is not None and self._reader.isRunning():
            # one frame at a time - and say so (G138: a second click used to be dropped without a word)
            self._summarise(f"still reading frame {int(s.frames[self._reading])} - open the next board in a "
                            "moment" if 0 <= self._reading < len(s.frames) else "still reading a frame")
            return
        self._summarise(f"reading frame {int(s.frames[i])} at full resolution...")
        th = _FrameReader(i, s.video, int(s.frames[i]))
        th.frame_ready.connect(self._on_frame)
        self._reader = th
        self._reader_scan = s
        self._reading = i
        th.start()

    def stop_reader(self):
        """Wait for a frame read in flight before the page goes away (a
        QThread collected while running takes the process down). A read ends by
        itself, so the wait has no cap (I200: a 15 s cap closed over a running
        thread)."""
        th = self._reader
        if th is not None and th.isRunning():
            th.wait()

    def _read_failed(self, i: int, why: str):
        # say it: a tile click that silently does nothing reads as a dead button
        s = self.scan
        frame = int(s.frames[i]) if s is not None and i < len(s.frames) else i
        msg = f"Frame {frame} could not be opened for editing: {why}."
        self._summarise(msg)
        more = "\n\nThe corners the scan found are kept and still used."
        if s is not None and s.video and not Path(s.video).exists():
            more += " To edit them, put the video back where it was and click the board again."
        QMessageBox.warning(self, "Cannot open this board", msg + more)

    def _on_frame(self, i: int, bgr, why: str):
        s = self.scan
        if s is None or s is not self._reader_scan or i >= len(s.corners):
            return                                  # a new scan arrived meanwhile
        if not self.isVisible() or not self.isEnabled():
            # (G114) the page was left, or a fit is running with the wizard disabled: an editor opened
            # now would change corners under a fit that has already read them
            self._summarise()
            return
        if bgr is None:
            self._read_failed(i, why)
            return
        self._summarise()
        orient = list(getattr(s, "orient", []) or [])
        dlg = CornerEditor(self, bgr, s.corners[i], self.pattern, self.square, self.prof,
                           s.frames[i], orient[i] if i < len(orient) else "")
        accepted = dlg.exec() == QDialog.Accepted
        moved, new_corners = dlg.moved(), dlg.corners.copy()
        dlg.deleteLater()          # (I249) each editor kept its full-resolution frame for the session
        if not accepted or not moved:
            return
        s.corners[i] = new_corners
        self.edited.add(i)
        self.corner_rev += 1
        try:
            self.errors[i] = lens.per_view_errors([s.corners[i]], self.pattern, self.square,
                                                  self.prof)[0]
        except Exception:                                        # noqa: BLE001
            pass
        self._tiles[i].set_pixmap(self._tile_image(i))
        self._tiles[i].set_error(self.errors[i])
        self._summarise("corners edited - refit to use them")
        self.changed.emit()

    def _summarise(self, note: str = ""):
        n = len(self.used)
        k = sum(self.used)
        e = self.errors[[i for i, u in enumerate(self.used) if u]] if k else np.zeros(0)
        med = float(np.nanmedian(e)) if len(e) and np.isfinite(e).any() else float("nan")
        bits = [f"{k} of {n} boards selected"]
        if np.isfinite(med):
            bits.append(f"median {med:.2f} px")
        if self.edited:
            bits.append(f"{len(self.edited)} edited by hand")
        if note:
            bits.append(note)
        self.summary.setText("  -  ".join(bits))
