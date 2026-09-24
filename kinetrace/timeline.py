"""Timeline panel: per-point track coverage + events, Premiere-style.

This panel IS the scrubber. One lane per point shows which frames are tracked
(point color; red overlay where the model's confidence dipped) and a top lane
holds named event windows plus the frame ruler. Click or drag to seek; click
an event ribbon to jump to it.

Time zoom: +/- (pointer over the panel) or Ctrl+wheel zooms the time axis
around the cursor so a precise frame window can be selected even on a
40k-frame video; middle-drag pans the zoomed view, which also auto-follows
the playhead.

Shift+drag draws a marquee over frames AND lanes: the horizontal extent is the
frame window, the vertical extent picks what Delete removes there — the segment
lane clears silhouettes, point lanes clear those points' tracks, a drag across
both clears both in one undoable step. A drag that stays in the ruler (or
clicking an event ribbon) selects the window without naming lanes: Delete then
falls back to the points selected in the side panel, as it always did.

Vertical: the panel is resizable via the splitter above it — visible lane
count comes from the widget height; the wheel scrolls only what still
overflows.

Painting reads the session arrays directly; the per-column aggregation is
cached and keyed by (session.data_version, view range, width), so scrubbing
repaints are O(width) and tracking-time updates recompute only when a chunk
actually landed.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QFontMetrics, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QInputDialog, QMenu, QSizePolicy, QWidget


def _pop(menu: QMenu, pos):
    """exec() a context menu and release it afterwards. A shown QMenu otherwise
    lives on as a child of the widget until the widget dies: one leaked menu
    (with all its actions) per right-click over a long digitizing session.
    Tooltips on: menus built per right-click missed the window-wide setting (G15)."""
    menu.setToolTipsVisible(True)
    chosen = menu.exec(pos)
    menu.deleteLater()          # deferred: `chosen` stays valid for the caller's comparisons
    return chosen

GUTTER_W = 96      # left name column, px
EVENTS_H = 18      # events lane height, px
ANIMAL_H = 13      # animal presence lane (only when the session has an animal)
LANE_H = 13        # per-point lane height, px
STRIP_H = 6        # scroll strip at the bottom while time-zoomed
DEFAULT_LANES = 10  # lanes assumed for the initial size hint (resize for more)
MIN_SPAN = 16      # tightest time zoom, frames
TIME_ZOOM_STEP = 1.5  # time-axis zoom factor per keypress / wheel notch
LOW_CONF = 0.5     # below this a tracked span gets the warning overlay

# colors come from the shared design tokens so the painted panel and the
# styled widgets read as one surface
from kinetrace import theme as _t  # noqa: E402

BG = QColor(_t.BG_WINDOW)
LANE_BG = QColor("#2C2C33")
LANE_SEP = QColor(_t.BG_CANVAS)
RULER_FG = QColor(_t.TEXT_DIM)
PLAYHEAD = QColor(245, 245, 246)
PENDING = QColor(_t.GREEN)
LOW_CONF_OVERLAY = _t.with_alpha(_t.RED, 130)        # confidence warning
MANUAL_FILL = QColor(255, 255, 255, 235)             # hand-placed frame marks
MANUAL_EDGE = QColor(20, 20, 24, 200)
OCCLUDED_WASH = QColor(22, 22, 26, 150)              # hand-marked hidden cells (hatched)
OCCLUDED_HATCH = QColor(235, 235, 240, 140)
DISAGREE_BAND = QColor(255, 80, 220, 200)            # this camera disagrees with the others (3D)
NOTE_MARK = QColor(_t.GREEN)                         # frame notes on the events lane
NOTE_MARK_W = 7                                      # px, base of the note triangle
SELECTION = _t.with_alpha(_t.ACCENT, 55)             # frame-window band (covered lanes)
SELECTION_WEAK = _t.with_alpha(_t.ACCENT, 18)        # same window, lanes it won't touch
SELECTION_EDGE = _t.with_alpha(_t.ACCENT, 230)
STRIP_TRACK = _t.with_alpha("#FFFFFF", 22)           # scroll strip background
STRIP_THUMB = _t.with_alpha(_t.ACCENT, 160)          # scroll strip view window


class TimelinePanel(QWidget):
    seek_requested = Signal(int)
    point_selected = Signal(int)
    events_changed = Signal()        # emitted after context-menu edits
    # (start, end, pids|None): clear tracked data here; None = "lanes unspecified",
    # which lets the app fall back to the point panel's selection
    clear_requested = Signal(int, int, object)
    occlude_requested = Signal(int, int, object, bool)   # (f0, f1, pids|None, hidden on/off)
    note_requested = Signal(int)                          # open the note editor for a frame
    clear_masks_requested = Signal(int, int)  # (start, end): clear the animal's masks here
    clear_both_requested = Signal(int, int, object)  # both, as ONE undo step

    def __init__(self):
        super().__init__()
        self.session = None
        self.current = 0
        self.selected: int | None = None
        self.pending_event: int | None = None  # E pressed once: start frame
        self.sel_range: tuple[int, int] | None = None  # frame-window selection
        # which lanes the marquee covers. sel_rows is None when the gesture did
        # not name lanes (ruler drag / event ribbon) and [] when it deliberately
        # covered no point lane (a segment-lane-only drag) — the two must stay
        # distinguishable or a silhouette-only delete would clear points too.
        self.sel_rows: list[int] | None = None
        self.sel_seg = False                   # marquee covers the segment lane
        self._view = (0, 1)                    # displayed frame range, inclusive
        self._scroll = 0                       # first visible lane row
        self._laid_out_n = -1
        self._cache_key = None
        self._trk_col = None                   # (W, N) uint8, any tracked in column
        self._conf_col = None                  # (W, N) float32, min conf in column
        self._man_col = None                   # (W, N) uint8, any hand-placed frame in column
        self._occ_col = None                   # (W, N) uint8, any hand-marked hidden cell in column
        # 3D disagreement: per cell, THIS camera's reprojection error against the
        # others' (from the last reconstruction, mapped by the app), NaN = no 3D
        self._disagree: np.ndarray | None = None      # (T, N) float32
        self._dis_col = None                   # (W, N) float32, column max of the above
        self._disagree_px = 5.0                # threshold, px (set_disagreement scales it by picture width)
        self._drag_seek = False
        self._drag_select_from: int | None = None
        self._drag_select_y: float | None = None
        self._pan_last_x: float | None = None
        self._strip_drag = False
        self.setMouseTracking(True)
        self.setMinimumHeight(EVENTS_H + LANE_H + 6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # clicking the panel takes focus so Shift+± zooms even after typing in
        # a spinbox (which would otherwise swallow the keystrokes)
        self.setFocusPolicy(Qt.ClickFocus)

    # ------------------------------------------------------------------- api

    def set_disagreement(self, arr: np.ndarray | None, threshold_px: float | None = None) -> None:
        """(T, N) per-cell reprojection error of THIS camera against the others
        (NaN = no 3D), from the app after a reconstruction; cells above the
        threshold (default 5 px scaled by picture width / 1920) get a magenta
        band. None clears it."""
        self._disagree = None if arr is None else np.asarray(arr, np.float32)
        if threshold_px is not None:
            self._disagree_px = float(threshold_px)
        elif self.session is not None and getattr(self.session, "width", 0):
            self._disagree_px = 5.0 * max(1.0, self.session.width / 1920.0)
        self._cache_key = None
        self.update()

    def set_session(self, session) -> None:
        self.session = session
        self._scroll = 0
        self._cache_key = None
        self.pending_event = None
        self.clear_selection()
        self._view = (0, max(1, (session.n_frames - 1) if session is not None else 1))
        self._laid_out_n = session.n_points if session is not None else 0
        self.updateGeometry()
        self.update()

    def _set_view(self, new0: int, span: int) -> None:
        """Move the displayed frame window to [new0, new0+span], clamped to the
        video. The ONE place that assigns _view — it invalidates the column
        cache and repaints, so no caller can forget and paint stale coverage."""
        T = self.session.n_frames
        span = int(np.clip(span, 1, max(T - 1, 1)))
        new0 = int(np.clip(new0, 0, T - 1 - span))
        if (new0, new0 + span) != self._view:
            self._view = (new0, new0 + span)
            self._cache_key = None
            self.update()

    def set_current(self, frame: int) -> None:
        if frame == self.current:
            return
        self.current = frame
        # a zoomed view follows the playhead (live tracking stays visible)
        v0, v1 = self._view
        if self.session is not None and not (v0 <= frame <= v1) and self._zoomed():
            span = v1 - v0
            self._set_view(frame - span // 4, span)
        self.update()

    def set_selected(self, pid: int | None) -> None:
        self.selected = pid
        self.update()

    def set_pending_event(self, frame: int | None) -> None:
        self.pending_event = frame
        self.update()

    def clear_selection(self) -> None:
        self.sel_range = None
        self.sel_rows = None
        self.sel_seg = False
        self._drag_select_from = None
        self._drag_select_y = None
        self.update()

    def select_event_window(self, index: int) -> None:
        """Select an event's frames WITHOUT naming lanes — Delete keeps falling
        back to the point panel's selection, which is what it has always done."""
        e = self.session.events[index]
        self.sel_range = (e.start, e.end)
        self.sel_rows = None
        self.sel_seg = False
        self.update()

    def select_all_lanes(self) -> None:
        """Extend the current selection over every lane (context menu)."""
        if self.sel_range is None or self.session is None:
            return
        self.sel_rows = list(range(self.session.n_points))
        self.sel_seg = self._animal_h() > 0
        self.update()

    def selection_targets(self) -> tuple[list[int] | None, bool]:
        """(point rows, segment) the current selection would delete. `None` rows
        means the gesture left the lanes unspecified."""
        return self.sel_rows, self.sel_seg

    def request_delete_selection(self) -> bool:
        """Delete key / menu default: ask the app to clear exactly the lanes the
        marquee covers. Returns False when there is nothing selected. The ONE
        place that maps a selection to a signal, so the key and the menu can
        never drift apart."""
        if self.sel_range is None:
            return False
        f0, f1 = self.sel_range
        self._emit_default_clear(f0, f1)
        return True

    def _emit_default_clear(self, f0: int, f1: int) -> None:
        rows, seg = self.sel_rows, self.sel_seg
        if seg and rows:
            self.clear_both_requested.emit(f0, f1, rows)
        elif seg:
            self.clear_masks_requested.emit(f0, f1)
        else:
            self.clear_requested.emit(f0, f1, rows if rows else None)

    def refresh(self) -> None:
        n = self.session.n_points if self.session is not None else 0
        if n != self._laid_out_n:
            # lane count changed: the layout caches sizeHint, so a bare
            # update() would leave new lanes clipped below the widget
            self._laid_out_n = n
            self._scroll = int(np.clip(self._scroll, 0,
                                       max(0, n - self._max_rows())))
            self.updateGeometry()
        self.update()

    def sizeHint(self):
        n = self.session.n_points if self.session is not None else 1
        h = EVENTS_H + self._animal_h() + LANE_H * max(1, min(n, DEFAULT_LANES)) + 6
        return QSize(400, h)

    def minimumSizeHint(self):
        return QSize(200, EVENTS_H + LANE_H + 6)

    # -------------------------------------------------------------- geometry

    def _animal_h(self) -> int:
        return ANIMAL_H if (self.session is not None and self.session.animal is not None) else 0

    def _lanes_y0(self) -> int:
        return EVENTS_H + 2 + self._animal_h()

    def _zoomed(self) -> bool:
        if self.session is None:
            return False
        return (self._view[1] - self._view[0]) < self.session.n_frames - 1

    def _max_rows(self) -> int:
        """Visible lane capacity from the CURRENT height (panel is resizable)."""
        return max(1, (self.height() - EVENTS_H - self._animal_h() - 6) // LANE_H)

    def _lane_w(self) -> int:
        return max(1, self.width() - GUTTER_W - 4)

    # Every frame of the view owns a CELL of lane_w / (v1 - v0 + 1) pixels, the
    # same cells the coverage is painted in. The mapping used to treat frames
    # as points spread over (v1 - v0) intervals, so when zoomed in up to half a
    # painted cell clicked / selected the NEIGHBOUR frame (I49).
    def _frame_at(self, x: float) -> int:
        v0, v1 = self._view
        cells = max(v1 - v0 + 1, 1)
        t = v0 + np.floor((x - GUTTER_W) * cells / self._lane_w())
        return int(np.clip(t, 0, self.session.n_frames - 1))

    def _x_of(self, frame: int) -> float:
        """Centre of `frame`'s cell (playhead, ticks, note marks)."""
        return self._x_edge(frame + 0.5)

    def _x_edge(self, frame: float) -> float:
        """Left edge of `frame`'s cell (`frame + 1` = its right edge)."""
        v0, v1 = self._view
        return GUTTER_W + (frame - v0) / max(v1 - v0 + 1, 1) * self._lane_w()

    def _in_animal_lane(self, y: float) -> bool:
        return self._animal_h() > 0 and EVENTS_H + 2 <= y < EVENTS_H + 2 + ANIMAL_H

    def _row_at(self, y: float) -> int | None:
        if y < self._lanes_y0() or self.session is None:
            return None
        row = int((y - self._lanes_y0()) // LANE_H) + self._scroll
        # bound to the rows actually drawn — the widget can be taller than the
        # lane stack, and that strip must not select an off-screen point
        return row if row in self._visible_rows() else None

    def _visible_rows(self) -> range:
        n = self.session.n_points if self.session is not None else 0
        return range(self._scroll, min(n, self._scroll + self._max_rows()))

    def _row_rect_y(self, row: int) -> tuple[int, int]:
        """Vertical extent of a point lane, in viewport px."""
        y0 = self._lanes_y0() + (row - self._scroll) * LANE_H
        return y0, y0 + LANE_H

    def _lanes_in_band(self, ya: float, yb: float) -> tuple[list[int] | None, bool]:
        """Which lanes a marquee spanning [ya, yb] vertically covers, as
        (point rows, segment lane). A band that never leaves the ruler returns
        (None, False): the frame window is selected but no lane is named, so
        Delete falls back to the point panel's selection.

        `[]` rows with segment True is the silhouette-only case and must stay
        distinct from None."""
        top, bot = (min(ya, yb), max(ya, yb))
        if bot < EVENTS_H:                      # ruler-only drag
            return None, False
        seg = self._animal_h() > 0 and top < EVENTS_H + 2 + ANIMAL_H and bot >= EVENTS_H + 2
        rows = []
        for r in self._visible_rows():
            ry0, ry1 = self._row_rect_y(r)
            if top < ry1 and bot >= ry0:
                rows.append(r)
        if not rows and not seg:
            # dropped in the empty strip below the last lane: don't make the
            # gesture a no-op, fall back to the unspecified selection
            return None, False
        return rows, seg

    def _sel_band_y(self) -> tuple[float, float]:
        """Vertical extent to paint the selection band over: the covered lanes,
        or the whole panel when no lane was named."""
        rows, seg = self.sel_rows, self.sel_seg
        if rows is None:
            return 0.0, float(self.height())
        tops, bots = [], []
        if seg:
            tops.append(EVENTS_H + 2)
            bots.append(EVENTS_H + 2 + ANIMAL_H)
        for r in rows:
            if r in self._visible_rows():
                y0, y1 = self._row_rect_y(r)
                tops.append(y0)
                bots.append(y1)
        if not tops:                            # covered lanes scrolled away
            return 0.0, float(self.height())
        return float(min(tops)), float(max(bots))

    def _sel_hint(self) -> str:
        """What Delete will do, for the band label and the tooltip."""
        rows, seg = self.sel_rows, self.sel_seg
        if rows is None:
            return "Del clears selected points"
        n = self.session.n_points if self.session is not None else 0
        rows = [r for r in rows if r < n]   # a deleted point can leave a stale row
        parts = []
        if seg:
            parts.append("silhouettes")
        if len(rows) == 1:
            parts.append(self.session.points[rows[0]].name)
        elif rows:
            parts.append(f"{len(rows)} points")
        return "Del clears " + (" + ".join(parts) if parts else "selected points")

    def _note_at(self, x: float, y: float) -> int | None:
        """Frame of the note mark under the cursor (events lane, lower edge)."""
        if self.session is None or not self.session.notes or not (EVENTS_H - 9 <= y <= EVENTS_H):
            return None
        best, bd = None, 5.0
        for nf in self.session.notes:
            d = abs(self._x_of(nf) - x)
            if d <= bd:
                best, bd = int(nf), d
        return best

    def _event_at(self, x: float, y: float) -> int | None:
        """Hit test in PIXEL space against the drawn ribbon: at 40k frames a
        short event's frame range covers well under a pixel, but its ribbon is
        drawn ≥3 px wide and must be clickable as drawn."""
        if y >= EVENTS_H or self.session is None:
            return None
        for i in reversed(range(len(self.session.events))):  # topmost drawn last
            e = self.session.events[i]
            x0 = self._x_edge(e.start)
            x1 = max(self._x_edge(e.end + 1), x0 + 3)
            if x0 - 1 <= x <= x1 + 1:
                return i
        return None

    # ------------------------------------------------------------- time zoom

    def zoom_time(self, factor: float, anchor_x: float | None = None) -> None:
        """Zoom the time axis by `factor` (>1 = in), anchored at `anchor_x`
        (viewport px; defaults to the playhead)."""
        if self.session is None:
            return
        T = self.session.n_frames
        v0, v1 = self._view
        span = v1 - v0
        new_span = int(np.clip(round(span / factor), MIN_SPAN, max(T - 1, 1)))
        if new_span == span:
            return
        if anchor_x is None:
            anchor_x = self._x_of(self.current)
        anchor_f = self._frame_at(anchor_x)
        frac = float(np.clip((anchor_x - GUTTER_W) / self._lane_w(), 0.0, 1.0))
        self._set_view(round(anchor_f - frac * new_span), new_span)

    def zoom_fit(self) -> None:
        """Show the whole video (undo any time zoom)."""
        if self.session is None:
            return
        self._set_view(0, max(1, self.session.n_frames - 1))

    def zoom_time_keyboard(self, factor: float) -> None:
        """Shift + +/-: anchor at the pointer when it is over the panel,
        else at the playhead."""
        x = None
        if self.underMouse():
            x = float(self.mapFromGlobal(QCursor.pos()).x())
        self.zoom_time(factor, x)

    def keyPressEvent(self, ev):
        # the panel handles its own zoom keys when focused (a focused spinbox
        # elsewhere can no longer swallow them); everything else propagates
        if ev.modifiers() == Qt.ShiftModifier and ev.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_time_keyboard(TIME_ZOOM_STEP)
            ev.accept()
            return
        if ev.modifiers() == Qt.ShiftModifier and ev.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_time_keyboard(1 / TIME_ZOOM_STEP)
            ev.accept()
            return
        super().keyPressEvent(ev)

    @staticmethod
    def _tick_step(span: int, width_px: int, target_px: int = 90) -> int:
        """Premiere-style adaptive ruler: the major tick step is the smallest
        1/2/5 x 10^k frame count that keeps ticks ~target_px apart."""
        raw = max(1.0, span * target_px / max(width_px, 1))
        mag = 10 ** int(np.floor(np.log10(raw)))
        for mult in (1, 2, 5, 10):
            if mag * mult >= raw:
                return max(1, int(mag * mult))
        return int(mag * 10)

    def _strip_to(self, x: float) -> None:
        """Center the zoomed view window on the strip position under `x`."""
        v0, v1 = self._view
        span = v1 - v0
        center = (x - GUTTER_W) / self._lane_w() * (self.session.n_frames - 1)
        self._set_view(round(center - span / 2), span)

    def _pan_time(self, dx_px: float) -> None:
        v0, v1 = self._view
        span = v1 - v0
        df = int(round(-dx_px / self._lane_w() * span))
        if df:
            self._set_view(v0 + df, span)

    # ------------------------------------------------------------- aggregate

    def _ensure_columns(self) -> None:
        s = self.session
        w = self._lane_w()
        v0, v1 = self._view
        key = (id(s), s.data_version, w, s.n_points, v0, v1, s.animal is not None)
        if key == self._cache_key:
            return
        N = s.n_points
        Tv = v1 - v0 + 1
        # column c covers view frames [c*Tv/W, (c+1)*Tv/W)
        bounds = np.minimum((np.arange(w, dtype=np.int64) * Tv) // w, Tv - 1)
        bounds = np.maximum.accumulate(bounds)  # reduceat needs non-decreasing
        if s.masks is not None and s.animal is not None:
            present = (s.masks.area[v0:v1 + 1] > 0)
            self._ani_col = np.maximum.reduceat(present.astype(np.uint8), bounds, axis=0)
            sc = np.where(present, np.nan_to_num(s.masks.score[v0:v1 + 1], nan=99.0), 99.0)
            self._ani_low_col = np.minimum.reduceat(sc.astype(np.float32), bounds, axis=0) < 2.0
        else:
            self._ani_col = None
            self._ani_low_col = None
        if N == 0:
            self._trk_col = np.zeros((w, 0), np.uint8)
            self._conf_col = np.ones((w, 0), np.float32)
            self._man_col = np.zeros((w, 0), np.uint8)
            self._occ_col = np.zeros((w, 0), np.uint8)
            self._cache_key = key
            return
        trk = s.tracked[v0:v1 + 1]
        self._trk_col = np.maximum.reduceat(trk.astype(np.uint8), bounds, axis=0)
        conf_masked = np.where(trk, s.confidence[v0:v1 + 1], 1.0).astype(np.float32)
        self._conf_col = np.minimum.reduceat(conf_masked, bounds, axis=0)
        man = (trk & s.manual[v0:v1 + 1]).astype(np.uint8)
        self._man_col = np.maximum.reduceat(man, bounds, axis=0)
        occ = s.occluded[v0:v1 + 1].astype(np.uint8)
        self._occ_col = np.maximum.reduceat(occ, bounds, axis=0)
        d = self._disagree
        if d is not None and d.shape == (s.n_frames, N):
            dd = np.nan_to_num(d[v0:v1 + 1], nan=0.0).astype(np.float32)
            self._dis_col = np.maximum.reduceat(dd, bounds, axis=0)
        else:
            self._dis_col = None
        self._cache_key = key

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """[(start, end_exclusive)] of True runs in a 1-D bool array."""
        if not mask.any():
            return []
        d = np.diff(mask.astype(np.int8), prepend=0, append=0)
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0]
        return list(zip(starts.tolist(), ends.tolist()))

    # ----------------------------------------------------------------- paint

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        s = self.session
        if s is None or s.n_frames <= 0:
            p.setPen(RULER_FG)
            p.drawText(self.rect(), Qt.AlignCenter, "timeline — open a video")
            p.end()
            return
        self._ensure_columns()
        w = self._lane_w()
        v0, v1 = self._view
        span = max(v1 - v0, 1)
        fm = QFontMetrics(self.font())

        # ---- events lane (doubles as the frame ruler) ----
        p.fillRect(QRect(GUTTER_W, 0, w + 4, EVENTS_H), LANE_BG.darker(115))
        p.setPen(RULER_FG)
        gutter_note = f"events ×{(s.n_frames - 1) / span:.0f}" if self._zoomed() else "events"
        p.drawText(QRect(2, 0, GUTTER_W - 6, EVENTS_H), Qt.AlignVCenter | Qt.AlignRight,
                   gutter_note)
        step = self._tick_step(span, w)
        minor = step // 5 if step >= 5 and (step / 5) / span * w >= 10 else 0
        p.setPen(QPen(RULER_FG.darker(170), 1))
        if minor:
            first_m = ((v0 + minor - 1) // minor) * minor
            for f in range(first_m, v1 + 1, minor):
                x = self._x_of(f)
                p.drawLine(QPointF(x, EVENTS_H - 2), QPointF(x, EVENTS_H))
        p.setPen(QPen(RULER_FG.darker(130), 1))
        first_tick = ((v0 + step - 1) // step) * step
        for f in range(first_tick, v1 + 1, step):
            x = self._x_of(f)
            p.drawLine(QPointF(x, EVENTS_H - 5), QPointF(x, EVENTS_H))
            if f + step <= v1:  # skip a label crowding the right edge
                p.drawText(QPointF(x + 2, EVENTS_H - 5), f"{f:,}")
        for e in s.events:
            if e.end < v0 or e.start > v1:
                continue
            x0, x1 = self._x_edge(e.start), self._x_edge(e.end + 1)
            rect = QRectF(max(x0, GUTTER_W), 2,
                          max(min(x1, GUTTER_W + w) - max(x0, GUTTER_W), 3), EVENTS_H - 5)
            c = QColor(*e.color, 170)
            p.setPen(QPen(QColor(*e.color), 1))
            p.setBrush(QBrush(c))
            p.drawRoundedRect(rect, 2, 2)
            if rect.width() > 18:
                p.setPen(QColor(15, 15, 15))
                p.drawText(rect.adjusted(3, 0, -2, 0), Qt.AlignVCenter,
                           fm.elidedText(e.name, Qt.ElideRight, int(rect.width()) - 5))
        # frame notes: a small triangle on the lower edge of the events lane
        if s.notes:
            p.setPen(QPen(QColor(15, 15, 15), 1))
            p.setBrush(QBrush(NOTE_MARK))
            hw = NOTE_MARK_W / 2
            for nf in s.notes:
                if v0 <= nf <= v1:
                    x = self._x_of(nf)
                    p.drawPolygon(QPolygonF([QPointF(x, EVENTS_H - 8), QPointF(x + hw, EVENTS_H - 1),
                                             QPointF(x - hw, EVENTS_H - 1)]))
        if self.pending_event is not None and v0 <= self.pending_event <= v1:
            x = self._x_of(self.pending_event)
            p.setPen(QPen(PENDING, 2))
            p.drawLine(QPointF(x, 0), QPointF(x, EVENTS_H))
            p.setBrush(QBrush(PENDING))
            p.drawPolygon(QPolygonF([QPointF(x, 2), QPointF(x + 8, 6), QPointF(x, 10)]))

        # ---- animal lane: where the silhouette exists (dim red = low presence score) ----
        y = EVENTS_H + 2
        if self._animal_h():
            lane = QRect(GUTTER_W, y, w + 4, ANIMAL_H - 1)
            p.fillRect(lane, LANE_BG.darker(108))
            col = QColor(*s.animal.color)
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(QRect(4, y + (ANIMAL_H - 8) // 2, 8, 8))
            p.setPen(QColor(210, 210, 215))
            p.drawText(QRect(16, y, GUTTER_W - 20, ANIMAL_H), Qt.AlignVCenter,
                       fm.elidedText(s.animal.name, Qt.ElideRight, GUTTER_W - 22))
            if self._ani_col is not None:
                pres = self._ani_col.astype(bool)
                p.setPen(Qt.NoPen)
                for c0, c1 in self._runs(pres):
                    p.fillRect(QRectF(GUTTER_W + c0, y + 2, max(c1 - c0, 1), ANIMAL_H - 5),
                               QColor(*s.animal.color, 200))
                for c0, c1 in self._runs(pres & self._ani_low_col):
                    p.fillRect(QRectF(GUTTER_W + c0, y + 2, max(c1 - c0, 1), ANIMAL_H - 5),
                               LOW_CONF_OVERLAY)
            p.setPen(QPen(LANE_SEP, 1))
            p.drawLine(lane.bottomLeft(), lane.bottomRight())
            y += ANIMAL_H

        # ---- point lanes ----
        for row in self._visible_rows():
            meta = s.points[row]
            lane = QRect(GUTTER_W, y, w + 4, LANE_H - 1)
            p.fillRect(lane, LANE_BG.lighter(112) if row == self.selected else LANE_BG)
            p.setBrush(QBrush(QColor(*meta.color)))
            p.setPen(Qt.NoPen)
            p.drawRect(QRect(4, y + (LANE_H - 8) // 2, 8, 8))
            f = self.font()
            f.setBold(row == self.selected)
            p.setFont(f)
            p.setPen(QColor(210, 210, 215) if meta.display else QColor(120, 120, 125))
            p.drawText(QRect(16, y, GUTTER_W - 20, LANE_H), Qt.AlignVCenter,
                       fm.elidedText(meta.name, Qt.ElideRight, GUTTER_W - 22))
            f.setBold(False)
            p.setFont(f)
            trk = self._trk_col[:, row].astype(bool)
            col = QColor(*meta.color, 235 if meta.display else 90)
            p.setPen(Qt.NoPen)
            # the selected point's lane draws its coverage bar thicker — the
            # panel selection and the lane read as one thing at a glance
            by, bh = (y + 1, LANE_H - 3) if row == self.selected else (y + 2, LANE_H - 5)
            for c0, c1 in self._runs(trk):
                p.fillRect(QRectF(GUTTER_W + c0, by, max(c1 - c0, 1), bh), col)
            low = trk & (self._conf_col[:, row] < LOW_CONF)
            for c0, c1 in self._runs(low):
                p.fillRect(QRectF(GUTTER_W + c0, by, max(c1 - c0, 1), bh),
                           LOW_CONF_OVERLAY)
            # hand-marked hidden cells: washed out + hatched — the data is
            # still there, but it will not be exported
            occ_run = self._occ_col[:, row].astype(bool)
            for c0, c1 in self._runs(occ_run):
                r_ = QRectF(GUTTER_W + c0, by, max(c1 - c0, 1), bh)
                p.fillRect(r_, OCCLUDED_WASH)
                p.fillRect(r_, QBrush(OCCLUDED_HATCH, Qt.BDiagPattern))
            # 3D disagreement: where THIS camera's position for the landmark does
            # not agree with the other cameras' (reprojection error above the
            # threshold) -- a magenta band along the bottom of the lane, so a
            # camera that slid onto the wrong body part shows up frame by frame
            if self._dis_col is not None and row < self._dis_col.shape[1]:
                bad = trk & (self._dis_col[:, row] > self._disagree_px)
                for c0, c1 in self._runs(bad):
                    p.fillRect(QRectF(GUTTER_W + c0, by + bh * 0.6, max(c1 - c0, 1), bh * 0.4), DISAGREE_BAND)
            # hand-placed frames: a small diamond on the lane (Shift+< / Shift+>
            # walk to the first / last one)
            man_cols = np.nonzero(self._man_col[:, row])[0]
            if len(man_cols):
                p.setPen(QPen(MANUAL_EDGE, 1))
                p.setBrush(QBrush(MANUAL_FILL))
                cy = by + bh / 2
                r = min(4.0, bh / 2)
                for c in man_cols.tolist():
                    cx = GUTTER_W + c + 0.5
                    p.drawPolygon(QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                                             QPointF(cx, cy + r), QPointF(cx - r, cy)]))
                p.setPen(Qt.NoPen)
            p.setPen(QPen(LANE_SEP, 1))
            p.drawLine(lane.bottomLeft(), lane.bottomRight())
            y += LANE_H
        if s.n_points > self._max_rows():
            p.setPen(RULER_FG)
            p.drawText(QRect(2, self.height() - 14, GUTTER_W - 6, 12),
                       Qt.AlignRight, f"{self._scroll + 1}–"
                       f"{min(s.n_points, self._scroll + self._max_rows())}/{s.n_points}")

        # ---- frame-window selection overlay ----
        if self.sel_range is not None:
            f0, f1 = self.sel_range
            if f1 >= v0 and f0 <= v1:
                x0 = max(self._x_edge(max(f0, v0)), GUTTER_W)
                x1 = min(self._x_edge(min(f1, v1) + 1), GUTTER_W + w)
                by0, by1 = self._sel_band_y()
                # the lanes the marquee covers are washed in full; the rest of
                # the window only gets a faint tint, so "what Delete will touch"
                # is visible without losing the frame range
                banded = by0 > 0 or by1 < self.height()
                if banded:
                    p.fillRect(QRectF(x0, 0, max(x1 - x0, 2), self.height()),
                               SELECTION_WEAK)
                p.fillRect(QRectF(x0, by0, max(x1 - x0, 2), max(by1 - by0, 2)), SELECTION)
                p.setPen(QPen(SELECTION_EDGE, 1.5))
                p.drawLine(QPointF(x0, 0), QPointF(x0, self.height()))
                p.drawLine(QPointF(x1, 0), QPointF(x1, self.height()))
                if banded:
                    # close the box horizontally too: the wash alone is nearly
                    # invisible on a lane whose point color is already the
                    # accent blue, and "which lanes" must never be a guess
                    p.drawLine(QPointF(x0, by0), QPointF(x1, by0))
                    p.drawLine(QPointF(x0, by1), QPointF(x1, by1))
                p.setPen(QColor(230, 240, 255))
                p.drawText(QPointF(x0 + 4, self.height() - 4),
                           f"{f0}–{f1} ({f1 - f0 + 1} frames) — {self._sel_hint()}")

        # ---- playhead ----
        if v0 <= self.current <= v1:
            x = self._x_of(self.current)
            p.setPen(QPen(PLAYHEAD, 1.5))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setBrush(QBrush(PLAYHEAD))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x - 4, 0), QPointF(x + 4, 0), QPointF(x, 6)]))

        # ---- scroll strip (only while zoomed): where the view window sits ----
        if self._zoomed():
            sy = self.height() - STRIP_H - 1
            T = max(s.n_frames - 1, 1)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(STRIP_TRACK))
            p.drawRoundedRect(QRectF(GUTTER_W, sy, w, STRIP_H), 3, 3)
            tx0 = GUTTER_W + v0 / T * w
            tx1 = GUTTER_W + v1 / T * w
            p.setBrush(QBrush(STRIP_THUMB))
            p.drawRoundedRect(QRectF(tx0, sy, max(tx1 - tx0, 20), STRIP_H), 3, 3)
        p.end()

    # ----------------------------------------------------------------- mouse

    def mousePressEvent(self, ev):
        if self.session is None:
            return
        x, y = ev.position().x(), ev.position().y()
        if ev.button() == Qt.MiddleButton:
            self._pan_last_x = x
            return
        if ev.button() == Qt.LeftButton:
            if (self._zoomed() and x >= GUTTER_W
                    and y >= self.height() - STRIP_H - 3):
                self._strip_drag = True     # scroll strip: jump + drag the view
                self._strip_to(x)
                return
            if ev.modifiers() & Qt.ShiftModifier and x >= GUTTER_W:
                f = self._frame_at(x)
                self._drag_select_from = f
                self._drag_select_y = y
                self.sel_range = (f, f)
                self.sel_rows, self.sel_seg = self._lanes_in_band(y, y)
                self.update()
                return
            if x < GUTTER_W:
                row = self._row_at(y)
                if row is not None:
                    self.point_selected.emit(row)
                return
            hit = self._event_at(x, y)
            if hit is not None:
                # clicking an event SELECTS its window (Delete then clears the
                # panel-selected points inside it) and jumps to its start
                self.select_event_window(hit)
                self.seek_requested.emit(self.session.events[hit].start)
                return
            self._drag_seek = True
            self.seek_requested.emit(self._frame_at(x))

    def mouseMoveEvent(self, ev):
        if self.session is None:
            return
        x, y = ev.position().x(), ev.position().y()
        if self._strip_drag and (ev.buttons() & Qt.LeftButton):
            self._strip_to(x)
            return
        if self._pan_last_x is not None and (ev.buttons() & Qt.MiddleButton):
            self._pan_time(x - self._pan_last_x)
            self._pan_last_x = x
            return
        if self._drag_select_from is not None and (ev.buttons() & Qt.LeftButton):
            f = self._frame_at(x)
            self.sel_range = (min(self._drag_select_from, f),
                              max(self._drag_select_from, f))
            self.sel_rows, self.sel_seg = self._lanes_in_band(self._drag_select_y, y)
            self.update()
            return
        if self._drag_seek and (ev.buttons() & Qt.LeftButton):
            self.seek_requested.emit(self._frame_at(x))
            return
        # hover tooltip: frame + timestamp + what's under the cursor
        if x >= GUTTER_W:
            f = self._frame_at(x)
            ts = f" · {f / self.session.fps:.2f}s" if self.session.fps else ""
            hit = self._event_at(x, y)
            note_f = self._note_at(x, y)
            if note_f is not None:
                n = self.session.notes[note_f]
                who = f" — {n.get('author')}" if n.get("author") else ""
                self.setToolTip(f"note at frame {note_f}{who}: {n.get('text', '')}\n"
                                "(right-click to edit or delete)")
            elif hit is not None:
                e = self.session.events[hit]
                same = [i for i, ev2 in enumerate(self.session.events)
                        if ev2.name == e.name]
                occ = (f" (occurrence {same.index(hit) + 1} of {len(same)})"
                       if len(same) > 1 else "")
                note = f"\n{e.note}" if e.note else ""
                who = f" — {e.author}" if e.author else ""
                self.setToolTip(f"{e.name}: frames {e.start}–{e.end}{occ}{who}{note}\n— click to "
                                "select + jump, right-click for options")
            elif self._in_animal_lane(y):
                m = self.session.masks
                if m is not None and m.has(f):
                    self.setToolTip(f"frame {f}{ts} · {self.session.animal.name}: silhouette "
                                    f"{int(m.area[f])} px², score {m.score[f]:.1f} — "
                                    "Shift+drag along this lane + Delete removes the "
                                    "silhouettes there")
                else:
                    self.setToolTip(f"frame {f}{ts} · {self.session.animal.name}: no silhouette "
                                    "here (press S and click it, then Track)")
            else:
                row = self._row_at(y)
                if row is not None and self.session.occluded[f, row]:
                    self.setToolTip(f"frame {f}{ts} · {self.session.points[row].name}: marked "
                                    "HIDDEN by hand — kept, but not exported and not used for 3D "
                                    "(Shift+X or the right-click menu unmarks it)")
                elif row is not None and self.session.tracked[f, row]:
                    c = self.session.confidence[f, row]
                    src = " (from silhouette)" if self.session.points[row].derived else ""
                    hand = " · placed by hand" if self.session.manual[f, row] else ""
                    dis = ""
                    if self._disagree is not None and f < self._disagree.shape[0] \
                            and row < self._disagree.shape[1] and np.isfinite(self._disagree[f, row]):
                        e = float(self._disagree[f, row])
                        if e > self._disagree_px:     # the magenta band (I52)
                            dis = (f"\nMAGENTA: in 3D this camera disagrees with the others by {e:.1f} px "
                                   f"(limit {self._disagree_px:.1f}) — the point has probably slid onto "
                                   "another spot here. Right-click it on the video → Snap to the other "
                                   "cameras' rays, or 3D → Re-track Disagreeing Stretches.")
                        else:
                            dis = f" · agrees with the other cameras ({e:.1f} px)"
                    self.setToolTip(f"frame {f}{ts} · {self.session.points[row].name}{src} "
                                    f"· conf {c:.2f}{hand}{dis}" + ("" if dis.startswith("\n") else
                                    " — Shift+drag along this lane + Delete removes its track there"))
                else:
                    self.setToolTip(f"frame {f}{ts} · Shift+drag across frames AND lanes: "
                                    "select, then Delete clears those lanes there "
                                    "· Shift+± / Ctrl+wheel: zoom time")
        else:
            self.setToolTip("")

    def mouseReleaseEvent(self, ev):
        self._drag_seek = False
        self._drag_select_from = None
        self._drag_select_y = None
        self._pan_last_x = None
        self._strip_drag = False

    def wheelEvent(self, ev):
        if self.session is None:
            ev.accept()
            return
        if ev.modifiers() & Qt.ControlModifier:
            factor = TIME_ZOOM_STEP if ev.angleDelta().y() > 0 else 1 / TIME_ZOOM_STEP
            self.zoom_time(factor, ev.position().x())
        elif self.session.n_points > self._max_rows():
            step = -1 if ev.angleDelta().y() > 0 else 1
            self._scroll = int(np.clip(self._scroll + step, 0,
                                       self.session.n_points - self._max_rows()))
            self.update()
        ev.accept()

    def contextMenuEvent(self, ev):
        if self.session is None:
            return
        x, y = ev.pos().x(), ev.pos().y()
        # inside the frame-window selection: offer the bulk clear
        if self.sel_range is not None and x >= GUTTER_W:
            f = self._frame_at(x)
            f0, f1 = self.sel_range
            if f0 <= f <= f1:
                rows, seg = self.sel_rows, self.sel_seg
                has_animal = self.session.animal is not None
                menu = QMenu(self)
                # the marquee's own target comes first and carries the (Del)
                # badge, so the menu and the key never disagree
                what = self._sel_hint().replace("Del clears ", "")
                act_default = menu.addAction(f"Clear {what} in frames {f0}–{f1} (Del)")
                menu.addSeparator()
                act_pts = menu.addAction(
                    f"Clear tracked points in frames {f0}–{f1}"
                    + ("" if rows else " (panel selection)"))
                act_masks = None
                act_both = None
                if has_animal:
                    act_masks = menu.addAction(
                        f"Clear {self.session.animal.name}'s silhouettes in frames {f0}–{f1}")
                    act_both = menu.addAction(f"Clear both in frames {f0}–{f1}")
                menu.addSeparator()
                act_hide = menu.addAction(
                    f"Mark HIDDEN in frames {f0}–{f1} (kept, not exported)"
                    + ("" if rows else " — panel selection"))
                act_unhide = menu.addAction(f"Unmark hidden in frames {f0}–{f1}")
                menu.addSeparator()
                act_all = menu.addAction("Extend selection to every lane")
                act_cancel = menu.addAction("Cancel selection (Esc)")
                chosen = _pop(menu, ev.globalPos())
                if chosen == act_hide:
                    self.occlude_requested.emit(f0, f1, rows if rows else None, True)
                elif chosen == act_unhide:
                    self.occlude_requested.emit(f0, f1, rows if rows else None, False)
                elif chosen == act_default:
                    self._emit_default_clear(f0, f1)
                elif chosen == act_pts:
                    self.clear_requested.emit(f0, f1, rows if rows else None)
                elif act_masks is not None and chosen == act_masks:
                    self.clear_masks_requested.emit(f0, f1)
                elif act_both is not None and chosen == act_both:
                    self.clear_both_requested.emit(f0, f1, rows if rows else None)
                elif chosen == act_all:
                    self.select_all_lanes()
                elif chosen == act_cancel:
                    self.clear_selection()
                return
        note_f = self._note_at(x, y)
        if note_f is not None:
            menu = QMenu(self)
            act_edit = menu.addAction(f"Edit note at frame {note_f}…")
            act_go = menu.addAction(f"Jump to frame {note_f}")
            act_del = menu.addAction("Delete note")
            chosen = _pop(menu, ev.globalPos())
            if chosen == act_edit:
                self.note_requested.emit(note_f)
            elif chosen == act_go:
                self.seek_requested.emit(note_f)
            elif chosen == act_del:
                self.session.set_note(note_f, "")
                self.events_changed.emit()
            self.update()
            return
        hit = self._event_at(x, y)
        if hit is None:
            if x >= GUTTER_W and y < EVENTS_H:
                f = self._frame_at(x)
                menu = QMenu(self)
                act_note = menu.addAction(f"Add a note at frame {f}… (Shift+N at the playhead)")
                if _pop(menu, ev.globalPos()) == act_note:
                    self.note_requested.emit(f)
            return
        e = self.session.events[hit]
        menu = QMenu(self)
        act_jump_s = menu.addAction(f"Jump to start ({e.start})")
        act_jump_e = menu.addAction(f"Jump to end ({e.end})")
        menu.addSeparator()
        act_select = menu.addAction("Select this window (for bulk clear)")
        menu.addSeparator()
        act_rename = menu.addAction("Rename event…")
        act_note = menu.addAction("Edit note…" if not e.note else f"Edit note… ({e.note[:40]})")
        act_start = menu.addAction("Set start to current frame")
        act_end = menu.addAction("Set end to current frame")
        menu.addSeparator()
        act_del = menu.addAction("Delete event")
        chosen = _pop(menu, ev.globalPos())
        if chosen == act_jump_s:
            self.seek_requested.emit(e.start)
        elif chosen == act_jump_e:
            self.seek_requested.emit(e.end)
        elif chosen == act_select:
            self.select_event_window(hit)
        elif chosen == act_rename:
            name, ok = QInputDialog.getText(self, "Rename event", "Name:", text=e.name)
            if ok and name.strip():
                self.session.update_event(hit, name=name.strip())
                self.events_changed.emit()
        elif chosen == act_note:
            text, ok = QInputDialog.getMultiLineText(
                self, "Event note", f"Note for \u201c{e.name}\u201d (frames {e.start}\u2013{e.end}):", e.note)
            if ok:
                self.session.update_event(hit, note=text)
                self.events_changed.emit()
        elif chosen == act_start:
            self.session.update_event(hit, start=self.current)
            self.events_changed.emit()
        elif chosen == act_end:
            self.session.update_event(hit, end=self.current)
            self.events_changed.emit()
        elif chosen == act_del:
            self.session.remove_event(hit)
            self.events_changed.emit()
        self.update()
