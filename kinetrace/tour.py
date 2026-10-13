"""The interface tour (G183, owner 2026-10-12: "the interface is very feature rich and the user is lost on how
to start. A quick interactive tour will be helpful with the user having the option to skip it").

A spotlight over the REAL window: everything is dimmed except one area at a time, and a card beside it says
what that area is for, in plain words, with Back / Next / Skip tour (keys: Right, Enter or Space = Next,
Left = Back, Esc = Skip). `MainWindow._start_tour` opens it (Help -> Take the Interface Tour, or the offer
at the first start, `offer_tour`); on an empty window it first opens the demo video (kinetrace/demo.py).

The tour READS the window and changes nothing in it (owner: "don't change any options selected by the user.
don't change the current state of the user"): a panel the user hid or floated is not shown or docked -- its
stop says how to bring it back instead. While the tour is up `MainWindow.eventFilter` holds every key, click
and shortcut aimed at the window back and hands the keys to `TourOverlay.key`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from kinetrace import theme

OFFERED_KEY = "tour/offered"     # settings.ini: absent = offer at the next start; "done" / "never" = not again
PAD = 4                          # the lit area reaches this far past its widget


@dataclass
class Stop:
    """One stop: `widget(win)` is the area it lights (None = no area: the card sits in the middle), or
    `rect(win)` a rectangle in window coordinates (the Help menu's title). `hidden(win)` is said instead of
    lighting the area when the user has hidden it or moved it to its own window."""
    key: str
    title: str
    text: str
    widget: Callable = lambda win: None
    rect: Callable | None = None
    hidden: Callable = lambda win: ""


def _panel_hidden(win) -> str:
    if win.dock.isFloating() and win.dock.isVisible():
        return ("You have moved this panel into its own window: it is the separate window titled "
                "<i>Layers</i>. Drag its title bar onto the right edge of this window to put it back. "
                "(View → Layers panel, Ctrl+1, shows or hides it.)")
    if win.dock.isVisible():          # the panel is there; this part of it shows once a video is open
        return "It appears in the panel on the right once a video is open."
    return "It is hidden now: <b>View → Layers panel</b> (Ctrl+1) shows it."


def _menu_bar(win):
    """The menu bar to light, or None when it is the system's own (macOS: at the top of the screen, outside
    the window -- the card then speaks of it without lighting anything)."""
    mb = win.menuBar()
    return None if mb.isNativeMenuBar() else mb


def _help_title_rect(win) -> QRect | None:
    mb = _menu_bar(win)
    if mb is None:
        return None
    r = mb.actionGeometry(win.m_help.menuAction())
    if r.isEmpty():
        return None
    return QRect(mb.mapTo(win, r.topLeft()), r.size())


def stops(win) -> list[Stop]:
    """The tour, in the order a new user meets the window (mirrors the manual's section 5)."""
    return [
        Stop("welcome", "Welcome to Kinetrace",
             "This short tour shows the parts of the window and what each is for — about a minute. "
             "<br><br><b>→</b> or <b>Enter</b> = next, <b>←</b> = back, <b>Esc</b> = leave the tour. "
             "Nothing you have set up is changed. Help → Take the Interface Tour shows it again."),
        Stop("menus", "The menus",
             "<b>File</b> opens a video or a project, saves and exports. <b>Edit</b> undoes. <b>View</b> shows "
             "and hides panels. <b>Skeleton</b> has ready-made landmark lists, <b>3D</b> is for several "
             "cameras, <b>Body</b> for people's joints. <b>Hover over any menu entry</b> and a line says "
             "what it does.",
             widget=_menu_bar),
        Stop("strip", "Getting started",
             "The four steps of a first project: <b>1 Open video → 2 Segment → 3 Skeleton → 4 Track</b>. "
             "Steps 2 and 3 are optional. Click a step to do it; the text on the right says what to do next.",
             widget=lambda win: win.onboarding,
             hidden=lambda win: "It is hidden now: <b>View → Getting started strip</b> shows it."),
        Stop("video", "The video",
             "Your video. <b>Click</b> places the point selected in LAYERS on this frame. <b>Mouse wheel</b> "
             "zooms, <b>drag</b> pans, <b>R</b> fits the whole picture back. With several cameras they share "
             "this space; the outlined one is the camera you work in.",
             widget=lambda win: win.grid),
        Stop("timeline", "The timeline",
             "One lane per point, showing on which frames it is tracked, corrected or missing. "
             "<b>Click or drag along it to move through the video</b> — it is the only scrubber. "
             "Drag the line above it to give it more room.",
             widget=lambda win: win.timeline),
        Stop("controls", "The control bar",
             "The frame box and ⏴ ▶ ⏵ (play is a preview only: nothing is tracked). Then the tools: "
             "<b>Point (N)</b> adds a point where you click next, <b>Segment (S)</b> outlines an animal, and "
             "the toggles beside them. Hover over any button for what it does.",
             widget=lambda win: win._controls),
        Stop("track", "Track",
             "<b>Track</b> follows the selected points (and silhouettes) forward from this frame, shown live. "
             "<b>X</b> or <b>Space</b> pauses at once: drag nothing — just click where a point should be to "
             "correct it, then Track again.",
             widget=lambda win: win.btn_track),
        Stop("cameras", "Cameras",
             "The cameras of this project. <b>+ Add video</b> adds another camera's video of the same moment; "
             "the 3D menu then syncs and calibrates them so positions come out in 3D.",
             widget=lambda win: win.cameras, hidden=_panel_hidden),
        Stop("layers", "LAYERS",
             "What is tracked: each <b>animal</b> (its silhouette) with its <b>points</b> under it, and Scene "
             "points that belong to no animal. <b>+ Animal</b>, <b>+ Point</b>, <b>Skeleton ▾</b> for named "
             "landmarks. Select rows to choose what Track follows.",
             widget=lambda win: win.layers, hidden=_panel_hidden),
        Stop("status", "The status bar",
             "The frame you are on, the last tracked frame, and the device tracking runs on "
             "(a graphics card is much faster).",
             widget=lambda win: win.statusBar()),
        Stop("help", "Help",
             "<b>F1</b> opens the full manual, written for someone new to tracking. Help also has the keyboard "
             "reference, a check of what this computer can run, and this tour again. You are ready: "
             "open a video with <b>File → Open Video</b> (Ctrl+O).",
             rect=_help_title_rect),
    ]


class TourOverlay(QWidget):
    """The dimmed window with one area lit and the card. A child of the main window covering all of it,
    menu bar included; it follows the window's size. `finished(completed)` when it closes: True after Done,
    False after Skip / Esc."""

    finished = Signal(bool)

    def __init__(self, win, tour_stops: list[Stop]):
        super().__init__(win)
        self._win = win
        self.stops = tour_stops
        self.index = 0
        self._hole: QRect | None = None
        self.setObjectName("tourOverlay")
        self.setAttribute(Qt.WA_NoMousePropagation)
        self.setFocusPolicy(Qt.NoFocus)
        self.card = QFrame(self)
        self.card.setObjectName("tourCard")
        self.card.setStyleSheet(
            f"#tourCard {{ background: {theme.BG_PANEL}; border: 1px solid {theme.HAIRLINE};"
            f" border-radius: 10px; }}")
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(20, 16, 20, 14)
        lay.setSpacing(8)
        self.counter = QLabel("")
        self.counter.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.title = QLabel("")
        self.title.setStyleSheet(f"color: {theme.TEXT}; font-size: 11pt; font-weight: 600;")
        self.title.setWordWrap(True)
        self.body = QLabel("")
        self.body.setStyleSheet(f"color: {theme.TEXT};")
        self.body.setTextFormat(Qt.RichText)
        self.body.setWordWrap(True)
        for w in (self.counter, self.title, self.body):
            lay.addWidget(w)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_skip = QPushButton("Skip tour")
        self.btn_skip.setToolTip("Leave the tour (Esc). Help → Take the Interface Tour shows it again")
        self.btn_back = QPushButton("Back")
        self.btn_next = QPushButton("Next")
        for b in (self.btn_skip, self.btn_back, self.btn_next):
            b.setFocusPolicy(Qt.NoFocus)        # the keys reach the tour through the window's event filter
        self.btn_next.setStyleSheet(f"QPushButton {{ background: {theme.ACCENT}; color: white;"
                                    f" border-radius: 6px; padding: 4px 14px; }}")
        row.addWidget(self.btn_skip)
        row.addStretch(1)
        row.addWidget(self.btn_back)
        row.addWidget(self.btn_next)
        lay.addLayout(row)
        self.btn_skip.clicked.connect(self.skip)
        self.btn_back.clicked.connect(self.back)
        self.btn_next.clicked.connect(self.next)
        win.installEventFilter(self)
        self.hide()

    # ---------------------------------------------------------------- steps
    def start(self) -> None:
        self.setGeometry(self._win.rect())
        self.show()
        self.raise_()
        self._go(0)

    def _go(self, i: int) -> None:
        self.index = max(0, min(i, len(self.stops) - 1))
        st = self.stops[self.index]
        self._hole = self.target_rect()
        text = st.text
        if self._hole is None and st.widget(self._win) is not None:
            text += "<br><br>" + st.hidden(self._win)
        self.counter.setText(f"{self.index + 1} / {len(self.stops)}")
        self.title.setText(st.title)
        self.body.setText(text)
        self.btn_back.setEnabled(self.index > 0)
        self.btn_next.setText("Done" if self.index == len(self.stops) - 1 else "Next")
        self._place_card()
        self.update()

    def target_rect(self) -> QRect | None:
        """The current stop's area in window coordinates, or None (no area, or the user hid it or moved it to
        its own window -- never shown or docked by the tour)."""
        st = self.stops[self.index]
        if st.rect is not None:
            return st.rect(self._win)
        w = st.widget(self._win)
        if w is None or not w.isVisible() or w.window() is not self._win:
            return None
        return QRect(w.mapTo(self._win, QPoint(0, 0)), w.size())

    def next(self) -> None:
        if self.index >= len(self.stops) - 1:
            self._close(True)
        else:
            self._go(self.index + 1)

    def back(self) -> None:
        if self.index > 0:
            self._go(self.index - 1)

    def skip(self) -> None:
        self._close(False)

    def key(self, ev) -> None:
        """A key pressed anywhere in the window while the tour is up."""
        k = ev.key()
        if k in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.next()
        elif k in (Qt.Key_Left, Qt.Key_Backspace):
            self.back()
        elif k == Qt.Key_Escape:
            self.skip()

    def _close(self, completed: bool) -> None:
        self._win.removeEventFilter(self)
        self.hide()
        try:
            self.finished.emit(completed)
        finally:
            self.setParent(None)        # out of the window at once (deleteLater waits for the event loop)
            self.deleteLater()

    # ---------------------------------------------------------------- geometry
    def eventFilter(self, obj, ev):
        if obj is self._win and ev.type() == QEvent.Resize and self.isVisible():
            self.setGeometry(self._win.rect())
            self._hole = self.target_rect()
            self._place_card()
            self.update()
        return False

    def _place_card(self) -> None:
        W, H = self.width(), self.height()
        cw = max(300, min(420, W // 3))
        self.card.setFixedWidth(cw)
        self.card.adjustSize()
        ch = self.card.height()
        hole, gap, m = self._hole, 12, 8
        if hole is None:
            x, y = (W - cw) // 2, (H - ch) // 2
        else:
            r = hole.adjusted(-PAD, -PAD, PAD, PAD)
            if r.bottom() + gap + ch <= H - m:
                x, y = r.left(), r.bottom() + gap
            elif r.top() - gap - ch >= m:
                x, y = r.left(), r.top() - gap - ch
            elif r.right() + gap + cw <= W - m:
                x, y = r.right() + gap, r.top()
            elif r.left() - gap - cw >= m:
                x, y = r.left() - gap - cw, r.top()
            else:                                    # a large area (the video): in its upper-right corner, off the
                x, y = r.right() - 16 - cw, r.top() + 16          # middle and off the demo's animal at frame 99
        self.card.move(max(m, min(x, W - cw - m)), max(m, min(y, H - ch - m)))
        self.card.raise_()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        if self._hole is not None:
            r = QRectF(self._hole.adjusted(-PAD, -PAD, PAD, PAD))
            path.addRoundedRect(r, 6, 6)
            path.setFillRule(Qt.OddEvenFill)
        p.fillPath(path, theme.with_alpha("#0A0A0C", 170))
        if self._hole is not None:
            p.setPen(QPen(theme.with_alpha(theme.ACCENT, 255), 2))
            p.drawRoundedRect(QRectF(self._hole.adjusted(-PAD, -PAD, PAD, PAD)), 6, 6)

    def mousePressEvent(self, ev):
        ev.accept()

    def mouseReleaseEvent(self, ev):
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        ev.accept()

    def wheelEvent(self, ev):
        ev.accept()


# ---------------------------------------------------------------- the first-start offer
def offer_due(settings) -> bool:
    """True until the user took the tour, skipped it, or said Don't ask again."""
    try:
        return not str(settings.value(OFFERED_KEY, "") or "")
    except Exception:      # noqa: BLE001 - unreadable settings: do not nag
        return False


def offer_tour(win) -> str:
    """The pop-up at the first start: "take", "later" (Not now, or the box closed: asked again at the next
    start) or "never" (Don't ask again)."""
    box = QMessageBox(win)
    box.setWindowTitle("Welcome to Kinetrace")
    box.setIcon(QMessageBox.Question)
    box.setTextFormat(Qt.RichText)
    box.setText("<p><b>New to Kinetrace? Take a one-minute tour of the window.</b></p>"
                "<p>It lights up each part of the window in turn and says what it is for. If no video is "
                "open, it opens a short demo video so every panel has something in it. You can leave it at "
                "any moment (Esc), and <b>Help → Take the Interface Tour</b> shows it again later.</p>")
    take = box.addButton("Take the tour", QMessageBox.AcceptRole)
    later = box.addButton("Not now", QMessageBox.RejectRole)
    never = box.addButton("Don't ask again", QMessageBox.DestructiveRole)
    box.setDefaultButton(take)
    box.setEscapeButton(later)
    box.exec()
    clicked = box.clickedButton()
    box.deleteLater()
    if clicked is take:
        return "take"
    if clicked is never:
        return "never"
    return "later"
