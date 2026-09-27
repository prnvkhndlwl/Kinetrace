"""Small reusable UI pieces: toast notifications and the settings dialog.

Toast: a non-blocking message that floats over the video for a few seconds
— for things the user must not miss (auto-pause, a model download, an error)
without a modal dialog stealing focus mid-work. The status bar keeps the
transient chatter; toasts are for the important stuff.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSlider, QTextBrowser, QToolButton, QVBoxLayout,
                               QWidget)

from kinetrace import theme

LEVEL_RANK = {"info": 0, "success": 1, "warn": 2, "error": 3}
LEVEL_COLORS = {
    "info": (theme.BG_PANEL, theme.TEXT, theme.ACCENT),
    "success": (theme.BG_PANEL, theme.TEXT, theme.GREEN),
    "warn": (theme.BG_PANEL, theme.TEXT, "#FFB340"),
    "error": (theme.BG_PANEL, theme.TEXT, theme.RED),
}


class Toast(QLabel):
    """Floating notification anchored to the top-center of `host`."""

    def __init__(self, host: QWidget):
        super().__init__(host)
        self._host = host
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("click to dismiss")
        self._level = "info"
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        host.installEventFilter(self)

    def show_message(self, text: str, level: str = "info", ms: int = 6000) -> None:
        # a second notice while one is still up used to REPLACE it, so the
        # first (often the verdict) was never read (G10): keep the latest two
        prev = self.text() if (self.isVisible() and self._timer.isActive()) else ""
        if prev and text not in prev.split("\n\n"):
            text = prev.split("\n\n")[-1] + "\n\n" + text
            ms = max(ms, self._timer.remainingTime())
            if LEVEL_RANK.get(self._level, 0) > LEVEL_RANK.get(level, 0):
                level = self._level
        self._level = level
        bg, fg, edge = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
        self.setStyleSheet(
            f"QLabel {{ background: {bg}; color: {fg}; border: 1px solid {theme.HAIRLINE};"
            f" border-left: 4px solid {edge}; border-radius: 6px; padding: 8px 14px;"
            f" font-size: 10pt; }}")
        self.setText(text)
        self._place()
        self.show()
        self.raise_()
        self._timer.start(max(1500, int(ms)))

    def mousePressEvent(self, ev):
        self.hide()

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if obj is self._host and ev.type() == QEvent.Resize and self.isVisible():
            self._place()
        return False

    def _place(self) -> None:
        w = min(max(320, self._host.width() - 80), 760)
        self.setFixedWidth(w)
        self.adjustSize()
        self.move((self._host.width() - w) // 2, 16)


class _Spinner(QWidget):
    """A turning arc in the accent colour: 'working, not frozen'."""

    def __init__(self, parent=None, size: int = 34):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._angle = (self._angle + 12) % 360
        self.update()

    def showEvent(self, ev):                 # noqa: N802 - Qt name
        self._timer.start()
        super().showEvent(ev)

    def hideEvent(self, ev):                 # noqa: N802 - Qt name
        self._timer.stop()
        super().hideEvent(ev)

    def paintEvent(self, ev):                # noqa: N802 - Qt name
        from PySide6.QtGui import QColor, QPainter, QPen
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(4, 4, -4, -4)
        p.setPen(QPen(QColor(theme.HAIRLINE), 3.5))
        p.drawEllipse(r)
        pen = QPen(QColor(theme.ACCENT), 3.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(r, -self._angle * 16, 100 * 16)
        p.end()


class LoadingOverlay(QWidget):
    """A card over the whole window while videos / a project open (owner,
    2026-09-26: the wait "seems like the app is frozen"): a turning spinner,
    what is being opened, the step it is on, a progress bar (counted when the
    number of steps is known, moving otherwise), a helpful hint, and Cancel where
    stopping is safe. Clicks on the window underneath are blocked (the app also
    disables its menus and hotkeys meanwhile). It appears after `DELAY_MS`, so an
    open that takes a blink does not flash."""

    DELAY_MS = 150

    def __init__(self, host: QWidget):
        from PySide6.QtWidgets import QFrame, QProgressBar
        super().__init__(host)
        self._host = host
        self._busy = False
        self._passive = False
        self.setAttribute(Qt.WA_NoMousePropagation, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._card = QFrame(self)
        self._card.setObjectName("loadingCard")
        self._card.setStyleSheet(
            f"#loadingCard {{ background: {theme.BG_PANEL}; border: 1px solid {theme.HAIRLINE};"
            f" border-radius: 10px; }}")
        lay = QVBoxLayout(self._card)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(14)
        self.spinner = _Spinner(self._card)
        top.addWidget(self.spinner, 0, Qt.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(4)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setStyleSheet(f"color: {theme.TEXT}; font-size: 11pt; font-weight: 600;")
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"color: {theme.TEXT};")
        texts.addWidget(self.title)
        texts.addWidget(self.detail)
        top.addLayout(texts, 1)
        lay.addLayout(top)
        self.bar = QProgressBar(self._card)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        lay.addWidget(self.bar)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {theme.TEXT_DIM}; font-style: italic;")
        lay.addWidget(self.hint)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFocusPolicy(Qt.NoFocus)
        self.btn_cancel.clicked.connect(self._on_cancel)
        row.addWidget(self.btn_cancel)
        lay.addLayout(row)
        self._cancel_cb = None
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.timeout.connect(self._appear)
        host.installEventFilter(self)
        self.hide()

    # ------------------------------------------------------------------ API

    def start(self, title: str, detail: str = "", total: int = 0, hint: str = "",
              on_cancel=None, immediate: bool = False) -> None:
        """Begin (or re-title) a wait. `on_cancel`: a callable -> Cancel is shown."""
        self._busy = True
        self.title.setText(title)
        self.step(detail, 0, total)
        self.set_hint(hint)
        self._cancel_cb = on_cancel
        self.btn_cancel.setVisible(on_cancel is not None)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("Cancel")
        if self.isVisible():
            return
        if immediate:
            self._appear()
            self.repaint()
        elif not self._delay.isActive():
            self._delay.start(self.DELAY_MS)

    def step(self, detail: str | None = None, value: int | None = None, total: int | None = None) -> None:
        if detail is not None:
            self.detail.setText(detail)
        if total is not None:
            self.bar.setRange(0, max(0, int(total)))       # 0 = a moving bar: length unknown
        if value is not None and self.bar.maximum() > 0:
            self.bar.setValue(int(value))
        if self.isVisible():
            self._place()                    # a longer line makes the card taller: keep it centred

    def set_hint(self, text: str) -> None:
        self.hint.setText(text or "")
        self.hint.setVisible(bool(text))
        if self.isVisible():
            self._place()

    def set_cancel(self, on_cancel) -> None:
        self._cancel_cb = on_cancel
        self.btn_cancel.setVisible(on_cancel is not None)

    def set_passive(self, on: bool) -> None:
        """The last, harmless part of a wait (the first picture): the card stays,
        lighter, and clicks go through to the window."""
        self._passive = bool(on)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, self._passive)
        if self._passive:
            self.btn_cancel.setVisible(False)
        self.update()

    def finish(self) -> None:
        self._busy = False
        self._delay.stop()
        self._cancel_cb = None
        self.set_passive(False)
        self.hide()

    def is_busy(self) -> bool:
        return self._busy

    # ------------------------------------------------------------ internals

    def _on_cancel(self) -> None:
        cb = self._cancel_cb
        if cb is None:
            return
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Cancelling…")
        cb()

    def _appear(self) -> None:
        if not self._busy:
            return
        self._place()
        self.show()
        self.raise_()

    def _place(self) -> None:
        mb = self._host.menuWidget() if hasattr(self._host, "menuWidget") else None
        top = mb.height() if mb is not None and mb.isVisible() else 0
        self.setGeometry(0, top, self._host.width(), max(0, self._host.height() - top))
        w = min(560, max(320, self.width() - 80))
        self._card.setFixedWidth(w)
        self._card.adjustSize()
        self._card.move((self.width() - w) // 2, max(20, (self.height() - self._card.height()) // 2 - 40))

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if obj is self._host and ev.type() == QEvent.Resize and self.isVisible():
            self._place()
        return False

    def paintEvent(self, ev):                # noqa: N802 - Qt name
        from PySide6.QtGui import QColor, QPainter
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(10, 10, 12, 50 if self._passive else 150))
        p.end()

    def mousePressEvent(self, ev):           # noqa: N802 - the window underneath is busy
        ev.accept()

    def mouseReleaseEvent(self, ev):         # noqa: N802
        ev.accept()

    def mouseDoubleClickEvent(self, ev):     # noqa: N802
        ev.accept()

    def wheelEvent(self, ev):                # noqa: N802
        ev.accept()


class ElidedLabel(QLabel):
    """A one-line label that shows the START of its text and ends in "…" when
    it does not fit, with the whole text as its tooltip. `text()` returns the
    whole text; `pad` = horizontal padding the stylesheet adds. (A plain
    right-aligned QLabel with an Ignored width clipped the onboarding hint's
    first words — the ones saying what to press — G5.)"""

    def __init__(self, text: str = "", pad: int = 0):
        super().__init__()
        self._full = ""
        self._pad = int(pad)
        self.setText(text)

    def text(self) -> str:                   # noqa: D102 - the full text, never the elided one
        return self._full

    def setText(self, text: str) -> None:    # noqa: N802 - Qt name
        self._full = text or ""
        self.setToolTip(self._full)
        self._elide()

    def resizeEvent(self, ev):               # noqa: N802 - Qt name
        super().resizeEvent(ev)
        self._elide()

    def sizeHint(self):                      # noqa: N802 - Qt name
        """The WHOLE text's width (the elided text would shrink the hint and the
        label could never widen again)."""
        h = super().sizeHint()
        m = self.contentsMargins()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + m.left() + m.right() + self._pad + 2,
                     h.height())

    def minimumSizeHint(self):               # noqa: N802 - Qt name
        """A few characters: a long text must never widen the window (G8)."""
        h = super().minimumSizeHint()
        return QSize(min(self.sizeHint().width(), self.fontMetrics().horizontalAdvance("frame 0…") + self._pad),
                     h.height())

    def _elide(self) -> None:
        m = self.contentsMargins()
        room = max(0, self.width() - m.left() - m.right() - self._pad)
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight, room) if room else ""
        QLabel.setText(self, shown if shown else self._full)


class OnboardingStrip(QWidget):
    """The four-step path for a new project — Open, Segment, Skeleton, Track — as a
    slim strip above the video. Done steps get a check, the next step is the
    accent, each chip is clickable, and the strip hides itself once a project
    has tracked data (View menu brings it back)."""

    def __init__(self, on_step):
        super().__init__()
        self._on_step = on_step
        self.setObjectName("onboarding")
        self.setStyleSheet(
            f"#onboarding {{ background: {theme.BG_WINDOW}; border-bottom: 1px solid {theme.HAIRLINE}; }}"
            f"QToolButton {{ border: none; padding: 4px 10px; color: {theme.TEXT_DIM}; }}"
            f"QToolButton[state='next'] {{ color: {theme.TEXT}; background: {theme.BG_PANEL};"
            f" border: 1px solid {theme.ACCENT}; border-radius: 6px; }}"
            f"QToolButton[state='done'] {{ color: {theme.TEXT_DIM}; }}"
            f"QLabel {{ color: {theme.TEXT_DIM}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 6, 4)
        lay.setSpacing(6)
        self._chips: list[QToolButton] = []
        self._labels = ["1  Open video  (Ctrl+O)", "2  Segment the animal  (S, optional)",
                        "3  Skeleton / landmarks  (optional)", "4  Track"]
        from PySide6.QtCore import QSize
        for i, text in enumerate(self._labels):
            b = QToolButton()
            b.setText(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setIconSize(QSize(16, 16))
            b.setFixedHeight(28)          # state changes (icon, border) must never move the layout
            b.clicked.connect(lambda _=False, k=i: self._on_step(k))
            b.setIcon(self._blank_icon())  # the check icon's room, from the start (G8)
            lay.addWidget(b)
            self._chips.append(b)
            if i < len(self._labels) - 1:
                arrow = QLabel("›")
                lay.addWidget(arrow)
        # the hint is guidance, NOT a fifth step: a rule separates it from the
        # numbered chips and it sits hard right, against the close button
        from PySide6.QtWidgets import QFrame, QSizePolicy
        lay.addSpacing(6)
        rule = QFrame()
        rule.setFrameShape(QFrame.VLine)
        rule.setStyleSheet(f"color: {theme.HAIRLINE};")
        rule.setFixedHeight(18)
        lay.addWidget(rule)
        self.hint = ElidedLabel("", pad=16)      # the stylesheet's 8 px padding each side
        self.hint.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.hint.setStyleSheet(f"color: {theme.TEXT_DIM}; padding: 0 8px;")
        # a single-line QLabel dictates the window minimum width (see CLAUDE.md):
        # ignore its width so a longer hint never resizes the window or the canvas
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.hint, 1)
        self.btn_close = QToolButton()
        self.btn_close.setText("×")
        self.btn_close.setToolTip("Hide this strip (View → Getting started strip brings it back)")
        self.btn_close.clicked.connect(self.hide)
        lay.addWidget(self.btn_close)
        self.setFixedHeight(28 + 8)       # a resize would refit a canvas the user has not zoomed
        self._last_state: tuple | None = None

    @staticmethod
    def _blank_icon() -> QIcon:
        from PySide6.QtGui import QPixmap
        pm = QPixmap(16, 16)
        pm.fill(Qt.transparent)
        return QIcon(pm)

    def set_state(self, done: list[bool], hint: str = "") -> None:
        from kinetrace import icons
        # segmenting and the skeleton are both optional (an animal a few pixels
        # across has no silhouette to outline): after opening a video, Track
        # is the next step, whatever else has been done
        nxt = 0 if not done[0] else (3 if not done[3] else None)
        key = (tuple(bool(d) for d in done), nxt)
        if key != self._last_state:
            self._last_state = key
            for i, b in enumerate(self._chips):
                state = "done" if done[i] else ("next" if i == nxt else "pending")
                b.setProperty("state", state)
                # an empty icon of the same size when not done: the ✓ must not make a
                # chip wider, or the strip's minimum (and the window) grows (G8)
                b.setIcon(icons.check() if done[i] else self._blank_icon())
                b.style().unpolish(b)
                b.style().polish(b)
        # "ⓘ" marks it as advice rather than the next numbered step
        text = f"ⓘ  {hint}" if hint else ""
        if text != self.hint.text():
            self.hint.setText(text)


MANUAL_PATH = Path(__file__).resolve().parent.parent / "docs" / "MANUAL.md"


def _slug(text: str) -> str:
    """GitHub's heading anchor rule, so the manual's own contents links work
    inside the dialog exactly as they do on disk."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    return "".join(out).strip("-")


class ManualDialog(QDialog):
    """The user manual, in the app.

    Reads `docs/MANUAL.md` and renders it with Qt's GitHub-markdown reader, so
    the file on disk stays the single source — there is no second copy to drift.
    A contents list down the left jumps to any section, the manual's own
    contents links work, and there is a find box, because a manual you have to
    scroll blindly through is one nobody reads.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kinetrace — user manual")
        self.resize(1000, 760)
        lay = QHBoxLayout(self)

        self.contents = QListWidget()
        self.contents.setMaximumWidth(250)
        self.contents.setToolTip("Jump to a section")
        self.contents.currentRowChanged.connect(self._jump)
        lay.addWidget(self.contents)

        right = QVBoxLayout()
        find_row = QHBoxLayout()
        self.find = QLineEdit()
        self.find.setPlaceholderText("Find in the manual…  (Enter for the next match)")
        self.find.returnPressed.connect(self._find_next)
        find_row.addWidget(self.find, 1)
        btn = QPushButton("Find")
        btn.clicked.connect(self._find_next)
        find_row.addWidget(btn)
        right.addLayout(find_row)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setOpenLinks(False)        # in-document jumps are handled here
        self.view.anchorClicked.connect(self._on_anchor)
        right.addWidget(self.view, 1)
        lay.addLayout(right, 1)

        self._anchors: dict[str, int] = {}   # heading slug -> character position
        self._load()

    def _load(self) -> None:
        try:
            text = MANUAL_PATH.read_text(encoding="utf-8")
        except OSError:
            self.view.setPlainText(
                f"The manual file could not be read:\n{MANUAL_PATH}\n\n"
                "It ships in the docs folder next to the program.")
            return
        self.view.setMarkdown(text)
        self._index_headings()

    def _index_headings(self) -> None:
        """Walk the rendered document for headings: one pass builds both the
        contents list and the anchor map."""
        self.contents.blockSignals(True)
        self.contents.clear()
        doc = self.view.document()
        block = doc.begin()
        while block.isValid():
            level = block.blockFormat().headingLevel()
            title = block.text().strip()
            if level and title:
                self._anchors.setdefault(_slug(title), block.position())
                if level <= 2:
                    item = QListWidgetItem(title if level == 1 else f"   {title}")
                    item.setData(Qt.UserRole, block.position())
                    self.contents.addItem(item)
            block = block.next()
        self.contents.blockSignals(False)

    def _scroll_to(self, position: int) -> None:
        cur = self.view.textCursor()
        cur.setPosition(position)
        self.view.setTextCursor(cur)
        # put the heading at the TOP of the view, not just barely on screen
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.value() + self.view.cursorRect().top())

    def _jump(self, row: int) -> None:
        item = self.contents.item(row)
        if item is not None:
            self._scroll_to(int(item.data(Qt.UserRole)))

    def _on_anchor(self, url) -> None:
        frag = url.fragment() or url.toString().lstrip("#")
        pos = self._anchors.get(_slug(frag))
        if pos is not None:
            self._scroll_to(pos)

    def _find_next(self) -> None:
        needle = self.find.text().strip()
        if not needle:
            return
        if not self.view.find(needle):       # wrap around to the top
            cur = self.view.textCursor()
            cur.movePosition(QTextCursor.Start)
            self.view.setTextCursor(cur)
            self.view.find(needle)


class SettingsDialog(QDialog):
    """Segmentation backend, Hugging Face token (for gated SAM 3), overlay opacity."""

    def __init__(self, parent, backend: str, opacity: float, token_present: bool, statuses: dict,
                 annotator: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.annotator = QLineEdit(annotator)
        self.annotator.setPlaceholderText("your name or initials")
        self.annotator.setToolTip("Recorded on every event and note you add, and in the exports, "
                                  "so a lab can tell who digitized what.")
        form.addRow("Annotator (your name)", self.annotator)
        self.backend = QComboBox()
        for key, (label, status, note) in statuses.items():
            self.backend.addItem(f"{label}   [{status}]", key)
            self.backend.setItemData(self.backend.count() - 1, note, Qt.ToolTipRole)
        idx = self.backend.findData(backend)
        self.backend.setCurrentIndex(max(idx, 0))
        self.backend.currentIndexChanged.connect(self._on_backend)
        form.addRow("Segmentation model", self.backend)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        form.addRow("", self.note)

        tok_row = QWidget()
        tl = QHBoxLayout(tok_row)
        tl.setContentsMargins(0, 0, 0, 0)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("token already stored" if token_present else "hf_… (only needed for SAM 3)")
        self.btn_token = QPushButton("Save token")
        tl.addWidget(self.token, 1)
        tl.addWidget(self.btn_token)
        form.addRow("Hugging Face token", tok_row)
        hint = QLabel("SAM 3 weights are gated: request access at huggingface.co/facebook/sam3, "
                      "create a read token in your Hugging Face settings, paste it here. It is "
                      "stored inside this folder (models/hf/token) and sent only to Hugging Face, "
                      "when the weights download.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        form.addRow("", hint)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 90)
        self.opacity.setValue(int(round(opacity * 100)))
        self.opacity.setToolTip("Silhouette overlay opacity")
        form.addRow("Mask opacity", self.opacity)
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._statuses = statuses
        self._on_backend()

    def _on_backend(self):
        key = self.backend.currentData()
        self.note.setText(self._statuses.get(key, ("", "", ""))[2])

    def chosen_backend(self) -> str:
        return str(self.backend.currentData())

    def chosen_opacity(self) -> float:
        return self.opacity.value() / 100.0

    def chosen_annotator(self) -> str:
        return self.annotator.text().strip()
