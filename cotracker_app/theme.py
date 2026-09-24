"""App-wide visual theme: a quiet, content-first dark design language.

Goals (in order): the video is the hero and all chrome recedes; ONE accent
color means active/selected/primary everywhere; consistent 6 px radii and a
4/8 px spacing grid; hierarchy from type and spacing, not boxes. The canvas
and timeline import these constants so custom painting and widget styling
can never drift apart.

Purely visual — applying (or not applying) the theme changes no behavior,
which is why the offscreen tests run with it on.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPalette, QPen, QPixmap

# ---- palette (single source of truth) ----
ACCENT = "#0A84FF"          # systemBlue (dark variant): active / selected / primary
ACCENT_HOVER = "#3395FF"
RED = "#FF453A"             # warnings (low confidence)
GREEN = "#30D158"           # event pending mark
BG_CANVAS = "#161619"       # behind the video: darkest, so frames pop
BG_WINDOW = "#1E1E23"       # window chrome
BG_PANEL = "#28282E"        # raised surfaces (menus, list, dialogs)
BG_INPUT = "#323238"        # editable fields
HAIRLINE = "#3A3A42"        # 1 px separators and borders
TEXT = "#E8E8EC"
TEXT_DIM = "#98989F"        # captions, hints, secondary labels
TEXT_DISABLED = "#5C5C64"

def with_alpha(hex_color: str, alpha: int) -> QColor:
    """A theme token at a given alpha, for painted overlays. Keeps the canvas
    and timeline tied to the palette — retuning a token here updates them too."""
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


STYLESHEET = f"""
QMainWindow, QDialog, QMessageBox {{ background: {BG_WINDOW}; }}
QWidget {{ color: {TEXT}; }}
QLabel {{ background: transparent; }}

/* --- buttons: borderless glass; blue tint = ON; one filled primary --- */
QToolButton, QPushButton {{
    background: transparent; border: 1px solid transparent; border-radius: 6px;
    padding: 4px 10px; color: {TEXT}; min-height: 18px;
}}
QToolButton[compact="true"] {{ padding: 4px 5px; }}
QToolButton[bar="true"] {{ padding: 4px 5px; }}
QToolButton:hover, QPushButton:hover {{ background: rgba(255,255,255,0.07); }}
QToolButton:pressed, QPushButton:pressed {{ background: rgba(255,255,255,0.13); }}
QToolButton:checked {{
    background: rgba(10,132,255,0.24); color: #CFE5FF;
    border-color: rgba(10,132,255,0.38);
}}
QToolButton:disabled, QPushButton:disabled {{ color: {TEXT_DISABLED}; }}
QToolButton#primary {{
    background: {ACCENT}; color: white; font-weight: 600; padding: 5px 14px;
}}
QToolButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QToolButton#primary:pressed {{ background: #0873DD; }}
QToolButton#primary:disabled {{ background: #2E2E36; color: {TEXT_DISABLED}; }}
QToolButton#primary::menu-button {{ border: none; width: 16px; }}

/* --- inputs (QAbstractSpinBox covers QSpinBox AND QDoubleSpinBox) --- */
QAbstractSpinBox, QLineEdit {{
    background: {BG_INPUT}; border: 1px solid {HAIRLINE}; border-radius: 6px;
    padding: 3px 6px; selection-background-color: {ACCENT}; selection-color: white;
}}
QAbstractSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QAbstractSpinBox:disabled {{ color: {TEXT_DISABLED}; }}
/* the step buttons: a visible column with a hairline, arrows drawn by
   SpinArrowStyle (a styled sub-control loses Fusion's own arrow glyphs) */
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 16px; border: none;
    border-left: 1px solid {HAIRLINE}; background: rgba(255,255,255,0.04);
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 6px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 6px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: rgba(255,255,255,0.12);
}}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{
    background: rgba(10,132,255,0.35);
}}
QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{ width: 9px; height: 9px; }}

/* --- point list (right panel) --- */
QListWidget {{
    background: {BG_PANEL}; border: none; padding: 4px; outline: none;
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 5px; }}
QListWidget::item:hover:!selected {{ background: rgba(255,255,255,0.06); }}
/* soft tint, not a solid accent fill — the accent CHECKBOX must stay legible */
QListWidget::item:selected {{
    background: rgba(10,132,255,0.28); color: white;
    border: 1px solid rgba(10,132,255,0.45);
}}
QListView::indicator {{
    width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid {HAIRLINE}; background: {BG_INPUT};
}}
QListView::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QDockWidget {{ color: {TEXT_DIM}; font-weight: 600; }}
QDockWidget::title {{ padding: 7px 10px; background: {BG_WINDOW}; }}

/* --- menus --- */
QMenuBar {{ background: {BG_WINDOW}; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 5px; background: transparent; }}
QMenuBar::item:selected {{ background: rgba(255,255,255,0.08); }}
QMenu {{ background: {BG_PANEL}; border: 1px solid {HAIRLINE}; padding: 5px; }}
QMenu::item {{ padding: 5px 26px 5px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: {TEXT_DISABLED}; }}
QMenu::separator {{ height: 1px; background: {HAIRLINE}; margin: 5px 8px; }}

/* --- progress: a thin quiet bar (position lives on the timeline) --- */
QProgressBar {{
    background: {BG_INPUT}; border: none; border-radius: 2px;
    min-height: 5px; max-height: 5px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

/* --- structure --- */
QSplitter::handle:vertical {{ background: {BG_WINDOW}; height: 5px; }}
QSplitter::handle:hover {{ background: rgba(10,132,255,0.40); }}
QStatusBar {{ background: {BG_WINDOW}; color: {TEXT_DIM}; }}
QStatusBar QLabel {{ color: {TEXT_DIM}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{
    background: #303036; color: {TEXT}; border: 1px solid {HAIRLINE};
    padding: 5px 8px;
}}
QTextBrowser {{ background: {BG_PANEL}; border: none; padding: 6px; }}

/* --- scrollbars: overlay-thin, no buttons --- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: rgba(255,255,255,0.22); border-radius: 3px; }}
QScrollBar::handle:hover {{ background: rgba(255,255,255,0.35); }}
QScrollBar::handle:vertical {{ min-height: 24px; }}
QScrollBar::handle:horizontal {{ min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


ARROW_DIR = Path(__file__).resolve().parent / "_theme_cache"


def _arrow_png(up: bool, color: str, name: str) -> str | None:
    """Draw a chevron and save it as a PNG the stylesheet can reference.

    A stylesheet that styles a spin box's step buttons makes Qt paint the
    buttons itself and drop the style's arrow primitives; without an
    `image:` rule the ± steppers are simply blank. Qt style sheets take images only from files,
    so the chevrons are rendered once, at 2x for high-DPI screens, into the
    app's own folder (the self-contained-folder rule: nothing outside it).
    Returns the file path with forward slashes, or None if it cannot be
    written (read-only install): the steppers then stay blank but work."""
    try:
        ARROW_DIR.mkdir(exist_ok=True)
        path = ARROW_DIR / f"{name}.png"
        if not path.exists():
            pm = QPixmap(18, 18)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(QColor(color), 2.4)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            tip_y, base_y = (5.5, 11.5) if up else (12.5, 6.5)
            path_ = QPainterPath(QPointF(4.0, base_y))
            path_.lineTo(QPointF(9.0, tip_y))
            path_.lineTo(QPointF(14.0, base_y))
            p.drawPath(path_)
            p.end()
            if not pm.save(str(path), "PNG"):
                return None
        return str(path).replace("\\", "/")
    except OSError:
        return None


def spin_arrow_rules() -> str:
    """The `image:` rules for the spin-box arrows (normal + disabled)."""
    up, down = _arrow_png(True, TEXT, "spin_up"), _arrow_png(False, TEXT, "spin_down")
    up_d, down_d = _arrow_png(True, TEXT_DISABLED, "spin_up_dim"), _arrow_png(False, TEXT_DISABLED, "spin_down_dim")
    if not (up and down):
        return ""
    rules = (f"QAbstractSpinBox::up-arrow {{ image: url({up}); width: 9px; height: 9px; }}\n"
             f"QAbstractSpinBox::down-arrow {{ image: url({down}); width: 9px; height: 9px; }}\n")
    if up_d and down_d:
        rules += (f"QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url({up_d}); }}\n"
                  f"QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url({down_d}); }}\n")
    return rules


def apply_theme(app) -> None:
    """Fusion base + dark palette + the stylesheet. Idempotent."""
    if app is None or getattr(app, "_cotrk_themed", False):
        return
    app.setStyle("Fusion")
    if sys.platform.startswith("win"):
        app.setFont(QFont("Segoe UI", 9))    # elsewhere the platform's own UI font
    pal = QPalette()
    roles = {
        QPalette.Window: BG_WINDOW, QPalette.WindowText: TEXT,
        QPalette.Base: BG_INPUT, QPalette.AlternateBase: BG_PANEL,
        QPalette.Text: TEXT, QPalette.Button: BG_WINDOW,
        QPalette.ButtonText: TEXT, QPalette.Highlight: ACCENT,
        QPalette.HighlightedText: "#FFFFFF", QPalette.ToolTipBase: "#303036",
        QPalette.ToolTipText: TEXT, QPalette.PlaceholderText: TEXT_DIM,
        QPalette.Link: ACCENT, QPalette.BrightText: RED,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        pal.setColor(QPalette.Disabled, role, QColor(TEXT_DISABLED))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET + spin_arrow_rules())
    app._cotrk_themed = True
