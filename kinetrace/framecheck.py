"""The window side of the frame check (I266): a worker that checks a camera's video off the GUI thread
and the dialog that shows the evidence. The measuring is `fingerprint.py` / `video_source.check_seeks`;
nothing here changes data.

`FrameCheckWorker` (one camera, one capture at a time, low priority): first the file's seeks
(`check_seeks`; the plan goes to `video_source.set_seek_plan` at once, so every decoder of this file
reads exact frames from then on), then either the fingerprint is made (a camera without one) or this
computer's frames are compared with it (`fingerprint.compare`).

`FrameCheckDialog`: the owner's idea -- for each saved moment, side by side, the picture saved where the
project was made and this computer's frames N-1, N, N+1, a difference view, the best match marked, one
plain sentence and a verdict on top. "Go to this frame" is the hand check; "Add the frame on screen" is
the fallback for a video that barely moves (only on the computer that made the fingerprint).
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from kinetrace import fingerprint as fpm
from kinetrace import theme
from kinetrace import video_source as vs


class FrameCheckWorker(QThread):
    """Check one camera's video: its seeks, then make or compare its fingerprint (see the module text)."""
    seeks_checked = Signal(object)      # video_source.SeekPlan
    made = Signal(object)               # fingerprint.VideoFingerprint
    checked = Signal(object)            # fingerprint.CheckResult
    failed = Signal(str)

    def __init__(self, path: str, n_frames: int, fingerprint, camera: str, *, make_missing: bool = True,
                 force: bool = False, data_before: bool = False):
        super().__init__()
        self.path, self.n_frames, self.fp, self.camera = str(path), int(n_frames), fingerprint, camera
        self.make_missing, self.force, self.data_before = make_missing, force, data_before
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            plan = vs.seek_plan(self.path)
            if plan is None or plan.file != vs.file_identity(self.path):
                plan = vs.check_seeks(self.path, self.n_frames, self._cancelled)
                if self._cancel:
                    return
                vs.set_seek_plan(self.path, plan)
                self.seeks_checked.emit(plan)
            if self.fp is None:
                if self.make_missing:
                    fp = fpm.make(self.path, self.n_frames, seeks="exact" if plan.exact else "read forward",
                                  data_before=self.data_before, should_cancel=self._cancelled)
                    if fp is not None and not self._cancel:
                        self.made.emit(fp)
            elif self.force or fpm.needs_check(self.fp, self.path):
                res = fpm.compare(self.fp, self.path, self.n_frames, self.camera, should_cancel=self._cancelled)
                if res is not None and not self._cancel:
                    self.checked.emit(res)
        except Exception as e:  # noqa: BLE001 -- said in words, never a crash of the window
            from kinetrace.errors import plain_error
            self.failed.emit(plain_error(e, f"{self.camera}: checking the video's frames failed", short=True))


VERDICT_COLOR = {"good": theme.GREEN, "ok": theme.AMBER, "poor": theme.RED}
_SIDE = 168                 # pixels per picture in the dialog


def _pixmap(rgb_or_grey: np.ndarray, side: int = _SIDE, frame_color: str | None = None,
            box=None, box_scale: float = 1.0) -> QPixmap:
    a = np.ascontiguousarray(rgb_or_grey)
    if a.ndim == 2:
        a = np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(img).scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    if frame_color or box is not None:
        p = QPainter(pm)
        if box is not None:
            k = pm.width() / float(w) * box_scale
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.drawRect(int(box[0] * k), int(box[1] * k), max(2, int(box[2] * k)), max(2, int(box[2] * k)))
        if frame_color:
            p.setPen(QPen(QColor(frame_color), 4))
            p.drawRect(2, 2, pm.width() - 4, pm.height() - 4)
        p.end()
    return pm


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| on grey, stretched so that 32 levels are white: where two pictures differ."""
    ga = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.clip(np.abs(ga - gb) * (255.0 / 32.0), 0, 255).astype(np.uint8)


class FrameCheckDialog(QDialog):
    """The evidence for one camera (see the module text). `go_to(frame)` and `add_frame()` are the
    app's: the dialog only shows and reports."""

    def __init__(self, parent, result, fp, camera: str, *, go_to=None, add_frame=None, reference: bool = False,
                 current_frame: int | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Video frames on this computer — {camera}")
        self.result, self.fp, self.camera = result, fp, camera
        self._go_to, self._add = go_to, add_frame
        self.k = self._first_interesting()
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.chip = QLabel(result.quality.upper())
        self.chip.setStyleSheet(f"background:{VERDICT_COLOR.get(result.quality, theme.TEXT_DIM)};color:#000;"
                                "font-weight:bold;padding:3px 8px;border-radius:4px;")
        self.chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top.addWidget(self.chip, 0, Qt.AlignTop)
        self.sentence = QLabel(result.sentence)
        self.sentence.setWordWrap(True)
        self.sentence.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.addWidget(self.sentence, 1)
        lay.addLayout(top)
        if reference:
            ref = QLabel("This computer made the fingerprint of this video, so it is the reference: the check "
                         "means something on another computer (it runs there by itself when the project is "
                         "opened). Here it shows which moments are kept.")
            ref.setWordWrap(True)
            ref.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            ref.setStyleSheet(f"color:{theme.TEXT_DIM};")
            lay.addWidget(ref)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Previous moment")
        self.btn_next = QPushButton("Next moment ▶")
        self.where = QLabel("")
        self.where.setAlignment(Qt.AlignCenter)
        self.btn_prev.clicked.connect(lambda: self.show_moment(self.k - 1))
        self.btn_next.clicked.connect(lambda: self.show_moment(self.k + 1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.where, 1)
        nav.addWidget(self.btn_next)
        lay.addLayout(nav)
        grid = QGridLayout()
        self.pics, self.caps = [], []
        for c in range(5):
            pic, cap = QLabel(), QLabel()
            pic.setAlignment(Qt.AlignCenter)
            pic.setMinimumSize(_SIDE + 8, _SIDE + 8)
            cap.setAlignment(Qt.AlignCenter)
            cap.setWordWrap(True)
            cap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            grid.addWidget(pic, 0, c)
            grid.addWidget(cap, 1, c)
            self.pics.append(pic)
            self.caps.append(cap)
        self.whole_saved, self.whole_here = QLabel(), QLabel()
        self.whole_cap_saved = QLabel("The whole frame where it was made (the crop's place in blue)")
        self.whole_cap_here = QLabel("The whole frame N here")
        for c, w in ((0, self.whole_saved), (1, self.whole_here)):
            w.setAlignment(Qt.AlignCenter)
            grid.addWidget(w, 2, c * 2, 1, 2)
        for c, w in ((0, self.whole_cap_saved), (1, self.whole_cap_here)):
            w.setAlignment(Qt.AlignCenter)
            w.setWordWrap(True)
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            grid.addWidget(w, 3, c * 2, 1, 2)
        lay.addLayout(grid)
        self.numbers = QLabel(result.detail)
        self.numbers.setWordWrap(True)
        self.numbers.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.numbers.setStyleSheet(f"color:{theme.TEXT_DIM};")
        self.numbers.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.numbers)
        what = QLabel("Nothing in the project was changed. If the pictures here are shifted, the tracks are "
                      "shown on the wrong frames on this computer: work on the computer that made the project, "
                      "or re-encode the video there once (File → Check Video Frames in the manual says how) "
                      "and track on the new file.")
        what.setWordWrap(True)
        what.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        if result.verdict in ("shifted", "mixed", "different"):
            lay.addWidget(what)
        buttons = QHBoxLayout()
        self.btn_go = QPushButton("Go to this frame")
        self.btn_go.setToolTip("Shows frame N of this camera in the main window: compare it with the saved picture")
        self.btn_go.clicked.connect(self._go)
        self.btn_go.setEnabled(go_to is not None)
        buttons.addWidget(self.btn_go)
        self.btn_add = None
        if add_frame is not None and current_frame is not None:
            self.btn_add = QPushButton("Add the frame on screen to the fingerprint")
            self.btn_add.setToolTip("For a video that barely moves: step the main window to a frame where something "
                                    "moves fast (this window stays open), then add it here. Only on the computer "
                                    "that made the fingerprint.")
            self.btn_add.clicked.connect(self._add_clicked)
            buttons.addWidget(self.btn_add)
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self.resize(5 * (_SIDE + 24), 640)
        self.show_moment(self.k)

    def _first_interesting(self) -> int:
        """The moment shown first: one whose best match is not its own frame, else the first."""
        for i, m in enumerate(self.result.marks):
            if m.best not in (0, None) and m.verdict in ("clear", "lean"):
                return i
        return 0

    def show_moment(self, k: int) -> None:
        marks = self.result.marks
        if not marks:
            self.where.setText("No moment to show")
            return
        k = int(np.clip(k, 0, len(marks) - 1))
        self.k = k
        m, saved = marks[k], self.fp.crops[k]
        box = self.fp.marks[k].box
        self.btn_prev.setEnabled(k > 0)
        self.btn_next.setEnabled(k < len(marks) - 1)
        self.where.setText(f"Moment {k + 1} of {len(marks)} — frame {m.frame} ({m.verdict}"
                           + (f", best match {m.best:+d}" if m.best is not None else "") + ")")
        self.pics[0].setPixmap(_pixmap(saved, frame_color=theme.ACCENT))
        self.caps[0].setText(f"Saved: frame {m.frame}\n(where the project was made)")
        for c, d in zip((1, 2, 3), (-1, 0, 1)):
            here = m.here.get(d)
            if here is None:
                self.pics[c].clear()
                self.caps[c].setText(f"Here: frame {m.frame + d}\n(not in this video)")
                continue
            best = d == m.best
            self.pics[c].setPixmap(_pixmap(here, frame_color=theme.GREEN if best else None))
            self.caps[c].setText(f"Here: frame {m.frame + d}\n" + (f"match {m.dists.get(d, float('nan')):.4f}")
                                 + ("  ✔ best match" if best else ""))
        cmp_d = m.best if (m.best is not None and m.best in m.here) else (0 if 0 in m.here else None)
        if cmp_d is not None:
            self.pics[4].setPixmap(_pixmap(difference(saved, m.here[cmp_d])))
            self.caps[4].setText(f"Difference: saved vs here {m.frame + cmp_d}\n(black = the same picture)")
        else:
            self.pics[4].clear()
            self.caps[4].setText("Difference: nothing to compare")
        th = self.fp.thumbs[k] if self.fp.thumbs is not None and k < len(self.fp.thumbs) else None
        if th is not None:
            self.whole_saved.setPixmap(_pixmap(th, side=2 * _SIDE, box=box,
                                               box_scale=th.shape[1] / float(max(1, self.fp.width))))
        here_th = m.here_thumb.get(0)
        if here_th is not None:
            self.whole_here.setPixmap(_pixmap(here_th, side=2 * _SIDE))
            self.whole_cap_here.setText(f"The whole frame {m.frame} here")
        else:
            self.whole_here.clear()

    def _go(self) -> None:
        if self._go_to is not None and self.result.marks:
            self._go_to(int(self.result.marks[self.k].frame))

    def _add_clicked(self) -> None:
        if self._add is not None:
            self._add()
            self.accept()
