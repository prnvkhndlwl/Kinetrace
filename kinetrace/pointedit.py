"""Edit → Point Tools… (G147): fix the identities of tracked points in the working camera.

Four tools, each one Ctrl+Z step (the app applies them, `session` does the work):
  * Swap two points -- a tracker swapped them (the left and the right foot);
  * Move one point's data to another -- a stretch tracked under the wrong name;
  * Fill one point's empty frames from another -- one landmark tracked in two pieces;
  * Split a point into two at a frame -- from there on it is really another part.
Over a frame range: the whole video, from a frame to the end, the timeline's selected window,
or two frames typed in. The dialog only reads the session (the counts it shows); nothing
changes until the app applies `result`.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QSpinBox,
                               QVBoxLayout, QWidget)

from kinetrace.theme import AMBER

OPS = {   # key: (label, what it does, needs a second point)
    "swap": ("Swap two points",
             "A and B exchange their data in the range: use it where the tracker swapped them (left and right "
             "foot, two balls that crossed).", True),
    "move": ("Move A's data to B",
             "On every frame of the range where A has data, B takes it (B's own data there is replaced) and A "
             "is cleared: a stretch that was tracked under the wrong name.", True),
    "fill": ("Fill B's empty frames from A",
             "Where B has no data in the range and A has, B takes A's data; A is left as it is (delete it "
             "afterwards if it was the same landmark tracked in two pieces).", True),
    "split": ("Split A into a new point",
              "From the chosen frame on, A's data becomes a new point (same kind and tracker) and A ends on the "
              "frame before: from there on it was really another part.", False),
}


class PointToolsDialog(QDialog):
    """`session` is read for names and counts only. `a` / `b` = the points to start with (ids),
    `frame` = the playhead, `window` = the timeline's selected (f0, f1) or None. After accept,
    `result` = dict(op, a, b, f0, f1, name)."""

    def __init__(self, parent, session, a: int = 0, b: int | None = None, op: str = "swap",
                 frame: int = 0, window: tuple[int, int] | None = None, camera: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Point tools")
        self.s = session
        self.frame = int(frame)
        self.window = window
        self.result: dict | None = None
        self._pids = list(range(session.n_points))
        lay = QVBoxLayout(self)
        intro = QLabel("Fix which point is which" + (f" in <b>{camera}</b>" if camera else "")
                       + ". Each change is one Ctrl+Z step. Only this camera changes: in other cameras the "
                         "same names keep their own data.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        ops = QWidget()
        og = QGridLayout(ops)
        og.setContentsMargins(0, 0, 0, 0)
        self.op_group = QButtonGroup(self)
        self.op_buttons: dict[str, QRadioButton] = {}
        for i, (key, (label, tip, _two)) in enumerate(OPS.items()):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            self.op_group.addButton(rb, i)
            self.op_buttons[key] = rb
            og.addWidget(rb, i // 2, i % 2)
        lay.addWidget(ops)
        self.op_text = QLabel()
        self.op_text.setWordWrap(True)
        lay.addWidget(self.op_text)

        form = QFormLayout()
        self.combo_a = QComboBox()
        self.combo_b = QComboBox()
        for q in self._pids:
            p = session.points[q]
            tag = " (silhouette)" if p.derived else " (ball)" if p.is_ball else ""
            self.combo_a.addItem(p.name + tag, q)
            self.combo_b.addItem(p.name + tag, q)
        self.combo_a.setCurrentIndex(max(0, min(a, session.n_points - 1)))
        if b is None:
            b = next((q for q in self._pids if q != a and not session.points[q].derived), a)
        self.combo_b.setCurrentIndex(max(0, min(b, session.n_points - 1)))
        form.addRow("A:", self.combo_a)
        self.label_b = QLabel("B:")
        form.addRow(self.label_b, self.combo_b)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("the new point's name")
        self.label_name = QLabel("New point:")
        form.addRow(self.label_name, self.name_edit)
        lay.addLayout(form)

        rng = QWidget()
        rg = QGridLayout(rng)
        rg.setContentsMargins(0, 0, 0, 0)
        n = session.n_frames
        self.range_group = QButtonGroup(self)
        self.r_all = QRadioButton(f"The whole video (0–{n - 1})")
        self.r_from = QRadioButton(f"From frame {self.frame} to the end")
        self.r_win = QRadioButton(f"The selected window ({window[0]}–{window[1]})" if window
                                  else "The selected window (Shift+drag on the timeline first)")
        self.r_win.setEnabled(window is not None)
        self.r_custom = QRadioButton("Frames")
        self.spin0 = QSpinBox()
        self.spin1 = QSpinBox()
        for sp, v in ((self.spin0, self.frame), (self.spin1, n - 1)):
            sp.setRange(0, n - 1)
            sp.setValue(v)
            sp.valueChanged.connect(lambda _v: (self.r_custom.setChecked(True), self._update()))
        for i, r in enumerate((self.r_all, self.r_from, self.r_win, self.r_custom)):
            self.range_group.addButton(r, i)
        rg.addWidget(QLabel("Frames:"), 0, 0)
        rg.addWidget(self.r_all, 0, 1)
        rg.addWidget(self.r_from, 1, 1)
        rg.addWidget(self.r_win, 2, 1)
        row = QHBoxLayout()
        row.addWidget(self.r_custom)
        row.addWidget(self.spin0)
        row.addWidget(QLabel("to"))
        row.addWidget(self.spin1)
        row.addStretch(1)
        rg.addLayout(row, 3, 1)
        (self.r_win if window is not None else self.r_from).setChecked(True)
        lay.addWidget(rng)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self.op_buttons.get(op, self.op_buttons["swap"]).setChecked(True)
        self.op_group.idToggled.connect(lambda *_: self._update())
        self.range_group.idToggled.connect(lambda *_: self._update())
        self.combo_a.currentIndexChanged.connect(lambda *_: self._update())
        self.combo_b.currentIndexChanged.connect(lambda *_: self._update())
        self.name_edit.textChanged.connect(lambda *_: self._update())
        self.resize(560, self.sizeHint().height())
        self._update()

    # ------------------------------------------------------------------ state
    def op(self) -> str:
        return next(k for k, rb in self.op_buttons.items() if rb.isChecked())

    def span(self) -> tuple[int, int]:
        n = self.s.n_frames
        if self.op() == "split":
            return self.spin0.value() if self.r_custom.isChecked() else self.frame, n - 1
        if self.r_all.isChecked():
            return 0, n - 1
        if self.r_from.isChecked():
            return self.frame, n - 1
        if self.r_win.isChecked() and self.window is not None:
            return tuple(sorted(self.window))
        return tuple(sorted((self.spin0.value(), self.spin1.value())))

    def _update(self) -> None:
        s, op = self.s, self.op()
        label, tip, two = OPS[op]
        self.op_text.setText(tip)
        split = op == "split"
        for w in (self.combo_b, self.label_b):
            w.setVisible(two)
        for w in (self.name_edit, self.label_name):
            w.setVisible(split)
        for w in (self.r_all, self.r_win, self.spin1):
            w.setEnabled(not split and (w is not self.r_win or self.window is not None))
        self.r_from.setText(f"From frame {self.frame} to the end" if not split else f"At frame {self.frame}")
        self.r_custom.setText("Frames" if not split else "At frame")
        if split and not (self.r_from.isChecked() or self.r_custom.isChecked()):
            self.r_from.setChecked(True)
        a = self.combo_a.currentData()
        b = self.combo_b.currentData() if two else None
        f0, f1 = self.span()
        why = s.point_tool_problem(a, b)
        na = int(s.tracked[f0:f1 + 1, a].sum()) if a is not None else 0
        nb = int(s.tracked[f0:f1 + 1, b].sum()) if b is not None else 0
        an = s.points[a].name if a is not None else "A"
        bn = s.points[b].name if b is not None else "B"
        if why is None:
            if op == "swap":
                why = None if na or nb else "neither point has data in these frames"
                text = f"{an} has data on {na:,} frame(s) in {f0}–{f1}, {bn} on {nb:,}: they exchange it."
            elif op == "move":
                over = int((s.tracked[f0:f1 + 1, a] & s.tracked[f0:f1 + 1, b]).sum())
                why = None if na else f"{an} has no data in these frames"
                text = (f"{bn} takes {an}'s data on {na:,} frame(s) in {f0}–{f1}"
                        + (f" ({over:,} of them replace {bn}'s own)" if over else "") + f"; {an} is cleared there.")
            elif op == "fill":
                k = int((s.tracked[f0:f1 + 1, a] & ~s.tracked[f0:f1 + 1, b]).sum())
                why = None if k else f"{bn} already has data wherever {an} has some in these frames"
                text = f"{bn} gets {an}'s data on {k:,} empty frame(s) in {f0}–{f1}; {an} stays as it is."
            else:
                why = None if na else f"{an} has no data from frame {f0} on"
                nm = self.name_edit.text().strip() or f"{an} (2)"
                text = (f"{an}'s data on {na:,} frame(s) from frame {f0} on becomes the new point “{nm}”; "
                        f"{an} ends at frame {f0 - 1}.")
        else:
            text = ""
        self.summary.setText(text if why is None else f"<span style='color:{AMBER}'>Cannot: {why}.</span>")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(why is None)
        self.buttons.button(QDialogButtonBox.Ok).setText({"swap": "Swap", "move": "Move", "fill": "Fill",
                                                          "split": "Split"}[op])

    def _accept(self) -> None:
        if not self.buttons.button(QDialogButtonBox.Ok).isEnabled():
            return
        f0, f1 = self.span()
        op = self.op()
        self.result = dict(op=op, a=self.combo_a.currentData(),
                           b=self.combo_b.currentData() if OPS[op][2] else None, f0=int(f0), f1=int(f1),
                           name=self.name_edit.text().strip() or None)
        self.accept()
