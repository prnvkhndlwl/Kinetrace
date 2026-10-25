"""The Track pop-up (G217): what a Track press does with the data that is already ahead of it.

Owner 2026-10-23: "when a user presses track - this should be pop-up". Asked when the points (or the silhouette)
the press tracks already have data after its start frame, in any camera of the press:
  * Keep my clicks, replace the rest (the default): the hand clicks stay and the trackers restart from them;
    every other frame the run reaches gets the trackers' new positions;
  * Fresh run: today's run exactly - it writes over the clicks it reaches and restarts nowhere - and then
    everything of these points after the last frame it reached is erased, clicks included. Never remembered;
  * Only fill the gaps (PR B): nothing that exists is touched; each empty stretch after the start is tracked from
    the frame before it, when that is a usable position (the app's `gap_plan`); offered when there is one.
The dialog only reads `info` (the counts the app made); the app acts on `choice()` after accept.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QLabel, QRadioButton,
                               QVBoxLayout)

CHOICES = {   # key: (label, what it does)
    "keep": ("Keep my clicks, replace the rest",
             "Your clicks stay exactly where they are (a Moving spot point also starts again from them); every "
             "other frame the run reaches gets the trackers' new positions. If the run stops early, the frames "
             "after the stop keep what they had. (A ball marker's clicks are prompts its run fits again.)"),
    "gaps": ("Only fill the gaps",
             "Nothing that exists is touched: each empty stretch after this frame is tracked from the frame before "
             "it. A stretch where the point left the picture or is marked hidden is skipped (the end says which); "
             "the frame or two missing between your clicks are left to Fill gaps. Inside a stretch Auto-pause is "
             "on: where the tracker loses the point, the stretch ends there."),
    "fresh": ("Fresh run",
              "The run goes over everything ahead as if there were no clicks: it writes over the clicks it "
              "reaches and restarts nowhere. When it has ended, everything of these points after the last frame "
              "it reached is erased too, clicks included. Ctrl+Z brings it all back."),
}


def _duration(secs: float) -> str:
    secs = max(1, int(round(float(secs))))
    return f"{secs // 60} min {secs % 60:02d} s" if secs >= 60 else f"{secs} s"


def ahead_sentence(info: dict) -> str:
    """'Ahead of frame 120 these 3 points have 4,300 tracked frames (in 1 of the 2 cameras); 37 are clicks, 12 of
    them imported.' from the app's counts: start, n_points, frames, clicks, imported, cameras (of the press),
    cameras_with (that have data ahead), masks."""
    n_pts, frames, clicks, imp, cams, with_ = (int(info.get(k, 0)) for k in ("n_points", "frames", "clicks",
                                                                             "imported", "cameras", "cameras_with"))
    who = f"these {n_pts} points have" if n_pts != 1 else "this point has"
    txt = f"Ahead of frame {int(info.get('start', 0))} {who} {frames:,} tracked frame{'s' if frames != 1 else ''}"
    if cams > 1:
        txt += f" (in all {cams} cameras)" if with_ >= cams else f" (in {with_} of the {cams} cameras)"
    if clicks:
        txt += f"; {clicks:,} {'are clicks' if clicks != 1 else 'is a click'}"
        if imp:
            txt += f", {imp:,} of them imported"
    masks = int(info.get("masks", 0))
    if masks:
        txt += f"; the silhouette has {masks:,} frame{'s' if masks != 1 else ''} ahead"
    return txt + "."


class TrackChoiceDialog(QDialog):
    """`info` = the press's counts (see `ahead_sentence`). After accept: `choice()` = "keep" | "fresh" and
    `remember_box.isChecked()` = use it every time (never for "fresh")."""

    def __init__(self, parent, info: dict):
        super().__init__(parent)
        self.setWindowTitle("Track: there is data ahead")
        self._info = dict(info)
        lay = QVBoxLayout(self)
        self.head = QLabel(self.message())
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QRadioButton] = {}
        for key, (label, text) in CHOICES.items():
            rb = QRadioButton(label)
            self.group.addButton(rb)
            self.buttons[key] = rb
            lay.addWidget(rb)
            why = QLabel(text)
            why.setWordWrap(True)
            why.setIndent(22)
            lay.addWidget(why)
        n_gaps, n_gf = int(info.get("gaps", 0)), int(info.get("gap_frames", 0))
        if n_gaps:
            secs = info.get("gap_seconds")
            self.buttons["gaps"].setText(
                f"Only fill the gaps ({n_gaps:,} stretch{'es' if n_gaps != 1 else ''}, {n_gf:,} frame"
                f"{'s' if n_gf != 1 else ''}" + (f", about {_duration(secs)}" if secs else "") + ")")
        else:
            self.buttons["gaps"].setText("Only fill the gaps (there is no gap to fill here)")
            self.buttons["gaps"].setEnabled(False)
        n_clicks = int(info.get("clicks", 0))
        if n_clicks:
            self.buttons["fresh"].setText(f"Fresh run (removes your {n_clicks:,} click{'s' if n_clicks != 1 else ''} "
                                          "ahead)")
        self.remember_box = QCheckBox("Use this every time (Track ▾ → When there is data ahead changes it)")
        lay.addWidget(self.remember_box)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("Track")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.group.buttonToggled.connect(lambda *_: self._sync())
        self.pick("keep")

    def message(self) -> str:
        note = (" (You chose to only fill the gaps every time, but there is no gap to fill here.)"
                if self._info.get("nothing_to_fill") else "")
        return ahead_sentence(self._info) + note + " What should this Track press do with it?"

    def pick(self, key: str) -> None:
        self.buttons.get(key, self.buttons["keep"]).setChecked(True)
        self._sync()

    def choice(self) -> str:
        return next((k for k, rb in self.buttons.items() if rb.isChecked()), "keep")

    def _sync(self) -> None:
        """A Fresh run is never remembered: it removes clicks, so it is asked for each time."""
        fresh = self.choice() == "fresh"
        if fresh:
            self.remember_box.setChecked(False)
        self.remember_box.setEnabled(not fresh)
