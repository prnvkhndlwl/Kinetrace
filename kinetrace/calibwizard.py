"""Wand calibration wizard — 3D → Calibrate Cameras with a Wand.

Written for someone who has never calibrated a camera. The pages explain the
one idea that matters at each step, pull everything they can from the open
project (which landmarks look like wand ends, how many frames they share,
image sizes, frame rates), run `wand.calibrate_wand` in a background thread,
and end with a report in plain language: a verdict (good / ok / poor), WHY,
and what to do about it. The numbers are there too, but the verdict rules
are what a newcomer should trust.

The data comes from the project the wizard is opened on: one camera per view,
the two wand ends tracked as two landmarks with the SAME names in every camera
(`wanddata.collect_wand`), optional extra landmarks, a dropped object for the
vertical, or three reference points for the axes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QRadioButton, QSpinBox,
                               QTextBrowser, QVBoxLayout, QWidget, QWizard, QWizardPage)

from kinetrace import lens as _lens
from kinetrace import theme
from kinetrace import wand as _wand
from kinetrace import wanddata as wd
from kinetrace.errors import plain_error
from kinetrace.lenswizard import VERDICT_COLORS, _dim, stop_page_threads

UNITS = [("metres (m)", "m"), ("centimetres (cm)", "cm"), ("millimetres (mm)", "mm"), ("inches (in)", "in"),
         ("wand lengths (not measured yet: enter 1)", "wand")]   # every distance then reads in wand lengths
VERDICT_WORDS = {"good": "GOOD — you can trust this calibration",
                 "ok": "USABLE — but read the notes below",
                 "poor": "NOT GOOD ENOUGH — fix what the notes say and calibrate again"}
# the wand frames: the wizard goes on from MIN_FRAMES, a stable result wants GOOD_FRAMES (G137: the
# intro said "30 is the minimum" while 10 passed - one wording everywhere)
MIN_FRAMES, GOOD_FRAMES = 10, 30


def lens_size_mismatch(prof, width: int, height: int, cam_name: str) -> str | None:
    """None when lens profile `prof` was made at this camera's picture size,
    else the sentence that says why it cannot be used (I31): its focal length
    and centre are in the other size's pixels, and with every camera profiled
    both are HELD FIXED in the solve - a 1280 x 960 profile on 640 x 480
    cameras turned a sound rig into verdict poor with the blame on the wand.
    The ONE size rule is `lens.size_mismatch` (R19); this is its wizard form."""
    if prof is None or not width or not height:
        return None
    return _lens.size_mismatch((prof.width, prof.height), (width, height), cam_name,
                               "this lens profile was measured on")


def _same_lens(a, b) -> bool:
    return _lens.same_profile(a, b)


def share_targets(project, view: int, prof=None) -> tuple[list[int], list[int]]:
    """The other cameras a lens profile of camera `view` can be shared with
    (G40: identical cameras, one checkerboard calibration): every camera with
    the same picture size. Returns (without a profile, with a DIFFERENT one) -
    the second only after the user agrees. `prof` = the profile (default: the
    one attached to `view`)."""
    lenses = list(getattr(project, "lenses", []) or [])
    if prof is None:
        prof = lenses[view] if view < len(lenses) else None
    s0 = project.sessions[view]
    fill, replace = [], []
    for c in range(project.n_views):
        if c == view:
            continue
        sc = project.sessions[c]
        if (int(sc.width), int(sc.height)) != (int(s0.width), int(s0.height)):
            continue
        other = lenses[c] if c < len(lenses) else None
        if other is None:
            fill.append(c)
        elif not _lens.same_profile(other, prof):
            replace.append(c)
    return fill, replace


def _clear_grid(grid) -> None:
    """Empty a QGridLayout, deleting the widgets it held."""
    while grid.count():
        it = grid.takeAt(0)
        if it.widget():
            it.widget().deleteLater()


def _guess_wand_names(names: list[str]) -> tuple[str | None, str | None]:
    """Two names that look like wand ends ("wand A"/"wand B", "wand1"/"wand2",
    anything containing 'wand' or 'ball'), else the first two."""
    hits = [n for n in names if "wand" in n.lower()] or [n for n in names if "ball" in n.lower()]
    if len(hits) >= 2:
        return hits[0], hits[1]
    if len(names) >= 2:
        return names[0], names[1]
    return (names[0] if names else None), None


class _Cancelled(Exception):
    """Raised from the progress callback once the wizard is closed (I36). Not a
    WandError on purpose: the focal-length search swallows those."""


class _CalibThread(QThread):
    # every signal carries the run's generation number (I28): a run started
    # with settings the user has since changed must not become the result
    progress = Signal(float, str, int)
    finished_ok = Signal(object, object, int)     # WandResult, gravity dict | None, generation
    error = Signal(str, int)

    def __init__(self, kwargs: dict, frame_mode: str, drop: dict | None, axes: dict | None,
                 lens_models: list | None = None, lens_summaries: list | None = None, gen: int = 0):
        super().__init__()
        self.kwargs = kwargs
        self.frame_mode = frame_mode
        self.drop = drop
        self.axes = axes
        self.lens_models = lens_models or []
        self.lens_summaries = lens_summaries or []
        self.gen = int(gen)
        self._cancel = False

    def cancel(self):
        """Stop at the next progress step (a focal-length guess or an adjustment
        pass): `WandWizard.done` then waits for a thread that is ending, not
        for a solve that may take a minute."""
        self._cancel = True

    def _note(self, f, m):
        if self._cancel:
            raise _Cancelled()
        self.progress.emit(float(f), str(m), self.gen)

    def run(self):
        wand = _wand
        try:
            res = wand.calibrate_wand(progress=self._note, **self.kwargs)
            grav = None
            # (I29) the wand solve stands on its own: a vertical / axes step that
            # cannot be applied keeps camera 1's axes and says why, instead of
            # throwing a good calibration away as "Calibration failed"
            try:
                if self.frame_mode == "drop" and self.drop is not None:
                    self._note(0.95, "Aligning the vertical to the dropped object")
                    res, grav = wand.align_gravity(res, self.drop["uv"], self.drop["fps"],
                                                   frame0=self.drop.get("t0"))
                elif self.frame_mode == "axes" and self.axes is not None:
                    self._note(0.95, "Aligning the axes to your reference points")
                    res = wand.align_axes(res, self.axes["origin"], self.axes["x"], self.axes["xy"])
            except wand.WandError as e:
                cam1 = (res.report.get("camera_names") or ["camera 1"])[0]
                if self.frame_mode == "drop":
                    what = "the vertical from the dropped object"
                    fix = ("pick the frames where it falls (at least 4, seen by two or more cameras)")
                else:
                    a = self.axes or {}
                    what = "the axes from the three reference points"
                    fix = (f"pick a frame where {a.get('names', 'all three')} are tracked in two or more "
                           f"cameras (the page suggests one)")
                    e = f"{e} on reference frame {a.get('t', '?')}"
                res = wand.keep_camera_frame(
                    res, f"{what[0].upper() + what[1:]} could NOT be set: {e}. The calibration itself is fine "
                         f"and keeps {cam1}'s axes. Go Back to 'Which way is up' and {fix}, or choose "
                         f"'Neither'.", what)
            # cameras with a lens profile were solved on UNDISTORTED points: the
            # calibration must undistort the tracks the same way at 3D time
            for c, m in enumerate(self.lens_models):
                if m is not None and c < len(res.cameras):
                    res.cameras[c].undistort = m
            if self.lens_summaries:
                res.report["lens_profiles"] = list(self.lens_summaries)
            if self._cancel:
                return
            self.finished_ok.emit(res, grav, self.gen)
        except _Cancelled:
            return
        except wand.WandError as e:
            self.error.emit(str(e), self.gen)           # already written for the user (G112)
        except Exception as e:      # noqa: BLE001
            # (G112) anything else ("index 3 is out of bounds") is a program fault, not a message
            # for the user: a sentence, and the traceback in the error log
            self.error.emit(plain_error(e, "The calibration could not be completed"), self.gen)


class IntroPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Calibrating your cameras with a wand")
        lay = QVBoxLayout(self)
        t = QTextBrowser()
        t.setOpenExternalLinks(False)
        t.setHtml(f"""
<p><b>What calibration is.</b> To turn the points you tracked in several cameras into
3D positions, the program has to know where every camera stands, which way it looks,
and how strongly its lens magnifies. Calibration works those numbers out. Without it
there is no 3D.</p>
<p><b>What a wand is.</b> A stick with a clearly visible mark at each end, a known
distance apart (say two table-tennis balls glued 40 cm apart on a rod — measure it
with a tape, centre to centre). You wave it through the space where the animal will
be, slowly, tilting it in every direction, while <b>all cameras record at once</b>.
Because the program knows the two marks are always the same distance apart, it can
solve for the cameras.</p>
<p><b>What you need in THIS project before running the wizard:</b></p>
<ol>
<li>the wand video of every camera opened as the cameras of this project (＋ Add video
in the CAMERAS panel), with their offsets aligned;</li>
<li>the two wand ends tracked as two landmarks with the <b>same names in every camera</b>
(for example <i>wand A</i> and <i>wand B</i>): place them with N (or Add ▾ → Ball marker
for a ball on the wand's end), press Track, correct
where needed. A few hundred frames spread over the recording is ideal; the wizard goes on from
{MIN_FRAMES} frames, but a stable result wants at least {GOOD_FRAMES};</li>
<li>optionally, a small object <b>dropped</b> in view of the cameras (a ball) tracked as
another landmark — it tells the program which way is up and gives an independent check
of the scale; or three reference points on the floor.</li>
</ol>
<p><b>What you get.</b> A calibration stored in this project (use it with 3D →
Reconstruct), files you can hand to DLTdv or keep for the animal projects
(3D → Export Calibration), and a report that says in plain words whether the result
can be trusted.</p>
<p style="color:{theme.TEXT_DIM}">Method: sparse bundle adjustment on the wand observations,
the same principle as easyWand (Theriault et al. 2014). The report's verdict rules are
conservative on purpose.</p>
""")
        lay.addWidget(t, 1)
        self.check = _dim(QLabel(""))
        lay.addWidget(self.check)

    def initializePage(self):
        p = self.wiz.project
        single = p.n_views < 2
        names = wd.common_point_names(p, min_views=1 if single else 2)
        parts = []
        ok = True
        if single:
            parts.append("• This project has ONE camera. You can review the wand tracking and its coverage "
                         "on the next page; the calibration itself needs the other cameras' wand videos in "
                         "this project (＋ Add video in the CAMERAS panel) — add them when they are tracked, "
                         "or export this camera's tracks (Ctrl+E) for easyWand / DLTdv.")
        else:
            parts.append(f"✓ {p.n_views} cameras in this project.")
        if len(names) >= 2:
            parts.append(f"✓ Landmarks tracked{'' if single else ' in at least two cameras'}: {', '.join(names[:8])}"
                         + (" …" if len(names) > 8 else "") + ".")
        else:
            parts.append("✗ Fewer than two landmarks are tracked" + ("" if single else " in two or more cameras")
                         + ". Track the two wand ends (same names in every camera) and come back.")
            ok = False
        self.check.setText("\n".join(parts))
        self._ok = ok
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return bool(getattr(self, "_ok", False))


class WandPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("The wand")
        lay = QVBoxLayout(self)
        lay.addWidget(_dim(QLabel("Which two landmarks are the ends of the wand, and how far apart are "
                                  "they? Measure centre to centre. The unit you choose here becomes the "
                                  "unit of every 3D result.")))
        form = QFormLayout()
        self.end_a = QComboBox()
        self.end_b = QComboBox()
        form.addRow("Wand end 1", self.end_a)
        form.addRow("Wand end 2", self.end_b)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        self.length = QDoubleSpinBox()
        self.length.setDecimals(4)
        self.length.setRange(0.0001, 100000.0)
        self.length.setValue(0.5)
        self.unit = QComboBox()
        for label, key in UNITS:
            self.unit.addItem(label, key)
        rl.addWidget(self.length, 1)
        rl.addWidget(self.unit)
        form.addRow("Distance between the ends", row)
        lay.addLayout(form)
        self.table = _dim(QLabel(""))
        lay.addWidget(self.table)
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        lay.addWidget(self.warn)
        lay.addStretch(1)
        for cb in (self.end_a, self.end_b):
            cb.currentIndexChanged.connect(self._refresh)
        self.length.valueChanged.connect(lambda _v: self.completeChanged.emit())

    def initializePage(self):
        p = self.wiz.project
        names = wd.common_point_names(p, min_views=1 if p.n_views < 2 else 2)
        prev = (self.end_a.currentText(), self.end_b.currentText())     # kept on a revisit (Back, Next)
        for cb in (self.end_a, self.end_b):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            cb.blockSignals(False)
        a, b = prev if (prev[0] in names and prev[1] in names) else _guess_wand_names(names)
        if a is not None:
            self.end_a.setCurrentText(a)
        if b is not None:
            self.end_b.setCurrentText(b)
        self._refresh()

    def _refresh(self):
        p = self.wiz.project
        a, b = self.end_a.currentText(), self.end_b.currentText()
        self._n_frames = 0
        if not a or not b or a == b:
            self.table.setText("")
            self.warn.setText("Pick two different landmarks." if a == b and a else "")
            self.completeChanged.emit()
            return
        rows = wd.coverage_summary(p, a, b)
        uv, _ = wd.collect_wand(p, a, b, max_frames=100000)
        self._n_frames = int(uv.shape[0])
        lines = ["<table>"]
        for nm, both, _any in rows:
            lines.append(f"<tr><td>{nm}</td><td style='padding-left:12px'>{both} frames with both ends</td></tr>")
        lines.append("</table>")
        if p.n_views < 2:
            both = rows[0][1] if rows else 0
            lines.append(f"<p>Frames with both ends in this one camera: <b>{both}</b>.</p>")
            self.table.setText("".join(lines))
            self.warn.setText(
                "<span style='color:#FFD60A'>One camera cannot be calibrated on its own: the wand's two "
                "ends give a direction and a scale only when a second camera sees them too. Add the "
                "other cameras' wand videos to this project (＋ Add video, then Sync) and track the "
                "same two names there; this page then counts the shared frames and lets you go on. "
                "Your tracking here is kept and exports normally (Ctrl+E).</span>")
            self.completeChanged.emit()
            return
        lines.append(f"<p>Frames where two or more cameras see both ends: <b>{self._n_frames}</b></p>")
        self.table.setText("".join(lines))
        if self._n_frames < MIN_FRAMES:
            self.warn.setText(f"<span style='color:{theme.RED}'>Too few frames (at least {MIN_FRAMES} are needed). "
                              "Track the wand ends in at least two cameras on the same stretch of the "
                              "recording.</span>")
        elif self._n_frames < GOOD_FRAMES:
            self.warn.setText(f"<span style='color:#FFD60A'>Under {GOOD_FRAMES} shared frames: it will run, but "
                              "the result will be rough. More frames, spread over the recording, help most."
                              "</span>")
        else:
            self.warn.setText("")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        a, b = self.end_a.currentText(), self.end_b.currentText()
        return bool(a and b and a != b and self.length.value() > 0
                    and getattr(self, "_n_frames", 0) >= MIN_FRAMES)


class CamerasPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("The cameras")
        lay = QVBoxLayout(self)
        lay.addWidget(_dim(QLabel("The program needs a starting idea of each lens's focal length (how "
                                  "strongly it magnifies, in pixels). It can find it on its own from the "
                                  "wand alone; that is the right choice unless you have calibrated these "
                                  "exact cameras before.")))
        gb = QGroupBox("Focal length")
        gl = QVBoxLayout(gb)
        self.r_auto = QRadioButton("Find it automatically (recommended)")
        self.r_known = QRadioButton("I know it — start from these values (pixels), then refine:")
        self.r_auto.setChecked(True)
        gl.addWidget(self.r_auto)
        gl.addWidget(self.r_known)
        self.grid = QGridLayout()
        gl.addLayout(self.grid)
        lay.addWidget(gb)
        self.distort = QCheckBox("Also estimate lens distortion (wide-angle / action cameras such as a GoPro)")
        self.distort.setToolTip("Straight lines look bent at the edges of the picture: that is lens "
                                "distortion. Tick this for wide-angle lenses. For ordinary lenses leave it "
                                "off — an extra unknown with nothing to pin it down only adds noise.")
        lay.addWidget(self.distort)
        gb3 = QGroupBox("Lens correction (needed for wide-angle / action cameras; optional otherwise)")
        g3 = QVBoxLayout(gb3)
        g3.addWidget(_dim(QLabel("A wide lens bends straight lines near the edges of the picture; the wand "
                                 "cannot measure that well on its own. Calibrate each such lens once with a "
                                 "printed checkerboard (3D → Calibrate a Lens), or load a saved profile. With a "
                                 "profile the focal length above is taken from it and the tracks are "
                                 "straightened before the solve.")))
        self.lens_grid = QGridLayout()
        g3.addLayout(self.lens_grid)
        lay.addWidget(gb3)
        gb2 = QGroupBox("Extra points (optional, they tighten the result)")
        g2 = QVBoxLayout(gb2)
        g2.addWidget(_dim(QLabel("Any other landmark tracked in two or more cameras — a mark on the "
                                 "floor, the dropped ball, even the animal — adds constraints. Untick "
                                 "anything you do not trust.")))
        self.extras = QListWidget()
        self.extras.setMaximumHeight(120)
        g2.addWidget(self.extras)
        lay.addWidget(gb2)
        lay.addStretch(1)
        self._spins: list[QSpinBox] = []
        self.r_known.toggled.connect(self._toggle_known)

    def initializePage(self):
        p = self.wiz.project
        # a revisit (Back, Next) keeps the user's focal values and unticked extras
        old_f = [sp.value() for sp in self._spins] if len(self._spins) == p.n_views else None
        unticked = {self.extras.item(i).text() for i in range(self.extras.count())
                    if self.extras.item(i).checkState() != Qt.Checked}
        _clear_grid(self.grid)
        self._spins = []
        for c, s in enumerate(p.sessions):
            self.grid.addWidget(QLabel(f"{p.name(c)}  ({s.width}×{s.height})"), c, 0)
            sp = QSpinBox()
            sp.setRange(50, 100000)
            sp.setValue(int(old_f[c]) if old_f else int(round(1.0 * s.width)))
            sp.setSuffix(" px")
            sp.setToolTip("A typical phone or camcorder lens is about 1.0-1.3 x the image width; an "
                          "action camera's wide lens about 0.5 x")
            self.grid.addWidget(sp, c, 1)
            self._spins.append(sp)
        self._toggle_known(self.r_known.isChecked())
        self._refresh_lens_rows()
        wand_names = {self.wiz.page_wand.end_a.currentText(), self.wiz.page_wand.end_b.currentText()}
        self.extras.clear()
        for nm in wd.common_point_names(p):
            if nm in wand_names:
                continue
            it = QListWidgetItem(nm)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked if nm in unticked else Qt.Checked)
            self.extras.addItem(it)

    def _toggle_known(self, on: bool):
        for sp in self._spins:
            sp.setEnabled(on)

    # ---- lens profiles per camera -------------------------------------------
    def _refresh_lens_rows(self):
        p = self.wiz.project
        _clear_grid(self.lens_grid)
        self.lens_labels = []
        self.lens_buttons = []
        for c in range(p.n_views):
            self.lens_grid.addWidget(QLabel(p.name(c)), c, 0)
            prof = p.lenses[c] if c < len(p.lenses) else None
            s = p.sessions[c]
            bad = lens_size_mismatch(prof, s.width, s.height, p.name(c))
            if prof is None:
                text = "no lens profile — the wand estimates the focal length, no distortion correction"
            elif bad:
                text = "attached, but NOT used: " + bad
            else:
                # the picture size (and an Argus file's line) in the row, so a wrong profile shows (I31)
                text = f"{prof.summary()}; for {int(prof.width)}×{int(prof.height)} pictures"
                if "line " in str(prof.source):
                    text += f" ({prof.source})"
                twins = [p.name(j) for j in range(p.n_views) if j != c and j < len(p.lenses)
                         and _same_lens(p.lenses[j], prof)]
                if twins:
                    text += f" — the same profile as {', '.join(twins)}"
            lab = _dim(QLabel(text))
            self.lens_grid.addWidget(lab, c, 1)
            self.lens_labels.append(lab)
            b1 = QPushButton("Calibrate…")
            b1.setToolTip("Open the lens wizard for this camera (needs a video of the printed checkerboard)")
            b1.clicked.connect(lambda _=False, k=c: self._calib_lens(k))
            b2 = QPushButton("Load file…")
            b2.setToolTip("A .klens.json saved earlier, or an Argus / DLTdv camera profile")
            b2.clicked.connect(lambda _=False, k=c: self._load_lens(k))
            b3 = QPushButton("Remove")
            b3.setEnabled(prof is not None)
            b3.clicked.connect(lambda _=False, k=c: self._remove_lens(k))
            # identical cameras share one checkerboard calibration (G40)
            b4 = QPushButton("Use for all")
            fill, replace = share_targets(p, c) if (prof is not None and not bad) else ([], [])
            b4.setEnabled(bool(fill or replace))
            b4.setToolTip(
                f"Use {p.name(c)}'s lens profile for every other camera with {s.width}×{s.height} pictures "
                "as well - right for identical cameras (same model, lens, zoom and recording mode). "
                "The wand still fine-tunes each camera's focal length."
                if (fill or replace) else
                "Attach a lens profile to this camera first; it can then be used for every other camera with "
                "the same picture size." if prof is None else
                "Every other camera with this picture size already uses this profile (or there is none).")
            b4.clicked.connect(lambda _=False, k=c: self._share_lens(k))
            self.lens_grid.addWidget(b1, c, 2)
            self.lens_grid.addWidget(b2, c, 3)
            self.lens_grid.addWidget(b3, c, 4)
            self.lens_grid.addWidget(b4, c, 5)
            self.lens_buttons.append((b1, b2, b3, b4))
        self.lens_grid.setColumnStretch(1, 1)
        # (I35) with a profile on EVERY camera there is nothing left for the tick to fit
        every = self.all_lensed()
        self.distort.setEnabled(not every)
        self.distort.setToolTip(
            "Every camera has a lens profile: its correction is used, so there is no distortion left to "
            "estimate." if every else
            "Straight lines look bent at the edges of the picture: that is lens distortion. Tick this for "
            "wide-angle lenses; cameras that have a lens profile below use the profile instead. For ordinary "
            "lenses leave it off — an extra unknown with nothing to pin it down only adds noise.")

    def _refuse_lens(self, c: int, why: str):
        """Leave camera c without the offered profile and say why on its row."""
        self._refresh_lens_rows()
        if c < len(self.lens_labels):
            self.lens_labels[c].setText(f"<span style='color:{theme.RED}'>not attached: {why}</span>")

    def _calib_lens(self, c: int):
        from kinetrace.lenswizard import LensWizard
        p = self.wiz.project
        lw = LensWizard(self, p, p.sessions[c].video_path, c, self.wiz.start_dir)
        accepted = lw.exec() == QWizard.Accepted
        prof, v, views = lw.result_profile, lw.result_view, (lw.result_views() if accepted else [])
        lw.deleteLater()            # (I249) the wizard holds its scan (~93 MB of thumbnails): release it
        if accepted and prof is not None and v is not None:
            s = p.sessions[v]
            bad = lens_size_mismatch(prof, s.width, s.height, p.name(v))
            if bad:
                self._refuse_lens(v, bad)
                return
            for k in views:          # + the cameras it is shared with (G40)
                p.lenses[k] = prof
            p.dirty = True
        self._refresh_lens_rows()

    def _share_lens(self, c: int):
        """Use camera c's lens profile for every other camera with the same
        picture size (G40); cameras that carry a DIFFERENT profile only after
        a Yes."""
        p = self.wiz.project
        prof = self.usable_lens(c)
        if prof is None:
            return
        fill, replace = share_targets(p, c)
        if replace:
            names = ", ".join(p.name(k) for k in replace)
            ans = QMessageBox.question(
                self, "Replace their lens profiles?",
                f"{names} already {'has' if len(replace) == 1 else 'have'} a lens profile of its own. Replace "
                f"{'it' if len(replace) == 1 else 'them'} with {p.name(c)}'s as well?\n\n"
                "Yes if the cameras are identical (same model, lens, zoom and recording mode); No keeps their "
                "own and shares only with the cameras that have none.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                replace = []
        for k in fill + replace:
            p.lenses[k] = prof
        if fill or replace:
            p.dirty = True
        self._refresh_lens_rows()

    def _load_lens(self, c: int):
        from kinetrace.lenswizard import LENS_FILTER
        p = self.wiz.project
        path, _ = QFileDialog.getOpenFileName(self, "Lens profile", self.wiz.start_dir, LENS_FILTER)
        if not path:
            return
        try:
            # the lens wizard's own dispatch (R19): for an Argus file the camera column decides
            # which line this camera gets, a camera it has no line for is refused (I80, I31)
            prof, which = _lens.read_lens_for(path, c, p.name(c))
        except Exception as e:      # noqa: BLE001
            self.lens_labels[c].setText(f"could not read {Path(path).name}: {e}")
            return
        if prof is None:
            self._refuse_lens(c, which)
            return
        if which:
            prof.source = f"{Path(path).name}{which}"
        s = p.sessions[c]
        bad = lens_size_mismatch(prof, s.width, s.height, p.name(c))
        if bad:
            self._refuse_lens(c, bad)
            return
        p.lenses[c] = prof
        p.dirty = True
        self._refresh_lens_rows()

    def _remove_lens(self, c: int):
        p = self.wiz.project
        if c < len(p.lenses):
            p.lenses[c] = None
            p.dirty = True
        self._refresh_lens_rows()

    def usable_lens(self, c: int):
        """Camera c's lens profile when it matches the camera's picture size,
        else None (I31: a profile attached elsewhere at another size is shown
        on its row as not used, never solved with)."""
        p = self.wiz.project
        prof = p.lenses[c] if c < len(p.lenses) else None
        if prof is None or c >= p.n_views:
            return None
        s = p.sessions[c]
        return None if lens_size_mismatch(prof, s.width, s.height, p.name(c)) else prof

    def usable_lenses(self) -> list:
        """`usable_lens` for every camera, in project order (R20: the list that
        seven callers each built for themselves)."""
        return [self.usable_lens(c) for c in range(self.wiz.project.n_views)]

    def lens_models(self) -> list:
        return [(l.undistort_model() if l is not None else None) for l in self.usable_lenses()]

    def lens_summaries(self) -> list:
        return [(l.summary() if l is not None else None) for l in self.usable_lenses()]

    def focal(self) -> list[float] | None:
        """Per-camera starting focal lengths: the lens profile's where there
        is one, the user's values where they chose "I know it"; NaN for a
        camera with neither (I221: "find it automatically" on a rig where only
        some cameras have a profile - the solve then searches the grid for those
        cameras alone; a width-based guess here made it skip the search and fail
        for most rigs). None when no camera's focal length is known."""
        p = self.wiz.project
        lenses = self.usable_lenses()
        known = self.r_known.isChecked()
        if not any(l is not None for l in lenses) and not known:
            return None
        out = []
        for c, l in enumerate(lenses):
            if l is not None:
                out.append(l.f_square)
            elif known and c < len(self._spins):
                out.append(float(self._spins[c].value()))
            else:
                out.append(float("nan"))
        return out

    def principal(self) -> list | None:
        p = self.wiz.project
        lenses = self.usable_lenses()
        if not any(l is not None for l in lenses):
            return None
        return [(l.principal if l is not None else ((s.width - 1) / 2.0, (s.height - 1) / 2.0))
                for l, s in zip(lenses, p.sessions)]

    def all_lensed(self) -> bool:
        p = self.wiz.project
        return p.n_views > 0 and all(l is not None for l in self.usable_lenses())

    def focal_free_per_camera(self):
        """What `calibrate_wand(estimate_focal=...)` gets (I144). A camera
        without a profile frees every focal length (unchanged, I35); with a
        profile on every camera, a camera SHARING its profile with another
        refines its focal length from the profile's (identical cameras still
        differ by up to ~1 %), a camera with its own profile keeps it."""
        if not self.all_lensed():
            return True
        lenses = self.usable_lenses()
        free = [any(j != c and _same_lens(lenses[j], l) for j in range(len(lenses))) for c, l in enumerate(lenses)]
        return free if any(free) else False

    def distortion_per_camera(self):
        """What `calibrate_wand(estimate_distortion=...)` gets (I35): the tick
        alone without profiles; with some, k1/k2 are fitted for the cameras
        WITHOUT one (a profiled camera's points arrive undistorted already)."""
        on = self.distort.isChecked()
        models = self.lens_models()
        if not any(m is not None for m in models):
            return bool(on)
        return [bool(on and m is None) for m in models]

    def extra_names(self) -> list[str]:
        return [self.extras.item(i).text() for i in range(self.extras.count())
                if self.extras.item(i).checkState() == Qt.Checked]


class FramePage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Which way is up")
        lay = QVBoxLayout(self)
        lay.addWidget(_dim(QLabel("The wand fixes the shape and the size of the world, but not which way "
                                  "is up or where zero is. Pick how to set that.")))
        self.r_drop = QRadioButton("A dropped object (recommended): gravity gives the vertical, and the "
                                   "measured fall checks the scale")
        self.r_axes = QRadioButton("Three reference points: an origin, a point along +X, a point in the "
                                   "XY plane")
        self.r_none = QRadioButton(f"Neither — use {wiz.project.name(0)}'s own axes (fine for shapes and distances)")
        self.r_none.setChecked(True)
        lay.addWidget(self.r_drop)
        self.g_drop = QWidget()
        gl = QFormLayout(self.g_drop)
        gl.setContentsMargins(24, 0, 0, 0)
        self.drop_name = QComboBox()
        self.drop_t0 = QSpinBox()
        self.drop_t1 = QSpinBox()
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.drop_t0)
        rl.addWidget(QLabel("to"))
        rl.addWidget(self.drop_t1)
        self.btn_fill = QPushButton("Find the fall")
        self.btn_fill.setToolTip("Look for the frames where the object falls freely: after it is let go, "
                                 "before it lands, bounces or is caught")
        rl.addWidget(self.btn_fill)
        rl.addStretch(1)
        gl.addRow("The dropped object", self.drop_name)
        gl.addRow("Frames while it falls (reference camera)", row)
        self.drop_note = _dim(QLabel(""))
        gl.addRow("", self.drop_note)
        lay.addWidget(self.g_drop)
        lay.addWidget(self.r_axes)
        self.g_axes = QWidget()
        al = QFormLayout(self.g_axes)
        al.setContentsMargins(24, 0, 0, 0)
        self.ax_o = QComboBox()
        self.ax_x = QComboBox()
        self.ax_y = QComboBox()
        self.ax_t = QSpinBox()
        al.addRow("Origin", self.ax_o)
        al.addRow("A point along +X", self.ax_x)
        al.addRow("A point in the XY plane (towards +Y)", self.ax_y)
        al.addRow("Reference frame where all three are visible", self.ax_t)
        self.ax_note = _dim(QLabel(""))
        al.addRow("", self.ax_note)
        lay.addWidget(self.g_axes)
        lay.addWidget(self.r_none)
        lay.addStretch(1)
        self._fall = None
        for r in (self.r_drop, self.r_axes, self.r_none):
            r.toggled.connect(self._toggle)
        self.btn_fill.clicked.connect(self._fill_drop)
        self.drop_name.currentIndexChanged.connect(self._fill_drop)
        # (I29) the reference frame follows the three points the user picks: it was
        # computed once for the default trio and then ran the solve on a frame where
        # the chosen floor marks did not exist
        for cb in (self.ax_o, self.ax_x, self.ax_y):
            cb.currentIndexChanged.connect(self._fill_axes)
        self.ax_t.valueChanged.connect(lambda _v: self._check_axes())
        for sp in (self.drop_t0, self.drop_t1):
            sp.valueChanged.connect(lambda _v: self.completeChanged.emit())

    def initializePage(self):
        p = self.wiz.project
        names = wd.common_point_names(p)
        wand_names = {self.wiz.page_wand.end_a.currentText(), self.wiz.page_wand.end_b.currentText()}
        others = [n for n in names if n not in wand_names]
        # a revisit (Back, Next) keeps what the user picked and typed here
        prev = None
        if getattr(self, "_visited", False):
            prev = {"drop": self.drop_name.currentText(), "t": (self.drop_t0.value(), self.drop_t1.value()),
                    "axes": self._axes_names(), "ax_t": self.ax_t.value()}
        self._visited = True
        # (I30) any instant two cameras share, not only those EVERY camera recorded;
        # negative when a camera started before the reference one
        t0, t1 = wd.sampling_window(p)
        if t1 < t0:
            t0 = t1 = 0
        for sp in (self.drop_t0, self.drop_t1, self.ax_t):
            sp.blockSignals(True)
            sp.setRange(t0, t1)
            sp.blockSignals(False)
        for cb in (self.drop_name, self.ax_o, self.ax_x, self.ax_y):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(others)
            cb.blockSignals(False)
        ball = [n for n in others if "ball" in n.lower() or "drop" in n.lower()]
        keep_drop = prev is not None and prev["drop"] in others
        keep_axes = prev is not None and all(n in others for n in prev["axes"])
        pick = prev["drop"] if keep_drop else (ball[0] if ball else None)
        if pick:
            self.drop_name.blockSignals(True)
            self.drop_name.setCurrentText(pick)
            self.drop_name.blockSignals(False)
        if keep_axes:
            for cb, nm in zip((self.ax_o, self.ax_x, self.ax_y), prev["axes"]):
                cb.blockSignals(True)
                cb.setCurrentText(nm)
                cb.blockSignals(False)
        elif len(others) >= 3:
            for cb, k in ((self.ax_x, 1), (self.ax_y, 2)):
                cb.blockSignals(True)
                cb.setCurrentIndex(k)
                cb.blockSignals(False)
        self.r_drop.setEnabled(bool(others))
        self.r_axes.setEnabled(len(others) >= 3)
        if keep_drop:
            for sp, v in zip((self.drop_t0, self.drop_t1), prev["t"]):
                sp.setValue(v)
            fall = self._fall
        else:
            fall = self._fill_drop()
        if keep_axes:
            self.ax_t.blockSignals(True)
            self.ax_t.setValue(prev["ax_t"])
            self.ax_t.blockSignals(False)
            self._check_axes()
        else:
            self._fill_axes()
        # (I26) the choice follows the data: a landmark NAMED like a dropped object
        # whose fall was found -> gravity; anything else -> Neither. It used to be
        # "dropped object" whenever any extra landmark existed, which set the
        # vertical from a floor mark's jitter for anyone who just pressed Next.
        # The choice left on an earlier visit is kept unless it is no longer possible.
        cur = self.r_drop if self.r_drop.isChecked() else (self.r_axes if self.r_axes.isChecked() else self.r_none)
        if prev is None or not cur.isEnabled():
            (self.r_drop if (ball and fall is not None) else self.r_none).setChecked(True)
        self._toggle()

    def _fill_drop(self):
        """Set the frames to the object's free fall (`wanddata.find_fall`) and
        say what was found; returns (first, last, note) or None."""
        p = self.wiz.project
        nm = self.drop_name.currentText()
        self._fall = None
        fps = p.sessions[0].fps if p.sessions else 0.0
        rate = f"Reference camera ({p.name(0)}) rate: {fps:.3f} fps. "
        if not nm:
            self.drop_note.setText(rate)
            self.completeChanged.emit()
            return None
        t0, t1 = wd.sampling_window(p)
        if t1 < t0:
            self.completeChanged.emit()
            return None
        fall = wd.find_fall(p, nm, t0, t1)
        if fall is not None:
            a, b, why = fall
            self.drop_t0.setValue(a)
            self.drop_t1.setValue(b)
            self.drop_note.setText(rate + f"“{nm}” falls freely on frames {a}–{b} ({b - a + 1} frames){why}. "
                                   "Only the frames between its release and the moment it lands, bounces or is "
                                   "caught belong here; correct them if they are wrong.")
        else:
            uv = wd.collect_drop(p, nm, t0, t1)
            hits = np.nonzero(np.isfinite(uv).all(axis=2).sum(axis=1) >= 2)[0]
            if len(hits):
                self.drop_t0.setValue(int(t0 + hits[0]))
                self.drop_t1.setValue(int(t0 + hits[-1]))
            self.drop_note.setText(
                rate + f"<span style='color:#FFD60A'>No free fall of “{nm}” was found: it never moves steadily "
                "in one direction for 6 frames or more in two cameras. If it really was dropped, set the frames "
                "of the fall by hand; otherwise choose another way below.</span>")
        self._fall = fall
        self.completeChanged.emit()
        return fall

    def _axes_names(self) -> list[str]:
        return [cb.currentText() for cb in (self.ax_o, self.ax_x, self.ax_y)]

    def _fill_axes(self):
        """Move the reference frame to one where the three chosen points are
        all tracked in two or more cameras (when there is one)."""
        names = self._axes_names()
        if all(names) and len(set(names)) == 3:
            t = wd.best_common_frame(self.wiz.project, names)
            if t is not None:
                self.ax_t.blockSignals(True)
                self.ax_t.setValue(int(t))
                self.ax_t.blockSignals(False)
        self._check_axes()

    def _axes_missing(self) -> list[str]:
        """The chosen reference points NOT seen by two cameras at the frame."""
        p = self.wiz.project
        t = self.ax_t.value()
        return [nm for nm in self._axes_names()
                if not nm or int((np.isfinite(wd.collect_at(p, nm, t)).all(axis=1)).sum()) < 2]

    def _check_axes(self):
        names = self._axes_names()
        if not all(names):
            self.ax_note.setText("")
        elif len(set(names)) < 3:
            self.ax_note.setText(f"<span style='color:{theme.RED}'>Pick three different landmarks.</span>")
        else:
            t = self.ax_t.value()
            missing = self._axes_missing()
            if not missing:
                self.ax_note.setText(f"All three are tracked in two or more cameras on frame {t}.")
            else:
                best = wd.best_common_frame(self.wiz.project, names)
                hint = (f"frame {best} has all three." if best is not None else
                        "they are never tracked together in two cameras: track all three on one common "
                        "frame, or choose another way.")
                self.ax_note.setText(f"<span style='color:{theme.RED}'>On frame {t}, "
                                     f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not tracked "
                                     f"in two cameras; {hint}</span>")
        self.completeChanged.emit()

    def _toggle(self):
        self.g_drop.setEnabled(self.r_drop.isChecked())
        self.g_axes.setEnabled(self.r_axes.isChecked())
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """What the chosen way needs before the run can use it (I29): four
        frames of the drop seen by two cameras, or three different reference
        points all seen by two cameras on the reference frame."""
        m = self.mode()
        if m == "drop":
            p = self.wiz.project
            uv = wd.collect_drop(p, self.drop_name.currentText(), self.drop_t0.value(), self.drop_t1.value())
            return bool(self.drop_name.currentText()) and int(
                (np.isfinite(uv).all(axis=2).sum(axis=1) >= 2).sum()) >= 4
        if m == "axes":
            names = self._axes_names()
            return all(names) and len(set(names)) == 3 and not self._axes_missing()
        return True

    def mode(self) -> str:
        return "drop" if self.r_drop.isChecked() else ("axes" if self.r_axes.isChecked() else "none")


class RunPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Calibrate")
        self.setFinalPage(True)
        lay = QVBoxLayout(self)
        self.summary = _dim(QLabel(""))
        lay.addWidget(self.summary)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Calibrate now")
        self.btn_run.setObjectName("primary")
        self.btn_save = QPushButton("Save calibration files…")
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("A Kinetrace calibration file (.kcal.json) for other projects, a "
                                 "dltCoefs.csv for DLTdv, and the report as text")
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.status = _dim(QLabel(""))
        lay.addWidget(self.status)
        self.report = QTextBrowser()
        lay.addWidget(self.report, 1)
        self.btn_run.clicked.connect(self._run)
        self.btn_save.clicked.connect(self._save)
        self._thread: _CalibThread | None = None
        self._gen = 0                  # bumped by every run and every change of settings (I28)
        self._run_key = None           # the settings the current result / run was made with

    def _settings_key(self) -> tuple:
        """Everything the result depends on: a result made with other settings
        than the pages show now is not the result of these settings (I28)."""
        w = self.wiz
        p = w.project
        f = w.page_frame
        mode = f.mode()
        frame = ((mode, f.drop_name.currentText(), f.drop_t0.value(), f.drop_t1.value()) if mode == "drop" else
                 (mode, *f._axes_names(), f.ax_t.value()) if mode == "axes" else (mode,))
        foc = w.page_cams.focal()
        return (w.page_wand.end_a.currentText(), w.page_wand.end_b.currentText(),
                float(w.page_wand.length.value()), str(w.page_wand.unit.currentData()),
                # NaN = "not known" (I221) must compare equal to itself, which a float NaN does not
                None if foc is None else tuple(float(v) if np.isfinite(v) else None for v in foc),
                tuple(_lens.profile_key(l) for l in w.page_cams.usable_lenses()),      # by content, R20
                bool(w.page_cams.distort.isChecked()), tuple(w.page_cams.extra_names()), frame)

    def initializePage(self):
        w = self.wiz
        a, b = w.page_wand.end_a.currentText(), w.page_wand.end_b.currentText()
        L = w.page_wand.length.value()
        unit = w.page_wand.unit.currentData()
        mode = w.page_frame.mode()
        how = {"drop": f"vertical from the dropped object “{w.page_frame.drop_name.currentText()}”",
               "axes": "axes from three reference points", "none": f"{w.project.name(0)}'s axes"}[mode]
        n_lens = sum(1 for m in w.page_cams.lens_models() if m is not None)
        f0 = w.page_cams.focal()
        guessed = [] if f0 is None else [w.project.name(c) for c, v in enumerate(f0) if not np.isfinite(v)]
        foc = ("focal lengths found automatically" if f0 is None
               else (f"lens profiles on {n_lens} camera(s)" if n_lens else "focal lengths from your values")
               + (f"; focal lengths found automatically for {', '.join(guessed)}" if guessed else ""))
        dist = ""
        if w.page_cams.distort.isChecked():
            # (I35) say what the tick really does once some cameras have a profile
            n_free = w.project.n_views - n_lens
            dist = (", with lens distortion" if not n_lens else
                    f", lens distortion fitted for the {n_free} camera(s) without a profile" if n_free else "")
        self.summary.setText(f"Wand: {a} ↔ {b}, {L:g} {unit}. {foc}{dist}"
                             + f". Extra points: {len(w.page_cams.extra_names())}. Frame: {how}.")
        # (I28) the pages may have changed since the last run (Back, edit, Next):
        # a result - or a run still going - made with other settings is dropped,
        # or Finish would hand over a calibration scaled by the OLD wand length
        running = self._thread is not None and self._thread.isRunning()
        if self._settings_key() != self._run_key or (self.wiz.result_calibration is None and not running):
            self._gen += 1
            self._run_key = None
            if running:
                self._thread.cancel()
            self.wiz.result = self.wiz.result_calibration = self.wiz.gravity = None
            self.btn_save.setEnabled(False)
            self.btn_run.setEnabled(not running)
            self.bar.setValue(0)
            self.status.setText("The earlier run, made with other settings, is stopping; 'Calibrate now' "
                                "comes back in a moment." if running else "")
            self.report.setHtml("<p style='color:%s'>Press <b>Calibrate now</b>. It takes a few seconds to a "
                                "minute.</p>" % theme.TEXT_DIM)
            self.completeChanged.emit()

    def _run(self):
        w = self.wiz
        p = w.project
        a, b = w.page_wand.end_a.currentText(), w.page_wand.end_b.currentText()
        uv, frames = wd.collect_wand(p, a, b, max_frames=400)
        extras = w.page_cams.extra_names()
        bg = wd.collect_extra_points(p, extras) if extras else None
        if bg is not None and bg.shape[0] == 0:
            bg = None
        models = w.page_cams.lens_models()
        any_lens = any(m is not None for m in models)

        def _und(arr):
            """Undistort every camera's pixels with its profile. Arrays are
            (F, C, 2, 2) wand ends, (B, C, 2) points or (N, C, 2) drops: the
            camera is always axis 1, the last axis is xy."""
            if arr is None or not any_lens:
                return arr
            out = np.array(arr, np.float64, copy=True)
            for c, m in enumerate(models):
                if m is None:
                    continue
                sl = out[:, c]
                out[:, c] = m.undistort(sl.reshape(-1, 2)).reshape(sl.shape)
            return out

        raw_uv, raw_bg = uv, bg                    # the clicks as digitized: coverage is measured on these
        uv = _und(uv)
        bg = _und(bg)
        kwargs = dict(wand_uv=uv, wand_length=float(w.page_wand.length.value()),
                      sizes=[(s.width, s.height) for s in p.sessions],
                      # (I247) coverage on the RAW points (a straightened box over the raw picture area read
                      # 33-52 % for a 25 % box); (I248) the report names the REFERENCE frame of an outlier
                      coverage_uv=raw_uv, coverage_bg_uv=raw_bg, frame_ids=frames,
                      focal=w.page_cams.focal(), principal=w.page_cams.principal(), bg_uv=bg,
                      estimate_focal=w.page_cams.focal_free_per_camera(),     # (I144) shared lenses refine
                      estimate_distortion=w.page_cams.distortion_per_camera(),
                      unit=str(w.page_wand.unit.currentData()),
                      names=[p.name(c) for c in range(p.n_views)])        # (I34) messages say cam1, not 0
        mode = w.page_frame.mode()
        drop = axes = None
        if mode == "drop":
            f = w.page_frame
            duv = wd.collect_drop(p, f.drop_name.currentText(), f.drop_t0.value(), f.drop_t1.value())
            drop = {"uv": _und(duv), "fps": float(p.sessions[0].fps),
                    "t0": int(min(f.drop_t0.value(), f.drop_t1.value()))}
        elif mode == "axes":
            f = w.page_frame
            t = f.ax_t.value()
            o, x, y = f._axes_names()
            axes = {"origin": _und(wd.collect_at(p, o, t)[None])[0],
                    "x": _und(wd.collect_at(p, x, t)[None])[0],
                    "xy": _und(wd.collect_at(p, y, t)[None])[0],
                    "names": f"{o}, {x} and {y}", "t": int(t)}
        self._gen += 1
        self._run_key = self._settings_key()
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.bar.setValue(0)
        self.status.setText("Working…")
        self.wiz.result = None
        self.wiz.result_calibration = None
        self.wiz.gravity = None
        self.completeChanged.emit()
        th = _CalibThread(kwargs, mode, drop, axes, models, w.page_cams.lens_summaries(), gen=self._gen)
        th.progress.connect(self._progress)
        th.finished_ok.connect(self._done)
        th.error.connect(self._fail)
        th.finished.connect(self._thread_ended)
        self._thread = th
        th.start()

    def _progress(self, f: float, m: str, gen: int):
        if gen == self._gen:
            self.bar.setValue(int(1000 * f))
            self.status.setText(m)

    def _thread_ended(self):
        """A run is over (finished, failed or cancelled): Calibrate can run again."""
        th = self._thread
        if th is not None:
            th.wait(5000)
        if th is None or not th.isRunning():
            self.btn_run.setEnabled(True)
            if th is not None and th.gen != self._gen and self.wiz.result_calibration is None:
                self.status.setText("")

    def _fail(self, msg: str, gen: int = -1):
        self.btn_run.setEnabled(True)
        if gen != self._gen:                  # a run for settings that have changed since (I28)
            return
        self.status.setText("")
        self.report.setHtml(f"<p style='color:{theme.RED}'><b>Calibration failed.</b> {msg.replace(chr(10), '<br>')}</p>"
                            "<p>Usual causes: a camera that shares too few wand frames with the "
                            "others (track more of the wand in it), the two wand names swapped in one "
                            "camera, or cameras whose offsets are not aligned (CAMERAS panel).</p>")

    def _done(self, res, grav, gen: int = -1):
        self.btn_run.setEnabled(True)
        if gen != self._gen:                  # a run for settings that have changed since (I28)
            return
        self.btn_save.setEnabled(True)
        self.bar.setValue(1000)
        self.status.setText("Done.")
        self.wiz.result = res
        self.wiz.gravity = grav
        # the unit the run was MADE with (res.unit), never the combo's value now (I28)
        cal = res.to_calibration()
        self.wiz.result_calibration = cal
        self.report.setHtml(report_html(res.report, grav, self.wiz.project, cal.unit))
        self.completeChanged.emit()

    def _save(self):
        if self.wiz.result is None:
            return
        start = str(Path(self.wiz.start_dir) / "cameras.kcal.json") if self.wiz.start_dir else "cameras.kcal.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", start,
                                              "Kinetrace calibration (*.kcal.json)")
        if not path:
            return
        written: list[str] = []
        try:
            save_calibration_files(self.wiz.result, self.wiz.gravity, path, written=written)
        except Exception as e:      # noqa: BLE001
            # (G113) a dltCoefs.csv open in Excel / MATLAB, a read-only folder, a full drive: the page
            # says what happened and which files were written before it, instead of a crash notice
            msg = plain_error(e, "The calibration files could not be saved", short=True)
            if written:
                msg += ". Written before that: " + ", ".join(Path(w).name for w in written) + " (incomplete set)"
            self.status.setText(msg)
            return
        msg = "Saved: " + ", ".join(Path(w).name for w in written)
        note = dlt_csv_caveat(self.wiz.result_calibration)
        if note:
            msg += ". " + note
        self.status.setText(msg)

    def isComplete(self) -> bool:
        return self.wiz.result_calibration is not None


# the calibration file itself (no Qt) lives in calibio; these names stay here too
from kinetrace.calibio import calibration_to_kcal, dlt_csv_matlab, load_kcal  # noqa: E402,F401


def _lensed_cameras(cal) -> list[tuple[int, float]]:
    """(camera index, how far its lens correction moves the picture's border
    in px, NaN if unknown) for every camera that carries an undistortion."""
    out = []
    for k, c in enumerate(getattr(cal, "cameras", []) or []):
        und = getattr(c, "undistort", None)
        if und is None or getattr(und, "kind", "none") == "none":
            continue
        bend = float("nan")
        try:
            w, h = int(c.width), int(c.height)
            if w > 0 and h > 0:
                n = 32
                xs, ys = np.linspace(0, w - 1, n), np.linspace(0, h - 1, n)
                border = np.concatenate([np.column_stack([xs, np.zeros(n)]), np.column_stack([xs, np.full(n, h - 1.0)]),
                                         np.column_stack([np.zeros(n), ys]), np.column_stack([np.full(n, w - 1.0), ys])])
                d = np.linalg.norm(np.asarray(und.undistort(border), np.float64) - border, axis=1)
                d = d[np.isfinite(d) & (d < 1e5)]
                if len(d):
                    bend = float(d.max())
        except Exception:      # noqa: BLE001 - the caveat stands without the number
            pass
        out.append((k, bend))
    return out


def dlt_csv_caveat(cal, names: list[str] | None = None) -> str | None:
    """One sentence for the status line / toast when the dltCoefs.csv cannot
    stand on its own (I32), else None: cameras whose calibration carries a
    lens correction have DLT coefficients for UNDISTORTED pixels, and a bare
    11-row csv has nowhere to keep the correction."""
    lensed = _lensed_cameras(cal)
    if not lensed:
        return None
    cams = ", ".join((names[k] if names and k < len(names) else f"camera {k + 1}") for k, _ in lensed)
    return (f"The dltCoefs.csv is for lens-corrected pixels in {cams}: raw video digitized with it in DLTdv "
            "or easyWand comes out wrong near the picture edges (see the _dltCoefs_README.txt; the "
            ".kcal.json keeps the correction).")


def _dlt_readme(cal, csv_name: str, names: list[str] | None) -> str | None:
    lensed = _lensed_cameras(cal)
    if not lensed:
        return None
    rows = []
    for k, bend in lensed:
        nm = names[k] if names and k < len(names) else f"camera {k + 1}"
        kind = getattr(cal.cameras[k].undistort, "kind", "?")
        amount = f"moves the picture's edges by up to {bend:.0f} px" if np.isfinite(bend) else "size not known"
        rows.append(f"  - {nm} (column {k + 1}): {kind} lens correction, {amount}")
    return "\n".join([
        f"About {csv_name}",
        "",
        "These DLT coefficients (11 rows, one column per camera, pixels counted from 1 as in",
        "MATLAB / DLTdv / easyWand) are valid for UNDISTORTED - lens-corrected - pixels in these cameras:",
        *rows,
        "",
        "A dltCoefs.csv has no room for a lens correction. Points digitized on the RAW videos of these",
        "cameras with this file alone come out in the wrong place in 3D, by up to the amounts above near",
        "the picture edges (much less near the centre).",
        "",
        "What to do:",
        "  - in Kinetrace, use the .kcal.json written next to this file: it keeps the correction;",
        "  - in DLTdv or easyWand, give each of these cameras its lens correction as well (or digitize",
        "    videos that were undistorted with the same lens profile).",
        "",
    ])


def _same_cameras(res, cal) -> bool:
    """True when `cal` is the calibration `res` produced: same number of
    cameras, identical DLT coefficients (a project saved and reopened keeps
    them bit for bit)."""
    a, b = list(res.cameras), list(cal.cameras)
    return len(a) == len(b) and all(np.array_equal(np.asarray(x.coefs, np.float64), np.asarray(y.coefs, np.float64))
                                    for x, y in zip(a, b))


def save_calibration_files(res, grav, path: str, cal=None, written: list | None = None) -> list[str]:
    """`*.kcal.json` (everything, for Kinetrace), `*_dltCoefs.csv` (MATLAB
    1-based pixels, for DLTdv / easyWand users), `*_report.txt` when there is
    a report and `*_dltCoefs_README.txt` when a camera carries a lens
    correction the csv cannot hold (I32). `res` may be None when exporting an
    imported calibration (`cal` then required). `written`, when given, is
    appended to AS EACH FILE LANDS, so a caller that catches an error half-way
    can say which files exist (G113)."""
    if cal is None:
        cal = res.to_calibration()
    elif res is not None and not _same_cameras(res, cal):
        # (I33) `cal` is NOT what this wand run produced (an imported calibration, or
        # another project's): its files must not carry the run's verdict and g check
        res, grav = None, None
    report = res.report if res is not None else getattr(cal, "report", {}) or {}
    names = report.get("camera_names") if isinstance(report, dict) else None
    p = Path(path)
    if not p.name.lower().endswith(".kcal.json"):
        p = p.with_name(p.stem + ".kcal.json")
    out = written if written is not None else []
    p.write_text(json.dumps(calibration_to_kcal(cal, report, grav), indent=1), encoding="utf-8")
    out.append(str(p))
    stem = p.name[:-len(".kcal.json")]
    csv = p.with_name(stem + "_dltCoefs.csv")
    dlt_csv_matlab(cal, csv)
    out.append(str(csv))
    readme = _dlt_readme(cal, csv.name, names)
    if readme:
        rd = p.with_name(stem + "_dltCoefs_README.txt")
        rd.write_text(readme, encoding="utf-8")
        out.append(str(rd))
    if report:
        rp = p.with_name(stem + "_report.txt")
        rp.write_text(report_text(report, grav, dlt_csv_caveat(cal, names)), encoding="utf-8")
        out.append(str(rp))
    return list(out)


def report_text(report: dict, grav: dict | None, csv_note: str | None = None) -> str:
    lines = [f"Kinetrace wand calibration report", f"verdict: {report.get('verdict', '?').upper()}", ""]
    for r in report.get("verdict_reasons", []):
        lines.append(f"- {r}")
    lines.append("")
    if csv_note:
        lines += [f"NOTE: {csv_note}", ""]
    keys = ("n_cameras", "camera_names", "n_frames_in", "n_frames_used", "n_bg_points", "outliers_removed",
            "wand_length", "wand_mean", "wand_sd", "wand_score_pct", "reproj_rmse_all", "reproj_rmse_px",
            "focal_px", "focal_estimated", "principal_px", "distortion_estimated", "distortion",
            "lens_profiles", "coverage_pct", "camera_distances", "frame")
    for k in keys:
        if k in report:
            lines.append(f"{k}: {json.dumps(report[k])}")
    if grav:
        lines.append(f"gravity_check: {json.dumps(grav)}")
    return "\n".join(lines) + "\n"


def gravity_colour(pct_off: float) -> str:
    """Colour of the gravity check's figure from how many % away from g it is, with
    wand.py's own bands: green while the sentence says "consistent", amber up to the
    point where the verdict is capped, red beyond (G137: the colours used 3 % / 8 %
    against sentences that used 2 % / 5 %)."""
    if pct_off <= _wand.G_CONSISTENT_PCT:
        return theme.GREEN
    return "#FFD60A" if pct_off <= _wand.G_SUSPECT_PCT else theme.RED


def report_html(report: dict, grav: dict | None, project, unit: str) -> str:
    v = str(report.get("verdict", "poor"))
    col = VERDICT_COLORS.get(v, theme.RED)
    parts = [f"<h2 style='color:{col}; margin-bottom:2px'>{VERDICT_WORDS.get(v, v)}</h2>"]
    parts.append("<ul>" + "".join(f"<li>{r}</li>" for r in report.get("verdict_reasons", [])) + "</ul>")
    if grav and grav.get("applied") is False:
        # (I25) the drop was not used: say so instead of printing a ratio as if it had been
        parts.append(f"<p><b>Gravity check:</b> <span style='color:{theme.RED}'>not used</span> — "
                     f"{grav.get('why', 'the dropped object did not fall freely on those frames')}. "
                     f"The world keeps {project.name(0)}'s axes; the note above says what to change.</p>")
    elif grav:
        g_exp = grav.get("g_expected")
        ratio = grav.get("g_ratio")
        g_meas = float(grav.get("g_measured", float("nan")))
        word = "wand lengths" if unit == "wand" else unit
        if ratio is None or g_exp is None:
            # (I27) no unit to compare with: the fall measures the wand instead
            L_m = grav.get("implied_wand_length_m")
            parts.append(f"<p><b>Gravity check:</b> the dropped object accelerated at {g_meas:.3f} {word}/s². "
                         + (f"If the frame rate is right, your wand is about <b>{float(L_m):.3f} m</b> long "
                            "centre to centre — measure it, then calibrate again with that length in metres."
                            if L_m else "") + "</p>")
        else:
            ratio = float(ratio)
            # (G137) the bands the sentences use (wand.py: consistent <= 2 %, suspect > 5 %)
            gcol = gravity_colour(abs(ratio - 1.0) * 100.0)
            parts.append(f"<p><b>Gravity check:</b> the dropped object accelerated at "
                         f"<span style='color:{gcol}'>{g_meas:.2f} {unit}/s²</span> "
                         f"against the expected {float(g_exp):.2f} {unit}/s² — a ratio of "
                         f"<b>{ratio:.3f}</b>. This is an independent check of the size of your world: "
                         "1.000 means the wand length you typed and the frame rate agree. A ratio far "
                         "from 1 usually means a wrong wand length, a wrong frame rate, or the object was "
                         "not in free fall on those frames.</p>")
    names = [project.name(c) for c in range(project.n_views)]
    rm = report.get("reproj_rmse_px", [])
    cov = report.get("coverage_pct", [])
    foc = report.get("focal_px", [])
    lenses = report.get("lens_profiles") or []
    parts.append("<h3>Per camera</h3><table cellpadding='3'>"
                 "<tr><th align='left'>camera</th><th>reprojection error (px)</th>"
                 "<th>wand coverage of the picture</th><th>focal length (px)</th><th align='left'>lens correction</th></tr>")
    for c, nm in enumerate(names):
        r_ = f"{rm[c]:.2f}" if c < len(rm) else "?"
        c_ = f"{cov[c]:.0f} %" if c < len(cov) else "?"
        f_ = f"{foc[c]:.0f}" if c < len(foc) else "?"
        l_ = (lenses[c] if c < len(lenses) and lenses[c] else "none (pinhole)")
        parts.append(f"<tr><td>{nm}</td><td align='center'>{r_}</td><td align='center'>{c_}</td>"
                     f"<td align='center'>{f_}</td><td>{l_}</td></tr>")
    parts.append("</table>")
    if any(lenses):
        # (I35) with some cameras unprofiled the focal lengths are free for every camera;
        # (I144) with every camera profiled, those SHARING a profile are refined
        ref = list(report.get("focal_refined") or [])
        refined = [names[c] for c in range(min(len(ref), len(names))) if ref[c]]
        if report.get("focal_estimated") and ref and len(refined) < len(ref):
            how_f = (f"come from the checkerboard, and were refined by the wand for {', '.join(refined)} "
                     "(cameras sharing one lens profile: identical cameras still differ a little)")
        else:
            how_f = ("were refined by the wand from the checkerboard's values" if report.get("focal_estimated")
                     else "come from the checkerboard")
        parts.append(f"<p style='color:{theme.TEXT_DIM}'>Cameras with a lens correction were solved on "
                     f"straightened points and keep that correction for 3D; their focal lengths {how_f}.</p>")
        per = report.get("distortion_estimated_per_camera") or []
        fitted = [names[c] for c in range(min(len(per), len(names))) if per[c]]
        if fitted:
            parts.append(f"<p style='color:{theme.TEXT_DIM}'>Lens distortion was fitted by the wand for "
                         f"{', '.join(fitted)} (no lens profile).</p>")
    parts.append(f"<h3>The wand</h3><p>Typed length {report.get('wand_length', 0):g} {unit}; recovered "
                 f"{report.get('wand_mean', 0):.4f} ± {report.get('wand_sd', 0):.4f} {unit} over "
                 f"{report.get('n_frames_used', 0)} of {report.get('n_frames_in', 0)} frames "
                 f"(wand score {report.get('wand_score_pct', 0):.2f} %; "
                 f"{report.get('outliers_removed', 0)} bad observations set aside; "
                 f"{report.get('n_bg_points', 0)} extra points).</p>")
    dists = report.get("camera_distances", [])
    if dists:
        parts.append("<h3>Camera spacing</h3><p>" + "; ".join(
            f"{names[int(i)]} ↔ {names[int(j)]}: {d:.2f} {unit}" for i, j, d in dists) +
            ". Check these against a tape measure between the cameras — a sanity check no number "
            "above can replace.</p>")
    parts.append(f"""<h3 style='margin-top:12px'>What the numbers mean</h3>
<p style='color:{theme.TEXT_DIM}'>
<b>Wand score</b>: how much the recovered wand length varies from frame to frame, as a percentage.
Under 1 % is good; it is the single best measure of the calibration's precision.
<b>Reprojection error</b>: after solving, how far (in pixels) each tracked wand end sits from where
the solved cameras say it should be. Around 1 px or less is good for HD video; large values in one
camera point at that camera (its tracking, its offset, or its lens).
<b>Coverage</b>: how much of each picture the wand visited. Cameras are only known well where the
wand went — keep it above a quarter of the picture and wave it through the whole animal volume.
</p>""")
    return "".join(parts)


class WandWizard(QWizard):
    def __init__(self, parent, project, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Calibrate cameras with a wand")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setMinimumSize(760, 620)
        self.project = project
        self.start_dir = start_dir
        self.result = None
        self.gravity = None
        self.result_calibration = None
        self.page_intro = IntroPage(self)
        self.page_wand = WandPage(self)
        self.page_cams = CamerasPage(self)
        self.page_frame = FramePage(self)
        self.page_run = RunPage(self)
        for pg in (self.page_intro, self.page_wand, self.page_cams, self.page_frame, self.page_run):
            self.addPage(pg)
        # (I36) labelled while the wizard is open - it used to be set in done(),
        # i.e. on a wizard that had already closed, so users only ever saw "Finish"
        self.setButtonText(QWizard.FinishButton, "Use this calibration")

    def done(self, r):
        # a running solve is CANCELLED first and waited for (I36, I200): `stop_page_threads`
        stop_page_threads(self)
        super().done(r)
