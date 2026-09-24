"""3D -> Sync Cameras: whole-frame offsets for hand-started cameras, from the
SOUND tracks (claps, voices - the usual way, `audiosync.py`) or from the
PICTURE (how much it changes frame to frame, `sync.py`).

Written for someone who has never synchronised cameras. The dialog says what
it is about to do, runs the estimator off the GUI thread, and shows one row
per camera: the offset it found, the offset the project has now, and a
verdict in words (clear / weak / none). Only rows the user leaves ticked are
applied, and a "none" row starts unticked. It ends with a hand check: step to
a moment every camera saw and confirm it shows at the same time in each view.
The sound method has a band-pass FILTER (background noise is the rule) and a
whitened correlation that listens to timing rather than loudness; it states
its one blind spot - sound needs time to travel, a few frames across a rig.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from cotracker_app import audiosync, sync, theme

VERDICT_WORDS = {"clear": "CLEAR - use it", "weak": "WEAK - check by eye first",
                 "none": "NONE - do not apply"}
VERDICT_COLORS = {"clear": theme.GREEN, "weak": "#FFD60A", "none": theme.RED}


class _SyncThread(QThread):
    progress = Signal(float)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, paths, rates, ref_range, search, prior=None, audio: dict | None = None):
        super().__init__()
        self.paths, self.rates, self.ref_range, self.search = list(paths), list(rates), ref_range, int(search)
        self.prior = prior
        self.audio = audio          # None = motion; else {fps, band, whiten}
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.audio is None:
                res = sync.estimate_offsets_from_motion(self.paths, self.rates, self.ref_range, self.search,
                                                        progress=self.progress.emit,
                                                        should_cancel=lambda: self._cancel, prior=self.prior)
            else:
                fps = self.audio["fps"]
                f0, f1 = self.ref_range
                res = audiosync.estimate_offsets_from_audio(
                    self.paths, fps, (f0 / fps[0], f1 / fps[0]), self.search / fps[0], prior=self.prior,
                    band=self.audio["band"], whiten=self.audio["whiten"], progress=self.progress.emit,
                    should_cancel=lambda: self._cancel)
        except Exception as e:      # noqa: BLE001
            self.error.emit(str(e))
            return
        self.done.emit(res)


class SyncDialog(QDialog):
    """Pick a stretch of the reference camera, a search range, run, review, apply."""

    def __init__(self, parent, project, paths: list[str], current_frame: int):
        super().__init__(parent)
        self.setWindowTitle("Sync cameras")
        self.project = project
        self.paths = list(paths)
        self.results: list | None = None
        self._thread: _SyncThread | None = None
        lay = QVBoxLayout(self)
        intro = QLabel(
            "<p><b>What this does.</b> Cameras started by hand begin recording seconds apart, so "
            "frame 1000 of one camera is not frame 1000 of another. This lines them up to the "
            "nearest whole frame, two ways:</p>"
            "<p><b>Sound</b> (the usual way): the cameras' microphones all heard the same claps, "
            "voices and knocks. Their sound tracks are compared and slid against each other until "
            "they match. Works whether or not the cameras see the same thing; needs sound tracks. "
            "One blind spot: sound takes time to travel, so a camera several metres farther from "
            "the clap hears it a few frames late at high frame rates.</p>"
            "<p><b>Motion</b> (the picture): <i>how much the picture changes</i> from frame to frame "
            "in every camera, slid until the curves match - a wand swung into view, a person "
            "walking past, a flash leave the same bump in every camera that SAW them.</p>"
            "<p><b>Then check by hand.</b> Step to a moment every camera saw and confirm it appears "
            "at the same time in each view. The fractional part comes later from <i>3D → Estimate "
            "Sub-frame Offsets</i>, once landmarks are tracked.</p>")
        intro.setWordWrap(True)
        intro.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(intro)
        form = QFormLayout()
        # method
        have_ffmpeg, ffmpeg_note = audiosync.ffmpeg_status()
        self.r_audio = QRadioButton("Sound (claps, voices) - recommended")
        self.r_motion = QRadioButton("Motion (how much the picture changes)")
        self.r_audio.setEnabled(have_ffmpeg)
        (self.r_audio if have_ffmpeg else self.r_motion).setChecked(True)
        meth = QWidget()
        mrow = QVBoxLayout(meth)           # stacked: side by side the two labels dictate the dialog width
        mrow.setContentsMargins(0, 0, 0, 0)
        mrow.setSpacing(2)
        mrow.addWidget(self.r_audio)
        mrow.addWidget(self.r_motion)
        form.addRow("Method", meth)
        self.method_note = QLabel("" if have_ffmpeg else ffmpeg_note)
        self.method_note.setWordWrap(True)
        self.method_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.method_note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        form.addRow("", self.method_note)
        # the sound filter: background noise is the rule, not the exception
        filt = QWidget()
        frow = QHBoxLayout(filt)
        frow.setContentsMargins(0, 0, 0, 0)
        self.band_lo = QSpinBox()
        self.band_lo.setRange(0, 3900)
        self.band_lo.setSingleStep(50)
        self.band_lo.setValue(int(audiosync.BAND_DEFAULT[0]))
        self.band_lo.setSuffix(" Hz")
        self.band_lo.setToolTip("Sounds below this are ignored: rumble, wind, traffic, mains hum. 0 = keep all.")
        self.band_hi = QSpinBox()
        self.band_hi.setRange(100, 4000)
        self.band_hi.setSingleStep(100)
        self.band_hi.setValue(int(audiosync.BAND_DEFAULT[1]))
        self.band_hi.setSuffix(" Hz")
        self.band_hi.setToolTip("Sounds above this are ignored (hiss). 4000 = keep all.")
        self.whiten = QCheckBox("noise-robust")
        self.whiten.setChecked(True)
        self.whiten.setToolTip("Listen to the TIMING of sounds, not their loudness (whitened, phase-transform "
                               "correlation): a loud fan or wind cannot out-vote a clap. Recommended; untick only "
                               "for a quiet room where it fails.")
        frow.addWidget(QLabel("keep"))
        frow.addWidget(self.band_lo)
        frow.addWidget(QLabel("to"))
        frow.addWidget(self.band_hi)
        frow.addSpacing(12)
        frow.addWidget(self.whiten)
        frow.addStretch(1)
        self.filter_row = filt
        form.addRow("Sound filter", filt)
        self.r_audio.toggled.connect(lambda on: self.filter_row.setEnabled(on))
        self.filter_row.setEnabled(self.r_audio.isChecked())
        ref_name = project.name(0) if project is not None else "camera 1"
        self.centre = QSpinBox()
        self.centre.setRange(0, 10_000_000)
        # (I12) `current_frame` is the playhead, which is in the ACTIVE camera's own
        # numbering; the stretch is picked in the reference's. Unmapped, a user standing
        # on the clap in a 240 fps camera beside a 120 fps reference got a stretch
        # centred twice as far in, which missed the clap.
        active = int(getattr(project, "active", 0) or 0) if project is not None else 0
        centre0 = int(current_frame)
        if active != 0:
            centre0 = int(round(project.map_frame_exact(active, 0, float(current_frame))))
            centre0 = min(max(centre0, 0), max(int(project.sessions[0].n_frames) - 1, 0))
        self.centre.setValue(centre0)
        self.centre.setToolTip(f"A frame of {ref_name} in the middle of the stretch to compare")
        form.addRow(f"Around frame of {ref_name}", self.centre)
        if active != 0:
            note = QLabel(f"= the instant on screen in {project.name(active)} (its frame {int(current_frame)}), "
                          f"through the current offsets.")
            note.setWordWrap(True)
            note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            note.setStyleSheet(f"color: {theme.TEXT_DIM};")
            form.addRow("", note)
        self.span = QSpinBox()
        self.span.setRange(30, 100_000)
        ref_fps0 = float(project.sessions[0].fps) if project is not None and project.sessions else 30.0
        # 40 s: on a real 8-camera wand recording a 20 s stretch held no
        # movement two given cameras both saw (verdict none, correctly) and
        # 40 s did (correlation 0.84). Longer costs decode time, nothing else.
        self.span.setValue(int(round(40.0 * ref_fps0)))
        self.span.setToolTip("How many frames of the reference camera to compare, centred on the frame above. "
                             "Longer is more reliable (more moments every camera saw or heard); 40 seconds is a "
                             "good default. Sound reads in seconds; the picture over a network share takes a "
                             "minute or two per camera.")
        form.addRow("Stretch (frames)", self.span)
        self.whole = QCheckBox("Use the whole recording instead (slow, but every shared moment counts)")
        self.whole.setToolTip("Reads every frame of every camera at postage-stamp size. Minutes per camera "
                              "over a network share; the surest choice when a shorter stretch finds nothing.")
        form.addRow("", self.whole)
        # The recording clocks in the file names (GoPro: _YYYYMMDD_HHMMSS_) give
        # each camera a prior to within a second; the motion search then only
        # has to cover that second either way instead of tens of seconds.
        fps = [s.fps for s in project.sessions] if project is not None else 30.0
        self.prior = sync.offsets_from_filenames(self.paths, fps) if project is not None else None
        have_prior = self.prior is not None and any(v is not None for v in self.prior[1:])
        ref_fps = float(project.sessions[0].fps) if project is not None and project.sessions else 30.0
        self.search = QSpinBox()
        self.search.setRange(1, 1_000_000)
        self.search.setValue(int(round(2.0 * ref_fps)) if have_prior else int(round(10.0 * ref_fps)))
        self.search.setToolTip("How far apart the cameras may have started, in reference frames each way, "
                               "beyond what the file names already say. "
                               f"At {ref_fps:.0f} fps, {int(round(ref_fps))} frames = 1 second.")
        form.addRow("Search up to (frames)", self.search)
        self.prior_note = QLabel("")
        self.prior_note.setWordWrap(True)
        self.prior_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.prior_note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        if have_prior:
            bits = []
            for i, v in enumerate(self.prior):
                if i == 0 or v is None:
                    continue
                secs = -v / float(fps[i] if isinstance(fps, list) else fps)
                bits.append(f"{project.name(i)} {abs(secs):.0f} s {'later' if secs > 0 else 'earlier'}")
            self.prior_note.setText("The file names carry the recording clocks: " + ", ".join(bits)
                                    + " than the reference (to the second). The search runs around those.")
        elif project is not None and project.n_views > 1:
            self.prior_note.setText("The file names carry no recording clock, so the search starts from "
                                    "the current offsets; widen it if the cameras started far apart.")
        form.addRow("", self.prior_note)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Find the offsets")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        row.addWidget(self.bar, 1)
        lay.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.status)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["apply", "camera", "offset found", "verdict", "offset now"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(140)
        lay.addWidget(self.table, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.btn_apply = QPushButton("Apply the ticked offsets")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        self.buttons.addButton(self.btn_apply, QDialogButtonBox.AcceptRole)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.resize(720, 640)

    # ------------------------------------------------------------------ run
    def _run(self):
        if self.project is None or len(self.paths) < 2:
            self.status.setText("Two cameras are needed.")
            return
        c, half = int(self.centre.value()), int(self.span.value()) // 2
        n_ref = self.project.sessions[0].n_frames
        f0, f1 = max(0, c - half), min(n_ref, c + half)
        if self.whole.isChecked():
            f0, f1 = 0, n_ref
        if f1 - f0 < 30:
            self.status.setText("The stretch is too short: choose a frame inside the reference video.")
            return
        self.btn_run.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.results = None
        self.table.setRowCount(0)
        use_audio = self.r_audio.isChecked()
        self.status.setText(f"Reading the sound of frames {f0}-{f1} of every camera…" if use_audio else
                            f"Reading frames {f0}-{f1} of every camera at postage-stamp size…")
        # search centre per camera: the file-name clock when there is one, else
        # the offset the project has now (the user's own nudging counts)
        prior = [(self.prior[i] if (self.prior is not None and i < len(self.prior) and self.prior[i] is not None)
                  else float(self.project.offsets[i])) for i in range(len(self.paths))]
        audio = None
        if use_audio:
            lo, hi = float(self.band_lo.value()), float(self.band_hi.value())
            band = None if (lo <= 0 and hi >= 4000) else (lo, hi)
            audio = {"fps": [float(s.fps) for s in self.project.sessions], "band": band,
                     "whiten": self.whiten.isChecked()}
        th = _SyncThread(self.paths, self.project.rates, (f0, f1), self.search.value(), prior, audio)
        th.progress.connect(lambda p: self.bar.setValue(int(1000 * p)))
        th.done.connect(self._done)
        th.error.connect(self._fail)
        self._thread = th
        th.start()

    def _fail(self, msg: str):
        self.btn_run.setEnabled(True)
        self.status.setText("Could not read the videos: " + msg)

    def _done(self, results):
        self.results = list(results)
        self.btn_run.setEnabled(True)
        self.bar.setValue(1000)
        self.table.setRowCount(len(self.results))
        n_clear = 0
        for r, cs in enumerate(self.results):
            v = cs.result.verdict
            n_clear += v == "clear"
            tick = QCheckBox()
            tick.setChecked(v != "none")
            self.table.setCellWidget(r, 0, tick)
            self.table.setItem(r, 1, QTableWidgetItem(self.project.name(cs.view)))
            self.table.setItem(r, 2, QTableWidgetItem(f"{cs.offset:+.0f}"))
            it = QTableWidgetItem(VERDICT_WORDS.get(v, v))
            it.setForeground(Qt.GlobalColor.white)
            it.setBackground(Qt.GlobalColor.transparent)
            it.setToolTip(cs.result.why)
            self.table.setItem(r, 3, it)
            self.table.setItem(r, 4, QTableWidgetItem(f"{self.project.offsets[cs.view]:+.3f}"))
        self.table.resizeColumnsToContents()
        n = len(self.results)
        # (I11) name the file that is actually silent, and a camera that did not record
        # the stretch, instead of calling every row "no sound track"
        if any(getattr(cs, "ref_silent", False) for cs in self.results):
            self.status.setText(f"{self.project.name(0)} (the reference): {self.results[0].result.why}")
            self.btn_apply.setEnabled(False)
            return
        n_silent = sum(1 for cs in self.results if getattr(cs, "has_audio", True) is False)
        reach = [self.project.name(cs.view) for cs in self.results if getattr(cs, "out_of_reach", False)]
        if n_silent or reach:
            bits = []
            if n_silent:
                bits.append(f"{n_silent} video(s) have no sound track: use the Motion method for them.")
            if reach:
                bits.append(f"{', '.join(reach)} did not record this stretch (stopped earlier, or started much "
                            "later than assumed): choose a stretch every camera recorded, or widen the search.")
            self.status.setText(" ".join(bits))
            self.btn_apply.setEnabled(any(cs.result.verdict != "none" for cs in self.results))
            return
        if self.r_audio.isChecked():
            fps0 = float(self.project.sessions[0].fps) if self.project.sessions else 30.0
            caveat = " " + audiosync.acoustic_caveat(fps0)
        else:
            caveat = ""
        if n_clear == n:
            msg = (f"Every camera lined up clearly. Press <b>Apply</b>, then step to a moment all cameras "
                   "saw and check it appears at the same time in each view." + caveat)
        elif n_clear:
            msg = (f"{n_clear} of {n} cameras lined up clearly; the others are weak or found nothing -- "
                   "hover a verdict for the reason. Apply the clear ones, then pick a stretch with an "
                   "event those cameras saw (a clap or a hand wave in front of every camera) for the rest."
                   + caveat)
        else:
            msg = ("No camera lined up clearly. The stretch probably holds nothing all cameras "
                   "saw or heard, or the offsets lie outside the search range. Try a stretch around a clap, "
                   "a flash or the wand entering, widen the search, or switch method.")
        self.status.setText(msg)
        self.btn_apply.setEnabled(any(cs.result.verdict != "none" for cs in self.results))

    def chosen(self) -> list[sync.CameraSync]:
        out = []
        for r, cs in enumerate(self.results or []):
            w = self.table.cellWidget(r, 0)
            if w is not None and w.isChecked():
                out.append(cs)
        return out

    def _apply(self):
        applied = 0
        for cs in self.chosen():
            self.project.set_offset(cs.view, float(round(cs.offset)))
            applied += 1
        self.applied = applied
        self.accept()

    def done(self, code):                       # noqa: D401 - Qt override
        if self._thread is not None and self._thread.isRunning():
            self._thread.request_cancel()
            self._thread.wait(30000)
        super().done(code)
