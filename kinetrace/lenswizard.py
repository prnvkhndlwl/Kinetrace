"""Lens calibration wizard — 3D → Calibrate a Lens (checkerboard).

For someone who has never heard the words "intrinsic" or "distortion": the
pages say when a lens needs calibrating at all, hand out a printable
checkerboard, tell the user how to film it, find the board in the video on
their behalf, and end with a verdict in words plus a before / after picture.
The result is a `lens.LensProfile` attached to one camera of the project
(`project.lenses[view]`) and, optionally, a `.klens.json` file for other
projects and other days with the same camera at the same zoom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton,
                               QSizePolicy, QSpinBox, QTextBrowser, QVBoxLayout, QWidget, QWizard,
                               QWizardPage)

from kinetrace import gpmf, lens, theme
from kinetrace.errors import plain_error
from kinetrace.video_source import display_rotation
from kinetrace.boardreview import BoardReview, pixmap

VERDICT_COLORS = {"good": theme.GREEN, "ok": "#FFD60A", "poor": theme.RED}
VERDICT_WORDS = {"good": "GOOD — attach this profile to the camera",
                 "ok": "USABLE — but read the notes below",
                 "poor": "NOT GOOD ENOUGH — film the board again as the notes say"}
LENS_FILTER = ("Lens profiles (*.klens.json *.json *.yml *.yaml *.txt);;Kinetrace lens (*.klens.json);;"
               "OpenCV lens (*.yml *.yaml *.json);;Argus / DLTdv camera profile (*.txt);;All files (*)")
UNITS = [("millimetres (mm)", 0.001), ("centimetres (cm)", 0.01), ("metres (m)", 1.0), ("inches (in)", 0.0254)]
ID_INTRO, ID_VIDEO, ID_REVIEW, ID_RESULT = range(4)


class _Call(QThread):
    def __init__(self, fn):
        super().__init__()
        self._fn, self.result, self.exc = fn, None, None

    def run(self):
        try:
            self.result = self._fn()
        except BaseException as e:  # noqa: BLE001 - raised again on the GUI thread
            self.exc = e


def _off_thread(owner: QWidget, fn):
    """`fn()` on a worker thread while the wizard keeps repainting (a local event
    loop; the wizard is disabled meanwhile, so nothing is clicked twice) -> its
    result. The fits of the review page ran on the GUI thread: seconds of a
    frozen wizard with a status line that never repainted (G51)."""
    th = _Call(fn)
    loop = QEventLoop()
    th.finished.connect(loop.quit)
    owner.setEnabled(False)
    QApplication.setOverrideCursor(Qt.BusyCursor)
    try:
        th.start()
        if not th.isFinished():
            loop.exec()
        th.wait()
    finally:
        QApplication.restoreOverrideCursor()
        owner.setEnabled(True)
    if th.exc is not None:
        raise th.exc
    return th.result


def _dim(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {theme.TEXT_DIM};")
    return label


def stop_page_threads(wiz) -> None:
    """Cancel, then WAIT for, the worker of every page of a wizard before it
    goes away - the one `done()` of the lens and the wand wizard (I200, R20). A
    QThread whose Python wrapper is collected while Qt still owns it takes the
    process down at exit with 0xC0000409; a worker is cancelled first so it
    stops at its next step, and the wait has no cap: the old 15 s cap closed the
    lens wizard over a scan that was still running."""
    for pid in wiz.pageIds():
        th = getattr(wiz.page(pid), "_thread", None)
        if th is None:
            continue
        if hasattr(th, "cancel"):
            th.cancel()
        if th.isRunning():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                th.wait()
            finally:
                QApplication.restoreOverrideCursor()


class _LensThread(QThread):
    """Looks for the board in the video: the scan only. The boards are then
    reviewed and a subset chosen before anything is fitted to them (the fit is
    the review page's, off this thread)."""
    progress = Signal(float, str)
    finished_ok = Signal(object)              # ScanResult
    error = Signal(str)

    def __init__(self, video: str, pattern: tuple[int, int]):
        super().__init__()
        self.video, self.pattern = video, pattern
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            scan = lens.scan_video(self.video, self.pattern, max_candidates=240,
                                   progress=lambda f, m: self.progress.emit(0.85 * f, m),
                                   should_cancel=lambda: self._cancel)
            if self._cancel:
                self.error.emit("cancelled")
                return
            if len(scan.corners) < 3:
                self.error.emit(f"the board was found in only {len(scan.corners)} of {scan.n_scanned} frames "
                                "looked at. Check the square count (inner corners, not squares), make sure the "
                                "whole board is in the picture and sharp, and that it is the board this wizard "
                                "printed (or any plain black-and-white checkerboard).")
                return
            self.progress.emit(1.0, f"{len(scan.corners)} boards found")
            self.finished_ok.emit(scan)
        except Exception as e:      # noqa: BLE001
            self.error.emit(str(e))


class IntroPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Calibrating a lens")
        lay = QVBoxLayout(self)
        t = QTextBrowser()
        t.setHtml(f"""
<p><b>Do you need this?</b> Every lens bends the picture a little; wide-angle lenses and
action cameras (GoPro and the like) bend it a lot — straight lines curve near the edges of
the frame. The wand calibration works out where the cameras are and how strongly they
magnify, but it assumes straight lines stay straight. If your lens bends them by more than a
few pixels, anything tracked near the edges of the picture would be placed wrongly in 3D by
about that much. <b>Calibrating the lens measures the bending so it can be undone.</b></p>
<ul>
<li><b>Ordinary lens</b> (a camcorder, a phone at 1x, a DSLR with a normal lens, a
machine-vision camera with a 8&nbsp;mm+ lens): usually not needed. The wand alone is fine.</li>
<li><b>Wide-angle or action camera</b> (GoPro, 360 rigs, "wide" or "superwide" modes, a
fisheye): needed. Do it once per camera and zoom setting; the profile is reused.</li>
<li><b>Not sure:</b> film the board and run this; the report tells you in pixels how much
your lens bends the edges, and whether it matters.</li>
</ul>
<p><b>What to do.</b></p>
<ol>
<li>Print the checkerboard (button below) at <b>100&nbsp;% / "actual size"</b>. Glue or tape it
to something perfectly flat and stiff — foam board, a clipboard, a piece of glass. Measure one
square with a ruler; it should be 24&nbsp;mm. Type the size you measure. If you use your own
board, keep <b>one count of squares odd and the other even</b> (the printed one is 10 × 7):
that is what lets the program tell which way up the board is on every frame.</li>
<li>With the camera set exactly as for the experiment (same zoom, same resolution, same
frame rate), film the board for 20–30 seconds: fill about a quarter of the picture with it,
move it slowly to <b>every edge and every corner</b>, near and far, and <b>tilt it</b> left,
right, up and down by 20–40 degrees. Keep it sharp: no motion blur. <b>Hold the board by its
edge or a handle behind it</b> — fingers over the squares or over the white border around
them make that frame unusable — and keep the whole board, border included, inside the
picture. Any rotation is fine: the program recognises the board's corners by the black square
beside them, so its axes stay attached to the board however you turn it.</li>
<li>Point the next page at that video. The program finds the board on its own.</li>
</ol>
<p style="color:{theme.TEXT_DIM}">Method: OpenCV's pinhole + radial model, or the fisheye model for wide
lenses (chosen automatically). The report's thresholds are conservative on purpose.</p>
""")
        lay.addWidget(t, 1)
        row = QHBoxLayout()
        self.btn_board = QPushButton("Save the checkerboard to print…")
        self.btn_board.clicked.connect(self._save_board)
        row.addWidget(self.btn_board)
        self.note = _dim(QLabel(""))
        row.addWidget(self.note, 1)
        lay.addLayout(row)

    def _save_board(self):
        start = str(Path(self.wiz.start_dir) / "kinetrace_checkerboard.png") if self.wiz.start_dir else "kinetrace_checkerboard.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save checkerboard", start, "PNG image (*.png)")
        if not path:
            return
        try:
            wmm, hmm = lens.save_checkerboard_png(path)
        except Exception as e:      # noqa: BLE001
            # (I222) a folder that cannot be written, a full drive: said on the page, not a crash notice
            self.note.setText(plain_error(e, "The checkerboard could not be saved", short=True))
            return
        self.note.setText(f"Saved {Path(path).name}: {wmm:.0f} × {hmm:.0f} mm at 300 dpi — print at 100 %, "
                          "landscape, on A4 or Letter.")


class VideoPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("The checkerboard video")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        prow = QWidget()
        pl = QHBoxLayout(prow)
        pl.setContentsMargins(0, 0, 0, 0)
        self.path = QLabel(self.wiz.default_video or "no video chosen")
        # the pre-filled video is simply the one open in the program, often the
        # animal or wand film rather than the checkerboard (G9): say so
        self.prefill_note = QLabel("This is the video open in the program. If it is not the film of the "
                                   "checkerboard, press Choose… and pick that one." if self.wiz.default_video else "")
        self.prefill_note.setWordWrap(True)
        self.prefill_note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.path.setWordWrap(True)
        btn = QPushButton("Choose…")
        btn.clicked.connect(self._pick)
        pl.addWidget(self.path, 1)
        pl.addWidget(btn)
        form.addRow("Video of the board", prow)
        form.addRow("", self.prefill_note)
        self.cam = QComboBox()
        for i in range(self.wiz.project.n_views if self.wiz.project is not None else 0):
            self.cam.addItem(self.wiz.project.name(i), i)
        self.cam.setToolTip("The camera this lens belongs to: the profile is attached to it")
        form.addRow("This is the lens of camera", self.cam)
        brow = QWidget()
        bl = QHBoxLayout(brow)
        bl.setContentsMargins(0, 0, 0, 0)
        self.cols = QSpinBox()
        self.cols.setRange(3, 30)
        self.cols.setValue(lens.DEFAULT_PATTERN[0])
        self.rows = QSpinBox()
        self.rows.setRange(3, 30)
        self.rows.setValue(lens.DEFAULT_PATTERN[1])
        bl.addWidget(self.cols)
        bl.addWidget(QLabel("×"))
        bl.addWidget(self.rows)
        bl.addWidget(_dim(QLabel("inner corners — where four squares meet; the printed board has 9 × 6")))
        bl.addStretch(1)
        form.addRow("Board size", brow)
        # A symmetric board (8 x 6) cannot show which way up it is; say so the
        # moment it is typed, not after a scan.
        self.pattern_note = _dim(QLabel(""))
        self.pattern_note.setWordWrap(True)
        self.pattern_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.pattern_note.setStyleSheet(f"color: {VERDICT_COLORS['ok']};")
        form.addRow("", self.pattern_note)
        self.cols.valueChanged.connect(self._pattern_changed)
        self.rows.valueChanged.connect(self._pattern_changed)
        self._pattern_changed()
        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        self.square = QDoubleSpinBox()
        self.square.setDecimals(3)
        self.square.setRange(0.001, 10000.0)
        self.square.setValue(lens.DEFAULT_SQUARE_MM)
        self.unit = QComboBox()
        for label, _ in UNITS:
            self.unit.addItem(label)
        sl.addWidget(self.square)
        sl.addWidget(self.unit)
        sl.addWidget(_dim(QLabel("measured with a ruler on the printed board")))
        sl.addStretch(1)
        form.addRow("One square", srow)
        mrow = QWidget()
        ml = QVBoxLayout(mrow)
        ml.setContentsMargins(0, 0, 0, 0)
        self.r_auto = QRadioButton("Not sure — try both models and keep the better (recommended)")
        self.r_std = QRadioButton("Ordinary lens")
        self.r_fish = QRadioButton("Wide-angle / action camera / fisheye")
        # (G146) shown only for GoPro footage that carries GoPro's lens model
        self.r_gopro = QRadioButton("GoPro lens from the video + your boards (recommended for this GoPro video)")
        self.r_gopro.setToolTip("The lens curve is GoPro's own model, read from the video, and holds over the whole "
                                "picture; the boards only measure this camera's focal length and centre, so they do "
                                "not need to reach the corners.")
        self.r_gopro.setVisible(False)
        self.r_auto.setChecked(True)
        for r in (self.r_gopro, self.r_auto, self.r_std, self.r_fish):
            ml.addWidget(r)
        self.gopro_note = _dim(QLabel(""))
        self.gopro_note.setWordWrap(True)
        self.gopro_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        ml.addWidget(self.gopro_note)
        form.addRow("Lens type", mrow)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Find the boards")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_load = QPushButton("I already have a lens file…")
        self.btn_load.setToolTip("A .klens.json from an earlier session, or an Argus / DLTdv camera profile")
        self.btn_load.clicked.connect(self._load)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_load)
        self.btn_gopro = QPushButton("Use GoPro's lens without boards")
        self.btn_gopro.setToolTip("GoPro's own lens model from the video, as it is: no checkerboard needed. It is "
                                  "the lens design, not this camera (about 1 % in focal length and ~10 px in centre "
                                  "between units); the wand calibration refines the focal length.")
        self.btn_gopro.clicked.connect(self._use_gopro_nominal)
        self.btn_gopro.setVisible(False)
        row.addWidget(self.btn_gopro)
        row.addStretch(1)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.status = _dim(QLabel(""))
        lay.addWidget(self.status)
        lay.addStretch(1)
        self._thread: _LensThread | None = None
        # the lens file chosen with "I already have a lens file", kept so an
        # Argus file's line can follow the camera combo (I79 / I80)
        self._loaded_path: str | None = None
        # the GoPro header read per (video, size, mtime): Back / Next does not read the file again
        self._gopro_read: dict = {}
        self.cam.currentIndexChanged.connect(self._camera_changed)

    def initializePage(self):
        if self.wiz.default_view is not None and 0 <= self.wiz.default_view < self.cam.count():
            self.cam.setCurrentIndex(self.wiz.default_view)
        self._detect_gopro(self.path.text().strip())

    def _detect_gopro(self, video: str) -> None:
        """(G145, G146) Is the board video GoPro footage carrying GoPro's lens model? Only then are the
        GoPro choices shown (other cameras' metadata is not known). Reads the file's header only."""
        info = None
        if video and Path(video).is_file():
            st = Path(video).stat()
            key = (video, st.st_size, st.st_mtime_ns)
            if key not in self._gopro_read:
                self._gopro_read[key] = gpmf.read_safe(video, sensors=False)
            info = self._gopro_read[key]
        self.wiz.gopro = info if (info is not None and info.has_lens) else None
        on = self.wiz.gopro is not None
        self.r_gopro.setVisible(on)
        self.btn_gopro.setVisible(on)
        if on:
            self.r_gopro.setChecked(True)
            self.gopro_note.setText(f"GoPro footage: {info.label}, stabilisation {info.stabilisation}. Its lens model "
                                    "is in the file." + (" Stabilisation was ON in this video: re-film the board with "
                                                         "HyperSmooth / EIS off, or no lens calibration will hold."
                                                         if info.stabilised else ""))
        else:
            if self.r_gopro.isChecked():
                self.r_auto.setChecked(True)
            self.gopro_note.setText("")
        self.gopro_note.setVisible(bool(self.gopro_note.text()))

    def _use_gopro_nominal(self) -> None:
        """GoPro's lens model as it is (no boards): straight to the result page, like a lens file."""
        info = self.wiz.gopro
        prof = gpmf.lens_profile(info) if info is not None else None
        if prof is None:
            self.status.setText("This video carries no GoPro lens model.")
            return
        self.wiz.scan = None
        self.wiz.used_boards = None
        self._loaded_path = None
        self.wiz.result_profile = prof
        self.wiz.result_view = self._view()
        problem = self._size_problem()
        self.status.setText(f"GoPro's lens model: {prof.summary()}. " + (problem if problem else "Press Next."))
        self.completeChanged.emit()

    def _pick(self):
        start = self.wiz.default_video or self.wiz.start_dir
        path, _ = QFileDialog.getOpenFileName(self, "Checkerboard video", start,
                                              "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.mpg);;All files (*)")
        if path:
            self.path.setText(path)
            self.prefill_note.setText("")      # chosen on purpose now (G9)
            self._detect_gopro(path)
            self.completeChanged.emit()

    def pattern(self) -> tuple[int, int]:
        return int(self.cols.value()), int(self.rows.value())

    def _pattern_changed(self, *_):
        self.pattern_note.setText(lens.orientation_note(self.pattern()))
        self.pattern_note.setVisible(bool(self.pattern_note.text()))
        if self.wiz.scan is not None and self.pattern() != tuple(self.wiz.pattern):
            # (G115) the boards were found for ANOTHER board size: they are not this board's corners
            self.wiz.scan = None
            self.wiz.used_boards = None
            self.wiz.result_profile = None
            self.status.setText("The board size changed: press 'Find the boards' again.")
            self.completeChanged.emit()

    def square_m(self) -> float:
        return float(self.square.value()) * UNITS[self.unit.currentIndex()][1]

    def model(self) -> str:
        if self.wiz.gopro is not None and self.r_gopro.isChecked():
            return "gopro"
        return "auto" if self.r_auto.isChecked() else ("fisheye" if self.r_fish.isChecked() else "standard")

    def _run(self):
        video = self.path.text().strip()
        if not video or not Path(video).exists():
            self.status.setText("Choose the video of the board first.")
            return
        self._lock_inputs(True)
        self.bar.setValue(0)
        self.wiz.result_profile = None
        self.wiz.scan = None
        self.wiz.used_boards = None
        self._loaded_path = None
        self.completeChanged.emit()
        th = _LensThread(video, self.pattern())
        th.progress.connect(lambda f, m: (self.bar.setValue(int(1000 * f)), self.status.setText(m)))
        th.finished_ok.connect(self._done)
        th.error.connect(self._fail)
        self._thread = th
        th.start()

    def _lock_inputs(self, scanning: bool):
        """While the scan runs, the board size and its buttons stay as they were
        when it started (G115: a size changed mid-scan made OpenCV raise, or fitted
        boards found for another size)."""
        for w_ in (self.btn_run, self.btn_load, self.cols, self.rows):
            w_.setEnabled(not scanning)

    def _fail(self, msg: str):
        self._lock_inputs(False)
        self.status.setText("Calibration failed: " + msg)

    def _done(self, scan):
        self._lock_inputs(False)
        self.bar.setValue(1000)
        self.wiz.scan = scan
        # (G115) the board size the scan USED, not whatever the boxes show now
        th = self._thread
        self.wiz.pattern = tuple(th.pattern) if th is not None else self.pattern()
        for box, v in ((self.cols, self.wiz.pattern[0]), (self.rows, self.wiz.pattern[1])):
            box.blockSignals(True)                  # the boxes show the size the boards were found for
            box.setValue(int(v))
            box.blockSignals(False)
        self._pattern_changed()
        self._read_settings()
        self.wiz.result_profile = None          # the review page fits
        self.wiz.result_view = self._view()
        self._show_scan_status()
        self.completeChanged.emit()

    def _read_settings(self):
        """The square size and lens type as the boxes show them NOW: read when Next
        is pressed, not when the scan ended (G115: a fisheye chosen after the scan
        was ignored)."""
        self.wiz.square = self.square_m()
        self.wiz.model = self.model()
        base = gpmf.lens_profile(self.wiz.gopro) if self.wiz.model == "gopro" else None
        scan = self.wiz.scan
        if base is not None and scan is not None:
            # GoPro's model is in the STORED picture: turned to the board video's pictures when its
            # rotation tag turns them (a board filmed with the camera on its side)
            base = lens.fit_profile(base, scan.size, scan.rotation)[0] or base
        self.wiz.base_lens = base

    def _view(self) -> int | None:
        return int(self.cam.currentData()) if self.cam.count() else None

    def _show_scan_status(self):
        scan = self.wiz.scan
        problem = self._size_problem()
        self.status.setText(f"{len(scan.corners)} boards found in {scan.n_scanned} frames. "
                            + (problem if problem else "Press Next to look through them."))

    def _size_problem(self) -> str | None:
        """Why the lens on this page cannot go to the chosen camera, or None.

        A profile is in pixels of the picture it was measured on; attached to
        a camera recording another size, its focal length and centre are
        wrong there and the wand solve holds them fixed (lens#2, the lens
        wizard's side of I31). This page is where the camera is chosen and
        the result page's Finish button says "Attach to" it, so the check
        lives here and covers every caller of the wizard."""
        p, view = self.wiz.project, self._view()
        if p is None or view is None or not (0 <= view < p.n_views):
            return None
        s = p.sessions[view]
        cam = (int(s.width), int(s.height))
        rot = display_rotation(s.video_path)
        if self.wiz.scan is not None:       # a board video filmed level fits a camera filmed on its side
            return lens.turn_needed(self.wiz.scan.size, self.wiz.scan.rotation, cam, rot, p.name(view),
                                    "The checkerboard video has")[1]
        prof = self.wiz.result_profile
        if prof is not None:
            fitted, why = lens.fit_profile(prof, cam, rot, p.name(view))
            return why if fitted is None else None
        return None

    def _camera_changed(self, *_):
        if self.wiz.scan is None and self._loaded_path is not None:
            self._apply_loaded()            # an Argus file has one line per camera (I80)
            return
        if self.wiz.scan is not None:
            self._show_scan_status()
        self.completeChanged.emit()

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Lens profile", self.wiz.start_dir, LENS_FILTER)
        if not path:
            return
        self._loaded_path = path
        self.wiz.scan = None
        self.wiz.used_boards = None
        self._apply_loaded()

    def _apply_loaded(self):
        """Read the chosen lens file for the camera the combo names."""
        path = self._loaded_path
        view = self._view()
        self.wiz.result_profile = None
        try:
            # (I80) an Argus file gives THIS camera its own line, or a refusal -- never the last
            # line reused for a camera the file does not list; the dispatch is the wand wizard's too
            name = self.wiz.project.name(view) if (self.wiz.project is not None and view is not None) else ""
            prof, which = lens.read_lens_for(path, view if view is not None else 0, name)
        except Exception as e:      # noqa: BLE001
            self.status.setText(f"Could not read {Path(path).name}: {e}")
            self.completeChanged.emit()
            return
        self.wiz.result_view = view
        if prof is None:
            self.status.setText(which)
        else:
            self.wiz.result_profile = prof
            problem = self._size_problem()
            self.status.setText(f"Loaded {Path(path).name}{which}: {prof.summary()}. "
                                + (problem if problem else "Press Next."))
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        if self.wiz.result_profile is None and self.wiz.scan is None:
            return False
        return self._size_problem() is None

    def validatePage(self) -> bool:
        # (I79) the lens goes to the camera the page shows when Next is
        # pressed, not the one the combo showed when the scan finished
        self.wiz.result_view = self._view()
        problem = self._size_problem()
        if problem:
            self.status.setText(problem)
            return False
        if self.wiz.scan is not None:
            self._read_settings()                  # (G115) the lens type and square as set NOW
        return True

    def nextId(self) -> int:
        # (I76) a profile loaded from a file has no boards to review: go
        # straight to the result page, which carries the Attach button
        if self.wiz.scan is None and self.wiz.result_profile is not None:
            return ID_RESULT
        return super().nextId()


class ReviewPage(QWizardPage):
    """Look at every board, fix a corner, choose which views to fit from.

    A lens calibration cannot be checked by reading its numbers, so this page
    shows the evidence: the corners, the order they were found in, the board's
    own axes, and how far each view reprojects. The automatic choice is the
    starting point, not the only option.
    """

    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Check the boards")
        self.setSubTitle("Every board found, with its corners and the board's own axes. "
                         "Untick any you do not trust, or click one to drag a corner.")
        lay = QVBoxLayout(self)
        self.review = BoardReview()
        self.review.runner = lambda fn: _off_thread(self.wiz, fn)      # "Best spread" off the GUI thread too
        self.review.changed.connect(self._review_changed)
        lay.addWidget(self.review, 1)
        row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit the lens from the ticked boards")
        self.btn_fit.setObjectName("primary")
        self.btn_fit.clicked.connect(self._fit)
        row.addWidget(self.btn_fit)
        row.addStretch(1)
        self.status = _dim(QLabel(""))
        row.addWidget(self.status, 1)
        lay.addLayout(row)
        self._ready = False
        # (ticked boards, corner-edit revision) the current profile was fitted from
        self._fitted: tuple | None = None

    def _mark_fitted(self, idx: list[int], prof, rev: int) -> None:
        """Record what `prof` was fitted from: the ticked boards and the corner-edit
        revision READ BEFORE the fit started (G114: read after it, an edit that
        landed meanwhile counted as fitted)."""
        self._fitted = (tuple(idx), int(rev))
        # the boards the fit really used, in scan numbering: calibrate_lens may
        # set a few aside (I78)
        used = (prof.report or {}).get("views_used")
        self.wiz.used_boards = ([idx[k] for k in used if 0 <= k < len(idx)] if used is not None
                                else list(idx))

    def _now(self) -> tuple:
        return (tuple(self.review.chosen()), self.review.corner_rev)

    def _review_changed(self):
        # (I74) the profile belongs to the boards it was fitted from. Unticking
        # a board or dragging a corner after the fit makes it stale: Next waits
        # for a refit, or for the choice to be put back as it was.
        was = self._ready
        self._ready = self._fitted is not None and self._now() == self._fitted
        if was and not self._ready:
            self.status.setText("The boards changed since the fit: press 'Fit the lens from the ticked "
                                "boards' before going on.")
        elif self._ready and not was and self.wiz.result_profile is not None:
            self.status.setText(f"Back to the boards of the fit: {self.wiz.result_profile.summary()}")
        self.completeChanged.emit()

    def initializePage(self):
        scan = self.wiz.scan
        self._ready = False
        self._fitted = None
        if scan is None:
            return
        self.status.setText(f"Choosing a good spread of the {len(scan.corners)} boards and fitting the lens "
                            "(a few seconds)…")
        pattern, square, model = self.wiz.pattern, self.wiz.square, self.wiz.model
        base = self.wiz.base_lens                         # (G146) GoPro's curve for model "gopro"
        corners = list(scan.corners)
        rev = self.review.corner_rev
        self.review.base = base

        def work():
            idx, err, why = lens.auto_select(corners, pattern, square, scan.size, model, base=base)
            prof = None
            if idx:
                try:
                    prof = lens.calibrate_lens([corners[i] for i in idx], pattern, square, scan.size, model,
                                               max_views=max(len(idx), lens.MAX_VIEWS), base=base)
                    prof.rotation = scan.rotation       # how the board video's pictures were turned
                    # the errors of the profile that is SHOWN, not of the provisional fit auto_select made
                    err = lens.per_view_errors(corners, pattern, square, prof)
                except Exception:                               # noqa: BLE001
                    prof = None
            return idx, err, why, prof
        idx, err, why, prof = _off_thread(self.wiz, work)
        self.review.set_scan(scan, pattern, square, prof, idx, err, model=model)
        if prof is not None:
            self.wiz.result_profile = prof
            self._mark_fitted(idx, prof, rev)
            self._ready = self._now() == self._fitted
            self.status.setText(f"Fitted: {prof.summary()}")
        else:
            self.status.setText(why)
        self.completeChanged.emit()

    def _fit(self):
        scan = self.wiz.scan
        idx = self.review.chosen()
        if scan is None or len(idx) < 3:
            self.status.setText("Tick at least three boards.")
            return
        self.btn_fit.setEnabled(False)
        self.status.setText(f"Fitting from {len(idx)} boards…")
        # (G114) the corners this fit is made from, and their revision, are read HERE, before the
        # off-thread fit: an edit that lands during it must make the profile stale, not "fitted"
        rev = self.review.corner_rev
        pattern, square, model = self.wiz.pattern, self.wiz.square, self.wiz.model
        chosen = [scan.corners[i].copy() for i in idx]
        everything = list(scan.corners)

        def work():
            # (G116) every ticked board is fitted: the boards were CHOSEN, so calibrate_lens must not
            # thin them to a 40-view spread behind a "Fitted from 120 boards"
            prof = lens.calibrate_lens(chosen, pattern, square, scan.size, model, max_views=len(chosen),
                                       base=self.wiz.base_lens)
            prof.rotation = scan.rotation               # how the board video's pictures were turned
            return prof, lens.per_view_errors(everything, pattern, square, prof)
        try:
            prof, err = _off_thread(self.wiz, work)
        except Exception as exc:                                # noqa: BLE001
            self.btn_fit.setEnabled(True)
            self.status.setText(plain_error(exc, "The lens could not be fitted from these boards", short=True))
            return
        self.review.set_fit(prof, err, "refitted")
        self.wiz.result_profile = prof
        self._mark_fitted(idx, prof, rev)
        self._ready = self._now() == self._fitted
        self.btn_fit.setEnabled(True)
        n_used = len(self.wiz.used_boards or idx)
        left = len(idx) - n_used
        self.status.setText(
            f"Fitted from {n_used} of the {len(idx)} ticked boards: {prof.summary()}"
            + (f" ({left} fitted far worse than the rest and were set aside: untick them to make that "
               "your choice.)" if left else "")
            + ("" if self._ready else " The boards changed while it was fitting: press the button again."))
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._ready and self.wiz.result_profile is not None


class ResultPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("The lens")
        self.setFinalPage(True)
        lay = QVBoxLayout(self)
        self.report = QTextBrowser()
        lay.addWidget(self.report, 1)
        pics = QHBoxLayout()
        self.pic_before = QLabel()
        self.pic_after = QLabel()
        self.pic_cov = QLabel()
        for w_, cap in ((self.pic_before, "as filmed"), (self.pic_after, "straightened by the profile"),
                        (self.pic_cov, "where the board went")):
            col = QVBoxLayout()
            col.addWidget(w_)
            col.addWidget(_dim(QLabel(cap)))
            pics.addLayout(col)
        lay.addLayout(pics)
        # identical cameras share one checkerboard calibration (G40); off by default:
        # the same picture size does not prove the same lens and zoom
        self.share = QCheckBox("")
        self.share.setVisible(False)
        self.share.toggled.connect(lambda _on: self._finish_text())
        lay.addWidget(self.share)
        self.share_views: list[int] = []
        row = QHBoxLayout()
        self.btn_save = QPushButton("Save lens file…")
        self.btn_save.clicked.connect(self._save)
        row.addWidget(self.btn_save)
        self.saved = _dim(QLabel(""))
        row.addWidget(self.saved, 1)
        lay.addLayout(row)

    def initializePage(self):
        prof = self.wiz.result_profile
        scan = self.wiz.scan
        if prof is None:
            return
        rep = prof.report or {}
        v = str(rep.get("verdict", "ok" if not rep else "poor"))
        col = VERDICT_COLORS.get(v, theme.TEXT_DIM)
        cam = self.wiz.project.name(self.wiz.result_view) if (self.wiz.project is not None and
                                                              self.wiz.result_view is not None) else "this camera"
        parts = []
        if rep:
            parts.append(f"<h2 style='color:{col}; margin-bottom:2px'>{VERDICT_WORDS.get(v, v)}</h2>")
            parts.append("<ul>" + "".join(f"<li>{r}</li>" for r in rep.get("verdict_reasons", [])) + "</ul>")
        else:
            parts.append(f"<h2 style='color:{theme.TEXT_DIM}'>Loaded from a file</h2><p>{prof.source}</p>")
        chk = prof.border_check()
        bend = chk["bend_px"]
        # (G144) the curvature is the lens's own (an edge pixel's distance from where a straight-line lens
        # would put it): said so, or "1076 px" beside "fit error 0.90 px" reads as a 1000-px error
        curv = "the lens's own curvature, not an error"
        if chk["runaway"]:
            bent = (f"the lens curves the picture edges by at least <b>{bend:.0f} px</b> ({curv}) and the model "
                    f"<b style='color:{VERDICT_COLORS['poor']}'>runs away in the corners</b> (see above)"
                    if np.isfinite(bend) else
                    f"<b style='color:{VERDICT_COLORS['poor']}'>the model runs away along the whole border</b>")
        else:
            bent = f"the lens curves the picture edges by up to <b>{bend:.0f} px</b> ({curv})"
        parts.append(f"<p><b>{cam}</b>: focal length {prof.f_square:.0f} px "
                     f"({'fisheye' if prof.fisheye else 'standard'} model), {bent}"
                     + (f", fit error {prof.rms:.2f} px over {prof.n_views} views" if np.isfinite(prof.rms) else "")
                     + f", picture {prof.width} × {prof.height}.</p>")
        if self.wiz.project is None:
            # opened with no video (G9): nothing to attach to -- the FILE is the result
            parts.append(f"<p><b>No video is open, so this profile is not attached to a camera.</b> Press "
                         "<b>Save lens file…</b> below and keep the file. To use it: open the project (or the "
                         "camera's video), then either 3D → Calibrate a Lens → <i>I already have a lens file…</i>, "
                         "or the <i>Load file…</i> button next to that camera on the wand wizard's Cameras page. "
                         "It is valid as long as the camera keeps the same lens, zoom and resolution.</p>")
        else:
            parts.append(f"<p style='color:{theme.TEXT_DIM}'>Press <b>Attach to {cam}</b> and the wand calibration "
                         "will undistort this camera's points with it automatically. Save the file to reuse the "
                         "profile in other projects: it is valid as long as this camera keeps the same lens, zoom and "
                         "resolution.</p>")
        self.report.setHtml("".join(parts))
        if scan is not None and scan.sample_bgr is not None:
            self.pic_before.setPixmap(pixmap(scan.sample_bgr, max_w=420))
            self.pic_after.setPixmap(pixmap(lens.undistort_image(scan.sample_bgr, prof), max_w=420))
            # (I78) light up the boards the profile was fitted from -- a fresh
            # diversity pick here showed unticked boards as used and could
            # contradict the report's coverage figure
            used = self.wiz.used_boards
            if used is None:
                used = self.wiz.page_review.review.chosen()
            self.pic_cov.setPixmap(pixmap(lens.coverage_image(scan, used), max_w=420))
        else:
            for w_ in (self.pic_before, self.pic_after, self.pic_cov):
                w_.clear()
        self.share_views = []
        if self.wiz.project is not None and self.wiz.result_view is not None:
            from kinetrace.calibwizard import share_targets
            self.share_views, _ = share_targets(self.wiz.project, self.wiz.result_view, prof)
        pr = self.wiz.project
        self.share.setVisible(bool(self.share_views))
        if self.share_views:
            names = ", ".join(pr.name(k) for k in self.share_views)
            sv = pr.sessions[self.wiz.result_view]
            self.share.setText(f"Also use it for the other cameras with {sv.width}×{sv.height} pictures that have "
                               f"no lens profile yet: {names}")
            self.share.setToolTip("Right for identical cameras - same model, lens, zoom and recording mode: one "
                                  "checkerboard calibration serves them all, and the wand still fine-tunes each "
                                  "camera's focal length. Leave it off for cameras with other lenses or zoom.")
        else:
            self.share.setChecked(False)
        self._finish_text()

    def _finish_text(self):
        pr = self.wiz.project
        if pr is None or self.wiz.result_view is None:
            self.wiz.setButtonText(QWizard.FinishButton, "Close")
            return
        cam = pr.name(self.wiz.result_view)
        more = len(self.share_views) if self.share.isChecked() else 0
        self.wiz.setButtonText(QWizard.FinishButton,
                               f"Attach to {cam}" + (f" and {more} more" if more else ""))

    def _save(self):
        prof = self.wiz.result_profile
        if prof is None:
            return
        cam = self.wiz.project.name(self.wiz.result_view) if (self.wiz.project is not None and
                                                              self.wiz.result_view is not None) else "camera"
        start = str(Path(self.wiz.start_dir) / f"{cam}{lens.LENS_SUFFIX}") if self.wiz.start_dir else f"{cam}{lens.LENS_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(self, "Save lens profile", start, "Kinetrace lens (*.klens.json)")
        if not path:
            return
        try:
            saved = prof.save(path)
        except Exception as e:      # noqa: BLE001
            # (I222) a read-only folder, a full drive: said on the page, not a crash notice
            self.saved.setText(plain_error(e, "The lens file could not be saved", short=True))
            return
        self.wiz.saved_path = str(saved)
        self.saved.setText("Saved " + Path(saved).name)

    def isComplete(self) -> bool:
        return self.wiz.result_profile is not None


class LensWizard(QWizard):
    def __init__(self, parent, project, default_video: str = "", default_view: int | None = None,
                 start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Calibrate a lens (checkerboard)")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setMinimumSize(820, 640)
        self.project = project
        self.default_video = default_video
        self.default_view = default_view
        self.start_dir = start_dir
        self.result_profile: lens.LensProfile | None = None
        self.result_view: int | None = None
        self.scan = None
        # scan indices of the boards the current profile was fitted from (I78)
        self.used_boards: list[int] | None = None
        self.pattern = lens.DEFAULT_PATTERN
        self.square = lens.DEFAULT_SQUARE_MM / 1000.0
        self.model = "auto"
        # (G146) the board video is GoPro footage with GoPro's lens model in it: `gopro` (gpmf.GoProInfo)
        # and its lens as Kinetrace's fisheye profile (`base_lens`) for "GoPro lens + your boards"
        self.gopro = None
        self.base_lens = None
        self.page_intro = IntroPage(self)
        self.page_video = VideoPage(self)
        self.page_review = ReviewPage(self)
        self.page_result = ResultPage(self)
        for pid, pg in ((ID_INTRO, self.page_intro), (ID_VIDEO, self.page_video),
                        (ID_REVIEW, self.page_review), (ID_RESULT, self.page_result)):
            self.setPage(pid, pg)

    def result_views(self) -> list[int]:
        """The cameras the result is attached to on Finish: the one it was made
        for, plus the identical ones ticked on the result page (G40)."""
        if self.result_view is None:
            return []
        pr = self.page_result
        return [self.result_view] + (list(pr.share_views) if pr.share.isChecked() else [])

    def done(self, r):
        # a scan still running is cancelled and WAITED for, however long (I200); so is a corner
        # editor's frame read in flight (I81)
        stop_page_threads(self)
        self.page_review.review.stop_reader()
        super().done(r)

