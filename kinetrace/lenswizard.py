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

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton,
                               QSizePolicy, QSpinBox, QTextBrowser, QVBoxLayout, QWidget, QWizard,
                               QWizardPage)

from kinetrace import lens, theme
from kinetrace.boardreview import BoardReview

VERDICT_COLORS = {"good": theme.GREEN, "ok": "#FFD60A", "poor": theme.RED}
VERDICT_WORDS = {"good": "GOOD — attach this profile to the camera",
                 "ok": "USABLE — but read the notes below",
                 "poor": "NOT GOOD ENOUGH — film the board again as the notes say"}
LENS_FILTER = ("Lens profiles (*.klens.json *.json *.yml *.yaml *.txt);;Kinetrace lens (*.klens.json);;"
               "OpenCV lens (*.yml *.yaml *.json);;Argus / DLTdv camera profile (*.txt);;All files (*)")
UNITS = [("millimetres (mm)", 0.001), ("centimetres (cm)", 0.01), ("metres (m)", 1.0), ("inches (in)", 0.0254)]
ID_INTRO, ID_VIDEO, ID_REVIEW, ID_RESULT = range(4)


def _dim(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {theme.TEXT_DIM};")
    return label


def _pix(bgr: np.ndarray, max_w: int = 420) -> QPixmap:
    h, w = bgr.shape[:2]
    sc = min(1.0, max_w / w)
    if sc < 1.0:
        bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())


class _LensThread(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(object, object)      # ScanResult, LensProfile
    error = Signal(str)

    def __init__(self, video: str, pattern: tuple[int, int], square: float, model: str,
                 fit: bool = True):
        super().__init__()
        self.video, self.pattern, self.square, self.model = video, pattern, square, model
        # `fit` False stops after the scan, so the boards can be reviewed and
        # a subset chosen before anything is fitted to them.
        self.fit = bool(fit)
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
            if not self.fit:
                self.progress.emit(1.0, f"{len(scan.corners)} boards found")
                self.finished_ok.emit(scan, None)
                return
            self.progress.emit(0.9, f"{len(scan.corners)} boards found — fitting the lens")
            prof = lens.calibrate_lens(scan.corners, self.pattern, self.square, scan.size, self.model)
            self.finished_ok.emit(scan, prof)
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
        wmm, hmm = lens.save_checkerboard_png(path)
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
        self.r_auto.setChecked(True)
        for r in (self.r_auto, self.r_std, self.r_fish):
            ml.addWidget(r)
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
        self.cam.currentIndexChanged.connect(self._camera_changed)

    def initializePage(self):
        if self.wiz.default_view is not None and 0 <= self.wiz.default_view < self.cam.count():
            self.cam.setCurrentIndex(self.wiz.default_view)

    def _pick(self):
        start = self.wiz.default_video or self.wiz.start_dir
        path, _ = QFileDialog.getOpenFileName(self, "Checkerboard video", start,
                                              "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.mpg);;All files (*)")
        if path:
            self.path.setText(path)
            self.prefill_note.setText("")      # chosen on purpose now (G9)
            self.completeChanged.emit()

    def pattern(self) -> tuple[int, int]:
        return int(self.cols.value()), int(self.rows.value())

    def _pattern_changed(self, *_):
        self.pattern_note.setText(lens.orientation_note(self.pattern()))
        self.pattern_note.setVisible(bool(self.pattern_note.text()))

    def square_m(self) -> float:
        return float(self.square.value()) * UNITS[self.unit.currentIndex()][1]

    def model(self) -> str:
        return "auto" if self.r_auto.isChecked() else ("fisheye" if self.r_fish.isChecked() else "standard")

    def _run(self):
        video = self.path.text().strip()
        if not video or not Path(video).exists():
            self.status.setText("Choose the video of the board first.")
            return
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.bar.setValue(0)
        self.wiz.result_profile = None
        self.wiz.scan = None
        self.wiz.used_boards = None
        self._loaded_path = None
        self.completeChanged.emit()
        th = _LensThread(video, self.pattern(), self.square_m(), self.model(), fit=False)
        th.progress.connect(lambda f, m: (self.bar.setValue(int(1000 * f)), self.status.setText(m)))
        th.finished_ok.connect(self._done)
        th.error.connect(self._fail)
        self._thread = th
        th.start()

    def _fail(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.status.setText("Calibration failed: " + msg)

    def _done(self, scan, prof):
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.bar.setValue(1000)
        self.wiz.scan = scan
        self.wiz.pattern = self.pattern()
        self.wiz.square = self.square_m()
        self.wiz.model = self.model()
        self.wiz.result_profile = prof          # None: the review page fits
        self.wiz.result_view = self._view()
        self._show_scan_status()
        self.completeChanged.emit()

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
        if self.wiz.scan is not None:
            return lens.size_mismatch(self.wiz.scan.size, cam, p.name(view), "The checkerboard video has")
        prof = self.wiz.result_profile
        if prof is not None:
            return lens.size_mismatch((prof.width, prof.height), cam, p.name(view))
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
            if path.lower().endswith((".json", ".yml", ".yaml")):
                from kinetrace import calibio
                prof, which = calibio.read_lens(path), ""      # Kinetrace or OpenCV lens file
            else:
                # (I80) the file's line for THIS camera, or a refusal -- never
                # the last line reused for a camera the file does not list
                name = self.wiz.project.name(view) if (self.wiz.project is not None and view is not None) else ""
                prof, which = lens.argus_profile_for(path, view if view is not None else 0, name)
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

    def _mark_fitted(self, idx: list[int], prof) -> None:
        self._fitted = (tuple(idx), self.review.corner_rev)
        # the boards the fit really used, in scan numbering: calibrate_lens may
        # thin the given views to a spread and set a few aside (I78)
        used = (prof.report or {}).get("views_used")
        self.wiz.used_boards = ([idx[k] for k in used if 0 <= k < len(idx)] if used is not None
                                else list(idx))

    def _review_changed(self):
        # (I74) the profile belongs to the boards it was fitted from. Unticking
        # a board or dragging a corner after the fit makes it stale: Next waits
        # for a refit, or for the choice to be put back as it was.
        was = self._ready
        now = (tuple(self.review.chosen()), self.review.corner_rev)
        self._ready = self._fitted is not None and now == self._fitted
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
        self.status.setText("Choosing a good spread…")
        QApplication.processEvents()
        idx, err, why = lens.auto_select(scan.corners, self.wiz.pattern, self.wiz.square,
                                         scan.size, self.wiz.model)
        prof = None
        if idx:
            try:
                prof = lens.calibrate_lens([scan.corners[i] for i in idx], self.wiz.pattern,
                                           self.wiz.square, scan.size, self.wiz.model)
            except Exception:                                   # noqa: BLE001
                prof = None
        self.review.set_scan(scan, self.wiz.pattern, self.wiz.square, prof, idx, err,
                             model=self.wiz.model)
        if prof is not None:
            self.wiz.result_profile = prof
            self._mark_fitted(idx, prof)
            self._ready = True
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
        QApplication.processEvents()
        try:
            prof = lens.calibrate_lens([scan.corners[i] for i in idx], self.wiz.pattern,
                                       self.wiz.square, scan.size, self.wiz.model)
        except Exception as exc:                                # noqa: BLE001
            self.btn_fit.setEnabled(True)
            self.status.setText(f"Could not fit: {exc}")
            return
        err = lens.per_view_errors(scan.corners, self.wiz.pattern, self.wiz.square, prof)
        self.review.prof = prof
        self.review.errors = err
        for i, tile in enumerate(self.review._tiles):
            tile.set_error(err[i])
            tile.set_pixmap(self.review._tile_image(i))
        self.review._summarise("refitted")
        self.wiz.result_profile = prof
        self._mark_fitted(idx, prof)
        self._ready = True
        self.btn_fit.setEnabled(True)
        self.status.setText(f"Fitted from {len(idx)} boards: {prof.summary()}")
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
        if chk["runaway"]:
            bent = (f"the edges of the picture are bent by at least <b>{bend:.0f} px</b> and the model "
                    f"<b style='color:{VERDICT_COLORS['poor']}'>runs away in the corners</b> (see above)"
                    if np.isfinite(bend) else
                    f"<b style='color:{VERDICT_COLORS['poor']}'>the model runs away along the whole border</b>")
        else:
            bent = f"the edges of the picture are bent by up to <b>{bend:.0f} px</b>"
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
            self.pic_before.setPixmap(_pix(scan.sample_bgr))
            self.pic_after.setPixmap(_pix(lens.undistort_image(scan.sample_bgr, prof)))
            # (I78) light up the boards the profile was fitted from -- a fresh
            # diversity pick here showed unticked boards as used and could
            # contradict the report's coverage figure
            used = self.wiz.used_boards
            if used is None:
                used = self.wiz.page_review.review.chosen()
            self.pic_cov.setPixmap(_pix(lens.coverage_image(scan, used)))
        else:
            for w_ in (self.pic_before, self.pic_after, self.pic_cov):
                w_.clear()
        self.wiz.setButtonText(QWizard.FinishButton,
                               f"Attach to {cam}" if self.wiz.project is not None else "Close")

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
        saved = prof.save(path)
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
        self.page_intro = IntroPage(self)
        self.page_video = VideoPage(self)
        self.page_review = ReviewPage(self)
        self.page_result = ResultPage(self)
        for pid, pg in ((ID_INTRO, self.page_intro), (ID_VIDEO, self.page_video),
                        (ID_REVIEW, self.page_review), (ID_RESULT, self.page_result)):
            self.setPage(pid, pg)

    def done(self, r):
        # Wait for any page's worker before the wizard (and then the
        # interpreter) goes away. A QThread whose Python wrapper is collected
        # while Qt still owns it takes the process down at exit with
        # 0xC0000409, which reads as a failed test even though everything
        # passed.
        for pid in self.pageIds():
            th = getattr(self.page(pid), "_thread", None)
            if th is None:
                continue
            if hasattr(th, "cancel"):
                th.cancel()
            if th.isRunning():
                th.wait(15000)
        self.page_review.review.stop_reader()       # the corner editor's frame read (I81)
        super().done(r)

