"""Track ▾ -> Test the point models on my clicks… (G57).

Which point model suits this footage? The user places one point by hand on
at least `spots.TEST_MIN_FRAMES` (20) frames in a row; this dialog starts
every point model from the first of them and follows it through the rest:
the Moving spot point model at a range of settings (one pass over the
frames, a few seconds) and, when ticked, AllTracker and CoTracker3 (re-tracked
from every correction, slower). Wherever a model drifts more than the drift
limit from a click, or stops, it is put back on that click -- what the user
would have to do -- and that is one correction. The model with the fewest
corrections is recommended, in a sentence; "Use it" makes it the project's
point model (and stores the Moving spot settings on the point).

The work runs on a QThread with its own capture (the GUI thread never touches
cv2 capture or torch); the dialog only shows progress, the table and the
verdict.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
                               QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from kinetrace import spots

MODEL_LABELS = {"alltracker": "AllTracker", "cotracker3": "CoTracker3", "spot": "Moving spot"}
_ORPHANS: list = []          # a test thread that outlived the dialog's wait: kept until it ends


class _ModelError(Exception):
    pass


class _TestRun(QThread):
    """Scores the point models against the clicks (off the GUI thread)."""

    progress = Signal(str, int, int)     # what, done, total
    finished_ok = Signal(object)         # list[spots.TestResult]
    failed = Signal(str)

    def __init__(self, video_path: str, cache, n_frames: int, clicks: dict, size, models, roi: bool):
        super().__init__()
        self.video_path = video_path
        self.cache = cache
        self.n_frames = int(n_frames)
        self.clicks = {int(f): np.asarray(xy, np.float64) for f, xy in clicks.items()}
        self.size = size
        self.models = list(models)
        self.roi = bool(roi)
        self.limit = None
        self.look = None
        self._cancel = False
        self._worker = None

    def cancel(self) -> None:
        self._cancel = True
        w = self._worker
        if w is not None:
            w.request_pause()

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._run())
        except Exception as e:      # noqa: BLE001 - said in words, logged by plain_error
            from kinetrace.errors import plain_error
            self.failed.emit(plain_error(e, "The test"))

    # ------------------------------------------------------------------ the work

    def _run(self) -> list:
        from kinetrace.video_source import VideoSource
        clicks = self.clicks
        frames = sorted(clicks)
        first, last = frames[0], frames[-1]
        w, h = self.size
        src = VideoSource(self.video_path, self.cache)
        try:
            rgb0 = src.get_frame(first)
            if rgb0 is None:
                raise RuntimeError(f"frame {first} of the video could not be read")
            look = spots.measure_spot(rgb0, clicks[first])
            sigma = look.sigma if look is not None else spots.DEFAULT_SIGMA
            self.limit = spots.drift_limit(w, 2.0 * np.sqrt(2.0) * sigma)
            gray = []
            for k in range(max(0, first - spots.HIST_FRAMES), first, spots.HIST_STEP):
                f = src.get_frame(k)
                if f is not None:
                    gray.append((k, _grey(f)))
            ahead = []
            if len(gray) < spots.MIN_HIST:
                for k in range(first + spots.HIST_STEP, min(self.n_frames, first + spots.HIST_FRAMES + 1),
                               spots.HIST_STEP):
                    f = src.get_frame(k)
                    if f is not None:
                        ahead.append((k, _grey(f)))
            src.seek(first)
            settings = spots.candidate_settings(sigma, clicks)
            self.look = look
            total = last - first + 1

            def frames_iter():
                for _ in range(total):
                    nxt = src.read_next()
                    if nxt is None:
                        return
                    yield nxt
            self.progress.emit(f"Moving spot: {len(settings)} settings", 0, total)
            res = spots.score_spot_settings(
                frames_iter(), clicks, settings, (w, h), self.limit, gray, ahead,
                cancel=lambda: self._cancel,
                progress=lambda d, t: self.progress.emit(f"Moving spot: {len(settings)} settings", d, t))
        finally:
            src.close()
        results = list(spots.best_spot(res).values())
        for model in self.models:
            if self._cancel:
                break
            results.append(self._model(model))
        return results

    def _model(self, model: str):
        """AllTracker / CoTracker3 through the clicks: a bare tracking worker
        from each correction (the app's own settings: LK refinement, ROI)."""
        from kinetrace.tracker import PointSpec, TrackingWorker
        label = MODEL_LABELS[model]
        clicks, limit = self.clicks, self.limit
        frames = sorted(clicks)
        last = frames[-1]
        runs = [0]

        def run_from(f, xy):
            runs[0] += 1
            self.progress.emit(f"{label}: run {runs[0]} (from frame {f})", f - frames[0], last - frames[0] + 1)
            out: dict = {}
            judged = {f}
            errs: list = []
            wk = TrackingWorker(self.video_path, f, None, None, self.cache, last + 1, refine=True,
                                specs=[PointSpec(0, np.asarray(xy, np.float32))], roi=self.roi,
                                autopause=False, point_backend=model)

            def chunk(w0, tr, vi, cf, mm, fr):
                for i in range(tr.shape[0]):
                    p = tr[i, 0]
                    out[w0 + i] = p.astype(np.float64) if np.isfinite(p).all() else None
                # a row is final once a later window starts after it (both models
                # rewrite the last window's overlap): judge only those, and stop
                # the run at the first final row that is off -- the next run
                # starts from the click there
                for g in frames:
                    if g in judged or g >= w0:
                        continue
                    judged.add(g)
                    p = out.get(g)
                    if p is None or float(np.hypot(*(p - clicks[g]))) > limit:
                        wk.request_pause()
                        break
            wk.chunk_ready.connect(chunk)
            wk.error.connect(errs.append)
            if self._cancel:
                return {}
            self._worker = wk
            try:
                wk.run()                    # synchronously, on this test thread
            finally:
                self._worker = None
            if errs and not self._cancel:
                raise _ModelError(errs[0])
            return {g: p for g, p in out.items() if p is not None}

        try:
            return spots.corrected_protocol(run_from, clicks, limit, label, model, cancel=lambda: self._cancel)
        except _ModelError as e:
            text = str(e).strip().splitlines()
            why = text[-1] if text else "it stopped with an error"
            return spots.TestResult(label, model, error=f"{label} could not run here: {why}")


def _grey(rgb):
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


class PointModelTest(QDialog):
    """The dialog: what the test does, whether the clicks are enough, a Run
    button, progress, the results table, the verdict and "Use it"."""

    def __init__(self, parent, point_name: str, video_path: str, cache, n_frames: int, size,
                 manual_clicks: dict, models, roi: bool, gpu: bool, on_use=None):
        super().__init__(parent)
        self.setWindowTitle(f"Test the point models on {point_name}'s clicks")
        self.point_name = point_name
        self.video_path, self.cache, self.n_frames, self.size = video_path, cache, n_frames, size
        self.models = list(models)          # [(key, available?, why not)]
        self.roi = roi
        self.on_use = on_use
        self.results = None
        self.winner = None
        self.error = ""
        self._thread: _TestRun | None = None
        ok, stretch, sentence = spots.click_requirement(manual_clicks.keys())
        self.enough = ok
        self.clicks = {f: manual_clicks[f] for f in stretch}

        lay = QVBoxLayout(self)
        intro = QLabel(
            f"This starts each point model from the first frame where you placed <b>{point_name}</b> by hand "
            "and follows it through the frames you placed. Wherever a model drifts away from your click, or "
            "stops, it is put back on your click &mdash; what you would have to do &mdash; and that counts as "
            "one <b>correction</b>. The model that needs the fewest corrections is recommended.<br><br>"
            "<b>Which one is for what?</b> " + spots.WHICH_MODEL)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)
        self.req = QLabel(("✓ " if ok else "⚠ ") + sentence)
        self.req.setWordWrap(True)
        self.req.setStyleSheet("color: #30d158;" if ok else "color: #ffb340;")
        lay.addWidget(self.req)
        avail = [k for k, a, _ in self.models if a]
        self.chk_models = QCheckBox("Also run " + " and ".join(MODEL_LABELS[k] for k in avail)
                                    + " (slower: they are re-tracked from every correction)"
                                    if avail else "AllTracker / CoTracker3 are not available here")
        self.chk_models.setChecked(bool(avail) and gpu)
        self.chk_models.setEnabled(bool(avail))
        why = [f"{MODEL_LABELS[k]}: {w}" for k, a, w in self.models if not a and w]
        self.chk_models.setToolTip(
            ("Without them the test only compares the Moving spot settings with each other. "
             + ("Off by default without a graphics card: they run slowly on the processor. " if not gpu else "")
             + " ".join(why)).strip())
        lay.addWidget(self.chk_models)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run the test")
        self.btn_run.setEnabled(ok)
        self.btn_run.clicked.connect(self.run_test)
        row.addWidget(self.btn_run)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        row.addWidget(self.bar, 1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_cancel)
        lay.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Point model", "Followed before the first correction",
                                              "Corrections", "of them silent drifts", "Median error"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setVisible(False)
        lay.addWidget(self.table, 1)
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(Qt.RichText)
        lay.addWidget(self.verdict)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_use = QPushButton("Use it")
        self.btn_use.setEnabled(False)
        self.btn_use.clicked.connect(self._use)
        bottom.addWidget(self.btn_use)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        bottom.addWidget(self.btn_close)
        lay.addLayout(bottom)
        self.resize(820, 560 if ok else 300)

    # ------------------------------------------------------------------ running

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def run_test(self) -> None:
        if not self.enough or self.running:
            return
        models = [k for k, a, _ in self.models if a] if self.chk_models.isChecked() else []
        th = _TestRun(self.video_path, self.cache, self.n_frames, self.clicks, self.size, models, self.roi)
        th.progress.connect(self._on_progress)
        th.finished_ok.connect(self._on_done)
        th.failed.connect(self._on_failed)
        self._thread = th
        self.results = None
        self.error = ""
        self.btn_run.setEnabled(False)
        self.btn_use.setEnabled(False)
        self.chk_models.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.btn_cancel.setVisible(True)
        self.status.setText("Starting…")
        th.start()

    def _on_progress(self, what: str, done: int, total: int) -> None:
        self.status.setText(what + "…")
        self.bar.setRange(0, max(1, total))
        self.bar.setValue(min(done, total))

    def _finish_ui(self) -> None:
        self.bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_run.setEnabled(self.enough)
        self.btn_run.setText("Run again")
        self.chk_models.setEnabled(any(a for _, a, _ in self.models))

    def _on_failed(self, text: str) -> None:
        self._finish_ui()
        self.error = text
        self.status.setText(text)

    def _on_done(self, results) -> None:
        cancelled = self._thread is not None and self._thread._cancel
        self._finish_ui()
        if cancelled:
            self.status.setText("Cancelled: nothing was changed.")
            return
        self.results = list(results)
        limit = self._thread.limit if self._thread is not None else None
        look = self._thread.look if self._thread is not None else None
        self.status.setText(
            f"Tested on {len(self.clicks)} hand-placed frames ({min(self.clicks)}-{max(self.clicks)}); "
            + (f"the spot measures about {look.diameter:.0f} px across; " if look is not None else "")
            + (f"a position more than {limit:.1f} px from your click counts as a drift." if limit else ""))
        self.winner = spots.recommend(self.results)
        order = sorted(self.results, key=lambda r: (r.key()[:3], r.key()[3]))
        self.table.setRowCount(len(order))
        for i, r in enumerate(order):
            label = r.label if not r.error else f"{r.label} (could not run)"
            cells = [label,
                     "" if r.error else f"{r.first_ok if r.corrections else r.frames} of {r.frames} frames",
                     "" if r.error else str(r.corrections),
                     "" if r.error else str(r.drifts),
                     "" if r.error or not r.errors else f"{r.median_error:.1f} px"]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if r is self.winner:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if r.error:
                    it.setToolTip(r.error)
                self.table.setItem(i, c, it)
        self.table.setVisible(True)
        text = spots.verdict_text(self.point_name, self.results)
        if look is not None and look.at_limit:
            text += (f"<br><b>{self.point_name} is about {look.diameter:.0f} px across or more: too big to be one "
                     "point.</b> Use AllTracker with a Segment (SAM 3); Moving spot is for single-point targets "
                     f"(up to about {spots.POINT_TARGET_MAX:.0f} px).")
        elif look is not None and look.diameter > spots.POINT_TARGET_MAX:
            text += (f"<br><b>{self.point_name} measures about {look.diameter:.0f} px across</b> &mdash; more than a "
                     f"single-point target (up to about {spots.POINT_TARGET_MAX:.0f} px). Moving spot would follow "
                     "only its centre, as one point. If you can see its shape, or want more than one landmark on "
                     "it, use AllTracker with a Segment.")
        errs = [r.error for r in self.results if r.error]
        self.verdict.setText(text + ("<br><i>" + " ".join(errs) + "</i>" if errs else ""))
        if self.winner is not None:
            self.btn_use.setText(f"Use {MODEL_LABELS.get(self.winner.model, self.winner.label)}"
                                 + (f" ({self.winner.settings.describe()})" if self.winner.model == "spot" else "")
                                 + " for this project")
            self.btn_use.setEnabled(True)

    def _use(self) -> None:
        if self.winner is None:
            return
        if callable(self.on_use):
            self.on_use(self.winner)
        self.accept()

    # ------------------------------------------------------------------ closing

    def _cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel()
            self.status.setText("Cancelling…")

    def _stop_thread(self) -> None:
        th = self._thread
        if th is None or not th.isRunning():
            return
        th.cancel()
        if not th.wait(5000):
            # a model step can take a moment: keep the thread referenced until it
            # ends -- dropping a running QThread aborts the program (I133)
            _ORPHANS.append(th)
            th.finished.connect(lambda: _ORPHANS.remove(th) if th in _ORPHANS else None)

    def reject(self) -> None:
        self._stop_thread()
        super().reject()

    def closeEvent(self, e) -> None:
        self._stop_thread()
        super().closeEvent(e)
