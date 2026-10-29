"""ChArUco camera calibration wizard — 3D → Calibrate → Calibrate Cameras with a ChArUco Board (I322, G221).

The same printed ChArUco board, filmed by every camera at once, gives each camera's lens (its
intrinsics: focal length, centre, distortion) and where the cameras are (the extrinsics). By
default both; the user may ask for the lenses only, or for the camera positions only with the
lens profiles the cameras already have.

How (the route Argus takes, Jackson et al. 2016, adapted with its authors' permission - see
`charuco.py` for the rules taken from it):
- `charuco.scan_rig` finds the board in every camera at the same instants (the project's
  offsets; a fractional offset is interpolated between two frames);
- lenses: `lens.calibrate_lens` per camera on that camera's own boards (Argus's 100 best, R3),
  the centre fitted and the fisheye model offered (Argus pins the centre);
- positions: the board's corners seen by two or more cameras are wand-style observations for
  `wand.calibrate_wand` - the first and last corner of a long edge as the wand (R4), its length
  the square size typed, every other shared corner an extra point - straightened with the
  lenses first, exactly as the wand wizard does (`calibwizard.solve`).
Nothing changes in the project before Finish; the app applies the lenses and the calibration
together, one Ctrl+Z.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QRadioButton, QSpinBox, QTextBrowser, QVBoxLayout,
                               QWidget, QWizard, QWizardPage)

from kinetrace import charuco, theme
from kinetrace import lens as _lens
from kinetrace.charucoui import CharucoFields
from kinetrace.calibwizard import _Cancelled, focal_free, principal_points, start_focals, undistort_views
from kinetrace.errors import plain_error
from kinetrace.lenswizard import VERDICT_COLORS, _dim, stop_page_threads

CHOICES = ("both", "lenses", "positions")
CHOICE_WORDS = {"both": "Lenses and camera positions (recommended)",
                "lenses": "Lenses only (each camera's focal length, centre and distortion)",
                "positions": "Camera positions only (keep the lens profiles the cameras have)"}
MIN_PAIRED = 10                 # the wand wizard's MIN_FRAMES: board instants seen by two or more cameras
GOOD_PAIRED = 30
MIN_LENS_VIEWS, GOOD_LENS_VIEWS = 3, 15


def _band(n: int, ok: int, good: int) -> str:
    return "good" if n >= good else ("ok" if n >= ok else "poor")


# ---------------------------------------------------------------- the core (no Qt widgets; any thread)


def fit_lenses(project, scan: charuco.RigScan, progress=None) -> dict:
    """{project camera: LensProfile} from each scan camera's own boards (R3: the 100 best),
    each recording its board, its video and the frames used. ValueError (in words) for a
    camera that saw too few usable boards."""
    spec = scan.spec
    out = {}
    for k, v in enumerate(scan.cameras):
        vs, fr = scan.camera_views(k)
        if len(vs) < MIN_LENS_VIEWS:
            raise ValueError(f"{scan.names[k]} saw the whole-enough board (6 corners, not in one line) at only "
                             f"{len(vs)} instants; a lens needs at least {MIN_LENS_VIEWS}, {GOOD_LENS_VIEWS} or more "
                             "for a trustworthy one. Look at more frames, or film the board nearer that camera")
        if progress is not None:
            progress(k / max(1, len(scan.cameras)), f"Fitting {scan.names[k]}'s lens from {len(vs)} boards")
        keep = charuco.best_frames(vs, scan.sizes[k])                    # R3 (Argus)
        prof = _lens.calibrate_lens([vs[i] for i in keep], spec.pattern, spec.square_m, scan.sizes[k],
                                    model="auto", source="ChArUco board (Kinetrace)")
        path = project.sessions[v].video_path
        try:
            _size, rotation, _oren = _lens.video_turn(path)
        except OSError:
            rotation = None
        prof.rotation = rotation
        used = (prof.report or {}).get("views_used", list(range(len(keep))))
        prof.board = {"video": Path(path).name, "path": os.path.abspath(path), "size": list(scan.sizes[k]),
                      "rotation": rotation, "frames": [int(fr[keep[i]]) for i in used],
                      "pattern": list(spec.pattern), "square_m": float(spec.square_m),
                      "kind": "charuco", "charuco": spec.to_json()}
        said = _lens.board_sentence(prof.board)
        if said and isinstance(prof.report.get("verdict_reasons"), list):
            prof.report["verdict_reasons"].append(said)
        out[int(v)] = prof
    return out


def rig_inputs(scan: charuco.RigScan, lenses: dict):
    """(kwargs for `wand.calibrate_wand`, lens models, lens summaries, wand frames) for the scan's
    cameras with `lenses` ({project camera: profile fitted to it, or absent}): the rules of the wand
    wizard (`calibwizard.build_inputs` / `CamerasPage`): profiled cameras straightened, their
    focal length and centre from the profile; a focal length refined only where cameras share a
    profile (I144) or some camera has none (I35). The rules ARE the wand wizard's functions (R59), with
    every scan camera ticked; distortion is never estimated here (the board's corners come from lenses
    already fitted or from none)."""
    wand_uv, L, bg_uv, frames, _bgf = charuco.pairs_and_extras(scan)
    ticked = list(range(len(scan.cameras)))
    lens_list = [lenses.get(v) for v in scan.cameras]
    models = [(l.undistort_model() if l is not None else None) for l in lens_list]
    summaries = [(l.summary() if l is not None else None) for l in lens_list]
    bg = bg_uv if len(bg_uv) else None
    focal = start_focals(lens_list, ticked)
    principal = principal_points(lens_list, scan.sizes, ticked)
    est = focal_free(lens_list, ticked)
    kwargs = dict(wand_uv=undistort_views(wand_uv, models), wand_length=L, sizes=list(scan.sizes),
                  coverage_uv=wand_uv, coverage_bg_uv=bg, frame_ids=frames, focal=focal, principal=principal,
                  bg_uv=undistort_views(bg, models),
                  estimate_focal=est, estimate_distortion=False, unit="m", names=list(scan.names))
    return kwargs, models, summaries, frames


def calibrate_rig(project, scan: charuco.RigScan, choice: str, existing: dict | None = None, progress=None) -> dict:
    """The wizard's run: {"lenses": {project camera: profile} (fitted, empty for "positions"),
    "result": WandResult | None ("lenses")}. `existing` = {project camera: the lens profile it has,
    fitted to its pictures} for "positions". WandError / ValueError carry a sentence."""
    from kinetrace.calibwizard import solve

    def note(f, m):
        if progress is not None:
            progress(f, m)
    lenses = {}
    if choice in ("both", "lenses"):
        lenses = fit_lenses(project, scan, lambda f, m: note(0.4 * f if choice == "both" else f, m))
    if choice == "lenses":
        return {"lenses": lenses, "result": None}
    use = lenses if choice == "both" else dict(existing or {})
    kwargs, models, summaries, frames = rig_inputs(scan, use)
    if len(kwargs["wand_uv"]) < MIN_PAIRED:
        raise ValueError(f"only {len(kwargs['wand_uv'])} instants show the board to two or more cameras with "
                         f"{charuco.MIN_SHARED_IDS} shared corners and a long edge's two end corners; at least "
                         f"{MIN_PAIRED} are needed, {GOOD_PAIRED} or more for a good result. Look at more frames, or "
                         "film the board where the cameras' views overlap")
    sub = (lambda f, m: note(0.4 + 0.6 * f, m)) if choice == "both" else note
    res, _grav = solve(kwargs, "none", None, None, models, summaries, sub)
    res.frame_ids = np.asarray(frames, np.int64)
    res.wand_names = ("board corner A", "board corner B")
    res.report["board"] = {"spec": scan.spec.to_json(), "words": scan.spec.words(),
                           "pair_length_m": float(charuco.wand_length_m(scan.spec)),
                           "n_extra": int(0 if kwargs["bg_uv"] is None else len(kwargs["bg_uv"]))}
    return {"lenses": lenses, "result": res}


def board_html(report: dict | None, lenses: dict, project) -> str:
    """The board's part of the report: what the 'wand' is here, and each fitted lens's verdict."""
    parts = []
    if report is not None and report.get("board"):
        b = report["board"]
        parts.append(f"<h3>The board</h3><p>{b['words']}. The 'wand' in the numbers below is the board's long edge: "
                     f"its first and last corner, <b>{1000 * b['pair_length_m']:.0f} mm</b> apart, so the size of the "
                     "whole calibration comes from the square size typed. Hand check: measure that edge on the "
                     f"printed board with a ruler; {b['n_extra']} more board corners seen by two or more cameras "
                     "were used as extra points.</p>")
    if lenses:
        rows = []
        for v in sorted(lenses):
            prof = lenses[v]
            rep = prof.report or {}
            verdict = str(rep.get("verdict", "?"))
            col = VERDICT_COLORS.get(verdict, theme.TEXT_DIM)
            rows.append(f"<tr><td>{project.name(v)}</td><td style='color:{col}'><b>{verdict.upper()}</b></td>"
                        f"<td align='center'>{prof.f_square:.0f}</td><td align='center'>{prof.rms:.2f}</td>"
                        f"<td align='center'>{prof.n_views}</td><td>{'fisheye' if prof.fisheye else 'standard'}</td>"
                        f"<td align='center'>{rep.get('coverage_pct', 0):.0f} %</td></tr>")
        parts.append("<h3>The lenses (from the board)</h3><table cellpadding='3'><tr><th align='left'>camera</th>"
                     "<th>verdict</th><th>focal length (px)</th><th>fit error (px)</th><th>boards</th><th>model</th>"
                     "<th>coverage</th></tr>" + "".join(rows) + "</table>")
        for v in sorted(lenses):
            rep = lenses[v].report or {}
            weak = [r for r in rep.get("verdict_reasons", []) if any(w in r for w in ("only", "too high", "RUNS AWAY",
                                                                                      "nearly square-on"))]
            if weak:
                parts.append(f"<p><b>{project.name(v)}</b>: " + " ".join(weak) + "</p>")
    return "".join(parts)


# ---------------------------------------------------------------- threads


class _ScanThread(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(object)              # charuco.RigScan
    error = Signal(str)

    def __init__(self, project, spec, first, last, n, views):
        super().__init__()
        self.project, self.spec, self.first, self.last, self.n, self.views = project, spec, first, last, n, views
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            scan = charuco.scan_rig(self.project, self.spec, None, self.first, self.last, self.n,
                                    progress=lambda f, m: self.progress.emit(float(f), str(m)),
                                    should_cancel=lambda: self._cancel, views=self.views)
            if not self._cancel:
                self.finished_ok.emit(scan)
        except InterruptedError:
            return
        except Exception as e:      # noqa: BLE001
            self.error.emit(plain_error(e, "The board could not be looked for", short=True))


class _RunThread(QThread):
    progress = Signal(float, str)
    finished_ok = Signal(object)              # calibrate_rig's dict
    error = Signal(str)

    def __init__(self, project, scan, choice, existing):
        super().__init__()
        self.project, self.scan, self.choice, self.existing = project, scan, choice, existing
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _note(self, f, m):
        """The progress callback: a Cancel stops the run at its next step (a lens fit, a focal-length
        guess, an adjustment pass), as the wand wizard's (I36) - `done()` then waits for a thread that is
        ending, not for a whole solve."""
        if self._cancel:
            raise _Cancelled()
        self.progress.emit(float(f), str(m))

    def run(self):
        from kinetrace import wand as _wand
        try:
            out = calibrate_rig(self.project, self.scan, self.choice, self.existing, progress=self._note)
            if not self._cancel:
                self.finished_ok.emit(out)
        except _Cancelled:
            return
        except (_wand.WandError, ValueError) as e:
            self.error.emit(str(e))
        except Exception as e:      # noqa: BLE001
            self.error.emit(plain_error(e, "The calibration could not be completed"))


# ---------------------------------------------------------------- pages


class BoardPage(QWizardPage):
    def __init__(self, wiz):
        super().__init__()
        self.wiz = wiz
        self.setTitle("The ChArUco board")
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Film the printed board with <b>every camera recording at once</b>: move it slowly through the space "
            "where the animal will be, tilted towards the cameras, near and far, into the corners of each picture. "
            "A board half out of a picture still counts (each corner is named by its markers). The cameras' "
            "offsets must already line them up (3D → Sync → Sync Cameras). Kinetrace's printable board is on the "
            "lens wizard's first page.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.fields = CharucoFields()
        lay.addWidget(self.fields)
        form = QFormLayout()
        frow = QWidget()
        fl = QHBoxLayout(frow)
        fl.setContentsMargins(0, 0, 0, 0)
        self.first = QSpinBox()
        self.last = QSpinBox()
        self.count = QSpinBox()
        for sp in (self.first, self.last):
            sp.setRange(0, 10 ** 8)
        self.count.setRange(5, 2000)
        self.count.setValue(charuco.DEFAULT_FRAMES)
        self.count.setToolTip("Instants spread evenly over the stretch (Argus looks at 50); more finds more boards "
                              "but takes longer")
        fl.addWidget(QLabel("from"))
        fl.addWidget(self.first)
        fl.addWidget(QLabel("to"))
        fl.addWidget(self.last)
        fl.addWidget(QLabel("look at"))
        fl.addWidget(self.count)
        fl.addWidget(_dim(QLabel("instants (frames of the first camera)")))
        fl.addStretch(1)
        form.addRow("Where the board is filmed", frow)
        lay.addLayout(form)
        self.cams_box = QGroupBox("Cameras")
        self.cam_lay = QVBoxLayout(self.cams_box)
        self.ticks: list[QCheckBox] = []
        lay.addWidget(self.cams_box)
        what = QGroupBox("What to calibrate")
        wl = QVBoxLayout(what)
        self.choice_group = QButtonGroup(self)
        self.r_choice = {}
        for k in CHOICES:
            r = QRadioButton(CHOICE_WORDS[k])
            self.r_choice[k] = r
            self.choice_group.addButton(r)
            wl.addWidget(r)
        self.r_choice["both"].setChecked(True)
        self.choice_note = _dim(QLabel(""))
        wl.addWidget(self.choice_note)
        lay.addWidget(what)
        row = QHBoxLayout()
        self.btn_scan = QPushButton("Find the board")
        self.btn_scan.setObjectName("primary")
        row.addWidget(self.btn_scan)
        row.addStretch(1)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self._thread: _ScanThread | None = None
        self._scan_key = None
        self.btn_scan.clicked.connect(self._scan)
        self.fields.changed.connect(self._changed)
        for sp in (self.first, self.last, self.count):
            sp.valueChanged.connect(self._changed)
        for r in self.r_choice.values():
            r.toggled.connect(lambda _on: self._choice_changed())

    def initializePage(self):
        p = self.wiz.project
        if not self.ticks:
            for c in range(p.n_views):
                cb = QCheckBox(p.name(c))
                cb.setChecked(True)
                cb.toggled.connect(self._changed)
                self.cam_lay.addWidget(cb)
                self.ticks.append(cb)
            lo, hi = charuco.shared_window(p)
            for sp in (self.first, self.last):
                sp.blockSignals(True)
            self.first.setValue(max(0, lo))
            self.last.setValue(max(0, hi))
            for sp in (self.first, self.last):
                sp.blockSignals(False)
        self._choice_changed()

    def views(self) -> list[int]:
        return [c for c, cb in enumerate(self.ticks) if cb.isChecked()]

    def choice(self) -> str:
        return next(k for k, r in self.r_choice.items() if r.isChecked())

    def _key(self):
        return (self.fields.spec(), int(self.first.value()), int(self.last.value()), int(self.count.value()),
                tuple(self.views()))

    def _changed(self, *_):
        if self.wiz.scan is not None and self._key() != self._scan_key:
            self.wiz.scan = None                   # boards found for other settings are not these (G115)
            self.status.setText("The board, the frames or the cameras changed: press 'Find the board' again.")
            self.bar.setValue(0)
        self.completeChanged.emit()

    def _choice_changed(self):
        k = self.choice()
        p = self.wiz.project
        if k == "positions":
            have = [p.name(c) for c in self.views() if self.wiz.usable_lens(c) is not None]
            self.choice_note.setText("Cameras with a lens profile keep it: " + (", ".join(have) if have else "none") +
                                     ". The others are solved as plain pinholes (no distortion correction).")
        elif k == "lenses":
            self.choice_note.setText("Each camera gets the lens measured from its own boards; the calibration stays "
                                     "as it is.")
        else:
            self.choice_note.setText("Each camera's lens from its own boards, then the cameras placed with those "
                                     "lenses. Lens profiles the cameras have now are replaced.")
        if self.wiz.scan is not None:
            self._show_scan()
        self.completeChanged.emit()

    def _scan(self):
        spec = self.fields.spec()
        if spec is None:
            self.status.setText("The board typed cannot exist: " + self.fields.problem() + ".")
            return
        views = self.views()
        need = 1 if self.choice() == "lenses" else 2
        if len(views) < need:
            self.status.setText(f"Tick at least {need} camera{'s' if need > 1 else ''}.")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self.wiz.scan = None
        self.bar.setValue(0)
        self.status.setText("Looking for the board…")
        self._lock(True)
        th = _ScanThread(self.wiz.project, spec, int(self.first.value()), int(self.last.value()),
                         int(self.count.value()), views)
        th.progress.connect(lambda f, m: (self.bar.setValue(int(1000 * f)), self.status.setText(m)))
        th.finished_ok.connect(lambda scan, k=self._key(): self._done(scan, k))
        th.error.connect(self._fail)
        th.finished.connect(lambda: self._lock(False))
        self._thread = th
        th.start()

    def _lock(self, on: bool):
        self.btn_scan.setEnabled(not on)
        self.fields.set_enabled_inputs(not on)
        for w_ in (self.first, self.last, self.count, *self.ticks):
            w_.setEnabled(not on)

    def _fail(self, msg: str):
        self.status.setText(f"<span style='color:{theme.RED}'>{msg}</span>")
        self.completeChanged.emit()

    def _done(self, scan, key):
        self.bar.setValue(1000)
        self.wiz.scan = scan
        if scan.spec != self.fields.spec():
            self.fields.set_spec(scan.spec)        # the layout the scan found (the boxes are locked meanwhile)
        self._scan_key = self._key()
        self._show_scan()
        self.completeChanged.emit()

    def _counts(self):
        """(lens boards per scan camera, paired instants) of the current scan."""
        scan = self.wiz.scan
        lens_n = [len(scan.camera_views(k)[0]) for k in range(len(scan.cameras))]
        wand_uv = charuco.pairs_and_extras(scan)[0]
        return lens_n, int(len(wand_uv))

    def _show_scan(self):
        scan = self.wiz.scan
        lens_n, paired = self._counts()
        k = self.choice()
        lines = [f"Looked at {len(scan.ref_frames)} instants. " + (scan.note or "")]
        for c in range(len(scan.cameras)):
            v = _band(lens_n[c], MIN_LENS_VIEWS, GOOD_LENS_VIEWS)
            col = VERDICT_COLORS.get(v, theme.TEXT_DIM)
            seen = scan.seen(c)
            fail = (f" <span style='color:{theme.RED}'>Frame {scan.failed[c]} did not decode.</span>"
                    if scan.failed[c] is not None else "")
            lines.append(f"{scan.names[c]}: the board in {seen} of {len(scan.ref_frames)} instants, "
                         f"<span style='color:{col}'>{lens_n[c]} good enough for its lens</span>{fail}")
        if k != "lenses":
            v = _band(paired, MIN_PAIRED, GOOD_PAIRED)
            col = VERDICT_COLORS.get(v, theme.TEXT_DIM)
            lines.append(f"Instants placing the cameras (two or more see a long edge and {charuco.MIN_SHARED_IDS}+ "
                         f"shared corners): <span style='color:{col}'><b>{paired}</b></span> "
                         f"(at least {MIN_PAIRED}, {GOOD_PAIRED}+ is good).")
        self.status.setText("<br>".join(lines))

    def isComplete(self) -> bool:
        scan = self.wiz.scan
        if scan is None or self._key() != self._scan_key:
            return False
        lens_n, paired = self._counts()
        k = self.choice()
        if k in ("both", "lenses") and min(lens_n) < MIN_LENS_VIEWS:
            return False
        if k in ("both", "positions") and (len(scan.cameras) < 2 or paired < MIN_PAIRED):
            return False
        return True


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
        self.btn_argus = QPushButton("Save for Argus…")
        self.btn_argus.setEnabled(False)
        self.btn_argus.setToolTip("camera_profile.txt (every camera's lens) and the board corners as Argus's paired / "
                                  "unpaired points files, to go on in Argus Wand")
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_argus)
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
        self._thread: _RunThread | None = None
        self._key = None
        self.btn_run.clicked.connect(self._run)
        self.btn_argus.clicked.connect(self._save_argus)

    def _settings_key(self):
        w = self.wiz
        bp = w.page_board
        existing = tuple(_lens.profile_key(w.usable_lens(c)) for c in bp.views()) if bp.choice() == "positions" else ()
        return (id(w.scan), bp.choice(), existing)

    def initializePage(self):
        w = self.wiz
        k = w.page_board.choice()
        names = ", ".join(w.project.name(c) for c in w.page_board.views())
        self.summary.setText(f"{w.scan.spec.words()}. Cameras: {names}. {CHOICE_WORDS[k]}.")
        w.setButtonText(QWizard.FinishButton, {"both": "Use these lenses and this calibration",
                                               "lenses": "Use these lenses",
                                               "positions": "Use this calibration"}[k])
        if self._settings_key() != self._key:
            w.result = w.result_calibration = None
            w.result_lenses = {}
            self.report.setHtml(f"<p style='color:{theme.TEXT_DIM}'>Press <b>Calibrate now</b>: a few seconds to a "
                                "minute.</p>")
            self.bar.setValue(0)
            self.status.setText("")
            self.btn_argus.setEnabled(False)
            self.completeChanged.emit()

    def _run(self):
        if self._thread is not None and self._thread.isRunning():
            return
        w = self.wiz
        k = w.page_board.choice()
        existing = {c: w.usable_lens(c) for c in w.page_board.views() if w.usable_lens(c) is not None}
        self._key = self._settings_key()
        w.result = w.result_calibration = None
        w.result_lenses = {}
        self.btn_run.setEnabled(False)
        self.btn_argus.setEnabled(False)
        self.status.setText("Working…")
        th = _RunThread(w.project, w.scan, k, existing)
        th.progress.connect(lambda f, m: (self.bar.setValue(int(1000 * f)), self.status.setText(m)))
        th.finished_ok.connect(self._done)
        th.error.connect(self._fail)
        th.finished.connect(lambda: self.btn_run.setEnabled(True))
        self._thread = th
        th.start()
        self.completeChanged.emit()

    def _fail(self, msg: str):
        self.status.setText("")
        self.report.setHtml(f"<p style='color:{theme.RED}'><b>Calibration failed.</b> {msg}</p>")

    def _done(self, out: dict):
        from kinetrace.calib import spread_calibration
        from kinetrace.calibwizard import report_html
        w = self.wiz
        p = w.project
        self.bar.setValue(1000)
        self.status.setText("Done.")
        w.result_lenses = dict(out["lenses"])
        res = out["result"]
        w.result = res
        parts = []
        if res is not None:
            views = list(w.scan.cameras)
            w.result_calibration = spread_calibration(res.to_calibration(), views, p.n_views)
            res.report["project_views"] = views
            res.report["project_names"] = [p.name(c) for c in range(p.n_views)]
            parts.append(report_html(res.report, None, p, res.unit))
        parts.insert(0 if res is None else 1, board_html(res.report if res is not None else None,
                                                         w.result_lenses, p))
        if res is None:
            lens_v = [str((pr.report or {}).get("verdict", "?")) for pr in w.result_lenses.values()]
            worst = "poor" if "poor" in lens_v else ("ok" if "ok" in lens_v else "good")
            parts.insert(0, f"<h2 style='color:{VERDICT_COLORS.get(worst, theme.TEXT_DIM)}'>Lenses: {worst.upper()}</h2>")
        self.report.setHtml("".join(parts))
        self.btn_argus.setEnabled(True)
        self.completeChanged.emit()

    def _save_argus(self):
        """(I323) Argus's three files into a folder, as its ChArUco script writes them: the board corners
        (paired_pts-xypts.csv, unpaired_pts-xypts.csv: raw pixels, y up) and camera_profile.txt (every
        camera's lens - the fitted ones, or the cameras' own for "positions only"), cameras numbered 1..C
        in the scan's order."""
        from PySide6.QtWidgets import QFileDialog

        from kinetrace import calibio
        w = self.wiz
        scan = w.scan
        if scan is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save for Argus: choose a folder", w.start_dir)
        if not folder:
            return
        written, notes = [], []
        try:
            wand_uv, _L, bg_uv, _fr, _bf = charuco.pairs_and_extras(scan)
            written += calibio.write_argus_xypts(folder, wand_uv, bg_uv, scan.sizes)
        except Exception as e:      # noqa: BLE001 - a read-only folder, a full drive: said on the page
            self.status.setText(plain_error(e, "The Argus files could not be saved", short=True))
            return
        lenses = [(k + 1, w.result_lenses.get(v) or w.usable_lens(v)) for k, v in enumerate(scan.cameras)]
        missing = [scan.names[n - 1] for n, prof in lenses if prof is None]
        if missing:
            notes.append(f"{calibio.ARGUS_PROFILE_NAME} is not written: {', '.join(missing)} "
                         f"{'has' if len(missing) == 1 else 'have'} no lens profile")
        else:
            path = Path(folder) / calibio.ARGUS_PROFILE_NAME
            try:
                calibio.write_argus_profiles(path, lenses)
                written.append(str(path))
            except calibio.CalibFormatError as e:
                notes.append(f"{calibio.ARGUS_PROFILE_NAME} is not written: {e}")
            except Exception as e:      # noqa: BLE001
                notes.append(plain_error(e, f"{calibio.ARGUS_PROFILE_NAME} could not be saved", short=True))
        cams = ", ".join(f"{k + 1} = {nm}" for k, nm in enumerate(scan.names))
        self.status.setText("Saved for Argus: " + ", ".join(Path(x).name for x in written)
                            + f" (cameras {cams}; pixels y up, as Argus counts them)."
                            + "".join(f" {n[0].upper() + n[1:]}." for n in notes))

    def isComplete(self) -> bool:
        w = self.wiz
        if self._settings_key() != self._key:
            return False
        k = w.page_board.choice()
        if k == "lenses":
            return bool(w.result_lenses)
        return w.result_calibration is not None


class CharucoWizard(QWizard):
    def __init__(self, parent, project, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Calibrate cameras with a ChArUco board")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setMinimumSize(780, 640)
        self.project = project
        self.start_dir = start_dir
        self.scan: charuco.RigScan | None = None
        self.result = None                          # WandResult (None for "lenses only")
        self.result_calibration = None              # over every project camera (None = not calibrated)
        self.result_lenses: dict = {}               # {project camera: LensProfile}, applied on Finish
        self.page_board = BoardPage(self)
        self.page_run = RunPage(self)
        self.addPage(self.page_board)
        self.addPage(self.page_run)

    def usable_lens(self, c: int):
        """Camera c's lens profile fitted to its pictures, else None (`calibwizard.lens_for_camera`)."""
        from kinetrace.calibwizard import lens_for_camera
        p = self.project
        prof = p.lenses[c] if c < len(p.lenses) else None
        return lens_for_camera(prof, p.sessions[c], p.name(c))[0] if prof is not None else None

    def choice(self) -> str:
        return self.page_board.choice()

    def done(self, r):
        stop_page_threads(self)
        super().done(r)
