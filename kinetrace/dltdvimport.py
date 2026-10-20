"""The two windows of File > Import > DLTdv Project (G211). The reading is `dltdvproject.py`, the measuring
`clickshift.py`, the building `app._import_dltdv_project`; nothing here changes data or opens a video.

`ClickShiftDialog` (only when some camera's clicks fit other frames clearly better): one row per camera with
the verdict in plain words, the hand check (one click cut out at DLTdv's frame and at the better one, the
click marked) and, for an offer, a tick "use the better fit" (on by default). Import builds the project with
the ticked shifts; Cancel imports nothing.

`ImportReportDialog`: what came in and what did not, every number with its verdict: the cameras (where each
video was found, offset, rate, lens, clicks, the frame check), the points, the 3D check against DLTdv's own 3D
with a hand check, anything left out and why.
"""
from __future__ import annotations

from html import escape

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QScrollArea, QSizePolicy, QTextBrowser, QVBoxLayout, QWidget)

CROP_SHOWN = 180           # px: a crop is scaled to this size on screen


def _pixmap(bgr: np.ndarray, xy) -> QPixmap:
    """A crop with the click drawn on it (a copy: the array is never drawn on), scaled for the screen."""
    img = np.ascontiguousarray(bgr).copy()
    s = CROP_SHOWN / max(img.shape[:2])
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                     interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    x, y = int(xy[0] * s + s / 2), int(xy[1] * s + s / 2)
    for col, w in (((0, 0, 0), 3), ((60, 220, 255), 1)):
        cv2.line(img, (x - 14, y), (x - 5, y), col, w, cv2.LINE_AA)
        cv2.line(img, (x + 5, y), (x + 14, y), col, w, cv2.LINE_AA)
        cv2.line(img, (x, y - 14), (x, y - 5), col, w, cv2.LINE_AA)
        cv2.line(img, (x, y + 5), (x, y + 14), col, w, cv2.LINE_AA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


def _wrap(text: str, rich: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.RichText if rich else Qt.PlainText)
    lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)      # a long caption never widens the window
    return lab


class ClickShiftDialog(QDialog):
    """`fits`: one `clickshift.ClickFit` (or None: no clicks / not checked) per camera; `names` the cameras'
    video names. `shifts()` = the whole-frame shift chosen for each camera (0 = DLTdv's frames)."""

    def __init__(self, parent, names: list[str], fits: list):
        super().__init__(parent)
        self.setWindowTitle("DLTdv project: do the clicks sit on the right pictures?")
        self.fits = list(fits)
        self.use_boxes: list[QCheckBox | None] = []
        lay = QVBoxLayout(self)
        lay.addWidget(_wrap(
            "A DLTdv project numbers frames the way the MATLAB that opened it read the video, and newer MATLAB "
            "versions number some videos (GoPro files among them) differently. Kinetrace looked at the pictures "
            "under each camera's clicks at DLTdv's frame numbers and at nearby frames. Where another frame fits "
            "clearly better, both pictures are shown: the click (cross) should sit on the animal. Ticked shifts "
            "move that camera's clicks AND its offset together, so the 3D is the same either way; only the "
            "pictures under the clicks change."))
        body = QWidget()
        grid = QVBoxLayout(body)
        for c, (name, fit) in enumerate(zip(names, self.fits)):
            box = QFrame()
            box.setFrameShape(QFrame.StyledPanel)
            g = QGridLayout(box)
            g.addWidget(_wrap(f"<b>Camera {c + 1}: {escape(name)}</b>", rich=True), 0, 0, 1, 2)
            g.addWidget(_wrap("Not checked: no clicks in this camera." if fit is None else fit.sentence), 1, 0, 1, 2)
            if fit is not None and fit.crops is not None:
                for k, (crop, xy, fr, cap) in enumerate(zip(fit.crops, fit.crop_xy, fit.crop_frames,
                                                            ("at DLTdv's frame", "at the best fit"))):
                    if k == 1 and fit.best == 0:
                        break
                    pic = QLabel()
                    pic.setPixmap(_pixmap(crop, xy))
                    col = QVBoxLayout()
                    col.addWidget(pic)
                    col.addWidget(QLabel(f"frame {fr} ({cap})"))
                    w = QWidget()
                    w.setLayout(col)
                    g.addWidget(w, 2, k)
            use = None
            if fit is not None and fit.offer is not None:
                use = QCheckBox(f"Use the better fit: move this camera's clicks and its offset by {fit.offer:+d} "
                                "frame(s)")
                use.setChecked(True)
                g.addWidget(use, 3, 0, 1, 2)
            self.use_boxes.append(use)
            grid.addWidget(box)
        grid.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Import")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Import nothing")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(640, 700)

    def shifts(self) -> list[int]:
        return [int(f.offer) if (box is not None and box.isChecked() and f is not None and f.offer is not None)
                else 0 for f, box in zip(self.fits, self.use_boxes)]


class ImportReportDialog(QDialog):
    """The import's report (HTML); `text` is the same in plain text."""

    def __init__(self, parent, title: str, html: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(html)
        self.text = view.toPlainText()
        lay.addWidget(view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.accept)
        bb.accepted.connect(self.accept)
        row.addWidget(bb)
        lay.addLayout(row)
        self.resize(760, 620)


def verdict_3d(cmp: dict) -> str:
    """The 3D check in one plain sentence (starts "3D check: <verdict>")."""
    if not cmp.get("n"):
        return f"3D check: not possible — {cmp.get('why', 'nothing to compare')}."
    v = cmp["verdict"]
    what = {"good": "the same 3D as DLTdv's, as it should be (the same clicks, calibration and lenses)",
            "ok": "close to DLTdv's but not the same: check the lenses and the cameras' order",
            "poor": "NOT DLTdv's 3D: the frames, the pixels or the lenses were read differently; check the "
                    "cameras' order and the frame check before using it"}[v]
    return (f"3D check: {v} — {what}. Kinetrace's 3D and DLTdv's stored 3D differ by {cmp['median']:.3g} "
            f"(median) / {cmp['p95']:.3g} (95 %) over {cmp['n']} points: {cmp['rel'] * 100:.3g} % of the size of "
            f"the clicked volume ({cmp['extent']:.3g}), in the calibration's own units."
            + (f" {cmp['missing']} of DLTdv's points have no Kinetrace 3D (a camera left out, or a click outside "
               "its video)." if cmp.get("missing") else ""))


def report_html(dv, kept: list[int], found: list, shifts: list[int], fits: list, left_out: dict,
                applied: list[dict], cmp: dict, cal_note: str, offsets: list[float] | None = None,
                skipped: bool = False) -> str:
    """The report. `kept` = the project's cameras (DLTdv video indices, in order), `applied` the
    `trackio.apply` summaries per kept camera, `left_out` {video: reason}, `offsets` the project's own
    offsets (after any shift taken), `skipped` = the user cancelled the frame check."""
    from pathlib import Path
    rows = []
    for i, c in enumerate(kept):
        fit = fits[i] if i < len(fits) else None
        lens = type(dv.lenses[c]).__name__ == "LWMUndistort"
        check = (("skipped (Cancel)" if skipped else "not checked (no clicks)") if fit is None else
                 {"fits": "fits DLTdv's frames (good)", "better": f"better {fit.best:+d}",
                  "cannot": "cannot tell", "unclear": f"unclear (best {fit.best:+d})"}[fit.verdict])
        if shifts[i]:
            check += f"; shift {shifts[i]:+d} TAKEN"
        elif fit is not None and fit.offer is not None:
            check += "; shift NOT taken"
        rows.append(f"<tr><td>{i + 1}</td><td>{escape(dv.video_files[c])}</td>"
                    f"<td>{escape(str(Path(found[c]).parent))}</td><td>{dv.offsets[c]:g}</td>"
                    f"<td>{'' if offsets is None else f'{offsets[i]:g}'}</td>"
                    f"<td>{dv.ratios[c]:g}</td><td>{'LWM profile' if lens else 'as filmed'}</td>"
                    f"<td>{dv.n_clicks(c)}</td><td>{escape(check)}</td></tr>")
    out = [f"<h3>{escape(Path(dv.path).name)}</h3>",
           f"<p>DLTdv {dv.version:g}; {len(dv.point_names)} point(s) in {dv.n_videos} video(s); "
           f"{len(dv.rows)} of its {dv.n_rows:,} rows hold clicks. The .mat is only read, never changed. "
           "This is a NEW, unsaved project: <b>File → Save Project As…</b> keeps it.</p>",
           "<h4>Cameras</h4><table border=1 cellspacing=0 cellpadding=3><tr><th>#</th><th>video</th><th>found in"
           "</th><th>DLTdv offset</th><th>Kinetrace offset</th><th>rate ratio</th><th>lens</th><th>clicks</th><th>frame check</th></tr>"
           + "".join(rows) + "</table>"]
    if left_out:
        out.append("<p><b>Left out:</b> " + "; ".join(f"video {c + 1} ({escape(dv.video_files[c])}): "
                                                     f"{escape(r)}" for c, r in sorted(left_out.items())) + "</p>")
    out.append("<h4>Points</h4><p>" + escape(", ".join(dv.point_names)) + "</p><ul>"
               + "".join(f"<li>Camera {i + 1}: {escape(a['sentence'])}</li>" for i, a in enumerate(applied))
               + "</ul>")
    shared = [(i, dv.shared_frames(c, shifts[i])) for i, c in enumerate(kept)]
    if any(n for _, n in shared):
        out.append("<p>" + "; ".join(f"camera {i + 1}: {n} click(s) shared a frame with another row (the later "
                                     "row was kept)" for i, n in shared if n) + ".</p>")
    out.append(f"<h4>Calibration and 3D</h4><p>{escape(cal_note)}</p><p>{escape(verdict_3d(cmp))}</p>")
    if cmp.get("hand"):
        out.append(f"<p><b>Hand check:</b> {escape(cmp['hand'])}. Scrub to that frame: Kinetrace's 3D view and "
                   "DLTdv's xyzpts should give that point.</p>")
    if dv.notes:
        out.append("<h4>Notes</h4><ul>" + "".join(f"<li>{escape(n)}</li>" for n in dv.notes) + "</ul>")
    return "".join(out)
