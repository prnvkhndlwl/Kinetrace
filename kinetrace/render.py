"""Overlay video render: the tracked result burned into an MP4.

For talks, for a collaborator without the app, and for QC at a glance: every
frame of a chosen range is decoded, the markers / names / skeleton bones /
silhouette / trails / frame counter / active events / notes are drawn on it
with OpenCV, and the result is written with `cv2.VideoWriter`.

`draw_overlay` is a pure function (numpy + cv2, no Qt) so it is testable and
reusable; `OverlayRenderer` is the QThread that runs it over a frame range
with its OWN VideoCapture (one capture per thread — the rule everywhere in
this app). Coordinates are native video pixels; the output can be scaled
down (`OverlayOptions.scale`) and everything drawn scales with it.

Codec: H.264 (`avc1`) is tried first because it plays everywhere; OpenCV's
bundled FFmpeg may not carry an encoder for it, in which case the writer
falls back to MPEG-4 part 2 (`mp4v`), which every player also opens.
"""

from __future__ import annotations

import sys
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class OverlayOptions:
    start: int = 0
    end: int = 0                 # inclusive
    scale: float = 1.0           # output size factor (1.0 = native)
    markers: bool = True
    names: bool = True
    bones: bool = True
    mask: bool = True
    trails: int = 30             # frames of trail, 0 = none
    frame_number: bool = True
    events: bool = True
    notes: bool = True
    marker_px: int = 7           # marker radius in OUTPUT pixels
    fps: float | None = None     # None = the video file's own rate (the session's `file_fps`)


def _bgr(color) -> tuple[int, int, int]:
    r, g, b = (int(c) for c in color)
    return (b, g, r)


def _text(img, text: str, org: tuple[int, int], scale: float, color, thick: int = 1) -> None:
    """Outlined text (dark halo) so it reads on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(bgr: np.ndarray, frame: int, session, opts: OverlayOptions,
                 bones: list[tuple[int, int]] | None = None,
                 mask_color=(77, 227, 176), mask_opacity: float = 0.35) -> np.ndarray:
    """Draw the session's state at `frame` on a BGR image (native size or a
    scaled copy — `opts.scale` says which). Returns the image (drawn in place
    when it is already the output size, else the scaled copy)."""
    sc = float(opts.scale)
    if sc != 1.0:
        bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    out = bgr
    h, w = out.shape[:2]
    # (G123) `w` is already the scaled width: scaling by `sc` again gave a half-size 4K overlay the minimum font
    font_scale = max(0.4, 0.55 * (w / 1280.0) ** 0.5) if w >= 640 else 0.4
    s = session
    T = s.n_frames
    if not (0 <= frame < T):
        return out
    r = max(2, int(round(opts.marker_px)))

    # silhouette (under everything else)
    if opts.mask and s.masks is not None and s.animal is not None:
        polys = s.masks.contours.get(frame)
        if polys:
            layer = out.copy()
            col = _bgr(getattr(s.animal, "color", mask_color))
            pts = [np.round(np.asarray(p, np.float64) * sc).astype(np.int32).reshape(-1, 1, 2)
                   for p in polys if len(p) >= 3]
            if pts:
                cv2.fillPoly(layer, pts, col)
                cv2.addWeighted(layer, float(mask_opacity), out, 1.0 - float(mask_opacity), 0, out)
                cv2.polylines(out, pts, True, col, 1, cv2.LINE_AA)

    occ = s.occluded[frame] if getattr(s, "occluded", None) is not None else np.zeros(s.n_points, bool)
    pos = s.tracks[frame] * sc
    valid = s.tracked[frame] & np.isfinite(s.tracks[frame]).all(axis=1)
    valid &= ~occ                       # cells marked hidden by hand are not drawn

    # trails
    if opts.trails > 0:
        lo = max(0, frame - int(opts.trails))
        for j, meta in enumerate(s.points):
            if not meta.display:
                continue
            seg = s.tracks[lo:frame + 1, j]
            ok = s.tracked[lo:frame + 1, j] & np.isfinite(seg).all(axis=1)
            if s.occluded is not None:
                ok &= ~s.occluded[lo:frame + 1, j]
            if ok.sum() < 2:
                continue
            col = _bgr(meta.color)
            pts = np.round(seg * sc).astype(np.int32)
            # break the polyline at gaps
            run: list[np.ndarray] = []
            for k in range(len(pts)):
                if ok[k]:
                    run.append(pts[k])
                elif len(run) >= 2:
                    cv2.polylines(out, [np.asarray(run).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
                    run = []
                else:
                    run = []
            if len(run) >= 2:
                cv2.polylines(out, [np.asarray(run).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)

    # bones
    if opts.bones and bones:
        for a, b in bones:
            if a < s.n_points and b < s.n_points and valid[a] and valid[b]:
                pa = tuple(int(v) for v in np.round(pos[a]))
                pb = tuple(int(v) for v in np.round(pos[b]))
                cv2.line(out, pa, pb, (235, 235, 235), 1, cv2.LINE_AA)

    # markers + names
    if opts.markers:
        for j, meta in enumerate(s.points):
            if not meta.display or not valid[j]:
                continue
            c = (int(round(pos[j][0])), int(round(pos[j][1])))
            col = _bgr(meta.color)
            if s.visibility[frame, j]:
                cv2.circle(out, c, r, col, -1, cv2.LINE_AA)
                cv2.circle(out, c, r, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, c, r, col, 2, cv2.LINE_AA)
            if meta.kind == "group":
                ol = meta.outline_at(pos[j] / sc) if hasattr(meta, "outline_at") else None
                if ol is not None:
                    cv2.polylines(out, [np.round(ol * sc).astype(np.int32).reshape(-1, 1, 2)], True,
                                  col, 1, cv2.LINE_AA)
                else:
                    cv2.circle(out, c, int(round(meta.radius * sc)), col, 1, cv2.LINE_AA)
            if opts.names:
                _text(out, meta.name, (c[0] + r + 3, c[1] - r - 2), font_scale, col)

    # frame counter + events + notes (top-left block, bottom-left notes)
    y = int(22 * max(font_scale / 0.55, 0.8)) + 4
    line_h = int(24 * max(font_scale / 0.55, 0.8))
    if opts.frame_number:
        t = f"frame {frame}"
        if s.fps:
            t += f"   {frame / s.fps:8.3f} s"
        _text(out, t, (8, y), font_scale, (255, 255, 255))
        y += line_h
    if opts.events:
        for e in s.events:
            if e.start <= frame <= e.end:
                _text(out, f"{e.name}  [{e.start}-{e.end}]", (8, y), font_scale, _bgr(e.color))
                y += line_h
    if opts.notes:
        n = s.notes.get(frame) if getattr(s, "notes", None) else None
        if n:
            _text(out, "note: " + str(n.get("text", ""))[:120], (8, h - 10), font_scale, (120, 255, 200))
    return out


def freeze_session(s, start: int = 0, end: int | None = None) -> SimpleNamespace:
    """A private copy of everything `draw_overlay` reads, taken on the GUI
    thread (I38). The renderer draws from its own thread while the app stays
    editable, and the session's mutators swap arrays one attribute at a time
    (a point is appended to `points` before the arrays grow), so drawing from
    the live session crashed on "index 10 is out of bounds" or mixed pre- and
    post-edit data. Silhouettes are copied only for `start..end`, the frames
    the render draws; the arrays whole (trails look back before `start`)."""
    end = s.n_frames - 1 if end is None else int(end)
    masks = None
    if s.masks is not None:
        masks = SimpleNamespace(
            contours={f: [np.array(p, copy=True) for p in ps]
                      for f, ps in s.masks.contours.items() if start <= f <= end},
            midline={f: np.array(m, copy=True)
                     for f, m in s.masks.midline.items() if start <= f <= end})
    occ = getattr(s, "occluded", None)
    return SimpleNamespace(
        n_frames=s.n_frames, n_points=s.n_points, fps=s.fps, width=s.width, height=s.height,
        file_fps=getattr(s, "file_fps", s.fps),
        tracks=s.tracks.copy(), tracked=s.tracked.copy(), visibility=s.visibility.copy(),
        occluded=None if occ is None else occ.copy(),
        points=[p.copy() for p in s.points],
        events=[copy.copy(e) for e in s.events],
        notes=copy.deepcopy(getattr(s, "notes", None) or {}),
        masks=masks, animal=None if s.animal is None else s.animal.copy())


def open_writer(path: str | Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 when the bundled FFmpeg can encode it, else MPEG-4. Returns the
    writer and the codec tag that worked."""
    path = str(path)
    # Windows Media Foundation encodes H.264 natively; asking for it FIRST
    # avoids OpenCV's FFmpeg backend logging a failed libopenh264 attempt
    # before it falls through to MSMF on its own
    attempts = [("avc1", cv2.CAP_ANY), ("mp4v", cv2.CAP_ANY)]
    if sys.platform.startswith("win"):           # Media Foundation exists only on Windows
        attempts.insert(0, ("avc1", cv2.CAP_MSMF))
    for tag, api in attempts:
        vw = cv2.VideoWriter(path, api, cv2.VideoWriter_fourcc(*tag), float(fps),
                             (int(size[0]), int(size[1])))
        if vw.isOpened():
            return vw, tag
        vw.release()
    raise RuntimeError("OpenCV could not open a video writer for " + path)


class OverlayRenderer(QThread):
    """Renders `opts.start..opts.end` of the session's video with the overlay
    into `out_path`. Emits progress (done, total), then finished_ok(path,
    codec) or error(message). `request_cancel` stops after the current
    frame and deletes the partial file.

    Built on the GUI thread: it draws from a FROZEN copy of the session
    (`freeze_session`, I38), so edits made while it renders are not in the
    video and cannot crash it. After `finished_ok`, `frames_written` is the
    number of frames really in the file and `note` a sentence saying why it is
    short of the range ("" when it is not) — a video can end before the range
    the session believed it had (I41)."""
    progress = Signal(int, int)
    finished_ok = Signal(str, str)
    error = Signal(str)

    def __init__(self, session, video_path: str, out_path: str, opts: OverlayOptions,
                 bones: list[tuple[int, int]] | None, mask_opacity: float = 0.35):
        super().__init__()
        self.session = freeze_session(session, opts.start, opts.end)
        self.video_path = str(video_path)
        self.out_path = str(out_path)
        self.opts = opts
        # (I44) the dialog's "Skeleton bones" tick decides, not the canvas's
        # View toggle: with the tick on, an empty list means the skeleton's bones
        if opts.bones and not bones and hasattr(session, "bones"):
            bones = session.bones()
        self.bones = list(bones or [])
        self.mask_opacity = float(mask_opacity)
        self.frames_written = 0
        self.note = ""
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def _discard(self) -> None:
        try:
            Path(self.out_path).unlink(missing_ok=True)
        except OSError:
            pass

    def _output_fps(self, header_fps: float) -> float:
        """The rate the overlay plays at (I226): the caller's `opts.fps`, else the
        rate the app measured for the video FILE (`file_fps`: from its timestamps
        when the header is wrong), else the container header's. A rate that is not a
        positive finite number is refused with a sentence -- NaN used to give a
        one-frame MP4 reported as complete."""
        if self.opts.fps is not None:
            candidates = [("the requested frame rate", self.opts.fps)]
        else:
            candidates = [("the frame rate measured for this video", getattr(self.session, "file_fps", None)),
                          ("the frame rate in the session", getattr(self.session, "fps", None)),
                          ("the frame rate in the video's header", header_fps)]
        for what, v in candidates:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                return v
            if self.opts.fps is not None:
                raise RuntimeError(f"{what} is {v:g}: it must be a number above 0 frames per second")
        raise RuntimeError("the video's frame rate is not known (none is stored in its file and none was "
                           "measured), so the overlay cannot be timed. Set the camera's frame rate with the "
                           "fps button in the CAMERAS panel first.")

    def run(self) -> None:
        from kinetrace.video_source import open_capture
        cap = None
        vw = None
        created = False             # the writer made / truncated out_path: ours to delete
        try:
            cap = open_capture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("could not open " + self.video_path)
            fps = self._output_fps(float(cap.get(cv2.CAP_PROP_FPS)))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            sc = float(self.opts.scale)
            size = (max(2, int(round(w * sc))), max(2, int(round(h * sc))))
            f0, f1 = int(self.opts.start), int(self.opts.end)
            total = max(0, f1 - f0 + 1)
            vw, codec = open_writer(self.out_path, fps, size)
            created = True
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            done = 0
            for f in range(f0, f1 + 1):
                if self._cancel:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                img = draw_overlay(bgr, f, self.session, self.opts, self.bones,
                                   mask_opacity=self.mask_opacity)
                if img.shape[1] != size[0] or img.shape[0] != size[1]:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                vw.write(img)
                done += 1
                if done % 5 == 0 or done == total:
                    self.progress.emit(done, total)
            vw.release()
            vw = None
            self.frames_written = done
            if self._cancel:
                self._discard()
                self.error.emit("cancelled")
                return
            if done == 0:
                # (I41) a range wholly past the frames that decode: an empty MP4
                # reported as "written" is worse than saying so
                self._discard()
                self.error.emit(f"frame {f0} could not be decoded (the video ends before it, or the "
                                "file is damaged there), so nothing was written. Choose a range "
                                "inside the video.")
                return
            if done < total:
                # (I41) the loop used to stop at the first failed read and report
                # the REQUESTED count; the app shows this sentence instead
                self.note = (f"Frame {f0 + done} could not be decoded (the video ends there, or the "
                             f"file is damaged), so this overlay stops at frame {f0 + done - 1}: "
                             f"{done} of {total} frames were written.")
                self.progress.emit(done, total)
            self.finished_ok.emit(self.out_path, codec)
        except Exception as e:      # noqa: BLE001
            if vw is not None:
                vw.release()
                vw = None
            if created:
                self._discard()     # a truncated file is not a result
            self.error.emit(str(e))
        finally:
            if vw is not None:
                vw.release()
            if cap is not None:
                cap.release()


# ------------------------------------------------------------------ dialog


class OverlayDialog:
    """Options for File → Export Overlay Video. Built lazily as a QDialog so
    the pure-function part of this module imports without QtWidgets."""

    def __new__(cls, parent, session, default_path: str, current: int,
                sel_range: tuple[int, int] | None = None, n_frames: int | None = None):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                                       QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                                       QRadioButton, QSpinBox, QVBoxLayout, QWidget)
        from kinetrace import theme

        # (I41) the frames that DECODE (the view's verified count) bound every
        # range: a project saved with a header's over-count keeps the longer
        # session, and a range in that phantom tail wrote a short or empty MP4
        T = session.n_frames if not n_frames else max(1, min(int(session.n_frames), int(n_frames)))
        if sel_range is not None:
            sel_range = None if sel_range[0] > T - 1 else (sel_range[0], min(sel_range[1], T - 1))
        events = [(e, min(e.end, T - 1)) for e in session.events if e.start <= T - 1]

        class _Dlg(QDialog):
            def __init__(self):
                super().__init__(parent)
                self.setWindowTitle("Export overlay video")
                self.setMinimumWidth(560)
                self.result_options: OverlayOptions | None = None
                self.result_path = ""
                lay = QVBoxLayout(self)
                intro = QLabel("Writes an MP4 with the tracked result drawn on every frame: markers, "
                               "names, skeleton, silhouette, trails, the frame counter, events and "
                               "notes. The original video is never modified.")
                intro.setWordWrap(True)
                lay.addWidget(intro)
                form = QFormLayout()
                # range
                rng = QWidget()
                rl = QVBoxLayout(rng)
                rl.setContentsMargins(0, 0, 0, 0)
                self.r_all = QRadioButton(f"Whole video (frames 0–{T - 1})")
                self.r_sel = QRadioButton("Selected frame window on the timeline"
                                          + (f" ({sel_range[0]}–{sel_range[1]})" if sel_range else ""))
                self.r_sel.setEnabled(sel_range is not None)
                self.r_ev = QRadioButton("An event:")
                self.ev_box = QComboBox()
                for e, end in events:
                    self.ev_box.addItem(f"{e.name}  [{e.start}–{end}]", (e.start, end))
                self.r_ev.setEnabled(bool(events))
                self.ev_box.setEnabled(bool(events))
                self.r_custom = QRadioButton("Frames:")
                self.f0 = QSpinBox()
                self.f0.setRange(0, T - 1)
                self.f1 = QSpinBox()
                self.f1.setRange(0, T - 1)
                self.f0.setValue(max(0, current - 150))
                self.f1.setValue(min(T - 1, current + 150))
                row_ev = QHBoxLayout()
                row_ev.addWidget(self.r_ev)
                row_ev.addWidget(self.ev_box, 1)
                row_c = QHBoxLayout()
                row_c.addWidget(self.r_custom)
                row_c.addWidget(self.f0)
                row_c.addWidget(QLabel("to"))
                row_c.addWidget(self.f1)
                row_c.addStretch(1)
                rl.addWidget(self.r_all)
                rl.addWidget(self.r_sel)
                rl.addLayout(row_ev)
                rl.addLayout(row_c)
                (self.r_sel if sel_range else self.r_all).setChecked(True)
                form.addRow("Range", rng)
                self.scale = QComboBox()
                for label, v in (("Full size", 1.0), ("Half size", 0.5), ("Quarter size", 0.25)):
                    self.scale.addItem(label, v)
                self.scale.setCurrentIndex(1 if session.width > 2000 else 0)
                self.scale.setToolTip("Half size is plenty for a talk and renders 4x faster")
                form.addRow("Output size", self.scale)
                opts = QWidget()
                ol = QVBoxLayout(opts)
                ol.setContentsMargins(0, 0, 0, 0)
                self.c_markers = QCheckBox("Markers")
                self.c_names = QCheckBox("Point names")
                self.c_bones = QCheckBox("Skeleton bones")
                self.c_mask = QCheckBox("Silhouette")
                self.c_trails = QCheckBox("Trails (last 30 frames)")
                self.c_frame = QCheckBox("Frame number and time")
                self.c_events = QCheckBox("Event names while active")
                self.c_notes = QCheckBox("Frame notes")
                for c in (self.c_markers, self.c_names, self.c_bones, self.c_mask, self.c_trails,
                          self.c_frame, self.c_events, self.c_notes):
                    c.setChecked(True)
                    ol.addWidget(c)
                self.c_mask.setEnabled(session.masks is not None and session.animal is not None)
                self.c_bones.setEnabled(bool(session.bones()))
                form.addRow("Draw", opts)
                self.marker = QSpinBox()
                self.marker.setRange(2, 30)
                self.marker.setValue(7)
                self.marker.setSuffix(" px")
                form.addRow("Marker size", self.marker)
                prow = QWidget()
                pl = QHBoxLayout(prow)
                pl.setContentsMargins(0, 0, 0, 0)
                self.path = QLineEdit(default_path)
                btn = QPushButton("Choose…")
                btn.clicked.connect(self._pick)
                pl.addWidget(self.path, 1)
                pl.addWidget(btn)
                form.addRow("Save as", prow)
                lay.addLayout(form)
                hint = QLabel("H.264 when this OpenCV build can encode it, otherwise MPEG-4; both play "
                              "in any player. Rendering runs in the background — you can keep working; "
                              "edits made while it renders are not in this video.")
                hint.setWordWrap(True)
                hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
                lay.addWidget(hint)
                btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                btns.button(QDialogButtonBox.Ok).setText("Render")
                btns.accepted.connect(self._accept)
                btns.rejected.connect(self.reject)
                lay.addWidget(btns)

            def _pick(self):
                p, _ = QFileDialog.getSaveFileName(self, "Overlay video", self.path.text(),
                                                   "MP4 video (*.mp4)")
                if p:
                    self.path.setText(p if p.lower().endswith(".mp4") else p + ".mp4")

            def _range(self) -> tuple[int, int]:
                if self.r_sel.isChecked() and sel_range:
                    return int(sel_range[0]), int(sel_range[1])
                if self.r_ev.isChecked() and self.ev_box.count():
                    a, b = self.ev_box.currentData()
                    return int(a), int(b)
                if self.r_custom.isChecked():
                    a, b = self.f0.value(), self.f1.value()
                    return (a, b) if a <= b else (b, a)
                return 0, T - 1

            def _accept(self):
                a, b = self._range()
                self.result_options = OverlayOptions(
                    start=a, end=b, scale=float(self.scale.currentData()),
                    markers=self.c_markers.isChecked(), names=self.c_names.isChecked(),
                    bones=self.c_bones.isChecked() and self.c_bones.isEnabled(),
                    mask=self.c_mask.isChecked() and self.c_mask.isEnabled(),
                    trails=30 if self.c_trails.isChecked() else 0,
                    frame_number=self.c_frame.isChecked(), events=self.c_events.isChecked(),
                    notes=self.c_notes.isChecked(), marker_px=self.marker.value())
                p = self.path.text().strip()
                self.result_path = p if p.lower().endswith(".mp4") else p + ".mp4"
                self.accept()

        return _Dlg()
