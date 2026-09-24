"""Kinematics from the 3D reconstruction: smoothed positions, velocity and
acceleration per landmark, with a report a first-time user can act on.

Why smoothing comes first: differentiating tracked positions amplifies their
noise - a 0.5 mm jitter at 240 fps becomes 0.085 m/s of velocity noise and
18 m/s^2 of acceleration noise (RMS), twice gravity. Every kinematics pipeline
low-passes first. The cutoff is chosen by Winter's residual analysis
(Biomechanics and Motor Control of Human Movement): filter at many cutoffs,
measure the RMS difference to the raw signal, fit a line to the high-cutoff
tail (0.5-0.9 x Nyquist, where only noise is being removed) and take the
lowest cutoff whose residual is no larger than that line's intercept - the
point where the filter starts eating signal instead of noise. Landmarks that
do not move are left out of that choice, and a movement whose main rhythm is
too fast to tell apart from jitter is kept and said to be uncertain. The
report states the cutoff and why, and names any landmark that moves faster
than the cutoff keeps, so the number is never silent.

Positions are in the project's unit (metres after a wand calibration in
metres; "wand lengths" when the wand length was not given); time is the
reference camera's frames / fps. Gaps stay gaps: each contiguous run of a
landmark is filtered on its own and derivatives never cross a gap. Pure
numpy / scipy, no Qt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MIN_RUN = 12             # samples: shorter runs are left unfiltered (filtfilt needs a few cycles)
ORDER = 4                # Butterworth order (2nd order applied forward + backward)
G = 9.81                 # m/s^2, for the hand check
TAIL_BAND = (0.5, 0.9)   # x Nyquist: the top of the spectrum, where the residual is noise only (Winter's linear tail)
RHYTHM_PEAKINESS = 100.0     # a spectral peak this many times the median power is a real rhythm (white noise: ~5)
KEEP_WARN = 0.95         # say so when the cutoff keeps less than this share of the movement's main rhythm
KEEP_MIN = 0.8           # a pick keeping less than this of the main rhythm measured movement as noise: distrust it
OWN_PICK_WARN = 1.5      # a landmark whose own pick is this far above the shared cutoff loses movement: say so


def contiguous_runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """[(a, b)] inclusive index ranges where `ok` (1-D bool) is True."""
    ok = np.asarray(ok, bool)
    if not ok.any():
        return []
    d = np.diff(ok.astype(np.int8))
    starts = list(np.nonzero(d == 1)[0] + 1)
    ends = list(np.nonzero(d == -1)[0])
    if ok[0]:
        starts = [0] + starts
    if ok[-1]:
        ends = ends + [len(ok) - 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _quad_extend(seg: np.ndarray, n_out: int, before: bool) -> np.ndarray:
    """`n_out` samples continuing the (k, D) `seg` with the quadratic fitted to
    it: before its first sample (`before`) or after its last."""
    k = seg.shape[0]
    t = np.arange(k, dtype=np.float64)
    c = np.polynomial.polynomial.polyfit(t, seg, 2)                 # (3, D)
    te = -np.arange(n_out, 0, -1, dtype=np.float64) if before else k + np.arange(n_out, dtype=np.float64)
    return np.polynomial.polynomial.polyval(te, c).T                # (n_out, D)


def lowpass(x: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass of a (T, ...) array along axis 0; a
    cutoff at or above Nyquist returns a copy.

    Each end is extended by the quadratic fitted to its last cutoff period
    before filtering (I5): scipy's default odd padding (9 samples) mirrors a
    parabola upside down about the end point, so a perfect free fall read
    1.2 m/s^2 at the start of a run and 18 at its end. A quadratic carries
    the local velocity and curvature on, and the pad (3 periods) outlasts the
    filter's start-up transient. The interior of a run is unchanged."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, np.float64)
    nyq = 0.5 * float(fps)
    if cutoff_hz <= 0 or cutoff_hz >= 0.98 * nyq or x.shape[0] < MIN_RUN:
        return x.copy()
    sos = butter(ORDER // 2, cutoff_hz / nyq, btype="low", output="sos")
    n = x.shape[0]
    period = float(fps) / float(cutoff_hz)                     # samples per cutoff period
    k = int(min(n, max(MIN_RUN, np.ceil(period))))             # edge samples the quadratic is fitted to
    pad = int(max(3 * MIN_RUN, np.ceil(3.0 * period)))
    x2 = x.reshape(n, -1)
    xe = np.vstack([_quad_extend(x2[:k], pad, True), x2, _quad_extend(x2[n - k:], pad, False)])
    y = sosfiltfilt(sos, xe, axis=0, padtype=None)
    return y[pad:pad + n].reshape(x.shape)


def kept_fraction(f_hz: float, cutoff_hz: float | None, fps: float) -> float:
    """Share of a movement's amplitude at `f_hz` that `lowpass` keeps: the
    forward-backward 2nd-order Butterworth is |H|^2 = 1 / (1 + r^4), r = the
    prewarped frequency ratio. 0.5 AT the cutoff, 0.94 at half of it."""
    nyq = 0.5 * float(fps)
    if cutoff_hz is None or cutoff_hz <= 0 or cutoff_hz >= 0.98 * nyq:
        return 1.0
    if f_hz >= nyq:
        return 0.0
    r = np.tan(np.pi * f_hz / fps) / np.tan(np.pi * cutoff_hz / fps)
    return float(1.0 / (1.0 + r ** 4))


def cutoff_keeping(f_hz: float, fps: float, keep: float = KEEP_WARN) -> float:
    """The lowest cutoff whose `lowpass` keeps `keep` of a movement at `f_hz`
    (capped at 0.9 x Nyquist)."""
    nyq = 0.5 * float(fps)
    ratio = (1.0 / keep - 1.0) ** 0.25
    fc = float(fps) / np.pi * np.arctan(np.tan(np.pi * min(f_hz, 0.999 * nyq) / fps) / ratio)
    return float(min(fc, 0.9 * nyq))


def dominant_rhythm(x: np.ndarray, fps: float) -> float | None:
    """The frequency (Hz) of the strongest spectral peak of a (T, D) stretch
    without gaps, linear trend removed; None when no peak stands out of the
    noise (a still landmark, or motion without a rhythm)."""
    x = np.asarray(x, np.float64).reshape(x.shape[0], -1)
    n = x.shape[0]
    if n < 3 * MIN_RUN or not np.isfinite(x).all():
        return None
    t = np.arange(n, dtype=np.float64)
    B = np.column_stack([t, np.ones(n)])
    det = x - B @ np.linalg.lstsq(B, x, rcond=None)[0]
    power = (np.abs(np.fft.rfft(det * np.hanning(n)[:, None], axis=0)) ** 2).sum(axis=1)[1:]
    if len(power) < 4 or not np.isfinite(power).all() or power.max() <= 0:
        return None
    k = int(np.argmax(power))
    if power[k] < RHYTHM_PEAKINESS * (float(np.median(power)) + 1e-300):
        return None
    return float(np.fft.rfftfreq(n, 1.0 / float(fps))[1:][k])


def residual_analysis(x: np.ndarray, fps: float, cutoffs: np.ndarray | None = None) -> tuple[float, dict]:
    """Winter's residual analysis on a (T, D) signal without gaps: returns the
    chosen cutoff (Hz) and {cutoffs, residuals, noise_rms, reason, still,
    unreliable, rhythm_hz, kept}. Falls back to fps / 10 when the signal is
    too short or flat. `still` = nothing moves above the jitter (keep it out
    of a shared choice); `unreliable` = the movement's main rhythm is too fast
    to tell apart from jitter, so the pick was replaced by the cutoff that
    keeps 95 % of that rhythm."""
    x = np.asarray(x, np.float64).reshape(x.shape[0], -1)
    nyq = 0.5 * float(fps)
    if cutoffs is None:
        cutoffs = np.geomspace(max(0.5, nyq / 200.0), 0.9 * nyq, 40)
    info = {"cutoffs": cutoffs, "still": False, "unreliable": False, "rhythm_hz": None, "kept": 1.0}
    if x.shape[0] < 3 * MIN_RUN or not np.isfinite(x).all() or np.ptp(x, axis=0).max() <= 0:
        info.update(residuals=np.full(len(cutoffs), np.nan), noise_rms=float("nan"),
                    reason="signal too short: default fps / 10")
        return float(min(fps / 10.0, 0.9 * nyq)), info

    def resid(fc):
        return float(np.sqrt(np.mean((x - lowpass(x, fps, fc)) ** 2)))

    res = np.array([resid(fc) for fc in cutoffs])
    # (I4) The noise-only line is fitted over a LINEARLY spaced band at the top
    # of the spectrum. The old fit took the top 40 % of the geometric grid,
    # which reaches down to 0.12-0.15 x Nyquist (9 Hz at 120 fps) where
    # wingbeats and undulation live: their energy bent the line, the "noise"
    # grew to the movement's own size and the pick removed the movement (a
    # 16 Hz wingbeat at 120 fps kept 18 %; 20 Hz was cut at 0.5 Hz).
    tail = np.linspace(TAIL_BAND[0] * nyq, TAIL_BAND[1] * nyq, 12)
    A = np.column_stack([tail, np.ones(len(tail))])
    slope, intercept = np.linalg.lstsq(A, np.array([resid(fc) for fc in tail]), rcond=None)[0]
    noise = float(max(intercept, 0.0))
    ok = np.nonzero(res <= noise + 1e-12)[0]
    rhythm = dominant_rhythm(x, fps)
    info.update(residuals=res, noise_rms=noise, rhythm_hz=rhythm)
    if not len(ok):
        fc = float(cutoffs[-1])
        reason = "residual analysis found no noise floor (the signal is very clean): almost no smoothing"
    else:
        fc = float(cutoffs[ok[0]])
        kept = kept_fraction(rhythm, fc, fps) if rhythm is not None else 1.0
        if rhythm is not None and kept < KEEP_MIN:
            # the pick would remove the movement's own rhythm: that rhythm reaches the
            # top of the spectrum, so the "noise" line measured movement, not jitter
            fc = cutoff_keeping(rhythm, fps)
            kept = kept_fraction(rhythm, fc, fps)
            info["unreliable"] = True
            reason = (f"the movement's main rhythm ({rhythm:.1f} Hz) is too fast for {fps:.0f} fps to tell apart "
                      f"from tracking jitter, so residual analysis cannot be trusted here; filtered lightly at "
                      f"{fc:.1f} Hz to keep {100 * kept:.0f} % of it - choose a cutoff by hand, and film faster "
                      "if you can")
        elif ok[0] == 0:
            # even the lowest cutoff removes no more than the jitter: nothing moves above the noise
            info["still"] = True
            reason = (f"this stretch barely moves: nothing faster than {fc:.1f} Hz rises above the tracking "
                      f"jitter (about {noise:.3g})")
        else:
            reason = (f"residual analysis: filtering at {fc:.1f} Hz removes about as much as the estimated noise "
                      f"({noise:.4g}); a lower cutoff would start removing real movement")
            if kept < KEEP_WARN:
                reason += (f"; careful: this keeps only {100 * kept:.0f} % of the movement's main rhythm "
                           f"({rhythm:.1f} Hz)")
        info["kept"] = kept
    info["reason"] = reason
    return fc, info


def smooth_xyz(xyz: np.ndarray, fps: float, cutoff: float | str | None = "auto",
               picks_out: dict | None = None) -> tuple[np.ndarray, float | None, str]:
    """(T, N, 3) positions -> smoothed copy (gaps preserved, each contiguous run
    filtered on its own). cutoff: "auto" (residual analysis, one cutoff for
    the whole set = the median of the per-landmark choices), a number in Hz,
    or None / 0 for no smoothing. Returns (smoothed, cutoff used, reason).
    `picks_out` (auto only) receives {landmark index: [(cutoff, info), ...]}
    per analysed stretch, for the report's per-landmark warnings."""
    xyz = np.asarray(xyz, np.float64)
    T, N = xyz.shape[:2]
    out = xyz.copy()
    if cutoff is None or (not isinstance(cutoff, str) and float(cutoff) <= 0):
        return out, None, "no smoothing: velocities and accelerations are raw differences (noisy)"
    ok = np.isfinite(xyz).all(axis=2)
    fc_used: float | None = None
    reason = ""
    if isinstance(cutoff, str):
        picks, still, n_unsure = [], [], 0
        for j in range(N):
            for a, b in contiguous_runs(ok[:, j]):
                if b - a + 1 >= 3 * MIN_RUN:
                    fc, info = residual_analysis(xyz[a:b + 1, j], fps)
                    if picks_out is not None:
                        picks_out.setdefault(j, []).append((fc, info))
                    # (I4) a stretch that does not move (a reference marker) asks for the
                    # lowest cutoff of the grid; two of them used to drag the shared median
                    # to 1.5 Hz and halve a 1.5 Hz movement. They are left out of the choice.
                    (still if info.get("still") else picks).append(fc)
                    n_unsure += bool(info.get("unreliable"))
        use = picks or still
        if use:
            fc_used = float(np.median(use))
            reason = (f"automatic: {fc_used:.1f} Hz, chosen by residual analysis (the median over "
                      f"{len(use)} landmark stretch(es), which asked for {min(use):.1f}-{max(use):.1f} Hz)")
            if picks and still:
                reason += (f"; {len(still)} stretch(es) that barely move (nothing above the tracking jitter) "
                           "were left out of that choice")
            elif still:
                reason += "; nothing here moves more than the tracking jitter, so it is smoothed heavily"
            if len(picks) > 1 and max(picks) > OWN_PICK_WARN * fc_used:
                reason += ("; the landmarks that asked for much more lose some of their fastest movement at "
                           "this cutoff - the report names them")
            if n_unsure:
                reason += (f"; on {n_unsure} stretch(es) the movement was too fast to tell apart from jitter, "
                           "so the automatic choice is uncertain there - see the report")
        else:
            fc_used = float(min(fps / 10.0, 0.45 * fps))
            reason = f"automatic fell back to {fc_used:.1f} Hz (fps / 10): no stretch long enough for residual analysis"
    else:
        fc_used = float(cutoff)
        if fc_used >= 0.98 * 0.5 * float(fps):
            # lowpass() returns such a signal unfiltered: say so instead of naming a cutoff
            return out, None, (f"no smoothing: the cutoff typed ({fc_used:.1f} Hz) is at the limit of what "
                               f"{float(fps):.0f} fps can hold, so velocities and accelerations are raw "
                               "differences (noisy)")
        reason = f"cutoff chosen by hand: {fc_used:.1f} Hz"
    for j in range(N):
        for a, b in contiguous_runs(ok[:, j]):
            if b - a + 1 >= MIN_RUN:
                out[a:b + 1, j] = lowpass(xyz[a:b + 1, j], fps, fc_used)
    return out, fc_used, reason


def derivatives(xyz: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration (T, N, 3) by central differences inside each
    contiguous run (second-order one-sided at its ends), NaN elsewhere."""
    xyz = np.asarray(xyz, np.float64)
    T, N = xyz.shape[:2]
    vel = np.full_like(xyz, np.nan)
    acc = np.full_like(xyz, np.nan)
    dt = 1.0 / float(fps)
    ok = np.isfinite(xyz).all(axis=2)
    for j in range(N):
        for a, b in contiguous_runs(ok[:, j]):
            if b - a + 1 >= 3:
                # edge_order=2 (I5): first-order ends read exactly a/2 and 3a/4 for a
                # parabola, so an unsmoothed free fall started and ended at 4.9 m/s^2
                v = np.gradient(xyz[a:b + 1, j], dt, axis=0, edge_order=2)
                vel[a:b + 1, j] = v
                acc[a:b + 1, j] = np.gradient(v, dt, axis=0, edge_order=2)
            elif b - a + 1 == 2:
                v = (xyz[b, j] - xyz[a, j]) / dt
                vel[a:b + 1, j] = v
    return vel, acc


def edge_width(fps: float, cutoff) -> int:
    """Frames at each end of a stretch that the report's peaks leave out: one
    cutoff period (the filter and the differences only see one side there);
    one frame without smoothing."""
    if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0:
        return 1
    return int(max(1, np.ceil(float(fps) / float(cutoff))))


def edge_mask(ok: np.ndarray, width: int) -> np.ndarray:
    """(T, N) bool: True on the first and last `width` frames of every
    contiguous run of `ok` (T, N)."""
    ok = np.asarray(ok, bool)
    out = np.zeros_like(ok)
    for j in range(ok.shape[1]):
        for a, b in contiguous_runs(ok[:, j]):
            out[a:min(b + 1, a + width), j] = True
            out[max(a, b + 1 - width):b + 1, j] = True
    return out


def kinematics_report(names, xyz_s, vel, acc, fps: float, unit: str, cutoff, reason: str, t0: int,
                      notes: dict | None = None, n_unsmoothed: int = 0) -> str:
    """The plain-language report. Peaks come from frames at least one cutoff
    period inside a stretch (I5: the ends of a stretch are the least certain,
    and the old peak was usually an end frame); `notes` {landmark index:
    sentence} adds a per-landmark warning; `n_unsmoothed` = frames in
    stretches too short to smooth (their derivatives are blank, I6)."""
    speed = np.linalg.norm(vel, axis=2)
    amag = np.linalg.norm(acc, axis=2)
    ok = np.isfinite(xyz_s).all(axis=2)
    u = unit or "units"
    e = edge_width(fps, cutoff)
    inner = ~edge_mask(ok, e)
    lines = [
        "Kinetrace 3D kinematics",
        "",
        f"Positions are in {u}; time runs at {fps:.3f} frames per second of the reference camera "
        f"(frame numbers of the reference camera, starting at {t0}).",
        f"Smoothing: {reason}.",
    ]
    if cutoff is not None:
        lines.append(f"What the cutoff means: movement slower than half of {cutoff:.1f} Hz keeps 94 % or more of its "
                     f"size; movement AT {cutoff:.1f} Hz keeps half; faster movement is removed as jitter.")
    if n_unsmoothed:
        lines.append(f"{n_unsmoothed} frame(s) lie in stretches shorter than {MIN_RUN} frames, too short to smooth: "
                     "their positions are exported as measured and their velocity and acceleration are left blank.")
    lines += [
        "Velocity = change of the smoothed position per second (central differences); acceleration = change of "
        "velocity per second. Neither is computed across a gap in the track.",
        "",
        f"Per landmark (frames with 3D, peak speed, peak acceleration). The peaks leave out the first and last "
        f"{e} frame(s) of every stretch, where the filter and the differences only see one side:",
    ]
    for j, nm in enumerate(names):
        n = int(ok[:, j].sum())
        if n == 0:
            lines.append(f"  {nm}: no 3D")
            continue
        sp = np.where(inner[:, j], speed[:, j], np.nan)
        am = np.where(inner[:, j], amag[:, j], np.nan)
        if not np.isfinite(sp).any():
            lines.append(f"  {nm}: {n} frames, but no stretch is longer than {2 * e} frames, so no trustworthy peak "
                         "(the values are in the CSV)")
        elif np.isfinite(am).any():
            lines.append(f"  {nm}: {n} frames, speed up to {np.nanmax(sp):.3f} {u}/s (median "
                         f"{np.nanmedian(sp):.3f}), acceleration up to {np.nanmax(am):.2f} {u}/s^2")
        else:
            lines.append(f"  {nm}: {n} frames, speed up to {np.nanmax(sp):.3f} {u}/s")
        if notes and notes.get(j):
            lines.append(f"    ! {notes[j]}")
    lines += ["", "How to check it by hand:"]
    if u in ("m", "metre", "metres", "meter", "meters"):
        lines.append(f"  - Anything in free fall (a dropped ball, an animal between take-off and landing, ignoring "
                     f"air) accelerates downwards at {G:.2f} m/s^2. If a dropped object reads far from that, the "
                     "wand length, the frame rate or the smoothing is wrong.")
    elif u == "wand":
        lines.append("  - The unit is WAND LENGTHS: every speed and acceleration scales with the real wand length. "
                     "Calibrate again with the measured length before comparing to published values.")
    else:
        lines.append(f"  - Convert to metres to compare with free fall ({G:.2f} m/s^2) or published speeds.")
    lines.append("  - A landmark that stands still should read near zero speed; if it reads tens of units per second, "
                 "the tracking jitters and a lower cutoff (or better tracking) is needed.")
    # np.gradient twice = (x[i+2] - 2 x[i] + x[i-2]) / (4 dt^2): white jitter s gives s * sqrt(6) / 4 * fps^2 RMS
    lines.append(f"  - Accelerations are sensitive to the cutoff: at {fps:.0f} fps, 0.5 mm of jitter unsmoothed "
                 f"becomes about {0.0005 * fps * fps * np.sqrt(6.0) / 4.0:.0f} m/s^2 (RMS) of fake acceleration, "
                 "with peaks three times that.")
    return "\n".join(lines) + "\n"


def export_kinematics(path: str | Path, rec, fps: float, unit: str, cutoff: float | str | None = "auto"
                      ) -> list[str]:
    """CSV (frame, time_s, per landmark X Y Z Vx Vy Vz speed Ax Ay Az acc) +
    a `_kinematics_report.txt` beside it. Returns the files written."""
    p = Path(path)
    raw = np.asarray(rec.xyz, np.float64)
    picks: dict = {}
    xyz_s, fc, reason = smooth_xyz(raw, fps, cutoff, picks_out=picks)
    vel, acc = derivatives(xyz_s, fps)
    ok = np.isfinite(raw).all(axis=2)
    n_unsmoothed = 0
    notes: dict[int, str] = {}
    if fc is not None:
        for j in range(ok.shape[1]):
            longest = None
            for a, b in contiguous_runs(ok[:, j]):
                if b - a + 1 < MIN_RUN:
                    # (I6) too short to filter: raw second differences of jitter are tens
                    # of m/s^2 at high frame rates and set the landmark's headline peak
                    # in a file that says it is smoothed. Positions stay as measured.
                    vel[a:b + 1, j] = np.nan
                    acc[a:b + 1, j] = np.nan
                    n_unsmoothed += b - a + 1
                if longest is None or b - a > longest[1] - longest[0]:
                    longest = (a, b)
            # (I4) the shared cutoff may be too low for THIS landmark: its own residual
            # analysis asked for much more (a wing tip beside the body), or its main
            # rhythm is cut (also for a cutoff chosen by hand)
            own = [f for f, inf in picks.get(j, []) if not inf.get("still")]
            f_d = dominant_rhythm(raw[longest[0]:longest[1] + 1, j], fps) if longest is not None else None
            if own and max(own) > OWN_PICK_WARN * fc:
                notes[j] = (f"on its own it asks for about {max(own):.0f} Hz (residual analysis of this landmark "
                            f"alone): at the shared {fc:.1f} Hz its fastest movement is weakened and its speeds and "
                            f"accelerations read too low. Export again with {max(own):.0f} Hz chosen by hand to "
                            "study it.")
            elif f_d is not None and kept_fraction(f_d, fc, fps) < KEEP_WARN:
                notes[j] = (f"its main rhythm ({f_d:.1f} Hz) keeps only {100 * kept_fraction(f_d, fc, fps):.0f} % "
                            f"of its size at {fc:.1f} Hz: its speeds and accelerations read too low. Export again "
                            f"with a cutoff chosen by hand of at least {cutoff_keeping(f_d, fps):.0f} Hz.")
    speed = np.linalg.norm(vel, axis=2)
    amag = np.linalg.norm(acc, axis=2)
    # the CSV sanitizer (as every other export, cf. I95): a comma in a landmark name
    # would split its column name and shift every column after it
    from cotracker_app.session import _sanitize
    safe = [_sanitize(str(nm)) for nm in rec.names]
    cols = ["frame", "time_s"]
    for nm in safe:
        cols += [f"{nm}_{k}" for k in ("X", "Y", "Z", "Vx", "Vy", "Vz", "speed", "Ax", "Ay", "Az", "acc")]
    head = "# unit " + (unit or "?") + f"; fps {fps:.6g}; smoothing: {reason}"
    if n_unsmoothed:
        head += (f"; {n_unsmoothed} frame(s) in stretches shorter than {MIN_RUN} frames could not be smoothed: "
                 "positions as measured, velocity and acceleration left blank")
    if notes:
        head += "; moving faster than this cutoff keeps: " + ", ".join(safe[j] for j in sorted(notes)) + " (see the report)"
    lines = [head, ",".join(cols)]

    def cell(v):
        return f"{v:.6f}" if np.isfinite(v) else "NaN"

    for i in range(rec.n_frames):
        f = rec.t0 + i
        row = [str(f), f"{f / fps:.6f}"]
        for j in range(len(rec.names)):
            row += [cell(v) for v in xyz_s[i, j]] + [cell(v) for v in vel[i, j]] + [cell(speed[i, j])]
            row += [cell(v) for v in acc[i, j]] + [cell(amag[i, j])]
        lines.append(",".join(row))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    rp = p.with_name(p.stem + "_report.txt")
    rp.write_text(kinematics_report(rec.names, xyz_s, vel, acc, fps, unit, fc, reason, rec.t0,
                                    notes=notes, n_unsmoothed=n_unsmoothed), encoding="utf-8")
    return [str(p), str(rp)]
