"""GoPro metadata (GPMF): the flag that identifies GoPro footage, and what the GoPro workflow reads
from it (G145, G146; owner 2026-10-03: "for GoPro only, since we don't know what data the other
cameras carry"). No Qt, no torch; numpy + OpenCV only for the lens.

A GoPro MP4 carries two kinds of GPMF:
  * the HEADER (moov/udta/GPMF): the camera model, firmware, recording settings (resolution,
    frame rate, lens mode, HyperSmooth / EIS, shutter, ISO limits, mounting orientation) and --
    for the mode it recorded in -- GoPro's own LENS MODEL: the incidence angle as a polynomial
    of the distance from the picture centre, theta = POLY(ZMPL * rho / half-diagonal), radially
    symmetric about the picture centre (measured on HERO12 Black + Mission 1 Pro: the corner
    radius solves POLY(r) = ZFOV / 2 at r = ZMPL, and the Mission 1's separate XFOV / YFOV agree);
  * the TIMED track ('gpmd'): accelerometer, gyroscope, the gravity direction in the camera's
    frame (GRAV), camera orientation (CORI: gyro-integrated, it drifts 3-60 deg over two minutes
    on a camera that never moved -- not used), shutter / ISO per frame, dropped-frame counters
    (MSKP), GPS on models that have it.
Plus a timecode track ('tmcd') with the start time on the camera clock (frames since midnight).

Only the bytes needed are read (the moov box, then the 'gpmd' samples), never the video: a few
hundred kB of a 2 GB file, so a file on a network drive reads in about a second.

`read(path)` -> GoProInfo, or None for footage that is not a GoPro's (no GoPro header and no
'gpmd' track). A re-exported copy usually loses the header (no lens model, no settings) and may
keep the sensors: `GoProInfo.has_lens` says which.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

VFOV_NAMES = {"W": "Wide", "L": "Linear", "N": "Narrow", "S": "SuperView", "H": "HyperView",
              "X": "Max SuperView", "M": "Max", "Z": "Linear + Horizon Lock"}
OREN_NAMES = {"U": "upright", "D": "upside down", "L": "on its side (left)", "R": "on its side (right)"}
TILT_DEV_DEG = 0.5      # the camera's tilt differs from its usual tilt by more than this = it moved (GRAV is steady to ~0.1-0.3 deg)
JOLT_MS2 = 3.0          # an accelerometer reading this far from gravity = a knock
SHUTTER_RATIO = 1.5     # shutters this many times apart = visibly different motion blur
FMT = {"b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I", "f": "f", "d": "d", "j": "q", "J": "Q",
       "q": "i", "Q": "q"}


class GoProReadError(Exception):
    """The file looked like GoPro footage but its metadata could not be read (said in words)."""


# --------------------------------------------------------------------------- MP4 boxes
class _MP4:
    def __init__(self, path: str):
        self.path = str(path)
        self.f = open(self.path, "rb")
        self.size = os.path.getsize(self.path)

    def close(self):
        self.f.close()

    def boxes(self, off: int, end: int):
        f = self.f
        while off + 8 <= end:
            f.seek(off)
            head = f.read(8)
            if len(head) < 8:
                return
            size, typ = struct.unpack(">I4s", head)
            hdr = 8
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                hdr = 16
            elif size == 0:
                size = end - off
            if size < hdr:
                return
            yield typ.decode("latin1"), off + hdr, off + size
            off += size

    def child(self, span, name: str):
        if span is None:
            return None
        for t, a, b in self.boxes(*span):
            if t == name:
                return a, b
        return None

    def read(self, off: int, n: int) -> bytes:
        self.f.seek(off)
        return self.f.read(n)

    def full(self, span) -> bytes:
        """A full box's payload past version / flags."""
        return self.read(span[0], span[1] - span[0])[4:]


def _tracks(m: _MP4):
    """(sample-entry fourcc, stbl span, media timescale) per track."""
    moov = m.child((0, m.size), "moov")
    out = []
    if moov is None:
        return out, None
    udta = None
    for t, a, b in m.boxes(*moov):
        if t == "udta":
            udta = m.child((a, b), "GPMF")
        if t != "trak":
            continue
        mdia = m.child((a, b), "mdia")
        mdhd = m.child(mdia, "mdhd")
        stbl = m.child(m.child(mdia, "minf"), "stbl")
        if mdhd is None or stbl is None:
            continue
        v = m.read(mdhd[0], 1)[0]
        d = m.full(mdhd)
        ts = struct.unpack(">I", d[8:12] if v == 0 else d[16:20])[0]
        sd = m.full(m.child(stbl, "stsd"))
        out.append((sd[8:12].decode("latin1"), stbl, ts))
    return out, udta


def _samples(m: _MP4, stbl, ts: int):
    """[(start s, duration s, bytes)] of one track, from its sample table."""
    stsz = m.full(m.child(stbl, "stsz"))
    fixed, n = struct.unpack(">II", stsz[:8])
    sizes = [fixed] * n if fixed else list(struct.unpack(f">{n}I", stsz[8:8 + 4 * n]))
    co = m.child(stbl, "stco")
    if co is not None:
        d = m.full(co)
        k = struct.unpack(">I", d[:4])[0]
        chunks = list(struct.unpack(f">{k}I", d[4:4 + 4 * k]))
    else:
        d = m.full(m.child(stbl, "co64"))
        k = struct.unpack(">I", d[:4])[0]
        chunks = list(struct.unpack(f">{k}Q", d[4:4 + 8 * k]))
    d = m.full(m.child(stbl, "stsc"))
    k = struct.unpack(">I", d[:4])[0]
    stsc = [struct.unpack(">III", d[4 + 12 * i:16 + 12 * i]) for i in range(k)]
    d = m.full(m.child(stbl, "stts"))
    k = struct.unpack(">I", d[:4])[0]
    durs = [dd for c, dd in (struct.unpack(">II", d[4 + 8 * i:12 + 8 * i]) for i in range(k)) for _ in range(c)]
    offs, si = [], 0
    for ci, coff in enumerate(chunks):
        per = [e for e in stsc if e[0] <= ci + 1][-1][1]
        o = coff
        for _ in range(per):
            if si >= n:
                break
            offs.append(o)
            o += sizes[si]
            si += 1
    t0 = np.concatenate([[0], np.cumsum(durs)])[:n] / float(ts)
    return [(float(t0[i]), durs[i] / float(ts), m.read(offs[i], sizes[i])) for i in range(len(offs))]


# --------------------------------------------------------------------------- GPMF KLV
def _klv(buf: bytes):
    i = 0
    while i + 8 <= len(buf):
        key = buf[i:i + 4].decode("latin1")
        typ = chr(buf[i + 4])
        size, rep = buf[i + 5], struct.unpack(">H", buf[i + 6:i + 8])[0]
        n = size * rep
        yield key, typ, size, rep, buf[i + 8:i + 8 + n]
        i += 8 + ((n + 3) & ~3)


def _values(typ: str, size: int, rep: int, body: bytes):
    if typ == "c":
        if size == 1:
            return body.decode("latin1", "replace").rstrip("\x00").strip()
        return [body[j * size:(j + 1) * size].decode("latin1", "replace").rstrip("\x00").strip() for j in range(rep)]
    if typ == "F":
        return [body[j:j + 4].decode("latin1") for j in range(0, len(body), 4)]
    if typ in FMT:
        fm = FMT[typ]
        k = struct.calcsize(fm)
        per = max(1, size // k)
        a = np.frombuffer(body[:per * rep * k], dtype=">" + fm).astype(np.float64).reshape(rep, per)
        if typ == "q":
            a = a / 65536.0
        elif typ == "Q":
            a = a / 4294967296.0
        return a
    return None


def _header(buf: bytes, out: dict, prefix: str = "") -> None:
    """The header's settings, flattened: the FIRST value of each key (the lens block and the
    global settings each name their own keys; DVID / DVNM repeat per block)."""
    for key, typ, size, rep, body in _klv(buf):
        if typ == "\x00":
            _header(body, out, prefix)
            continue
        v = _values(typ, size, rep, body)
        if isinstance(v, np.ndarray):
            v = v.ravel().tolist()
            v = v[0] if len(v) == 1 else v
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
            v = v[0]
        out.setdefault(key, v)


SENSOR_KEYS = frozenset({"SHUT", "ISOE", "MSKP", "GRAV", "ACCL"})   # the streams `_sensor_summary` reads


def _streams(samples) -> dict:
    """{data key: (times (n,), values (n, k))} of the numeric timed streams, SCAL applied. Only the
    streams in SENSOR_KEYS are decoded; any other stream found is listed with None (a gyroscope,
    orientation, ... nothing reads)."""
    acc: dict = {}
    meta = {"STMP", "TSMP", "STNM", "SIUN", "UNIT", "SCAL", "ORIN", "ORIO", "MTRX", "TYPE", "TMPC", "EMPT",
            "DVID", "DVNM", "TICK", "TOCK", "RMRK", "QUAN", "VERS", "GPSF", "GPSP", "GPSU", "GPSA"}
    for t, dur, payload in samples:
        for k, typ, size, rep, body in _klv(payload):
            if k != "DEVC":
                continue
            for k2, _t2, _s2, _r2, b2 in _klv(body):
                if k2 != "STRM":
                    continue
                scal, data = None, None
                for k3, typ3, s3, r3, b3 in _klv(b2):
                    if k3 == "SCAL":
                        scal = (typ3, s3, r3, b3)
                    elif k3 not in meta and typ3 in FMT:
                        data = (k3, typ3, s3, r3, b3)       # the stream's LAST numeric key is its data
                if data is None:
                    continue
                if data[0] not in SENSOR_KEYS:
                    if data[3]:
                        acc.setdefault(data[0], None)
                    continue
                key, v = data[0], _values(*data[1:])
                if v is None or not len(v):
                    continue
                scal = _values(*scal) if scal is not None else None
                if scal is not None:
                    sc = np.ravel(np.asarray(scal, np.float64))
                    v = v / (sc if sc.size == v.shape[1] else sc[0])
                tt = t + dur * np.arange(len(v)) / len(v)
                acc.setdefault(key, []).append((tt, v))
    return {k: None if parts is None else (np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
            for k, parts in acc.items()}


def _timecode(m: _MP4, tracks) -> tuple[int, float] | None:
    """(start frame since midnight on the camera clock, its frame rate) from the 'tmcd' track."""
    for kind, stbl, _ts in tracks:
        if kind != "tmcd":
            continue
        sd = m.full(m.child(stbl, "stsd"))
        # sample entry: size, 'tmcd', 6 reserved, data-ref index (2), reserved (4), flags (4),
        # timescale (4), frame duration (4), frames per second (1)
        e = sd[4:]
        tscale = int.from_bytes(e[24:28], "big")
        fdur = int.from_bytes(e[28:32], "big")
        smp = _samples(m, stbl, max(1, tscale))
        if not smp or len(smp[0][2]) < 4 or not fdur:
            return None
        return int.from_bytes(smp[0][2][:4], "big"), tscale / float(fdur)
    return None


# --------------------------------------------------------------------------- the result
@dataclass
class GoProInfo:
    """What one GoPro video says about itself. `has_lens` = GoPro's lens model is in the file."""
    path: str
    model: str = ""                  # "HERO12 Black", "MISSION 1 PRO", ...
    firmware: str = ""
    width: int = 0
    height: int = 0
    fps: float = float("nan")
    lens_mode: str = ""              # "Wide", "Linear", ...
    stabilised: bool = False         # HyperSmooth or EIS on: the camera re-warps every frame
    stabilisation: str = ""          # in words
    orientation: str = ""            # OREN: U / D / L / R
    zfov_deg: float = float("nan")   # diagonal field of view of the lens model
    poly: list = field(default_factory=list)
    zmpl: float = float("nan")
    projection: str = ""
    shutter_s: float = float("nan")  # median exposure time
    iso: float = float("nan")        # median sensor ISO
    dropped_frames: int = 0          # MSKP: main-video frames the camera skipped
    timecode: tuple | None = None    # (start frame since midnight, rate) on the camera clock
    duration_s: float = 0.0
    gravity: np.ndarray | None = None        # unit gravity direction in the sensor frame (median)
    moves: list = field(default_factory=list)  # [{"start_s", "end_s", "frame", "end_frame", "tilt_deg", "jolt"}]
    jolts: list = field(default_factory=list)  # [{"time_s", "frame", "ms2"}]
    header_found: bool = False
    sensors_found: bool = False

    @property
    def has_lens(self) -> bool:
        return len(self.poly) >= 2 and np.isfinite(self.zmpl) and self.zmpl > 0 and self.width > 0

    @property
    def label(self) -> str:
        bits = [self.model or "GoPro"]
        if self.lens_mode:
            bits.append(self.lens_mode)
        if self.width:
            bits.append(f"{self.width}x{self.height}")
        if np.isfinite(self.fps):
            bits.append(f"{self.fps:.2f} fps")
        return ", ".join(bits)

    def frame_of(self, t: float) -> int:
        return int(round(t * self.fps)) if np.isfinite(self.fps) and self.fps > 0 else 0

    def tilt(self) -> tuple[float, float] | None:
        """(degrees the lens points below the horizon -- negative = above, roll in degrees) from
        the gravity sensor, taking the mounting (OREN) into account: an upside-down camera flips its
        picture, one on its side records a landscape picture turned 90 degrees from the sensor."""
        g = self.gravity
        if g is None:
            return None
        gx, gy, gz = (float(v) for v in g)
        o = self.orientation
        if o == "D":
            down, side = -gy, -gx
        elif o == "L":
            down, side = -gx, gy
        elif o == "R":
            down, side = gx, -gy
        else:
            down, side = gy, gx
        return float(np.degrees(np.arctan2(gz, down))), float(np.degrees(np.arcsin(np.clip(side, -1, 1))))


def is_gopro(path) -> bool:
    """THE flag (G145): this video is GoPro footage -- GoPro's header box or its 'gpmd' sensor
    track is in the file. Reads only the moov box; False for anything unreadable."""
    try:
        m = _MP4(str(path))
    except OSError:
        return False
    try:
        tracks, udta = _tracks(m)
        if any(k == "gpmd" for k, _s, _t in tracks):
            return True
        if udta is not None:
            hdr: dict = {}
            _header(m.read(udta[0], udta[1] - udta[0]), hdr)
            return bool(hdr.get("MINF") or hdr.get("FMWR"))
        return False
    except Exception:       # noqa: BLE001 - a damaged or foreign file is simply not GoPro footage here
        return False
    finally:
        m.close()


def read(path, sensors: bool = True) -> GoProInfo | None:
    """GoProInfo for a GoPro video, None for any other. `sensors=False` skips the timed track
    (the header and the timecode only: what the lens needs)."""
    try:
        m = _MP4(str(path))
    except OSError:
        return None
    try:
        tracks, udta = _tracks(m)
        gpmd = next(((s, t) for k, s, t in tracks if k == "gpmd"), None)
        hdr: dict = {}
        if udta is not None:
            _header(m.read(udta[0], udta[1] - udta[0]), hdr)
        if gpmd is None and not (hdr.get("MINF") or hdr.get("FMWR")):
            return None
        info = GoProInfo(str(path), header_found=bool(hdr))
        info.model = str(hdr.get("MINF") or "")
        info.firmware = str(hdr.get("FMWR") or "")
        vres = hdr.get("VRES")
        if isinstance(vres, list) and len(vres) == 2:
            info.width, info.height = int(vres[0]), int(vres[1])
        vfps = hdr.get("VFPS")
        if isinstance(vfps, list) and len(vfps) == 2 and vfps[1]:
            info.fps = float(vfps[0]) / float(vfps[1])
        info.lens_mode = VFOV_NAMES.get(str(hdr.get("VFOV") or ""), str(hdr.get("VFOV") or ""))
        eis = str(hdr.get("EISE") or "").upper() == "Y"
        hs = str(hdr.get("HSGT") or "").upper()
        info.stabilised = eis or (hs not in ("", "OFF", "N", "NONE"))
        info.stabilisation = ("HyperSmooth " + hs.title() if hs not in ("", "OFF", "N", "NONE")
                              else "EIS on" if eis else "off")
        info.orientation = str(hdr.get("OREN") or "")
        poly = hdr.get("POLY")
        if isinstance(poly, list) and len(poly) >= 2:
            info.poly = [float(v) for v in poly]
        for key, attr in (("ZMPL", "zmpl"), ("ZFOV", "zfov_deg")):
            v = hdr.get(key)
            if isinstance(v, (int, float)):
                setattr(info, attr, float(v))
        prj = hdr.get("PRJT")
        info.projection = prj if isinstance(prj, str) else (prj[0] if isinstance(prj, list) and prj else "")
        info.timecode = _timecode(m, tracks)
        if sensors and gpmd is not None:
            smp = _samples(m, *gpmd)
            if smp:
                info.duration_s = smp[-1][0] + smp[-1][1]
                _sensor_summary(info, _streams(smp))
        return info
    except Exception as e:      # noqa: BLE001 - a GoPro file whose metadata is damaged: said, not a crash
        raise GoProReadError(f"the GoPro data in {os.path.basename(str(path))} could not be read ({e})") from e
    finally:
        m.close()


def read_safe(path, sensors: bool = True) -> GoProInfo | None:
    """`read` that never raises: None when the metadata cannot be read (the video opens as usual)."""
    try:
        return read(path, sensors)
    except Exception:       # noqa: BLE001
        return None


def _sensor_summary(info: GoProInfo, st: dict) -> None:
    info.sensors_found = bool(st)
    if "SHUT" in st:
        info.shutter_s = float(np.median(st["SHUT"][1][:, 0]))
    if "ISOE" in st:
        info.iso = float(np.median(st["ISOE"][1][:, 0]))
    if "MSKP" in st:
        info.dropped_frames = int(np.nansum(st["MSKP"][1][:, 0]))
    if "GRAV" in st:
        t, g = st["GRAV"]
        g = g[:, :3] / np.maximum(np.linalg.norm(g[:, :3], axis=1, keepdims=True), 1e-9)
        med = np.median(g, axis=0)
        med /= max(np.linalg.norm(med), 1e-9)
        info.gravity = med
        info.moves = _moves(t, g, med, info)
    if "ACCL" in st:
        t, a = st["ACCL"]
        mag = np.linalg.norm(a[:, :3], axis=1)
        dev = np.abs(mag - np.median(mag))
        hits = np.nonzero(dev > JOLT_MS2)[0]
        groups: list[list[int]] = []
        for i in hits.tolist():
            if groups and t[i] - t[groups[-1][-1]] < 1.0:
                groups[-1].append(i)
            else:
                groups.append([i])
        info.jolts = [{"time_s": float(t[g[int(np.argmax(dev[g]))]]), "ms2": float(dev[g].max()),
                       "frame": info.frame_of(float(t[g[int(np.argmax(dev[g]))]]))} for g in groups]


def _moves(t: np.ndarray, g: np.ndarray, usual: np.ndarray, info: GoProInfo) -> list:
    """Stretches where the camera's tilt differs from its usual tilt (the recording's median
    gravity direction) by more than TILT_DEV_DEG, judged on quarter-second means (the sensor is
    steady to ~0.1-0.3 deg on a camera that does not move)."""
    if len(t) < 4:
        return []
    edges = np.arange(t[0], t[-1] + 0.25, 0.25)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 1)
    # each bucket's samples as one slice (a stable sort keeps their order: the same rows as `idx == k`)
    order = np.argsort(idx, kind="stable")
    gs, ks = g[order], np.arange(len(edges))
    lo, hi = np.searchsorted(idx[order], ks, side="left"), np.searchsorted(idx[order], ks, side="right")
    out, cur = [], None
    for k in range(len(edges)):
        if lo[k] == hi[k]:
            continue
        m = gs[lo[k]:hi[k]].mean(axis=0)
        m /= max(np.linalg.norm(m), 1e-9)
        dev = float(np.degrees(np.arccos(np.clip(m @ usual, -1, 1))))
        if dev > TILT_DEV_DEG:
            if cur is None:
                cur = {"start_s": float(edges[k]), "end_s": float(edges[k] + 0.25), "tilt_deg": dev}
            else:
                cur["end_s"] = float(edges[k] + 0.25)
                cur["tilt_deg"] = max(cur["tilt_deg"], dev)
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    for mv in out:
        mv["frame"] = info.frame_of(mv["start_s"])
        mv["end_frame"] = info.frame_of(mv["end_s"])
        mv["to_end"] = bool(mv["end_s"] >= t[-1] - 0.3)
    return out


# --------------------------------------------------------------------------- the lens
def lens_profile(info: GoProInfo):
    """GoPro's lens model for this recording mode as Kinetrace's fisheye lens (OpenCV's Kannala-
    Brandt form): the polynomial sampled from the centre to the corner and f, k1..k4 fitted to it
    (0.21 px on a HERO12 2.7K Wide, 0.32 px on a Mission 1 Pro 4K Wide); the centre is the picture
    centre (GoPro's model has no other). NOMINAL: the lens design, not this unit -- units differ by
    about 1 % in focal length and ~10 px in centre, which the boards or the wand calibration
    measure. None when the file carries no lens model."""
    from scipy.optimize import least_squares

    from kinetrace import lens as lens_mod
    if not info.has_lens:
        return None
    w, h = info.width, info.height
    hd = 0.5 * float(np.hypot(w, h))
    rho = np.linspace(0.0, hd, 3000)
    th = np.polyval(np.asarray(info.poly, float)[::-1], info.zmpl * rho / hd)
    if not np.all(np.diff(th) > 0):
        return None                       # not a monotone angle curve: not a model we understand

    def res(p):
        t2 = th * th
        return p[0] * th * (1 + p[1] * t2 + p[2] * t2 ** 2 + p[3] * t2 ** 3 + p[4] * t2 ** 4) - rho
    sol = least_squares(res, [hd / th[-1], 0, 0, 0, 0])
    f, k = float(sol.x[0]), np.asarray(sol.x[1:], np.float64)
    fit_px = float(np.max(np.abs(res(sol.x))))
    K = np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
    prof = lens_mod.LensProfile(w, h, K, k, True, float("nan"), 0,
                                f"GoPro's lens model from the video ({info.label}): nominal, not measured on this camera")
    # VRES is the size the camera STORES (3840 x 2160 on a 4K clip filmed on its side, whose rotation
    # tag makes it play 2160 x 3840): the model is in the stored picture, so `lens.fit_profile` turns
    # it for a camera whose video is turned
    prof.rotation = 0
    prof.report = {
        "verdict": "ok", "gopro_nominal": True, "model": "fisheye", "gopro_poly_fit_px": fit_px,
        "focal_px": [f, f], "principal_px": [K[0, 2], K[1, 2]], "dist": k.tolist(), "width": w, "height": h,
        "fov_diag_deg": float(info.zfov_deg),
        "verdict_reasons": [
            f"GoPro's own lens model for this recording mode ({info.label}), read from the video: it covers the "
            "whole picture, corners included.",
            "It is the lens DESIGN, not this camera: real units differ by about 1 % in focal length and up to ~10 px "
            "in where the centre is. The wand calibration refines the focal length; for the centre too, film a "
            "checkerboard and choose 'GoPro lens + your boards'.",
            f"Field of view {info.zfov_deg:.0f} degrees corner to corner (GoPro's figure)."
            if np.isfinite(info.zfov_deg) else "",
        ]}
    prof.report["verdict_reasons"] = [r for r in prof.report["verdict_reasons"] if r]
    return prof


# --------------------------------------------------------------------------- checks
def problems(info: GoProInfo, name: str = "") -> list[str]:
    """What in one GoPro video can spoil tracking or 3D, in words (empty = nothing)."""
    who = name or os.path.basename(info.path)
    out = []
    if info.stabilised:
        out.append(f"{who}: stabilisation was ON ({info.stabilisation}). The camera re-warps every frame, so no "
                   "lens calibration holds and 3D from this camera will be off: record with HyperSmooth / EIS off.")
    if info.dropped_frames:
        out.append(f"{who}: the camera reports {info.dropped_frames} dropped frame(s): frame numbers after them "
                   "are shifted against the other cameras.")
    passing = []
    for mv in info.moves:
        if mv["to_end"] and mv["start_s"] > 0.5:
            out.append(f"{who} moved at frame {mv['frame']} ({mv['start_s']:.1f} s) and stayed {mv['tilt_deg']:.1f} "
                       "degrees off its earlier tilt: a calibration made before then does not hold for it after.")
        else:
            passing.append(mv)
    if passing:
        spans = [f"{m['frame']}–{m['end_frame']}" for m in passing]
        spans = ", ".join(spans[:-1]) + " and " + spans[-1] if len(spans) > 1 else spans[0]
        out.append(f"{who} tilted by up to {max(m['tilt_deg'] for m in passing):.1f} degrees in frames {spans}: "
                   "3D from this camera is off there.")
    return out


def rig_problems(infos: dict) -> list[str]:
    """Settings that differ between the GoPro cameras of one project ({camera name: GoProInfo})."""
    infos = {k: v for k, v in infos.items() if v is not None and v.header_found}
    if len(infos) < 2:
        return []
    out = []

    def spread(attr, words, fmt=str):
        vals: dict = {}
        for k, v in infos.items():
            vals.setdefault(fmt(getattr(v, attr)), []).append(k)
        if len(vals) > 1:
            out.append(f"The cameras recorded with different {words}: "
                       + "; ".join(f"{val} ({', '.join(ks)})" for val, ks in vals.items()) + ".")
    spread("lens_mode", "lens modes")
    spread("fps", "frame rates", lambda x: f"{x:.3f} fps")
    shut = {k: v.shutter_s for k, v in infos.items() if np.isfinite(v.shutter_s) and v.shutter_s > 0}
    if len(shut) > 1 and max(shut.values()) / min(shut.values()) >= SHUTTER_RATIO:
        out.append("The shutter speeds differ (" + ", ".join(f"{k} 1/{1 / s:.0f} s" for k, s in shut.items())
                   + "): a fast animal blurs differently in each camera.")
    return out


def timecode_prior(infos: list, fps: list) -> list | None:
    """Kinetrace offsets from the cameras' timecode tracks (camera 0 = the reference): a camera that
    started later has a negative offset. A SEARCH WINDOW for the sound / motion sync, not a sync --
    on a 10-camera rig the clocks were 0.2-0.7 s apart from what the sound sync found.
    None unless every camera has a timecode."""
    if not infos or any(i is None or i.timecode is None for i in infos):
        return None
    t = [i.timecode[0] / i.timecode[1] for i in infos]        # seconds since midnight
    return [0.0] + [float((t[0] - t[k]) * fps[k]) for k in range(1, len(infos))]
