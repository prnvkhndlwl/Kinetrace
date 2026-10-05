"""Test helper (G145): a playable MP4 that carries GoPro-style metadata -- the header box
(moov/udta/GPMF: model, settings, GoPro's lens polynomial), a 'gpmd' sensor track (gravity,
accelerometer, shutter, ISO, dropped frames) and a 'tmcd' timecode track -- so the GoPro workflow is
tested without anyone's footage. The video itself is written by OpenCV; the GoPro boxes are added to
its moov, their samples in a second mdat placed before the moov (the video's chunk offsets do not move).

The lens numbers are the HERO12 Black 2.7K Wide model as a HERO12 writes it into every file.
"""
from __future__ import annotations

import struct

import cv2
import numpy as np

H12_POLY = [0.0, 1.814842700958252, 0.10586126148700714, -0.6543740034103394, 0.35001200437545776,
            -1.0506910418746004e-13, 4.411807358784166e-14]
H12_ZMPL = 0.6548827886581421
H12_ZFOV = 127.66272735595703


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def _full(typ: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _box(typ, bytes([version]) + flags.to_bytes(3, "big") + payload)


def klv(key: str, typ: str, values, size: int | None = None) -> bytes:
    """One GPMF entry. `values`: bytes (nested, typ '\\0'), a str (typ 'c'), or numbers."""
    if typ == "\0":
        data, size, rep = values, 1, len(values)
    elif typ == "c":
        data = values.encode("latin1")
        size, rep = len(data), 1
    else:
        fmt = {"s": "h", "S": "H", "l": "i", "L": "I", "f": "f", "B": "B", "b": "b"}[typ]
        arr = np.asarray(values, dtype=">" + fmt)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1) if size is None else arr.reshape(1, -1)
        data = arr.tobytes()
        size, rep = arr.shape[1] * arr.itemsize, arr.shape[0]
    pad = (-len(data)) % 4
    return key.encode("latin1") + typ.encode("latin1") + bytes([size]) + struct.pack(">H", rep) + data + b"\0" * pad


def header_gpmf(model="HERO12 Black", w=2704, h=1520, fps=(240000, 1001), lens="W", eis="N", hs="OFF",
                oren="U", poly=None, zmpl=H12_ZMPL, zfov=H12_ZFOV) -> bytes:
    poly = H12_POLY if poly is None else poly
    glob = b"".join([klv("DVID", "L", [1]), klv("DVNM", "c", "Global Settings"), klv("FMWR", "c", "H23.01.02.40.70"),
                     klv("MINF", "c", model), klv("OREN", "c", oren), klv("EISE", "c", eis), klv("HSGT", "c", hs),
                     klv("VRES", "L", [w, h], size=8), klv("VFPS", "L", list(fps), size=8),
                     klv("PRJT", "c", "GPRO")])
    fovl = b"".join([klv("DVID", "c", "FOVL"), klv("DVNM", "c", "Large FOV"), klv("ZFOV", "f", [zfov]),
                     klv("VFOV", "c", lens), klv("POLY", "f", poly, size=4 * len(poly)), klv("ZMPL", "f", [zmpl])])
    return klv("DEVC", "\0", glob) + klv("DEVC", "\0", fovl)


def sensor_payloads(seconds: int, gravity=(0.189, 0.557, 0.809), bump_at: float | None = None,
                    tilt_after_deg: float = 1.3, jolt_ms2: float = 25.0, shutter=1 / 1920, iso=1600,
                    dropped_at: int | None = None) -> list[bytes]:
    """One GPMF payload per second: GRAV (240 Hz), ACCL (200 Hz), SHUT / ISOE (30 Hz), MSKP (240 Hz).
    A knock at `bump_at` s: an accelerometer spike and the gravity direction turned by
    `tilt_after_deg` from then on."""
    g0 = np.asarray(gravity, float)
    g0 /= np.linalg.norm(g0)
    ax = np.cross(g0, [1.0, 0, 0])
    ax /= np.linalg.norm(ax)
    a = np.radians(tilt_after_deg)
    g1 = g0 * np.cos(a) + np.cross(ax, g0) * np.sin(a)          # g0 turned about ax
    rng = np.random.default_rng(0)
    out = []
    for s in range(seconds):
        tg = s + np.arange(240) / 240.0
        ta = s + np.arange(200) / 200.0
        grav = np.array([g1 if (bump_at is not None and t >= bump_at) else g0 for t in tg])
        grav += rng.normal(0, 0.0005, grav.shape)
        acc = np.array([(g1 if (bump_at is not None and t >= bump_at) else g0) * 9.81 for t in ta])
        acc += rng.normal(0, 0.03, acc.shape)
        if bump_at is not None and s <= bump_at < s + 1:
            k = int((bump_at - s) * 200)
            acc[k:k + 3] += np.array([jolt_ms2, 0, 0])
        mskp = np.zeros(240)
        if dropped_at is not None and s == dropped_at:
            mskp[10] = 1
        devc = b"".join([
            klv("DVID", "L", [1]), klv("DVNM", "c", "Camera"),
            klv("STRM", "\0", klv("STNM", "c", "Gravity Vector") + klv("SCAL", "s", [32767])
                + klv("GRAV", "s", np.round(grav * 32767).astype(int), size=6)),
            klv("STRM", "\0", klv("STNM", "c", "Accelerometer") + klv("SCAL", "s", [418])
                + klv("ACCL", "s", np.clip(np.round(acc * 418), -32768, 32767).astype(int), size=6)),
            klv("STRM", "\0", klv("STNM", "c", "Exposure time") + klv("SHUT", "f", [shutter] * 30)),
            klv("STRM", "\0", klv("STNM", "c", "Sensor ISO") + klv("ISOE", "S", [iso] * 30)),
            klv("STRM", "\0", klv("STNM", "c", "MRV Frame Skip") + klv("MSKP", "B", mskp.astype(int))),
        ])
        out.append(klv("DEVC", "\0", devc))
    return out


def _trak(track_id: int, handler: bytes, entry: bytes, sizes: list[int], offsets: list[int], delta: int,
          timescale: int) -> bytes:
    n = len(sizes)
    tkhd = _full(b"tkhd", struct.pack(">IIIII", 0, 0, track_id, 0, n * delta) + b"\0" * 8
                 + struct.pack(">hhhh", 0, 0, 0, 0)
                 + struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000) + struct.pack(">II", 0, 0), flags=3)
    mdhd = _full(b"mdhd", struct.pack(">IIIIHH", 0, 0, timescale, n * delta, 0x55C4, 0))
    hdlr = _full(b"hdlr", struct.pack(">I", 0) + handler + b"\0" * 12 + b"GoPro\0")
    dref = _full(b"dref", struct.pack(">I", 1) + _full(b"url ", b"", flags=1))
    stbl = _box(b"stbl", b"".join([
        _full(b"stsd", struct.pack(">I", 1) + entry),
        _full(b"stts", struct.pack(">III", 1, n, delta)),
        _full(b"stsc", struct.pack(">IIII", 1, 1, 1, 1)),     # one sample per chunk
        _full(b"stsz", struct.pack(">II", 0, n) + struct.pack(f">{n}I", *sizes)),
        _full(b"stco", struct.pack(">I", n) + struct.pack(f">{n}I", *offsets)),
    ]))
    minf = _box(b"minf", _full(b"nmhd", b"") + _box(b"dinf", dref) + stbl)
    return _box(b"trak", tkhd + _box(b"mdia", mdhd + hdlr + minf))


def make_gopro_video(path: str, w: int = 320, h: int = 240, seconds: int = 6, fps: int = 30,
                     header: dict | None = None, sensors: dict | None = None, timecode_frame: int = 1_000_000,
                     with_header: bool = True, with_sensors: bool = True) -> str:
    """A `seconds`-long w x h video at `fps` with GoPro metadata. `header` / `sensors`: keyword
    arguments of `header_gpmf` / `sensor_payloads` (VRES / VFPS default to the video's)."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (h + 40, w + 4 * seconds * fps, 3), dtype=np.uint8)
    for f in range(seconds * fps):
        vw.write(np.ascontiguousarray(base[20:20 + h, 4 * f:4 * f + w]))
    vw.release()
    data = open(path, "rb").read()
    # the top-level boxes: everything before moov stays byte for byte (the video's chunk offsets hold)
    off, moov = 0, None
    while off + 8 <= len(data):
        size, typ = struct.unpack(">I4s", data[off:off + 8])
        if typ == b"moov":
            moov = (off, off + size)
        off += size
    assert moov is not None and moov[1] == len(data), "expected the moov box at the end"
    head, moov_payload = data[:moov[0]], data[moov[0] + 8:moov[1]]
    payloads = sensor_payloads(seconds, **(sensors or {})) if with_sensors else []
    tc = struct.pack(">I", int(timecode_frame))
    mdat_body = b"".join(payloads) + tc
    mdat_start = len(head)
    first = mdat_start + 8
    offs, o = [], first
    for p_ in payloads:
        offs.append(o)
        o += len(p_)
    tc_off = o
    extra = b""
    if with_sensors:
        gp_entry = _box(b"gpmd", b"\0" * 6 + struct.pack(">H", 1))
        extra += _trak(100, b"meta", gp_entry, [len(p_) for p_ in payloads], offs, 1000, 1000)
    tc_entry = _box(b"tmcd", b"\0" * 6 + struct.pack(">H", 1) + struct.pack(">I", 0) + struct.pack(">I", 2)
                    + struct.pack(">II", fps * 1000, 1000) + bytes([fps, 0]))
    extra += _trak(101, b"tmcd", tc_entry, [4], [tc_off], seconds * fps, fps)
    if with_header:
        hk = dict(w=w, h=h, fps=(fps * 1000, 1000))
        hk.update(header or {})
        extra += _box(b"udta", _box(b"GPMF", header_gpmf(**hk)))
    new_moov = _box(b"moov", moov_payload + extra)
    with open(path, "wb") as fh:
        fh.write(head + _box(b"mdat", mdat_body) + new_moov)
    return path
