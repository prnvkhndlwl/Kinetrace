"""Pixel-centre arithmetic shared by the tracker, the segmenter, the silhouette, the fingerprint and the
window (R37, R39). Pixel i's centre is at i (OpenCV's convention, Kinetrace's scene coordinates), so a
picture scaled by s maps p to (p + 0.5) * s - 0.5, not p * s. No dtype is forced here: every caller keeps
its own casts and rounding, so the results are the ones its inline copy gave, bit for bit. Imports no
other Kinetrace module."""
from __future__ import annotations

import numpy as np


def centre_scale(p, s):
    """A position in a picture scaled by `s` (working -> native: s = native / working)."""
    return (p + 0.5) * s - 0.5


def centred_origin(c, size, extent) -> int:
    """(R39) The first pixel of a `size`-wide window centred on `c`, kept inside 0 .. `extent`: the
    tracker's ROI crop, the balls' crop and the fingerprint's crop. Python's `round` (ties to even) is
    what all three used; `project.half_up` would move the ties."""
    return int(np.clip(round(c - size / 2), 0, max(0, extent - size)))


def centre_unscale(p, s):
    """The inverse of `centre_scale` (native -> working). A division, never `* (1 / s)`: not the same floats."""
    return (p + 0.5) / s - 0.5
