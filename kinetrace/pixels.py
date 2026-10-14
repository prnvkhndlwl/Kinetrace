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


def centre_unscale(p, s):
    """The inverse of `centre_scale` (native -> working). A division, never `* (1 / s)`: not the same floats."""
    return (p + 0.5) / s - 0.5
