"""Procedural, redistributable scene for learning; not training/test data."""
from __future__ import annotations

import numpy as np


def learning_pattern(size: int = 256) -> np.ndarray:
    """A linear-intensity scene with lines, a slanted edge, texture and points."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 160:
        raise ValueError("size must be an integer >= 160")
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    image = np.full((size, size), 0.08)
    # Top left: three line spacings, all with identical original contrast.
    for y0, y1, period in [(0.10, 0.20, 0.025), (0.23, 0.33, 0.05), (0.36, 0.46, 0.10)]:
        region = (xx > 0.08) & (xx < 0.46) & (yy > y0) & (yy < y1)
        stripes = ((xx / period) % 1 < 0.5)
        image[region] = np.where(stripes[region], 0.9, 0.15)
    # Top right: a smooth surface cut by a slanted edge.
    region = (xx > 0.55) & (xx < 0.92) & (yy > 0.10) & (yy < 0.46)
    image[region] = np.where((xx - 0.3 * yy)[region] > 0.64, 0.85, 0.20)
    # Bottom left: smooth variation makes random fluctuations easy to see.
    region = (xx > 0.08) & (xx < 0.46) & (yy > 0.55) & (yy < 0.90)
    texture = 0.47 + 0.19 * np.sin(2 * np.pi * 5 * xx) * np.cos(2 * np.pi * 4 * yy)
    image[region] = texture[region]
    # Bottom right: edges and two bright point-like disks.
    r = np.hypot(xx - 0.72, yy - 0.70)
    image[(r > 0.075) & (r < 0.11)] = 0.88
    image[(xx > 0.56) & (xx < 0.89) & (yy > 0.85) & (yy < 0.875)] = 0.80
    for cx in (0.67, 0.78):
        image[np.hypot(xx - cx, yy - 0.70) < 0.008] = 1.0
    return image
