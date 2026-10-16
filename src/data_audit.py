"""Duplicate screening helpers, separate from restoration evaluation."""
import numpy as np
from PIL import Image
from scipy.fft import dctn


def perceptual_signatures(image: Image.Image) -> list[int]:
    """63-bit low-frequency hashes for four rotations and their mirrors.

    This is a screening heuristic, not proof of different scene provenance.
    It can miss arbitrary recrops or different exposures of the same scene.
    """
    small = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    signatures = []
    for mirror in (False, True):
        base = small[:, ::-1] if mirror else small
        for rotation in range(4):
            coefficients = dctn(np.rot90(base, rotation), norm='ortho')[:8,:8].ravel()[1:]
            bits = coefficients > np.median(coefficients)
            signatures.append(int(''.join('1' if bit else '0' for bit in bits), 2))
    return signatures


def hash_distance(first: list[int], second: list[int]) -> int:
    return min((a ^ b).bit_count() for a in first for b in second)
