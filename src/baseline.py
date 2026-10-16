"""Constant-spectrum-ratio Wiener baseline with periodic boundaries."""

import numpy as np
from scipy import fft

from .optics import _real, _real_array_2d, psf_to_otf


def wiener_restore(observation: np.ndarray, psf: np.ndarray,
                   regularization: float = 0.001) -> np.ndarray:
    """Return F^-1[conj(H) Y / (|H|^2 + K)], without output clipping.

    K > 0 approximates a constant noise-to-signal power-spectrum ratio.
    Equivalently this is identity-Tikhonov regularization with periodic
    boundaries. K is NOT the Gaussian noise standard deviation or simply
    its variance. It must be selected using validation data for research.
    This differs from a Wiener-Hunt filter using a Laplacian penalty.

    Observations may legitimately lie outside [0, 1]. All arrays retain
    floating-point values; clipping is a separate display/evaluation rule.
    """
    image = _real_array_2d(observation, "observation")
    k = _real(regularization, "regularization", minimum=0.0)
    if k == 0:
        raise ValueError("regularization must be > 0; direct inversion is singular at OTF zeros.")
    transfer = psf_to_otf(psf, image.shape)
    restored_spectrum = np.conj(transfer) * fft.fft2(image) / (np.abs(transfer)**2 + k)
    return fft.ifft2(restored_spectrum).real
