"""Scalar, monochromatic, incoherent imaging with a defocused circular pupil.

The pupil phase is ``2*pi*a*sqrt(3)*(2*rho**2-1)``.  ``a`` is the
continuous-pupil RMS wavefront error in waves, not a distance in millimetres.
Image-plane spacing is ``q_step = pupil_diameter / fft_size`` in units of
lambda*f/D.  This is a point-sampled, dimensionless teaching model.

Every kernel is centred at index ``(height//2, width//2)``.  Convolution uses
explicit periodic boundaries, including when either dimension is even.
"""

from numbers import Integral, Real

import numpy as np
from scipy import fft


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _real(value: float, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    value = float(value)
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}." if minimum is not None else "."))
    return value


def _real_array_2d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numbers.")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return array


def make_psf_with_info(
    defocus_waves: float = 0.15,
    pupil_diameter: int = 128,
    fft_size: int = 512,
    kernel_size: int = 129,
) -> tuple[np.ndarray, dict]:
    """Return a centred, normalized PSF and its numerical sampling metadata.

    ``pupil_diameter`` counts pupil-plane sampling intervals across D; it is
    not the number of nonzero pixels on an arbitrary row.  ``fft_size`` is
    the padded pupil grid side length.  To refine pupil sampling while
    holding image-plane sampling fixed, multiply both by the same factor.

    ``crop_energy_fraction`` measures retained energy relative to the full
    *discrete FFT grid*.  It is recorded before renormalizing the crop and
    is not a claim about energy outside an infinitely sampled optical plane.
    Kernel support is a deliberate approximation; inspect this fraction.
    Even kernels place their origin at ``kernel_size//2`` and have one more
    negative-coordinate sample than positive-coordinate sample.
    """
    a = _real(defocus_waves, "defocus_waves")
    diameter = _integer(pupil_diameter, "pupil_diameter", minimum=4)
    n = _integer(fft_size, "fft_size", minimum=4)
    k = _integer(kernel_size, "kernel_size")
    if diameter > n:
        raise ValueError("pupil_diameter must not exceed fft_size.")
    if k > n:
        raise ValueError("kernel_size must not exceed fft_size.")

    coordinate = (np.arange(n, dtype=np.float64) - n // 2) / (diameter / 2.0)
    rho2 = coordinate[:, None] ** 2 + coordinate[None, :] ** 2
    aperture = rho2 <= 1.0
    wavefront = a * np.sqrt(3.0) * (2.0 * rho2[aperture] - 1.0)
    pupil = np.zeros((n, n), dtype=np.complex128)
    pupil[aperture] = np.exp(2j * np.pi * wavefront)
    amplitude = fft.fftshift(fft.fft2(fft.ifftshift(pupil)))
    full_intensity = np.abs(amplitude) ** 2
    full_intensity /= full_intensity.sum()

    start = n // 2 - k // 2
    crop = full_intensity[start : start + k, start : start + k].copy()
    retained = float(crop.sum())
    if not np.isfinite(retained) or retained <= 0:
        raise ValueError("The selected PSF crop has no finite positive energy.")
    psf = crop / retained
    metadata = {
        "defocus_waves": a,
        "defocus_unit": "RMS waves on a continuous unit circular pupil",
        "pupil_diameter": diameter,
        "fft_size": n,
        "kernel_size": k,
        "q_step": diameter / n,
        "q_unit": "lambda*f/D",
        "crop_energy_fraction": retained,
        "lost_energy_fraction": max(0.0, 1.0 - retained),
        "energy_reference": "full discrete FFT grid before crop normalization",
        "pupil_nonzero_pixels": int(aperture.sum()),
        "sampled_wavefront_rms_waves": float(np.std(wavefront)),
        "kernel_origin": [k // 2, k // 2],
        "boundary": "periodic",
        "sampling": "point-sampled image plane; no pixel-area integration",
    }
    return psf, metadata


def make_psf(
    defocus_waves: float = 0.15,
    pupil_diameter: int = 128,
    fft_size: int = 512,
    kernel_size: int = 129,
) -> np.ndarray:
    """Return a centred PSF.  See :func:`make_psf_with_info` for conventions."""
    return make_psf_with_info(defocus_waves, pupil_diameter, fft_size, kernel_size)[0]


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad a centred PSF, move its origin to (0, 0), and take its FFT.

    The supplied kernel is neither cropped nor renormalized.  It must fit
    inside ``shape``.  The origin is ``(kh//2, kw//2)`` for odd and even
    dimensions alike; this convention avoids half-pixel/one-pixel shifts.
    """
    kernel = _real_array_2d(psf, "psf")
    if np.any(kernel < 0) or not np.isfinite(kernel.sum()) or kernel.sum() <= 0:
        raise ValueError("psf must be nonnegative with finite positive total energy.")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("shape must contain two positive integer dimensions.")
    target = tuple(_integer(size, "shape dimension") for size in shape)
    if any(k > n for k, n in zip(kernel.shape, target)):
        raise ValueError("psf must fit within the requested image shape.")
    padded = np.zeros(target, dtype=np.float64)
    padded[: kernel.shape[0], : kernel.shape[1]] = kernel
    padded = np.roll(padded, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1))
    return fft.fft2(padded)


def circular_convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve using periodic boundaries, preserving the image's shape.

    Inputs and results are floating point.  No clipping or contrast
    stretching is applied, and only round-off-size imaginary FFT residue
    is discarded.  A normalized PSF preserves a constant image.
    """
    scene = _real_array_2d(image, "image")
    return fft.ifft2(fft.fft2(scene) * psf_to_otf(psf, scene.shape)).real


def simulate_observation(
    image: np.ndarray,
    defocus_waves: float = 0.15,
    noise_sigma: float = 0.01,
    seed: int = 42,
    pupil_diameter: int = 128,
    fft_size: int = 512,
    kernel_size: int = 129,
) -> dict:
    """Blur a [0, 1] linear grayscale scene and add seeded Gaussian noise.

    The noise realization is repeatable for a given seed; use distinct,
    recorded seeds for distinct experimental images.  Negative or >1 noisy
    values are expected and retained.  Clip only a copy for display.
    ``image`` is copied so changing a returned array cannot edit the input.
    """
    scene = _real_array_2d(image, "image")
    if np.any(scene < 0) or np.any(scene > 1):
        raise ValueError("image must already be linear grayscale in [0, 1]; no automatic rescaling is performed.")
    sigma = _real(noise_sigma, "noise_sigma", minimum=0.0)
    random_seed = _integer(seed, "seed", minimum=0)
    psf, metadata = make_psf_with_info(defocus_waves, pupil_diameter, fft_size, kernel_size)
    blurred = circular_convolve(scene, psf)
    noisy = blurred + np.random.default_rng(random_seed).normal(0.0, sigma, scene.shape)
    metadata.update({"noise_sigma": sigma, "seed": random_seed, "noise_model": "additive independent Gaussian", "observation_clipped": False})
    return {"image": scene.copy(), "psf": psf, "blurred": blurred, "noisy": noisy, "metadata": metadata}
