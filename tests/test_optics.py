"""Numerical checks required before using the simulator to train a model."""

import numpy as np
import pytest
from scipy.special import j1

from src.optics import circular_convolve, make_psf, make_psf_with_info, psf_to_otf, simulate_observation


def direct_periodic_convolution(image, kernel):
    """Independent spatial-domain reference; deliberately not an FFT."""
    result = np.zeros_like(image, dtype=float)
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            result += kernel[row, column] * np.roll(
                image,
                (row - kernel.shape[0] // 2, column - kernel.shape[1] // 2),
                axis=(0, 1),
            )
    return result


@pytest.mark.parametrize("kernel_shape", [(3, 3), (4, 2), (3, 4), (2, 3)])
@pytest.mark.parametrize("image_shape", [(7, 8), (8, 7)])
def test_fft_matches_asymmetric_direct_periodic_convolution(kernel_shape, image_shape):
    rng = np.random.default_rng(12)
    image = rng.random(image_shape)
    kernel = rng.random(kernel_shape)
    kernel /= kernel.sum()
    np.testing.assert_allclose(circular_convolve(image, kernel), direct_periodic_convolution(image, kernel), atol=5e-15, rtol=0)


@pytest.mark.parametrize("size", [32, 33])
def test_centred_impulse_reproduces_full_size_psf(size):
    psf = make_psf(pupil_diameter=16, fft_size=64, kernel_size=size)
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1
    np.testing.assert_allclose(circular_convolve(impulse, psf), psf, atol=5e-17, rtol=0)


def test_psf_energy_and_constant_preservation():
    psf, info = make_psf_with_info()
    assert psf.shape == (129, 129)
    assert np.isfinite(psf).all() and (psf >= 0).all()
    assert abs(psf.sum() - 1) < 1e-14
    assert 0.95 < info["crop_energy_fraction"] < 1
    assert info["lost_energy_fraction"] > 0
    assert info["q_step"] == 0.25
    np.testing.assert_allclose(circular_convolve(np.full((160, 170), 0.42), psf), 0.42, atol=1e-14, rtol=0)
    assert abs(psf_to_otf(psf, (160, 170))[0, 0] - 1) < 1e-14


def test_zero_defocus_agrees_with_analytic_airy_profile_and_first_dark_ring():
    # Refined image-plane sampling, dq=1/16, resolves the first Airy minimum.
    psf, info = make_psf_with_info(0.0, pupil_diameter=128, fft_size=2048, kernel_size=129)
    centre = psf.shape[0] // 2
    q = np.arange(centre + 1) * info["q_step"]
    argument = np.pi * q
    analytic = np.ones_like(q)
    analytic[1:] = (2 * j1(argument[1:]) / argument[1:]) ** 2
    numerical = psf[centre, centre:] / psf[centre, centre]
    assert np.max(np.abs(numerical - analytic)) < 0.003
    interval = np.where((q > 0.9) & (q < 1.5))[0]
    measured_minimum = q[interval[np.argmin(numerical[interval])]]
    assert abs(measured_minimum - 1.21966989) <= info["q_step"]


def test_grid_refinement_preserves_image_plane_scale_and_blurred_scene():
    coarse, coarse_info = make_psf_with_info(0.15, 128, 512, 129)
    refined, refined_info = make_psf_with_info(0.15, 256, 1024, 129)
    assert coarse_info["q_step"] == refined_info["q_step"] == 0.25
    assert np.sum(np.abs(coarse - refined)) < 0.01
    scene = np.zeros((256, 256))
    scene[80:176, 80:176] = 1.0
    difference = circular_convolve(scene, coarse) - circular_convolve(scene, refined)
    assert np.max(np.abs(difference)) < 0.002


def test_support_extension_retains_more_energy_and_small_image_change():
    small, small_info = make_psf_with_info(0.15, 128, 512, 129)
    large, large_info = make_psf_with_info(0.15, 128, 512, 257)
    assert large_info["crop_energy_fraction"] > small_info["crop_energy_fraction"]
    scene = np.zeros((320, 320))
    scene[80:240, 80:240] = 1
    difference = circular_convolve(scene, small) - circular_convolve(scene, large)
    assert np.max(np.abs(difference)) < 0.015


def test_seeded_noise_is_repeatable_unclipped_and_does_not_modify_input():
    image = np.zeros((40, 40))
    kwargs = {"pupil_diameter": 16, "fft_size": 64, "kernel_size": 33}
    first = simulate_observation(image, seed=7, **kwargs)
    repeated = simulate_observation(image, seed=7, **kwargs)
    changed = simulate_observation(image, seed=8, **kwargs)
    np.testing.assert_array_equal(first["noisy"], repeated["noisy"])
    assert not np.array_equal(first["noisy"], changed["noisy"])
    assert first["noisy"].min() < 0 and first["noisy"].max() > 0
    assert not first["metadata"]["observation_clipped"]
    first["image"][0, 0] = 1
    assert image[0, 0] == 0
    noiseless = simulate_observation(image, noise_sigma=0, **kwargs)
    np.testing.assert_array_equal(noiseless["noisy"], noiseless["blurred"])


@pytest.mark.parametrize("kwargs", [
    {"defocus_waves": np.nan}, {"defocus_waves": np.inf}, {"defocus_waves": True},
    {"pupil_diameter": 3}, {"pupil_diameter": 20.5}, {"pupil_diameter": True},
    {"pupil_diameter": 600}, {"fft_size": 0}, {"kernel_size": 513}, {"kernel_size": 0},
])
def test_invalid_psf_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        make_psf(**kwargs)


@pytest.mark.parametrize("image", [np.zeros((2, 2, 3)), np.zeros((0, 5)), np.full((8, 8), np.nan), np.ones((8, 8), dtype=complex), np.full((8, 8), 255.0), np.full((8, 8), -0.1)])
def test_invalid_scene_raises_instead_of_silently_rescaling(image):
    with pytest.raises(ValueError):
        simulate_observation(image, pupil_diameter=4, fft_size=8, kernel_size=5)


@pytest.mark.parametrize("kwargs", [{"noise_sigma": -1}, {"noise_sigma": np.inf}, {"seed": -1}, {"seed": 1.5}, {"seed": True}])
def test_invalid_observation_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_observation(np.zeros((40, 40)), pupil_diameter=16, fft_size=64, kernel_size=33, **kwargs)


def test_invalid_kernel_or_output_shape_raises():
    for kernel in [np.zeros((3, 3)), np.array([[1.0, -0.1]]), np.full((3, 3), np.nan)]:
        with pytest.raises(ValueError):
            psf_to_otf(kernel, (4, 4))
    for shape in [(2, 2), (4,), (4.5, 4), (True, 4), (0, 4)]:
        with pytest.raises(ValueError):
            psf_to_otf(np.ones((3, 3)), shape)
