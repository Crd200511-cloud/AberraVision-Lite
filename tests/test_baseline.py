"""Independent algebraic checks of the restoration baseline."""

import numpy as np
import pytest

from src.baseline import wiener_restore


def test_delta_psf_has_expected_regularized_gain_without_clipping():
    image = np.array([[-0.3, 1.8, 0.1], [0.9, -0.2, 1.1]])
    np.testing.assert_allclose(wiener_restore(image, np.ones((1, 1)), .2), image/1.2, atol=1e-14)


def test_agrees_with_independent_dense_ridge_solution():
    # Construct the periodic imaging matrix directly, without any FFT helper.
    rows, cols = 5, 6
    kernel = np.array([[0.05, 0.2, 0.03], [0.1, 0.3, 0.1], [0.07, 0.1, 0.05]])
    matrix = np.zeros((rows*cols, rows*cols))
    for y in range(rows):
        for x in range(cols):
            for ky in range(3):
                for kx in range(3):
                    sy, sx = (y-(ky-1)) % rows, (x-(kx-1)) % cols
                    matrix[y*cols+x, sy*cols+sx] += kernel[ky, kx]
    observation = np.random.default_rng(13).normal(size=(rows, cols))
    k = .07
    expected = np.linalg.solve(matrix.T@matrix+k*np.eye(rows*cols), matrix.T@observation.ravel())
    actual = wiener_restore(observation, kernel, k)
    np.testing.assert_allclose(actual.ravel(), expected, atol=1e-13)


def test_otf_zeros_are_finite_and_input_is_unchanged():
    observation = np.arange(64, dtype=float).reshape(8, 8)/64
    saved = observation.copy()
    result = wiener_restore(observation, np.ones((2, 2))/4, 1e-4)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(observation, saved)


@pytest.mark.parametrize("k", [0, -0.1, np.nan, np.inf, True])
def test_invalid_regularization(k):
    with pytest.raises(ValueError):
        wiener_restore(np.ones((4, 4)), np.ones((1, 1)), k)
