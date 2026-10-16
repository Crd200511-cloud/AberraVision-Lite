"""Shared, prespecified scoring: full input -> central crop -> equal clipping."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SSIM_SETTINGS = {'data_range': 1.0, 'channel_axis': None, 'gaussian_weights': True,
                 'sigma': 1.5, 'win_size': 11, 'use_sample_covariance': False,
                 'K1': .01, 'K2': .03}


def score_reconstruction(target, prediction):
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if target.shape != (384,384) or prediction.shape != target.shape:
        raise ValueError('Expected matching 384x384 full images.')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Images must be finite.')
    if target.min() < 0 or target.max() > 1:
        raise ValueError('Target must be in [0,1].')
    x = target[64:320,64:320]
    estimate = np.clip(prediction[64:320,64:320],0,1)
    mse = float(np.mean((estimate-x)**2))
    return {'psnr_db': float(peak_signal_noise_ratio(x,estimate,data_range=1.0)) if mse else float('inf'),
            'ssim': float(structural_similarity(x,estimate,**SSIM_SETTINGS)),
            'l1': float(np.mean(np.abs(estimate-x)))}
