import numpy as np
import pytest
from src.metrics import score_reconstruction


def test_known_constant_error_gives_twenty_db_and_correct_ssim():
    target=np.full((384,384),.5)
    result=score_reconstruction(target,np.full_like(target,.6))
    assert result['psnr_db']==pytest.approx(20)
    assert result['l1']==pytest.approx(.1)
    assert result['ssim']==pytest.approx((2*.5*.6+.01**2)/(.5**2+.6**2+.01**2),abs=1e-10)


def test_border_is_excluded_and_clipping_does_not_mutate_raw():
    target=np.ones((384,384))
    prediction=np.full_like(target,2.)
    prediction[:64]=-100
    original=prediction.copy()
    scores=score_reconstruction(target,prediction)
    assert scores['psnr_db']==float('inf') and scores['ssim']==pytest.approx(1)
    np.testing.assert_array_equal(original,prediction)


def test_metrics_reject_shape_and_nan_instead_of_silently_scoring():
    with pytest.raises(ValueError):
        score_reconstruction(np.zeros((384,384)),np.zeros((256,256)))
    with pytest.raises(ValueError):
        score_reconstruction(np.zeros((384,384)),np.full((384,384),np.nan))
