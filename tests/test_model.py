import numpy as np
import pytest
import torch

from src.model import ResidualCNN
from src.training import extract_training_patch, load_training_sources

torch.set_num_threads(2)


def test_initial_prediction_is_unclipped_wiener():
    model = ResidualCNN()
    inputs = torch.randn(2,2,17,19)
    with torch.inference_mode():
        output = model(inputs)
    torch.testing.assert_close(output, inputs[:,1:2], rtol=0, atol=0)
    assert output.min() < 0 and output.max() > 1


def test_network_does_not_accept_target_as_extra_input_channel():
    with pytest.raises(ValueError, match='Expected'):
        ResidualCNN()(torch.zeros(1,3,16,16))


def test_training_loader_rejects_test_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match='only train and validation'):
        load_training_sources(tmp_path, {'test': 1})


def test_halo_patch_prediction_matches_full_image_interior():
    torch.manual_seed(11)
    model = ResidualCNN()
    with torch.no_grad():
        model.correction[-1].weight.normal_(0,.03)
    sample = {'inputs': torch.randn(2,192,192), 'target': torch.randn(1,192,192)}
    inp, target = extract_training_patch(sample, 64, 72, size=32)
    with torch.inference_mode():
        full = model(sample['inputs'][None])[:,:,64:96,72:104]
        patch = model(inp[None])[:,:,5:-5,5:-5]
    torch.testing.assert_close(patch, full, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(target, sample['target'][:,64:96,72:104], atol=0, rtol=0)


def test_gradient_updates_learn_a_simple_residual():
    torch.manual_seed(7)
    model = ResidualCNN()
    inputs = torch.rand(2,2,26,26)
    target = inputs[:,1:2,5:-5,5:-5] + .05
    initial_hidden = model.correction[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    initial_loss = torch.nn.functional.l1_loss(model(inputs)[:,:,5:-5,5:-5],target).item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(inputs)[:,:,5:-5,5:-5],target)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    final_loss = torch.nn.functional.l1_loss(model(inputs)[:,:,5:-5,5:-5],target).item()
    assert final_loss < initial_loss/2
    assert not torch.equal(initial_hidden,model.correction[0].weight)
