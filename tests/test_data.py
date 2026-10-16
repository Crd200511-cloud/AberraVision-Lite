import numpy as np
from PIL import Image

from src.data import candidate_split, prepare_scene, srgb_to_linear
from src.data_audit import perceptual_signatures, hash_distance


def test_split_has_independent_source_ids_and_expected_counts():
    split = candidate_split()
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [100, 20, 30]
    assert len(set(sum(split.values(), []))) == 150
    assert split == candidate_split()
    assert all(1 <= int(i) <= 800 for group in split.values() for i in group)


def test_srgb_reference_values():
    np.testing.assert_allclose(srgb_to_linear([0, .04045, .5, 1]),
                               [0, .00313080495, .21404114048, 1], atol=1e-10)


def test_average_in_linear_light_not_encoded_srgb(tmp_path):
    # Equal white/black columns average to linear 0.5, not inverse_sRGB(0.5).
    rgb = np.zeros((258,258,3), dtype=np.uint8)
    rgb[:,::2] = 255
    path = tmp_path/'stripe.png'
    Image.fromarray(rgb).save(path)
    scene, info = prepare_scene(path, size=129)
    np.testing.assert_allclose(scene, .5, atol=1e-7)
    assert info['per_image_contrast_stretch'] is False


def test_linear_luminance_and_no_per_image_normalization(tmp_path):
    rgb = np.zeros((129,129,3), dtype=np.uint8)
    rgb[:,:,0] = 255
    path = tmp_path/'red.png'
    Image.fromarray(rgb).save(path)
    scene, _ = prepare_scene(path, size=129)
    np.testing.assert_allclose(scene, .2126, atol=1e-7)


def test_duplicate_screen_finds_rotated_and_mirrored_source():
    pixels = np.random.default_rng(9).integers(0,256,(64,64), dtype=np.uint8)
    source = Image.fromarray(pixels)
    signatures = perceptual_signatures(source)
    rotated = source.transpose(Image.Transpose.ROTATE_90)
    mirrored = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    assert hash_distance(signatures, perceptual_signatures(rotated)) <= 2
    assert hash_distance(signatures, perceptual_signatures(mirrored)) <= 2
