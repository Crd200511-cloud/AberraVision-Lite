"""Source-image splits and fixed linear grayscale scene preparation."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps


def candidate_split(seed: int = 20261012) -> dict[str, list[str]]:
    """Project-specific split within official DIV2K train HR, before cropping.

    This separates source IDs, but does not replace duplicate-content audit.
    It is not the official DIV2K challenge split or benchmark.
    """
    ids = np.random.default_rng(seed).permutation(np.arange(1, 801))[:150]
    return {name: [f'{i:04d}' for i in group] for name, group in
            zip(('train', 'validation', 'test'), (ids[:100], ids[100:120], ids[120:]))}


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if not np.all(np.isfinite(rgb)) or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError('sRGB values must be finite and in [0, 1].')
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)


def prepare_scene(path: Path, size: int = 384) -> tuple[np.ndarray, dict]:
    """Color-manage if tagged; linearize, luminance, BOX resize, center crop.

    Natural images serve as scene proxies, not measured RAW irradiance.
    Preprocessing happens before simulated optical blur and Gaussian noise.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 129:
        raise ValueError('size must be an integer >= 129.')
    raw_bytes = path.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as source:
        icc = source.info.get('icc_profile')
        photo = ImageOps.exif_transpose(source).convert('RGB')
        if icc:
            photo = ImageCms.profileToProfile(photo, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                              ImageCms.createProfile('sRGB'), outputMode='RGB')
        width, height = photo.size
        if min(width, height) < size:
            raise ValueError('Source must be at least the target size; no upsampling.')
        rgb = np.asarray(photo, dtype=np.float64) / 255
    luminance = srgb_to_linear(rgb) @ np.array([.2126, .7152, .0722])
    scale = size / min(width, height)
    resized_shape = (max(size, round(width * scale)), max(size, round(height * scale)))
    resized = Image.fromarray(luminance.astype(np.float32)).resize(resized_shape, Image.Resampling.BOX)
    left, top = (resized.width - size)//2, (resized.height - size)//2
    scene = np.asarray(resized.crop((left, top, left+size, top+size)), dtype=np.float64)
    return scene, {
        'source_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'oriented_source_size_wh': [width, height], 'scene_size_hw': [size, size],
        'color_rule': 'embedded ICC to sRGB' if icc else 'untagged RGB assumed sRGB',
        'preprocessing': 'sRGB inverse transfer -> linear Rec.709 luminance -> BOX resize shorter side -> center crop',
        'resized_size_wh': list(resized_shape), 'crop_box_ltrb': [left, top, left+size, top+size],
        'per_image_contrast_stretch': False,
    }
