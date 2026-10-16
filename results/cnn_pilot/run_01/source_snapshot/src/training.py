"""Training helpers that preserve source splits and full-image degradation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import wiener_restore
from .optics import simulate_observation


def load_training_sources(root: Path, limits: dict[str, int]) -> tuple[dict, dict]:
    """Only load train/validation scenes; this API deliberately rejects test."""
    if not limits or not set(limits).issubset({'train', 'validation'}):
        raise ValueError('Training loader accepts only train and validation splits.')
    for value in limits.values():
        if type(value) is not int or value < 1:
            raise ValueError('Source counts must be positive integers.')
    manifest_path = root/'results/data_preparation/data_manifest.json'
    selection_path = root/'results/validation/wiener_selection.json'
    manifest_bytes, selection_bytes = manifest_path.read_bytes(), selection_path.read_bytes()
    manifest, selection = json.loads(manifest_bytes), json.loads(selection_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest['status'] != 'source_split_frozen_after_duplicate_review':
        raise ValueError('Source split must be frozen.')
    if selection['data_manifest_sha256'] != manifest_sha:
        raise ValueError('Wiener selection belongs to a different source split.')
    all_ids = [r['id'] for r in manifest['files']]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Source ID occurs in multiple records.')
    # Verify that the selected K really follows the saved validation rule.
    means = np.asarray([r['l1_by_candidate'] for r in selection['per_image']]).mean(axis=0)
    expected_k = selection['protocol']['k_candidates'][int(np.argmin(means))]
    if selection['selected_k'] != expected_k:
        raise ValueError('Selected K does not match saved validation scores.')
    protocol = selection['protocol']
    if protocol['score_crop_ltrb'] != [64,64,320,320] or protocol['scene_size'] != 384:
        raise ValueError('Current training pipeline expects the frozen 384/256 spatial protocol.')
    sources = {}
    for split, count in limits.items():
        records = [r for r in manifest['files'] if r['project_split'] == split]
        if len(records) < count:
            raise ValueError(f'Not enough {split} source photos.')
        sources[split] = []
        for record in records[:count]:
            path = root/record['scene_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['scene_file_sha256']:
                raise ValueError(f'Prepared scene checksum mismatch: {record["id"]}')
            scene = np.load(path, allow_pickle=False).astype(np.float64)
            if scene.shape != (384,384):
                raise ValueError('Expected 384x384 scene.')
            seed = 100000+int(record['id'])
            simulation = simulate_observation(scene, **protocol['optics'],
                                               noise_sigma=protocol['noise_sigma'], seed=seed)
            wiener = wiener_restore(simulation['noisy'], simulation['psf'], selection['selected_k'])
            # Retain unclipped observations and baseline as the two input channels.
            inputs = np.stack([simulation['noisy'], wiener]).astype(np.float32)
            sources[split].append({'id': record['id'], 'inputs': torch.from_numpy(inputs),
                                   'target': torch.from_numpy(scene.astype(np.float32)[None]),
                                   'noise_seed': seed})
    provenance = {'data_manifest_sha256': manifest_sha,
                  'wiener_selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
                  'selected_k': selection['selected_k'], 'protocol': protocol,
                  'source_ids': {group: [s['id'] for s in samples] for group, samples in sources.items()},
                  'test_sources_loaded': False}
    return sources, provenance


def extract_training_patch(sample: dict, top: int, left: int, size: int = 128,
                           halo: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Read surrounding pixels as context; score only the central patch.

    The 5-pixel halo matches five stride-one 3x3 convolutions. No new blur
    or noise is introduced at patch boundaries. Loss pixels lie inside
    the same central region used for full-image monitoring.
    """
    if halo != 5 or size < 1 or top < 64 or left < 64 or top+size > 320 or left+size > 320:
        raise ValueError('Patch must fit inside [64:320]^2 and have a 5-pixel halo.')
    inputs = sample['inputs'][:, top-halo:top+size+halo, left-halo:left+size+halo]
    target = sample['target'][:, top:top+size, left:left+size]
    return inputs, target


def epoch_batches(samples: list[dict], rng: np.random.Generator,
                  patches_per_image: int = 4, batch_size: int = 4):
    indices = np.repeat(np.arange(len(samples)), patches_per_image)
    rng.shuffle(indices)
    inputs, targets = [], []
    for index in indices:
        top, left = rng.integers(64, 193, size=2)
        inp, target = extract_training_patch(samples[int(index)], int(top), int(left))
        inputs.append(inp)
        targets.append(target)
        if len(inputs) == batch_size:
            yield torch.stack(inputs), torch.stack(targets)
            inputs, targets = [], []
    if inputs:
        yield torch.stack(inputs), torch.stack(targets)


@torch.inference_mode()
def monitor(model, samples: list[dict], device: torch.device) -> list[dict]:
    """Equal-weight per-source L1 on full-image inference, shared score rule."""
    model.eval()
    rows = []
    for sample in samples:
        prediction = model(sample['inputs'][None].to(device)).cpu()[0,0]
        if not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite model output.')
        target = sample['target'][0,64:320,64:320]
        cnn_l1 = (prediction[64:320,64:320].clamp(0,1)-target).abs().mean().item()
        wiener_l1 = (sample['inputs'][1,64:320,64:320].clamp(0,1)-target).abs().mean().item()
        rows.append({'id': sample['id'], 'cnn_l1': cnn_l1, 'wiener_l1': wiener_l1})
    return rows
