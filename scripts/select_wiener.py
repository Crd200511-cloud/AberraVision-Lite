"""Choose K using validation mean L1 only, after the data split is frozen."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import wiener_restore
from src.optics import make_psf, make_psf_with_info, circular_convolve, simulate_observation

# Prespecified before looking at validation restoration results.
PROTOCOL = {
    'k_candidates': [.0001, .001, .003, .01, .03],
    'selection_metric': 'minimum mean per-image validation L1; first ascending K wins exact ties',
    'optics': {'defocus_waves': .15, 'pupil_diameter': 128, 'fft_size': 512, 'kernel_size': 129},
    'noise_sigma': .01, 'noise_seed_rule': '100000 + integer source ID',
    'scene_size': 384, 'score_crop_ltrb': [64,64,320,320],
    'score_output_processing': 'clip predictions to [0,1]; retain raw observations and restorations',
    'convergence_training_ids': ['0356','0223'],
    'convergence_max_abs_tolerances': {'pupil_refinement': .002, 'kernel_extension': .015},
    'test_restorations_accessed': False,
}


def load_scene(record):
    path = ROOT/record['scene_path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['scene_file_sha256']:
        raise ValueError('Prepared scene checksum mismatch.')
    scene = np.load(path, allow_pickle=False).astype(np.float64)
    if scene.shape != (384,384):
        raise ValueError('Expected 384x384 scene.')
    return scene


def main():
    manifest_path = ROOT/'results/data_preparation/data_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['status'] != 'source_split_frozen_after_duplicate_review':
        raise ValueError('Complete data review first.')
    output = ROOT/'results/validation'
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/'wiener_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != PROTOCOL:
        raise ValueError('Existing protocol differs; do not silently change selection rules.')
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    psf, psf_info = make_psf_with_info(**PROTOCOL['optics'])
    refined = make_psf(.15, 256, 1024, 129)
    extended = make_psf(.15, 128, 512, 257)
    training_records = {r['id']: r for r in manifest['files'] if r['project_split'] == 'train'}
    convergence = []
    for image_id in PROTOCOL['convergence_training_ids']:
        scene = load_scene(training_records[image_id])
        nominal = circular_convolve(scene, psf)
        differences = {'pupil_refinement': float(np.max(np.abs(nominal-circular_convolve(scene, refined)))),
                       'kernel_extension': float(np.max(np.abs(nominal-circular_convolve(scene, extended))))}
        if any(differences[key] > tolerance for key, tolerance in PROTOCOL['convergence_max_abs_tolerances'].items()):
            raise ValueError(f'Numerical convergence failed on training image {image_id}: {differences}')
        convergence.append({'id': image_id, **differences})
    validation = [r for r in manifest['files'] if r['project_split'] == 'validation']
    if len(validation) != 20:
        raise ValueError('Expected 20 validation source photos.')
    crop = np.s_[64:320,64:320]
    per_image = []
    for record in validation:
        scene = load_scene(record)
        sample = simulate_observation(scene, **PROTOCOL['optics'], noise_sigma=.01, seed=100000+int(record['id']))
        scores = [float(np.mean(np.abs(np.clip(wiener_restore(sample['noisy'], psf, k)[crop],0,1)-scene[crop])))
                  for k in PROTOCOL['k_candidates']]
        per_image.append({'id': record['id'], 'l1_by_candidate': scores})
    means = np.asarray([r['l1_by_candidate'] for r in per_image]).mean(axis=0)
    best_index = int(np.argmin(means))
    selected_k = PROTOCOL['k_candidates'][best_index]
    result = {'purpose': 'validation parameter selection, not final test results',
              'data_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'protocol': PROTOCOL, 'psf_info': psf_info, 'convergence': convergence,
              'validation_source_count': len(validation), 'per_image': per_image,
              'mean_l1_by_candidate': means.tolist(), 'selected_k': selected_k}
    (output/'wiener_selection.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(7,4.3), layout='constrained')
    ax.semilogx(PROTOCOL['k_candidates'], means, 'o-', color='#246a9e')
    ax.scatter([selected_k], [means[best_index]], color='#bc482d', s=65, zorder=3)
    for k, score in zip(PROTOCOL['k_candidates'], means):
        ax.annotate(f'{score:.4f}', (k, score), xytext=(0,9), textcoords='offset points', ha='center')
    ax.set(xlabel='Wiener K (log scale)', ylabel='Mean L1 (lower is better)',
           title='20 validation photos | Same observations for every K')
    ax.margins(y=.2)
    ax.grid(alpha=.2)
    fig.savefig(output/'wiener_validation.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'k_candidates': PROTOCOL['k_candidates'], 'mean_l1': means.tolist(), 'selected_k': selected_k}, indent=2))


if __name__ == '__main__':
    main()
