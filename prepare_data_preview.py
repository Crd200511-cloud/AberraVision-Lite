"""Prepare local natural-image teaching samples, not final benchmark results."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import wiener_restore
from src.data import prepare_scene
from src.optics import simulate_observation


def main():
    records = json.loads((ROOT/'results/data_preparation/preview_acquisition.json').read_text())['files']
    output = ROOT/'results/data_preview'
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    fig, axes = plt.subplots(len(records), 3, figsize=(11, 4*len(records)), squeeze=False, layout='constrained')
    for row, record in enumerate(records):
        scene, preprocessing = prepare_scene(ROOT/record['path'], size=384)
        if preprocessing['source_sha256'] != record['sha256']:
            raise ValueError('Source hash mismatch.')
        noise_seed = 100000+int(record['id'])
        simulation = simulate_observation(scene, defocus_waves=.15, noise_sigma=.01, seed=noise_seed)
        restored = wiener_restore(simulation['noisy'], simulation['psf'], regularization=.001)
        arrays = [scene, simulation['noisy'], restored]
        for ax, array, title in zip(axes[row], arrays, ['A: Original target', 'B: Blur + noise', 'C: Wiener (K = 0.001)']):
            ax.imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'DIV2K {record["id"]} | {title}', fontsize=10)
            ax.set_axis_off()
        np.savez_compressed(output/f'{record["id"]}.npz', original=scene, observed=simulation['noisy'],
                            wiener=restored, psf=simulation['psf'])
        # The same preselected center region for all panels, no outcome-driven crop selection.
        detail, detail_axes = plt.subplots(1, 3, figsize=(10, 3.6), layout='constrained')
        for ax, array, title in zip(detail_axes, arrays, ['A: Original target', 'B: Blur + noise', 'C: Wiener']):
            ax.imshow(array[128:256,128:256], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(title)
            ax.set_axis_off()
        detail.suptitle(f'DIV2K {record["id"]} | Same center crop | Teaching only', fontsize=11)
        detail.savefig(output/f'{record["id"]}_detail.png', dpi=150)
        plt.close(detail)
        # Connect the L1 lesson to this preselected crop; not a test-set score.
        target_crop = scene[128:256,128:256]
        teaching_l1 = {
            name: float(np.mean(np.abs(np.clip(array[128:256,128:256], 0, 1)-target_crop)))
            for name, array in zip(('observed', 'wiener'), arrays[1:])
        }
        metadata.append({'id': record['id'], 'project_split': 'train', 'preprocessing': preprocessing,
                         'optics': simulation['metadata'], 'wiener_k': .001, 'display_range': [0,1],
                         'raw_values_clipped': False, 'detail_crop_ltrb': [128,128,256,256],
                         'teaching_l1': teaching_l1,
                         'teaching_l1_rule': 'mean absolute error on the fixed center 128x128; both predictions clipped to [0,1]; not formal evaluation'})
    fig.suptitle('Natural-photo scene proxies | Fixed defocus 0.15 waves, noise sigma 0.01 | No CNN', fontsize=12)
    fig.savefig(output/'natural_samples.png', dpi=150)
    plt.close(fig)
    (ROOT/'results/data_preparation/preview_processing.json').write_text(json.dumps({
        'purpose': 'training-candidate teaching preview only', 'formal_settings_frozen': False,
        'samples': metadata}, indent=2)+'\n')
    print(output/'natural_samples.png')


if __name__ == '__main__':
    main()
