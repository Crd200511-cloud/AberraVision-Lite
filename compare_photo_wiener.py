"""Compare two K values on saved teaching inputs without regenerating noise."""
from __future__ import annotations

import argparse
import hashlib
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


def main(k: float):
    records = json.loads((ROOT/'results/data_preparation/preview_processing.json').read_text())['samples']
    output = ROOT/'results/data_preview'
    metrics = []
    for record in records:
        path = output/f'{record["id"]}.npz'
        with np.load(path) as stored:
            original, observed, psf, before = (stored[key].copy() for key in ('original', 'observed', 'psf', 'wiener'))
        old_k = record['wiener_k']
        # Confirm the cached baseline belongs to this observation and PSF.
        np.testing.assert_allclose(before, wiener_restore(observed, psf, old_k), rtol=1e-12, atol=1e-12)
        after = wiener_restore(observed, psf, k)
        if not np.all(np.isfinite(after)):
            raise ValueError('Nonfinite restored values.')
        panels = [original, before, after]
        labels = ['Original target', f'Before: K = {old_k:g}', f'After: K = {k:g}']
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
        for ax, array, label in zip(axes, panels, labels):
            ax.imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(label)
            ax.set_axis_off()
        fig.suptitle(f'DIV2K {record["id"]} | Same observation and PSF | Only K changes | Teaching only', fontsize=11)
        fig.savefig(output/f'{record["id"]}_k_comparison.png', dpi=150)
        plt.close(fig)
        crop = np.s_[128:256,128:256]
        scores = {name: float(np.mean(np.abs(np.clip(array[crop], 0, 1)-original[crop])))
                  for name, array in zip(('before', 'after'), (before, after))}
        np.savez_compressed(output/f'{record["id"]}_k_comparison.npz', before=before, after=after)
        metrics.append({'id': record['id'], 'old_k': old_k, 'new_k': k,
                        'source_npz_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'teaching_l1': scores})
        print(record['id'], scores)
    report = {'purpose': 'user-directed teaching comparison, not validation or test evaluation',
              'only_changed_parameter': 'K', 'raw_outputs_clipped': False,
              'score_rule': 'clip predictions to [0,1], then L1 on fixed center [128:256,128:256]',
              'samples': metrics}
    (ROOT/'results/data_preparation/wiener_k_comparison.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=float, default=.01)
    args = parser.parse_args()
    try:
        main(args.k)
    except ValueError as exc:
        parser.error(str(exc))
