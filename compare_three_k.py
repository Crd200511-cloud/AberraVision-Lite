"""Teaching comparison: one saved training observation, three Wiener K values."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.baseline import wiener_restore


def main():
    sample_id = '0356'
    manifest = json.loads((ROOT / 'results/data_preparation/data_manifest.json').read_text())
    assert any(r['id'] == sample_id and r['project_split'] == 'train' for r in manifest['files'])
    source = ROOT / f'results/data_preview/{sample_id}.npz'
    with np.load(source) as saved:
        target, observed, psf = (saved[k].copy() for k in ('original', 'observed', 'psf'))
    ks = [0.001, 0.01, 0.1]
    outputs = [wiener_restore(observed, psf, k) for k in ks]
    assert all(a.shape == target.shape and np.isfinite(a).all() for a in outputs)
    panels = [target, observed] + outputs
    labels = ['Original', 'Same degraded input'] + [f'K = {k:g}' for k in ks]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5), layout='constrained')
    for col, (array, label) in enumerate(zip(panels, labels)):
        for row, region in enumerate((array, array[128:256, 128:256])):
            axes[row, col].imshow(region, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[row, col].set_title(label if row == 0 else 'Same center crop')
            axes[row, col].axis('off')
    fig.suptitle('Training photo 0356 | Fixed PSF and exact noise realization | Teaching experiment')
    output = ROOT / 'results/data_preview/0356_three_k.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    record = {'purpose': 'teaching only; no formal parameter reselection', 'id': sample_id,
              'split': 'train', 'k_values': ks, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'only_changed_parameter': 'K', 'display_limits': [0, 1], 'crop': [128, 256, 128, 256],
              'input_and_psf': 'same saved arrays for all three reconstructions'}
    (output.with_suffix('.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
