"""Teaching comparison of Wiener regularization; not a validation sweep."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import wiener_restore
from src.optics import simulate_observation
from src.patterns import learning_pattern


def run_lesson(output: Path, defocus: float = .15, noise: float = .01):
    result = simulate_observation(learning_pattern(), defocus_waves=defocus, noise_sigma=noise, seed=42)
    # Preset teaching levels, not a claim that any is optimal for natural images.
    levels = [1e-5, 1e-3, 1e-2]
    restorations = [wiener_restore(result['noisy'], result['psf'], k) for k in levels]
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), layout='constrained')
    panels = [result['image'], result['noisy']] + restorations
    labels = ['Original', 'Blur + noise', 'Wiener: K = 0.00001', 'Wiener: K = 0.001', 'Wiener: K = 0.01']
    for ax, array, label in zip(axes, panels, labels):
        ax.imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f'Learning only | Defocus {defocus:.2f} waves | Noise sigma {noise:.3f} | Same observation', fontsize=12)
    fig.savefig(output/'wiener_comparison.png', dpi=160)
    plt.close(fig)
    # A compact, larger two-result view for side-by-side visual inspection.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), layout='constrained')
    for ax, array, label in zip(axes, [restorations[0], restorations[-1]], ['A: K = 0.00001', 'B: K = 0.01']):
        ax.imshow(array, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(label, fontsize=12)
        ax.set_axis_off()
    fig.suptitle('Same blurred input; only regularization changes', fontsize=12)
    fig.savefig(output/'wiener_tradeoff.png', dpi=140)
    plt.close(fig)
    np.savez_compressed(output/'wiener_arrays.npz', original=result['image'],
                        observed=result['noisy'], psf=result['psf'],
                        **{f'restored_{i}': arr for i, arr in enumerate(restorations)})
    metadata = dict(result['metadata'])
    metadata.update(purpose='teaching_only_not_formal_benchmark', regularization_values=levels,
                    wiener_variant='constant noise-to-signal spectrum ratio / identity Tikhonov',
                    display_range=[0, 1], raw_outputs_clipped=False,
                    noise_seed=42, formal_parameter_selected=False)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Learning figure:', (output/'wiener_comparison.png').resolve())
    print('Detail comparison:', (output/'wiener_tradeoff.png').resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--defocus', type=float, default=.15)
    parser.add_argument('--noise', type=float, default=.01)
    parser.add_argument('--output', type=Path, default=ROOT/'results/wiener_learning')
    args = parser.parse_args()
    try:
        run_lesson(args.output, args.defocus, args.noise)
    except ValueError as exc:
        parser.error(str(exc))
