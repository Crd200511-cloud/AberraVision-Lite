"""One small, explicitly nonformal CNN learning and runtime pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import ResidualCNN
from src.training import load_training_sources, epoch_batches, monitor

PILOT = {'purpose': 'small-sample pipeline and timing pilot; not formal research run',
         'train_sources': 8, 'validation_sources': 4, 'epochs': 12,
         'seed': 20261012, 'learning_rate': .0005, 'optimizer': 'Adam',
         'adam_betas': [.9, .999], 'adam_eps': 1e-8, 'weight_decay': 0,
         'patches_per_image_per_epoch': 4, 'batch_size': 4,
         'scored_patch_size': 128, 'context_halo': 5,
         'loss': 'L1 on unclipped outputs; inner 128x128 of 138x138 input patch',
         'monitoring': 'both curves use full-image center 256x256 L1 after clipping predictions',
         'checkpoint_rule': 'minimum validation monitoring L1, including epoch 0; earliest on exact ties',
         'architecture': 'five 3x3 convolutions: 2-32-32-32-32-1, ReLU after first four; residual sum',
         'initialization': 'PyTorch default hidden layers; zero final convolution',
         'noise': 'one fixed realization per source; seed 100000+ID',
         'augmentation': 'none; random crop positions only', 'cpu_threads': 4}


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def save_checkpoint(path, model, epoch, config, provenance):
    torch.save({'state_dict': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'epoch': epoch, 'config': config, 'provenance': provenance}, path)


def run(output: Path, requested_device: str):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is nonempty; use a new directory to preserve previous pilot records.')
    if requested_device == 'auto':
        requested_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested_device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available in this process; use --device cpu.')
    device = torch.device(requested_device)
    torch.set_num_threads(PILOT['cpu_threads'])
    torch.manual_seed(PILOT['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(PILOT['seed'])
    output.mkdir(parents=True, exist_ok=True)
    config = {**PILOT, 'device': str(device), 'deterministic_algorithms': True}
    sources, provenance = load_training_sources(ROOT, {'train': 8, 'validation': 4})
    model = ResidualCNN().to(device)
    config['parameter_count'] = sum(p.numel() for p in model.parameters())
    environment = {'torch': str(torch.__version__), 'numpy': np.__version__,
                   'platform': platform.platform(), 'python': platform.python_version(),
                   'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available()}
    code_paths = ['pilot_train.py', 'src/model.py', 'src/training.py', 'src/optics.py', 'src/baseline.py']
    code_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths}
    (output/'run_config.json').write_text(json.dumps({'config': config, 'provenance': provenance,
              'environment': environment, 'code_sha256': code_hashes}, indent=2)+'\n')
    optimizer = torch.optim.Adam(model.parameters(), lr=PILOT['learning_rate'],
                                  betas=tuple(PILOT['adam_betas']), eps=PILOT['adam_eps'],
                                  weight_decay=0, foreach=False)
    history, detailed_monitoring = [], []
    best_loss, best_epoch = float('inf'), 0
    started = time.perf_counter()
    for epoch in range(PILOT['epochs']+1):
        synchronize(device)
        epoch_start = time.perf_counter()
        patch_loss_sum, patch_count, gradient_norms = 0., 0, []
        if epoch:
            model.train()
            for inputs, target in epoch_batches(sources['train'], rng):
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)[:,:,5:-5,5:-5]
                loss = torch.nn.functional.l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss.')
                loss.backward()
                # Detect invalid gradients without clipping or otherwise changing the optimizer.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
                patch_loss_sum += loss.item()*inputs.shape[0]
                patch_count += inputs.shape[0]
        synchronize(device)
        training_seconds = time.perf_counter()-epoch_start
        monitoring_start = time.perf_counter()
        train_rows = monitor(model, sources['train'], device)
        validation_rows = monitor(model, sources['validation'], device)
        synchronize(device)
        row = {'epoch': epoch,
               'patch_optimization_l1': patch_loss_sum/patch_count if patch_count else None,
               'train_monitor_l1': float(np.mean([r['cnn_l1'] for r in train_rows])),
               'validation_monitor_l1': float(np.mean([r['cnn_l1'] for r in validation_rows])),
               'train_wiener_l1': float(np.mean([r['wiener_l1'] for r in train_rows])),
               'validation_wiener_l1': float(np.mean([r['wiener_l1'] for r in validation_rows])),
               'training_seconds': training_seconds,
               'monitoring_seconds': time.perf_counter()-monitoring_start,
               'max_gradient_norm': max(gradient_norms) if gradient_norms else None}
        history.append(row)
        detailed_monitoring.append({'epoch': epoch, 'train': train_rows, 'validation': validation_rows})
        if row['validation_monitor_l1'] < best_loss:
            best_loss, best_epoch = row['validation_monitor_l1'], epoch
            save_checkpoint(output/'best.pt', model, epoch, config, provenance)
        with (output/'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f'Epoch {epoch:02d}: train={row["train_monitor_l1"]:.6f}, val={row["validation_monitor_l1"]:.6f}, '
              f'optimization={training_seconds:.2f}s', flush=True)
    total_seconds = time.perf_counter()-started
    save_checkpoint(output/'last.pt', model, PILOT['epochs'], config, provenance)
    # Check that save/load preserves actual predictions of the trained last model.
    reloaded = ResidualCNN().to(device)
    reloaded.load_state_dict(torch.load(output/'last.pt', map_location='cpu', weights_only=True)['state_dict'])
    probe = sources['validation'][0]['inputs'][None].to(device)
    with torch.inference_mode():
        difference = float((model(probe)-reloaded(probe)).abs().max().cpu())
    if difference > 1e-6:
        raise ValueError('Checkpoint reload changed predictions.')
    best_checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    reloaded.load_state_dict(best_checkpoint['state_dict'])
    best_rows = monitor(reloaded, sources['validation'], device)
    reloaded_score = float(np.mean([r['cnn_l1'] for r in best_rows]))
    if abs(reloaded_score-best_loss) > 1e-6:
        raise ValueError('Reloaded best checkpoint does not reproduce its validation score.')
    summary = {'purpose': PILOT['purpose'], 'best_epoch': best_epoch,
               'initial_train_l1': history[0]['train_monitor_l1'],
               'initial_validation_l1': history[0]['validation_monitor_l1'],
               'best_validation_l1': best_loss, 'last_train_l1': history[-1]['train_monitor_l1'],
               'last_validation_l1': history[-1]['validation_monitor_l1'],
               'total_train_and_monitor_seconds': total_seconds,
               'median_training_epoch_seconds_excluding_first': float(np.median([r['training_seconds'] for r in history[2:]])),
               'median_monitoring_epoch_seconds_excluding_initial': float(np.median([r['monitoring_seconds'] for r in history[1:]])),
               'reload_prediction_max_abs_difference': difference,
               'reloaded_best_validation_l1': reloaded_score,
               'test_sources_loaded': False, 'formal_training_completed': False}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'per_image_monitoring.json').write_text(json.dumps(detailed_monitoring, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(8,4.8), layout='constrained')
    epochs = [r['epoch'] for r in history]
    ax.plot(epochs, [r['train_monitor_l1'] for r in history], 'o-', label='8 training photos', color='#246a9e')
    ax.plot(epochs, [r['validation_monitor_l1'] for r in history], 'o-', label='4 validation photos', color='#bd522a')
    ax.axhline(history[0]['validation_wiener_l1'], color='#bd522a', linestyle='--', alpha=.6, label='Validation Wiener baseline')
    ax.scatter([best_epoch], [best_loss], s=130, facecolors='none', edgecolors='black', zorder=4, label='Selected pilot checkpoint')
    ax.set(xlabel='Epoch (0 = before training)', ylabel='Mean L1 (lower is better)',
           title='Small-sample pilot | Same full-image scoring rule for both curves')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(output/'learning_curve.png', dpi=150)
    plt.close(fig)
    # Preselected validation photo: first ID, regardless of improvement.
    example = sources['validation'][0]
    with torch.inference_mode():
        prediction = reloaded(example['inputs'][None].to(device)).cpu()[0,0].numpy()
    preview_dir = ROOT/'results/data_preview/cnn_pilot'/output.name
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1,3,figsize=(11,4),layout='constrained')
    panels = [example['target'][0].numpy(), example['inputs'][1].numpy(), prediction]
    for ax, array, title in zip(axes, panels, ['Original target','Wiener K=0.01','Wiener + pilot CNN']):
        ax.imshow(array[64:320,64:320],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f'Validation {example["id"]} | Best pilot epoch {best_epoch} | Not a final test result',fontsize=11)
    fig.savefig(preview_dir/'example.png',dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/cnn_pilot/run_01')
    parser.add_argument('--device', choices=['auto','cpu','mps'], default='auto')
    args = parser.parse_args()
    try:
        run(args.output, args.device)
    except ValueError as exc:
        parser.error(str(exc))
