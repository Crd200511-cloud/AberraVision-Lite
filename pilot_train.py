"""One small, explicitly nonformal CNN learning and runtime pilot."""
import argparse
from pathlib import Path
from src.runner import run

ROOT = Path(__file__).resolve().parent

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/cnn_pilot/run_01')
    parser.add_argument('--device', choices=['auto','cpu','mps'], default='auto')
    args = parser.parse_args()
    try:
        run(args.output, args.device, PILOT)
    except ValueError as exc:
        parser.error(str(exc))
