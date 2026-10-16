"""Run the single prespecified formal experiment from a fresh initialization."""
import argparse
import hashlib
import json
from pathlib import Path

from src.runner import run

ROOT = Path(__file__).resolve().parent


def main(output):
    raw = (ROOT/'config.json').read_bytes()
    protocol = json.loads(raw)
    if protocol['status'] != 'frozen_before_formal_training_and_test':
        raise ValueError('Freeze the experiment protocol first.')
    for key, name in [('data_manifest_sha256','results/data_preparation/data_manifest.json'),
                      ('wiener_selection_sha256','results/validation/wiener_selection.json')]:
        if protocol[key] != hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
            raise ValueError(f'Frozen prerequisite changed: {name}')
    settings = {**protocol['training'], 'formal_training': True,
                'protocol_sha256': hashlib.sha256(raw).hexdigest()}
    run(output, 'cpu', settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/formal/run_01')
    args = parser.parse_args()
    main(args.output)
