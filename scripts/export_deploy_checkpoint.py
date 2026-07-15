"""Export a training checkpoint as a compact inference-only checkpoint.

The deployment server only needs the composed Hydra config and EMA policy
state. Raw model weights, optimizer moments, and workspace state are omitted.
"""

import argparse
import os
import pathlib

import dill
import torch


def format_size(num_bytes):
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.2f} {unit}'
        value /= 1024


def main():
    parser = argparse.ArgumentParser(
        description='Strip a DP3 training checkpoint for deployment.')
    parser.add_argument('input', type=pathlib.Path,
                        help='Full training checkpoint')
    parser.add_argument('output', type=pathlib.Path,
                        help='Inference-only checkpoint to create')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite the output if it already exists')
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f'input checkpoint does not exist: {input_path}')
    if input_path == output_path:
        parser.error('input and output paths must be different')
    if output_path.exists() and not args.force:
        parser.error(f'output already exists: {output_path} (use --force)')

    print(f'Loading {input_path} ({format_size(input_path.stat().st_size)})')
    payload = torch.load(
        input_path.open('rb'), pickle_module=dill, map_location='cpu',
        weights_only=False)
    if 'cfg' not in payload:
        raise KeyError('checkpoint has no cfg')
    state_dicts = payload.get('state_dicts', {})
    if 'ema_model' not in state_dicts:
        raise KeyError('checkpoint has no state_dicts["ema_model"]')

    deploy_payload = {
        'format': 'dp3_deploy_v1',
        'cfg': payload['cfg'],
        'state_dicts': {
            'ema_model': state_dicts['ema_model'],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + '.tmp')
    if temp_path.exists():
        temp_path.unlink()
    try:
        torch.save(
            deploy_payload, temp_path.open('wb'), pickle_module=dill)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    output_size = output_path.stat().st_size
    input_size = input_path.stat().st_size
    print(f'Saved {output_path} ({format_size(output_size)})')
    print(f'Size reduction: {input_size / output_size:.2f}x')


if __name__ == '__main__':
    main()
