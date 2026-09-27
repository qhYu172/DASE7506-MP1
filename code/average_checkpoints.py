"""Uniformly average model weights from Iteration 14 checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, sha


DEFAULT_STEPS = (17760, 18240, 18720, 19200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent/'runs/iter_14')
    parser.add_argument('--steps', type=int, nargs='+', default=DEFAULT_STEPS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = [args.run_dir/f'checkpoint_step_{step}.pt' for step in args.steps]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error(f'Missing checkpoints: {missing}')

    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    reference = checkpoints[0]
    for checkpoint, step in zip(checkpoints, args.steps):
        if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != reference['implementation']:
            raise ValueError(f'Incompatible checkpoint at step {step}.')
        if checkpoint['config'] != reference['config'] or checkpoint['step'] != step:
            raise ValueError(f'Configuration or step mismatch at step {step}.')

    averaged = {}
    for name in reference['model']:
        tensors = [checkpoint['model'][name] for checkpoint in checkpoints]
        if tensors[0].is_floating_point():
            averaged[name] = torch.stack([tensor.float() for tensor in tensors]).mean(0)
        else:
            if not all(torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f'Non-floating state differs for {name}.')
            averaged[name] = tensors[0].clone()

    output = args.output or args.run_dir/'checkpoint.pt'
    result = dict(reference)
    last_checkpoint = checkpoints[args.steps.index(max(args.steps))]
    result.update(model=averaged, step=max(args.steps), averaged_steps=list(args.steps),
                  train_tokens=last_checkpoint.get('train_tokens'))
    torch.save(result, output)
    metadata = {
        'protocol': PROTOCOL,
        'method': 'uniform_fp32_checkpoint_average',
        'steps': list(args.steps),
        'input_sha256': {str(step): sha(path) for step, path in zip(args.steps, paths)},
        'output': str(output),
        'output_sha256': sha(output),
    }
    (args.run_dir/'swa_metrics.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
