"""Inference-only validation ablations for the frozen Iteration 13 checkpoint."""

import argparse
import json
from pathlib import Path
import types

import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score


def neural_only(self, ids):
    return F.softmax(self(ids).float() / self.temperature, dim=-1).log()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).resolve().parent / 'runs/iter_13/checkpoint.pt')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'runs/iter_13/ablation_validation_cpu_fp32.json')
    parser.add_argument('--threads', type=int, default=32)
    args = parser.parse_args()

    device, precision = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] != 'student':
        raise ValueError('Expected the frozen student checkpoint for this protocol.')
    model, implementation_sha = make_model('student', checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    original_predict = model.predict_log_probs
    validation = load_data()['validation']

    variants = (
        ('base_neural_t1', False, 1.0),
        ('neural_t1_10', False, 1.10),
        ('cache_t1', True, 1.0),
        ('cache_t1_10', True, 1.10),
    )
    results = {}
    for name, cache_on, temperature in variants:
        model.temperature = temperature
        model.predict_log_probs = original_predict if cache_on else types.MethodType(neural_only, model)
        with torch.inference_mode():
            result = score(model, *validation, device, precision)
        result.pop('window_nll_nats')
        results[name] = {'cache': cache_on, 'temperature': temperature, **result}
        print(f'{name}: {result["bpb"]:.9f} BPB, {result["seconds"]:.3f} s', flush=True)

    output = {
        'protocol': PROTOCOL,
        'split': 'validation',
        'precision': precision,
        'checkpoint_sha256': sha(args.checkpoint),
        'implementation_sha256': implementation_sha,
        'ablation_script_sha256': sha(Path(__file__)),
        'threads_requested': args.threads,
        'variants': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
