"""Train the student model, with optional R-Drop and scheduled snapshots."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/iter_14.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=19200)
    p.add_argument('--cosine-horizon', type=int, default=19200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--rdrop-alpha', type=float, default=1.0)
    p.add_argument('--save-steps', default='17760,18240,18720,19200',
                   help='Comma-separated update numbers for checkpoint snapshots.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.cosine_horizon < 1:
        p.error('Batch size, step count, and cosine horizon must be positive.')
    if args.rdrop_alpha < 0:
        p.error('R-Drop alpha must be non-negative.')
    try:
        save_steps = {int(value) for value in args.save_steps.split(',') if value.strip()}
    except ValueError:
        p.error('--save-steps must be a comma-separated list of integers.')
    if any(step < 1 or step > args.steps for step in save_steps):
        p.error('Every save step must be between 1 and --steps.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    def save_checkpoint(path, step):
        torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                    'model':model.state_dict(),'seed':args.seed,
                    'train_tokens':step*args.batch_size*256,'step':step},path)

    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_validation_bpb = float('inf')
    best_step = None
    best_checkpoint = args.run_dir/'best_validation_checkpoint.pt'
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.cosine_horizon)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits1 = model(batch[:,:-1]).float()
            logits2 = model(batch[:,:-1]).float()
            targets = batch[:,1:].flatten()
            ce1 = F.cross_entropy(logits1.flatten(0,1), targets)
            ce2 = F.cross_entropy(logits2.flatten(0,1), targets)
            logp1 = F.log_softmax(logits1, dim=-1)
            logp2 = F.log_softmax(logits2, dim=-1)
            p1 = logp1.exp()
            p2 = logp2.exp()
            kl12 = F.kl_div(logp2, p1, reduction='batchmean') / logits1.shape[1]
            kl21 = F.kl_div(logp1, p2, reduction='batchmean') / logits1.shape[1]
            ce = 0.5 * (ce1 + ce2)
            kl = 0.5 * (kl12 + kl21)
            loss = ce + args.rdrop_alpha * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'ce':ce.item(),'kl':kl.item(),
                   'learning_rate':learning_rate,
                   'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if step + 1 in save_steps:
            save_checkpoint(args.run_dir/f'checkpoint_step_{step+1}.pt', step+1)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if intermediate['bpb'] < best_validation_bpb:
                best_validation_bpb = intermediate['bpb']
                best_step = step + 1
                save_checkpoint(best_checkpoint, best_step)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    if validation['bpb'] < best_validation_bpb:
        best_validation_bpb = validation['bpb']
        best_step = args.steps
        save_checkpoint(best_checkpoint, best_step)
    checkpoint = args.run_dir/'checkpoint.pt'
    save_checkpoint(checkpoint, args.steps)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'cosine_horizon':args.cosine_horizon,'rdrop_alpha':args.rdrop_alpha,
              'save_steps':sorted(save_steps),
              'best_validation_bpb':best_validation_bpb,'best_step':best_step,
              'best_checkpoint_sha256':sha(best_checkpoint),
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
