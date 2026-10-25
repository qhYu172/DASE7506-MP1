"""Measure fresh official prediction and scoring processes; fail on either process error."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--split',choices=['validation','test'],required=True)
args=parser.parse_args()
cp=Path('runs/model_bundle_v3')
manifest=json.loads(Path('runs/v3-freeze.json').read_text())
for name,expected in manifest['assets'].items():
    assert hashlib.sha256((cp/name).read_bytes()).hexdigest()==expected,'Asset differs from freeze'
for name,expected in manifest['source'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,'Source differs from freeze'
env=os.environ.copy()
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    env[name]='4'
prediction=f'runs/{args.split}-v3-predictions.json'
score=f'runs/{args.split}-v3-score.json'
commands=[[sys.executable,'predict.py','--method','submission.py','--checkpoint',str(cp),
           '--split',args.split,'--output',prediction],
          [sys.executable,'score_predictions.py','--split',args.split,'--predictions',prediction,'--output',score]]
durations=[]
for command in commands:
    tick=time.perf_counter();subprocess.run(command,env=env,check=True)
    durations.append(time.perf_counter()-tick)
record={'split':args.split,'fresh_process_seconds':durations,'total_seconds':sum(durations),
        'commands':commands,'python':platform.python_version(),'platform':platform.platform(),
        'local_cpu':'Intel Core i9-14900HX','threads':4,
        'asset_bytes':sum(p.stat().st_size for p in cp.iterdir() if p.is_file())}
Path(f'runs/{args.split}-v3-resources.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
