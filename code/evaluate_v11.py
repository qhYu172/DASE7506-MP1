"""Frozen v11 official evaluation; no model selection here."""
import os,json,hashlib,time,sys,subprocess
from pathlib import Path

root=Path('runs/experiment_v11')
freeze=json.loads((root/'freeze.json').read_text())
for name,digest in freeze['sha256'].items():
    assert hashlib.sha256(Path(name.replace('\\','/')).read_bytes()).hexdigest()==digest,name
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
start=time.perf_counter()
for split in ('validation','test'):
    t=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--method','repair_v11.py','--checkpoint',str(root),'--split',split,'--output',f'runs/{split}-v11-predictions.json'],env=env,check=True)
    subprocess.run([sys.executable,'score_predictions.py','--split',split,'--predictions',f'runs/{split}-v11-predictions.json','--output',f'runs/{split}-v11-score.json'],env=env,check=True)
    resources=dict(fresh_process_total_seconds=time.perf_counter()-t,inference_asset_bytes=(root/'model.npz').stat().st_size,threads=4,hardware='Intel Core i9-14900HX; reference Xeon not measured')
    Path(f'runs/{split}-v11-resources.json').write_text(json.dumps(resources,indent=2))
    print(json.dumps(resources))
