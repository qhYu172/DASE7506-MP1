"""Select complementary residual fusion exclusively on validation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,hashlib,datetime
from pathlib import Path
import joblib
import numpy as np
from common import load_inputs
from answer_ranker import build_groups
from train_answer_ranker import targets
from boost_v8 import features
from neural_v6 import transform,logits

out=Path('runs/model_bundle_v9');out.mkdir(exist_ok=True)
tree=joblib.load('runs/experiment_v8/model.joblib');nn=joblib.load('runs/experiment_v6/model.joblib')
b,a=load_inputs('validation');x,m,g=build_groups(b,a,'numeric');y=targets(b,g,'validation')
f=features(x,m);z=transform(x,m,nn['mean'],nn['scale'])
t=np.mean([md.predict(f.reshape(-1,f.shape[-1]),num_threads=4).reshape(m.shape) for md in tree['models']],axis=0)
n=np.mean([logits(z,md) for md in nn['models']],axis=0)
base=x@tree['linear'];best=None;records=[]
for tw in (0.,.25,.5,.75,1.):
 for nw in (0.,.25,.5,.75,1.):
  s=base+tw*t+nw*n;s[~m]=-np.inf;p=s.argmax(1);hard=m.sum(1)>1
  row=dict(tree_weight=tw,neural_weight=nw,validation_correct=int((p==y).sum()),validation_hard_correct=int((p[hard]==y[hard]).sum()))
  records.append(row)
  # Prefer fewer residual components if validation accuracies tie.
  key=(row['validation_correct'],-int(tw>0)-int(nw>0),-tw-nw)
  if best is None or key>best[0]:best=(key,row)
selected=best[1];print(json.dumps(selected),flush=True)
joblib.dump(dict(linear=tree['linear'],tree_weight=selected['tree_weight'],neural_weight=selected['neural_weight'],trees=tree['models'] if selected['tree_weight'] else [],networks=nn['models'] if selected['neural_weight'] else [],mean=nn['mean'],scale=nn['scale']),out/'model.joblib')
(out/'config.json').write_text(json.dumps(dict(selected=selected,validation_n=len(y),validation_hard_n=int(hard.sum()),experiments=records,selection='validation only; before test evaluation'),indent=2))
sources=['final_v9.py','boost_v8.py','neural_v6.py','answer_ranker.py','features.py','train_v4.py','train_v5.py','train_v6.py','train_v7.py','train_v8.py','select_v9.py']
files=sources+[str(out/'model.joblib'),str(out/'config.json')]
freeze=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},inference_asset_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),test_access='Not accessed by training or selection scripts')
Path('runs/v9-freeze.json').write_text(json.dumps(freeze,indent=2))
