"""Validation comparison of three predeclared choices: v2, group ranker, equal fusion."""
import json
from pathlib import Path
import shutil
import numpy as np
from common import load_inputs
from train_answer_ranker import targets
from fusion_ranker import components

bank,arrays=load_inputs('validation')
p,q,groups=components(bank,arrays,'runs/model_bundle_v2','runs/augmented_ranker')
y=targets(bank,groups,'validation');hard=np.array([len(g)>1 for g in groups])
records=[]
for w in (0,.5,1):
    pick=((1-w)*p+w*q).argmax(axis=1)
    r={'group_weight':w,'correct':int((pick==y).sum()),'n':len(y),
       'hard_correct':int((pick[hard]==y[hard]).sum()),'hard_n':int(hard.sum())}
    records.append(r);print(json.dumps(r),flush=True)
best=max(records,key=lambda r:r['hard_correct'])
out=Path('runs/fusion_v3');out.mkdir(parents=True,exist_ok=True)
shutil.copytree('runs/model_bundle_v2',out/'v2',dirs_exist_ok=True)
(out/'group').mkdir(exist_ok=True)
shutil.copy2('runs/augmented_ranker/ranker.npz',out/'group'/'ranker.npz')
(out/'config.json').write_text(json.dumps({'group_weight':best['group_weight'],
    'selection':'validation hard accuracy; three predeclared options','comparisons':records},indent=2))
print('selected',best,flush=True)
