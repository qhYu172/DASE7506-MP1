"""Synthetic repair and validation-input invariance checks; no test references."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import copy,json
from pathlib import Path
import numpy as np
from common import load_inputs,parse_answer
from repair_v11 import predict,trace

assert trace('2 * 3 = 5\n5 * 4 = 20')[-1][2]==24
assert trace('2 * 3 = 6\n6 * 4 = 24')[-1][2]==24
assert not trace('2x + 3 = 9')
bank,arrays=load_inputs('validation');bank=bank[:64];arrays={'embeddings':arrays['embeddings'][:64]}
checkpoint=Path('runs/experiment_v11');p=predict(bank,arrays,checkpoint)
permutation=[2,0,3,1];shuffled=copy.deepcopy(bank)
for row in shuffled:
    row['candidates']=[row['candidates'][j] for j in permutation]
    for j,c in enumerate(row['candidates']):c['index']=j
q=predict(shuffled,{'embeddings':arrays['embeddings'][:,permutation]},checkpoint)
answer=lambda b,p:[parse_answer(row['candidates'][j]['text']) for row,j in zip(b,p)]
assert answer(bank,p)==answer(shuffled,q)
parts=[]
for start in range(0,64,16):parts+=predict(bank[start:start+16],{'embeddings':arrays['embeddings'][start:start+16]},checkpoint)
assert p==parts
result={'synthetic_arithmetic':'passed','candidate_permutation_64_validation_inputs':'passed','batch_invariance_64_validation_inputs':'passed','reference_answers_used':False}
Path('runs/experiment_v11/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
