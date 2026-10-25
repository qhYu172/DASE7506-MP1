"""Contract checks on validation inputs only; no reference labels needed."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from common import load_inputs, parse_answer
from answer_ranker import predict
from submission import predict as old_predict

bank,arrays=load_inputs('validation')
cp=Path('runs/augmented_ranker')
small=bank[:40];ea={'embeddings':arrays['embeddings'][:40]}
chosen=predict(small,ea,cp)
assert predict([],{'embeddings':np.zeros((0,4,1024))},cp)==[]
assert all(type(j) is int and 0<=j<4 for j in chosen)
renamed=deepcopy(small)
for i,row in enumerate(renamed):row['id']=f'unseen-audit-{i}'
assert chosen==predict(renamed,ea,cp)
permutation=np.array([2,0,3,1]);permuted=deepcopy(small)
for row in permuted:
    row['candidates']=[row['candidates'][j] for j in permutation]
    for j,c in enumerate(row['candidates']):c['index']=j
other=predict(permuted,{'embeddings':ea['embeddings'][:,permutation]},cp)
for row,j,prow,k in zip(small,chosen,permuted,other):
    assert parse_answer(row['candidates'][j]['text'])==parse_answer(prow['candidates'][k]['text'])
for start in range(0,40,10):
    assert chosen[start:start+10]==predict(small[start:start+10],{'embeddings':ea['embeddings'][start:start+10]},cp)
saved=json.loads(Path('runs/validation-v2-predictions.json').read_text())['selected']
assert old_predict(bank,arrays,Path('runs/model_bundle_v2'))==saved
result={'empty_bank':True,'valid_output':True,'unseen_ids':True,'candidate_permutation':True,
        'batch_independence':True,'v2_refactor_all_validation_choices_unchanged':True}
Path('runs/augmented_ranker/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
