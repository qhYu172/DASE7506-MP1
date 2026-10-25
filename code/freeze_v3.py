"""Freeze the validation-selected answer ranker before any new test scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

source=Path('runs/augmented_ranker')
target=Path('runs/model_bundle_v3')
target.mkdir(parents=True,exist_ok=True)
shutil.copy2(source/'ranker.npz',target/'ranker.npz')
development=json.loads((source/'development.json').read_text())
config={'method':'numeric-answer-group-pairwise-subsets','version':3,'seed':7506,
        'selected':development['selected'],'validation_correct':746,'validation_n':798,
        'validation_hard_correct':303,'validation_hard_n':355,
        'source_ideas':'../../PUBLIC_METHODS.md','prediction_entrypoint':'submission.py'}
(target/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'method_selection':'Train-question CV and validation; before v3 test scoring',
          'prior_exposure':'V1 and V2 aggregate test scores already known; no per-question test error inspection',
          'assets':{p.name:digest(p) for p in target.iterdir() if p.is_file()},
          'source':{name:digest(Path(name)) for name in ['answer_ranker.py','features.py','submission.py',
              'train_answer_ranker.py','train_augmented_ranker.py','select_fusion.py']}}
Path('runs/v3-freeze.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'asset_bytes':sum(p.stat().st_size for p in target.iterdir()),'sha256':manifest['assets']}))
