"""Download the pinned official course data and verify every manifest hash."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATA=Path(__file__).resolve().parent/'data'
COMMIT='bb801f4f2e7629d270df765ab12c333b2dc616bb'
manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
expected=manifest['sha256']
def valid(name,digest):
    p=DATA/name
    return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
missing=[name for name,digest in expected.items() if not valid(name,digest)]
if missing:
    url=f'https://codeload.github.com/xudongwu-0/DASE7506-MP2/zip/{COMMIT}'
    print('Downloading pinned official course data...',flush=True)
    with urlopen(url,timeout=120) as response:
        archive=ZipFile(BytesIO(response.read()))
    prefix=f'DASE7506-MP2-{COMMIT}/code/data/'
    for name in missing:
        content=archive.read(prefix+name)
        if hashlib.sha256(content).hexdigest()!=expected[name]:
            raise ValueError(f'Official data hash mismatch: {name}')
        destination=DATA/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
assert all(valid(name,digest) for name,digest in expected.items())
print('All official data files present and SHA-256 verified.')
