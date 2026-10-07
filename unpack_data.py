"""Restore the tables shipped as .csv.gz (next to the compressed copies) and check their SHA-256."""
import gzip, hashlib, json, shutil
from pathlib import Path
R = Path(__file__).resolve().parent
expected = json.loads((R / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))['unpacked']
for rel, digest in sorted(expected.items()):
    out = R / rel; gz = out.with_name(out.name + '.gz')
    if not out.exists():
        with gzip.open(gz, 'rb') as f, out.open('wb') as g:
            shutil.copyfileobj(f, g, 1 << 20)
    h = hashlib.sha256()
    with out.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    assert h.hexdigest() == digest, rel
    print('ok', rel)
