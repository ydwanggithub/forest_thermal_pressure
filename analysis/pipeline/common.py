"""Versioned long-record extension of the protected-forest analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/longterm'
for name in ['', 'remote_cache', 'models', 'source_data', 'qa', 'figures', 'editable']:
    (OUT / name).mkdir(parents=True, exist_ok=True)
YEARS = list(range(2003, 2026))
PML_YEARS = list(range(2003, 2024))
BANDS = ['boreal', 'temperate', 'tropical']
SEED = 20260922

def save(d, name):
    d = d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
    d.to_csv(OUT / name, index=False)
    return d

def dump(x, name):
    (OUT / name).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def master():
    m = pd.read_csv(ROOT / 'data/tables/master_delta_5km_v2.csv')
    m = m[m.biome.isin(BANDS)].copy().reset_index(drop=True)
    m['window_id'] = m.win_y.astype(str) + '_' + m.win_x.astype(str)
    assert not m.window_id.duplicated().any()
    return m
