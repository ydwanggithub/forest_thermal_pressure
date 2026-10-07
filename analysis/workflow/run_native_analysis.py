"""Run the existing analysis implementations against revised native-grid tables."""
from pathlib import Path
import sys, argparse, json, hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/pipeline'))
import common
common.OUT=HERE/'native_analysis'
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['statistics','prediction','display']);args=parser.parse_args()
name={'statistics':'analyze','prediction':'machine_learning','display':'prepare_display'}[args.stage]
import importlib
m=importlib.import_module(name)
m.OUT=common.OUT
if hasattr(m,'SRC'):m.SRC=common.OUT/'source_data'
if args.stage=='prediction':
    # These are the predictive comparisons actually used by the current paper.
    # Common-product comparisons in the paper use the separately refitted OLS
    # paths, not the historical optional XGBoost fits for those subsets.
    m.save(m.verify_algorithms(),'qa/response_algorithm_checks.csv')
    rows=m.run_spatial('longterm');m.save(rows,'validation.csv')
    rows+=m.run_annual();m.save(rows,'validation.csv')
    m.dump({'status':'complete','primary_predictors':m.FULL,
      'annual_predictors':[f for f in m.FULL if f!='UP_trend']+['year'],
      'spatial_cv':'GroupKFold with 5 degree block groups',
      'temporal_validation':'train 2003 to 2020; predict 2021 to 2025',
      'annual_window_leakage':False,'XGB_parameters':m.XGB,
      'reported_comparisons_only':['longterm','annual']},'qa/ml_completion.json')
else:m.main()
(common.OUT/'qa'/f'{args.stage}_implementation.json').write_text(json.dumps({'module':name,'original_source_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
    'versioned_output':str(common.OUT),'scope':'Existing implementation; native-resolution inputs substituted; original output directory untouched'},indent=2),encoding='utf-8')
