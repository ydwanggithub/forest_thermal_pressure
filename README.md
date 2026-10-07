# Protected forest canopy and surface temperature near cities

This repository accompanies the manuscript

> **Canopy-temperature associations in protected forests near cities vary with climate and season**
>
> Yuandong Wang, Hengkai Li, Jiahui Zhou, Yuyao Wang and Lihong Meng

submitted to *Agricultural and Forest Meteorology*. It contains the derived data, the analysis and
extraction code, the result tables and the figure source data of the study.

## Study design

The study compares protected forest within 5 km of a fixed urban boundary with protected forest beyond
5 km in 0.25 degree windows, on a common 0.01 degree grid. For each window and year from 2003 to 2025 it
forms near-minus-reference contrasts of MODIS leaf area index (MOD15A2H), modelled evapotranspiration
(MOD16A2GF, with PML-V2 as an alternative), elevation-adjusted day and night land surface temperature
(MYD11A2) and GAIA impervious-surface urban pressure, together with ERA5-Land climate. Product quality and
valid ranges are evaluated at native resolution before transfer to the common grid. The analyses comprise

1. long-term and annual spatial models: linked regressions of the canopy, evapotranspiration and thermal
   contrasts with spatial block bootstrap intervals, and XGBoost and random forest models evaluated with
   cross-validation grouped by 5 degree blocks;
2. within-window models with window and year fixed effects and two-way clustered covariance;
3. alternative annual constructions and composite-date reconstructions that trace how contributing cells,
   quarter weights and observation dates enter the annual means;
4. calendar-quarter models, MODIS albedo (MCD43A3), snow (MOD10A2) and ERA5-Land radiation and snow cover,
   and robustness checks for extreme canopy anomalies, Global Forest Change forest loss and the share of
   observation dates not shared between the two forest portions.

## Repository contents

```text
data/
  tables/master_delta_5km_v2.csv      window frame: window identifiers, centres and static contrasts
  geometry/frozen_zones.geojson       frozen near and reference forest cell unions of the windows
  longterm/                           annual and long-term window tables of the original pipeline,
                                      ERA5-Land climate and PML-V2 annual tables
analysis/
  frame/                              construction of the zones and window pairing (needs the global
                                      zone and urban rasters, which are not redistributed)
  pipeline/                           shared modules: panel preparation, statistical models,
                                      machine learning and response summaries, annual extraction
  workflow/                           extraction, preparation and analysis scripts, run in the order
                                      below, with their inputs and outputs:
    native_support/                   native-resolution composite support and frozen 5-10 km ring geometry
    native_analysis/                  annual panels, model estimates, validation, fitted models
                                      (models/), quality checks (qa/) and analysis outputs (source_data/)
    native_sensitivity/               added-covariate and observation-support models
    native_decomposition/             cell-level annual constructions, quarterly models, urban-pressure
                                      interactions and Supplementary Tables S7 to S11 (table_S*.csv)
    native_dates/                     composite-date portion means and date-matching models
    native_process/                   quarterly albedo, snow, snow-flag-free LAI and radiation
    native_robustness/                annual forest loss and the robustness models
figures/
  Fig01.png ... Fig09.png, Graphical_Abstract.png
  source_data/                        the values displayed in each figure, named by figure
unpack_data.py                        restores the tables shipped as .csv.gz
tests/verify_release.py               reruns the workflow analyses and compares the results
```

The file names in `analysis/workflow/native_analysis/source_data/` follow the analysis stages; the figure
numbers of the article are used only in `figures/source_data/`. Figure rendering code is outside the
scope of this release.

## Reproducing the results

Create the environment and restore the compressed tables:

```bash
conda env create -f environment.yml
conda activate forest-thermal-pressure
python unpack_data.py
```

`tests/verify_release.py` checks every file against `MANIFEST_SHA256.json`, rebuilds the prepared annual
tables and reruns the observation-construction, observation-date, urban-pressure, albedo and snow, and
robustness analyses in a temporary copy, and compares each rebuilt result table with the released one:

```bash
python tests/verify_release.py
```

The individual analyses can also be run from the repository root in this order:

```bash
python analysis/workflow/prepare_native_analysis.py
python analysis/workflow/run_native_analysis.py statistics
python analysis/workflow/run_native_analysis.py prediction
python analysis/workflow/analyze_native_additional_controls.py
python analysis/workflow/analyze_native_observation_support.py
python analysis/workflow/analyze_native_decomposition.py
python analysis/workflow/analyze_native_dates.py
python analysis/workflow/analyze_unshared_dates.py
python analysis/workflow/prepare_supplementary_tables.py
python analysis/workflow/analyze_urban_modulation.py
python analysis/workflow/analyze_process_evidence.py
python analysis/workflow/analyze_robustness_checks.py
python analysis/workflow/prepare_supplementary_tables.py
```

`prepare_supplementary_tables.py` runs twice because Supplementary Table S8 is an input of the two
following analyses, whose estimates then enter Supplementary Tables S9 to S11. The `prediction` stage
refits the XGBoost and random forest models and takes several hours.

## Earth Engine extractions

The `fetch_*.py` scripts document the Earth Engine extractions and need an Earth Engine project of your
own:

```bash
set EE_PROJECT=your-earth-engine-project
python analysis/workflow/fetch_native_observation_support.py
```

The extracted tables are included, so the extractions are not needed to reproduce the analyses. A new
extraction can differ slightly if the public collections have changed.

## Source data

The source products are MODIS MOD15A2H, MYD11A2, MOD16A2GF, MCD43A3 and MOD10A2 (Collection 6.1), PML-V2
v0.1.8, ERA5-Land monthly aggregates, GAIA impervious surface, Global Urban Boundaries 2018, Copernicus
Global Land Cover CGLS-LC100 Collection 3, SRTMGL1, Global Forest Change v1.13 and the World Database on
Protected Areas. Access and licensing remain with the respective providers. The geometry in
`data/geometry/` consists of 0.01 degree cell unions derived from these layers, not of original
protected-area boundaries. The monthly release of the protected-area database used to build the frame
was not recorded; the frozen geometry fixes the analysed locations.

## License

The code is released under the MIT License (see `LICENSE`).
