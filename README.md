# PRC 2026 taxi-out prediction

Two-level residual model for predicting the taxi-out time (`TAXITIME_SEC_mvt`) of departures at ten European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM). The final run scores the January, February, June and July 2026
departures of `final_ranking.parquet`.

This repository contains the code to train, cross-validate and predict, plus the trained artifacts that are small enough to
version (models, out-of-fold predictions, per-component predictions). Large tabular inputs (`*.parquet`) are not versioned.

## Approach

1. **Level 1, two base models** trained on all of 2025:
   - Base A: a chain of CatBoost and LightGBM models fitted on the residual against a schedule-based baseline
     (`prc/basea`).
   - Base B: an ensemble of LightGBM, CatBoost and an MLP defined by a YAML config, followed by a refit of the segment of
     LIRF departures without a matched flight plan (`prc/baseb`).
   - `v36 = 0.5 * A + 0.5 * B` (`prc/blend.py`).
2. **Level 2, residual layers** applied on top of `v36` (`prc/layers`, `prc/otimista`):
   rule and model layers for LIRF, LTFM and LEBL, and an ADS-B based corrector ("layer D") for EDDF, EDDM, EHAM, LSZH,
   LEBL, EGLL and LEMD, trained on neighbouring months of 2025 with a seasonal-distance weighting.
3. **Stack** (`prc/stack`): ridge/lasso combiners fitted per airport cell on out-of-fold 2025 predictions, a small
   population of neural combiners, and 15 members (base-A and layer deltas, five LightGBM variants, five sequence networks, an MLP).
   Safety rules fall back to the layer output where the stack correction is out of scale.
4. **Residual specialists** (`prc/espera_rot`): a LightGBM on aircraft rotation, stand-slot and historical stand profiles
   for non-LIRF airports, and a separate LIRF-only specialist.
5. **Post-processing rules** (`prc/layers/lirf_night_rule.py`, `prc/layers/lirf_eco_rule.py`): fixed rules for LIRF departures
   without an actual off-block time, each moving the model output a fixed fraction of the way to `MVT - SCHED`.
6. **Daily layer** (`prc/daily`): a per-airport, per-day posterior update that uses the published daily total of
   pre-departure delay as a constraint on the departures of that day, followed by a second pass for LIRF
   (`prc/daily/second_pass.py`).
7. **Regime layer** (`prc/regime`): LightGBM increment fitted on the out-of-fold residual from runway- and airport-level
   neighbourhood aggregates over 15 to 120 minute windows.

## Repository layout

```
prc/paths.py            all input/output locations (environment variables below)
prc/basea               base A: features, component fits, assembly, out-of-fold validation
prc/baseb               base B: trimmed model framework, YAML configs, build/holdout/refit CLI
prc/adsb                ADS-B surface extraction and per-flight feature extraction
prc/layers              level-2 layers, layer composition, LIRF rules
prc/otimista            layer D: contexts, leave-month-out validation, application, H_base members
prc/stack               member table, combiners, neural population, final stack application
prc/espera_rot          residual specialists (all airports, and LIRF only)
prc/daily               daily layer: parameter fit, application, constants, LIRF second pass
prc/regime              regime layer: neighbourhood features, fit, application
prc/blend.py            v36 blend of the two bases
prc/compose.py          assembly of the final pre-specialist prediction for the four months
prc/terms.py            builds delta terms between two versions of a component
prc/pipeline.py         stage definitions used by the three entry points
scripts/train.py        runs training stages
scripts/cv.py           runs validation stages
scripts/predict.py      runs prediction/composition stages
artifacts/              versioned models and out-of-fold predictions (see docs/ARTIFACTS.md)
docs/                   stage reference and artifact manifest
```

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or newer. CatBoost and LightGBM use CPU; the sequence networks and the MLP run on CPU, CUDA or Apple MPS
(`--device`).

## Data and locations

`prc/paths.py` reads these environment variables (defaults relative to the repository):

| Variable | Default | Content |
|---|---|---|
| `PRC_DATA_ROOT` | `data/` | `raw/` monthly training files, `final/final_ranking.parquet`, `final/final_submitting.parquet`, `processed/` |
| `PRC_RANKING`, `PRC_SUBMITTING` | `data/final/...` | ranking and submission template |
| `PRC_TRAINING_DIR` | `data/raw` | the twelve `training_2025-*.parquet` files |
| `PRC_SURFACE_DIR` | `data/processed/adsb_surface` | daily ADS-B surface extracts |
| `PRC_EUROCONTROL_DIR` | `data/external/aiu_csv` | `all_pre_departure_delays_2025.csv`, `all_pre_departure_delays_2026.csv` |
| `PRC_WORK` | `work/` | intermediates |
| `PRC_ARTIFACTS` | `artifacts/` | models, out-of-fold predictions, references |

The code reads labels only from the 2025 training files. The ranking files are read for features and identifiers.

## Running

All entry points accept `--dry-run`, which prints the commands without executing them.

```
python scripts/train.py features basea baseb blend layers otimista v53 stack espera daily regime
python scripts/cv.py basea baseb layers otimista stack espera
python scripts/predict.py compose espera daily second_pass regime
```

`python scripts/<entry>.py all` runs every stage of that entry point. Stage definitions and the exact commands are in
`docs/STAGES.md`. The final file is `artifacts/submission_regime.parquet`, with columns `MVT_ID_mvt` and `TAXITIME_SEC_mvt`.

## Months and composition

February and June 2026 have no 2025 hold-out of their own, so their layer-D models use the other eleven months of 2025
with seasonal-distance weights and a proxy prediction for the own month. January and July use the 2025 hold-out of the
corresponding month.

For January and July the prediction is assembled as a reference layer plus model deltas. The reference is a prediction file
for these rows (`artifacts/reference/`); the deltas are the differences between the retrained models and the
earlier models for the same rows (stack, ADS-B layer D, pool, LIRF physical layer, layer-D second-stage term). They are read
from precomputed files in `artifacts/stack_inputs/`, which `prc/terms.py` builds. February and June use the retrained model
chain only.

## Required inputs

- `data/raw/training_2025-*.parquet`, `data/final/final_ranking.parquet`, `data/final/final_submitting.parquet`
- `data/processed/adsb_surface/` daily ADS-B surface extracts
- `data/external/aiu_csv/all_pre_departure_delays_{2025,2026}.csv`
- `artifacts/reference/` and `artifacts/stack_inputs/` parquet files read by `predict.py compose`
- `artifacts/oof/daily/oof_2025.npz`, `artifacts/daily/params.json` and `artifacts/models/regime/` for the daily and regime layers (`train.py daily regime` regenerates them)

## In development

**Reconstructing the CTOT slot window with the airport graph as reference** (branch [`feature/in_dev`](https://github.com/vilelabruno/prc_taxiout_pred/tree/feature/in_dev/in_development); the code of this README's pipeline is unchanged).

Part of the taxi-out error is time spent at the stand that the clocks do not explain. A regulated flight holds at the stand until the slot issued by the Network
Manager opens and must take off inside a tolerance window of **−5/+10 min around its CTOT**; if it cannot make the window it loses the slot and receives the next free one.
We reconstruct that window per flight: `ETOT = EOBT + EXOT` with EXOT estimated from the taxiway graph (Dijkstra path, turns, crossings, simulated stand-to-runway movement),
a runway timeline from ADS-B and movements calibrated so that ≥ 90 % of real departures fall in an open window, a look-back scan from the MVT (`CTOT ∈ [MVT − 10, MVT + 5]` min) that counts
available, blocked and lost windows, and per-airport weights fitted against the stand hold. As a proxy of the regulation we replicate the dynamic indicators of the paper below.

**Paper:** Mas-Pujol, S., & Delgado, L. (2024). *Prediction of ATFM impact for individual flights: A machine learning approach.* Expert Systems with Applications, 252.
[doi/ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417424010121). Also: EUROCONTROL, *ATFCM Users Manual* ed. 22.1 (slot tolerance window, ETOT, TTOT/TSAT/EXOT).

**Simulated graph (EGLL, largest-error flight of the day, BAW105 on 2025-01-17).** Edges coloured by the block field of the other aircraft taxiing at that instant; blue is the
route simulated backwards from the MVT to the stand:

![Largest error on the simulated EGLL graph](https://github.com/vilelabruno/prc_taxiout_pred/blob/feature/in_dev/in_development/plots/rev_maior_erro.png?raw=true)

The same flight decomposed: `y` = 1971 s = (MVT − ADS-B anchor) 549 s + apron 62 s + simulated stand→anchor 11 s + a remaining wait of 1349 s that the graph movement does not explain:

![Anchor decomposition for BAW105](https://github.com/vilelabruno/prc_taxiout_pred/blob/feature/in_dev/in_development/plots/rev_ancora_BAW105.png?raw=true)

**A departure fitting a CTOT window (3D).** Real ADS-B climb-out of the same flight (colour = minutes since take-off) crossing a gate in mid-air that stands for the slot window `[CTOT − 5, CTOT + 10]` min; grey is neighbouring traffic:

![BAW105 crossing the CTOT window gate](https://github.com/vilelabruno/prc_taxiout_pred/blob/feature/in_dev/in_development/plots/ctot_gate_3d.png?raw=true)

**Early results** (out-of-fold, hold-out January + July 2025, residual of the champion): the CTOT-window features give −0.17 s [−0.27; −0.08] and the per-airport hold weights a further −0.20 s [−0.29; −0.10] in RMSE
over the earlier groups. The gains are small; the variants that did not help (sky-occupancy grid, sky proximity, tail weights and doses) are documented in
[`in_development/README.md`](https://github.com/vilelabruno/prc_taxiout_pred/blob/feature/in_dev/in_development/README.md).
