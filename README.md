# Sparse-input audiogram completion: accuracy, triage performance and subgroup audit (NHANES)

This folder holds the code and outputs behind the manuscript in `manuscript/`. It replaces the analysis in `4_code/hearingproject_final-3.ipynb`, which was rejected by Scientific Reports. All analyses were rebuilt from scratch.

## How to reproduce

```bash
pip install pandas numpy scipy scikit-learn xgboost statsmodels matplotlib requests
python -I code/00_download.py         # downloads 24 NHANES .xpt files into data_raw/ (about 70 MB)
python -I code/01_build_cohorts.py    # writes data_derived/cohorts.csv and cohort_flow.csv
python -I code/02_models_and_audit.py # writes outputs/*.csv and outputs/results.json (about 15-25 min on a laptop CPU)
python -I code/02b_triage_extras.py   # AUC differences, matched specificity, group detection models
python -I code/03_figures.py          # writes figures/*.png and figures/*.pdf
python -I code/04_numbers_for_paper.py # writes manuscript/numbers.tex and manuscript/tables/*.tex
python -I code/05_make_submission_version.py # journal version (figures as separate files)
```

Fixed seed: 2026. Package versions used are recorded in `outputs/results.json` under `versions`.

## Cohorts

| Cohort | NHANES cycles | Ages | Role |
|---|---|---|---|
| Development | 2011–2012, 2015–2016 | 20–69 | Model fitting, 5-fold cross-validation |
| External A | 1999–2000, 2001–2002, 2003–2004 | 20–69 | Earlier-era external test |
| External B | 2005–2006, 2009–2010, 2017–2018 | 70+ | Older, hearing-loss-enriched external test |

NHANES tested adults aged 20–69 only in the cycles listed under Development and External A. Adults aged 70+ were tested in 2005–06, 2009–10 and 2017–18. 2007–08 tested young people only, and no 2013–14 audiometry file was released.

## Key definitions

- **Inputs:** measured 2 and 4 kHz thresholds in both ears, age and sex.
- **Targets:** 0.5, 1, 3, 6 and 8 kHz thresholds in both ears.
- **Threshold codes:** 666 (no response) is set to 5 dB above the highest valid level at that frequency. 888 (could not obtain) is treated as missing. No upper cap is applied to valid values. The first 1 kHz measurement is used, and the 1 kHz retest is kept to estimate test–retest variability.
- **Hearing loss:** WHO 2021 grading. This uses the better-ear four-frequency average (0.5, 1, 2 and 4 kHz): ≥20 dB HL counts as any hearing loss and ≥35 dB HL as moderate or worse.
- **Models:** Copy (nearest measured frequency), Age-sex norm (median audiogram by age decade and sex), Ridge (RidgeCV), and XGBoost (`xgboost.XGBRegressor`, tuned by nested CV).

## Mapping from manuscript to outputs

| Manuscript item | File |
|---|---|
| Figure 1 (cohort flow) | `figures/fig1_cohort_flow.*` ← `data_derived/cohort_flow.csv` |
| Table 1 (descriptives) | `outputs/descriptives.csv` |
| Figure 2, Table 2 (accuracy by frequency) | `outputs/mae_by_freq.csv`, `outputs/mae_by_target.csv` |
| Figure 3 (accuracy by WHO grade) | `outputs/mae_by_severity.csv` |
| Table 3 (triage) | `outputs/triage_overall.csv` |
| Figure 4 (triage by group) | `outputs/triage_by_group.csv` |
| Table 4, Figure 5 (subgroup audit) | `outputs/group_audit_xgb.csv`, `outputs/conditional_group_effects.csv`, `outputs/binned_error_calibration.csv` |
| Sensitivity analyses, survey-weighted results, SHAP | `outputs/results.json`, `outputs/shap_mean_abs.csv` |
| Per-participant predictions | `outputs/predictions.csv` |

## Data

The data are public-use NHANES files from the US National Center for Health Statistics (https://wwwn.cdc.gov/nchs/nhanes/). They are not redistributed here; `code/00_download.py` fetches them.
