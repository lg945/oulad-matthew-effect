# Replication Materials: Engagement, Educational Background and Learning Outcomes in Online Programming Education

This repository is a self-contained replication package for a quantitative study of
how online engagement relates to learning outcomes in programming education, and
how those relationships differ by learners' educational background. The scripts
below regenerate the descriptive statistics, regression and interaction models,
robustness checks, mediation analysis, and cross-dataset analysis reported in the
accompanying manuscript.

## Contents

```
data/             processed analysis tables (OULAD CCC sample; BePKT per-user features and problem metadata)
figures/          static PNG outputs of all figures plus the graphical abstract
scripts/          analysis scripts, numbered in execution order
requirements.txt  Python dependencies
```

## Data sources

- **OULAD** ("Learn to Code for Data Analysis" module, CCC subset). Publicly
  available under CC-BY 4.0 at https://analyse.kmi.open.ac.uk/open_dataset.
  Download the eight `.parquet` tables into `data/raw/` before running the
  Stage-4 scripts (`05_blocked_analyses.py`). The raw OULAD tables are not
  redistributed here.
- **BePKT** online-judge platform. Two files are included:
  `data/bepkt/raw_data/problem.csv` (problem metadata) and
  `data/bepkt/bepkt_user_features.csv`, a compact per-user feature table
  (N = 906) covering submission volume, problem breadth, practice regularity and
  the early-standing indicator. The raw submission log (~203 MB) and the
  behavioural activity log (~82 MB) are not redistributed here; they can be
  obtained from the original BePKT dataset (Zhu et al., 2022). The cross-dataset
  scripts use the raw log when it is present and otherwise fall back to the
  derived table, which reproduces the same numbers; the behavioural check is
  skipped with a notice when `behavior.csv` is absent.
  The raw submission log is expected at
  `data/bepkt/raw_data/submission.csv` if you want to re-run the raw-data
  path; it is excluded from version control (see `.gitattributes`).
- **Processed OULAD CCC samples** used by most scripts: `data/ccc_final.csv`
  and `data/ccc_complete.csv`.

## Scripts

| Script | Purpose |
|--------|---------|
| 00_clean_data.py | Builds the OULAD CCC analysis sample |
| 00b_build_bepkt_features.py | Builds the per-user BePKT feature table from the raw submission log (needed only if you have the raw log) |
| 01_descriptive.py | Descriptive statistics and group comparisons |
| 02_main_ols_models.py | OLS interaction models: participation threshold, participation intensity, time windows, and the multi-pathway model (withdrawal and achievement outcomes) |
| 02b_engagement_profiles.py | Multi-dimensional engagement profiles by education group |
| 02c_vif_diagnostics.py | Collinearity (VIF) diagnostics |
| 03_psm.py | Propensity-score matching and per-subgroup ATT estimates |
| 03b_psm_verification.py | Independent re-computation of the matching results |
| 04_mediation.py | Bootstrap mediation of the education–outcome association |
| 05_blocked_analyses.py | Sensitivity analyses: time-ordered survival models, conditional mediation, PSM bootstrap |
| 06_bepkt_replication.py | Cross-dataset analysis on the BePKT sample |
| 06b_verify_bepkt.py | Independent re-computation of the cross-dataset results |
| 06c_bepkt_psm.py | BePKT propensity-score audit |

## Figures

Static PNG outputs of the figures in the accompanying manuscript are provided in
`figures/` for viewing without re-running the code. Filenames are descriptive:

- `fig_theory_multpath.png` — conceptual framework
- `fig4_workflow_v2.png` — analytical workflow
- `fig2_descriptive_v2.png` — descriptive panels
- `fig5_multidimensional_engagement.png` — engagement profiles by group
- `fig9_assessment_patterns.png` — assessment-submission patterns
- `fig1_matthew_effect_v2.png` — outcome gaps by group and participation
- `fig6_forum_trajectory.png` — weekly discussion-activity trajectories
- `fig10_correlation_heatmap.png` — correlations among engagement variables
- `fig8_mediation_pathway.png` — mediation path diagram
- `fig3a_psm_balance_v2.png`, `fig3b_common_support_v2.png` — matching diagnostics
- `fig7_interaction_comparison.png` — interaction coefficients across models
- `fig11a_bepkt_matthew_effect.png`, `fig11b_bepkt_interaction_comparison.png`,
  `fig11c_bepkt_coefficients.png` — cross-dataset results
- `fig_graphical_abstract.png` — graphical abstract

Figure-generation scripts are not included; the figures are provided as static
outputs of the analyses above.

## Run

```
pip install -r requirements.txt
python scripts/00_clean_data.py
python scripts/01_descriptive.py
python scripts/02_main_ols_models.py
# ... (see the script table above)
```

All paths are resolved relative to the repository root; no absolute paths are
required. Intermediate outputs are written to `artifacts/`.

## License

Analysis code: MIT. Data: OULAD is distributed under CC-BY 4.0; the BePKT data
are subject to the terms of the original dataset (Zhu et al., 2022).
