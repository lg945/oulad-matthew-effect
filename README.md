# Replication Materials: Multi-Pathway Matthew Effect in Online Programming Education

This repository contains the analysis scripts and processed data needed to
reproduce the results reported in the manuscript *"Educational Background and
the Multi-Pathway Matthew Effect Associated with Disengagement in Online Programming
Education"* (submitted to *Education and Information Technologies*).

## Data sources

- **OULAD** ("Learn to Code for Data Analysis" module, CCC subset). Publicly
  available under CC-BY 4.0 at https://analyse.kmi.open.ac.uk/open_dataset.
  Download the eight `.parquet` tables into `data/raw/` before running the
  Stage-4 scripts (`05_blocked_analyses.py`).
- **BePKT** online-judge replication sample. Processed files are provided under
  `data/bepkt/raw_data/` (submission.csv, problem.csv).
- Processed OULAD CCC samples used by most scripts: `data/ccc_final.csv` and
  `data/ccc_complete.csv`.

## Script order

| Script | Produces |
|--------|----------|
| 00_clean_data.py | OULAD CCC cleaning |
| 01_descriptive.py | Table 1 + descriptive figures |
| 02_main_ols_models.py | OLS interaction / continuous / time-window / multi-pathway models (Tables 2-5, 8-11a) |
| 02b_engagement_profiles.py | Multi-dimensional engagement profiles (Fig. 4) |
| 02c_vif_diagnostics.py | Variance inflation diagnostics (Table 8) |
| 03_psm.py | Propensity-score matching (Tables 6, 6b, 7) |
| 03b_psm_verification.py | Independent PSM re-computation |
| 04_mediation.py | Bootstrap mediation (Table 5b) |
| 05_blocked_analyses.py | Stage-4 sensitivity (time-ordered, conditional mediation, PSM bootstrap) |
| 06_bepkt_replication.py | BePKT cross-dataset replication (Tables 11-14) |
| 06b_verify_bepkt.py | Independent BePKT re-computation |
| 06c_bepkt_psm.py | BePKT PSM audit |

## Figures

The 15 numbered manuscript figures plus the graphical abstract are provided as
static PNG outputs in `figures/` for convenient viewing without re-running the
code:

- `fig_theory_multpath.png` — Fig. 1 conceptual framework
- `fig4_workflow_v2.png` — Fig. 2 analytical workflow
- `fig2_descriptive_v2.png` — Fig. 3 descriptive panels
- `fig5_multidimensional_engagement.png` — Fig. 4 engagement profiles
- `fig9_assessment_patterns.png` — Fig. 5 assessment patterns
- `fig1_matthew_effect_v2.png` — Fig. 6 Matthew-effect plot
- `fig6_forum_trajectory.png` — Fig. 7 forum trajectory
- `fig10_correlation_heatmap.png` — Fig. 8 correlation heatmap
- `fig8_mediation_pathway.png` — Fig. 9 mediation pathway
- `fig3a_psm_balance_v2.png` / `fig3b_common_support_v2.png` — Fig. 10a/b PSM diagnostics
- `fig7_interaction_comparison.png` — Fig. 11 interaction comparison
- `fig11c_bepkt_coefficients.png` / `fig11a_bepkt_matthew_effect.png` / `fig11b_bepkt_interaction_comparison.png` — Fig. 12a-c BePKT results

These are generated from the analysis outputs by the figure scripts (not bundled
here); the data and analysis scripts above are sufficient to reproduce every
reported statistic.

## Run

```
pip install -r requirements.txt
python scripts/00_clean_data.py
python scripts/01_descriptive.py
python scripts/02_main_ols_models.py
# ... (see table above)
```

All paths are resolved relative to the repository root; no absolute paths are
required. Intermediate outputs are written to `artifacts/`.

## License

Analysis code: MIT. Data: see the respective sources above (OULAD CC-BY 4.0).
