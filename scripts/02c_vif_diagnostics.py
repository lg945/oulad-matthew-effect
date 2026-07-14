"""
脚本17b：修正多重共线性后的多路径模型
1. 去掉 total_clicks_log（与 vle_diversity r=0.94，VIF=43）
2. 对连续变量做均值中心化（降低交互项VIF）
3. 重新跑多路径模型
"""
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['MPLCONFIGDIR'] = os.path.join(REPO_ROOT, ".matplotlib")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = os.path.join(REPO_ROOT, "data")
ART = os.path.join(REPO_ROOT, "artifacts")

print("=" * 60)
print("Corrected Multi-path Model (VIF fix)")
print("=" * 60)

# ============================================================
# 1. Load and build (same as script 16/17)
# ============================================================
df_ccc = pd.read_csv(f"{DATA}/ccc_complete.csv", encoding='utf-8-sig')
vle = pd.read_parquet(f"{DATA}/vle.parquet")
ccc_vle = vle[vle['code_module'] == 'CCC'].copy()
ccc_vle_ids = set(ccc_vle['id_site'].unique())

sv_2014J = pd.read_parquet(f"{DATA}/studentVle_p2014J_.parquet")
sv_2014B = pd.read_parquet(f"{DATA}/studentVle_p2014B_.parquet")
sv_all = pd.concat([sv_2014J, sv_2014B], ignore_index=True)
ccc_ids = set(df_ccc['id_student'].unique())
sv_ccc = sv_all[sv_all['id_student'].isin(ccc_ids)].copy()
sv_ccc = sv_ccc[sv_ccc['id_site'].isin(ccc_vle_ids)].copy()
sv_ccc = sv_ccc.merge(ccc_vle[['id_site', 'activity_type']], on='id_site', how='left')

sa = pd.read_parquet(f"{DATA}/studentAssessment.parquet")
sa_ccc = sa[sa['id_student'].isin(ccc_ids)].copy()
asmt = pd.read_parquet(f"{DATA}/assessments.parquet")
asmt_ccc = asmt[asmt['code_module'] == 'CCC'].copy()

df = df_ccc.copy()
df['D_forumng'] = (df['forumng_clicks'] > 0).astype(int)
df['Y_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)
df['female'] = (df['gender'] == 'F').astype(int)
df['disability_bin'] = (df['disability'] == 'Y').astype(int)

imd_map = {'0-10%': 5, '10-20%': 15, '20-30%': 25, '30-40%': 35, '40-50%': 45,
           '50-60%': 55, '60-70%': 65, '70-80%': 75, '80-90%': 85, '90-100%': 95}
df['imd_num'] = df['imd_band'].map(imd_map)
age_map = {'0-35': 25, '35-55': 45, '55<=': 65}
df['age_mid'] = df['age_band'].map(age_map)
df['prev_attempts'] = df['num_of_prev_attempts']

df['edu_group'] = 'Other'
df.loc[df['highest_education'] == 'Lower Than A Level', 'edu_group'] = 'Low'
df.loc[df['highest_education'].isin(['A Level or Equivalent', 'HE Qualification']), 'edu_group'] = 'High'
df_sub = df[df['edu_group'].isin(['Low', 'High'])].copy()
df_sub['edu_high'] = (df_sub['edu_group'] == 'High').astype(int)

vle_div = sv_ccc.groupby('id_student')['activity_type'].nunique().reset_index()
vle_div.columns = ['id_student', 'vle_diversity']
df_sub = df_sub.merge(vle_div, on='id_student', how='left')
df_sub['vle_diversity'] = df_sub['vle_diversity'].fillna(0)

asmt_nonexam = asmt_ccc[asmt_ccc['assessment_type'] != 'Exam']
asmt_nonexam_count = asmt_nonexam.groupby('code_presentation')['id_assessment'].nunique().reset_index()
asmt_nonexam_count.columns = ['code_presentation', 'nonexam_assessments']
nonexam_map = asmt_nonexam_count.set_index('code_presentation')['nonexam_assessments'].to_dict()

sa_ccc_valid = sa_ccc[sa_ccc['date_submitted'] >= 0]
submitted = sa_ccc_valid.groupby('id_student')['id_assessment'].nunique().reset_index()
submitted.columns = ['id_student', 'submitted_assessments']
df_sub = df_sub.merge(submitted, on='id_student', how='left')
df_sub['submitted_assessments'] = df_sub['submitted_assessments'].fillna(0)
df_sub['total_nonexam'] = df_sub['code_presentation'].map(nonexam_map)
df_sub['assessment_ratio'] = df_sub['submitted_assessments'] / df_sub['total_nonexam']
df_sub['assessment_ratio'] = df_sub['assessment_ratio'].clip(0, 1)

study_days = sv_ccc.groupby('id_student')['date'].nunique().reset_index()
study_days.columns = ['id_student', 'study_days']
df_sub = df_sub.merge(study_days, on='id_student', how='left')
df_sub['study_days'] = df_sub['study_days'].fillna(0)
df_sub['imd_num'] = df_sub['imd_num'].fillna(df_sub['imd_num'].median())

# ============================================================
# 2. CORRECTION: Drop total_clicks_log, center continuous vars
# ============================================================
print("\n[1/3] Applying corrections:")
print("  - DROPPED total_clicks_log (r=0.94 with vle_diversity, VIF=43)")
print("  - Mean-centered continuous variables")

# Center continuous variables
for v in ['vle_diversity', 'assessment_ratio', 'study_days']:
    df_sub[f'{v}_c'] = df_sub[v] - df_sub[v].mean()

# Center edu_high (binary, but centering helps interaction VIF)
df_sub['edu_high_c'] = df_sub['edu_high'] - df_sub['edu_high'].mean()

# Corrected engage vars
engage_vars_corr = ['D_forumng', 'vle_diversity_c', 'assessment_ratio_c', 'study_days_c']
var_labels_corr = {
    'D_forumng': 'Forum participation',
    'vle_diversity_c': 'VLE diversity (centered)',
    'assessment_ratio_c': 'Assessment completion (centered)',
    'study_days_c': 'Study regularity (centered)',
}

# Interactions with centered edu
interact_vars_corr = []
for v in engage_vars_corr:
    iv = f"{v}_x_edu"
    df_sub[iv] = df_sub[v] * df_sub['edu_high_c']
    interact_vars_corr.append(iv)

# ============================================================
# 3. VIF check (corrected)
# ============================================================
print("\n[2/3] VIF after correction:")
covariates = ['age_mid', 'female', 'disability_bin', 'imd_num',
              'prev_attempts', 'studied_credits']
all_vars = covariates + engage_vars_corr + ['edu_high_c'] + interact_vars_corr
X_full = sm.add_constant(df_sub[all_vars])

vif_rows = []
for i, col in enumerate(X_full.columns):
    if col == 'const':
        continue
    vif_val = variance_inflation_factor(X_full.values, i)
    label = var_labels_corr.get(col, col)
    status = 'OK' if vif_val < 5 else 'Moderate' if vif_val < 10 else 'HIGH'
    vif_rows.append({'Variable': label, 'VIF': round(vif_val, 2), 'Status': status})
    print(f"  {label:<35s}: VIF = {vif_val:.2f} [{status}]")

pd.DataFrame(vif_rows).to_csv(f"{ART}/table_vif_corrected.csv", encoding='utf-8-sig', index=False)

# ============================================================
# 4. Corrected multi-path model
# ============================================================
print("\n[3/3] Corrected multi-path OLS:")
y = df_sub['Y_withdrawn']
X_multi = sm.add_constant(df_sub[all_vars])
ols_multi = sm.OLS(y, X_multi).fit(cov_type='HC1')

print(f"\n  R-sq = {ols_multi.rsquared:.4f}")
print(f"  N = {len(df_sub):,}")
print(f"\n  Key interaction effects:")

key_vars = interact_vars_corr + engage_vars_corr
for v in key_vars:
    label = var_labels_corr.get(v, v)
    beta = ols_multi.params[v]
    p = ols_multi.pvalues[v]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {label:<35s}: beta = {beta:.4f}, p = {p:.4f} [{sig}]")

# Save corrected results
rows = []
for var in ols_multi.params.index:
    if var == 'const':
        continue
    label = var_labels_corr.get(var, var)
    rows.append({
        'Variable': label,
        'Beta': round(ols_multi.params[var], 4),
        'SE': round(ols_multi.bse[var], 4),
        'p': round(ols_multi.pvalues[var], 4),
        'Sig': '***' if ols_multi.pvalues[var] < 0.001 else
               '**' if ols_multi.pvalues[var] < 0.01 else
               '*' if ols_multi.pvalues[var] < 0.05 else
               '' if ols_multi.pvalues[var] >= 0.1 else '.',
    })
pd.DataFrame(rows).to_csv(f"{ART}/table_multipath_ols_corrected.csv", encoding='utf-8-sig', index=False)

print(f"\n  Saved: table_vif_corrected.csv, table_multipath_ols_corrected.csv")
