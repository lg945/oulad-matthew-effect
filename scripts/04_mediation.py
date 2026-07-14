"""
Bootstrap mediation analysis for education -> forum participation (binary) -> withdrawal
Using ccc_final.csv (which has Y_withdrawn column)
"""
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = os.path.join(REPO_ROOT, "data")
ART_DIR = os.path.join(REPO_ROOT, "artifacts")
DATA_FILE = os.path.join(DATA_DIR, "ccc_final.csv")

# Load data
df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
print(f"Loaded data shape: {df.shape}")

# Build analysis dataset
# Outcome: withdrawal (we have Y_withdrawn column, but we'll recompute from final_result to be safe)
df['Y_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)

# Education high: Low = Lower Than A Level, High = A Level or Equivalent or HE Qualification
df['edu_group'] = 'Other'
df.loc[df['highest_education'] == 'Lower Than A Level', 'edu_group'] = 'Low'
df.loc[df['highest_education'].isin(['A Level or Equivalent', 'HE Qualification']), 'edu_group'] = 'High'
df_sub = df[df['edu_group'].isin(['Low', 'High'])].copy()
df_sub['edu_high'] = (df_sub['edu_group'] == 'High').astype(int)

# Binary forum participation (already 0/1)
df_sub['D_forumng'] = (df_sub['forumng_clicks'] > 0).astype(int)
# Optional: continuous forum (log)
df_sub['forum_clicks_log'] = np.log1p(df_sub['forumng_clicks'])

# Covariates
# demographics
df_sub['female'] = (df_sub['gender'] == 'F').astype(int)
# disability binary
df_sub['disability_bin'] = (df_sub['disability'] == 'Y').astype(int)
# IMD numeric
imd_map = {'0-10%': 5, '10-20%': 15, '20-30%': 25, '30-40%': 35, '40-50%': 45,
           '50-60%': 55, '60-70%': 65, '70-80%': 75, '80-90%': 85, '90-100%': 95}
df_sub['imd_num'] = df_sub['imd_band'].map(imd_map)
# age midpoint
age_map = {'0-35': 25, '35-55': 45, '55<=': 65}
df_sub['age_mid'] = df_sub['age_band'].map(age_map)
# previous attempts (already column)
df_sub['prev_attempts'] = df_sub['num_of_prev_attempts']
# studied credits (use standardized version as covariate)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_sub['studied_credits'] = sc.fit_transform(df_sub[['studied_credits']])
# Handle missing imd_num
df_sub['imd_num'] = df_sub['imd_num'].fillna(df_sub['imd_num'].median())

covariates = ['age_mid', 'female', 'disability_bin', 'imd_num',
              'prev_attempts', 'studied_credits']

print(f"Analysis sample size: {len(df_sub)}")
print(f"Education distribution: Low={ (df_sub['edu_high']==0).sum() }, High={ (df_sub['edu_high']==1).sum() }")
print(f"Forum participation mean: {df_sub['D_forumng'].mean():.3f}")
print(f"Withdrawal mean: {df_sub['Y_withdrawn'].mean():.3f}")

# Function to compute mediation effects given a dataframe
def compute_mediation_binary(data):
    # Step 1: edu -> forum participation (path a)
    X1 = sm.add_constant(data[['edu_high'] + covariates])
    M = data['D_forumng']
    model_m = sm.OLS(M, X1).fit()
    a = model_m.params['edu_high']
    se_a = model_m.bse['edu_high']
    
    # Step 2: forum + edu -> withdrawal (paths b and c')
    X2 = sm.add_constant(data[['D_forumng', 'edu_high'] + covariates])
    Y = data['Y_withdrawn']
    model_y = sm.OLS(Y, X2).fit()
    b = model_y.params['D_forumng']
    se_b = model_y.bse['D_forumng']
    c_prime = model_y.params['edu_high']
    
    # Step 3: edu -> withdrawal only (total effect c)
    X3 = sm.add_constant(data[['edu_high'] + covariates])
    model_y_total = sm.OLS(Y, X3).fit()
    c_total = model_y_total.params['edu_high']
    
    indirect = a * b
    # Sobel SE
    sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    sobel_z = indirect / sobel_se
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    # Proportion mediated
    prop_mediated = indirect / c_total if c_total != 0 else np.nan
    
    return {
        'a': a, 'se_a': se_a,
        'b': b, 'se_b': se_b,
        "c'": c_prime,
        'c_total': c_total,
        'indirect': indirect,
        'sobel_se': sobel_se,
        'sobel_z': sobel_z,
        'sobel_p': sobel_p,
        'prop_mediated': prop_mediated
    }

# Point estimates
point = compute_mediation_binary(df_sub)
print("\n=== Point estimates (binary mediator) ===")
for k, v in point.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

# Bootstrap
n_boot = 5000
boot_indirect = []
boot_prop = []
rng = np.random.default_rng(seed=42)
n = len(df_sub)

print(f"\nRunning {n_boot} bootstrap samples...")
for i in tqdm(range(n_boot)):
    # Resample with replacement
    sample_idx = rng.integers(0, n, n)
    df_boot = df_sub.iloc[sample_idx].copy()
    df_boot.reset_index(drop=True, inplace=True)
    try:
        res = compute_mediation_binary(df_boot)
        boot_indirect.append(res['indirect'])
        if not np.isnan(res['prop_mediated']):
            boot_prop.append(res['prop_mediated'])
    except Exception as e:
        # In case of numerical issues, skip
        continue

boot_indirect = np.array(boot_indirect)
boot_prop = np.array(boot_prop)

print(f"\nBootstrap completed. Valid indirect samples: {len(boot_indirect)}; proportion samples: {len(boot_prop)}")

# Percentile confidence intervals
def perc_ci(samples, alpha=0.05):
    if len(samples) == 0:
        return np.nan, np.nan
    lower = np.percentile(samples, 100 * alpha/2)
    upper = np.percentile(samples, 100 * (1 - alpha/2))
    return lower, upper

ci_indirect = perc_ci(boot_indirect)
ci_prop = perc_ci(boot_prop)

# Bootstrap p-value (proportion of samples where effect is zero or opposite sign)
def bootstrap_p(est, samples):
    if len(samples) == 0:
        return np.nan
    if est > 0:
        p = (np.sum(samples <= 0) * 2) / len(samples)
    else:
        p = (np.sum(samples >= 0) * 2) / len(samples)
    return min(p, 1.0)

p_indirect = bootstrap_p(point['indirect'], boot_indirect)
p_prop = bootstrap_p(point['prop_mediated'], boot_prop)

print("\n=== Bootstrap results ===")
print(f"Indirect effect: {point['indirect']:.4f}")
print(f"  95% CI: [{ci_indirect[0]:.4f}, {ci_indirect[1]:.4f}]")
print(f"  p-value: {p_indirect:.4f}")
print(f"Proportion mediated: {point['prop_mediated']:.4f}")
print(f"  95% CI: [{ci_prop[0]:.4f}, {ci_prop[1]:.4f}]")
print(f"  p-value: {p_prop:.4f}")

# Save results to CSV
out_df = pd.DataFrame({
    'parameter': ['indirect_effect', 'prop_mediated'],
    'estimate': [point['indirect'], point['prop_mediated']],
    'boot_ci_lower': [ci_indirect[0], ci_prop[0]],
    'boot_ci_upper': [ci_indirect[1], ci_prop[1]],
    'bootstrap_p': [p_indirect, p_prop],
    'sobel_p': [point['sobel_p'], np.nan]
})
out_path = os.path.join(ART_DIR, 'bootstrap_mediation_results.csv')
out_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

# Suggested text for manuscript
text = (
    f"The bootstrap-estimated indirect effect (education → forum participation → withdrawal) "
    f"was {point['indirect']:.3f} (95% CI [{ci_indirect[0]:.3f}, {ci_indirect[1]:.3f}], p={p_indirect:.3f}), "
    f"accounting for {point['prop_mediated']*100:.1f}% of the total effect "
    f"(95% CI [{ci_prop[0]*100:.1f}%, {ci_prop[1]*100:.1f}%], p={p_prop:.3f})."
)
print("\n=== Suggested manuscript text ===")
print(text)