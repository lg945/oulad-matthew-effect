import os, warnings
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
BASE = rREPO_ROOT

sub = pd.read_csv(os.path.join(BASE, 'data/bepkt/raw_data/submission.csv'))
sub['create_time'] = pd.to_datetime(sub['create_time'], utc=True)
prob = pd.read_csv(os.path.join(BASE, 'data/bepkt/raw_data/problem.csv'))
sub = sub.merge(prob[['id', 'difficulty']], left_on='problem_id', right_on='id', how='left', suffixes=('', '_prob'))
user = sub.groupby('user_id').agg(
    total_submissions=('id', 'count'),
    unique_problems_attempted=('problem_id', 'nunique'),
    accepted_problems=('problem_id', lambda x: x[sub.loc[x.index, 'result'] == 0].nunique()),
    accepted_count=('result', lambda x: (x == 0).sum())).reset_index()
fs = sub.sort_values(['user_id', 'create_time']).groupby('user_id').first().reset_index()
user['first_result'] = fs.set_index('user_id').loc[user['user_id'], 'result'].values
user['D_prior'] = (user['first_result'] == 0).astype(int)
user['acceptance_rate'] = user['accepted_count'] / user['total_submissions']
user['Y_low_completion'] = (user['accepted_problems'] < 5).astype(int)
user['log_submissions'] = np.log1p(user['total_submissions'])
user['log_problems'] = np.log1p(user['unique_problems_attempted'])
ad = sub.groupby('user_id')['create_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='unique_active_days')
user = user.merge(ad, on='user_id', how='left')
user['log_active_days'] = np.log1p(user['unique_active_days'])
for v in ['log_submissions', 'log_problems', 'log_active_days']:
    user[v + '_mc'] = user[v] - user[v].mean()

# ---- EXACT OLD-SCRIPT PSM ----
user['D_engagement'] = (user['total_submissions'] >= user['total_submissions'].median()).astype(int)
treated = user[user['D_engagement'] == 1]; control = user[user['D_engagement'] == 0]
scaler = StandardScaler(); X_psm = scaler.fit_transform(user[['D_prior', 'acceptance_rate']])
nn = NearestNeighbors(n_neighbors=1); nn.fit(X_psm[user['D_engagement'] == 0])
dist, indices = nn.kneighbors(X_psm[user['D_engagement'] == 1])
matched_control = control.iloc[indices.flatten()]
att = treated['Y_low_completion'].mean() - matched_control['Y_low_completion'].mean()
print('=== EXACT OLD-SCRIPT PSM (median split, [D_prior, acceptance_rate], 1:1 NN w/replacement, NO caliper) ===')
print(f'  N_treated={len(treated)} N_control(matched, w/replacement)={len(matched_control)}')
print(f'  unique control units matched = {matched_control.index.nunique()}')
print(f'  Treated rate={treated["Y_low_completion"].mean():.4f}  Matched control rate={matched_control["Y_low_completion"].mean():.4f}')
print(f'  ATT={att:.4f}')
print()

def run_psm(covariates, caliper, label):
    sc = StandardScaler()
    Xtr = sc.fit_transform(treated[covariates])
    Xco = sc.transform(control[covariates])
    n = NearestNeighbors(n_neighbors=1); n.fit(Xco)
    d, idx = n.kneighbors(Xtr)
    if caliper:
        keep = d.flatten() <= 0.25
        mt = treated[keep]; mc = control.iloc[idx.flatten()[keep]]
    else:
        mt = treated; mc = control.iloc[idx.flatten()]
    a = mt['Y_low_completion'].mean() - mc['Y_low_completion'].mean()
    print(f'  [{label}] cov={covariates}\n      -> N_t={len(mt)} N_c={len(mc)} ATT={a:.4f} (T={mt["Y_low_completion"].mean():.3f} C={mc["Y_low_completion"].mean():.3f})')

print('=== ROBUSTNESS GRID (median split) ===')
for covs in [['D_prior', 'acceptance_rate'],
             ['D_prior', 'acceptance_rate', 'log_problems_mc', 'log_active_days_mc'],
             ['D_prior', 'acceptance_rate', 'log_problems', 'log_active_days'],
             ['log_problems_mc', 'log_active_days_mc']]:
    run_psm(covs, False, 'no-caliper')
    run_psm(covs, True, 'caliper.25')
print()

print('=== SUBGROUP ATT by D_prior (exact old-script matching) ===')
for d in [0, 1]:
    st = treated[treated['D_prior'] == d]; sm_c = matched_control[matched_control['D_prior'] == d]
    if len(st) > 5 and len(sm_c) > 5:
        a = st['Y_low_completion'].mean() - sm_c['Y_low_completion'].mean()
        print(f'  D_prior={d}: ATT={a:.4f} (T={st["Y_low_completion"].mean():.3f} C={sm_c["Y_low_completion"].mean():.3f} N_t={len(st)} N_c={len(sm_c)})')
