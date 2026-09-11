"""Independent re-computation of BePKT replication numbers to verify the manuscript."""
import os, json, re, warnings
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')

BASE = REPO_ROOT

# ---- load like redraw script ----
RAW_LOG = os.path.join(BASE, 'data/bepkt/raw_data/submission.csv')
DERIVED = os.path.join(BASE, 'data/bepkt/bepkt_user_features.csv')
if os.path.exists(RAW_LOG):
    sub = pd.read_csv(RAW_LOG)
    sub['create_time'] = pd.to_datetime(sub['create_time'], utc=True)
    prob = pd.read_csv(os.path.join(BASE, 'data/bepkt/raw_data/problem.csv'))
    sub = sub.merge(prob[['id', 'difficulty']], left_on='problem_id', right_on='id', how='left', suffixes=('', '_prob'))

    user = sub.groupby('user_id').agg(
        total_submissions=('id', 'count'),
        first_sub_time=('create_time', 'min'),
        last_sub_time=('create_time', 'max'),
        unique_problems_attempted=('problem_id', 'nunique'),
        accepted_problems=('problem_id', lambda x: x[sub.loc[x.index, 'result'] == 0].nunique()),
        accepted_count=('result', lambda x: (x == 0).sum()),
        unique_active_days=('create_time', lambda x: x.dt.date.nunique()),
        mid_high_attempted=('difficulty', lambda x: (x.isin(['Mid', 'High'])).sum()),
    ).reset_index()

    first_sub = sub.sort_values(['user_id', 'create_time']).groupby('user_id').first().reset_index()
    user['first_result'] = first_sub.set_index('user_id').loc[user['user_id'], 'result'].values
    user['D_prior'] = (user['first_result'] == 0).astype(int)
    user['acceptance_rate'] = user['accepted_count'] / user['total_submissions']
    user['Y_low_completion'] = (user['accepted_problems'] < 5).astype(int)
elif os.path.exists(DERIVED):
    print('[info] raw submission log not present; loading precomputed per-user features from', DERIVED)
    user = pd.read_csv(DERIVED)
    for c in ['first_sub_time', 'last_sub_time']:
        user[c] = pd.to_datetime(user[c], utc=True, errors='coerce')
else:
    raise FileNotFoundError(
        'Neither the raw submission log nor the derived feature table was found; see README for data setup.')
user['log_submissions'] = np.log1p(user['total_submissions'])
user['log_problems'] = np.log1p(user['unique_problems_attempted'])
user['log_active_days'] = np.log1p(user['unique_active_days'])

print('='*70)
print('BASIC: N, D_prior, Table 11 descriptives')
print('='*70)
N = len(user)
print(f'N students = {N}')
hi = user['D_prior'].sum(); lo = N - hi
print(f'D_prior High = {hi} ({hi/N:.1%}), Low = {lo} ({lo/N:.1%})')
for g, lbl in [(0,'Low'),(1,'High')]:
    gg = user[user['D_prior']==g]
    print(f'  {lbl}: N={len(gg)} acc_rate={gg["acceptance_rate"].mean():.3f} subs={gg["total_submissions"].mean():.1f} '
          f'probs={gg["unique_problems_attempted"].mean():.1f} days={gg["unique_active_days"].mean():.1f} '
          f'lowcomp={gg["Y_low_completion"].mean():.1%}')

# ---- centered pathways ----
for var in ['log_submissions','log_problems','log_active_days']:
    user[f'{var}_mc'] = user[var] - user[var].mean()
user['D_x_submissions'] = user['D_prior'] * user['log_submissions_mc']
user['D_x_problems'] = user['D_prior'] * user['log_problems_mc']
user['D_x_days'] = user['D_prior'] * user['log_active_days_mc']

def ols_report(y, X_cols, label):
    X = sm.add_constant(user[X_cols])
    m = sm.OLS(user[y], X).fit(cov_type='HC1')
    print(f'\n--- {label} (R2={m.rsquared:.3f}) ---')
    for c in X_cols:
        print(f'  {c:28s} beta={m.params[c]: .4f} SE={m.bse[c]:.4f} p={m.pvalues[c]:.4f}')

print('\n'+'='*70)
print('TABLE 12: Individual Pathway Models')
print('='*70)
ols_report('Y_low_completion', ['D_prior','log_submissions_mc','D_x_submissions'], 'Model1 Submission Volume')
ols_report('Y_low_completion', ['D_prior','log_problems_mc','D_x_problems'], 'Model2 Problem Breadth')
ols_report('Y_low_completion', ['D_prior','log_active_days_mc','D_x_days'], 'Model3 Practice Regularity')

print('\n'+'='*70)
print('TABLE 13: Multi-Path Model')
print('='*70)
mp_cols = ['D_prior','log_submissions_mc','log_problems_mc','log_active_days_mc','D_x_submissions','D_x_problems','D_x_days']
X = sm.add_constant(user[mp_cols])
m5 = sm.OLS(user['Y_low_completion'], X).fit(cov_type='HC1')
print(f'R2={m5.rsquared:.3f}, N={int(m5.nobs)}')
for c in mp_cols:
    print(f'  {c:28s} beta={m5.params[c]: .4f} SE={m5.bse[c]:.4f} p={m5.pvalues[c]:.4f}')

print('\n'+'='*70)
print('TABLE 14: VIF-corrected (Submission Volume removed)')
print('='*70)
vc_cols = ['D_prior','log_problems_mc','log_active_days_mc','D_x_problems','D_x_days']
X = sm.add_constant(user[vc_cols])
m6 = sm.OLS(user['Y_low_completion'], X).fit(cov_type='HC1')
print(f'R2={m6.rsquared:.3f}, N={int(m6.nobs)}')
for c in vc_cols:
    print(f'  {c:28s} beta={m6.params[c]: .4f} SE={m6.bse[c]:.4f} p={m6.pvalues[c]:.4f}')

print('\n'+'='*70)
print('PSM: top-quartile submission volume -> Y_low_completion (1:1 NN, 0.25 caliper)')
print('='*70)
q = user['log_submissions'].quantile(0.75)
user['treat'] = (user['log_submissions'] >= q).astype(int)
print(f'Top-quartile threshold (log_submissions) = {q:.3f}')
print(f'Treated (top quartile) count = {user["treat"].sum()}, Control = {(user["treat"]==0).sum()}')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
psX = user[['D_prior','log_problems_mc','log_active_days_mc','log_submissions_mc']]
psX = sm.add_constant(psX)
lr = LogisticRegression(max_iter=1000).fit(psX, user['treat'])
user['ps'] = lr.predict_proba(psX)[:,1]
treated = user[user['treat']==1]; control = user[user['treat']==0]
nn = NearestNeighbors(n_neighbors=1).fit(control[['ps']])
dist, idx = nn.kneighbors(treated[['ps']])
matched_c = control.iloc[idx.flatten()]
# caliper 0.25 SD of PS
ps_sd = user['ps'].std()
keep = dist.flatten() <= 0.25*ps_sd
mt = treated[keep]; mc = matched_c[keep]
att = mt['Y_low_completion'].mean() - mc['Y_low_completion'].mean()
print(f'After 0.25-SD caliper: matched treated={len(mt)}, matched control={len(mc)}')
print(f'  treated low-completion rate = {mt["Y_low_completion"].mean():.3f}')
print(f'  control low-completion rate = {mc["Y_low_completion"].mean():.3f}')
print(f'  ATT = {att:.3f}')

print('\n'+'='*70)
print('MEDIATION: Baron-Kenny submission volume mediates D_prior -> Y_low_completion')
print('='*70)
# total effect c
Xc = sm.add_constant(user[['D_prior']]); mc_tot = sm.OLS(user['Y_low_completion'], Xc).fit(cov_type='HC1')
c = mc_tot.params['D_prior']; c_p = mc_tot.pvalues['D_prior']
# a: D_prior -> mediator (log_submissions)
Xa = sm.add_constant(user[['D_prior']]); ma = sm.OLS(user['log_submissions_mc'], Xa).fit(cov_type='HC1')
a = ma.params['D_prior']
# b: mediator -> Y controlling for D_prior
Xb = sm.add_constant(user[['log_submissions_mc','D_prior']]); mb = sm.OLS(user['Y_low_completion'], Xb).fit(cov_type='HC1')
b = mb.params['log_submissions_mc']
# c prime
cprime = mb.params['D_prior']; cp_p = mb.pvalues['D_prior']
ab = a*b
# Sobel
a_se = ma.bse['D_prior']; b_se = mb.bse['log_submissions_mc']
z = ab / np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
print(f'  c (total) = {c:.4f} (p={c_p:.4f})')
print(f'  a = {a:.4f}, b = {b:.4f}, ab (indirect) = {ab:.4f}')
print(f'  c\' (direct) = {cprime:.4f} (p={cp_p:.4f})')
print(f'  Sobel z = {z:.3f}')
print(f'  mediation ratio (ab/c) = {ab/c:.3f}')

print('\n'+'='*70)
print('BEHAVIORAL: Contest Participation x D_prior (N=651)')
print('='*70)
BEHAV = os.path.join(BASE, 'data/bepkt/raw_data/behavior.csv')
if not os.path.exists(BEHAV):
    print('[info] behavior.csv (large raw activity log) is not redistributed with this')
    print('       repository; the behavioural contest-engagement check is skipped.')
    print('       See README > Data sources for how to obtain it.')
    print('\nDONE')
    raise SystemExit(0)
beh = pd.read_csv(BEHAV)
beh['timestamp'] = pd.to_datetime(beh['timestamp'], utc=True)
def ext_sid(a):
    try:
        d = json.loads(a.replace("'", '"')); dig = re.findall(r'\d+', d.get('name',''))
        return dig[0] if dig else None
    except: return None
def ext_verb(v):
    try: return json.loads(v.replace("'", '"'))['id'].split('/')[-1]
    except: return 'unknown'
beh['student_id'] = beh['actor'].apply(ext_sid)
beh['verb_type'] = beh['verb'].apply(ext_verb)
beh = beh.dropna(subset=['student_id'])
bs = beh.groupby('student_id').agg(
    contest_clicks=('verb_type', lambda x: (x=='clickContestProblem').sum()),
    contest_submits=('verb_type', lambda x: (x=='submitContestProblem').sum()),
).reset_index()
bs['contest_engagement'] = bs['contest_clicks'] + bs['contest_submits']
# merge D_prior by student_id (need mapping user_id <-> student_id?) Use first_sub user matching
# In redraw, behavior D_early_high used median split of early_activity (separate). But paper's Contest x D_prior
# requires linking behavior student to submission D_prior. We approximate by matching on id overlap.
user['student_id'] = user['user_id'].astype(str)
merged = bs.merge(user[['student_id','D_prior','Y_low_completion']], on='student_id', how='inner')
print(f'  Behavioral matched N = {len(merged)}')
if len(merged) > 10:
    merged['contest_mc'] = merged['contest_engagement'] - merged['contest_engagement'].mean()
    merged['D_x_contest'] = merged['D_prior'] * merged['contest_mc']
    Xb2 = sm.add_constant(merged[['D_prior','contest_mc','D_x_contest']])
    mb2 = sm.OLS(merged['Y_low_completion'], Xb2).fit(cov_type='HC1')
    print(f'  Contest x D_prior beta={mb2.params["D_x_contest"]:.4f} p={mb2.pvalues["D_x_contest"]:.4f}')
print('\nDONE')
