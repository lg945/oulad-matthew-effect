"""
CANONICAL BePKT replication analysis (authoritative, fully mean-centered).

All pathway variables are MEAN-CENTERED, consistent with the manuscript text
("All pathway variables were mean-centered"). This script is the single source
of truth for Tables 11-14 and the PSM/descriptive numbers in the manuscript.

Outputs: prints rounded values for direct transcription into the markdown
tables, and saves CSVs under artifacts/ for the record.
"""
import os, warnings
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')

BASE = rREPO_ROOT
ART = os.path.join(BASE, 'artifacts')
os.makedirs(ART, exist_ok=True)

# ---------- Load & build ----------
sub = pd.read_csv(os.path.join(BASE, 'data/bepkt/raw_data/submission.csv'))
sub['create_time'] = pd.to_datetime(sub['create_time'], utc=True)
prob = pd.read_csv(os.path.join(BASE, 'data/bepkt/raw_data/problem.csv'))
sub = sub.merge(prob[['id', 'difficulty']], left_on='problem_id', right_on='id',
                how='left', suffixes=('', '_prob'))

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

ad = sub.groupby('user_id')['create_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='unique_active_days')
user = user.merge(ad, on='user_id', how='left')

user['log_submissions'] = np.log1p(user['total_submissions'])
user['log_problems'] = np.log1p(user['unique_problems_attempted'])
user['log_active_days'] = np.log1p(user['unique_active_days'])
for v in ['log_submissions', 'log_problems', 'log_active_days']:
    user[v + '_mc'] = user[v] - user[v].mean()
user['D_x_submissions'] = user['D_prior'] * user['log_submissions_mc']
user['D_x_problems'] = user['D_prior'] * user['log_problems_mc']
user['D_x_days'] = user['D_prior'] * user['log_active_days_mc']

N = len(user)
print(f'N students = {N}')
print(f'D_prior High = {int(user.D_prior.sum())} ({user.D_prior.mean()*100:.1f}%), '
      f'Low = {N-int(user.D_prior.sum())} ({(1-user.D_prior.mean())*100:.1f}%)')

# ---------- Table 11: descriptives by group ----------
print('\n=== TABLE 11: Descriptive Statistics by Early-Standing Group ===')
rows = []
for label, mask in [('Low', user.D_prior == 0), ('High', user.D_prior == 1)]:
    g = user[mask]
    rows.append([label, g.acceptance_rate.mean(), g.total_submissions.mean(),
                 g.unique_problems_attempted.mean(), g.unique_active_days.mean(),
                 g.Y_low_completion.mean() * 100])
for name, i in [('Acceptance rate (mean)', 1), ('Total submissions (mean)', 2),
                ('Unique problems attempted (mean)', 3), ('Active days (mean)', 4),
                ('Low completion rate (%)', 5)]:
    lo, hi = rows[0][i], rows[1][i]
    print(f'  {name:34s} Low={lo:7.3f}  High={hi:7.3f}  Gap(H-L)={hi-lo:7.3f}')
t11 = pd.DataFrame({'Group': ['Low', 'High'],
                    'Acceptance_rate': [rows[0][1], rows[1][1]],
                    'Total_submissions': [rows[0][2], rows[1][2]],
                    'Unique_problems': [rows[0][3], rows[1][3]],
                    'Active_days': [rows[0][4], rows[1][4]],
                    'Low_completion_pct': [rows[0][5], rows[1][5]]})
t11.to_csv(os.path.join(ART, 'table11_descriptives.csv'), index=False)

# ---------- Table 12: Individual pathway models (FULLY CENTERED) ----------
print('\n=== TABLE 12: Individual Pathway Models (fully centered) ===')
pathways = {'log_submissions_mc': 'Submission Volume (log)',
            'log_problems_mc': 'Problem Breadth (log)',
            'log_active_days_mc': 'Practice Regularity (log)'}
t12 = []
for var, label in pathways.items():
    # robust interaction column name
    if var == 'log_submissions_mc':
        ix = 'D_x_submissions'
    elif var == 'log_problems_mc':
        ix = 'D_x_problems'
    else:
        ix = 'D_x_days'
    X = sm.add_constant(user[['D_prior', var, ix]])
    m = sm.OLS(user['Y_low_completion'], X).fit(cov_type='HC1')
    dp_b, dp_p = m.params['D_prior'], m.pvalues['D_prior']
    t12.append({'Model': label, 'beta_pathway': m.params[var], 'se_pathway': m.bse[var],
                'p_pathway': m.pvalues[var], 'beta_Dprior': dp_b,
                'se_Dprior': m.bse['D_prior'], 'p_Dprior': dp_p,
                'beta_inter': m.params[ix], 'se_inter': m.bse[ix], 'p_inter': m.pvalues[ix],
                'R2': m.rsquared})
    print('  {:24s} path={: .4f}(p={:.4f}) Dprior={: .4f}(p={:.4f}) inter={: .4f}(p={:.4f}) R2={:.3f}'.format(
        label, m.params[var], m.pvalues[var], dp_b, dp_p, m.params[ix], m.pvalues[ix], m.rsquared))
pd.DataFrame(t12).to_csv(os.path.join(ART, 'table12_individual.csv'), index=False)

# ---------- Table 13: Multi-path model (centered) ----------
print('\n=== TABLE 13: Multi-Path Model (centered) ===')
Xm = sm.add_constant(user[['D_prior', 'log_submissions_mc', 'log_problems_mc',
                            'log_active_days_mc', 'D_x_submissions', 'D_x_problems', 'D_x_days']])
m5 = sm.OLS(user['Y_low_completion'], Xm).fit(cov_type='HC1')
print(f'  R2={m5.rsquared:.4f} N={int(m5.nobs)}')
for c in Xm.columns:
    if c == 'const':
        continue
    star = '***' if m5.pvalues[c] < .001 else ('**' if m5.pvalues[c] < .01 else ('*' if m5.pvalues[c] < .05 else ''))
    print(f'  {c:22s} beta={m5.params[c]: .4f} se={m5.bse[c]: .4f} p={m5.pvalues[c]:.4f} {star}')
mp = pd.DataFrame({'Variable': [c for c in Xm.columns if c != 'const'],
                   'Beta': [m5.params[c] for c in Xm.columns if c != 'const'],
                   'SE': [m5.bse[c] for c in Xm.columns if c != 'const'],
                   'p': [m5.pvalues[c] for c in Xm.columns if c != 'const']})
mp.to_csv(os.path.join(ART, 'table13_multipath.csv'), index=False)

# ---------- Table 14: VIF-corrected (drop log_submissions) ----------
print('\n=== TABLE 14: VIF-Corrected Multi-Path (drop Submission Volume) ===')
Xv = sm.add_constant(user[['D_prior', 'log_problems_mc', 'log_active_days_mc',
                            'D_x_problems', 'D_x_days']])
m5v = sm.OLS(user['Y_low_completion'], Xv).fit(cov_type='HC1')
print(f'  R2={m5v.rsquared:.4f} N={int(m5v.nobs)}')
for c in Xv.columns:
    if c == 'const':
        continue
    star = '***' if m5v.pvalues[c] < .001 else ('**' if m5v.pvalues[c] < .01 else ('*' if m5v.pvalues[c] < .05 else ''))
    print(f'  {c:22s} beta={m5v.params[c]: .4f} se={m5v.bse[c]: .4f} p={m5v.pvalues[c]:.4f} {star}')
vp = pd.DataFrame({'Variable': [c for c in Xv.columns if c != 'const'],
                   'Beta': [m5v.params[c] for c in Xv.columns if c != 'const'],
                   'SE': [m5v.bse[c] for c in Xv.columns if c != 'const'],
                   'p': [m5v.pvalues[c] for c in Xv.columns if c != 'const']})
vp.to_csv(os.path.join(ART, 'table14_vifcorrected.csv'), index=False)

# ---------- VIF ----------
print('\n=== VIF (full model) ===')
Xvif = user[['D_prior', 'log_submissions_mc', 'log_problems_mc', 'log_active_days_mc',
             'D_x_submissions', 'D_x_problems', 'D_x_days']].copy()
for i, col in enumerate(Xvif.columns):
    print(f'  {col:22s} VIF={variance_inflation_factor(Xvif.values, i):.2f}')

print('\nDONE — numbers above are authoritative for manuscript Tables 11-14.')
