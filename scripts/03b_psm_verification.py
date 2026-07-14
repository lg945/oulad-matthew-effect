"""
Independent recomputation to verify OULAD Tables 4 (R2), 6 (PSM/IPW), 7 (balance), 9 (R2), 10 (Rosenbaum).
Spec: simple-covariate PSM matching forum vs non-forum on 6 covariates, 1:1 NN, caliper 0.25*sd(ps).
Data: data/ccc_complete.csv  (N=4,282 Low/High)
"""
import os, sys, warnings
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
warnings.filterwarnings('ignore')

BASE = REPO_ROOT
DATA = os.path.join(BASE, "data")
sys.stdout.reconfigure(encoding='utf-8')

# ---------- build data exactly per script 16 ----------
df = pd.read_csv(os.path.join(DATA, "ccc_complete.csv"), encoding='utf-8-sig')
df['D_forumng'] = (df['forumng_clicks'] > 0).astype(int)
df['Y_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)
df['female'] = (df['gender'] == 'F').astype(int)
df['disability_bin'] = (df['disability'] == 'Y').astype(int)
imd_map = {'0-10%':5,'10-20%':15,'20-30%':25,'30-40%':35,'40-50%':45,'50-60%':55,
           '60-70%':65,'70-80%':75,'80-90%':85,'90-100%':95}
df['imd_num'] = df['imd_band'].map(imd_map)
age_map = {'0-35':25,'35-55':45,'55<=':65}
df['age_mid'] = df['age_band'].map(age_map)
df['prev_attempts'] = df['num_of_prev_attempts']
df['edu_group'] = 'Other'
df.loc[df['highest_education']=='Lower Than A Level','edu_group']='Low'
df.loc[df['highest_education'].isin(['A Level or Equivalent','HE Qualification']),'edu_group']='High'
df_sub = df[df['edu_group'].isin(['Low','High'])].copy()
df_sub['edu_high'] = (df_sub['edu_group']=='High').astype(int)
df_sub['imd_num'] = df_sub['imd_num'].fillna(df_sub['imd_num'].median())

covariates = ['age_mid','female','disability_bin','imd_num','prev_attempts','studied_credits']

print("="*70)
print("VERIFY Table 4 / 9 R-squared")
print("="*70)
# Build multi-path vars
df_sub['total_clicks_log'] = np.log1p(df_sub['total_clicks'])
df_sub['forum_clicks_log'] = np.log1p(df_sub['forumng_clicks'])
# need vle_diversity, assessment_ratio, study_days for Table 4 (raw)
# small reuse: recompute from raw parquet quickly
import pyarrow.parquet as pq
vle = pd.read_parquet(os.path.join(DATA,"vle.parquet"))
ccc_vle = vle[vle['code_module']=='CCC']
ccc_vle_ids = set(ccc_vle['id_site'])
sv = pd.concat([pd.read_parquet(os.path.join(DATA,"studentVle_p2014J_.parquet")),
                pd.read_parquet(os.path.join(DATA,"studentVle_p2014B_.parquet"))], ignore_index=True)
ccc_ids = set(df_sub['id_student'])
sv = sv[sv['id_student'].isin(ccc_ids) & sv['id_site'].isin(ccc_vle_ids)]
sv = sv.merge(ccc_vle[['id_site','activity_type']], on='id_site', how='left')
vle_div = sv.groupby('id_student')['activity_type'].nunique()
df_sub['vle_diversity'] = df_sub['id_student'].map(vle_div).fillna(0)
sa = pd.read_parquet(os.path.join(DATA,"studentAssessment.parquet"))
asmt = pd.read_parquet(os.path.join(DATA,"assessments.parquet"))
asmt_ccc = asmt[asmt['code_module']=='CCC']
nonexam = asmt_ccc[asmt_ccc['assessment_type']!='Exam'].groupby('code_presentation')['id_assessment'].nunique()
df_sub['total_nonexam'] = df_sub['code_presentation'].map(nonexam)
sa_valid = sa[sa['date_submitted']>=0]
subd = sa_valid.groupby('id_student')['id_assessment'].nunique()
df_sub['submitted_assessments'] = df_sub['id_student'].map(subd).fillna(0)
df_sub['assessment_ratio'] = (df_sub['submitted_assessments']/df_sub['total_nonexam']).clip(0,1)
study_days = sv.groupby('id_student')['date'].nunique()
df_sub['study_days'] = df_sub['id_student'].map(study_days).fillna(0)

y = df_sub['Y_withdrawn']
# Table 4: 5 vars raw (no centering)
engage4 = ['D_forumng','total_clicks_log','vle_diversity','assessment_ratio','study_days']
inter4 = []
for v in engage4:
    iv=f"{v}_x_edu"; df_sub[iv]=df_sub[v]*df_sub['edu_high']; inter4.append(iv)
X4 = sm.add_constant(df_sub[covariates+engage4+['edu_high']+inter4])
m4 = sm.OLS(y, X4).fit(cov_type='HC1')
print(f"Table 4 R2 (manuscript 0.395): {m4.rsquared:.4f}")
print(f"  D_forumng x edu p = {m4.pvalues['D_forumng_x_edu']:.4f}  (manuscript .62)")
print(f"  assessment_ratio x edu p = {m4.pvalues['assessment_ratio_x_edu']:.4f}  (manuscript .006)")
print(f"  study_days x edu p = {m4.pvalues['study_days_x_edu']:.4f}  (manuscript <.001)")

# Table 9: centered, drop total_clicks_log
for v in ['vle_diversity','assessment_ratio','study_days']:
    df_sub[f'{v}_c'] = df_sub[v]-df_sub[v].mean()
df_sub['edu_high_c'] = df_sub['edu_high']-df_sub['edu_high'].mean()
engage9 = ['D_forumng','vle_diversity_c','assessment_ratio_c','study_days_c']
inter9=[]
for v in engage9:
    iv=f"{v}_x_edu"; df_sub[iv]=df_sub[v]*df_sub['edu_high_c']; inter9.append(iv)
X9 = sm.add_constant(df_sub[covariates+engage9+['edu_high_c']+inter9])
m9 = sm.OLS(y, X9).fit(cov_type='HC1')
print(f"Table 9 R2 (manuscript 0.393): {m9.rsquared:.4f}")
print(f"  assessment x edu p = {m9.pvalues['assessment_ratio_c_x_edu']:.4f}  (manuscript .002)")
print(f"  study_days x edu p = {m9.pvalues['study_days_c_x_edu']:.4f}  (manuscript <.001)")

print()
print("="*70)
print("VERIFY Table 6 (PSM + IPW) and Table 7 (balance) and Table 10 (Rosenbaum)")
print("="*70)

def fit_ps(data, treat, out, covs):
    X = data[covs].values.astype(float)
    ps = LogisticRegression(max_iter=2000).fit(X, data[treat].values).predict_proba(X)[:,1]
    return ps

def match_1to1(ps, T, caliper_sd=0.25, seed=42):
    rng = np.random.default_rng(seed)
    cal = caliper_sd*ps.std()
    treated = np.where(T==1)[0]
    rng.shuffle(treated)
    control = np.where(T==0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(ps[control].reshape(-1,1))
    mt, mc = [], []
    used=set()
    for ti in treated:
        d, idx = nn.kneighbors([[ps[ti]]])
        ci = control[idx[0,0]]
        if d[0,0] < cal and ci not in used:
            mt.append(ti); mc.append(ci); used.add(ci)
    return np.array(mt), np.array(mc), cal

def smd(t, c):
    sd = np.sqrt((t.var(ddof=1)+c.var(ddof=1))/2)
    return (t.mean()-c.mean())/sd if sd>0 else 0.0

# ---- Full-sample PSM (forum vs non-forum) ----
ps_full = fit_ps(df_sub, 'D_forumng', 'Y_withdrawn', covariates)
mt, mc, cal = match_1to1(ps_full, df_sub['D_forumng'].values)
print(f"\n[Full PSM] caliper={cal:.4f}  N_matched_pairs={len(mt)}")
att_full = (df_sub['Y_withdrawn'].values[mt]-df_sub['Y_withdrawn'].values[mc]).mean()*100
print(f"  ATT = {att_full:.2f} pp   (manuscript Table 6: -39.0)")

# subgroup
for grp in ['Low','High']:
    sub = df_sub[df_sub['edu_group']==grp]
    ps = fit_ps(sub,'D_forumng','Y_withdrawn',covariates)
    a,b,_ = match_1to1(ps, sub['D_forumng'].values)
    att = (sub['Y_withdrawn'].values[a]-sub['Y_withdrawn'].values[b]).mean()*100
    print(f"  [{grp} PSM] N_pairs={len(a)} ATT={att:.2f} pp   (manuscript: Low -31.6, High -41.4)")

# ---- IPW ATT ----
def ipw_att(data, treat, out, covs):
    ps = fit_ps(data, treat, out, covs)
    T = data[treat].values; Y = data[out].values
    w = np.where(T==1, 1.0, ps/(1-ps))
    yt = Y[T==1].mean()
    yc = np.sum(w[T==0]*Y[T==0])/np.sum(w[T==0])
    return (yt-yc)*100
print(f"\n[IPW] Full={ipw_att(df_sub,'D_forumng','Y_withdrawn',covariates):.2f} "
      f"(manuscript -41.9)")
print(f"  Low={ipw_att(df_sub[df_sub.edu_group=='Low'],'D_forumng','Y_withdrawn',covariates):.2f} "
      f"(manuscript -33.9)")
print(f"  High={ipw_att(df_sub[df_sub.edu_group=='High'],'D_forumng','Y_withdrawn',covariates):.2f} "
      f"(manuscript -44.9)")

# ---- Table 7 balance (full-sample matched) ----
print("\n[Table 7 balance] covariate | SMD_before | SMD_after | (manuscript before/after)")
matched = df_sub.iloc[np.concatenate([mt,mc])].copy()
matched['set'] = ['T']*len(mt)+['C']*len(mc)
for v in covariates:
    t_full = df_sub.loc[df_sub['D_forumng']==1, v]
    c_full = df_sub.loc[df_sub['D_forumng']==0, v]
    t_m = matched.loc[matched['set']=='T', v]
    c_m = matched.loc[matched['set']=='C', v]
    sb = smd(t_full, c_full); sa = smd(t_m, c_m)
    print(f"  {v:<16s} {sb:+.3f}   {sa:+.3f}   abs_after={abs(sa):.3f}")

# ---- Table 10 Rosenbaum bounds on full matched pair diffs ----
diffs = (df_sub['Y_withdrawn'].values[mt]-df_sub['Y_withdrawn'].values[mc])
from scipy.stats import norm
n=len(diffs); ranks=stats.rankdata(np.abs(diffs)); Tplus=np.sum(ranks*(diffs>0))
E_T=n*(n+1)/4; V_T=n*(n+1)*(2*n+1)/24
print("\n[Table 10 Rosenbaum] Gamma | p_upper (manuscript: 2.5->.0003, 3.0->.031)")
for gamma in [1.0,1.5,2.0,2.5,3.0]:
    if gamma==1.0:
        z=(Tplus-E_T)/np.sqrt(V_T); pu=1-norm.cdf(z)
    else:
        E_u=np.sum(ranks)*gamma/(1+gamma); V_u=np.sum(ranks**2)*gamma/(1+gamma)**2
        z=(Tplus-E_u)/np.sqrt(V_u); pu=1-norm.cdf(z)
    print(f"  Gamma={gamma}: p_upper={pu:.4f}")
