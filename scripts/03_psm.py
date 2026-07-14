"""
W2 升级版：完整 PSM 分析
================================================
- 升级协变量（含非线性项 + 交互项）
- Caliper 调参（0.05 / 0.1 / 0.2 / 0.5 SD）
- 平衡检验（SMD）
- ATT 估计（配对 t-test + 配对 OLS）
- 稳健性：替代处理变量 / 替代协变量集
- Rosenbaum bounds 敏感性分析
- 输出：Table 2/3 + Fig 2/3
"""
import pandas as pd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import statsmodels.api as sm
import os
# 将 matplotlib 缓存放到项目目录内，避免沙箱权限问题
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path("data")
OUT  = Path("artifacts")
OUT.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(DATA / "df_ccc_treatments.csv", encoding='utf-8-sig')
print(f"[Input] N={df.shape[0]}")

# ============================================================
# 协变量构造：基础 + 非线性 + 交互项
# ============================================================
df['pres_2014J'] = (df['code_presentation'] == '2014J').astype(int)
df['female_x_age'] = df['female'] * df['age_mid']
df['edu_x_affluence'] = df['edu_ord'] * df['affluence']
df['disability_x_age'] = df['disability'] * df['age_mid']

# 4 个之前没平衡的变量
# female (binary) - 可加与其他变量的交互
# edu_ord (ordinal) - 可加平方项
# affluence (continuous) - 可加平方项
# disability (binary) - 可加与其他变量的交互
df['edu_ord_sq'] = df['edu_ord'] ** 2
df['affluence_sq'] = df['affluence'] ** 2

base_covs = ['age_mid', 'female', 'edu_ord', 'affluence',
             'disability', 'prev_attempts', 'studied_credits_std',
             'reg_offset', 'unreg_flag',
             'D_oucontent', 'D_quiz', 'D_resource']
nonlinear = ['edu_ord_sq', 'affluence_sq',
             'female_x_age', 'edu_x_affluence', 'disability_x_age']
pres = ['pres_2014J']

covariates = base_covs + nonlinear + pres
print(f"协变量数: {len(covariates)}")

# ============================================================
# 1) 估计倾向得分
# ============================================================
print("\n" + "="*70)
print("1) Logit 倾向得分估计")
print("="*70)
D = 'D_forumng_q75'
X = df[covariates].values
T = df[D].values
Y = df['Y_score'].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(X_s, T)
ps = lr.predict_proba(X_s)[:, 1]
df['ps'] = ps

print(f"PS 分布: min={ps.min():.3f}  max={ps.max():.3f}")
print(f"  mean(T) = {ps[T==1].mean():.3f}  mean(C) = {ps[T==0].mean():.3f}")
print(f"  AUC = {lr.score(X_s, T):.3f}")

# 共同支撑检查
ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())
print(f"共同支撑: [{ps_min:.3f}, {ps_max:.3f}]")

# ============================================================
# 2) Caliper 调参
# ============================================================
print("\n" + "="*70)
print("2) Caliper 调参")
print("="*70)


def do_match(ps, T, caliper_val, seed=None):
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ps[control_idx].reshape(-1, 1))
    matched_t, matched_c = [], []
    for ti in treated_idx:
        p_t = ps[ti]
        dist, idx = nn.kneighbors([[p_t]])
        if dist[0, 0] < caliper_val:
            matched_t.append(ti)
            matched_c.append(control_idx[idx[0, 0]])
    return matched_t, matched_c


caliper_results = []
for c_sd in [0.05, 0.1, 0.2, 0.5, 1.0]:
    cal = c_sd * ps.std()
    mt, mc = do_match(ps, T, cal)
    n_dropped = (T == 1).sum() - len(mt)
    caliper_results.append({
        'caliper_SD': c_sd,
        'caliper_val': round(cal, 4),
        'N_matched_T': len(mt),
        'N_matched_C': len(mc),
        'N_dropped_T': n_dropped,
        'match_rate_T': round(len(mt) / (T == 1).sum(), 3),
    })
    print(f"  caliper={c_sd}*SD ({cal:.4f}): N_T={len(mt)}, N_dropped_T={n_dropped}")

caliper_df = pd.DataFrame(caliper_results)
caliper_df.to_csv(OUT / "table_caliper_tuning.csv", index=False, encoding='utf-8-sig')

# 选 0.2 * SD 作为主结果
CHOSEN_SD = 0.2
caliper = CHOSEN_SD * ps.std()
print(f"\n主分析采用 caliper = {CHOSEN_SD} * SD = {caliper:.4f}")

# 对 affluence 做 5 分箱粗化（用于 CEM 辅助分层）
df['affluence_q'] = pd.qcut(df['affluence'], 5, labels=False, duplicates='drop')

# ============================================================
# 3) 主匹配 + 平衡检验
# ============================================================
print("\n" + "="*70)
print(f"3) 主匹配 (caliper = {CHOSEN_SD}*SD)")
print("="*70)

# 对 female 和 disability 做联合精确分层，对 affluence 做粗化分箱控制
print("采用联合精确分层：female × disability × affluence_q × edu_ord，每层内做 PS 1:1 NN 匹配")
all_matched_t, all_matched_c = [], []
strata = df.groupby(['female', 'disability', 'affluence_q', 'edu_ord'])
for keys, group in strata:
    idx_g = group.index.to_numpy()
    ps_g = ps[idx_g]
    T_g = T[idx_g]
    treated_g = np.where(T_g == 1)[0]
    control_g = np.where(T_g == 0)[0]
    if len(treated_g) == 0 or len(control_g) == 0:
        continue
    nn_g = NearestNeighbors(n_neighbors=1)
    nn_g.fit(ps_g[control_g].reshape(-1, 1))
    for ti_local in treated_g:
        ti_global = idx_g[ti_local]
        p_t = ps[ti_global]
        dist, idx = nn_g.kneighbors([[p_t]])
        if dist[0, 0] < caliper:
            ci_global = idx_g[control_g[idx[0, 0]]]
            all_matched_t.append(ti_global)
            all_matched_c.append(ci_global)

matched_treated = all_matched_t
matched_control = all_matched_c
print(f"匹配后: N_T={len(matched_treated)}, N_C={len(matched_control)}")
print(f"匹配率: T={len(matched_treated)/(T==1).sum():.1%}, C={len(matched_control)/(T==0).sum():.1%}")

df_m = df.iloc[matched_treated + matched_control].copy()
df_m['matched_set'] = ['T'] * len(matched_treated) + ['C'] * len(matched_control)
df_m['pair_id'] = list(range(len(matched_treated))) + list(range(len(matched_treated)))
df_m.to_csv(OUT / "df_matched.csv", index=False, encoding='utf-8-sig')

# 平衡检验
print("\n" + "="*70)
print("4) 平衡检验 (SMD)")
print("="*70)
rows = []
for var in covariates:
    t = df_m.loc[df_m['matched_set']=='T', var]
    c = df_m.loc[df_m['matched_set']=='C', var]
    sd_pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    smd = (t.mean() - c.mean()) / sd_pooled if sd_pooled > 0 else 0
    rows.append({
        'Variable': var,
        'T_mean': round(t.mean(), 3),
        'C_mean': round(c.mean(), 3),
        'SMD':     round(smd, 3),
        'SMD_abs': round(abs(smd), 3),
        'Balanced': 'YES' if abs(smd) < 0.1 else 'NO',
    })
smd_table = pd.DataFrame(rows)
print(smd_table.to_string(index=False))

n_balanced = smd_table['Balanced'].value_counts().get('YES', 0)
print(f"\n匹配后 SMD<0.1: {n_balanced}/{len(smd_table)} 个变量")
unbalanced_vars = smd_table[smd_table['Balanced']=='NO']['Variable'].tolist()
if unbalanced_vars:
    print(f"未平衡: {unbalanced_vars}")
else:
    print("ALL VARIABLES BALANCED")

smd_table.to_csv(OUT / "table2_balance.csv", index=False, encoding='utf-8-sig')

# ============================================================
# 5) ATT 估计
# ============================================================
print("\n" + "="*70)
print("5) ATT 估计")
print("="*70)

# 配对 t-test
t_paired = df_m.loc[df_m['matched_set']=='T', 'Y_score'].values
c_paired = df_m.loc[df_m['matched_set']=='C', 'Y_score'].values
att = t_paired - c_paired
att_mean = att.mean()
att_std = att.std(ddof=1)
tstat, pval = stats.ttest_rel(t_paired, c_paired)
print(f"配对差值均值: {att_mean:.2f}, SE = {att_std/np.sqrt(len(att)):.2f}")
print(f"配对 t 检验:  t={tstat:.3f}, p={pval:.6f}")
print(f"ATT (配对) = {att_mean:.2f}")

# 未配对 t 检验 (对照)
t_unp = df_m.loc[df_m['matched_set']=='T', 'Y_score']
c_unp = df_m.loc[df_m['matched_set']=='C', 'Y_score']
tstat_unp, pval_unp = stats.ttest_ind(t_unp, c_unp, equal_var=False)
print(f"未配对 t:     t={tstat_unp:.3f}, p={pval_unp:.6f}")

# 匹配前 OLS 基准
X_ols_full = sm.add_constant(df[[D] + covariates])
ols_unmatched = sm.OLS(df['Y_score'], X_ols_full).fit()
print(f"\n匹配前 OLS (D={D} 系数): {ols_unmatched.params[D]:.2f}, "
      f"SE={ols_unmatched.bse[D]:.2f}, p={ols_unmatched.pvalues[D]:.4f}")

# 匹配后 OLS (with covariates)
X_ols_m = sm.add_constant(df_m[[D] + covariates])
ols_matched = sm.OLS(df_m['Y_score'], X_ols_m).fit()
print(f"匹配后 OLS (D={D} 系数): {ols_matched.params[D]:.2f}, "
      f"SE={ols_matched.bse[D]:.2f}, p={ols_matched.pvalues[D]:.4f}")

# 配对 OLS with clustered SE (by pair_id)
# 用 entity effects 吸收 pair
df_m['treat'] = (df_m['matched_set'] == 'T').astype(int)
ols_paired = sm.OLS(df_m['Y_score'],
                     sm.add_constant(df_m[['treat']])).fit(
                     cov_type='cluster', cov_kwds={'groups': df_m['pair_id']})
print(f"配对回归 (clustered SE): coef={ols_paired.params['treat']:.2f}, "
      f"SE={ols_paired.bse['treat']:.2f}, p={ols_paired.pvalues['treat']:.4f}")

# ============================================================
# 6) 稳健性检验
# ============================================================
print("\n" + "="*70)
print("6) 稳健性检验")
print("="*70)

robust = []

# R1: 切分阈值换成 top50%（和主 D 的 q75 对比）
print(f"\nR1: Cutoff = q50 (> {df['forumng_clicks'].quantile(0.5):.0f} clicks)")
T_alt = df['D_forumng_median'].values
ps_alt = LogisticRegression(max_iter=2000, C=1.0).fit(
    StandardScaler().fit_transform(X), T_alt).predict_proba(
    StandardScaler().fit_transform(X))[:, 1]
cal_alt = 0.2 * ps_alt.std()
mt_alt, mc_alt = do_match(ps_alt, T_alt, cal_alt)
df_alt = df.iloc[mt_alt + mc_alt].copy()
df_alt['matched_set'] = ['T']*len(mt_alt) + ['C']*len(mc_alt)
t_alt = df_alt.loc[df_alt['matched_set']=='T', 'Y_score'].values
c_alt = df_alt.loc[df_alt['matched_set']=='C', 'Y_score'].values
att_alt = (t_alt - c_alt).mean()
pval_alt = stats.ttest_rel(t_alt, c_alt)[1]
print(f"   ATT = {att_alt:.2f}, p = {pval_alt:.4f}, N_matched = {len(mt_alt)}")
robust.append({'name': 'R1: cutoff=q50', 'ATT': round(att_alt, 2),
               'p': round(pval_alt, 4), 'N': len(mt_alt)})

# R2: D 换成 D_oucontent_high (高频 oucontent 互动, 按 0.5*median 切分)
print(f"\nR2: D = D_oucontent_high (0.5*median)")
median_ou = df['oucontent_clicks'].median()
df['D_oucontent_high'] = (df['oucontent_clicks'] > 0.5 * median_ou).astype(int)
print(f"   D=1 占比: {df['D_oucontent_high'].mean():.1%}")
T_o = df['D_oucontent_high'].values
ps_o = LogisticRegression(max_iter=2000, C=1.0).fit(
    StandardScaler().fit_transform(X), T_o).predict_proba(
    StandardScaler().fit_transform(X))[:, 1]
cal_o = 0.2 * ps_o.std()
mt_o, mc_o = do_match(ps_o, T_o, cal_o)
df_o = df.iloc[mt_o + mc_o].copy()
df_o['matched_set'] = ['T']*len(mt_o) + ['C']*len(mc_o)
t_o = df_o.loc[df_o['matched_set']=='T', 'Y_score'].values
c_o = df_o.loc[df_o['matched_set']=='C', 'Y_score'].values
att_o = (t_o - c_o).mean()
pval_o = stats.ttest_rel(t_o, c_o)[1]
print(f"   ATT = {att_o:.2f}, p = {pval_o:.4f}, N_matched = {len(mt_o)}")
robust.append({'name': 'R2: D=oucontent_high', 'ATT': round(att_o, 2),
               'p': round(pval_o, 4), 'N': len(mt_o)})

# R3: D 换成 q90 (更严格的高参与定义)
print(f"\nR3: D = D_forumng_q90 (>{df['forumng_clicks'].quantile(0.9):.0f} clicks, D=1={df['D_forumng_q90'].mean():.1%})")
T_b = df['D_forumng_q90'].values
ps_b = LogisticRegression(max_iter=2000, C=1.0).fit(
    StandardScaler().fit_transform(X), T_b).predict_proba(
    StandardScaler().fit_transform(X))[:, 1]
cal_b = 0.2 * ps_b.std()
mt_b, mc_b = do_match(ps_b, T_b, cal_b)
df_b = df.iloc[mt_b + mc_b].copy()
df_b['matched_set'] = ['T']*len(mt_b) + ['C']*len(mc_b)
t_b = df_b.loc[df_b['matched_set']=='T', 'Y_score'].values
c_b = df_b.loc[df_b['matched_set']=='C', 'Y_score'].values
att_b = (t_b - c_b).mean()
pval_b = stats.ttest_rel(t_b, c_b)[1]
print(f"   ATT = {att_b:.2f}, p = {pval_b:.4f}, N_matched = {len(mt_b)}")
robust.append({'name': 'R3: cutoff=q90', 'ATT': round(att_b, 2),
               'p': round(pval_b, 4), 'N': len(mt_b)})

# R4: 仅 2014B 学期
print(f"\nR4: 仅 2014B 学期")
df_2014B = df[df['code_presentation']=='2014B'].copy()
T_bb = df_2014B[D].values
X_bb = df_2014B[covariates].values
ps_bb = LogisticRegression(max_iter=2000, C=1.0).fit(
    StandardScaler().fit_transform(X_bb), T_bb).predict_proba(
    StandardScaler().fit_transform(X_bb))[:, 1]
cal_bb = 0.2 * ps_bb.std()
mt_bb, mc_bb = do_match(ps_bb, T_bb, cal_bb)
df_bb = df_2014B.iloc[mt_bb + mc_bb].copy()
df_bb['matched_set'] = ['T']*len(mt_bb) + ['C']*len(mc_bb)
t_bb = df_bb.loc[df_bb['matched_set']=='T', 'Y_score'].values
c_bb = df_bb.loc[df_bb['matched_set']=='C', 'Y_score'].values
att_bb = (t_bb - c_bb).mean()
pval_bb = stats.ttest_rel(t_bb, c_bb)[1]
print(f"   ATT = {att_bb:.2f}, p = {pval_bb:.4f}, N_matched = {len(mt_bb)}")
robust.append({'name': 'R4: 2014B only', 'ATT': round(att_bb, 2),
               'p': round(pval_bb, 4), 'N': len(mt_bb)})

# R5: 协变量简化
print(f"\nR5: 协变量简化（去非线性）")
covs_simple = base_covs + pres
X_s2 = StandardScaler().fit_transform(df[covs_simple].values)
ps_s2 = LogisticRegression(max_iter=2000, C=1.0).fit(X_s2, T).predict_proba(X_s2)[:, 1]
cal_s2 = 0.2 * ps_s2.std()
mt_s2, mc_s2 = do_match(ps_s2, T, cal_s2)
df_s2 = df.iloc[mt_s2 + mc_s2].copy()
df_s2['matched_set'] = ['T']*len(mt_s2) + ['C']*len(mc_s2)
t_s2 = df_s2.loc[df_s2['matched_set']=='T', 'Y_score'].values
c_s2 = df_s2.loc[df_s2['matched_set']=='C', 'Y_score'].values
att_s2 = (t_s2 - c_s2).mean()
pval_s2 = stats.ttest_rel(t_s2, c_s2)[1]
print(f"   ATT = {att_s2:.2f}, p = {pval_s2:.4f}, N_matched = {len(mt_s2)}")
robust.append({'name': 'R5: simple covariates', 'ATT': round(att_s2, 2),
               'p': round(pval_s2, 4), 'N': len(mt_s2)})

# R6: 协变量加全 (加 imd_band 哑变量)
print(f"\nR6: 加 imd_band 哑变量")
imd_dummies = pd.get_dummies(df['imd_band'], prefix='imd', drop_first=True)
for col in imd_dummies.columns:
    df[col] = imd_dummies[col]
covs_full = base_covs + nonlinear + pres + imd_dummies.columns.tolist()
X_f = StandardScaler().fit_transform(df[covs_full].values)
ps_f = LogisticRegression(max_iter=2000, C=1.0).fit(X_f, T).predict_proba(X_f)[:, 1]
cal_f = 0.2 * ps_f.std()
mt_f, mc_f = do_match(ps_f, T, cal_f)
df_f = df.iloc[mt_f + mc_f].copy()
df_f['matched_set'] = ['T']*len(mt_f) + ['C']*len(mc_f)
t_f = df_f.loc[df_f['matched_set']=='T', 'Y_score'].values
c_f = df_f.loc[df_f['matched_set']=='C', 'Y_score'].values
att_f = (t_f - c_f).mean()
pval_f = stats.ttest_rel(t_f, c_f)[1]
print(f"   ATT = {att_f:.2f}, p = {pval_f:.4f}, N_matched = {len(mt_f)}")
robust.append({'name': 'R6: add imd_band', 'ATT': round(att_f, 2),
               'p': round(pval_f, 4), 'N': len(mt_f)})

robust_df = pd.DataFrame(robust)
robust_df.to_csv(OUT / "table3_robustness.csv", index=False, encoding='utf-8-sig')
print("\n稳健性结果汇总:")
print(robust_df.to_string(index=False))

# ============================================================
# 7) Rosenbaum bounds 敏感性分析
# ============================================================
print("\n" + "="*70)
print("7) Rosenbaum bounds 敏感性分析")
print("="*70)
# Wilcoxon signed-rank (更稳健)
w_stat, w_p = stats.wilcoxon(att, alternative='greater')
print(f"Wilcoxon signed-rank: W={w_stat:.0f}, p={w_p:.6f}")

# Rosenbaum bounds
# 标准实现: 配对差值 dt = Y_T - Y_C
# Gamma = 偏倚因子（1.0 = 无偏倚）
# 假设: 每对匹配中, P(T=1) / P(T=0) <= Gamma
# 用 Wilcoxon signed-rank statistic
def rosenbaum_bounds_pvalues(diffs, gamma_list):
    """
    配对差值的 Rosenbaum bounds
    公式: T+ = sum(rank(|d_i|) * I(d_i > 0))
    E[T+ | Gamma=1] = N(N+1)/4
    Var[T+ | Gamma=1] = N(N+1)(2N+1)/24
    在 Gamma 下: 偏倚因子 = 1 + (Gamma-1)/(1+(Gamma-1)) * (1-|d_i|/d_max)
    """
    from scipy.stats import norm
    n = len(diffs)
    abs_diffs = np.abs(diffs)
    # Wilcoxon signed-rank: 计算 T+
    ranks = stats.rankdata(abs_diffs)
    T_plus = np.sum(ranks * (diffs > 0))
    # 期望和方差 (under H0 with Gamma=1)
    E_T = n * (n + 1) / 4
    V_T = n * (n + 1) * (2 * n + 1) / 24
    # Gamma=1: 上界 p 和下界 p 一致
    results = []
    for gamma in gamma_list:
        if gamma == 1.0:
            # 标准 Wilcoxon
            z_lower = (T_plus - E_T) / np.sqrt(V_T)
            p_one = 1 - norm.cdf(z_lower)  # 单侧
            results.append({
                'Gamma': gamma,
                'p_upper': round(p_one, 4),
                'p_lower': round(p_one, 4),
            })
        else:
            # 上界: 假设所有 P(T=1) = gamma/(1+gamma)
            # E_upper = sum( rank * gamma/(1+gamma) )
            E_upper = np.sum(ranks) * gamma / (1 + gamma)
            V_upper = np.sum(ranks**2) * gamma / (1 + gamma)**2
            z_upper = (T_plus - E_upper) / np.sqrt(V_upper) if V_upper > 0 else 0
            p_upper = 1 - norm.cdf(z_upper)
            # 下界: 假设所有 P(T=1) = 1/(1+gamma)
            E_lower = np.sum(ranks) * 1 / (1 + gamma)
            V_lower = np.sum(ranks**2) * 1 / (1 + gamma)**2
            z_lower = (T_plus - E_lower) / np.sqrt(V_lower) if V_lower > 0 else 0
            p_lower = 1 - norm.cdf(z_lower)
            results.append({
                'Gamma': gamma,
                'p_upper': round(p_upper, 4),
                'p_lower': round(p_lower, 4),
            })
    return results


gamma_list = [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
rb_results = rosenbaum_bounds_pvalues(att, gamma_list)
rb_df = pd.DataFrame(rb_results)
print(rb_df.to_string(index=False))
rb_df.to_csv(OUT / "table_rosenbaum.csv", index=False, encoding='utf-8-sig')

# 找 Gamma* (p_upper > 0.05)
gamma_critical = None
for r in rb_results:
    if r['p_upper'] > 0.05:
        gamma_critical = r['Gamma']
        break
print(f"\nGamma* (p_upper 突破 0.05): {gamma_critical}")

# ============================================================
# 8) 可视化
# ============================================================
print("\n" + "="*70)
print("8) 出版级图表")
print("="*70)

# Fig 2: PS 分布图
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(ps[T==0], bins=30, alpha=0.6, label='Control (D=0)', color='steelblue', edgecolor='black')
axes[0].hist(ps[T==1], bins=30, alpha=0.6, label='Treatment (D=1)', color='coral', edgecolor='black')
axes[0].set_xlabel('Propensity Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('(a) PS Distribution (Before Matching)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 匹配后
ps_m = ps[np.concatenate([matched_treated, matched_control])]
T_m = np.concatenate([[1]*len(matched_treated), [0]*len(matched_control)])
axes[1].hist(ps_m[T_m==0], bins=30, alpha=0.6, label='Control (matched)', color='steelblue', edgecolor='black')
axes[1].hist(ps_m[T_m==1], bins=30, alpha=0.6, label='Treatment (matched)', color='coral', edgecolor='black')
axes[1].set_xlabel('Propensity Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('(b) PS Distribution (After Matching)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig2_ps_distribution.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: artifacts/fig2_ps_distribution.png")

# Fig 3: SMD 对比图
fig, ax = plt.subplots(figsize=(7, 7))
plot_df = smd_table.copy()
plot_df = plot_df.sort_values('SMD_abs', ascending=True)
colors = ['green' if x == 'YES' else 'red' for x in plot_df['Balanced']]
ax.barh(plot_df['Variable'], plot_df['SMD_abs'], color=colors, edgecolor='black')
ax.axvline(0.1, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.1)')
ax.set_xlabel('|Standardized Mean Difference|')
ax.set_ylabel('Covariate')
ax.set_title('Balance Test (After Matching)')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig3_balance.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: artifacts/fig3_balance.png")

# Fig 4: ATT 稳健性
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = ['Main'] + [r['name'].split(': ')[1] if ': ' in r['name'] else r['name'] for r in robust]
atts = [att_mean] + [r['ATT'] for r in robust]
ax.barh(labels, atts, color=['steelblue'] + ['lightblue']*len(robust), edgecolor='black')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('ATT (Weighted Final Score, points)')
ax.set_title('Robustness of ATT Estimate')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig4_robustness.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: artifacts/fig4_robustness.png")

# ============================================================
# 9) 结果汇总
# ============================================================
print("\n" + "="*70)
print("9) 完整结果汇总")
print("="*70)
result = {
    "D_variable": D,
    "caliper_SD": CHOSEN_SD,
    "caliper_val": round(caliper, 4),
    "N_unmatched": int(df.shape[0]),
    "N_matched":   int(df_m.shape[0]),
    "N_treated_matched": int((df_m['matched_set']=='T').sum()),
    "N_control_matched": int((df_m['matched_set']=='C').sum()),
    "n_vars_balanced": int(n_balanced),
    "n_vars_total":    int(len(smd_table)),
    "unbalanced_vars": unbalanced_vars,
    "ATT_paired_t":  round(att_mean, 2),
    "ATT_SE":        round(att_std/np.sqrt(len(att)), 2),
    "pvalue_paired": round(pval, 6),
    "pvalue_unpaired": round(pval_unp, 6),
    "OLS_unmatched": round(ols_unmatched.params[D], 2),
    "OLS_matched":   round(ols_matched.params[D], 2),
    "OLS_paired_clustered": round(ols_paired.params['treat'], 2),
    "wilcoxon_p":    float(w_p),
    "robustness":    robust,
}
with open(OUT / "psm_result.json", "w", encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"\nSaved: artifacts/psm_result.json")
