import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""
W1.3 — Table 1: 描述统计 (处理组 vs 对照组)
==========================================
- 按 D_high_engagement 分组，输出各协变量的均值/标准差/差异显著性
- 输出 publication 级别的 Table 1 到 artifacts/table1_descriptive.csv
- 同时画 1 张分布图 (treatment 对比) 到 artifacts/fig1_distribution.png

W1.3 输出:
  artifacts/table1_descriptive.csv    <-- 投稿 Table 1 候选
  artifacts/fig1_distribution.png      <-- 投稿图 1 候选
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Chinese 字体（如果系统装了）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

DATA = Path("data")
OUT  = Path("artifacts")
OUT.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(DATA / "df_ccc.csv", encoding='utf-8-sig')
print(f"清洗后子样本: N = {df.shape[0]}, D = {df['D_high_engagement'].sum()}, Control = {(df['D_high_engagement']==0).sum()}")

# ============================================================
# 1. 协变量分组统计
# ============================================================
covariates_cont = ['age_mid', 'edu_ord', 'affluence', 'imd_num',
                   'prev_attempts', 'studied_credits',
                   'total_clicks', 'active_days', 'clicks_per_day', 'engagement_ratio',
                   'reg_offset']
covariates_bin   = ['female', 'disability', 'unreg_flag']

rows = []
for var in covariates_cont:
    t = df.loc[df['D_high_engagement']==1, var].dropna()
    c = df.loc[df['D_high_engagement']==0, var].dropna()
    # Welch's t-test (不等方差)
    tstat, pval = stats.ttest_ind(t, c, equal_var=False)
    pooled_sd = np.sqrt(((t.std(ddof=1)**2 + c.std(ddof=1)**2) / 2))
    # Cohen's d
    if pooled_sd > 0:
        d = (t.mean() - c.mean()) / pooled_sd
    else:
        d = 0.0
    rows.append({
        'Variable': var,
        'Type': 'Continuous',
        'Treated (D=1) Mean': round(t.mean(), 3),
        'Treated (D=1) SD':   round(t.std(ddof=1), 3),
        'Control (D=0) Mean': round(c.mean(), 3),
        'Control (D=0) SD':   round(c.std(ddof=1), 3),
        'Diff (T-C)':         round(t.mean() - c.mean(), 3),
        "Cohen's d":          round(d, 3),
        't / chi2':           round(tstat, 3),
        'p-value':            round(pval, 4),
        'Sig.':               '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else '',
    })

for var in covariates_bin:
    t = df.loc[df['D_high_engagement']==1, var].dropna()
    c = df.loc[df['D_high_engagement']==0, var].dropna()
    # 两组比例差异的卡方检验
    t1, t0 = int((t==1).sum()), int((t==0).sum())
    c1, c0 = int((c==1).sum()), int((c==0).sum())
    contingency = np.array([[t1, t0], [c1, c0]])
    try:
        chi2, pval, dof, _ = stats.chi2_contingency(contingency)
    except Exception:
        chi2, pval = np.nan, np.nan
    # Cohen's h
    p1, p2 = t1/len(t) if len(t) else 0, c1/len(c) if len(c) else 0
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    rows.append({
        'Variable': var,
        'Type': 'Binary',
        'Treated (D=1) Mean': round(p1, 3),
        'Treated (D=1) SD':   round(np.sqrt(p1*(1-p1)), 3),
        'Control (D=0) Mean': round(p2, 3),
        'Control (D=0) SD':   round(np.sqrt(p2*(1-p2)), 3),
        'Diff (T-C)':         round(p1 - p2, 3),
        "Cohen's d":          round(h, 3),  # 这里实际是 Cohen's h
        't / chi2':           round(chi2, 3) if not np.isnan(chi2) else None,
        'p-value':            round(pval, 4) if not np.isnan(pval) else None,
        'Sig.':               '***' if (not np.isnan(pval) and pval<0.001) else '**' if (not np.isnan(pval) and pval<0.01) else '*' if (not np.isnan(pval) and pval<0.05) else '',
    })

# Y (结果) 单独放最后一行
t = df.loc[df['D_high_engagement']==1, 'Y_score']
c = df.loc[df['D_high_engagement']==0, 'Y_score']
tstat, pval = stats.ttest_ind(t, c, equal_var=False)
rows.append({
    'Variable': 'Y_score (weighted final)',
    'Type': 'Outcome',
    'Treated (D=1) Mean': round(t.mean(), 3),
    'Treated (D=1) SD':   round(t.std(ddof=1), 3),
    'Control (D=0) Mean': round(c.mean(), 3),
    'Control (D=0) SD':   round(c.std(ddof=1), 3),
    'Diff (T-C)':         round(t.mean() - c.mean(), 3),
    "Cohen's d":          round((t.mean()-c.mean())/np.sqrt((t.std(ddof=1)**2+c.std(ddof=1)**2)/2), 3),
    't / chi2':           round(tstat, 3),
    'p-value':            round(pval, 4),
    'Sig.':               '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else '',
})

table1 = pd.DataFrame(rows)
print("\n" + "="*100)
print("Table 1: 描述统计 (处理组 vs 对照组) - PSM 匹配前的差异")
print("="*100)
print(table1.to_string(index=False))

# 导出 CSV
table1.to_csv(OUT / "table1_descriptive.csv", index=False, encoding='utf-8-sig')
print(f"\n[OK] Table 1 已保存 → artifacts/table1_descriptive.csv")

# ============================================================
# 2. Figure 1: 关键变量分布图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 2x2 panels: Y_score, total_clicks, age_mid, edu_ord
panels = [
    ('Y_score',  'Weighted Final Score',          True),
    ('total_clicks', 'Total VLE Clicks (log)',    True),
    ('age_mid',  'Age (mid-point)',               False),
    ('edu_ord',  'Education Level (ordinal)',     False),
]
for ax, (var, label, log_x) in zip(axes.flat, panels):
    t_data = df.loc[df['D_high_engagement']==1, var].dropna()
    c_data = df.loc[df['D_high_engagement']==0, var].dropna()
    if log_x and (t_data > 0).all() and (c_data > 0).all():
        t_data = np.log1p(t_data); c_data = np.log1p(c_data)
    ax.hist(c_data, bins=30, alpha=0.55, label=f'Control (D=0, n={len(c_data)})', color='#4C78A8', density=True)
    ax.hist(t_data, bins=30, alpha=0.55, label=f'Treated (D=1, n={len(t_data)})', color='#F58518', density=True)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Figure 1. Distribution of key variables by treatment status (before PSM)', fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig1_distribution.png", dpi=180, bbox_inches='tight')
print(f"[OK] Figure 1 已保存 → artifacts/fig1_distribution.png")

# ============================================================
# 3. 关键数据点摘要
# ============================================================
summary = {
    "N_total": int(df.shape[0]),
    "N_treated": int((df['D_high_engagement']==1).sum()),
    "N_control": int((df['D_high_engagement']==0).sum()),
    "mean_outcome_treated": float(df.loc[df['D_high_engagement']==1, 'Y_score'].mean()),
    "mean_outcome_control": float(df.loc[df['D_high_engagement']==0, 'Y_score'].mean()),
    "raw_outcome_gap": float(df.loc[df['D_high_engagement']==1, 'Y_score'].mean() - df.loc[df['D_high_engagement']==0, 'Y_score'].mean()),
    "presentations": sorted(df['code_presentation'].unique().tolist()),
    "n_significant_imbalance_p05": int(((table1['p-value'] < 0.05) & (table1['Variable'] != 'Y_score (weighted final)')).sum()),
    "n_imbalanced_vars": int(((table1['p-value'] < 0.05) & (table1['Variable'] != 'Y_score (weighted final)')).sum()),
}
with open(OUT / "table1_summary.json", "w", encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")
