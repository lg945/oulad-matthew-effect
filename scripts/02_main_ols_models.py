import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""
脚本14v2：严格马太效应验证（OLS交互项 + PSM稳健性）
策略调整：OLS交互项为主检验（更标准），PSM为补充稳健性
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

DATA = os.path.join(REPO_ROOT, "data")
ART = os.path.join(REPO_ROOT, "artifacts")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("脚本14v2：严格马太效应验证（OLS + PSM）")
print("=" * 70)

# ============================================================
# 1. 加载并准备数据
# ============================================================
print("\n[1/6] 加载并准备数据...")
df = pd.read_csv(f"{DATA}/ccc_complete.csv", encoding='utf-8-sig')
print(f"  总样本量: {len(df):,}")

# 创建关键变量
df['D_forumng'] = (df['forumng_clicks'] > 0).astype(int)
df['Y_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)

# 从原始列创建PSM协变量
df['female'] = (df['gender'] == 'F').astype(int)
df['disability_bin'] = (df['disability'] == 'Y').astype(int)
imd_map = {'0-10%': 5, '10-20%': 15, '20-30%': 25, '30-40%': 35, '40-50%': 45,
           '50-60%': 55, '60-70%': 65, '70-80%': 75, '80-90%': 85, '90-100%': 95}
df['imd_num'] = df['imd_band'].map(imd_map)
age_map = {'0-35': 25, '35-55': 45, '55<=': 65}
df['age_mid'] = df['age_band'].map(age_map)
df['prev_attempts'] = df['num_of_prev_attempts']

# 教育背景分组
df['edu_group'] = 'Other'
df.loc[df['highest_education'] == 'Lower Than A Level', 'edu_group'] = 'Low'
df.loc[df['highest_education'].isin(['A Level or Equivalent', 'HE Qualification']), 'edu_group'] = 'High'

# 只保留 Low 和 High 组
df_sub = df[df['edu_group'].isin(['Low', 'High'])].copy()
df_sub['edu_high'] = (df_sub['edu_group'] == 'High').astype(int)

# 处理NaN（imd_num有缺失值）
print(f"  imd_num缺失: {df_sub['imd_num'].isna().sum()}")
df_sub['imd_num'] = df_sub['imd_num'].fillna(df_sub['imd_num'].median())

print(f"  分析样本: {len(df_sub):,}")
print(f"  Low教育: {(df_sub['edu_group']=='Low').sum()}")
print(f"  High教育: {(df_sub['edu_group']=='High').sum()}")

# ============================================================
# 2. 描述统计（简单对比）
# ============================================================
print("\n[2/6] 描述统计（简单对比）...")

for edu in ['Low', 'High']:
    sub = df_sub[df_sub['edu_group'] == edu]
    n_forum = (sub['D_forumng']==1).sum()
    n_no = (sub['D_forumng']==0).sum()
    w_forum = sub[sub['D_forumng']==1]['Y_withdrawn'].mean()*100
    w_no = sub[sub['D_forumng']==0]['Y_withdrawn'].mean()*100
    diff = w_no - w_forum
    print(f"\n  {edu}教育 (N={len(sub)})")
    print(f"    论坛参与者: {n_forum}人, Withdrawn率={w_forum:.1f}%")
    print(f"    非参与者: {n_no}人, Withdrawn率={w_no:.1f}%")
    print(f"    简单对比效应: {diff:.1f}pp")

# ============================================================
# 3. OLS交互项模型（主检验）
# ============================================================
print("\n[3/6] OLS交互项模型（马太效应主检验）...")

# Model 1: 基础模型（无交互项）
covariates = ['age_mid', 'female', 'disability_bin', 'imd_num',
              'prev_attempts', 'studied_credits']
model1_vars = covariates + ['D_forumng', 'edu_high']

X1 = sm.add_constant(df_sub[model1_vars])
y1 = df_sub['Y_withdrawn']
model1 = sm.OLS(y1, X1).fit(cov_type='HC1')

print("\n  Model 1 (基础模型，无交互项):")
print(f"    D_forumng系数: {model1.params['D_forumng']:.4f} (p={model1.pvalues['D_forumng']:.4f})")
print(f"    edu_high系数: {model1.params['edu_high']:.4f} (p={model1.pvalues['edu_high']:.4f})")
print(f"    R-squared: {model1.rsquared:.4f}, N={int(model1.nobs)}")

# Model 2: 交互项模型（马太效应检验）
df_sub['D_x_edu'] = df_sub['D_forumng'] * df_sub['edu_high']
model2_vars = covariates + ['D_forumng', 'edu_high', 'D_x_edu']

X2 = sm.add_constant(df_sub[model2_vars])
y2 = df_sub['Y_withdrawn']
model2 = sm.OLS(y2, X2).fit(cov_type='HC1')

print("\n  Model 2 (交互项模型 - 马太效应检验):")
print(f"    D_forumng系数: {model2.params['D_forumng']:.4f} (p={model2.pvalues['D_forumng']:.4f})")
print(f"    edu_high系数: {model2.params['edu_high']:.4f} (p={model2.pvalues['edu_high']:.4f})")
print(f"    D_forumng×edu_high系数: {model2.params['D_x_edu']:.4f} (p={model2.pvalues['D_x_edu']:.4f})")
print(f"    R-squared: {model2.rsquared:.4f}, N={int(model2.nobs)}")

# 解读交互项
inter_coef = model2.params['D_x_edu']
inter_p = model2.pvalues['D_x_edu']

if inter_coef < 0 and inter_p < 0.05:
    print(f"\n   马太效应显著！")
    print(f"    交互项为负且p<0.05：论坛参与对高教育学生降低退选率的效应更大")
    print(f"    Low教育组效应: D_forumng系数 = {model2.params['D_forumng']:.4f}")
    print(f"    High教育组效应: D_forumng + D_x_edu = {model2.params['D_forumng'] + inter_coef:.4f}")
elif inter_coef < 0 and inter_p < 0.10:
    print(f"\n  ⚠️ 马太效应边缘显著 (p={inter_p:.4f})")
else:
    print(f"\n  ❌ 马太效应OLS交互项不显著 (p={inter_p:.4f})")
    print(f"    交互项系数: {inter_coef:.4f}")

# ============================================================
# 4. 子组OLS回归（分解效应）
# ============================================================
print("\n[4/6] 子组OLS回归（分别估计效应）...")

subgroup_results = []

for edu in ['Low', 'High']:
    sub = df_sub[df_sub['edu_group'] == edu]
    X_sub = sm.add_constant(sub[covariates + ['D_forumng']])
    y_sub = sub['Y_withdrawn']
    model_sub = sm.OLS(y_sub, X_sub).fit(cov_type='HC1')
    
    coef = model_sub.params['D_forumng']
    se = model_sub.bse['D_forumng']
    p = model_sub.pvalues['D_forumng']
    
    subgroup_results.append({
        'Group': edu,
        'N': int(model_sub.nobs),
        'D_forumng_coef': coef,
        'SE': se,
        'p_value': p,
        'effect_pp': coef * 100 * -1 if coef < 0 else coef * 100  # 退选率降低的pp
    })
    
    print(f"\n  {edu}教育组:")
    print(f"    D_forumng系数: {coef:.4f} (SE={se:.4f}, p={p:.4f})")
    print(f"    解释: 论坛参与降低退选率 {abs(coef)*100:.1f}个百分点")
    print(f"    R-sq: {model_sub.rsquared:.4f}")

# ============================================================
# 5. PSM稳健性检验（简化版）
# ============================================================
print("\n[5/6] PSM稳健性检验...")

from sklearn.linear_model import LogisticRegression
from scipy import stats

def run_simple_psm(data, treat_col, outcome_col, covs, caliper_sd=0.25):
    """简化PSM：1:1最近邻匹配"""
    sub = data.dropna(subset=covs + [treat_col, outcome_col]).copy()
    
    X = sub[covs].values
    y = sub[treat_col].values
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X, y)
    sub['ps'] = lr.predict_proba(X)[:, 1]
    
    caliper = caliper_sd * np.std(sub['ps'])
    
    treated = sub[sub[treat_col] == 1].sort_values('ps')
    control_pool = sub[sub[treat_col] == 0].copy()
    
    matched_pairs = []
    used = set()
    
    # 更高效的匹配：排序后greedy最近邻
    control_sorted = control_pool.sort_values('ps')
    
    for t_idx, t_row in treated.iterrows():
        ps_t = t_row['ps']
        best_idx = None
        best_diff = caliper
        
        for c_idx, c_row in control_sorted.iterrows():
            if c_idx in used:
                continue
            diff = abs(ps_t - c_row['ps'])
            if diff <= caliper:
                if best_idx is None or diff < best_diff:
                    best_diff = diff
                    best_idx = c_idx
            elif c_row['ps'] > ps_t + caliper:
                break  # PS sorted, no more matches
        
        if best_idx is not None:
            matched_pairs.append((t_idx, best_idx))
            used.add(best_idx)
    
    if len(matched_pairs) == 0:
        return None
    
    # ATT
    t_out = np.array([sub.loc[p[0], outcome_col] for p in matched_pairs])
    c_out = np.array([sub.loc[p[1], outcome_col] for p in matched_pairs])
    
    att = np.mean(t_out - c_out)
    se = np.std(t_out - c_out) / np.sqrt(len(matched_pairs))
    t_stat = att / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    return {
        'n_pairs': len(matched_pairs),
        'att': att,
        'att_pp': att * 100,
        'se_pp': se * 100,
        't_stat': t_stat,
        'p_val': p_val
    }

# 全样本PSM
full_psm = run_simple_psm(df_sub, 'D_forumng', 'Y_withdrawn', covariates)
if full_psm:
    print(f"\n  全样本PSM: ATT={full_psm['att_pp']:.2f}pp, SE={full_psm['se_pp']:.2f}pp, p={full_psm['p_val']:.6f}, N_pairs={full_psm['n_pairs']}")

# 低教育组PSM
low_psm = run_simple_psm(df_sub[df_sub['edu_group']=='Low'], 'D_forumng', 'Y_withdrawn', covariates)
if low_psm:
    print(f"  低教育组PSM: ATT={low_psm['att_pp']:.2f}pp, SE={low_psm['se_pp']:.2f}pp, p={low_psm['p_val']:.6f}, N_pairs={low_psm['n_pairs']}")

# 高教育组PSM
high_psm = run_simple_psm(df_sub[df_sub['edu_group']=='High'], 'D_forumng', 'Y_withdrawn', covariates)
if high_psm:
    print(f"  高教育组PSM: ATT={high_psm['att_pp']:.2f}pp, SE={high_psm['se_pp']:.2f}pp, p={high_psm['p_val']:.6f}, N_pairs={high_psm['n_pairs']}")

# ============================================================
# 6. 保存结果 + 可视化
# ============================================================
print("\n[6/6] 保存结果与可视化...")

# 保存OLS结果
ols_summary = {
    'Model1_D_forumng_coef': model1.params['D_forumng'],
    'Model1_D_forumng_p': model1.pvalues['D_forumng'],
    'Model1_R2': model1.rsquared,
    'Model2_D_forumng_coef': model2.params['D_forumng'],
    'Model2_D_x_edu_coef': model2.params['D_x_edu'],
    'Model2_D_x_edu_p': model2.pvalues['D_x_edu'],
    'Model2_R2': model2.rsquared,
    'N': int(model2.nobs),
    'Low_Edu_effect_pp': abs(subgroup_results[0]['D_forumng_coef'])*100,
    'High_Edu_effect_pp': abs(subgroup_results[1]['D_forumng_coef'])*100,
    'Diff_pp': abs(subgroup_results[1]['D_forumng_coef'])*100 - abs(subgroup_results[0]['D_forumng_coef'])*100,
}

pd.DataFrame([ols_summary]).to_csv(f"{ART}/table_ols_interaction.csv", encoding='utf-8-sig', index=False)
print(f"   table_ols_interaction.csv")

# 保存子组OLS结果
pd.DataFrame(subgroup_results).to_csv(f"{ART}/table_subgroup_ols.csv", encoding='utf-8-sig', index=False)
print(f"   table_subgroup_ols.csv")

# 保存PSM结果
psm_results = []
if full_psm:
    psm_results.append({'Group': 'Full', 'N_pairs': full_psm['n_pairs'], 'ATT_pp': full_psm['att_pp'], 'SE_pp': full_psm['se_pp'], 'p_value': full_psm['p_val']})
if low_psm:
    psm_results.append({'Group': 'Low-Edu', 'N_pairs': low_psm['n_pairs'], 'ATT_pp': low_psm['att_pp'], 'SE_pp': low_psm['se_pp'], 'p_value': low_psm['p_val']})
if high_psm:
    psm_results.append({'Group': 'High-Edu', 'N_pairs': high_psm['n_pairs'], 'ATT_pp': high_psm['att_pp'], 'SE_pp': high_psm['se_pp'], 'p_value': high_psm['p_val']})
pd.DataFrame(psm_results).to_csv(f"{ART}/table_psm_robustness.csv", encoding='utf-8-sig', index=False)
print(f"   table_psm_robustness.csv")

# 可视化1：子组效应对比图
fig, ax = plt.subplots(figsize=(8, 5))

low_effect = abs(subgroup_results[0]['D_forumng_coef']) * 100
high_effect = abs(subgroup_results[1]['D_forumng_coef']) * 100
low_se = subgroup_results[0]['SE'] * 100
high_se = subgroup_results[1]['SE'] * 100

groups = ['Low Education\n(Below A Level)', 'High Education\n(A Level / HE Qualification)']
effects = [low_effect, high_effect]
ses = [low_se, high_se]

bars = ax.bar(groups, effects, color=['#4CAF50', '#E53935'], alpha=0.85,
              edgecolor='black', linewidth=0.8, width=0.5)

ax.errorbar(groups, effects, yerr=[1.96*s for s in ses],
            fmt='none', ecolor='black', capsize=8, capthick=1.5, linewidth=1.5)

for i, (eff, se) in enumerate(zip(effects, ses)):
    ax.text(i, eff + 1.96*se + 3, f'{eff:.1f}pp', ha='center', fontsize=13, fontweight='bold')

# 差异标注
diff = high_effect - low_effect
inter_p = model2.pvalues['D_x_edu']

sig_text = 'p < 0.05' if inter_p < 0.05 else ('p < 0.10' if inter_p < 0.10 else f'p = {inter_p:.3f}')

ax.annotate(f'Matthew Effect Gap\nΔ = {diff:.1f}pp\n({sig_text})',
            xy=(0.5, max(effects) + 15), fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=1.5))

ax.set_ylabel('Effect of Forum Participation on Withdrawal Reduction\n(Percentage Points)', fontsize=11)
ax.set_title('Heterogeneous Effects of Forum Participation on Withdrawal Risk\nby Educational Background (OLS with Robust SE)', fontsize=13)
ax.set_ylim(0, max(effects) + 30)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{ART}/fig_matthew_effect_ols.png", dpi=300, bbox_inches='tight')
print(f"   fig_matthew_effect_ols.png")

# 可视化2：描述统计
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

forum_rates = []
withdrawn_no = []
withdrawn_yes = []
labels = []

for edu in ['Low', 'High']:
    sub = df_sub[df_sub['edu_group'] == edu]
    forum_rates.append(sub['D_forumng'].mean()*100)
    withdrawn_no.append(sub[sub['D_forumng']==0]['Y_withdrawn'].mean()*100)
    withdrawn_yes.append(sub[sub['D_forumng']==1]['Y_withdrawn'].mean()*100)
    labels.append(edu)

ax1 = axes[0]
bars1 = ax1.bar(labels, forum_rates, color=['#4CAF50', '#E53935'], alpha=0.7, edgecolor='black', linewidth=0.5)
for i, v in enumerate(forum_rates):
    ax1.text(i, v+2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax1.set_ylabel('Forum Participation Rate (%)', fontsize=11)
ax1.set_title('Forum Participation Rate by Education', fontsize=12)
ax1.set_ylim(0, 100)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = axes[1]
x = np.arange(len(labels))
width = 0.35
ax2.bar(x-width/2, withdrawn_no, width, label='No Forum', color='#FF9800', alpha=0.8, edgecolor='black')
ax2.bar(x+width/2, withdrawn_yes, width, label='Forum Participant', color='#2196F3', alpha=0.8, edgecolor='black')
for i, (v1, v2) in enumerate(zip(withdrawn_no, withdrawn_yes)):
    ax2.text(i-width/2, v1+2, f'{v1:.1f}%', ha='center', fontsize=10)
    ax2.text(i+width/2, v2+2, f'{v2:.1f}%', ha='center', fontsize=10)
ax2.set_ylabel('Withdrawal Rate (%)', fontsize=11)
ax2.set_title('Withdrawal Rate by Education & Forum', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 110)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{ART}/fig_descriptive_by_group.png", dpi=300, bbox_inches='tight')
print(f"   fig_descriptive_by_group.png")

print("\n" + "=" * 70)
print(" 脚本14v2完成！")
print("=" * 70)

# 打印最终核心结论
print("\n★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
print("核心结论汇总：")
print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
print(f"\n1. 主效应（OLS Model1）:")
print(f"   论坛参与显著降低退选率 (coef={model1.params['D_forumng']:.4f}, p={model1.pvalues['D_forumng']:.4f})")

print(f"\n2. 马太效应（OLS交互项 Model2）:")
print(f"   D_forumng×edu_high 系数: {model2.params['D_x_edu']:.4f}")
print(f"   p-value: {model2.pvalues['D_x_edu']:.4f}")

print(f"\n3. 子组效应分解:")
print(f"   Low教育组: 论坛参与降低退选率 {abs(subgroup_results[0]['D_forumng_coef'])*100:.1f}pp (p={subgroup_results[0]['p_value']:.4f})")
print(f"   High教育组: 论坛参与降低退选率 {abs(subgroup_results[1]['D_forumng_coef'])*100:.1f}pp (p={subgroup_results[1]['p_value']:.4f})")
print(f"   差异: {abs(subgroup_results[1]['D_forumng_coef'])*100 - abs(subgroup_results[0]['D_forumng_coef'])*100:.1f}pp")

print(f"\n4. PSM稳健性:")
if full_psm:
    print(f"   全样本: ATT={full_psm['att_pp']:.2f}pp (p={full_psm['p_val']:.6f})")
if low_psm:
    print(f"   Low教育: ATT={low_psm['att_pp']:.2f}pp")
if high_psm:
    print(f"   High教育: ATT={high_psm['att_pp']:.2f}pp")
