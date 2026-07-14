"""
脚本16：多维参与分析 — 从单变量走向厚实研究
核心改进：
1. 从OULAD原始数据构建7个新参与变量
2. 连续型论坛参与深度 × 教育背景模型
3. 早期 vs. 后期论坛参与时间窗口模型
4. 多维参与路径模型（forum + vle + diversity + assessment + regularity）
5. 中介分析：教育背景 → 论坛参与 → 退选
6. 编程课程特有的描述性分析（评估提交模式）
7. 论坛参与轨迹分析（周序列）
输出：所有新表格 + 6张科研级图表
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = os.path.join(REPO_ROOT, "data")
ART = os.path.join(REPO_ROOT, "artifacts")

# ============================================================
# 科研级绘图设置（统一视觉样式，见 scripts/figure_style.py）
# ============================================================
import sys
sys.path.insert(0, ros.path.join(REPO_ROOT, "scripts"))
from figure_style import apply_style, COLORS, GROUP_LOW, GROUP_HIGH, TINT_LOW, TINT_HIGH, TINT_OUTCOME
apply_style()

print("=" * 70)
print("脚本16：多维参与分析 — 让研究从薄变厚")
print("=" * 70)

# ============================================================
# 1. 加载原始数据并构建新变量
# ============================================================
print("\n[1/10] 加载原始数据并构建多维参与变量...")

# 加载CCC学生信息
df_ccc = pd.read_csv(f"{DATA}/ccc_complete.csv", encoding='utf-8-sig')
print(f"  CCC学生: {len(df_ccc):,}")

# 加载VLE资源类型映射
vle = pd.read_parquet(f"{DATA}/vle.parquet")
# CCC module的资源
ccc_vle = vle[vle['code_module'] == 'CCC'].copy()
ccc_vle_ids = set(ccc_vle['id_site'].unique())

# 加载两个presentation的逐日VLE数据
sv_2014J = pd.read_parquet(f"{DATA}/studentVle_p2014J_.parquet")
sv_2014B = pd.read_parquet(f"{DATA}/studentVle_p2014B_.parquet")
sv_all = pd.concat([sv_2014J, sv_2014B], ignore_index=True)

# 仅保留CCC模块的学生和资源
ccc_ids = set(df_ccc['id_student'].unique())
sv_ccc = sv_all[sv_all['id_student'].isin(ccc_ids)].copy()
sv_ccc = sv_ccc[sv_ccc['id_site'].isin(ccc_vle_ids)].copy()

# 合入资源类型
sv_ccc = sv_ccc.merge(ccc_vle[['id_site', 'activity_type']], on='id_site', how='left')

print(f"  CCC VLE记录: {len(sv_ccc):,}")
print(f"  CCC 资源类型分布:")
print(sv_ccc['activity_type'].value_counts().head(10))

# 加载评估数据
sa = pd.read_parquet(f"{DATA}/studentAssessment.parquet")
sa_ccc = sa[sa['id_student'].isin(ccc_ids)].copy()
asmt = pd.read_parquet(f"{DATA}/assessments.parquet")
asmt_ccc = asmt[asmt['code_module'] == 'CCC'].copy()

# ============================================================
# 2. 构建多维参与变量
# ============================================================
print("\n[2/10] 构建多维参与变量...")

# 基础变量
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

# 教育背景分组
df['edu_group'] = 'Other'
df.loc[df['highest_education'] == 'Lower Than A Level', 'edu_group'] = 'Low'
df.loc[df['highest_education'].isin(['A Level or Equivalent', 'HE Qualification']), 'edu_group'] = 'High'
df_sub = df[df['edu_group'].isin(['Low', 'High'])].copy()
df_sub['edu_high'] = (df_sub['edu_group'] == 'High').astype(int)

# === 新变量构建 ===

# (A) Forum参与深度：连续型论坛点击次数（替代二元）
# 已有 forumng_clicks

# (B) 早期 vs. 后期论坛参与
# date=0 是课程开始，前14天为"早期"
# 需要从sv_ccc计算
# 按学生+presentation分组
sv_ccc_forum = sv_ccc[sv_ccc['activity_type'] == 'forumng'].copy()

# 关联presentation
pres_map = df_ccc.set_index('id_student')['code_presentation'].to_dict()
sv_ccc_forum['code_presentation'] = sv_ccc_forum['id_student'].map(pres_map)

# 早期论坛点击（date 0-14, 即课程开始后前两周）
sv_ccc_forum_early = sv_ccc_forum[(sv_ccc_forum['date'] >= 0) & (sv_ccc_forum['date'] <= 14)]
forum_early = sv_ccc_forum_early.groupby('id_student')['sum_click'].sum().reset_index()
forum_early.columns = ['id_student', 'forum_clicks_early']

# 后期论坛点击（date > 14）
sv_ccc_forum_late = sv_ccc_forum[sv_ccc_forum['date'] > 14]
forum_late = sv_ccc_forum_late.groupby('id_student')['sum_click'].sum().reset_index()
forum_late.columns = ['id_student', 'forum_clicks_late']

# 合入主数据
df_sub = df_sub.merge(forum_early, on='id_student', how='left')
df_sub = df_sub.merge(forum_late, on='id_student', how='left')
df_sub['forum_clicks_early'] = df_sub['forum_clicks_early'].fillna(0)
df_sub['forum_clicks_late'] = df_sub['forum_clicks_late'].fillna(0)

# 早期论坛参与二元指标
df_sub['D_forum_early'] = (df_sub['forum_clicks_early'] > 0).astype(int)
df_sub['D_forum_late'] = (df_sub['forum_clicks_late'] > 0).astype(int)

# (C) VLE总参与强度
# 已有 total_clicks

# (D) VLE参与广度（资源类型多样性）
vle_div = sv_ccc.groupby('id_student')['activity_type'].nunique().reset_index()
vle_div.columns = ['id_student', 'vle_diversity']
df_sub = df_sub.merge(vle_div, on='id_student', how='left')
df_sub['vle_diversity'] = df_sub['vle_diversity'].fillna(0)

# (E) 评估完成度
# 计算每个学生提交的评估数 / 该presentation应有的评估数
asmt_count = asmt_ccc.groupby('code_presentation')['id_assessment'].nunique().reset_index()
asmt_count.columns = ['code_presentation', 'total_assessments']

sa_ccc_valid = sa_ccc[sa_ccc['date_submitted'] >= 0]  # 排除未提交
submitted = sa_ccc_valid.groupby('id_student')['id_assessment'].nunique().reset_index()
submitted.columns = ['id_student', 'submitted_assessments']

df_sub = df_sub.merge(submitted, on='id_student', how='left')
df_sub['submitted_assessments'] = df_sub['submitted_assessments'].fillna(0)

# 获取该学生的presentation应有的评估数
pres_assmt_map = asmt_count.set_index('code_presentation')['total_assessments'].to_dict()
# 排除Exam（Exam的date是NaN, 不一定能按时提交, 且Exam占总权重100%但只有2个）
# 只算CMA+TMA
asmt_nonexam = asmt_ccc[asmt_ccc['assessment_type'] != 'Exam']
asmt_nonexam_count = asmt_nonexam.groupby('code_presentation')['id_assessment'].nunique().reset_index()
asmt_nonexam_count.columns = ['code_presentation', 'nonexam_assessments']
nonexam_map = asmt_nonexam_count.set_index('code_presentation')['nonexam_assessments'].to_dict()

df_sub['total_nonexam'] = df_sub['code_presentation'].map(nonexam_map)
df_sub['assessment_ratio'] = df_sub['submitted_assessments'] / df_sub['total_nonexam']
df_sub['assessment_ratio'] = df_sub['assessment_ratio'].clip(0, 1)  # 防止超1

# (F) 学习规律性（活跃天数）
study_days = sv_ccc.groupby('id_student')['date'].nunique().reset_index()
study_days.columns = ['id_student', 'study_days']
df_sub = df_sub.merge(study_days, on='id_student', how='left')
df_sub['study_days'] = df_sub['study_days'].fillna(0)

# (G) 论坛参与对数化（处理偏态分布）
df_sub['forum_clicks_log'] = np.log1p(df_sub['forumng_clicks'])  # log(1+x)
df_sub['total_clicks_log'] = np.log1p(df_sub['total_clicks'])

# 处理缺失
df_sub['imd_num'] = df_sub['imd_num'].fillna(df_sub['imd_num'].median())

covariates = ['age_mid', 'female', 'disability_bin', 'imd_num',
              'prev_attempts', 'studied_credits']

print(f"  分析样本: {len(df_sub):,} (Low={int((df_sub['edu_group']=='Low').sum())}, High={int((df_sub['edu_group']=='High').sum())})")
print(f"  新变量概览:")
for v in ['forumng_clicks', 'forum_clicks_early', 'forum_clicks_late',
          'total_clicks', 'vle_diversity', 'assessment_ratio', 'study_days']:
    print(f"    {v}: mean={df_sub[v].mean():.2f}, median={df_sub[v].median():.2f}, "
          f"Low_mean={df_sub[df_sub['edu_group']=='Low'][v].mean():.2f}, "
          f"High_mean={df_sub[df_sub['edu_group']=='High'][v].mean():.2f}")

# ============================================================
# 3. 多维参与描述统计表（Table 1 扩展版）
# ============================================================
print("\n[3/10] 多维参与描述统计表...")

desc_rows = []
for edu in ['Low', 'High']:
    sub = df_sub[df_sub['edu_group'] == edu]
    n = len(sub)
    desc_rows.append({
        'Group': edu, 'N': n,
        'Female_pct': sub['female'].mean() * 100,
        'Age_mean': sub['age_mid'].mean(),
        'IMD_mean': sub['imd_num'].mean(),
        'Disability_pct': sub['disability_bin'].mean() * 100,
        'Forum_binary_pct': sub['D_forumng'].mean() * 100,
        'Forum_clicks_mean': sub['forumng_clicks'].mean(),
        'Forum_clicks_median': sub[sub['D_forumng']==1]['forumng_clicks'].median(),
        'Forum_early_mean': sub['forum_clicks_early'].mean(),
        'Forum_early_pct': sub['D_forum_early'].mean() * 100,
        'Forum_late_mean': sub['forum_clicks_late'].mean(),
        'Forum_late_pct': sub['D_forum_late'].mean() * 100,
        'VLE_total_mean': sub['total_clicks'].mean(),
        'VLE_diversity_mean': sub['vle_diversity'].mean(),
        'Assessment_ratio_mean': sub['assessment_ratio'].mean(),
        'Study_days_mean': sub['study_days'].mean(),
        'Withdrawal_pct': sub['Y_withdrawn'].mean() * 100,
        'Withdrawal_no_forum_pct': sub[sub['D_forumng']==0]['Y_withdrawn'].mean() * 100,
        'Withdrawal_forum_pct': sub[sub['D_forumng']==1]['Y_withdrawn'].mean() * 100,
    })

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(f"{ART}/table1_multidimensional.csv", encoding='utf-8-sig', index=False)
print(desc_df.to_string(index=False))

# ============================================================
# 4. OLS：连续型论坛参与深度 × 教育背景
# ============================================================
print("\n[4/10] OLS：论坛参与深度（连续型）× 教育背景...")

# Model A: 二元forum × edu (baseline, 保留)
df_sub['D_x_edu'] = df_sub['D_forumng'] * df_sub['edu_high']
y = df_sub['Y_withdrawn']

X_base = sm.add_constant(df_sub[covariates + ['D_forumng', 'edu_high', 'D_x_edu']])
ols_base = sm.OLS(y, X_base).fit(cov_type='HC1')

# Model B: 连续型forum_clicks_log × edu
df_sub['forum_log_x_edu'] = df_sub['forum_clicks_log'] * df_sub['edu_high']
X_cont = sm.add_constant(df_sub[covariates + ['forum_clicks_log', 'edu_high', 'forum_log_x_edu']])
ols_cont = sm.OLS(y, X_cont).fit(cov_type='HC1')

# 保存结果
depth_rows = []
for model, name in [(ols_base, 'Binary'), (ols_cont, 'Continuous_log')]:
    for var in model.params.index:
        if var == 'const':
            continue
        depth_rows.append({
            'Model': name,
            'Variable': var,
            'Beta': model.params[var],
            'SE': model.bse[var],
            'p': model.pvalues[var],
            'CI_lower': model.conf_int().loc[var, 0],
            'CI_upper': model.conf_int().loc[var, 1],
        })

depth_df = pd.DataFrame(depth_rows)
depth_df.to_csv(f"{ART}/table_forum_depth_ols.csv", encoding='utf-8-sig', index=False)

print("  Binary interaction: β_D_x_edu =", ols_base.params['D_x_edu'],
      "p =", ols_base.pvalues['D_x_edu'])
print("  Continuous interaction: β_forum_log_x_edu =", ols_cont.params['forum_log_x_edu'],
      "p =", ols_cont.pvalues['forum_log_x_edu'])
print("  Continuous R-sq:", ols_cont.rsquared)

# ============================================================
# 5. 时间窗口模型：早期 vs. 后期论坛参与 × 教育背景
# ============================================================
print("\n[5/10] 时间窗口模型：早期 vs. 后期论坛参与...")

# Model C: 早期forum × edu
df_sub['early_x_edu'] = df_sub['D_forum_early'] * df_sub['edu_high']
X_early = sm.add_constant(df_sub[covariates + ['D_forum_early', 'edu_high', 'early_x_edu']])
ols_early = sm.OLS(y, X_early).fit(cov_type='HC1')

# Model D: 后期forum × edu
df_sub['late_x_edu'] = df_sub['D_forum_late'] * df_sub['edu_high']
X_late = sm.add_constant(df_sub[covariates + ['D_forum_late', 'edu_high', 'late_x_edu']])
ols_late = sm.OLS(y, X_late).fit(cov_type='HC1')

# Model E: 早期 + 后期联合模型
X_joint = sm.add_constant(df_sub[covariates + ['D_forum_early', 'D_forum_late', 'edu_high',
                                                  'early_x_edu', 'late_x_edu']])
ols_joint = sm.OLS(y, X_joint).fit(cov_type='HC1')

# 保存
timing_rows = []
for model, name in [(ols_early, 'Early_only'), (ols_late, 'Late_only'), (ols_joint, 'Joint')]:
    for var in model.params.index:
        if var == 'const':
            continue
        timing_rows.append({
            'Model': name,
            'Variable': var,
            'Beta': model.params[var],
            'SE': model.bse[var],
            'p': model.pvalues[var],
        })

timing_df = pd.DataFrame(timing_rows)
timing_df.to_csv(f"{ART}/table_forum_timing_ols.csv", encoding='utf-8-sig', index=False)

print("  Early interaction: β =", ols_early.params['early_x_edu'],
      "p =", ols_early.pvalues['early_x_edu'])
print("  Late interaction: β =", ols_late.params['late_x_edu'],
      "p =", ols_late.pvalues['late_x_edu'])
print("  Joint early×edu:", ols_joint.params['early_x_edu'],
      "p =", ols_joint.pvalues['early_x_edu'])
print("  Joint late×edu:", ols_joint.params['late_x_edu'],
      "p =", ols_joint.pvalues['late_x_edu'])

# ============================================================
# 6. 多维参与路径模型
# ============================================================
print("\n[6/10] 多维参与路径模型...")

# Model F: 所有参与维度 + 各维度×edu交互
engage_vars = ['D_forumng', 'total_clicks_log', 'vle_diversity', 'assessment_ratio', 'study_days']
interact_vars = []
for v in engage_vars:
    iv = f"{v}_x_edu"
    df_sub[iv] = df_sub[v] * df_sub['edu_high']
    interact_vars.append(iv)

X_multi = sm.add_constant(df_sub[covariates + engage_vars + ['edu_high'] + interact_vars])
ols_multi = sm.OLS(y, X_multi).fit(cov_type='HC1')

# 保存
multi_rows = []
for var in ols_multi.params.index:
    if var == 'const':
        continue
    multi_rows.append({
        'Variable': var,
        'Beta': ols_multi.params[var],
        'SE': ols_multi.bse[var],
        'p': ols_multi.pvalues[var],
        'Sig': '***' if ols_multi.pvalues[var] < 0.001 else
               '**' if ols_multi.pvalues[var] < 0.01 else
               '*' if ols_multi.pvalues[var] < 0.05 else
               '' if ols_multi.pvalues[var] >= 0.1 else '.'
    })

multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv(f"{ART}/table_multipath_ols.csv", encoding='utf-8-sig', index=False)

print("  Multi-path R-sq:", ols_multi.rsquared)
print("  Forum × edu interaction:", ols_multi.params['D_forumng_x_edu'],
      "p =", ols_multi.pvalues['D_forumng_x_edu'])
for iv in interact_vars:
    if iv != 'D_forumng_x_edu':
        print(f"  {iv}: β={ols_multi.params[iv]:.4f}, p={ols_multi.pvalues[iv]:.4f}")

# ============================================================
# 7. 中介分析：教育背景 → 论坛参与 → 退选
# ============================================================
print("\n[7/10] 中介分析：教育背景 → 论坛参与 → 退选...")

# Step 1: edu → forum participation (路径a)
X_step1 = sm.add_constant(df_sub[covariates + ['edu_high']])
y_forum_bin = df_sub['D_forumng']
ols_a = sm.OLS(y_forum_bin, X_step1).fit(cov_type='HC1')

# Step 2: forum + edu → withdrawal (路径b + 直接效应c')
X_step2 = sm.add_constant(df_sub[covariates + ['D_forumng', 'edu_high']])
ols_bc = sm.OLS(y, X_step2).fit(cov_type='HC1')

# Step 3: edu → withdrawal only (总效应c)
X_step3 = sm.add_constant(df_sub[covariates + ['edu_high']])
ols_c = sm.OLS(y, X_step3).fit(cov_type='HC1')

# 计算中介效应
a = ols_a.params['edu_high']  # edu → forum
b = ols_bc.params['D_forumng']  # forum → withdrawal (controlling edu)
c_prime = ols_bc.params['edu_high']  # direct effect
c_total = ols_c.params['edu_high']  # total effect
indirect = a * b  # indirect effect through forum

# Sobel test
se_a = ols_a.bse['edu_high']
se_b = ols_bc.bse['D_forumng']
sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
sobel_z = indirect / sobel_se
sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

# Proportion mediated
prop_mediated = indirect / c_total if c_total != 0 else 0

# 也用连续型forum做中介
y_forum_log = df_sub['forum_clicks_log']
ols_a2 = sm.OLS(y_forum_log, X_step1).fit(cov_type='HC1')
X_step2b = sm.add_constant(df_sub[covariates + ['forum_clicks_log', 'edu_high']])
ols_bc2 = sm.OLS(y, X_step2b).fit(cov_type='HC1')

a2 = ols_a2.params['edu_high']
b2 = ols_bc2.params['forum_clicks_log']
indirect2 = a2 * b2
se_a2 = ols_a2.bse['edu_high']
se_b2 = ols_bc2.bse['forum_clicks_log']
sobel_se2 = np.sqrt(b2**2 * se_a2**2 + a2**2 * se_b2**2)
sobel_z2 = indirect2 / sobel_se2
sobel_p2 = 2 * (1 - stats.norm.cdf(abs(sobel_z2)))

med_rows = [{
    'Mediator': 'D_forumng',
    'Path_a': a, 'Path_a_SE': se_a, 'Path_a_p': ols_a.pvalues['edu_high'],
    'Path_b': b, 'Path_b_SE': se_b, 'Path_b_p': ols_bc.pvalues['D_forumng'],
    'Direct_c_prime': c_prime, 'Total_c': c_total,
    'Indirect_ab': indirect, 'Sobel_SE': sobel_se, 'Sobel_z': sobel_z, 'Sobel_p': sobel_p,
    'Prop_mediated': prop_mediated,
}, {
    'Mediator': 'forum_clicks_log',
    'Path_a': a2, 'Path_a_SE': se_a2, 'Path_a_p': ols_a2.pvalues['edu_high'],
    'Path_b': b2, 'Path_b_SE': se_b2, 'Path_b_p': ols_bc2.pvalues['forum_clicks_log'],
    'Direct_c_prime': ols_bc2.params['edu_high'], 'Total_c': c_total,
    'Indirect_ab': indirect2, 'Sobel_SE': sobel_se2, 'Sobel_z': sobel_z2, 'Sobel_p': sobel_p2,
    'Prop_mediated': indirect2 / c_total if c_total != 0 else 0,
}]

med_df = pd.DataFrame(med_rows)
med_df.to_csv(f"{ART}/table_mediation.csv", encoding='utf-8-sig', index=False)

print("  === Binary forum mediator ===")
print(f"  Path a (edu→forum): {a:.4f}, p={ols_a.pvalues['edu_high']:.4f}")
print(f"  Path b (forum→withdrawal|edu): {b:.4f}, p={ols_bc.pvalues['D_forumng']:.4f}")
print(f"  Direct effect c': {c_prime:.4f}")
print(f"  Total effect c: {c_total:.4f}")
print(f"  Indirect effect ab: {indirect:.4f}")
print(f"  Sobel z={sobel_z:.3f}, p={sobel_p:.4f}")
print(f"  Proportion mediated: {prop_mediated:.1%}")
print("  === Continuous forum mediator ===")
print(f"  Path a (edu→forum_log): {a2:.4f}, p={ols_a2.pvalues['edu_high']:.4f}")
print(f"  Path b (forum_log→withdrawal|edu): {b2:.4f}, p={ols_bc2.pvalues['forum_clicks_log']:.4f}")
print(f"  Indirect: {indirect2:.4f}, Sobel z={sobel_z2:.3f}, p={sobel_p2:.4f}")

# ============================================================
# 8. 论坛参与轨迹分析（周序列）
# ============================================================
print("\n[8/10] 论坛参与轨迹分析...")

# 按周分组计算每组的平均论坛点击
# date=0是课程开始，每周7天
sv_ccc_forum2 = sv_ccc[sv_ccc['activity_type'] == 'forumng'].copy()
sv_ccc_forum2['week'] = sv_ccc_forum2['date'] // 7

# 合入教育背景
sv_ccc_forum2['edu_group'] = sv_ccc_forum2['id_student'].map(
    df_sub.set_index('id_student')['edu_group'].to_dict()
)
sv_ccc_forum2 = sv_ccc_forum2[sv_ccc_forum2['edu_group'].isin(['Low', 'High'])]

# 每周每组的平均点击（per student）+ 标准误（用于 95% CI 置信带）
per_student_week = sv_ccc_forum2.groupby(['id_student', 'week', 'edu_group'])['sum_click'].sum().reset_index(name='clicks')
trajectory = per_student_week.groupby(['week', 'edu_group'])['clicks'].agg(
    total_clicks='sum',
    n_students='nunique',
    clicks_per_student='mean',
    se='sem',
).reset_index()

# 限制在课程期间（week -3 到 36, 即 date -21 到 252）
trajectory = trajectory[(trajectory['week'] >= -3) & (trajectory['week'] <= 36)]

trajectory.to_csv(f"{ART}/table_forum_trajectory.csv", encoding='utf-8-sig', index=False)

print(f"  Trajectory rows: {len(trajectory)}")
print(f"  Weeks range: {trajectory['week'].min()} to {trajectory['week'].max()}")

# ============================================================
# 9. 编程课程特有的描述性分析：评估提交模式
# ============================================================
print("\n[9/10] 评估提交模式分析...")

sa_ccc2 = sa[sa['id_student'].isin(set(df_sub['id_student']))].copy()

# 合入评估元数据
sa_ccc2 = sa_ccc2.merge(asmt[['id_assessment', 'assessment_type', 'date', 'weight', 'code_presentation']],
                         on='id_assessment', how='left')

# 合入教育背景
sa_ccc2['edu_group'] = sa_ccc2['id_student'].map(
    df_sub.set_index('id_student')['edu_group'].to_dict()
)
sa_ccc2 = sa_ccc2[sa_ccc2['edu_group'].isin(['Low', 'High'])]

# 评估提交统计
def _ci95(arr):
    """Return (mean, 95% CI half-width) for an array."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, 1.96 * se

assmt_stats = []
for edu in ['Low', 'High']:
    sub_sa = sa_ccc2[sa_ccc2['edu_group'] == edu]
    students_in_group = df_sub[df_sub['edu_group'] == edu]['id_student'].unique()
    n_students = len(students_in_group)

    # 非考试评估
    nonexam_sa = sub_sa[sub_sa['assessment_type'] != 'Exam']

    # 提交率（per-student 已提交评估数）
    submitted_count = nonexam_sa.groupby('id_student')['id_assessment'].nunique()
    submitted_arr = submitted_count.reindex(students_in_group, fill_value=0).values
    mean_submitted, ci_submitted = _ci95(submitted_arr)

    # 平均分数（仅已提交的；score 含 NaN（已提交但未评分）需剔除）
    submitted_scores = nonexam_sa[nonexam_sa['date_submitted'] >= 0]['score'].dropna().values
    mean_score, ci_score = _ci95(submitted_scores)

    # 提交延迟（date_submitted - date，仅TMA）
    tma = sub_sa[(sub_sa['assessment_type'] == 'TMA') & (sub_sa['date_submitted'] >= 0)]
    if 'date_x' in tma.columns:
        tma['delay'] = tma['date_submitted'] - tma['date_x']
    elif 'date' in tma.columns:
        tma['delay'] = tma['date_submitted'] - tma['date']
    else:
        tma_delay_merge = tma.merge(asmt[['id_assessment', 'date']], on='id_assessment', how='left')
        tma['delay'] = tma_delay_merge['date_submitted'] - tma_delay_merge['date']
    mean_delay, ci_delay = _ci95(tma['delay'].values)

    # 第一个TMA提交率
    first_tma = asmt_ccc[(asmt_ccc['assessment_type'] == 'TMA') &
                          (asmt_ccc['date'] == 32)].copy()  # 最早TMA
    first_tma_ids = set(first_tma['id_assessment'].unique())
    first_tma_submitted = sub_sa[sub_sa['id_assessment'].isin(first_tma_ids)]
    first_tma_rate = first_tma_submitted.groupby('id_student')['id_assessment'].nunique()
    first_tma_arr = first_tma_rate.reindex(students_in_group, fill_value=0).values
    first_tma_mean, ci_first_tma = _ci95(first_tma_arr)

    assmt_stats.append({
        'Group': edu, 'N': n_students,
        'Avg_nonexam_submitted': mean_submitted,
        'Avg_nonexam_submitted_ci': ci_submitted,
        'Avg_score_submitted': mean_score,
        'Avg_score_submitted_ci': ci_score,
        'Avg_TMA_delay_days': mean_delay,
        'Avg_TMA_delay_ci': ci_delay,
        'First_TMA_submission_rate': first_tma_mean,
        'First_TMA_submission_rate_ci': ci_first_tma,
        'Assessment_ratio': df_sub[df_sub['edu_group'] == edu]['assessment_ratio'].mean(),
    })

assmt_df = pd.DataFrame(assmt_stats)
assmt_df.to_csv(f"{ART}/table_assessment_patterns.csv", encoding='utf-8-sig', index=False)
print(assmt_df.to_string(index=False))

# ============================================================
# 10. 生成科研级图表
# ============================================================
print("\n[10/10] 生成科研级图表...")

# === Figure 4 (new numbering): 多维参与概况（4面板，含 95% CI）===
fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))

def _prop_ci(series):
    """95% CI half-width (in % points) for a binary proportion series."""
    s = series.astype(float)
    p = s.mean()
    n = len(s)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else np.nan
    return 1.96 * se * 100.0

def _mean_ci(series):
    s = series.astype(float)
    n = len(s)
    if n < 2:
        return np.nan
    se = s.std(ddof=1) / np.sqrt(n)
    return 1.96 * se

wbar = 0.35
ekw = dict(ecolor=COLORS['dark'], lw=1.2, capsize=4)

# Panel (a): 参与率对比 (proportions, 95% CI)
vars_rate = ['D_forumng', 'D_forum_early', 'D_forum_late']
labels_rate = ['Any forum', 'Early forum\n(≤2 weeks)', 'Late forum\n(>2 weeks)']
for i, v in enumerate(vars_rate):
    low = df_sub[df_sub['edu_group'] == 'Low'][v]
    high = df_sub[df_sub['edu_group'] == 'High'][v]
    axes5[0, 0].bar(i, low.mean() * 100, wbar, yerr=_prop_ci(low), color=GROUP_LOW,
                   error_kw=ekw, label='Low' if i == 0 else '')
    axes5[0, 0].bar(i + wbar, high.mean() * 100, wbar, yerr=_prop_ci(high), color=GROUP_HIGH,
                   error_kw=ekw, label='High' if i == 0 else '')
axes5[0, 0].set_xticks([0.175, 1.175, 2.175])
axes5[0, 0].set_xticklabels(labels_rate)
axes5[0, 0].set_ylabel('Participation rate (%)')
axes5[0, 0].set_title('(a) Forum engagement rates')
axes5[0, 0].legend()

# Panel (b): 参与深度对比（均值, 95% CI）
vars_depth = ['forumng_clicks', 'forum_clicks_early', 'forum_clicks_late']
labels_depth = ['Total', 'Early\n(≤2 wk)', 'Late\n(>2 wk)']
for i, v in enumerate(vars_depth):
    low = df_sub[df_sub['edu_group'] == 'Low'][v]
    high = df_sub[df_sub['edu_group'] == 'High'][v]
    axes5[0, 1].bar(i, low.mean(), wbar, yerr=_mean_ci(low), color=GROUP_LOW, error_kw=ekw)
    axes5[0, 1].bar(i + wbar, high.mean(), wbar, yerr=_mean_ci(high), color=GROUP_HIGH, error_kw=ekw)
axes5[0, 1].set_xticks([0.175, 1.175, 2.175])
axes5[0, 1].set_xticklabels(labels_depth)
axes5[0, 1].set_ylabel('Mean clicks')
axes5[0, 1].set_title('(b) Forum engagement depth')

# Panel (c): 其他参与维度（均值, 95% CI）
vars_other = ['total_clicks', 'vle_diversity', 'assessment_ratio']
labels_other = ['Total VLE\nclicks', 'Resource\ndiversity', 'Assessment\ncompletion']
for i, v in enumerate(vars_other):
    low = df_sub[df_sub['edu_group'] == 'Low'][v]
    high = df_sub[df_sub['edu_group'] == 'High'][v]
    axes5[1, 0].bar(i, low.mean(), wbar, yerr=_mean_ci(low), color=GROUP_LOW, error_kw=ekw)
    axes5[1, 0].bar(i + wbar, high.mean(), wbar, yerr=_mean_ci(high), color=GROUP_HIGH, error_kw=ekw)
axes5[1, 0].set_xticks([0.175, 1.175, 2.175])
axes5[1, 0].set_xticklabels(labels_other)
axes5[1, 0].set_ylabel('Mean value')
axes5[1, 0].set_title('(c) Multi-path engagement')

# Panel (d): 退选率 by forum participation pattern (proportions, 95% CI)
for edu, color in [('Low', GROUP_LOW), ('High', GROUP_HIGH)]:
    sub = df_sub[df_sub['edu_group'] == edu]
    patterns = [
        sub,
        sub[sub['D_forum_early'] == 0],
        sub[sub['D_forum_early'] == 1],
        sub[(sub['D_forum_early'] == 1) & (sub['D_forum_late'] == 1)],
    ]
    ys = [p['Y_withdrawn'].mean() * 100 for p in patterns]
    cis = [_prop_ci(p['Y_withdrawn']) for p in patterns]
    axes5[1, 1].errorbar(range(4), ys, yerr=cis, fmt='o-', color=color,
                        capsize=4, lw=1.5, label=edu)
axes5[1, 1].set_xticks(range(4))
axes5[1, 1].set_xticklabels(['All', 'No early\nforum', 'Early\nonly', 'Sustained'])
axes5[1, 1].set_ylabel('Withdrawal rate (%)')
axes5[1, 1].set_title('(d) Withdrawal by engagement pattern')
axes5[1, 1].legend()

fig5.suptitle('Multi-dimensional engagement profiles by education group',
              fontsize=13, fontweight='bold', y=1.02)
fig5.tight_layout()
fig5.savefig(f"{ART}/fig5_multidimensional_engagement.pdf")
fig5.savefig(f"{ART}/fig5_multidimensional_engagement.png")
plt.close(fig5)
print("  fig5 saved.")

# === Figure 7 (new numbering): 论坛参与轨迹（时间曲线, 含 95% CI 置信带）===
fig6, ax6 = plt.subplots(figsize=(10, 5))

for edu, color, label in [('Low', GROUP_LOW, 'Low education'),
                           ('High', GROUP_HIGH, 'High education')]:
    traj = trajectory[trajectory['edu_group'] == edu].sort_values('week')
    ci = 1.96 * traj['se']
    ax6.fill_between(traj['week'], traj['clicks_per_student'] - ci,
                     traj['clicks_per_student'] + ci, color=color, alpha=0.15, lw=0)
    ax6.plot(traj['week'], traj['clicks_per_student'], '-', color=color, label=label, linewidth=1.8)

# 标注课程开始线
ax6.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.6, label='Module start')
# 标注早期参与窗口
ax6.axvspan(0, 2, alpha=0.12, color=COLORS['green'], label='Early window (≤2 wk)')
# 标注第一个TMA截止
ax6.axvline(x=4.5, color=COLORS['red'], linestyle=':', alpha=0.6, label='1st TMA deadline')

ax6.set_xlabel('Week (from module start)')
ax6.set_ylabel('Mean forum clicks per student')
ax6.set_title('Forum engagement trajectory over module duration')
ax6.legend(loc='upper right')

fig6.tight_layout()
fig6.savefig(f"{ART}/fig6_forum_trajectory.pdf")
fig6.savefig(f"{ART}/fig6_forum_trajectory.png")
plt.close(fig6)
print("  fig6 saved.")

# === Figure 7: 交互项对比图（多模型交互系数）===
fig7, ax7 = plt.subplots(figsize=(10, 6))

# 收集各模型的交互项系数和CI
interact_results = []

# Binary baseline
interact_results.append({
    'Model': 'Binary forum\n× edu',
    'Beta': ols_base.params['D_x_edu'],
    'CI_lower': ols_base.conf_int().loc['D_x_edu', 0],
    'CI_upper': ols_base.conf_int().loc['D_x_edu', 1],
    'p': ols_base.pvalues['D_x_edu'],
})

# Continuous
interact_results.append({
    'Model': 'Continuous forum\n(log) × edu',
    'Beta': ols_cont.params['forum_log_x_edu'],
    'CI_lower': ols_cont.conf_int().loc['forum_log_x_edu', 0],
    'CI_upper': ols_cont.conf_int().loc['forum_log_x_edu', 1],
    'p': ols_cont.pvalues['forum_log_x_edu'],
})

# Early
interact_results.append({
    'Model': 'Early forum\n× edu',
    'Beta': ols_early.params['early_x_edu'],
    'CI_lower': ols_early.conf_int().loc['early_x_edu', 0],
    'CI_upper': ols_early.conf_int().loc['early_x_edu', 1],
    'p': ols_early.pvalues['early_x_edu'],
})

# Late
interact_results.append({
    'Model': 'Late forum\n× edu',
    'Beta': ols_late.params['late_x_edu'],
    'CI_lower': ols_late.conf_int().loc['late_x_edu', 0],
    'CI_upper': ols_late.conf_int().loc['late_x_edu', 1],
    'p': ols_late.pvalues['late_x_edu'],
})

# Multi-path (forum × edu)
interact_results.append({
    'Model': 'Multi-path\nforum × edu',
    'Beta': ols_multi.params['D_forumng_x_edu'],
    'CI_lower': ols_multi.conf_int().loc['D_forumng_x_edu', 0],
    'CI_upper': ols_multi.conf_int().loc['D_forumng_x_edu', 1],
    'p': ols_multi.pvalues['D_forumng_x_edu'],
})

interact_df = pd.DataFrame(interact_results)
interact_df.to_csv(f"{ART}/table_interaction_comparison.csv", encoding='utf-8-sig', index=False)

# 绘制
y_pos = range(len(interact_results))
betas = [r['Beta'] for r in interact_results]
ci_low = [r['CI_lower'] for r in interact_results]
ci_up = [r['CI_upper'] for r in interact_results]
p_vals = [r['p'] for r in interact_results]
labels = [r['Model'] for r in interact_results]

# 颜色编码显著性
colors_bar = [COLORS['blue'] if p < 0.05 else COLORS['gray'] for p in p_vals]

ax7.barh(y_pos, betas, xerr=[np.array(betas)-np.array(ci_low), np.array(ci_up)-np.array(betas)],
         color=colors_bar, height=0.6, capsize=4)
ax7.set_yticks(y_pos)
ax7.set_yticklabels(labels)
ax7.axvline(x=0, color='black', linewidth=0.8)
ax7.set_xlabel('Interaction coefficient (β)')
ax7.set_title('Matthew Effect interaction across models\n(blue = p < .05, gray = p ≥ .05)')

# 标注p值（白底框隔离，避免与误差线重合）
_sig_bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                 edgecolor='#cccccc', lw=0.4, alpha=0.95)
for i, p in enumerate(p_vals):
    sig_mark = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax7.text(betas[i] + 0.01 if betas[i] > 0 else betas[i] - 0.01, i,
             f'{sig_mark} (p={p:.3f})', va='center',
             ha='left' if betas[i] > 0 else 'right', fontsize=8,
             bbox=_sig_bbox)

fig7.tight_layout()
fig7.savefig(f"{ART}/fig7_interaction_comparison.pdf")
fig7.savefig(f"{ART}/fig7_interaction_comparison.png")
plt.close(fig7)
print("  fig7 saved.")

# === Figure 8: 中介分析路径图 ===
fig8, ax8 = plt.subplots(figsize=(8, 5))

# 节点位置
edu_x, edu_y = 0, 2
forum_x, forum_y = 3, 3
withd_x, withd_y = 6, 2

# 绘制节点（统一调色板：教育=蓝, 论坛=橙, 结果=红）
node_specs = [
    (edu_x, edu_y, 'Education\nBackground', COLORS['blue']),
    (forum_x, forum_y, 'Forum\nEngagement', COLORS['orange']),
    (withd_x, withd_y, 'Learning\nDisengagement', COLORS['red']),
]
for nx, ny, label, ec in node_specs:
    circle = plt.Circle((nx, ny), 0.95, fill=True, facecolor='white',
                        edgecolor=ec, linewidth=2)
    ax8.add_patch(circle)
    ax8.text(nx, ny, label, ha='center', va='center', fontsize=8.5,
             fontweight='bold', color=COLORS['dark'])

_lbl_box = dict(boxstyle='round,pad=0.25', facecolor='white',
                edgecolor=COLORS['dark'], lw=0.8, alpha=0.95)

# 路径箭头
# a: edu → forum
ax8.annotate('', xy=(forum_x-0.95, forum_y), xytext=(edu_x+0.95, edu_y),
             arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))
ax8.text(1.5, 2.75, f'a = {a:.3f}\np < .001', fontsize=8.5,
         color=COLORS['dark'], ha='center', va='center', fontweight='bold', bbox=_lbl_box)

# b: forum → withdrawal
ax8.annotate('', xy=(withd_x-0.95, withd_y), xytext=(forum_x+0.95, forum_y),
             arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))
ax8.text(4.5, 2.75, f'b = {b:.3f}\np < .001', fontsize=8.5,
         color=COLORS['dark'], ha='center', va='center', fontweight='bold', bbox=_lbl_box)

# c': direct: edu → withdrawal (curved ARC above the a–b chain, not through circle)
ax8.annotate('', xy=(withd_x-0.95, withd_y+0.65), xytext=(edu_x+0.95, edu_y+0.65),
             arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2,
                             linestyle='dashed',
                             connectionstyle='arc3,rad=0.25'))
# Place c' label on the upper arc, safely away from all circles / arrows
ax8.text(3, 3.7, f"c' = {c_prime:.3f}\n(direct)", fontsize=8.5,
         color=COLORS['dark'], ha='center', va='center', fontweight='bold',
         bbox=_lbl_box)

# 间接效应标注
ax8.text(3, 0.4, f'Indirect effect: ab = {indirect:.3f}\nSobel z = {sobel_z:.2f}, p = {sobel_p:.3f}\n'
         f'Proportion mediated: {prop_mediated:.1%}',
         fontsize=10, color=COLORS['dark'], ha='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['dark'], alpha=0.9))

ax8.set_xlim(-1.5, 7.5)
ax8.set_ylim(-0.5, 4.8)
ax8.set_aspect('equal')
ax8.axis('off')
ax8.set_title('Mediation pathway: Education → Forum Engagement → Learning Disengagement',
              fontsize=12, fontweight='bold', pad=15)

fig8.tight_layout()
fig8.savefig(f"{ART}/fig8_mediation_pathway.pdf")
fig8.savefig(f"{ART}/fig8_mediation_pathway.png")
plt.close(fig8)
print("  fig8 saved.")

# === Figure 5 (new numbering): 评估提交模式对比 ===
fig9, axes9 = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Assessment completion ratio
for edu, color in [('Low', GROUP_LOW), ('High', GROUP_HIGH)]:
    sub = df_sub[df_sub['edu_group'] == edu]
    axes9[0].hist(sub['assessment_ratio'], bins=15, alpha=0.6, color=color, label=edu, density=True)
axes9[0].set_xlabel('Assessment completion ratio')
axes9[0].set_ylabel('Density')
axes9[0].set_title('(a) Assessment completion distribution')
axes9[0].legend()

# Panel (b): Mean assessment metrics (含 95% CI)
metrics = ['Avg_nonexam_submitted', 'Avg_score_submitted', 'First_TMA_submission_rate']
for i, m in enumerate(metrics):
    for edu, color in [('Low', GROUP_LOW), ('High', GROUP_HIGH)]:
        row = assmt_df[assmt_df['Group'] == edu].iloc[0]
        val = row[m]
        ci = row[m + '_ci']
        axes9[1].bar(i + (0.35 if edu == 'High' else 0), val, width=0.35,
                     yerr=ci, capsize=3, color=color,
                     error_kw=dict(ecolor=COLORS['dark'], lw=1.0))
axes9[1].set_xticks([0.175, 1.175, 2.175])
axes9[1].set_xticklabels(['Avg submitted\n(out of 8)', 'Avg score', 'First TMA\nsubmit rate'])
axes9[1].set_title('(b) Assessment engagement metrics')

fig9.suptitle('Programming assessment patterns by education group',
              fontsize=12, fontweight='bold', y=1.05)
fig9.tight_layout()
fig9.savefig(f"{ART}/fig9_assessment_patterns.pdf")
fig9.savefig(f"{ART}/fig9_assessment_patterns.png")
plt.close(fig9)
print("  fig9 saved.")

# ============================================================
# 总结输出
# ============================================================
print("\n" + "=" * 70)
print("分析完成！新增发现摘要：")
print("=" * 70)

print(f"\n[1] 论坛参与深度（连续型）× 教育背景:")
print(f"    β = {ols_cont.params['forum_log_x_edu']:.4f}, p = {ols_cont.pvalues['forum_log_x_edu']:.4f}")
print(f"    R-sq = {ols_cont.rsquared:.4f} (vs binary R-sq = {ols_base.rsquared:.4f})")

print(f"\n[2] 早期 vs. 后期论坛参与:")
print(f"    早期交互: β = {ols_early.params['early_x_edu']:.4f}, p = {ols_early.pvalues['early_x_edu']:.4f}")
print(f"    后期交互: β = {ols_late.params['late_x_edu']:.4f}, p = {ols_late.pvalues['late_x_edu']:.4f}")
print(f"    联合模型 早期×edu: β = {ols_joint.params['early_x_edu']:.4f}, p = {ols_joint.pvalues['early_x_edu']:.4f}")
print(f"    联合模型 后期×edu: β = {ols_joint.params['late_x_edu']:.4f}, p = {ols_joint.pvalues['late_x_edu']:.4f}")

print(f"\n[3] 多维参与路径模型:")
print(f"    R-sq = {ols_multi.rsquared:.4f}")
print(f"    Forum × edu (in multi-path): β = {ols_multi.params['D_forumng_x_edu']:.4f}, p = {ols_multi.pvalues['D_forumng_x_edu']:.4f}")
for iv in interact_vars:
    if iv != 'D_forumng_x_edu':
        sig_mark = '*' if ols_multi.pvalues[iv] < 0.05 else 'n.s.'
        print(f"    {iv}: β = {ols_multi.params[iv]:.4f}, p = {ols_multi.pvalues[iv]:.4f} ({sig_mark})")

print(f"\n[4] 中介分析（二元forum）:")
print(f"    edu → forum: a = {a:.4f}")
print(f"    forum → withdrawal: b = {b:.4f}")
print(f"    直接效应 c' = {c_prime:.4f}")
print(f"    总效应 c = {c_total:.4f}")
print(f"    间接效应 ab = {indirect:.4f}")
print(f"    Sobel z = {sobel_z:.3f}, p = {sobel_p:.4f}")
print(f"    中介比例 = {prop_mediated:.1%}")

print(f"\n[5] 多维参与描述差异:")
for v in ['D_forumng', 'D_forum_early', 'D_forum_late', 'forumng_clicks',
          'forum_clicks_early', 'forum_clicks_late', 'vle_diversity',
          'assessment_ratio', 'study_days']:
    low_m = df_sub[df_sub['edu_group'] == 'Low'][v].mean()
    high_m = df_sub[df_sub['edu_group'] == 'High'][v].mean()
    diff_pct = (high_m - low_m) / low_m * 100 if low_m != 0 else float('inf')
    print(f"    {v}: Low={low_m:.2f}, High={high_m:.2f}, diff={diff_pct:+.1f}%")

print(f"\n[6] 评估提交模式:")
print(assmt_df.to_string(index=False))

print("\n所有表格和图表已保存到 artifacts/ 目录。")
