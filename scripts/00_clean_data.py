import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""
W1.2 — OULAD CCC 子样本构建
==========================
- 读取 7 张 OULAD 表
- 筛选 CCC 模块（编程入门课）
- 按学生-课程-学期（id_student × code_presentation）合并
- 构建处理变量 (D=高频VLE互动)、结果变量 (Y=加权最终成绩)、协变量 (X)
- 输出清洗后的子样本到 data/df_ccc.csv
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 强制 stdout/stderr 用 utf-8，避免 Windows GBK 编码崩溃
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

DATA = Path("data")
OUT  = Path("artifacts")
OUT.mkdir(exist_ok=True, parents=True)

# ============================================================
# 1. 读取 OULAD 全部表
# ============================================================
print("=" * 60)
print("1. 读取 OULAD 表")
print("=" * 60)

df_info     = pd.read_parquet(DATA / "studentInfo.parquet")
df_reg      = pd.read_parquet(DATA / "studentRegistration.parquet")
df_assess   = pd.read_parquet(DATA / "studentAssessment.parquet")
df_vle_meta = pd.read_parquet(DATA / "vle.parquet")
df_courses  = pd.read_parquet(DATA / "courses.parquet")
df_assess_def= pd.read_parquet(DATA / "assessments.parquet")

# 合并 CCC 模组的 VLE（切片文件不含 code_module 列，从文件名恢复）
vle_2014B = pd.read_parquet(DATA / "studentVle_p2014B_.parquet").assign(code_module='CCC', code_presentation='2014B')
vle_2014J = pd.read_parquet(DATA / "studentVle_p2014J_.parquet").assign(code_module='CCC', code_presentation='2014J')
df_vle = pd.concat([vle_2014B, vle_2014J], ignore_index=True)

print(f"studentInfo:        {df_info.shape}")
print(f"studentRegistration:{df_reg.shape}")
print(f"studentAssessment:  {df_assess.shape}")
print(f"vle:                {df_vle.shape}")
print(f"vle_meta:           {df_vle_meta.shape}")
print(f"courses:            {df_courses.shape}")
print(f"assessments:        {df_assess_def.shape}")
print(f"studentVle CCC:     {df_vle.shape}")

# ============================================================
# 2. 筛选 CCC 模块
# ============================================================
print("\n" + "=" * 60)
print("2. 筛选 CCC 模块（编程入门课）")
print("=" * 60)

ccc_info = df_info[df_info['code_module'] == 'CCC'].copy()
ccc_vle  = df_vle[df_vle['code_module'] == 'CCC'].copy()

# studentAssessment 缺 code_module，先 merge 拿评估定义（含 weight）
ccc_assess = df_assess.merge(
    df_assess_def[['code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'weight']],
    on=['id_assessment'],
    how='left',
    suffixes=('', '_def')
)
# 优先使用 merge 自带的 code_module；如果没有就回退到 _def 列
if 'code_module_def' in ccc_assess.columns:
    ccc_assess['code_module'] = ccc_assess['code_module'].fillna(ccc_assess['code_module_def'])
    ccc_assess['code_presentation'] = ccc_assess['code_presentation'].fillna(ccc_assess['code_presentation_def'])
ccc_assess = ccc_assess[ccc_assess['code_module'] == 'CCC'].copy()

print(f"CCC studentInfo:        {ccc_info.shape}")
print(f"CCC studentVle:         {ccc_vle.shape}")
print(f"CCC studentAssessment:  {ccc_assess.shape}")

# 检查 presentation 范围
print(f"\nCCC presentations: {sorted(ccc_info['code_presentation'].unique())}")
print(f"CCC students: {ccc_info['id_student'].nunique()}")

# ============================================================
# 3. 计算每个学生的 VLE 互动统计量
# ============================================================
print("\n" + "=" * 60)
print("3. 计算 VLE 互动统计量")
print("=" * 60)

# 按 (id_student, code_presentation) 聚合
vle_agg = ccc_vle.groupby(['id_student', 'code_presentation']).agg(
    total_clicks  = ('sum_click', 'sum'),
    active_days   = ('date', 'nunique'),
    n_sites       = ('id_site', 'nunique'),
).reset_index()

# 课程长度（天数）
course_len = df_courses.set_index(['code_module', 'code_presentation'])['module_presentation_length'].to_dict()
vle_agg['course_length'] = vle_agg.apply(
    lambda r: course_len.get(('CCC', r['code_presentation']), 270), axis=1
)
vle_agg['clicks_per_day'] = vle_agg['total_clicks'] / vle_agg['course_length']
vle_agg['engagement_ratio'] = vle_agg['active_days'] / vle_agg['course_length']

print(f"VLE 聚合后: {vle_agg.shape}")
print(vle_agg.describe())

# ============================================================
# 4. 计算每个学生的加权最终成绩
# ============================================================
print("\n" + "=" * 60)
print("4. 计算加权最终成绩")
print("=" * 60)

# 把每个评估的 score 按 weight 加权求和
ccc_assess['weighted'] = ccc_assess['score'] * ccc_assess['weight'] / 100.0
score_agg = ccc_assess.groupby(['id_student', 'code_presentation', 'code_module']).agg(
    final_score = ('weighted', 'sum'),
    n_assessments = ('id_assessment', 'nunique'),
).reset_index()

print(f"成绩聚合: {score_agg.shape}")
print(score_agg['final_score'].describe())

# ============================================================
# 5. 合并到主表
# ============================================================
print("\n" + "=" * 60)
print("5. 合并到主表（学生-课程-学期粒度）")
print("=" * 60)

df = ccc_info.merge(
    score_agg[['id_student', 'code_presentation', 'final_score']],
    on=['id_student', 'code_presentation'],
    how='left'
).merge(
    vle_agg,
    on=['id_student', 'code_presentation'],
    how='left'
).merge(
    df_reg[['id_student', 'code_presentation', 'date_registration', 'date_unregistration']],
    on=['id_student', 'code_presentation'],
    how='left'
)

print(f"合并后: {df.shape}")
print(f"列: {list(df.columns)}")

# ============================================================
# 6. 变量构造
# ============================================================
print("\n" + "=" * 60)
print("6. 变量构造")
print("=" * 60)

# 处理变量 D: 高频 VLE 互动（按总点击数中位数切分）
median_clicks = df['total_clicks'].median()
df['D_high_engagement'] = (df['total_clicks'] > median_clicks).astype(int)
print(f"median_clicks = {median_clicks:.0f}")
print(f"处理组占比 = {df['D_high_engagement'].mean():.3f}")

# 结果变量 Y: 加权最终成绩
df['Y_score'] = df['final_score']

# 协变量
# age_band: 0-35=1, 35-55=2, 55<= =3
df['age_mid'] = df['age_band'].map({'0-35': 25, '35-55': 45, '55<=': 60})
df['female'] = (df['gender'] == 'F').astype(int)
df['disability'] = (df['disability'] == 'Y').astype(int)

# highest_education 编码（有序）
edu_map = {
    'No Formal quals': 0,
    'Lower Than A Level': 1,
    'A Level or Equivalent': 2,
    'HE Qualification': 3,
    'Post Graduate Qualification': 4
}
df['edu_ord'] = df['highest_education'].map(edu_map)

# IMD band（贫困指数）数字编码：取区间下界
df['imd_num'] = df['imd_band'].str.replace('%', '', regex=False).str.split('-').str[0]
df['imd_num'] = pd.to_numeric(df['imd_num'], errors='coerce')
# IMD 是贫困指数百分位，数值越大越贫困；重新编码为"富裕度"=100-imd
df['affluence'] = 100 - df['imd_num']

# 注册时点相对课程开始
df['reg_offset'] = df['date_registration']  # 负数=提前注册
df['unreg_flag'] = df['date_unregistration'].notna().astype(int)

# 之前尝试次数
df['prev_attempts'] = df['num_of_prev_attempts']

# 学习负担
df['studied_credits_std'] = (df['studied_credits'] - df['studied_credits'].mean()) / df['studied_credits'].std()

print(f"处理组占比 = {df['D_high_engagement'].mean():.3f}")
print(f"平均最终成绩 = {df['Y_score'].mean():.2f}")

# ============================================================
# 7. 缺失值与样本筛选
# ============================================================
print("\n" + "=" * 60)
print("7. 缺失值处理与样本筛选")
print("=" * 60)

print("缺失值统计：")
print(df[['Y_score', 'total_clicks', 'active_days', 'age_mid', 'edu_ord', 'imd_num', 'prev_attempts']].isna().sum())

# 删去缺失关键变量的行
df_clean = df.dropna(subset=['Y_score', 'total_clicks', 'age_mid', 'edu_ord', 'imd_num', 'prev_attempts']).copy()
print(f"\n清洗前: {df.shape[0]}  清洗后: {df_clean.shape[0]}")

# 排除 Withdrawn 状态的学生（无成绩）
df_clean = df_clean[df_clean['final_result'] != 'Withdrawn'].copy()
print(f"排除 Withdrawn 后: {df_clean.shape[0]}")

# ============================================================
# 8. 保存
# ============================================================
print("\n" + "=" * 60)
print("8. 保存清洗后的子样本")
print("=" * 60)

# 选择需要的列
keep_cols = [
    'id_student', 'code_presentation',
    'D_high_engagement', 'Y_score',
    'total_clicks', 'active_days', 'clicks_per_day', 'engagement_ratio', 'n_sites',
    'age_mid', 'female', 'edu_ord', 'affluence', 'imd_num',
    'disability', 'prev_attempts', 'studied_credits', 'studied_credits_std',
    'reg_offset', 'unreg_flag', 'final_result',
    'region', 'gender', 'age_band', 'highest_education', 'imd_band'
]
df_final = df_clean[keep_cols].reset_index(drop=True)
df_final.to_csv(DATA / 'df_ccc.csv', index=False, encoding='utf-8-sig')
print(f"已保存 data/df_ccc.csv  ({df_final.shape})")

# 记录样本量
sample_log = [
    ("0. CCC raw students (studentInfo)",  len(ccc_info)),
    ("1. Merge VLE → 32K interactions",     df.shape[0]),
    ("2. Drop missing Y/X",                  df_clean.shape[0]),
    ("3. Drop Withdrawn",                    df_final.shape[0]),
]
with open(OUT / "sample_construction.json", "w") as f:
    json.dump(sample_log, f, indent=2)

print("\n样本构造日志：")
for k, v in sample_log:
    print(f"  {k:35s}  N = {v:>7,}")

print("\n[OK] W1.2 data cleaning done")
