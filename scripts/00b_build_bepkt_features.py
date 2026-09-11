"""Build the compact per-user BePKT feature table used by the cross-dataset scripts.

Run this only if you have obtained the raw BePKT submission log
(`data/bepkt/raw_data/submission.csv`, ~203 MB). It writes
`data/bepkt/bepkt_user_features.csv`, the small table that
06_bepkt_replication.py / 06b_verify_bepkt.py / 06c_bepkt_psm.py fall back to when
the raw log is not present.

The aggregation below is identical to the one embedded in
06_bepkt_replication.py, so both paths produce the same numbers.
"""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "data", "bepkt", "raw_data")
OUT = os.path.join(REPO_ROOT, "data", "bepkt", "bepkt_user_features.csv")

sub = pd.read_csv(os.path.join(RAW, "submission.csv"))
sub["create_time"] = pd.to_datetime(sub["create_time"], utc=True)
prob = pd.read_csv(os.path.join(RAW, "problem.csv"))
sub = sub.merge(prob[["id", "difficulty"]], left_on="problem_id", right_on="id",
                how="left", suffixes=("", "_prob"))

user = sub.groupby("user_id").agg(
    total_submissions=("id", "count"),
    first_sub_time=("create_time", "min"),
    last_sub_time=("create_time", "max"),
    unique_problems_attempted=("problem_id", "nunique"),
    accepted_problems=("problem_id", lambda x: x[sub.loc[x.index, "result"] == 0].nunique()),
    accepted_count=("result", lambda x: (x == 0).sum()),
    unique_active_days=("create_time", lambda x: x.dt.date.nunique()),
    mid_high_attempted=("difficulty", lambda x: (x.isin(["Mid", "High"])).sum()),
).reset_index()

first_sub = sub.sort_values(["user_id", "create_time"]).groupby("user_id").first().reset_index()
user["first_result"] = first_sub.set_index("user_id").loc[user["user_id"], "result"].values
user["D_prior"] = (user["first_result"] == 0).astype(int)
user["acceptance_rate"] = user["accepted_count"] / user["total_submissions"]
user["Y_low_completion"] = (user["accepted_problems"] < 5).astype(int)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
user.to_csv(OUT, index=False)
print(f"N = {len(user)}")
print(f"D_prior High = {int(user.D_prior.sum())} ({user.D_prior.mean()*100:.1f}%)")
print("written:", OUT)
