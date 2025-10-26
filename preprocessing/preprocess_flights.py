#!/usr/bin/env python3
"""
scaler_flights.py
-----------------
Flight Delays 전처리:
- 결과 특성 수: 38 (bias 미포함)
- One-hot encoding + Standard Scaling(수치 5개만)
- Outputs: X_train.txt, X_val.txt, y_train.txt, y_val.txt
  (이 스크립트는 bias 열 추가/초기 가중치 생성은 하지 않습니다)
"""

import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# [1/6] 데이터 로드
# ------------------------------------------------------------
print("=" * 80)
print("Flight Delays Data Preprocessing (Python version of scaler_flights.R)")
print("=" * 80)
data_path = "/workspace/SGD/data/flights.csv"
usecols = [
    "MONTH", "DAY_OF_WEEK", "AIRLINE",
    "TAXI_OUT", "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "DEPARTURE_DELAY",
]
print(f"[1/6] Loading data from {data_path} ...")
df = pd.read_csv(data_path, usecols=usecols, low_memory=False)
before = len(df)
df = df.dropna()
after = len(df)
print(f"  Records: {after:,} (dropped {before-after:,} NaNs)")

# ------------------------------------------------------------
# [2/6] 원-핫 인코딩(카테고리 고정 포함)
# ------------------------------------------------------------
print("[2/6] One-hot encoding with fixed categories ...")

# 카테고리 고정(항상 동일한 38개 피처가 나오도록)
month_dtype = CategoricalDtype(categories=list(range(1, 13)), ordered=True)
dow_dtype   = CategoricalDtype(categories=list(range(1, 8)), ordered=True)
airlines = ['AA','AS','B6','DL','EV','F9','HA','MQ','NK','OO','UA','US','VX','WN']
air_dtype   = CategoricalDtype(categories=airlines, ordered=False)

df["MONTH"]       = df["MONTH"].astype(month_dtype)
df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype(dow_dtype)
df["AIRLINE"]     = df["AIRLINE"].astype(air_dtype)

# 타깃/피처 분리
y = df["DEPARTURE_DELAY"].to_numpy(dtype=np.float32)
X = df.drop(columns=["DEPARTURE_DELAY"]).copy()

# 더미 생성(구분자 '.'로 고정)
X_enc = pd.get_dummies(
    X,
    columns=["MONTH", "DAY_OF_WEEK", "AIRLINE"],
    prefix=["MONTH", "DAY_OF_WEEK", "AIRLINE"],
    prefix_sep="."
)

# 누락된 더미 열 자동 보완(0으로 채움)
num_cols   = ["TAXI_OUT", "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE"]
month_cols = [f"MONTH.{i}" for i in range(1, 13)]
dow_cols   = [f"DAY_OF_WEEK.{i}" for i in range(1, 8)]
air_cols   = [f"AIRLINE.{a}" for a in airlines]

for col in month_cols + dow_cols + air_cols:
    if col not in X_enc.columns:
        X_enc[col] = 0

# 최종 38개 피처 순서 고정
X_final = pd.concat([X_enc[num_cols], X_enc[month_cols + dow_cols + air_cols]], axis=1)
assert X_final.shape[1] == 38, f"Expected 38 features, got {X_final.shape[1]}"

# ------------------------------------------------------------
# [3/6] Train / Validation split
# ------------------------------------------------------------
print("[3/6] Train/Validation split (70/30) ...")
X_train, X_val, y_train, y_val = train_test_split(
    X_final.to_numpy(dtype=np.float32),
    y,
    test_size=0.30,
    random_state=42,
    shuffle=True
)
print(f"  Train: {X_train.shape[0]:,} rows, Val: {X_val.shape[0]:,} rows, Features: {X_train.shape[1]}")

# ------------------------------------------------------------
# [4/6] 스케일링(수치 5개만 표준화; 더미는 그대로)
# ------------------------------------------------------------
print("[4/6] Scaling numeric features with StandardScaler ...")
scaler = StandardScaler()
# 앞 5개가 수치, 이후는 원-핫(0/1) → 수치만 스케일
X_train_num = X_train[:, :5]
X_val_num   = X_val[:, :5]
X_train[:, :5] = scaler.fit_transform(X_train_num)
X_val[:,   :5] = scaler.transform(X_val_num)

# ------------------------------------------------------------
# [5/6] y 표준화 (예시처럼 RMSE를 1.0대 스케일로 만들기)
# ------------------------------------------------------------
print("[5/6] Standardizing target variable y (DEPARTURE_DELAY) ...")
y_mean = y_train.mean()
y_std  = y_train.std() if y_train.std() != 0 else 1.0

y_train_scaled = (y_train - y_mean) / y_std
y_val_scaled   = (y_val   - y_mean) / y_std

print(f"  y_train: mean={y_mean:.6f}, std={y_std:.6f}")
print(f"  y_train_scaled: mean={y_train_scaled.mean():.6f}, std={y_train_scaled.std():.6f}")
print(f"  y_val_scaled: mean={y_val_scaled.mean():.6f}, std={y_val_scaled.std():.6f}")

# ------------------------------------------------------------
# [6/7] 저장
# ------------------------------------------------------------
print("[6/7] Saving files to /workspace/SGD ...")
out = "/workspace/SGD"
os.makedirs(out, exist_ok=True)

np.savetxt(f"{out}/X_train.txt", X_train, fmt="%.6f")
np.savetxt(f"{out}/y_train.txt", y_train_scaled, fmt="%.6f")
np.savetxt(f"{out}/X_val.txt",   X_val,   fmt="%.6f")
np.savetxt(f"{out}/y_val.txt",   y_val_scaled,   fmt="%.6f")

# y 복원용 통계 저장
with open(f"{out}/y_stats.txt", "w") as f:
    f.write(f"mean={y_mean}\n")
    f.write(f"std={y_std}\n")

# (참고) 피처 이름 저장
with open(f"{out}/feature_names.txt", "w") as f:
    f.write("\n".join(
        num_cols + month_cols + dow_cols + air_cols
    ))

# (참고) 스케일러 파라미터 저장
with open(f"{out}/scaler_params.txt", "w") as f:
    f.write("StandardScaler on numeric 5 features only\n")
    for i, c in enumerate(num_cols):
        f.write(f"{c:20s} mean={scaler.mean_[i]:10.6f}  std={scaler.scale_[i]:10.6f}\n")

# ------------------------------------------------------------
# [7/7] 완료
# ------------------------------------------------------------
print("[7/7] Done.")
print("Outputs:")
print("  - X_train.txt, X_val.txt (features=38; bias NOT included)")
print("  - y_train.txt, y_val.txt (STANDARDIZED target variable)")
print("  - y_stats.txt (mean, std for inverse transformation)")
print("Next: run `preproc_flights.sh` to add bias (make 39 columns) and create b_bh.txt")
print("=" * 80)
