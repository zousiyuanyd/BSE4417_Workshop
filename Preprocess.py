import pandas as pd
import numpy as np

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv("/Users/yudian-sunny/Desktop/BSE4417_Workshop/Building_electricial_load_dataset.csv")
df.columns = [c.strip() for c in df.columns]

# ============================================================
# 2. 重命名列（用 degC 代替 ℃）
# ============================================================
rename_map = {
    "Month":                        "Month",
    "Day type":                     "Day",
    "Hour":                         "Hour",
    "airTemperature":               "Air Temp._t (degC)",
    "airTemperature t-1":           "Air Temp._t-1 (degC)",
    "Electricial load t-1 (kW)":    "Elec. Load_t-1 (kW)",
    "Electricial load (kW)":        "Elec. Load_t (kW)",
}
df = df.rename(columns=rename_map)

# 列名常量
COL_MONTH = "Month"
COL_DAY   = "Day"
COL_HOUR  = "Hour"
COL_T     = "Air Temp._t (degC)"
COL_T1    = "Air Temp._t-1 (degC)"
COL_DT    = "delta_Temp. (degC)"
COL_L1    = "Elec. Load_t-1 (kW)"
COL_L     = "Elec. Load_t (kW)"

print("重命名后列顺序：")
print(list(df.columns))
print("-" * 50)

# ============================================================
# 3. 重建 Hour 和 Day
# ============================================================
df[COL_HOUR] = df.index % 24

first_valid_idx = df[COL_DAY].first_valid_index()
first_day_type  = int(df.loc[first_valid_idx, COL_DAY])
day_offset = (df.index - first_valid_idx) // 24
df[COL_DAY] = ((first_day_type - 1 + day_offset) % 7) + 1

df[COL_HOUR] = df[COL_HOUR].astype(int)
df[COL_DAY]  = df[COL_DAY].astype(int)

print("Hour / Day 填补后缺失情况：")
print(df[[COL_HOUR, COL_DAY]].isna().sum())
print("-" * 50)

# ============================================================
# 4. 检测 t 和 t-1 同时缺失（填补前）
# ============================================================
def check_simultaneous_missing(df, col_t, col_t1, name):
    both      = df[col_t].isna() & df[col_t1].isna()
    n_t_only  = (df[col_t].isna()  & df[col_t1].notna()).sum()
    n_t1_only = (df[col_t].notna() & df[col_t1].isna()).sum()
    n_both    = both.sum()
    print(f"【{name}】")
    print(f"  仅 {col_t} 缺失  : {n_t_only}")
    print(f"  仅 {col_t1} 缺失 : {n_t1_only}")
    print(f"  两者同时缺失     : {n_both}")
    if n_both > 0:
        print(f"  同时缺失的 index : {df.index[both].tolist()}")
    print("-" * 50)

check_simultaneous_missing(df, COL_T, COL_T1, "温度")
check_simultaneous_missing(df, COL_L, COL_L1, "负荷")

# ============================================================
# 5. 温度列互相填补
# ============================================================
df[COL_T1] = df[COL_T1].fillna(df[COL_T].shift(1))
df[COL_T]  = df[COL_T].fillna(df[COL_T1].shift(-1))

for _ in range(5):
    before = df[[COL_T, COL_T1]].isna().sum().sum()
    df[COL_T1] = df[COL_T1].fillna(df[COL_T].shift(1))
    df[COL_T]  = df[COL_T].fillna(df[COL_T1].shift(-1))
    after = df[[COL_T, COL_T1]].isna().sum().sum()
    if before == after:
        break

df[COL_T]  = df[COL_T].interpolate(method="linear", limit_direction="both")
df[COL_T1] = df[COL_T1].interpolate(method="linear", limit_direction="both")

print("温度列填补后缺失情况：")
print(df[[COL_T, COL_T1]].isna().sum())
print("-" * 50)

# ============================================================
# 6. 负荷列互相填补
# ============================================================
df[COL_L1] = df[COL_L1].fillna(df[COL_L].shift(1))
df[COL_L]  = df[COL_L].fillna(df[COL_L1].shift(-1))

for _ in range(5):
    before = df[[COL_L, COL_L1]].isna().sum().sum()
    df[COL_L1] = df[COL_L1].fillna(df[COL_L].shift(1))
    df[COL_L]  = df[COL_L].fillna(df[COL_L1].shift(-1))
    after = df[[COL_L, COL_L1]].isna().sum().sum()
    if before == after:
        break

df[COL_L]  = df[COL_L].interpolate(method="linear", limit_direction="both")
df[COL_L1] = df[COL_L1].interpolate(method="linear", limit_direction="both")

print("负荷列填补后缺失情况：")
print(df[[COL_L, COL_L1]].isna().sum())
print("-" * 50)

# ============================================================
# 7. 互换温度两列位置（t-1 在前，t 在后）
# ============================================================
cols = list(df.columns)
i_t  = cols.index(COL_T)
i_t1 = cols.index(COL_T1)
if i_t < i_t1:
    cols[i_t], cols[i_t1] = cols[i_t1], cols[i_t]
    df = df[cols]

print("温度两列互换后列顺序：")
print(list(df.columns))
print("-" * 50)

# ============================================================
# 8. 插入 delta_Temp 列（位置：Air Temp._t 后、Elec. Load_t-1 前）
# ============================================================
df[COL_DT] = df[COL_T] - df[COL_T1]

cols = list(df.columns)
cols.remove(COL_DT)
i_t = cols.index(COL_T)
cols.insert(i_t + 1, COL_DT)
df = df[cols]

print("插入 delta_Temp 后列顺序：")
print(list(df.columns))
print("-" * 50)

# ============================================================
# 9. ★ 统一保留两位小数 + 转 float ★
# ============================================================
numeric_cols = [COL_T, COL_T1, COL_DT, COL_L1, COL_L]

for col in numeric_cols:
    # 先确保是数值类型（防止有字符串/NaN 混入）
    df[col] = pd.to_numeric(df[col], errors="coerce")
    # 保留两位小数，并显式转 float
    df[col] = df[col].round(2).astype(float)

print("数值列保留两位小数后的 dtype：")
print(df[numeric_cols].dtypes)
print("-" * 50)

print("数值列前 10 行预览：")
print(df[numeric_cols].head(10).to_string(index=False))
print("-" * 50)

# ============================================================
# 10. 自检 + 最终检查 + 保存
# ============================================================
check = (df[COL_L].shift(1) - df[COL_L1]).abs().dropna()
print(f"负荷 t-1 与上一行 t 的最大偏差: {check.max():.6f}")
check_t = (df[COL_T].shift(1) - df[COL_T1]).abs().dropna()
print(f"温度 t-1 与上一行 t 的最大偏差: {check_t.max():.6f}")

# delta 校验（保留两位小数后误差可能不为 0，属于正常）
delta_check = (df[COL_T] - df[COL_T1] - df[COL_DT]).abs().max()
print(f"delta_Temp 计算校验最大误差: {delta_check:.6f}")

print("\n最终缺失情况：")
print(df.isna().sum())

print("\n前 10 行预览：")
print(df.head(10).to_string(index=False))

df.to_csv("/Users/yudian-sunny/Desktop/Building_electricial_load_dataset_clean.csv", index=False)
print("\n清洗完成，已保存。")