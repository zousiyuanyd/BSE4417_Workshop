import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. 读取清洗后的数据
# ============================================================
df = pd.read_csv("/Users/yudian-sunny/Desktop/BSE4417_Workshop/Building_electricial_load_dataset_clean.csv")

# 列名常量
COL_MONTH = "Month"
COL_DAY   = "Day"
COL_HOUR  = "Hour"
COL_T     = "Air Temp._t (degC)"
COL_L1    = "Elec. Load_t-1 (kW)"
COL_L     = "Elec. Load_t (kW)"

# ============================================================
# 2. 构造输入 X 和目标 y（优化：严格按照 组合2 筛选特征，移除了 COL_DT）
# ============================================================
features = [
    COL_MONTH,   # 月份（分类，1-12）
    COL_DAY,     # 星期（分类，1-7）
    COL_HOUR,    # 小时（分类，0-23）
    COL_T,       # 当前时刻气温
    COL_L1,      # 上一时刻负荷
]
target = COL_L

# 检查缺失
assert df[features + [target]].isna().sum().sum() == 0, "数据里还有 NaN，请先清洗"

X = df[features].copy()
y = df[target].copy()

print("样本总数:", len(df))
print("经调优选择的输入特征:", features)
print("目标:", target)
print("-" * 50)

# ============================================================
# 3. 周期性编码（Month / Day / Hour 是周期变量，必须 sin/cos）
# ============================================================
X["Month_sin"] = np.sin(2 * np.pi * X[COL_MONTH] / 12)
X["Month_cos"] = np.cos(2 * np.pi * X[COL_MONTH] / 12)
X["Day_sin"]   = np.sin(2 * np.pi * X[COL_DAY]   / 7)
X["Day_cos"]   = np.cos(2 * np.pi * X[COL_DAY]   / 7)
X["Hour_sin"]  = np.sin(2 * np.pi * X[COL_HOUR]  / 24)
X["Hour_cos"]  = np.cos(2 * np.pi * X[COL_HOUR]  / 24)

# 最终输入列：连续量（剔除了 COL_DT） + 周期编码
feature_cols = [
    COL_T, COL_L1,                     # 连续量
    "Month_sin", "Month_cos",
    "Day_sin",   "Day_cos",
    "Hour_sin",  "Hour_cos",
]
X = X[feature_cols]

print("最终输入列数:", len(feature_cols))
print(feature_cols)
print("-" * 50)

# ============================================================
# 4. 按时间顺序 7:3 划分
# ============================================================
split_idx = int(len(df) * 0.7)

X_train = X.iloc[:split_idx].reset_index(drop=True)
X_test  = X.iloc[split_idx:].reset_index(drop=True)
y_train = y.iloc[:split_idx].reset_index(drop=True)
y_test  = y.iloc[split_idx:].reset_index(drop=True)

print(f"训练集: {len(X_train)} 条 ({len(X_train)/len(df)*100:.1f}%)")
print(f"测试集: {len(X_test)}  条 ({len(X_test)/len(df)*100:.1f}%)")
print("-" * 50)

# ============================================================
# 5. 归一化（只在训练集上 fit，防止泄漏）
# ============================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

# y 也归一化，有助于 MLP 收敛
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# ============================================================
# 6. 建立并训练 MLP（优化：采用最佳胜出拓扑架构 (64, 32)）
# ============================================================
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # 经验证最优的双隐藏层金字塔架构
    activation="relu",
    solver="adam",
    alpha=1e-4,                    # L2 正则
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,           # 自动切一部分训练数据做验证，防过拟合
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=42,
    verbose=False,
)

model.fit(X_train_scaled, y_train_scaled)
print(f"训练完成，迭代次数: {model.n_iter_}")
print("-" * 50)

# ============================================================
# 7. 预测（记得把 y 反归一化）
# ============================================================
y_train_pred = scaler_y.inverse_transform(
    model.predict(X_train_scaled).reshape(-1, 1)
).ravel()

y_test_pred = scaler_y.inverse_transform(
    model.predict(X_test_scaled).reshape(-1, 1)
).ravel()

# ============================================================
# 8. 评估
# ============================================================
def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE：注意 y_true 里有接近 0 的值会爆炸，这里仅参考
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"【{name}】")
    print(f"  MAE  = {mae:.3f} kW")
    print(f"  RMSE = {rmse:.3f} kW")
    print(f"  R2   = {r2:.4f}")
    print(f"  MAPE = {mape:.3f} %")
    print("-" * 50)
    return mae, rmse, r2

evaluate(y_train, y_train_pred, "训练集")
evaluate(y_test,  y_test_pred,  "测试集")

# ============================================================
# 9. 可视化
# ============================================================
# 9.1 测试集：真实 vs 预测（横坐标显示 Month/Day Hour）

plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label="Actual",    color="black", linewidth=1.5)
plt.plot(y_test_pred,   label="Predicted", color="red",   linewidth=1.2)
plt.xlabel("Test sample", fontsize=14)
plt.ylabel("Elec. Load_t (kW)", fontsize=14)
plt.title("ANN Prediction on Test Set (Optimized Model)", fontsize=16)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---- 星期几映射：1-7 → Mon-Sun ----
WEEKDAY_MAP = {1: "Mon", 2: "Tue", 3: "Wed",
               4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# ---- 构造 "MM Wed 02:00" 标签 ----
df_reset = df.reset_index(drop=True)
df_reset["Day_str"] = df_reset[COL_DAY].map(WEEKDAY_MAP)
df_reset["label"] = (
    df_reset[COL_MONTH].astype(str).str.zfill(2) + " " +
    df_reset["Day_str"] + " " +
    df_reset[COL_HOUR].astype(str).str.zfill(2) + ":00"
)

# ---- 取出测试集对应的 label ----
test_label = df_reset["label"].iloc[split_idx:].reset_index(drop=True)

# ---- 画前 7 天 ----
N_HOURS = 24 * 7

plt.figure(figsize=(16, 6))

plt.plot(range(N_HOURS), y_test.values[:N_HOURS],
         label="Actual", color="black", linewidth=1.5)
plt.plot(range(N_HOURS), y_test_pred[:N_HOURS],
         label="Predicted", color="red", linewidth=1.2)

# ---- 横坐标：每 12 小时一个刻度 ----
tick_pos = list(range(0, N_HOURS, 12))
tick_lab = [test_label.iloc[i] for i in tick_pos]
plt.xticks(tick_pos, tick_lab, rotation=45, ha="right", fontsize=10)

plt.xlabel("Day Hour", fontsize=14)
plt.ylabel("Elec. Load_t (kW)", fontsize=14)
plt.title("ANN Prediction on Test Set (first 7 days - Optimized)", fontsize=16)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 9.2 散点图：真实 vs 预测
plt.figure(figsize=(6, 6))
plt.scatter(y_test.values, y_test_pred, s=8, alpha=0.5)
lims = [min(y_test.min(), y_test_pred.min()),
        max(y_test.max(), y_test_pred.max())]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual (kW)", fontsize=12)
plt.ylabel("Predicted (kW)", fontsize=12)
plt.title("Actual vs Predicted (Optimized)", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 9.3 残差分布
residuals = y_test.values - y_test_pred
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=50, color="steelblue", edgecolor="white")
print(f"测试集最终盲测完成！")
plt.xlabel("Residual (kW)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Residual Distribution (Test - Optimized)", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()