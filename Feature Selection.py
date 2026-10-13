import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# 1. 读取清洗后的数据
# ============================================================
df = pd.read_csv("/Users/yudian-sunny/Desktop/BSE4417_Workshop/Building_electricial_load_dataset_clean.csv")

# 列名常量
COL_MONTH = "Month"
COL_DAY   = "Day"
COL_HOUR  = "Hour"
COL_T     = "Air Temp._t (degC)"
COL_DT    = "delta_Temp. (degC)"
COL_L1    = "Elec. Load_t-1 (kW)"
COL_L     = "Elec. Load_t (kW)"

target = COL_L

# 检查整体缺失
assert df[[COL_MONTH, COL_DAY, COL_HOUR, COL_T, COL_DT, COL_L1, target]].isna().sum().sum() == 0, "数据里还有 NaN"

# ============================================================
# 2. 严格隔离：只切出训练集进行特征组合搜索（不加入任何测试集）
# ============================================================
split_idx = int(len(df) * 0.7)
df_train_full = df.iloc[:split_idx].reset_index(drop=True)

print(f"参与特征交叉验证的训练集样本数: {len(df_train_full)} 条")
print("-" * 60)

# ============================================================
# 3. 定义 5 组不同的特征组合候选池（已将全特征调至组合 4）
# ============================================================
feature_combinations = [
    {
        "label": "组合1: 时间 + 历史负荷 (基准惯性驱动)",
        "raw_features": [COL_MONTH, COL_DAY, COL_HOUR, COL_L1]
    },
    {
        "label": "组合2: 时间 + 历史负荷 + 当前气温 (加入绝对温度影响)",
        "raw_features": [COL_MONTH, COL_DAY, COL_HOUR, COL_L1, COL_T]
    },
    {
        "label": "组合3: 时间 + 历史负荷 + 温度变化 (加入温度动态趋势)",
        "raw_features": [COL_MONTH, COL_DAY, COL_HOUR, COL_L1, COL_DT]
    },
    {
        "label": "组合4: 完整全特征组合 (同时具备时间、历史负荷、气温及温度变化)",
        "raw_features": [COL_MONTH, COL_DAY, COL_HOUR, COL_L1, COL_T, COL_DT]
    },
    {
        "label": "组合5: 纯环境与时间驱动 (作为对照组：剔除上一时刻负荷惯性)",
        "raw_features": [COL_MONTH, COL_DAY, COL_HOUR, COL_T, COL_DT]
    }
]

# ============================================================
# 4. 定义时间序列交叉验证（3-Fold TimeSeriesSplit）
# ============================================================
tscv = TimeSeriesSplit(n_splits=3)

best_r2 = -float("inf")
best_feature_combo = None

print("开始对不同的【特征组合】进行 3-Fold 交叉验证...\n")

for combo in feature_combinations:
    fold_r2_scores = []
    raw_feats = combo["raw_features"]
    
    # 模拟 3 折滚动切分
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_train_full)):
        df_tr, df_val = df_train_full.iloc[train_idx], df_train_full.iloc[val_idx]
        
        # --- 局部动态构造 X 和 y ---
        X_tr_raw = df_tr[raw_feats].copy()
        X_val_raw = df_val[raw_feats].copy()
        
        y_tr = df_tr[target].copy()
        y_val = df_val[target].copy()
        
        # --- 周期性编码（只有当组合里包含时间特征时才编码） ---
        # 训练折编码
        if COL_MONTH in raw_feats:
            X_tr_raw["Month_sin"] = np.sin(2 * np.pi * X_tr_raw[COL_MONTH] / 12)
            X_tr_raw["Month_cos"] = np.cos(2 * np.pi * X_tr_raw[COL_MONTH] / 12)
        if COL_DAY in raw_feats:
            X_tr_raw["Day_sin"]   = np.sin(2 * np.pi * X_tr_raw[COL_DAY]   / 7)
            X_tr_raw["Day_cos"]   = np.cos(2 * np.pi * X_tr_raw[COL_DAY]   / 7)
        if COL_HOUR in raw_feats:
            X_tr_raw["Hour_sin"]  = np.sin(2 * np.pi * X_tr_raw[COL_HOUR]  / 24)
            X_tr_raw["Hour_cos"]  = np.cos(2 * np.pi * X_tr_raw[COL_HOUR]  / 24)
            
        # 验证折编码
        if COL_MONTH in raw_feats:
            X_val_raw["Month_sin"] = np.sin(2 * np.pi * X_val_raw[COL_MONTH] / 12)
            X_val_raw["Month_cos"] = np.cos(2 * np.pi * X_val_raw[COL_MONTH] / 12)
        if COL_DAY in raw_feats:
            X_val_raw["Day_sin"]   = np.sin(2 * np.pi * X_val_raw[COL_DAY]   / 7)
            X_val_raw["Day_cos"]   = np.cos(2 * np.pi * X_val_raw[COL_DAY]   / 7)
        if COL_HOUR in raw_feats:
            X_val_raw["Hour_sin"]  = np.sin(2 * np.pi * X_val_raw[COL_HOUR]  / 24)
            X_val_raw["Hour_cos"]  = np.cos(2 * np.pi * X_val_raw[COL_HOUR]  / 24)
            
        # 剔除未编码的原始时间离散特征
        drop_cols = [c for c in [COL_MONTH, COL_DAY, COL_HOUR] if c in raw_feats]
        X_tr = X_tr_raw.drop(columns=drop_cols)
        X_val = X_val_raw.drop(columns=drop_cols)
        
        # --- 严格的数据归一化 ---
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_tr_scaled = scaler_X.fit_transform(X_tr)
        X_val_scaled = scaler_X.transform(X_val)
        y_tr_scaled = scaler_y.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
        
        # --- 建立并训练固定参数的 MLP ---
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), # 保持你原本的神经网络层结构
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        
        model.fit(X_tr_scaled, y_tr_scaled)
        
        # --- 预测与 R2 计算 ---
        y_val_pred_scaled = model.predict(X_val_scaled)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
        
        r2 = r2_score(y_val, y_val_pred)
        fold_r2_scores.append(r2)
    
    # 计算当前特征组合的 3 折平均 R2
    avg_r2 = np.mean(fold_r2_scores)
    print(f"【{combo['label']}】")
    print(f"  -> 输入特征包含: {[f for f in raw_feats]}")
    print(f"  -> 3折验证集平均 R2 = {avg_r2:.4f}  (各折得分: {[round(x, 4) for x in fold_r2_scores]})\n")
    
    if avg_r2 > best_r2:
        best_r2 = avg_r2
        best_feature_combo = combo

print("-" * 60)
print("【特征选择完成 - 最佳特征组合结果】")
print(f" 胜出组合: {best_feature_combo['label']}")
print(f" 最终选用的原始输入特征: {best_feature_combo['raw_features']}")
print(f" 最高平均验证集 R2: {best_r2:.4f}")