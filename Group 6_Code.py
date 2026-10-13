import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

## I. Preprocessing

# 1. Read raw data and rename columns
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Building_Electrical_Load.csv")) # Ensure local path is used
df.columns = [c.strip() for c in df.columns] # Strip leading and trailing whitespace from all column names

df = df.rename(columns={
    "Day type":                  "Day",
    "airTemperature":            "Air Temp._t (degC)",
    "airTemperature t-1":        "Air Temp._t-1 (degC)",
    "Electricial load t-1 (kW)": "Elec. Load_t-1 (kW)",
    "Electricial load (kW)":     "Elec. Load_t (kW)",
})

# 2. Fill missing values
# 2.1 Hour and Day
df["Hour"] = df.index % 24
first = df["Day"].first_valid_index()
first_day = int(df.loc[first, "Day"])
df["Day"] = ((first_day - 1 + (df.index - first) // 24) % 7) + 1
# 2.2 Temperature and Load
for a, b in [("Air Temp._t (degC)", "Air Temp._t-1 (degC)"),
             ("Elec. Load_t (kW)",  "Elec. Load_t-1 (kW)")]:
    df[b] = df[b].fillna(df[a].shift(1))
    df[a] = df[a].fillna(df[b].shift(-1))
    df[a] = df[a].interpolate(limit_direction="both")
    df[b] = df[b].interpolate(limit_direction="both")

# 3. Add delta temperature and round to 2 decimals
df["delta_Temp. (degC)"] = (df["Air Temp._t (degC)"] - df["Air Temp._t-1 (degC)"]).round(2)
for c in ["Air Temp._t (degC)", "Air Temp._t-1 (degC)", "delta_Temp. (degC)",
          "Elec. Load_t (kW)", "Elec. Load_t-1 (kW)"]:
    df[c] = df[c].round(2).astype(float)

df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Building_Electrical_Load_clean.csv"), index=False)

## II. Feature Selection

# 4. Define features and encoding
T  = "Air Temp._t (degC)"
L1 = "Elec. Load_t-1 (kW)"
L  = "Elec. Load_t (kW)"
DT = "delta_Temp. (degC)"
## Encoding: Demonstrate cyclical characteristics of Month, Day, Hour
def encode(data, cols):
    X = pd.DataFrame(index=data.index)
    if "Month" in cols:
        X["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
        X["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    if "Day" in cols:
        X["Day_sin"]   = np.sin(2 * np.pi * data["Day"]   / 7)
        X["Day_cos"]   = np.cos(2 * np.pi * data["Day"]   / 7)
    if "Hour" in cols:
        X["Hour_sin"]  = np.sin(2 * np.pi * data["Hour"]  / 24)
        X["Hour_cos"]  = np.cos(2 * np.pi * data["Hour"]  / 24)
    if L1 in cols:
        X[L1] = data[L1]
    if T in cols:
        X[T] = data[T]
    if DT in cols:
        X[DT] = data[DT]
    return X

## 5. Search best feature combination with 3-fold CV
print("\nResults:")

split = int(len(df) * 0.7)
train = df.iloc[:split].reset_index(drop=True)

combos = [
    ["Month", "Day", "Hour", L1],
    ["Month", "Day", "Hour", L1, T],
    ["Month", "Day", "Hour", L1, DT],
    ["Month", "Day", "Hour", L1, T, DT],
    ["Month", "Day", "Hour", T, DT],
]
tscv = TimeSeriesSplit(n_splits=3)
best_r2, best_combo = -1, None
print("\n1. Feature Cross-Validation")
print("  -- Coefficients of determination (R2) for different feature combinations:\n")

for cols in combos:
    scores = []
    for tr_idx, val_idx in tscv.split(train):
        tr, val = train.iloc[tr_idx], train.iloc[val_idx]
        sx, sy = MinMaxScaler(), MinMaxScaler()
        Xtr_s  = sx.fit_transform(encode(tr,  cols))
        Xval_s = sx.transform(   encode(val, cols))
        ytr_s  = sy.fit_transform(tr[[L]].values.reshape(-1, 1)).ravel()

        m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                         max_iter=1500, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=20,
                         random_state=42)
        m.fit(Xtr_s, ytr_s)
        pred = sy.inverse_transform(m.predict(Xval_s).reshape(-1, 1)).ravel()
        scores.append(r2_score(val[L], pred))

    avg = np.mean(scores)
    print(f"{cols} -> R2: {avg:.4f}")
    if avg > best_r2:
        best_r2, best_combo = avg, cols

print(f"\nBest combo: {best_combo}")
print(f"Best R2:    {best_r2:.4f}")
print("-" * 100)

## III. Model Development

# 6. Search best hidden-layer topology
topos = [(32, 16), (64, 32), (128, 64)]
best_top_r2, best_top = -1, None

print("\n2. Hidden-Layer Topology Comparison")
print("  -- Coefficients of determination (R2) for different hidden-layer topologies:\n")

for top in topos:
    scores = []
    for tr_idx, val_idx in tscv.split(train):
        tr, val = train.iloc[tr_idx], train.iloc[val_idx]
        sx, sy = MinMaxScaler(), MinMaxScaler()
        Xtr_s  = sx.fit_transform(encode(tr,  best_combo))
        Xval_s = sx.transform(   encode(val, best_combo))
        ytr_s  = sy.fit_transform(tr[[L]].values.reshape(-1, 1)).ravel()

        m = MLPRegressor(hidden_layer_sizes=top, activation="relu",
                         solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                         max_iter=1500, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=20,
                         random_state=42)
        m.fit(Xtr_s, ytr_s)
        pred = sy.inverse_transform(m.predict(Xval_s).reshape(-1, 1)).ravel()
        scores.append(r2_score(val[L], pred))

    avg = np.mean(scores)
    print(f"topology {top} -> R2: {avg:.4f}")
    if avg > best_top_r2:
        best_top_r2, best_top = avg, top

print(f"\nBest topology: {best_top}")
print(f"Best R2:       {best_top_r2:.4f}")
print("-" * 100)

# 7. Train final model
X = encode(df, best_combo)
y = df[L]
## train-test split
split = int(len(df) * 0.7)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

sx, sy = MinMaxScaler(), MinMaxScaler()
X_train_s = sx.fit_transform(X_train)
X_test_s  = sx.transform(X_test)
y_train_s = sy.fit_transform(y_train.values.reshape(-1, 1)).ravel()
## ANN built-up
model = MLPRegressor(
    hidden_layer_sizes=best_top,
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=42,
)
model.fit(X_train_s, y_train_s)

y_train_pred = sy.inverse_transform(model.predict(X_train_s).reshape(-1, 1)).ravel()
y_test_pred  = sy.inverse_transform(model.predict(X_test_s).reshape(-1, 1)).ravel()

## IV. Model Evaluation

# 8. Evaluate the model
def report(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"[{name}]")
    print(f"MAE:  {mae:.3f} kW")
    print(f"RMSE: {rmse:.3f} kW")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.3f} %")
    print("\n")

print("3. Model Performance:\n")
print("  (1) Evaluation Metrics:\n")
report(y_train, y_train_pred, "Train")
report(y_test,  y_test_pred,  "Test")

# 9. Multi-seed robustness check
rng = np.random.default_rng(seed=2026)
available_seeds = list(range(10, 101))
seeds = rng.choice(available_seeds, size=5, replace=False).tolist()

print("  (2) Robustness:")
print(f"\nRandom selected seeds for initial weights:\n{seeds}")

r2_list = []

for seed in seeds:
    m = MLPRegressor(
        hidden_layer_sizes=best_top,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    m.fit(X_train_s, y_train_s)
    pred = sy.inverse_transform(m.predict(X_test_s).reshape(-1, 1)).ravel()
    r2_list.append(r2_score(y_test, pred))

print("\nConfidence Interval Result:")
print(f"R2 Mean +/- Std: {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}")
print(f"Min R2: {np.min(r2_list):.4f}   Max R2: {np.max(r2_list):.4f}")
print("-" * 100)

## V. Result Visualization

# 10. Plot 1: full test set
plt.figure(figsize=(14, 6))
plt.plot(y_test.to_numpy(), label="Actual", color="black", linewidth=1.5)
plt.plot(y_test_pred, label="Predicted", color="red", linewidth=1.2)
plt.xlabel("Test sample", fontsize=14)
plt.ylabel("Electrical Load (kW)", fontsize=14)
plt.title("ANN Prediction on Test Set", fontsize=16)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 11. Plot 2: first 7 days
weekday = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
d = df.reset_index(drop=True)
labels = (d["Month"].astype(str).str.zfill(2) + " " +
          d["Day"].map(weekday) + " " +
          d["Hour"].astype(str).str.zfill(2) + ":00").iloc[split:].reset_index(drop=True)

n = 24 * 7
plt.figure(figsize=(16, 6))
plt.plot(range(n), y_test.to_numpy()[:n], label="Actual", color="black", linewidth=1.5)
plt.plot(range(n), y_test_pred[:n], label="Predicted", color="red", linewidth=1.2)
pos = list(range(0, n, 12))
plt.xticks(pos, [labels.iloc[i] for i in pos], rotation=45, ha="right", fontsize=10)
plt.xlabel("Timestamp", fontsize=14)
plt.ylabel("Electrical Load (kW)", fontsize=14)
plt.title("ANN Prediction on Test Set (first 7 days)", fontsize=16)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 12. Plot 3: actual vs predicted scatter
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, s=8, alpha=0.5)
lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual (kW)", fontsize=12)
plt.ylabel("Predicted (kW)", fontsize=12)
plt.title("Actual vs Predicted", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 13. Plot 4: residual distribution
residuals = y_test.to_numpy() - y_test_pred
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=50, color="steelblue", edgecolor="white")
plt.xlabel("Residual (kW)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Residual Distribution for Test Set (Actual - Prediction)", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()