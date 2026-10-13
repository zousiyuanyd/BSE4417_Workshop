# Building Energy Consumption Prediction Using Artificial Neural Networks (ANN)

## Group 6

An end-to-end artificial neural network (ANN) pipeline for short-term building electrical load forecasting. Using historical temperature, lagged load, and cyclical time features, the model is built with `sklearn`'s `MLPRegressor`, covering feature selection, hyperparameter search, model evaluation, and visualization.

---

## Table of Contents

- [Overview](#overview)
- [Data Requirements](#data-requirements)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Feature Engineering & Selection](#2-feature-engineering--selection)
  - [3. Model Development](#3-model-development)
  - [4. Model Evaluation](#4-model-evaluation)
  - [5. Result Visualization](#5-result-visualization)
- [Output Files](#output-files)
- [Configuration](#configuration)

---

## Overview

This project implements an end-to-end short-term load forecasting pipeline:

1. Read and clean raw load/temperature data;
2. Construct cyclical and lagged features, then automatically search for the best feature combination;
3. Search for the best hidden-layer topology;
4. Train the final ANN model and run a multi-seed robustness check;
5. Output error metrics and four visualization plots.

---

## Data Requirements

The script reads `Building_Electrical_Load.csv` from the same directory by default. It must contain at least the following columns (leading/trailing whitespace is stripped before renaming):

| Raw Column | Renamed To | Description |
|---|---|---|
| `Day type` | `Day` | Day type / day of week |
| `airTemperature` | `Air Temp._t (degC)` | Air temperature at time t |
| `airTemperature t-1` | `Air Temp._t-1 (degC)` | Air temperature at time t-1 |
| `Electricial load t-1 (kW)` | `Elec. Load_t-1 (kW)` | Electrical load at time t-1 |
| `Electricial load (kW)` | `Elec. Load_t (kW)` | Electrical load at time t (target) |
| `Month` | `Month` | Month (used for cyclical encoding) |

> The data is assumed to be **hourly** (`Hour` is derived from `index % 24`, and days from `// 24`).

---

## Requirements

- Python 3.8+
- Dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Quick Start

```bash
# 1. Place Building_Electrical_Load.csv in the same directory as the script
# 2. Run the script
python your_script_name.py
```

The script will automatically train, evaluate, and display 4 visualization plots.

---

## Project Structure

```
├── Building_Electrical_Load.csv          # Raw input data
├── Building_Electrical_Load_clean.csv    # Cleaned output data
└── your_script_name.py                   # Main script
```

The main script is organized into five parts:

| Part | Content |
|---|---|
| I. Preprocessing | Read, rename, impute missing values, derive features |
| II. Feature Selection | Cyclical encoding + 3-fold time series CV to search for the best feature combination |
| III. Model Development | Hidden-layer topology search + final model training |
| IV. Model Evaluation | MAE / RMSE / R² / MAPE + multi-seed robustness check |
| V. Result Visualization | 4 plots |

---

## Methodology

### 1. Data Preprocessing

- **Column normalization**: strip whitespace and unify column names.
- **Time feature reconstruction**:
  - `Hour = index % 24`
  - `Day` is recursively derived with a 7-day cycle.
- **Missing value imputation** (temperature and load handled pairwise):
  1. Fill `t-1` using the forward value of `t`;
  2. Fill `t` using the backward value of `t-1`;
  3. Fill remaining gaps with bidirectional interpolation `interpolate(limit_direction="both")`.
- **Derived feature**: `delta_Temp. (degC) = T_t - T_{t-1}`.
- **Numeric formatting**: round to 2 decimals and cast to `float`.
- Cleaned data is saved as `Building_Electrical_Load_clean.csv`.

### 2. Feature Engineering & Selection

`Month` / `Day` / `Hour` are encoded with **sine/cosine cyclical encoding** to avoid misinterpreting periodic patterns as linear:

```
x_sin = sin(2π · x / period)
x_cos = cos(2π · x / period)
```

Candidate feature combinations:

```python
combos = [
    ["Month", "Day", "Hour", L1],
    ["Month", "Day", "Hour", L1, T],
    ["Month", "Day", "Hour", L1, DT],
    ["Month", "Day", "Hour", L1, T, DT],
    ["Month", "Day", "Hour", T, DT],]
```

`TimeSeriesSplit(n_splits=3)` is used for time series cross-validation (only on the first 70% of data), and the best combination is selected by mean R².

### 3. Model Development

- Under the best feature combination, three hidden-layer topologies are compared:

```python
topos = [(32, 16), (64, 32), (128, 64)]
```

- Once the best topology is chosen, the final model is trained on the full training set (first 70%).

Key `MLPRegressor` settings:

| Parameter | Value |
|---|---|
| activation | `relu` |
| solver | `adam` |
| alpha (L2) | `1e-4` |
| learning_rate_init | `1e-3` |
| max_iter | 1500 (search) / 2000 (final) |
| early_stopping | `True` |
| validation_fraction | `0.1` |
| n_iter_no_change | 20 (search) / 30 (final) |
| random_state | 42 |

Features and target are normalized with `MinMaxScaler`; predictions are inverse-transformed afterward.

### 4. Model Evaluation

For both training and test sets, the following metrics are computed:

- **MAE** (mean absolute error, kW)
- **RMSE** (root mean squared error, kW)
- **R²** (coefficient of determination)
- **MAPE** (mean absolute percentage error, %)

A **multi-seed robustness check** is also performed: 5 seeds are randomly drawn from 10–100, the model is retrained, and the mean, standard deviation, min, and max of the test-set R² are reported to assess sensitivity to initial weights.

### 5. Result Visualization

1. **Full test set prediction**: actual vs predicted.
2. **First 7 days prediction**: with `MM Day HH:00` time ticks.
3. **Actual vs predicted scatter**: with a 45° reference line.
4. **Residual distribution histogram**: `Actual - Prediction`.

---

## Output Files

| File | Description |
|---|---|
| `Building_Electrical_Load_clean.csv` | Cleaned data with derived features |

Console output includes:

- CV R² for each feature combination
- CV R² for each hidden-layer topology
- MAE, RMSE, R², MAPE for train/test sets
- Multi-seed robustness statistics (mean ± std, min/max R²)

---

## Configuration

To tune the pipeline, modify the following:

| Location | Description |
|---|---|
| `combos` | Candidate feature combinations |
| `topos` | Candidate hidden-layer topologies |
| `split = int(len(df) * 0.7)` | Train/test split ratio |
| `MLPRegressor(...)` params | Learning rate, regularization, iterations, etc. |
| `seeds` / `rng` | Random seed strategy for robustness check |
| `n = 24 * 7` | Display window length for the second plot |

---

## Notes

- `TimeSeriesSplit` is used so that cross-validation strictly respects temporal order, preventing future information leakage.
- Missing value imputation relies on mutual inference between `t` and `t-1`. If the data has severe gaps at the boundaries, manual inspection is recommended.
- The `Month` column must exist in the raw data; `Day` and `Hour` are generated or completed by the script.