import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Read data
df = pd.read_csv("/Users/yudian-sunny/Desktop/Building_electricial_load_dataset.csv")

# 2. Select model inputs and output
features = ["airTemperature"]

target = "Electricial load (kW)"

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

X = df[features]
y = df[target]

# 3. Split data in time order
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Apply Min-Max normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create the MLP model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=1000,
    random_state=42
)

# 6. Train the model
model.fit(X_train_scaled, y_train)

# 7. Predict the test set
y_pred = model.predict(X_test_scaled)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.3f} kW")
print(f"RMSE: {rmse:.3f} kW")
print(f"R2:   {r2:.3f}")

# 9. Plot actual and predicted load
plt.figure(figsize=(14, 6))

plt.plot(
    y_test.to_numpy(),
    label="Actual load",
    color="black"
)

plt.plot(
    y_pred,
    label="MLP prediction",
    color="red"
)

plt.xlabel("Test sample", fontsize=18)
plt.ylabel("Electrical load (kW)", fontsize=18)
plt.title("MLP Electrical Load Prediction", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()