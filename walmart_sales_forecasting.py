import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ----------------------------
# 1. Load and prepare data
# ----------------------------
df = pd.read_csv("Walmart.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Sort values for time-based analysis
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# One store for ARIMA example
store_id = 1
df_store = df[df["Store"] == store_id].copy().sort_values("Date").reset_index(drop=True)


# ----------------------------
# 2. ARIMA forecast (Store 1)
# ----------------------------
# Train-test split for store-level series
split_index = int(len(df_store) * 0.8)

train_store = df_store.iloc[:split_index].copy()
test_store = df_store.iloc[split_index:].copy()

train_series = train_store["Weekly_Sales"]
test_series = test_store["Weekly_Sales"]

# Fit ARIMA model
# (1,1,1) is a simple baseline choice for demonstration
arima_model = ARIMA(train_series, order=(1, 1, 1))
arima_fitted = arima_model.fit()

# Forecast the test period
arima_forecast = arima_fitted.forecast(steps=len(test_store))

# Evaluate ARIMA
arima_mae = mean_absolute_error(test_series, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_series, arima_forecast))

print(f"ARIMA MAE: {arima_mae:.2f}")
print(f"ARIMA RMSE: {arima_rmse:.2f}")

# ----------------------------
# 3. Feature engineering for ML
# ----------------------------
df_ml = df.copy()

# Lag features by store
df_ml["lag_1"] = df_ml.groupby("Store")["Weekly_Sales"].shift(1)
df_ml["lag_7"] = df_ml.groupby("Store")["Weekly_Sales"].shift(7)
df_ml["lag_14"] = df_ml.groupby("Store")["Weekly_Sales"].shift(14)

# Rolling features by store
df_ml["rolling_mean_7"] = (
    df_ml.groupby("Store")["Weekly_Sales"]
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)

df_ml["rolling_mean_14"] = (
    df_ml.groupby("Store")["Weekly_Sales"]
    .rolling(14)
    .mean()
    .reset_index(level=0, drop=True)
)

df_ml["rolling_std_7"] = (
    df_ml.groupby("Store")["Weekly_Sales"]
    .rolling(7)
    .std()
    .reset_index(level=0, drop=True)
)

# Calendar features
df_ml["month"] = df_ml["Date"].dt.month
df_ml["week"] = df_ml["Date"].dt.isocalendar().week.astype(int)
df_ml["day_of_week"] = df_ml["Date"].dt.dayofweek

# Interaction feature
df_ml["temp_x_fuel"] = df_ml["Temperature"] * df_ml["Fuel_Price"]

# Drop rows with missing values from lag/rolling creation
df_ml = df_ml.dropna().reset_index(drop=True)


# ----------------------------
# 4. Random Forest on Store 1
# ----------------------------
df_store_ml = df_ml[df_ml["Store"] == store_id].copy().sort_values("Date").reset_index(drop=True)

feature_cols = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "month",
    "week",
    "day_of_week",
    "temp_x_fuel",
]

target_col = "Weekly_Sales"

split_index_ml = int(len(df_store_ml) * 0.8)

train_ml = df_store_ml.iloc[:split_index_ml].copy()
test_ml = df_store_ml.iloc[split_index_ml:].copy()

X_train = train_ml[feature_cols]
X_test = test_ml[feature_cols]
y_train = train_ml[target_col]
y_test = test_ml[target_col]

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"Random Forest MAE: {rf_mae:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")


# ----------------------------
# 5. Plot Random Forest results
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(test_ml["Date"], y_test.values, label="Actual")
plt.plot(test_ml["Date"], rf_pred, label="Random Forest Predicted")
plt.title("Random Forest: Actual vs Predicted Sales (Store 1)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ----------------------------
# 6. Compare ARIMA vs Random Forest
# ----------------------------
comparison_dates = test_store["Date"].reset_index(drop=True)

# Align actual values from ARIMA split
comparison_df = pd.DataFrame({
    "Date": comparison_dates,
    "Actual": test_series.reset_index(drop=True),
    "ARIMA_Pred": pd.Series(arima_forecast).reset_index(drop=True)
})

# Align RF predictions by joining on date
rf_compare = test_ml[["Date"]].copy().reset_index(drop=True)
rf_compare["RF_Pred"] = rf_pred

comparison_df = comparison_df.merge(rf_compare, on="Date", how="left")

plt.figure(figsize=(12, 6))
plt.plot(comparison_df["Date"], comparison_df["Actual"], label="Actual", linewidth=2)
plt.plot(comparison_df["Date"], comparison_df["ARIMA_Pred"], label="ARIMA Forecast", linewidth=2)
plt.plot(comparison_df["Date"], comparison_df["RF_Pred"], label="Random Forest Forecast", linewidth=2)

plt.title("ARIMA vs Random Forest Forecast (Store 1)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()