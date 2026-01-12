# Decision.py
import pandas as pd
import xgboost as xgb
import os
import requests

def train_model():
    # Load CSV, create features, train XGBoost (same as your current code)
    CSV_PATH = os.path.join(os.getcwd(), "Netherlands.csv")
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)").reset_index(drop=True)
    df["hour"] = df["Datetime (UTC)"].dt.hour

    df["roll_mean_72"] = df["Price (EUR/MWhe)"].rolling(72).mean()
    df["hour_rel"] = df["Price (EUR/MWhe)"] / df["roll_mean_72"]
    hour_shape = df.dropna().groupby("hour")["hour_rel"].mean()

    for lag in range(1, 25):
        df[f"lag_{lag}"] = df["Price (EUR/MWhe)"].shift(lag)

    df["residual"] = df["Price (EUR/MWhe)"] - df["roll_mean_72"] * df["hour"].map(hour_shape)
    df = df.dropna().reset_index(drop=True)

    features = [f"lag_{i}" for i in range(1, 25)]
    X = df[features]
    y = df["residual"]

    split = int(len(X) * 0.8)
    dtrain = xgb.DMatrix(X.iloc[:split], label=y.iloc[:split])
    dval = xgb.DMatrix(X.iloc[split:], label=y.iloc[split:])

    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": 42
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    return model, hour_shape

def forecast_next_hours(model, hour_shape):
    # Fetch live data, compute baseline, predict next 12 hours (same as your current code)
    api = requests.get("https://api.energy-charts.info/price", params={"bzn": "NL"}).json()
    live = pd.DataFrame({
        "Datetime (UTC)": pd.to_datetime(api["unix_seconds"], unit="s"),
        "price": api["price"]
    }).sort_values("Datetime (UTC)").reset_index(drop=True)

    live["Datetime (UTC)"] = pd.to_datetime(api["unix_seconds"], unit="s", utc=True)
    live = live[live["Datetime (UTC)"] <= pd.Timestamp.utcnow()]

    baseline = live["price"].tail(72).mean()
    history = live["price"].tolist()
    current_time = pd.Timestamp.utcnow().floor("h")

    preds = []
    times = []

    for step in range(12):
        hour = current_time.hour
        shape = hour_shape.get(hour, 1.0)
        lags = history[-24:]
        X_input = pd.DataFrame([{f"lag_{i+1}": lags[-(i+1)] for i in range(24)}])
        residual = model.predict(xgb.DMatrix(X_input))[0]

        price_pred = baseline * shape + residual
        preds.append(price_pred)
        times.append(current_time + pd.Timedelta(hours=1))
        history.append(price_pred)
        current_time += pd.Timedelta(hours=1)

    forecast_df = pd.DataFrame({
        "Datetime_Local": times,
        "Predicted_Price": preds
    })
    return forecast_df
