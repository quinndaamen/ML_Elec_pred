# decision.py
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import os

def run_forecast():
    # -------------------------
    # CONFIG
    # -------------------------
    
    CSV_PATH = os.path.join(os.path.dirname(__file__), "Netherlands.csv")

    BZN = "NL"
    HOURS_BACK = 96
    PRED_HOURS = 12
    PRICE_COL = "Price (EUR/MWhe)"
    UTC_OFFSET = 1  # Netherlands local time

    # -------------------------
    # 1. LOAD HISTORICAL CSV
    # -------------------------
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)").reset_index(drop=True)
    df["hour"] = df["Datetime (UTC)"].dt.hour

    # Rolling baseline & hourly shape
    df["roll_mean_72"] = df[PRICE_COL].rolling(72).mean()
    df["hour_rel"] = df[PRICE_COL] / df["roll_mean_72"]
    hour_shape = df.dropna().groupby("hour")["hour_rel"].mean()

    # -------------------------
    # 2. ML RESIDUAL MODEL
    # -------------------------
    for lag in range(1, 25):
        df[f"lag_{lag}"] = df[PRICE_COL].shift(lag)

    df["residual"] = df[PRICE_COL] - df["roll_mean_72"] * df["hour"].map(hour_shape)
    df = df.dropna().reset_index(drop=True)

    features = [f"lag_{i}" for i in range(1, 25)]
    X = df[features]
    y = df["residual"]

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

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

    # -------------------------
    # 3. FETCH LIVE DATA (past prices only)
    # -------------------------
    api = requests.get("https://api.energy-charts.info/price", params={"bzn": BZN}).json()
    live = pd.DataFrame({
        "Datetime (UTC)": pd.to_datetime(api["unix_seconds"], unit="s"),
        "price": api["price"]
    }).sort_values("Datetime (UTC)").reset_index(drop=True)

    live["Datetime (UTC)"] = pd.to_datetime(api["unix_seconds"], unit="s", utc=True)
    live = live[live["Datetime (UTC)"] <= pd.Timestamp.utcnow()]  # both UTC now tz-aware

    baseline = live["price"].tail(72).mean()
    history = live["price"].tolist()

    # Start forecast from **current UTC hour**
    current_time = pd.Timestamp.utcnow().floor("h")

    # -------------------------
    # 4. FORECAST NEXT 12 HOURS
    # -------------------------
    preds = []
    times = []

    for step in range(12):  # PRED_HOURS
        hour = current_time.hour
        shape = hour_shape.get(hour, 1.0)

        # 24 lag features from history
        lags = history[-24:]
        X_input = pd.DataFrame([{f"lag_{i+1}": lags[-(i+1)] for i in range(24)}])
        residual = model.predict(xgb.DMatrix(X_input))[0]

        price_pred = baseline * shape + residual
        preds.append(price_pred)
        times.append(current_time + pd.Timedelta(hours=UTC_OFFSET))  # NL local
        history.append(price_pred)
        current_time += pd.Timedelta(hours=1)

    # -------------------------
    # 5. RETURN RESULTS
    # -------------------------
    forecast_df = pd.DataFrame({
        "Datetime_Local": times,
        "Predicted_Price": preds
    })

    return forecast_df
