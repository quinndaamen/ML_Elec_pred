# app.py
from fastapi import FastAPI
from Decision import train_model, forecast_next_hours
import threading
import time

app = FastAPI(title="Electricity Price Predictor")

model_data = {"model": None, "hour_shape": None}

def hourly_trainer():
    while True:
        print("Starting hourly retraining...")
        model, hour_shape = train_model()
        model_data["model"] = model
        model_data["hour_shape"] = hour_shape
        print("Hourly retraining done.")
        # wait until next hour
        time_to_next_hour = 3600 - time.time() % 3600
        time.sleep(time_to_next_hour)

@app.on_event("startup")
def startup_event():
    # First training before accepting requests
    print("Starting first model training...")
    model, hour_shape = train_model()
    model_data["model"] = model
    model_data["hour_shape"] = hour_shape
    print("First model training done.")

    # Start background hourly retraining
    thread = threading.Thread(target=hourly_trainer, daemon=True)
    thread.start()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/predict")
async def get_forecast():
    # Now model is guaranteed to exist
    forecast_df = forecast_next_hours(model_data["model"], model_data["hour_shape"])
    return forecast_df.to_dict(orient="records")
