# app.py
from fastapi import FastAPI
from Decision import train_model, forecast_next_hours
import threading
import time

app = FastAPI(title="Electricity Price Predictor")

model_data = {"model": None, "hour_shape": None}

def hourly_trainer():
    while True:
        model, hour_shape = train_model()
        model_data["model"] = model
        model_data["hour_shape"] = hour_shape
        # wait until next hour
        time_to_next_hour = 3600 - time.time() % 3600
        time.sleep(time_to_next_hour)

@app.on_event("startup")
def startup_event():
    # run trainer in background
    thread = threading.Thread(target=hourly_trainer, daemon=True)
    thread.start()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/predict")
async def get_forecast():
    model = model_data["model"]
    hour_shape = model_data["hour_shape"]
    if model is None:
        return {"error": "Model not trained yet. Try again in a few seconds."}
    forecast_df = forecast_next_hours(model, hour_shape)
    return forecast_df.to_dict(orient="records")
