# app.py
import os
import threading
import time
from fastapi import FastAPI
from Decision import train_model, forecast_next_hours

app = FastAPI(title="Electricity Price Predictor")

# Shared model storage
model_data = {"model": None, "hour_shape": None}

def hourly_trainer():
    while True:
        print("[Trainer] Starting hourly retraining...")
        model, hour_shape = train_model()
        model_data["model"] = model
        model_data["hour_shape"] = hour_shape
        print("[Trainer] Hourly retraining done.")
        # wait until the next hour
        time_to_next_hour = 3600 - time.time() % 3600
        time.sleep(time_to_next_hour)

@app.on_event("startup")
def startup_event():
    # Initial training before serving requests
    print("[Startup] Starting first model training...")
    model, hour_shape = train_model()
    model_data["model"] = model
    model_data["hour_shape"] = hour_shape
    print("[Startup] First model training done.")

    # Start background hourly retraining
    thread = threading.Thread(target=hourly_trainer, daemon=True)
    thread.start()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/predict")
async def get_forecast():
    if model_data["model"] is None:
        return {"error": "Model not trained yet. Try again in a few seconds."}

    forecast_df = forecast_next_hours(model_data["model"], model_data["hour_shape"])
    return forecast_df.to_dict(orient="records")

if __name__ == "__main__":
    # Render requires binding to 0.0.0.0 and the port from the environment
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)
