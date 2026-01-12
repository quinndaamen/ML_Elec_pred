from fastapi import FastAPI
from Decision import run_forecast

app = FastAPI(title="Electricity Price Predictor")

@app.get("/predict")
async def get_forecast():
    forecast_df = run_forecast()
    return forecast_df.to_dict(orient="records")
