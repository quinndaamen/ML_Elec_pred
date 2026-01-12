from fastapi import FastAPI
from Decision import run_forecast

app = FastAPI(title="Electricity Price Predictor")

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/predict")
async def get_forecast():
    df = run_forecast()
    # convert everything to native types
    records = df.copy()
    for col in records.columns:
        if records[col].dtype.name.startswith("datetime"):
            records[col] = records[col].astype(str)
        elif "float" in records[col].dtype.name or "int" in records[col].dtype.name:
            records[col] = records[col].astype(float)
    return records.to_dict(orient="records")
