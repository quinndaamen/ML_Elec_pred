from fastapi import FastAPI
import Decision  # your existing ML script

app = FastAPI(title="Electricity Price Predictor")

@app.get("/predict")
async def get_forecast():
    # Run your existing script logic and return the results
    forecast_df = Decision.run_forecast()  # we’ll need to wrap your code in a function
    # Convert DataFrame to JSON
    return forecast_df.to_dict(orient="records")
