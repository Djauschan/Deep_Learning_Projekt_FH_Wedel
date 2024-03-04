import pandas as pd
from fastapi import FastAPI
from CNN.prediction.model_exe import ModelExe

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the CNN group :)))"}


@app.get("/predict/")
def predict_cnn(stock_symbol: str, start_date: str, end_date: str, resolution: str):
    cnn_interface = ModelExe()

    prediction = cnn_interface.predict(stock_symbol, pd.to_datetime(start_date),
                                       pd.to_datetime(end_date), resolution)

    prediction.set_index('Timestamp', inplace=True)
    prediction = prediction.astype("Float64")

    prediction_data = prediction[stock_symbol.upper()]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data
