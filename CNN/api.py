import pandas as pd
from fastapi import FastAPI

from src.prediction.abstract_model import resolution as resolutionType
from src.prediction.model_cnn_implementation import ModelExe

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the CNN group :)))"}


@app.get("/predict/")
def predict_cnn(stock_symbol: str, start_date: str, end_date: str, resolution: resolutionType):
    stock_symbols = stock_symbol[1:-1].split(", ")
    cnn_interface = ModelExe()
    prediction = cnn_interface.predict(stock_symbols, pd.to_datetime(start_date),
                                       pd.to_datetime(end_date), resolution)

    data = {}
    for symbol in stock_symbols:
        symbol_prediction = prediction[f"{symbol}"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data
