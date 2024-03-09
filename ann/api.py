import pandas as pd
from fastapi import FastAPI
from ML_Modelle.abstract_model import resolution as resolution_enum
from ML_Modelle.ML_PredictionInterface import (
    ML_PredictionInterface_GradientBoostingModel,
    ML_PredictionInterface_RandomForest,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the ANN group :)))"}


@app.get("/predict/randomForest")
def predict_random_forest(stock_symbols: str, start_date: str, end_date: str, resolution: resolution_enum):
    symbols_list = stock_symbols[1:-1].split(", ")

    random_forest_interface = ML_PredictionInterface_RandomForest()
    prediction = random_forest_interface.predict(symbols_list, pd.to_datetime(
        start_date), pd.to_datetime(end_date), resolution)

    data = {}
    for symbol in symbols_list:
        symbol_prediction = prediction[f"{symbol}_Predicted_Close"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data


@app.get("/predict/gradientBoost")
def predict_gradient_boost(stock_symbols: str, start_date: str, end_date: str, resolution: resolution_enum):
    symbols_list = stock_symbols[1:-1].split(", ")

    gradient_boost_interface = ML_PredictionInterface_GradientBoostingModel()
    prediction = gradient_boost_interface.predict(
        symbols_list, pd.to_datetime(start_date), pd.to_datetime(end_date), resolution)

    data = {}
    for symbol in symbols_list:
        symbol_prediction = prediction[f"{symbol}_Predicted_Close"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data
