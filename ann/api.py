import pandas as pd
from fastapi import FastAPI
import warnings
warnings.filterwarnings("ignore")

from ML_Modelle.ML_PredictionInterface import ML_PredictionInterface_GradientBoostingModel, ML_PredictionInterface_RandomForest
from ML_Modelle.abstract_model import resolution
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the ANN group :)))"}

@app.get("/predict/randomForest")
def predict_randomForest(stock_symbols: str, start_date: str, end_date: str, resolution : resolution):
    random_forest_interface = ML_PredictionInterface_RandomForest()
    stock_symbols = stock_symbols[1:-1].split(", ")
    prediction = random_forest_interface.predict(stock_symbols, pd.to_datetime(start_date), pd.to_datetime(end_date), resolution)
    
    data = {}
    for symbol in stock_symbols:
        symbol_prediction = prediction[f"{symbol}_Predicted_Close"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data

@app.get("/predict/gradientBoost")
def predict_randomForest(stock_symbols: str, start_date: str, end_date: str, resolution : resolution):
    random_forest_interface = ML_PredictionInterface_GradientBoostingModel()
    stock_symbols = stock_symbols[1:-1].split(", ")
    prediction = random_forest_interface.predict(stock_symbols, pd.to_datetime(start_date), pd.to_datetime(end_date), resolution)
    
    data = {}
    for symbol in stock_symbols:
        symbol_prediction = prediction[f"{symbol}_Predicted_Close"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data

