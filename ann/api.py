import pandas as pd
from fastapi import FastAPI
from ML_Modelle.ML_PredictionInterface import (
    ABC_GradientBoostingModel,
    ABC_LinearRegressionModel,
    ABC_RandomForestModel,
    ABC_SVMModel,
)
from statisticmodels.PredicitonInterface import (
    ArimaInterface,
    ETSInterface,
    NaiveInterface,
    ThetaInterface,
    WindowAverageInterface,
    historicAverageInterface,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the ANN group :)))"}


@app.get("/predict/linearRegression/")
def predict_linearRegression(stock_symbol: str, start_date: str, end_date: str):
    linear_regression_interface = ABC_LinearRegressionModel()

    prediction = linear_regression_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}_Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data
