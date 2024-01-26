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


@app.get("/predict/arima")
def predict_arima(stock_symbol: str, start_date: str, end_date: str):
    arima_interface = ArimaInterface()

    prediction = arima_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
    prediction.set_index('ds', inplace=True)

    prediction = prediction.astype("Float64")

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/ETS")
def predict_ets(stock_symbol: str, start_date: str, end_date: str):
    ets_interface = ETSInterface()

    prediction = ets_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
    prediction.set_index('ds', inplace=True)

    prediction = prediction.astype("Float64")

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/historicAverage")
def predict_historicAverage(stock_symbol: str, start_date: str, end_date: str):
    historic_average_interface = historicAverageInterface()

    prediction = historic_average_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
    prediction.set_index('ds', inplace=True)

    prediction = prediction.astype("Float64")

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/theta")
def predict_theta(stock_symbol: str, start_date: str, end_date: str):
    theta_interface = ThetaInterface()

    prediction = theta_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
    prediction.set_index('ds', inplace=True)

    prediction = prediction.astype("Float64")

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/naive")
def predict_naive(stock_symbol: str, start_date: str, end_date: str):
    naive_interface = NaiveInterface()

    prediction = naive_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
    prediction.set_index('ds', inplace=True)

    prediction = prediction.astype("Float64")

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/windowAverage")
def predict_windowAverage(stock_symbol: str, start_date: str, end_date: str):
    window_average_interface = WindowAverageInterface()

    prediction = window_average_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
    prediction.set_index('ds', inplace=True)

    prediction = prediction.astype("Float64")

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"{stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


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


@app.get("/predict/randomForest")
def predict_randomForest(stock_symbol: str, start_date: str, end_date: str):
    random_forest_interface = ABC_RandomForestModel()

    prediction = random_forest_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/gradientBoost")
def predict_gradientBoost(stock_symbol: str, start_date: str, end_date: str):
    gradient_boost_interface = ABC_GradientBoostingModel()

    prediction = gradient_boost_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    print(prediction)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/svm")
def predict_svm(stock_symbol: str, start_date: str, end_date: str):
    svm_interface = ABC_SVMModel()

    prediction = svm_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data
