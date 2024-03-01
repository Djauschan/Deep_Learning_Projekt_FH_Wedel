import pandas as pd
from fastapi import FastAPI

# from statisticmodels.PredicitonInterface import (
#     ArimaInterface,
#     ETSInterface,
#     NaiveInterface,
#     ThetaInterface,
#     WindowAverageInterface,
#     historicAverageInterface,
# )

from ML_Modelle.ML_PredictionInterface_daily import (
    # ABC_LinearRegressionModel_daily,
    ABC_RandomForestModel_daily,
    ABC_GradientBoostingModel_daily,
    # ABC_SVMModel_daily,
)

from ML_Modelle.ML_PredictionInterface_hour import (
    # ABC_LinearRegressionModel_hour,
    ABC_RandomForestModel_hour,
    ABC_GradientBoostingModel_hour,
    # ABC_SVMModel_hour,
)

from ML_Modelle.ML_PredictionInterface_min import (
    # ABC_LinearRegressionModel_min,
    ABC_RandomForestModel_min,
    ABC_GradientBoostingModel_min,
    # ABC_SVMModel_min,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the ANN group :)))"}

########################################## 
########### statisticmodels ############ 
 
# @app.get("/predict/arima")
# def predict_arima(stock_symbol: str, start_date: str, end_date: str):
#     arima_interface = ArimaInterface()

#     prediction = arima_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
#     prediction.set_index('ds', inplace=True)

#     prediction = prediction.astype("Float64")

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


# @app.get("/predict/ETS")
# def predict_ets(stock_symbol: str, start_date: str, end_date: str):
#     ets_interface = ETSInterface()

#     prediction = ets_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
#     prediction.set_index('ds', inplace=True)

#     prediction = prediction.astype("Float64")

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


# @app.get("/predict/historicAverage")
# def predict_historicAverage(stock_symbol: str, start_date: str, end_date: str):
#     historic_average_interface = historicAverageInterface()

#     prediction = historic_average_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
#     prediction.set_index('ds', inplace=True)

#     prediction = prediction.astype("Float64")

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


# @app.get("/predict/theta")
# def predict_theta(stock_symbol: str, start_date: str, end_date: str):
#     theta_interface = ThetaInterface()

#     prediction = theta_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
#     prediction.set_index('ds', inplace=True)

#     prediction = prediction.astype("Float64")

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


# @app.get("/predict/naive")
# def predict_naive(stock_symbol: str, start_date: str, end_date: str):
#     naive_interface = NaiveInterface()

#     prediction = naive_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
#     prediction.set_index('ds', inplace=True)

#     prediction = prediction.astype("Float64")

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


# @app.get("/predict/windowAverage")
# def predict_windowAverage(stock_symbol: str, start_date: str, end_date: str):
#     window_average_interface = WindowAverageInterface()

#     prediction = window_average_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)
#     prediction.set_index('ds', inplace=True)

#     prediction = prediction.astype("Float64")

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


########################################## 
############## daily model ############### 

# @app.get("/predict/linearRegression_daily")
# def predict_linearRegression_daily(stock_symbol: str, start_date: str, end_date: str):
#     linear_regression_interface = ABC_LinearRegressionModel_daily()

#     prediction = linear_regression_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}_Predicted_Close"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data

@app.get("/predict/randomForest_daily")
def predict_randomForest_daily(stock_symbol: str, start_date: str, end_date: str):
    random_forest_interface = ABC_RandomForestModel_daily()

    prediction = random_forest_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/gradientBoost_daily")
def predict_gradientBoost_daily(stock_symbol: str, start_date: str, end_date: str):
    gradient_boost_interface = ABC_GradientBoostingModel_daily()

    prediction = gradient_boost_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    print(prediction)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data

# @app.get("/predict/svm_daily")
# def predict_svm_daily(stock_symbol: str, start_date: str, end_date: str):
#     svm_interface = ABC_SVMModel_daily()

#     prediction = svm_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction["Predicted_Close"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data

########################################## 
############### hour model ############### 

# @app.get("/predict/linearRegression_hour")
# def predict_linearRegression_hour(stock_symbol: str, start_date: str, end_date: str):
#     linear_regression_interface = ABC_LinearRegressionModel_hour()

#     prediction = linear_regression_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}_Predicted_Close"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data

@app.get("/predict/randomForest_hour")
def predict_randomForest_hour(stock_symbol: str, start_date: str, end_date: str):
    random_forest_interface = ABC_RandomForestModel_hour()

    prediction = random_forest_interface.predict(stock_symbol, pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/gradientBoost_hour")
def predict_gradientBoost_hour(stock_symbol: str, start_date: str, end_date: str):
    gradient_boost_interface = ABC_GradientBoostingModel_hour()

    prediction = gradient_boost_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    print(prediction)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


# @app.get("/predict/svm_hour")
# def predict_svm_hour(stock_symbol: str, start_date: str, end_date: str):
#     svm_interface = ABC_SVMModel_hour()

#     prediction = svm_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction["Predicted_Close"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data


########################################## 
############### min model ############### 

# @app.get("/predict/linearRegression_min")
# def predict_linearRegression_min(stock_symbol: str, start_date: str, end_date: str):
#     linear_regression_interface = ABC_LinearRegressionModel_min()

#     prediction = linear_regression_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction[f"{stock_symbol.upper()}_Predicted_Close"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data

@app.get("/predict/randomForest_min")
def predict_randomForest_min(stock_symbol: str, start_date: str, end_date: str):
    random_forest_interface = ABC_RandomForestModel_min()

    prediction = random_forest_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


@app.get("/predict/gradientBoost_min")
def predict_gradientBoost_min(stock_symbol: str, start_date: str, end_date: str):
    gradient_boost_interface = ABC_GradientBoostingModel_min()

    prediction = gradient_boost_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

    print(prediction)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction["Predicted_Close"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


# @app.get("/predict/svm_min")
# def predict_svm_min(stock_symbol: str, start_date: str, end_date: str):
#     svm_interface = ABC_SVMModel_min()

#     prediction = svm_interface.predict(
#         pd.to_datetime(start_date), pd.to_datetime(end_date), 120)

#     # Convert the prediction to a list of dictionaries
#     prediction_data = prediction["Predicted_Close"]
#     data = [{"date": date, "value": value}
#             for date, value in prediction_data.items()]

#     return data
