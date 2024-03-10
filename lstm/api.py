import pandas as pd
from fastapi import FastAPI
import warnings
import os
warnings.filterwarnings("ignore")

import pickle
from prediction import predict_intervals
from datapreprocessor import DataProcessor
from enum import Enum

class resolution(Enum):
    """Enum for the resolution of the stock data.
    """
    DAILY = 'D'
    TWO_HOURLY = 'H'
    MINUTE = 'M'

app = FastAPI()

class ML_Prediction_LSTMBasicModel():

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, resolution: resolution) -> pd.DataFrame:

        #Get Model
        self.load_model(symbol, resolution)
        print("loaded model")
        
        # Get Data
        pd_data = self.load_data(symbol, resolution)

        
        # scaler = StandardScaler()
        # self.scaler.transform(self.data[['Close']])  

        back_transform_test_data = pd_data['back_transform_test_data']
        X_test = pd_data['X_test']

        predicted_values = []
        indices = []

        # Bestimmung des letzten bekannten Close-Wertes zum Startzeitpunkt
        # Verwendung von Codebausteinen der ANN Gruppe 
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='D')

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                X_test_row = X_test.loc[timestamp]

                # Vorhersage machen
                # predicted_pct_change = self.model.predict([X_test_row])[0]
                
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)

                # Aktualisierung des letzten bekannten Close-Werts
                last_known_close_value = predicted_close

                # Vorhersagewerte und Indizes speichern
                predicted_values.append(predicted_close)
                indices.append(timestamp)


        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)

        return prediction_df

    def preprocess(self) -> None:
        pass

    def load_data(self, symbol, resolution):
        file_path = f'Data/{resolution}/{symbol}.txt'
        data = {}

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.startswith(symbol) and filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.replace(f"{symbol}_", "").split(".")[0]
                    data[key] = pd.read_pickle(full_path)

        return data

    def load_model(self, symbol, resolution) -> None:
        model_path = f'saved_pkl_model/{resolution}/{symbol}_lstm_model_20240113-212614.keras'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

# FÜRS BACKEND: @johann
# @app.get("/")
# async def root():
#     return {"message": "Prediction API for the LSTM))"}


# @app.get("/predict/lstm")
# def predict_lstm(stock_symbols: str, start_date: str, end_date: str, resolution: resolution):

#     print("LSTM prediction")

#     symbols_list = stock_symbols[1:-1].split(", ")
#     basic_lstm = ML_Prediction_LSTMBasicModel()

#     results = []
#     if resolution == resolution.DAILY or resolution == resolution.MINUTE:
#         interval = 1
#     elif resolution == resolution.TWO_HOURLY:
#         interval = 2
#     else:
#         raise NotImplementedError()
#     for stock_symbol in symbol_list:
#         results.append(basic_lstm.predict(stock_symbol, pd.to_datetime(start_date), pd.to_datetime(end_date), interval))

#     result_df = pd.DataFrame()

#     for result in results:
#         result_df = pd.concat([result_df, result], axis=1)

#     prediction = result_df.round(2)

#     # Set the index of the prediction dataframe to be shifted by 20 hours for daily predictions
#     if resolution == resolution_enum.DAILY:
#         prediction.index = prediction.index + pd.Timedelta(hours=20)

#     data = {}
#     for symbol in symbols_list:
#         symbol_prediction = prediction[f"{symbol}_Predicted_Close"]
#         data[symbol] = [{"date": date, "value": value}
#                         for date, value in symbol_prediction.items()]

#     return data


# @app.get("/predict/lstm")
# def predict_lstm(stock_symbols: str = "[AAPL, NVDA]",
#                          start_date: str = "2021-01-04",
#                          resolution: str = "D"):
#     if resolution == "M":
#         start_date += " 10:01:00"
#     end_date = calculate_end_date(start_date, resolution)
#     if resolution == "H":
#         start_date += " 10:00:00"
#         end_date += " 16:00:00"

#     data_to_send = {"stock_symbols": stock_symbols,
#                     "start_date": start_date,
#                     "end_date": end_date,
#                     "resolution": resolution}

#     api_url = "http://predict_lstm:8000/predict/lstm"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()



#   predict_lstm:
#   # Specify the build context for this service (current directory)
#     build:
#       context: ./lstm
#       dockerfile: Dockerfile



#     # Bind mount the current directory to /app in the container
#     volumes:
#       - ./lstm:/app



#     # Map port 8003 on the host to port 8000 in the container
#     ports:
#       - 8005:8000



#     # Set the entrypoint of the container to start a Uvicorn server running the app defined in 'api' of 'src_transformers' directory
#     entrypoint:
#       [
#         "bash",
#         "-c",
#         "uvicorn api:app --host 0.0.0.0 --port 8000"
#       ]
#     networks:
#       - backend_network