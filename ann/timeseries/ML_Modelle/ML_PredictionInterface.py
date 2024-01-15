import pandas as pd
import os
import pickle

from abc import ABC, abstractmethod
from abstract_model import AbstractModel

from ml_model_abc import LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

#Implementieren der ML-Modelle von: LR, RF, GBM, SVM

class ABC_LinearRegressionModel(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model()
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end)
        predicted_values = []
        indices = []

        loaded_data = self.load_data()
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        #last_known_close_value = back_transform_test_data['close'].iloc[0]
        #last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        if timestamp_start == pd.Timestamp('2021-01-04'):
            last_known_close_value = back_transform_test_data['close'].iloc[0]
        else:
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        max_points = min(len(prediction_dates), len(X_test))
        for i in range(max_points):
            X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)  # Umwandlung der Daten in DataFrame
            predicted_pct_change = self.model.predict(X_test_row_df)[0]             # Vorhersage für den nächsten Tag (prozentuale Veränderung)
            
            # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
            predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
            predicted_values.append(predicted_close)

            # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
            last_known_close_value = predicted_close
            indices.append(X_test.index[i])

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({'AAPL_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self) -> None:
        file_path = 'saved_pkl/Data'
        data = {}  # Dictionary zum Speichern der geladenen Dataframes

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.split(".")[0]  # Schlüssel ohne Dateiendung
                    data[key] = pd.read_pickle(full_path)
        return data

    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        model_path = 'saved_pkl/LR-Model/lr_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print("Modell-Datei nicht gefunden.")
            self.model = None


class ABC_RandomForestModel(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:

        #Model laden
        self.load_model()

        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end)
        predicted_values = []
        indices = []

        loaded_data = self.load_data()
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        #last_known_close_value = back_transform_test_data['close'].iloc[0]
        #last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        if timestamp_start == pd.Timestamp('2021-01-04'):
            last_known_close_value = back_transform_test_data['close'].iloc[0]
        else:
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        max_points = min(len(prediction_dates), len(X_test))
        for i in range(max_points):
            X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)  # Umwandlung der Daten in DataFrame
            predicted_pct_change = self.model.predict(X_test_row_df)[0]             # Vorhersage für den nächsten Tag (prozentuale Veränderung)
            
            # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
            predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
            predicted_values.append(predicted_close)

            # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
            last_known_close_value = predicted_close
            indices.append(X_test.index[i])

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({'Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self) -> None:
        file_path = 'saved_pkl/Data'
        data = {}  # Dictionary zum Speichern der geladenen Dataframes

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.split(".")[0]  # Schlüssel ohne Dateiendung
                    data[key] = pd.read_pickle(full_path)
        return data

    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        model_path = 'saved_pkl/RF-Model/rf_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print("Modell-Datei nicht gefunden.")
            self.model = None

class ABC_GradientBoostingModel(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:

        #Model laden
        self.load_model()

        #Modelle sind bis zum 3.1.21trainiert -> prediction ab 4.1 möglich
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end)
        predicted_values = []
        indices = []

        loaded_data = self.load_data()
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        #last_known_close_value = back_transform_test_data['close'].iloc[0]
        #last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        if timestamp_start == pd.Timestamp('2021-01-04'):
            last_known_close_value = back_transform_test_data['close'].iloc[0]
        else:
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        max_points = min(len(prediction_dates), len(X_test))
        for i in range(max_points):
            X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)  # Umwandlung der Daten in DataFrame
            predicted_pct_change = self.model.predict(X_test_row_df)[0]             # Vorhersage für den nächsten Tag (prozentuale Veränderung)
            
            # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
            predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
            predicted_values.append(predicted_close)

            # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
            last_known_close_value = predicted_close
            indices.append(X_test.index[i])

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({'Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self) -> None:
        file_path = 'saved_pkl/Data'
        data = {}  # Dictionary zum Speichern der geladenen Dataframes

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.split(".")[0]  # Schlüssel ohne Dateiendung
                    data[key] = pd.read_pickle(full_path)
        return data

    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        model_path = 'saved_pkl/GBM-Model/gbm_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print("Modell-Datei nicht gefunden.")
            self.model = None


class ABC_SVMModel(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model()

        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end)
        predicted_values = []
        indices = []

        loaded_data = self.load_data()
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        #last_known_close_value = back_transform_test_data['close'].iloc[0]
        #last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        if timestamp_start == pd.Timestamp('2021-01-04'):
            last_known_close_value = back_transform_test_data['close'].iloc[0]
        else:
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        max_points = min(len(prediction_dates), len(X_test))
        for i in range(max_points):
            X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)  # Umwandlung der Daten in DataFrame
            predicted_pct_change = self.model.predict(X_test_row_df)[0]             # Vorhersage für den nächsten Tag (prozentuale Veränderung)
            
            # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
            predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
            predicted_values.append(predicted_close)

            # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
            last_known_close_value = predicted_close
            indices.append(X_test.index[i])

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({'Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self) -> None:
        file_path = 'saved_pkl/Data'
        data = {}  # Dictionary zum Speichern der geladenen Dataframes

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.split(".")[0]  # Schlüssel ohne Dateiendung
                    data[key] = pd.read_pickle(full_path)
        return data

    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        model_path = 'saved_pkl/SVM-Model/svm_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print("Modell-Datei nicht gefunden.")
            self.model = None


  