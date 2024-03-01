import pandas as pd
import os
import pickle

from abc import ABC, abstractmethod
from ML_Modelle.abstract_model import AbstractModel

from ML_Modelle.ml_model_min import LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

#Implementieren der ML-Modelle von: LR, RF, GBM, SVM

class ABC_LinearRegressionModel_min(AbstractModel):

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='min')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}min') #intervall 20 min
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Überprüfen, ob Startzeitpunkt in den Daten vorhanden ist
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Letzten bekannten Wert vor dem Startzeitpunkt verwenden
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                # Daten für den aktuellen Zeitpunkt extrahieren
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Prozentuale Veränderung vorhersagen und in absoluten Close-Wert umwandeln
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
                
                # Vorhersagewert und Zeitstempel speichern
                predicted_values.append(predicted_close)
                indices.append(timestamp)
                
                # Letzten bekannten Close-Wert aktualisieren
                last_known_close_value = predicted_close

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_min/Data'
        data = {}

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.startswith(symbol) and filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.replace(f"{symbol}_", "").split(".")[0]
                    data[key] = pd.read_pickle(full_path)

        return data

    def preprocess(self) -> None:
        pass

    def load_model(self, symbol) -> None:
        model_path = f'ML_Modelle/saved_pkl_model_min/LR-Model/{symbol}_lr_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

class ABC_RandomForestModel_min(AbstractModel):

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='min')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}min') #intervall 20 min
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Überprüfen, ob Startzeitpunkt in den Daten vorhanden ist
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Letzten bekannten Wert vor dem Startzeitpunkt verwenden
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                # Daten für den aktuellen Zeitpunkt extrahieren
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Prozentuale Veränderung vorhersagen und in absoluten Close-Wert umwandeln
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
                
                # Vorhersagewert und Zeitstempel speichern
                predicted_values.append(predicted_close)
                indices.append(timestamp)
                
                # Letzten bekannten Close-Wert aktualisieren
                last_known_close_value = predicted_close

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_min/Data'
        data = {}

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.startswith(symbol) and filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.replace(f"{symbol}_", "").split(".")[0]
                    data[key] = pd.read_pickle(full_path)

        return data

    def preprocess(self) -> None:
        pass

    def load_model(self, symbol) -> None:
        model_path = f'ML_Modelle/saved_pkl_model_min/RF-Model/{symbol}_rf_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

class ABC_GradientBoostingModel_min(AbstractModel):

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='min')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}min') #intervall 20 min
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Überprüfen, ob Startzeitpunkt in den Daten vorhanden ist
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Letzten bekannten Wert vor dem Startzeitpunkt verwenden
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                # Daten für den aktuellen Zeitpunkt extrahieren
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Prozentuale Veränderung vorhersagen und in absoluten Close-Wert umwandeln
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
                
                # Vorhersagewert und Zeitstempel speichern
                predicted_values.append(predicted_close)
                indices.append(timestamp)
                
                # Letzten bekannten Close-Wert aktualisieren
                last_known_close_value = predicted_close

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_min/Data'
        data = {}

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.startswith(symbol) and filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.replace(f"{symbol}_", "").split(".")[0]
                    data[key] = pd.read_pickle(full_path)

        return data

    def preprocess(self) -> None:
        pass

    def load_model(self, symbol) -> None:
        model_path = f'ML_Modelle/saved_pkl_model_min/GBM-Model/{symbol}_gbm_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None


class ABC_SVMModel_min(AbstractModel):

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='min')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}min') #intervall 20 min
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Überprüfen, ob Startzeitpunkt in den Daten vorhanden ist
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Letzten bekannten Wert vor dem Startzeitpunkt verwenden
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                # Daten für den aktuellen Zeitpunkt extrahieren
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Prozentuale Veränderung vorhersagen und in absoluten Close-Wert umwandeln
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
                
                # Vorhersagewert und Zeitstempel speichern
                predicted_values.append(predicted_close)
                indices.append(timestamp)
                
                # Letzten bekannten Close-Wert aktualisieren
                last_known_close_value = predicted_close

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_min/Data'
        data = {}

        if os.path.exists(file_path):
            for filename in os.listdir(file_path):
                if filename.startswith(symbol) and filename.endswith(".pkl"):
                    full_path = os.path.join(file_path, filename)
                    key = filename.replace(f"{symbol}_", "").split(".")[0]
                    data[key] = pd.read_pickle(full_path)

        return data

    def preprocess(self) -> None:
        pass

    def load_model(self, symbol) -> None:
        model_path = f'ML_Modelle/saved_pkl_model_min/SVM-Model/{symbol}_svm_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

  