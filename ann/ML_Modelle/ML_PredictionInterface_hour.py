import pandas as pd
import os
import pickle

#Implementieren der ML-Modelle von: LR, RF, GBM, SVM

class ABC_LinearRegressionModel_hour():
    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}H') #intervall 2h
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        #print(X_test)

        # Bestimmung des letzten bekannten Close-Wertes zum Startzeitpunkt
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Falls kein genauer Übereinstimmungswert vorhanden ist, verwenden Sie den letzten verfügbaren Wert vor dem Startzeitpunkt
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)

                # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
                last_known_close_value = predicted_close
                predicted_values.append(predicted_close)
                indices.append(timestamp)

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_hour/Data'
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
        model_path = f'ML_Modelle/saved_pkl_model_hour/LR-Model/{symbol}_lr_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

class ABC_RandomForestModel_hour():

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}H') #intervall 2h
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Bestimmung des letzten bekannten Close-Wertes zum Startzeitpunkt
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Falls kein genauer Übereinstimmungswert vorhanden ist, verwenden Sie den letzten verfügbaren Wert vor dem Startzeitpunkt
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)

                # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
                last_known_close_value = predicted_close
                predicted_values.append(predicted_close)
                indices.append(timestamp)

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_hour/Data'
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
        model_path = f'ML_Modelle/saved_pkl_model_hour/RF-Model/{symbol}_rf_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

class ABC_GradientBoostingModel_hour():

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}H') #intervall 2h
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Bestimmung des letzten bekannten Close-Wertes zum Startzeitpunkt
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Falls kein genauer Übereinstimmungswert vorhanden ist, verwenden Sie den letzten verfügbaren Wert vor dem Startzeitpunkt
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)

                # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
                last_known_close_value = predicted_close
                predicted_values.append(predicted_close)
                indices.append(timestamp)

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_hour/Data'
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
        model_path = f'ML_Modelle/saved_pkl_model_hour/GBM-Model/{symbol}_gbm_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None


class ABC_SVMModel_hour():

    def predict(self, symbol, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        #Model laden
        self.load_model(symbol)
        
        #Modelle sind bis zum 3.1.21 trainiert -> prediction ab 4.1 möglich
        #prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq='H')           #stündliche prediciton
        prediction_dates = pd.date_range(start=timestamp_start, end=timestamp_end, freq=f'{interval}H') #intervall 2h
        predicted_values = []
        indices = []

        loaded_data = self.load_data(symbol)
        back_transform_test_data = loaded_data['back_transform_test_data']
        X_test = loaded_data['X_test']

        # Bestimmung des letzten bekannten Close-Wertes zum Startzeitpunkt
        if timestamp_start in back_transform_test_data.index:
            last_known_close_value = back_transform_test_data.loc[timestamp_start, 'close']
        else:
            # Falls kein genauer Übereinstimmungswert vorhanden ist, verwenden Sie den letzten verfügbaren Wert vor dem Startzeitpunkt
            last_known_close_value = back_transform_test_data[back_transform_test_data.index < timestamp_start]['close'].iloc[-1]

        # Iteration über die Vorhersagedaten
        for timestamp in prediction_dates:
            if timestamp in X_test.index:
                X_test_row_df = pd.DataFrame([X_test.loc[timestamp]], columns=X_test.columns)
                
                # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
                predicted_pct_change = self.model.predict(X_test_row_df)[0]
                predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)

                # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
                last_known_close_value = predicted_close
                predicted_values.append(predicted_close)
                indices.append(timestamp)

        # Erstellen eines DataFrames für die Vorhersageergebnisse
        prediction_df = pd.DataFrame({f'{symbol}_Predicted_Close': predicted_values}, index=indices)
        return prediction_df

    def load_data(self, symbol):
        file_path = 'ML_Modelle/saved_pkl_model_hour/Data'
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
        model_path = f'ML_Modelle/saved_pkl_model_hour/SVM-Model/{symbol}_svm_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print(f"Modell-Datei für {symbol} nicht gefunden.")
            self.model = None

  