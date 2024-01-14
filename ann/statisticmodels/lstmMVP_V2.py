# Import der benötigten Bibliotheken
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignoriert die Tensorflow-Warnungen
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import datetime
from pandas.tseries.offsets import BDay

class StockModel:
    def __init__(self, file_path, cut_off_date, load_model_path=None):
        self.file_path = file_path
        self.cut_off_date = cut_off_date
        self.data = self.load_data(file_path)
        self.train_data, self.test_data = self.split_data(self.data, cut_off_date)
        self.scaled_train_data, self.scaler = self.preprocess_data(self.train_data.copy())
        self.last_known_close = self.train_data['Close'].iloc[-1]
        #Entscheiden ob ein neues Modell erstellt werden soll oder ein bereits vorhandenes geladen werden soll
        if load_model_path:
            self.model = self.load_model(load_model_path)
        else:
            self.model = self.build_model((30, 1))

    #Load data from Djauschan & Kevin
    # def loadDataNew(self):
        
    #     data = DataReader({"READ_ALL_FILES": "READ_ALL_FILES"})
    #     txt_files, symbols = data.get_txt_files()
 
    #     # Test for-Schleife später löschen
    #     for i in symbols:
    #         print(i)
    #     # 0 = AAL, 1 = AAPL, ...   #von den 10 datas
    #     data.current_file_idx = 1
    #     df = data.read_next_txt()
    #     print("\nData:", df.symbol[0], "\n")

    #     data_columns = df.columns
    #     print(data_columns) #nur zum testen

    #     #Data clean
    #     cleaner = DataCleaner(df)
        
    #     return data

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        data.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data.set_index('DateTime', inplace=True)
        data = data[data.index.time == pd.to_datetime('16:00').time()]
        return data

    def calculate_price_difference(self, data):
        data['PriceDiff'] = data['Close'].diff()
        return data.dropna()

    def preprocess_data(self, data):
        scaler = StandardScaler()
        data = self.calculate_price_difference(data)
        scaled_data = scaler.fit_transform(data['PriceDiff'].values.reshape(-1, 1))
        return scaled_data, scaler

    def create_dataset(self, data, time_step=30):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def split_data(self, data, cut_off_date):
        train_data = data.loc[data.index < cut_off_date]
        test_data = data.loc[data.index >= cut_off_date]
        return train_data, test_data

    def train_model(self, epochs, batch_size):
        X_train, y_train = self.create_dataset(self.scaled_train_data, 30)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_next_day(self, scaled_data):
        if len(scaled_data) < 30:
            raise ValueError("Nicht genügend Daten für eine Vorhersage verfügbar.")

        last_sequence = scaled_data[-30:].reshape(1, 30, 1)
        predicted_scaled = self.model.predict(last_sequence)
        predicted_price_change = self.scaler.inverse_transform(predicted_scaled)[0, 0]

        predicted_close_price = self.last_known_close + predicted_price_change
        return predicted_close_price

    def predict_x_days(self, x_days, timestamp_start):
        # Eine Liste zur Speicherung der Vorhersagedaten
        prediction_data = []
        scaled_data = self.scaled_train_data.copy()
        last_known_close = self.last_known_close

        current_timestamp = pd.to_datetime(timestamp_start)

        for _ in range(x_days):
            last_sequence = scaled_data[-30:].reshape(1, 30, 1)
            predicted_scaled = self.model.predict(last_sequence)
            predicted_price_change = self.scaler.inverse_transform(predicted_scaled)[0, 0]

            predicted_close_price = last_known_close + predicted_price_change

            # Hinzufügen des Datums und des geschätzten Schlusskurses zur Liste
            prediction_data.append({'DateTime': current_timestamp, 'Close': predicted_close_price})

            new_row = np.array([predicted_price_change])
            scaled_data = np.append(scaled_data, new_row)[-30:]
            last_known_close = predicted_close_price

            # Update des Zeitstempels zum nächsten Geschäftstag
            current_timestamp += BDay(1)

        predictions = pd.DataFrame(prediction_data)
        return predictions
    
        #Speicherung des Modells im H5-Format
    def save_model_asH5(self, file_path_saving, stock_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"{stock_name}_lstm_model_{timestamp}.h5"
        model_save_path = f'{file_path_saving}/{model_name}'  # Pfad zum Speichern des Modells
        self.model.save(model_save_path)
        print(f"Modell gespeichert als: {model_name}")    

    #Speicherung des Modells im Keras-Format
    def save_model_asKeras(self, file_path_saving, stock_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"{stock_name}_lstm_model_{timestamp}.keras"
        model_save_path = f'{file_path_saving}/{model_name}'  # Pfad zum Speichern des Modells
        self.model.save(model_save_path)
        print(f"Modell gespeichert als: {model_name}")    

    #Laden des Modells    
    def load_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Modell geladen von: {model_path}")
            return model
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return None

# Instanz der StockModel Klasse erstellen
# Wenn die TF Nachrichten zu viel werden: tf.keras.utils.disable_interactive_logging()
file_path = "/Users/umutkurucay/Documents/Developer/LSTM_testing/Data/SQ_1min.txt"  # Pfad zur Datei
cut_off_date = '2021-01-03'  # Cut-off-Datum
stock_name = file_path.split('/')[-1].split('_')[0] # Hier wird der Name der Aktie aus dem Dateipfad extrahiert
#bei laden wird das Modell geladen, bei generieren wird ein neues Modell erstellt
laden_statt_generieren = True #oder "False" für generieren 


if laden_statt_generieren:
    print("Vorhandenes Modell wird geladen")
    #lokales Testen:
    #load_model_path = "/Users/umutkurucay/Documents/Developer/LSTM_testing/saved_models/AAPL_lstm_model_20240113-174652.h5" 
    load_model_path = f"statisticmodels/models/savedModelsLSTM/{stock_name}_lstm_model.h5"
    # nun soll ein relativer pfad zum modell angegeben werden, welcher mithilfe des Symbols 
    print("load_model wurde aufgerufen")
    print(load_model_path)
else:
    print("Neues Modell wird generiert")

stock_model = StockModel(file_path, cut_off_date)

# Echten Close-Preis für die nächsten X Tage holen (nach dem Cut-off-Datum) mit dem Timestamp


# Trainieren des Modells
if not laden_statt_generieren:
    print("Es wird generiert!")
    
    stock_model.train_model(epochs=100, batch_size=64) # Hier kann die Anzahl der Epochen und die Batch-Größe geändert werden
    stock_name = file_path.split('/')[-1].split('_')[0] # Hier wird der Name der Aktie aus dem Dateipfad extrahiert
# Vorhersage des nächsten Tages
# next_day_prediction = stock_model.predict_next_day(stock_model.scaled_train_data)
# print(f"Vorhergesagter Close-Preis für den nächsten Tag: {next_day_prediction}")

# Wenn True, dann wird das Modell gespeichert
save_model = not laden_statt_generieren
if save_model:
    load_model_path = None
    stock_model.save_model_asH5("/Users/umutkurucay/Documents/Developer/LSTM_testing/saved_models", stock_name)
    stock_model.save_model_asKeras("/Users/umutkurucay/Documents/Developer/LSTM_testing/saved_models", stock_name)

# Pfad zum geladenen Modell, None wenn ein neues Modell erstellt werden soll
stock_model = StockModel(file_path, cut_off_date, load_model_path=load_model_path)

def manualStart():
    timestamp_start = '2021-01-04'
    x_days = 2  # Anzahl der Tage, die vorhergesagt werden sollen
    print("Echter Close-Preis für die nächsten x Tage:")
    for day, price in enumerate(stock_model.test_data['Close'].values[:x_days], start=1):
        print(f"Tag {day}: {price}")
    predictions = stock_model.predict_x_days(x_days, timestamp_start)
    print("...........")
    print(predictions)
    print("...........")
    
if False: #AUF TRUE SETZEN WENN DIESE KLASSE MANUELL GESTARTET WERDEN SOLL
    manualStart()