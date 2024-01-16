# Für die Ausführung in Google Colab (Verfügbarkein von T4 GPU und TPU) muss folgender pip Befehl ausgeführt werden: 
# !pip install keras-tuner -q
# Alle anderen Pakete sind bereits in Colab verfügbar

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import datetime
from tensorflow.keras.models import load_model

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)
    data = data[data.index.time == pd.to_datetime('16:00').time()]
    return data

# Da wir nicht die Preise direkt vorhersagen, sondern die prozentuale Veränderung, müssen wir diese berechnen
# Eventuell sollte statt der prozentualen Veränderung der Logarithmus der prozentualen Veränderung verwendet werden
# weil die prozentuale Veränderung bei kleinen Werten stark schwankt
# Ein anderer Ansatz wäre die absolute Differenz zu verwenden
def calculate_price_difference(data):
    data['PriceDiff'] = data['Close'].diff()
    return data.dropna()

def calculate_percentage_change(data):
    data['PctChange'] = data['Close'].pct_change()
    return data.dropna()

# für prozentuale Veränderung
def preprocess_data_for_percantage(data):
    scaler = StandardScaler()
    data = calculate_percentage_change(data)
    scaled_data = scaler.fit_transform(data['PctChange'].values.reshape(-1, 1))
    return scaled_data, scaler

# für absolute Differenz
def preprocess_data_for_pricedifference(data):
    scaler = StandardScaler()
    data = calculate_price_difference(data)
    scaled_data = scaler.fit_transform(data['PriceDiff'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Modellerstellung:
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#Speicherung des Modells
def save_model(model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"stock_predictor_model_{timestamp}.keras"
    model.save(model_name)
    print(f"Modell gespeichert als: {model_name}")


#Ausführung
file_path = '/content/drive/MyDrive/data/AMD_1min.txt'
data = load_data(file_path)
scaled_data, scaler = preprocess_data(data)
X, y = create_dataset(scaled_data, 30)

train_size = int(len(X) * 0.70)
X_train, X_valid = X[:train_size], X[train_size:]
y_train, y_valid = y[:train_size], y[train_size:]

model = build_model((30, 1))
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid))

# Modell speichern
# Kommentar entfernen, wenn Modell gespeichert werden soll: 
# save_model(model)

#alte predict_next_day Funktion: für prozentuale Veränderung
# def predict_next_day(model, data, scaler, last_close_price):
#     if len(data) < 30:
#         raise ValueError("Nicht genügend Daten für eine Vorhersage verfügbar.")
#     last_sequence = data[-30:].reshape(1, 30, 1)
#     predicted_scaled = model.predict(last_sequence)
#     predicted_pct_change = scaler.inverse_transform(predicted_scaled)[0, 0]
#     predicted_absolute_change = last_close_price * predicted_pct_change
#     predicted_close_price = last_close_price + predicted_absolute_change
#     return predicted_close_price
load_saved_model = "nein"  # Oder "ja"

if load_saved_model.lower() == "ja":
    model_filename = input("Geben Sie den Dateinamen des zu ladenden Modells ein: ")
    model = load_model(model_filename)
else:
    # Hier verwenden wir das in dieser Sitzung trainierte Modell
    print("Verwendung des in dieser Sitzung trainierten Modells")

def predict_next_day(model, data, scaler, last_close_price, last_date_index):
    if len(data) < 30:
        raise ValueError("Nicht genügend Daten für eine Vorhersage verfügbar.")
    last_sequence = data[-30:].reshape(1, 30, 1)
    predicted_scaled = model.predict(last_sequence)
    predicted_price_change = scaler.inverse_transform(predicted_scaled)[0, 0]
    predicted_close_price = last_close_price + predicted_price_change
    next_day_index = last_date_index + pd.DateOffset(days=1)
    return next_day_index, predicted_close_price

last_close_price = data['Close'].iloc[-31]
last_date_index = data.index[-31]

next_day_index, predicted_close = predict_next_day(model, scaled_data, scaler, last_close_price, last_date_index)
print(f"Vorhergesagter Close-Preis für {next_day_index.date()}: {predicted_close}")

# Ausgabe der Vorhersage für den nächsten Tag
def get_actual_next_day_close(data, predicted_date):
    if predicted_date in data.index:
        actual_next_day_close = data.loc[predicted_date]['Close']
        return actual_next_day_close
    else:
        return None  # bzw. Fehlerbehandlung einbauen, falls das Datum nicht gefunden wird

# 
predicted_date = next_day_index  # Das Datum, das von der Vorhersagefunktion zurückgegeben wird
actual_close_price = get_actual_next_day_close(data, predicted_date)
if actual_close_price is not None:
    print(f"Tatsächlicher Close-Preis für {predicted_date.date()}: {actual_close_price}")
else:
    print(f"Kein Datenpunkt gefunden für das Datum {predicted_date.date()}")
