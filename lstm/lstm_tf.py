# Für die Ausführung in Google Colab (Verfügbarkein von T4 GPU und TPU) muss folgender pip Befehl ausgeführt werden: 
# !pip install keras-tuner -q
# Alle anderen Pakete sind bereits in Colab verfügbar

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keras_tuner as kt

class StockPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.name = file_path.split('/')[-1].split('.')[0]

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data.set_index('DateTime', inplace=True)
        data = data[data.index.time == pd.to_datetime('16:00').time()]
        return data

    def preprocess_data(self, data):
        values = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values)
        return scaled_data

    def create_dataset(self, data, time_step=100):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def split_data(self, X, y):
        train_size = int(len(X) * 0.80)
        valid_test_size = len(X) - train_size
        valid_size = test_size = int(valid_test_size / 2)
        X_train, X_valid_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_valid_test = y[0:train_size], y[train_size:len(y)]
        X_valid, X_test = X_valid_test[:valid_size], X_valid_test[valid_size:]
        y_valid, y_test = y_valid_test[:valid_size], y_valid_test[valid_size:]
        return X_train.reshape(-1, 100, 1), y_train, X_valid.reshape(-1, 100, 1), y_valid, X_test.reshape(-1, 100, 1), y_test

    def build_model_automated_hpt(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(
            units=hp.Int('units', min_value=2, max_value=3, step=1),
            return_sequences=True,
            input_shape=(100, 1)))
        model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=2, max_value=3, step=1), return_sequences=False))
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=2, max_value=3, step=1)))
        model.add(tf.keras.layers.Dense(1))
        #TODO: Nur bei ARM Macs den legacy Adam optimizer nehmen.-> ansonsten ohne legacy
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')), loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, X_valid, y_valid):
        #Early Stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=5)
        #Automated Hyperparameter Tuning
        tuner = kt.RandomSearch(
            self.build_model_automated_hpt,
            objective='val_loss',
            #Standardwert max_trials ist 10
            max_trials=3,
            executions_per_trial=3,
            directory='my_dir',
            #TODO FileName in Abhängigkeit von der Aktie ändern

            project_name='lstm_stock_prediction_')
        tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.get_best_models(num_models=1)[0]
        history = self.model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[es])

        self.plot_loss(history)

    def predict(self, X_test):
        test_predictions = self.model.predict(X_test)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return test_predictions

    def plot_predictions(self, y_test, test_predictions):
        y_test_scaled = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        mse = mean_squared_error(y_test_scaled, test_predictions)
        plt.figure(figsize=(10,6))
        plt.plot(y_test_scaled, label='Tatsächliche Werte', color='blue')
        plt.plot(test_predictions, label='Vorhersagen', color='red')
        plt.title(f'Aktienpreisvorhersage (MSE beträgt: {mse})')
        plt.xlabel('Zeit')
        plt.ylabel('Preis')
        plt.legend()
        plt.show()

    def plot_loss(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Trainingsverlust')
        plt.plot(history.history['val_loss'], label='Validierungsverlust')
        plt.title('Trainings- und Validierungsverlust')
        plt.xlabel('Epochen')
        plt.ylabel('Verlust')
        plt.legend()
        plt.show()

    def save_model(self, model_name):
        self.model.save(model_name)

# for file in os.listdir('/Users/umutkurucay/Documents/Developer/LSTM_testing/Data'):
#   if file.endswith('.csv'):
predictor = StockPredictor(f'/content/drive/MyDrive/data/AMD_1min.txt')
data = predictor.load_data()
scaled_data = predictor.preprocess_data(data)
X, y = predictor.create_dataset(scaled_data)
X_train, y_train, X_valid, y_valid, X_test, y_test = predictor.split_data(X, y)
predictor.train_model(X_train, y_train, X_valid, y_valid)
test_predictions = predictor.predict(X_test)
predictor.plot_predictions(y_test, test_predictions)
predictor.save_model('LSTM_autotunedTEST.keras')
