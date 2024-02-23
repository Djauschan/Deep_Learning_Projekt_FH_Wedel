import numpy as np
import sys


class QLearningPredictor():
    def __init__(self, model_path):
        if 'ma' in model_path.lower():
            self.model = MAPredictor(model_path)
        elif 'rsi' in model_path.lower():
            self.model = RSIPredictor(model_path)
        try:
            self.name = self.model.name
        except AttributeError:
            self.name = 'unsupported'
    def predict(self, data) -> str:
        return self.model.predict(data)
            

class MAPredictor():
    def __init__(self, model_path):
        self.ma_variant = int(model_path.split('/')[-1].split('_')[0].replace('ma', ''))
        self.q_table = np.load(model_path)
        self.name = f'Q_learning_MA{self.ma_variant}'
    
    
    def calculate_MA(self, data) -> float: #Mit period = 5, 30 oder 200
        if len(data) < self.ma_variant:
            # Nicht genügend Daten für den vollständigen MA
            return data.mean()
        else:
            # Berechnen des Durchschnitts der letzten 'period' Werte
            return data.iloc[-self.ma_variant:].mean()
    
    def calculate_state(self, current_price, ma_value):
        deviation = (current_price - ma_value) / ma_value * 100
        max_deviation = 2 * deviation.std() 
        scaled_deviation = (deviation + max_deviation) / (2 * max_deviation)
        state = max(0, min(int(scaled_deviation * 20), 20 - 1)) # n_bins = 20
        return state
    
    def predict(self, data) -> str:
        current_price = data.iloc[-1].Close
        state_index = self.calculate_state(current_price, self.calculate_MA(data.Close))
        action = np.argmax(self.q_table[state_index])
        if action == 0:
            return 'hold'
        elif action == 1:
            return 'buy'
        elif action == 2:
            return 'sell'
        else:
            raise ValueError(f'Invalid action: {action}')

class RSIPredictor():
    def __init__(self, model_path) -> float:
        self.q_table = np.load(model_path)
        self.name = 'Q_learning_RSI'
        self.low_threshold = 30
        self.high_threshold = 70
    
    def calculate_RSI(self, data, period=14):
        if len(data) < period:
            return 50  # Neutraler RSI-Wert bei unzureichenden Daten

        # Berechnung der Differenzen
        delta = data.Close.diff()
        
        # Ermittlung von Gewinnen und Verlusten über die letzten 'period' Tage
        gain = delta.clip(lower=0).iloc[-period:].mean()
        loss = -delta.clip(upper=0).iloc[-period:].mean()

        # Vermeidung der Division durch Null
        if loss == 0:
            return 100 if gain > 0 else 50

        # Berechnung des RSI
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        return RSI
    
    def predict(self, data) -> str:
        rsi_value = self.calculate_RSI(data)
        if rsi_value < self.low_threshold:
            return 'buy'  # Kaufen
        elif rsi_value > self.high_threshold:
            return 'sell'  # Verkaufen
        else:
            return 'hold'  # Halten