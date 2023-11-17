import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import math
import copy
from torch.utils.data import Dataset

##### Daten aufbereiten

# Entfernen von NaN-Werten, die durch die rollenden Funktionen eingeführt wurden
stock_data2 = stock_data2.dropna()
stock_data2 = stock_data2.iloc[29:]  # Entfernt die ersten 29 Zeilen, die NaN-Werte wegen des 30-day MA haben könnten

stock_data2 = stock_data2.dropna()

# Aufteilen in Trainings- und Testdaten
train_size = int(0.7 * len(stock_data2))
train_data = stock_data2.iloc[:train_size]
test_data = stock_data2.iloc[train_size:]

# Datenstruktur
# Data: close  5-day MA  30-day MA  200-day MA   RSI