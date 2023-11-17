import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import math
import copy
from torch.utils.data import Dataset

# ##### Gleitende Durchschnitte
stock_data2['5-day MA'] = stock_data2['close'].rolling(window=5).mean()
stock_data2['30-day MA'] = stock_data2['close'].rolling(window=30).mean()
stock_data2['200-day MA'] = stock_data2['close'].rolling(window=200).mean()

def compute_RSI(data, window=14):
    delta = data['close'].diff()

    # Verwenden von `loc` für bedingte Zuweisungen, um Probleme mit nicht eindeutiger Indizierung zu vermeiden
    loss = pd.Series(index=delta.index, dtype=float)
    gain = pd.Series(index=delta.index, dtype=float)

    loss.loc[delta < 0] = -delta[delta < 0]
    gain.loc[delta > 0] = delta[delta > 0]

    avg_loss = loss.rolling(window=window, min_periods=1).mean().abs()
    avg_gain = gain.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Behandeln Sie den Fall, bei dem avg_loss 0 ist, um Division durch Null zu vermeiden
    rsi[avg_loss == 0] = 100

    # Behandeln Sie den Fall, bei dem avg_gain 0 ist
    rsi[avg_gain == 0] = 0

    return rsi

# Berechnung des RSI für den DataFrame `stock_data2`
stock_data2['RSI'] = compute_RSI(stock_data2)

