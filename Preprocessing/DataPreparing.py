import pandas as pd

def read_data_from_txt(file_name):
    """
    Liest Daten aus einer txt-Datei und erstellt einen DataFrame.
    """
    df = pd.read_csv(file_name, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def compute_RSI(data, window=14):
    delta = data.diff()
    loss = delta.clip(upper=0)
    gain = delta.clip(lower=0)
    avg_loss = loss.rolling(window=window, min_periods=1).mean().abs()
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    rsi[avg_gain == 0] = 0
    return rsi

def split_data(df, train_ratio=0.7, test_ratio=0.15):
    """
    Teilt die Daten in Trainings-, Test- und Validierungsdatensätze auf.
    """
    train_end = int(len(df) * train_ratio)
    test_end = train_end + int(len(df) * test_ratio)

    train_data = df.iloc[:train_end]
    test_data = df.iloc[train_end:test_end]
    valid_data = df.iloc[test_end:]

    return train_data, test_data, valid_data

# Lese Daten aus der Datei
data = read_data_from_txt(r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Preprocessing\AAL_1min.txt')  # Pfad aktualisieren

# Aggregieren der Daten zu Tagesdaten
daily_data = data.resample('D').last()

# Entfernen von Zeilen mit leeren Werten
daily_data = daily_data.dropna()

# Berechnen der gleitenden Durchschnitte und RSI für Tagesdaten
daily_data['ma5'] = daily_data['close'].rolling(window=5, min_periods=1).mean()
daily_data['ma30'] = daily_data['close'].rolling(window=30, min_periods=1).mean()
daily_data['ma200'] = daily_data['close'].rolling(window=200, min_periods=1).mean()
daily_data['rsi'] = compute_RSI(daily_data['close'])

# Auswahl der gewünschten Spalten
final_data = daily_data[['close', 'ma5', 'ma30', 'ma200', 'rsi']]

# Aufteilung der Daten
train_data, test_data, valid_data = split_data(final_data)

# Speichern der Datensätze als CSV-Dateien
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
valid_data.to_csv('valid_data.csv')
