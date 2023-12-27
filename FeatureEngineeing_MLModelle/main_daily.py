#Package einlesen
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#classe / other scripts
from txtReader import DataReader
from data_cleaning import DataCleaner
from feature_ts import FeatureEngineering
from pipeline import ClassPipeline
from split import DataSplitter
from split_xy import Xy_DataSplitter


#################################
# Daten einlesen
data = DataReader({"READ_ALL_FILES": "READ_ALL_FILES"})
txt_files, symbols = data.get_txt_files()
 
# Test for-Schleife später löschen
for i in symbols:
    print(i)
# 0 = AAL, 1 = AAPL, ...   #von den 10 datas
data.current_file_idx = 1
df = data.read_next_txt()
print("\nData:", df.symbol[0], "\n")

data_columns = df.columns
print(data_columns) #nur zum testen

#Data clean
cleaner = DataCleaner(df)

df_b_daily = cleaner.busi() #daily business days, Wert von 16 Uhr

data_columns = df_b_daily.columns
print(data_columns, "\n") # ['open', 'high', 'low', 'close', 'volume']


#features + pipeline
data = df_b_daily

# Anwendung der Pipeline
pipeline = ClassPipeline(data)

data_pip = pipeline.fit_transform(data, 'busdaily')
print(data_pip.columns)
print(data_pip)
print("pip läuft \n")


############################# start #####################
# Verwendung der Klasse zum splitten der Daten aktuell erstmal train test 80-20
splitter = DataSplitter(data)

#splitter.split_by_ratio(split_ratio=0.8)  # Split für 80% Trainingsdaten
#splitter.split_by_date(pd.Timestamp('2019-12-31 16:00:00')) #Split zu diesen Datum

splitter.split_by_date_lag20d(pd.Timestamp('2019-12-31 16:00:00')) #Split zu diesen Datum mit beachtung der 20 Tage

train_data = splitter.get_train_data()
test_data = splitter.get_test_data()

print("Maximaler Index train_data:", train_data.index.max()) #2019-12-31
print("Minimaler Index test_data:", test_data.index.min()) #2019-12-04 -> richtig beim splitt - 20 BD
print("\n")

print("Trainingsdaten:\n", train_data)
print("Testdaten:\n", test_data)

# Plot um die aufteilung anzusehen
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['close'], label='Trainingsdaten', linewidth=2)
plt.plot(test_data.index, test_data['close'], label='Testdaten')
plt.title('Aufteilung der Trainings- und Testdaten')
plt.xlabel('Index')
plt.ylabel('close')
plt.legend()
#plt.show()

####
#spltt data durch pipeline
train_data = pipeline.fit_transform(train_data, 'busdaily')
test_data = pipeline.fit_transform(test_data, 'busdaily')

print("Maximaler Index train_data nach pip:", train_data.index.max()) #2019-12-31
print("Minimaler Index test_data nach pip:", test_data.index.min()) #2020-01-01 -> passt
print("\n")

print(train_data.columns)
print("split pip done \n") #nur zum testen

###############################################################
# data_columns = ['open', 'high', 'low', 'close', 'volume']

back_transform_train_data = train_data[['open', 'close']] #backup für zurück transfomieren der realen werten 
back_transform_test_data = test_data[['open', 'close']] #backup für zurück transfomieren der realen werten 

# Feature (X = unabhängige Variable) und Ziel (y = abhängige Variable) 
# Zielvariable 'close'
###
splitter = Xy_DataSplitter(train_data, test_data)
splitter.split_into_features_and_target('close') # Zielvariable 'close'

# Zugriff auf die aufgeteilten Daten
X_train = splitter.get_X_train()
X_test = splitter.get_X_test()
y_train = splitter.get_y_train()
y_test = splitter.get_y_test()

#prints nur zum testen
print("Maximaler Index train_data:", train_data.index.max()) #2019-12-31 
print("Minimaler Index test_data:", test_data.index.min()) # 2020-01-01
print("Maximaler Index X_train:", X_train.index.max()) #2019-12-31
print("Minimaler Index X_test:", X_test.index.min()) #2020-01-01

print("X_train")
print(X_train.columns)
print(X_train)
print("X_test")
print(X_test.columns)
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)

print("done, now ml-model \n") #nur zum testen


###########################################
####### Verwendung der ML-Modelle #########
# Initialisierung und Training des linearen Regressionsmodells

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

last_known_open_value = back_transform_test_data['open'].iloc[0]
last_known_close_value = back_transform_test_data['close'].iloc[0] # Letzter bekannter Close-Wert

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Erstellen von leeren Listen für die gespeicherten Werte
predicted_close_values = []
actual_close_values = []
indices = []

######
#for i in range(len(X_test)):  # Für jeden Zeitpunkt im Testdatensatz
for i in range(min(20, len(X_test))): #Für die nächten 10 Index
    # Umwandlung der Daten in DataFrame mit Feature-Namen
    X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)

    # Vorhersage für den nächsten Tag (prozentuale Veränderung)
    predicted_pct_change = model.predict(X_test_row_df)[0]

    # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
    predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
    predicted_close_values.append(predicted_close)

    # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
    last_known_close_value = back_transform_test_data["close"].iloc[i]

    # Tatsächlicher Close-Wert für den entsprechenden Zeitpunkt
    actual_close = back_transform_test_data["close"].iloc[i]
    actual_close_values.append(actual_close)

    # Speichern des Indizes
    indices.append(X_test.index[i])

# Berechnung des MAE
mae = mean_absolute_error(actual_close_values, predicted_close_values)

# Berechnung des RMSE
rmse = np.sqrt(mean_squared_error(actual_close_values, predicted_close_values))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Visualisierung der vorhergesagten und tatsächlichen Close-Werte
plt.figure(figsize=(12, 6))
plt.plot(indices, predicted_close_values, label='Vorhergesagte Close-Werte', marker='.')
plt.plot(indices, actual_close_values, label='Tatsächliche Close-Werte', marker='x')
plt.title('Vergleich der vorhergesagten und tatsächlichen Close-Werte')
plt.xlabel('Zeitpunkt')
plt.ylabel('Close-Wert')
plt.legend()
plt.xticks(rotation=45)  # Drehen der X-Achsen-Beschriftungen für bessere Lesbarkeit
plt.show()


