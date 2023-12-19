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
from ml_model import LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

### main

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

df_normal = cleaner.normal()
df_hourly = cleaner.hourly()
df_minute = cleaner.minute()
df_daily = cleaner.daily()

df_normal

data_columns = df_normal.columns
print(data_columns, "\n")


#features + pipeline
data = df_normal

# Anwendung der Pipeline
pipeline = ClassPipeline(data)

data_pip = pipeline.fit_transform(data, 'test')
print(data_pip.columns)
print(data_pip)
print("pip läuft \n")

# Verwendung der Klasse zum splitten der Daten aktuell erstmal train test 80-20
splitter = DataSplitter(data)

splitter.split()
train_data = splitter.get_train_data()
test_data = splitter.get_test_data()

print("Trainingsdaten:\n", train_data)
print("Testdaten:\n", test_data)

# Plot um die aufteilung anzusehen
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['close'], label='Trainingsdaten')
plt.plot(test_data.index, test_data['close'], label='Testdaten')
plt.title('Aufteilung der Trainings- und Testdaten')
plt.xlabel('Index')
plt.ylabel('close')
plt.legend()
#plt.show()
print("plot done \n") #nur zum testen

####
#spltt data durch pipeline
train_data = pipeline.fit_transform(train_data, 'test')
test_data = pipeline.fit_transform(test_data, 'test')

print(train_data)
print(train_data.columns)
print("split pip done \n") #nur zum testen

######## Prüfen von NaN / zu große Werte / oder infintiv value ################
# Überprüfen auf extrem große Werte
max_float64 = np.finfo(np.float64).max
min_float64 = np.finfo(np.float64).min
print('Maximaler Wert in train_data:', train_data.max().max())
print('Maximaler Wert in test_data:', test_data.max())
print('Werte in train_data außerhalb des float64 Bereichs:', ((train_data > max_float64) | (train_data < min_float64)).sum().sum())
print('Werte in test_data außerhalb des float64 Bereichs:', ((test_data > max_float64) | (test_data < min_float64)).sum())

print('Anzahl der NaN-Werte in train_data:', train_data.isna().sum().sum()) # Überprüfung auf NaN Werte
print('Anzahl der unendlichen Werte in train_data:', np.isinf(train_data).sum().sum()) # Überprüfung auf unendliche Werte
print('Anzahl der NaN-Werte in test_data:', test_data.isna().sum().sum()) # Überprüfung auf NaN Werte
print('Anzahl der unendlichen Werte in test_data:', np.isinf(test_data).sum().sum()) # Überprüfung auf unendliche Werte
print("\prüfung done \n")
#########################

# Feature (X = unabhängige Variable) und Ziel (y = abhängige Variable) 
# Zielvariable 'Close'
X_train = train_data[train_data.index < test_data.index.min()]
X_test = test_data[test_data.index >= test_data.index.min()]

#X_train.drop(["close"], axis=1, inplace=True)
#X_test.drop(["close"], axis=1, inplace=True)

X_train.drop(["close_PctChange"], axis=1, inplace=True)
X_test.drop(["close_PctChange"], axis=1, inplace=True)
X_train.drop(["close_Differenz"], axis=1, inplace=True)
X_test.drop(["close_Differenz"], axis=1, inplace=True)

print(X_train.columns)

#y_train = train_data[train_data.index < test_data.index.min()]['close']
#y_test = test_data[test_data.index >= test_data.index.min()]['close']

y_train = train_data[train_data.index < test_data.index.min()]['close_PctChange']
y_test = test_data[test_data.index >= test_data.index.min()]['close_PctChange']

print(X_train)
print(y_train)

print("done, now ml-model \n") #nur zum testen

####### Verwendung der ML-Modelle #########
# Initialisierung und Training des linearen Regressionsmodells
lr_model = LinearRegressionModel()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_evaluation_results = lr_model.evaluate(lr_predictions, y_test)
lr_variance_score = lr_model.vs(X_test, y_test)
lr_coefficients = lr_model.coef()
print("lr done\n")

# Initialisierung und Training des Random Forest-Modells
rf_model = RandomForestModel( # Hyperparameter beim nächsten mal weiter anpassen
    n_estimators=10,      # Anpassen der Anzahl der Bäume im Wald
    max_depth=10,         # Anpassen der maximalen Tiefe der Bäume (None bedeutet keine Begrenzung)
    min_samples_split=2,  # Anpassen der Mindestanzahl von Beispielen, die für einen Split erforderlich sind
    min_samples_leaf=1    # Anpassen der Mindestanzahl von Beispielen in einem Blattknoten
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_evaluation_results = rf_model.evaluate(rf_predictions, y_test)
rf_feature_importances = rf_model.fi()
print("rf done\n")

# Initialisierung und Training des Gradient Boosting-Modells
gbm_model = GradientBoostingModel( # Hyperparameter beim nächsten mal weiter anpassen
    n_estimators=10,    # Die Anzahl der Bäume im GBM-Ensemble
    learning_rate=0.1,  # Die Lernrate, die angibt, wie stark jeder Baum die Vorhersagen beeinflusst
    max_depth=3         # Die maximale Tiefe der Bäume im GBM
    )
gbm_model.fit(X_train, y_train)
gbm_predictions = gbm_model.predict(X_test)
gbm_evaluation_results = gbm_model.evaluate(gbm_predictions, y_test)
gbm_feature_importances = gbm_model.fi()
print("gbm done\n")

# Initialisierung und Training des SVM-Modells
#svm_model = SVMModel(kernel='rbf', C=1.0, epsilon=0.1)
#svm_model.fit(X_train, y_train)
#svm_predictions = svm_model.predict(X_test)
#svm_evaluation_results = svm_model.evaluate(svm_predictions, y_test)
#print("svm done\n")

print("modelle fertig trainiert\n") #nur zum testen
print("Ausgabe der Evaluierungsergebnisse \n") #nur zum testen

#### Ausgabe der Evaluierungsergebnisse
# LR
print("Linear Regression Evaluation:")
for key, value in lr_evaluation_results.items():
    print(f"{key}: {value}")

print("\nLinear Regression Variance Score:")
for key, value in lr_variance_score.items():
    print(f"{key}: {value}")

print("\nLinear Regression Coefficients:")
for key, value in lr_coefficients.items():
    print(f"{key}: {value}")

#####
# RF
print("\nRandom Forest Evaluation:")
for key, value in rf_evaluation_results.items():
    print(f"{key}: {value}")

print("\nRandom Forest Feature Importances:")
for key, value in rf_feature_importances.items():
    print(f"{key}: {value}")

# GBM
print("\nGradient Boosting Machine Evaluation:")
for key, value in gbm_evaluation_results.items():
    print(f"{key}: {value}")

print("\nGBM Feature Importances:")
for key, value in gbm_feature_importances.items():
    print(f"{key}: {value}")

# SVM
#print("\nSVM Evaluation:")
#for key, value in svm_evaluation_results.items():
 #   print(f"{key}: {value}")

print("\n")

###### plot der ergebnisse
evaluation_results = {
    'Linear Regression': lr_evaluation_results,
    'Random Forest': rf_evaluation_results,
    'Gradient Boosting': gbm_evaluation_results,
    #'SVM': svm_evaluation_results
}
# Metriken, die Sie vergleichen möchten
metrics = ['MAE', 'RMSE'] #, 'MSLE', 'Median AE']
n_models = len(evaluation_results)
n_metrics = len(metrics)

# Daten für das gruppierte Balkendiagramm vorbereiten
data = np.array([[evaluation_results[model][metric] for metric in metrics] for model in evaluation_results])

# X-Achsen-Positionen für jedes Modell
x = np.arange(n_metrics)
bar_width = 0.2  # Breite der Balken
plt.figure(figsize=(12, 8))
for i in range(n_models):
    bars = plt.bar(x + i * bar_width, data[i], width=bar_width, label=list(evaluation_results.keys())[i])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom')

plt.xlabel('Metriken')
plt.ylabel('Werte')
plt.title('Vergleich der Modelle basierend auf verschiedenen Metriken')
plt.xticks(x + bar_width * (n_models - 1) / 2, metrics)
plt.legend()
#plt.show()

print("plot done\n")

#plot ml modelle
plt.figure(figsize=(12, 8))
plt.plot(y_test.index, y_test, color='black', label='Actual Values') # Tatsächliche Werte
plt.plot(y_test.index, lr_predictions, color='blue', label='Linear Regression Predictions')# Vorhersagen der linearen Regression
plt.plot(y_test.index, rf_predictions, color='green', label='Random Forest Predictions')# Vorhersagen des Random Forest
plt.plot(y_test.index, gbm_predictions, color='red', label='GBM Predictions') # Vorhersagen der GBM
#plt.plot(y_test.index, svm_predictions, color='orange', label='SVM Predictions') # Vorhersagen der SVM
plt.title('Model Predictions vs Actual Values')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()