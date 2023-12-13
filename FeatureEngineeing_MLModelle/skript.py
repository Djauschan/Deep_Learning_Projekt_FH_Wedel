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
print(data_columns)


#features + pipeline
data = df_hourly

# Anwendung der Pipeline
pipeline = ClassPipeline(data)
data_pip = pipeline.fit_transform(data, 'h')
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
plt.show()
print("plot done \n") #nur zum testen

####
#spltt data durch pipeline
train_data = pipeline.fit_transform(train_data, 'h')
test_data = pipeline.fit_transform(test_data, 'h')

print(train_data)
print(train_data.columns)
print("split pip done \n") #nur zum testen

# Feature (X = unabhängige Variable) und Ziel (y = abhängige Variable) 
# Zielvariable 'Close'
X_train = train_data[train_data.index < test_data.index.min()]
X_test = test_data[test_data.index >= test_data.index.min()]

y_train = train_data[train_data.index < test_data.index.min()]['close']
y_test = test_data[test_data.index >= test_data.index.min()]['close']

print("done, now ml-model \n") #nur zum testen



####### Verwendung der ML-Modelle #########
# Initialisierung und Training des linearen Regressionsmodells
lr_model = LinearRegressionModel()
lr_model.fit(X_train, y_train)
lr_evaluation_results = lr_model.evaluate(X_test, y_test)
lr_variance_score = lr_model.vs(X_test, y_test)
lr_coefficients = lr_model.coef()
print("lr done\n")

# Initialisierung und Training des Random Forest-Modells
rf_model = RandomForestModel()
rf_model.fit(X_train, y_train)
rf_evaluation_results = rf_model.evaluate(X_test, y_test)
rf_feature_importances = rf_model.fi()
print("rf done\n")

# Initialisierung und Training des Gradient Boosting-Modells
gbm_model = GradientBoostingModel()
gbm_model.fit(X_train, y_train)
gbm_evaluation_results = gbm_model.evaluate(X_test, y_test)
gbm_feature_importances = gbm_model.fi()
print("gbm done\n")

# Initialisierung und Training des SVM-Modells
#svm_model = SVMModel(kernel='rbf', C=1.0, epsilon=0.1)
#svm_model.fit(X_train, y_train)
#svm_evaluation_results = svm_model.evaluate(X_test, y_test)
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

### plot der ergebnisse
evaluation_results = {
    'Linear Regression': lr_evaluation_results,
    'Random Forest': rf_evaluation_results,
    'Gradient Boosting': gbm_evaluation_results,
    #'SVM': svm_evaluation_results
}
# Metriken, die Sie vergleichen möchten
metrics = ['MAE', 'RMSE', 'MSLE', 'Median AE']
n_models = len(evaluation_results)
n_metrics = len(metrics)

# Daten für das gruppierte Balkendiagramm vorbereiten
data = np.array([[evaluation_results[model][metric] for metric in metrics] for model in evaluation_results])

# X-Achsen-Positionen für jedes Modell
x = np.arange(n_metrics)
bar_width = 0.2  # Breite der Balken

plt.figure(figsize=(12, 8))

for i in range(n_models):
    plt.bar(x + i * bar_width, data[i], width=bar_width, label=list(evaluation_results.keys())[i])

plt.xlabel('Metriken')
plt.ylabel('Werte')
plt.title('Vergleich der Modelle basierend auf verschiedenen Metriken')
plt.xticks(x + bar_width * (n_models - 1) / 2, metrics)
plt.legend()

plt.show()

print("plot done\n")