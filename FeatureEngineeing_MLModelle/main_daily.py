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
from ml_model import ActualValues, BaseModel, LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

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
# Verwendung der Klasse zum splitten der Daten
splitter = DataSplitter(data)
#splitter.split_by_ratio(split_ratio=0.8)  # Split für 80% Trainingsdaten
#splitter.split_by_date(pd.Timestamp('2019-12-31 16:00:00')) #Split zu diesen Datum
splitter.split_by_date_lag20d(pd.Timestamp('2021-01-03')) #Split zu diesen Datum mit beachtung der 20 Tage
    #Final 03.01.2021 

train_data = splitter.get_train_data()
test_data = splitter.get_test_data()

print("Maximaler Index train_data:", train_data.index.max()) #2021-01-01
print("Minimaler Index test_data:", test_data.index.min()) #2020-12-07 -> beim splitt - 20 Business Days wegen den LagFeatures
print("\n")

print("Trainingsdaten:\n", train_data)
print("Testdaten:\n", test_data)

# Plot um die aufteilung anzusehen
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['close'], label='Trainingsdaten')
plt.plot(test_data.index, test_data['close'], label='Testdaten')
plt.title('Aufteilung der Trainings- und Testdaten')
plt.xlabel('Index')
plt.ylabel('close')
plt.legend()
#plt.show()

####
#splitt train und test data durch pipeline
train_data = pipeline.fit_transform(train_data, 'busdaily')
test_data = pipeline.fit_transform(test_data, 'busdaily')

print("Maximaler Index train_data nach pip:", train_data.index.max()) #2021-01-01 #Freitag
print("Minimaler Index test_data nach pip:", test_data.index.min()) #2021-01-04 #Monatg -> passt, da Wochende kein Traden
print("\n")

print(train_data.columns)
print("split pip done \n") #nur zum testen

###############################################################
# data_columns = ['open', 'high', 'low', 'close', 'volume']

back_transform_train_data = train_data[['open', 'close']] #backup für zurück transfomieren der realen werten 
back_transform_test_data = test_data[['open', 'close']] #backup für zurück transfomieren der realen werten 

# Feature (X = unabhängige Variable) und Ziel (y = abhängige Variable) 
### split xy 
splitter = Xy_DataSplitter(train_data, test_data)
splitter.split_into_features_and_target('close') # Zielvariable 'close'

X_train = splitter.get_X_train()
X_test = splitter.get_X_test()
y_train = splitter.get_y_train()
y_test = splitter.get_y_test()

#prints nur zum testen
print("Maximaler Index train_data:", train_data.index.max())    # 2021-01-01
print("Minimaler Index test_data:", test_data.index.min())      # 2021-01-04
print("Maximaler Index X_train:", X_train.index.max())          # 2021-01-01
print("Minimaler Index X_test:", X_test.index.min())            # 2021-01-04

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

############################################################################################################################################################################
###################################### Verwendung der ML-Modelle ####################################################

#Länge der Prediction
prediction_length = 20

# Letzter bekannter Wert
last_known_open_value = back_transform_test_data['open'].iloc[0] # Letzter bekannter Open-Wert
last_known_close_value = back_transform_test_data['close'].iloc[0] # Letzter bekannter Close-Wert

# die Tatsächlicher Close-Wert für den entsprechenden Zeitpunkt
actual_values_class = ActualValues(pred_length=prediction_length)
actual_close_values, actual_indices = actual_values_class.get_actual_values_test_close(X_test, back_transform_test_data)

####### Initialisierung und Training der Modelle #######
############### LR ##################### 
lr_model = LinearRegressionModel(pred_length=prediction_length)
lr_model.fit(X_train, y_train)
lr_train_predictions = lr_model.train_predict(X_train)
lr_predictions, lr_indices = lr_model.predict(X_test, back_transform_test_data) #return predicted_close_values, indices # test prediction
print("LR Model done \n")
############### RF ##################### 
rf_model = RandomForestModel(pred_length=prediction_length)
rf_model.fit(X_train, y_train)
rf_train_predictions = rf_model.train_predict(X_train)
rf_predictions, rf_indices = rf_model.predict(X_test, back_transform_test_data)
print("RF Model done \n")
############### GBM ##################### 
gbm_model = GradientBoostingModel(pred_length=prediction_length)
gbm_model.fit(X_train, y_train)
gbm_train_predictions = gbm_model.train_predict(X_train)
gbm_predictions, gbm_indices = gbm_model.predict(X_test, back_transform_test_data)
print("GBM Model done \n")
############### SVM ##################### 
svm_model = SVMModel(pred_length=prediction_length)
svm_model.fit(X_train, y_train)
svm_train_predictions = svm_model.train_predict(X_train)
svm_predictions, svm_indices = svm_model.predict(X_test, back_transform_test_data)
print("SVM Model done \n")

#######################################################
############### Model extra Infos ##################### 
print("Mehr Infos zu den Modellen:\n")
lr_train_variance_score = lr_model.calculate_variance_score(X_train, y_train)
print("LR Train: Variance Score:", lr_train_variance_score)
lr_train_coefficients = lr_model.get_coefficients()
print("LR Train: Coefficients:", lr_train_coefficients)

lr_variance_score = lr_model.calculate_variance_score(X_test, y_test)
print("LR Test: Variance Score:", lr_variance_score)
lr_coefficients = lr_model.get_coefficients()
print("LR Test:Coefficients:", lr_coefficients)

rf_feature_importances = rf_model.get_feature_importances()
print("RF Feature Importances:", rf_feature_importances)

gbm_feature_importances = gbm_model.get_feature_importances()
print("GBM Feature Importances:", gbm_feature_importances)

svm_support_vectors = svm_model.get_support_vectors()
svm_n_support = svm_model.get_n_support()
svm_dual_coef = svm_model.get_dual_coef()
print("SVM Support Vectors:", svm_support_vectors)
print("SVM Number of Support Vectors per Class:", svm_n_support)
print("SVM Dual Coefficients:", svm_dual_coef)
print("\n")
##########################################################
############### Evaluation MAE, RMSE##################### 
print("Evaluation der Fehlermetricen:\n")
############ Train 
### LR 
lr_train_evaluation_results = lr_model.calculate_training_error(X_train, y_train)
print("Trainingsfehler für das LR-Modell:", lr_train_evaluation_results)

### RF
rf_train_evaluation_results = rf_model.calculate_training_error(X_train, y_train)
print("Trainingsfehler für das RF-Modell:", rf_train_evaluation_results)

### GBM
gbm_train_evaluation_results = gbm_model.calculate_training_error(X_train, y_train)
print("Trainingsfehler für das GBM-Modell:", gbm_train_evaluation_results)

### SVM
svm_train_evaluation_results = svm_model.calculate_training_error(X_train, y_train)
print("Trainingsfehler für das SVM-Modell:", svm_train_evaluation_results)

############### Test  
### LR evaluation  
lr_evaluation_results = lr_model.evaluate(lr_predictions, actual_close_values)
print("Testfehler für das LR-Modell:", lr_evaluation_results)

### RF evaluation  
rf_evaluation_results = rf_model.evaluate(rf_predictions, actual_close_values)
print("Testfehler für das RF-Modell:", rf_evaluation_results)

### GBM evaluation  
gbm_evaluation_results = gbm_model.evaluate(gbm_predictions, actual_close_values)
print("Testfehler für das GBM-Modell:", gbm_evaluation_results)

### SVM evaluation  
svm_evaluation_results = svm_model.evaluate(svm_predictions, actual_close_values)
print("Testfehler für das SVM-Modell:", svm_evaluation_results)

####################### plot ######################## 
###### Train + Test plot der evaluation
evaluation_results_plot = {
    'LR_Train': lr_train_evaluation_results,
    'LR_Test': lr_evaluation_results,
    'RF_Train': rf_train_evaluation_results,
    'RF_Test': rf_evaluation_results,
    'GBM_Train': gbm_train_evaluation_results,
    'GBM_Test': gbm_evaluation_results,
    'SVM_Train': svm_train_evaluation_results,
    'SVM_Test': svm_evaluation_results,
}

# Metriken, die Sie vergleichen möchten
metrics = ['MAE', 'RMSE'] #, 'MSLE', 'Median AE']
n_models = len(evaluation_results_plot)
n_metrics = len(metrics)

# X-Achsen-Positionen für jedes Modell
x = np.arange(n_metrics)
bar_width = 0.1  # Breite der Balken

model_colors = {
    'LR_Train': '#FF9999',  # Light Red
    'LR_Test': '#FF0000',   # Red
    'RF_Train': '#99FF99',  # Light Green
    'RF_Test': '#008000',   # Green
    'GBM_Train': '#ADD8E6', # Light Blue
    'GBM_Test': '#0000FF',  # Blue
    'SVM_Train': '#FFCC99', # Light Orange
    'SVM_Test': '#FFA500',  # Orange
}

plt.figure(figsize=(12, 6))

# Berechnen Sie die X-Achsen-Positionen für jedes Modell
x = np.arange(n_metrics) * (n_models + 1) * bar_width

# Erstellung der Balkendiagramme für Trainings- und Testfehler
for i, model in enumerate(evaluation_results_plot.keys()):
    for j, metric in enumerate(metrics):
        # Berechnen Sie die Position für jeden Balken
        position = x[j] + i * bar_width

        # Plot
        plot_val = evaluation_results_plot[model][metric]
        bar_plot = plt.bar(position, plot_val, width=bar_width, color=model_colors[model], label=model if j == 0 else "")

        # Beschriftung der Balken
        plt.text(position, plot_val, f'{plot_val:.2f}', ha='center', va='bottom')

plt.xlabel('Metriken')
plt.ylabel('Werte')
plt.title('Vergleich der Modelle basierend auf verschiedenen Metriken - Trainings- und Testfehler')
plt.xticks(x + bar_width * n_models / 2 - bar_width, metrics)
plt.legend()
#plt.show()

#######################################################################################
### plot der vorhergesagten und tatsächlichen Werte

### alle prediction der modelle in einen datensatz
predictions_data = {
    "lr_prediction": lr_predictions,
    "rf_prediction": rf_predictions,
    "gbm_prediction": gbm_predictions,
    "svm_prediction": svm_predictions,
}
predictions_data

# Visualisierung der vorhergesagten und tatsächlichen Close-Werte
plt.figure(figsize=(12, 6))

plt.plot(actual_indices, actual_close_values, label='Tatsächliche Werte', color='black')
plt.plot(actual_indices, lr_predictions, label='LR Vorhergesagte Werte', color='red')
plt.plot(actual_indices, rf_predictions, label='RF Vorhergesagte Werte', color='green')
plt.plot(actual_indices, gbm_predictions, label='GBM Vorhergesagte Werte', color='blue')
plt.plot(actual_indices, svm_predictions, label='SVM Vorhergesagte Werte', color='orange')

plt.scatter(actual_indices, actual_close_values, color='black')
plt.scatter(actual_indices, lr_predictions, color='red')
plt.scatter(actual_indices, rf_predictions, color='green')
plt.scatter(actual_indices, gbm_predictions, color='blue')
plt.scatter(actual_indices, svm_predictions, color='orange')

plt.title('Vergleich der vorhergesagten und tatsächlichen Close-Werte')
plt.xlabel('Zeitpunkt')
plt.ylabel('Close-Wert')
plt.legend()
plt.show()

