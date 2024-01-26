#Package einlesen
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

#classe / other scripts
from txtReader import DataReader
from data_cleaning import DataCleaner
from feature_ts import FeatureEngineering
from pipeline import ClassPipeline
from split import DataSplitter
from split_xy import Xy_DataSplitter

#from ml_model import ActualValues, BaseModel, LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

from ml_model_abc import LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

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

#Data clean
cleaner = DataCleaner(df)
df_b_daily = cleaner.busi() #daily business days, Wert von 16 Uhr
data_columns = df_b_daily.columns

#features + pipeline
data = df_b_daily

# Anwendung der Pipeline
pipeline = ClassPipeline(data)
data_pip = pipeline.fit_transform(data, 'busdaily')

# Verwendung der Klasse zum splitten der Daten
splitter = DataSplitter(data)
splitter.split_by_date_lag20d(pd.Timestamp('2021-01-03')) #Split zu diesen Datum mit beachtung der 20 Tage  #Final 03.01.2021 

train_data = splitter.get_train_data()  #2021-01-01
test_data = splitter.get_test_data()    #2020-12-07 -> beim splitt - 20 Business Days wegen den LagFeatures

#splitt train und test data durch pipeline
train_data = pipeline.fit_transform(train_data, 'busdaily')
test_data = pipeline.fit_transform(test_data, 'busdaily')

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

###################################### Verwendung der ML-Modelle ####################################################

last_known_close_value = back_transform_test_data['close'].iloc[0] # Letzter bekannter Close-Wert

pfad_data = 'saved_pkl/Data'
back_transform_test_data.to_pickle(os.path.join(pfad_data, 'back_transform_test_data.pkl'))
X_test.to_pickle(os.path.join(pfad_data, 'X_test.pkl'))

####### Initialisierung und Training der Modelle , sowie speichern #######

# Modelle trainiert bis Timestamp('2021-01-03') -> ab nächsten Werktag 2021-01-05 möglich

############### LR ##################### 
lr_model = LinearRegressionModel()
lr_model.fitandsave(X_train, y_train)

############### RF ##################### 
rf_model = RandomForestModel()
rf_model.fitandsave(X_train, y_train)

############### GBM ##################### 
gbm_model = GradientBoostingModel()
gbm_model.fitandsave(X_train, y_train)

############### SVM ##################### 
svm_model = SVMModel()
svm_model.fitandsave(X_train, y_train)

print("Alle Modelle und Daten als pkl gesichert.")
print("Aktuell aber nur auf den Datensatz:")
print(df.symbol[0])


""" 
################## prediction test

############### LR ##################### 
lr_predictions = lr_model.predict(pd.Timestamp('2021-01-08'), pd.Timestamp('2021-01-10'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("lr_predictions")
print(lr_predictions)

############### RF ##################### 
rf_predictions = rf_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-7'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("rf_predictions")
print(rf_predictions)

############### GBM ##################### 
gbm_predictions = gbm_model.predict(pd.Timestamp('2021-01-04'), pd.Timestamp('2021-01-6'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("gbm_predictions")
print(gbm_predictions)

############### SVM ##################### 
svm_predictions = svm_model.predict(pd.Timestamp('2021-01-04'), pd.Timestamp('2021-01-6'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("svm_predictions")
print(svm_predictions)

"""
