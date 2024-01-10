#main gridsearchCV

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

from gridsearchCV_ml import ActualValues, BaseModel, LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

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

splitter.split_by_date_lag20d(pd.Timestamp('2019-12-31 16:00:00')) #Split zu diesen Datum mit beachtung der 20 Tage

train_data = splitter.get_train_data()
test_data = splitter.get_test_data()

print("Maximaler Index train_data:", train_data.index.max()) #2019-12-31
print("Minimaler Index test_data:", test_data.index.min()) #2019-12-04 -> richtig beim splitt - 20 BD wegen den LagFeatures
print("\n")

print("Trainingsdaten:\n", train_data)
print("Testdaten:\n", test_data)

####
#splitt train und test data durch pipeline
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
### split xy 
splitter = Xy_DataSplitter(train_data, test_data)
splitter.split_into_features_and_target('close') # Zielvariable 'close'

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

#####################################################################################################################
###################################### Verwendung der ML-Modelle ####################################################

#

#########################################################################################
######################### Hyperparameter suche for RF and GBM ##########################

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

print("Hyperparameter suche")

# Definieren des Parametergitters -> suche der besten Hyperparameter
search_space = {
    'n_estimators': [10, 50, 100, 500],                             # Anzahl der Bäume im Wald
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],        # Maximale Tiefe der Bäume
    'min_samples_split': [2, 10, 20, 50],                           # Mindestanzahl der Samples zum Teilen eines Knotens
    'min_samples_leaf': [1, 5, 10, 50]                              # Mindestanzahl der Samples in einem Blatt
}

### RF
print("RF start:")
rf = RandomForestRegressor(random_state=11)

# Erstellen des Grid Search Objekts
CV_rf = GridSearchCV(estimator=rf, param_grid=search_space, cv=5)

# Durchführung des Grid Search
CV_rf.fit(X_train, y_train)

# Ausgabe der besten Parameter
print("Beste Hyperparameter:", CV_rf.best_params_)

# Optional: Ausgabe der besten Modellgenauigkeit
print("Beste Genauigkeit:", CV_rf.best_score_)


### GBM
print("GBM start:")
gbm = GradientBoostingRegressor(random_state=11)

# Erstellen des Grid Search Objekts
CV_gbm = GridSearchCV(estimator=gbm, param_grid=search_space, cv=5)

# Durchführung des Grid Search
CV_gbm.fit(X_train, y_train)

# Ausgabe der besten Parameter
print("Beste Hyperparameter:", CV_gbm.best_params_)

# Optional: Ausgabe der besten Modellgenauigkeit
print("Beste Genauigkeit:", CV_gbm.best_score_)
