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

from gridsearchCV_ml import BaseModel, LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

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

splitter.split_by_date_lag20d(pd.Timestamp('2021-01-03')) #Split zu diesen Datum mit beachtung der 20 Tage
    #Final 03.01.2021 

train_data = splitter.get_train_data()
test_data = splitter.get_test_data()

print("Maximaler Index train_data:", train_data.index.max()) #2021-01-01
print("Minimaler Index test_data:", test_data.index.min()) #2020-12-07 -> beim splitt - 20 Business Days wegen den LagFeatures
print("\n")

print("Trainingsdaten:\n", train_data)
print("Testdaten:\n", test_data)

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

print("done, now ml-model \n") #nur zum testen

#####################################################################################################################
###################################### Verwendung der ML-Modelle ####################################################

#

#########################################################################################
######################### Hyperparameter suche for RF and GBM ##########################

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

print("Hyperparameter suche für RF und GBM")
# Definieren des Parametergitters für RF und GBM -> suche der besten Hyperparameter
search_space = {
    'n_estimators': [10, 20],                                       # Anzahl der Bäume im Wald
    'max_depth': [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],     # Maximale Tiefe der Bäume
    'min_samples_split': [2, 3, 4, 5, 10, 15, 20],                  # Mindestanzahl der Samples zum Teilen eines Knotens
    'min_samples_leaf': [1, 5, 10, 25, 50, 75 ],                    # Mindestanzahl der Samples in einem Blatt
    'max_features': ['auto', 'sqrt'],                               # Anzahl Features, die bei der Suche nach dem besten Split berücksichtigt werden
}

### RF
print("RF start:")
rf = RandomForestRegressor()

# Erstellen des Grid Search Objekts
CV_rf = GridSearchCV(estimator=rf, param_grid=search_space, cv=5)
CV_rf.fit(X_train, y_train) 

print("Beste Hyperparameter:", CV_rf.best_params_)  # Ausgabe der besten Parameter
print("Beste Genauigkeit:", CV_rf.best_score_)      # Ausgabe der besten Modellgenauigkeit


### GBM
print("GBM start:")
gbm = GradientBoostingRegressor()

# Erstellen des Grid Search Objekts
CV_gbm = GridSearchCV(estimator=gbm, param_grid=search_space, cv=5)
CV_gbm.fit(X_train, y_train)

print("Beste Hyperparameter:", CV_gbm.best_params_) # Ausgabe der besten Parameter
print("Beste Genauigkeit:", CV_gbm.best_score_)     # Ausgabe der besten Modellgenauigkeit

########
print("Hyperparameter suche für SVM")
# Definieren des Parametergitters für SVM -> suche der besten Hyperparameter
search_space_svm = {
    'C': [0.1, 1, 10, 100],                             # Regularisierungsparameter
    'epsilon': [0.1, 0.2, 0.5, 1],                      # Epsilon im Verlustfunktion, relevant für SVR
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],     # Kernel-Art
}

### SVM
print("SVM start:")
svm_model = SVR()

# Erstellen des Grid Search Objekts
CV_svm = GridSearchCV(estimator=svm_model, param_grid=search_space_svm, cv=5) #Kreuzvalidierung 5
CV_svm.fit(X_train, y_train)

print("Beste Hyperparameter:", CV_svm.best_params_) # Ausgabe der besten Parameter
print("Beste Genauigkeit:", CV_svm.best_score_)     # Ausgabe der besten Modellgenauigkeit




#aktuellste:

'''
Hyperparameter suche
RF start:
Beste Hyperparameter: {'max_depth': 2, 'max_features': 'auto', 'min_samples_leaf': 50, 'min_samples_split': 2, 'n_estimators': 10}
Beste Genauigkeit: 0.00045184978621377604

GBM start:
Beste Hyperparameter: {'max_depth': 1, 'max_features': 'auto', 'min_samples_leaf': 75, 'min_samples_split': 2, 'n_estimators': 10}
Beste Genauigkeit: 0.0007969530261381586

Hyperparameter suche für SVM
SVM start:
Beste Hyperparameter: {'C': 0.1, 'epsilon': 0.5, 'kernel': 'rbf'}
Beste Genauigkeit: -0.003339281165786012
'''

