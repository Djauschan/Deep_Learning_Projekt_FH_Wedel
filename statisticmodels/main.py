import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from preprocessing.txtReader import DataReader
from preprocessing.data_cleaning import DataCleaner
from models.visualisation import VisualizeStatsModel
from models.arima import Arima
from sklearn.model_selection import TimeSeriesSplit
from preprocessing.evaluation import Evaluation
 
 
 
 
test = DataReader({"READ_ALL_FILES": "READ_ALL_FILES"})
 
txt_files, symbols = test.get_txt_files()
 
# Test for-Schleife später löschen
for i in symbols:
    print(i)
 
test.current_file_idx = 1
 
df = test.read_next_txt()
clean = DataCleaner(df)
 
 
 
print(clean.df_at_16.info())
print(clean.df_at_09.info())
df16Cleaned=clean.df_at_16
df09Cleaned=clean.df_at_09
 
arimaData= clean.transformForNixtla(df16Cleaned)
print(arimaData.head())
print(arimaData.info())
 
yWerte=arimaData['y'].values
 
 
 
 
 
 
 
crossValidierung=Evaluation(30,3,yWerte)
 
testarray, Resultdic= crossValidierung.CrossValidation('arima')
 
print(Resultdic.keys())
print(Resultdic.values())
print('ich bin Kevin')
print(testarray)
 
 
 
# for keys,value in Resultdic.items():
#     print('Das sind die Metriken für ',keys)
#     temp=crossValidierung.metricsCalculation(testarray,value)
#     print(temp)
 
viz=VisualizeStatsModel(arimaData,testarray,testarray,testarray)
 
viz.arimaCrossvalidation(testarray,Resultdic)
 
# plt.style.use('fivethirtyeight')
# colors = ['b', 'g', 'r', 'c', 'm']
 
 
# plt.figure(figsize=(10, 6))
# for i,(key, values) in enumerate(Resultdic.items()):
#     plt.plot(values, label=key)
 
# # Achsentitel und Legende hinzufügen
# plt.plot(testarray,label='ECHTE DATEN AMK',color='k')
# plt.title('Prognosen')
# plt.xlabel('Zeitpunkt')
# plt.ylabel('Wert')
# plt.legend()
 
# plt.show()
 
 
# arimaData['ds'] = pd.to_datetime(arimaData['ds'])
# oberGrenze= arimaData[arimaData['ds']<="2017-11-30"]['ds']
# unterGrenze=arimaData[arimaData['ds']>"2017-11-30"]['ds']
 
 
# train=arimaData[arimaData['ds']<="2017-11-30"]['y'].values
# test=arimaData[arimaData['ds']>"2017-11-30"]['y'].values
# # Normal Arima:  
# ArimaModell=Arima(train)
 
 
# forecast= ArimaModell.forecast(5,train)
 
# chrrr=VisualizeStatsModel(arimaData, forecast,train,test )
 
# chrrr.simpleViz(unterGrenze[0:5],5)
 
 
# #pct_Change Forecast
# forecast_pct=ArimaModell.percent_change(100,train)
# chrrr.simpleViz(unterGrenze[0:100],100,prediction=forecast_pct)
 
# #sqrt Forecast
# forecast_sqr=ArimaModell.squareTransformer(30,train)
# chrrr.simpleViz(unterGrenze[0:30],30,prediction=forecast_sqr)
 
# #log Forecast
# forecast_log=ArimaModell.logTransformer(30,train)
# chrrr.simpleViz(unterGrenze[0:30],30,prediction=forecast_log)
 
# #log_shift Forecast
# forecast_log_shift=ArimaModell.shift_logTransformer(30,train)
# chrrr.simpleViz(unterGrenze[0:30],30,prediction=forecast_log_shift)
 
 
 
 
 
 
 
 
# # Extract Closing values and Create TimeSeriesSplit splits
# X = arimaData['y'].values
# splits = TimeSeriesSplit(n_splits=5,test_size=30)
 
 
# plt.figure(1)
# index = 0
# # Loop over splits to update train-test splits and models
# fig,ax=plt.subplots(nrows=5)
# for train_index, test_index in splits.split(X):
#   # Create and update train-test splits
#   train = X[train_index]
#   test = X[test_index]
#   print('Observations: %d' % (len(train) + len(test)))
#   print('Training Observations: %d' % (len(train)))
#   print('Testing Observations: %d' % (len(test)))
#   # Enter model and evaluation technique here
 
 
#   # Subplots for each train-test splits
   
#   ax[index].plot(train,'green')
#   ax[index].plot([None for i in train] + [x for x in test],'blue')
#   index += 1
# plt.show()
 
 
 
# Extract Closing values and Create TimeSeriesSplit splits
# X = arimaData['y'].values
# splits = TimeSeriesSplit(n_splits=3,test_size=100)
 
 
# plt.figure(1)
# index = 0
# # Loop over splits to update train-test splits and models
# fig,ax=plt.subplots(nrows=3)
# for train_index, test_index in splits.split(X):
#   # Create and update train-test splits
#   train = X[train_index]
#   test = X[test_index]
#   print('Observations: %d' % (len(train) + len(test)))
#   print('Training Observations: %d' % (len(train)))
#   print('Testing Observations: %d' % (len(test)))
#   # Enter model and evaluation technique here
 
 
#   # Subplots for each train-test splits
   
#   ax[index].plot(train,'green')
#   ax[index].plot([None for i in train] + [x for x in test],'blue')
#   index += 1
# plt.show()

