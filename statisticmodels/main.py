import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from preprocessing.txtReader import DataReader
from preprocessing.data_cleaning import DataCleaner
from models.arima import Arima
from sklearn.model_selection import TimeSeriesSplit
 
 
 
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
 
 
arimaData['ds'] = pd.to_datetime(arimaData['ds'])
oberGrenze= arimaData[arimaData['ds']<="2017-11-30"]['ds']
unterGrenze=arimaData[arimaData['ds']>"2017-11-30"]['ds']
 
train=arimaData[arimaData['ds']<="2017-11-30"]['y'].values
test=arimaData[arimaData['ds']>"2017-11-30"]['y'].values
 
ArimaModell=Arima(train)
 
 
forecast= ArimaModell.forecast(5)
 
plt.plot(arimaData[arimaData['ds']>"2017-11-30"]['ds'][0:2] ,arimaData[arimaData['ds']>'2017-11-30']['y'][0:2].values, color='red', label='echte Daten')
plt.plot( arimaData[arimaData['ds']>"2017-11-30"]['ds'][0:2],forecast['mean'][0:2], color='blue', label='Prognose')
 
plt.plot(arimaData[arimaData['ds']<="2017-11-30"]['ds'][-3:],train[-3:],color='purple', label='Echte WERT VERGANGENHEIT')
plt.plot(arimaData[arimaData['ds']<="2017-11-30"]['ds'][-3:],ArimaModell.fittedValues['fitted'][-3:],color='green', label='FIT AMK')
plt.legend()
plt.show()
 
 
 
 
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

