import pandas as pd  
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from models.arima import Arima
from sklearn import metrics
 
 
 
class Evaluation:
 
    def __init__(self, splits, testlength, y_values):
        self.testlength=testlength
        self.splitter= TimeSeriesSplit(n_splits=splits,test_size=testlength)
        self.y_values=y_values
       
 
    def CrossValidation(self, model: str):
        testArray = np.array([])
        data = self.y_values
        prognoseDic = {}
        if model == 'arima':
            statsModell = Arima(self.y_values)
            prognoseDic = {'defaultArima': np.array([]),
                        'logTransformer': np.array([]),
                        'squareTransformer': np.array([]),
                        'pct_change': np.array([]),
                        'log_shift': np.array([])}
 
            for train_index, test_index in self.splitter.split(data):
                # Hier werden alle Testsets in einem Array gespeichert
                testArray = np.append(testArray, data[test_index])
 
                # Ab hier werden die Prognose für dieses Trainingsset generiert
                defaultArima = statsModell.forecast(self.testlength, data[train_index])
                prognoseDic['defaultArima'] = np.append(prognoseDic['defaultArima'], defaultArima)
 
                logTransformer = statsModell.logTransformer(self.testlength, data[train_index])
                prognoseDic['logTransformer'] = np.append(prognoseDic['logTransformer'], logTransformer)
 
                squareTransformer = statsModell.squareTransformer(self.testlength, data[train_index])
                prognoseDic['squareTransformer'] = np.append(prognoseDic['squareTransformer'], squareTransformer)
 
                pct_change = statsModell.percent_change(self.testlength, data[train_index])
                prognoseDic['pct_change'] = np.append(prognoseDic['pct_change'], pct_change)
 
                log_shift = statsModell.shift_logTransformer(self.testlength, data[train_index])
                prognoseDic['log_shift'] = np.append(prognoseDic['log_shift'], log_shift)
 
        return testArray, prognoseDic
 
    def metricsCalculation(self, testArray, prognoseArray):
        metricDic={'MAE':"",'MSE':"",'RMSE':""}
        metricDic['MAE']=metrics.mean_absolute_error(testArray,prognoseArray)
        metricDic['MSE']=metrics.mean_squared_error(testArray,prognoseArray)
        metricDic['RMSE']=np.sqrt(metricDic['MSE'])
 
        return metricDic