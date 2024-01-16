import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt
import pickle
from statsforecast import StatsForecast
import os

class Arima():
   
    def __init__(self, y_values):

        """
        Initializes the Arima Class

        Args:
        y_values: list of the values that the model will be trained on
        """
        self.y_values= y_values
        self.arimaModell= AutoARIMA(max_p=15,max_q=15)
        self.path='statisticmodels\models\savedModels'

    def fitAndSave(self, trainData, currentSymbol):
        modell=StatsForecast([AutoARIMA(max_p=15,max_q=15)],freq='B')
        modell.fit(trainData)
        path="statisticmodels\models\savedModels\\"+currentSymbol+'.pkl' 
        return modell.save(path)

    def loadAndPredict(self, path,horizon):
        model= StatsForecast.load(path)
        return model.predict(horizon)
       
 
    def forecast(self, forecasthorizon:int, trainData):
        """
        Calculates the forecast for a given forecasthorizon
        args:
        forcasthorizon(int): days that are to be predicted
        trainData: data to for the model training

        return:
        Forecast for the forecasthorizon
        """
       
        model = self.arimaModell.forecast(trainData,forecasthorizon)
        return model['mean']
 
    def logTransformer(self, forecasthorizon:int, trainData):
        """
        Transforms train data with the log function to make data stationary. 
        Generates a forecast with transformed data and backtransforms these data.

        args:
        forcasthorizon(int): days that are to be predicted
        trainData: data to for the model training

        return:
        Forecast for the forecasthorizon
        """
        lg_transformer=np.log(trainData)
        forecast =  self.forecast(forecasthorizon,lg_transformer)
        backtransformed_forecast=np.exp(forecast)
        return backtransformed_forecast
 
    def squareTransformer(self, forecasthorizon:int,trainData):
        """
        Transforms train data by squaring them to make data stationary. 
        Generates a forecast with transformed data and backtransforms these data.

        args:
        forcasthorizon(int): days that are to be predicted
        trainData: data to for the model training

        return:
        Forecast for the forecasthorizon
        """
        square_transformer= np.sqrt(trainData)
        forecast =  self.forecast(forecasthorizon,square_transformer)
        backtransformed_forecast=np.square(forecast)
        return backtransformed_forecast
 
 
    def percent_change(self, forecasthorizon:int,trainData):
        """
        Transforms train data by calculating the percentage change them to make data stationary. 
        Generates a forecast with transformed data and backtransforms these data.

        args:
        forcasthorizon(int): days that are to be predicted
        trainData: data to for the model training

        return:
        Forecast for the forecasthorizon
        """
        pct_change_transform= np.array(pd.Series(trainData).pct_change().dropna())
        forecast =  self.forecast(forecasthorizon,pct_change_transform)
        # last known value from the actual time series
        last_value = trainData[-1]
 
        # initialising the list for the backtransformed predictions
        backtransformed_forecast = np.array([last_value])
 
        # backtransformation for every predicition
        for pct_change in forecast:
            new_value = backtransformed_forecast[-1] * (1 + pct_change)
            backtransformed_forecast = np.append(backtransformed_forecast,new_value)
 
        # removing the last known value (last known value is not a part of the prediction)
        backtransformed_forecast = backtransformed_forecast[1:]
        return backtransformed_forecast
 
           
   
    def shift_logTransformer(self, forecasthorizon:int,trainData):
        """
        Transforms train data by using a shif log transformation to make data stationary. 
        Generates a forecast with transformed data and backtransforms these data.

        args:
        forcasthorizon(int): days that are to be predicted
        trainData: data to for the model training

        return:
        Forecast for the forecasthorizon
        """
        shift_log= np.log(np.array((trainData/pd.Series(trainData).shift(1)).dropna()))
       
        forecast= self.forecast(forecasthorizon,shift_log)
        # last known value from the actual time series
        last_value=trainData[-1]
       
        # initialising the list for the backtransformed predictions
        backtransformed_forecast= np.array([last_value])
       
        # backtransformation for every predicition
        for i in forecast:
            new_value= backtransformed_forecast[-1]*np.exp(i)
            backtransformed_forecast = np.append(backtransformed_forecast,new_value)
        
        backtransformed_forecast = backtransformed_forecast[1:]
 
        return backtransformed_forecast

    

    

