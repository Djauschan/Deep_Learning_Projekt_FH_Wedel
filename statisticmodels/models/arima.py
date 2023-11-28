import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt
 
class Arima():
   
    def __init__(self, y_values):
       
        self.y_values= y_values
        self.arimaModell= AutoARIMA(max_p=15,max_q=15)
       
       
       
 
    def forecast(self, forecasthorizon:int, trainData):
       
        model = self.arimaModell.forecast(trainData,forecasthorizon)
        return model['mean']
 
    def logTransformer(self, forecasthorizon:int, trainData):
        lg_transformer=np.log(trainData)
        forecast =  self.forecast(forecasthorizon,lg_transformer)
        rücktransformierte_werte=np.exp(forecast)
        return rücktransformierte_werte
 
    def squareTransformer(self, forecasthorizon:int,trainData):
        square_transformer= np.sqrt(trainData)
        forecast =  self.forecast(forecasthorizon,square_transformer)
        rücktransformierte_werte=np.square(forecast)
        return rücktransformierte_werte
 
 
    def percent_change(self, forecasthorizon:int,trainData):
        pct_change_transform= np.array(pd.Series(trainData).pct_change().dropna())
        forecast =  self.forecast(forecasthorizon,pct_change_transform)
        # Letzter bekannter Wert der ursprünglichen Zeitreihe
        letzter_wert = trainData[-1]
 
        # Initialisieren der Liste für die rücktransformierten Vorhersagen
        rücktransformierte_werte = np.array([letzter_wert])
 
        # Rücktransformation für jede Vorhersage
        for pct_change in forecast:
            neuer_wert = rücktransformierte_werte[-1] * (1 + pct_change)
            rücktransformierte_werte = np.append(rücktransformierte_werte,neuer_wert)
 
        # Entfernen des ersten Werts (ursprünglicher letzter bekannter Wert)
        rücktransformierte_werte = rücktransformierte_werte[1:]
        return rücktransformierte_werte
 
           
   
    def shift_logTransformer(self, forecasthorizon:int,trainData):
        shift_log= np.log(np.array((trainData/pd.Series(trainData).shift(1)).dropna()))
       
        forecast= self.forecast(forecasthorizon,shift_log)
        letzter_wert=trainData[-1]
        rücktransformierte_werte = np.array([letzter_wert])
       
 
        for i in forecast:
            neuer_wert= rücktransformierte_werte[-1]*np.exp(i)
            rücktransformierte_werte = np.append(rücktransformierte_werte,neuer_wert)
        rücktransformierte_werte = rücktransformierte_werte[1:]
 
        return rücktransformierte_werte

    

    

