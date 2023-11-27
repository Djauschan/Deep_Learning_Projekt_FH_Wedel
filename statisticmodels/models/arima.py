import numpy as np
import pandas as pd 
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

class Arima():
    
    def __init__(self, y_values):
        self.y_values= y_values
        self.arimaModell= AutoARIMA(max_p=30,max_q=15)
        self.fittedModel= self.arimaModell.fit(self.y_values)
        self.fittedValues= self.fittedModel.predict_in_sample()

    def forecast(self, forecasthorizon:int):
        model = self.arimaModell.forecast(self.y_values,forecasthorizon)
        return model

    

    

