import numpy as np
import pandas as pd 
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

class Arima():
    
    def __init__(self, data:pd.DataFrame()):
        self.data= data
        self.y_values= np.array(data['y'])
        self.arimaModell= AutoARIMA()

    def fitArima(self):
                

    

