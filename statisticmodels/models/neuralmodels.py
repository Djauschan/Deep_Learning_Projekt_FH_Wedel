import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS


class Nbeats(): 
    
    def __init__(self, horizon, ar):
        self.nbeatsModel= NBEATS(h=horizon,input_size=ar,max_steps=100)
        self.nf=NeuralForecast(models=[self.nbeatsModel],freq='B')

    def forecast(self,y_train):
        self.nf.fit(df=y_train)
        y_hat=self.nf.predict().reset_index()
        return y_hat

class Nhits(): 
    
    def __init__(self, horizon, ar):
        self.nhitsModel= NHITS(h=horizon,input_size=ar,max_steps=100)
        self.nf=NeuralForecast(models=[self.nhitsModel],freq='B')

    def forecast(self,y_train):
        self.nf.fit(df=y_train)
        y_hat=self.nf.predict().reset_index()
        return y_hat




