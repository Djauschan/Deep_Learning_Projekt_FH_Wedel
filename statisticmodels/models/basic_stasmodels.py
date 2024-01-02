import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsforecast.models import AutoTheta
from statsforecast.models import HistoricAverage
from statsforecast.models import Naive
from statsforecast.models import WindowAverage
# from statsforecast import GARCH -> Volatilität unzureichend geschätzt
# from statsforecast import ARCH -> wie oben
import matplotlib.pyplot as plt

class Exponentialsmoothing():

    def __init__(self):
        """
        Initializes the Exponentialsmoothing class.
        """
        self.exponentialModel=AutoETS()

    def forecast(self, forecasthorizon:int, trainData):
        """
        Generates a forecast for a given forecasthorizon.

        args:
        forecasthorizon(int): days that have to be predicted
        trainData: data for the model training

        return:
        predicitons for the forecast horizon
        """
        prediction= self.exponentialModel.forecast(trainData,forecasthorizon)
        return prediction ['mean']

class Theta():

    def __init__(self):
        """
        Initializes the Theta Class.
        """
        self.thetaModel=AutoTheta()

    def forecast(self, forecasthorizon:int, trainData):
        """
        Generates a forecast for a given forecasthorizon.

        args:
        forecasthorizon(int): days that have to be predicted
        trainData: data for the model training

        return:
        predicitons for the forecast horizon
        """
        prediction=self.thetaModel.forecast(trainData,forecasthorizon)
        return prediction['mean']

class Historic_average():

    def __init__(self):
        self.histModel= HistoricAverage()

    def forecast(self, forecasthorizon:int, trainData):
        prediciton=self.histModel.forecast(trainData,forecasthorizon)
        return prediciton['mean']

class Naive_(): 
    def __init__(self):
        self.naivModel= Naive() 

    def forecast(self, forecasthorizon:int, trainData):
        prediciton=self.naivModel.forecast(trainData,forecasthorizon)
        return prediciton['mean']

class Window_average(): 
    def __init__(self):
        self.windowModel= WindowAverage(window_size=30) 

    def forecast(self, forecasthorizon:int, trainData):
        prediciton=self.windowModel.forecast(trainData,forecasthorizon)
        return prediciton['mean']




