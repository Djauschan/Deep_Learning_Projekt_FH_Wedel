import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsforecast.models import AutoTheta
from statsforecast import GARCH
from statsforecast import ARCH
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




