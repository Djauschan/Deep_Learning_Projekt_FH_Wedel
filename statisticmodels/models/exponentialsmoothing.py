import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsforecast.models import AutoTheta
import matplotlib.pyplot as plt

class Exponentialsmoothing():

    def __init__(self):

        self.exponentialModel=AutoETS()

    def forecast(self, forecasthorizon:int, trainData):
        prediction= self.exponentialModel.forecast(trainData,forecasthorizon)
        return prediction ['mean']

class Theta:
    def __init__(self):
        self.thetaModel=AutoTheta()

    def forecast(self, forecasthorizon:int, trainData):
        prediction=self.thetaModel.forecast(trainData,forecasthorizon)
        return prediction['mean']




