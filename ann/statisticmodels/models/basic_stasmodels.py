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
import os

class Exponentialsmoothing():

    def __init__(self):
        """
        Initializes the Exponentialsmoothing class.
        """
        self.exponentialModel=AutoETS()

    def fitAndSave(self, trainData, currentSymbol):
        model=StatsForecast([AutoETS()], freq='B')
        model.fit(trainData)
        path="statisticmodels\models\savedModelsETS\\"+currentSymbol+'.pkl' 
        return model.save(path)

    def loadAndPredict(self, path,horizon):
        model= StatsForecast.load(path)
        return model.predict(horizon)

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

    def fitAndSave(self, trainData, currentSymbol):
        model=StatsForecast([AutoTheta()], freq='B')
        model.fit(trainData)
        path="statisticmodels\models\savedModels\\"+currentSymbol+'_Theta.pkl' 
        return model.save(path)

    def loadAndPredict(self, path,horizon):
        model= StatsForecast.load(path)
        return model.predict(horizon)

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

    def fitAndSave(self, trainData, currentSymbol):
        model=StatsForecast([HistoricAverage()], freq='B')
        model.fit(trainData)
        path="statisticmodels\models\savedModels\\"+currentSymbol+'_historicAverage.pkl' 
        return model.save(path)

    def loadAndPredict(self, path,horizon):
        model= StatsForecast.load(path)
        return model.predict(horizon)

    def forecast(self, forecasthorizon:int, trainData):
        prediciton=self.histModel.forecast(trainData,forecasthorizon)
        return prediciton['mean']

class Naive_(): 
    def __init__(self):
        self.naivModel= Naive() 

    def fitAndSave(self, trainData, currentSymbol):
        model=StatsForecast([Naive()], freq='B')
        model.fit(trainData)
        path="statisticmodels\models\savedModels\\"+currentSymbol+'_Naive.pkl' 
        return model.save(path)

    def loadAndPredict(self, path,horizon):
        model= StatsForecast.load(path)
        return model.predict(horizon)

    def forecast(self, forecasthorizon:int, trainData):
        prediciton=self.naivModel.forecast(trainData,forecasthorizon)
        return prediciton['mean']

class Window_average(): 
    def __init__(self):
        self.windowModel= WindowAverage(window_size=30) 

    def fitAndSave(self, trainData, currentSymbol):
        model=StatsForecast([WindowAverage(window_size=30)], freq='B')
        model.fit(trainData)
        path="statisticmodels\models\savedModels\\"+currentSymbol+'_WindowAverage.pkl' 
        return model.save(path)

    def loadAndPredict(self, path,horizon):
        model= StatsForecast.load(path)
        return model.predict(horizon)   

    def forecast(self, forecasthorizon:int, trainData):
        prediciton=self.windowModel.forecast(trainData,forecasthorizon)
        return prediciton['mean']




