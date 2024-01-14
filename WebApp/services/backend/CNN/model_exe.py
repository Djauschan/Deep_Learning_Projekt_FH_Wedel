from abc import ABC

import numpy as np
import pandas as pd

from CNN.abstract_model import AbstractModel
from CNN.preprocessingServices import Preprocessor, ModelImportService, ConfigService


class ModelExe(AbstractModel):

    def __init__(self):
        self.configService = ConfigService()
        configPath = "./CNN/configDir/PredictionConfig.yml"
        #configPath = "C:\\Projekte\ProjectDeepLearning_CNN\\project_deeplearning\\src\\predictionApi\\configDir\\PredictionConfig.yml"
        self.parameters = self.configService.loadModelConfig(configPath)
        self.preprocessor = Preprocessor(self.parameters)
        self.gafData = np.zeros((1, 1))
        """
            a collection of multiple models. Each Model can only predict a single value,
            in order to get a series of multiple timeStamps, multiple Models will predict values 
            within a range of 480min
            => model 1 => predict 1* 480min ahead
            => model 2 => predict 2* 480min ahead
            => model 2 => predict 3* 480min ahead
            => model 6 => predict 6* 480min ahead = 2Tage
        """
        self.modelCollection = []
        self.load_model()

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        """predict stock price for a given time interval

        Args:
            timestamp_start (pd.Timestamp): start time of the time period
            timestamp_end (pd.Timestamp): end time of the time period
            interval (int): interval in minutes

        Returns:
            pd.DataFrame: dataframe with columns: timestamp, 1-n prices of stock_symbols
        """
        print("TEST1++++++++++++++++++++++++++++++")
        toReturnDataFrame = pd.DataFrame(columns=["Timestamp", "AAPL"])
        interval = 480  # can not be changed, model is trained on this specific interval
        modelInputList, endPriceList = self.preprocessor.pipeline(timestamp_start, timestamp_end)
        i = 0
        for model in self.modelCollection:
            calcTimeStamp = timestamp_end + pd.Timedelta(minutes=(i+1) * interval)
            y_change = model.forward(modelInputList[i])
            y_price = self._calcThePriceFromChange(y_change.item(), endPriceList[i])
            toReturnDataFrame.loc[i] = [calcTimeStamp, y_price]
            i += 1
        print("TEST2++++++++++++++++++++++++++++++")
        return toReturnDataFrame


    def _calcThePriceFromChange(self, y_change, endPrice):
        return 1 + y_change * endPrice

    def load_model(self) -> None:
        modelImportService = ModelImportService(self.parameters)
        listOfModelsToLoad = modelImportService.getSavedModelsPaths()
        for model_path in listOfModelsToLoad:
            self.modelCollection.append(modelImportService.loadModel(model_path))

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable"""
        """ not used, bc start & endtime needed, directly in "predict executed" 
        """
        pass

    def load_data(self) -> None:
        """load data from database and stores it in a class variable

        """
        pass
