from abc import ABC
from datetime import timedelta

import numpy as np
import pandas as pd
from CNN.abstract_model import AbstractModel
from CNN.preprocessingServices import ConfigService, ModelImportService, Preprocessor


class ModelExe(AbstractModel):
    def __init__(self):
        self.configService = ConfigService()
        configPath = "./configDir/PredictionConfig.yml"
        # configPath = "C:\\Projekte\ProjectDeepLearning_CNN\\project_deeplearning\\src\\CNN\\configDir\\PredictionConfig.yml"
        self.parameters = self.configService.loadModelConfig(configPath)
        self.preprocessor = Preprocessor(self.parameters)
        self.gafData = np.zeros((1, 1))
        """
            a collection of multiple models. Each Model can only predict a single value,
            in order to get a series of multiple timeStamps, multiple Models will predict values 
            within a range of 480min
            => model 1 => predict 1* 120min ahead
            => model 2 => predict 4* 120min ahead
            => model 3 => predict 8* 120min ahead
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
        # 10.01.2022 start
        # 13.01.2022 end
        tmpTimeStampStart = timestamp_start
        timestamp_start = timestamp_start - pd.Timedelta(days=13)
        ts_timeStampEnd = tmpTimeStampStart
        toReturnDataFrame = pd.DataFrame(columns=["Timestamp", "AAPL"])
        timeStamp = [pd.Timestamp(
            "2021-01-04 9:30"), pd.Timestamp("2021-01-04 16:00"), pd.Timestamp("2021-01-05 9:30")]
        # ahead = [1, 4, 8] #3 models predict, 1*interval ahead, 3*interval..
        interval = 120  # fixed, model trained on
        '''
        ModelInput muss (1,9,20,20) sein => (batch, anz_feat, length_singleTs, length_singleTs)
        '''
        modelInputList, endPriceList = self.preprocessor.pipeline(
            timestamp_start, ts_timeStampEnd)
        # toReturnDataFrame.loc[1] = [timestamp_end, -1.0]
        i = 0
        calcTimeStamp = tmpTimeStampStart
        for model in self.modelCollection:
            # calcTimeStamp = calcTimeStamp + pd.Timedelta(minutes=(interval * ahead[i]))
            # if 9 > calcTimeStamp.hour or calcTimeStamp.hour > 17:
            #     nextDay: pd.Timestamp = calcTimeStamp + timedelta(days=1)
            #     nextDayStartDay: pd.Timestamp = pd.Timestamp(year=nextDay.year, month=nextDay.month, day=nextDay.day,
            #     hour=9, minute=30, second=0)

            calcTimeStamp = timeStamp[i]
            # different input per Modell possible, but not for MVP
            y_change = model.forward(modelInputList[0])
            y_price = self._calcThePriceFromChange(
                y_change.item(), endPriceList[0])
            toReturnDataFrame.loc[i] = [calcTimeStamp, y_price]
            i += 1
        return toReturnDataFrame

    def _calcThePriceFromChange(self, y_change, endPrice):
        # because the differencing is in 1000, modelled trained that way
        return (1 + (y_change / 100)) * endPrice

    def load_model(self) -> None:
        modelImportService = ModelImportService(self.parameters)
        listOfModelsToLoad = modelImportService.getSavedModelsPaths()
        for model_path in listOfModelsToLoad:
            self.modelCollection.append(
                modelImportService.loadModel(model_path))

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable"""
        """ not used, bc start & endtime needed, directly in "predict executed" 
        """
        pass

    def load_data(self) -> None:
        """load data from database and stores it in a class variable"""
        pass


"""
model_Exe = ModelExe()
# model_Exe.predict(pd.Timestamp(year=2022, month=1, ))
#startDate = pd.Timestamp("2021-02-05 04:00:00") valid
#endDate = pd.Timestamp("2021-02-13 04:00:00") valid 21-02-12.. invalid
startDate = pd.Timestamp("2021-02-05 04:00:00")
endDate = pd.Timestamp("2021-02-18 04:00:00")
t = model_Exe.predict(startDate, endDate, 120)
"""
