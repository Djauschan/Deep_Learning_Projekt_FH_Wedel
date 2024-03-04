import pandas as pd

from CNN.prediction.abstract_model import AbstractModel
from CNN.prediction.modelWrapper import ModelWrapper
from CNN.preprocessing.services.ConfigService import ConfigService
from CNN.prediction.abstract_model import resolution
from CNN.preprocessing.services.DifferencingService import differencingService


class ModelExe(AbstractModel):
    def __init__(self):
        self.configService = ConfigService()
        # configPath = "./configDir/PredictionConfig.yml"
        configPath = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\execution\\PredictionConfig.yml"
        self.config = self.configService.loadModelConfig(configPath)
        self.modelWrapper = ModelWrapper(self.config)

    def predict(self, symbol_list: str, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp,
                res) -> pd.DataFrame:
        stock_symbols = symbol_list[1:-1].split(", ")
        return self.modelWrapper.collective_predict(stock_symbols, res, timestamp_start, timestamp_end)

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable"""
        """ not used, bc start & endtime needed, directly in "predict executed" 
        """
        pass

    def load_data(self) -> None:
        """load data from database and stores it in a class variable"""
        pass

    def load_model(self) -> None:
        """load model from file and stores it in a class variable

        """
        pass


"""
    Test Code
"""
model_Exe = ModelExe()
startDate = pd.Timestamp("2021-02-01 04:00:00")
endDate = pd.Timestamp("2021-02-18 16:00:00")
t = model_Exe.predict("[AAL, AAPL]", startDate, endDate, resolution.MINUTE)
