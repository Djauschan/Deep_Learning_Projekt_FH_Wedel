import pandas as pd

from src.prediction.abstract_model import AbstractModel
from src.prediction.modelWrapper import ModelWrapper
from src.preprocessing.services.ConfigService import ConfigService
from src.prediction.abstract_model import resolution
from src.preprocessing.services.DifferencingService import differencingService


class ModelExe(AbstractModel):
    def __init__(self):
        self.configService = ConfigService()
        configPath = "./configs/execution/docker_PredictionConfig.yml"
        #configPath = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\execution\\PredictionConfig.yml"
        self.config = self.configService.loadModelConfig(configPath)
        self.modelWrapper = ModelWrapper(self.config)

    def predict(self, symbol_list: list, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp,
                res) -> pd.DataFrame:
        return self.modelWrapper.collective_predict(symbol_list, res, timestamp_start, timestamp_end)

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
#Test Code

model_Exe = ModelExe()
startDate = pd.Timestamp("2021-02-01 04:00:00")
endDate = pd.Timestamp("2021-02-18 16:00:00")
t = model_Exe.predict(["AAL", "AAPL", "TSLA"], startDate, endDate, resolution.TWO_HOURLY)
print(t)
"""