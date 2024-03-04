import os

import pandas as pd

from CNN.prediction.Model import Model
from CNN.prediction.abstract_model import resolution
from CNN.prediction.services.DataLoaderService import DataLoaderService
from CNN.prediction.services.ModelImportService import ModelImportService
from CNN.prediction.services.preprocessingServices import Preprocessor
from CNN.preprocessing.services.DifferencingService import differencingService
from CNN.preprocessing.services.TimeModificationService import TimeModificationService

TRADING_PARAMS = {
    'M': {'interval': 15, 'length': 20},  # dayTrading = minute
    'H': {'interval': 120, 'length': 20},  # swingTrading = hourly
    'D': {'interval': 300, 'length': 20}  # longTrading = day
}

STOCK_PARAMS = {

}


class ModelWrapper:
    """
        class that loads all modells and distributes requests to the correct modell
    """

    def __init__(self, config):
        self.config = config
        self.modelsToLoad = config['MODELS_TO_LOAD']
        self.MODEL_FOLDER = config['MODEL_FOLDER']
        self.modelCollection = []
        self.modelResults = []
        self.modelInputData = []
        self.loadModels()

    def collective_predict(self, symbol_list: list, trading_type: resolution, startDate: pd.Timestamp,
                           endDate: pd.Timestamp) -> pd.DataFrame:
        """
            redirect predict request from api/backend to correct model
        """
        predictions = pd.DataFrame([], columns=symbol_list)
        interval = TRADING_PARAMS.get(trading_type.value).get('interval')
        preprocessor = Preprocessor(self.config)
        for stock_symbol in symbol_list:
            modelInputData, endRawPrice = preprocessor.pipeline(stock_symbol, startDate, endDate,
                                                                length=20, interval=interval)
            stockResultArr = self.getAllPredictionsForSingleStock(stock_symbol, modelInputData, endDate, endRawPrice,
                                                                  interval)
            self.modelResults.append({stock_symbol: stockResultArr})

        return self.createPredictionDataframe(self.modelResults)

    def getAllPredictionsForSingleStock(self, stock_symbol, modelInputData, endDate, rawEndPrice, interval):
        stockResult = []
        for modelEntry in self.modelCollection:
            if modelEntry.get('stock_Symbol') == stock_symbol:
                # (1, 5, 20, 20) size
                # predict single model
                modelResult = modelEntry.get('model').predict(modelInputData)
                horizon = modelEntry.get('horizon')
                predictedEndPrice = differencingService.calcThePriceFromChange(True, modelResult, rawEndPrice)
                dateTimeOfPrediction = TimeModificationService.calcDateTimeFromStartDateAndInterval(endDate, interval,
                                                                                                    horizon)
                stockResult.append({'DateTime': dateTimeOfPrediction, 'result': predictedEndPrice})

        return stockResult

    def createPredictionDataframe(self, resultMap) -> pd.DataFrame:
        """
            (look up, Gruppenleiter-Meeting protokol
            creates dataframe to return to Interface Impl.
            in form of (return values in €)
                                AAL, APL...
            01.01.2022:15:00    12  22
            01.01.2022:16:00    14  24
            01.01.2022:17:00    16  25
        """
        pass


    def loadModels(self):
        for items in self.modelsToLoad:
            for tradingType in items.values():
                for horizon in tradingType:
                    for (folder, models) in horizon.values():
                        for model_entry in models.values():
                            for item in model_entry:
                                stock_symbol = list(item.keys())[0]
                                model_path = list(item.values())[0]
                                folder_path = os.path.join(self.MODEL_FOLDER, list(folder.values())[0])
                                model = Model(stock_symbol, horizon, os.path.join(folder_path, model_path))
                                self.modelCollection.append({'stock_Symbol': stock_symbol, 'horizon': horizon,
                                                         'model': model})
