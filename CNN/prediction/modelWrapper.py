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
        modelsToExecute = []
        if trading_type == resolution.MINUTE:
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'dayTrading']
        elif trading_type == resolution.TWO_HOURLY:
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'swingTrading']
        elif trading_type == resolution.DAILY:
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'longTrading']
        else:
            print("error")

        interval = TRADING_PARAMS.get(trading_type.value).get('interval')
        preprocessor = Preprocessor(self.config)
        for stock_symbol in symbol_list:
            modelInputData, endRawPrice = preprocessor.pipeline(stock_symbol, startDate, endDate,
                                                                length=20, interval=interval)
            stockResultArr = self.getAllPredictionsForSingleStock(modelsToExecute, stock_symbol, modelInputData, endDate, endRawPrice,
                                                                  interval)
            self.modelResults.append({'stockSymbol': stock_symbol, 'result': stockResultArr})

        return self.createPredictionDataframe(self.modelResults)

    @staticmethod
    def getAllPredictionsForSingleStock(modelsToExecute, stock_symbol, modelInputData, endDate, rawEndPrice, interval):
        stockResult = []
        for modelEntry in modelsToExecute:
            if modelEntry.get('stockSymbol') == stock_symbol:
                # (1, 5, 20, 20) size
                # predict single model
                model = modelEntry.get('model').predict(modelInputData)
                modelResult = model.get('prediction')
                horizon = modelEntry.get('horizon')
                predictedEndPrice = differencingService.calcThePriceFromChange(True, modelResult.item(),
                                                                               rawEndPrice)
                dateTimeOfPrediction = TimeModificationService.calcDateTimeFromStartDateAndInterval(endDate, interval,
                                                                                                    horizon)
                stockResult.append({'DateTime': dateTimeOfPrediction, 'result': predictedEndPrice})

        return stockResult

    def createPredictionDataframe(self, resultMap) -> pd.DataFrame:
        """
            IN: resultMap:
                    AAL:
                        DateTime: 01.01.2023:15:00, result: 16,02
                        DateTime: 01.01.2023:16:00, result: 17,08
                        DateTime: 01.01.2023:17:00, result: 18,01
                    AAPL:
                        DateTime: 01.01.2023:15:00, result: 16,02
                        DateTime: 01.01.2023:16:00, result: 17,08
                        DateTime: 01.01.2023:17:00, result: 18,21

            OUT:                    AAL,    APL...
                01.01.2022:15:00    12      22
                01.01.2022:16:00    14      24
                01.01.2022:17:00    16      25
        """
        stockList = ['Timestamp']
        timestamps = []
        predicts = []
        for item in resultMap:
            stockList.append(item.get('stock_symbol'))
            for result in item.get('result'):
                timestamps.append(result.get('DateTime'))
                predicts.append(result.get('result'))


        # prediction.set_index('Timestamp', inplace=True)
        # prediction = prediction.astype("Float64")
        return pd.DataFrame()

    def loadModels(self):
        for tradingTypes in self.modelsToLoad.items():
            tradingType = tradingTypes[0]
            horizons = tradingTypes[1]
            for horizonData in horizons:
                for horizon in horizonData:
                    item = horizonData[horizon]
                    folderPath = item.get('folder')
                    models = item.get('models')
                    for model in models:
                        for stock_symbol in model:
                            modelPath = model[stock_symbol]
                            path = os.path.join(self.MODEL_FOLDER, folderPath, modelPath)
                            modelObj = Model(stock_symbol, horizon, path)
                            self.modelCollection.append({'tradingType': tradingType, 'stockSymbol': stock_symbol,
                                                         'horizon': horizon, 'model': modelObj})
