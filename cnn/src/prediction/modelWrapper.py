import os

import pandas as pd

from src.prediction.modelObject import Model
from src.prediction.abstract_model import resolution
from src.prediction.services.preprocessingServices import Preprocessor
from src.preprocessing.services.DifferencingService import differencingService
from src.preprocessing.services.TimeModificationService import TimeModificationService

TRADING_PARAMS = {
    'M': {'interval': 15, 'length': 20},  # dayTrading = minute
    'H': {'interval': 120, 'length': 20},  # swingTrading = hourly
    'D': {'interval': 300, 'length': 20}  # longTrading = day
}


class ModelWrapper:
    """
        class that loads all specified models from config,
        distributes requests to the correct model and finally stores desired results in
        dataframe to return back
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
            predicts the requested stock regarding trading type
            redirects the single predict to each respective model and collects the result
        """
        interval = TRADING_PARAMS.get(trading_type.value).get('interval')
        # the model input = the start of the dateTimes to predict is the last input of the model
        endDate = startDate
        startDate = startDate - pd.Timedelta(days=30)

        modelsToExecute = []
        timestamps_array = []
        if trading_type == resolution.MINUTE:
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'dayTrading']
            timestamps_array = [TimeModificationService.getSameDay1020am(endDate)]
        elif trading_type == resolution.TWO_HOURLY:
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'swingTrading']
            startDate_dateTime = TimeModificationService.getSameDay10am(endDate)
            timestamps_array = [startDate_dateTime + pd.Timedelta(hours=2 * i) for i in range(4)]
        elif trading_type == resolution.DAILY:
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'longTrading']
            startDate_dateTime = TimeModificationService.getSameDay8pm(endDate)
            timestamps_array = [startDate_dateTime]
            #when time ahead model 5 was executed
            #timestamps_array.extend([startDate_dateTime + pd.Timedelta(days=1) for i in range(3)])
            timestamps_array.extend([startDate_dateTime + pd.Timedelta(days=1) for i in range(2)])
            lastEle = timestamps_array[len(timestamps_array) - 1]
            timestamps_array.append(lastEle + pd.Timedelta(days=2))
        else:
            print("error")

        preprocessor = Preprocessor(self.config)
        datesTimes = []
        for stock_symbol in symbol_list:
            modelInputData, endRawPrice = preprocessor.pipeline(stock_symbol, startDate, endDate,
                                                                length=20, interval=interval)
            predictions = self.getAllPredictionsForSingleStock(modelsToExecute, stock_symbol,
                                                               modelInputData, endDate, endRawPrice,
                                                               interval)
            self.modelResults.append({stock_symbol: predictions})

        return self.createPredictionDataframe(self.modelResults, timestamps_array)

    @staticmethod
    def getAllPredictionsForSingleStock(modelsToExecute, stock_symbol, modelInputData, endDate, rawEndPrice, interval):
        """
            execute predictions iteratively for all horizons defined
            for a single Model for a specified stock_symbol
        """
        predictionList = []
        dateTimeList = []
        for modelEntry in modelsToExecute:
            if modelEntry.get('stockSymbol') == stock_symbol:
                # (1, 5, 20, 20) size
                # predict single model
                model = modelEntry.get('model').predict(modelInputData)
                modelResult = model.get('prediction')
                predictionEndPrice = differencingService.calcThePriceFromChange(True, modelResult.item(),
                                                                                rawEndPrice)
                predictionList.append(predictionEndPrice)

        return predictionList

    @staticmethod
    def createPredictionDataframe(resultMap, dateTimes) -> pd.DataFrame:
        """
            reformats collected predictions to the defined DataFrame type
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
        toReturn = pd.DataFrame(dateTimes, columns=['Timestamp'])
        for stock in resultMap:
            for key in stock:
                toReturn[key] = stock[key]

        toReturn.set_index('Timestamp', inplace=True)
        toReturn = toReturn.astype("Float64")
        return toReturn

    def loadModels(self):
        """
            load models out of prediction config and stores it in class variable
        """
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
