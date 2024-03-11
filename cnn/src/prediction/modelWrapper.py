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
        self.modelResults = {}
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
        AVAIL_MODELS = 0
        EXECUTION_ITERATION = 0
        modelsToExecute = []
        timestamps_array = []
        if trading_type == resolution.MINUTE:
            """
                Bspw: 04.01
                startDate = 10:00am
                1.) EndDate = 10:05
                2.) EndDate = 10:10
                3.) EndDate = 10:15
                4.) EndDate = 10:20
            """
            EXECUTION_ITERATION = 5
            AVAIL_MODELS = 1
            StartDate = TimeModificationService.getSameDay10am(endDate)
            # list of endates to recursively call the models
            timestamps_array = [StartDate + pd.Timedelta(minutes=5 * i) for i in range(EXECUTION_ITERATION)]
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'dayTrading']
        elif trading_type == resolution.TWO_HOURLY:
            EXECUTION_ITERATION = 2
            AVAIL_MODELS = 4
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'swingTrading']
            startDate_dateTime = TimeModificationService.getSameDay10am(endDate)
            i = 1
            timestamps_array = [startDate_dateTime]
            while i < (AVAIL_MODELS * EXECUTION_ITERATION):
                if i == 6:
                    timestamps_array.append(TimeModificationService.getNextDay10am(timestamps_array[i - 1]))
                else:
                    timestamps_array.append(timestamps_array[i - 1] + pd.Timedelta(hours=2))
                i = i + 1
        elif trading_type == resolution.DAILY:
            EXECUTION_ITERATION = 4
            AVAIL_MODELS = 5
            modelsToExecute = [d for d in self.modelCollection if d.get('tradingType') == 'longTrading']
            startDate_dateTime = TimeModificationService.getSameDay8pm(endDate)
            timestamps_array = [startDate_dateTime + pd.Timedelta(days=1 * i) for i in range(AVAIL_MODELS *
                                                                                             EXECUTION_ITERATION)]
        else:
            print("error")

        preprocessor = Preprocessor(self.config)
        # times how often the models are called and how often the data has to be prepared
        idx = 0
        predictionMap = {}
        while idx < EXECUTION_ITERATION:
            for stock_symbol in symbol_list:
                _endDate = pd.Timestamp(timestamps_array[idx])
                modelInputData, endRawPrice = preprocessor.pipeline(stock_symbol, startDate, _endDate,
                                                                    length=20, interval=interval)
                predictions = self.getAllPredictionsForSingleStock(modelsToExecute, stock_symbol,
                                                                   modelInputData, endRawPrice)
                if stock_symbol in predictionMap:
                    predictionMap[stock_symbol].extend(predictions)
                else:
                    predictionMap[stock_symbol] = predictions

            idx = idx + 1

        return self.createPredictionDataframe(predictionMap, timestamps_array)

    @staticmethod
    def getAllPredictionsForSingleStock(modelsToExecute, stock_symbol, modelInputData, rawEndPrice):
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
        for (stockSymbol, predictions) in resultMap.items():
            toReturn[stockSymbol] = predictions

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
