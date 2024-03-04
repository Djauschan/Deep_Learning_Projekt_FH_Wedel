import os

import pandas as pd

from CNN.prediction.Model import Model
from CNN.prediction.abstract_model import resolution
from CNN.prediction.services.DataLoaderService import DataLoaderService
from CNN.prediction.services.ModelImportService import ModelImportService
from CNN.prediction.services.preprocessingServices import Preprocessor

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
        self.modelCollection = []
        self.modelResult = []
        self.modelInputData = []
        self.loadModels()

    def collective_predict(self, symbol_list: list, trading_type: resolution, startDate: pd.Timestamp,
                           endDate: pd.Timestamp):
        """
            redirect predict request from api/backend to correct model
        """
        interval = TRADING_PARAMS.get(trading_type.value).get('interval')
        preprocessor = Preprocessor(self.config)
        for stock_symbol in symbol_list:
            modelInputData = preprocessor.pipeline(stock_symbol, startDate, endDate, length=20, interval=interval)
            self.modelInputData.append(modelInputData)

            for stock, horizon, model in self.modelCollection:
                if stock == stock_symbol:
                    self.modelResult.append({'horizon': horizon, 'result': model.predict(modelInputData)})

    def loadModels(self):
        for (k, v) in self.modelsToLoad:
            for (t, v1) in v:
                for (f, m) in v1:
                    for (stock_Symbol, model_path) in m:
                        model = Model(k, t, stock_Symbol, os.path.join(f, model_path))
                        self.modelCollection.append({'stock_Symbol': stock_Symbol, 'horizon': t, 'model': model})
