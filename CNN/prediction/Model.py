from typing import Iterable

from torch.jit import ScriptModule

from CNN.prediction.services.ModelImportService import ModelImportService


class Model:

    def __init__(self, type: str, horizon: int, stock_symbol: str, completePath):
        self.modelImportService = ModelImportService(completePath)
        self.model: ScriptModule = self.modelImportService.loadModel()
        self.trading_type = type
        self.stock_symbol = stock_symbol
        self.model_horizon = horizon
        self.model_length = 0
        self.model_interval = 0

    def predict(self, modelInput: Iterable):
        prediction = self.model.forward(modelInput)
        toReturn = {'stock_symbol': self.stock_symbol, 'timestamp': self.model_horizon, 'prediction': prediction}
        return toReturn
