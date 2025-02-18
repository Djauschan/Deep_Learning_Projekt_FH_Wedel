from typing import Iterable

from torch.jit import ScriptModule

from src.prediction.services.ModelImportService import ModelImportService


class Model:
    """
        loads and stores a single pytorch model from jit
        executes predict method and return structured object with prediction result back
    """

    def __init__(self, stock_symbol: str, horizon: int, completePath):
        self.modelImportService = ModelImportService(completePath)
        self.model: ScriptModule = self.modelImportService.loadModel()
        self.stock_symbol = stock_symbol
        self.model_horizon = horizon
        self.model_length = 0
        self.model_interval = 0

    def predict(self, modelInput: Iterable):
        prediction = self.model.forward(modelInput)
        toReturn = {'stock_symbol': self.stock_symbol, 'horizon': self.model_horizon, 'prediction': prediction}
        return toReturn
