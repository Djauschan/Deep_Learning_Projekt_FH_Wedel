from CNN.prediction.services.ModelImportService import ModelImportService


class ModelWrapper:
    """
        class that loads all modells and distributes requests to the correct modell
    """

    def __init__(self, config):
        self.parameters = config
        self.modelCollection = []

    def predict(self, stockName: str, tradingType: str):
        """
            redirect predict request from api/backend to correct model
        """
        pass

    def loadModells(self):
        modelImportService = ModelImportService(self.parameters)
        listOfModelsToLoad = modelImportService.getSavedModelsPaths()
        for model_path in listOfModelsToLoad:
            self.modelCollection.append(
                modelImportService.loadModel(model_path))
