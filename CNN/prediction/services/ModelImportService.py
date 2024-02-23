import os
import torch


class ModelImportService:
    """
        service that loads pytorch models from given config
    """

    def __init__(self, modelParameters):
        self.modelParameters = modelParameters

    def getSavedModelsPaths(self) -> list:
        MODEL_FOLDER = self.modelParameters["MODEL_FOLDER"]
        MODELS_TO_LOAD = self.modelParameters["MODELS_TO_LOAD"]
        MODELS_PATH_LIST = []
        for i in MODELS_TO_LOAD:
            MODELS_PATH_LIST.append(os.path.join(MODEL_FOLDER, i))

        return MODELS_PATH_LIST

    def loadModel(self, full_path):
        """
            load jit torch model from given path
        """
        device = torch.device("cpu")
        model = torch.jit.load(full_path, map_location=device)
        model.eval()
        return model
