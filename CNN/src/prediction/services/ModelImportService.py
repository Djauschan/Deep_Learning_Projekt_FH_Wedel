import os
import torch
from torch.jit import ScriptModule


class ModelImportService:
    """
        service that loads pytorch models from given config
    """

    def __init__(self, path: str):
        self.model_path = path

    def loadModel(self) -> ScriptModule:
        """
            load jit torch model from given path
        """
        device = torch.device("cpu")
        model = torch.jit.load(self.model_path, map_location=device)
        model.eval()
        return model
