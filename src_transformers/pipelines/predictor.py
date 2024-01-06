from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src_transformers.models.loss import RMSELoss, RMSLELoss
from src_transformers.pipelines.model_service import ModelService
from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset
from src_transformers.utils.logger import Logger
from src_transformers.utils.viz_training import plot_evaluation

"""
This module contains the Predictor class which is used to train a PyTorch model.
"""


@dataclass
class Predictor():

    model: nn.Module
    device: torch.device
    dataset: MultiSymbolDataset

    @classmethod
    def create_predictor_from_config(cls,
                                     model: nn.Module,
                                     device: torch.device,
                                     dataset: MultiSymbolDataset) -> "Predictor":

        return cls(model=model,
                   device=device,
                   dataset=dataset)

    def predict(self):
        datalaoder = DataLoader(self.dataset, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for input, target in datalaoder:
                input = input.to(self.device)
                output = self.model(input)
                print(output)
