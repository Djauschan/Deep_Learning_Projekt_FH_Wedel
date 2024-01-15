"""
This module contains the Predictor class which is used to predict using a PyTorch model.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset


@dataclass
class Predictor():
    """
    A class used to make predictions using a PyTorch model.

    Attributes:
        model (nn.Module): The PyTorch model to use for predictions.
        device (torch.device): The device (CPU or GPU) where the model and data should be loaded.
        dataset (MultiSymbolDataset): The dataset to use for predictions.
    """
    model: nn.Module
    device: torch.device
    dataset: MultiSymbolDataset

    def predict(self) -> torch.Tensor:
        """
        Makes predictions on the dataset using the model.

        This method iterates over the dataset, makes a prediction each iteration,
        and concatenates the predictions into a single tensor.

        Returns:
            torch.Tensor: A tensor containing the predictions for the entire dataset.
        """
        predictions = []
        data_loader = DataLoader(self.dataset, shuffle=False)
        self.model.eval()

        with torch.no_grad():
            for encoder_input, _ in data_loader:
                encoder_input = encoder_input.to(self.device)
                output = self.model(encoder_input)
                # Squeeze the batch dimension
                predictions.append(torch.squeeze(output, 0))

        return torch.cat(predictions, dim=0)
