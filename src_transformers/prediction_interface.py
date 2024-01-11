"""
This module contains the TransformerInterface class and the MultiSymbolDataset class.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src_transformers.abstract_model import AbstractModel
from src_transformers.preprocessing.prediction_dataset import PredictionDataset


class TransformerInterface(AbstractModel):
    """
    A class used to make predictions using a Transformer model.

    This class inherits from the AbstractModel class and overrides its methods to implement
    functionality specific to Transformer models. It includes methods for loading data,
    preprocessing, data, loading a model, and making predictions.

    The predict method takes a start and end timestamp and an optional interval, and returns a
    DataFrame with predicted stock prices for the given time period.

    Attributes:
        model (nn.Module): The PyTorch model to use for predictions.
        dataset (MultiSymbolDataset): The dataset to use for predictions.
    """

    def predict(
        self,
        timestamp_start: pd.Timestamp,
        timestamp_end: pd.Timestamp,
        interval: int = 0,
    ) -> pd.DataFrame:
        """predict stock price for a given time interval

        Args:
            timestamp_start (pd.Timestamp): start time of the time period
            timestamp_end (pd.Timestamp): end time of the time period
            interval (int, optional): interval in minutes. Defaults to 0.
            file_path (str, optional): path to the config file. Defaults to DEFAULT_PATH.

        Returns:
            pd.DataFrame: dataframe with columns: timestamp, 1-n prices of stock_symbols
        """
        # TODO: Pass first and last date to load_data
        # first_date = timestamp_start.strftime("%Y-%m-%d")
        # last_date = timestamp_end.strftime("%Y-%m-%d")
        # TODO: Use interval?

        self.load_data(timestamp_start)
        # self.preprocess()
        self.load_model()

        predictions = []
        data_loader = DataLoader(self.dataset, shuffle=False)
        self.model.eval()

        with torch.no_grad():
            for encoder_input, _ in data_loader:
                encoder_input = encoder_input.to(torch.device("cuda"))
                output = self.model(encoder_input)
                # Squeeze the batch dimension
                predictions.append(torch.squeeze(output, 0))

        prediction = torch.cat(predictions, dim=0).cpu()
        # Set the column names to the symbol names and the index to the timestamps
        prediction = pd.DataFrame(prediction.numpy())

        # TODO: Calculate absolute prices (start_price from where?)
        # self.calculate_absolut_prices(prediction)

        return prediction

    def load_data(self, timestamp_start: pd.Timestamp) -> None:
        """load data from database and stores it in a class variable"""
        data_path = Path("data", "output", "Multi_Symbol.csv")
        data = pd.read_csv(data_path.as_posix())

        print(data.info())

        PredictionDataset.create_from_config(timestamp_start, data)

        with open(data_path, "r", newline="", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            # Get the column names of the csv file (includes posix_time)
            encoder_dimensions = len(next(csv_reader)) - 1
            length = sum(1 for _ in csv_reader)

        decoder_dimensions = 1

        self.dataset = MultiSymbolDataset(
            length=length,
            encoder_dimensions=encoder_dimensions,
            decoder_dimensions=decoder_dimensions,
            encoder_input_length=30,
            decoder_target_length=30,
            data_file=data_path.as_posix(),
            time_resolution=30,
        )

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable"""
        pass

    def load_model(self) -> None:
        """load model from file and stores it in a class variable"""
        model_path = Path("data", "output", "models", "TransformerModel_v4.pt")
        self.model = torch.load(model_path)
