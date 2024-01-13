"""
This module contains the TransformerInterface class and the MultiSymbolDataset class.
"""
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src_transformers.abstract_model import AbstractModel


class PredictionDataset(Dataset):

    def __init__(self, data: pd.DataFrame, first_index: int) -> None:
        self.data = data.iloc[first_index - 96 : first_index, :]
        self.data.set_index("posix_time", inplace=True)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.data.to_numpy(), dtype=torch.float32)


class TransformerInterface(AbstractModel):

    def __init__(self) -> None:
        self.interval_minutes = 120
        self.num_intervals = 24
        self.model_path = Path("data", "output", "models", "TransformerModel_v7.pt")
        self.data_path = Path("data", "output", "Multi_Symbol_Train.csv")
        self.prices_path = Path("data", "output", "prices.pkl")

    def predict(
        self,
        timestamp_start: pd.Timestamp,
        timestamp_end: pd.Timestamp,
        interval: int = 0,
    ) -> pd.DataFrame:

        prices_before_prediction, dataset = self.load_data(timestamp_start)
        data_loader = DataLoader(dataset, shuffle=False)

        model = self.load_model()
        model_input = next(iter(data_loader))

        model.eval()
        with torch.no_grad():
            output = model(model_input)

        output = torch.squeeze(output, 0).cpu().numpy()
        columns = ["close AAPL", "close AAL", "close AMD", "close C", "close MRNA",
                   "close NIO", "close NVDA", "close SNAP", "close SQ", "close TSLA"]
        prediction = pd.DataFrame(output, columns=columns)

        timestamps = self.generate_timestamps(timestamp_start,
                                              self.interval_minutes,
                                              self.num_intervals)
        prediction.set_index(timestamps, inplace=True)
        # Fallback if it does not work
        # prediction.index = timestamps

        for symbol, price in prices_before_prediction.items():
            if f"close {symbol}" in columns:
                relative_prices = list(prediction[f"close {symbol}"])
                absolute_prices = self.calculate_absolute_prices(prices=relative_prices,
                                                                 start_price=price)
                prediction[f"close {symbol}"] = np.round(absolute_prices, decimals=2)

        return prediction

    def load_data(self, timestamp_start: pd.Timestamp) -> tuple[dict[str, float], PredictionDataset]:
        """load data from database and stores it in a class variable"""
        data = pd.read_csv(self.data_path)

        data_after_start = data[data["posix_time"] > timestamp_start.value / 1e9]
        first_index = data_after_start.index[0]
        dataset = PredictionDataset(data, first_index)

        prices_before_data = pickle.load(open(self.prices_path, "rb"))
        data_before_start = data.iloc[:first_index, :]

        prices_before_prediction = {}
        for symbol, price in prices_before_data.items():
            relative_prices = list(data_before_start[f"close {symbol}"])
            absolute_prices = self.calculate_absolute_prices(prices=relative_prices,
                                                             start_price=price)
            prices_before_prediction[symbol] = round(absolute_prices[-1], 2)

        return prices_before_prediction, dataset

    def preprocess(self) -> None:
        """Not implemented in this interface as stored data is already preprocessed."""

    def load_model(self) -> nn.Module:
        """load model from file and stores it in a class variable"""
        model = torch.load(self.model_path)
        model.device = torch.device("cpu")
        model.to(torch.device("cpu"))
        model.eval()

        return model

    def generate_timestamps(self, start_timestamp: pd.Timestamp, interval_minutes: int, num_intervals: int) -> list[pd.Timestamp]:
        """
        Generate a list of timestamps starting at `start_timestamp` and ending at
        `start_timestamp + interval_minutes * (num_intervals - 1)`.

        Args:
            start_timestamp (pd.Timestamp): Datetime to start at
            interval_minutes (int): Duration of each interval in minutes
            num_intervals (int): Number of intervals to generate

        Returns:
            List[pd.Timestamp]: List of timestamps
        """
        timestamps = [start_timestamp + timedelta(minutes=i * interval_minutes)
                      for i in range(num_intervals)]

        return timestamps


if __name__ == "__main__":
    pred = TransformerInterface().predict(pd.to_datetime('2021-01-04'),
                                          pd.to_datetime('2021-01-06'))
    print(pred)
