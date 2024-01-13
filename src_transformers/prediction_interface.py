"""
This module contains the TransformerInterface class and the MultiSymbolDataset class.
"""
import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src_transformers.abstract_model import AbstractModel


class TransformerInterface(AbstractModel):

    def __init__(self) -> None:
        self.interval_minutes = 120
        self.num_intervals = 24

        self.model_path = Path("data", "output", "models", "TransformerModel_v7.pt")
        self.data_path = Path("data", "output", "Multi_Symbol_Train.csv")
        self.prices_path = Path("data", "output", "prices.pkl")

        self.dataset = None
        self.model = None
        self.prices_before_prediction = None

    def predict(
        self,
        timestamp_start: pd.Timestamp,
        timestamp_end: pd.Timestamp,
        interval: int = 0,
    ) -> pd.DataFrame:
        timestamps = _generate_timestamps(timestamp_start, self.interval_minutes, self.num_intervals)

        self.load_data(timestamp_start)
        self.load_model()
        self.model.eval()

        data_loader = DataLoader(self.dataset, shuffle=False)
        model_input = next(iter(data_loader))

        with torch.no_grad():
            output = self.model(model_input)

        output = torch.squeeze(output, 0)
        prediction = output.cpu().numpy()

        columns = []
        for dataset_column in self.dataset.data.columns:
            if "close" in dataset_column:
                columns.append(dataset_column)

        columns = ["close AAPL", "close AAL", "close AMD", "close C", "close MRNA", "close NIO", "close NVDA", "close SNAP", "close SQ", "close TSLA"]

        prediction = pd.DataFrame(prediction, columns=columns)
        # Set timestamps as index
        prediction.index = timestamps

        for symbol, price in self.prices_before_prediction.items():
            if symbol in f"close {symbol}" in columns:
                absolute_prices = self.calculate_absolut_prices(prediction[f"close {symbol}"], price)
                prediction[f"close {symbol}"] = round(absolute_prices, 2)

        return prediction

    def load_data(self, timestamp_start: pd.Timestamp) -> None:
        """load data from database and stores it in a class variable"""
        data = pd.read_csv(self.data_path.as_posix())

        prices_before_data = pickle.load(open(self.prices_path.as_posix(), "rb"))

        data_after_start = data[data["posix_time"] > timestamp_start.value / 1e9].copy()
        first_index = data_after_start.index[0]

        data_before_start = data.iloc[:first_index, :].copy()

        prices_before_prediction = {}
        for symbol, price in prices_before_data.items():
            absolut_prices = self.calculate_absolut_prices(data_before_start[f"close {symbol}"], price)
            prices_before_prediction[symbol] = round(absolut_prices.values[-1], 2)

        self.prices_before_prediction = prices_before_prediction
        self.dataset = PredictionDataset.create_from_config(data, first_index)

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable

        """

    def load_model(self) -> None:
        """load model from file and stores it in a class variable"""
        self.model = torch.load(self.model_path)
        self.model.device = torch.device("cpu")

        self.model.to(torch.device("cpu"))


class PredictionDataset(Dataset):

    def __init__(self, data: pd.DataFrame, first_index: int) -> None:
        self.data = data.iloc[first_index - 96 : first_index, :]
        self.data.set_index("posix_time", inplace=True)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.data.to_numpy(), dtype=torch.float32)


def _generate_timestamps(start_timestamp: pd.Timestamp, interval_minutes: int, num_intervals: int) -> list[pd.Timestamp]:
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
    timestamps = [start_timestamp + timedelta(minutes=i * interval_minutes) for i in range(num_intervals)]
    return timestamps


if __name__ == "__main__":
    pred = TransformerInterface().predict(pd.to_datetime('2021-01-04'),
                                          pd.to_datetime('2021-01-06'))
    print(pred)
