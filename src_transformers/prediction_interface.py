"""
This module contains the TransformerInterface class and the MultiSymbolDataset class.
"""
import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src_transformers.abstract_model import AbstractModel

INTERVAL_MINUTES = 120
NUM_INTERVALS = 24


class TransformerInterface(AbstractModel):

    def predict(
        self,
        timestamp_start: pd.Timestamp,
        timestamp_end: pd.Timestamp,
        interval: int = 0,
    ) -> pd.DataFrame:
        # TODO: Pass first and last date to load_data
        # first_date = timestamp_start.strftime("%Y-%m-%d")
        # last_date = timestamp_end.strftime("%Y-%m-%d")
        # TODO: Use interval?
        timestamps = _generate_timestamps(timestamp_start, INTERVAL_MINUTES, NUM_INTERVALS)

        self.load_data(timestamp_start)
        # self.preprocess()
        self.load_model()
        self.model.eval()

        data_loader = DataLoader(self.dataset, shuffle=False)
        model_input = next(iter(data_loader))

        with torch.no_grad():
            output = self.model(model_input)

        output = torch.squeeze(output, 0)
        prediction = output.cpu().numpy()

        # TODO: Calculate absolute prices (start_price from where?)
        # self.calculate_absolut_prices(prediction)

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
        data_path = Path("data", "output", "Multi_Symbol_Train.csv")
        data = pd.read_csv(data_path.as_posix())

        print(data)

        prices_before_data = pickle.load(open("data/output/prices.pkl", "rb"))
        print(prices_before_data)

        # self.dataset = PredictionDataset.create_from_config(timestamp_start, data, prices_before_data)

        data_after_start = data[data["posix_time"] > timestamp_start.value / 1e9].copy()
        first_index = data_after_start.index[0]

        data_before_start = data.iloc[:first_index, :].copy()

        prices_before_prediction = {}
        for symbol, price in prices_before_data.items():
            absolut_prices = self.calculate_absolut_prices(data_before_start[f"close {symbol}"], price)
            prices_before_prediction[symbol] = round(absolut_prices.values[-1], 2)

        print(prices_before_prediction)
        self.prices_before_prediction = prices_before_prediction
        self.dataset = PredictionDataset.create_from_config(data, first_index)

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable"""

    def load_model(self) -> None:
        """load model from file and stores it in a class variable"""
        model_path = Path("data", "output", "models", "TransformerModel_v7.pt")
        self.model = torch.load(model_path)
        self.model.device = torch.device("cpu")

        self.model.to(torch.device("cpu"))


@dataclass
class PredictionDataset(Dataset):

    data: pd.DataFrame

    @classmethod
    def create_from_config(
        cls: type["PredictionDataset"],
        data: pd.DataFrame,
        first_index: int
    ) -> "PredictionDataset":

        # print(data)

        # Load the data from the csv (timestamp start - encoder length)
        dataset_data = data.iloc[first_index - 96 : first_index, :]
        dataset_data = dataset_data.set_index("posix_time")

        # print(dataset_data)

        # Store the start prices in a class variable

        return cls(
            data=dataset_data
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.data.to_numpy()
        data = torch.tensor(data, dtype=torch.float32)

        return data


def _generate_timestamps(start_timestamp: pd.Timestamp, interval_minutes: int, num_intervals: int) -> List[pd.Timestamp]:
    """
    Generate a list of timestamps starting at start_timestamp and ending at start_timestamp + interval_minutes * (num_intervals - 1)

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
    pred = TransformerInterface().predict(pd.to_datetime('2021-01-04'), pd.to_datetime('2021-01-06'))
    print(pred)
