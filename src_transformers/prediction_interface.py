"""
This module contains the TransformerInterface class and the MultiSymbolDataset class.
"""
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

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

        prediction = pd.DataFrame(prediction, columns=columns)

        for symbol, price in self.prices_before_prediction.items():
            absolute_prices = self.calculate_absolut_prices(prediction[f"close {symbol}"], price)
            prediction[f"close {symbol}"] = round(absolute_prices, 2)

        return prediction

    def load_data(self, timestamp_start: pd.Timestamp) -> None:
        """load data from database and stores it in a class variable"""
        data_path = Path("data", "output", "Multi_Symbol_Predict3.csv")
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
        model_path = Path("data", "output", "models", "TransformerModel_v6.pt")
        self.model = torch.load(model_path)
        self.model.device = torch.device("cpu")

        self.model.to(torch.device("cpu"))


if __name__ == "__main__":
    pred = TransformerInterface().predict(pd.to_datetime('2021-01-30'), pd.to_datetime('2021-02-01'))
    print(pred)
