"""
This module provides an interface for making predictions with a Transformer model.
It contains two classes: PredictionDataset and TransformerInterface.

The PredictionDataset class is a PyTorch Dataset for making predictions.

The TransformerInterface class provides methods for loading data, loading a model, making
predictions, and converting relative prices to absolute prices.
"""
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from abstract_model import AbstractModel
from torch.utils.data import DataLoader, Dataset


class PredictionDataset(Dataset):
    """
    A PyTorch Dataset for making predictions.

    This class inherits from the PyTorch Dataset class and overrides its methods to provide data for
    making predictions. It takes a DataFrame and an index as input, and provides a single sample
    containing the 96 rows (matching our Transformer's encoder input length) of data preceding the
    given index.

    Attributes:
        data (pd.DataFrame): The DataFrame containing the data for making predictions.
    """

    def __init__(self, data: pd.DataFrame, first_index: int) -> None:
        """
        Initializes the PredictionDataset with the given data and index.

        Args:
            data (pd.DataFrame): The DataFrame containing the data for making predictions.
            first_index (int): The index of the first row of data not included in the sample.
        """
        self.data = data.iloc[first_index - 96: first_index, :]
        self.data.set_index("posix_time", inplace=True)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Since this dataset is designed to provide a single sample for making predictions,
        this method always returns 1.

        Returns:
            int: The length of the dataset.
        """
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Returns the sample.

        Since this dataset is designed to provide a single sample for making predictions, the index
        is ignored and this method always returns the same sample.

        Args:
            index (int): The index. Ignored.

        Returns:
            torch.Tensor: The sample.
        """
        return torch.tensor(self.data.to_numpy(), dtype=torch.float32)


class TransformerInterface(AbstractModel):
    """
    An interface for making predictions with a Transformer model.

    This class provides methods for loading data, loading a model, making predictions, and
    converting relative prices to absolute prices. It inherits from the AbstractModel class.

    Attributes:
        interval_minutes (int): The interval between predictions in minutes.
        num_intervals (int): The number of intervals.
        model_path (Path): The path to the model file.
        data_path (Path): The path to the data file.
        prices_path (Path): The path to the prices file.
    """

    def __init__(self) -> None:
        """
        Initializes the TransformerInterface with default values.

        This method sets the interval between predictions, the number of intervals,
        and the paths to the model file, the data file, and the prices file.
        """
        self.interval_minutes = 120
        self.num_intervals = 24
        self.model_path = Path(Path.cwd(), "transformer",
                               "prediction_resources", "torch_transformer.pt")
        self.data_path = Path(Path.cwd(), "transformer",
                              "prediction_resources", "preprocessed_data_prediction.csv")
        self.prices_path = Path(Path.cwd(), "transformer",
                                "prediction_resources", "prices_prediction.pkl")

    def predict(self,
                timestamp_start: pd.Timestamp,
                timestamp_end: pd.Timestamp,
                interval: int = 0) -> pd.DataFrame:
        """
        Makes predictions for the given time period.

        This method loads the data for making predictions, makes predictions using the model, and
        converts the predictions to absolute prices. It then returns a DataFrame with the predicted
        prices for each stock.

        Args:
            timestamp_start (pd.Timestamp): The start timestamp for the predictions.
            timestamp_end (pd.Timestamp): The end timestamp for the predictions.
            interval (int, optional): The interval between predictions in minutes. Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame with the predicted prices for each stock.
        """
        # Load the data for making predictions
        prices_before_prediction, dataset = self.load_data(timestamp_start)
        data_loader = DataLoader(dataset, shuffle=False)
        # Get the input for the model
        model_input = next(iter(data_loader))
        model = self.load_model()

        # Make predictions using the model
        with torch.no_grad():
            output = model(model_input)

        # Squeeze the batch dimension and convert the output to a 2 dimensional numpy array
        output = torch.squeeze(output, 0).cpu().numpy()
        # Create a DataFrame with the predictions (and mapping to the correct column names)
        columns = ["close AAPL", "close AAL", "close AMD", "close C", "close MRNA",
                   "close NIO", "close NVDA", "close SNAP", "close SQ", "close TSLA"]
        prediction = pd.DataFrame(output, columns=columns)

        # Generate the timestamps for the predictions
        timestamps = self.generate_timestamps(timestamp_start,
                                              self.interval_minutes,
                                              self.num_intervals)
        prediction.index = pd.Index(timestamps)

        # Convert the relative prices to absolute prices
        for symbol, price in prices_before_prediction.items():
            if f"close {symbol}" in columns:
                relative_prices = list(prediction[f"close {symbol}"])
                absolute_prices = self.calculate_absolute_prices(prices=relative_prices,
                                                                 start_price=price)
                prediction[f"close {symbol}"] = np.round(
                    absolute_prices, decimals=2)

        return prediction

    def load_data(self, timestamp_start: pd.Timestamp) -> tuple[dict[str, float], PredictionDataset]:
        """
        Loads the data for making predictions.

        This method loads the data from the csv file specified by the `data_path` attribute.
        It then creates a PredictionDataset containing the 96 rows of data preceding the given
        start timestamp. Furthermore, it calculates the prices of the stocks just before the
        start timestamp.

        Args:
            timestamp_start (pd.Timestamp): The start timestamp for the predictions.

        Returns:
            tuple: A tuple containing a dictionary with the prices of the stocks just before the
                   start timestamp, and a PredictionDataset with the data for making predictions.
        """
        data = pd.read_csv(self.data_path)

        # Find the index of the first timestamp on the given start date
        data_after_start = data[data["posix_time"]
                                >= timestamp_start.value / 1e9]
        first_index = data_after_start.index[0]
        dataset = PredictionDataset(data, first_index)

        # Load the prices of the stocks before the data
        prices_before_data = pickle.load(open(self.prices_path, "rb"))
        # Select the data before the start timestamp
        data_before_start = data.iloc[:first_index, :]

        # Calculate the prices of the stocks just before the start timestamp
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
        """
        Loads a PyTorch model from a file.

        This method loads a PyTorch model from the file specified by the `model_path` attribute.
        It then sets the model's device to CPU and switches the model to evaluation mode.

        Returns:
            nn.Module: The loaded PyTorch model.
        """
        model = torch.load(self.model_path)
        model.to(torch.device("cpu"))
        # Set model device attribute to CPU so that masks are on CPU as well
        model.device = torch.device("cpu")
        model.eval()

        return model

    def generate_timestamps(self,
                            timestamp_start: pd.Timestamp,
                            interval_minutes: int,
                            num_intervals: int) -> list[pd.Timestamp]:
        """
        Generate a list of timestamps starting at `timestamp_start` and ending at
        `timestamp_start + interval_minutes * (num_intervals - 1)`.

        Args:
            timestamp_start (pd.Timestamp): Datetime to start at
            interval_minutes (int): Duration of each interval in minutes
            num_intervals (int): Number of intervals to generate

        Returns:
            list[pd.Timestamp]: List of timestamps.
        """
        timestamps = [timestamp_start + timedelta(minutes=i * interval_minutes)
                      for i in range(num_intervals)]

        return timestamps
