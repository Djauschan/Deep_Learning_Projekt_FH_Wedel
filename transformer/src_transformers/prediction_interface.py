"""
This module provides an interface for making predictions with a Transformer model.
It contains two classes: PredictionDataset and TransformerInterface.

The PredictionDataset class is a PyTorch Dataset for making predictions.

The TransformerInterface class provides methods for loading data, loading a model, making
predictions, and converting relative prices to absolute prices.
"""
import datetime as dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from src_transformers.abstract_model import AbstractModel
from src_transformers.abstract_model import resolution as resolution_enum
from src_transformers.pipelines.constants import MODEL_NAME_MAPPING
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

    def __init__(self, data: pd.DataFrame, first_index: int, encoder_intervals: int) -> None:
        """
        Initializes the PredictionDataset with the given data and index.

        Args:
            data (pd.DataFrame): The DataFrame containing the data for making predictions.
            first_index (int): The index of the first row of data not included in the sample.
        """
        self.data = data.iloc[first_index - encoder_intervals: first_index, :]
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

    def __init__(self, resolution: resolution_enum) -> None:
        """
        Initializes the TransformerInterface with default values.

        This method sets the interval between predictions, the number of intervals,
        and the paths to the model file, the data file, and the prices file.
        """
        self.resolution = resolution

        # Set paths to the directories for readability in the following if-else statements
        data_path = Path("data", "output")
        models_path = Path("data", "output", "models")
        configs_path = Path("data", "test_configs")

        # Set the correct model, data and config paths based on the requested resolution
        # The division sign is used to join paths in pathlib
        # self.num_intervals defines the number of intervals for the prediction
        if resolution == resolution.MINUTE:
            # Not implemented for minute resolution
            raise NotImplementedError()
        elif resolution == resolution.TWO_HOURLY:
            self.num_intervals = 24
            self.model_path = models_path / "TransformerModel_v2.pt"
            self.data_path = data_path / "120_min_input_data.csv"
            self.prices_path = data_path / "tt_prices_for_120_min.pkl"
            config_path = configs_path / "config_tt_hourly.yaml"
        elif resolution == resolution.DAILY:
            self.num_intervals = 30
            self.model_path = models_path / "TransformerModel_v5.pt"
            self.data_path = data_path / "1440_min_input_data.csv"
            self.prices_path = data_path / "tt_prices_for_1440_min.pkl"
            config_path = configs_path / "config_tt_daily.yaml"
        else:
            raise ValueError("Invalid resolution")

        # Load configuration file of the chosen model
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Get the encoder length as it is needed for the prediction dataset
        model_parameters = config.pop('model_parameters').popitem()[1]
        self.encoder_length = model_parameters['seq_len_encoder']

        # Get the time resolution and the start and end day from the dataset parameters
        # They are also needed for the timestamp generation
        dataset_parameters = config.pop('dataset_parameters')
        self.time_resolution = dataset_parameters['time_resolution']
        self.start_day = dataset_parameters["data_selection_config"]["start_day_time"]
        self.end_day = dataset_parameters["data_selection_config"]["end_day_time"]
        # The symbols are needed to map the predictions to the correct stock symbols
        self.symbols = dataset_parameters["decoder_symbols"]

    def predict(self, symbol_list: list, timestamp_start: pd.Timestamp) -> pd.DataFrame:
        """predicts the stock prices for the given symbols and time range.

        Args:
            symbol_list (list): The list of symbols for which the stock prices should be predicted.
            timestamp_start (pd.Timestamp): The start of the time range for which the stock prices should be predicted.
            timestamp_end (pd.Timestamp): The end of the time range for which the stock prices should be predicted.

        Returns:
            pd.DataFrame: The predicted stock prices.
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
        prediction = pd.DataFrame(output, columns=self.symbols)

        # Generate the timestamps for the predictions
        timestamps = self.generate_timestamps(timestamp_start,
                                              self.time_resolution,
                                              self.num_intervals)

        if self.resolution == resolution_enum.DAILY:
            # Set the timestamps as the index of the DataFrame and set the time to 20:00
            prediction.index = pd.Index(timestamps) + pd.Timedelta(hours=20)
        if self.resolution == resolution_enum.TWO_HOURLY:
            # When dealing with 2-hourly data, we need to add in the weeken days
            start_day = dt.datetime.strptime(
                self.start_day, '%H:%M').time().hour
            end_day = dt.datetime.strptime(self.end_day, '%H:%M').time().hour

            empty_df = pd.DataFrame([], index=pd.Index(
                timestamps), columns=prediction.columns)

            i = 0
            for timestamp in timestamps:
                # The aggregation returns the start of the interval as the timestamp.
                # Since we are predicting closing prices, we set the timestamp of the interval to the end of the interval.
                if timestamp.hour >= start_day + 2 and timestamp.hour < end_day + 2:
                    # Insert prediction where timestamp matches index
                    # Insert data from prediction at row i
                    empty_df.loc[timestamp, :] = prediction.iloc[i]
                    i += 1

            prediction = empty_df.fillna(0)

        # Convert the relative prices to absolute prices
        for symbol, price in prices_before_prediction.items():
            if symbol in self.symbols:
                relative_prices = list(prediction[f"{symbol}"])
                absolute_prices = self.calculate_absolute_prices(prices=relative_prices,
                                                                 start_price=price)
                prediction[f"{symbol}"] = np.round(
                    absolute_prices, decimals=2)

        # Only return the predictions for the requested stock symbols
        return prediction[symbol_list]

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
        dataset = PredictionDataset(data, first_index, self.encoder_length)

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

    def load_model(self, model_name: str = "torch_transformer") -> nn.Module:
        """
        Loads a PyTorch model from a file.

        This method loads a PyTorch model from the file specified by the `model_path` attribute.
        It then sets the model's device to CPU and switches the model to evaluation mode.

        Args:
            model_name (str, optional): The name of the model to be loaded. Defaults to "torch_transformer".

        Returns:
            nn.Module: The loaded PyTorch model.
        """

        state_dict, params = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        model = MODEL_NAME_MAPPING[model_name](**params)
        model.load_state_dict(state_dict)
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
        timestamps = [timestamp_start + pd.Timedelta(minutes=i * interval_minutes)
                      for i in range(num_intervals)]

        return timestamps
