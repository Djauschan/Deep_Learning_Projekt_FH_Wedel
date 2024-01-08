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

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int = 0) -> pd.DataFrame:
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

        self.load_data()
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

    def load_data(self) -> None:
        """load data from database and stores it in a class variable

        """
        data_path = Path("data", "output", "Multi_Symbol_Train.csv")

        with open(data_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            # Get the column names of the csv file (includes posix_time)
            encoder_dimensions = len(next(csv_reader)) - 1
            length = sum(1 for _ in csv_reader)

        decoder_dimensions = 1

        self.dataset = MultiSymbolDataset(length=length,
                                          encoder_dimensions=encoder_dimensions,
                                          decoder_dimensions=decoder_dimensions,
                                          encoder_input_length=30,
                                          decoder_target_length=30,
                                          data_file=data_path.as_posix(),
                                          time_resolution=30)

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable

        """
        pass

    def load_model(self) -> None:
        """load model from file and stores it in a class variable

        """
        model_path = Path("data", "output", "models", "TransformerModel_v4.pt")
        self.model = torch.load(model_path)


@dataclass
class MultiSymbolDataset(Dataset):
    """
    A PyTorch Dataset for multi-symbol financial data.

    This class handles multi-symbol financial data. It supports creating a new dataset
    from a configuration file and loading existing data from a csv file, if set to do so
    in the configuration file. The dataset is designed to deliver data for a Transformer.

    Attributes:
        length (int): The number of samples in the dataset.
        encoder_dimensions (int): The number of dimensions in the encoder input.
        decoder_dimensions (int): The number of dimensions in the decoder input.
        encoder_input_length (int): The length of the encoder input sequence.
        decoder_input_length (int): The length of the decoder input sequence.
        data_file (str): The path to the file to store the data in or to load the data from.
    """

    length: int
    encoder_dimensions: int
    decoder_dimensions: int
    encoder_input_length: int
    decoder_target_length: int
    data_file: str
    time_resolution: int

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        The number of samples is calculated as the total length of the data minus
        the length of the encoder and decoder input sequences plus one.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.length - self.encoder_input_length - self.decoder_target_length + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns time series sequences as input for the encoder and decoder from the dataset
        starting at the specified index.

        The encoder input sequence starts at the specified index and has a length of
        `self.encoder_input_length`. The decoder target sequence starts immediately after the
        input sequence and has a length of `self.decoder_target_length`.

        The method reads a chunk of data from the data file, starting at `encoder_input_start`
        and ending at `decoder_target_end`. It then extracts the encoder input and decoder target
        from this chunk and converts them to PyTorch tensors.

        Args:
            index (int): The index of the sample in the dataset.

        Returns:
            torch.Tensor: The encoder input sequence.
            torch.Tensor: The decoder target sequence.
        """
        # Calculate the start of the encoder input and the end of the decoder target
        # from the given index to load the correct data from the file
        encoder_input_start, encoder_input_end = index, index + self.encoder_input_length
        decoder_target_end = encoder_input_end + self.decoder_target_length

        num_rows_to_read = decoder_target_end - encoder_input_start

        # Load the data from the file, offset by the given index
        data = pd.read_csv(self.data_file, skiprows=encoder_input_start,
                           nrows=num_rows_to_read, index_col=0)

        encoder_input = data.to_numpy()

        # Get the encoder input from 0 to self.input_length
        encoder_input = encoder_input[0:self.encoder_input_length]
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)

        # Only keep the decoder symbols (which are at the end) for the decoder target
        decoder_target = data.iloc[:, -self.decoder_dimensions:]
        decoder_target = decoder_target.to_numpy()

        # Get the decoder target (starting from the end of encoder input)
        decoder_target = decoder_target[self.encoder_input_length:self.encoder_input_length
                                        + self.decoder_target_length]
        decoder_target = torch.tensor(decoder_target, dtype=torch.float32)

        return encoder_input, decoder_target


if __name__ == "__main__":
    pred = TransformerInterface().predict(pd.to_datetime('2021-01-04'), pd.to_datetime('2021-02-01'))
    print(pred)
