"""
This module contains the MultiSymbolDataset class which is used to handle multi-symbol
financial data.
"""
import pickle
from dataclasses import dataclass
from datetime import date

import pandas as pd
import torch
from torch.utils.data import Dataset

from src_transformers.preprocessing.csv_io import get_csv_shape, read_csv_chunk
from src_transformers.preprocessing.data_processing import (
    add_time_information,
    fill_dataframe,
    get_all_dates,
)
from src_transformers.preprocessing.preprocessing_constants import SCALER_OPTIONS
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.logger import Logger


@dataclass
class PredictionDataset(Dataset):
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

    data: pd.DataFrame
    encoder_dimensions: int
    decoder_dimensions: int
    encoder_input_length: int
    decoder_target_length: int

    @classmethod
    def create_from_config(
        cls: type["PredictionDataset"],
        timestamp_start: pd.DataFrame,
        data: pd.DataFrame,
    ) -> "PredictionDataset":
        print(timestamp_start)
        print(timestamp_start.date())
        print(timestamp_start.value)
        print(data.iloc[0, 0])

        print(data[data["posix_time"] > timestamp_start.value / 1e9])

        # Load the data from the csv (timestamp start - encoder length)

        # Store the start prices in a class variable

        return cls(
            data=data,
            encoder_dimensions=10,
            decoder_dimensions=1,
            encoder_input_length=96,
            decoder_target_length=24,
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        The number of samples is calculated as the total length of the data minus
        the length of the encoder and decoder input sequences plus one.

        Returns:
            int: The number of samples in the dataset.
        """
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
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
        data = self.data.to_numpy()
        data = torch.tensor(data, dtype=torch.float32)

        return data
