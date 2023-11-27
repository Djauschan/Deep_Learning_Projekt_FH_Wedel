import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src_transformers.preprocessing.dataProcessing import (
    add_time_information,
    create_one_hot_vector,
    lookup_symbol,
)
from src_transformers.preprocessing.txtReader import DataReader


class PerSymbolDataset(Dataset):
    """
    Data stored as tensors
    Pytorch uses the 3 functions [__init__, __len__, __getitem__]
    """

    def __init__(self, data_frame: pd.DataFrame, symbols: list, config: dict, input_length: int, target_length: int):
        """
        Initializes the Pytorch data set.

        Args:
            data_frame (pd.DataFrame): Data frame that contains the data for the dataset.
            symbols (list): List that contains all symbols.
            config (dict): Dictionary for the configuration of data preprocessing.
            input_length (int): Length of the input sequence.
            target_length (int): Length of the target sequence.
        """
        # Dictionary for the configuration of data preprocessing is saved.
        self.config = config
        self.input_length = input_length
        self.target_length = target_length

        # The type of the ticker symbol is read out.
        self.type = data_frame["type"][1]
        # Then it is removed from the data frame, since it is the same for all lines.
        data_frame = data_frame.drop("type", axis=1)

        # The ticker symbol is read out.
        self.symbol = data_frame["symbol"][1]
        # The name of the ticker symbol is then looked up.
        self.name = lookup_symbol(self.symbol, self.type)
        # Then it is removed from the data frame, since it is the same for all lines.
        data_frame = data_frame.drop("symbol", axis=1)

        data_frame = add_time_information(data_frame)

        if self.config["DEBUG_OUTPUT"]:
            print(data_frame)

        # Only the numerical data is used for machine learning.
        # columns from position 1 (inclusive) to position 9 (exclusive)
        numeric_values = data_frame.iloc[:, 1:9].to_numpy()

        # Represent symbol as one-hot vector
        one_hot_vec = create_one_hot_vector(symbols, self.symbol)

        # A one-hot vector is added to each entry in the time series,
        # indicating the ticker symbol to which the transaction belongs.
        numeric_values = np.concatenate((numeric_values, np.tile(
            one_hot_vec, (numeric_values.shape[0], 1))), axis=1)

        # If the number of columns is odd, a column with zeros is added.
        # This is necessary because the number of columns must be even for the transformer model.
        if numeric_values.shape[1] % 2 != 0:
            zero_column = np.zeros((numeric_values.shape[0], 1))
            numeric_values = np.concatenate(
                (numeric_values, zero_column), axis=1)

        # The data of the current transaction is used to predict the data of the next transaction.
        input = numeric_values
        output = numeric_values

        # For the use of pytorch, the data is converted into tensors.
        self.input_data = torch.tensor(input, dtype=torch.float32)
        self.output_data = torch.tensor(output, dtype=torch.float32)

        # The dimensions of the input and output data are required
        # to dimension the input and output layers of the model.
        self.input_dim = self.input_data.shape[1]
        self.output_dim = self.output_data.shape[1]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        This is the length of the Input data minus the length of
        the Input for one sample minus the length of the target for one sample plus one.

        Returns:
            int: number of samples in the dataset
        """
        return len(self.input_data) - self.input_length - self.target_length + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampel of input and target sequences. The input and target sequences are of the
        length defined in the config. The start of the target sequence is the first entry after
        the input sequence.

        Args:
            idx (int): position of sample in the tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sampel (X, Y) at position idx
        """
        # Get the input data of length INPUT_LEN
        start_input = idx
        end_input = idx + self.input_length
        input = self.input_data[start_input:end_input]

        # Get the output target data of length TARGET_LEN after the input period
        start_target = end_input
        end_target = start_target + self.target_length
        target = self.output_data[start_target:end_target]

        return input, target


# Code for debugging
if __name__ == "__main__":

    # Check if the path to the configuration file was passed as an argument.
    assert len(sys.argv) == 2

    # Read in the configuration file.
    config_file_path = sys.argv[1]
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Create dataset
    txt_reader = DataReader(config)
    data = txt_reader.read_next_txt()
    dataset = PerSymbolDataset(data, txt_reader.symbols, config, 200, 3)
    # Create data loader
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    # Print the first sample.
    test_sample = next(iter(dataloader))[1]
    print("TEST SAMPLE:")
    print(test_sample)
