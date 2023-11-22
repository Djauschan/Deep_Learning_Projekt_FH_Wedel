import torch
import sys
import yaml
from torch.utils.data import Dataset
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.preprocessing.dataProcessing import lookup_symbol, add_time_information, create_one_hot_vector
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class PerSymbolDataset(Dataset):
    """
    Data stored as tensors
    Pytorch uses the 3 functions [__init__, __len__, __getitem__]
    """

    def __init__(self, data_frame: pd.DataFrame, symbols: list, config: dict):
        """
        Initializes the Pytorch data set.

        Args:
            data_frame (pd.DataFrame): Data frame that contains the data for the dataset.
            symbols (list): List that contains all symbols.
            config (dict): Dictionary for the configuration of data preprocessing.
        """
        # Dictionary for the configuration of data preprocessing is saved.
        self.config = config

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
        numeric_values = data_frame.iloc[:, 1:8].to_numpy()
        #TODO Dimensionen fixen (waren vorher 9) -> müssen gerade sein

        # Represent symbol as one-hot vector
        one_hot_vec = create_one_hot_vector(symbols, self.symbol)

        # A one-hot vector is added to each entry in the time series,
        # indicating the ticker symbol to which the transaction belongs.
        numeric_values = np.concatenate((numeric_values, np.tile(
            one_hot_vec, (numeric_values.shape[0], 1))), axis=1)

        # The data of the current transaction is used to predict the data of the next transaction.
        output = numeric_values[:-1]
        input = numeric_values[1:]
        #TODO langfristige Lösung finden

        # For the use of pytorch, the data is converted into tensors.
        self.input_data = torch.tensor(input, dtype=torch.float32)
        self.output_data = torch.tensor(output, dtype=torch.float32)

        # The dimensions of the input and output data are required
        # to dimension the input and output layers of the model.
        self.input_dim = self.input_data.shape[1]
        self.output_dim = self.output_data.shape[1]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns:
            int: number of samples in the dataset
        """
        return len(self.input_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampel (X, Y) at position idx

        Args:
            idx (int): position of sample in the tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sampel (X, Y) at position idx
        """
        return self.input_data[idx], self.output_data[idx]


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
    dataset = PerSymbolDataset(data, txt_reader.symbols, config)
    # Create data loader
    dataloader = DataLoader(
        dataset, batch_size=dataset.config["BATCH_SIZE"], shuffle=False)
    # Print the first sample.
    test_sample = next(iter(dataloader))[0]
    print("INPUT:")
    print(test_sample[0])
    print("OUTPUT:")
    print(test_sample[1])
