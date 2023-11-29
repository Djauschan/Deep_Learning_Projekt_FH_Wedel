import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src_transformers.preprocessing.csv_io import (
    count_rows,
    get_column_count,
    read_csv_chunk,
)
from src_transformers.preprocessing.dataProcessing import (
    add_time_information,
    fill_dataframe,
    get_all_dates,
)
from src_transformers.preprocessing.txtReader import DataReader


class MultiSymbolDataset(Dataset):
    """
    Data stored as tensors
    Pytorch uses the 3 functions [__init__, __len__, __getitem__]
    """

    def __init__(self, reader: DataReader, config: dict, input_length: int, target_length: int):
        """
        Initializes the Pytorch data set.

        Args:
            reader (DataReader): Data Reader to read the files specified in the configuration file.
            config (dict): Dictionary for the configuration of data preprocessing.
            input_length (int): Length of the input sequence.
            target_length (int): Length of the target sequence.
        """
        self.config = config
        # A new file with input data is created.
        if self.config['CREATE_NEW_FILE']:
            print("[MULTI_SYMBOL_DATASET]: Data Preprocessing started")
            date_df = get_all_dates(reader)
            stocks, date_df = fill_dataframe(date_df, reader)

            # Sort the stock symbols alphabetically to ensure that the order is always the same.
            stocks.sort()
            self.stocks = stocks

            date_df = add_time_information(date_df)
            date_df.set_index('posix_time', inplace=True)
            date_df.drop(columns=['timestamp'], inplace=True)
            # Add a column with zeros for each stock symbol.
            # date_df.loc[:, stocks] = 0

            # If the number of columns is odd, a column with zeros is added.
            # This is necessary because the number of columns must be even for the transformer model.
            # Check if the number of columns is odd ( + 1 to account for the index column)
            if (len(date_df.columns)) % 2 != 0:
                # Add a new column 'even' with zeros
                date_df['even'] = 0

            self.length = len(date_df)  # * len(stocks)

            current_columns = list(date_df.columns.values)
            target_columns = []
            targets = 0

            # Tread edge case: File is not available
            for symbol in self.config["target_symbols"]:
                current_columns.remove(f'close {symbol}')
                target_columns.append(f'close {symbol}')
                targets += 1

            date_df = date_df[current_columns + target_columns]

            self.input_dim = date_df.shape[1]
            self.output_dim = targets

            date_df.to_csv(
                self.config["DATA_FILE_PATH"], mode='w', header=True)

            # data = date_df.copy()
            # data.loc[:, 'Target'] = data[f'close {stock}']
            # for i, stock in enumerate(stocks):
            #     print(
            #         f"[MULTI_SYMBOL_DATASET]: Load and process data for {stock}")
            #     # Add a column with ones for each stock symbol.
            #     # data = date_df.copy()
            #     # data.loc[:, stock] = 1
            #     data.loc[:, 'Target'] = data[f'close {stock}']

            #     if i == 0:
            #         # The first stock is written to the file with the header.
            #         data.to_csv(self.config["DATA_FILE_PATH"],
            #                     mode='w', header=True)
            #         # The dimensions of the input and output data are required
            #         # to dimension the input and output layers of the model.
            #         self.input_dim = data.shape[1]
            #         self.output_dim = 1
            #     # else:
            #         # The other stocks are appended to the file without a header.
            #         # data.to_csv(self.config["DATA_FILE_PATH"],
            #         #            mode='a', header=False)
            print("File: \"" + self.config["DATA_FILE_PATH"] + "\" created.")
        else:
            # The existing file with input data is used.
            self.length = count_rows(self.config["DATA_FILE_PATH"])
            self.output_dim = 1
            self.input_dim = get_column_count(
                self.config["DATA_FILE_PATH"]) - self.output_dim

        # Define Sequence length for encoder and decoder
        self.seq_len_encoder = input_length
        self.seq_len_decoder = target_length

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        This is the length of the Input data minus the length of
        the Input for one sample minus the length of the target for one sample plus one.

        Returns:
            int: number of samples in the dataset
        """

        return self.length - self.seq_len_encoder - self.seq_len_decoder + 1

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
        start_input = idx
        end_input = idx + self.seq_len_encoder
        start_target = end_input
        end_target = start_target + self.seq_len_decoder
        input_length = self.seq_len_encoder
        target_length = self.seq_len_decoder

        data = read_csv_chunk(
            self.config["DATA_FILE_PATH"], start_input, end_target)

        # The last column contains the target data.
        target_data = data.iloc[:, -self.output_dim:].to_numpy()

        # for _ in range(self.output_dim):
        #     data.pop(data.columns[-1])

        # The target data must be a 2D array.
        target_data = np.array([np.array(element) for element in target_data])

        # The other columns contain the input data.
        input_data = data.to_numpy()

        # Get the input data of length INPUT_LEN
        input = input_data[0:input_length]
        input = torch.tensor(input, dtype=torch.float32)

        # Get the output target data of length TARGET_LEN after the input period
        target = target_data[input_length:input_length+target_length]
        target = torch.tensor(target, dtype=torch.float32)

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

    dataset = MultiSymbolDataset(txt_reader, config)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    test_sample = next(iter(dataloader))[1]
