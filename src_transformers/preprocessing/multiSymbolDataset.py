import torch
from torch.utils.data import Dataset

from src_transformers.preprocessing.csv_io import get_csv_shape, read_csv_chunk
from src_transformers.preprocessing.dataProcessing import (
    add_time_information,
    fill_dataframe,
    get_all_dates,
)
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.logger import Logger


class MultiSymbolDataset(Dataset):
    """
    Data stored as tensors
    Pytorch uses the 3 functions [__init__, __len__, __getitem__]
    """

    def __init__(self, config: dict, encoder_input_length: int, decoder_input_length: int):
        """
        Initializes the Pytorch data set.

        Args:
            config (dict): Dictionary for the configuration of data preprocessing.
            input_length (int): Length of the input sequence.
            target_length (int): Length of the target sequence.
        """

        self.config = config
        self.encoder_input_length = encoder_input_length
        self.decoder_input_length = decoder_input_length

        if not self.config['CREATE_NEW_FILE']:
            # Read the existing file if the user wants to skip creating a new file
            Logger.log_text(
                "Loading pre-processed data from file for the multi symbol dataset.")
Logger.log_text(
                "Added more precise time information to the pre-processed data.")
            self.decoder_dimensions = len(self.config["target_symbols"])
        else:
            Logger.log_text(
                "Data pre-processing for the multi symbol dataset has started.")

            data_reader = DataReader(self.config)

            date_df = get_all_dates(data_reader)
            self.stocks, date_df = fill_dataframe(date_df, data_reader)
            Logger.log_text(
                "Added the time stamps from all loaded files and filled missing values to the pre-processed data.")

            date_df = add_time_information(date_df)
            date_df.set_index('posix_time', inplace=True)
            date_df.drop(columns=['timestamp'], inplace=True)
            Logger.log_text(
                "Added more precise time information to the pre-processed data.")

            # If the number of columns is odd, a column with zeros is added.
            # This is necessary because the number of columns must be even for the
            # positional encoding of the transformer model.
            if len(date_df.columns) % 2 != 0:
                date_df['even'] = 0

            current_columns = list(date_df.columns.values)
            target_columns = []
            targets = 0

            for symbol in self.config["target_symbols"]:
                current_columns.remove(f'close {symbol}')
                target_columns.append(f'close {symbol}')
                targets += 1

            # Re-ordering the target columns to be at the end
            date_df = date_df[current_columns + target_columns]

            self.length = len(date_df)
            self.encoder_dimensions = date_df.shape[1]
            self.decoder_dimensions = targets

            file = self.config["DATA_FILE_PATH"]
            date_df.to_csv(file, mode='w', header=True)
            Logger.log_text(
                f"Pre-processed data was stored in the file '{file}'.")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        This is the length of the Input data minus the length of
        the Input for one sample minus the length of the target for one sample plus one.

        Returns:
            int: number of samples in the dataset
        """
        return self.length - self.encoder_input_length - self.decoder_input_length + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sample of input and target sequences. The input and target sequences are of the
        length defined in the config. The start of the target sequence is the first entry after
        the input sequence.

        Args:
            idx (int): position of sample in the tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sampel (X, Y) at position idx
        """
        input_start, input_end = index, index + self.encoder_input_length
        target_end = input_end + self.decoder_input_length

        # Load the data from the file, offset by the given index
        data = read_csv_chunk(
            self.config["DATA_FILE_PATH"], input_start, target_end)

        encoder_input = data.to_numpy()
        # Get the encoder input from 0 to self.input_length
        encoder_input = encoder_input[0:self.encoder_input_length]
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)

        decoder_input = data.iloc[:, -self.decoder_dimensions:].to_numpy()
        # Get the decoder input (starting from the end of encoder input)
        decoder_input = decoder_input[self.encoder_input_length:self.encoder_input_length +
                                      self.decoder_input_length]
        decoder_input = torch.tensor(decoder_input, dtype=torch.float32)

        return encoder_input, decoder_input
