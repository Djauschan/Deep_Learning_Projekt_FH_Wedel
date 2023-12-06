"""
This module contains the MultiSymbolDataset class which is used to handle multi-symbol
financial data.
"""
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src_transformers.preprocessing.csv_io import get_csv_shape, read_csv_chunk
from src_transformers.preprocessing.data_processing import (
    add_time_information,
    fill_dataframe,
    get_all_dates,
)
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.logger import Logger


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
    decoder_input_length: int
    data_file: str

    @classmethod
    def create_from_config(cls,
                           read_all_files: bool,
                           data_usage_ratio: float,
                           create_new_file: bool,
                           data_file: str,
                           encoder_symbols: list[str],
                           decoder_symbols: list[str],
                           encoder_input_length: int,
                           decoder_input_length: int):
        """
        This method either creates a new MultiSymbolDataset by preprocessing financial data for
        multiple symbols (using information from the configuration file) or creates the dataset
        using existing data from a csv file. If data is preprocessed using the configuration file,
        the preprocessed data is stored for later use.

        Args:
            read_all_files (bool): Whether to read all files in the data directory.
            create_new_file (bool): Whether to create a new file for the preprocessed data.
            data_file (str): The path to the file to store the data in or to load the data from.
            encoder_symbols (list[str]): The symbols to use for the encoder input.
            decoder_symbols (list[str]): The symbols to use for the decoder input.
            encoder_input_length (int): The length of the encoder input sequence.
            decoder_input_length (int): The length of the decoder input sequence.

        Returns:
            MultiSymbolDataset: The created or loaded dataset.
        """
        if not create_new_file:
            # Read the existing file if the user wants to skip creating a new
            # file
            Logger.log_text(
                "Loading pre-processed data from file for the multi symbol dataset.")

            length, encoder_dimensions = get_csv_shape(data_file)
            decoder_dimensions = len(decoder_symbols)
        else:
            Logger.log_text(
                "Data pre-processing for the multi symbol dataset was started.")

            data_reader = DataReader(
                read_all_files, encoder_symbols, decoder_symbols)

            data_df = get_all_dates(data_reader, data_usage_ratio)
            Logger.log_text(
                f"Created a dataframe from the selected {len(data_df)} timestamps, "
                + f"since the user specified a data usage ratio of {data_usage_ratio}.")
            cls.stocks, data_df = fill_dataframe(data_df, data_reader)
            Logger.log_text(
                "Filled the timestamp dataframe with data from the selected stocks and symbols.")
            data_df = add_time_information(data_df)
            Logger.log_text(
                "Added more precise time information to the dataframe.")

            # If the number of columns is odd, a column with zeros is added.
            # This is necessary because the number of columns must be even for the
            # positional encoding of the transformer model.
            if len(data_df.columns) % 2 != 0:
                data_df['even'] = 0

            current_columns = data_df.columns.to_list()
            target_columns = []

            for symbol in decoder_symbols:
                current_columns.remove(f'close {symbol}')
                target_columns.append(f'close {symbol}')

            # Re-ordering the target columns to be at the end
            data_df = data_df[current_columns + target_columns]

            length = len(data_df)
            encoder_dimensions = data_df.shape[1]
            decoder_dimensions = len(decoder_symbols)

            data_df.to_csv(data_file, mode='w', header=True)
            Logger.log_text(
                f"Dataframe holding the preprocessed data was stored to the file '{data_file}'.")

        return cls(length=length,
                   encoder_dimensions=encoder_dimensions,
                   decoder_dimensions=decoder_dimensions,
                   encoder_input_length=encoder_input_length,
                   decoder_input_length=decoder_input_length,
                   data_file=data_file)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        The number of samples is calculated as the total length of the data minus
        the length of the encoder and decoder input sequences plus one.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.length - self.encoder_input_length - self.decoder_input_length + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns time series sequences as input for the encoder and decoder from the dataset
        starting at the specified index.

        The encoder input sequence starts at the specified index and has a length of
        `self.encoder_input_length`. The decoder input sequence starts immediately after the
        input sequence and has a length of `self.decoder_input_length`.

        The method reads a chunk of data from the data file, starting at `encoder_input_start`
        and ending at `decoder_input_end`. It then extracts the encoder and decoder inputs
        from this chunk and converts them to PyTorch tensors.

        Args:
            index (int): The index of the sample in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Time series sequences as input for the encoder
                and decoder starting at the specified index.
        """
        encoder_input_start, encoder_input_end = index, index + self.encoder_input_length
        decoder_input_end = encoder_input_end + self.decoder_input_length

        # Load the data from the file, offset by the given index
        data = read_csv_chunk(
            self.data_file, encoder_input_start, decoder_input_end)

        encoder_input = data.to_numpy()
        # Get the encoder input from 0 to self.input_length
        encoder_input = encoder_input[0:self.encoder_input_length]
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)

        decoder_input = data.iloc[:, -self.decoder_dimensions:].to_numpy()
        # Get the decoder input (starting from the end of encoder input)
        decoder_input = decoder_input[self.encoder_input_length:self.encoder_input_length +
                                      self.decoder_input_length]
        decoder_input = torch.tensor(decoder_input, dtype=torch.float32)
        # TODO decoder_input umbennen in decoder_target

        return encoder_input, decoder_input
