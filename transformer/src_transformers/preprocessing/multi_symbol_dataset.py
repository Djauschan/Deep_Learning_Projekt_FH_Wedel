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
    subseries_amount: int
    validation_split: float
    encoder_dimensions: int
    decoder_dimensions: int
    encoder_input_length: int
    decoder_target_length: int
    data_file: str
    scaler: str
    time_resolution: int
    ignore_nights: bool

    @classmethod
    def create_from_config(cls,
                           read_all_files: bool,
                           first_date: date,
                           last_date: date,
                           data_usage_ratio: float,
                           subseries_amount: int,
                           validation_split: float,
                           create_new_file: bool,
                           data_file: str,
                           encoder_symbols: list[str],
                           decoder_symbols: list[str],
                           encoder_input_length: int,
                           decoder_target_length: int,
                           scaler: str = "MinMaxScaler",
                           time_resolution: int = 1,
                           ignore_nights: bool = False):
        """
        This method either creates a new MultiSymbolDataset by preprocessing financial data for
        multiple symbols (using information from the configuration file) or creates the dataset
        using existing data from a csv file. If data is preprocessed using the configuration file,
        the preprocessed data is stored for later use.

        Args:
            read_all_files (bool): Whether to read all files in the data directory.
            first_date (date): The first date to use in the dataset.
            last_date (date): The last date to use in the dataset.
            data_usage_ratio (float): The ratio of data to use for the dataset.
            create_new_file (bool): Whether to create a new file for the preprocessed data.
            data_file (str): The path to the file to store the data in or to load the data from.
            encoder_symbols (list[str]): The symbols to use for the encoder input.
            decoder_symbols (list[str]): The symbols to use for the decoder input.
            encoder_input_length (int): The length of the encoder input sequence.
            decoder_target_length (int): The length of the decoder target sequence.
            scaler (str, optional): The scaler to use for the data. Defaults to "MinMaxScaler".
            time_resolution (int, optional): The time resolution of the data in minutes.
            ignore_nights (bool, optional): Whether to ignore the night hours in the data.

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

            # Create the instance of the MultiSymbolDataset
            instance_multi_symbol_dataset = cls(length=length,
                                                subseries_amount=subseries_amount,
                                                validation_split=validation_split,
                                                encoder_dimensions=encoder_dimensions,
                                                decoder_dimensions=decoder_dimensions,
                                                encoder_input_length=encoder_input_length,
                                                decoder_target_length=decoder_target_length,
                                                data_file=data_file,
                                                scaler=scaler,
                                                time_resolution=time_resolution,
                                                ignore_nights=ignore_nights)

        else:
            Logger.log_text(
                "Data pre-processing for the multi symbol dataset was started.")

            data_reader = DataReader(
                read_all_files, encoder_symbols, decoder_symbols, first_date, last_date)

            data_df = get_all_dates(data_reader, data_usage_ratio)
            Logger.log_text(
                f"Created a dataframe from the selected {len(data_df)} timestamps, "
                + f"since the user specified a data usage ratio of {data_usage_ratio}.")
            cls.stocks, cls.prices, data_df = fill_dataframe(
                data_df, data_reader, time_resolution, ignore_nights)

            # Select Data for Prediction Interface
            # Only Select timestamp index is greater or equal to 2020-12-22 04:00:00
            # data_df = data_df[(data_df['timestamp'] >= pd.to_datetime("2020-12-22 04:00:00", format="%Y-%m-%d %H:%M:%S"))]

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

            # Create the instance of the MultiSymbolDataset
            instance_multi_symbol_dataset = cls(length=length,
                                                subseries_amount=subseries_amount,
                                                validation_split=validation_split,
                                                encoder_dimensions=encoder_dimensions,
                                                decoder_dimensions=decoder_dimensions,
                                                encoder_input_length=encoder_input_length,
                                                decoder_target_length=decoder_target_length,
                                                data_file=data_file,
                                                scaler=scaler,
                                                time_resolution=time_resolution,
                                                ignore_nights=ignore_nights)

            # Normalization
            scaler = SCALER_OPTIONS[scaler]()

            # Get all columns that contain volume and indeces in train set
            volume_cols = [
                item for item in data_df.columns if "volume" in item]
            train_indeces, validation_indecies = instance_multi_symbol_dataset.get_subset_indices()

            # NOTE: This is only to create the dataset for the prediction interface
            # scaler = pickle.load(open("data/output/Multi_Symbol_Train_scaler.pkl", "rb"))
            # with open("data/output/prices.pkl", 'wb') as file:
            #     pickle.dump(instance_multi_symbol_dataset.prices, file)

            # Fit scaler to each volume column only with train data
            scaler.fit(data_df[volume_cols].iloc[train_indeces])

            # Transform train and test data
            data_df[volume_cols] = scaler.transform(data_df[volume_cols])

            # Store scaler in pickle file
            scaler_path = instance_multi_symbol_dataset.data_file.replace(
                '.csv', f'_scaler.pkl')
            with open(scaler_path, 'wb') as file:
                pickle.dump(scaler, file)

            # Store data in csv file
            data_df.to_csv(data_file, mode='w', header=True)
            Logger.log_text(
                f"Dataframe holding the preprocessed data was stored to the file '{data_file}'.")

        return instance_multi_symbol_dataset

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

        # Load the data from the file, offset by the given index
        data, _ = read_csv_chunk(
            self.data_file, encoder_input_start, decoder_target_end)
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

    def get_subset_indices(self) -> tuple[list[int], list[int]]:
        """
        Computes and returns the indices for the training and validation sets of all subseries.

        This method calculates the lengths of the subseries, as well as their training and
        validation sets. It then generates the indices for the training and validation subsets,
        ensuring that the training and validation sets do not overlap by considering the
        transformer's encoder and decoder input lengths.

        Returns:
            list[int]: A list containing the indices for the training sets of all subseries.
            list[int]: A list containing the indices for the validation sets of all subseries.
        """
        # Compute the lengths of the subseries, as well as their training and validation parts
        subseries_length = int(self.length / self.subseries_amount)
        validation_length = int(subseries_length * self.validation_split)
        training_length = subseries_length - validation_length

        training_indices = []
        validation_indices = []
        # Set start index for the first training subseries to the remainder (X) of the modulo
        # This is done to skip the first X elements which we cannot allocate to a subseries
        training_start = self.length % subseries_length

        # Iterate over all subseries
        for _ in range(self.subseries_amount):
            # The validation set of the subseries starts right after the training set
            validation_start = training_start + training_length
            # The training set of the subseries ends before the validation set, keeping the
            # encoder and decoder input lengths in mind to not overlap training and validation sets
            training_end = validation_start - \
                self.decoder_target_length - self.encoder_input_length + 1

            # Create the training indices for this subseries and append them to the list
            subseries_training_indices = range(training_start, training_end)
            training_indices.extend(subseries_training_indices)

            # The training set of the *next* subseries starts right after the validation set
            training_start = validation_start + validation_length
            # The validation set of the subseries ends before the next subseries' training set,
            # keeping the encoder and decoder input lengths in mind to not overlap the subseries
            validation_end = training_start - \
                self.decoder_target_length - self.encoder_input_length + 1

            # Create the validation indices for this subseries and append them to the list
            subseries_validation_indices = range(
                validation_start, validation_end)
            validation_indices.extend(subseries_validation_indices)

        return training_indices, validation_indices
