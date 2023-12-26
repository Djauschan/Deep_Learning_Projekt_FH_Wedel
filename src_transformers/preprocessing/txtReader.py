import os
from typing import Optional
import pandas as pd
from datetime import date

type_dict = {"etf-complete": "ETF", "us3000": "stock", "usindex": "index"}


class DataReader():
    """
    The class is used to read the data from the files.
    """

    def __init__(self,
                 read_all_files: bool,
                 encoder_symbols: list[str],
                 decoder_symbols: list[str],
                 last_date: date,
                 data_dir_name: str = "data/input"):
        """
        Initializes a data reader.

        Args:
            read_all_files (bool): Boolean whether all files should be processed.
            encoder_symbols (list[str]): Symbols that are to be used as input features
            decoder_symbols (list[str]): Symbols that are to be predicted.
            last_date (date): Last date that is read in.
            data_dir_name (str, optional): Directory in which the data is located.

        It is expected that the files containing the data are located in subdirectories. \n
        C:. \n
        ├───data \n
        │   ├───input \n
        │   │   ├───etf-complete_tickers_A-C_1min_w1q7w \n
        │   │   ├───s3000_tickers_A-B_1min_iqksn \n
        │   │   ├───usindex_1min_u8d0l \n
        """

        # Dictionary for the configuration of data preprocessing is saved.
        self.read_all_files = read_all_files
        self.encoder_symbols = encoder_symbols
        self.decoder_symbols = decoder_symbols
        # Get last minute of the day
        self.last_date = pd.to_datetime(last_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

        self.root_folder = data_dir_name
        # A list containing all file names and a list containing all symbols
        # are created.
        self.txt_files, self.input_symbols, self.target_symbols = self.get_txt_files()
        # Counter, so that one file after the other can be read from the list
        # of file names.
        self.current_file_idx = 0

    def get_txt_files(self) -> tuple[list, list, list]:
        """
        Returns a list with the paths of all read txt files and a list with all symbols.

        Returns:
            tuple[list, list]: List with paths of all read txt files and list with all symbols
        """
        txt_files, input_symbols, target_symbols = [], [], []
        # Traverse subdirectories.
        for root, dirs, files in os.walk(self.root_folder):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                # In the subdirectory all files are traversed.
                for inner_root, inner_dirs, inner_files in os.walk(dir_path):
                    for file in inner_files:
                        symbol = file.split("_")[0]
                        if self.read_all_files or self.root_folder.endswith(
                                "test"):
                            # The paths of all text files are stored in a list.
                            if file.endswith(".txt"):
                                txt_files.append(os.path.join(dir_path, file))
                                input_symbols.append(symbol)
                        else:
                            # The paths of all text files selected via the symbols
                            # in the configuration file are saved in a list.
                            if file.endswith(
                                    ".txt") and symbol in self.encoder_symbols:
                                txt_files.append(os.path.join(dir_path, file))
                                input_symbols.append(symbol)
                            elif file.endswith(".txt") and symbol in self.decoder_symbols:
                                txt_files.append(os.path.join(dir_path, file))
                                target_symbols.append(symbol)
        return txt_files, input_symbols, target_symbols

    @classmethod
    def _get_type(cls, file_path: str) -> Optional[str]:
        """
        Returns the type of the data based on the path of the file.

        Args:
            file_path (str): path of the file

        Returns:
            str: type of data (EFT, stock, index)
        """
        directory = os.path.dirname(file_path)
        folder_name = os.path.basename(directory)
        for key in type_dict:
            if key in folder_name:
                return type_dict.get(key)
        return None

    def read_next_txt(self) -> Optional[pd.DataFrame]:
        """
        Reads in the next file if there are files that have not yet been read in.

        Returns:
            pd.DataFrame: Read in file.
        """
        # Check if there are files that have not been read in yet.
        if self.current_file_idx < len(self.txt_files):
            file_to_read = self.txt_files[self.current_file_idx]

            # Read in file.
            data = pd.read_csv(file_to_read, names=[
                               "timestamp", "open", "high", "low", "close", "volume"])

            # Replace Nan values in volume column with 0. Indices do not have a
            # volume.
            data["volume"] = data["volume"].fillna(0)

            # The ticker symbol to which the data belongs is included in the
            # file name.
            filename = os.path.basename(file_to_read)
            data["symbol"] = filename.split("_")[0]

            # The type of the data is determined based on the path of the file.
            data["type"] = DataReader._get_type(file_to_read)

            # Convert timestamp to python timestamp.
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], format="%Y-%m-%d %H:%M:%S")

            # Only Use data to the last timestamp of the last day
            data = data[data["timestamp"] <= self.last_date]

            self.current_file_idx += 1
            return data
        else:
            # No more files to read
            return None

    def reset_index(self):
        """
        Resets the index to start reading the files from the beginning again.
        """
        self.current_file_idx = 0
