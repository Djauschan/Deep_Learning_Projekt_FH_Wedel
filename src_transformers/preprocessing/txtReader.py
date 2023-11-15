import os
import pandas as pd
from src_transformers.preprocessing.config import config

type_dict = {"etf-complete": "ETF", "us3000": "stock", "usindex": "index"}


class DataReader():
    """
    The class is used to read the data from the files.
    """

    def __init__(self, data_dir_name: str = "data"):
        """
        Initializes a data reader.

        Args: \n
            data_dir_name (str, optional): Directory in which the data is located. Defaults to "data". 

        It is expected that the files containing the data are located in subdirectories. \n
        C:. \n
        ├───data \n
        │   ├───etf-complete_tickers_A-C_1min_w1q7w \n
        │   ├───... \n
        │   ├───us3000_tickers_Y-Z_1min_v264r \n      
        │   └───usindex_1min_u8d0l \n
        """
        # The path of the current file is determined.
        current_file_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the 'data' directory located two levels up from the current file's directory
        data_directory = os.path.join(
            current_file_directory, os.pardir, os.pardir, 'data')

        self.root_folder = data_directory
        # A list containing all file names and a list containing all symbols are created.
        self.txt_files, self.symbols = self.get_txt_files()
        # Counter, so that one file after the other can be read from the list of file names.
        self.current_file_idx = 0

    def get_txt_files(self) -> tuple[list, list]:
        """
        Returns a list with the paths of all read txt files and a list with all symbols.

        Returns:
            tuple[list, list]: List with paths of all read txt files and list with all symbols
        """
        txt_files, symbols = [], []
        # Traverse subdirectories.
        for root, dirs, files in os.walk(self.root_folder):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                # In the subdirectory all files are traversed.
                for inner_root, inner_dirs, inner_files in os.walk(dir_path):
                    for file in inner_files:
                        symbol = file.split("_")[0]
                        if config["READ_ALL_FILES"] or self.root_folder.endswith("test"):
                            # The paths of all text files are stored in a list.
                            if file.endswith(".txt"):
                                txt_files.append(os.path.join(dir_path, file))
                                symbols.append(symbol)
                        else:
                            # The paths of all text files selected via the symbols
                            # in the configuration file are saved in a list.
                            if file.endswith(".txt") and symbol in config["SYMBOLS_TO_READ"]:
                                txt_files.append(os.path.join(dir_path, file))
                                symbols.append(symbol)
        return txt_files, symbols

    def _get_type(self, file_path: str) -> str:
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

    def read_next_txt(self) -> pd.DataFrame:
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

            # Replace Nan vaules in voulme column with 0. Indices do not have a volume.
            data["volume"] = data["volume"].fillna(0)

            # The ticker symbol to which the data belongs is included in the file name.
            filename = os.path.basename(file_to_read)
            data["symbol"] = filename.split("_")[0]

            # The type of the data is determined based on the path of the file.
            data["type"] = self._get_type(file_to_read)

            # Convert timestamp to python timestamp.
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], format="%Y-%m-%d %H:%M:%S")

            self.current_file_idx += 1
            return data
        else:
            # No more files to read
            return None
