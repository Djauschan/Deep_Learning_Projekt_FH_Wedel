import os
import pandas as pd


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
        │   └───etf-complete_tickers_T-Z_1min_mkvx9 \n
        """
        # The data directory must be in the same folder as the reader.
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir_path = os.path.join(current_file_dir, data_dir_name)
        self.root_folder = data_dir_path
        # A list is created that contains all file names.
        self.txt_files = self.get_txt_files()
        # Counter, so that one file after the other can be read from the list of file names.
        self.current_file_idx = 0

    def get_txt_files(self) -> list:
        """
        Returns a list with the paths of all text files.

        Returns:
            list: List with the paths of all text files.
        """
        txt_files = []
        # Traverse subdirectories.
        for root, dirs, files in os.walk(self.root_folder):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                # In the subdirectory all files are traversed.
                for inner_root, inner_dirs, inner_files in os.walk(dir_path):
                    for file in inner_files:
                        # The paths of all text files are stored in a list.
                        if file.endswith(".txt"):
                            txt_files.append(os.path.join(dir_path, file))
        return txt_files

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
                               "timestamp", "value 1", "value 2", "value 3", "value 4", "volume"])

            # The ticker symbol to which the data belongs is included in the file name.
            filename = os.path.basename(file_to_read)
            data["symbol"] = filename.split("_")[0]

            # Convert timestamp to python timestamp.
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], format="%Y-%m-%d %H:%M:%S")

            self.current_file_idx += 1
            return data
        else:
            # No more files to read
            return None
