import csv
from itertools import islice
from txtReader import DataReader


class virtualCsvFile:
    """
    Class, which provides a list of CSV files as a virtual CSV file.
    """

    def __init__(self, file_paths: list, symbols: list):
        """
        Initializes a virtual CSV file

        Args:
            file_paths (list): List of all file paths of the files that are to be concatenated to a virtual file
            symbols (list): For each file, this list saves the corresponding symbol at the same index.
        """
        # Dictionary to map index ranges to file paths
        self.index_to_file = {}
        # Number of lines in the concatenated file.
        self.total_rows = 0

        # Dictionaries are filled and the total number of lines is counted.
        for file_path, symbol in zip(file_paths, symbols):
            file_handle = open(file_path, 'r')
            csv_reader = csv.reader(file_handle)
            file_rows = sum(1 for _ in csv_reader)

            self.index_to_file[self.total_rows] = {
                'symbol': symbol,
                'file_handle': file_handle,
                'file_rows': file_rows
            }
            self.total_rows += file_rows

    def __getitem__(self, index: int) -> tuple[str, str]:
        """
        Returns the line at the Index position.

        Args:
            index (int): Index of the line to be read.

        Returns:
            tuple[str, str]: Symbol, read line
        """
        for start_index, file_features in self.index_to_file.items():
            if index < start_index + file_features['file_rows']:
                file_handle = file_features['file_handle']
                # Reduce the index by the number of lines of the skipped files.
                adjusted_index = index - start_index
                # Set pointer to start of file
                file_handle.seek(0)
                # Use islice to efficiently skip lines up to the desired line
                for _ in islice(file_handle, adjusted_index):
                    pass
                # Read in the desired line
                item = next(file_handle)
                return file_features['symbol'], item

    def __len__(self) -> int:
        """
        Returns the length of the virtual CSV file

        Returns:
            int: length of the virtual CSV file
        """
        return self.total_rows

    def close(self):
        """
        Closes all files that were merged in the virtual file.
        """
        for info in self.index_to_file.values():
            info['file_handle'].close()


# Code for testing & debugging
if __name__ == "__main__":
    reader = DataReader("test")
    file_paths, symbols = reader.get_txt_files()
    concatenated_csv = virtualCsvFile(file_paths, symbols)
    for i in range(20):
        print(concatenated_csv[i])
    concatenated_csv.close()
