import pandas as pd
import csv


def read_csv_chunk(file_path: str, start_index: int, stop_index: int) -> pd.DataFrame:
    """
    The lines of the CSV file from start index (inclusive) to stop index (exclusive) are read in.

    Args:
        file_path (str): Path of the CSV file to be read.
        start_index (int): Start index of the extract from the CSV file.
        stop_index (int): Stop index of the extract from the CSV file.

    Returns:
        pf.DataFrame: Extract from the CSV file.
    """

    # Calculate the number of rows to read
    num_rows_to_read = stop_index - start_index

    # Read the specified range from the CSV file, starting from the header
    df = pd.read_csv(file_path, skiprows=start_index,
                     nrows=num_rows_to_read)
    return df


def count_rows(csv_file_path: str) -> int:
    """
    Counts the number of rows in the CSV file.

    Args:
        csv_file_path (str): Path of the CSV file.

    Returns:
        int: Number of rows in the CSV file.
    """
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # skip the header
        row_count = sum(1 for row in csv_reader)
    return row_count


def get_column_count(csv_file_path: str) -> int:
    """
    Counts the number of columns in the CSV file.

    Args:
        csv_file_path (str): Path of the CSV file.

    Returns:
        int: Number of columns in the CSV file.
    """
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        return len(header)
