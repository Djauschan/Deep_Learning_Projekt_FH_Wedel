import os
import pandas as pd
import numpy as np

# If CSV files containing the mapping of symbols to names are available,
# they are read and dictionaries are created.
# The dictionaries are expected in the data directory.

# The path of the current file is determined.
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'data' directory located two levels up from the current file's directory
data_dir_path = os.path.join(
    current_file_directory, os.pardir, os.pardir, 'data')


# The source of the data is: https://stockanalysis.com/
etf_dict_path = os.path.join(data_dir_path, "etf_mapping.csv")
stock_dict_path = os.path.join(data_dir_path, "stock_mapping.csv")
# This dictionary was created manually.
index_dict_path = os.path.join(data_dir_path, "index_mapping.csv")
dict_of_dicts: dict = {}

for file_path, type in [(index_dict_path, "index"), (etf_dict_path, "ETF"), (stock_dict_path, "stock")]:
    if os.path.exists(file_path):
        dict_file = pd.read_csv(file_path)
        dict_of_dicts[type] = (
            dict(zip(dict_file.iloc[:, 0], dict_file.iloc[:, 1])))


def lookup_symbol(key: str, type: str) -> str:
    """
    Looks up the name of the passed symbol in the dictionary of the passed type.
    Returns the name of the symbol or None if the symbol is not in the dictionary or the dictionary does not exist.

    Args:
        key (str): Symbol for which the name is to be looked up.
        type (str): Type of the symbol.

    Returns:
        str: Name of the symbol or None if the symbol is not in the dictionary or the dictionary does not exist.
    """
    if type in dict_of_dicts:
        dictionary: dict = dict_of_dicts.get(type)
        return dictionary.get(key)
    else:
        return None


def add_time_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional time information to the passed data frame in the form of additional columns.

    Args:
        df (pd.DataFrame): Data frame to which additional time information is to be added.

    Returns:
        pd.DataFrame: Data frame with additional time information.
    """
    # It is checked whether a transaction is the first or the last of the day.
    df['first of day'] = df.groupby(
        df['timestamp'].dt.date).cumcount() == 0
    df['last of day'] = df.groupby(
        df['timestamp'].dt.date).cumcount(ascending=False) == 0
    # Convert boolean to integer.
    df['first of day'] = df['first of day'].astype(int)
    df['last of day'] = df['last of day'].astype(int)
    # Convert python time to posix time.
    df['posix_time'] = df['timestamp'].apply(
        lambda x: x.timestamp())
    return df


def create_one_hot_vector(symbols: list, symbol: str) -> np.array:
    """
    A one-hot vector is created from the list of all categories and the passed category.
    The 1 is the index of the category in the ascending sorted category list.

    Args:
        symbols (list): List of all categories
        symbol (str): Category for which a one-hot vector is to be created.

    Raises:
        ValueError: If the passed category is not in the category list.

    Returns:
        np.array: Passed category represented as one-hot vector.
    """
    unique_symbols = list(set(symbols))
    unique_symbols.sort()

    if symbol in unique_symbols:
        index = unique_symbols.index(symbol)
        one_hot_vector = np.zeros(len(unique_symbols), dtype=int)
        one_hot_vector[index] = 1
        return one_hot_vector
    else:
        raise ValueError(f"Symbol: '{symbol}' not found!")
