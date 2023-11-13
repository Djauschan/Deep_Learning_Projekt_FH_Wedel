import os
import pandas as pd
import numpy as np

# If CSV files containing the mapping of symbols to names are available,
# they are read and dictionaries are created.
# The dictionaries are expected in the data directory.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = os.path.join(current_file_dir, "data")
# The source of the data is: https://stockanalysis.com/
etf_dict_path = os.path.join(data_dir_path, "etf_mapping.csv")
stock_dict_path = os.path.join(data_dir_path, "stock_mapping.csv")
# This dictionary was created manually.
index_dict_path = os.path.join(data_dir_path, "index_mapping.csv")
dicts = []

for file_path in [index_dict_path, etf_dict_path, stock_dict_path]:
    if os.path.exists(file_path):
        dict_file = pd.read_csv(file_path)
        dicts.append(dict(zip(dict_file.iloc[:, 0], dict_file.iloc[:, 1])))


def lookup_symbol(key: str) -> str:
    """
    Returns the name assigned to the symbol, 
    or None if the name is not contained in any dictionary 
    or if no dictionary has been created.

    Args:
        key (str): Symbol whose name is being searched for.

    Returns:
        str: Name of the symbol or None
    """
    for dictionary in dicts:
        if dictionary is not None and key in dictionary:
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
