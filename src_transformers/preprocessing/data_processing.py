from pathlib import Path
from typing import Final

import pandas as pd

from src_transformers.preprocessing.txtReader import DataReader

# If CSV files containing the mapping of symbols to names are available,
# they are read and dictionaries are created. The files are expected in the data directory.
# The source of the data is: https://stockanalysis.com/
ETF_MAPPING_FILE: Final[Path] = Path("data", "etf_mapping.csv")
STOCK_MAPPING_FILE: Final[Path] = Path("data", "stock_mapping.csv")
INDEX_MAPPING_FILE: Final[Path] = Path("data", "index_mapping.csv")

dict_of_dicts: dict = {}

for file, symbol_type in [(INDEX_MAPPING_FILE, "index"), (ETF_MAPPING_FILE, "ETF"), (STOCK_MAPPING_FILE, "stock")]:
    if Path.exists(file):
        dict_file = pd.read_csv(file)
        dict_of_dicts[symbol_type] = (
            dict(zip(dict_file.iloc[:, 0], dict_file.iloc[:, 1])))


def lookup_symbol(symbol: str, symbol_type: str) -> str:
    """
    Looks up the name of the passed symbol in the dictionary of the passed type.
    Returns the name of the symbol or None if the symbol is not in the dictionary or the dictionary does not exist.

    Args:
        key (str): Symbol for which the name is to be looked up.
        type (str): Type of the symbol.

    Returns:
        str: Name of the symbol or None if the symbol is not in the dictionary or the dictionary does not exist.
    """
    if symbol_type in dict_of_dicts:
        dictionary: dict = dict_of_dicts.get(symbol_type)
        return dictionary.get(symbol)
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


def get_all_dates(reader: DataReader) -> pd.DataFrame:
    """
    Reads in all files that are to be imported and creates a data frame that contains the union of all timestamps of all read-in files.

    Args:
        reader (DataReader): DataReader object that reads in the files.

    Returns:
        pd.DataFrame: Data frame that contains the union of all timestamps of all read-in files.
    """
    all_timestamps = set()

    while True:
        file_df = reader.read_next_txt()
        if file_df is None:
            break

        all_timestamps.update(file_df['timestamp'])
        # Explicitly delete the data frame to free up memory.
        del file_df

    reader.reset_index()

    # Create an empty DataFrame with timestamps
    return pd.DataFrame({'timestamp': sorted(list(all_timestamps))})


def fill_dataframe(all_dates: pd.DataFrame, reader: DataReader) -> tuple[list, pd.DataFrame]:
    """
    A data frame is created that contains the values required for training for all files that are to be read in. 
    The columns are filled so that values are available for all files for all timestamps.
    For stocks and ETFs, the closing price and the volume per minute are loaded. 
    For indices, only the closing price per minute is taken into account.


    Args:
        all_dates (pd.DataFrame): Data frame that contains the union of all timestamps of all read-in files.
        reader (DataReader): DataReader object that reads in the files.

    Returns:
        tuple[list, pd.DataFrame]: Symbols of all stocks in the data frame, data frame containing the values required for training.
    """
    stocks = []

    while True:
        file_df = reader.read_next_txt()
        if file_df is None:
            break

        symbol = file_df['symbol'].iloc[0]
        symbol_type = file_df['type'].iloc[0]

        if symbol_type == 'index':
            # Only the closing price is used for indices.
            merged_df = pd.merge(all_dates, file_df[['timestamp', 'close']],
                                 how='left', on='timestamp', suffixes=('', f'_{symbol}'))
            # ffill: forward fill, bfill: backward fill
            all_dates[f'close {symbol}'] = merged_df['close'].ffill().bfill()

        if symbol_type == 'stock' or symbol_type == 'ETF':
            # The closing price and the volume are used for stocks and ETFs.
            merged_df = pd.merge(all_dates, file_df[['timestamp', 'close', 'volume']],
                                 how='left', on='timestamp', suffixes=('', f'_{symbol}'))
            # ffill: forward fill, bfill: backward fill
            all_dates[f'close {symbol}'] = merged_df['close'].ffill().bfill()
            all_dates[f'volume {symbol}'] = merged_df['volume'].fillna(0)
            # The symbols of all stocks are saved in a list as they are used as target variables.
            if symbol_type == 'stock':
                stocks.append(symbol)

        # Explicitly delete the data frame to free up memory.
        del file_df

    return stocks, all_dates
