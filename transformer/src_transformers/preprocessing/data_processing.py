from pathlib import Path
from typing import Final, Optional

import pandas as pd

from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.logger import Logger

# If CSV files containing the mapping of symbols to names are available,
# they are read and dictionaries are created. The files are expected in the data directory.
# The source of the data is: https://stockanalysis.com/
ETF_MAPPING_FILE: Final[Path] = Path("data", "etf_mapping.csv")
STOCK_MAPPING_FILE: Final[Path] = Path("data", "stock_mapping.csv")
INDEX_MAPPING_FILE: Final[Path] = Path("data", "index_mapping.csv")

dict_of_dicts: dict = {}

for file, symbol_type in [(INDEX_MAPPING_FILE, "index"),
                          (ETF_MAPPING_FILE, "ETF"), (STOCK_MAPPING_FILE, "stock")]:
    if Path.exists(file):
        dict_file = pd.read_csv(file)
        dict_of_dicts[symbol_type] = (
            dict(zip(dict_file.iloc[:, 0], dict_file.iloc[:, 1])))


def lookup_symbol(symbol: str, symbol_type: str) -> Optional[str]:
    """
    Looks up the name of the passed symbol in the dictionary of the passed type.
    Returns the name of the symbol or None if the symbol is not in the dictionary or the dictionary does not exist.

    Args:
        symbol (str): Symbol for which the name is to be looked up.
        symbol_type (str): Type of the symbol.

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

    df.set_index('posix_time', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)

    return df


def get_all_dates(reader: DataReader, data_usage_ratio: float) -> pd.DataFrame:
    """
    Reads in all files that are to be imported and creates a data frame
    that contains the union of all timestamps of all read-in files.

    Args:
        reader (DataReader): DataReader object that reads in the files.

    Returns:
        pd.DataFrame: Data frame that contains the union of all timestamps of all read-in files.
    """
    all_timestamps = set()
    read_files = 0

    while True:
        file_df = reader.read_next_txt()
        if file_df is None:
            break

        all_timestamps.update(file_df['timestamp'])
        # Explicitly delete the data frame to free up memory.
        del file_df

        read_files += 1
        Logger.log_text(
            f"Read {read_files} file(s), totalling {len(all_timestamps)} timestamps.")

    reader.reset_index()

    all_timestamps = sorted(all_timestamps)
    all_timestamps_length = len(all_timestamps)
    used_timestamps_length = int(all_timestamps_length * data_usage_ratio)

    if all_timestamps_length == used_timestamps_length:
        start_index = 0
    else:
        start_index = all_timestamps_length - used_timestamps_length - 1

    used_timestamps = all_timestamps[start_index:all_timestamps_length]
    used_timestamps_df = pd.DataFrame({'timestamp': used_timestamps})

    return used_timestamps_df


def fill_dataframe(all_dates: pd.DataFrame,
                   reader: DataReader, time_resolution: int) -> tuple[list, pd.DataFrame]:
    """
    A data frame is created that contains the values required for training for all files that are to be read in.
    The columns are filled so that values are available for all files for all timestamps.
    For stocks and ETFs, the closing price and the volume per minute are loaded.
    For indices, only the closing price per minute is taken into account.
    The data frame is then resampled to the desired time resolution.


    Args:
        all_dates (pd.DataFrame): Data frame that contains the union of all timestamps of all read-in files.
        reader (DataReader): DataReader object that reads in the files.
        time_resolution (int): Time resolution to which the data frame is to be resampled.

    Returns:
        tuple[list, dict, pd.DataFrame]:
        Symbols of all stocks in the data frame
        Dictionary that assigns an absolute price value to all symbols.
        Data frame containing the values required for training.
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

            # The symbols of all stocks are saved in a list as they are used as
            # target variables.
            if symbol_type == 'stock':
                stocks.append(symbol)

        # Explicitly delete the data frame to free up memory.
        del file_df

    # Set Timestamp as index.
    all_dates.set_index('timestamp', inplace=True)

    # Absolute prices of the last known time step (offset) are saved.
    offset = 0
    # Fill the Dictionary with the absolute price values of all symbols at position "offset".
    prices = {}
    for symbol in all_dates.columns:
        if 'close' in symbol:
            only_symbol = symbol.split(' ')[1]
            prices[only_symbol] = all_dates[symbol].iloc[offset]

    # Change time resolution of data frame.
    # The volume is summed up, the last closing price of the intervall is used.
    # The selectied timesamp is the first of the intervall.
    # The values beginning with the selected timestamt to the last of the intervall are considered.
    # Example: time reslution = 30 timestamps = 2019-08-02 04:00:00
    # volume = sum between 2019-08-02 04:00:00 and 2019-08-02 04:29:00
    # close = last value between 2019-08-02 04:00:00 and 2019-08-02 04:29:00

    # Find all 'close' and 'volume' columns
    close_columns = [col for col in all_dates.columns if 'close' in col]
    volume_columns = [col for col in all_dates.columns if 'volume' in col]
    # Create a dictionary for  aggregation
    agg_dict = {col: 'last' for col in close_columns}
    agg_dict.update({col: 'sum' for col in volume_columns})
    # Resample the DataFrame with a frequency of time_resolution entries and aggregate dynamically
    agg_df = all_dates.resample(f"{time_resolution}Min").agg(agg_dict)

    # Drop rows which contain any NaN values.
    # This removes rows that were added by the aggregation but for which there are no values in the real data.
    agg_df.dropna(inplace=True)

    # Apply relative differencing on close colum to get the change between timestamps.
    for colum in close_columns:
        # Apply relative differencing on close colum to get the change between timestamps.
        agg_df[colum] = agg_df[colum].pct_change(fill_method=None)
        # If the vaules do not differ, the result is NaN. This is replaced by 0.
        agg_df[colum] = agg_df[colum].fillna(0)

    # Drop the first row, because it contains NaN values due to the differencing.
    agg_df = agg_df.drop(agg_df.index[0]).reset_index()

    return stocks, prices, agg_df
