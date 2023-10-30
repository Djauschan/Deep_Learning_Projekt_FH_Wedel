import os
import pandas as pd

# If there is a CSV file that contains the mapping of the symbol to the ETF name, 
# it will be read and a dictionary will be created.
# The source of the data is: https://stockanalysis.com/etf/
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = os.path.join(current_file_dir, "data")
etf_dict_path = os.path.join(data_dir_path, "ETF_Mapping.csv")
my_dict = None
if os.path.exists(etf_dict_path):
    etf_dict = pd.read_csv(etf_dict_path, delimiter=";")
    my_dict = dict(zip(etf_dict['Symbol'], etf_dict['Fund Name']))
    


def lookup_symbol(key: str) -> str:
    """
    Returns the name associated with the symbol, 
    or None if the name is not in the dictionary or no dictionary has been created.

    Args:
        key (str): Symbol whose name is being searched for.

    Returns:
        str: Name of the symbol or None
    """
    if my_dict is not None:
        return my_dict.get(key)
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