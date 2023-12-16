import pickle
import sys

import pandas as pd
import yaml

from src_transformers.preprocessing.preprocessing_constants import SCALER_OPTIONS
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.plot import plot_df


def generate_scaler(config: dict) -> None:
    """
    Generates a scaler for each symbol in the data set and stores it in a pickle file.
    Args:
        config: The configuration file.

    Returns: None

    """
    # Get the parameters for the data set.
    data_parameters = config["dataset_parameters"]

    # Remove parameter that the datareader does not need.
    data_parameters.pop("create_new_file")
    data_parameters.pop("data_file")
    data_parameters.pop("data_usage_ratio")
    data_parameters.pop("scaler")
    data_parameters.pop("subseries_amount")
    data_parameters.pop("validation_split")

    # As long as there is data and the user does not stop, Scaler is created for each symbol.
    scaler_dict = {}
    for scaler in SCALER_OPTIONS:
        txt_reader = DataReader(**data_parameters)
        data = txt_reader.read_next_txt()
        while data is not None:
            symbol = data.iloc[0]["symbol"]
            if data.iloc[0]['type'] == 'stock' or data.iloc[0]['type'] == 'ETF':
                scaler_volume = scaler()
                scaler_volume.fit(data['volume'].values.reshape(-1, 1))
                scaler_dict[f'volume {symbol}'] = scaler_volume

            data = txt_reader.read_next_txt()

        # Save all scaler in a pickle file.
        with open(f"data/output/scaler_{type(scaler_volume).__name__}.pkl", 'wb') as file:
            pickle.dump(scaler_dict, file)

        print(scaler)


if __name__ == "__main__":

    # Check if the path to the configuration file was passed as an argument.
    assert len(sys.argv) == 2

    # Read in the configuration file.
    config_file_path = sys.argv[1]
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Generate scaler
    generate_scaler(config)
