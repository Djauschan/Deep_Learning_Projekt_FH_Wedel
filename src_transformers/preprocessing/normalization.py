import pandas as pd
from sklearn import preprocessing
import yaml
import sys
import pickle
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.plot import plot_df

def generate_scaler(config: dict) -> None:
    """
    Visualizes the files and saves the result in the directory data/output

    Args:
        config (dict): Dictionary for the configuration of data preprocessing.
    """
    # Get the parameters for the data set.
    data_parameters = config["dataset_parameters"]
    # Remove parameter that the datareader does not need.
    data_parameters.pop("create_new_file")
    data_parameters.pop("data_file")
    data_parameters.pop("data_usage_ratio")
    txt_reader = DataReader(**data_parameters)
    data = txt_reader.read_next_txt()
    # As long as there is data and the user does not stop, data will be
    # visualized.
    scaler_dict = {}
    while data is not None:
        symbol = data.iloc[0]["symbol"]
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(data['close'].values.reshape(-1, 1))
        scaler_dict[symbol] = scaler

        data = txt_reader.read_next_txt()

    with open("data/output/scaler.pkl", 'wb') as file:
        pickle.dump(scaler_dict, file)

    # Laden der Pickle-Datei
    with open('data/output/scaler.pkl', 'rb') as file:
        loaded_scalers = pickle.load(file)
    print(loaded_scalers)


if __name__ == "__main__":

    # Check if the path to the configuration file was passed as an argument.
    assert len(sys.argv) == 2

    # Read in the configuration file.
    config_file_path = sys.argv[1]
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Visualize data.
    generate_scaler(config)
