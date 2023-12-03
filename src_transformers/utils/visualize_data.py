import yaml
import sys
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.plot import plot_df

# Code to visualize the data.


def visualize_data(config: dict) -> None:
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
    txt_reader = DataReader(**data_parameters)
    data = txt_reader.read_next_txt()
    # As long as there is data and the user does not stop, data will be visualized.
    while data is not None:
        plot_df(data)
        data = txt_reader.read_next_txt()


if __name__ == "__main__":

    # Check if the path to the configuration file was passed as an argument.
    assert len(sys.argv) == 2

    # Read in the configuration file.
    config_file_path = sys.argv[1]
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Visualize data.
    visualize_data(config)
