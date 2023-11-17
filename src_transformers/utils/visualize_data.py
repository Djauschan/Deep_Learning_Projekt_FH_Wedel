import yaml
import sys
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.plot import plot_df

# Code to visualize the data.


def vizualize_data(config: dict) -> None:
    """
    Visualizes the files and saves the result in the directory data/output

    Args:
        config (dict): Dictionary for the configuration of data preprocessing.
    """

    # Read in data.
    txt_reader = DataReader(config)
    data = txt_reader.read_next_txt()
    choice = ""
    # As long as there is data and the user does not stop, data will be visualized.
    while (data is not None) and (choice != "e"):
        if config["DEBUG_OUTPUT"]:
            print(data)
        plot_df(data, config)
        # If all data is to be visualized, the user does not have to confirm.
        if not config["VISUALIZE_ALL_DATA"]:
            choice = input("exit (e) or continue (other key)?").strip().lower()
        if choice != "e":
            data = txt_reader.read_next_txt()


if __name__ == "__main__":

    # Check if the path to the configuration file was passed as an argument.
    assert len(sys.argv) == 2

    # Read in the configuration file.
    config_file_path = sys.argv[1]
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Visualize data.
    vizualize_data(config)
