from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.preprocessing.plot import plot_df
from src_transformers.preprocessing.config import config

# Code to visualize the data.
if __name__ == "__main__":
    # Read in data.
    txt_reader = DataReader()
    data = txt_reader.read_next_txt()
    choice = ""
    # As long as there is data and the user does not stop, data will be visualized.
    while (data is not None) and (choice != "e"):
        if config["DEBUG_OUTPUT"]:
            print(data)
        plot_df(data)
        # If all data is to be visualized, the user does not have to confirm.
        if not config["VISUALIZE_ALL_DATA"]:
            choice = input("exit (e) or continue (other key)?").strip().lower()
        if choice != "e":
            data = txt_reader.read_next_txt()
