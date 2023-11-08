from txtReader import DataReader
from plot import plot_df
from config import config

# Code to visualize the data2.
if __name__ == "__main__":
    # Read in data2.
    txt_reader = DataReader()
    data = txt_reader.read_next_txt()
    choice = ""
    # As long as there is data2 and the user does not stop, data2 will be visualized.
    while (data is not None) and (choice != "e"):
        if config["DEBUG_OUTPUT"]:
            print(data)
        plot_df(data)
        # If all data2 is to be visualized, the user does not have to confirm.
        if not config["VISUALIZE_ALL_DATA"]:
            choice = input("exit (e) or continue (other key)?").strip().lower()
        if choice != "e":
            data = txt_reader.read_next_txt()
