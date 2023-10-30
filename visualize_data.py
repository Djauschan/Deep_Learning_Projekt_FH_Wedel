from txtReader import DataReader
from plot import plot_df
from config import DEBUG, VISUALIZE_ALL

# Code to visualize the data.
if __name__ == "__main__":
    # Read in data.
    txt_reader = DataReader()
    data = txt_reader.read_next_txt()
    choice = ""
    # As long as there is data and the user does not stop, data will be visualized.
    while (data is not None) and (choice != "e"):
        if DEBUG:
            print(data)
        plot_df(data)
        # If all data is to be visualized, the user does not have to confirm.
        if not VISUALIZE_ALL:
            choice = input("exit (e) or continue (other key)?").strip().lower()
        if choice != "e":
            data = txt_reader.read_next_txt()
