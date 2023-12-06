import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
import os
from src_transformers.preprocessing.data_processing import lookup_symbol
import pandas as pd

# The path of the current file is determined.
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'data' directory located two levels up from
# the current file's directory
data_dir_path = os.path.join(
    current_file_directory, os.pardir, os.pardir, 'data')

# Construct the path to the 'output' directory located in the 'data' directory
output_dir_path = os.path.join(data_dir_path, 'output')


def plot_df(df: pd.DataFrame) -> None:
    """
    Creates a plot for the relevant columns of the passed data frame.

    Args:
        df (pd.DataFrame): Data frame for which a plot is to be created.
    """

    # The type of the ticker symbol is read out.
    df_type = df["type"][1]
    # Then it is removed from the data frame, since it is the same for all
    # lines.
    df = df.drop("type", axis=1)

    # The ticker symbol is read out.
    symbol = df["symbol"][1]
    # Then it is removed from the data frame, since it is the same for all
    # lines.
    df = df.drop("symbol", axis=1)

    # Determine the number of rows and columns in the plot needed to visualize
    # the desired columns.
    count: int = len(df.columns) - 1
    rows: int = 2
    columns: int = math.ceil(count / 2)

    # Create plot
    fig, axes = plt.subplots(rows, columns, figsize=(
        columns * 6, rows * 4))

    # Determine the title of the plot.
    # If possible, the name of the ETF is used as the title.
    symbol_name: str = lookup_symbol(symbol, df_type)

    if symbol_name is not None:
        title = symbol_name + " [" + symbol + "]" + " {" + df_type + "}"
    else:
        title = "Symbol " + symbol + " {" + df_type + "}"
    fig.suptitle(title)

    # The desired columns of the data frame are visualized in the form of
    # subplots.
    i = 0
    for i in range(count):
        row = i // columns
        col = i % columns
        axes[row, col].plot(df.iloc[:, 0], df.iloc[:, i + 1])
        axes[row, col].set_title(df.columns[i + 1])

        # Display only years on the X axis.
        axes[row, col].xaxis.set_major_locator(mdates.YearLocator())
        axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Rotate the tick labels for better readability
        plt.setp(axes[row, col].get_xticklabels(), rotation=45)

    # The subplots that are not used are hidden.
    for z in range(i + 1, rows * columns):
        row = z // columns
        col = z % columns
        axes[row, col].set_axis_off()

    # Add some space between subplots for readability
    plt.tight_layout()

    # The created plot is saved to a file.
    title = title.replace("/", "_")  # Replace "/" with "_"
    path = os.path.join(output_dir_path, title + ".png")
    plt.savefig(path, bbox_inches='tight', dpi=400)
    print(title + ".png" + " saved")

    # Close plot to reduce memory usage
    plt.close()
