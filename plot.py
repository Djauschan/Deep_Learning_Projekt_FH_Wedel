import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
import os
from dataProcessing import lookup_symbol
from config import config
import pandas as pd

# If there is no output directory yet, one will be created.
if not os.path.exists("./output"):
    os.makedirs("./output")


def plot_df(df: pd.DataFrame):
    """
    Creates a plot for the relevant columns of the passed data frame.

    Args:
        df (pd.DataFrame): Data frame for which a plot is to be created.
    """

    # The type of the ticker symbol is read out.
    df_type = df["type"][1]
    # Then it is removed from the data frame, since it is the same for all lines.
    df = df.drop("type", axis=1)

    # The ticker symbol is read out.
    symbol = df["symbol"][1]
    # Then it is removed from the data frame, since it is the same for all lines.
    df = df.drop("symbol", axis=1)

    # Determine the number of rows and columns in the plot needed to visualize the desired columns.
    count: int = len(df.columns)-1
    rows: int = 2
    colums: int = math.ceil(count / 2)

    # Create plot
    fig, axes = plt.subplots(rows, colums, figsize=(
        colums*6, rows*4))

    # Determine the title of the plot.
    # If possible, the name of the ETF is used as the title.
    etf_name: str = lookup_symbol(symbol, df_type)
    title: str = ""
    if (etf_name is not None):
        title = etf_name + " [" + symbol + "]" + " {" + df_type + "}"
    else:
        title = "Symbol " + symbol + " {" + df_type + "}"
    fig.suptitle(title)

    # The desired columns of the data frame are visualized in the form of subplots.
    i = 0
    for i in range(count):
        row = i // colums
        col = i % colums
        axes[row, col].plot(df.iloc[:, 0], df.iloc[:, i+1])
        axes[row, col].set_title(df.columns[i+1])

        # Display only years on the X axis.
        axes[row, col].xaxis.set_major_locator(mdates.YearLocator())
        axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Rotate the tick labels for better readability
        plt.setp(axes[row, col].get_xticklabels(), rotation=45)

    # The subplots that are not used are hidden.
    for z in range(i+1, rows*colums):
        row = z // colums
        col = z % colums
        axes[row, col].set_axis_off()

    # Add some space between subplots for readability
    plt.tight_layout()

    # The created plot is saved to a file.
    title = title.replace("/", "_")  # Replace "/" with "_"
    path = "./output/" + title + ".png"
    plt.savefig(path, bbox_inches='tight', dpi=400)
    print(title + ".png" + " saved")

    if config["SHOW_PLOT"]:
        plt.show()

    # Close plot to reduce memory usage
    plt.close()
