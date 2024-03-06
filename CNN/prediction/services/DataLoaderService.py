import pandas as pd


class DataLoaderService:
    """
    class to load data from file and store it in dataframe
    """

    def __init__(self):
        pass

    def loadDataFromFile(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        rsc_completePath: str,
        ALL_DATA_COLUMNS: list,
        COLUMNS_TO_KEEP: list,
    ) -> pd.DataFrame:
        df = pd.read_csv(
            rsc_completePath, sep=",", names=ALL_DATA_COLUMNS, index_col=False
        )
        toRemove = []
        for col in df:
            if col not in COLUMNS_TO_KEEP:
                toRemove.append(col)

        data = df.drop(toRemove, axis=1)
        data["DateTime"] = pd.to_datetime(data["DateTime"])
        data = data[(data["DateTime"] >= start_date) & (data["DateTime"] <= end_date)]
        return data