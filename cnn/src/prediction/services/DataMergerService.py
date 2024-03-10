import pandas as pd


class DataMergerService:
    def __init__(self):
        pass

    @staticmethod
    def mergeFeatureData(
        main_df: pd.DataFrame, df_toMerge: pd.DataFrame
    ) -> pd.DataFrame:
        """
        merge the dataframe in the list
        """
        return main_df.merge(
            df_toMerge, how="inner", left_on=["posixMinute"], right_on=["posixMinute"]
        )