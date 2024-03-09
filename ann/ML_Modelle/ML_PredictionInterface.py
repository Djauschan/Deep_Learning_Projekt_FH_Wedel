import pandas as pd
from ML_Modelle.abstract_model import AbstractModel, resolution
from ML_Modelle.ML_PredictionInterface_daily import (
    ABC_GradientBoostingModel_daily,
    ABC_LinearRegressionModel_daily,
    ABC_RandomForestModel_daily,
    ABC_SVMModel_daily,
)
from ML_Modelle.ML_PredictionInterface_hour import (
    ABC_GradientBoostingModel_hour,
    ABC_LinearRegressionModel_hour,
    ABC_RandomForestModel_hour,
    ABC_SVMModel_hour,
)
from ML_Modelle.ML_PredictionInterface_min import (
    ABC_GradientBoostingModel_min,
    ABC_LinearRegressionModel_min,
    ABC_RandomForestModel_min,
    ABC_SVMModel_min,
)


class ML_PredictionInterface_RandomForest(AbstractModel):

    def predict(self, symbol_list: list, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, resolution: resolution) -> pd.DataFrame:
        """predicts the stock prices for the given symbols and time range.

        Args:
            symbol_list (list): The list of symbols for which the stock prices should be predicted.
            timestamp_start (pd.Timestamp): The start of the time range for which the stock prices should be predicted.
            timestamp_end (pd.Timestamp): The end of the time range for which the stock prices should be predicted.
            resolution (resolution): The resolution of the stock data.

        Returns:
            pd.DataFrame: The predicted stock prices.
        """
        results = []
        if resolution == resolution.DAILY:
            chosen_interface = ABC_RandomForestModel_daily
            interval = 1
        elif resolution == resolution.TWO_HOURLY:
            chosen_interface = ABC_RandomForestModel_hour
            interval = 2
        elif resolution == resolution.MINUTE:
            chosen_interface = ABC_RandomForestModel_min
            interval = 20
        else:
            raise NotImplementedError()
        for stock_symbol in symbol_list:
            results.append(chosen_interface().predict(
                stock_symbol, timestamp_start, timestamp_end, interval))

        result_df = pd.DataFrame()

        for result in results:
            result_df = pd.concat([result_df, result], axis=1)

        return result_df.round(2)

    def load_data(self) -> None:
        """load data from database and stores it in a class variable

        """
        pass

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable

        """
        pass

    def load_model(self) -> None:
        """load model from file and stores it in a class variable

        """
        pass


class ML_PredictionInterface_GradientBoostingModel(AbstractModel):

    def predict(self, symbol_list: list, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, resolution: resolution) -> pd.DataFrame:
        """predicts the stock prices for the given symbols and time range.

        Args:
            symbol_list (list): The list of symbols for which the stock prices should be predicted.
            timestamp_start (pd.Timestamp): The start of the time range for which the stock prices should be predicted.
            timestamp_end (pd.Timestamp): The end of the time range for which the stock prices should be predicted.
            resolution (resolution): The resolution of the stock data.

        Returns:
            pd.DataFrame: The predicted stock prices.
        """
        results = []
        if resolution == resolution.DAILY:
            chosen_interface = ABC_GradientBoostingModel_daily
            interval = 1
        elif resolution == resolution.TWO_HOURLY:
            chosen_interface = ABC_GradientBoostingModel_hour
            interval = 2
        elif resolution == resolution.MINUTE:
            chosen_interface = ABC_GradientBoostingModel_min
            interval = 20
        else:
            raise NotImplementedError()
        for stock_symbol in symbol_list:
            results.append(chosen_interface().predict(
                stock_symbol, timestamp_start, timestamp_end, interval))

        result_df = pd.DataFrame()

        for result in results:
            result_df = pd.concat([result_df, result], axis=1)

        return result_df.round(2)

    def load_data(self) -> None:
        """load data from database and stores it in a class variable

        """
        pass

    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable

        """
        pass

    def load_model(self) -> None:
        """load model from file and stores it in a class variable

        """
        pass
