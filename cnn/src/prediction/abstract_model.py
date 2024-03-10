from abc import ABC, abstractmethod
import pandas as pd
import typing
from enum import Enum


class resolution(Enum):
    """Enum for the resolution of the stock data.
    """
    DAILY = "D"
    TWO_HOURLY = "H"
    MINUTE = "M"


class AbstractModel(ABC):
    """This is the absctract class for all models. It defines the methods that should be implemented in the child classes.
    You can define more methods if you need them but these should not be called from the outside (backend).

    Please give feedback if you think that you need more abstract methods.
    If you have useful methods that will be usefull for all child classes, please talk to me and we can
    add them as non abstract methods here.

    """

    @abstractmethod
    def predict(self, symbol_list: list, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp,
                resolution: resolution) -> pd.DataFrame:
        """predicts the stock prices for the given symbols and time range.

        Args:
            symbol_list (list): The list of symbols for which the stock prices should be predicted.
            timestamp_start (pd.Timestamp): The start of the time range for which the stock prices should be predicted.
            timestamp_end (pd.Timestamp): The end of the time range for which the stock prices should be predicted.
            resolution (resolution): The resolution of the stock data.

        Returns:
            pd.DataFrame: The predicted stock prices.
        """
        pass

    @abstractmethod
    def load_data(self) -> None:
        """load data from database and stores it in a class variable

        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """preprocess data and stores it in a class variable

        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """load model from file and stores it in a class variable

        """
        pass

    @staticmethod
    def calculate_absolute_prices(prices: typing.Iterable, start_price: float) -> list:
        """
        Calculates the absolut prices from the relative prices.
        The start price is the price before the first relative price is applied.

        Args:
            prices (typing.Iterable): Iterable of relative prices.
            start_price (float): Absolute start price before the first relative price is applied.

        Returns:
            list: List of absolute prices.
        """
        for i, price_change in enumerate(prices):
            if i == 0:
                prices[i] = start_price * (1 + price_change)
            else:
                prices[i] = prices[i - 1] * (1 + price_change)
        return prices
