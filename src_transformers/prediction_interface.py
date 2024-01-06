import pandas as pd
import yaml
from src_transformers.abstract_model import AbstractModel
import subprocess

from pathlib import Path
from typing import Final, Optional

DEFAULT_PATH: Final[Path] = Path("data", "test_configs", "training_config_tt_predict.yaml")


class prediction_interface(AbstractModel):
    """This is the abstract class for all models. It defines the methods that should be implemented in the child classes.
    You can define more methods if you need them but these should not be called from the outside (backend).

    Please give feedback if you think that you need more abstract methods.
    If you have useful methods that will be useful for all child classes, please talk to me, and we can add them as non abstract methods here.

    """

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int = 0, file_path=DEFAULT_PATH) -> pd.DataFrame:
        """predict stock price for a given time interval

        Args:
            timestamp_start (pd.Timestamp): start time of the time period
            timestamp_end (pd.Timestamp): end time of the time period
            interval (int, optional): interval in minutes. Defaults to 0.
            file_path (str, optional): path to the config file. Defaults to DEFAULT_PATH.

        Returns:
            pd.DataFrame: dataframe with columns: timestamp, 1-n prices of stock_symbols
        """
        # Write timestamp_start and timestamp_end to config file
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        # Set the first and last date in the config file for the main
        config['dataset_parameters']['first_date'] = timestamp_start.strftime("%Y-%m-%d")
        config['dataset_parameters']['last_date'] = timestamp_end.strftime("%Y-%m-%d")

        # Save adjusted config file
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

        subprocess.run(["python", "-m", "src_transformers.main", "-c", file_path, "-p", "predict"])

        return None

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


if __name__ == "__main__":
    prediction_interface().predict(pd.to_datetime('2021-01-04'), pd.to_datetime('2021-02-01'))
