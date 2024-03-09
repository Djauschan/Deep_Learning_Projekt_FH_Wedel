from prediction_interface.q_learning_predictor import QLearningPredictor
from prediction_interface.abstract_model import resolution as resolution_type

from glob import glob
import pandas as pd
import requests

class RLInterface:
    """RL Interface for the prediction API. This class is used to predict trading actions.
    """
    def __init__(self) -> None:
        """Initializes the RLInterface by loading all available models.
        """
        models = [QLearningPredictor(model) for model in glob('prediction_interface/api_models/*.npy')]
        self.models = {model.name : model for model in models if model.name != 'unsupported'}
        
        if 'ensemble' in self.models.keys():
            self.ensemble_mode = True
        else:
            self.ensemble_mode = False
        
        self.action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        self.data = self._read_data()
        
        
    def predict(self, stock_symbol : str, start_date : str , end_date : str, resolution: resolution_type) -> dict[str, dict[str, str]]:        
        """Predicts trading actions for a given stock symbol and time frame with every avialible model.

        Args:
            stock_symbol (str): The stock symbol to predict trading actions for.
            start_date (str): The start date of the time frame to predict trading actions for.
            end_date (str): The end date of the time frame to predict trading actions for.

        Returns:
            dict[pd.Timestamp, dict[str, str]]: A dictionary containing the predictions for every model for every hour in the given time frame.
        """
        predictions = {}
        relevant_data = self.data[stock_symbol.upper()]
        if resolution == resolution_type.DAILY:
            time_range = pd.date_range(f'{start_date} 20:00:00', f'{end_date} 20:00:00', freq='D')
            resolution = 'D'
        elif resolution == resolution_type.TWO_HOURLY:
            time_range = pd.date_range(start_date, end_date, freq='2h')
            resolution = 'H'
        elif resolution == resolution_type.MINUTE:
            time_range = pd.date_range(start_date, end_date, freq='min')
            resolution = 'M'
        else:
            raise ValueError(f'Resolution {resolution} not supported')
        last_timestamp = None
        for current_date in time_range:
            if current_date not in relevant_data.index and resolution_type(resolution) != resolution_type.DAILY:
                continue
            if last_timestamp is not None:
                data = relevant_data.loc[relevant_data.index <= current_date]
                ensemble_input = []
                predictions[current_date] = {}
                for model_name, model in self.models.items():
                    if model_name != 'ensemble' and 'ML' not in model_name:
                        current_prediction = model.predict(data)
                        ensemble_input.append(current_prediction)
                        predictions[current_date][model_name] = self.action_map[current_prediction]
                    elif 'ML' in model_name:
                        model_prediction = self._get_prediction_from_dict_list(
                            self._get_data_from_api(model.get_api_url(), stock_symbol, current_date, resolution)[stock_symbol.upper()],
                            current_date)
                        data_to_send = [data.Close[-1], model_prediction]
                        current_prediction = model.predict(data_to_send)
                        ensemble_input.append(current_prediction)
                        predictions[current_date][model_name.split('-')[0]] = self.action_map[current_prediction]
            
                if self.ensemble_mode:
                    predictions[current_date]['ensemble'] = self.action_map[self.models['ensemble'].predict(ensemble_input)]
                else:
                    lst = list(predictions[current_date].values())
                    predictions[current_date]['election'] = max(set(lst), key=lst.count)
            last_timestamp = current_date
        return predictions
    
    def _read_data(self) -> dict[str, pd.DataFrame]:
        """Reads the stock data from the data folder.

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing the stock data for every stock symbol.
        """
        dataDict = {}
        for filepath in glob('../data/aktien/*.txt'):
            df = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True, names=["DateTime", "Open", "Close", "High", "Low", "_"]).drop("_", axis=1)
            stock_symbol = filepath.split('/')[-1].split('_')[0]
            dataDict[stock_symbol] = df
        
        return dataDict
    
    def _get_data_from_api(self, url : str, stock_symbol : str, start_date : pd.Timestamp, resolution) -> pd.DataFrame:
        """Gets the stock data from the given API.

        Args:
            url (str): The API URL to get the stock data from.
            stock_symbol (str): The stock symbol to get the data for.
            start_date (pd.Timestamp): The start date of the time frame to get the data for.

        Returns:
            pd.DataFrame: The stock data for the given stock symbol and time frame.
        """
        response = requests.get(url, params={'stock_symbols': f'[{stock_symbol.upper()}]',
                                             'start_date': start_date.strftime('%Y-%m-%d'),
                                             'resolution': resolution})
        return response.json()
    
    def _get_prediction_from_dict_list(self, dict_list : list[dict[str : str]], desired_date : pd.Timestamp) -> float:
        """Gets the value for the given date from the given dictionary list.

        Args:
            dict_list (list): The dictionary list to get the value from.
            desired_date (pd.Timestamp): The date to get the value for.

        Returns:
            float: The desired prediction value.
        """
        try:
            return pd.DataFrame(dict_list, index=[pd.Timestamp(dictionary['date']) for dictionary in dict_list]).loc[desired_date].value
        except KeyError:
            return None