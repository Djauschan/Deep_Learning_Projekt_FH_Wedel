from prediction_interface.q_learning_predictor import QLearningPredictor

from glob import glob
import pandas as pd

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
        self.data = self.read_data()
        
        
    def predict(self, stock_symbol : str, start_date : str , end_date : str) -> dict[str, dict[str, str]]:        
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
        for current_date in pd.date_range(start_date, end_date, freq='h'):
            data = relevant_data.loc[:current_date]
            if current_date in data.index:
                ensemble_input = []
                predictions[current_date] = {}
                for model_name, model in self.models.items():
                    if model_name != 'ensemble':
                        current_prediction = model.predict(data)
                        ensemble_input.append(current_prediction)
                        predictions[current_date][model_name] = self.action_map[current_prediction]
            
                if self.ensemble_mode:
                    predictions[current_date]['ensemble'] = self.action_map[self.models['ensemble'].predict(ensemble_input)]
                else:
                    lst = list(predictions[current_date].values())
                    predictions[current_date]['election'] = max(set(lst), key=lst.count)
        
        return predictions
    
    def read_data(self) -> dict[str, pd.DataFrame]:
        """Reads the stock data from the data folder.

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing the stock data for every stock symbol.
        """
        dataDict = {}
        for filepath in glob('../data/aktien/*.txt'):
            df = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True, names=["DateTime", "Open", "Close", "High", "Low", "_"]).drop("_", axis=1)
            df = df[df.index.minute == 0]
            stock_symbol = filepath.split('/')[-1].split('_')[0]
            dataDict[stock_symbol] = df
        
        return dataDict