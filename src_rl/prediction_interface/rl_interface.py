from prediction_interface.q_learning_predictor import QLearningPredictor

from glob import glob
import pandas as pd

class RLInterface:
    def __init__(self) -> None:
        models = [QLearningPredictor(model) for model in glob('prediction_interface/api_models/*.npy')]
        self.models = {model.name : model for model in models if model.name != 'unsupported'}
        
        print(self.models)
        
        if 'ensemble' in self.models.keys():
            self.ensemble_mode = True
        else:
            self.ensemble_mode = False
        
        self.action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        self.data = self.read_data()
        
        
    def predict(self, stock_symbol, start_date, end_date) -> dict:
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
    
    def read_data(self) -> dict:
        dataDict = {}
        for filepath in glob('../data/aktien/*.txt'):
            df = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True, names=["DateTime", "Open", "Close", "High", "Low", "_"]).drop("_", axis=1)
            df = df[df.index.minute == 0]
            stock_symbol = filepath.split('/')[-1].split('_')[0]
            dataDict[stock_symbol] = df
        
        return dataDict