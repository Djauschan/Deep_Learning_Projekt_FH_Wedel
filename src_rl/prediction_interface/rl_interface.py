from prediction_interface.q_learning_predictor import QLearningPredictor

from glob import glob
import pandas as pd

class RLInterface:
    def __init__(self) -> None:
        models = [QLearningPredictor(model) for model in glob('prediction_interface/api_models/ma*.npy')]
        self.models = {model.name : model for model in models}
        
        self.data = self.read_data()
        
        
    def predict(self, stock_symbol, start_date, end_date) -> dict:
        prediction = {}
        
        for model_name, model in self.models.items():
            current_date = start_date
            while current_date <= end_date:
                data = self.data[stock_symbol.upper()].loc[:current_date]
                prediction[model_name] = model.predict(data)
                current_date = current_date + pd.Timedelta('1h')
        
        return prediction
    
    def read_data(self) -> dict:
        dataDict = {}
        for filepath in glob('../data/aktien/*.txt'):
            df = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True, names=["DateTime", "Open", "Close", "High", "Low", "_"]).drop("_", axis=1)
            df = df[df.index.minute == 0]
            stock_symbol = filepath.split('/')[-1].split('_')[0]
            dataDict[stock_symbol] = df
        
        return dataDict