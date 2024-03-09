import numpy as np
import pandas as pd

class QLearningPredictor():
    """QLearningPredictor class for the prediction API. This class is used to predict trading actions.
    """
    def __init__(self, model_path : str) -> None:
        """Initializes the QLearningPredictor by loading the model from the given path.

        Args:
            model_path (str): The path to the model to load.
        """
        if 'ma' in model_path.lower():
            self.model = MAPredictor(model_path)
        elif 'rsi' in model_path.lower():
            self.model = RSIPredictor()
        elif 'aggregation' in model_path.lower():
            self.model = EnsemblePredictor(model_path)
        else:
            self.model = MLModelPredictor(model_path)
        try:
            self.name = self.model.name
        except AttributeError:
            self.name = 'unsupported'
    
    def predict(self, data : object) -> int:
        """Predicts trading actions for the given data.

        Args:
            data (object): The data to predict trading actions for. Can be a pd.DataFrame, list[int] or an dict depending on the underlying model.

        Returns:
            int: The predicted trading action.
        """
        return self.model.predict(data)
    
    def get_api_url(self) -> str:
        """Returns the API URL for the underlying model.

        Returns:
            str: The API URL for the underlying model.
        """
        try:
            return self.model.get_api_url()
        except AttributeError:
            return "unsupported"

class MAPredictor():
    """MAPredictor class for the prediction API. This class is used to predict trading actions based on the moving average.
    """
    def __init__(self, model_path : str) -> None:
        """Initializes the MAPredictor by loading the model from the given path.

        Args:
            model_path (str): The path to the model to load.
        """
        self.ma_variant = int(model_path.split('/')[-1].split('_')[1].replace('ma', ''))
        self.q_table = np.load(model_path)
        self.name = f'Q_learning_MA{self.ma_variant}'
    
    def _calculate_MA(self, data : pd.DataFrame) -> float:
        """Calculates the moving average for the given data.

        Args:
            data (pd.DataFrame): The data to calculate the moving average for.

        Returns:
            float: The calculated moving average.
        """
        if len(data) < self.ma_variant:
            # Not enough data for the complete MA
            return data.mean()
        else:
            # Calcuate the MA for the last X days. The Variant specifies the number of days
            return data.iloc[-self.ma_variant:].mean()
    
    def _calculate_state(self, current_price : np.array, ma_value : float) -> int:
        """Calculates the state for the given current price and moving average value.

        Args:
            current_price (np.array): The last 200 closing prices to calculate the state for.
            ma_value (float): The moving average value to calculate the state for.

        Returns:
            int: The calculated state.
        """
        deviation = (current_price - ma_value) / ma_value * 100
        max_deviation = 2 * deviation.std()
        scaled_deviation = ((deviation + max_deviation) / (2 * max_deviation))[-1]
        state = max(0, min(int(scaled_deviation * 20), 20 - 1)) # n_bins = 20
        return state
    
    def predict(self, data : pd.DataFrame) -> int:
        """Predicts trading actions for the given data.

        Args:
            data (pd.DataFrame): The data to predict trading actions for.

        Returns:
            int: The predicted trading action.
        """
        current_price = data.iloc[-221:-1].Close
        state_index = self._calculate_state(np.array(current_price), self._calculate_MA(data.Close))
        return np.argmax(self.q_table[state_index])

class RSIPredictor():
    """RSIPredictor class for the prediction API. This class is used to predict trading actions based on the RSI.
    """
    def __init__(self) -> None:
        """Initializes the RSIPredictor by loading the model from the given path.
        """
        self.name = 'RSI_decision'
        self.low_threshold = 30
        self.high_threshold = 70
    
    def _calculate_RSI(self, data : str, period : int = 14) -> float:
        """Calculates the RSI for the given data with a given periodlength.

        Args:
            data (str): The data to calculate the RSI for.
            period (int, optional): Periodlength which is used to calculate the RSI. Defaults to 14.

        Returns:
            float: The calculated RSI.
        """
        if len(data) < period:
            return 50  # Neutraler RSI-Wert bei unzureichenden Daten

        # Calculate the difference between the closing prices
        delta = data.Close.diff()
        
        # Calculate the gain and loss for the given period
        gain = delta.clip(lower=0).iloc[-period:].mean()
        loss = -delta.clip(upper=0).iloc[-period:].mean()

        # Avoid division by zero
        if loss == 0:
            return 100 if gain > 0 else 50

        # Calculate the RSI
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        return RSI
    
    def predict(self, data : pd.DataFrame) -> int:
        """Predicts trading actions for the given data.

        Args:
            data (pd.DataFrame): The data to predict trading actions for.

        Returns:
            int: The predicted trading action.
        """
        rsi_value = self._calculate_RSI(data)
        if rsi_value < self.low_threshold:
            return 1  # buy
        elif rsi_value > self.high_threshold:
            return 2  # sell
        else:
            return 0  # hold

class EnsemblePredictor():
    """ensemblePredictor class for the prediction API. This class is used to predict trading actions based on the ensemble of all models.
    """
    def __init__(self, model_path : str) -> None:
        """Initializes the ensemblePredictor by loading the model from the given path.

        Args:
            model_path (str): The path to the model to load.
        """
        self.q_table = np.load(model_path)
        self.name = 'ensemble'
        
    def predict(self, actions : list[int]) -> int:
        """Predicts trading actions for the given actions of the other agents.

        Args:
            actions (list[int]): The actions of the other agents to predict trading actions for.

        Returns:
            int: The predicted trading action.
        """
        weighted_actions = np.zeros(3)  # Angenommen, es gibt 3 mögliche Aktionen (kaufen, nichts tun, und verkaufen)
    
        for i, action in enumerate(actions):
            weighted_actions[action] += self.q_table[i, action]
        # Return a random action if all actions have the same weight
        if np.all(weighted_actions == weighted_actions[0]):
            return np.random.choice(len(weighted_actions))
        return np.argmax(weighted_actions)

class MLModelPredictor():
    """MLModelPredictor class for the prediction API. This class is used to predict trading actions based on ML Model predictions.
    """
    def __init__(self, model_path : str) -> None:
        """Initializes the MLModelPredictor by loading the model from the given path.

        Args:
            model_path (str): The path to the model to load.
        """
        # Check if the model is supported
        if 'rf_agent' in model_path.lower():
            self.model = np.load(model_path)
            self.name = "Q_learning_RandomForest-ML"
            self.api_url = "http://backend:8000/predict/randomForest"
        elif 'gbm_agent' in model_path.lower():
            self.model = np.load(model_path)
            self.name = "Q_learning_GradientBoostingTree-ML"
            self.api_url = "http://backend:8000/predict/gradientBoost"
        elif 'trans_agent' in model_path.lower():
            self.model = np.load(model_path)
            self.name = "Q_learning_Transformer-ML"
            self.api_url = "http://backend:8000/predict/transformer"
        else:
            self.model = None
            self.name = "unsupported"
        
    def predict(self, data : list[int]) -> int:
        """Predicts trading actions for the given data.

        Args:
            data (list[int]): The data to predict trading actions for. The first element is the last price, the second element is the prediction for the next price.

        Returns:
            int: The predicted trading action.
        """
        last_price = data[0]
        prediction = data[1]
        if prediction == None:
            return 0  # Hold because no prediction is available
        if prediction > last_price:
            return 1  # buy if the prediction for t+1 is higher than the price at t
        elif prediction < last_price:
            return 2  # sell if the prediction for t+1 is lower than the price at t
        return 0 
    
    def get_api_url(self) -> str:
        return self.api_url