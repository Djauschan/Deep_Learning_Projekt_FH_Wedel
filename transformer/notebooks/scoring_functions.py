import numpy as np
import pickle
import torch

def total_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean absolute error of the prediction.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))

def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean error of the prediction.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean error.
    """
    return np.mean(y_true - y_pred)

def mae_by_feature(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the mean absolute error for each feature.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Mean absolute error for each feature.
    """
    return np.mean(np.mean(np.abs(y_true - y_pred), axis=0), axis=0)

def mean_error_by_feature(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the mean error for each feature.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Mean absolute error for each feature.
    """
    return np.mean(np.mean((y_true - y_pred), axis=0), axis=0)

def mae_by_timestep(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the mean absolute error for each timestep.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Mean absolute error for each timestep.
    """
    return np.mean(np.mean(np.abs(y_true - y_pred), axis=0), axis=1)

def mean_error_by_timestep(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the mean error for each timestep.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Mean absolute error for each timestep.
    """
    return np.mean(np.mean((y_true - y_pred), axis=0), axis=1)

def mae_by_feature_timestep(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the mean absolute error for each feature and timestep.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Mean absolute error for each feature and timestep.
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)

def mean_error_by_feature_timestep(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the mean error for each feature and timestep.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Mean absolute error for each feature and timestep.
    """
    return np.mean((y_true - y_pred), axis=0)

class portfolio:
    def __init__(self, n_stocks: np.ndarray):
        self.stocks = np.zeros(n_stocks)
        self.invested = 0.0
        self.cash_out = 0.0
        self.value = 0.0
        self.profit = 0.0

    def buy(self, stock: int, amount: int, price: float):
        self.stocks[stock] += amount
        self.invested += amount * price

    def sell(self, stock: int, price: float):
        self.cash_out += self.stocks[stock] * price
        self.stocks[stock] = 0

    def get_result(self, prices: np.ndarray):
        self.value = np.sum(self.stocks * prices)
        self.profit = self.value + self.cash_out - self.invested
        return self.profit

def calc_profit(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the profit for an investment based on the predictions for the whole dataset.

    Args:
        y_true (np.ndarray): True values (absolute).
        y_pred (np.ndarray): Predicted values (absolute).

    Returns:
        np.ndarray: Profit.
    """
    porfolio = portfolio(y_true.shape[2])

    for sample in range(1, y_true.shape[0]):
        current_prices = y_true[sample-1][0]
        #absoute_predictions = np.cumprod(1 + y_pred[sample], axis=0)

        max_prediction = np.max(y_pred, axis=(0,1))
        min_prediction = np.min(y_pred, axis=(0,1))

        for stock in range(y_true.shape[2]):

            # Buy if all predictions are higher than the current price
            if min_prediction[stock] > current_prices[stock]:
                porfolio.buy(stock, 5, current_prices[stock])

            # Sell if all predictions are lower than the current price
            elif max_prediction[stock] < current_prices[stock]:
                porfolio.sell(stock, current_prices[stock])

    profit = porfolio.get_result(current_prices)

    return profit




if __name__ == "__main__":
    with open("target_data.pkl", "rb") as f:
        y_true = pickle.load(f)
    with open("prediction.pkl", "rb") as f:
        y_pred = pickle.load(f)

    with open("rl_pred.pkl", "rb") as f:
        y_rl = pickle.load(f)

    # Generate artificial target date from y_rl
    y_true_rl = np.zeros((y_rl.shape[0], y_rl.shape[1], y_rl.shape[2]))
    for t in range(y_rl.shape[0]):
        for s in range(y_rl.shape[1]):
            y_true_rl[t, s, :] = y_rl[t, s, :]*np.random.uniform(0.9, 1.1, y_rl.shape[2])

    # Convert tensor to numpy
    #y_true = y_true.numpy()
    #y_pred = y_pred.numpy()

    calc_profit(y_true_rl, y_rl)

    print(total_mae(y_true, y_pred))

    print(mae_by_feature(y_true, y_pred))

    print(mae_by_timestep(y_true, y_pred))