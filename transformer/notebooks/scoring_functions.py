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


if __name__ == "__main__":
    with open("target_data.pkl", "rb") as f:
        y_true = pickle.load(f)
    with open("prediction.pkl", "rb") as f:
        y_pred = pickle.load(f)

    # Convert tensor to numpy
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    print(total_mae(y_true, y_pred))

    print(mae_by_feature(y_true, y_pred))

    print(mae_by_timestep(y_true, y_pred))