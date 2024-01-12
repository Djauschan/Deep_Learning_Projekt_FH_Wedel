import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('Agg')
import numpy as np


def plot_evaluation(targets: np.array, predictions: np.array):
    """
    Plots the predictions and targets for the evaluation.

    Args:
        targets (np.array): targets
        predictions (np.array): predictions
    """

    n_target_features = targets.shape[2]

    fig, axes = plt.subplots(
        ncols=1, nrows=n_target_features, figsize=(20, n_target_features * 2))

    colors = cm.tab20(range(20))

    if n_target_features == 1:
        axes = [axes]

    for feature_idx in range(n_target_features):
        color_idx = (feature_idx * 2) % 20
        axes[feature_idx].plot(targets[:, 0, feature_idx],
                               c=colors[color_idx], label='target')
        axes[feature_idx].plot(
            predictions[:, 0, feature_idx], c=colors[color_idx + 1], label='prediction')
        axes[feature_idx].legend()

    return fig


def plot_loss_horizon(targets: np.array, predictions: np.array):
    """
    Plots the predictions and targets for the evaluation.

    Args:
        targets (np.array): targets
        predictions (np.array): predictions
    """

    n_target_features = targets.shape[2]

    err = np.abs(targets - predictions)
    mean_err = np.mean(err, axis=0)

    fig, axes = plt.subplots(ncols=1,
                             nrows=n_target_features,
                             figsize=(20, n_target_features * 2))

    colors = cm.tab20(range(20))

    if n_target_features == 1:
        axes = [axes]

    for feature_idx in range(n_target_features):
        color_idx = (feature_idx * 2) % 20
        for i in range(err.shape[0]):
            axes[feature_idx].plot(err[i, :, feature_idx],
                                   c=colors[color_idx + 1],
                                   alpha=0.2)
        axes[feature_idx].plot(mean_err[:, feature_idx],
                               c='red',
                               label='mean error')
        axes[feature_idx].legend(loc='upper right')

    plt.xlabel('Prediction horizon')

    return fig

def plot_absolute_predictions(targets: np.array, predictions: np.array):
    """
    Plots the predictions and targets of relative change as absolute values. The plots start
    at 100. Then for each time step the target and prediction of relative change is applied to
    the predious value.

    Args:
        targets: Targets of relative change.
        predictions: Predictions of relative change.

    Returns:

    """

    n_target_features = targets.shape[2]
    prediction_horizon = targets.shape[1]-1
    half_prediction_horizon = prediction_horizon // 2

    fig, axes = plt.subplots(
        ncols=1, nrows=n_target_features, figsize=(20, n_target_features * 2.5))

    colors = cm.tab20c(range(20))

    if n_target_features == 1:
        axes = [axes]

    for feature_idx in range(n_target_features):
        color_idx = (feature_idx * 4) % 20
        axes[feature_idx].plot(
            np.cumprod(predictions[:, 0, feature_idx] + 1) * 100, c=colors[color_idx], label='prediction 1 step')
        axes[feature_idx].plot(
            np.cumprod(predictions[:, half_prediction_horizon, feature_idx] + 1) * 100, c=colors[color_idx + 1], label=f'prediction {half_prediction_horizon} step')
        axes[feature_idx].plot(
            np.cumprod(predictions[:, prediction_horizon, feature_idx] + 1) * 100, c=colors[color_idx + 2], label=f'prediction {prediction_horizon} step')
        axes[feature_idx].plot(np.cumprod(targets[:, 0, feature_idx] + 1) * 100,
                               c='black', label='target')
        axes[feature_idx].legend()

    return fig
