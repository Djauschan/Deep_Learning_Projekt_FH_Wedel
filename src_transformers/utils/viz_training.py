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
