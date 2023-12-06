import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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

    for feature_idx in range(n_target_features):
        color_idx = (feature_idx * 2) % 20
        axes[feature_idx].plot(targets[:, 0, feature_idx],
                               c=colors[color_idx], label='target')
        # ax.plot(targets[:, 1, feature_idx], c=colors[color_idx])
        axes[feature_idx].plot(
            predictions[:, 0, feature_idx], c=colors[color_idx + 1], label='prediction')
        # ax.plot(predictions[:, 1, feature_idx], c=colors[color_idx+1])
        axes[feature_idx].legend()

    return fig
