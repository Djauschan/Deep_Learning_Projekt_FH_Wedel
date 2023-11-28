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

    fig, ax = plt.subplots(figsize=(20, 10))

    colors = cm.tab20(np.linspace(0, 1, targets.shape[2]*2))

    for feature_idx in range(targets.shape[2]):
        color_idx = feature_idx*2
        ax.plot(targets[:, 0, feature_idx], c='r')
        #ax.plot(targets[:, 1, feature_idx], c=colors[color_idx])
        ax.plot(predictions[:, 0, feature_idx], c='b')
        #ax.plot(predictions[:, 1, feature_idx], c=colors[color_idx+1])

    fig.show()

    fig.close()