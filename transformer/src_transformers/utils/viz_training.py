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
        ncols=1, nrows=n_target_features, figsize=(18, n_target_features * 2.5))

    colors = cm.tab10(range(4))

    if n_target_features == 1:
        axes = [axes]

    abs_target = np.ones((len(targets) + 1, targets.shape[2])) * 100

    abs_pred_step1 = np.ones((len(predictions) + 1, predictions.shape[2])) * 100
    abs_pred_step2 = np.ones((len(predictions) + 1, predictions.shape[2])) * 100
    abs_pred_step3 = np.ones((len(predictions) + 1, predictions.shape[2])) * 100
    abs_pred_changes = np.ones_like(predictions) * 100


    for i in range(len(targets)):
        abs_target[i + 1] = abs_target[i] * (1 + targets[i, 0])
        abs_pred_step1[i + 1] = abs_target[i] * (1 + predictions[i, 0])
        for j in range(prediction_horizon):
            if i <= j:
                abs_pred_changes[i, j+1] = abs_pred_changes[i, j] * (1 + predictions[i, j])
        if i <= len(targets) - half_prediction_horizon:
            abs_pred_step2[i+half_prediction_horizon] = abs_target[i,0] * abs_pred_changes[i, half_prediction_horizon] / 100
        if i <= len(targets) - prediction_horizon:
            abs_pred_step3[i+prediction_horizon] = abs_target[i,0] * abs_pred_changes[i, prediction_horizon] / 100


    for feature_idx in range(n_target_features):
        y_ax_min = abs_target[:, feature_idx].min()
        y_ax_max = abs_target[:, feature_idx].max()
        y_ax_range = y_ax_max - y_ax_min
        y_ax_min -= y_ax_range * 0.1
        y_ax_max += y_ax_range * 0.1

        color_idx = feature_idx % 4
        axes[feature_idx].plot(
            abs_pred_step1[:, feature_idx], c="#33608D", label='prediction 1 step')
        axes[feature_idx].plot(
            abs_pred_step2[:, feature_idx], c="#26AC81", label=f'prediction {half_prediction_horizon} step')
        axes[feature_idx].plot(
            abs_pred_step3[:, feature_idx], c="#95D73F", label=f'prediction {prediction_horizon} step')
        axes[feature_idx].plot(abs_target[:, feature_idx],
                               c="#471567", label='target')
        axes[feature_idx].legend()
        axes[feature_idx].set_ylim(y_ax_min, y_ax_max)

    return fig
