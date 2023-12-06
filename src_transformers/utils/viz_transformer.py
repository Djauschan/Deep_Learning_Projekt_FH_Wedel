import os

import plotly.graph_objects as go
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_attention_mask_plotly(attention_mask):
    """
    Visualizes a 3D attention mask in an interactive scatter plot using plotly.

    Parameters:
    - attention_mask: torch.Tensor, 3D tensor with True or False values.
    """
    # Make sure the input is a torch.Tensor
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # Ensure the tensor has exactly 3 dimensions
    if attention_mask.dim() != 3:
        raise ValueError("Attention mask must be a 3D tensor")

    # Get the dimensions of the tensor
    depth, rows, cols = attention_mask.shape

    # Get the coordinates of True and False values
    true_coords = torch.nonzero(attention_mask)
    false_coords = torch.nonzero(~attention_mask)

    # Create a 3D scatter plot using plotly
    fig = go.Figure()

    # Scatter plot for True values (green)
    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 2],  # Column Index
        y=true_coords[:, 1],  # Row Index
        z=true_coords[:, 0],  # Depth Index
        mode='markers',
        marker=dict(color='green', size=5),
        name='True'
    ))

    # Scatter plot for False values (red)
    fig.add_trace(go.Scatter3d(
        x=false_coords[:, 2],  # Column Index
        y=false_coords[:, 1],  # Row Index
        z=false_coords[:, 0],  # Depth Index
        mode='markers',
        marker=dict(color='red', size=5),
        name='False'
    ))

    # Set axis labels
    fig.update_layout(scene=dict(
        xaxis_title='Column Index',
        yaxis_title='Row Index',
        zaxis_title='Depth Index'
    ))

    # Set plot title
    fig.update_layout(title_text='3D Attention Mask Visualization')

    # Show the plot
    fig.show(renderer="browser")


def plot_tensor(tensor: torch.tensor, title: str = None, viz: bool = False):
    """
    Plots a 3D tensor as a 3D bar chart.

    Args:
        tensor (torch.tensor): tensor to plot
        title (str): title of the plot
        viz (bool): if True, the plot is shown in a new window
    """
    if viz:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        if tensor.requires_grad:
            data = tensor.detach()
        else:
            data = tensor
        data = data.numpy()[0]

        axes = [None, None]

        fig = plt.figure(figsize=(12, 5))
        axes[0] = fig.add_subplot(121, projection='3d')
        axes[1] = fig.add_subplot(122, projection='3d')

        # Create a meshgrid for x and y coordinates
        y, x = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

        # Flatten the tensor values for z coordinates
        z = data.flatten()

        # Create a 3D bar chart
        axes[0].bar3d(
            x.flatten(),
            y.flatten(),
            np.ones_like(z) *
            np.min(z),
            1,
            0.1,
            z,
            shade=True)
        axes[0].set_xlabel('Timeline')
        axes[0].set_ylabel('Feature')

        axes[1].bar3d(
            y.flatten(),
            x.flatten(),
            np.ones_like(z) *
            np.min(z),
            0.1,
            1,
            z,
            shade=True)
        axes[1].set_xlabel('Feature')
        axes[1].set_ylabel('Timeline')

        # Set axis labels
        for ax in axes:
            ax.set_zlabel('Value')

        fig.suptitle(title)

        # Show the plot
        plt.show()
