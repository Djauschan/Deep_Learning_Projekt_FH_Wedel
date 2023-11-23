import plotly.graph_objects as go
import torch

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
