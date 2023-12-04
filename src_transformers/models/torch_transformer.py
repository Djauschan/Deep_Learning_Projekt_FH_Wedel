import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

# Enable anomaly detection to detect invalid gradients
# torch.autograd.set_detect_anomaly(True)


def handle_nan_gradients(grad: torch.Tensor) -> torch.Tensor:
    """
    Replace NaN gradients with zero gradients.

    Args:
        grad (torch.Tensor): Gradient tensor

    Returns:
        torch.Tensor: Gradient tensor or zero tensor if NaN
    """
    if torch.isnan(grad).any():
        return torch.zeros_like(grad)
    return grad


class TransformerModel(nn.Module):

    def __init__(self, dim_encoder: int, dim_decoder: int, num_heads: int, num_layers: int, d_ff: int,
                 seq_len_encoder: int, seq_len_decoder: int, dropout: float, device: torch.device):

        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(
            dim_encoder, dropout, max(seq_len_encoder, seq_len_decoder))
        encoder_layers = TransformerEncoderLayer(
            dim_encoder, num_heads, d_ff, dropout)
        encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers)
        self.d_model = dim_encoder
        self.linear = nn.Linear(dim_encoder, dim_decoder)
        self.device = device

        # Initialize the weights of the first layer's self-attention module
        nn.init.xavier_uniform_(
            self.transformer_encoder.layers[0].self_attn.in_proj_weight)
        nn.init.constant_(
            self.transformer_encoder.layers[0].self_attn.in_proj_bias, val=0.0)

        self.init_weights()

        # Register a hook to handle NaN gradients in the first layer's self-attention module
        self.transformer_encoder.layers[0].self_attn.in_proj_weight.register_hook(
            handle_nan_gradients)
        self.transformer_encoder.layers[0].self_attn.in_proj_bias.register_hook(
            handle_nan_gradients)

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, dim_decoder]``
        """
        src_mask = None  # TODO find proper solution

        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(
                len(src)).to(self.device)
            # src_mask = zeros_tensor = torch.zeros((50, 50)).to(self.device)
        # TODO all values NaN after one epoch
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
