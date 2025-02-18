import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

# Enable anomaly detection to detect invalid gradients
torch.autograd.set_detect_anomaly(True)


class TransformerModel(nn.Module):

    def __init__(self, dim_encoder: int, dim_decoder: int, num_heads: int, num_layers: int, d_ff: int,
                 seq_len_encoder: int, seq_len_decoder: int, dropout: float, device: torch.device):
        """
        Transformer model for time series forecasting based on PyTorch's TransformerEncoder.

        Args:
            dim_encoder (int): Input dimension of the encoder.
            dim_decoder (int): Output dimension of the decoder.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            d_ff (int): Dimension of the feed forward layer.
            seq_len_encoder (int): Length of the input sequence.
            seq_len_decoder (int): Length of the output sequence.
            dropout (float): Dropout probability.
            device (torch.device): Whether to use GPU or CPU.
        """
        super().__init__()

        # Throw value Error if seq_len_decoder is larger than seq_len_encoder
        if seq_len_decoder > seq_len_encoder:
            raise ValueError(
                'seq_len_decoder must be smaller or equal to seq_len_encoder')

        self.model_type = 'Transformer'
        self.seq_len_encoder = seq_len_encoder
        self.seq_len_decoder = seq_len_decoder
        self.pos_encoder = PositionalEncoding(
            dim_encoder, dropout, max(seq_len_encoder, seq_len_decoder))
        encoder_layers = TransformerEncoderLayer(
            dim_encoder, num_heads, d_ff, dropout)
        encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers)
        self.embedding = nn.Linear(dim_encoder, dim_encoder)
        self.d_model = dim_encoder
        self.linear = nn.Linear(dim_encoder, dim_decoder)
        self.device = device

        self.init_weights()

        # save constructor arguments to enable model saving/loading
        self.params = {
            'dim_encoder': dim_encoder,
            'dim_decoder': dim_decoder,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'd_ff': d_ff,
            'seq_len_encoder': seq_len_encoder,
            'seq_len_decoder': seq_len_decoder,
            'dropout': dropout,
            'device': device
        }

    def init_weights(self) -> None:
        """
        Initialize weights of the model.
        """
        initrange = 0.2
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): Input tensor of shape (seq_len, batch_size, dim_encoder).
            src_mask (torch.Tensor, optional): Mask tensor of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, dim_decoder).
        """
        src_mask = nn.Transformer.generate_square_subsequent_mask(
            src.size(1)).to(self.device)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)

        # The output sequence is artificially shortened so that it is possible
        # to have a shorter output sequence than the input sequence.
        output_sequence = output[:, :self.seq_len_decoder, :]

        return output_sequence


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int):
        """
        Positional encoding for the transformer model.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout probability.
            max_len (int): Maximum length of the input sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.squeeze())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_encoder).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_encoder).
        """
        # self.pe is sliced not only by seq_len but also by dim_encoder before it's added to x.
        # This ensures that the dimensions of self.pe match those of x.
        # The unsqueeze(0) is used to add an extra dimension to self.pe to
        # match the batch size dimension of x.
        x = x + self.pe[:x.size(1), :x.size(2)].unsqueeze(0)
        return self.dropout(x)
