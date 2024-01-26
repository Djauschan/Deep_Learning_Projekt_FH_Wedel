import math
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

torch.autograd.set_detect_anomaly(True)


class TransformerBatchNormEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward network with batch normalization.
    """

    def __init__(self, dim_encoder: int, num_heads: int, d_ff: int, dropout: float):
        """
        TransformerEncoderLayer is made up of self-attn and feedforward network with batch normalization.

        Args:
            dim_encoder (int): Input dimension of the encoder.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout probability.
        """
        super(TransformerBatchNormEncoderLayer, self).__init__()

        # Self attention block
        self.self_attn = MultiheadAttention(
            dim_encoder, num_heads, dropout=dropout, batch_first=True)

        # MLP Block
        self.linear1 = nn.Linear(dim_encoder, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, dim_encoder)

        # Batch normalization and dropout
        self.norm1 = nn.BatchNorm1d(dim_encoder)
        self.norm2 = nn.BatchNorm1d(dim_encoder)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            src:
            src_mask:
            src_key_padding_mask:

        Returns:

        """
        # Batch norm
        src = src.permute(0, 2, 1)
        src = self.norm1(src)
        src = src.permute(0, 2, 1)

        # Self attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # Batch norm
        src = src.permute(0, 2, 1)
        src = self.norm2(src)
        src = src.permute(0, 2, 1)

        # MLP Block
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder is a stack of N encoder layers
    """

    def __init__(self, dim_encoder: int,
                 dim_decoder: int,
                 num_heads: int,
                 num_layers: int,
                 d_ff: int,
                 seq_len_encoder: int,
                 seq_len_decoder: int,
                 dropout: float,
                 device: torch.device,
                 norm: str = "layer_norm"):
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
            norm (str): Type of normalization to use. Can be "layer_norm" or "batch_norm".
        """
        super().__init__()

        # Throw value Error if seq_len_decoder is larger than seq_len_encoder
        if seq_len_decoder > seq_len_encoder:
            raise ValueError(
                'seq_len_decoder must be smaller or equal to seq_len_encoder')

        self.model_type = 'TransformerEncoder'
        self.seq_len_encoder = seq_len_encoder
        self.seq_len_decoder = seq_len_decoder
        self.norm = norm

        self.pos_encoder = PositionalEncoding(
            dim_encoder, dropout, max(seq_len_encoder, seq_len_decoder))

        if self.norm == 'layer_norm':
            encoder_layer = nn.TransformerEncoderLayer(
                dim_encoder, num_heads, d_ff, dropout, norm_first=True, batch_first=True)
        elif self.norm == 'batch_norm':
            encoder_layer = TransformerBatchNormEncoderLayer(
                dim_encoder, num_heads, d_ff, dropout)
        else:
            raise ValueError(
                'norm must be either "layer_norm" or "batch_norm".')

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

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
            'device': device,
            'norm': norm
        }

    def init_weights(self) -> None:
        """
        Initialize weights of the model.
        """
        initrange = 0.2
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.normal_(mean=0.0, std=0.2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src_mask = None  # TODO find proper solution

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(
                src.size(1)).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)

        # The output sequence is artificially shortened so that it is possible to have a shorter output sequence than the input sequence.
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
