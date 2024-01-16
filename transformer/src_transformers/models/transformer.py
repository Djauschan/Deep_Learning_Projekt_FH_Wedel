import math

import pandas as pd
import torch
import torch.nn as nn
import numpy as np

torch.autograd.set_detect_anomaly(True)

class MultiHeadAttention(nn.Module):
    """
    This module contains one multi-head attention layer of the transformer model.
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model (int): The model's dimension (number of features in one timestep).
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number
        # of heads

        assert d_model % num_heads == 0, f"d_model (here: {d_model}) must be divisible by num_heads (here: {num_heads})"
        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = (
            d_model // num_heads
        )  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        This function calculates the Self-Attention for one head.
        """
        # Calculate attention scores (i.e. similarity scores between query and
        # keys)
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain
        # parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities (i.e. Attention
        # weights)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads,
                      self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class MultiHeadAttention_Modified(nn.Module):
    """
    This module contains one multi-head attention layer of the transformer model.
    """

    def __init__(self, dim_encoder, dim_decoder, num_heads, device):
        """
        Args:
            d_model (int): The model's dimension (number of features in one timestep).
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention_Modified, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number
        # of heads
        assert dim_encoder % num_heads == 0, f"d_model (here: {dim_encoder}) must be divisible by num_heads (here: {num_heads})"
        assert dim_decoder % num_heads == 0, f"d_model (here: {dim_decoder}) must be divisible by num_heads (here: {num_heads})"

        # Initialize dimensions
        self.dim_encoder = dim_encoder  # Encoder dimension
        self.dim_decoder = dim_decoder  # Decoder dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = (
            dim_decoder // num_heads
        )  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(dim_decoder, dim_decoder)  # Query transformation
        self.W_k = nn.Linear(dim_encoder, dim_decoder)  # Key transformation
        self.W_v = nn.Linear(dim_encoder, dim_decoder)  # Value transformation
        self.W_o = nn.Linear(dim_decoder, dim_decoder)  # Output transformation

        self.device = device

    def scaled_dot_product_attention(
            self, Q, K, V, mask=None):
        """
        This function calculates the Self-Attention for one head.
        """
        # Calculate attention scores (i.e. similarity scores between query and
        # keys)
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_scores = attn_scores.to(self.device)

        # Apply mask if provided (useful for preventing attention to certain
        # parts like padding)
        if mask is not None:
            mask = mask.to(self.device)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities (i.e. Attention
        # weights)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads,
                      self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.dim_decoder)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(
            Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    This module contains one position-wise feed forward layer with 2 layers of a
    transformer model. The activations function is ReLU. The input and output dimensions
    are the same and can be specified with init parameters. The hidden dimension is specified
    as d_ff.
    """

    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model (int): The model's dimension (number of features in one timestep).
            d_ff (int): The dimension of the feed forward layer.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
    This class produces a positional encoding for a transformer model.
    """

    def __init__(self, d_model: int, seq_length: int):
        """
        Args:
            d_model (int): The model's dimension (number of features in one timestep).
            seq_length (int): The maximum sequence length (number of timesteps).
        """
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix with shape (seq_len_encoder,
        # d_model)
        pe = torch.zeros(seq_length, d_model)

        # Creates postitional encoding for one timestep of the length of the
        # max sequence
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        # Creates a divisor for the positional encoding along the model's dimension
        # This is to make the positional encoding's values decay along the
        # model's dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -
            (math.log(100.0) / d_model)
        )

        # Apply the div term to the positionoal encoding to create the
        # positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(2)]  # Old: 1


class EncoderLayer(nn.Module):
    """
    This module contains one encoder layer of the transformer model.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Args:
            d_model (int): The model's dimension (number of features in one timestep).
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feed forward layer.
            dropout (float): The dropout rate. For regularization
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Forward attention layer
        x = self.norm1(x)
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        # Add + normalize + dropout

        # Forward feed forward layer
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_encoder, dim_decoder,
                 num_heads, d_ff, dropout, device):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(dim_decoder, num_heads)
        self.cross_attn = MultiHeadAttention_Modified(
            dim_encoder, dim_decoder, num_heads, device)
        self.feed_forward = PositionWiseFeedForward(dim_decoder, d_ff)
        self.norm1 = nn.LayerNorm(dim_decoder)
        self.norm2 = nn.LayerNorm(dim_decoder)
        self.norm3 = nn.LayerNorm(dim_decoder)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask,
                tgt_mask):
        # Forward self attention layer for tgt inputs
        x = self.norm1(x)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)

        # Forward cross attention layer for encoder outputs
        # Encoders outputs are used as keys and values
        # The decoder's outputs are used as queries
        # TODO: unmcommend the following lines
        x = self.norm2(x)
        attn_output = self.cross_attn(
            x, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_output)

        # Forward feed forward layer
        # x = self.norm3(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            dim_decoder,
            dim_encoder,
            num_heads,
            num_layers,
            d_ff,
            seq_len_encoder,
            seq_len_decoder,
            dropout,
            device: torch.device
    ):
        super(Transformer, self).__init__()

        self.device = device

        # Positional encoding
        self.positional_encoding_encoder = PositionalEncoding(
            dim_encoder, seq_len_encoder)
        self.positional_encoding_decoder = PositionalEncoding(
            dim_encoder, seq_len_decoder)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(dim_encoder, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dim_encoder, dim_decoder, num_heads, d_ff, dropout, self.device)
             for _ in range(num_layers)]
        )

        self.fc1 = nn.Linear(dim_decoder, dim_decoder)
        self.fc2 = nn.Linear(d_ff, dim_decoder)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, seq, no_peak: bool):
        """
        Generates an attention Mask for the Attention Layer The mask will be
        a square of the sequence with boolean values.
        Args:
            seq: sequence of samples that need to be masked
            no_peak: true, if the attention should be masked diagonally

        Returns: Attention Mask as torch tensor.

        """
        mask = (seq != 0).unsqueeze(1).unsqueeze(3)
        seq_length = seq.size(1)

        # generate squared tensor from sequence
        nopeak_mask = torch.ones(1, seq_length, seq_length)

        # add diagonal no peak mask if required
        if no_peak:
            nopeak_mask = 1 - torch.triu(nopeak_mask, diagonal=1)

        nopeak_mask = nopeak_mask.bool()
        nopeak_mask = nopeak_mask.to(self.device)

        # some formating for dimensionality (no clue why, just dont touch it)
        mask = mask & nopeak_mask.unsqueeze(3)
        mask = mask[:, :, :, :, 0]

        return mask.to(self.device)

    def forward(self, src, tgt):

        # Generate masks for Inputs (src) and Targets (tgt)
        src_mask = self.generate_mask(src, no_peak=False)
        tgt_mask = self.generate_mask(tgt, no_peak=True)
        dec_mask = torch.ones(tgt_mask.size(0), tgt_mask.size(
            1), tgt_mask.size(2), src_mask.size(3)).bool()

        # Create the input for the decoder
        # Targets are shifted one to the right and last entry of targets is
        # filled on idx 0
        n_tgt_feature = tgt.shape[2]
        dec_input = torch.cat(
            (src[:, -1, -n_tgt_feature:].unsqueeze(1), tgt[:, :-1, :]), dim=1)
        #dec_input = tgt #TODO: remove this line

        dec_mask.to(self.device)
        dec_input.to(self.device)

        # Embed inputs and apply positional encoding
        src_embedded = self.dropout(
            self.positional_encoding_encoder(src)
        )

        # Embed target and apply positional encoding
        tgt_embedded = self.dropout(
            self.positional_encoding_decoder(dec_input)
        )
        # src_embedded = src
        # tgt_embedded = dec_input

        # Forward encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(
                enc_output,
                mask=src_mask)

        # Forward decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(
                dec_output,
                enc_output,
                dec_mask,
                tgt_mask)

        output = self.dropout(self.fc1(dec_output))

        return output
