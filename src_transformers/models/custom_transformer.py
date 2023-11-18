import math

import pandas as pd
import torch
import torch.nn as nn


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
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

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
        # Calculate attention scores (i.e. similarity scores between query and keys)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities (i.e. Attention weights)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

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
    def __init__(self, d_model:int, max_seq_length:int):
        """
        Args:
            d_model (int): The model's dimension (number of features in one timestep).
            max_seq_length (int): The maximum sequence length (number of timesteps).
        """
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix with shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # Creates postitional encoding for one timestep of the length of the max sequence
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Creates a divisor for the positional encoding along the model's dimension
        # This is to make the positional encoding's values decay along the model's dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Apply the div term to the positionoal encoding to create the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)] #TODO, Old: 1


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
        attn_output = self.self_attn(x, x, x, mask)

        # Add + normalize + dropout
        x = self.norm1(x + self.dropout(attn_output)) #TODO: Norm lieber vorher?

        # Forward feed forward layer
        ff_output = self.feed_forward(x)

        # Add + normalize + dropout
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):

        # Forward self attention layer for tgt inputs
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Forward cross attention layer for encoder outputs
        # Encoders outputs are used as keys and values
        # The decoder's outputs are used as queries
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Forward feed forward layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer_C(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        output_seq_length,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer_C, self).__init__()

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        px = pd.DataFrame(self.positional_encoding.pe[0].numpy())

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, output_seq_length)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # Generate masks for Inputs (src) and Targets (tgt)
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embed inputs and apply positional encoding
        src_embedded = self.dropout(
            self.positional_encoding(src)
        )

        # Embed target and apply positional encoding
        tgt_embedded = self.dropout(
            self.positional_encoding(tgt)
        )

        # Forward encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Forward decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


#test = Transformer(max_seq_length=200, d_model=10, num_heads=2, num_layers=3, d_ff=2048, dropout=0.1, src_vocab_size=10000, tgt_vocab_size=10000)