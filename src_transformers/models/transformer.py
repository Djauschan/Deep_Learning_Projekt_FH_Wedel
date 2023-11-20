import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        # d_model: Inputdimensionen
        # num_heads = Anzahl der Attention heads, auf die Input aufgeteilt werden soll 
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        # Prüfung: d_model durch Anzahl attention heads teilbar? 
        self.d_k = (
            d_model // num_heads
        )  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        # Transformationsgewichte definiert: Query, Key, Value, Output
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

        # Attention mit Vektoren berechnen
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # Attention scores berechnet: Produkt aus Querys und Keys 
        # Skaliert mit Wurzel der Key-Dimension d_k 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        # Masked Attention, wenn angegeben: Aus Attention Score angewendet, maskiert best. Werte
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Attention Werte durch Softmax geben -> in Wahrscheinlichkeiten transformierten die sich zu 1 addieren 
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Finaler Output: Attention weights mit Output multiplizieren
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

        # Formt Eingabe x um -> Ermöglicht Modell meheree Attention Heads gleichzeitig zu verarbeiten
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # Anschließend Ergebnisse wieder zu einem Tensor der Form batch_size, seq_length, d_model zusammenfügen
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        # Forward Funktion -> Berechnung an sich wird vorgenommen
    def forward(self, Q, K, V, mask=None):

        # lineare Transformation auf Q,K,V mit der in Intitialisierung festgelegten Gewichte
        # dann die transformierten Q,K,V mit in mehrere Heads aufteilen 
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        # Scaled dot product attention auf die Split heads anwenden
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        # Ergebnisse jedes Heads wieder zu einem Gesamt-Tensor zusammenfügen
        # und urch output linear Transformation leiten
        output = self.W_o(self.combine_heads(attn_output))
        return output

# definiert position-wise feed forward NN mit 2 linearen Layern und ReLU Funktion 
class PositionWiseFeedForward(nn.Module):

    # d_model = Dimensionen der Ein- und Ausgabe des Modells
    # d_ff = Dimensionen des inner Layser im Feed-forward netword
    # self.fc1 und 2 = Zwei fully connected linear layers mit Input und Output Dimensionen wie in d_model und d_ff definiert
    # Mit ReLu nichtlinearität zwischen den beiden linearen Schichten 
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    # x = Eingabe für das Feed Forward Netz
    # self.fc1 = Eingabe wird durch erste lineare Schicht (fc1) geleitet
    # self.relu = Ausgabe von fc1 wird durch ReLu geleitet (alle negativen Werte werden durch 0 ersetzt)
    # self.fc2 = aktivierte Ausgabe wird durch zweite lineare Schicht (fc2) geleitet und erzeugt endgültige Ausgabe
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Positionsinfo jedes Token in Inputsequenz bringen 
# dafür Sinus und Kosinusfunktionen unterschiedl. Frequenzen verwendet um Positionskodierung zu erzeugen
class PositionalEncoding(nn.Module):

    # d_model: Die Dimension der Eingabe des Modells.
    # max_seq_length: Die maximale Länge der Sequenz, für die positional encodings vorberechnet werden.
    # pe: Mit Nullen gefüllter Tensor, der mit positional encodings aufgefüllt wird.
    # position: Ein Tensor, der die Positionsindizes für jede Position in der Sequenz enthält.
    # div_term: Ein Term, der verwendet wird, um die Positionsindizes auf eine bestimmte Weise zu skalieren.
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Sinusfunktion wird auf die geraden Indizes und  Kosinusfunktion auf die ungeraden Indizes von pe angewendet.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe als Puffer registriert, d. h. er ist Teil des Modulstatus, wird aber nicht als trainierbarer Parameter betrachtet
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# Building Encoder Blocks
# Klasse definiert einen einzelnen Layer des Encoders 
class EncoderLayer(nn.Module):

    # num_heads: Anzahl der attention heads im multi-head-attention
    # d_ff: Dimensionalität d. inneren Layer im position-wise feed forward network
    # dropout: dropout rate für Regularisierung 
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        # Multihead Attention Mechanismus
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Position-wise feed-forward NN
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # norm1 und 2: Layer normalisation -> Glättung vom Layerinput
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout Layer -> random Aktivierungen auf null setzen -> Overfitting reduzieren
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, mask):
        # Input x durch multi-head self-attention geleitet
        attn_output = self.self_attn(x, x, x, mask)
        # Add & Normalize nach Attention: 
        # Attention Output zum ursprünglichen Input hinzufügen (residual connection)
        # anschließend Dropout und Normalisierung mit norm1
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-Forward Network: Output aus vorherigem Schritt durch position-wise feed-forward network leiten
        ff_output = self.feed_forward(x)
        # Add & Normalize nach Feed Forward:
        # Feed Forward Output zum Input hinzufügen (residual connection)
        # anschließend Dropout und Normalisierung mit norm2
        x = self.norm2(x + self.dropout(ff_output))
        # verarbeiteter Tensor wird als Output des Encoder Layer zurückgegeben
        return x

# Building Decoder Blocks
class DecoderLayer(nn.Module):

    # 
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # Multihead Attention wie im Encoder 
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Multi-head Attention der Output des Encoder berücksichtigt
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # Feed-Forward Network:
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        #Dropout Layer für Regularisierung
        self.dropout = nn.Dropout(dropout)

    # x: Input für Decoder Layer
    # enc_output: Ausgabe des entsprechenden Decoders (für cross-attention Schritt)
    # src_mask: Source mask -> Teile des Encoder Outputs maskieren
    # tgt_mask: Target mask -> Teile des Decorder Input ignorieren
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-Attention auf Target Sequenz: Input x durch self-attention Mechanismus verarbeitet
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # Add & Normalize nach self-attention: 
        # Attention Output zum ursprünglichen Input hinzufügen (residual connection)
        # anschließend Dropout und Normalisierung mit norm1
        x = self.norm1(x + self.dropout(attn_output))
        # Cross-attention mit Encoder Output: Normalisierter Output aus vorherigem Schritt
        # durch cross-attention Mechanismus verarbeitet, der Output enc_output des Encoders berücksichtigt
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        # Add & Normalize nach cross-attention: 
        # cross-attention Output zum Input dieser Stufe hinzugefügt
        # anschließend Dropout und Normalisierung mit norm2
        x = self.norm2(x + self.dropout(attn_output))
        # Feed-Forward network: Output aus vorherigem Schritt durch feed-forward netz geleitet
        ff_output = self.feed_forward(x)
        # Add & Normalize nach Feed-forward: Feed forward Output zum Input dieser Stufe hinzugefügt
        # anschließend Dropout und Normalisierung mit norm3
        x = self.norm3(x + self.dropout(ff_output))
        # Output des Decoder-Layer: verarbeiteter Tensor
        return x


# Encoder und Decoder kombinieren, um komplettes Transformer Network zu erhalten
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# #Performance Evaluation
# transformer.eval()

# val_src_data =
# val_tgt_data =

# with torch.no_grad():

#     val_output = transformer(val_src_data, val_tgt_data[:, :-1])
#     val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
#     print(f"Validation Loss: {val_loss.item()}")
# """

# # https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
