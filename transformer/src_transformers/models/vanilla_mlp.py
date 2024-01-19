import math

import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class ResidualBlock(nn.Module):
    def __init__(self,
                 hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Batch Normalisierung
        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x
        out = self.linear(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = out + residual  # ursprünglichen Input zum Output addieren
        return out


class Multi_Layer_Perceptron(nn.Module):
    def __init__(self,
                 seq_len_encoder,
                 seq_len_decoder,
                 dim_decoder,
                 dim_encoder,
                 hidden_dim,
                 dropoutrate,  # bei BatchNorm erstmal kein Dropout um kein Noise zu erzeugen
                 device: torch.device):
        super(Multi_Layer_Perceptron, self).__init__()
        self.device = device

        self.seq_len_encoder = seq_len_encoder  # Required to save the model
        self.seq_len_decoder = seq_len_decoder  # Required to save the model
        self.dim_encoder = dim_encoder  # Required to save the model
        self.dim_decoder = dim_decoder  # Required to save the model

        # Erster Layer
        layers = [nn.Linear(seq_len_encoder * dim_encoder,
                            hidden_dim), nn.ReLU()]

        # Residual Layers
        for _ in range(2):  # 6-mal für die 6 versteckten Layer
            layers.append(ResidualBlock(hidden_dim))
            # Optional: Dropout einfügen
            # if dropoutrate > 0:
            #     layers.append(nn.Dropout(dropoutrate))

        # Letzter Layer
        layers.append(nn.Linear(hidden_dim, seq_len_decoder * dim_decoder))

        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x, _):
        x.to(self.device)  # TODO: Check if required else: delete (@Eira)

        # Flatten the input (batch_size, seq_len, dim) -> (batch_size, seq_len * dim)
        # The volume of the stock and the encoding for start off date an end of date are still in the input
        x = x.view(x.size(0), -1)
        output = self.layer_stack(x)
        return output.unsqueeze(2)
