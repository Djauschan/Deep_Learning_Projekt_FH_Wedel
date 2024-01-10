import math

import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class Multi_Layer_Perceptron(nn.Module):

    def __init__(self,
                 seq_len_encoder,
                 seq_len_decoder,
                 dim_decoder,
                 dim_encoder,
                 hidden_dim,
                 dropoutrate,
                 device: torch.device):
        super(Multi_Layer_Perceptron, self).__init__()
        self.device = device
        self.layer_stack = nn.Sequential(
            nn.Linear(seq_len_encoder * dim_encoder, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropoutrate),  
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len_decoder * dim_decoder),
        )

    def forward(self, x, _):
        x.to(self.device)

        # Flatten the input (batch_size, seq_len, dim) -> (batch_size, seq_len * dim)
        # The volume of the stock and the encoding for start off date an end of date are still in the input
        x = x.view(x.size(0), -1)
        output = self.layer_stack(x)
        return output.unsqueeze(2)
