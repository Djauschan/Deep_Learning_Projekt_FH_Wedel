import torch
import torch.nn as nn


class My_loss: 

    @classmethod
    def rmse (cls, prediction: torch.tensor, target:torch.tensor):
        mse_loss = nn.MSELoss()(prediction, target)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss