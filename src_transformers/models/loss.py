import torch
import torch.nn as nn
import torch.nn.functional as F


class My_loss: 

    # Root Mean Squared Error
    @classmethod
    def rmse (cls, prediction: torch.tensor, target:torch.tensor):
        mse_loss = nn.MSELoss()(prediction, target)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss
    
    
    @classmethod
    def rmsle(cls, prediction: torch.tensor, target: torch.tensor):
        # Anwenden des Logarithmus auf prediction und target
        log_prediction = torch.log1p(prediction)
        log_target = torch.log1p(target)

        # Berechnung des RMSE
        mse_loss = nn.MSELoss()(log_prediction, log_target)
        rmsle_loss = torch.sqrt(mse_loss)

        return rmsle_loss 