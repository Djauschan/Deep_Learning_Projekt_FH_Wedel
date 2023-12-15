import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class RMSELoss(nn.Module):

    def __init__(self):
        """
        Initializes the RMSELoss class.
        """
        super().__init__()

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        Calculates the root mean squared error (RMSE) between the prediction and the target.

        Args:
            prediction (torch.tensor): Prediction of the model.
            target (torch.tensor): Correct target values.

        Returns:
            torch.tensor: The RMSE loss.
        """

        mse_loss = nn.MSELoss()(prediction, target, min=0.0001)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss


class RMSLELoss(nn.Module):

    def __init__(self):
        """
        Initializes the RMSLELoss class.
        """
        super().__init__()

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        Calculates the root mean squared logarithmic error (RMSLE) between the prediction and the target.

        Args:
            prediction (torch.tensor): Prediction of the model.
            target (torch.tensor): Correct target values.

        Returns:
            torch.tensor: The RMSLE loss.
        """
        # Ensure that the values are not negative to avoid NaNs
        prediction = torch.clamp(prediction, min=0.0001)
        target = torch.clamp(target, min=0.0001)

        # Applying the logarithm to prediction and target
        log_prediction = torch.log1p(prediction)
        log_target = torch.log1p(target)

        # Calculation of the RMSE
        mse_loss = nn.MSELoss()(log_prediction, log_target)
        rmsle_loss = torch.sqrt(mse_loss)

        return rmsle_loss
