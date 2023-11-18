import pandas as pd
import torch
from torch.utils.data import Dataset


class SingleStockDataset(Dataset):
    """
    A dataset class that extends the generic
    torch dataset and contains just one time series
    of one stock.
    """

    def __init__(self, ts: pd.DataFrame, input_len: int, target_len: int):
        """
        Initializes the dataset with a time series as pandas series.
        Args:
            ts: Time series as pandas series
            input_len: Lenght of the input for the model
            target_len: Length of the target for the model
        """
        self.ts = ts
        self.input_len = input_len
        self.target_len = target_len

    def __len__(self):
        return len(self.ts) - self.input_len - self.target_len - 1

    def __getitem__(self, item):
        # Get the input data
        start_input = item
        end_input = item + self.input_len
        input = self.ts[start_input:end_input].values

        # Get the output target data
        start_target = end_input + 1
        end_target = start_target + self.target_len
        target = self.ts[start_target:end_target].values

        # Wrap as item of tensors
        item = {
            "input": torch.tensor(input, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.float),
        }

        return item
