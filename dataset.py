import torch
from torch.utils.data import Dataset
from txtReader import DataReader
from dataProcessing import lookup_symbol, add_time_information
from torch.utils.data import DataLoader
import pandas as pd
from config import DEBUG, BATCH_SIZE


class PerSymbolETFDataset(Dataset):
    """
    Data stored as tensors
    Pytorch uses the 3 functions [__init__, __len__, __getitem__]
    """

    def __init__(self, data_frame: pd.DataFrame):
        """
        Initializes the Pytorch data set.

        Args:
            data_frame (pd.DataFrame): Data frame that contains the data for the dataset.
        """
        # The symbol of the ETF is read out.
        self.symbol = data_frame["symbol"][1]
        # Then the name for the ETF is looked up.
        self.name = lookup_symbol(self.symbol)
        # Then it is removed from the data frame, since it is the same for all lines.
        data_frame = data_frame.drop("symbol", axis=1)

        data_frame = add_time_information(data_frame)

        if DEBUG:
            print(data_frame)

        # Only the numerical data is used for machine learning.
        numeric_values = data_frame.iloc[:, 1:9].to_numpy()

        # The data of the current transaction is used to predict the data of the next transaction.
        input = numeric_values[:-1]
        output = numeric_values[1:]

        # For the use of pytorch, the data is converted into tensors.
        self.input_data = torch.tensor(input, dtype=torch.float32)
        self.output_data = torch.tensor(output, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns:
            int: number of samples in the dataset
        """
        return len(self.input_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampel (X, Y) at position idx

        Args:
            idx (int): position of sample in the tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sampel (X, Y) at position idx
        """
        return self.input_data[idx], self.output_data[idx]


# Code for debugging
if __name__ == "__main__":
    # Create dataset
    txt_reader = DataReader()
    data = txt_reader.read_next_txt()
    dataset = PerSymbolETFDataset(data)
    # Create data loader
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Print the first sample.
    test_sample = next(iter(dataloader))
    print("Input: ", test_sample[0])
    print("Output: ", test_sample[1])
