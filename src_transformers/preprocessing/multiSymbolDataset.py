import torch
import sys
import yaml
from torch.utils.data import Dataset
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.preprocessing.dataProcessing import lookup_symbol, add_time_information, create_one_hot_vector, get_all_dates, fill_dataframe
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class MultiSymbolDataset(Dataset):
    """
    Data stored as tensors
    Pytorch uses the 3 functions [__init__, __len__, __getitem__]
    """

    def __init__(self, reader: DataReader, config: dict):
        date_df = get_all_dates(reader)
        date_df = fill_dataframe(date_df, reader)

        date_df.to_csv("dates.csv", index=False)

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass
