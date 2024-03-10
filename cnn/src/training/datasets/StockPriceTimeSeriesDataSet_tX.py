
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


class StockPriceTimeSeriesDataSet(Dataset):
    """
        needs data array like
            (a,b,c,d) = (1500, 9, 20, 20) = (länge_aller_ts, features, länge_single_ts, länge_single_Ts)
    """
    def __init__(self, data, labels, transform):
        self.transform = transform
        self.tsData = data
        self.tsLabels = labels

    def __len__(self):
        return len(self.tsLabels)

    def getData(self):
        return self.tsData

    def __getitem__(self, idx):
        item = {
            'data': self.tsData[idx - 1],
            'label': self.tsLabels[idx - 1]
        }
        if self.transform is not None:
            sample = self.transform(item)
        else:
            sample = item

        return sample
