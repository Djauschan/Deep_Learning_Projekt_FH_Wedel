import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from torch.utils.data import Dataset
from txtReader import DataReader
from dataProcessing import lookup_symbol, add_time_information, create_one_hot_vector
from torch.utils.data import DataLoader
import pandas as pd
from config import config
import numpy as np
from perSymbolETFDataset import PerSymbolETFDataset


# Initialize the DataReader and read data
txt_reader = DataReader()
data = txt_reader.read_next_txt()

# Create a dataset from the read data
dataset = PerSymbolETFDataset(data, txt_reader.symbols)

# Create DataLoader for batch processing
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False  # Batch size set to 1 for individual stock processing
)

# Testen der Datenstruktur aus dem DataLoader
for batch in dataloader:
    src_data, tgt_data = batch
    print("Shape of src_data:", src_data.shape)
    print("First few rows of src_data:", src_data[:5])
    break  # Dies bricht die Schleife nach dem ersten Batch ab

def dataloader_to_dataframe(dataloader):
    frames = []
    for batch in dataloader:
        src_data, tgt_data = batch
        # nur den "close"-Kurs
        close_prices = src_data.numpy()[:, 1]  # Angenommene Position des "close"-Kurses
        frames.append(pd.DataFrame(close_prices, columns=['close']))
    return pd.concat(frames)

# DataLoader initialisieren und Daten in DataFrame umwandeln
stock_data = dataloader_to_dataframe(dataloader)

# ##### Daten auf Y-Achse Spiegeln

stock_data2 = stock_data.iloc[::-1].copy()

