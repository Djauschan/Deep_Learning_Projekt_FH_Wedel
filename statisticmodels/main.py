import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.txtReader import DataReader
from preprocessing.data_cleaning import DataCleaner
from models.arima import Arima


test = DataReader({"READ_ALL_FILES": "READ_ALL_FILES"})

txt_files, symbols = test.get_txt_files()

# Test for-Schleife später löschen
for i in symbols:
    print(i)

test.current_file_idx = 1

df = test.read_next_txt()

clean = DataCleaner(df)



print(clean.df_at_16.info())
print(clean.df_at_09.info())
df16Cleaned=clean.df_at_16
df09Cleaned=clean.df_at_09

arimaData= clean.transformForNixtla(df16Cleaned)
ArimaModell=Arima(arimaData)
print(ArimaModell.data.head())
arimaData.plot()


