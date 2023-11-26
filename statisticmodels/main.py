import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.txtReader import DataReader
from preprocessing.data_cleaning import DataCleaner


test = DataReader({"READ_ALL_FILES": "READ_ALL_FILES"})

txt_files, symbols = test.get_txt_files()

# Test for-Schleife später löschen
for i in symbols:
    print(i)

test.current_file_idx = 1

df = test.read_next_txt()

clean = DataCleaner(df)


clean.calculate_diff(clean.df_at_16,[" 15:00:00", " 16:30:00"])


df16Cleaned=clean.datafiller(' 16:00:00')
print(df16Cleaned.info())

clean.calculate_diff(clean.df_at_09,[" 08:30:00", " 10:00:00"])
df09Cleaned=clean.datafiller(' 09:30:00')
print(df09Cleaned.info())

#Kleine Visualisierung
plt.style.use('fivethirtyeight')
chrr,ax=plt.subplots(ncols=2,figsize=(10,10))
ax[0].plot(df16Cleaned['close'])
ax[1].plot(df09Cleaned['close'])
plt.show()