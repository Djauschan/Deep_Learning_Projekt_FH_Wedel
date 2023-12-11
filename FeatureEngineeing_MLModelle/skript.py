#Package einlesen
import pandas as pd
import numpy as np

#classe
from txtReader import DataReader
from data_cleaning import DataCleaner

from feature_ts import FeatureEngineering
from pipeline import ClassPipeline


### main

data = DataReader({"READ_ALL_FILES": "READ_ALL_FILES"})
txt_files, symbols = data.get_txt_files()
 
# Test for-Schleife später löschen
for i in symbols:
    print(i)


# 0 = AAL, 1 = AAPL, ...   #von den 10 datas
data.current_file_idx = 1
df = data.read_next_txt()

#Data clean
cleaner = DataCleaner(df)

df_normal = cleaner.normal()
df_hourly = cleaner.hourly()
df_minute = cleaner.minute()
df_daily = cleaner.daily()

df_normal

#features
data = df_hourly

#pipeline
# Verwendung der ClassPipeline
pipeline = ClassPipeline(data)

# Anwenden der Test-Pipeline
#data_pip_test = pipeline.fit_transform(data, 'test')

# Anwenden der Cor-Pipeline
#data_pip_cor = pipeline.fit_transform(data, 'cor')

# Anwenden der Model-Pipeline
data_pip_model = pipeline.fit_transform(data, 'model')


print(data_pip_model)

print(data_pip_model.columns)
#modelle