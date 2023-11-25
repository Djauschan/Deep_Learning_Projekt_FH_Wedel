import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.txtReader import DataReader

test= DataReader({"READ_ALL_FILES":"READ_ALL_FILES"})

txt_files, symbols=test.get_txt_files()

for i in symbols: 
    print(i)