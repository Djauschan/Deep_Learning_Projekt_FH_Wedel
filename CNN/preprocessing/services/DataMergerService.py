import os

import numpy as np
import pandas as pd

from CNN.preprocessing.services.ImportService import importService


class DataMergerService:

    def __init__(self):
        pass

    def mergeFeatureData(self, main_df: pd.DataFrame, df_toMerge: pd.DataFrame) -> pd.DataFrame:
        '''
            merge the dataframe in the list
        '''
        return main_df.merge(df_toMerge, how='inner', left_on=['posixMinute'], right_on=['posixMinute'])
    '''
        param:
            @dataArr1, (n x m) array like, the array to be merged with
            @dataArr2, (n x m) array like, the array to be merged with
            the dim of both arrays musst be the same
        return
            @result
    '''
    def _mergeData(self, dataArr1, dataArr2):
        pass
