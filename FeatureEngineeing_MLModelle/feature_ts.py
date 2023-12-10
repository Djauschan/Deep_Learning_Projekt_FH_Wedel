### Features erstellen
import pandas as pd
import numpy as np

from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (LagFeatures, WindowFeatures,)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#aktuell für hourly

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_columns = ['open', 'high', 'low', 'close', 'volume']
            
    #Datetime 

    #erstelle Feature für Wochentag
    def add_day_name():
        def transform(df):
            df = df.copy()
            df["day_name"] = df.index.day_name()
            return df
        return transform
    day_name_transformer = FunctionTransformer(add_day_name())  # anpassen, sodass Funktion in eine Pipeline integriert werden kann

    dtf = DatetimeFeatures(
        # the datetime variable
        variables="index",
        # the features we want to create
        features_to_extract=[
            "week",
            "month",
            "day_of_week",  #0=monday, ... 6=sunday
            "day_of_month",
            "hour",
            "weekend",
        ],
    )

    #lag feature backward mit 1h - 7h 
    def create_lag_features_backward(data, max_lag_hours=7):
        df = data.copy()
        for col in data_columns:
            for lag_hour in range(1, max_lag_hours + 1):
                df[f'{col}_lag_{lag_hour}H_back'] = data[col].shift(lag_hour)
        return df
    lag_backward_features= FunctionTransformer(create_lag_features_backward) # Erstellen einer  FunctionTransformer-Instanz um es in die Pipeline aufnehmen zu können

    #lag feature foreward mit 1h - 7h 
    def create_lag_features_forward(data, max_lag_hours=7):
        df = data.copy()
        for col in data_columns:
            for lag_hour in range(1, max_lag_hours + 1):
                df[f'{col}_lag_{lag_hour}H_forward'] = data[col].shift(-lag_hour)
        return df

    lag_forward_features= FunctionTransformer(create_lag_features_forward)

    def replace_weekend_volume_with_zero(data):
        # Kopie des DataFrame erstellen, um das Original nicht zu ändern
        df = data.copy()
        # Ersetze das 'Volume'-Attribut auf 0 für Wochenenden (wenn 'weekend' gleich 1 ist)
        df.loc[df['weekend'] == 1, 'Volume'] = 0
        return df
    replace_weekend_volume = FunctionTransformer(replace_weekend_volume_with_zero)

    # Drop missing data
    imputer = DropMissingData()

    # Drop original time series
    drop_ts = DropFeatures(features_to_drop=['open', 'high', 'low', 'close', 'volume'])

    #drop_ts = DropFeatures(features_to_drop=data_columns)
