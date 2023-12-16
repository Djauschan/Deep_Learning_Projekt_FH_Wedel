### Features erstellen

import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer

from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import (LagFeatures, WindowFeatures,)
from feature_engine.creation import CyclicalFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures

data_columns = ['open', 'high', 'low', 'close', 'volume']

####################################################################
### jeden df
def add_day_name():
    def transform(data):
        df = data.copy()
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

def create_differenz_value(df):
    for col in data_columns:
        df[col + '_Differenz'] = df[col].diff()
    return df
differenz_value = FunctionTransformer(create_differenz_value)

def create_differenz_and_pct_change(data):
    df = data.copy()
    for col in data_columns:
        if col in data_columns:
            df.loc[:, col + '_Differenz'] = df[col].diff() # Berechnung der Differenz
            df.loc[:, col + '_PctChange'] = df[col].pct_change() * 100 # Berechnung der prozentualen Änderung # Multipliziert mit 100, um es in Prozent auszudrücken
    return df
differenz_pct_change_transformer = FunctionTransformer(create_differenz_and_pct_change)

# Drop missing data
imputer = DropMissingData()

#drop infinitiv valuess
def remove_infinite_values(df):
    # Ersetzt sowohl positive als auch negative unendliche Werte durch NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True) # Entfernt alle Zeilen mit NaN-Werten
    return df
remove_infinite = FunctionTransformer(remove_infinite_values)

# Drop original time series
drop_ts = DropFeatures(features_to_drop=['open', 'high', 'low', 'close', 'volume'])



####################################################################
### df normal / minütlich && if dtf is made
def create_window_feature(data):
    df = data.copy()
    for col in data_columns:
        if col in data_columns:
            # Berechnen des durchschnittlichen Wertes der Spalte pro Stunde
            average_value_per_hour = df.groupby('hour')[col].transform('mean')
            df[col + '_average_per_hour'] = average_value_per_hour

    return df
window_feature_transformer = FunctionTransformer(create_window_feature) #kw_args={'data_columns': data_columns}

####################################################################
### df hourly 

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
    df.loc[df['weekend'] == 1, 'volume'] = 0
    return df
replace_weekend_volume = FunctionTransformer(replace_weekend_volume_with_zero)


####################################################################
# Class FeatureEngineering
class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_columns = ['open', 'high', 'low', 'close', 'volume']
    
