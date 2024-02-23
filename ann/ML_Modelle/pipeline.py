import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_ts import FeatureEngineering

#all
from feature_ts import day_name_transformer, dtf, replace_weekend_volume, imputer, drop_ts, remove_infinite
from feature_ts import differenz_value, pct_change_transformer, differenz_pct_change_transformer

#daily
from feature_ts import lag_backward_20d_features, monthly_average_feature

#hour
from feature_ts import lag_backward_7h_features, remove_weekend

#min
from feature_ts import window_feature_transformer, lag_backward_60min

class ClassPipeline:
    def __init__(self, data):

        self.data = data
        # Initialisierung der FeatureEngineering-Klasse mit den Daten
        feature_engineering = FeatureEngineering(self.data)

        self.pipe_all = Pipeline( ##alle Features die es aktuell gibt
            [
                # for all
                ("day_name", day_name_transformer),
                ("datetime_features", dtf),
                ("replace_weekend_volume", replace_weekend_volume), #for daily -> resample "D"
                ("differenz_value", differenz_value), #dif
                ("pct_change", pct_change_transformer), #pct
                ("pct_dif", differenz_pct_change_transformer), #dif + pct

                #for normal / minütlich, only with dtf
                ("window_feature", window_feature_transformer), #mit dtf ! -> ("datetime_features", dtf),
                ("lag_backward_60min", lag_backward_60min),

                # for hourly
                ("lag_features_back", lag_backward_7h_features),
                ("remove_weekend", remove_weekend),

                #bus daily
                ("lag_backward_20d_features", lag_backward_20d_features),
                ("monthly_average_feature", monthly_average_feature),

                # last ones
                ("dropna", imputer),
                ("remove_infinite", remove_infinite),
                ("drop_ts", drop_ts),
            ]
        )
        ################################################################
        self.pipe_min = Pipeline( #min df    
            #model min prediction
            [
                ("datetime_features", dtf),
                ("pct_change", pct_change_transformer),
                ("lag_backward_60min", lag_backward_60min),
                ("dropna", imputer),
            ]
        )
        self.pipe_hour = Pipeline( #hourly df 
            #model hour prediction
            [
                ("datetime_features", dtf),
                #("remove_weekend", remove_weekend),
                ("pct_change", pct_change_transformer),
                ("lag_features_back", lag_backward_7h_features),
                ("dropna", imputer),
            ]
        )

        self.pipe_busdaily = Pipeline( #daily business df 
            #model daily prediction
            [
                ("datetime_features", dtf),
                ("pct_change", pct_change_transformer),
                ("lag_backward_20d_features", lag_backward_20d_features),
                ("dropna", imputer),
            ]
        )

    def fit_transform(self, data, pipeline_name):
        if pipeline_name == 'all': #alle
            return self.pipe_all.fit_transform(data)
        elif pipeline_name == 'min': #min
            return self.pipe_min.fit_transform(data)
        elif pipeline_name == 'hour': #hourly
            return self.pipe_hour.fit_transform(data)
        elif pipeline_name == 'busdaily': #business daily
            return self.pipe_busdaily.fit_transform(data)
        else:
            raise ValueError("Unbekannter Pipeline-Name")
