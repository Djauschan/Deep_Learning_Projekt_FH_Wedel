import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_ts import FeatureEngineering

from feature_ts import day_name_transformer, dtf, replace_weekend_volume, imputer, drop_ts, remove_infinite
from feature_ts import differenz_value, pct_change_transformer, differenz_pct_change_transformer
from feature_ts import lag_backward_20d_features, monthly_average_feature
from feature_ts import lag_backward_7h_features, lag_forward_7h_features
from feature_ts import window_feature_transformer

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
                
                # for hourly
                ("lag_features_back", lag_backward_7h_features),
                ("lag_features_for", lag_forward_7h_features),

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
        self.pipe_h = Pipeline( #hourly df
            [
                ("create_differenz", differenz_value),
                ("datetime_features", dtf),
                ("lag_features_back", lag_backward_7h_features),
                ("replace_weekend_volume", replace_weekend_volume),
                ("dropna", imputer),
            ]
        )

        self.pipe_busdaily = Pipeline( #daily business df
            [
                ("datetime_features", dtf),
                ("pct_change", pct_change_transformer),
                ("lag_backward_20d_features", lag_backward_20d_features),
                ("monthly_average_feature", monthly_average_feature),
                ("dropna", imputer),
            ]
        )

        self.pipe_test = Pipeline(
            [
                ("datetime_features", dtf),
                ("replace_weekend_volume", replace_weekend_volume),
                ("window_feature_transformer", window_feature_transformer),
                ("pct_dif", differenz_pct_change_transformer),
                ("remove_infinite", remove_infinite),
                ("dropna", imputer),
            ]
        )

    def fit_transform(self, data, pipeline_name):
        if pipeline_name == 'all': #alle
            return self.pipe_all.fit_transform(data)
        elif pipeline_name == 'h': #hourly
            return self.pipe_h.fit_transform(data)
        elif pipeline_name == 'busdaily': #business daily
            return self.pipe_busdaily.fit_transform(data)
        elif pipeline_name == 'test': #normal
            return self.pipe_test.fit_transform(data)
        else:
            raise ValueError("Unbekannter Pipeline-Name")
