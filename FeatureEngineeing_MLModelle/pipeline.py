import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#siehe feature_ts.py
from feature_ts import FeatureEngineering
from feature_ts import day_name_transformer, dtf, lag_backward_features, lag_forward_features,replace_weekend_volume, imputer, drop_ts


class ClassPipeline:
    def __init__(self, data):

        self.data = data
        # Initialisierung der FeatureEngineering-Klasse mit den Daten
        feature_engineering = FeatureEngineering(self.data)

        pipe_test = Pipeline(
            [
                ("day_name", day_name_transformer),
                ("datetime_features", dtf),
                ("lag_features_back", lag_backward_features),
                ("lag_features_for", lag_forward_features),
                ("replace_weekend_volume", replace_weekend_volume),
                ("dropna", imputer),
                ("drop_ts", drop_ts),
            ]
        )

        pipe_cor = Pipeline(
            [
                ("day_name", day_name_transformer),
                ("datetime_features", dtf),
                ("lag_features_back", lag_backward_features),
                ("lag_features_for", lag_forward_features),
                ("replace_weekend_volume", replace_weekend_volume),
                ("dropna", imputer),
                #("drop_ts", drop_ts),
            ]
        )

        #pipeline features mit orignail spalten -> alles
        pipe_model = Pipeline(
            [
                #("day_name", day_name_transformer),
                ("datetime_features", dtf),
                ("lag_features_back", lag_backward_features),
                #("lag_features_for", lag_forward_features),
                ("replace_weekend_volume", replace_weekend_volume),
                ("dropna", imputer),
                #("drop_ts", drop_ts),
            ]
        )

    def fit_transform(self, data, pipeline_name):
        if pipeline_name == 'test':
            return self.pipe_test.fit_transform(data)
        elif pipeline_name == 'cor':
            return self.pipe_cor.fit_transform(data)
        elif pipeline_name == 'model':
            return self.pipe_model.fit_transform(data)
        else:
            raise ValueError("Unbekannter Pipeline-Name")
