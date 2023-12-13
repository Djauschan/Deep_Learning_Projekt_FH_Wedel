import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_ts import FeatureEngineering
from feature_ts import day_name_transformer, dtf, lag_backward_features, lag_forward_features,replace_weekend_volume, imputer, drop_ts, differenz_value

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
                ("replace_weekend_volume", replace_weekend_volume),
                ("value_differenz", differenz_value),

                # for hourly
                ("lag_features_back", lag_backward_features),
                ("lag_features_for", lag_forward_features),

                # last ones
                ("dropna", imputer),
                ("drop_ts", drop_ts),
            ]
        )

        self.pipe_h = Pipeline( #hourly df
            [
                ("create_differenz", differenz_value),
                ("datetime_features", dtf),
                ("lag_features_back", lag_backward_features),
                ("replace_weekend_volume", replace_weekend_volume),
                ("dropna", imputer),
            ]
        )

        self.pipe_dif = Pipeline(
            [
                ("dropna", imputer),
            ]
        )

    def fit_transform(self, data, pipeline_name):
        if pipeline_name == 'all':
            return self.pipe_test.fit_transform(data)
        elif pipeline_name == 'h':
            return self.pipe_h.fit_transform(data)
        elif pipeline_name == 'dif':
            return self.pipe_model.fit_transform(data)
        else:
            raise ValueError("Unbekannter Pipeline-Name")
