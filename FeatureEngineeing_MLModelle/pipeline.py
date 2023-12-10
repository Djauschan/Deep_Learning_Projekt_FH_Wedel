from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_ts import FeatureEngineering


class ClassPipeline:
    def __init__(self):

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
