import pandas as pd
import numpy as np

#modelle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# evaluation metric 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error

##################### Actual value #####################
class ActualValues:
    #Länge der prediction --> i
    num_points = 100     #20 für 1 Monat -> 20 Business Days #None, für den gesmaten X_test lang
    
    def get_actual_values_test_close(self, X_test, back_transform_test_data):
        actual_close_values = []
        indices = []

        max_points = min(self.num_points, len(X_test))
        for i in range(max_points):
            actual_close = back_transform_test_data["close"].iloc[i]
            actual_close_values.append(actual_close)
            indices.append(X_test.index[i])

        return actual_close_values, indices

    def get_actual_values_test_open(self, X_test, back_transform_test_data):
        actual_open_values = []
        indices = []

        max_points = min(self.num_points, len(X_test))
        for i in range(max_points):
            actual_open = back_transform_test_data["open"].iloc[i]
            actual_open_values.append(actual_open)
            indices.append(X_test.index[i])

        return actual_open_values, indices


##################### Basis Model#######################
class BaseModel:
    #Länge der prediction --> i
    num_points = 100     #20 für 1 Monat -> 20 Business Days #None, für den gesmaten X_test lang
    
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train): # Model trainieren mit Train
        self.model.fit(X_train, y_train)

    #test
    def predict(self, X_test, back_transform_test_data): # Prediction erstellen
        predicted_close_values = []
        indices = []
        last_known_close_value = back_transform_test_data['close'].iloc[0]

        max_points = min(self.num_points, len(X_test)) 
        for i in range(max_points):
            X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)  # Umwandlung der Daten in DataFrame mit Feature-Namen
            predicted_pct_change = self.model.predict(X_test_row_df)[0]             # Vorhersage für den nächsten Tag (prozentuale Veränderung)
            # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
            predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
            predicted_close_values.append(predicted_close)
            
            # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
            last_known_close_value = back_transform_test_data["close"].iloc[i]
            indices.append(X_test.index[i])

        return predicted_close_values, indices

    #test
    def evaluate(self, predictions, y_test):
        evaluation_results = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions, squared=False),
            #"MSLE": mean_squared_log_error(y_test, predictions),
            #"Median AE": median_absolute_error(y_test, predictions),
        }
        return evaluation_results
    
    ############### ergänzung von train model, um train und test fehler zu vergleichen
    #train
    def train_predict(self, X):
        return self.model.predict(X)
    
    #train
    def calculate_training_error(self, X_train, y_train):
        # Vorhersagen für Trainingsdaten erstellen
        y_pred_train = self.train_predict(X_train)
        # Fehlermetriken berechnen
        training_error = {
            "MAE": mean_absolute_error(y_train, y_pred_train),
            "RMSE": mean_squared_error(y_train, y_pred_train, squared=False),
        }
        return training_error

############### LR ##################### 
class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LinearRegression())
    
    def calculate_variance_score(self, X_test, y_test):
        variance_score = {
            "Variance Score": self.model.score(X_test, y_test),
        }
        return variance_score

    def get_coefficients(self):
        coef_results = {
            "Coefficients": self.model.coef_,
        }
        return coef_results

############### RF ##################### 
class RandomForestModel(BaseModel):
    hyperparameters = {"n_estimators": 10, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 10}

    def __init__(self):
        super().__init__(RandomForestRegressor(**RandomForestModel.hyperparameters))
    
    def get_feature_importances(self):
        feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return feature_importances

############### GBM ##################### 
class GradientBoostingModel(BaseModel):
    hyperparameters = {"n_estimators": 10, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 10}

    def __init__(self):
        super().__init__(GradientBoostingRegressor(**GradientBoostingModel.hyperparameters))

    def get_feature_importances(self):
        feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return feature_importances

############### SVM ##################### 
class SVMModel(BaseModel):
    hyperparameters = {"kernel": 'rbf', "C": 1.0, "epsilon": 0.1}

    def __init__(self):
        super().__init__(SVR(**SVMModel.hyperparameters))
    
    def get_support_vectors(self):
        return self.model.support_vectors_

    def get_n_support(self):
        return self.model.n_support_

    def get_dual_coef(self):
        return self.model.dual_coef_

