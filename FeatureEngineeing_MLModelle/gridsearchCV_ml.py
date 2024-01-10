#ml model
import pandas as pd
import numpy as np

#modelle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


##################### Basis Model#######################
class BaseModel:
    #Länge der prediction --> i
    num_points = 20     #20 für 1 Monat -> 20 Business Days #None, für den gesmaten X_test lang
    
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train): # Model trainieren mit Train
        self.model.fit(X_train, y_train)

    #Hyperparameter
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        return self

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
    hyperparameters = {"n_estimators": 1, "max_depth": 1, "min_samples_split": 1, "min_samples_leaf": 1}

    def __init__(self, random_state=None, **kwargs):
        # Aktualisieren der hyperparameters mit zusätzlichen übergebenen kwargs
        updated_hyperparameters = RandomForestModel.hyperparameters.copy()
        updated_hyperparameters.update(kwargs)

        super().__init__(RandomForestRegressor(**RandomForestModel.hyperparameters))
    
    def get_feature_importances(self):
        feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return feature_importances

    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        # Rückgabe der Parameter des Modells
        return {**self.hyperparameters, **self.model.get_params(deep=deep)}

    def set_params(self, **params):
        # Aktualisieren der hyperparameters und des Modells mit neuen Parametern
        self.hyperparameters.update(params)
        self.model.set_params(**params)
        return self

############### GBM ##################### 
class GradientBoostingModel(BaseModel):
    hyperparameters = {"n_estimators": 1, "max_depth": 1, "min_samples_split": 1, "min_samples_leaf": 1}

    def __init__(self, random_state=None, **kwargs):
        # Aktualisieren der hyperparameters mit zusätzlichen übergebenen kwargs
        updated_hyperparameters = RandomForestModel.hyperparameters.copy()
        updated_hyperparameters.update(kwargs)

        super().__init__(GradientBoostingRegressor(**GradientBoostingModel.hyperparameters))

    def get_feature_importances(self):
        feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return feature_importances
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        # Rückgabe der Parameter des Modells
        return {**self.hyperparameters, **self.model.get_params(deep=deep)}

    def set_params(self, **params):
        # Aktualisieren der hyperparameters und des Modells mit neuen Parametern
        self.hyperparameters.update(params)
        self.model.set_params(**params)
        return self

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

