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

############### RF ##################### 
class RandomForestModel(BaseModel):
    hyperparameters = {"n_estimators": 10, "max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 50, "random_state": 11}

    def __init__(self, **kwargs):
        # Aktualisieren der hyperparameters mit zusätzlichen übergebenen kwargs
        updated_hyperparameters = RandomForestModel.hyperparameters.copy()
        updated_hyperparameters.update(kwargs)

        super().__init__(RandomForestRegressor(**RandomForestModel.updated_hyperparameters))
    
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
    hyperparameters = {"n_estimators": 1, "max_depth": 1, "min_samples_split": 1, "min_samples_leaf": 1, "random_state": 11}

    def __init__(self, **kwargs):
        # Aktualisieren der hyperparameters mit zusätzlichen übergebenen kwargs
        updated_hyperparameters = RandomForestModel.hyperparameters.copy()
        updated_hyperparameters.update(kwargs)

        super().__init__(GradientBoostingRegressor(**GradientBoostingModel.updated_hyperparameters))
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        return {**self.hyperparameters, **self.model.get_params(deep=deep)}

    def set_params(self, **params):
        self.hyperparameters.update(params)
        self.model.set_params(**params)
        return self

############### SVM ##################### 
class SVMModel(BaseModel):
    hyperparameters = {"kernel": 'rbf', "C": 1.0, "epsilon": 0.1}

    def __init__(self, **kwargs):
        # Aktualisieren der hyperparameters mit zusätzlichen übergebenen kwargs
        updated_hyperparameters = RandomForestModel.hyperparameters.copy()
        updated_hyperparameters.update(kwargs)

        super().__init__(SVR(**SVMModel.updated_hyperparameters))

