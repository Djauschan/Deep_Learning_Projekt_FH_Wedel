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

############### LR ##################### 
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test):
        evaluation_results = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions, squared=False),
            #"MSLE": mean_squared_log_error(y_test, predictions),
            #"Median AE": median_absolute_error(y_test, predictions),
        }
        return evaluation_results

    def vs (self, X_test, y_test):
        variance_score = {
            "Variance Score": self.model.score(X_test, y_test),
        }
        return variance_score
    
    def coef(self):
        coef_results = {
            "Coefficients": self.model.coef_,
        }
        return coef_results

############### RF ##################### 
class RandomForestModel:
    hyperparameters = {"n_estimators": 10, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1}

    def __init__(self):
        self.model = RandomForestRegressor(**RandomForestModel.hyperparameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test):
        evaluation_results = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions, squared=False),
            #"MSLE": mean_squared_log_error(y_test, predictions),
            #"Median AE": median_absolute_error(y_test, predictions),
        }
        return evaluation_results
    
    def fi(self):
        rf_feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return rf_feature_importances


############### GBM ##################### 
class GradientBoostingModel:
    hyperparameters = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}

    def __init__(self):
        self.model = GradientBoostingRegressor(**GradientBoostingModel.hyperparameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test):
        evaluation_results = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions, squared=False),
            #"MSLE": mean_squared_log_error(y_test, predictions),
            #"Median AE": median_absolute_error(y_test, predictions),
        }
        return evaluation_results
    
    def fi(self):
        gbm_feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return gbm_feature_importances


############### SVM ##################### 
class SVMModel:
    hyperparameters = {"kernel": 'rbf', "C": 1.0, "epsilon": 0.1}

    def __init__(self):
        self.model = SVR(**SVMModel.hyperparameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, predictions, y_test):
        evaluation_results = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions, squared=False),
            #"MSLE": mean_squared_log_error(y_test, predictions),
            #"Median AE": median_absolute_error(y_test, predictions),
        }
        return evaluation_results
