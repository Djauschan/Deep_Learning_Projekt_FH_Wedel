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


################ Prediction länge ##################
class PredictionLength:
    ''' Die jeweilige Länge der Prediction.'''
    def __init__(self, pred_length=1):
        self.pred_length = pred_length

##################### Actual value #####################
class ActualValues(PredictionLength):
    def __init__(self, pred_length=1):
        super().__init__(pred_length=pred_length)
    
    def get_actual_values_test_close(self, X_test, back_transform_test_data):
        actual_close_values = []
        indices = []

        max_points = min(self.pred_length, len(X_test))
        for i in range(max_points):
            actual_close = back_transform_test_data["close"].iloc[i]
            actual_close_values.append(actual_close)
            indices.append(X_test.index[i])

        return actual_close_values, indices

    def get_actual_values_test_open(self, X_test, back_transform_test_data):
        actual_open_values = []
        indices = []

        max_points = min(self.pred_length, len(X_test))
        for i in range(max_points):
            actual_open = back_transform_test_data["open"].iloc[i]
            actual_open_values.append(actual_open)
            indices.append(X_test.index[i])

        return actual_open_values, indices

##################### Basis Model#######################
class BaseModel(PredictionLength):
    def __init__(self, model, pred_length=1):
        super().__init__(pred_length=pred_length)
        self.model = model

    def fit(self, X_train, y_train): # Model trainieren mit Train
        self.model.fit(X_train, y_train)

    #test
    def predict(self, X_test, back_transform_test_data): # Prediction erstellen
        predicted_close_values = []
        indices = []
        last_known_close_value = back_transform_test_data['close'].iloc[0]

        max_points = min(self.pred_length, len(X_test)) 
        for i in range(max_points):
            X_test_row_df = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)  # Umwandlung der Daten in DataFrame mit Feature-Namen
            predicted_pct_change = self.model.predict(X_test_row_df)[0]             # Vorhersage für den nächsten Tag (prozentuale Veränderung)
            # Umwandlung der prozentualen Veränderung in einen absoluten Close-Wert
            predicted_close = last_known_close_value * (1 + predicted_pct_change / 100)
            predicted_close_values.append(predicted_close)
            
            # Aktualisieren des letzten bekannten Close-Werts für die nächste Vorhersage
            #last_known_close_value = back_transform_test_data["close"].iloc[i]
            last_known_close_value = predicted_close
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
    
    ### Ergänzungen von Train Model, um Train und Test Fehler zu vergleichen
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
    ''' 
        Ein einfaches statistisches Modell zur Vorhersage einer abhängigen Variable (y, hier close) basierend auf einer oder mehreren unabhängigen Variablen (X, die restlichen Variablen).
        -> Einsatzbereich gut für Fälle, in denen eine lineare Beziehung zwischen den Variablen angenommen wird.

        # 
        Variance Score: Wie gut das LR Model ist 
            -> 0:schlechtes Model,  1: super Model -> genau 1 overfitted
        
        Coefficients: Einfluss der einzelnen Variablen auf die Zielvariable
    '''
    def __init__(self, pred_length=1):
        super().__init__(LinearRegression(), pred_length=pred_length)
    
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
    ''' 
        Ein Ensemble-Modell, das aus vielen unabhängingen Entscheidungsbäumen besteht und durchschnittliche Vorhersagen liefert.
        -> Einsatzgebiet: Effektiv in Situationen, wo hohe Genauigkeit und die Berücksichtigung komplexer Zusammenhänge zwischen Variablen gefragt sind.

        Ein Ensemble-Modell kombiniert die Vorhersagen mehrerer verschiedener Modelle (hier Bäume), um die Gesamtleistung zu verbessern. 
        Statt sich auf ein einzelnes Modell (einem Baum) zu verlassen, nutzt es die Stärken mehrerer Modelle (Bäume -> Wald), um genauere und robustere Ergebnisse zu erzielen.
    
        Dabei nutzt ein RF-Modell das Bagging (Bootsstap-Stichproben), um unabhängiger Bäume zu erstellen und zu verwenden. 
        Das Bagging hilft dabei die Varianz zu verringern, da die Modelle jeweils unabhänging voneinander erstellt werden.

        #
        Feature Importances: ist die Wichtigkeit der einzelnen Variablen
    '''
    hyperparameters = {"n_estimators": 10, "max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 50, "max_features": "auto", "random_state": 11}

    def __init__(self, pred_length=1):
        super().__init__(RandomForestRegressor(**RandomForestModel.hyperparameters), pred_length=pred_length)
    
    def get_feature_importances(self):  
        feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return feature_importances

############### GBM ##################### 
class GradientBoostingModel(BaseModel):
    '''
        Ein fortgeschrittenes (verbessertes) Ensemble-Modell, das Fehler sequenziell durch aufeinanderfolgende Bäume korrigiert.
        -> Bäume werden nacheinander erstellt, wobei jeder Baum versucht, die Fehler des vorherigen Baumes zu korrigieren.
           (Im Vergleich zum RF, der hier zufällige Bäume erstellt (Bagging /Bootsstap); -> GBM nicht)

        #
        Feature Importances: ist die Wichtigkeit der einzelnen Variablen
    '''
    hyperparameters = {"n_estimators": 10, "max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 50, "max_features": "auto", "random_state": 11}

    def __init__(self, random_state=11, pred_length=1):
        super().__init__(GradientBoostingRegressor(**GradientBoostingModel.hyperparameters), pred_length=pred_length)

    def get_feature_importances(self):  
        feature_importances = {
            "Feature Importances": self.model.feature_importances_
        }
        return feature_importances

############### SVM ##################### 
class SVMModel(BaseModel):
    '''
        Support Vector Maschine (SVM) ist ein leistungsfähiges und flexibles Klassifikationsmodell.
        
        Es versucht, eine Hyperebene zu finden, die die Klassen in einem Merkmalsraum am besten trennt. 
        Dabei maximiert sie den Abstand zwischen den Datenpunkten der verschiedenen Klassen, die am nächsten an dieser Hyperebene liegen (-> Support-Vektoren).
        
        -> SVM eignet sich sehr gut für komplexe Klassifikationsaufgaben mit einer klaren Trennung & funktioniert gut bei linearen und nichtlinearen Beziehungen. 

        -> Gedanke: Trennung zwischen Aktienpunkte die Steigung bzw die Sinken (nicht steigen).
    
        #
        Support Vektora: wie das SVM-Modell die Klassen trennt
        
        n_support: Anzahl der Support-Vektoren für jede Klasse 
            -> Model mit hoher Anzahl kann overfit sein

        dual_coef: gibt die Koeffizienten der Support-Vektoren in der dualen Darstellung zurück
            -> ein Maß, wie stark jeder Support-Vektor die Position der Entscheidungsgrenze beeinflusst
    '''
    hyperparameters = {"kernel": 'rbf', "C": 1.0, "epsilon": 0.1}

    def __init__(self, pred_length=1):
        super().__init__(SVR(**SVMModel.hyperparameters), pred_length=pred_length)
    
    def get_support_vectors(self):    
        return self.model.support_vectors_

    def get_n_support(self):            
        return self.model.n_support_

    def get_dual_coef(self):            
        return self.model.dual_coef_   