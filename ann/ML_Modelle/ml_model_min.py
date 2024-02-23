import pandas as pd
import numpy as np
import pickle
import os

#modelle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

#skript
#from abstract_model import AbstractModel

##################### Basis Model#######################
class BaseModel():
    def __init__(self, model):
        super().__init__()
        self.model = model
    
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
    def __init__(self):
        super().__init__(LinearRegression())

    def fitandsave(self, X_train, y_train, symbol):
        self.model.fit(X_train, y_train)
        path = f"ML_Modelle/saved_pkl_model_min/LR-Model/{symbol}_lr_model.pkl"
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
    
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
    hyperparameters = {"n_estimators": 1, "max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "auto", "random_state": 11}

    def __init__(self):
        super().__init__(RandomForestRegressor(**RandomForestModel.hyperparameters))

    def fitandsave(self, X_train, y_train, symbol):
        self.model.fit(X_train, y_train)
        path = f"ML_Modelle/saved_pkl_model_min/RF-Model/{symbol}_rf_model.pkl"
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

############### GBM ##################### 
class GradientBoostingModel(BaseModel):
    '''
        Ein fortgeschrittenes (verbessertes) Ensemble-Modell, das Fehler sequenziell durch aufeinanderfolgende Bäume korrigiert.
        -> Bäume werden nacheinander erstellt, wobei jeder Baum versucht, die Fehler des vorherigen Baumes zu korrigieren.
           (Im Vergleich zum RF, der hier zufällige Bäume erstellt (Bagging /Bootsstap); -> GBM nicht)

        #
        Feature Importances: ist die Wichtigkeit der einzelnen Variablen
    '''
    hyperparameters = {"n_estimators": 1, "max_depth": 1, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "auto", "random_state": 11}

    def __init__(self):
        super().__init__(GradientBoostingRegressor(**GradientBoostingModel.hyperparameters))

    def fitandsave(self, X_train, y_train, symbol):
        self.model.fit(X_train, y_train)
        path = f"ML_Modelle/saved_pkl_model_min/GBM-Model/{symbol}_gbm_model.pkl"
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

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
    hyperparameters = {"kernel": 'rbf', "C": 0.1, "epsilon": 0.5}

    def __init__(self):
        super().__init__(SVR(**SVMModel.hyperparameters))
    
    def fitandsave(self, X_train, y_train, symbol):
        self.model.fit(X_train, y_train)
        path = f"ML_Modelle/saved_pkl_model_min/SVM-Model/{symbol}_svm_model.pkl"
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
