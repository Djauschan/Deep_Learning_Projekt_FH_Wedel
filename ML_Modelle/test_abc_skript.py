# Importieren der erforderlichen Bibliotheken
import pandas as pd

#from ml_model_ab import BaseModel, LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

from ML_PredictionInterface import ABC_LinearRegressionModel, ABC_RandomForestModel, ABC_GradientBoostingModel, ABC_SVMModel

#test LR
lr_model = ABC_LinearRegressionModel()
lr_model.load_model()

lr_predictions = lr_model.predict(pd.Timestamp('2021-01-04'), pd.Timestamp('2021-01-06'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("lr_predictions")
print(lr_predictions)

#test RF
rf_model = ABC_RandomForestModel()
rf_model.load_model()

rf_predictions = rf_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("rf_predictions")
print(rf_predictions)

#test LR
gbm_model = ABC_GradientBoostingModel()
gbm_model.load_model()

gbm_predictions = gbm_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("gbm_predictions")
print(gbm_predictions)

#test LR
svm_model = ABC_SVMModel()
svm_model.load_model()

svm_predictions = svm_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  # Vorhersagen für die Zeit nach dem 3. Januar
print("svm_predictions")
print(svm_predictions)



