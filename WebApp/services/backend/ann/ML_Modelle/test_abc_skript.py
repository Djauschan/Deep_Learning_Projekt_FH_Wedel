import pandas as pd

#from ml_model_ab import BaseModel, LinearRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel

from ML_PredictionInterface import ABC_LinearRegressionModel, ABC_RandomForestModel, ABC_GradientBoostingModel, ABC_SVMModel

#test LR
lr_model = ABC_LinearRegressionModel()
lr_predictions = lr_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  
print("lr_predictions")
print(lr_predictions)

#test RF
rf_model = ABC_RandomForestModel()
rf_predictions = rf_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  
print("rf_predictions")
print(rf_predictions)

#test GBM
gbm_model = ABC_GradientBoostingModel()
gbm_predictions = gbm_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  
print("gbm_predictions")
print(gbm_predictions)

#test SVM
svm_model = ABC_SVMModel()
svm_predictions = svm_model.predict(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 2)  
print("svm_predictions")
print(svm_predictions)


