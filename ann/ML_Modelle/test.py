#### zum testen

################## prediction test daily
import pandas as pd
from ML_PredictionInterface_daily import ABC_RandomForestModel_daily, ABC_GradientBoostingModel_daily

############### RF ##################### 
rf_model = ABC_RandomForestModel_daily()

rf_predictions = rf_model.predict("AAPL", pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'), 1)
print("rf_predictions")
print(rf_predictions)

############### GBM ##################### 
gbm_model = ABC_GradientBoostingModel_daily()

gbm_predictions = gbm_model.predict("AAPL", pd.Timestamp('2021-01-10'), pd.Timestamp('2021-01-13'), 1)
print("gbm_predictions")
print(gbm_predictions)




#######
import pandas as pd
from ML_PredictionInterface_hour import ABC_LinearRegressionModel_hour, ABC_RandomForestModel_hour, ABC_GradientBoostingModel_hour, ABC_SVMModel_hour

############### RF ##################### 
rf_model = ABC_RandomForestModel_hour()

rf_predictions = rf_model.predict("AAPL", pd.Timestamp('2021-01-05 10:00:00'), pd.Timestamp('2021-01-05 14:00:00'), 2)
print("rf_predictions")
print(rf_predictions)

############### GBM ##################### 
gbm_model = ABC_GradientBoostingModel_hour()

gbm_predictions = gbm_model.predict("AAPL", pd.Timestamp('2021-01-05 10:00:00'), pd.Timestamp('2021-01-05 14:00:00'), 2)
print("gbm_predictions")
print(gbm_predictions)



#################### 
import pandas as pd
from ML_PredictionInterface_min import ABC_LinearRegressionModel_min, ABC_RandomForestModel_min, ABC_GradientBoostingModel_min, ABC_SVMModel_min

############### RF ##################### 
rf_model = ABC_RandomForestModel_min()

rf_predictions = rf_model.predict("AAPL", pd.Timestamp('2021-01-05 10:00:00'), pd.Timestamp('2021-01-05 10:40:00'), 20) #intervall alle 20 min
print("rf_predictions")
print(rf_predictions)

############### GBM ##################### 
gbm_model = ABC_GradientBoostingModel_min()

gbm_predictions = gbm_model.predict("AAPL", pd.Timestamp('2021-01-05 10:08:00'), pd.Timestamp('2021-01-05 10:49:00'), 20)
print("gbm_predictions")
print(gbm_predictions)


