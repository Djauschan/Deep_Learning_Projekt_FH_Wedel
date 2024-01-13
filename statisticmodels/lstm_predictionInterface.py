from abstract_model import AbstractModel
import statisticmodels.lstmMVPV1 as lstmMVPV1
from tensorflow.keras.models import load_model
import pandas as pd 
import os


class LstmInterface(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelsLSTM/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)

        for filename in os.listdir(path):
            if filename.endswith(".h5"):  
                model_path = os.path.join(path, filename)
                # LSTM modell laden
                model = lstmMVPV1.load_model(model_path)                
                prediction = model.predict_x_days(days_difference)  

                prediction['ds'] =  business_days_series
                prediction['symbol'] = filename[:-4]

                all_predictions = pd.concat([all_predictions, prediction], ignore_index=True)

        return all_predictions 
    
    def load_data(self) -> None:
        pass
    
    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        pass
