from pandas.core.api import DataFrame as DataFrame
from abstract_model import AbstractModel
from statsforecast import StatsForecast
import pandas as pd 
import os


class ArimaInterface(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelsARIMA/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
    

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)  

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


class ETSInterface(AbstractModel):
    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelsETS/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
    

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)  

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


class historicAverageInterface(AbstractModel):
    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelshistoricAverage/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
    

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)  

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

class ThetaInterface(AbstractModel):
    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelsTheta/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
    

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)  

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

class NaiveInterface(AbstractModel):
    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelsNaive/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
    

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)  

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

class WindowAverageInterface(AbstractModel):
    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> DataFrame:
        
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/models/savedModelsWindowAverage/"
        all_predictions = pd.DataFrame()
        
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
    

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)  

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