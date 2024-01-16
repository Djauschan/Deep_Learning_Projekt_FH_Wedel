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
    
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        merged_df = pd.DataFrame(business_days_series, columns=['ds'])

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)

                symbol = filename[:-4]  

                prediction.rename(columns={'AutoARIMA': symbol}, inplace=True)
                prediction['ds']=business_days_series


                merged_df = pd.merge(merged_df, prediction[['ds', symbol]], on='ds', how='left')
    

        return merged_df
    
    
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
    
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        merged_df = pd.DataFrame(business_days_series, columns=['ds'])

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)

                symbol = filename[:-4]  

                prediction.rename(columns={'AutoETS': symbol}, inplace=True)
                prediction['ds']=business_days_series


                merged_df = pd.merge(merged_df, prediction[['ds', symbol]], on='ds', how='left')
    

        return merged_df
   
    
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
    
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        merged_df = pd.DataFrame(business_days_series, columns=['ds'])

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)

                symbol = filename[:-4]  

                prediction.rename(columns={'HistoricAverage': symbol}, inplace=True)
                prediction['ds']=business_days_series


                merged_df = pd.merge(merged_df, prediction[['ds', symbol]], on='ds', how='left')
    

        return merged_df
    
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
    
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        merged_df = pd.DataFrame(business_days_series, columns=['ds'])

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)

                symbol = filename[:-4]  

                prediction.rename(columns={'AutoTheta': symbol}, inplace=True)
                prediction['ds']=business_days_series


                merged_df = pd.merge(merged_df, prediction[['ds', symbol]], on='ds', how='left')
    

        return merged_df
    
    
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
    
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        merged_df = pd.DataFrame(business_days_series, columns=['ds'])

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)

                symbol = filename[:-4]  

                prediction.rename(columns={'Naive': symbol}, inplace=True)
                prediction['ds']=business_days_series


                merged_df = pd.merge(merged_df, prediction[['ds', symbol]], on='ds', how='left')
    

        return merged_df
    
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
    
        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(start=timestamp_start, end=timestamp_end)

        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        merged_df = pd.DataFrame(business_days_series, columns=['ds'])

        for filename in os.listdir(path):
            if filename.endswith(".pkl"):  
                model_path = os.path.join(path, filename)
                model = StatsForecast.load(model_path)
                prediction = model.predict(days_difference)

                symbol = filename[:-4]  

                prediction.rename(columns={'WindowAverage': symbol}, inplace=True)
                prediction['ds']=business_days_series


                merged_df = pd.merge(merged_df, prediction[['ds', symbol]], on='ds', how='left')
    

        return merged_df
    
    
    def load_data(self) -> None:
        pass
    
    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        pass