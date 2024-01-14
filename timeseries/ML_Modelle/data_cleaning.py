import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data.set_index("timestamp")
        self.raw_data = self.raw_data.drop(["type"], axis=1)
        self.raw_data = self.raw_data.drop(["symbol"], axis=1)
        self.difference = []

        # Die Daten auf offizielle Handelszeiten setzen
        self.raw_data = self.raw_data.between_time('09:30', '16:00')

        # Erstellen verschiedener Zeitintervall-DataFrames
        self.df_normal = self.normal()  
        self.df_hourly = self.hourly()
        self.df_minute = self.minute()
        self.df_daily = self.daily()
        self.df_busi = self.busi()
    
    def normal(self):
        return self.raw_data
    
    def hourly(self):
        hourly = self.raw_data.resample('H').mean()
        return hourly.ffill()
    
    def minute(self):
        minute = self.raw_data.resample('min').mean()
        minute = minute.between_time('09:30', '16:00')
        return minute.ffill()
    
    def daily(self):
        daily = self.raw_data.between_time('16:00:00', '16:00:00')
        daily = daily.resample('D').mean()
        return daily.ffill()
    
    def busi(self):
        busi = self.raw_data.between_time('16:00:00', '16:00:00')
        busi = busi.resample('B').mean()
        return busi.ffill()