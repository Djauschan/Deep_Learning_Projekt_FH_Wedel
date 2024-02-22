import pandas as pd

class DataSplitter:
    def __init__(self, data):
        self.data = data
        self.train_data = None
        self.test_data = None

    def split_by_ratio(self, split_ratio=0.8):
        """
            Split der Daten in Train und Test nach einem prozentualen Verhältnis.
            Hier 80-20 -> 80 Train
        """
        split_index = int(len(self.data) * split_ratio)
        self.train_data = self.data.iloc[:split_index]
        self.test_data = self.data.iloc[split_index:]

    def split_by_date(self, split_date):
        """
            Split der Daten in Train und Test an einem spezifischen Datum.
            Zeitraum wird im Hauptskript definiert.
        """
        self.train_data = self.data[self.data.index < split_date]
        self.test_data = self.data[self.data.index >= split_date]
    
    #daily model
    def split_by_date_lag20d(self, split_date):
        """
            Split der Daten in Train und Test an einem spezifischen Datum.
            Zeitraum wird im Hauptskript definiert.

            Testdaten beginnen 20 Indizes vor dem split_date 
            -> warum: damit die Lag_features noch die Infos erhalten der letzten 20 Tage und nicht rausgeschmissen werden weil NaN
            #udemy 3.25
        """
        self.train_data = self.data[self.data.index < split_date]
        self.test_data = self.data[self.data.index >= split_date - pd.offsets.BusinessDay(20)]

    #hour model
    def split_by_hour_7h(self, split_date):
        """
            Split der Daten in Train und Test an einem spezifischen Datum.
            Zeitraum wird im Hauptskript definiert.

            Testdaten beginnen 7 Stunden vor dem split_date 
            -> warum: damit die Lag_features noch die Infos erhalten der letzten 7 Stunden und nicht rausgeschmissen werden weil NaN
            #udemy 3.25
        """
        self.train_data = self.data[self.data.index < split_date]
        #self.test_data = self.data[self.data.index >= split_date - pd.Timedelta(hours=7)]
        self.test_data = self.data[self.data.index >= split_date - pd.DateOffset(hours=7)]
        #self.test_data = self.data[self.data.index >= split_date - pd.offsets.DateOffset(hours=7)]

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

