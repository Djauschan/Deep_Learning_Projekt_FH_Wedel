import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, raw_data: pd.DataFrame):
        """
        initializes the DataCleaner class
        args:
        raw_data(pd.Dataframe): data with missing time stamps
        """
        self.raw_data = raw_data.set_index("timestamp").drop(["type"], axis=1)
        self.difference = []
        self.nunique_dates = len(np.unique(self.raw_data.index.date))
        self.days_to_remove=[]
        self.df_at_16 = self.closeCleanedData()
        #self.df_at_09 = self.openCleanedData()
        

    def calculate_diff(self, time:str, intervall):
        """
        calculates the how many time stamps are missing
        args:
        time(str): string that indicates, whether 16.00 or 9.30 is about to be filled
        intervall: interval to check whether there are any data for these days
        """
        #Creating the difference between all days from the data set and days with the needed timestamp
        #missing timestamps will be recognized here
        data = self.raw_data.between_time(time, time)
        small = data.index.date
        big = self.raw_data.index.date
        self.difference = list(set(big) - set(small))

        
        for date in self.difference:
            # checking if there are any available timestamps within a 2 hour range from opening or close
            #if there are no available timestamps day will be removed
            time_range = pd.date_range(
                str(date) + intervall[0], str(date) + intervall[1], freq="1T"
            )
            if not self.raw_data.index.isin(time_range).any():
                self.days_to_remove.append(date)

        self.difference = [i for i in self.difference if i not in self.days_to_remove]

    def datafiller(self, fillTime):
        """
        function to fill the missing time stamps
        args:
        fillTime: either 16:00:00 or 09:30:00

        """
        #filling missing timestamps with none beforehand
        fill_dates = pd.to_datetime(np.array([str(i) + fillTime for i in self.difference]))
        for i in fill_dates:
            self.raw_data.loc[i] = None

        self.raw_data = self.raw_data.sort_index()
        tempTicker= np.array(self.raw_data['symbol'])[0]
        self.raw_data=self.raw_data.drop(['symbol'], axis=1)
        interCounter, ffCounter, bfCounter = 0,0,0
        #indexing = list(self.raw_data.index)

        #filling missing time stamps
        for null_value in fill_dates: 
            nullValueIndex=  null_value_index = self.raw_data.index.get_indexer([null_value])[0]
            lowerBound, upperBound=nullValueIndex - 1,nullValueIndex+2
            temp_df = self.raw_data.iloc[lowerBound: upperBound]
            #nullValueIndex=self.raw_data.index.get_indexer([null_value])[0]  indexing.index(nullValues) 
            


            timeDiff = (temp_df.index[-1] - temp_df.index[0]) / pd.Timedelta(minutes=1)
            if timeDiff<=5:
                temp_df= temp_df.interpolate()
                self.raw_data.iloc[nullValueIndex]=temp_df.iloc[1]
                interCounter+=1
            else: 
                if ((temp_df.index[-1] - null_value)/pd.Timedelta(minutes=1))<((null_value - temp_df.index[0])/pd.Timedelta(minutes=1)):
                    temp_df=temp_df.bfill()
                    self.raw_data.iloc[nullValueIndex]=temp_df.iloc[1]
                    bfCounter+=1
                else: 
                    temp_df=temp_df.ffill()
                    self.raw_data.iloc[nullValueIndex]=temp_df.iloc[1]
                    ffCounter+=1
        print('Das sind die Ergebnisse fürs Preprocessing ')
        print(interCounter)
        print(ffCounter)
        print(bfCounter)
        self.raw_data= self.raw_data.assign(symbol=tempTicker)
        if fillTime==' 16:00:00':
            return self.raw_data.between_time("16:00", "16:00")
        else:
            #geht hier noch deutlich eleganter, kann man später nochmal drüber nachdenken!
            check=self.raw_data.between_time("09:30", "09:30")
            return check.drop([str(i)+' 09:30:00' for i in self.days_to_remove])

    def closeCleanedData(self):
        self.calculate_diff("16:00", [" 15:00:00", " 16:30:00"])
        return self.datafiller(" 16:00:00")

    def openCleanedData(self):
        self.calculate_diff("09:30", [" 08:30:00", " 10:00:00"])
        return self.datafiller(" 09:30:00")

    def transformForNixtla(self, data:pd.DataFrame):
        transformedData=pd.DataFrame()
        transformedData['unique_id']=data['symbol']
        transformedData['ds']=data.index
        transformedData['y']=data['close']
        transformedData.reset_index(drop=True,inplace=True)
        return transformedData