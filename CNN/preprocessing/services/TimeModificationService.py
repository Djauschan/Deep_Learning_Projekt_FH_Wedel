from datetime import timedelta

import numpy as np
import pandas as pd


class TimeModificationService:

    def __init__(self):
        pass

    def transformTimestap(self, data, dropDateTime: bool):
        return self._dataAugmentationAddContiniousMinuteOfDay(data, dropDateTime)

    def _dataAugmentationAddContiniousMinuteOfDay(self, data, dropDateTime: bool):
        dateTimeArr = data['DateTime']
        conti_minute_arr = self.addPosixTimeStamp(dateTimeArr)
        #data['posixMinute'] = conti_minute_arr
        data.insert(loc=0, column='posixMinute', value=conti_minute_arr)
        if dropDateTime:
            data.drop(columns=['DateTime'], inplace=True)

        print(data.columns)
        return data

    @staticmethod
    def addPosixTimeStamp(df_DateTimeColumn):
        # Convert python time to posix time in minutes
        return df_DateTimeColumn.apply(lambda x: (x.timestamp()) / 60)

    @staticmethod
    def calcDateTimeFromStartDateAndInterval(startDate: pd.Timestamp, interval, horizon) -> pd.Timestamp:
        difference = interval * horizon
        nextDateTime: pd.Timestamp = startDate + timedelta(minutes=difference)
        return nextDateTime

    @staticmethod
    def getNextDayVal(previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        nextDay: pd.Timestamp = previousDayDateTime + timedelta(days=countOfDay)
        nextDayStartDay: pd.Timestamp = pd.Timestamp(year=nextDay.year, month=nextDay.month, day=nextDay.day,
                                                     hour=9, minute=30, second=0)
        return nextDayStartDay

    @staticmethod
    def getPriviousDayVal(previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        prevDay: pd.Timestamp = previousDayDateTime - timedelta(days=countOfDay)
        prevDayStartOfDay: pd.Timestamp = pd.Timestamp(year=prevDay.year, month=prevDay.month, day=prevDay.day,
                                                       hour=9, minute=30, second=0)
        return prevDayStartOfDay


