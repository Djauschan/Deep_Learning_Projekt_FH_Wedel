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
        # data['posixMinute'] = conti_minute_arr
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
    def calcDateTimeFromStartDateAndInterval(endTimeSeries: pd.Timestamp, interval, horizon) -> pd.Timestamp:
        difference = interval * horizon
        nextDateTime: pd.Timestamp = endTimeSeries + timedelta(minutes=difference)
        return nextDateTime

    @staticmethod
    def reArrangeDateTimeList(dateTimeList: list):
        toReturnList = []
        TRADING_START = 8
        TRADING_END = 18
        firstEle = dateTimeList[0]
        if firstEle.hour > TRADING_END or firstEle.hour < TRADING_START:
            firstEle = TimeModificationService.getNextDayVal(firstEle, 1)
        toReturnList.append(firstEle)
        i = 1
        while i < len(dateTimeList):
            previousTime = toReturnList[i - 1]  # 9:30 | 15:45
            previousRawTime = dateTimeList[i - 1]  # 9:30 | 15:45
            currentRawTime = dateTimeList[i]
            timeDiff = currentRawTime - previousRawTime
            if previousTime != previousRawTime:
                modifiedTime = previousTime + timeDiff
                if modifiedTime.hour > TRADING_END or modifiedTime.hour < TRADING_START:
                    modifiedTime = TimeModificationService.getNextDayVal(modifiedTime, 1)
                toReturnList.append(modifiedTime)
            else:
                toReturnList.append(currentRawTime)

            i += 1
        return toReturnList

    @staticmethod
    def getNextDayVal(previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        nextDay: pd.Timestamp = previousDayDateTime + timedelta(days=countOfDay)
        nextDayStartDay: pd.Timestamp = pd.Timestamp(year=nextDay.year, month=nextDay.month, day=nextDay.day,
                                                     hour=9, minute=30, second=0)
        return nextDayStartDay

    @staticmethod
    def getPreviousDayVal(previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        prevDay: pd.Timestamp = previousDayDateTime - timedelta(days=countOfDay)
        prevDayStartOfDay: pd.Timestamp = pd.Timestamp(year=prevDay.year, month=prevDay.month, day=prevDay.day,
                                                       hour=16, minute=00, second=0)
        return prevDayStartOfDay
