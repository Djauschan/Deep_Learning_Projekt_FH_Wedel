from datetime import timedelta

import pandas as pd


class TimeModificationService:
    """
        service that executes time operations
    """

    def __init__(self):
        pass

    def transformTimestap(self, data, dropDateTime: bool):
        return self._dataAugmentationAddContinuousMinuteOfDay(data, dropDateTime)

    def _dataAugmentationAddContinuousMinuteOfDay(self, data, dropDateTime: bool):
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
    def getNextDay10am(date: pd.Timestamp) -> pd.Timestamp:
        startDate: pd.Timestamp = pd.Timestamp(year=date.year, month=date.month, day=date.day + 1,
                                               hour=10, minute=00, second=0)
        return startDate

    @staticmethod
    def getSameDay8pm(date: pd.Timestamp) -> pd.Timestamp:
        startDate: pd.Timestamp = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                                               hour=20, minute=00, second=0)
        return startDate

    @staticmethod
    def getSameDay10am(date: pd.Timestamp) -> pd.Timestamp:
        startDate: pd.Timestamp = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                                               hour=10, minute=0, second=0)
        return startDate

    """
    @staticmethod
    def reArrangeDateTimeList(dateTimeList: list):
        toReturnList = []
        TRADING_START = 10
        TRADING_END = 20
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
                if currentRawTime.hour > TRADING_END or currentRawTime.hour < TRADING_START:
                    currentRawTime = TimeModificationService.getNextDayVal(currentRawTime, 1)
                toReturnList.append(currentRawTime)

            i += 1
        return toReturnList """

    @staticmethod
    def reArrangeDateTimeList(dateTimeList: list):
        toReturnList = []
        TRADING_START = 10
        TRADING_END = 20
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
                if currentRawTime.hour > TRADING_END or currentRawTime.hour < TRADING_START:
                    currentRawTime = TimeModificationService.getNextDayVal(currentRawTime, 1)
                toReturnList.append(currentRawTime)

            i += 1
        return toReturnList

    @staticmethod
    def getNextDayVal(previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        nextDay: pd.Timestamp = previousDayDateTime + timedelta(days=countOfDay)
        nextDayStartDay: pd.Timestamp = pd.Timestamp(year=nextDay.year, month=nextDay.month, day=nextDay.day,
                                                     hour=10, minute=00, second=0)
        return nextDayStartDay

    @staticmethod
    def getPreviousDayVal(previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        prevDay: pd.Timestamp = previousDayDateTime - timedelta(days=countOfDay)
        prevDayStartOfDay: pd.Timestamp = pd.Timestamp(year=prevDay.year, month=prevDay.month, day=prevDay.day,
                                                       hour=16, minute=00, second=0)
        return prevDayStartOfDay
