import numpy as np


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

    def addPosixTimeStamp(self, df_DateTimeColumn):
        # Convert python time to posix time in minutes
        return df_DateTimeColumn.apply(lambda x: (x.timestamp()) / 60)


