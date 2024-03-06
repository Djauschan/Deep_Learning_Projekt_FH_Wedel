import numpy as np


class differencingService:

    def __init__(self, ENHANCE_DIFFERENCE):
        self.ENHANCE_DIFFERENCE = ENHANCE_DIFFERENCE

    def transformSeriesAndLabel(self, data, label):
        i = 0
        len_data = len(data)
        # Y = anz daten; anz ele in  zeitreihe, anz features in einem ele
        differencedData = np.zeros((len(data), len(data[0]) - 1, len(data[0][0])))
        while i < len_data:
            differencedData[i], label[i] = self._transformSingleSeries(data[i], label[i])
            i = i + 1

        data = differencedData
        return data, label

    '''
        transforms a given timeseries into series of percentage differences 
    '''

    def _transformSingleSeries(self, series, label):
        # dimension = length of series -1; anz features. Differencing removes first ele
        differenceArray = np.zeros((len(series) - 1, len(series[0])))
        i = 1
        curr = series[i]
        while i < len(series):
            prev = series[i - 1]
            curr = series[i]
            differenceArray[i - 1] = self._calcPercentageDifference(prev, curr)
            i = i + 1

        # the first = [0] element in the row, is the "Open" = "the Label" the last of the series
        currentLabelVal = curr[0]
        labelVal = self._calcPercentageDifference(currentLabelVal, label)
        return differenceArray, labelVal

    def _calcPercentageDifference(self, baseVal, newVal):
        val = 100
        if self.ENHANCE_DIFFERENCE:
            val = 1000

        return ((newVal - baseVal) / baseVal) * val

    def transformSingleSeries(self, data):
        len_data = len(data)
        # Y = anz daten; anz ele in  zeitreihe, anz features in einem ele
        differencedData = np.zeros((len_data, len(data[0])))
        currRow = data[0]
        i = 1
        while i < len_data:
            prevRow = data[i - 1]
            currRow = data[i]
            x = 0
            while x < len(prevRow):
                prevValInRow = prevRow[x]
                currentValInRow = currRow[x]
                diffVal = self._calcPercentageDifference(prevValInRow, currentValInRow)
                differencedData[i][x] = diffVal
                x += 1
            i = i + 1

        data = differencedData
        return data

    @staticmethod
    def calcThePriceFromChange(ENHANCE_DIFFERENCE: bool, y_change, endPrice):
        # because the differencing is in 1000, modelled trained that way
        # ((newVal - baseVal)/baseVal) * val=1000
        val = 100
        if ENHANCE_DIFFERENCE:
            val = 1000

        return round((1 + (y_change / val)) * endPrice, 2)
