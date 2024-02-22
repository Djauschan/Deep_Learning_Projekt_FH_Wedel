import numpy as np


class AverageService:
    """
        calc the current avarage of the timeseries and add it as a feature
    """

    # 50200, 11, 4 => 50200, 11, 5 =(5 <- 1 feature dazu...)
    def calcAvg(self, arr: np.ndarray):
        len_df = len(arr)
        i: int = 0
        while i < len_df:
            arr[i] = self._calcAvgOnSingleTs(arr[i])
        return arr

    def _calcAvgOnSingleTs(self, timeSeries: np.ndarray):
        """
            adds a row to the single timeseries numpy array
            containing the current avarage to each timestep
            respectively

            param:
                timeSeries = (x, y)
                    x: länge der timeSeries
                    y: anz der feature je timeseries

        """
        len_series = len(timeSeries)
        # 1 sinlge timeseries = (11, 4) = 11 entries, 4 features
        mvgAvgRow = np.zeros((len_series, 1))
        tmpSum = 0
        counter = 0

        while counter < len_series:
            # open value; 0 = open value index
            openVal = timeSeries[counter][0]
            tmpSum += openVal
            counter += 1
            mvgAvgRow[counter - 1] = round(tmpSum / counter, 3)

        extendedTimeSeries = np.concatenate((timeSeries, mvgAvgRow), axis=1)
        return extendedTimeSeries
