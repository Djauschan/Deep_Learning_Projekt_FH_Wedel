import numpy as np


class AverageService:
    """
        calc the current avarage of the timeseries and add it as a feature
    """

    # 50200, 11, 4 => 50200, 11, 5 =( 5 <- 1 feature dazu...)
    @staticmethod
    def calcAvg(arr: np.ndarray):
        """
            extends the numpy array of all timeseries to add 1 more feature
        """
        newArr = np.zeros((len(arr), len(arr[0]), len(arr[0][0])+1))
        len_df = len(arr)
        i: int = 0
        while i < len_df:
            newArr[i] = AverageService.calcAvgOnSingleTs(arr[i])
            i += 1
        return newArr

    @staticmethod
    def calcAvgOnSingleTs(timeSeries: np.ndarray):
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
