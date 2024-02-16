'''
    Normalisieren im Bezug auf minimum von Series oder gesamte Datengrundlage
'''
import numpy as np


class NormalisationService:
    """
        class to normalize a dataframe
    """

    def __init__(self):
        pass

    '''
        normalizes data into a scale from -1 -> +1
    '''

    def normMinusPlusOne(self, data: np.ndarray):
        """
            normalisiert in jeder einzelnen zeitreihe #richtig
            für jedes feature von -1, +1

            praman:
                data: dimension (anz_aller_ts, länge_single_ts, features)
            return:
                data: dimension (anz_aller_ts, länge_single_ts, features), nur normalisiert
        """
        toReturn = data.copy()
        print('normMinusPlusOne DATA')
        i = 0
        len_data = len(data)
        while i < len_data:
            singleSeriesData = data[i].transpose()
            x = 0
            while x < len(singleSeriesData):
                featureRow = singleSeriesData[x]
                _min = np.min(featureRow)
                _max = np.max(featureRow)
                featureRow = ((featureRow - _max) + (featureRow - _min)) / (_max - _min)
                singleSeriesData[x] = featureRow
                x += 1

            toReturn[i] = singleSeriesData.transpose()
            i = i + 1

        return toReturn
