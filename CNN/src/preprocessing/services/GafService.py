from pyts.image import GramianAngularField
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

#matplotlib.use('TkAgg')


class gafService:

    def __init__(self):
        pass

    def createGAFfromMultivariateTimeSeries(self, data: list) -> np.ndarray:
        '''
            param:
                @data: eine List (elemente sind die features) mit np.Arrays die jeweils die Dimension
                                bspw = (3740, 5) = (länge alle ts, länge einer ts)
            return:
                @return = np.array (len_of_feature, anz_aller_ts, länge_einzelner_ts, länge_einzelner_ts)
        '''
        gafDataFeatureSeparated = np.zeros((len(data), len(data[0]), len(data[0][0]), len(data[0][0])))
        i = 0
        while i < len(data):
            gafDataFeatureSeparated[i] = self._createGAFfromUnivariateTimeSeries(data[i])
            i = i + 1

        return gafDataFeatureSeparated

    def _createGAFfromUnivariateTimeSeries(self, data: np.ndarray) -> np.ndarray:
        '''
            @data: Dimension (anz einzelner ts, len einer ts, anz feature je ts): bspw = (3740, 5)
        '''
        print("IN GAF SERVICE")
        len_series = len(data)
        gafData = np.zeros((len_series, data[0].size, data[0].size))
        i = 0
        while i < len_series:
            # np.concatenate(data[i]) führt von (5, 3) -> (15)
            # x_arr = np.concatenate(data[i]) #unnötig weil multivariat nicht geht
            x_arr = data[i]

            # logging prgress
            perc_done = int((i * 100) / len_series)
            if perc_done % 100 == 0:
                print('% done: ' + str(perc_done))

            gafData[i] = self._createSingleGAF(x_arr)
            i = i + 1
        return gafData

    def createSingleGAFfromMultivariateTimeSeries(self, data: np.ndarray) -> np.ndarray:
        """
            param:
                @data: numpy arr with dimensions of (length, features) => (30, 10)
                @return = np.array (len_of_feature, länge_ts, länge_ts)
            return:
                numpyArray with elements containing GAF data
        """
        data = data.transpose()  # (10, 30)
        gafDataFeatureSeparated = np.zeros(
            (len(data), len(data[0]), len(data[0]))
        )  # 10x30x30
        i = 0
        while i < len(data):
            gafDataFeatureSeparated[i] = self._createSingleGAF(data[i])
            i = i + 1

        return gafDataFeatureSeparated

    @staticmethod
    def _createSingleGAF(x_arr: np.ndarray):
        """
            method to create a single GAF, based on preprocessed numpy array
        """
        nanCount = np.count_nonzero(~np.isnan(x_arr))
        if nanCount < (len(x_arr)):
            print("NaN in _createSingleGAF")
            x_arr = np.nan_to_num(x_arr)

        # Compute Gramian angular fields
        gasf = GramianAngularField(method='summation')
        # takes (features, anz_timpstamp)
        # x_arr = [-0.16094916 -0.21443595 -0.14302942 -0.08939169 -0.12541136 -0.10782134]
        X_gasf = gasf.fit_transform([x_arr])

        '''
        X_gasf =
        [[[-0.95823126  0.14451426 -0.99999702 -0.14451428 -0.95746452,   -0.80345206],  
        [ 0.14451426  1.         -0.14210009 -1.         -0.42388929,   -0.70522997],  
        [-0.99999702 -0.14210009 -0.95961513  0.14210007 -0.83628839,   -0.60157087],  
        [-0.14451428 -1.          0.14210007  1.          0.42388927,    0.70522995],  
        [-0.95746452 -0.42388929 -0.83628839  0.42388927 -0.64063574,   -0.34319245],  
        [-0.80345206 -0.70522997 -0.60157087  0.70522995 -0.34319245,   -0.00530138]]]
        '''
        # print("After X_gasf")
        # print(X_gasf)
        # gadf = GramianAngularField(method='difference')
        # X_gadf = gadf.fit_transform(x_arr)
        return X_gasf

    @staticmethod
    def saveGAFimg(data, savePath):
        """
            creates a image from given GAF data to test / visualize
        """
        print('data: single imageData')
        # [[-0.95823126  0.14451426 -0.99999702 -0.14451428 -0.95746452 -0.80345206],
        # [ 0.14451426  1.         -0.14210009 -1.         -0.42388929 -0.70522997],
        # [-0.99999702 -0.14210009 -0.95961513  0.14210007 -0.83628839 -0.60157087],
        # [-0.14451428 -1.          0.14210007  1.          0.42388927  0.70522995],
        # [-0.95746452 -0.42388929 -0.83628839  0.42388927 -0.64063574 -0.34319245],
        # [-0.80345206 -0.70522997 -0.60157087  0.70522995 -0.34319245 -0.00530138]]
        plt.imshow(data, cmap='rainbow', origin='lower')
        plt.savefig(savePath)
