import os

import numpy as np
import pandas as pd
import torch


class importService:

    def __init__(self, path):
        self.path = path

    def loadFromNpy(self, fileName):
        return np.load(os.path.join(self.path, fileName))


    def loadTimeSeriesFromNumpy(self, featureList):
        """
            param:
                liste alles features und paths:
            return:
                array: (länge_aller_ts, länge_feature, länge_sinlge_ts, länge_sinlge_ts)
        """
        firstData = np.load(os.path.join(self.path, 'Open__GAF_DATA.npy'))
        dimensions_shape = firstData.shape
        #FROM shape: (anz_feature, length_all_ts, length_single_ts, length_single_ts)
        tmpCompleteData = np.zeros((len(featureList), dimensions_shape[0], dimensions_shape[1], dimensions_shape[2]))
        #NEED shape (length_all_ts. anz_feature, length_single_ts, length_single_test)
        #completeData = np.zeros((dimensions_shape[0], len(featureList), dimensions_shape[1], dimensions_shape[2]))

        #todo schöner bauen
        i = 0
        for filename in os.listdir(self.path):
            if filename in featureList:
                data = np.load(os.path.join(self.path, filename))
                tmpCompleteData[i] = data
                i = i + 1

        completeData = np.transpose(tmpCompleteData, (1, 0, 2, 3))
        return completeData

    def loadLabelsFromNumpy(self):
        labels = []
        for filename in os.listdir(self.path):
            if filename.endswith("_LABELS.npy"):
                labels = np.load(os.path.join(self.path, filename))

        return labels

    def loadDataFromFile(self, start_date, end_date, AMOUNT_OF_DATA, rsc_completePath, DATA_COLUMNS, COLUMNS_TO_LOAD):

        print('LOAD PURE DATA IN MEMORY')
        #Data must not contain other coulmn but the "COLUMNS_TO_LOAD" in the correct Order
        df = pd.read_csv(rsc_completePath, sep=",", names=DATA_COLUMNS, index_col=False)

        toRemove = []
        for col in df:
            if col not in COLUMNS_TO_LOAD:
                toRemove.append(col)

        data = df.drop(toRemove, axis=1)

        data['DateTime'] = pd.to_datetime(data['DateTime'])
        # Filtering based on the datetime range
        if start_date != "*" and end_date != "*":
            data = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)]

        print(len(data))
        len_data = int((len(data)*AMOUNT_OF_DATA))
        print(len_data)
        data = data.iloc[:int((len(data)*AMOUNT_OF_DATA))]
        print(len(data))
        print(data)
        return data

    def loadModel(self, full_path):
        model = torch.jit.load(full_path)
        model.eval()
        return model