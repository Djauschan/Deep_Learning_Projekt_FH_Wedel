import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


class ExportService:
    '''
        param:
            @folderPath: base path to export to, checks if already exist otherwise create folder
    '''

    def __init__(self, folderPath):
        self.folderPath = folderPath
        if not os.path.exists(self.folderPath):
            os.makedirs(self.folderPath)

    def logEpochResult(self, arr, subFolder, pathToSave):
        folderPath = os.path.join(self.folderPath, subFolder)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(folderPath, pathToSave)
        df = pd.DataFrame(arr, columns=['epoch', 'modelOut', 'label', 'loss', 'running_avg'])
        df.to_csv(filePath, sep=';', index=False)
        return df

    def createLossPlot(self, df, LOSS_PLOT_INTERVAL, subFolder, pathToSave):
        folderPath = os.path.join(self.folderPath, subFolder)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(folderPath, pathToSave)
        df.plot(y='running_avg', kind="line", figsize=(20, 5), lw=1)
        plt.title("running_avg", color='red')
        plt.savefig(filePath + 'running_avg' + '.png')
        #
        df.plot(y='loss', kind="line", figsize=(20, 5), lw=1)
        plt.title("loss absolute", color='red')
        plt.savefig(filePath + 'absolute_loss' + '.png')
        #
        df = df.iloc[::LOSS_PLOT_INTERVAL, :]
        df.plot(y=['modelOut', 'label'], kind="line", figsize=(20, 5), lw=1, alpha=0.4)
        plt.title("soll_ist vergleich", color='red')
        plt.savefig(filePath + 'soll_ist' + '.png')
        #

    def storeGAFNumpyFeatureDataSeperatly(self, arr, FEATURE_LIST, pathToSave):
        """
            param:
                arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)

            speichert anz_features oft (len_allSeries, singleSeries, singleSeries) = 1000x20x20 das 9 mal gespeichert
        """
        i = 0
        for fArr in arr:
            self.storeNumpyTimeSeries(fArr, FEATURE_LIST[i] + '_' + pathToSave)
            i = i + 1
        print('All saved')

    def storeNumpyTimeSeries(self, arr, pathToSave):
        """
            #tod rename into "storeSeries"
            von storeTimeSeriesNumpyFeatureDataSeperatly
            wird x mal aufgerufen x=feature List

            param:
                arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)
        """
        filePath = os.path.join(self.folderPath, pathToSave)
        np.save(filePath, arr)

    def storeModel(self, model, modelName: str):
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        now = datetime.now()
        dateTimeAsStr = str(now.day) + str(now.month) + str(now.year) + "_" + str(now.hour) + str(now.minute)
        model_scripted.save(os.path.join(self.folderPath, modelName + dateTimeAsStr + '.pt'))  # Save
