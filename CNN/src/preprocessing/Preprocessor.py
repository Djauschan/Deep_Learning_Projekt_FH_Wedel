import sys
import time

import pandas as pd
import os
import numpy as np

from src.preprocessing.services.AverageService import AverageService
from src.preprocessing.services.DataMergerService import DataMergerService
from src.preprocessing.services.ExportService import ExportService
from src.preprocessing.services.GafService import gafService
from src.preprocessing.services.TimeBuildService import TimeSeriesBuilder
from src.preprocessing.services.ConfigService import ConfigService
from src.preprocessing.services.DifferencingService import differencingService
from src.preprocessing.services.TimeModificationService import TimeModificationService
from src.preprocessing.services.GafService import gafService
from src.preprocessing.services.ImportService import importService
from src.preprocessing.services.NormalisationService import NormalisationService


class Preprocessor:
    """
        a class to preprocess data defined a @ymlModelConfigPath
        and executes a collection of service methods on it (avaraging, normalisation..)
        and finally saves the preprocessed data as .npy file in defined folders
    """

    def __init__(self, ymlModelConfigPath):
        confService = ConfigService()
        self.modelParameters = confService.loadModelConfig(ymlModelConfigPath)
        self.TO_SAVE_RSC_FOLDER = self.modelParameters['TO_SAVE_RSC_FOLDER']
        self.TO_FIND_RSC_FOLDER = self.modelParameters['TO_FIND_RSC_FOLDER']
        self.RSC_DATA_FILES = self.modelParameters['RSC_DATA_FILES']
        self.TS_TOLERANCE = self.modelParameters['TOLERANCE']
        self.NEXT_DAY_RETRY_THRESHOLD = self.modelParameters['NEXT_DAY_RETRY_THRESHOLD']
        self.TS_INTERVAL = self.modelParameters['TIME_STEP_INTERVAL']
        self.TS_LENGTH = self.modelParameters['TIMESERIES_SEQUENCE_LEN']
        self.TS_AHEAD = self.modelParameters['TIMESTEPS_AHEAD']
        self.FEATURES = self.modelParameters['FEATURES']
        self.MVG_AVG = self.modelParameters['MVG_AVG']
        self.AMOUNT_OF_DATA = self.modelParameters['AMOUNT_OF_DATA']
        self.TIME_SPAN_BEGIN = self.modelParameters['TIME_SPAN_BEGIN']
        self.TIME_SPAN_END = self.modelParameters['TIME_SPAN_END']
        self.DATA_FEATURE_NAME = self.modelParameters['DATA_FEATURE_NAME']
        """
            OTHER FEATURES IN OTHER FILES:
        """
        self.TO_FIND_OTHER_FEATURE_RSC_FOLDER = self.modelParameters['TO_FIND_OTHER_FEATURE_RSC_FOLDER']
        self.OTHER_FEATURES_TO_LOAD = self.modelParameters['OTHER_FEATURES_TO_LOAD']
        self.ENHANCE_DIFFERENCE = self.modelParameters['ENHANCE_DIFFERENCE']

        """
            All Services Used during Preprocessing
        """
        self.featureDataMergeService = DataMergerService()
        self.timeSeriesBuilderService = TimeSeriesBuilder()
        self.timeModificationService = TimeModificationService()
        self.normalisationService = NormalisationService()
        self.averagingService = AverageService()
        self.differenceService = differencingService(self.ENHANCE_DIFFERENCE)
        self.GAFservice = gafService()

        # MAIN DATA & LABEL PREPROCESSING
        trainingDataPath = str(self.TS_LENGTH) + '_' + str(self.TS_INTERVAL) + '_' + str(self.TS_AHEAD) + '_' + \
            self.DATA_FEATURE_NAME + '_' + self.TIME_SPAN_BEGIN + '_' + self.TIME_SPAN_END
        sub_path = os.path.join(self.TO_SAVE_RSC_FOLDER, trainingDataPath)
        if not os.path.exists(self.TO_SAVE_RSC_FOLDER):
            os.makedirs(self.TO_SAVE_RSC_FOLDER)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

        for i in self.RSC_DATA_FILES:
            for k, v in i.items():
                kuerzel_folder_path = os.path.join(sub_path, k)
                if not os.path.exists(kuerzel_folder_path):
                    os.makedirs(kuerzel_folder_path)
                fileName = v[0]
                allDataColumns = v[1]
                column_featureName = v[2]
                label_name = v[3]
                imgDataPath = os.path.join(kuerzel_folder_path, 'ImageGafData')
                df = self.loadMainData(fileName, allDataColumns, column_featureName)
                """ 
                    In the meaning of Performance this architecture is horrible because
                    multiple steps of preprocessing could be done in 1 loop but are here
                    split in x loops for each service..
                """
                # merges the main data (the data with the label values)
                # with all the feature data (the etf, gold index...)
                df = self.getAndMergeFeatureDataWithMainData(df)
                data, labels = self.createSeries(df, self.FEATURES, label_name)
                # create a feature Row with avg vals for open
                data = self.averagingService.calcAvg(data)
                # All feature Data will be differenced
                data, labels = self.differenceService.transformSeriesAndLabel(data, labels)
                # Only the Data will be normalised
                data = self.normalisationService.normMinusPlusOne(data)
                #########################
                #### EXPORT THE DATA ####
                # self.exportLabelsToNpy(kuerzel_folder_path, labels) #no need to save TS
                # data in format (count_of_feature, anz vo ts, length single ts)
                countOfFeatures = len(self.FEATURES)
                if self.MVG_AVG:
                    countOfFeatures += 1

                featureShapedData = self.reshapeDataToFeatureList(data, countOfFeatures)
                # list(features) to -> np.array(len_of_feature, anz_aller_ts, länge_einzelner_ts, länge_einzelner_ts)
                gafData = self.GAFservice.createGAFfromMultivariateTimeSeries(featureShapedData)
                # swap shape of gafData
                # (features, datapoints, len_ts, len_ts) -> (datapoints, features, len_ts, len_Ts)
                # from (5, 5000, 10, 10) ->  (5000, 5, 10, 10)
                gafData = np.transpose(gafData, (1, 0, 2, 3))
                if not os.path.exists(imgDataPath):
                    os.makedirs(imgDataPath)

                np.save(os.path.join(imgDataPath, self.DATA_FEATURE_NAME + '.npy'), gafData)
                np.save(os.path.join(imgDataPath, 'LABELS' + '.npy'), labels)

                # Test Images for Visualisation
                # create imgaes for first 3 feautre to Visualize
                self.createSingleGafImg(gafData[0][0], imgDataPath + '0_gaf.png')
                self.createSingleGafImg(gafData[1][0], imgDataPath + '1_gaf.png')
                self.createSingleGafImg(gafData[2][0], imgDataPath + '2_gaf.png')
                # self.exportGafData(imgDataPath, gafData, self.FEATURES) deprecated

    def loadMainData(self, fileName, allDataColumns, column_featureName) -> pd.DataFrame:
        loadService = importService("")
        df = loadService.loadDataFromFile(self.TIME_SPAN_BEGIN, self.TIME_SPAN_END, self.AMOUNT_OF_DATA,
                                          self.TO_FIND_RSC_FOLDER + fileName, allDataColumns, column_featureName)

        df = self.timeModificationService.transformTimestap(df, False)
        return df

    def getAndMergeFeatureDataWithMainData(self, main_df: pd.DataFrame) -> pd.DataFrame:
        mergedDf = main_df
        for oData in self.OTHER_FEATURES_TO_LOAD:
            for k, v in oData.items():
                fileName = v[0]
                allColumns = v[1]
                column_featureName = v[2]
                loadService = importService("")
                feature_df = loadService.loadDataFromFile(self.TIME_SPAN_BEGIN, self.TIME_SPAN_END, self.AMOUNT_OF_DATA,
                                                          self.TO_FIND_OTHER_FEATURE_RSC_FOLDER + fileName, allColumns,
                                                          column_featureName)
                feature_df = self.timeModificationService.transformTimestap(feature_df, True)
                mergedDf = self.featureDataMergeService.mergeFeatureData(mergedDf, feature_df)

        return mergedDf

    def createSeries(self, data, FEATURES, LABEL):
        # designered length + 1 => differecing removes first element
        # returs: (a=5000, l=11, f=8) array;
        # a=anz der einzelnen TS; l= die länge jeder TS; f=die enthaltenen Feature
        data, labels = self.timeSeriesBuilderService.buildTimeSeries(data, self.TS_LENGTH + 1,
                                                                     self.TS_INTERVAL,
                                                                     self.TS_AHEAD,
                                                                     self.TS_TOLERANCE,
                                                                     self.NEXT_DAY_RETRY_THRESHOLD,
                                                                     FEATURES,
                                                                     LABEL)
        # data, labels = self.timeSeriesBuilderService.timeSeriesBuilderSimple(data, 9, 16, self.TS_LENGTH, FEATURES,
        #                                                                     self.TS_AHEAD)

        print('LEN OF TS:')
        print(len(labels))
        return data, labels

    def exportGafData(self, path, data, features):
        exporter = ExportService(path)
        exporter.storeGAFNumpyFeatureDataSeperatly(data, features, '_GAF_DATA')

    def createSingleGafImg(self, data, path):
        GAFservice = gafService()
        GAFservice.saveGAFimg(data, path)

    def reshapeDataToFeatureList(self, arr, COUNT_OF_FEATURES: int) -> list:
        """
            param:
                arr = dimension (länge_aller_ts, länge_einzelner_ts, features)
                bspw = (3740, 20, 9)
                    the @arr containing the numpy array with format like:
                    r = anz von einzelnen timeseries
                    length = länge der einzelnen zeitrihe
                    COUNT_OF_FEATURES = menge an features
                    np.zeros((r, length, COUNT_OF_FEATURES))
                        COUNT_OF_FEATURES = len("Open", "Open_EWA", "Open_EWC", "Open_EWG", "Open_EWJ", "Open_EWU",
                                            "Open_INDA", "Open_MCHI")
                FEATURE_LIST, is a string list where every feauter should be stored in separate .npmy file

            return:
                list = count of (features (len_all_series, len_ts))
        """
        # Splitting the array along the last dimension into three separate arrays
        # split from: (3740, 5, 3)     to: list(3), each = (3740, 5, 1)
        split_arrays = np.split(arr, COUNT_OF_FEATURES, axis=2)
        toReturnList = []  # new list with the correct Shaped Data
        i = 0
        for sArr in split_arrays:
            t_arr = np.squeeze(sArr, axis=2)  # removes the (1) redundant dimension
            toReturnList.append(t_arr)
            # self.storeNumpyTimeSeries(t_arr, FEATURE_LIST[i] + '_' + pathToSave)
            i = i + 1
        return toReturnList
