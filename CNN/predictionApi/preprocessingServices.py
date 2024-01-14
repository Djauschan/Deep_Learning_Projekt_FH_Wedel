import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from pyts.image import GramianAngularField
from yaml import SafeLoader

'''
@auther ayk.gue
'''


class ConfigService:
    """
        class to load model config
    """

    def __init__(self):
        pass

    def loadModelConfig(self, path: str):
        # opening a file
        parameter = {}
        with open(path, 'r') as stream:
            try:
                # Converts yaml document to python object
                parameter = yaml.load(stream, Loader=SafeLoader)
            except yaml.YAMLError as e:
                print(e)
                # todo feherlbehandung einbauen

        return parameter


class DataMergerService:

    def __init__(self):
        pass

    def mergeFeatureData(self, main_df: pd.DataFrame, df_toMerge: pd.DataFrame) -> pd.DataFrame:
        """
            merge the dataframe in the list
        """
        return main_df.merge(df_toMerge, how='inner', left_on=['posixMinute'], right_on=['posixMinute'])


class DifferencingService:
    """
        class to calc the difference between items in a given array and return array of differences
    """

    def __init__(self):
        pass

    def transformSeries(self, data):
        i = 0
        len_data = len(data)
        # Y = anz daten; anz ele in  zeitreihe, anz features in einem ele
        differencedData = np.zeros((len(data), len(data[0]) - 1, len(data[0][0])))
        while i < len_data:
            differencedData[i], _ = self._transformSingleSeries(data[i], 0.0)
            i = i + 1

        data = differencedData
        return data

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
        return ((newVal - baseVal) / baseVal) * 100


class GafService:
    """
        service that create based Grammian Angular Fields
    """

    def __init__(self):
        pass

    def createGAFfromMultivariateTimeSeries(self, data: np.ndarray) -> np.ndarray:
        """
            param:
            @data: numpy arr with dimensions of (length, features) => (30, 10)
            @return = np.array (len_of_feature, länge_ts, länge_ts)
        """
        data = data.transpose()  # (10, 30)
        gafDataFeatureSeparated = np.zeros((len(data), len(data[0]), len(data[0])))  # 10x30x30
        i = 0
        while i < len(data):
            gafDataFeatureSeparated[i] = self._createSingleGAF(data[i])
            i = i + 1

        return gafDataFeatureSeparated

    def _createSingleGAF(self, x_arr: np.ndarray):
        """
            x_arr = one single timeSeries, length=10
        """
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
        return X_gasf

    def saveGAFimg(self, data: np.ndarray, savePath: str):
        print('data: single imageData')
        # [[-0.95823126  0.14451426 -0.99999702 -0.14451428 -0.95746452 -0.80345206],
        # [ 0.14451426  1.         -0.14210009 -1.         -0.42388929 -0.70522997],
        # [-0.99999702 -0.14210009 -0.95961513  0.14210007 -0.83628839 -0.60157087],
        # [-0.14451428 -1.          0.14210007  1.          0.42388927  0.70522995],
        # [-0.95746452 -0.42388929 -0.83628839  0.42388927 -0.64063574 -0.34319245],
        # [-0.80345206 -0.70522997 -0.60157087  0.70522995 -0.34319245 -0.00530138]]
        plt.imshow(data, cmap='rainbow', origin='lower')
        plt.savefig(savePath)


class NormalisationService:
    """
        class to normalize a dataframe
    """

    def __init__(self):
        pass

    '''
        normalizes data into a scale from -1 -> +1
    '''

    def normMinusPlusOne(self, data: np.ndarray) -> np.ndarray:
        print('normMinusPlusOne DATA')
        i = 0
        len_data = len(data)
        _min = np.min(data)
        _max = np.max(data)
        while i < len_data:
            data[i] = ((data[i] - _max) + (data[i] - _min)) / (_max - _min)
            i = i + 1

        return data


class TimeModificationService:
    """
        class to parse all given timeStamps to posixTime/60
    """

    def __init__(self):
        pass

    def transformTimestap(self, data: pd.DataFrame, dropDateTime: bool) -> pd.DataFrame:
        dateTimeArr = data['DateTime']
        conti_minute_arr = self.addPosixTimeStamp(dateTimeArr)
        # data['posixMinute'] = conti_minute_arr
        data.insert(loc=0, column='posixMinute', value=conti_minute_arr)
        if dropDateTime:
            data.drop(columns=['DateTime'], inplace=True)
        print(data.columns)
        return data

    def addPosixTimeStamp(self, df_DateTimeColumn):
        # Convert python time to posix time in minutes
        return df_DateTimeColumn.apply(lambda x: (x.timestamp()) / 60)


class TimeSeriesBuilder:
    """
        class to build a timeseries based from given Data.
        the data is already filtered start & end date are fixed and
        !!! WARNING, the used Modells require a length for = 10 ITEMS !!!
        !!! Interval should be 480min !!!
    """

    def __init__(self):
        pass

    def buildTimeSeries(self, df: pd.DataFrame, features, length: int = 20, interval: int = 120,
                        tolerance: int = 50) -> np.ndarray:
        """
            create 1 single valid series, out of given dataframe.
            :param
                pf: the dataframe already filtered by start & end date.
                features: a list of features to filter out of the dataframe
                length: the length of the series ! NEEDS TO BE EQUAL TO MODEL_INPUT_SIZE !
                interval: the optimal distance between two datapoints, should be equal to training data,
                    but not required
            :returns
                a numpy array, with dimension (length, len(features)), where the filtered series is in
        """
        COUNT_OF_FEATURES = len(features)
        RETRY_TRESHOLD = 5
        df_length = len(df)
        first_ele = df[0]
        last_ele = df[df_length - 1]
        current_ele = first_ele
        resultNp = np.zeros((length, COUNT_OF_FEATURES))
        # First and Last element are selected from User and are filtered within dataLoadingProcess
        resultNp[0] = first_ele[[*features]].values
        resultNp[9] = last_ele[[*features]].values
        previousCandidate = first_ele
        previousCandidateIdx = 0
        i: int = 0 #counter for entire df <=> index of df
        x: int = 0 #counter for series entry
        while i < df_length:
            while x < length:
                optimal_minute = int(previousCandidate['posixMinute']) + interval
                #get the index of candidate with optimal distance, or the clostest alternative
                candidateIdx, minimalAbw, closestCandidateIdx = self._binary_search(df, i, df_length - 1,
                                                                                    optimal_minute, 1440, -1)
                #if optimal candidate not found
                if candidateIdx == -1:
                    #check if in tolerance range, if so take alternative value
                    if abs(minimalAbw) < tolerance:
                        candidateIdx = closestCandidateIdx
                    else:
                        reTryCounter: int = 2
                        while reTryCounter < RETRY_TRESHOLD and abs(minimalAbw) > tolerance:
                            optimalDateTimeFromPosix = pd.Timestamp.fromtimestamp(optimal_minute * 60)
                            nextDayStartDay = self._getNextDayVal(optimalDateTimeFromPosix, reTryCounter)
                            self._binary_search_dateTime(df, previousCandidateIdx, df_length - 1, nextDayStartDay,
                                                         1000000, -1, True)

                if candidateIdx != -1:
                    candidate = df.iloc[candidateIdx]
                    resultNp[i] = candidate[[*features]].values
                    previousCandidate = candidate
                    previousCandidateIdx = candidateIdx
                else:
                    previousCandidate = df.iloc[i]
                    previousCandidateIdx = i

                x += 1

            i += 1

        return resultNp

    def _binary_search(self, df: pd.DataFrame, lowIdx: int, highIdx: int, optimalTimeToFind: int, minimal_abw: int,
                       closestIdx: int) -> tuple[int, int, int]:
        """
            recursive binarySeach within a dataFrame, start from lowIdx and max from highIdx.
            Returns the clostestIdx found as well if the exact item was not found

            param:
                df: the Dataframe to look in
                lowIdx: the index up which the search starts
                highIdx: the maximum index to look for
                optimalTimeToFind: the exact posixTime to find (posixTime * 60)
                minimal_abw: the absolute distance to the optimalTimeToFind
                closestIdx: the index to the closest element

            return:
                closestIdx: the index to the closest element, -1 if exact item was found
                exactIdx: the index of the searched element, -1 if not found
                absoluteDistance: the distance to the exact element if searched for ele not found
        """
        if highIdx >= lowIdx:
            midIdx = (highIdx + lowIdx) // 2
            midVal = int(df.iloc[midIdx]['posixMinute'])

            tmpAbw = abs(midVal - optimalTimeToFind)
            if tmpAbw < minimal_abw:
                minimal_abw = tmpAbw
                closestIdx = midIdx

            # If element is present at the middle itself
            if midVal == optimalTimeToFind:
                return midIdx, minimal_abw, closestIdx

            if midVal > optimalTimeToFind:
                return self._binary_search(df, lowIdx, midIdx - 1, optimalTimeToFind, minimal_abw, closestIdx)
            else:
                return self._binary_search(df, midIdx + 1, highIdx, optimalTimeToFind, minimal_abw, closestIdx)
        else:
            return -1, minimal_abw, closestIdx

    def _binary_search_dateTime(self, df: pd.DataFrame, lowIdx: int, highIdx: int, optimalDateToFind: pd.Timestamp,
                                minimal_abw: int, closestIdx: int, closestVal_isBigger: bool) -> tuple[int, int, int]:
        """
            binary search on dateTime
        """
        if highIdx >= lowIdx:
            midIdx = (highIdx + lowIdx) // 2
            midVal = df.iloc[midIdx]['DateTime']

            tmpAbw = pd.Timedelta(midVal - optimalDateToFind).seconds
            if tmpAbw < minimal_abw:
                if closestVal_isBigger:
                    if optimalDateToFind < midVal:
                        minimal_abw = tmpAbw
                        closestIdx = midIdx
                else:
                    minimal_abw = tmpAbw
                    closestIdx = midIdx

            # If element is present at the middle itself
            if tmpAbw < 10*60: #if two dates in same minute => they are equal
                return midIdx, minimal_abw, closestIdx

            if midVal > optimalDateToFind:
                return self._binary_search_dateTime(df, lowIdx, midIdx - 1, optimalDateToFind, minimal_abw, closestIdx,
                                                    closestVal_isBigger)
            else:
                return self._binary_search_dateTime(df, midIdx + 1, highIdx, optimalDateToFind, minimal_abw, closestIdx,
                                                    closestVal_isBigger)
        else:
            return -1, minimal_abw, closestIdx

    def _getNextDayVal(self, previousDayDateTime: pd.Timestamp, countOfDay: int) -> pd.Timestamp:
        """
            get index of candidate from next Day 9:30 pm
        """
        nextDay: pd.Timestamp = previousDayDateTime + timedelta(days=countOfDay)
        nextDayStartDay: pd.Timestamp = pd.Timestamp(year=nextDay.year, month=nextDay.month, day=nextDay.day,
                                                     hour=9, minute=30, second=0)
        return nextDayStartDay


class DataLoaderService:
    """
        class to load data from file and store it in dataframe
    """

    def __init__(self):
        pass

    def loadDataFromFile(self, start_date: pd.Timestamp, end_date: pd.Timestamp, rsc_completePath: str,
                         ALL_DATA_COLUMNS: list, COLUMNS_TO_KEEP: list) -> pd.DataFrame:
        df = pd.read_csv(rsc_completePath, sep=",", names=ALL_DATA_COLUMNS, index_col=False)
        toRemove = []
        for col in df:
            if col not in COLUMNS_TO_KEEP:
                toRemove.append(col)

        data = df.drop(toRemove, axis=1)
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)]
        return data


class ModelImportService:

    def __init__(self, modelParameters):
        self.modelParameters = modelParameters

    def getSavedModelsPaths(self) -> list:
        MODEL_FOLDER = self.modelParameters['MODEL_FOLDER']
        MODELS_TO_LOAD = self.modelParameters['MODELS_TO_LOAD']
        MODELS_PATH_LIST = []
        for i in MODELS_TO_LOAD:
            MODELS_PATH_LIST.append(os.path.join(MODEL_FOLDER, i))

        return MODELS_PATH_LIST

    """
        load jit torch model from given path
    """
    def loadModel(self, full_path):
        model = torch.jit.load(full_path)
        model.eval()
        return model


class Preprocessor:
    '''
        class to preprocess a given timeSeries
    '''

    def __init__(self, modelParameters):
        self.configService = ConfigService()
        self.modelParameters = modelParameters
        self.dataLoaderService = DataLoaderService()
        self.featureDataMergeService = DataMergerService()
        self.timeSeriesBuilderService = TimeSeriesBuilder()
        self.timeModificationService = TimeModificationService()
        self.differenceService = DifferencingService()
        self.normalisationService = NormalisationService()
        self.GAFservice = GafService()



    def pipeline(self, startDate: pd.Timestamp, endDate: pd.Timestamp):
        RSC_ROOT = self.modelParameters['RSC_ROOT']
        STOCK_FOLDER = self.modelParameters['STOCK_FOLDER']
        ETF_FOLDER = self.modelParameters['ETF_FOLDER']
        RSC_DATA_FILES = self.modelParameters['RSC_DATA_FILES']
        FEATURES_DATA_TO_LOAD = self.modelParameters['FEATURES_DATA_TO_LOAD']
        FEATURES = self.modelParameters['FEATURES']
        gafData = []
        lastRow = []
        for i in RSC_DATA_FILES:
            for k, v in i.items():
                fileName = v[0]
                allDataColumns = v[1]
                column_featureName = v[2]
                stock_path: str = os.path.join(RSC_ROOT, STOCK_FOLDER)
                feature_path: str = os.path.join(RSC_ROOT, ETF_FOLDER)
                stock_data = self.dataLoaderService.loadDataFromFile(startDate, endDate, stock_path + fileName,
                                                                     allDataColumns, column_featureName)
                stock_data = self.timeModificationService.transformTimestap(stock_data, False)
                lastRow = stock_data[len(stock_data-1)]['Open']
                # load ETF-feature data & join data & features
                data = self.__getAndMergeFeatureDataWithMainData(startDate, endDate, feature_path,
                                                                 FEATURES_DATA_TO_LOAD, stock_data)
                data = self.timeSeriesBuilderService.buildTimeSeries(data, FEATURES)
                # All feature Data will be differenced
                data, labels = self.differenceService.transformSeries(data)
                # Only the Data will be normalised
                data = self.normalisationService.normMinusPlusOne(data)
                # the final model inputData
                gafData = self.GAFservice.createGAFfromMultivariateTimeSeries(data)

        return gafData, lastRow

    def reshapeData(self, data: np.ndarray) -> list:
        dataList = [data]
        return dataList

    def __getAndMergeFeatureDataWithMainData(self, startate: pd.Timestamp, endDate, rsc_folder: str, FEATURES_TO_LOAD: list,
                                             main_df: pd.DataFrame) -> pd.DataFrame:
        mergedDf = main_df
        for oData in FEATURES_TO_LOAD:
            for k, v in oData.items():
                fileName = v[0]
                allColumns = v[1]
                column_featureName = v[2]
                feature_df = self.dataLoaderService.loadDataFromFile(startate, endDate, rsc_folder + fileName,
                                                                     allColumns, column_featureName)
                feature_df = self.timeModificationService.transformTimestap(feature_df, True)
                mergedDf = self.featureDataMergeService.mergeFeatureData(mergedDf, feature_df)
        return mergedDf
