import os
import numpy as np
import pandas as pd
import torch

from numpy import float32
from CNN.prediction.services.DataLoaderService import DataLoaderService
from CNN.prediction.services.DataMergerService import DataMergerService
from CNN.preprocessing.services.AverageService import AverageService
from CNN.preprocessing.services.GafService import gafService
from CNN.preprocessing.services.NormalisationService import NormalisationService
from CNN.preprocessing.services.TimeBuildService import TimeSeriesBuilder
from CNN.preprocessing.services.TimeModificationService import TimeModificationService
from CNN.preprocessing.services.DifferencingService import differencingService

"""
@auther ayk.gue
"""
class Preprocessor:
    """
    class to preprocess a given timeSeries
    """

    def __init__(self, config):
        self.config = config
        self.dataLoaderService = DataLoaderService()
        self.featureDataMergeService = DataMergerService()
        self.averagingService = AverageService()
        self.timeSeriesBuilderService = TimeSeriesBuilder()
        self.timeModificationService = TimeModificationService()
        self.differenceService = differencingService(True)
        self.normalisationService = NormalisationService()
        self.GAFservice = gafService()

    def pipeline(self, stock_symbol, startDate: pd.Timestamp, endDate: pd.Timestamp, length: int, interval: int):
        RSC_ROOT = self.config["RSC_ROOT"]
        RSC_DATA_FILES = self.config["RSC_DATA_FILES"]
        STOCK_DATA = [entry for entry in RSC_DATA_FILES if entry.get(stock_symbol) is not None]
        if len(STOCK_DATA) < 1:
            print('no validation data found')
            print(2/0)
        FEATURES_DATA_TO_LOAD = self.config["FEATURES_DATA_TO_LOAD"]
        FEATURES = self.config["FEATURES"]
        modelInput = torch.zeros(1)
        endPrice = 0
        for item in STOCK_DATA:
            filePath = item.get(stock_symbol)[0]
            allColumns = item.get(stock_symbol)[1]
            featureColumns = item.get(stock_symbol)[2]
            filePath = os.path.join(RSC_ROOT, filePath)
            stock_data = self.dataLoaderService.loadDataFromFile(
                startDate,
                endDate,
                filePath,
                allColumns,
                featureColumns,
            )
            stock_data = self.timeModificationService.transformTimestap(stock_data, False)
            endPrice = stock_data.iloc[len(stock_data) - 1]["Open"]
            # load ETF-feature data & join data & features
            data = self.__getAndMergeFeatureDataWithMainData(
                startDate, endDate, RSC_ROOT, FEATURES_DATA_TO_LOAD, stock_data
            )
            data, dateTimeArr = self.timeSeriesBuilderService.buildSingleTimeSeries(
                data, FEATURES, length, interval, tolerance=30
            )
            data = self.averagingService.calcAvgOnSingleTs(data)

            # if not (length + 1) -> data not correct
            #filteredSize = data dropd NaN
            if len(data) != length+1:
                return -1, -1

            data = self.differenceService.transformSingleSeries(data)
            # remove first item difference of 0
            data = data[1:]
            # Only the Data will be normalised
            data = self.normalisationService.normSingleSeriesMinusPlusOne(data)
            # the final model inputData
            gafData = self.GAFservice.createSingleGAFfromMultivariateTimeSeries(data)
            # toTensor
            arr = np.array(gafData).astype(float32)
            modelInput = torch.unsqueeze(torch.from_numpy(arr), 0).to(torch.device("cpu"))

        return modelInput, endPrice

    def __getAndMergeFeatureDataWithMainData(
        self,
        startate: pd.Timestamp,
        endDate,
        rsc_folder: str,
        FEATURES_TO_LOAD: list,
        main_df: pd.DataFrame,
    ) -> pd.DataFrame:
        mergedDf = main_df
        for oData in FEATURES_TO_LOAD:
            for k, v in oData.items():
                fileName = v[0]
                allColumns = v[1]
                column_featureName = v[2]
                feature_df = self.dataLoaderService.loadDataFromFile(
                    startate,
                    endDate,
                    rsc_folder + fileName,
                    allColumns,
                    column_featureName,
                )
                feature_df = self.timeModificationService.transformTimestap(
                    feature_df, True
                )
                mergedDf = self.featureDataMergeService.mergeFeatureData(
                    mergedDf, feature_df
                )
        return mergedDf


class ModelImportService:
    def __init__(self, modelParameters):
        self.modelParameters = modelParameters

    def getSavedModelsPaths(self) -> list:
        MODEL_FOLDER = self.modelParameters["MODEL_FOLDER"]
        MODELS_TO_LOAD = self.modelParameters["MODELS_TO_LOAD"]
        MODELS_PATH_LIST = []
        for i in MODELS_TO_LOAD:
            MODELS_PATH_LIST.append(os.path.join(MODEL_FOLDER, i))

        return MODELS_PATH_LIST

    """
        load jit torch model from given path
    """

    def loadModel(self, full_path):
        device = torch.device("cpu")
        model = torch.jit.load(full_path, map_location=device)
        model.eval()
        return model