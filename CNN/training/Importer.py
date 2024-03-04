import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

### DeepLearning Project ###
from CNN.preprocessing.services.ConfigService import ConfigService
from CNN.preprocessing.services.ImportService import importService

# RESCOURCE DIRS
RSC_ROOT = "rsc/"
DATA_FOLDER_NAME = "timeSeriesData/"
RSC_FOLDER_NAME = "data/"
RSC_SERIES = "timeSeriesData/"
RSC_SUB_FOLDER_NAME = "etf-complete_tickers_A-C_1min_w1q7w/"
### DeepLearning Project ###

RESULT_PATH = 'results'
FEATURE_ENG_PATH = 'featureEng'
DATA_ENG_PATH = 'dataEng'
MODEL_RESULT_PATH = 'modelResult'
PATH_CSV = "csv"
PATH_IMG = "img"

file_name_data_csv = "_data.csv"
file_test_name_result_csv = "test_results.csv"
file_train_name_result_csv = "training_results.csv"
file_name_loss = "training_loss.csv"
file_name_kum_tr_loss = "training_kummuliert_loss.csv"
file_name_kum_tr_loss_img = "training_kummuliert_loss.jpg"
file_name_kum_test_loss = "test_kummuliert_loss.csv"
file_name_kum_test_loss_img = "test_kummuliert_loss.jpg"
overViewCsvName = "__RESULT_OVERVIEW.csv"
traingLossFileName_img = "training_loss.jpg"
file_name_plot = "plot.png"
file_train_name_plot = "train_plot.png"
file_test_name_plot = "test_plot.png"
saved_model_statedict_name = "model_state.pt"
saved_model_name = "model.pt"
saved_model_scripted_name = "model_scripted.pt"
threshold = 50

EXPORT_PNG_LABEL_SIZE = 24
plt.rcParams['agg.path.chunksize'] = 10000

'''
    @rscFileName: the file from where the data is coming from 
    @projectDirName: the name of the created file to store the results in
    @modelType: a Suffix for the Used ModelType to structure the results
'''


class Importer:

    def __init__(self, ymlModelConfig):
        confService = ConfigService()
        self.modelParameters = confService.loadModelConfig(ymlModelConfig)
        self.PROJECT_ROOT = self.modelParameters['PROJECT_ROOT']
        self.RSC_DATA_ROOT = self.modelParameters['RSC_DATA_ROOT']
        self.TRAINING_FOLDER = self.modelParameters['TRAINING_FOLDER']
        self.TRAINING_DATA_LIST = self.modelParameters['TRAINING_DATA_LIST']
        self.FEATURE_LIST = self.modelParameters['FEATURE_LIST']
        training_data_root = os.path.join(self.RSC_DATA_ROOT, self.TRAINING_FOLDER)
        '''
        steps to do, in order to prep the data for the model
        1.) load main data
        2.) merge with corrospoding feature data
        3.) repeat 1. and 2. until no data left in TRAINING_DATA_LIST
        => result should be:
        array(length_of_all_tr_data, anz_feature, tsl, tsl)
        '''
        # (length_all_ts, feature_count, tsl, tsl)
        self.data = None
        self.labels = None
        self.modelResourcePaths = []
        # list of pair entires (each pair = k=Kürzel, v=list with ele=0 data, ele=1 label
        for i in self.TRAINING_DATA_LIST:
            # getting data for 1 stock
            for k, v in i.items():
                data_fileName = v[0]
                label_fileName = v[1]
                data_folder_path = os.path.join(training_data_root, data_fileName)
                label_folder_path = os.path.join(training_data_root, label_fileName)
                print(label_folder_path)
                if not os.path.exists(label_folder_path) or not os.path.exists(data_folder_path):
                    print('NO FILE FOUND !!')
                    sys.exit()
                self.modelResourcePaths.append([k, training_data_root, data_fileName, label_fileName])
                #importer = importService(training_data_root)
                #self.data = importer.loadFromNpy(data_fileName)
                #self.labels = importer.loadFromNpy(label_fileName)

        # creating timeStamp dir for new dataImports & results
        self.currTime_folder = self.getCurrentTimeName()
        self.result_completePath = os.path.join(self.PROJECT_ROOT, RESULT_PATH, self.currTime_folder)
        self.rsc_completePath = os.path.join(self.PROJECT_ROOT, RSC_ROOT, DATA_FOLDER_NAME)

    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels

    def getModelParameter(self):
        return self.modelParameters

    def getCurrentTimeName(self):
        now = datetime.now()
        return str(now.day) + str(now.month) + str(now.year) + "_" + str(now.hour) + str(now.minute)

    def _helperCountVal(self, column, decimalPrecision=1):
        column.round(decimalPrecision)
        count = 0
        # key: val1=count; val2=index in df
        vals = {0.00: [0, 0]}
        for x in range(len(column)):
            tmpKey = column[x]
            if tmpKey < 250:
                continue

            tmpEntry = vals.get(tmpKey, 0)
            if tmpEntry == 0:
                vals[tmpKey] = [1, x]
            else:
                tmpEntry[0] = tmpEntry[0] + 1
                tmpEntry[1] = x

        sortedVal = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)}
        valList = list(sortedVal.values())
        maxCount = valList[0][0]
        return sortedVal, maxCount

    def getModelResources(self):
        return self.modelResourcePaths

    @staticmethod
    def loadModelData(root_folder, data_fileName, label_fileName):
        importer = importService(root_folder)
        data = importer.loadFromNpy(data_fileName)
        labels = importer.loadFromNpy(label_fileName)
        modelData = ModelData(data, labels)
        return modelData


class ModelData:

    def __init__(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels

    def getTrainingData(self):
        return self.trainingData

    def getTrainingLabels(self):
        return self.trainingLabels
