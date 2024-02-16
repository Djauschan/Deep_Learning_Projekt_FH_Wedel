from CNN.training.CNN_TimeSeriesModel_exe import CNN_TimeSeriesModel
from preprocessing.Preprocessor import Preprocessor
from training.Importer import Importer
from preprocessing.services.ExportService import ExportService

'''
    Verwaltet die Ausführung verschiedener ModelTypen und ihr Training.
    Einstiegsdatei für jegliche ausführung
'''
class executer:

    def __init__(self):
        pass

    def buildTimeSeriesData_w(self):
        CONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\baseModel_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH1)
        print('BUILD ALL DATA with CONFIG_PATH1')


    def startTraining_w(self):
        CONFIG_PATH_1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\training\\baseModel_1.yml"
        #start 12:50
        importer = Importer(CONFIG_PATH_1)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

exe = executer()
exe.buildTimeSeriesData_w()
#exe.startTraining_w()