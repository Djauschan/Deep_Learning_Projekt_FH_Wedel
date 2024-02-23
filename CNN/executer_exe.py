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
        CONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH1)
        print('BUILD ALL DATA with CONFIG_PATH 1')

        CONFIG_PATH2 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH2)
        print('BUILD ALL DATA with CONFIG_PATH 2')

        CONFIG_PATH3 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH3)
        print('BUILD ALL DATA with CONFIG_PATH 3')

        CONFIG_PATH4 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH4)
        print('BUILD ALL DATA with CONFIG_PATH 4')

        CONFIG_PATH5 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH5)
        print('BUILD ALL DATA with CONFIG_PATH 5')

        CONFIG_PATH6 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH6)
        print('BUILD ALL DATA with CONFIG_PATH 6')


    def startTraining_w(self):
        CONFIG_PATH_1 = "/CNN/configs/training/baseModel_dayTraiding.yml"
        #start 12:50
        importer = Importer(CONFIG_PATH_1)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

exe = executer()
exe.buildTimeSeriesData_w()
#exe.startTraining_w()