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
        """CONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH1)
        print('BUILD ALL DATA with CONFIG_PATH 1')

        CONFIG_PATH2 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_swingTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH2)
        print('SWING 1 -> BUILD ALL DATA with CONFIG_PATH 2')

        CONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_longTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH1)
        print('SWING 1 -> BUILD ALL DATA with CONFIG_PATH 1')

        CONFIG_PATH2 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_longTrayding_2.yml"
        preProcessingService = Preprocessor(CONFIG_PATH2)
        print('SWING 2 -> BUILD ALL DATA with CONFIG_PATH 2')

        CONFIG_PATH3 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_longTrayding_3.yml"
        preProcessingService = Preprocessor(CONFIG_PATH3)
        print('SWING 3 -> BUILD ALL DATA with CONFIG_PATH 3')

        CONFIG_PATH4 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\preprocessing\\dataCreating_longTrayding_4.yml"
        preProcessingService = Preprocessor(CONFIG_PATH4)
        print('SWING 4 -> BUILD ALL DATA with CONFIG_PATH 4')"""

    def startTraining_w(self):
        DAY1_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\training\\baseModel_dayTraiding.yml"
        #start 12:50
        importer: Importer = Importer(DAY1_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

        SWING1_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\training\\baseModel_SwingTrading_1.yml"
        #start 12:50
        importer: Importer = Importer(SWING1_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

        SWING2_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\training\\baseModel_SwingTrading_2.yml"
        #start 12:50
        importer: Importer = Importer(SWING2_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

        SWING3_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\training\\baseModel_SwingTrading_3.yml"
        #start 12:50
        importer: Importer = Importer(SWING3_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

        SWING4_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\CNN\\configs\\training\\baseModel_SwingTrading_4.yml"
        #start 12:50
        importer: Importer = Importer(SWING4_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')

exe = executer()
#exe.buildTimeSeriesData_w()
exe.startTraining_w()