import time

import numpy as np

from src.training.CNN_TimeSeriesModel_exe import CNN_TimeSeriesModel
from Preprocessor import Preprocessor
from Importer import Importer

'''
    Verwaltet die Ausführung verschiedener ModelTypen und ihr Training.
    Einstiegsdatei für jegliche ausführung
'''
class executer:

    def __init__(self):
        pass

    """
    SOLL = 
        Bei D:
            jeden tag 20uhr
        Bei H:
            10:00 - 18 alle 2h, aber gerade zahlen
        Bei M:
            10:20uhr
            
            
    todo:
    swing3 und 4
    long 5
    """
    def buildTimeSeriesData_w(self):
        """CONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_dayTrayding_1.yml"
        preProcessingService = Preprocessor(CONFIG_PATH1)
        print('BUILD ALL DATA with CONFIG_PATH 1')

        SCONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_swingTrayding_1.yml"
        preProcessingService = Preprocessor(SCONFIG_PATH1)
        print('SWING 1 -> BUILD ALL DATA with CONFIG_PATH 2')
        SCONFIG_PATH2 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_swingTrayding_2.yml"
        preProcessingService = Preprocessor(SCONFIG_PATH2)
        print('SWING 2 -> BUILD ALL DATA with CONFIG_PATH 2')
        SCONFIG_PATH3 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_swingTrayding_3.yml"
        preProcessingService = Preprocessor(SCONFIG_PATH3)
        print('SWING 3 -> BUILD ALL DATA with CONFIG_PATH 2')"""
        SCONFIG_PATH4 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_swingTrayding_4.yml"
        preProcessingService = Preprocessor(SCONFIG_PATH4)
        print('SWING 4 -> BUILD ALL DATA with CONFIG_PATH 2')
        ###############################
        """
        LCONFIG_PATH1 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_longTrayding_1.yml"
        preProcessingService = Preprocessor(LCONFIG_PATH1)
        print('LONG 1 -> BUILD ALL DATA with CONFIG_PATH 1')

        LCONFIG_PATH2 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_longTrayding_2.yml"
        preProcessingService = Preprocessor(LCONFIG_PATH2)
        print('LONG 2 -> BUILD ALL DATA with CONFIG_PATH 2')

        LCONFIG_PATH3 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_longTrayding_3.yml"
        preProcessingService = Preprocessor(LCONFIG_PATH3)
        print('LONG 3 -> BUILD ALL DATA with CONFIG_PATH 3')

        LCONFIG_PATH4 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_longTrayding_4.yml"
        preProcessingService = Preprocessor(LCONFIG_PATH4)
        print('LONG 4 -> BUILD ALL DATA with CONFIG_PATH 4')

        LCONFIG_PATH5 = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\preprocessing\\dataCreating_longTrayding_5.yml"
        preProcessingService = Preprocessor(LCONFIG_PATH5)
        print('LONG 4 -> BUILD ALL DATA with CONFIG_PATH 4')"""

    def startTraining_w(self):
        #Swing 3
        SWING3_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\training\\baseModel_SwingTrading_3.yml"
        #start 12:50
        importer: Importer = Importer(SWING3_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')
        time.sleep(900)
        #Swing 4
        SWING4_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\training\\baseModel_SwingTrading_4.yml"
        #start 12:50
        importer: Importer = Importer(SWING4_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')
        time.sleep(900)
        #long 5
        SWING1_CONFIG_PATH = "C:\\Projekte\\__PorjectDeepLearningMain\\Deep_Learning\\cnn\\configs\\training\\baseModel_Langfristig_5.yml"
        #start 12:50
        importer: Importer = Importer(SWING1_CONFIG_PATH)
        trainExe = CNN_TimeSeriesModel(importer)
        trainExe.exe()
        print('END 1: train & export Model')


exe = executer()
#exe.buildTimeSeriesData_w()
exe.startTraining_w()