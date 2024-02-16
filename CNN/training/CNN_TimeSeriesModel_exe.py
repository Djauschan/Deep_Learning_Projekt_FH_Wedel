import os
from datetime import datetime

from CNN.training.Trainer import ModelTrainer
from CNN.training.datasets.StockPriceTimeSeriesDataSet_tX import StockPriceTimeSeriesDataSet
from CNN.modelStructures.CNN_multiVariat import CNN_Model
from CNN.preprocessing.services.ExportService import ExportService
from CNN.preprocessing.services.TimeSeriesTensorTransformer import ToTensor, CorrectData
from torchvision import transforms
import torch


class CNN_TimeSeriesModel:
    def __init__(self, importer):
        self.parameters = importer.getModelParameter()
        now = datetime.now()
        begin_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("Beginn Time: " + begin_time)
        self.EXPORT_PATH = os.path.join(self.parameters['RESULT_FOLDER'], begin_time)
        self.exporter = ExportService(self.EXPORT_PATH)
        transTensor = ToTensor()
        transCorrectData = CorrectData()
        dataset_train = StockPriceTimeSeriesDataSet(importer, transform=transforms.Compose([
                                                                                transTensor, transCorrectData]))

        # init = mit anz.
        # featureslen(dataset_train[0]['x'][0]) = input size of img
        #dataset_train  = len = 2 => idx: 0 = data; idx: 1 = label
        len_features = len(dataset_train[0]['x'])
        len_tsl = len(dataset_train[0]['x'][0])
        model = CNN_Model(len_tsl, len_features)
        #loss_func = torch.nn.CrossEntropyLoss()  # classyfi
        loss_func = torch.nn.MSELoss()  # MSE
        LR = self.parameters['LR']
        L2RegularisationFactor = self.parameters['L2RegularisationFactor']
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2RegularisationFactor)
        self.trainer = ModelTrainer(self.parameters, dataset_train, model, loss_func, optimizer)

    def exe(self):
        model = self.trainer.run(self.exporter)
        self.exporter.storeModel(model, self.parameters['MODEL_NAME'])
