import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from CNN.preprocessing.services.ExportService import ExportService

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

device = torch.device(dev)

INTERVAL_LOGGING = 1000

class ModelTrainer:

    def __init__(self, parameters, dataset, model, loss_func, optimizer):
        self.parameters = parameters
        self.LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
        self.NUM_EPOCH = parameters['NUM_EPOCH']
        self.LR = parameters['LR']
        self.BATCH_SIZE = parameters['BATCH_SIZE']
        self.TRAIN_TEST_SPLIT_RATIO = parameters['TRAIN_TEST_SPLIT_RATIO']
        self.L2RegularisationFactor = parameters['L2RegularisationFactor']
        self.dataset_train = dataset
        self.train_size = int(self.TRAIN_TEST_SPLIT_RATIO * len(dataset))
        self.test_size = len(dataset) - self.train_size
        self.train_dataset, test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True,
                                           num_workers=0)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0)
        self.model = model
        model.to(device)
        self.loss = loss_func
        self.optimizer = optimizer
        self.printer = 100000

        print("GPU OUTPUT: ")
        name = torch.cuda.get_device_name(0)
        print(name)


    '''
        Ausführung:
            Für jede Epoche:
                * Training (model adjust)
                * Test (model freeze)
            
        Gespeichert wird:
            Für Jede Epoche:
                * Training:
                    csv: Soll_Werte Array, Ist_Werte Array, laufender loss, laufender durchschnittliche Loss
                    img: soll-ist Vergleich mit Running Mean, laufender loss
                * Test:
                    csv: Soll_Werte Array, Ist_Werte Array, laufender loss, laufender durchschnittliche Loss
                    img: soll-ist Vergleich mit Running Mean, laufender loss
            Für Alle Epochen Zusammen:
                * Training:
                    csv: Soll_Werte Array, Ist_Werte Array, laufender Loss, laufender avg loss --Intervalliert
                    img: soll-ist Vergleich mit Running Mean, laufender loss
                * Test
                    csv: Soll_Werte Array, Ist_Werte Array, laufender Loss, laufender avg loss --Intervalliert
                    img: soll-ist Vergleich mit Running Mean, laufender loss
    '''

    def run(self, exporter: ExportService):
        # np array mit dim = (x,y) alle x-te trainingschritte werden gelogged, dabei werden y messwerte gespeichert
        # y1 = 'epoch', y2='modelOut', y3='label', y4='loss', y5='running_avg'
        train_logging_arr = np.zeros(
            (int((len(self.train_dataloader) * self.NUM_EPOCH / self.LOGGING_INTERVAL)) + 5, 5))
        rRunningAvgLoss = 0.0
        # running var for each model exec step
        c = 0
        # logging running var
        logIdx = 0
        print('LENGTH OF DATALOADER')
        print(len(self.train_dataloader))
        for e in range(self.NUM_EPOCH):
            self.model.train()
            ############## FOR EPOCH #############
            ######## EPOCH Train #######
            ### ______________ TRAIN ______________ ###
            for i, z in enumerate(self.train_dataloader):
                c = c + 1
                x = z['x']
                y = z['y']
                loss_val, model_out = self.training_step(self.model, x, y, self.loss, self.optimizer)
                rLabel = round(y.item(), 4)
                rModel_out = round(model_out.item(), 4)
                rLoss = round(loss_val.item(), 4)
                rRunningAvgLoss = round(((rRunningAvgLoss + rLoss) / c), 3)
                if i % self.LOGGING_INTERVAL == 0:
                    # Logging Single EPOCH TRAIN
                    train_logging_arr[logIdx] = [int(e), rModel_out, rLabel, rLoss, rRunningAvgLoss]
                    logIdx = logIdx + 1
                    if i % 10000 == 0:
                        print("Epoch: {}, Batch: {}".format(e + 1, i))
                        #'''
                        print('IN INTERVAL LOGGING')
                        print('model_INPUT:')
                        print(x)
                        print('model_out')
                        print(model_out)
                        print('y')
                        print(y)
                        print("Training Loss: {}".format(loss_val))
                        print('________________')
                        #'''

            ##### TEST FOR EACH EPOCH
            # with torch.no_grad():
            #    test_epoch_y_arr = self.test_model(self.test_dataloader, self.model)
            # add logging

            # FÜR GESAMT RESULT

        LOSS_PLOT_INTERVAL = self.parameters['LOSS_PLOT_INTERVAL']
        df = exporter.logEpochResult(train_logging_arr, 'TRAINING', 'TRAINING_LOGGING.csv')
        exporter.createLossPlot(df, LOSS_PLOT_INTERVAL, 'TRAINING', 'TRAINING_LOGGING')
        now = datetime.now()
        end_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("End Time =", end_time)
        return self.model

    def test_model(self, test_data, model):
        ### ______________ TEST ______________ ###
        with torch.no_grad():
            # Model should not improve
            model.eval()
            index = 0
            sum_losses = 0.0
            for i, z in enumerate(test_data):
                inputs = z['x']
                y = z['y']
                # check each item in batch
                # eigtl nunnötig da Batch_size = 1
                for j in range(len(y)):
                    model_out = model(inputs.float())
                    # todo anpassen loss.item() anstatt selber zu berechnen.
                    y_val = y[j].item()
                    model_out_val = model_out[j].item()
                    # bei anderer Fehlerfunktion zu ändern, bzw. über loss.item() definieren
                    current_loss = (y_val - model_out_val) ** 2
                    index = index + 1
                    sum_losses = sum_losses + current_loss

            mse = round((sum_losses / index), 4)
            return mse

    def training_step(self, model, x, y, loss, optimizer):
        x = x.to(device)
        y = y.to(device)
        model_out = model(x.float())
        model_out = torch.squeeze(model_out, 1)
        #to put model out in same shape as y

        # MSE
        loss_val = loss(model_out, y.float())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        return loss_val, model_out