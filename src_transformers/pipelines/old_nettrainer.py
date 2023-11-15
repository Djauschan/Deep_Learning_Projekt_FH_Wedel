import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torchData
import torch.optim as optim

from src_transformers.pipelines.constants import MODEL_NAME_MAPPING
from src_transformers.preprocessing.datasets import Dataset_single_stock
from src_transformers.utils.logger import Logger

# TODO implement correctly
PATH_FILE = "C:/Users/nikla/OneDrive/Uni/23WS/Projekt/Deep_Learning/data/raw_data/AAPL_1min.txt"


class NetTrainer():
    """
    Trainer class to train and/or evaluate a model
    """

    def __init__(self,
                 batch_size: int,
                 criterion: str,
                 optimizer: str,
                 learning_rate: float,
                 momentum: float,
                 gpu: bool,
                 model_name: str,
                 model_parameters: dict
                 ) -> None:
        """
        Initializes nettrainer instance
        Args:
            batch_size: Batchsize for train and evaluation
            criterion: String for loss function ("mse"|"crossentropy")
            optimizer: String for optimizer ("sgd"|"adam")
            learning_rate: Learning rate for sgd optimizer
            momentum: Momentum for sgd optimizer
            gpu: Boolean if gpu should be used
            model_name: Name of the model to get the model class from the model mapping
            model_parameters: All relevant model parameters to initialize the model
        """
        # Initialize model
        self.model = MODEL_NAME_MAPPING[model_name](**model_parameters)

        # Initialize loss function
        if criterion == "mse":
            self.criterion = nn.MSELoss()
        elif criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()

        # self.seed = seed
        # # RNG Seed setzen
        # torch.manual_seed(self.seed)
        # if self.gpu:
        #     torch.cuda.manual_seed(self.seed)
        #

        # Load optimizer
        #TODO: (Niklas) Tests für beide Fälle schreiben
        self.optimizer = None
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters())

        # Initilize Logger
        self.logger = Logger()

        # Log first information for nettrainer initializations
        nettrainer_settings_str = f"batch_size: {batch_size} - criterion: {criterion} - optimizer: {optimizer} - " \
                                  f"learning_rate: {learning_rate} - momentum: {momentum}"
        self.logger.log_string('Configs_nettrainer', nettrainer_settings_str)
        self.logger.log_string('Configs_model', str(model_parameters))
        self.logger.model_text(self.model)
        #self.logger.summary("seed", self.seed)

        # Datenset vorbereiten
        self.train_loader, self.validation_loader = self.prepare_dataset(0.2, batch_size)

        # Initialize GPU
        self.gpu = gpu and torch.cuda.is_available()
        self.gpu_setup()

    def prepare_dataset(self, validation_split: float, batch_size: int) -> tuple[torchData.DataLoader, torchData.DataLoader]:
        """
        Function to read the data and prepare it as a torch dataset. Also, generates the data loader for the train and test set.
        Args:
            validation_split: Part of the data that should be saved for the evaluation
            batch_size: Size of the train and evaluation batches

        Returns: train and evaluation data loader

        """

        ts = pd.read_csv(PATH_FILE, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.dataset = Dataset_single_stock(ts['close'].iloc[:250], 197)

        dataset_size = len(self.dataset)
        validation_size = int(np.floor(validation_split * dataset_size))
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = torchData.random_split(self.dataset,
                                                                   [train_size,
                                                                    validation_size])

        train_loader = torchData.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=False)
        validation_loader = torchData.DataLoader(validation_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        return train_loader, validation_loader

    def gpu_setup(self) -> None:
        """
        Loades model and criterion to GPU if hardware is available
        Returns: None

        """
        if self.gpu:
            print("[TRAINER]: Write model and criterion on GPU")
            self.model.to('cuda')
            self.criterion.to('cuda')

        # Log if cpu or gpu is used
        self.logger.log_string("device usage", ("GPU" if self.gpu else "CPU"))

    def train(self, epochs: int, patience: int=0, inflation: int=1) -> None:
        """
        Function that runs the training of the model initialized with this class. It calls calc_epoch for each epoch given as a parameter.
        Args:
            epochs: Number of train epochs
            patience: not yet implemented
            inflation: not yet impemented

        Returns: None

        """
        # Log stat of the training
        self.logger.train_start()

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        # self.inflation = inflation

        # Daten fuer Early stopping
        # min_loss = float('inf')
        # cur_patience = 0
        # finish_reason = 'Training did not start'

        # Run epoch step for the amount of epochs given as parameter
        for epoch in range(epochs):
            # Try block to be able to stop the training with CTRL + C
            try:

                train_loss = self.calc_epoch()
                # Log train loss
                self.logger.train_loss(train_loss, epoch)

                # Calculate loss on validation set
                # if(epoch % 10) == 0:
                validation_loss = self.validate(epoch)

                # Early stopping
                # if (min_loss > validation_loss):
                #     min_loss = validation_loss
                #     cur_patience = 0
                #     # Speicher das aktuell beste Netz
                #     self.logger.save_net(self.model)
                # else:
                #     if (patience > 0):
                #         cur_patience += 1
                #         if (cur_patience == patience):
                #             finish_reason = 'Training finished because of early stopping'
                #             break

                # Save the model
                self.logger.save_net(self.model)

                # Print epoch result
                print(f"train loss: {train_loss}")
                print(f"validation loss: {validation_loss}")

            except KeyboardInterrupt:
                # Give reason if Training was interrupted by user
                finish_reason = 'Training interrupted by user'
                break
            else:
                finish_reason = 'Training finished normally'

        # Log finish
        self.logger.train_end(finish_reason)

        # Training Cleanup
        self.logger.close()

    def calc_epoch(self) -> float:
        """
        Function to calculate inference for all batches in the train loader and return the average loss of all batches in the epoch as epoch loss
        Returns: Epoch loss

        """
        epoch_loss = 0.0
        step_count = 0

        # Set model in train mode
        self.model.train()
        # self.dataset.set_validation(False)
        # self.dataset.set_rng_back()

        # for _ in range(self.inflation):
        # Run through all batches of the dataset and accumulate the loss
        for batch in self.train_loader:
            epoch_loss += self.calc_step(batch)
            step_count += 1

        return (epoch_loss / step_count)

    def calc_step(self, batch: torchData.BatchSampler) -> float:
        """
        Function that calculates on inference with one batch, returns the loss and runs one optimization step on the model parameter
        Args:
            batch: Train batch

        Returns: Loss of the batch inference

        """
        # Reset optimizer
        self.optimizer.zero_grad()

        # If required load data to gpu
        if self.gpu:
            batch = batch.to('cuda')

        # Forward step
        prediction = self.model.forward(batch['input'])

        # Loss calculation
        loss = self.criterion(prediction, batch['target'].float())

        # Backpropagation
        loss.backward()

        # Optimazation step
        self.optimizer.step()

        return loss.item()

    def validate(self, epoch: int = 0) -> float:
        """
        Function to run inferences on validation set and calculate average loss per batch.
        Args:
            epoch: Current epoch to log that loss on that step

        Returns: Validation loss
        """

        # Set model to evaluation mode
        self.model.eval()
        # Setze das Dataset in den Validierungszustand
        # self.dataset.set_validation(True)

        valid_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                # If required load data to gpu
                if self.gpu:
                    batch = batch.to('cuda')

                prediction = self.model.forward(batch['input'])
                loss = self.criterion(prediction, batch['target'].float())

                valid_loss += loss.sum().item()

                steps += 1

        # Log validation loss
        self.logger.val_loss((valid_loss / steps), epoch)
        # Log das Image wie das Dataset gerade Klassifieziert wird
        # self.logger.save_cur_image(self.model, epoch,
        # self.dataset.data_input,
        # self.dataset.data_output)

        return (valid_loss / steps)
