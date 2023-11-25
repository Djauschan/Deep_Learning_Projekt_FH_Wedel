"""
This module contains the Trainer class which is used to train a PyTorch model.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from src_transformers.pipelines.constants import MODEL_NAME_MAPPING
from src_transformers.utils.logger import Logger


@dataclass
class Trainer:
    """
    A class used to represent a Trainer for a PyTorch model.

    This class handles the training process for a PyTorch model, including setting up the
    model, loss function, and optimizer from a configuration, moving the model and loss
    function to the GPU if available, setting up the training and validation data loaders,
    and training the model for a specified number of epochs.

    Attributes:
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        validation_split (float): The fraction of the data to use for validation.
        loss (nn.MSELoss | nn.CrossEntropyLoss): The loss function to use.
        optimizer (optim.SGD | optim.Adam): The optimizer to use.
        gpu_activated (bool): Whether to use a GPU for training.
        model (nn.Module): The PyTorch model to train.
        logger (Logger): The logger to use for logging training information.
    """
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    loss: nn.MSELoss | nn.CrossEntropyLoss
    optimizer: optim.SGD | optim.Adam
    gpu_activated: bool
    model: nn.Module
    logger: Logger

    @classmethod
    def create_trainer_from_config(
        cls: type["Trainer"],
        batch_size: int,
        epochs: int,
        learning_rate: float,
        validation_split: float,
        loss: Optional[str] = "mse",
        optimizer: Optional[str] = "adam",
        momentum: Optional[float] = 0,
        use_gpu: Optional[bool] = True,
        **kwargs,
    ) -> "Trainer":
        """
        Creates a Trainer instance from an unpacked configuration file.

        This method sets up the loss function, model, and optimizer based on the provided
        parameters. It also checks if a GPU is available and if it should be used for training.
        The other parameters from the config are simply passed through to the Trainer instance.

        Args:
            batch_size (int): The batch size for training.
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate for the optimizer.
            validation_split (float): The fraction of the data to use for validation.
            loss (str, optional): The name of the loss function to use. Defaults to "mse".
            optimizer (str, optional): The name of the optimizer to use. Defaults to "adam".
            momentum (float, optional): The momentum for the "sgd" optimizer. Defaults to 0.
            use_gpu (bool, optional): Whether to use a GPU for training. Defaults to True.
            **kwargs: Additional keyword arguments.
                      These should include the model name and model parameters.

        Returns:
            Trainer: A Trainer instance with the specified configuration.
        """

        # Setting up the loss
        if loss == "mse":
            loss_instance = nn.MSELoss()
        elif loss == "crossentropy":
            loss_instance = nn.CrossEntropyLoss()
        else:
            print(
                f"Trainer initialization: Loss {loss} is not valid, defaulting to MSELoss"
            )
            loss_instance = nn.MSELoss()

        # Setting up the model from the model name and parameters in the config
        try:
            model_name, model_parameters = kwargs.popitem()
            model = MODEL_NAME_MAPPING[model_name](**model_parameters)
        except KeyError as parse_error:
            raise (
                KeyError(f"The model '{model_name}' does not exist!")
            ) from parse_error
        except TypeError as model_error:
            raise (
                TypeError(
                    f"The creation of the {model_name} model failed with the following error message {model_error}"
                )
            ) from model_error

        # Setting up the optimizer
        if optimizer == "adam":
            optimizer_instance = optim.Adam(
                model.parameters(), lr=learning_rate)
            if momentum != 0:
                print(
                    f"Trainer initialization: Momentum {momentum} is not used since the optimizer is set to Adam"
                )
        if optimizer == "sgd":
            optimizer_instance = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum
            )

        # Setting up GPU based on availability and usage preference
        gpu_activated = use_gpu and torch.cuda.is_available()

        instance = cls(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_split=validation_split,
            loss=loss_instance,
            optimizer=optimizer_instance,
            gpu_activated=gpu_activated,
            model=model,
            logger=Logger(),
        )

        return instance

    def start_training(self, dataset: Dataset) -> None:
        """
        This is the entrypoint method to start the training process for the model.

        This method first moves the model and loss function to the GPU if `gpu_activated`
        is True. It then logs the trainer configuration and the model architecture. The
        method sets up the training and validation data loaders using the `setup_dataloaders`
        method. Afterwards, it starts the actual training using the `train_model` method and
        logs the reason for finishing the training. After the training process is finished,
        the method closes the logger.

        Args:
            dataset (Dataset): Dataset to be used for training, optimizing and validating the model.
        """

        if self.gpu_activated:
            self.model.to("cuda")
            self.loss.to("cuda")

        config_str = f"batch_size: {self.batch_size}\
            epochs: {self.epochs}\
            learning rate: {self.learning_rate}\
            loss: {self.loss}\
            optimizer: {self.optimizer}\
            gpu activated: {self.gpu_activated}"
        self.logger.log_string("Trainer configuration", config_str)
        self.logger.model_text(self.model)

        # Creating training and validation data loaders from the given data source
        self.setup_dataloaders(dataset)

        # Perform model training
        self.logger.train_start()
        finish_reason = self.train_model()
        self.logger.train_end(finish_reason)

        self.logger.close()

    def setup_dataloaders(self, dataset: Dataset) -> None:
        """
        Sets up the training and validation data loaders.

        This function creates data loaders from the passed dataset.
        It splits the dataset into training and validation sets based on the
        `self.validation_split` attribute, and creates data loaders for both of these sets.
        The data loaders are stored in the `self.train_loader` and `self.validation_loader`
        attributes, respectively.

        Args:
            dataset (Dataset): Dataset to be used for training, optimizing and validating the model.
        """

        # determine train and val set size
        dataset_size = len(dataset)
        validation_size = int(np.floor(self.validation_split * dataset_size))
        train_size = dataset_size - validation_size

        # Split dataset by index not random
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        validation_dataset = torch.utils.data.Subset(
            dataset, range(train_size, train_size + validation_size))

        # create data torch loader
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train_model(self, patience: int = 10, inflation: int = 1) -> str:
        """
        Trains the model for a specified number of epochs. For each epoch, the method calculates
        the training loss and validation loss, logs these losses, and saves the current state
        of the model.

        If a `KeyboardInterrupt` is raised during training, the method catches it and sets the
        finish reason to `"Training interrupted by user"`. If the training finishes without
        interruption, the finish reason is set to `"Training finished normally"`.

        Args:
            patience (int): The number of epochs to wait for improvement before stopping training.
                            Defaults to 10.
            inflation (int): The factor by which to inflate the number of epochs. Defaults to 1.

        Returns:
            str: The reason the training ended.
        """
        # self.inflation = inflation
        # Daten fuer Early stopping
        min_loss = float('inf')
        cur_patience = 0

        for epoch in tqdm(range(self.epochs)):
            try:
                # for _ in range(self.inflation):
                train_loss = self.calculate_train_loss()
                self.logger.train_loss(train_loss, epoch)

                validation_loss = self.calculate_validation_loss()
                self.logger.val_loss(validation_loss, epoch)

                # TODO: Implement early stopping
                # Early stopping
                if (min_loss > validation_loss):
                    min_loss = validation_loss
                    cur_patience = 0
                else:
                    if (patience > 0):
                        cur_patience += 1
                        if (cur_patience == patience):
                            finish_reason = 'Training finished because of early stopping'
                            break

                # TODO: Move this method to this class (breach of logger competencies)
                self.logger.save_net(self.model)
            except KeyboardInterrupt:
                finish_reason = "Training interrupted by user."
                break
            else:
                finish_reason = "Training finished normally."

        return finish_reason

    def calculate_train_loss(self) -> float:
        """
        Calculates the training loss for the model. This method is called during each epoch.

        This method iterates over each batch in the training loader. For each batch, it
        resets the optimizer, calculates the loss between the predictions and the actual
        targets, performs backpropagation, and updates the model's parameters.
        The method accumulates the total training loss and returns the average training
        loss per batch.

        If `gpu_activated` is True, the method moves the batch to the GPU before computing the
        predictions and loss.

        Returns:
            float: The average training loss per batch.
        """
        self.model.train()
        train_loss: float = 0
        step_count: int = 0

        for input, target in self.train_loader:
            # Reset optimizer
            self.optimizer.zero_grad()

            # Create the input for the decoder
            # Targets are shifted one to the right and last entry of targets is filled on idx 0
            dec_input = torch.cat(
                (input[:, -1, :].unsqueeze(1), target[:, :-1, :]), dim=1)

            if self.gpu_activated:
                input = input.to("cuda")
                target = target.to("cuda")
                dec_input = dec_input.to("cuda")

            prediction = self.model.forward(input, dec_input)
            loss = self.loss(prediction, target.float())

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            step_count += 1

            # print(f'Batch {step_count} loss: {loss.item()}')

        return train_loss / step_count

    def calculate_validation_loss(self) -> float:
        """
        Calculates the validation loss for the model. This method is called during each epoch.

        This method iterates over each batch in the validation loader, computes the model's
        predictions for the batch, calculates the loss between the predictions and the actual
        targets, and accumulates the total validation loss. The method returns the average
        validation loss per batch.

        If `gpu_activated` is True, the method moves the batch to the GPU before computing the
        predictions and loss.

        Returns:
            float: The average validation loss per batch.
        """
        self.model.eval()
        validation_loss: float = 0
        step_count: int = 0

        with torch.no_grad():
            for input, target in self.validation_loader:

                # prepare decoder input
                dec_input = torch.cat(
                    (input[:, -1, :].unsqueeze(1), target[:, :-1, :]), dim=1)

                if self.gpu_activated:
                    input = input.to("cuda")
                    target = target.to("cuda")
                    dec_input = dec_input.to("cuda")

                prediction = self.model.forward(input, dec_input)
                loss = self.loss(prediction, target.float())

                validation_loss += loss.sum().item()
                step_count += 1

        return validation_loss / step_count
