"""
This module contains the Trainer class which is used to train a PyTorch model.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_transformers.pipelines.model_service import ModelService
from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset
from src_transformers.utils.logger import Logger
from src_transformers.utils.viz_training import plot_evaluation

FIG_OUTPUT_PATH: Final[Path] = Path("./data/output/eval_plot")

# create directory if it does not exist
FIG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


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
        eval_mode (bool): Is set to True, if the evaluation function is called.
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
    eval_mode: bool

    @classmethod
    def create_trainer_from_config(
        cls: type["Trainer"],
        dataset: MultiSymbolDataset,
        model: nn.Module,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        validation_split: float,
        loss: str = "mse",
        optimizer: str = "adam",
        momentum: float = 0,
        gpu_activated: bool = True,
        eval_mode: bool = False
    ) -> "Trainer":
        """
        Creates a Trainer instance from an unpacked configuration file.

        This method sets up the loss function, model, and optimizer based on the provided
        parameters. It also checks if a GPU is available and if it should be used for training.
        The other parameters from the config are simply passed through to the Trainer instance.

        Args:
            batch_size (int): The batch size for tra*"sgd" optimizer. Defaults to 0.
            use_gpu (bool): Whether to use a GPU for training. Defaults to True.
            eval_mode (bool): should be set to true for evaluation.
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
            Logger.log_text(f"Loss {loss} is not valid, defaulting to MSELoss")
            loss_instance = nn.MSELoss()

        # Setting up the optimizer
        if optimizer == "adam":
            optimizer_instance = optim.Adam(
                model.parameters(), lr=learning_rate)
            if momentum != 0:
                Logger.log_text(
                    f"Momentum {momentum} is not used since the optimizer is set to Adam")
        elif optimizer == "sgd":
            optimizer_instance = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum
            )
        else:
            Logger.log_text(
                f"Optimizer {optimizer} is not valid, defaulting to Adam")
            optimizer_instance = optim.Adam(
                model.parameters(), lr=learning_rate)

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
            eval_mode=eval_mode
        )

        cls._dataset = dataset
        Logger.log_text("Trainer was successfully set up.")

        return instance

    def start_training(self) -> None:
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
        self.logger.write_text("Trainer configuration", config_str)
        self.logger.write_model(self.model)

        # Creating training and validation data loaders from the given data source
        train_loader, validation_loader = self.setup_dataloaders()

        # Perform model training
        self.logger.write_training_start()
        finish_reason = self.train_model(train_loader, validation_loader)
        self.logger.write_training_end(finish_reason)

    def setup_dataloaders(self) -> tuple[DataLoader, DataLoader]:
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
        dataset_size = len(self._dataset)
        validation_size = int(np.floor(self.validation_split * dataset_size))
        train_size = dataset_size - validation_size

        # Split dataset by index not random
        train_dataset = torch.utils.data.Subset(
            self._dataset, range(train_size))
        validation_dataset = torch.utils.data.Subset(
            self._dataset, range(train_size, train_size + validation_size))

        # Create torch data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, validation_loader

    def train_model(self, train_loader, validation_loader, patience: int = 500) -> str:
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

        Returns:
            str: The reason the training ended.
        """
        # Setup for early stopping
        min_loss = float('inf')
        cur_patience = 0

        finish_reason = "Training terminated before training loop ran through."
        for epoch in tqdm(range(self.epochs)):
            try:
                train_loss = self.calculate_train_loss(train_loader)
                self.logger.log_training_loss(train_loss, epoch)

                validation_loss, _ = self.calculate_validation_loss(
                    validation_loader)
                self.logger.log_validation_loss(validation_loss, epoch)

                # Early stopping
                if min_loss > validation_loss:
                    min_loss = validation_loss
                    cur_patience = 0
                else:
                    if patience > 0:
                        cur_patience += 1
                        if cur_patience == patience:
                            finish_reason = "Training finished because of early stopping."
                            self.save_model()
                            break

                self.save_model()
            except KeyboardInterrupt:
                finish_reason = "Training interrupted by user input."
                break

        # Overwrite finish reason if training was not finished due to early stopping or user input
        if finish_reason == "Training terminated before training loop ran through.":
            finish_reason = "Training was normally completed."

        return finish_reason

    def calculate_train_loss(self, train_loader) -> float:
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

        loder_len = len(train_loader)

        for input, target in train_loader:

            # Reset optimizer
            self.optimizer.zero_grad()

            n_tgt_feature = target.shape[2]

            # Create the input for the decoder
            # Targets are shifted one to the right and last entry of targets is filled on idx 0
            dec_input = torch.cat(
                (input[:, -1, -n_tgt_feature:].unsqueeze(1), target[:, :-1, :]), dim=1)

            #
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

            if step_count % 50 == 0:
                print(f'Batch {step_count}/{loder_len} loss: {loss.item()}')

        return train_loss / step_count

    def calculate_validation_loss(self, validation_loader) -> tuple[float, np.array]:
        """
        Calculates the validation loss for the model. This method is called during each epoch.

        This method iterates over each batch in the validation loader, computes the model's
        predictions for the batch, calculates the loss between the predictions and the actual
        targets, and accumulates the total validation loss. The method returns the average
        validation loss per batch.

        If `gpu_activated` is True, the method moves the batch to the GPU before computing the
        predictions and loss.

        Returns:
            tuple:
                float: The average validation loss per batch.
                np.array: Results for predictions (prediction & targets)
        """
        self.model.eval()
        validation_loss: float = 0
        step_count: int = 0

        # create an array to store the predictions and targets of all samples
        if self.eval_mode:
            samples = len(validation_loader.dataset)
            prediction_len = validation_loader.dataset.dataset.seq_len_decoder
            dim = validation_loader.dataset.dataset.output_dim
            results = np.zeros((2, samples, prediction_len, dim))
        else:
            results = None

        loder_len = len(validation_loader)

        with torch.no_grad():
            for input, target in validation_loader:

                # Create the input for the decoder
                # Targets are shifted one to the right and last entry of targets is filled on idx 0
                n_tgt_feature = target.shape[2]
                dec_input = torch.cat(
                    (input[:, -1, -n_tgt_feature:].unsqueeze(1), target[:, :-1, :]), dim=1)

                if self.gpu_activated:
                    input = input.to("cuda")
                    target = target.to("cuda")
                    dec_input = dec_input.to("cuda")

                prediction = self.model.forward(input, dec_input)
                loss = self.loss(prediction, target.float())

                if self.eval_mode:
                    start_idx = step_count * self.batch_size
                    end_idx = start_idx + self.batch_size
                    results[0, start_idx:end_idx, :, :] = prediction.cpu()
                    results[1, start_idx:end_idx, :, :] = target.cpu()

                validation_loss += loss.sum().item()
                step_count += 1

                if step_count % 10 == 0:
                    print(
                        f'Batch {step_count}/{loder_len} loss: {loss.item()}')

        loss = validation_loss / step_count

        return loss, results

    def save_model(self) -> None:
        """
        This method uses the `save_model` function to save the trained model to a file.
        After the model is saved, the method logs a message to the console with the path to the file.
        """
        path = ModelService.save_model(self.model)
        print(f"[TRAINER]: Model saved to '{path}'")

    def evaluate(self) -> None:
        """


        """
        if self.gpu_activated:
            self.model.to("cuda")
            self.loss.to("cuda")

        # Creating training and validation data loaders from the given data source
        self.validation_split = 1
        train_loader, validation_loader = self.setup_dataloaders()

        self.eval_mode = True

        loss, results = self.calculate_validation_loss(validation_loader)

        predictions = results[0]
        targets = results[1]

        plot = plot_evaluation(targets, predictions)
        plot.show()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # TODO @duwe: hier deinen Pfad einfügen
        # path = f'C:/Users/nikla/OneDrive/Uni/23WS/Projekt/Deep_Learning/data/output/eval_plot/plot{now}.png'
        path = Path(FIG_OUTPUT_PATH, f'plot{now}.png')

        plot.savefig(path)

        print(f"[TRAINER]: Evaluation plot saved to '{path}'")

        print(loss)
