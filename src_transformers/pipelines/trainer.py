from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src_transformers.pipelines.constants import MODEL_NAME_MAPPING
from src_transformers.preprocessing.datasets import SingleStockDataset
from src_transformers.utils.logger import Logger


@dataclass
class Trainer:
    """
    trainer class
    """

    source_data: str
    batch_size: int
    epochs: int
    learning_rate: float
    loss: nn.MSELoss | nn.CrossEntropyLoss
    optimizer: optim.SGD | optim.Adam
    gpu_activated: bool
    model: nn.Module
    logger: Logger

    @classmethod
    def create_trainer(
        cls: type["Trainer"],
        source_data: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        loss: Optional[str] = "mse",
        optimizer: Optional[str] = "adam",
        momentum: Optional[float] = 0,
        use_gpu: Optional[bool] = True,
        **kwargs,
    ) -> "Trainer":
        """_summary_

        Args:
            batch_size (int): _description_
            epochs (int): _description_
            learning_rate (float): _description_
            model_name (str): _description_
            parameters (dict): _description_
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
            optimizer_instance = optim.Adam(model.parameters())
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
            source_data=source_data,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            loss=loss_instance,
            optimizer=optimizer_instance,
            gpu_activated=gpu_activated,
            model=model,
            logger=Logger(),
        )

        return instance

    def start_training(self) -> None:
        """_summary_"""

        if self.gpu_activated:
            self.model.to("cuda")
            self.loss.to("cuda")

        config_str = f"source_data: {self.source_data}\
            batch_size: {self.batch_size}\
            epochs: {self.epochs}\
            learning rate: {self.learning_rate}\
            loss: {self.loss}\
            optimizer: {self.optimizer}\
            gpu activated: {self.gpu_activated}"

        self.logger.log_string("Trainer configuration", config_str)
        self.logger.model_text(self.model)

        # TODO: Make validation split based on config
        validation_split: float = 0.2
        self.setup_dataloaders(validation_split)

        # Log stat of the training
        self.logger.train_start()
        try:
            self.train_model()
            finish_reason = "Training ended normally."
        except KeyboardInterrupt:
            finish_reason = "Training interrupted by user."

        self.logger.train_end(finish_reason)
        self.logger.close()

    def setup_dataloaders(self, validation_split) -> None:
        """_summary_

        Args:
            validation_split (_type_): _description_

        Returns:
            tuple[DataLoader, DataLoader]: _description_
        """
        # Setting up the dataset
        time_series = pd.read_csv(
            self.source_data,
            names=["timestamp", "open", "high", "low", "close", "volume"],
        )
        dataset = SingleStockDataset(time_series["close"].iloc[:250], 197)

        dataset_size = len(dataset)
        validation_size = int(np.floor(validation_split * dataset_size))
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = random_split(
            dataset, [train_size, validation_size]
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train_model(self, patience: int = 0, inflation: int = 1) -> None:
        # self.inflation = inflation
        # Daten fuer Early stopping
        # min_loss = float('inf')
        # cur_patience = 0

        for epoch in tqdm(range(self.epochs)):
            # for _ in range(self.inflation):
            train_loss = self.calculate_train_loss()
            self.logger.train_loss(train_loss, epoch)

            validation_loss = self.calculate_validation_loss()
            self.logger.val_loss(validation_loss, epoch)

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

            self.logger.save_net(self.model)

    def calculate_train_loss(self) -> float:
        self.model.train()
        train_loss: float = 0
        step_count: int = 0

        for batch in self.train_loader:
            # Reset optimizer
            self.optimizer.zero_grad()

            if self.gpu_activated:
                batch = batch.to("cuda")

            prediction = self.model.forward(batch["input"])
            loss = self.loss(prediction, batch["target"].float())

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            step_count += 1

        return train_loss / step_count

    def calculate_validation_loss(self) -> float:
        self.model.eval()
        validation_loss: float = 0
        step_count: int = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                if self.gpu_activated:
                    batch = batch.to("cuda")

                prediction = self.model.forward(batch["input"])
                loss = self.loss(prediction, batch["target"].float())

                validation_loss += loss.sum().item()
                step_count += 1

        return validation_loss / step_count
