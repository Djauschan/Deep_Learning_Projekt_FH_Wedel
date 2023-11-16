from dataclasses import dataclass

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, Dataset

from src_transformers.pipelines.constants import MODEL_NAME_MAPPING
from src_transformers.utils.logger import Logger


@dataclass
class Trainer:
    """
    trainer class
    """

    batch_size: int
    epochs: int
    learning_rate: float
    model: any
    logger: Logger

    @classmethod
    def create_trainer(
        cls: type["Trainer"],
        batch_size: int,
        epochs: int,
        learning_rate: float,
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

        instance = cls(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            model=model,
            logger=Logger(),
        )
        return instance

        # # Create data loader

        # VALIDATION_SPLIT_SIZE: float = 0.2
        # train_loader, validation_loader = cls.prepare_dataset(
        #     VALIDATION_SPLIT_SIZE, batch_size
        # )

        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # batch = next(iter(dataloader))
        # src_data = batch[0].long()
        # tgt_data = batch[1].long()

        # # Modelltraining
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        # optimizer = optim.Adam(
        #     model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
        # )

        # model.train()

        # for epoch in range(epochs):
        #     optimizer.zero_grad()
        #     print(tgt_data.shape)
        #     output = model(src_data, tgt_data[:, :-1])
        #     loss = criterion(
        #         output.contiguous().view(-1, model.tgt_vocab_size),
        #         tgt_data[:, 1:].contiguous().view(-1),
        #     )
        #     loss.backward()
        #     optimizer.step()
        #     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        # # TODO: Save model

    def prepare_dataset(self, validation_split) -> tuple[DataLoader, DataLoader]:
        return _, _

    def gpu_setup(self) -> None:
        pass

    def train_model(self, patience: int = 0, inflation: int = 1) -> None:
        pass

    def calculate_epoch(self) -> float:
        pass

    def calculate_step(self, batch: BatchSampler) -> float:
        pass

    def validate(self) -> float:
        pass
