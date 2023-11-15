from dataclasses import dataclass

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src_transformers.pipelines.constants import MODEL_NAME_MAPPING


@dataclass
class Trainer:
    """
    trainer class
    """

    @classmethod
    def start_training(
        cls: type["Trainer"],
        batch_size: int,
        epochs: int,
        learning_rate: float,
        model_name: str,
        parameters: dict,
        dataset: Dataset
    ) -> None:
        """_summary_

        Args:
            batch_size (int): _description_
            epochs (int): _description_
            learning_rate (float): _description_
            model_name (str): _description_
            parameters (dict): _description_
        """
        model = MODEL_NAME_MAPPING[model_name](**parameters)

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        batch = next(iter(dataloader))
        src_data = batch[0].long()
        tgt_data = batch[1].long()

        # Modelltraining
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
        )

        model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            print(tgt_data.shape)
            output = model(src_data, tgt_data[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, model.tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        # TODO: Save model
