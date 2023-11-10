from pydantic import BaseModel, Field, PositiveInt


class Trainer(BaseModel):
    """
    This class is used to train a given transformer model.

    Args:
        BaseModel (_type_): _description_
    """

    batch_size: PositiveInt
    epochs: PositiveInt
    learning_rate: float = Field(default=0.001, ge=0, le=1)

    def start_training(self):
        print(f"{self.batch_size}, {self.epochs}, {self.learning_rate}")

        # TODO: Save model
