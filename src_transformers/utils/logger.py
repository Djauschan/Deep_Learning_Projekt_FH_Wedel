from datetime import datetime
from typing import Optional

from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


class Logger():
    """
    Class that enables logging.
    This saves all information in the Summarywrite of pytorch.
    """

    def __init__(self) -> None:
        """
        Initializer of the Logger class.
        Args:
            name: Name of the logger/the run
            locdir: Directory where the logs are saved
            time_stamp: [True] if a time stamp should be added to the name
        """
        current_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._logger = SummaryWriter(f"runs/{current_time_string}")
        self._training_start: Optional[datetime] = None

    def log_text(self, tag: str, text: str) -> None:
        """
        Logges a string TEXT in the token DESC
        """
        self._logger.add_text(tag, text)

    def log_training_start(self) -> None:
        """
        Logges the start time of the training
        """
        tqdm.write("[LOG]: Training was started.")
        self._training_start = datetime.now()
        self._logger.add_text(
            "training_duration/start_time", str(self._training_start))

    def log_training_end(self, reason: str) -> None:
        """
        Logges the end time and duration of the training
        """
        training_end = datetime.now()

        if self._training_start is None:
            # Set training duration to None if training start is None (if log_training_start was not called)
            training_duration = None
        else:
            training_duration = training_end - self._training_start

        self._logger.add_text("training_duration/end_time", str(training_end))
        self._logger.add_text("training_duration/duration",
                              str(training_duration))
        self._logger.add_text("training_duration/reason", reason)

        tqdm.write(
            f"[LOG]: Training finished with a runtime of {training_duration}. Finish reason: {reason}")
        tqdm.write("[LOG]: Closing logger.")
        self._logger.close()

    def log_validation_loss(self, value: float, step: int) -> None:
        """
        Logges the loss of the validation
        """
        tqdm.write(f"[LOG]: Validation Step {step} logged. Loss {value}")
        self._logger.add_scalar("loss/val", value, step)

    def log_training_loss(self, value: float, step: int):
        """
        Logges the loss of the training
        """
        tqdm.write(f"[LOG]: Training Step {step} logged. Loss {value}")
        self._logger.add_scalar("loss/train", value, step)

    def log_model_string(self, model) -> None:
        """
        Logges the model in textual form
        """
        self._logger.add_text("model", str(model))
