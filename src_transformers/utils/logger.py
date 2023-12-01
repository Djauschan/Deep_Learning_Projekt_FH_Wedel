"""
This module contains the Logger class which is used to log training information.
"""
import io
import os
from datetime import datetime
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from PIL import Image

from torchvision.transforms import ToTensor

import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src_transformers.utils.viz_training import plot_evaluation


class Logger():
    """
    A class used to log training information.

    This class handles logging of training information to the console and to a TensorBoard log file. 
    It logs the start and end times of the training, the training and validation losses for each
    epoch, and the model architecture. Additionally, it handles closing the logger after training is
    finished.
    """

    def __init__(self) -> None:
        """
        Initializes the Logger.

        This init method sets up the TensorBoard writer with the name of the target directory set
        to the current date and time. It also initializes the training start time to None.
        """
        current_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._summary_writer = SummaryWriter(f"runs/{current_time_string}")
        self._training_start: Optional[datetime] = None

    def write_text(self, tag: str, text: str) -> None:
        """
        Writes a custom text to a custom tag in the TensorBoard log file.

        Args:
            tag (str): The tag for the text.
            text (str): The text to write.
        """
        self._summary_writer.add_text(tag, text)

    def write_model(self, model: nn.Module) -> None:
        """
        Writes the model architecture formatted as text to the TensorBoard log file.

        Args:
            model (nn.Module): The model.
        """
        self._summary_writer.add_text("model", str(model))

    def write_training_start(self) -> None:
        """
        Logs the start time of the training.

        This method writes a message to the console, sets the training start time to the current
        time, and writes the start time to the TensorBoard log file.
        """
        tqdm.write("[LOGGER]: Training was started.")

        self._training_start = datetime.now()
        self._summary_writer.add_text(
            "training_duration/start_time", str(self._training_start))

    def write_training_end(self, reason: str) -> None:
        """
        Logs the end time of the training and the reason the training finished.

        This method calculates the training duration, writes a message to the console, 
        and writes the end time, training duration, and finish reason to the TensorBoard log file. 
        It also closes the TensorBoard writer.

        Args:
            reason (str): The reason the training finished.
        """
        training_end = datetime.now()

        if self._training_start is None:
            # Set training duration to None if training start is None (if log_training_start was not called)
            training_duration = None
        else:
            training_duration = training_end - self._training_start
        tqdm.write(
            f"[LOGGER]: Training finished with a runtime of {training_duration}. Finish reason: {reason}")

        self._summary_writer.add_text(
            "training_duration/end_time", str(training_end))
        self._summary_writer.add_text("training_duration/duration",
                                      str(training_duration))
        self._summary_writer.add_text("training_duration/reason", reason)

        tqdm.write("[LOGGER]: Closing logger.")
        self._summary_writer.close()

    def log_training_loss(self, value: float, epoch: int):
        """
        Logs the training loss for an epoch.

        This method writes a message to the console and writes the training loss to the
        TensorBoard log file.

        Args:
            value (float): The training loss.
            epoch (int): The epoch number.
        """
        tqdm.write(f"[LOGGER]: Epoch {epoch}: Training Loss = {value}")
        self._summary_writer.add_scalar("loss/train", value, epoch)

    def log_validation_loss(self, value: float, epoch: int) -> None:
        """
        Logs the validation loss for an epoch.

        This method writes a message to the console and writes the validation loss to the
        TensorBoard log file.

        Args:
            value (float): The validation loss.
            epoch (int): The epoch number.
        """
        tqdm.write(f"[LOGGER]: Epoch {epoch}: Validation Loss = {value}")
        self._summary_writer.add_scalar("loss/val", value, epoch)

    def save_loss_chart(self, targets: np.array, predictions: np.array, epoch: int):
        """
        Logged ein Bild der atkuellen Klassifiezierung
        """
        fig = plot_evaluation(targets, predictions)

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Save image in Logger
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image)
        self._summary_writer.add_image("image/ts_chart", image, epoch)

        print("[Logger]: Charts saved.")
        plt.close('all')

    def save_model(self, targets: np.array, predictions: np.array, epoch: int):
        """
        Logged ein Bild der atkuellen Klassifiezierung
        """
        fig = plot_evaluation(targets, predictions)

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Save image in Logger
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image)
        self._summary_writer.add_image("image/ts_chart", image, epoch)

        print("[Logger]: Charts saved.")
        plt.close('all')
