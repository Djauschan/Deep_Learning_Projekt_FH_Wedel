"""
This module contains functions for saving and loading PyTorch models.
"""

from pathlib import Path
from typing import Final

import torch
import torch.nn as nn

MODEL_OUTPUT_PATH: Final[Path] = Path("./data/output/models")


def get_latest_version(name: str) -> int:
    """
    Returns the highest version number of a model with the specified name.

    This method looks for files in the MODEL_OUTPUT_PATH directory that start with the
    specified name followed by '_v', and returns the highest version number found.
    If no such files are found, it returns 0.

    Args:
        name (str): The name of the model.

    Returns:
        int: The highest version number of the specified model, or 0 if no model is found.
    """
    relevant_file_names = list(map(lambda f: f.name, filter(lambda f: f.name.startswith(
        f'{name}_v'), MODEL_OUTPUT_PATH.iterdir())))

    # Split off the substrings before the v and after the . in the file name
    version_numbers = list(
        map(lambda f: int(f.split('v')[1].split('.')[0]), relevant_file_names))

    if version_numbers:
        return max(version_numbers)

    return 0


def save_model(model: nn.Module) -> str:
    """
    Saves the specified model to the MODEL_OUTPUT_PATH directory.

    This method saves the model to a file with the name of the model's class followed by '_v'
    and the next available version number. The file is saved in the MODEL_OUTPUT_PATH directory
    and has the extension '.pt'.

    Args:
        model (nn.Module): The model to be saved.

    Returns:
        str: The absolute path to the saved model file.
    """
    # Get the class name of the model as string
    model_class_name = type(model).__name__
    new_version = get_latest_version(model_class_name) + 1

    path = Path(MODEL_OUTPUT_PATH, f'{model_class_name}_v{new_version}.pt')
    torch.save(model, path)

    return str(path.absolute())


def load_newest_model(model_class: nn.Module) -> nn.Module:
    """
    Loads the newest model of the specified class from the MODEL_OUTPUT_PATH directory.

    This method loads the model file with the highest version number for the specified class. 
    The model is loaded in evaluation mode.

    Args:
        model_class (nn.Module): The class of the model to be loaded.

    Raises:
        ValueError: If no model of the specified class is found in the MODEL_OUTPUT_PATH directory.

    Returns:
        torch.nn.Module: The loaded model.
    """
    version = get_latest_version(model_class.__name__)
    if version == 0:
        raise ValueError(
            f'No model of class {model_class.__name__} found in {MODEL_OUTPUT_PATH}')

    model = torch.load(
        Path(MODEL_OUTPUT_PATH, f"{model_class.__name__}_v{version}.pt"))
    model.eval()

    return model
