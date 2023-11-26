import os

import torch

# The path of the current file is determined.
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'data' directory located two levels up from the current file's directory
data_dir_path = os.path.join(
    current_file_directory, os.pardir, os.pardir, 'data')

# Construct the path to the "models" directory which is located in the "output" directory
# which is located in the "data" directory
models_dir_path = os.path.join(data_dir_path, 'output', "models")


def get_latest_version(name: str, directory: str = models_dir_path) -> int:
    """
    Returns the highest version number of a model with the specified name in the specified directory.

    Args:
        name (str): The name of the model.
        directory (str, optional): Directory in which the models are saved. Defaults to models_dir_path.

    Returns:
        int: The highest version number of a model with the specified name in the specified directory or 0 if there is no version yet.
    """

    # Determines the parts of the file names after the version identifier 'v'.
    versions = [f.split(f'{name}_v')[1] for f in os.listdir(
        directory) if f.startswith(f'{name}_v')]

    # Determines the version numbers and converts them into int values.
    versions = [int(f.split('.')[0]) for f in versions]

    # Returns the highest version number or 0 if there is no version yet.
    if versions:
        return max(versions)
    else:
        return 0


def save_model(model: torch.nn.Module, directory: str = models_dir_path) -> str:
    """
    Saves the specified model in the specified directory.
    The new model saved receives the highest version number.

    Args:
        model (torch.nn.Module): The model to be saved.
        directory (str, optional): Directory in which the model is saved. Defaults to models_dir_path.
    """

    # Get the class name of the model as sting
    name = model.__class__.__name__

    current_version = get_latest_version(name)
    version = current_version + 1

    # save the model in the passed direcotry in the format <name>_v<version>.pt
    path = os.path.join(directory, f'{name}_v{version}.pt')
    torch.save(model, path)
    return path


def load_newest_model(model_class: torch.nn.Module) -> torch.nn.Module:
    """
    Loads the newest model of the specified class from the models directory.

    Args:
        model_class (torch.nn.Module): The class of the model to be loaded.

    Raises:
        ValueError: If no model of the specified class is found in the models directory.

    Returns:
        torch.nn.Module: The loaded model.
    """
    version = get_latest_version(model_class.__name__)
    if version == 0:
        raise ValueError(
            f'No model of class {model_class.__name__} found in {models_dir_path}')
    model = torch.load(os.path.join(
        models_dir_path, f'{model_class.__name__}_v{version}.pt'))
    model.eval()
    return model
