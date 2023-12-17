import torch
import yaml
from torch import cuda
import sys
import numpy as np
import os

from src_transformers.pipelines.model_service import ModelService
from src_transformers.pipelines.trainer import Trainer
from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset

import sklearn.datasets
import sklearn.metrics

import autosklearn.regression

import matplotlib.pyplot as plt

# The path of the current file is determined.
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'data' directory located two levels up from
# the current file's directory
data_dir_path = os.path.join(
    current_file_directory, os.pardir, os.pardir, 'data')

# Construct the path to the 'output' directory located in the 'data' directory
output_dir_path = os.path.join(data_dir_path, 'output', 'images')


def get_data(config: dict) -> None:
    """
    Visualizes the files and saves the result in the directory data/output

    Args:
        config (dict): Dictionary for the configuration of data preprocessing.
    """
    # Get the parameters for the data set.
    data_parameters = config["dataset_parameters"]

    model_parameters = config.pop("model_parameters")
    model_name, model_attributes = model_parameters.popitem()

    gpu_activated = config.pop("use_gpu") and cuda.is_available()
    if gpu_activated:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = MultiSymbolDataset.create_from_config(encoder_input_length=model_attributes.get("seq_len_encoder"), decoder_target_length=model_attributes.get("seq_len_decoder"), **config.pop("dataset_parameters"))

    model = ModelService.create_model(device=device, encoder_dimensions=dataset.encoder_dimensions, decoder_dimensions=dataset.decoder_dimensions, model_name=model_name, model_attributes=model_attributes)

    trainer = Trainer.create_trainer_from_config(
        dataset=dataset,
        model=model,
        device=device,
        **config.pop("training_parameters"))

    # Creating training and validation data loaders from the given data source.
    train_loader, validation_loader = trainer.setup_dataloaders()

    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []

    for inputs, targets in train_loader:
        train_inputs.append(inputs.numpy())
        train_targets.append(targets.numpy())

    for inputs, targets in validation_loader:
        val_inputs.append(inputs.numpy())
        val_targets.append(targets.numpy())

    # Concatenate all the batches
    train_inputs = np.concatenate(train_inputs, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    val_inputs = np.concatenate(val_inputs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    return train_inputs, train_targets, val_inputs, val_targets


if __name__ == "__main__":

    # Check if the path to the configuration file was passed as an argument.
    assert len(sys.argv) == 2

    # Read in the configuration file.
    config_file_path = sys.argv[1]
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get data.
    train_inputs, train_targets, test_inputs, test_targets = get_data(config)

    # Get initial shapes.
    train_inputs_shape = train_inputs.shape
    train_targets_shape = train_targets.shape
    test_inputs_shape = test_inputs.shape
    test_targets_shape = test_targets.shape

    # Transform a list of sequences of samples into a list of samples each containing a sequence.
    train_inputs = train_inputs.reshape(train_inputs.shape[0], -1)
    train_targets = train_targets.reshape(train_targets.shape[0], -1)
    test_inputs = test_inputs.reshape(test_inputs.shape[0], -1)
    test_targets = test_targets.reshape(test_targets.shape[0], -1)

    # multiply all walues by 1000 to get significant loss
    train_inputs = train_inputs * 1000
    train_targets = train_targets * 1000
    test_inputs = test_inputs * 1000
    test_targets = test_targets * 1000

    automl = autosklearn.regression.AutoSklearnRegressor(memory_limit=1024 * 100, n_jobs=1)
    automl.fit(train_inputs, train_targets)
    print(automl.leaderboard())
    print(automl.show_models())
    train_predictions = automl.predict(train_inputs)
    print("Train R2 score:", sklearn.metrics.r2_score(train_targets, train_predictions))
    test_predictions = automl.predict(test_inputs)
    print("Test R2 score:", sklearn.metrics.r2_score(test_targets, test_predictions))

    # Reshape back to original shapes.
    test_predictions = test_predictions.reshape(test_targets_shape)
    test_targets = test_targets.reshape(test_targets_shape)
    train_predictions = train_predictions.reshape(train_targets_shape)
    train_targets = train_targets.reshape(train_targets_shape)

    # Plot test
    test_targets_first = test_targets[:, 0, :]
    test_predictions_first = test_predictions[:, 0, :]

    # Create 5 line plots
    for i in range(test_targets_shape[2]):
        plt.figure(figsize=(10, 6))
        plt.plot(test_targets_first[:, i], label='Actual')
        plt.plot(test_predictions_first[:, i], label='Predicted')
        plt.title(f'Feature {i+1}')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(output_dir_path, f'feature_test_{i+1}.png'))

        # Close plot to reduce memory usage
        plt.close()

    # Plot train
    train_targets_first = train_targets[:, 0, :]
    train_predictions_first = train_predictions[:, 0, :]

    # Create 5 line plots
    for i in range(train_targets_shape[2]):
        plt.figure(figsize=(10, 6))
        plt.plot(train_targets_first[:, i], label='Actual')
        plt.plot(train_predictions_first[:, i], label='Predicted')
        plt.title(f'Feature {i+1}')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(output_dir_path, f'feature_train_{i+1}.png'))

        # Close plot to reduce memory usage
        plt.close()
