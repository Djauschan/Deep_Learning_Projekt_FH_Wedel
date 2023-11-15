import argparse
from typing import Final
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.preprocessing.perSymbolDataset import PerSymbolDataset
import os

import yaml

from src_transformers.pipelines.trainer import Trainer
from src_transformers.pipelines.old_nettrainer import NetTrainer

TRAIN_COMMAND: Final[str] = "train"
# Second version to work parallel (niklas / luca) -> to be joint
TRAIN_COMMAND_NK: Final[str] = "train_nk"
PREDICT_COMMAND: Final[str] = "predict"


def setup_parser() -> argparse.ArgumentParser:
    """
    Sets up the argument parser with the "config" and "pipeline" arguments.

    Returns:
        ArgumentParser: The argument parser with the added arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file with specified options.",
    )
    parser.add_argument("pipeline", choices=[
                        TRAIN_COMMAND, TRAIN_COMMAND_NK, PREDICT_COMMAND])

    return parser


def main() -> None:
    """
    The main function of the script. It sets up the parser, reads the configuration file,
    and starts the training or prediction process based on the provided pipeline argument.
    """
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # TODO: Load dataset (and pass to trainer class?, could also be subconfig in trainer config)
    # TODO: Load Optimizer (could be part of training config)

    # Create dataset
    txt_reader = DataReader()
    data = txt_reader.read_next_txt()
    dataset = PerSymbolDataset(data, txt_reader.symbols)
    # Create data loader
    # dataloader = DataLoader( dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    config["dataset"] = dataset
    print(config)

    if args.pipeline == TRAIN_COMMAND:
        Trainer.start_training(**config)
    if args.pipeline == TRAIN_COMMAND_NK:
        # niklas' version

        # TODO: Put into main function
        with open("C:/Users/nikla/OneDrive/Uni/23WS/Projekt/Deep_Learning/data/training_configs/training_config.yaml", "r", encoding="utf-8") as f:
            config_tr = yaml.safe_load(f)
        with open("C:/Users/nikla/OneDrive/Uni/23WS/Projekt/Deep_Learning/data/model_configs/slp_config.yaml", "r", encoding="utf-8") as f:
            config_mdl = yaml.safe_load(f)

        # Initialize nettrainer with parameters from config
        trainer = NetTrainer(**config_tr['trainer_parameters'],
                             model_name=config_mdl['model_name'],
                             model_parameters=config_mdl['model_parameters'])
        # TODO To be discussed: two configs, option: pass model config

        # Start training
        trainer.train(epochs=config['epochs'])
    else:
        print("placeholder for predict")


if __name__ == "__main__":
    main()
