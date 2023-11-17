import argparse
from typing import Final
from torch.utils.data import DataLoader

import yaml

from src_transformers.pipelines.trainer import Trainer
from src_transformers.preprocessing.perSymbolDataset import PerSymbolDataset
from src_transformers.preprocessing.txtReader import DataReader

TRAIN_COMMAND: Final[str] = "train"
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
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file with the specified pipeline options.",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        required=True,
        choices=[TRAIN_COMMAND, PREDICT_COMMAND],
        help="Pipeline to be started",
    )
    parser.add_argument(
        "--data_config",
        "-d",
        type=str,
        required=True,
        help="Path to the configuration file with the specified data processing options.",
    )

    return parser


def main() -> None:
    """
    The main function of the script. It sets up the parser, reads the configuration files,
    and starts the training or prediction process based on the provided pipeline argument.

    Usage examples:
        python -m src_transformers.main --config data/test_configs/training_config.yaml --pipeline train --data_config data/test_configs/data_config.yaml
        python -m src_transformers.main -c data/test_configs/training_config_slp.yaml -p train -d data/test_configs/data_config.yaml
    """
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(args.data_config, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    txt_reader = DataReader(data_config)
    data = txt_reader.read_next_txt()
    dataset = PerSymbolDataset(data, txt_reader.symbols, data_config)
    # Create data loader
    dataloader = DataLoader(
        dataset, batch_size=dataset.config["BATCH_SIZE"], shuffle=False)
    # Print the first sample.
    test_sample = next(iter(dataloader))[0]
    print("INPUT:")
    print(test_sample[0])
    print("OUTPUT:")
    print(test_sample[1])

    if args.pipeline == TRAIN_COMMAND:
        trainer = Trainer.create_trainer_from_config(**config)
        trainer.start_training()
    else:
        print("placeholder for predict")


if __name__ == "__main__":
    main()
