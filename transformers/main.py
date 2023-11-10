import argparse
from typing import Final

import yaml

from transformers.pipelines.trainer import Trainer

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
        type=str,
        required=True,
        help="Path to the configuration file with specified options.",
    )
    parser.add_argument("pipeline", choices=[TRAIN_COMMAND, PREDICT_COMMAND])

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

    if args.pipeline == TRAIN_COMMAND:
        Trainer.start_training(**config)
    else:
        print("placeholder for predict")


if __name__ == "__main__":
    main()
