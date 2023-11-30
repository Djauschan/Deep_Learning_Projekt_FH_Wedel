import argparse
from typing import Final

import yaml
from torch import cuda

from src_transformers.pipelines.trainer import Trainer
from src_transformers.preprocessing.multiSymbolDataset import MultiSymbolDataset

TRAIN_COMMAND: Final[str] = "train"
EVAL_COMMAND: Final[str] = "evaluate"
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
        choices=[TRAIN_COMMAND, PREDICT_COMMAND, EVAL_COMMAND],
        help="Pipeline to be started",
    )

    return parser


def main() -> None:
    """
    The main function of the script. It sets up the parser, reads the configuration files,
    and starts the training or prediction process based on the provided pipeline argument.

    Usage examples:
        python -m src_transformers.main --config data/test_configs/training_config.yaml --pipeline train
        python -m src_transformers.main -c data/test_configs/training_config.yaml -p train
    """
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setting up GPU based on availability and usage preference
    gpu_activated = config.use_gpu and cuda.is_available()

    if args.pipeline == TRAIN_COMMAND:
        dataset = MultiSymbolDataset.create_from_config(
            **config.dataset_parameters)

        # model = create_model = ()

        trainer = Trainer.create_trainer_from_config(
            dataset=dataset,
            gpu_activated=gpu_activated,
            **config.training_parameters)

        trainer.start_training()
        trainer.save_model()
    if args.pipeline == EVAL_COMMAND:
        trainer = Trainer.create_trainer_from_config(**config)
        trainer.start_training()
        # Validation split = 1 as the model will not be trained
        config['validation_split'] = 1
        # trainer = Trainer.create_trainer_from_config(**config, eval_mode=True)
        trainer.evaluate()
    else:
        # dataloader = DataLoader(dataset, shuffle=False)
        # model_class = MODEL_NAME_MAPPING[last_key]
        # model = load_newest_model(model_class
        print("placeholder")


if __name__ == "__main__":
    main()
