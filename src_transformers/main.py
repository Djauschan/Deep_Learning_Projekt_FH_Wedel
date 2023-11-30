import argparse
from typing import Final

import yaml
from torch import cuda

from src_transformers.pipelines.model_io import create_model
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
    gpu_activated = config.pop("use_gpu") and cuda.is_available()

    model_parameters = config.pop("model_parameters")
    model_name, model_attributes = model_parameters.popitem()

    dataset = MultiSymbolDataset.create_from_config(
        encoder_input_length=model_attributes.get("seq_len_encoder"),
        decoder_input_length=model_attributes.get("seq_len_decoder"),
        **config.pop("dataset_parameters"))

    model = create_model(gpu_activated=gpu_activated,
                         encoder_dimensions=dataset.encoder_dimensions,
                         decoder_dimensions=dataset.decoder_dimensions,
                         model_name=model_name,
                         model_attributes=model_attributes)

    if args.pipeline == TRAIN_COMMAND:
        trainer = Trainer.create_trainer_from_config(
            dataset=dataset,
            model=model,
            gpu_activated=gpu_activated,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.save_model()
    if args.pipeline == EVAL_COMMAND:
        trainer = Trainer.create_trainer_from_config(
            dataset=dataset,
            model=model,
            gpu_activated=gpu_activated,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.evaluate()
    else:
        print("Prediction placeholder")


if __name__ == "__main__":
    main()
