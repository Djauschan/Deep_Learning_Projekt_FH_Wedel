import argparse
from typing import Final

import torch
import yaml
from torch import cuda

from src_transformers.pipelines.model_service import ModelService
from src_transformers.pipelines.predictor import Predictor
from src_transformers.pipelines.trainer import Trainer
from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset
from src_transformers.utils.logger import Logger

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
    if gpu_activated:
        device = torch.device('cuda')
        Logger.log_text(
            f"Using the device '{cuda.get_device_name(device)}' for the started pipeline.")
    else:
        device = torch.device('cpu')
        Logger.log_text(
            "GPU was either deactivated or is not available, using the CPU for the started pipeline.")

    seed = config["training_parameters"]["seed"]
    # Setting random seed for torch
    if seed is not None:
        torch.manual_seed(seed)
        if device == torch.device("cuda"):
            torch.cuda.manual_seed(seed)

    model_parameters = config.pop("model_parameters")
    model_name, model_attributes = model_parameters.popitem()

    dataset = MultiSymbolDataset.create_from_config(
        encoder_input_length=model_attributes.get("seq_len_encoder"),
        decoder_target_length=model_attributes.get("seq_len_decoder"),
        **config.pop("dataset_parameters"))

    if args.pipeline == TRAIN_COMMAND:
        model = ModelService.create_model(device=device,
                                          encoder_dimensions=dataset.encoder_dimensions,
                                          decoder_dimensions=dataset.decoder_dimensions,
                                          model_name=model_name,
                                          model_attributes=model_attributes)

        trainer = Trainer.create_trainer_from_config(
            dataset=dataset,
            model=model,
            device=device,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.save_model()
    if args.pipeline == EVAL_COMMAND:
        model = ModelService.create_model(device=device,
                                          encoder_dimensions=dataset.encoder_dimensions,
                                          decoder_dimensions=dataset.decoder_dimensions,
                                          model_name=model_name,
                                          model_attributes=model_attributes)

        trainer = Trainer.create_trainer_from_config(
            dataset=dataset,
            model=model,
            device=device,
            **config.pop("training_parameters"))

        trainer.start_training()
        trainer.evaluate()
    elif args.pipeline == PREDICT_COMMAND:
        model = ModelService.load_newest_model(model_name=model_name)
        predictor = Predictor(model=model,
                              device=device,
                              dataset=dataset)

        prediction = predictor.predict()
        print(prediction)


if __name__ == "__main__":
    main()
