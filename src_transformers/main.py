import argparse
from typing import Final
import yaml
import torch
from torch.utils.data import DataLoader
from src_transformers.pipelines.trainer import Trainer
from src_transformers.preprocessing.perSymbolDataset import PerSymbolDataset
from src_transformers.preprocessing.multiSymbolDataset import MultiSymbolDataset
from src_transformers.preprocessing.txtReader import DataReader
from src_transformers.utils.model_io import save_model, load_newest_model
from src_transformers.pipelines.constants import MODEL_NAME_MAPPING

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
        python -m src_transformers.main --config data/test_configs/training_config.yaml --pipeline train --data_config data/test_configs/data_config_multivariat.yaml
        python -m src_transformers.main -c data/test_configs/training_config.yaml -p train -d data/test_configs/data_config_multivariat.yaml
    """
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(args.data_config, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # TODO Niklas: move dataloading in trainer class to avoid loading if not required for other pipelines @Duwe spricht da was gegen?

    txt_reader = DataReader(data_config)
    test = MultiSymbolDataset(txt_reader, data_config)

    txt_reader.reset_index()
    data = txt_reader.read_next_txt()
    dataset = PerSymbolDataset(data, txt_reader.symbols, data_config)

    # Pop the last key-value pair
    last_key, inner_dict = config.popitem()

    # Add new key-value pairs to the inner dictionary
    inner_dict['input_dim'] = dataset.input_dim
    inner_dict['output_dim'] = dataset.output_dim
    inner_dict['seq_len_encoder'] = dataset.seq_len_encoder
    inner_dict['seq_len_decoder'] = dataset.seq_len_decoder
    inner_dict['use_gpu'] = config['use_gpu']

    # Add the modified inner dictionary back to the outer dictionary
    config[last_key] = inner_dict

    if args.pipeline == TRAIN_COMMAND:
        trainer = Trainer.create_trainer_from_config(**config)
        trainer.start_training(dataset)
        save_model(trainer.model)  # TODO raus? Wird durch logger erledigt

    else:
        dataloader = DataLoader(dataset, shuffle=False)
        model_class = MODEL_NAME_MAPPING[last_key]
        model = load_newest_model(model_class)


if __name__ == "__main__":
    main()
