import yaml

from src_transformers.preprocessing.multiSymbolDataset import MultiSymbolDataset

# from src_transformers.preprocessing.perSymbolDataset import PerSymbolDataset
from src_transformers.preprocessing.txtReader import DataReader


def create_dataset_from_config(config_path: str, model_name: str, model_parameters: dict) -> MultiSymbolDataset:
    # test = MultiSymbolDataset(txt_reader, config)

    with open(config_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    txt_reader = DataReader(data_config)
    txt_reader.reset_index()
    data = txt_reader.read_next_txt()

    # Insert if case determining lengths based on model_name when multiple models exist
    input_length = model_parameters.get("seq_len_encoder")
    target_length = model_parameters.get("seq_len_decoder")

    # Fallback to defaults if no values were extracted from model_parameters
    if input_length is None or target_length is None:
        input_length = 200
        target_length = 3

    # TODO: Adjust for both dataset and use config to determine which dataset to use
    dataset = MultiSymbolDataset(
        txt_reader, data_config, input_length, target_length)

    return dataset
