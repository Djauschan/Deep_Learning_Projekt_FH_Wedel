
from typing import Final

from src_transformers.models.transformer import Transformer
from src_transformers.models.single_layer_perceptron import Single_layer_perceptron

MODEL_NAME_MAPPING: Final[dict[str, any]] = {"transformer": Transformer}
MODEL_NAME_MAPPING: Final[dict[str, any]] = {"test_slp": Single_layer_perceptron}
