from typing import Final

from src_transformers.models.single_layer_perceptron import SingleLayerPerceptron
from src_transformers.models.transformer import Transformer
from src_transformers.models.custom_transformer import Transformer_C

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "transformer": Transformer,
    "single_layer_perceptron": SingleLayerPerceptron,
    "transformer_c": Transformer_C,
}
