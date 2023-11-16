from typing import Final

from src_transformers.models.single_layer_perceptron import SingleLayerPerceptron
from src_transformers.models.transformer import Transformer

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "transformer": Transformer,
    "single_layer_perceptron": SingleLayerPerceptron,
}
