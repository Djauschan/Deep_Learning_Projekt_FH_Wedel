from typing import Final

from src_transformers.models.transformer import Transformer
from src_transformers.models.torch_transformer import TransformerModel
from src_transformers.models.vanilla_mlp import Multi_Layer_Perceptron

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "transformer": Transformer,
    "torch_transformer": TransformerModel,
    "mlp": Multi_Layer_Perceptron
}
