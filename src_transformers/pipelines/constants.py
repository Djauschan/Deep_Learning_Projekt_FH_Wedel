from typing import Final

from src_transformers.models.transformer import Transformer
from src_transformers.models.torch_transformer import TransformerModel

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "transformer": Transformer,
    "torch_transformer": TransformerModel
}
