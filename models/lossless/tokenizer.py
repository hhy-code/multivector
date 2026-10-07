from models.factories import register_tokenizer
from models.tokenpruning.tokenizer import TokenPruningTokenizer


@register_tokenizer("lossless")
class LosslessTokenizer(TokenPruningTokenizer):
    """ColBERT tokenizer used by the Lossless pruning model."""
