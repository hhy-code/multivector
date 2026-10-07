from models.factories import register_tokenizer
from models.tokenpruning.tokenizer import TokenPruningTokenizer


@register_tokenizer("agc")
class AGCTokenizer(TokenPruningTokenizer):
    """ColBERT tokenizer used by AGC."""
