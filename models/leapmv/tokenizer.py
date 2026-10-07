from models.factories import register_tokenizer
from models.tokenpruning.tokenizer import TokenPruningTokenizer


@register_tokenizer("leapmv")
class LeapMVTokenizer(TokenPruningTokenizer):
    """ColBERT tokenizer used by LeapMV."""
