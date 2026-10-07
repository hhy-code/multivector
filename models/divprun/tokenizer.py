from models.factories import register_tokenizer
from models.tokenpruning.tokenizer import TokenPruningTokenizer


@register_tokenizer("divprun")
class DivPruneTokenizer(TokenPruningTokenizer):
    """ColBERT tokenizer used by DivPrune."""
