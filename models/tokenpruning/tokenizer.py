from models.colbert.tokenizer import ColBERTTokenizer
from models.factories import register_tokenizer


@register_tokenizer("tokenpruning")
class TokenPruningTokenizer(ColBERTTokenizer):
    pass