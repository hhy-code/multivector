from typing import Any

from models.factories import register_tokenizer
from models.tokenpooling.tokenizer import TokenPoolingTokenizer


@register_tokenizer("kmeans")
class KMeansTokenizer(TokenPoolingTokenizer):
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KMeansTokenizer":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 256)

        return cls(pretrained_model, qry_maxlen, doc_maxlen)
