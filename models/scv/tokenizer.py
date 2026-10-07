from typing import Any

from models.colbert.tokenizer import ColBERTTokenizer
from models.factories import register_tokenizer


@register_tokenizer("scv")
class SCVTokenizer(ColBERTTokenizer):
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SCVTokenizer":
        pretrained_model = config.get(
            "pretrained_model",
            "distilbert-base-uncased",
        )
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 512)

        return cls(pretrained_model, qry_maxlen, doc_maxlen)
