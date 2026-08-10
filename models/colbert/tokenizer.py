from typing import Any

from torch import Tensor

from utils.utils import insert_prefix_token_id
from models.base import BaseTokenizer
from models.factories import register_tokenizer


@register_tokenizer("colbert")
class ColBERTTokenizer(BaseTokenizer):
    def tokenize_qry(self, texts: list[str], add_special_tokens: bool = False) -> list[list[str]]:
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in texts]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_token], [self.sep_token]
        tokens = [
            prefix + lst + suffix + [self.mask_token for _ in range((self.qry_maxlen - len(lst) - 3))]
            for lst in tokens
        ]

        return tokens

    def tokenize_doc(self, texts: list[str], add_special_tokens: bool = False) -> list[list[str]]:
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in texts]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def tensorize_qry(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        obj = self.tok(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.qry_maxlen - 1
        )

        ids = insert_prefix_token_id(obj["input_ids"], self.Q_token_id)
        masks = insert_prefix_token_id(obj["attention_mask"], 1)

        ids[ids == self.pad_token_id] = self.mask_token_id

        return ids, masks

    def tensorize_doc(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        obj = self.tok(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.doc_maxlen - 1
        )

        ids = insert_prefix_token_id(obj["input_ids"], self.D_token_id)
        masks = insert_prefix_token_id(obj["attention_mask"], 1)

        return ids, masks

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ColBERTTokenizer":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 256)

        return cls(pretrained_model, qry_maxlen, doc_maxlen)
