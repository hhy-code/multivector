from torch import Tensor

from utils.utils import insert_prefix_token_id, insert_prefix_token_ids
from models.base import BaseTokenizer
from models.factories import register_tokenizer


@register_tokenizer("memtoken")
class MemTokenTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model: str, qry_maxlen: int,
                 doc_maxlen: int, mem_num: int) -> None:
        super().__init__(pretrained_model, qry_maxlen, doc_maxlen)
        self.mem_num = mem_num
        self.mem_token = "[M]"
        self.mem_token_id = self.tok.convert_tokens_to_ids("[unused2]")

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

        prefix, suffix = [self.cls_token, self.D_token] + [self.mem_token for _ in range(self.mem_num)], [self.sep_token]
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
            max_length=self.doc_maxlen - self.mem_num - 1
        )

        ids = insert_prefix_token_ids(
            obj["input_ids"],
            [self.D_token_id] + [self.mem_token_id for _ in range(self.mem_num)]
        )
        masks = insert_prefix_token_ids(
            obj["attention_mask"],
            [1 for _ in range(self.mem_num + 1)]
        )

        return ids, masks

    @classmethod
    def from_config(cls, config) -> "MemTokenTokenizer":
        return cls(
            config.get("pretrained_model", "bert-base-uncased"),
            config.get("qry_maxlen", 32),
            config.get("doc_maxlen", 300),
            config.get("mem_num", 32)
        )
