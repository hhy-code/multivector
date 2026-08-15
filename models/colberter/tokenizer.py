from typing import Any

import torch
from nltk.stem.porter import PorterStemmer
from tokenizers.pre_tokenizers import BertPreTokenizer
from torch import Tensor

from models.base import BaseTokenizer
from models.factories import register_tokenizer


@register_tokenizer("colberter")
class ColBERTerTokenizer(BaseTokenizer):
    def __init__(
        self,
        pretrained_model: str,
        qry_maxlen: int,
        doc_maxlen: int,
        stem_words: bool = True,
    ) -> None:
        super().__init__(pretrained_model, qry_maxlen, doc_maxlen)

        self.stem_words = stem_words
        self.pre_tokenizer = BertPreTokenizer()
        self.stemmer = PorterStemmer()

    def tokenize_qry(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
    ) -> list[list[str]]:
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in texts]

        if not add_special_tokens:
            return tokens

        return [[self.cls_token] + tokens_ + [self.sep_token] for tokens_ in tokens]

    def tokenize_doc(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
    ) -> list[list[str]]:
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in texts]

        if not add_special_tokens:
            return tokens

        return [[self.cls_token] + tokens_ + [self.sep_token] for tokens_ in tokens]

    def _create_word_ids(self, texts: list[str], offsets: Tensor) -> Tensor:
        word_ids = torch.zeros(offsets.shape[:2], dtype=torch.long)

        for batch_idx, text in enumerate(texts):
            words = self.pre_tokenizer.pre_tokenize_str(text)
            unique_words = {}

            for word, span in words:
                word = word.lower()
                if self.stem_words:
                    word = self.stemmer.stem(word)

                if word not in unique_words:
                    unique_words[word] = len(unique_words) + 1

                for token_idx, token_span in enumerate(offsets[batch_idx].tolist()):
                    if token_span[0] >= span[0] and token_span[1] <= span[1]:
                        if token_span[0] != token_span[1]:
                            word_ids[batch_idx, token_idx] = unique_words[word]

        return word_ids

    def _tensorize(
        self,
        texts: list[str],
        max_length: int,
        padding: str,
    ) -> tuple[Tensor, Tensor]:
        obj = self.tok(
            texts,
            padding=padding,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=max_length,
        )

        word_ids = self._create_word_ids(texts, obj["offset_mapping"])
        input_ids = torch.stack([obj["input_ids"], word_ids], dim=1)

        return input_ids, obj["attention_mask"]

    def tensorize_qry(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        return self._tensorize(texts, self.qry_maxlen, "max_length")

    def tensorize_doc(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        return self._tensorize(texts, self.doc_maxlen, "longest")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ColBERTerTokenizer":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        qry_maxlen = config.get("qry_maxlen", 32)
        doc_maxlen = config.get("doc_maxlen", 300)
        stem_words = config.get("stem_words", True)

        return cls(pretrained_model, qry_maxlen, doc_maxlen, stem_words)
