from typing import Any

import torch


class TriplesCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [f["query"] for f in features]
        documents = [
            doc
            for f in features
            for doc in f["document"]
        ]

        if "label" in features[0]:
            return {
                "query": self.tokenizer.tensorize_qry(queries),
                "document": self.tokenizer.tensorize_doc(documents),
                "label": [f["label"] for f in features]
            }

        return {
            "query": self.tokenizer.tensorize_qry(queries),
            "document": self.tokenizer.tensorize_doc(documents),
        }