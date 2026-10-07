from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .index import PLAIDIndex


@dataclass
class PLAIDSearchResult:
    document_ids: list[str]
    scores: Tensor


class PLAIDSearcher:
    """Search interface for model-produced multi-vector representations."""

    def __init__(self, index: PLAIDIndex) -> None:
        self.index = index

    def search(
        self,
        query_embeddings: Tensor,
        k: int = 10,
        query_mask: Tensor | None = None,
        subset: list[str] | None = None,
    ) -> PLAIDSearchResult:
        return self.search_batch(
            query_embeddings.unsqueeze(0),
            k=k,
            query_mask=None if query_mask is None else query_mask.unsqueeze(0),
            subset=subset,
        )[0]

    def search_batch(
        self,
        query_embeddings: Tensor,
        k: int = 10,
        query_mask: Tensor | None = None,
        subset: list[str] | list[list[str]] | None = None,
    ) -> list[PLAIDSearchResult]:
        results = self.index.search(
            query_embeddings,
            k=k,
            query_mask=query_mask,
            subset=subset,
        )
        return [
            PLAIDSearchResult(
                document_ids=[result["id"] for result in query_results],
                scores=torch.tensor(
                    [result["score"] for result in query_results],
                    dtype=torch.float32,
                ),
            )
            for query_results in results
        ]
