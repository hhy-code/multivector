from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor


def prepare_embeddings(
    embeddings: Tensor | Sequence[Tensor],
    mask: Tensor | None = None,
) -> list[Tensor]:
    """Convert padded model output into PyLate's variable-length format."""

    if isinstance(embeddings, Tensor):
        items = list(embeddings)
    else:
        items = list(embeddings)

    if mask is not None:
        items = [item[item_mask.bool()] for item, item_mask in zip(items, mask)]

    return [item.detach().cpu() for item in items]


class PLAIDIndex:
    """Project-facing adapter around PyLate's FastPLAID index."""

    def __init__(
        self,
        index_folder: str,
        index_name: str,
        override: bool = False,
        nbits: int = 4,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_samples_kmeans: int | None = None,
        seed: int = 42,
        use_triton: bool | None = None,
        n_ivf_probe: int = 8,
        n_full_scores: int = 8192,
        batch_size: int = 1 << 18,
        num_threads: int | None = None,
        show_progress: bool = True,
        device: str | list[str] | None = None,
        low_memory: bool = False,
    ) -> None:
        # Keep PyLate lazy: importing this module should not load FastPLAID.
        from pylate import indexes

        self.backend = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=override,
            use_fast=True,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            use_triton=use_triton,
            n_ivf_probe=n_ivf_probe,
            n_full_scores=n_full_scores,
            batch_size=batch_size,
            num_threads=num_threads,
            show_progress=show_progress,
            device=device,
            low_memory=low_memory,
        )

    @property
    def is_indexed(self) -> bool:
        return bool(self.backend._index.is_indexed)

    def add_documents(
        self,
        document_ids: str | list[str],
        document_embeddings: Tensor | Sequence[Tensor],
        document_mask: Tensor | None = None,
    ) -> "PLAIDIndex":
        self.backend.add_documents(
            documents_ids=document_ids,
            documents_embeddings=prepare_embeddings(
                document_embeddings,
                document_mask,
            ),
        )
        return self

    def remove_documents(self, document_ids: list[str]) -> "PLAIDIndex":
        self.backend.remove_documents(document_ids)
        return self

    def update_documents(
        self,
        document_ids: list[str],
        document_embeddings: Tensor | Sequence[Tensor],
        document_mask: Tensor | None = None,
    ) -> "PLAIDIndex":
        self.backend.update_documents(
            documents_ids=document_ids,
            documents_embeddings=prepare_embeddings(
                document_embeddings,
                document_mask,
            ),
        )
        return self

    def search(
        self,
        query_embeddings: Tensor | Sequence[Tensor],
        k: int = 10,
        query_mask: Tensor | None = None,
        subset: list[str] | list[list[str]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        return self.backend(
            prepare_embeddings(query_embeddings, query_mask),
            k=k,
            subset=subset,
        )

    def get_documents_embeddings(
        self,
        document_ids: list[list[str]],
    ) -> list[list[list[int | float]]]:
        return self.backend.get_documents_embeddings(document_ids)
