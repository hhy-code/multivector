from typing import Any

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel
from scipy.cluster.hierarchy import linkage, fcluster

from models.base import BaseModel, colbert_score
from models.factories import register_model


@register_model("tokenpooling")
class TokenPooling(BaseModel):
    def __init__(self, pretrained_model: str, dim: int, pooling_factor: int) -> None:
        super().__init__()

        if pooling_factor <= 0:
            raise ValueError("pooling_factor must be greater than 0")

        self.bert = AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)
        self.pooling_factor = pooling_factor

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.linear(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)

        return {
            "mv_repr": Q,
        }

    def _hierarchical_pooling(self, D: Tensor, D_mask: Tensor) -> dict[str, Tensor]:
        B, L, dim = D.size()

        # Use a batch-independent width so outputs from separately encoded
        # mini-batches can still be concatenated.
        max_k = max(L // self.pooling_factor, 1)

        valid_mask = D_mask.bool()
        doc_lens = valid_mask.sum(dim=1, dtype=torch.long)
        passage_embeddings = D[valid_mask]

        # Transfer all valid embeddings to CPU once. Ward clustering itself is
        # CPU-bound in SciPy, but this avoids one GPU synchronization per doc.
        X = passage_embeddings.detach().float().cpu().numpy()
        doc_lens_cpu = doc_lens.cpu().tolist()

        labels = np.empty(X.shape[0], dtype=np.int64)
        start = 0
        for length in doc_lens_cpu:
            end = start + length
            k = max(length // self.pooling_factor, 1)
            Z = linkage(X[start:end], metric="euclidean", method="ward")
            labels[start:end] = fcluster(Z, t=k, criterion="maxclust") - 1
            start = end

        # Move all labels back to the model device in one transfer, then pool
        # every document and cluster with a single index_add operation.
        labels_tensor = torch.from_numpy(labels).to(
            device=D.device,
            dtype=torch.long,
        )
        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=D.device),
            doc_lens,
        )
        global_labels = batch_ids * max_k + labels_tensor

        cluster_sums = D.new_zeros((B * max_k, dim))
        cluster_sums.index_add_(0, global_labels, passage_embeddings)
        counts = torch.bincount(global_labels, minlength=B * max_k)

        pooled_repr = cluster_sums / counts.clamp_min(1).to(D.dtype).unsqueeze(1)
        pooled_repr = F.normalize(pooled_repr, p=2, dim=-1)
        pooled_repr = pooled_repr.view(B, max_k, dim)
        pooled_mask = counts.view(B, max_k).gt(0).to(D_mask.dtype)

        return {
            "mv_repr": pooled_repr,
            "mv_mask": pooled_mask,
        }

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        D = self.linear(D)  # B, L, D
        D = F.normalize(D, p=2, dim=-1)
        return self._hierarchical_pooling(D, attention_mask)

    @classmethod
    def score(cls, Q: dict[str, Tensor], D: dict[str, Tensor], pairwise: bool = True) -> Tensor:
        return colbert_score(
            Q=Q["mv_repr"],
            D=D["mv_repr"],
            D_mask=D["mv_mask"],
            Q_mask=None,
            pairwise=pairwise
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TokenPooling":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        pooling_factor = config.get("pooling_factor", 4)

        return cls(pretrained_model, dim, pooling_factor)
