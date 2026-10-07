from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model


@register_model("kmeans")
class KMeans(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        pooling_factor: int,
        num_iters: int = 10,
        temp: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        if pooling_factor <= 0:
            raise ValueError("pooling_factor must be greater than 0")
        if num_iters <= 0:
            raise ValueError("num_iters must be greater than 0")

        self.bert = AutoModel.from_pretrained(
            pretrained_model,
            add_pooling_layer=False,
        )
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)
        self.pooling_factor = pooling_factor
        self.num_iters = num_iters

    def encode_qry(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H
        Q = F.normalize(self.linear(Q), p=2, dim=-1)  # B, L, D

        return {"mv_repr": Q}

    @torch.no_grad()
    def _cluster_labels(
        self,
        x: Tensor,
        mask: Tensor,
        num_clusters: Tensor,
        max_clusters: int,
    ) -> Tensor:
        B, L, dim = x.shape
        x = x.detach().float()
        valid_mask = mask.bool()
        active_clusters = (
            torch.arange(max_clusters, device=x.device)[None, :]
            < num_clusters[:, None]
        )

        mean = (x * valid_mask[..., None]).sum(dim=1)
        mean = mean / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        distance_to_mean = (x - mean[:, None]).square().sum(dim=-1)
        first_indices = distance_to_mean.masked_fill(~valid_mask, -1).argmax(dim=1)
        first_centers = x.gather(
            1,
            first_indices[:, None, None].expand(-1, 1, dim),
        ).squeeze(1)
        centers = x.new_zeros(B, max_clusters, dim)
        centers[:, 0] = first_centers

        min_distances = (x - centers[:, :1]).square().sum(dim=-1)
        for cluster_index in range(1, max_clusters):
            next_indices = min_distances.masked_fill(~valid_mask, -1).argmax(dim=1)
            next_centers = x.gather(
                1,
                next_indices[:, None, None].expand(-1, 1, dim),
            ).squeeze(1)
            is_active = active_clusters[:, cluster_index]
            centers[:, cluster_index] = next_centers * is_active[:, None]

            distances = (x - next_centers[:, None]).square().sum(dim=-1)
            min_distances = torch.where(
                is_active[:, None],
                torch.minimum(min_distances, distances),
                min_distances,
            )

        batch_ids = torch.arange(B, device=x.device)[:, None].expand(B, L)
        for _ in range(self.num_iters):
            distances = (
                x.square().sum(dim=-1, keepdim=True)
                + centers.square().sum(dim=-1)[:, None]
                - 2 * torch.bmm(x, centers.transpose(1, 2))
            )
            distances = distances.masked_fill(~active_clusters[:, None], torch.inf)
            labels = distances.argmin(dim=-1)

            global_labels = (batch_ids * max_clusters + labels)[valid_mask]
            sums = x.new_zeros(B * max_clusters, dim)
            sums.index_add_(0, global_labels, x[valid_mask])
            counts = torch.bincount(global_labels, minlength=B * max_clusters)
            counts = counts.view(B, max_clusters)

            new_centers = sums.view(B, max_clusters, dim)
            new_centers = new_centers / counts.clamp_min(1)[..., None]
            centers = torch.where(counts[..., None] > 0, new_centers, centers)

        distances = (
            x.square().sum(dim=-1, keepdim=True)
            + centers.square().sum(dim=-1)[:, None]
            - 2 * torch.bmm(x, centers.transpose(1, 2))
        )
        distances = distances.masked_fill(~active_clusters[:, None], torch.inf)
        return distances.argmin(dim=-1)

    def _kmeans_pooling(self, D: Tensor, D_mask: Tensor) -> dict[str, Tensor]:
        B, L, dim = D.shape
        max_clusters = max(L // self.pooling_factor, 1)

        valid_mask = D_mask.bool()
        doc_lens = valid_mask.sum(dim=1, dtype=torch.long)
        num_clusters = (doc_lens // self.pooling_factor).clamp(
            min=1,
            max=max_clusters,
        )
        labels = self._cluster_labels(
            D,
            valid_mask,
            num_clusters,
            max_clusters,
        )

        batch_ids = torch.arange(B, device=D.device)[:, None].expand(B, L)
        global_labels = (batch_ids * max_clusters + labels)[valid_mask]
        cluster_sums = D.new_zeros(B * max_clusters, dim)
        cluster_sums.index_add_(0, global_labels, D[valid_mask])
        counts = torch.bincount(global_labels, minlength=B * max_clusters)

        pooled = cluster_sums / counts.clamp_min(1).to(D.dtype).unsqueeze(1)
        pooled = F.normalize(pooled, p=2, dim=-1)

        return {
            "mv_repr": pooled.view(B, max_clusters, dim),
            "mv_mask": counts.view(B, max_clusters).gt(0).to(D_mask.dtype),
        }

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H
        D = F.normalize(self.linear(D), p=2, dim=-1)  # B, L, D

        return self._kmeans_pooling(D, attention_mask)

    @classmethod
    def score(
        cls,
        Q: dict[str, Tensor],
        D: dict[str, Tensor],
        pairwise: bool = True,
    ) -> Tensor:
        return colbert_score(
            Q=Q["mv_repr"],
            D=D["mv_repr"],
            D_mask=D["mv_mask"],
            Q_mask=None,
            pairwise=pairwise,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KMeans":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        pooling_factor = config.get("pooling_factor", 4)
        num_iters = config.get("num_iters", 10)
        temp = config.get("temp")

        return cls(pretrained_model, dim, pooling_factor, num_iters=num_iters, temp=temp)
