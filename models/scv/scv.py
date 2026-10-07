import math
from typing import Any

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel

from models.base import BaseModel, cls_score, colbert_score
from models.factories import register_model
from utils.dist import dist_gather, dist_gather_with_grads


@register_model("scv")
class SCV(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        span_hidden_dim: int,
        span_width: int = 8,
        overlap_rate: float = 0.2,
        cls_loss_weight: float = 1.0,
        dropout: float = 0.05,
        temp: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        self.bert = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        n_heads = self.bert.config.num_attention_heads

        self.span_width = span_width
        self.span_stride = max(1, int((1 - overlap_rate) * span_width))
        self.cls_loss_weight = cls_loss_weight

        self.token_linear = nn.Linear(hidden_size, dim)
        self.cls_linear = nn.Linear(hidden_size, dim)
        self.boundary_linear = nn.Linear(hidden_size, span_hidden_dim)
        self.context_linear = nn.Linear(span_hidden_dim * 2, span_hidden_dim)
        self.span_attention = nn.MultiheadAttention(
            hidden_size,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        span_input_dim = span_hidden_dim * 4 + hidden_size * 3
        self.span_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(span_input_dim, dim),
            nn.GELU(),
        )

    def encode_qry(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        return {
            "mv_repr": self.token_linear(hidden[:, 2:]),
            "mv_mask": attention_mask[:, 2:],
            "cls_repr": self.cls_linear(hidden[:, 0]),
        }

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        token_hidden = hidden[:, 2:]
        token_mask = attention_mask[:, 2:]
        span_repr, span_mask = self._compress(token_hidden, token_mask)
        return {
            "mv_repr": span_repr,
            "mv_mask": span_mask,
            "token_repr": self.token_linear(token_hidden),
            "token_mask": token_mask,
            "cls_repr": self.cls_linear(hidden[:, 0]),
        }

    def _compress(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length, hidden_size = hidden.shape
        max_spans = max(
            1,
            math.ceil(
                (sequence_length - self.span_width) / self.span_stride + 1
            ),
        )
        starts = torch.arange(max_spans, device=hidden.device) * self.span_stride
        offsets = torch.arange(self.span_width, device=hidden.device)
        raw_indices = starts[:, None] + offsets[None]
        in_bounds = raw_indices < sequence_length
        indices = raw_indices.clamp_max(sequence_length - 1)

        windows = hidden[:, indices]
        window_mask = attention_mask[:, indices].bool() & in_bounds[None]
        lengths = attention_mask.sum(dim=-1)
        span_counts = torch.ceil(
            (lengths - self.span_width).clamp_min(0) / self.span_stride
        ).long() + 1
        span_mask = torch.arange(max_spans, device=hidden.device)
        span_mask = span_mask[None] < span_counts[:, None]
        window_mask = window_mask & span_mask[:, :, None]

        start_hidden = windows[:, :, 0]
        end_offsets = window_mask.sum(dim=-1).clamp_min(1) - 1
        end_hidden = windows.gather(
            2,
            end_offsets[:, :, None, None].expand(-1, -1, 1, hidden_size),
        ).squeeze(2)

        gs = self.boundary_linear(start_hidden)
        ge = self.boundary_linear(end_hidden)
        gm = gs * ge
        gc = nn.functional.gelu(self.context_linear(torch.cat([gs, ge], dim=-1)))

        mask = window_mask[..., None]
        sum_pool = (windows * mask).sum(dim=2)
        max_pool = windows.masked_fill(~mask, -1e4).max(dim=2).values
        max_pool = max_pool * span_mask[..., None]

        flat_windows = windows.flatten(0, 1)
        flat_mask = window_mask.flatten(0, 1)
        flat_mask = flat_mask.clone()
        flat_mask[~flat_mask.any(dim=-1), 0] = True
        attended = self.span_attention(
            flat_windows,
            flat_windows,
            flat_windows,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )[0]
        attended = attended.masked_fill(~flat_mask[..., None], -1e4)
        salient = attended.max(dim=1).values.view(batch_size, max_spans, -1)
        salient = salient * span_mask[..., None]

        compressed = torch.cat(
            [gs, ge, gm, gc, sum_pool, max_pool, salient],
            dim=-1,
        )
        span_repr = self.span_linear(compressed)
        span_repr = span_repr * span_mask[..., None]
        return span_repr, span_mask

    @classmethod
    def score(
        cls,
        Q: dict[str, Tensor],
        D: dict[str, Tensor],
        pairwise: bool = True,
    ) -> Tensor:
        scores = colbert_score(
            Q["mv_repr"],
            D["mv_repr"],
            D["mv_mask"],
            Q["mv_mask"],
            pairwise=pairwise,
        )
        return scores * Q["mv_mask"].sum(dim=-1, keepdim=True)

    def compute_loss(self, Q: dict[str, Tensor], D: dict[str, Tensor]) -> Tensor:
        all_documents = self._gather_documents(D)
        n_queries = Q["mv_repr"].size(0)
        documents_per_query = D["mv_repr"].size(0) // n_queries
        labels = torch.arange(n_queries, device=Q["mv_repr"].device)
        labels = labels * documents_per_query
        if dist.is_initialized():
            labels = labels + dist.get_rank() * D["mv_repr"].size(0)

        span_scores = self.score(Q, all_documents, pairwise=False)
        token_scores = colbert_score(
            Q["mv_repr"],
            all_documents["token_repr"],
            all_documents["token_mask"],
            Q["mv_mask"],
            pairwise=False,
        )
        token_scores = token_scores * Q["mv_mask"].sum(dim=-1, keepdim=True)
        cls_scores = cls_score(
            Q["cls_repr"],
            all_documents["cls_repr"],
            pairwise=False,
        )
        if self.temp is not None:
            span_scores = span_scores / self.temp
            token_scores = token_scores / self.temp
            cls_scores = cls_scores / self.temp

        span_loss = nn.functional.cross_entropy(span_scores, labels)
        token_loss = nn.functional.cross_entropy(token_scores, labels)
        cls_loss = nn.functional.cross_entropy(cls_scores, labels)
        loss = span_loss + token_loss + self.cls_loss_weight * cls_loss
        self.loss_dict = {
            "span_loss": span_loss.detach(),
            "token_loss": token_loss.detach(),
            "cls_loss": cls_loss.detach(),
        }
        return loss

    @staticmethod
    def _gather_documents(D: dict[str, Tensor]) -> dict[str, Tensor]:
        if not dist.is_initialized():
            return D

        gathered = {}
        for key, value in D.items():
            if "mask" in key:
                gathered[key] = dist_gather(
                    value,
                    dist.get_rank(),
                    dist.get_world_size(),
                )
            else:
                gathered[key] = dist_gather_with_grads(
                    value,
                    dist.get_world_size(),
                )
        return gathered

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SCV":
        pretrained_model = config.get("pretrained_model", "distilbert-base-uncased")
        dim = config.get("dim", 128)
        span_hidden_dim = config.get("span_hidden_dim", 384)
        span_width = config.get("span_width", 8)
        overlap_rate = config.get("overlap_rate", 0.2)
        cls_loss_weight = config.get("cls_loss_weight", 1.0)
        dropout = config.get("dropout", 0.05)
        temp = config.get("temp")

        return cls(pretrained_model, dim, span_hidden_dim, span_width, overlap_rate, cls_loss_weight, dropout, temp=temp)
