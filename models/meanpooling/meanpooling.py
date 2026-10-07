import math
from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model


@register_model("meanpooling")
class MeanPooling(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        span_size: int | None = None,
        temp: float | None = None,
        protected_tokens: int = 0,
        pooling_mode: str = "fixed",
        keep_ratio: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        if pooling_mode not in {"fixed", "adaptive"}:
            raise ValueError("pooling_mode must be 'fixed' or 'adaptive'")
        if pooling_mode == "fixed":
            if type(span_size) is not int or span_size <= 0:
                raise ValueError("span_size must be a positive integer")
        elif keep_ratio is None or not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")

        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)
        self.span_size = span_size
        self.protected_tokens = protected_tokens
        self.pooling_mode = pooling_mode
        self.keep_ratio = keep_ratio

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H
        Q = F.normalize(self.linear(Q), p=2, dim=-1)  # B, L, D

        return {
            "mv_repr": Q,
        }

    def _mean_pooling(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Mean-pool unprotected tokens in consecutive spans, excluding padding."""
        B, T, D = x.shape
        S = self.span_size

        pad_size = (-T) % S
        if pad_size:
            x = F.pad(x, (0, 0, 0, pad_size))
            mask = F.pad(mask, (0, pad_size), value=0)
            T = T + pad_size

        N = T // S
        x = x.reshape(B, N, S, D)
        mask = mask.reshape(B, N, S)

        pooled = (x * mask[..., None]).sum(dim=2)
        pooled = pooled / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        valid_spans = mask.any(dim=-1)
        pooled = pooled.masked_fill(~valid_spans[..., None], 0.0)

        return pooled, valid_spans

    def _adaptive_mean_pooling(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Pool each right-padded document into ceil(valid_length * keep_ratio) spans."""
        B, T, D = x.shape
        lengths = mask.sum(dim=-1, dtype=torch.long)  # B
        num_spans = (lengths.to(torch.float64) * self.keep_ratio).ceil().long()  # B

        max_spans = math.ceil(T * self.keep_ratio)
        indices = torch.arange(max_spans, device=x.device)[None, :]  # 1, max_spans
        valid_spans = indices < num_spans[:, None]  # B, max_spans
        denominators = num_spans.clamp_min(1)[:, None]  # B, 1
        lengths = lengths[:, None]  # B, 1

        starts = (indices * lengths) // denominators  # B, max_spans
        ends = ((indices + 1) * lengths + denominators - 1) // denominators  # B, max_spans
        starts = starts.masked_fill(~valid_spans, 0)
        ends = ends.masked_fill(~valid_spans, 0)

        sum_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        prefix_sums = x.masked_fill(~mask.bool().unsqueeze(-1), 0).cumsum(dim=1, dtype=sum_dtype)  # B, T, D
        prefix_sums = F.pad(prefix_sums, (0, 0, 1, 0))  # B, T+1, D
        pooled = prefix_sums.gather(1, ends.unsqueeze(-1).expand(B, max_spans, D))  # B, max_spans, D
        pooled = pooled - prefix_sums.gather(1, starts.unsqueeze(-1).expand(B, max_spans, D))  # B, max_spans, D
        pooled = pooled / (ends - starts).clamp_min(1).unsqueeze(-1)  # B, max_spans, D

        return pooled.to(x.dtype), valid_spans  # (B, max_spans, D), (B, max_spans)

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        prefix = D[:, : self.protected_tokens]
        prefix_mask = attention_mask[:, : self.protected_tokens].bool()
        pooling = self._mean_pooling if self.pooling_mode == "fixed" else self._adaptive_mean_pooling
        body, body_mask = pooling(
            D[:, self.protected_tokens :],
            attention_mask[:, self.protected_tokens :].bool(),
        )
        D = torch.cat((prefix, body), dim=1)
        mask = torch.cat((prefix_mask, body_mask), dim=1)
        D = self.linear(D)
        D = F.normalize(D, p=2, dim=-1)

        return {
            "mv_repr": D,
            "mv_mask": mask
        }

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
    def from_config(cls, config: dict[str, Any]) -> "MeanPooling":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        span_size = config.get("span_size")
        protected_tokens = config.get("protected_tokens", 0)
        pooling_mode = config.get("pooling_mode", "fixed")
        keep_ratio = config.get("keep_ratio")
        temp = config.get("temp")

        return cls(pretrained_model, dim, span_size, temp=temp, protected_tokens=protected_tokens, pooling_mode=pooling_mode, keep_ratio=keep_ratio)
