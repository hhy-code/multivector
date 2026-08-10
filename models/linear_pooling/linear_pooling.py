from typing import Any

from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model
from utils.utils import mask_sep


class SpanAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, span_size: int, dropout: float) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.span_size = span_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        if D != self.hidden_size:
            raise ValueError(
                f"Expected the last dimension of x to be {self.hidden_size}, "
                f"got {D}."
            )

        scores = self.linear(x).squeeze(-1)  # [B, T]

        S = self.span_size
        pad_size = (-T) % S
        if pad_size:
            x = F.pad(x, (0, 0, 0, pad_size))
            scores = F.pad(scores, (0, pad_size))
            attention_mask = F.pad(attention_mask, (0, pad_size), value=0)
            T += pad_size

        N = T // S
        x = x.reshape(B, N, S, D)
        scores = scores.reshape(B, N, S)
        span_mask = attention_mask.reshape(B, N, S).bool()
        valid_spans = span_mask.any(dim=-1)  # [B, N]

        # Give empty spans finite logits so softmax is well-defined; their output
        # is explicitly zeroed below.
        safe_mask = span_mask | ~valid_spans.unsqueeze(-1)
        scores = scores.masked_fill(~safe_mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))

        pooled = (weights.unsqueeze(-1) * x).sum(dim=2)
        pooled = pooled.masked_fill(~valid_spans.unsqueeze(-1), 0.0)

        return pooled, valid_spans


@register_model("linear_pooling")
class LinearPooling(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        span_size: int,
        dropout: float = 0.1,
        temp: float | None = None,
        learnable_temp: bool = False
    ) -> None:
        super().__init__(temp, learnable_temp=learnable_temp)

        self.bert = AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, dim, bias=False)
        self.pooling = SpanAttentionPooling(
            hidden_size,
            span_size,
            dropout
        )

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.linear(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)

        return {
            "mv_repr": Q,
        }

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        mask = mask_sep(attention_mask)
        D, mask = self.pooling(D[:, 2:], mask[:, 2:])

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
    def from_config(cls, config: dict[str, Any]) -> "LinearPooling":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        span_size = config.get("span_size", 4)
        dropout = config.get("dropout", 0.1)
        temp = config.get("temp")
        learnable_temp = config.get("learnable_temp", False)

        return cls(pretrained_model, dim, span_size, dropout, temp=temp, learnable_temp=learnable_temp)
