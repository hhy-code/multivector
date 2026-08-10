from typing import Any

from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model
from utils.utils import mask_sep


@register_model("meanpooling")
class MeanPooling(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        span_size: int,
        temp: float | None = None,
        learnable_temp: bool = False,
    ) -> None:
        super().__init__(temp=temp, learnable_temp=learnable_temp)

        self.bert = AutoModel.from_pretrained(
            pretrained_model,
            add_pooling_layer=False,
        )
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)
        self.span_size = span_size

    def encode_qry(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H
        Q = F.normalize(self.linear(Q), p=2, dim=-1)  # B, L, D

        return {"mv_repr": Q}

    def _mean_pooling(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pool document content tokens into consecutive fixed-size spans."""
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

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        mask = mask_sep(attention_mask)
        D, mask = self._mean_pooling(
            D[:, 2:],
            mask[:, 2:]
        )
        D = self.linear(D)
        D = F.normalize(D, p=2, dim=-1)

        return {
            "mv_repr": D,
            "mv_mask": mask,
        }

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
    def from_config(cls, config: dict[str, Any]) -> "MeanPooling":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        span_size = config.get("span_size", 16)
        temp = config.get("temp")
        learnable_temp = config.get("learnable_temp", False)

        return cls(
            pretrained_model,
            dim,
            span_size,
            temp=temp,
            learnable_temp=learnable_temp,
        )
