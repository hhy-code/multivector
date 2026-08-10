from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model


@register_model("constbert")
class ConstBERT(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        doc_maxlen: int,
        n_out: int = 32,
        temp: float | None = None,
        learnable_temp: bool = False
    ) -> None:
        super().__init__(temp=temp, learnable_temp=learnable_temp)

        self.doc_maxlen = doc_maxlen
        self.bert = AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)
        self.doc_project = nn.Linear(doc_maxlen, n_out)

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.linear(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)

        return {
            "mv_repr": Q
        }

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        D = D.permute(0, 2, 1)  # B, H, L
        D = self.doc_project(D)  # B, H, C
        D = D.permute(0, 2, 1)  # B, C, H
        D = self.linear(D)
        D = F.normalize(D, p=2, dim=2)

        mask = torch.ones(D.shape[0], D.shape[1], device=D.device, dtype=attention_mask.dtype)

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
    def from_config(cls, config: dict[str, Any]) -> "ConstBERT":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        doc_maxlen = config.get("doc_maxlen", 256)
        n_out = config.get("n_out", 32)
        temp = config.get("temp")
        learnable_temp = config.get("learnable_temp", False)

        return cls(pretrained_model, dim, doc_maxlen, n_out, temp=temp, learnable_temp=learnable_temp)
