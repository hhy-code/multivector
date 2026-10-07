from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model


@register_model("memtoken")
class MemToken(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        mem_num: int,
        temp: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        self.mem_num = mem_num

        self.bert = AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.linear(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)

        return {
            "mv_repr": Q,
        }

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        D = self.linear(D[:, 2: 2 + self.mem_num])  # B, L, D
        D = F.normalize(D, p=2, dim=-1)

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
    def from_config(cls, config: dict[str, Any]) -> "MemToken":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        mem_num = config.get("mem_num", 32)
        temp = config.get("temp")

        return cls(pretrained_model, dim, mem_num, temp=temp)
