import math
from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model
from utils.utils import mask_sep


@register_model("tokenpruning")
class TokenPruning(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        pruning: str,
        pruning_factor: int,
        idf_path: str | None = None,
        temp: float | None = None,
        learnable_temp: bool = False
    ) -> None:
        super().__init__(temp=temp, learnable_temp=learnable_temp)

        if pruning not in {"first", "attn", "idf"}:
            raise ValueError("pruning must be either 'first', 'attn' or 'idf'")
        if pruning_factor <= 0 or pruning_factor > 1:
            raise ValueError("pruning_factor must be (0, 1]")
        if pruning == "idf" and idf_path is None:
            raise ValueError("The IDF pruning method requires the `idf_path` parameter.")

        self.pruning = pruning
        self.pruning_factor = pruning_factor

        self.bert = AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)

        self.register_buffer("idf_lookup", None, persistent=False)
        if idf_path is not None:
            idf_mapping = self._load_idf_scores(idf_path)
            idf_lookup = torch.full((self.bert.config.vocab_size,), -1.0)
            for tok_id, score in idf_mapping.items():
                if tok_id < 0 or tok_id >= idf_lookup.numel():
                    raise ValueError(
                        f"Token ID {tok_id} is outside the model vocabulary."
                    )
                idf_lookup[tok_id] = score
            self.idf_lookup = idf_lookup

    def _load_idf_scores(self, path: str) -> dict[int, float]:
        idf_mapping: dict[int, float] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().split("\t")

                if len(items) != 2:
                    raise ValueError("Failed to parse the idf_score file. Expected format: token_id\tscore.")

                tok_id, score = int(items[0]), float(items[1])
                if tok_id not in idf_mapping:
                    idf_mapping[tok_id] = score
                else:
                    raise ValueError(f"Duplicate token ID: {tok_id}")

        return idf_mapping

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.linear(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)

        return {
            "mv_repr": Q,
        }

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        D = self.linear(D)  # B, L, D
        D = F.normalize(D, p=2, dim=-1)

        return self._pruning(input_ids, D, attention_mask)

    def _pruning(self, input_ids: Tensor, D: Tensor, D_mask: Tensor) -> dict[str, Tensor]:
        prefix = D[:, :2]

        input_ids = input_ids[:, 2:]
        D = D[:, 2:]
        D_mask = mask_sep(D_mask[:, 2:])

        valid_mask = D_mask.bool()
        doc_lens = valid_mask.sum(dim=1, dtype=torch.long)
        keep_lens = (doc_lens * self.pruning_factor).ceil().int().clamp_min_(1)
        # Documents are padded to the longest sequence in the batch, so this
        # equals max(keep_lens) without a GPU-to-CPU synchronization.
        max_k = max(math.ceil((D.size(1) - 1) * self.pruning_factor), 1)  # -1 excluding [SEP]
        keep_mask = (
            torch.arange(max_k, device=D.device)[None, :] < keep_lens[:, None]
        )

        if self.pruning == "first":
            # The tokenizer right-pads documents, so the first k valid tokens
            # are a contiguous slice and require no sorting or gathering.
            pruned_repr = D[:, :max_k]
        elif self.pruning == "attn":
            scores = D @ D.transpose(-1, -2)  # [B, T, T]
            mask_2d = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
            scores.masked_fill_(~mask_2d, float("-inf"))
            weights = scores.softmax(dim=-1).sum(dim=1)  # [B, T]
            token_indices = weights.topk(max_k, dim=1).indices
            pruned_repr = D.gather(
                dim=1,
                index=token_indices.unsqueeze(-1).expand(-1, -1, D.size(-1)),
            )
        else:  # idf
            assert self.idf_lookup is not None
            idf_scores = self.idf_lookup[input_ids]  # [B, T]
            idf_scores = idf_scores.masked_fill(~valid_mask, float("-inf"))
            token_indices = idf_scores.topk(max_k, dim=1).indices
            pruned_repr = D.gather(
                dim=1,
                index=token_indices.unsqueeze(-1).expand(-1, -1, D.size(-1)),
            )

        # Dense batches require a shared max_k width. Remove the extra values
        # of shorter documents instead of merely relying on the score mask.
        pruned_repr = pruned_repr.masked_fill(~keep_mask.unsqueeze(-1), 0)
        pruned_repr = torch.cat([prefix, pruned_repr], dim=1)
        keep_mask = torch.cat(
            [keep_mask.new_ones((prefix.size(0), 2)), keep_mask], 
            dim=1
        )

        return {
            "mv_repr": pruned_repr,
            "mv_mask": keep_mask.to(D_mask.dtype),
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
    def from_config(cls, config: dict[str, Any]) -> "TokenPruning":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        pruning = config.get("pruning", "first")
        pruning_factor = config.get("pruning_factor", 4)
        idf_path = config.get("idf_path", None)
        temp = config.get("temp")
        learnable_temp = config.get("learnable_temp", False)

        return cls(
            pretrained_model,
            dim,
            pruning,
            pruning_factor,
            idf_path,
            temp=temp,
            learnable_temp=learnable_temp,
        )
