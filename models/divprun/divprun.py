from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from models.colbert import ColBERT
from models.factories import register_model
from utils.utils import mask_sep


@register_model("divprun")
class DivPrune(ColBERT):
    """ColBERT with training-free Max-Min diversity document pruning."""

    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        pruning_factor: float,
        protected_tokens: int = 2,
        temp: float | None = None,
    ) -> None:
        super().__init__(pretrained_model, dim, temp=temp)

        if pruning_factor <= 0 or pruning_factor > 1:
            raise ValueError("pruning_factor must be in (0, 1]")

        self.pruning_factor = pruning_factor
        self.protected_tokens = protected_tokens

    @staticmethod
    @torch.no_grad()
    def _diverse_indices(D: Tensor, D_mask: Tensor, max_k: int) -> Tensor:
        """Apply the paper's greedy Max-Min selection to a batch."""
        B, L, _ = D.shape
        valid_mask = D_mask.bool()

        vectors = F.normalize(D.detach().float(), p=2, dim=-1)
        distances = 1.0 - torch.bmm(vectors, vectors.transpose(1, 2))

        valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        diagonal = torch.eye(L, dtype=torch.bool, device=D.device)[None]
        nearest = distances.masked_fill(~valid_pairs | diagonal, torch.inf).amin(dim=-1)
        nearest = nearest.masked_fill(~valid_mask, -torch.inf)

        selected_indices = torch.empty(B, max_k, dtype=torch.long, device=D.device)
        selected_mask = torch.zeros(B, L, dtype=torch.bool, device=D.device)
        batch_indices = torch.arange(B, device=D.device)

        next_indices = nearest.argmax(dim=-1)
        selected_indices[:, 0] = next_indices
        selected_mask[batch_indices, next_indices] = True
        min_distances = distances[batch_indices, next_indices]

        for index in range(1, max_k):
            scores = min_distances.masked_fill(~valid_mask | selected_mask, -torch.inf)
            next_indices = scores.argmax(dim=-1)
            selected_indices[:, index] = next_indices
            selected_mask[batch_indices, next_indices] = True
            min_distances = torch.minimum(
                min_distances,
                distances[batch_indices, next_indices],
            )

        return selected_indices

    def _pruning(self, D: Tensor, D_mask: Tensor) -> dict[str, Tensor]:
        prefix = D[:, : self.protected_tokens]
        prefix_mask = D_mask[:, : self.protected_tokens]

        D = D[:, self.protected_tokens :]
        D_mask = mask_sep(D_mask[:, self.protected_tokens :])

        doc_lens = D_mask.sum(dim=1, dtype=torch.long)
        keep_lens = (doc_lens * self.pruning_factor).round().long().clamp_min_(1)
        max_k = max(round((D.size(1) - 1) * self.pruning_factor), 1)
        keep_mask = torch.arange(max_k, device=D.device)[None] < keep_lens[:, None]

        token_indices = self._diverse_indices(D, D_mask, max_k)
        pruned_repr = D.gather(
            dim=1,
            index=token_indices.unsqueeze(-1).expand(-1, -1, D.size(-1)),
        )
        pruned_repr = pruned_repr.masked_fill(~keep_mask.unsqueeze(-1), 0)

        return {
            "mv_repr": torch.cat((prefix, pruned_repr), dim=1),
            "mv_mask": torch.cat((prefix_mask.bool(), keep_mask), dim=1).to(D_mask.dtype),
        }

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        D = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H
        D = F.normalize(self.linear(D), p=2, dim=-1)  # B, L, D

        return self._pruning(D, attention_mask)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DivPrune":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        pruning_factor = config.get("pruning_factor", 0.1)
        protected_tokens = config.get("protected_tokens", 2)
        temp = config.get("temp")

        return cls(pretrained_model, dim, pruning_factor, protected_tokens=protected_tokens, temp=temp)
