import os
from typing import Any

from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel

from models.base import BaseModel, colbert_score
from models.factories import register_model
from utils.utils import load_config


@register_model("leapmv")
class LeapMV(BaseModel):
    """Learnable document-token pruning for ColBERT representations."""

    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        dropout: float = 0.1,
        gumbel_temp: float = 2 / 3,
        keep_ratio: float | None = None,
        protected_tokens: int = 0,
        temp: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False)

        self.gumbel_temp = gumbel_temp
        self.keep_ratio = keep_ratio
        self.protected_tokens = protected_tokens
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, 2)

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.linear(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)
        return {
            "mv_repr": Q,
        }

    def _straight_through_gate(self, logits: Tensor) -> Tensor:
        soft = F.gumbel_softmax(
            logits,
            tau=self.gumbel_temp,
            hard=False,
            dim=-1,
        )
        self._gumbel_probabilities = soft.detach()
        # Same straight-through gate as hard=True; retain this sample's soft probabilities.
        index = soft.max(dim=-1, keepdim=True)[1]
        hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)
        return (hard - soft.detach() + soft)[..., 0]

    @staticmethod
    def _compact(
        D: Tensor,
        D_mask: Tensor,
        keep_scores: Tensor,
        keep_lens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        max_k = max(int(keep_lens.max().item()), 1)
        keep_scores = keep_scores.masked_fill(~D_mask.bool(), -torch.inf)
        indices = keep_scores.topk(max_k, dim=1).indices
        keep_mask = torch.arange(max_k, device=D.device)[None] < keep_lens[:, None]
        kept = D.gather(1, indices.unsqueeze(-1).expand(-1, -1, D.size(-1)))
        selected_mask = torch.zeros_like(D_mask, dtype=torch.bool).scatter(1, indices, keep_mask) & D_mask.bool()
        return kept.masked_fill(~keep_mask.unsqueeze(-1), 0), keep_mask, selected_mask

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        raw_D = self.linear(hidden)
        D = F.normalize(raw_D, p=2, dim=-1)

        prefix = D[:, : self.protected_tokens]
        prefix_mask = attention_mask[:, : self.protected_tokens].bool()
        body = D[:, self.protected_tokens :]
        body_mask = attention_mask[:, self.protected_tokens :].bool()

        # original
        # classifier_input = raw_D[:, self.protected_tokens :]
        # l2_norm
        # classifier_input = body
        # layer_norm
        # classifier_input = F.layer_norm(raw_D[:, self.protected_tokens :], (raw_D.size(-1),))
        # layer_norm / sqrt(dim)
        classifier_input = F.layer_norm(raw_D[:, self.protected_tokens :], (raw_D.size(-1),)) / (raw_D.size(-1) ** 0.5)
        
        logits = self.classifier(self.dropout(classifier_input * body_mask.unsqueeze(-1)))

        if self.training:
            gate = self._straight_through_gate(logits) * body_mask
            probabilities = F.softmax(logits.detach(), dim=-1)
            keep_mask = torch.logical_and(gate.detach(), body_mask)
            drop_mask = torch.logical_xor(body_mask, keep_mask)
            keep_count = keep_mask.sum()
            valid_count = body_mask.sum()
            drop_count = valid_count - keep_count
            self.pruning_metrics = {
                "keep_prob_mean": (probabilities[..., 0] * keep_mask).sum() / keep_count.clamp_min(1),
                "drop_prob_mean": (probabilities[..., 1] * drop_mask).sum() / drop_count.clamp_min(1),
                "gumbel_keep_prob_mean": (
                    (self._gumbel_probabilities[..., 0] * keep_mask).sum(dtype=torch.float32)
                    / keep_count.clamp_min(1)
                ),
                "gumbel_drop_prob_mean": (
                    (self._gumbel_probabilities[..., 1] * drop_mask).sum(dtype=torch.float32)
                    / drop_count.clamp_min(1)
                ),
                "gate_keep_ratio": keep_count / valid_count.clamp_min(1),
            }
            body = body * gate.unsqueeze(-1)

            # original
            # output_mask = body_mask
            # gate
            output_mask = keep_mask

        else:
            # Plain softmax (no noise):
            # probabilities = F.softmax(logits, dim=-1)
            # Gumbel-Softmax (with noise):
            probabilities = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=False, dim=-1)
            keep_scores = probabilities[..., 0]
            valid_counts = body_mask.sum(dim=1)
            if self.keep_ratio is None:
                keep_lens = torch.logical_and(keep_scores.ge(0.5), body_mask).sum(dim=1)
            else:
                keep_lens = (valid_counts * self.keep_ratio).ceil().long().clamp_min_(1)
            body, output_mask, keep_mask = self._compact(
                body,
                body_mask,
                keep_scores,
                keep_lens,
            )
            drop_mask = torch.logical_xor(body_mask, keep_mask)
            keep_counts = keep_mask.sum(dim=1)
            drop_counts = valid_counts - keep_counts
            detached_probabilities = probabilities.detach()
            self.eval_pruning_stats = torch.stack((
                (detached_probabilities[..., 0] * keep_mask).sum(dim=1, dtype=torch.float32),
                (detached_probabilities[..., 1] * drop_mask).sum(dim=1, dtype=torch.float32),
                keep_counts,
                drop_counts,
                valid_counts,
            ), dim=1)
            keep_prob_sum, drop_prob_sum, keep_count, drop_count, valid_count = self.eval_pruning_stats.sum(dim=0)
            self.pruning_metrics = {
                "keep_prob_mean": keep_prob_sum / keep_count.clamp_min(1),
                "drop_prob_mean": drop_prob_sum / drop_count.clamp_min(1),
                "gate_keep_ratio": keep_count / valid_count.clamp_min(1),
            }

        return {
            "mv_repr": torch.cat((prefix, body), dim=1),
            "mv_mask": torch.cat((prefix_mask, output_mask), dim=1).to(
                attention_mask.dtype
            ),
        }

    def compute_loss(self, Q: dict[str, Tensor], D: dict[str, Tensor]) -> Tensor:
        loss = super().compute_loss(Q, D)
        self.loss_dict.update(self.pruning_metrics)
        return loss

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
    def from_config(cls, config: dict[str, Any]) -> "LeapMV":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        dropout = config.get("dropout", 0.1)
        gumbel_temp = config.get("gumbel_temp", 2 / 3)
        keep_ratio = config.get("keep_ratio")
        protected_tokens = config.get("protected_tokens", 0)
        temp = config.get("temp")

        return cls(pretrained_model=pretrained_model, dim=dim, dropout=dropout, gumbel_temp=gumbel_temp, keep_ratio=keep_ratio, protected_tokens=protected_tokens, temp=temp)

    @classmethod
    def from_pretrained(cls, config: str | dict) -> "LeapMV":
        config = load_config(config)
        model = cls.from_config(config)

        if "ckpt_path" in config:
            ckpt_path = config["ckpt_path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file `{ckpt_path}` does not exist")

            checkpoint = load_file(ckpt_path)
            checkpoint.pop("bert.embeddings.position_ids", None)
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            allowed_missing = {"classifier.weight", "classifier.bias"}
            if set(missing) - allowed_missing or unexpected:
                raise RuntimeError(
                    f"Missing keys: {missing}; unexpected keys: {unexpected}"
                )

        return model
