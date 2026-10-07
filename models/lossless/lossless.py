import os
from typing import Any

import torch
from safetensors.torch import load_file
from scipy.optimize import linprog
from torch import nn, Tensor
from transformers import AutoModel

from models.base import BaseModel
from models.factories import register_model
from utils.utils import load_config, mask_sep


def lossless_score(
    Q: Tensor,
    D: Tensor,
    D_mask: Tensor,
    pairwise: bool = False,
) -> Tensor:
    """ColBERTP MaxSim with a zero lower bound on each query-token score."""
    num_queries, _, dim = Q.shape
    num_documents = D.size(0)

    if pairwise:
        assert num_documents % num_queries == 0
        documents_per_query = num_documents // num_queries
        D = D.view(num_queries, documents_per_query, -1, dim)
        similarities = torch.einsum("bid,bnjd->binj", Q.to(D.dtype), D)
        similarities.masked_fill_(
            ~D_mask.view(num_queries, 1, documents_per_query, -1).bool(),
            float("-inf"),
        )
    else:
        similarities = Q.view(-1, dim).to(D.dtype) @ D.view(-1, dim).T
        similarities = similarities.view(
            *Q.shape[:2], num_documents, D.size(-2)
        )
        similarities.masked_fill_(
            ~D_mask[None, None].bool(),
            float("-inf"),
        )

    return similarities.amax(dim=-1).relu().sum(dim=1)


@register_model("lossless")
class Lossless(BaseModel):
    """ColBERTP with trainable pruning regularization and inference pruning."""

    REGULARIZATION_DEFAULTS = {
        "none": 0.0,
        "l1": 0.01,
        "doc_sim": 0.8,
        "nuclear": 0.1,
    }

    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        projector_dim: int = 32,
        projector_norm: float = 1e-3,
        pruning: str = "none",
        norm_threshold: float = 0.5,
        lp_threshold: float = 0.7,
        protected_tokens: int = 2,
        regularization: str = "none",
        regularization_weight: float = 0.0,
        doc_similarity_eps: float = 1e-2,
        temp: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        if pruning not in {"none", "norm", "lp"}:
            raise ValueError("pruning must be 'none', 'norm' or 'lp'")
        if regularization not in self.REGULARIZATION_DEFAULTS:
            raise ValueError(
                "regularization must be 'none', 'l1', 'doc_sim' or 'nuclear'"
            )

        # Keep the backbone identical to this project's ColBERT so its
        # checkpoints can initialize Lossless without renaming parameters.
        self.bert = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, dim, bias=False)
        self.projector = nn.Linear(hidden_size, projector_dim, bias=False)
        with torch.no_grad():
            self.projector.weight.copy_(
                nn.functional.normalize(self.projector.weight, p=2, dim=-1)
                * projector_norm
            )

        self.pruning = pruning
        self.norm_threshold = norm_threshold
        self.lp_threshold = lp_threshold
        self.protected_tokens = protected_tokens
        self.regularization = regularization
        self.regularization_weight = regularization_weight
        self.doc_similarity_eps = doc_similarity_eps

    def _project(self, hidden: Tensor) -> Tensor:
        base = self.linear(hidden)
        residual = self.projector(hidden)
        denominator = torch.sqrt(
            base.square().sum(dim=-1, keepdim=True)
            + residual.square().sum(dim=-1, keepdim=True)
        )
        return base / denominator.clamp_min(1e-12)

    def encode_qry(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        return {"mv_repr": self._project(hidden)}

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        document_repr = self._project(hidden)
        document_mask = mask_sep(attention_mask)

        if not self.training:
            if self.pruning == "norm":
                document_mask = self._norm_pruning(
                    document_repr,
                    document_mask,
                )
            elif self.pruning == "lp":
                document_mask = self._lp_pruning(
                    document_repr,
                    document_mask,
                )

        return {
            "mv_repr": document_repr,
            "mv_mask": document_mask,
        }

    def _protect_prefix(self, pruned_mask: Tensor, original_mask: Tensor) -> Tensor:
        prefix_length = min(self.protected_tokens, pruned_mask.size(1))
        pruned_mask[:, :prefix_length] = original_mask[:, :prefix_length]
        return pruned_mask

    def _norm_pruning(self, D: Tensor, D_mask: Tensor) -> Tensor:
        pruned_mask = D.norm(dim=-1).ge(self.norm_threshold)
        pruned_mask = pruned_mask & D_mask.bool()
        pruned_mask = self._protect_prefix(pruned_mask, D_mask.bool())
        return pruned_mask.to(D_mask.dtype)

    @staticmethod
    def _is_dominated(document: Tensor, token_index: int) -> bool:
        token = document[:, token_index]
        constraint = document - token[:, None]
        constraint = torch.cat(
            (constraint[:, :token_index], constraint[:, token_index + 1 :]),
            dim=1,
        )
        result = linprog(
            torch.zeros(document.size(1) - 1).numpy(),
            A_eq=constraint.numpy(),
            b_eq=token.numpy(),
            bounds=[(0, None)] * (document.size(1) - 1),
            method="highs",
        )
        return bool(result.success)

    @torch.no_grad()
    def _lp_pruning(self, D: Tensor, D_mask: Tensor) -> Tensor:
        output_mask = D_mask.bool().clone()

        for batch_index in range(D.size(0)):
            valid_indices = D_mask[batch_index].bool().nonzero().flatten()
            if valid_indices.numel() <= 1:
                continue

            document = D[batch_index, valid_indices].float().T
            _, singular_values, right_vectors = torch.linalg.svd(
                document,
                full_matrices=False,
            )
            cumulative = torch.cumsum(
                singular_values / singular_values.sum().clamp_min(1e-12),
                dim=0,
            )
            reduced_dim = int((cumulative <= self.lp_threshold).sum()) + 1
            reduced = right_vectors[:reduced_dim].detach().cpu()

            max_values, max_indices = reduced.max(dim=-1)
            min_values, min_indices = reduced.min(dim=-1)
            required = max_indices[max_values > 0]
            required = torch.cat((required, min_indices[min_values < 0])).unique()
            protected = (valid_indices < self.protected_tokens).nonzero().flatten().cpu()
            required = torch.cat((required, protected)).unique()

            for token_index in range(reduced.size(1)):
                if bool((required == token_index).any()):
                    continue
                if self._is_dominated(reduced, token_index):
                    output_mask[batch_index, valid_indices[token_index]] = False

        return output_mask.to(D_mask.dtype)

    @classmethod
    def score(
        cls,
        Q: dict[str, Tensor],
        D: dict[str, Tensor],
        pairwise: bool = True,
    ) -> Tensor:
        return lossless_score(
            Q["mv_repr"],
            D["mv_repr"],
            D["mv_mask"],
            pairwise=pairwise,
        )

    def _regularization_loss(self, D: dict[str, Tensor]) -> Tensor:
        document_repr = D["mv_repr"].float()
        document_mask = D["mv_mask"].bool()
        mask = document_mask.to(document_repr.dtype)

        if self.regularization == "l1":
            values = document_repr.norm(p=1, dim=-1) * mask
            return values.sum() / mask.sum().clamp_min(1)

        if self.regularization == "nuclear":
            matrices = document_repr * mask[..., None]
            nuclear_norm = torch.linalg.matrix_norm(
                matrices.transpose(1, 2),
                ord="nuc",
            )
            lengths = document_mask.sum(dim=-1).clamp_max(document_repr.size(-1))
            return (nuclear_norm / lengths.clamp_min(1)).mean()

        norms = document_repr.norm(dim=-1)
        similarities = torch.bmm(document_repr, document_repr.transpose(1, 2))
        similarities = similarities.relu()
        pair_mask = document_mask[:, :, None] & document_mask[:, None, :]
        diagonal = torch.eye(
            document_repr.size(1),
            device=document_repr.device,
            dtype=torch.bool,
        )
        similarities = similarities * (pair_mask & ~diagonal).to(similarities.dtype)
        weights = (1 - norms) / (norms + self.doc_similarity_eps)
        numerator = (weights * similarities.sum(dim=-1) * mask).sum(dim=-1)
        lengths = document_mask.sum(dim=-1)
        denominator = (lengths * (lengths - 1)).clamp_min(1)
        return -(numerator / denominator).mean()

    def compute_loss(self, Q: dict[str, Tensor], D: dict[str, Tensor]) -> Tensor:
        retrieval_loss = super().compute_loss(Q, D)
        self.loss_dict["retrieval_loss"] = retrieval_loss.detach()

        if self.regularization == "none" or self.regularization_weight == 0:
            return retrieval_loss

        regularization_loss = self._regularization_loss(D)
        self.loss_dict["regularization_loss"] = regularization_loss.detach()
        return retrieval_loss + self.regularization_weight * regularization_loss

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Lossless":
        regularization = config.get("regularization", "none")
        regularization_weight = config.get(
            "regularization_weight",
            cls.REGULARIZATION_DEFAULTS.get(regularization, 0.0),
        )
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        projector_dim = config.get("projector_dim", 32)
        projector_norm = config.get("projector_norm", 1e-3)
        pruning = config.get("pruning", "none")
        norm_threshold = config.get("norm_threshold", 0.5)
        lp_threshold = config.get("lp_threshold", 0.7)
        protected_tokens = config.get("protected_tokens", 2)
        doc_similarity_eps = config.get("doc_similarity_eps", 1e-2)
        temp = config.get("temp")

        return cls(pretrained_model=pretrained_model, dim=dim, projector_dim=projector_dim, projector_norm=projector_norm, pruning=pruning, norm_threshold=norm_threshold, lp_threshold=lp_threshold, protected_tokens=protected_tokens, regularization=regularization, regularization_weight=regularization_weight, doc_similarity_eps=doc_similarity_eps, temp=temp)

    @classmethod
    def from_pretrained(cls, config: str | dict[str, Any]) -> "Lossless":
        config = load_config(config)
        model = cls.from_config(config)
        checkpoint_path = config.get("ckpt_path")
        if checkpoint_path is None:
            return model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file `{checkpoint_path}` does not exist"
            )

        state_dict = load_file(checkpoint_path)
        state_dict.pop("bert.embeddings.position_ids", None)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing - {"projector.weight"} or unexpected:
            raise RuntimeError(
                "Checkpoint is incompatible with Lossless: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return model
